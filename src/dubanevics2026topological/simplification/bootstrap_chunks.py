"""Chunked, resumable bootstrap runs (ADR 0008, follow-up on chunking and resume).

A 10,000-draw bootstrap runs for hours, so the draws are split into chunks of
``chunk_size`` draws. Each chunk is a Topographer ``BootstrapChunk`` saved as an
``.npz`` file the moment it is finished; a rerun computes only the missing draws.
Two series are run, ``observed`` (the real landscape) and ``flat`` (every raw
fitness equal, the chance level), and each has its own chunks.

Directory layout, one directory per scientific fingerprint::

    <base>/<sha256 of the scientific fingerprint, first 16 hex>/
        manifest.json
        observed-000000-000250.npz
        flat-000000-000250.npz
        ...

``<base>`` is ``$FITNESSTREE_BOOTSTRAP_CHUNK_DIR`` when set, else
``$XDG_DATA_HOME/fitnesstree/bootstrap`` when that is set, else
``~/.local/share/fitnesstree/bootstrap``. The first 8 hex of the directory name
are the suffix ``checkpoint_identity`` gives a run that has to share its name with
another one, so both name the same run. Chunks are never deleted automatically;
``fresh=True`` moves the whole directory aside to ``<dir>.replaced-<uuid>``.

``chunk_size`` is not part of the fingerprint: the draws are seeded one by one, so
any partition of ``0..n_draws-1`` gives the same result bit for bit. A rerun with
another ``chunk_size`` keeps the chunks on disk and splits only the missing draws
into the new size.

``manifest.json`` records the scientific fingerprint, ``n_draws``, ``seed``, the
last ``chunk_size``, the Topographer version and revision, a sha256 per source
file of the effective Topographer package, a setup digest per series (see
``setup_digest``) and, per chunk, its series, draw range, file and sha256. It is
written atomically after every chunk. A manifest whose fingerprint, ``n_draws``,
``seed`` or Topographer sources differ from the run asked for, or whose series
digest differs from the network about to be drawn, is refused (``ValueError``;
``--fresh`` is the way out); so are chunks of one series that disagree on their
Topographer setup fingerprint. A chunk whose file is missing, fails its checksum
or cannot be read is warned about and recomputed, also when it only fails on the
final load. A chunk file that never reached the manifest (a crash between the
two writes) is simply recomputed and replaced.

While a run uses the store it holds an exclusive ``flock`` on ``<dir>/.lock``
(``--fresh`` takes the same lock); a filesystem that cannot lock only warns.

SIGTERM, SIGUSR1 and SIGINT stop the run gracefully: the chunk in progress
finishes and is saved, the log names the signal and says how many chunks are done,
and the process exits with code 124 (the exit code of ``timeout``), so a scheduler
can resubmit it. If a pool worker dies (``BrokenProcessPool``) while a stop is
pending, that is a stop too: the saved chunks are kept and the exit code is 124.

``FITNESSTREE_BOOTSTRAP_NO_DRAW`` is a guard for runs that must not draw (an
acceptance check on a laptop, a trial of a new layout against the tracked 10,000-draw
run). Set to ``1`` (any value but empty, ``0``, ``false``, ``no`` or ``off``), it
makes ``ChunkedBootstrap`` raise ``DrawRefused`` immediately before every call of
``bootstrap.run``, the one call that draws. A run whose chunks are all saved asks for
no range, so a re-classification from complete chunks still passes; a run that still
needs a range stops there, with the chunk directory and the mirror untouched. The
guard lives here, outside every fingerprint, so setting it changes no key.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import signal
import threading
import time
import uuid
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Self

import numpy as np
import rustworkx
import topographer
from topographer.statistics import bootstrap
from topographer.statistics.bootstrap import BootstrapChunk
from tqdm import tqdm

from ..analysis import package_source_hashes
from . import checkpoint_identity
from .bootstrap_aggregation import BootstrapDistances, CoreFreeLeafCooccurrence

ENV_CHUNK_DIR = "FITNESSTREE_BOOTSTRAP_CHUNK_DIR"
ENV_NO_DRAW = "FITNESSTREE_BOOTSTRAP_NO_DRAW"
ENV_XDG_DATA = "XDG_DATA_HOME"
DEFAULT_CHUNK_BASE = Path("~/.local/share/fitnesstree/bootstrap")
MANIFEST_NAME = "manifest.json"
LOCK_NAME = ".lock"
MANIFEST_FORMAT = 3
SERIES = ("observed", "flat")
EXIT_RESUMABLE = 124
MAX_REDRAW_ROUNDS = 3  # a chunk that fails its final load is redrawn this often
STOP_SIGNALS = tuple(
    getattr(signal, name)
    for name in ("SIGTERM", "SIGUSR1", "SIGINT")
    if hasattr(signal, name)
)


class BootstrapStopped(SystemExit):
    """Raised after a stop signal, once the chunk in progress has been saved."""

    def __init__(self, message: str) -> None:
        super().__init__(EXIT_RESUMABLE)
        self.message = message


class DrawRefused(RuntimeError):
    """Raised instead of a draw while ``FITNESSTREE_BOOTSTRAP_NO_DRAW`` is set."""


def no_draw_requested(*, context: Any | None = None) -> bool:
    """Whether drawing is forbidden by RunContext or FITNESSTREE_BOOTSTRAP_NO_DRAW.

    A guard fails closed: any value is read as "do not draw" except the explicit
    off values (empty, ``0``, ``false``, ``no``, ``off``, in any case).
    """
    if context is not None and hasattr(context, "no_draw_allowed"):
        return bool(context.no_draw_allowed)
    value = os.environ.get(ENV_NO_DRAW, "").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def chunk_base(*, context: Any | None = None) -> Path:
    """The directory that holds one subdirectory per scientific fingerprint."""
    if context is not None:
        if getattr(context, "bootstrap_chunk_dir", None) is not None:
            return Path(context.bootstrap_chunk_dir)
        return DEFAULT_CHUNK_BASE.expanduser()
    override = os.environ.get(ENV_CHUNK_DIR)
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get(ENV_XDG_DATA)
    if xdg:
        return Path(xdg).expanduser() / "fitnesstree" / "bootstrap"
    return DEFAULT_CHUNK_BASE.expanduser()


def chunk_directory(scientific: Mapping[str, Any], base: Path | None = None) -> Path:
    """Return the current chunk path or a matching pre-band-independent path."""
    root = base or chunk_base()
    current = root / _chunk_key(scientific)
    if current.is_dir():
        return current
    matches: list[Path] = []
    for candidate in sorted(root.iterdir()) if root.is_dir() else ():
        manifest = candidate / MANIFEST_NAME
        if not candidate.is_dir() or not manifest.is_file():
            continue
        try:
            stored = json.loads(manifest.read_text(encoding="utf-8"))[
                "scientific_fingerprint"
            ]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not isinstance(stored, Mapping) or candidate.name != _chunk_key(stored):
            continue
        if _without_band_quantile(stored) == _without_band_quantile(scientific):
            matches.append(candidate)
    if len(matches) > 1:
        raise ValueError(
            "Several legacy bootstrap chunk directories match this draw identity: "
            + ", ".join(map(str, matches))
        )
    if matches:
        return matches[0]
    return current


def _chunk_key(scientific: Mapping[str, Any]) -> str:
    text = json.dumps(scientific, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _without_band_quantile(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove derived band and retired core settings from a draw fingerprint."""
    normalized = json.loads(json.dumps(value, default=str))
    if not isinstance(normalized, dict):
        return {}
    config = normalized.get("config")
    if isinstance(config, dict):
        config.pop("band_quantile", None)
        config.pop("core_threshold", None)
        config.pop("consensus_peak_frequency", None)
    return normalized


def missing_ranges(
    covered: Sequence[tuple[int, int]], n_draws: int, chunk_size: int
) -> list[range]:
    """The draws not yet covered, cut into ranges of at most ``chunk_size``."""
    taken = np.zeros(n_draws, dtype=bool)
    for start, stop in covered:
        taken[start:stop] = True
    ranges: list[range] = []
    position = 0
    while position < n_draws:
        if taken[position]:
            position += 1
            continue
        end = position
        while end < n_draws and not taken[end]:
            end += 1
        ranges += [
            range(a, min(a + chunk_size, end)) for a in range(position, end, chunk_size)
        ]
        position = end
    return ranges


def topographer_identity() -> dict[str, str | None]:
    """Version and git revision of the installed Topographer, when known."""
    try:
        distribution = metadata.distribution("topographer")
        version: str | None = distribution.version
        direct = distribution.read_text("direct_url.json")
        rev = (
            json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
        )
    except (metadata.PackageNotFoundError, ValueError, OSError):
        return {"version": None, "rev": None}
    return {"version": version, "rev": rev}


def library_versions() -> dict[str, str]:
    """Versions of the libraries that decide a draw's random stream and its tree.

    NumPy does not promise the same ``Generator.normal`` stream across versions,
    and rustworkx builds the trees, so chunks resume only under identical ones.
    Polars and SciPy take no part in drawing, so they are not recorded.
    """
    return {"numpy": np.__version__, "rustworkx": rustworkx.__version__}


def topographer_sources() -> dict[str, str]:
    """sha256 of every Python source of the effective Topographer package."""
    return package_source_hashes(topographer)


def setup_digest(network: Any, statistic: Any, se_attr: str = "se") -> dict[str, str]:
    """What decides a series' draws: Topographer's fingerprint plus named parts.

    ``fingerprint`` is ``bootstrap.fingerprint(network, prepared_statistic)``, the
    value every chunk carries and the one that decides a resume. ``values``, ``se``
    and ``edges`` are sha256 of the vertex scalars, the standard errors and the edge
    list as little-endian float64 / int64 arrays, in node order, and ``statistic``
    is the prepared statistic's fingerprint. The parts only exist so a refusal can
    name what changed; they never override the fingerprint.
    """
    prepared = statistic.prepare(network)
    graph = network.graph
    nodes = list(graph.node_indices())

    def hashed(array: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

    values = np.array([graph[i][network.scalar_attr] for i in nodes], dtype="<f8")
    se = np.array([graph[i][se_attr] for i in nodes], dtype="<f8")
    edges = np.asarray(graph.edge_list(), dtype="<i8").reshape(-1, 2)
    return {
        "fingerprint": str(bootstrap.fingerprint(network, prepared, se_attr=se_attr)),
        "values": hashed(values),
        "se": hashed(se),
        "edges": hashed(edges),
        "statistic": str(prepared.fingerprint()),
    }


@contextlib.contextmanager
def directory_lock(directory: Path) -> Iterator[bool]:
    """Exclusive ``flock`` on ``<directory>/.lock``; yields whether it is held.

    Same ENOSYS/ENOLCK/EOPNOTSUPP tolerance and timeout as ``checkpoint_identity``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with checkpoint_identity._timed_lock(
        directory / "chunks",
        lock_path=directory / LOCK_NAME,
        what="The bootstrap chunk store",
    ) as held:
        yield held


def _write_json_atomic(path: Path, value: Any) -> None:
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class ChunkStore:
    """The saved chunks of one run, with a manifest; in memory when ``directory`` is None."""

    def __init__(
        self,
        directory: Path | None,
        *,
        scientific: Mapping[str, Any] | None,
        n_draws: int,
        seed: int,
        chunk_size: int,
        fresh: bool = False,
        hold_lock: bool = False,
        log: Callable[[str], None] = print,
    ) -> None:
        self.directory = directory
        self.scientific = json.loads(json.dumps(scientific or {}, default=str))
        self.n_draws = n_draws
        self.seed = seed
        self.chunk_size = chunk_size
        self._log = log
        self._entries: list[dict[str, Any]] = []
        self._memory: dict[str, BootstrapChunk] = {}
        self._setups: dict[str, dict[str, str]] = {}
        self._sources: dict[str, str] | None = None
        self._legacy_manifest = False
        self._locks = contextlib.ExitStack()
        if directory is None:
            return
        try:
            if fresh and directory.exists():
                with directory_lock(directory):  # never under a running bootstrap
                    aside = directory.with_name(
                        f"{directory.name}.replaced-{uuid.uuid4().hex}"
                    )
                    directory.rename(aside)
                log(f"Bootstrap chunks: moved {directory} aside to {aside}.")
            directory.mkdir(parents=True, exist_ok=True)
            if hold_lock:
                self._locks.enter_context(directory_lock(directory))
            self._read_manifest()
        except BaseException:
            self._locks.close()
            raise

    def close(self) -> None:
        """Release the directory lock taken with ``hold_lock=True``."""
        self._locks.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _topographer_sources(self) -> dict[str, str]:
        if self._sources is None:
            self._sources = topographer_sources()
        return self._sources

    # -- manifest ------------------------------------------------------------

    def _manifest(self) -> dict[str, Any]:
        return {
            "format": MANIFEST_FORMAT,
            "scientific_fingerprint": self.scientific,
            "n_draws": self.n_draws,
            "seed": self.seed,
            "chunk_size": self.chunk_size,
            "libraries": library_versions(),
            "topographer": topographer_identity(),
            "topographer_sources": self._topographer_sources(),
            "setups": self._setups,
            "updated": datetime.now(UTC).isoformat(timespec="seconds"),
            "chunks": self._entries,
        }

    def _read_manifest(self) -> None:
        assert self.directory is not None
        path = self.directory / MANIFEST_NAME
        if not path.is_file():
            return
        try:
            found = json.loads(path.read_text(encoding="utf-8"))
            entries = list(found["chunks"])
            stored = found["scientific_fingerprint"]
            n_draws, seed = found["n_draws"], found["seed"]
            if found.get("format") != MANIFEST_FORMAT:
                raise ValueError(
                    f"format {found.get('format')} is not {MANIFEST_FORMAT}"
                )
            sources, setups = found["topographer_sources"], dict(found["setups"])
            libraries = dict(found["libraries"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(
                f"The bootstrap chunk manifest {path} is unreadable ({error}); "
                "move the directory aside (--fresh) to start again."
            ) from error
        parts = checkpoint_identity.differences(stored, self.scientific)
        self._legacy_manifest = stored != self.scientific and _without_band_quantile(
            stored
        ) == _without_band_quantile(self.scientific)
        if self._legacy_manifest:
            parts = []
        if stored != self.scientific and not parts and not self._legacy_manifest:
            parts = ["scientific fingerprint"]
        if n_draws != self.n_draws:
            parts.append(f"n_draws ({n_draws} stored, {self.n_draws} asked)")
        if seed != self.seed:
            parts.append(f"seed ({seed} stored, {self.seed} asked)")
        if parts:
            raise ValueError(
                f"Refusing to resume the bootstrap chunks in {self.directory}: "
                f"they were made with a different {', '.join(parts)}. Rerun with "
                "--fresh to move them aside and start again."
            )
        now = library_versions()
        for name in sorted(libraries.keys() | now.keys()):
            if libraries.get(name) != now.get(name):
                raise ValueError(
                    f"Refusing to resume the bootstrap chunks in {self.directory}: "
                    f"{name} is {now.get(name)} now but {libraries.get(name)} made "
                    "them, and NumPy does not promise the same random stream across "
                    "versions. Use the same uv.lock environment, or rerun with "
                    "--fresh to move the chunks aside and start again."
                )
        current = self._topographer_sources()
        if sources != current:
            changed = sorted(
                name
                for name in sources.keys() | current.keys()
                if sources.get(name) != current.get(name)
            )
            shown = ", ".join(changed[:3]) + (", ..." if len(changed) > 3 else "")
            raise ValueError(
                f"Refusing to resume the bootstrap chunks in {self.directory}: "
                f"Topographer's sources differ from the ones that made them "
                f"({len(changed)} files: {shown}; manifest {found.get('topographer')}, "
                f"now {topographer_identity()}). Rerun with --fresh to move the "
                "chunks aside and start again."
            )
        self._setups = setups
        self._entries = entries

    def _save_manifest(self) -> None:
        if self.directory is not None:
            if self._legacy_manifest:
                raise ValueError(
                    "Legacy bootstrap chunks are read-only; complete the saved "
                    "draw set before changing band settings."
                )
            _write_json_atomic(self.directory / MANIFEST_NAME, self._manifest())

    def check_setup(self, series: str, digest: Mapping[str, str]) -> None:
        """Refuse a ``series`` whose network or statistic differs from the stored one.

        Called before anything is drawn. The first digest of a series is recorded
        in the manifest at once, so a later session compares against it.
        """
        digest = dict(digest)
        stored = self._setups.get(series)
        if stored is None:
            if self._ranges(series) and self.directory is not None:
                raise ValueError(
                    f"The {series} chunks in {self.directory} have no recorded "
                    "setup digest; rerun with --fresh to start again."
                )
        elif stored != digest:
            parts = sorted(
                k
                for k in stored.keys() | digest.keys()
                if k != "fingerprint" and stored.get(k) != digest.get(k)
            ) or ["the Topographer fingerprint"]
            raise ValueError(
                f"Refusing to resume the {series} bootstrap chunks in "
                f"{self.directory}: their setup differs in {', '.join(parts)}. "
                "Rerun with --fresh to move them aside and start again."
            )
        if stored is None:
            self._setups[series] = digest
            self._save_manifest()

    # -- chunks --------------------------------------------------------------

    def _ranges(self, series: str) -> list[tuple[int, int]]:
        if self.directory is None:
            return sorted(
                (int(c.draws.min()), int(c.draws.max()) + 1)
                for name, c in self._memory.items()
                if name.startswith(f"{series}-")
            )
        return sorted(
            (e["start"], e["stop"]) for e in self._entries if e["series"] == series
        )

    def missing(self, series: str) -> list[range]:
        """Ranges of ``series`` still to compute, cut to the current ``chunk_size``."""
        return missing_ranges(self._ranges(series), self.n_draws, self.chunk_size)

    def refuse_legacy_missing(
        self, series: str, *, window: range | None = None
    ) -> None:
        """Refuse incomplete legacy stores before a draw can touch their directory."""
        if not self._legacy_manifest:
            return
        missing = (
            self.missing(series)
            if window is None
            else self.missing_within(series, window)
        )
        if missing:
            raise ValueError(
                f"Legacy bootstrap chunks in {self.directory} are read-only and "
                f"series {series!r} is incomplete; refusing to draw or modify the "
                "archive. Restore the matching legacy chunks or use a fresh run."
            )

    def missing_within(self, series: str, window: range) -> list[range]:
        """Gaps of ``series`` inside ``window``, each gap one range (one chunk).

        Draws outside the window count as taken, so a window already covered by
        saved chunks has no gap. Used by a draw slice (``bootstrap_slices``).
        """
        outside = [(0, window.start), (window.stop, self.n_draws)]
        return missing_ranges(
            [*self._ranges(series), *outside], self.n_draws, max(len(window), 1)
        )

    def done_chunks(self, series: str) -> int:
        return len(self._ranges(series))

    def add(self, series: str, chunk: BootstrapChunk) -> None:
        """Save ``chunk`` (atomically; Topographer sets its mode) and record it in the manifest."""
        start, stop = int(chunk.draws.min()), int(chunk.draws.max()) + 1
        name = f"{series}-{start:06d}-{stop:06d}.npz"
        if self.directory is None:
            self._memory[name] = chunk
            return
        if self._legacy_manifest:
            raise ValueError(
                "Legacy bootstrap chunks are read-only; refusing to append a draw."
            )
        path = self.directory / name
        chunk.save(path)
        self._entries = [
            e for e in self._entries if (e["series"], e["start"]) != (series, start)
        ]
        self._entries.append(
            {
                "series": series,
                "start": start,
                "stop": stop,
                "file": name,
                "sha256": checkpoint_identity.sha256_of(path),
            }
        )
        self._save_manifest()

    def load(self, series: str) -> list[BootstrapChunk]:
        """Every valid chunk of ``series``; a bad one is warned about and forgotten."""
        if self.directory is None:
            return [
                c for n, c in sorted(self._memory.items()) if n.startswith(f"{series}-")
            ]
        chunks: list[BootstrapChunk] = []
        kept: list[dict[str, Any]] = []
        for entry in self._entries:
            if entry["series"] != series:
                kept.append(entry)
                continue
            problem = self._problem(entry)
            if problem is None:
                try:
                    chunks.append(BootstrapChunk.load(self.directory / entry["file"]))
                    kept.append(entry)
                    continue
                except (OSError, ValueError, KeyError) as error:
                    problem = f"cannot be read ({error})"
            warnings.warn(
                f"Bootstrap chunk {entry['file']} {problem}; recomputing draws "
                f"{entry['start']}-{entry['stop'] - 1}.",
                RuntimeWarning,
                stacklevel=2,
            )
        if len(kept) != len(self._entries):
            self._entries = kept
            self._save_manifest()
        if len({chunk.fingerprint for chunk in chunks}) > 1:
            raise ValueError(
                f"The {series} chunks in {self.directory} come from different "
                "landscapes (their Topographer setup fingerprints disagree); rerun "
                "with --fresh."
            )
        return chunks

    def _problem(self, entry: Mapping[str, Any]) -> str | None:
        assert self.directory is not None
        path = self.directory / entry["file"]
        if not path.is_file():
            return "is missing"
        try:
            if checkpoint_identity.sha256_of(path) != entry["sha256"]:
                return "fails its checksum"
        except OSError as error:
            return f"cannot be read ({error})"
        return None

    def complete(self) -> bool:
        """Whether every series has all its draws saved, each chunk present and intact."""
        return self.directory is not None and all(
            not self.missing(series)
            and all(
                self._problem(entry) is None
                for entry in self._entries
                if entry["series"] == series
            )
            for series in SERIES
        )

    def verify(self) -> None:
        """Drop (with a warning) every recorded chunk that is missing or corrupt."""
        for series in SERIES:
            self.load(series)


def chunks_complete(
    scientific: Mapping[str, Any],
    *,
    n_draws: int,
    seed: int,
    chunk_size: int,
    base: Path | None = None,
) -> bool:
    """Whether the saved chunks of this scientific run hold every draw.

    Read-only: nothing is created, locked or drawn. A manifest that would be refused
    for a resume (other versions or Topographer sources) counts as incomplete.
    """
    directory = chunk_directory(scientific, base)
    if not (directory / MANIFEST_NAME).is_file():
        return False
    try:
        store = ChunkStore(
            directory,
            scientific=scientific,
            n_draws=n_draws,
            seed=seed,
            chunk_size=chunk_size,
        )
    except (OSError, ValueError):
        return False
    return store.complete()


class StopFlag:
    """Set by SIGTERM, SIGUSR1 or SIGINT; installed only from the main thread."""

    def __init__(self) -> None:
        self.requested = False
        self.signal: int | None = None

    def _handler(self, number: int, _frame: Any) -> None:
        self.requested = True
        self.signal = number

    @contextlib.contextmanager
    def installed(self) -> Iterator[StopFlag]:
        if threading.current_thread() is not threading.main_thread():
            yield self
            return
        previous = {n: signal.signal(n, self._handler) for n in STOP_SIGNALS}
        try:
            yield self
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


@dataclass
class ChunkedBootstrap:
    """Run the series of one bootstrap chunk by chunk and combine them.

    ``store`` decides where chunks live (a directory, or memory when it has none);
    ``stop`` is polled after every saved chunk.
    """

    store: ChunkStore
    n_jobs: int = 1
    log: Callable[[str], None] = print
    stop: StopFlag = field(default_factory=StopFlag)
    series: tuple[str, ...] = SERIES
    session_draws: dict[str, int] = field(default_factory=dict)
    # A draw slice: only these draws are drawn, as one chunk per series, and
    # nothing is combined (``draw_window``).
    window: range | None = None
    # Whether drawing is forbidden; ``None`` reads the environment when a draw is due.
    no_draw: bool | None = None

    def _pending(self, name: str) -> list[range]:
        if self.window is None:
            return self.store.missing(name)
        return self.store.missing_within(name, self.window)

    def remaining_chunks(self) -> int:
        return sum(len(self._pending(name)) for name in self.series)

    def done_chunks(self) -> int:
        return sum(self.store.done_chunks(name) for name in self.series)

    def _signal_name(self) -> str:
        try:
            return signal.Signals(self.stop.signal).name
        except ValueError:
            return f"signal {self.stop.signal}"

    def _stop(self, reason: str) -> BootstrapStopped:
        done = self.done_chunks()
        message = f"resumable: {done}/{done + self.remaining_chunks()} chunks done"
        self.log(f"Bootstrap stopped by {reason}; {message}.")
        return BootstrapStopped(message)

    def _check_stop(self) -> None:
        if self.stop.requested and self.remaining_chunks() > 0:
            raise self._stop(self._signal_name())

    def run(
        self,
        name: str,
        network: Any,
        statistic: Any,
        *,
        label: str | None = None,
    ) -> BootstrapDistances:
        """Compute the missing chunks of series ``name`` and return the combined result."""
        store = self.store
        label = label or name
        # Before any draw: the stored setup must be the one about to be drawn.
        store.check_setup(name, setup_digest(network, statistic))
        store.verify()
        store.refuse_legacy_missing(name)
        self._check_stop()
        for attempt in range(MAX_REDRAW_ROUNDS):
            self._draw(name, network, statistic, label)
            chunks = store.load(name)  # a chunk that fails now is redrawn next round
            if not store.missing(name):
                break
        else:
            raise ValueError(
                f"The {label} chunks keep failing to load after "
                f"{MAX_REDRAW_ROUNDS} rounds; check the disk or rerun with --fresh."
            )
        prepared = CoreFreeLeafCooccurrence(statistic).prepare(network)
        expected = prepared.fingerprint()
        for position, chunk in enumerate(chunks):
            if chunk.statistic_fingerprint != expected:
                raise ValueError(
                    f"chunk {position} has a different statistic fingerprint; "
                    "chunks were made with a different statistic or reference."
                )
        return bootstrap.combine(chunks, prepared)

    def draw_window(
        self, name: str, network: Any, statistic: Any, *, label: str | None = None
    ) -> int:
        """Draw the missing part of ``self.window`` of series ``name``; return the draws made.

        The same checks as ``run`` come first (setup digest, damaged chunks). A
        window that is already saved and intact draws nothing. The window is one
        chunk unless saved chunks cut it up, and nothing is combined.
        """
        assert self.window is not None
        store = self.store
        store.check_setup(name, setup_digest(network, statistic))
        store.verify()
        store.refuse_legacy_missing(name, window=self.window)
        self._check_stop()
        before = self.session_draws.get(name, 0)
        self._draw(name, network, statistic, label or name)
        return self.session_draws.get(name, 0) - before

    def _draw(self, name: str, network: Any, statistic: Any, label: str) -> None:
        store = self.store
        pending = self._pending(name)
        total = store.n_draws if self.window is None else len(self.window)
        done_draws = total - sum(len(r) for r in pending)
        bar = tqdm(
            total=total,
            initial=done_draws,
            desc=label,
            unit="draw",
            disable=None,
        )
        started, session_draws = time.perf_counter(), 0
        try:
            for draws in pending:
                last = {"done": 0}

                def advance(done: int, total: int, last: dict[str, int] = last) -> None:
                    bar.update(done - last["done"])
                    last["done"] = done

                if self.no_draw if self.no_draw is not None else no_draw_requested():
                    raise DrawRefused(
                        f"{ENV_NO_DRAW} is set, so nothing is drawn, but series "
                        f"{name!r} ({label}) still needs draws "
                        f"{draws.start}-{draws.stop - 1} of {store.n_draws}. Unset "
                        f"{ENV_NO_DRAW} to draw them; the saved chunks are untouched."
                    )
                try:
                    chunk = bootstrap.run(
                        network,
                        statistic,
                        se_attr="se",
                        n_draws=store.n_draws,
                        seed=store.seed,
                        draws=draws,
                        n_jobs=self.n_jobs,
                        progress=advance,
                    )
                except BrokenProcessPool:
                    # Workers die on the scheduler's SIGTERM; with a stop pending
                    # that is the stop, and the saved chunks are all still good.
                    if self.stop.requested:
                        raise self._stop(
                            f"{self._signal_name()} (a worker died)"
                        ) from None
                    raise
                bar.update(len(draws) - last["done"])
                store.add(name, chunk)
                session_draws += len(draws)
                done_draws += len(draws)
                self.session_draws[name] = self.session_draws.get(name, 0) + len(draws)
                self._log_chunk(label, draws, done_draws, total, session_draws, started)
                self._check_stop()
        finally:
            bar.close()

    def _log_chunk(
        self,
        label: str,
        draws: range,
        done: int,
        total: int,
        session: int,
        started: float,
    ) -> None:
        elapsed = max(time.perf_counter() - started, 1e-9)
        rate = session / elapsed
        left = total - done
        eta = "done" if left <= 0 else f"ETA {_duration(left / rate)}"
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        self.log(
            f"[{stamp}] {label}: draws {draws.start}-{draws.stop - 1} saved; "
            f"{done}/{total} draws, {rate:.3g} draws/s, {eta}."
        )


def _duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes}m{secs:02d}s"
