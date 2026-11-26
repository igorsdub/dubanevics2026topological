"""Figure labels read from the manuscript quantity registry.

A figure that states a number asks the registry for it by key (``display``), so the
label cannot diverge from the manuscript. Every refusal names the key and the
registry path. Staleness is checked lazily, per requested key, against the current
fingerprint of the row's ``source_stage``: the same fingerprints the registry
recorded (``Upstream.fingerprint``, read again through ``load_sources``). The
current fingerprints are computed once, on the first key that needs them, because
that reads the upstream outputs through their cached loaders; a figure that asks
for no key pays nothing. Rows sourced only from ``config/analysis.toml`` carry no
fingerprint and are not checked here; the whole-registry check
(``load_current_manuscript_quantities``, and ``build_figures``) covers them, since a
config change changes the registry's own stage fingerprint.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import polars as pl

from ..analysis.manuscript_quantities import STALE, Upstream, combine, gap_kind

REGISTRY_PARTS = ("tables", "manuscript_quantities", "manuscript_quantities.parquet")
RERUN = "run `just manuscript-quantities` first"

CurrentFingerprints = Callable[[], Mapping[str, str | None]]


class QuantityError(Exception):
    """Base of every refusal to read a manuscript quantity."""


class QuantityRegistryMissing(QuantityError, FileNotFoundError):
    """The registry file does not exist."""


class QuantityKeyMissing(QuantityError):
    """The key is not a row of the registry."""


class QuantityGapError(QuantityError):
    """The key is a gap row: the registry could not source it."""

    def __init__(self, key: str, reason: str, path: Path) -> None:
        super().__init__(f"{key}: {reason} (registry {path})")
        self.key = key
        self.reason = reason


class QuantityStaleError(QuantityError):
    """The row was built from an upstream that has since changed."""


def registry_path(output: Path) -> Path:
    return output.joinpath(*REGISTRY_PARTS)


def registry_fingerprint(output: Path) -> str:
    """The registry checkpoint's fingerprint, from its export record (for figure records).

    The record and its currency are the analysis module's (``require_current_registry``
    is what Table 1 runs); this only translates a missing record into this module's error.
    """
    from ..analysis.manuscript_quantities import read_registry_fingerprint

    try:
        return read_registry_fingerprint(output)
    except FileNotFoundError as error:
        raise QuantityRegistryMissing(str(error)) from error


def current_upstream_fingerprints(output: Path) -> dict[str, str | None]:
    """Fingerprint now of each upstream stage, by stage name (reuses ``load_sources``)."""
    from ..analysis.manuscript_quantities import load_sources

    return {
        up.stage: up.fingerprint
        for up in load_sources(output).values()
        if up is not None
    }


class ManuscriptQuantities:
    """Read-only view of the registry: ``display(key)`` and the keys asked for."""

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        path: Path,
        current_fingerprints: CurrentFingerprints | None = None,
    ) -> None:
        """``current_fingerprints`` returns the stage-to-fingerprint map used for the
        staleness check; ``None`` skips the check (tests, in-memory rows)."""
        self._rows = {str(row["key"]): dict(row) for row in rows}
        self._path = path
        self._current_fingerprints = current_fingerprints
        self._current: Mapping[str, str | None] | None = None
        self._requested: dict[str, None] = {}

    @property
    def path(self) -> Path:
        return self._path

    @property
    def requested_keys(self) -> tuple[str, ...]:
        """Every key asked for, in first-request order (failed requests included)."""
        return tuple(self._requested)

    def display(self, key: str) -> str:
        """The registry's ``display_value`` for ``key``, or a ``QuantityError``."""
        return str(self._sourced(key)["display_value"])

    def value(self, key: str) -> str:
        """The registry's exact ``value`` text for ``key``, or a ``QuantityError``.

        For comparing a figure's own number with the registry, never for printing.
        """
        return str(self._sourced(key)["value"])

    def _sourced(self, key: str) -> Mapping[str, Any]:
        """The row of ``key`` once it is known to be sourced and current."""
        self._requested[key] = None
        row = self._rows.get(key)
        if row is None:
            raise QuantityKeyMissing(
                f"{key}: not a key of the manuscript quantity registry ({self._path})."
            )
        reason = row.get("gap_reason")
        shown = row.get("display_value")
        if not reason and shown is None:
            raise QuantityError(
                f"{key}: the registry row has no display value and no gap reason, so "
                f"the registry is malformed ({self._path}); {RERUN}."
            )
        if reason:
            if gap_kind(reason) == STALE:
                # ``stale:`` is the registry's own signal that an upstream output
                # exists but is not current: the same refusal as a stale row.
                raise QuantityStaleError(f"{key}: {reason} (registry {self._path})")
            raise QuantityGapError(key, reason, self._path)
        self._check_fresh(key, row)
        return row

    def _check_fresh(self, key: str, row: Mapping[str, Any]) -> None:
        recorded = row.get("source_fingerprint")
        if recorded is None or self._current_fingerprints is None:
            return
        if self._current is None:
            self._current = self._current_fingerprints()
        # A derived row joins its inputs' stages and fingerprints with ``+``; build the
        # current value with ``combine`` itself, so a part without a fingerprint (a
        # config input, or a stage that records none) reads ``none`` exactly as there.
        stages = str(row["source_stage"]).split("+")
        joined = combine(
            *(
                Upstream(
                    stage,
                    None if stage.startswith("config/") else self._current.get(stage),
                )
                for stage in stages
            )
        )
        now = None if joined is None else joined.fingerprint
        if now != recorded:
            raise QuantityStaleError(
                f"{key}: the registry row is stale ({self._path}); its source "
                f"{row['source_stage']} recorded fingerprint {recorded!r} but is now "
                f"{now!r}; {RERUN}."
            )


def load_current_manuscript_quantities(
    output: Path,
    settings: dict[str, Any] | None = None,
) -> ManuscriptQuantities:
    """Run the full registry-currency check, then load the registry.

    The per-key check cannot see a row with no fingerprint (a config row such as the
    Published threshold), so a reader outside ``build_figures`` (the notebooks) loads
    through this: it raises unless the registry's export record matches the current
    stage fingerprint, the same check Table 1 and ``build_figures`` run.
    """
    from ..analysis import load_settings
    from ..build import require_current_registry

    require_current_registry(settings or load_settings(), output)
    return load_manuscript_quantities(output)


def from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    path: Path = Path("<memory>"),
    current_fingerprints: CurrentFingerprints | None = None,
) -> ManuscriptQuantities:
    """A view over in-memory rows; staleness is checked only when a map is given."""
    return ManuscriptQuantities(
        rows, path=path, current_fingerprints=current_fingerprints
    )


def load_manuscript_quantities(
    output: Path,
    *,
    current_fingerprints: CurrentFingerprints | None = None,
) -> ManuscriptQuantities:
    """Read the exported registry under ``output``; never builds it."""
    path = registry_path(output)
    if not path.is_file():
        raise QuantityRegistryMissing(
            f"The manuscript quantity registry is missing ({path}); {RERUN}."
        )
    return ManuscriptQuantities(
        pl.read_parquet(path).to_dicts(),
        path=path,
        current_fingerprints=current_fingerprints
        or (lambda: current_upstream_fingerprints(output)),
    )
