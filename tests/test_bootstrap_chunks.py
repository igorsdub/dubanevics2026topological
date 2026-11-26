"""Chunked, resumable bootstrap: equivalence, resume, refusals, signals (ADR 0008)."""

from __future__ import annotations

import errno
import json
import os
import signal
import stat
import warnings
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest
from test_bootstrap_smoothed_tree import (
    CONFIG,
    RAW_FITNESS,
    _growth,
    _network,
    _patched_compute,
    _toy,
)
from dubanevics2026topological.simplification.cooccurrence_leaf import LeafCooccurrence
from dubanevics2026topological.simplification.cooccurrence_terminal import (
    bootstrap_terminal_cooccurrence,
)
from topographer.statistics import bootstrap as topographer_bootstrap
from topographer.topology import ScalarNetworkTopology

from dubanevics2026topological.simplification import bootstrap_chunks, checkpoint_identity
from dubanevics2026topological.simplification import bootstrap_smoothed_tree as module
from dubanevics2026topological.simplification.bootstrap_chunks import (
    BootstrapStopped,
    ChunkedBootstrap,
    ChunkStore,
    chunk_base,
    chunk_directory,
    missing_ranges,
)

N = 40
SIZE = 7
SCIENTIFIC = {"config": {"n_replicates": N, "seed": CONFIG["seed"]}}
TABLES = (
    "hills",
    "draw_stats",
)


HILLS = {"b": [1, 0], "d": [3], "f": [5, 6]}  # node indices, as the pipeline passes


def _statistic(network=None):
    reference = ScalarNetworkTopology(network or _network()).split_tree()
    return LeafCooccurrence("split", reference=reference, nominal_hills=HILLS)


def _chunk(draws=range(7), **network_options):
    return topographer_bootstrap.run(
        _network(**network_options),
        _statistic(),
        se_attr="se",
        n_draws=N,
        seed=CONFIG["seed"],
        draws=draws,
    )


def _store(directory, *, chunk_size=SIZE, scientific=SCIENTIFIC, **kwargs):
    return ChunkStore(
        directory,
        scientific=scientific,
        n_draws=N,
        seed=CONFIG["seed"],
        chunk_size=chunk_size,
        log=lambda message: None,
        **kwargs,
    )


def _runner(directory, **kwargs):
    return ChunkedBootstrap(_store(directory, **kwargs), log=lambda message: None)


def _build(runner):
    prepared, tree = _toy()
    return module.build_bootstrap_smoothed_tree(
        prepared,
        tree,
        _growth(),
        {
            **{
                key: value
                for key, value in CONFIG.items()
                if key not in {"core_threshold", "consensus_peak_frequency"}
            },
            "n_replicates": N,
        },
        log=lambda message: None,
        chunks=runner,
    )


def _same_tables(left, right):
    for name in TABLES:
        assert getattr(left, name).equals(getattr(right, name)), name

    def comparable_summary(value):
        summary = {k: v for k, v in value.items() if k != "timing"}
        evidence = dict(summary.get("evidence_availability", {}))
        evidence.pop("chunks_present", None)
        if evidence:
            summary["evidence_availability"] = evidence
        else:
            summary.pop("evidence_availability", None)
        return summary

    assert comparable_summary(left.summary) == comparable_summary(right.summary)


@pytest.fixture(scope="module")
def unchunked():
    """The one-chunk run every other run must equal."""
    store = ChunkStore(
        None,
        scientific=None,
        n_draws=N,
        seed=CONFIG["seed"],
        chunk_size=N,
    )
    return _build(ChunkedBootstrap(store, log=lambda message: None))


class _Calls(list):
    """The draw ranges asked for, plus callbacks run before each call."""

    def __init__(self) -> None:
        super().__init__()
        self.hooks: list = []


@pytest.fixture
def calls(monkeypatch):
    """Record the draw ranges Topographer is asked for, and let a test hook in."""
    record = _Calls()
    original = topographer_bootstrap.run

    def run(*args, **kwargs):
        record.append(kwargs["draws"])
        for hook in record.hooks:
            hook(len(record))
        return original(*args, **kwargs)

    monkeypatch.setattr(bootstrap_chunks.bootstrap, "run", run)
    return record


# -- equivalence ----------------------------------------------------------------


def test_chunked_equals_unchunked(tmp_path, unchunked):
    chunked = _build(_runner(tmp_path / "chunks"))
    _same_tables(chunked, unchunked)


def test_chunk_size_does_not_change_the_result(unchunked):
    _same_tables(_build(_runner(None, chunk_size=3)), unchunked)
    _same_tables(_build(_runner(None, chunk_size=1000)), unchunked)


def test_chunked_path_matches_deprecated_measurements():
    """The retired upstream method and core-free adapter share measurements."""
    network = _network()
    statistic = _statistic(network)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = bootstrap_terminal_cooccurrence(
            network,
            se_attr="se",
            tree_type="split",
            n_replicates=N,
            seed=CONFIG["seed"],
            core_cutoff=0.95,
            reference=statistic.reference,
            nominal_hills=statistic.nominal_hills,
            n_jobs=1,
        )
    result = _runner(None).run("observed", network, statistic)
    old_draws = old.draw_stats.rename({"peak_count": "leaf_count"})
    exact = ["draw", "leaf_count", "reproduced_pair_fraction"]
    assert result.draw_stats.select(exact).equals(old_draws.select(exact))
    for name in ("wasserstein_1", "bottleneck"):
        assert result.draw_stats[name].to_numpy() == pytest.approx(
            old_draws[name].to_numpy(), rel=1e-12, abs=0
        )
    assert result.nominal_hill_support.equals(old.nominal_hill_support)


# -- resume ---------------------------------------------------------------------


def test_a_crashed_run_resumes_without_recomputing_finished_chunks(
    tmp_path, calls, unchunked
):
    directory = tmp_path / "chunks"

    def crash(count):
        if count == 4:
            raise RuntimeError("killed")

    calls.hooks.append(crash)
    with pytest.raises(RuntimeError, match="killed"):
        _build(_runner(directory))
    assert calls == [range(7), range(7, 14), range(14, 21), range(21, 28)]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert [(c["start"], c["stop"]) for c in manifest["chunks"]] == [
        (0, 7),
        (7, 14),
        (14, 21),
    ]

    calls.clear()
    calls.hooks.clear()
    resumed = _build(_runner(directory))
    assert calls[0] == range(21, 28)  # nothing finished is drawn again
    assert sum(len(r) for r in calls) == 2 * N - 21
    _same_tables(resumed, unchunked)

    calls.clear()
    _build(_runner(directory))  # everything is saved: no draw at all
    assert calls == []


def test_a_new_chunk_size_keeps_the_saved_chunks(tmp_path, calls, unchunked):
    directory = tmp_path / "chunks"
    calls.hooks.append(lambda count: count == 3 and (_ for _ in ()).throw(KeyError()))
    with pytest.raises(KeyError):
        _build(_runner(directory))
    calls.clear()
    calls.hooks.clear()
    resumed = _build(_runner(directory, chunk_size=10))
    assert calls[0] == range(14, 24)
    _same_tables(resumed, unchunked)


# -- the no-draw guard (ADR 0013, Phase 3) -----------------------------------------


def _saved(directory: Path) -> dict[str, bytes]:
    """Every file of a chunk directory, by name (the lock file is empty)."""
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir()) if p.is_file()}


def test_no_draw_guard_raises_before_drawing(tmp_path, monkeypatch, calls):
    """``FITNESSTREE_BOOTSTRAP_NO_DRAW`` stops a run just before it would draw.

    The guard stands before every ``bootstrap.run`` call, so a run that still has a
    range to draw refuses at once, without calling Topographer and without touching
    what the chunk directory already holds.
    """
    directory = tmp_path / "chunks"
    assert bootstrap_chunks.ENV_NO_DRAW == "FITNESSTREE_BOOTSTRAP_NO_DRAW"
    monkeypatch.setenv(bootstrap_chunks.ENV_NO_DRAW, "1")

    with pytest.raises(bootstrap_chunks.DrawRefused) as refused:
        _build(_runner(directory))
    assert isinstance(refused.value, RuntimeError)
    message = str(refused.value)
    assert bootstrap_chunks.ENV_NO_DRAW in message
    assert "observed" in message and "0-6" in message  # the first range it refused
    assert calls == []  # not a single draw was asked for
    assert not [p for p in directory.iterdir() if p.suffix == ".npz"]

    # With some chunks already saved, the guard refuses the first missing range and
    # leaves every saved byte as it was.
    monkeypatch.delenv(bootstrap_chunks.ENV_NO_DRAW)
    calls.hooks.append(lambda count: count == 4 and (_ for _ in ()).throw(KeyError()))
    with pytest.raises(KeyError):
        _build(_runner(directory))
    assert calls == [range(7), range(7, 14), range(14, 21), range(21, 28)]
    saved = _saved(directory)
    assert sum(name.endswith(".npz") for name in saved) == 3

    calls.clear()
    calls.hooks.clear()
    monkeypatch.setenv(bootstrap_chunks.ENV_NO_DRAW, "1")
    with pytest.raises(bootstrap_chunks.DrawRefused, match="21-27"):
        _build(_runner(directory))
    assert calls == []
    assert _saved(directory) == saved


def test_no_draw_guard_lets_a_fully_saved_run_through(
    tmp_path, monkeypatch, calls, unchunked
):
    """A re-classification from complete chunks draws nothing, so the guard allows it."""
    directory = tmp_path / "chunks"
    _build(_runner(directory))
    saved = _saved(directory)
    calls.clear()

    monkeypatch.setenv(bootstrap_chunks.ENV_NO_DRAW, "1")
    runner = _runner(directory)
    rebuilt = _build(runner)
    _same_tables(rebuilt, unchunked)
    assert rebuilt.summary["evidence_availability"]["chunks_present"] is True
    assert calls == []
    assert runner.session_draws == {}
    assert _saved(directory) == saved


@pytest.mark.parametrize(
    ("value", "refuses"),
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("anything", True),
        ("0", False),
        ("false", False),
        ("No", False),
        ("off", False),
        ("", False),
        ("  ", False),
    ],
)
def test_no_draw_guard_reads_its_value(tmp_path, monkeypatch, calls, value, refuses):
    """Anything but an explicit "off" refuses: a safety guard fails closed."""
    monkeypatch.setenv(bootstrap_chunks.ENV_NO_DRAW, value)
    runner = _runner(tmp_path / "chunks")
    if refuses:
        with pytest.raises(bootstrap_chunks.DrawRefused):
            _build(runner)
        assert calls == []
    else:
        _build(runner)
        assert calls  # it drew


def test_the_guard_is_unset_by_default(monkeypatch):
    monkeypatch.delenv(bootstrap_chunks.ENV_NO_DRAW, raising=False)
    assert bootstrap_chunks.no_draw_requested() is False
    monkeypatch.setenv(bootstrap_chunks.ENV_NO_DRAW, "1")
    assert bootstrap_chunks.no_draw_requested() is True


def test_missing_ranges_cut_the_gaps_to_the_chunk_size():
    assert missing_ranges([], 10, 4) == [range(4), range(4, 8), range(8, 10)]
    assert missing_ranges([(0, 4), (6, 8)], 10, 3) == [
        range(4, 6),
        range(8, 10),
    ]
    assert missing_ranges([(0, 10)], 10, 3) == []


# -- refusals and repairs ---------------------------------------------------------


def test_a_different_fingerprint_refuses_to_resume(tmp_path):
    directory = tmp_path / "chunks"
    _runner(directory).store._save_manifest()
    other = {"config": {"n_replicates": N, "seed": 99}}
    with pytest.raises(ValueError, match=r"different config.*--fresh"):
        _store(directory, scientific=other)
    with pytest.raises(ValueError, match="n_draws"):
        ChunkStore(
            directory,
            scientific=SCIENTIFIC,
            n_draws=N + 1,
            seed=CONFIG["seed"],
            chunk_size=SIZE,
        )
    with pytest.raises(ValueError, match=r"seed \(7 stored, 8 asked\)"):
        ChunkStore(
            directory,
            scientific=SCIENTIFIC,
            n_draws=N,
            seed=8,
            chunk_size=SIZE,
        )


def test_an_unreadable_manifest_is_refused(tmp_path):
    directory = tmp_path / "chunks"
    directory.mkdir()
    (directory / "manifest.json").write_text("{not json")
    with pytest.raises(ValueError, match="unreadable"):
        _store(directory)


def test_a_corrupt_chunk_is_recomputed(tmp_path, calls, unchunked):
    directory = tmp_path / "chunks"
    _build(_runner(directory))
    calls.clear()
    victim = directory / "observed-000014-000021.npz"
    victim.write_bytes(victim.read_bytes()[:-20] + b"garbage-garbage-gar")
    with pytest.warns(RuntimeWarning, match="observed-000014-000021.*checksum"):
        resumed = _build(_runner(directory))
    assert calls == [range(14, 21)]
    _same_tables(resumed, unchunked)


def test_a_missing_chunk_file_is_recomputed(tmp_path, calls, unchunked):
    directory = tmp_path / "chunks"
    _build(_runner(directory))
    calls.clear()
    (directory / "flat-000035-000040.npz").unlink()
    with pytest.warns(RuntimeWarning, match="is missing"):
        resumed = _build(_runner(directory))
    assert calls == [range(35, 40)]
    _same_tables(resumed, unchunked)


def test_fresh_moves_the_old_directory_aside(tmp_path):
    directory = tmp_path / "chunks"
    _build(_runner(directory))
    saved = sorted(p.name for p in directory.iterdir())
    store = _store(directory, fresh=True)
    assert store.missing("observed")[0] == range(7)
    assert not (directory / "manifest.json").exists()
    (aside,) = [p for p in tmp_path.iterdir() if ".replaced-" in p.name]
    assert aside.name.startswith("chunks.replaced-")
    kept = sorted(p.name for p in aside.iterdir() if p.name != ".lock")
    assert kept == saved  # never deleted


def test_chunk_files_are_group_readable(tmp_path):
    directory = tmp_path / "chunks"
    _build(_runner(directory))
    files = [p for p in directory.iterdir() if p.is_file()]
    assert {p.name for p in files} >= {"manifest.json", "observed-000000-000007.npz"}
    assert not [p for p in files if p.name.endswith(".tmp")]
    for path in files:
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode & 0o044 == 0o044, (path.name, oct(mode))


# -- signals -----------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="needs POSIX signals")
@pytest.mark.parametrize("number", [signal.SIGTERM, getattr(signal, "SIGUSR1", 0)])
def test_a_stop_signal_saves_the_chunk_and_exits_with_124(
    tmp_path, calls, unchunked, number
):
    directory = tmp_path / "chunks"
    messages: list[str] = []
    calls.hooks.append(lambda count: count == 2 and os.kill(os.getpid(), number))
    runner = ChunkedBootstrap(_store(directory), log=messages.append)
    previous = signal.getsignal(number)
    with pytest.raises(SystemExit) as stopped, runner.stop.installed():
        _build(runner)
    assert isinstance(stopped.value, BootstrapStopped)
    assert stopped.value.code == 124
    assert signal.getsignal(number) == previous  # handlers are restored
    assert any("resumable: 2/12 chunks done" in m for m in messages)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert len(manifest["chunks"]) == 2  # the chunk in progress was saved

    calls.clear()
    calls.hooks.clear()
    _same_tables(_build(_runner(directory)), unchunked)
    assert calls[0] == range(14, 21)


def test_a_stop_during_the_last_chunk_still_finishes(tmp_path, calls, unchunked):
    directory = tmp_path / "chunks"
    last = 2 * len(missing_ranges([], N, SIZE))
    calls.hooks.append(
        lambda count: count == last and os.kill(os.getpid(), signal.SIGTERM)
    )
    runner = _runner(directory)
    with runner.stop.installed():
        _same_tables(_build(runner), unchunked)  # no chunk left: nothing to stop


# -- where chunks live -------------------------------------------------------------


def test_the_environment_variable_sets_the_chunk_base(tmp_path, monkeypatch):
    monkeypatch.setenv(bootstrap_chunks.ENV_CHUNK_DIR, str(tmp_path / "elsewhere"))
    assert chunk_base() == tmp_path / "elsewhere"
    directory = chunk_directory(SCIENTIFIC)
    assert directory.parent == tmp_path / "elsewhere"
    assert len(directory.name) == 16
    # The first 8 hex are the suffix checkpoint_identity gives the same run.
    assert directory.name[:8] == checkpoint_identity.digest8(SCIENTIFIC)
    assert chunk_directory({"config": {"seed": 1}}) != directory

    monkeypatch.delenv(bootstrap_chunks.ENV_CHUNK_DIR)
    monkeypatch.setenv(bootstrap_chunks.ENV_XDG_DATA, str(tmp_path / "xdg"))
    assert chunk_base() == tmp_path / "xdg/fitnesstree/bootstrap"

    monkeypatch.delenv(bootstrap_chunks.ENV_XDG_DATA)
    assert chunk_base() == Path.home() / ".local/share/fitnesstree/bootstrap"


def test_the_stage_saves_chunks_under_its_fingerprint(tmp_path, monkeypatch):
    original = ChunkedBootstrap.run
    settings = _patched_compute(tmp_path, monkeypatch, [])
    monkeypatch.setattr(ChunkedBootstrap, "run", original)
    settings["analysis"][module.SECTION]["chunk_size"] = 7
    module.compute_bootstrap_smoothed_tree(settings)

    (directory,) = (tmp_path / "chunks").iterdir()
    manifest = json.loads((directory / "manifest.json").read_text())
    assert directory.name == chunk_directory(manifest["scientific_fingerprint"]).name
    assert (manifest["n_draws"], manifest["seed"], manifest["chunk_size"]) == (
        20,
        7,
        7,
    )
    assert sorted({c["series"] for c in manifest["chunks"]}) == ["flat", "observed"]
    assert len(manifest["chunks"]) == 6  # 3 chunks of 7, 7 and 6 per series
    assert set(manifest["topographer"]) == {"version", "rev"}
    # The settings that are not in the fingerprint do not name the directory.
    settings["analysis"][module.SECTION]["chunk_size"] = 5
    module.compute_bootstrap_smoothed_tree(settings, force=True)
    assert [p.name for p in (tmp_path / "chunks").iterdir()] == [directory.name]

    module.compute_bootstrap_smoothed_tree(settings, force=True, fresh=True)
    names = sorted(p.name for p in (tmp_path / "chunks").iterdir())
    assert len(names) == 2 and names[1].startswith(f"{directory.name}.replaced-")


def test_chunk_directory_finds_pre_band_independent_manifest_read_only(tmp_path):
    current = {"config": {"seed": 7}}
    legacy = {"config": {"seed": 7, "band_quantile": 0.95}}
    legacy_dir = tmp_path / bootstrap_chunks._chunk_key(legacy)
    legacy_dir.mkdir()
    manifest = legacy_dir / "manifest.json"
    manifest.write_text(json.dumps({"scientific_fingerprint": legacy}))
    original = manifest.read_bytes()

    assert chunk_directory(current, tmp_path) == legacy_dir
    assert manifest.read_bytes() == original


def test_pre_band_quantile_chunks_are_read_only_and_need_no_draw(tmp_path, monkeypatch):
    directory = tmp_path / "legacy"
    _build(_runner(directory))
    manifest = directory / "manifest.json"
    found = json.loads(manifest.read_text())
    found["scientific_fingerprint"]["config"]["band_quantile"] = 0.95
    manifest.write_text(json.dumps(found))
    original = manifest.read_bytes()

    monkeypatch.setenv(bootstrap_chunks.ENV_NO_DRAW, "1")
    monkeypatch.setattr(
        topographer_bootstrap,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected draw")
        ),
    )
    _build(_runner(directory))
    assert manifest.read_bytes() == original


def test_incomplete_legacy_chunks_refuse_before_drawing_and_writing(
    tmp_path, monkeypatch, calls
):
    directory = tmp_path / "legacy-incomplete"
    _build(_runner(directory))
    manifest_path = directory / "manifest.json"
    found = json.loads(manifest_path.read_text())
    found["scientific_fingerprint"]["config"]["band_quantile"] = 0.95
    missing = next(entry for entry in found["chunks"] if entry["series"] == "observed")
    (directory / missing["file"]).unlink()
    found["chunks"].remove(missing)
    manifest_path.write_text(json.dumps(found))
    before = _saved(directory)
    calls.clear()

    with pytest.raises(
        ValueError, match="Legacy bootstrap chunks.*read-only.*incomplete"
    ):
        _build(_runner(directory))

    assert calls == []
    assert _saved(directory) == before


# -- H1: the setup digest ------------------------------------------------------------


def _run(runner, name="observed", **network_options):
    network = _network(**network_options)
    return runner.run(name, network, _statistic(network))


def test_a_changed_se_refuses_before_any_draw(tmp_path, calls):
    directory = tmp_path / "chunks"
    _run(_runner(directory))
    calls.clear()
    with pytest.raises(ValueError, match=r"observed.*differs in se.*--fresh"):
        _run(_runner(directory), se=0.06)
    assert calls == []  # nothing was drawn


def test_changed_values_refuse_before_any_draw(tmp_path, calls):
    directory = tmp_path / "chunks"
    _run(_runner(directory))
    calls.clear()
    moved = [*RAW_FITNESS[:1], 0.55, *RAW_FITNESS[2:]]
    with pytest.raises(ValueError, match=r"observed.*differs in .*values.*--fresh"):
        _run(_runner(directory), fitness=moved)
    assert calls == []


def test_the_digest_is_stored_before_the_first_draw_and_a_consistent_store_resumes(
    tmp_path, calls
):
    directory = tmp_path / "chunks"
    calls.hooks.append(lambda count: count == 2 and (_ for _ in ()).throw(KeyError()))
    with pytest.raises(KeyError):
        _run(_runner(directory))
    manifest = json.loads((directory / "manifest.json").read_text())
    assert set(manifest["setups"]["observed"]) == {
        "fingerprint",
        "values",
        "se",
        "edges",
        "statistic",
    }
    calls.clear()
    calls.hooks.clear()
    _run(_runner(directory))
    assert calls[0] == range(7, 14)  # resumed, not redrawn


def test_series_have_their_own_digests(tmp_path):
    directory = tmp_path / "chunks"
    runner = _runner(directory)
    _run(runner)
    _run(runner, "flat")  # same network, another series: its own digest
    setups = json.loads((directory / "manifest.json").read_text())["setups"]
    assert set(setups) == {"observed", "flat"}
    _run(_runner(directory), "flat")  # and it resumes


def test_chunks_that_disagree_on_their_fingerprint_are_refused(tmp_path, calls):
    directory = tmp_path / "chunks"
    _run(_runner(directory))
    foreign = _chunk(range(14, 21), se=0.09)  # another landscape's chunk
    calls.clear()
    path = directory / "observed-000014-000021.npz"
    foreign.save(path)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["chunks"]:
        if entry["file"] == path.name:
            entry["sha256"] = checkpoint_identity.sha256_of(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"different landscapes.*--fresh"):
        _run(_runner(directory))
    assert calls == []


def test_a_manifest_without_digests_is_refused(tmp_path):
    directory = tmp_path / "chunks"
    _runner(directory).store._save_manifest()
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["format"] = 1
    del manifest["setups"], manifest["topographer_sources"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"unreadable.*--fresh"):
        _store(directory)


# -- H2: the Topographer sources -----------------------------------------------------


@pytest.mark.parametrize("rev", ["abc123", None])
def test_changed_topographer_sources_refuse_to_resume(tmp_path, monkeypatch, rev):
    directory = tmp_path / "chunks"
    identity = {"version": "1.0", "rev": rev}  # an editable install has no rev
    monkeypatch.setattr(bootstrap_chunks, "topographer_identity", lambda: identity)
    monkeypatch.setattr(bootstrap_chunks, "topographer_sources", lambda: {"a.py": "1"})
    _runner(directory).store._save_manifest()
    _store(directory)  # same identity, same sources: resumes

    monkeypatch.setattr(
        bootstrap_chunks, "topographer_sources", lambda: {"a.py": "2", "b.py": "3"}
    )
    with pytest.raises(ValueError, match=r"sources differ.*2 files.*a\.py.*--fresh"):
        _store(directory)
    _store(directory, fresh=True)  # the way out


@pytest.mark.parametrize("package", ["numpy", "rustworkx"])
def test_a_library_version_change_refuses_to_resume(tmp_path, monkeypatch, package):
    directory = tmp_path / "chunks"
    _runner(directory).store._save_manifest()
    _store(directory)  # same versions: resumes
    same = bootstrap_chunks.library_versions()
    monkeypatch.setattr(
        bootstrap_chunks,
        "library_versions",
        lambda: {**same, package: "0.0.1"},
    )
    with pytest.raises(ValueError, match=rf"{package} is 0\.0\.1 now but .*--fresh"):
        _store(directory)
    _store(directory, fresh=True)  # the way out


def test_a_manifest_without_library_versions_is_refused(tmp_path):
    directory = tmp_path / "chunks"
    _runner(directory).store._save_manifest()
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["format"] = 2
    del manifest["libraries"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"unreadable.*--fresh"):
        _store(directory)


# -- H3: signals and dying workers ----------------------------------------------------


def test_sigint_is_a_stop_signal_and_the_log_names_the_signal(tmp_path, calls):
    assert signal.SIGINT in bootstrap_chunks.STOP_SIGNALS
    messages: list[str] = []
    calls.hooks.append(lambda count: count == 1 and os.kill(os.getpid(), signal.SIGINT))
    runner = ChunkedBootstrap(_store(tmp_path / "chunks"), log=messages.append)
    previous = signal.getsignal(signal.SIGINT)
    with pytest.raises(BootstrapStopped), runner.stop.installed():
        _build(runner)
    assert signal.getsignal(signal.SIGINT) == previous
    assert any("stopped by SIGINT" in m for m in messages)


def test_a_dead_worker_during_a_stop_keeps_the_chunks_and_exits_124(tmp_path, calls):
    directory = tmp_path / "chunks"
    messages: list[str] = []
    runner = ChunkedBootstrap(_store(directory), log=messages.append)

    def die(count):
        if count == 3:
            runner.stop.requested, runner.stop.signal = True, signal.SIGTERM
            raise BrokenProcessPool("worker killed by SIGTERM")

    calls.hooks.append(die)
    with pytest.raises(BootstrapStopped) as stopped:
        _build(runner)
    assert stopped.value.code == 124
    assert any("SIGTERM (a worker died)" in m for m in messages)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert len(manifest["chunks"]) == 2  # the saved ones stay


def test_a_dead_worker_without_a_stop_is_a_real_error(tmp_path, calls):
    calls.hooks.append(lambda count: (_ for _ in ()).throw(BrokenProcessPool("boom")))
    with pytest.raises(BrokenProcessPool):
        _build(_runner(tmp_path / "chunks"))


# -- M3: --fresh implies force --------------------------------------------------------


def test_fresh_implies_force(tmp_path, monkeypatch):
    builds: list = []
    settings = _patched_compute(tmp_path, monkeypatch, builds)
    module.compute_bootstrap_smoothed_tree(settings)
    module.compute_bootstrap_smoothed_tree(settings)
    assert len(builds) == 1  # a current checkpoint is reused
    module.compute_bootstrap_smoothed_tree(settings, fresh=True)
    assert len(builds) == 2  # --fresh alone recomputes it


# -- L1: the lock ---------------------------------------------------------------------


def test_the_store_lock_excludes_a_second_run_and_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_identity, "LOCK_TIMEOUT", 0.3)
    directory = tmp_path / "chunks"
    with _store(directory, hold_lock=True):
        with pytest.raises(RuntimeError, match="locked by another process"):
            _store(directory, hold_lock=True)
        with pytest.raises(RuntimeError, match="locked by another process"):
            _store(directory, fresh=True)
        assert (directory / ".lock").exists()
    _store(directory, hold_lock=True).close()  # released


def test_a_lock_on_a_directory_moved_aside_is_retaken(tmp_path, monkeypatch):
    import fcntl

    directory = tmp_path / "chunks"
    directory.mkdir()
    real = fcntl.flock
    calls = []

    def flock_then_move(fd, operation):
        real(fd, operation)
        calls.append(operation)
        if len(calls) == 1:  # another --fresh moves the directory aside meanwhile
            directory.rename(tmp_path / "chunks.replaced-x")
            directory.mkdir()

    monkeypatch.setattr(fcntl, "flock", flock_then_move)
    with bootstrap_chunks.directory_lock(directory) as held:
        assert held
        assert (directory / ".lock").exists()  # the current directory's lock file
    assert sum(c == fcntl.LOCK_EX | fcntl.LOCK_NB for c in calls) == 2  # retried


def test_a_filesystem_without_flock_only_warns(tmp_path, monkeypatch):
    import fcntl

    def refuse(*args):
        raise OSError(errno.ENOSYS, "no locks here")

    monkeypatch.setattr(fcntl, "flock", refuse)
    with pytest.warns(RuntimeWarning, match="working without a lock"):
        _store(tmp_path / "chunks", hold_lock=True).close()


# -- L3, L4: unreadable and flaky chunks ------------------------------------------------


def test_an_unreadable_chunk_file_is_recomputed(
    tmp_path, monkeypatch, calls, unchunked
):
    directory = tmp_path / "chunks"
    _build(_runner(directory))
    calls.clear()
    original = checkpoint_identity.sha256_of

    failed: list[str] = []

    def flaky(path):
        if path.name == "observed-000007-000014.npz" and not failed:
            failed.append(path.name)  # only the check of the saved chunk fails
            raise PermissionError(errno.EACCES, "denied")
        return original(path)

    monkeypatch.setattr(checkpoint_identity, "sha256_of", flaky)
    with pytest.warns(RuntimeWarning, match="observed-000007-000014.*cannot be read"):
        resumed = _build(_runner(directory))
    assert calls == [range(7, 14)]
    _same_tables(resumed, unchunked)


def test_a_chunk_that_fails_its_final_load_is_recomputed(
    tmp_path, monkeypatch, calls, unchunked
):
    directory = tmp_path / "chunks"
    _build(_runner(directory))
    calls.clear()
    seen: dict[str, int] = {}
    original = bootstrap_chunks.BootstrapChunk.load

    def load(path):
        seen[Path(path).name] = count = seen.get(Path(path).name, 0) + 1
        if Path(path).name == "observed-000014-000021.npz" and count == 2:
            raise OSError(
                errno.EIO, "disk hiccup"
            )  # passes verify, fails the final load
        return original(path)

    monkeypatch.setattr(bootstrap_chunks.BootstrapChunk, "load", staticmethod(load))
    with pytest.warns(RuntimeWarning, match="observed-000014-000021.*cannot be read"):
        resumed = _build(_runner(directory))
    assert calls == [range(14, 21)]
    _same_tables(resumed, unchunked)


# -- L6, L7 ---------------------------------------------------------------------------


def test_seconds_per_draw_count_only_this_sessions_draws(tmp_path):
    directory = tmp_path / "chunks"
    first = _build(_runner(directory))
    assert first.summary["timing"]["real_seconds_per_draw"] > 0
    second = _build(_runner(directory))  # every draw resumed
    assert second.summary["timing"]["real_seconds_per_draw"] is None
    assert second.summary["timing"]["flat_seconds_per_draw"] is None


def test_series_names_are_matched_with_their_separator():
    store = _store(None)
    store.add("observed", _chunk())
    assert store.done_chunks("observed") == 1
    assert store.done_chunks("obs") == 0
    assert store.load("obs") == []
    assert len(store.load("observed")) == 1


def test_the_noise_model_decides_the_chunk_directory_and_the_dataset_name(tmp_path):
    """A pooled run never shares chunks or a result-dataset name with a replicate-SE run."""
    from dubanevics2026topological.simplification import (
        bootstrap_chunks,
        checkpoint_identity,
        bootstrap_smoothed_tree,
    )

    config = {"n_replicates": 10000, "seed": 20261019}
    pooled = checkpoint_identity.scientific_fingerprint(
        {"config": {**config, "noise_model": bootstrap_smoothed_tree.NOISE_MODEL}}
    )
    replicate = checkpoint_identity.scientific_fingerprint({"config": config})
    assert pooled["config"]["noise_model"] == "pooled"
    assert bootstrap_chunks.chunk_directory(
        pooled, tmp_path
    ) != bootstrap_chunks.chunk_directory(replicate, tmp_path)
    # fingerprint8 of the result dataset is the digest of the same fingerprint.
    assert checkpoint_identity.digest8(pooled) != checkpoint_identity.digest8(replicate)
