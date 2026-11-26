from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dubanevics2026topological.results import processed
from dubanevics2026topological.results.context import RunContext
from dubanevics2026topological.results.hashing import compute_scientific_hash
from dubanevics2026topological.results.lookup import MissingProductError
from dubanevics2026topological.results.manifest import Manifest, write_manifest


def _product(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "data.txt").write_bytes(b"hello")
    identity = {"thr": "reflected"}
    digest, _ = compute_scientific_hash("tree", identity)
    write_manifest(
        Manifest(
            product_type="tree",
            scientific_identity=identity,
            scientific_id_sha256=digest,
            parents=[],
            data_provenance={},
            artifacts={
                "data.txt": {
                    "sha256": hashlib.sha256(b"hello").hexdigest(),
                    "size_bytes": 5,
                }
            },
            run_fields={},
            code_provenance={},
            environment={},
            created_at="2026-01-01T00:00:00Z",
            evidence_availability={
                "checkpoint_present": True,
                "distances_present": False,
                "chunks_present": False,
            },
        ),
        directory,
    )


def _context(root: Path) -> RunContext:
    return RunContext(root=root, output_root=root / "output", env={})


def test_states_are_the_seven_reviewed_states() -> None:
    assert len(processed.STATES) == 7
    assert processed.STATES[0] == "full"


def test_path_is_kind_and_state_under_root(tmp_path: Path) -> None:
    assert processed.path("tree", "full", root=tmp_path) == tmp_path / "tree" / "full"
    assert processed.path("growth_rates", root=tmp_path) == tmp_path / "growth_rates"
    assert processed.path("tree", "full") == processed.PROCESSED_ROOT / "tree" / "full"


def test_load_verifies_a_processed_product(tmp_path: Path) -> None:
    _product(tmp_path / "data" / "processed" / "tree" / "reflected")
    context = _context(tmp_path)
    loaded = processed.load("tree", "reflected", context)
    assert loaded.report.status == "full"
    assert loaded.path.name == "reflected"


def test_load_missing_names_the_recipes(tmp_path: Path) -> None:
    with pytest.raises(MissingProductError, match="just all.*just fetch-data"):
        processed.load("tree", "full", _context(tmp_path))


def test_load_rejects_a_modified_artifact(tmp_path: Path) -> None:
    directory = tmp_path / "data" / "processed" / "tree" / "reflected"
    _product(directory)
    (directory / "data.txt").write_bytes(b"HELLO")
    with pytest.raises(MissingProductError, match="failed verification"):
        processed.load("tree", "reflected", _context(tmp_path))


def _tree(root: Path) -> None:
    (root / "a").mkdir(parents=True)
    (root / "a" / "x.txt").write_text("x")
    (root / "y.txt").write_text("yy")


def test_manifest_roundtrip_and_header(tmp_path: Path) -> None:
    _tree(tmp_path)
    target = processed.write_manifest(tmp_path)
    lines = target.read_text().splitlines()
    assert lines[0] == "path\tbytes\tsha256"
    assert [line.split("\t")[0] for line in lines[1:]] == ["a/x.txt", "y.txt"]
    assert processed.check_manifest(tmp_path) == []


def test_check_manifest_reports_missing_extra_size_and_hash(tmp_path: Path) -> None:
    _tree(tmp_path)
    (tmp_path / "z.txt").write_text("zz")
    processed.write_manifest(tmp_path)
    (tmp_path / "z.txt").unlink()
    (tmp_path / "new.txt").write_text("n")
    (tmp_path / "y.txt").write_text("y")  # size change
    (tmp_path / "a" / "x.txt").write_text("q")  # same size, different hash
    assert processed.check_manifest(tmp_path) == [
        "missing: z.txt",
        "extra: new.txt",
        "hash: a/x.txt",
        "size: y.txt",
    ]


def test_check_manifest_without_manifest_names_the_recipes(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="just fetch-data"):
        processed.check_manifest(tmp_path)


def test_cli_check_exit_codes(tmp_path: Path) -> None:
    _tree(tmp_path)
    assert processed.main(["check", "--root", str(tmp_path)]) == 1
    assert processed.main(["write", "--root", str(tmp_path)]) == 0
    assert processed.main(["check", "--root", str(tmp_path)]) == 0


def test_publish_replaces_the_target(tmp_path: Path) -> None:
    source = tmp_path / "src"
    _tree(source)
    root = tmp_path / "processed"
    processed.publish(source, "leaves", "full", root=root)
    (source / "y.txt").write_text("changed")
    target = processed.publish(source, "leaves", "full", root=root)
    assert (target / "y.txt").read_text() == "changed"


def test_state_for_selection_names_the_processed_state() -> None:
    by = {"rule": "fdr_fixed_cutoff", "method": "by", "q": 0.05}
    assert (
        processed.state_for_selection({**by, "cutoff": "median_reflected"}, True)
        == "reflected_by"
    )
    assert (
        processed.state_for_selection({**by, "cutoff": "papkou_original"})
        == "published_by"
    )
    assert processed.state_for_selection({"rule": "median_reflected"}) == "reflected"
    assert processed.state_for_selection({"rule": "mixture_quantile"}) == "legacy_mixture"
    with pytest.raises(ValueError, match="band tree"):
        processed.state_for_selection({"rule": "median_reflected"}, True)
    with pytest.raises(ValueError, match="No processed"):
        processed.state_for_selection({"rule": "unknown"})


def test_lookup_scans_the_processed_root(tmp_path: Path) -> None:
    from dubanevics2026topological.results import lookup

    _product(tmp_path / "data" / "processed" / "tree" / "reflected")
    found = lookup.load("tree", context=_context(tmp_path))
    assert found.path.name == "reflected"
    assert found.report.status == "full"


def test_product_loader_reads_leaves_by_state(tmp_path: Path) -> None:
    _product(tmp_path / "data" / "processed" / "leaves" / "full")
    loader = processed.product_loader(_context(tmp_path))
    assert loader("leaves", "full").path.name == "full"


def _write_output_product(context: RunContext, kind: str, directory: str, identity: dict, created: str, *, variant: str | None = None) -> Path:
    root = context.output_root if variant is None else context.output_root / "variants" / variant
    target = root / kind / directory
    target.mkdir(parents=True)
    (target / "data.txt").write_text(directory)
    digest, _ = compute_scientific_hash(kind, identity)
    write_manifest(
        Manifest(
            product_type=kind,
            scientific_identity=identity,
            scientific_id_sha256=digest,
            parents=[],
            data_provenance={},
            artifacts={
                "data.txt": {
                    "sha256": hashlib.sha256(directory.encode()).hexdigest(),
                    "size_bytes": len(directory),
                }
            },
            run_fields={},
            code_provenance={},
            environment={},
            created_at=created,
            evidence_availability={
                "checkpoint_present": False,
                "distances_present": False,
                "chunks_present": False,
            },
        ),
        target,
        check_collision=False,
    )
    return target


@pytest.mark.parametrize(
    ("identity", "state"),
    [
        ({"graph": "full", "tree": "nominal"}, "full"),
        ({"thr": "published", "fdr": "none", "tree": "nominal"}, "published"),
        ({"thr": "reflected", "fdr": "none"}, "reflected"),
        ({"thr": "published", "fdr": "by", "q": 0.05}, "published_by"),
        ({"thr": "reflected", "fdr": "by", "tree": "band", "bandq": 0.95}, "reflected_by_band"),
    ],
)
def test_state_of_identity(identity, state) -> None:
    assert processed.state_of_identity(identity) == state


def test_state_of_identity_refuses_an_unknown_selection() -> None:
    with pytest.raises(ValueError, match="No processed state"):
        processed.state_of_identity({"thr": "mixture", "fdr": "none"})


def test_collect_names_products_by_state_and_keeps_the_newest(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _write_output_product(context, "tree", "old", {"thr": "reflected", "fdr": "by", "q": 0.05, "tree": "nominal", "v": 1}, "2026-01-01T00:00:00Z")
    _write_output_product(context, "tree", "new", {"thr": "reflected", "fdr": "by", "q": 0.05, "tree": "nominal", "v": 2}, "2026-02-01T00:00:00Z")
    _write_output_product(context, "tree", "full", {"graph": "full", "tree": "nominal"}, "2026-01-01T00:00:00Z")
    written = processed.collect(context)
    assert {path.relative_to(context.processed_root).as_posix() for path in written} == {
        "tree/reflected_by",
        "tree/full",
    }
    assert (context.processed_root / "tree" / "reflected_by" / "data.txt").read_text() == "new"
    assert processed.load("tree", "full", context).report.status == "full"


def test_collect_names_variant_products_by_their_variant_root(tmp_path: Path) -> None:
    context = _context(tmp_path)
    identity = {"thr": "published", "fdr": "none"}
    _write_output_product(context, "pooled_scatter", "a", identity, "2026-01-01T00:00:00Z", variant="threshold=reflected_fdr=by_q=0.05")
    _write_output_product(context, "pooled_scatter", "b", identity, "2026-01-01T00:00:00Z")
    written = {p.relative_to(context.processed_root).as_posix() for p in processed.collect(context)}
    # The default root and the variant that builds the same graph keep separate folders.
    assert written == {"pooled_scatter/default", "pooled_scatter/reflected_by"}


def test_collect_and_restore_plain_stage_outputs(tmp_path: Path) -> None:
    context = _context(tmp_path)
    growth = context.output_root / "analysis" / "sequence_growth_rates"
    growth.mkdir(parents=True)
    (growth / "growth_rates.parquet").write_bytes(b"rates")
    tables = context.output_root / "tables"
    tables.mkdir()
    (tables / "papkou2023_codon_entropy_metadata.json").write_text("{}")
    (tables / "unrelated.csv").write_text("x")
    processed.collect(context)
    assert (context.processed_root / "growth_rates" / "growth_rates.parquet").read_bytes() == b"rates"
    assert (context.processed_root / "codon_entropy" / "papkou2023_codon_entropy_metadata.json").is_file()
    assert not (context.processed_root / "codon_entropy" / "unrelated.csv").exists()

    import shutil

    shutil.rmtree(context.output_root)
    restored = processed.restore(context)
    assert (growth / "growth_rates.parquet").read_bytes() == b"rates"
    assert (tables / "papkou2023_codon_entropy_metadata.json").is_file()
    assert processed.restore(context) == []  # identical files are left alone
    assert len(restored) == 2


def test_cli_collect_and_restore(tmp_path: Path, monkeypatch) -> None:
    if not (Path(__file__).resolve().parents[1] / "data/raw/papkou2023_fitness_wt.parquet").is_file():
        pytest.skip("needs the gitignored raw tables under data/raw/ (absent in a clean public clone)")
    monkeypatch.chdir(tmp_path)
    assert processed.main(["collect", "--root", str(tmp_path / "p")]) == 0
    assert processed.main(["restore", "--root", str(tmp_path / "p")]) == 0


def test_manifest_lists_dotfiles_and_files_behind_symlinked_directories(tmp_path: Path) -> None:
    _tree(tmp_path)
    (tmp_path / ".hidden.json").write_text("{}")
    real = tmp_path.parent / (tmp_path.name + "-real")
    real.mkdir()
    (real / "inside.txt").write_text("i")
    (tmp_path / "linked").symlink_to(real)
    processed.write_manifest(tmp_path)
    listed = (tmp_path / "MANIFEST.tsv").read_text()
    assert ".hidden.json" in listed and "linked/inside.txt" in listed
    (tmp_path / ".hidden.json").write_text("changed")
    assert processed.check_manifest(tmp_path) == ["size: .hidden.json"]
    (tmp_path / ".extra").write_text("e")
    assert "extra: .extra" in processed.check_manifest(tmp_path)


def test_collect_skips_identities_outside_the_seven_states_with_a_warning(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _write_output_product(context, "tree", "mixture", {"thr": "mixture", "fdr": "none"}, "2026-01-01T00:00:00Z")
    _write_output_product(context, "tree", "full", {"graph": "full", "tree": "nominal"}, "2026-01-01T00:00:00Z")
    with pytest.warns(UserWarning, match="Not collected"):
        written = processed.collect(context)
    assert {p.relative_to(context.processed_root).as_posix() for p in written} == {"tree/full"}
