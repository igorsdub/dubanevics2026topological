"""The per-variant peak analysis, pooled_scatter and panel summary of the four-graph figure.

The stages themselves need the Papkou data and the private packages, so the collaborators
are replaced by recorders here; what is tested is the variant parameterisation (which
declaration, which root, which context), the additive guarantee for the one variant that
already existed, the summary's cross-checks and the figure-side loader.
"""

from __future__ import annotations

import json
import tomllib
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological.analysis import ROOT, load_settings
from dubanevics2026topological.build import FDR_POOLED_VARIANT, peak_panels
from dubanevics2026topological.figures.bootstrap_four_graphs import PANEL_ORDER
from dubanevics2026topological.figures.peak_uncertainty import PANEL_VARIANTS
from dubanevics2026topological.results import MissingProductError, RunContext

VARIANTS = tuple(PANEL_VARIANTS)


def _base(tmp_path: Path | None = None, **kwargs) -> RunContext:
    return RunContext(output_root=tmp_path, env={}, **kwargs)


# -- what each variant is ----------------------------------------------------------


def test_the_four_variants_are_the_ones_of_the_bootstrap_four_graphs_in_the_same_order() -> None:
    assert VARIANTS == tuple(name for name, _ in PANEL_ORDER)
    assert set(peak_panels.PEAK_PANEL_RULES) == set(VARIANTS)


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_variant_declares_what_its_name_says(variant: str) -> None:
    context = _base(variant_name=variant)
    settings = load_settings(context=context)
    declaration = peak_panels.require_declaration(variant, settings)
    expected = peak_panels.PEAK_PANEL_RULES[variant]
    assert {key: declaration.get(key) for key in expected} == expected
    # The declaration comes from the variant's own overlay, not from the main config.
    overlay = tomllib.loads((ROOT / "config/variants" / f"{variant}.toml").read_text())
    assert declaration == overlay["clamped"]["candidate_functional_threshold"]


def test_a_variant_that_declares_another_rule_stops_the_stage() -> None:
    settings = load_settings(context=_base(variant_name="threshold=published_fdr=none"))
    with pytest.raises(ValueError, match="threshold=reflected_fdr=none"):
        peak_panels.require_declaration("threshold=reflected_fdr=none", settings)
    with pytest.raises(ValueError, match="threshold=published_fdr=by_q=0.05"):
        peak_panels.require_declaration("threshold=published_fdr=by_q=0.05", settings)


def test_the_by_variants_differ_only_in_their_cutoff_and_the_point_ones_have_no_fdr() -> None:
    rules = peak_panels.PEAK_PANEL_RULES
    assert rules["threshold=published_fdr=none"]["method"] is None
    assert rules["threshold=reflected_fdr=none"]["q"] is None
    by = [rules["threshold=published_fdr=by_q=0.05"], rules["threshold=reflected_fdr=by_q=0.05"]]
    assert {(r["rule"], r["method"], r["q"]) for r in by} == {("fdr_fixed_cutoff", "by", 0.05)}
    assert {r["cutoff"] for r in by} == {"papkou_original", "median_reflected"}


# -- contexts and the additive guarantee --------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_variant_root_is_under_variants_with_the_variants_own_settings(
    tmp_path: Path, variant: str
) -> None:
    base = _base(tmp_path, host="cluster", run_id="9")
    context = peak_panels.variant_context(base, variant)
    assert context.output_root == (tmp_path / "variants" / variant).resolve()
    assert context.variant_name == variant
    for field in ("processed_root", "raw_data_dir", "host", "run_id", "root"):
        assert getattr(context, field) == getattr(base, field), field


def test_the_existing_variant_gets_exactly_the_context_the_old_stage_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``fdr-pooled-scatter-product`` builds under env variant + ``output/variants/<v>``.

    The peak checkpoint fingerprint and the pooled_scatter identity are functions of the
    settings and the context only, so an equal context keeps both unchanged: the new
    stage reuses the product the old one wrote instead of writing a second one.
    """
    # ``dataclasses.replace`` re-reads the process environment for the fields that are
    # None, as the old stage's ``RunContext()`` does, so the comparison needs one
    # environment: the cluster's own (barseq export, chunk dir, N_JOBS) would otherwise
    # sit on one side only.
    for name in (
        "FITNESSTREE_OUTPUT_ROOT", "OUTPUT", "FITNESSTREE_ANALYSIS_VARIANT",
        "FITNESSTREE_BOOTSTRAP_CHUNK_DIR", "FITNESSTREE_N_JOBS", "PAPKOU2023_BARSEQ_EXPORT",
    ):
        monkeypatch.delenv(name, raising=False)
    old = RunContext(
        output_root=f"output/variants/{FDR_POOLED_VARIANT}",
        variant_name=FDR_POOLED_VARIANT,
    )
    new = peak_panels.variant_context(RunContext(), FDR_POOLED_VARIANT)
    assert new == old


def test_an_unknown_variant_has_no_context(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown peak panel variant"):
        peak_panels.variant_context(_base(tmp_path), "nominal")


def test_the_reader_context_is_the_variant_context(tmp_path: Path) -> None:
    base = _base(tmp_path)
    reader = peak_panels.reader_context(base, VARIANTS[0])
    assert reader == peak_panels.variant_context(base, VARIANTS[0])


def test_the_stage_edits_no_file_the_existing_identities_hash() -> None:
    """Additive by construction: the new module is a caller of the unchanged builders.

    The peak checkpoint fingerprint hashes ``analysis/peak_uncertainty.py`` and the files
    ``peak_stage`` lists; the pooled_scatter identity hashes its parent, tolerances and the
    peak table bytes. None of that is in ``peak_panels.py``, which only calls them.
    """
    source = (ROOT / "src/dubanevics2026topological/build/peak_panels.py").read_text()
    peak_source = (ROOT / "src/dubanevics2026topological/analysis/peak_uncertainty.py").read_text()
    assert "peak_panels" not in peak_source
    assert "load_or_compute_peak_uncertainty(settings=settings, context=context)" in source
    assert "build_pooled(context=context)" in source
    # No second pipeline: nothing here opens a checkpoint writer, a fingerprint or the
    # pooled_scatter publisher itself.
    for forbidden in ('"pooled_scatter",\n        {}', "publish_checkpoint", "stage_fingerprint"):
        assert forbidden not in source


# -- the stage, with collaborators recorded ------------------------------------------


@pytest.fixture
def stage(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        peak_panels,
        "load_or_compute_peak_uncertainty",
        lambda **kw: calls.append(("peak", kw["context"].variant_name, kw["settings"])) or {},
    )
    monkeypatch.setattr(peak_panels, "_fingerprint", lambda settings, **kw: "fp")
    pooled = SimpleNamespace(manifest=SimpleNamespace(scientific_id_sha256="a" * 64))
    monkeypatch.setattr(
        peak_panels,
        "load_pooled_scatter_product",
        lambda context, **kw: calls.append(("load", context.variant_name, kw)) or pooled,
    )
    monkeypatch.setattr(
        peak_panels,
        "build_peak_panel_summary",
        lambda context, settings, product: calls.append(("summary", context.variant_name, product))
        or Path(f"/summary/{context.variant_name}"),
    )
    return calls, pooled


def _build_pooled(calls):
    def build(*, context):
        calls.append(("pooled", context.variant_name, context.output_root))
        return Path("/pooled")

    return build


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_variant_runs_peak_then_pooled_then_summary_under_its_own_root(
    tmp_path: Path, stage, variant: str
) -> None:
    calls, pooled = stage
    context = peak_panels.variant_context(_base(tmp_path), variant)
    path = peak_panels.build_variant_panel(context, _build_pooled(calls))
    assert path == Path(f"/summary/{variant}")
    assert [call[:2] for call in calls] == [
        ("peak", variant),
        ("pooled", variant),
        ("load", variant),
        ("summary", variant),
    ]
    # The unchanged builders get the variant's own settings and root.
    declared = peak_panels.PEAK_PANEL_RULES[variant]["rule"]
    assert calls[0][2]["analysis"]["clamped"]["candidate_functional_threshold"]["rule"] == declared
    assert calls[1][2] == (tmp_path / "variants" / variant).resolve()
    tolerances = load_settings(context=context)["analysis"]
    assert calls[2][2] == {
        "expected": {
            "identity_tolerance": float(tolerances["sequence_growth_rates"]["identity_tolerance"]),
            "standard_error_tolerance": float(
                tolerances["peak_uncertainty_scatter"]["standard_error_tolerance"]
            ),
        },
        "expected_checkpoint_fingerprint": "fp",
    }
    assert calls[3][2] is pooled


def test_every_panel_variant_is_built_in_figure_order_and_a_subset_can_be_chosen(
    tmp_path: Path, stage
) -> None:
    calls, _ = stage
    built = peak_panels.build_peak_panels(_base(tmp_path), _build_pooled(calls))
    assert list(built) == list(VARIANTS)
    assert [call[1] for call in calls if call[0] == "summary"] == list(VARIANTS)
    calls.clear()
    only = peak_panels.build_peak_panels(
        _base(tmp_path), _build_pooled(calls), variants=("threshold=reflected_fdr=none",)
    )
    assert list(only) == ["threshold=reflected_fdr=none"]
    assert {call[1] for call in calls} == {"threshold=reflected_fdr=none"}


# -- the sequence counts --------------------------------------------------------------


class _Graph:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads

    def node_indices(self):
        return range(len(self.payloads))

    def __getitem__(self, index: int) -> dict:
        return self.payloads[index]


def _prepared(raw: dict[str, float], threshold: float, *, functional: int | None = None):
    payloads = [{"dna_sequence": name, "raw_fitness": value} for name, value in raw.items()]
    counted = (
        sum(value >= threshold for value in raw.values()) if functional is None else functional
    )
    return SimpleNamespace(
        network=SimpleNamespace(graph=_Graph(payloads), scalar_attr="raw_fitness"),
        threshold=threshold,
        diagnostics={
            "selected_component_vertex_count": len(raw),
            "selected_component_functional_vertex_count": counted,
            "peak_count": 2,
        },
    )


def test_a_scalar_selection_counts_functional_at_or_above_the_threshold_and_has_no_uncertain() -> None:
    prepared = _prepared({"a": 0.0, "b": -0.2, "c": -0.5, "d": -0.7}, -0.5)
    counts = peak_panels.selected_sequence_counts(prepared, SimpleNamespace(classes=None))
    assert counts == {"functional": 3, "non_functional": 1, "uncertain": None, "vertices": 4}


def test_a_classification_counts_each_class_of_the_selected_vertices_only() -> None:
    prepared = _prepared({"a": 0.0, "b": -0.2, "c": -0.5, "d": -0.7}, -0.5, functional=2)
    # The classification names a sequence outside the selected graph too: not counted.
    classes = {"a": "functional", "b": "functional", "c": "uncertain", "d": "non_functional", "z": "uncertain"}
    counts = peak_panels.selected_sequence_counts(prepared, SimpleNamespace(classes=classes))
    assert counts == {"functional": 2, "non_functional": 1, "uncertain": 1, "vertices": 4}


def test_counts_that_disagree_with_the_landscape_diagnostics_stop_the_stage() -> None:
    prepared = _prepared({"a": 0.0, "b": -0.7}, -0.5, functional=2)
    with pytest.raises(ValueError, match="Functional vertex count"):
        peak_panels.selected_sequence_counts(prepared, SimpleNamespace(classes=None))
    prepared = _prepared({"a": 0.0, "b": -0.7}, -0.5)
    prepared.diagnostics["selected_component_vertex_count"] = 3
    with pytest.raises(ValueError, match="diagnostics say 3"):
        peak_panels.selected_sequence_counts(prepared, SimpleNamespace(classes=None))


# -- the summary product ---------------------------------------------------------------


@pytest.fixture
def summary_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    published: dict = {}

    @contextmanager
    def no_lock(*args, **kwargs):
        yield

    monkeypatch.setattr(peak_panels, "checkpoint_lock", no_lock)
    monkeypatch.setattr(peak_panels, "cache_paths", lambda context: (tmp_path / "peak", tmp_path / "pending"))
    monkeypatch.setattr(peak_panels, "_peak_inputs", lambda context: {"tables/x.parquet": "hash"})
    monkeypatch.setattr(
        peak_panels,
        "selection_identity",
        lambda selection: {"rule": selection.rule, "threshold": selection.threshold.threshold},
    )

    def publish(product_type, base_params, identity, **kwargs):
        published.update(product_type=product_type, base=base_params, identity=identity, **kwargs)
        directory = tmp_path / "published"
        directory.mkdir(exist_ok=True)
        kwargs["write_artifacts"](directory)
        return directory, None

    monkeypatch.setattr(peak_panels, "_publish", publish)

    def no_saved_summary(*args, **kwargs):
        raise MissingProductError("no saved summary")

    monkeypatch.setattr(peak_panels, "load", no_saved_summary)

    scatter = tmp_path / "candidate_clamped_scatter.parquet"
    pl.DataFrame({"is_peak": [True, True, False]}).write_parquet(scatter)
    pooled_diagnostics = tmp_path / "diagnostics.json"
    pooled_diagnostics.write_text(
        json.dumps({"thresholds": {"candidate_clamped": -0.5}, "wild_type_sequence": "wt"})
    )
    pooled = SimpleNamespace(
        label="pooled_scatter_x",
        manifest=SimpleNamespace(scientific_id_sha256="b" * 64),
        artifacts={
            "candidate_clamped_scatter.parquet": scatter,
            "diagnostics.json": pooled_diagnostics,
        },
    )

    def configure(
        variant: str, *, classes, peaks=2, threshold=-0.5, checkpoint_peaks=2, functional=None,
        study_peaks=2,
    ):
        monkeypatch.setattr(
            peak_panels,
            "_load_peak_checkpoint",
            lambda directory, context=None: {
                "provenance": {"fingerprint": "fp"},
                "diagnostics": {
                    "peak_counts": {
                        "candidate_clamped": checkpoint_peaks,
                        "study_clamped": study_peaks,
                    },
                    "clamped_thresholds": {"study_clamped": -0.5077, "candidate_clamped": -0.508},
                },
            },
        )
        prepared = _prepared({"a": 0.0, "b": -0.2, "c": -0.5, "d": -0.7}, -0.5, functional=functional)
        prepared.diagnostics["peak_count"] = peaks
        selection = SimpleNamespace(
            rule="x", classes=classes, threshold=SimpleNamespace(threshold=threshold)
        )
        monkeypatch.setattr(peak_panels, "_prepare_selected_graph", lambda settings, context: (prepared, selection))
        context = peak_panels.variant_context(_base(tmp_path), variant)
        return context, load_settings(context=context)

    return configure, pooled, published, tmp_path


def _diagnostics(directory: Path) -> dict:
    return json.loads((directory / "published/diagnostics.json").read_text())


def test_the_summary_of_a_point_variant_has_no_uncertain_class_and_no_fdr(summary_env) -> None:
    configure, pooled, published, tmp = summary_env
    context, settings = configure("threshold=reflected_fdr=none", classes=None)
    peak_panels.build_peak_panel_summary(context, settings, pooled)
    diagnostics = _diagnostics(tmp)
    assert diagnostics == {
        "variant": "threshold=reflected_fdr=none",
        "graph": "candidate_clamped",
        "rule": "median_reflected",
        "cutoff": None,
        "fdr_method": None,
        "fdr_q": None,
        "threshold": -0.5,
        "peak_count": 2,
        "functional_sequence_count": 3,
        "nonfunctional_sequence_count": 1,
        "uncertain_sequence_count": None,
        "largest_component_vertex_count": 4,
        "wild_type_sequence": "wt",
    }
    assert published["product_type"] == "peak_panel_summary"
    assert published["identity"] == {
        "variant": "threshold=reflected_fdr=none",
        "graph": "candidate_clamped",
        "selection": {"rule": "x", "threshold": -0.5},
        "peak_checkpoint_fingerprint": "fp",
        "pooled_scatter_id": "b" * 64,
    }
    assert published["input_data_sha256s"] == {"tables/x.parquet": "hash"}
    (parent,) = published["parents"]
    assert parent["product_type"] == "pooled_scatter"
    assert parent["scientific_id_sha256"] == "b" * 64


def test_the_summary_of_a_by_variant_carries_the_fdr_setting_and_the_uncertain_count(summary_env) -> None:
    configure, pooled, published, tmp = summary_env
    classes = {"a": "functional", "b": "functional", "c": "uncertain", "d": "non_functional"}
    # Under a classification the landscape's functional count is that of the Functional class.
    context, settings = configure("threshold=published_fdr=by_q=0.05", classes=classes, functional=2)
    peak_panels.build_peak_panel_summary(context, settings, pooled)
    diagnostics = _diagnostics(tmp)
    assert (diagnostics["fdr_method"], diagnostics["fdr_q"]) == ("by", 0.05)
    assert diagnostics["cutoff"] == "papkou_original"
    assert (
        diagnostics["functional_sequence_count"],
        diagnostics["uncertain_sequence_count"],
        diagnostics["nonfunctional_sequence_count"],
    ) == (2, 1, 1)


def test_a_peak_count_that_differs_between_the_scatter_the_checkpoint_and_the_landscape_stops_the_stage(
    summary_env,
) -> None:
    configure, pooled, published, _ = summary_env
    for kwargs in ({"checkpoint_peaks": 3}, {"peaks": 5}):
        context, settings = configure("threshold=reflected_fdr=none", classes=None, **kwargs)
        with pytest.raises(ValueError, match="Peak counts of threshold=reflected_fdr=none disagree"):
            peak_panels.build_peak_panel_summary(context, settings, pooled)
    assert published == {}


def test_a_threshold_that_differs_from_the_scatter_stops_the_stage(summary_env) -> None:
    configure, pooled, published, _ = summary_env
    context, settings = configure("threshold=reflected_fdr=none", classes=None, threshold=-0.4)
    with pytest.raises(ValueError, match="Threshold of threshold=reflected_fdr=none differs"):
        peak_panels.build_peak_panel_summary(context, settings, pooled)
    assert published == {}


def test_the_published_graph_peak_count_must_equal_the_candidate_count_of_the_papkou_point_variant(
    summary_env,
) -> None:
    configure, pooled, published, tmp = summary_env
    context, settings = configure("threshold=published_fdr=none", classes=None)
    peak_panels.build_peak_panel_summary(context, settings, pooled)
    assert _diagnostics(tmp)["peak_count"] == 2
    published.clear()
    context, settings = configure("threshold=published_fdr=none", classes=None, study_peaks=3)
    with pytest.raises(ValueError, match=r"has 3 peaks at cutoff -0\.5077.*has 2 peaks at cutoff -0\.5"):
        peak_panels.build_peak_panel_summary(context, settings, pooled)
    assert published == {}


def test_a_saved_summary_of_the_same_identity_is_returned_without_building_the_graph(
    summary_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure, pooled, published, tmp = summary_env
    context, settings = configure("threshold=reflected_fdr=none", classes=None)
    saved = SimpleNamespace(
        path=tmp / "saved_summary",
        report=SimpleNamespace(status="full", byte_integrity=True),
        manifest=SimpleNamespace(parents=[{"scientific_id_sha256": "b" * 64}]),
    )
    monkeypatch.setattr(peak_panels, "load", lambda *args, **kwargs: saved)

    def builder(settings, context):
        raise AssertionError("the selected graph must not be rebuilt")

    monkeypatch.setattr(peak_panels, "_prepare_selected_graph", builder)
    assert peak_panels.build_peak_panel_summary(context, settings, pooled) == saved.path
    assert published == {}


# -- the figure-side loader -------------------------------------------------------------


@pytest.fixture
def loader_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seen: dict = {"pooled": [], "summary": []}

    @contextmanager
    def no_lock(*args, **kwargs):
        yield

    monkeypatch.setattr(peak_panels, "checkpoint_lock", no_lock)
    monkeypatch.setattr(peak_panels, "cache_paths", lambda context: (tmp_path / "peak", tmp_path / "p"))
    monkeypatch.setattr(peak_panels, "_peak_inputs", lambda context: {"hash": context.variant_name})
    monkeypatch.setattr(peak_panels, "_fingerprint", lambda settings, **kw: f"fp-{kw['context'].variant_name}")

    pooled: dict[str, SimpleNamespace] = {}
    summaries: dict[str, SimpleNamespace] = {}
    for index, variant in enumerate(VARIANTS):
        scatter = tmp_path / f"{variant}.parquet"
        pl.DataFrame({"dna_sequence": ["wt", f"x{index}"], "is_peak": [False, True]}).write_parquet(scatter)
        pooled_id = f"{index}" * 64
        pooled[variant] = SimpleNamespace(
            manifest=SimpleNamespace(scientific_id_sha256=pooled_id),
            artifacts={"candidate_clamped_scatter.parquet": scatter},
        )
        diagnostics = tmp_path / f"{variant}.json"
        diagnostics.write_text(json.dumps({"variant": variant, "peak_count": index}))
        summaries[variant] = SimpleNamespace(
            report=SimpleNamespace(status="full", byte_integrity=True),
            manifest=SimpleNamespace(
                scientific_id_sha256=f"{index + 4}" * 64,
                parents=[{"scientific_id_sha256": pooled_id}],
            ),
            artifacts={"diagnostics.json": diagnostics},
        )

    def load_pooled(context, **kwargs):
        seen["pooled"].append((context, kwargs))
        return pooled[context.variant_name]

    def load(product, *, context, expected):
        assert product == "peak_panel_summary"
        seen["summary"].append((context, expected))
        return summaries[context.variant_name]

    monkeypatch.setattr(peak_panels, "load_pooled_scatter_product", load_pooled)
    monkeypatch.setattr(peak_panels, "load", load)
    settings = load_settings(context=_base())
    return seen, pooled, summaries, settings, tmp_path


def test_the_loader_reads_each_variant_from_its_own_root_and_checks_the_identity(loader_env) -> None:
    seen, pooled, summaries, settings, tmp = loader_env
    panels, ids = peak_panels.load_peak_panels(_base(tmp), settings)
    assert list(panels) == list(VARIANTS) and list(ids) == list(VARIANTS)
    for index, variant in enumerate(VARIANTS):
        assert panels[variant]["diagnostics"] == {"variant": variant, "peak_count": index}
        assert panels[variant]["scatter"]["dna_sequence"].to_list() == ["wt", f"x{index}"]
        assert ids[variant] == f"{index + 4}" * 64
    roots = [context.output_root for context, _ in seen["summary"]]
    assert roots == [(tmp / "variants" / v).resolve() for v in VARIANTS]
    assert len(set(roots)) == 4
    for (context, expected), variant in zip(seen["summary"], VARIANTS, strict=True):
        assert expected["scientific_identity"] == {
            "variant": variant,
            "graph": "candidate_clamped",
            "peak_checkpoint_fingerprint": f"fp-{variant}",
            "pooled_scatter_id": pooled[variant].manifest.scientific_id_sha256,
        }
        assert expected["data_provenance"] == {"hash": variant}
    for (context, kwargs), variant in zip(seen["pooled"], VARIANTS, strict=True):
        assert kwargs["expected_checkpoint_fingerprint"] == f"fp-{variant}"


def test_a_missing_or_stale_product_names_the_variant_and_the_stage(loader_env, monkeypatch) -> None:
    _, _, _, settings, tmp = loader_env

    def stale(product, *, context, expected):
        raise MissingProductError("no match")

    monkeypatch.setattr(
        peak_panels,
        "load_pooled_scatter_product",
        lambda context, **kw: SimpleNamespace(manifest=SimpleNamespace(scientific_id_sha256="a" * 64)),
    )
    monkeypatch.setattr(peak_panels, "load", stale)
    with pytest.raises(MissingProductError, match="threshold=published_fdr=none") as error:
        peak_panels.load_peak_panels(_base(tmp), settings)
    assert "just peak-panels-product" in str(error.value)


def test_a_summary_that_does_not_descend_from_its_pooled_scatter_is_refused(loader_env) -> None:
    _, _, summaries, settings, tmp = loader_env
    summaries["threshold=published_fdr=none"].manifest.parents = [{"scientific_id_sha256": "f" * 64}]
    with pytest.raises(ValueError, match="does not descend from its pooled_scatter"):
        peak_panels.load_peak_panels(_base(tmp), settings)


def test_a_summary_that_is_not_fully_verified_is_refused(loader_env) -> None:
    _, _, summaries, settings, tmp = loader_env
    summaries["threshold=reflected_fdr=none"].report = SimpleNamespace(status="partial", byte_integrity=True)
    with pytest.raises(MissingProductError, match="not fully verified") as error:
        peak_panels.load_peak_panels(_base(tmp), settings)
    message = str(error.value)
    assert "threshold=reflected_fdr=none" in message and "stale or damaged" in message
    assert "just peak-panels-product" in message and "Missing or stale" not in message


def test_the_builder_rebuilds_on_an_unverified_summary_instead_of_failing(monkeypatch) -> None:
    class Rebuild(Exception):
        pass

    def unverified(*args, **kwargs):
        raise peak_panels.UnverifiedPeakPanelError("not fully verified")

    def proceed(*args, **kwargs):
        raise Rebuild

    monkeypatch.setattr(peak_panels, "require_declaration", lambda variant, settings: {"rule": "r"})
    monkeypatch.setattr(peak_panels, "cache_paths", lambda context: (Path("."), None))
    monkeypatch.setattr(peak_panels, "checkpoint_lock", lambda *a, **k: nullcontext())
    monkeypatch.setattr(
        peak_panels, "_load_peak_checkpoint", lambda *a, **k: {"provenance": {"fingerprint": "fp"}}
    )
    monkeypatch.setattr(peak_panels, "_peak_inputs", lambda context: {})
    monkeypatch.setattr(peak_panels, "variant_reader", lambda context: context)
    monkeypatch.setattr(peak_panels, "_verified_summary", unverified)
    monkeypatch.setattr(peak_panels, "_prepare_selected_graph", proceed)
    context = SimpleNamespace(variant_name="threshold=reflected_fdr=none")
    with pytest.raises(Rebuild):
        peak_panels.build_peak_panel_summary(context, {}, SimpleNamespace())
