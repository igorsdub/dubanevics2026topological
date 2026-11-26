"""Selection guards for the saved Figure 6 products."""

import pytest

from dubanevics2026topological.build.reflected_fdr_inputs import validate_selection


@pytest.mark.parametrize(
    "state,identity",
    [
        ("reflected", {"thr": "reflected", "fdr": "none", "tree": "nominal"}),
        ("reflected_by", {"thr": "reflected", "fdr": "by", "q": 0.05, "tree": "nominal"}),
    ],
)
def test_nominal_selection_is_accepted(state, identity):
    validate_selection(state, identity)


@pytest.mark.parametrize("change", [{"tree": "band"}, {"thr": "published"}, {"q": 0.1}, {"fdr": "none"}])
def test_wrong_by_selection_is_rejected(change):
    identity = {"thr": "reflected", "fdr": "by", "q": 0.05, "tree": "nominal", **change}
    with pytest.raises(ValueError, match="reflected_by"):
        validate_selection("reflected_by", identity)


def test_corrupt_classification_snapshot_is_rejected(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import polars as pl
    from dubanevics2026topological.build import reflected_fdr_inputs as module

    diagnostic = tmp_path / "diagnostics.json"
    diagnostic.write_text(json.dumps({"threshold": -0.25, "functional_set_sha256": "set", "selected_component_vertex_count": 1}))
    snapshot = tmp_path / "classes.parquet"
    snapshot.write_bytes(b"corrupt")
    population = pl.DataFrame({"dna_sequence": ["a"], "scalar": [0.1]})

    def tree_for(state, context):
        identity = {"thr": "reflected", "fdr": "by" if state == "reflected_by" else "none", "tree": "nominal", "landscape_selection": {"threshold": -0.25, "functional_set_sha256": "set"}}
        if state == "reflected_by":
            identity["q"] = 0.05
        return SimpleNamespace(manifest=SimpleNamespace(scientific_identity=identity, scientific_id_sha256=state))

    def landscape_for(products, tree):
        identity = {key: value for key, value in tree.manifest.scientific_identity.items() if key != "tree"}
        return SimpleNamespace(manifest=SimpleNamespace(scientific_identity=identity, scientific_id_sha256="landscape"), artifacts={"diagnostics.json": diagnostic})

    monkeypatch.setattr(module, "PresentationProducts", lambda context: SimpleNamespace(landscape_population=lambda landscape: population))
    monkeypatch.setattr(module, "_tree_for_leaf", tree_for)
    monkeypatch.setattr(module, "_landscape_for_tree", landscape_for)
    monkeypatch.setattr(module, "_layout", lambda products, tree: (object(), "layout"))
    monkeypatch.setattr(module, "_snapshot", lambda *args: (snapshot, "0" * 64))
    with pytest.raises(ValueError, match="classification snapshot.*mismatched"):
        module.load_reflected_fdr_inputs(object())