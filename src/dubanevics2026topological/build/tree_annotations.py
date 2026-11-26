"""Publish the existing biological description for an identified nominal tree."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..analysis.describe import describe
from ..landscape.products import _publish, _sha256
from ..results import ALLOWED_KEYS, RunContext, load

_LABEL_KEYS = frozenset(ALLOWED_KEYS) - {"host", "run", "hash", "style"}


def build_tree_annotations(
    tree_analysis: Any, *, tree: Any, layout_label: str, context: RunContext
) -> Path:
    """Save unchanged annotation tables after the layout stage validates the tree."""
    if (
        tree.product_type != "tree"
        or tree.report.status != "full"
        or not tree.report.byte_integrity
        or tree.manifest.scientific_identity.get("tree") != "nominal"
    ):
        raise ValueError("Annotations require a fully verified nominal tree.")
    layout = load("tree_layout", layout_label, context=context)
    tree_id = tree.manifest.scientific_id_sha256
    if (
        layout.report.status != "full"
        or not layout.report.byte_integrity
        or layout.manifest.scientific_identity.get("tree_scientific_id_sha256")
        != tree_id
    ):
        raise ValueError("Annotations require a verified layout for the same tree.")
    annotations = describe(
        products={"annotations"},
        landscape=SimpleNamespace(trees={"split": tree_analysis}),
        landscape_name="candidate_clamped",
    ).annotations
    params = {
        key: value
        for key, value in tree.manifest.scientific_identity.items()
        if key in _LABEL_KEYS
    }
    code_paths = (
        Path(inspect.getfile(describe))
        .resolve()
        .with_name("biological_annotations.py"),
        Path(inspect.getfile(describe)).resolve(),
        Path(__file__).resolve(),
    )
    identity = {
        **params,
        "tree_scientific_id_sha256": tree_id,
        "annotation_code_digest": {
            str(index): _sha256(path) for index, path in enumerate(code_paths)
        },
    }
    parent = {
        "product_type": "tree",
        "label": tree.label,
        "scientific_id_sha256": tree_id,
    }

    def write(directory: Path) -> None:
        for name in ("branches", "pairs", "critical_points"):
            getattr(annotations, name).write_parquet(directory / f"{name}.parquet")
        (directory / "summary.json").write_text(
            json.dumps(annotations.summary, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    path, _ = _publish(
        "tree_annotations",
        params,
        identity,
        context=context,
        input_data_sha256s={},
        parents=(parent,),
        code_paths=code_paths,
        write_artifacts=write,
    )
    return path
