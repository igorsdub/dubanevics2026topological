"""Publish one ``landscape_summary`` product per state from saved products only."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..analysis.describe import SUMMARY_COLUMNS, describe, landscape_summary_column
from ..landscape.products import _publish, _sha256
from ..results import ALLOWED_KEYS, RunContext
from . import landscape_summary_inputs
from .landscape_summary_inputs import Loader, load_landscape_summary_inputs
from ..results.processed import LEAVES

_LABEL_KEYS = frozenset(ALLOWED_KEYS) - {"host", "run", "hash", "style"}
SUMMARY_FILE = "summary.json"


def _reference(product: Any) -> dict[str, str]:
    return {
        "product_type": product.product_type,
        "label": product.label,
        "scientific_id_sha256": product.manifest.scientific_id_sha256,
    }


def build_landscape_summary_products(
    context: RunContext,
    *,
    labels: Mapping[str, str] = LEAVES,
    loader: Loader | None = None,
) -> dict[str, Path]:
    """Save each state's summary column under the label keys of its own tree.

    The seven states are loaded together and summarised through ``describe`` so
    the cross-state checks of the seven-column summary still run; each column is
    then published separately because one label carries one tree's keys.
    """
    loaded = load_landscape_summary_inputs(context, labels=labels, loader=loader)
    columns = describe(
        products={"landscape_summary"},
        landscape_inputs=loaded.inputs,
        full_fitness=loaded.full_fitness,
    ).landscape_summary
    full_landscape = loaded.landscapes[SUMMARY_COLUMNS[0]]
    code_paths = (
        Path(inspect.getfile(landscape_summary_column)).resolve(),
        Path(inspect.getfile(describe)).resolve(),
        Path(landscape_summary_inputs.__file__).resolve(),
        Path(__file__).resolve(),
    )
    outputs: dict[str, Path] = {}
    for name in SUMMARY_COLUMNS:
        tree = loaded.trees[name]
        params = {
            key: value
            for key, value in tree.manifest.scientific_identity.items()
            if key in _LABEL_KEYS
        }
        parents = [
            _reference(loaded.leaves[name]),
            _reference(tree),
            _reference(loaded.landscapes[name]),
        ]
        if name != SUMMARY_COLUMNS[0]:
            parents.append(_reference(full_landscape))
        classification = loaded.classification_sha256.get(name)
        identity = {
            **params,
            "column": name,
            "code_digest": {str(i): _sha256(path) for i, path in enumerate(code_paths)},
        }
        if classification is not None:
            identity["classification_table_sha256"] = classification
        # Metric order is the table's row order, so it is stored as a list.
        rows = [{"metric": m, "value": v} for m, v in columns[name].items()]

        def write(directory: Path, rows: list[dict[str, Any]] = rows) -> None:
            (directory / SUMMARY_FILE).write_text(
                json.dumps(rows, allow_nan=False) + "\n", encoding="utf-8"
            )

        outputs[name], _ = _publish(
            "landscape_summary",
            params,
            identity,
            context=context,
            input_data_sha256s=(
                {}
                if classification is None
                else {"classification_table": classification}
            ),
            parents=parents,
            code_paths=code_paths,
            write_artifacts=write,
        )
    return outputs
