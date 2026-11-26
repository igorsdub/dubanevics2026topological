"""Validate the exclusion tables of config/figures.toml."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

TABLE = "bootstrap_identity_exclusions"
MISSING_PRODUCT_TABLE = "missing_product_exclusions"


def bootstrap_identity_exclusions(
    figures_settings: Mapping[str, Any],
    allowed: Collection[str],
    table_name: str = TABLE,
) -> dict[str, str]:
    """Figure id -> reason; a malformed table is an error, never a silent no-op."""
    table = figures_settings.get(table_name, {})
    if not isinstance(table, Mapping):
        raise ValueError(f"[{table_name}] must be a table of figure id = reason.")
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise ValueError(
            f"[{table_name}] names {unknown}, which this table cannot exclude; "
            f"allowed: {sorted(allowed)}."
        )
    for figure_id, reason in table.items():
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"[{table_name}] {figure_id} needs a non-empty string reason.")
    return dict(table)
