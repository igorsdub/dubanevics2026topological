"""Result label grammar, formatting, and parsing per ADR 0016 and Spec 1.

Grammar Rules:
- Format: ``<name>__<params>`` where ``__`` separates product/view name from parameters.
- Parameters: ``key=value`` pairs joined by single underscores ``_``.
- Pipeline data-flow order:
  ``graph`` -> ``thr`` -> ``fdr`` -> ``q`` -> ``p`` -> ``draws`` -> ``seed`` -> ``tree`` ->
  ``bandq`` -> ``host`` -> ``run`` -> ``hash`` (+ ``style`` for figures).
- Allowed keys: ``graph``, ``thr``, ``fdr``, ``q``, ``p``, ``draws``, ``seed``, ``tree``,
  ``bandq``, ``host``, ``run``, ``hash``, ``style``.
- No ``_`` or ``-`` inside keys or values, except a leading minus on numbers.
- Numbers appear strictly after ``=``, never following ``-``.
- Required keys per standard product type:
  - ``fitness``: ``thr=<published|reflected|mixture>_fdr=<none|by>[_q=<q>][_p=<p>]``
  - ``landscape``: ``graph=full`` OR fitness keys
  - ``tree`` nominal: landscape keys + ``tree=nominal``
  - ``bootstrap``: landscape keys + ``draws=<n>_seed=<s>``
  - ``tree`` band: landscape keys + ``draws=<n>_seed=<s>_tree=band_bandq=<x>``
  - ``leaves``, ``landscape_summary``, ``ridges``, ``ridge_entropy_sweeps``, ``merge_flow``: corresponding tree keys
- ``graph`` only ever takes ``full``.
- ``fdr=none`` is always written explicitly when no multiple testing adjustment is applied.
"""

from __future__ import annotations

import decimal
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

ALLOWED_KEYS: tuple[str, ...] = (
    "graph",
    "thr",
    "fdr",
    "q",
    "p",
    "draws",
    "seed",
    "tree",
    "bandq",
    "host",
    "run",
    "hash",
    "style",
)

PIPELINE_KEY_ORDER: tuple[str, ...] = ALLOWED_KEYS

STANDARD_PRODUCT_TYPES: tuple[str, ...] = (
    "growth_rates",
    "pooled_scatter",
    "threshold_diagnostics",
    "fitness",
    "landscape",
    "tree",
    "tree_layout",
    "tree_annotations",
    "bootstrap",
    "leaves",
    "landscape_summary",
    "ridges",
    "ridge_entropy_sweeps",
    "merge_flow",
)

ALLOWED_THR_VALUES: frozenset[str] = frozenset({"published", "reflected", "mixture"})
ALLOWED_FDR_VALUES: frozenset[str] = frozenset({"none", "by"})
ALLOWED_TREE_VALUES: frozenset[str] = frozenset({"nominal", "band"})
ALLOWED_HOST_VALUES: frozenset[str] = frozenset({"local"})
ALLOWED_STYLE_VALUES: frozenset[str] = frozenset({"annotated", "vanilla"})

_HEX_8_PATTERN = re.compile(r"^[0-9a-f]{8}$")


class MalformedLabelError(ValueError):
    """Raised when a result label violates ADR 0016 grammar or contracts."""


def _is_numeric(val: str) -> bool:
    """Check if string is a valid finite integer or float."""
    try:
        f = float(val)
        return math.isfinite(f)
    except ValueError:
        return False


def _parse_typed_value(key: str, val_str: str) -> Any:
    """Convert raw string parameter to a typed Python object."""
    if key in ("draws", "seed"):
        return int(val_str)
    if key in ("q", "p", "bandq"):
        f = float(val_str)
        if not math.isfinite(f):
            raise MalformedLabelError(
                f"Non-finite numeric value for '{key}': '{val_str}'"
            )
        return f
    if key == "run":
        return int(val_str) if val_str.isdigit() else val_str
    return val_str


def _format_value(key: str, val: Any) -> str:
    """Format a typed Python value to its canonical label string."""
    if isinstance(val, float):
        if not math.isfinite(val):
            raise MalformedLabelError(
                f"Non-finite float value '{val}' is not allowed for key '{key}'"
            )
        # Shortest round-trip fixed-point decimal without scientific notation
        s = repr(val)
        if "e" in s or "E" in s:
            d = decimal.Decimal(s)
            return format(d, "f")
        return s
    return str(val)


def _validate_token_characters(key: str, val_str: str) -> None:
    """Enforce character restrictions on keys and values."""
    if not (key.isalpha() and key.islower()):
        raise MalformedLabelError(
            f"Key '{key}' must be strictly lowercase alphabetic without digits, "
            f"underscores, or hyphens."
        )

    val_lower = val_str.lower()
    if val_lower in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity"):
        raise MalformedLabelError(
            f"Non-finite numeric value for '{key}' is forbidden: '{val_str}'"
        )

    if "_" in val_str:
        raise MalformedLabelError(
            f"Value for '{key}' cannot contain underscores: '{val_str}'"
        )

    if "-" in val_str:
        if not val_str.startswith("-"):
            raise MalformedLabelError(
                f"Hyphen in value for '{key}' is only permitted as a leading minus "
                f"on numbers: '{val_str}'"
            )
        remainder = val_str[1:]
        if not _is_numeric(remainder):
            raise MalformedLabelError(
                f"Leading minus on '{key}' value is only permitted on numeric values: "
                f"'{val_str}'"
            )

    # Numbers only after '=':
    # Digits are only allowed in numeric parameters, run IDs, or the 8-hex hash
    allowed_numeric_keys = {"q", "p", "bandq", "draws", "seed", "run", "hash"}
    has_digits = any(c.isdigit() for c in val_str)
    if has_digits:
        if key in allowed_numeric_keys:
            pass
        else:
            raise MalformedLabelError(
                f"Parameter '{key}' cannot contain digits in value '{val_str}'. "
                f"Numbers appear strictly after '=' on numeric fields."
            )

    # Specific key validations
    if key in ("q", "p", "bandq") and not _is_numeric(val_str):
        raise MalformedLabelError(
            f"Value for numeric parameter '{key}' must be a finite number, got '{val_str}'"
        )

    # Specific key validations
    if key == "hash" and not _HEX_8_PATTERN.match(val_str):
        raise MalformedLabelError(
            f"'hash' must be exactly 8 lowercase hexadecimal characters, got '{val_str}'"
        )
    if key == "graph" and val_str != "full":
        raise MalformedLabelError(
            f"'graph' only ever takes value 'full', got '{val_str}'"
        )
    if key == "fdr" and val_str not in ALLOWED_FDR_VALUES:
        raise MalformedLabelError(
            f"'fdr' must be one of {sorted(ALLOWED_FDR_VALUES)}, got '{val_str}'"
        )
    if key == "thr" and val_str not in ALLOWED_THR_VALUES:
        raise MalformedLabelError(
            f"'thr' must be one of {sorted(ALLOWED_THR_VALUES)}, got '{val_str}'"
        )
    if key == "tree" and val_str not in ALLOWED_TREE_VALUES:
        raise MalformedLabelError(
            f"'tree' must be one of {sorted(ALLOWED_TREE_VALUES)}, got '{val_str}'"
        )
    if key == "host" and val_str not in ALLOWED_HOST_VALUES:
        raise MalformedLabelError(
            f"'host' must be one of {sorted(ALLOWED_HOST_VALUES)}, got '{val_str}'"
        )
    if key == "style" and val_str not in ALLOWED_STYLE_VALUES:
        raise MalformedLabelError(
            f"'style' must be one of {sorted(ALLOWED_STYLE_VALUES)}, got '{val_str}'"
        )


def _validate_product_contracts(
    product_name: str, raw_params: Mapping[str, str]
) -> None:
    """Validate that parameters satisfy required and forbidden keys per product type."""
    # Check host/run pairing
    has_host = "host" in raw_params
    has_run = "run" in raw_params
    if has_host != has_run:
        raise MalformedLabelError(
            f"Run fields 'host' and 'run' must both be present if either is specified, "
            f"got host={has_host}, run={has_run}"
        )

    # Style is only allowed for figures, never standard data products
    if product_name in STANDARD_PRODUCT_TYPES and "style" in raw_params:
        raise MalformedLabelError(
            f"Visual 'style' parameter is not permitted on scientific data product "
            f"'{product_name}'"
        )

    # If style is present, it must be after hash
    if "style" in raw_params and "hash" not in raw_params:
        raise MalformedLabelError("'style' must appear after 'hash' in figure labels")

    # Standard data product rules
    if product_name not in STANDARD_PRODUCT_TYPES:
        return

    if product_name == "growth_rates":
        if raw_params.get("thr") != "published" or raw_params.get("fdr") != "none":
            raise MalformedLabelError(
                "Product 'growth_rates' requires 'thr=published_fdr=none'"
            )
        if set(raw_params) - {"thr", "fdr", "hash", "host", "run"}:
            raise MalformedLabelError(
                "Product 'growth_rates' has unsupported parameters"
            )
        return

    if product_name == "pooled_scatter":
        if raw_params.get("thr") != "published" or raw_params.get("fdr") != "none":
            raise MalformedLabelError(
                "Product 'pooled_scatter' requires 'thr=published_fdr=none'"
            )
        if set(raw_params) - {"thr", "fdr", "hash", "host", "run"}:
            raise MalformedLabelError(
                "Product 'pooled_scatter' has unsupported parameters"
            )
        return

    if product_name == "threshold_diagnostics":
        if raw_params.get("graph") != "full":
            raise MalformedLabelError(
                "Product 'threshold_diagnostics' requires 'graph=full'"
            )
        if set(raw_params) - {"graph", "hash", "host", "run"}:
            raise MalformedLabelError(
                "Product 'threshold_diagnostics' has unsupported parameters"
            )
        return

    # Check fitness keys
    def check_fitness_keys() -> None:
        if "thr" not in raw_params:
            raise MalformedLabelError(
                f"Product '{product_name}' requires threshold key 'thr'"
            )
        if "fdr" not in raw_params:
            raise MalformedLabelError(
                f"Product '{product_name}' requires multiple testing key 'fdr'"
            )
        fdr_val = raw_params["fdr"]
        if fdr_val == "by" and "q" not in raw_params:
            raise MalformedLabelError(
                f"Product '{product_name}' with 'fdr=by' requires nominal level 'q'"
            )
        if fdr_val == "none" and "q" in raw_params:
            raise MalformedLabelError(
                f"Product '{product_name}' with 'fdr=none' must omit 'q'"
            )

        thr_val = raw_params["thr"]
        if thr_val == "mixture":
            if "p" not in raw_params:
                raise MalformedLabelError(
                    f"Product '{product_name}' with 'thr=mixture' requires mixture probability 'p'"
                )
        elif thr_val in ("published", "reflected"):
            if "p" in raw_params:
                raise MalformedLabelError(
                    f"Parameter 'p' is forbidden when thr='{thr_val}'"
                )
        elif "p" in raw_params:
            raise MalformedLabelError(
                "Parameter 'p' is only permitted when thr='mixture'"
            )

    # Check landscape keys
    def check_landscape_keys() -> None:
        if "graph" in raw_params:
            if "thr" in raw_params or "fdr" in raw_params or "p" in raw_params:
                raise MalformedLabelError(
                    f"Product '{product_name}' cannot combine 'graph=full' with "
                    f"threshold/fdr/mixture keys"
                )
        else:
            check_fitness_keys()

    if product_name == "fitness":
        check_fitness_keys()
        forbidden = {"graph", "draws", "seed", "tree", "bandq"}
        for k in forbidden:
            if k in raw_params:
                raise MalformedLabelError(
                    f"Product 'fitness' must not contain key '{k}'"
                )

    elif product_name == "landscape":
        check_landscape_keys()
        forbidden = {"draws", "seed", "tree", "bandq"}
        for k in forbidden:
            if k in raw_params:
                raise MalformedLabelError(
                    f"Product 'landscape' must not contain key '{k}'"
                )

    elif product_name == "bootstrap":
        check_landscape_keys()
        if "draws" not in raw_params or "seed" not in raw_params:
            raise MalformedLabelError(
                "Product 'bootstrap' requires 'draws' and 'seed' keys"
            )
        forbidden = {"tree", "bandq"}
        for k in forbidden:
            if k in raw_params:
                raise MalformedLabelError(
                    f"Product 'bootstrap' must not contain key '{k}' (draw identity only)"
                )

    elif product_name in (
        "tree",
        "tree_layout",
        "tree_annotations",
        "leaves",
        "landscape_summary",
        "ridges",
        "ridge_entropy_sweeps",
        "merge_flow",
    ):
        check_landscape_keys()
        if "tree" not in raw_params:
            raise MalformedLabelError(
                f"Product '{product_name}' requires 'tree' key ('nominal' or 'band')"
            )
        tree_val = raw_params["tree"]
        if tree_val == "nominal":
            forbidden = {"draws", "seed", "bandq"}
            for k in forbidden:
                if k in raw_params:
                    raise MalformedLabelError(
                        f"Nominal '{product_name}' must not contain key '{k}'"
                    )
        elif tree_val == "band":
            if "draws" not in raw_params or "seed" not in raw_params:
                raise MalformedLabelError(
                    f"Band '{product_name}' requires upstream bootstrap keys 'draws' and 'seed'"
                )
            if "bandq" not in raw_params:
                raise MalformedLabelError(
                    f"Band '{product_name}' requires persistence band quantile 'bandq'"
                )


@dataclass(frozen=True)
class ResultLabel:
    """Parsed DrWatson-style result label per ADR 0016."""

    product_name: str
    params: dict[str, Any]
    raw_params: dict[str, str]

    @property
    def short_hash(self) -> str | None:
        """First 8 hex characters of scientific hash, if present."""
        return self.raw_params.get("hash")

    @property
    def host(self) -> str | None:
        """Execution host ('local'), if present."""
        return self.raw_params.get("host")

    @property
    def run(self) -> Any | None:
        """Execution run ID, if present."""
        return self.params.get("run")

    @property
    def style(self) -> str | None:
        """Figure visual style ('annotated' or 'vanilla'), if present."""
        return self.raw_params.get("style")

    @property
    def scientific_params(self) -> dict[str, Any]:
        """Parameters strictly contributing to scientific identity (excluding run/style/hash)."""
        excluded = {"host", "run", "hash", "style"}
        return {k: v for k, v in self.params.items() if k not in excluded}

    def format(self, validate: bool = True) -> str:
        """Format to canonical string."""
        return format_label(self.product_name, self.params, validate=validate)


def parse_label(label_str: str, validate: bool = True) -> ResultLabel:
    """Parse and validate a result label per ADR 0016 grammar."""
    if not isinstance(label_str, str):
        raise MalformedLabelError(f"Label must be a string, got {type(label_str)}")

    if "__" not in label_str:
        raise MalformedLabelError(
            f"Label must contain '__' separating product/view name from parameters: '{label_str}'"
        )

    parts = label_str.split("__")
    if len(parts) != 2:
        raise MalformedLabelError(
            f"Label must contain exactly one '__' separator, got {len(parts) - 1}: '{label_str}'"
        )

    product_name, param_str = parts[0], parts[1]
    if not product_name:
        raise MalformedLabelError(f"Empty product/view name in label: '{label_str}'")
    if not param_str:
        raise MalformedLabelError(f"Empty parameters in label: '{label_str}'")

    tokens = param_str.split("_")
    raw_params: dict[str, str] = {}
    typed_params: dict[str, Any] = {}

    prev_order_idx = -1

    for token in tokens:
        if "=" not in token:
            raise MalformedLabelError(
                f"Invalid parameter token '{token}' in '{label_str}' (missing '=')"
            )
        subparts = token.split("=")
        if len(subparts) != 2:
            raise MalformedLabelError(
                f"Multiple '=' characters in token '{token}' in '{label_str}'"
            )
        key, val_str = subparts[0], subparts[1]

        if not key:
            raise MalformedLabelError(f"Empty key in token '{token}' in '{label_str}'")
        if not val_str:
            raise MalformedLabelError(f"Empty value for key '{key}' in '{label_str}'")

        if key not in ALLOWED_KEYS:
            raise MalformedLabelError(
                f"Unknown parameter key '{key}' in '{label_str}'. "
                f"Allowed keys: {', '.join(ALLOWED_KEYS)}"
            )

        if key in raw_params:
            raise MalformedLabelError(
                f"Duplicate parameter key '{key}' in label '{label_str}'"
            )

        _validate_token_characters(key, val_str)

        # Enforce pipeline data-flow order
        curr_order_idx = PIPELINE_KEY_ORDER.index(key)
        if curr_order_idx < prev_order_idx:
            raise MalformedLabelError(
                f"Parameter key '{key}' is out of pipeline data-flow order in '{label_str}'. "
                f"Expected order: {', '.join(PIPELINE_KEY_ORDER)}"
            )
        prev_order_idx = curr_order_idx

        raw_params[key] = val_str
        typed_params[key] = _parse_typed_value(key, val_str)

    if validate:
        _validate_product_contracts(product_name, raw_params)

    return ResultLabel(
        product_name=product_name,
        params=typed_params,
        raw_params=raw_params,
    )


def format_label(
    product_name: str, params: Mapping[str, Any], validate: bool = True
) -> str:
    """Format product name and parameter mapping into a canonical result label."""
    if not product_name:
        raise MalformedLabelError("Product name cannot be empty")

    raw_tokens: list[tuple[str, str]] = []
    raw_params: dict[str, str] = {}

    for key in PIPELINE_KEY_ORDER:
        if key in params:
            val = params[key]
            val_str = _format_value(key, val)
            _validate_token_characters(key, val_str)
            raw_tokens.append((key, val_str))
            raw_params[key] = val_str

    # Check for any extra unrecognised keys in params
    for key in params:
        if key not in ALLOWED_KEYS:
            raise MalformedLabelError(
                f"Unknown parameter key '{key}'. Allowed keys: {', '.join(ALLOWED_KEYS)}"
            )

    if not raw_tokens:
        raise MalformedLabelError(
            f"No valid parameters provided for product '{product_name}'"
        )

    if validate:
        _validate_product_contracts(product_name, raw_params)

    param_str = "_".join(f"{k}={v}" for k, v in raw_tokens)
    return f"{product_name}__{param_str}"
