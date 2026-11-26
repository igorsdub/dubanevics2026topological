"""Story-view catalog and the figure file naming grammar (Spec 2, Sections 2 and 6).

Single-result figures are ``<view>__<label>_style=<annotated|vanilla>.<ext>``;
comparison figures are ``<view>_<comparison>__hash=<8hex>.<ext>`` with a JSON
sidecar. Names are built with ``results.labels.format_label``, so key order,
character rules and the 8-hex hash are the result-label grammar's own; style is
appended after ``hash`` and never enters a scientific hash. Pure naming: nothing
here reads a product or the filesystem except the sidecar writer.

Interpretations of the spec, recorded here:

- Bootstrap-draw and band key sets carry ``thr``/``fdr`` and no ``graph=full``:
  the bootstrap exists only for the BY graphs (every spec example is of that
  form), so there is no full-graph bootstrap name.
- A ``tree`` value follows the key set: ``tree=nominal`` without ``bandq``,
  ``tree=band`` with it.
- The view ``leaves`` shares its name with a data product, for which the result-label
  grammar forbids ``style``. Figure names therefore skip the product-contract check
  (``validate=False``: key order, character rules and the hash format still apply)
  and use this module's per-view key sets instead.
- Comparison names (``<comparison>``) are "defined in configuration"; they are
  restricted to lowercase letters and digits (no underscore) so the
  ``<view>_<comparison>__hash=`` form parses unambiguously.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..results.hashing import canonical_json
from ..results.labels import (
    ALLOWED_STYLE_VALUES,
    MalformedLabelError,
    format_label,
    parse_label,
)

_THR_FDR = frozenset({"thr", "fdr"})
_DRAWS = frozenset({"draws", "seed"})
_GRAPH = frozenset({"graph"})
_NOMINAL_KEYS = (frozenset({"graph", "tree"}), _THR_FDR | {"tree"})
_LANDSCAPE_KEYS = (_GRAPH, _THR_FDR)
_BAND_KEYS = (_THR_FDR | _DRAWS | {"tree", "bandq"},)
_DRAWS_KEYS = (_THR_FDR | _DRAWS,)


@dataclass(frozen=True)
class D10View:
    """One view of the catalog.

    ``key_sets`` are the allowed scientific key sets of a single-result view
    (``hash`` is always required; ``q`` and ``p`` are optional after ``fdr``).
    """

    name: str
    kind: str  # "single" or "comparison"
    layers: tuple[int, ...]
    key_sets: tuple[frozenset[str], ...] = ()
    status: str = "available"


D10_VIEWS: dict[str, D10View] = {
    view.name: view
    for view in (
        D10View("tree_persistence", "single", (1, 2), _NOMINAL_KEYS),
        # The annotated style drawn as one chart (stops and synonymous leaves together),
        # the extra beside the two-row default; same keys, own name.
        D10View("tree_persistence_combined", "single", (1, 2), _NOMINAL_KEYS),
        D10View(
            "fitness_histogram",
            "single",
            (1,),
            (_GRAPH,),
            status="pending_landmark_product",
        ),
        D10View("tree_comparison_thresholds", "comparison", (2,)),
        D10View("fitness_uncertainty", "single", (3,), (_THR_FDR,)),
        D10View("tree_comparison_fdr", "comparison", (3,)),
        # The annotated style is two files of two rows each (nominal then band): one
        # marking stop codons, one marking synonymous leaves. The vanilla style marks
        # neither leaf kind and stays one two-row file under the plain view name.
        D10View("bootstrap_band", "single", (4,), _BAND_KEYS),
        D10View("bootstrap_band_stops", "single", (4,), _BAND_KEYS),
        D10View("bootstrap_band_synonymous", "single", (4,), _BAND_KEYS),
        D10View("bootstrap_band_combined", "single", (4,), _BAND_KEYS),
        D10View("tree_comparison_band", "comparison", (4,)),
        D10View("leaves", "single", (4,), (*_NOMINAL_KEYS, *_BAND_KEYS)),
        D10View("band_quantile_scan", "single", (4,), _DRAWS_KEYS),
        D10View(
            "ridge_entropy",
            "single",
            (4,),
            _BAND_KEYS,
        ),
        D10View("graph_codon_entropy", "comparison", (4,)),
        D10View("edge_evidence", "single", (3,), _LANDSCAPE_KEYS),
    )
}

_NAME_TOKEN = re.compile(r"^[a-z][a-z0-9]*(_combined)?$")
_EXTENSION = re.compile(r"^[a-z0-9]+$")
_HASH8 = re.compile(r"^[0-9a-f]{8}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OPTIONAL = frozenset({"q", "p"})
_FORBIDDEN = frozenset({"host", "run", "style"})


def _view(name: str, kind: str) -> D10View:
    view = D10_VIEWS.get(name)
    if view is None:
        raise MalformedLabelError(f"Unknown view '{name}'")
    if view.kind != kind:
        raise MalformedLabelError(f"View '{name}' is a {view.kind} view, not {kind}")
    return view


def _extension(ext: str) -> str:
    if not _EXTENSION.match(ext):
        raise MalformedLabelError(f"Invalid figure extension '{ext}'")
    return ext


def single_result_figure_name(
    view: str, params: Mapping[str, Any], style: str, ext: str
) -> str:
    """``<view>__<label>_style=<style>.<ext>`` for a single-result view."""
    spec = _view(view, "single")
    if style not in ALLOWED_STYLE_VALUES:
        raise MalformedLabelError(f"Unknown figure style '{style}'")
    clash = _FORBIDDEN & params.keys()
    if clash:
        raise MalformedLabelError(
            f"Figure names never carry {sorted(clash)}; style is a separate argument"
        )
    if "hash" not in params:
        raise MalformedLabelError("Figure names must carry the scientific 'hash'")
    scientific = set(params) - {"hash"}
    if scientific - _OPTIONAL not in spec.key_sets:
        raise MalformedLabelError(
            f"View '{view}' needs one of {[sorted(k) for k in spec.key_sets]} "
            f"(plus optional q, p) and 'hash'; got {sorted(scientific)}"
        )
    if scientific & _OPTIONAL and "fdr" not in scientific:
        raise MalformedLabelError("'q' and 'p' require 'fdr'")
    if "tree" in scientific:
        expected = "band" if "bandq" in scientific else "nominal"
        if params["tree"] != expected:
            raise MalformedLabelError(
                f"View '{view}' with keys {sorted(scientific)} needs "
                f"tree={expected}, got tree={params['tree']}"
            )
    label = format_label(view, {**params, "style": style}, validate=False)
    return f"{label}.{_extension(ext)}"


def comparison_hash(
    input_digests: Sequence[str], specification: Mapping[str, Any]
) -> str:
    """8-hex SHA-256 over the ordered full input digests and the specification.

    The hashed bytes are ``results.hashing.canonical_json`` (keys sorted at all
    depths, compact separators, UTF-8, NaN and inf rejected) of
    ``{"inputs": [<full sha256, in slot order>], "specification": {...}}``; the
    result is the first 8 hex characters of the SHA-256. Input order matters,
    mapping key order does not.
    """
    if not input_digests or any(not _SHA256.match(d) for d in input_digests):
        raise MalformedLabelError("Comparison inputs must be full sha256 hex digests")
    document = {"inputs": list(input_digests), "specification": dict(specification)}
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()[:8]


def _comparison_stem(view: str, comparison: str, digest: str) -> str:
    _view(view, "comparison")
    if not _NAME_TOKEN.match(comparison):
        raise MalformedLabelError(
            f"Comparison name '{comparison}' must be lowercase alphanumeric, optionally followed by _combined"
        )
    if not _HASH8.match(digest):
        raise MalformedLabelError(f"Comparison hash must be 8 hex, got '{digest}'")
    return f"{view}_{comparison}__hash={digest}"


def comparison_figure_name(view: str, comparison: str, digest: str, ext: str) -> str:
    """``<view>_<comparison>__hash=<8hex>.<ext>`` for a comparison view."""
    return f"{_comparison_stem(view, comparison, digest)}.{_extension(ext)}"


def comparison_sidecar_name(view: str, comparison: str, digest: str) -> str:
    """Name of the JSON sidecar that accompanies a comparison figure."""
    return f"{_comparison_stem(view, comparison, digest)}.json"


def comparison_sidecar_document(
    *,
    view: str,
    comparison: str,
    inputs: Sequence[tuple[str, str]],
    commit: str | None,
    notes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """View, comparison, ordered ``(label, sha256)`` inputs and the commit.

    ``notes``, when given, is carried under ``"notes"`` (what a view left out or
    changed in the drawing).
    """
    _view(view, "comparison")
    if any(not _SHA256.match(digest) for _, digest in inputs):
        raise MalformedLabelError("Sidecar inputs need full sha256 hex digests")
    document = {
        "view": view,
        "comparison": comparison,
        "inputs": [{"label": label, "sha256": digest} for label, digest in inputs],
        "commit": commit,
    }
    if notes is not None:
        document["notes"] = notes
    return document


def write_comparison_sidecar(path: Path, **document_arguments: Any) -> None:
    """Write ``comparison_sidecar_document(**document_arguments)`` to ``path``."""
    document = comparison_sidecar_document(**document_arguments)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def figure_relative_path(view: str, filename: str) -> Path:
    """``figures/<view>/<filename>`` after checking the whole file name grammar.

    A single-result file is ``<view>__<label>_style=<s>.<ext>`` with a label
    that parses for the view; a comparison file is
    ``<view>_<comparison>__hash=<8hex>.<ext>`` (figure or ``.json`` sidecar).
    """
    spec = D10_VIEWS.get(view)
    if spec is None:
        raise MalformedLabelError(f"Unknown view '{view}'")
    stem, dot, ext = filename.rpartition(".")
    if not dot or not _EXTENSION.match(ext):
        raise MalformedLabelError(f"File '{filename}' has no valid extension")
    if spec.kind == "single" and ext == "json":
        raise MalformedLabelError(f"Single-result figure '{filename}' has no sidecar")
    if spec.kind == "comparison":
        pattern = rf"{re.escape(view)}_[a-z][a-z0-9]*(?:_combined)?__hash=[0-9a-f]{{8}}"
        if not re.fullmatch(pattern, stem):
            raise MalformedLabelError(f"File '{filename}' is not a '{view}' name")
    else:
        label = parse_label(stem, validate=False)
        if label.product_name != view or label.style is None:
            raise MalformedLabelError(f"File '{filename}' is not a '{view}' name")
        params = {k: v for k, v in label.params.items() if k != "style"}
        single_result_figure_name(view, params, label.style, ext)
    return Path("figures") / view / filename


_HASH_PART = re.compile(r"_{1,2}hash=[0-9a-f]{8}")


def stable_figure_path(root: Path, view: str, filename: str) -> Path:
    """``<root>/figures/<filename without its hash>``: one flat, re-run-stable folder.

    The name is checked against the whole file name grammar first
    (``figure_relative_path``); the scientific ``hash`` is then dropped, so a
    re-run overwrites the figure instead of leaving a second one beside it.
    """
    return root / "figures" / "canonical" / stable_name(view, filename)


def stable_name(view: str, filename: str) -> str:
    """``filename`` without its hash, on the shelf: the name ``stable_figure_path`` writes."""
    figure_relative_path(view, filename)
    from ..output_paths import shelf_stem

    return shelf_stem(_HASH_PART.sub("", filename))


def stable_stem(view: str, filename: str) -> str:
    """``stable_name`` without its extension: a figure's shelf stem and render-record name."""
    return stable_name(view, filename).rpartition(".")[0]
