"""The story views as multi-output registry figures (Decision 10, the presentation specification).

``tree_persistence``, ``bootstrap_band``, ``tree_comparison_views``, ``ridge_entropy`` and
``graph_codon_entropy`` are each one registry figure that writes several files: one per saved
tree (or comparison, or ridge product), per layout and per style. The ``figure_id`` names the
family; each :class:`FigureOutput` names one file by its shelf stem, which is also its render
record name, so ``figure_return`` finds every one.

The functions here only draw. They read the payloads of ``build/story_view_inputs.py`` (saved
products, verified and loaded) and the settings, and never open ``output/``. Which saved trees a
view draws is configuration (``[story_views]`` in ``config/figures.toml``).

Reading guide: a band tree's "surviving peaks" count is its persistence-pair count, so it
includes the global maximum's pair (closed at the global minimum by convention).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..results import parse_label
from ._count_labels import count_label_text
from .bootstrap_band_view import build_bootstrap_band_figure
from .graph_codon_entropy import build_graph_codon_entropy_figure
from .labels import (
    fdr_level_text,
    settings_fdr_q,
    PAPKOU_FDR_BOOTSTRAPPED,
    TOPOGRAPHER_FDR_BOOTSTRAPPED,
    graph_label,
    state_label,
)
from .ridge_entropy_view import build_ridge_entropy_figure
from .tree_comparison_view import COLUMNS, ComparisonPanel
from .tree_comparison_view import (
    build_tree_comparison_figure as build_tree_comparison_chart,
)
from .tree_persistence_view import build_tree_persistence_figure
from .view_naming import (
    comparison_figure_name,
    comparison_hash,
    comparison_sidecar_document,
    single_result_figure_name,
    stable_stem,
)

TREE_PERSISTENCE_ID = "tree_persistence"
BOOTSTRAP_BAND_ID = "bootstrap_band"
TREE_COMPARISON_VIEWS_ID = "tree_comparison_views"
RIDGE_ENTROPY_ID = "ridge_entropy"
GRAPH_CODON_ENTROPY_ID = "graph_codon_entropy"
STORY_VIEW_FIGURE_IDS = frozenset(
    {
        TREE_PERSISTENCE_ID,
        BOOTSTRAP_BAND_ID,
        TREE_COMPARISON_VIEWS_ID,
        RIDGE_ENTROPY_ID,
        GRAPH_CODON_ENTROPY_ID,
    }
)
FULL_PUBLISHED_COMPARISON = "fullpublished"


@dataclass(frozen=True)
class FigureOutput:
    """One file of a multi-output figure.

    ``stem`` is the shelf stem (no extension): the name of the file under
    ``canonical/`` and of its render record. ``build`` draws the chart when called, so a
    failure names the output it belongs to. ``sidecar``, when given, is the JSON document
    written beside the figure as ``canonical/<stem>.json``.
    """

    stem: str
    build: Callable[[], Any]
    sidecar: Mapping[str, Any] | None = None


_NOT_LANDSCAPE_KEYS = frozenset({"hash", "tree", "draws", "seed"})
_COMPARISON_TITLES = {
    "tree_comparison_thresholds": "Threshold selections on one fitness axis",
    "tree_comparison_fdr": "Point estimate versus BY selection",
    "tree_comparison_band": f"Bootstrapped trees: {PAPKOU_FDR_BOOTSTRAPPED} versus {TOPOGRAPHER_FDR_BOOTSTRAPPED}",
}


def population_title(params: Mapping[str, Any], *, fdr_q: float | None = None) -> str:
    """Display title of a nominal tree's population from its label keys."""
    return state_label(_state(params, fdr_q=fdr_q))


def _state(
    params: Mapping[str, Any], *, band: bool = False, fdr_q: float | None = None
) -> str:
    """The graph state name of a saved product's label keys (``reflected_by_band`` and so on).

    The state names carry no q: a BY product maps to ``_by`` only when its recorded q is the
    analysis level ``fdr_q`` (the level the figure prints once), otherwise it is refused.
    """
    if params.get("graph") == "full":
        return "full"
    state = str(params["thr"])
    if params.get("fdr") == "by":
        if fdr_q is not None and float(params.get("q", fdr_q)) != float(fdr_q):
            raise ValueError(
                f"A BY tree at q = {params['q']} is not the analysis level q = {fdr_q}; "
                "the state labels do not record q."
            )
        state += "_by"
        if band or params.get("tree") == "band":
            state += "_band"
    return state


def panel_title(params: Mapping[str, Any], *, fdr_q: float | None = None) -> str:
    """Panel heading: the population, plus the band state for a band tree."""
    return population_title(params, fdr_q=fdr_q)


def _with_fdr(subtitle: str, params_list: Sequence[Mapping[str, Any]], fdr_q: float | None) -> str:
    """Append the FDR level once when any drawn state is selected with FDR."""
    level = fdr_level_text(fdr_q)
    if level and any(params.get("fdr") == "by" for params in params_list):
        return f"{subtitle} · {level}"
    return subtitle


def split_comparison_id(comparison_id: str) -> tuple[str, str]:
    """``(view, name)`` of a ``[comparisons.<view>_<name>]`` table name."""
    view, _, name = comparison_id.rpartition("_")
    if view not in COLUMNS or not name:
        raise ValueError(f"{comparison_id!r} is not <tree comparison view>_<name>.")
    return view, name


def resolve_panel_labels(
    comparison: Mapping[str, Any], tree_labels: Sequence[str]
) -> list[str]:
    """The one saved tree label of each declared panel, in declared order."""
    state = comparison["tree_state"]
    parsed = {
        label: dict(parse_label(label, validate=False).params) for label in tree_labels
    }
    resolved = []
    for panel in comparison["panels"]:
        wanted = {key: str(value) for key, value in panel.items()}
        found = [
            label
            for label, params in parsed.items()
            if str(params.get("tree")) == state
            and {k: str(v) for k, v in params.items() if k not in _NOT_LANDSCAPE_KEYS}
            == wanted
        ]
        if len(found) != 1:
            raise ValueError(
                f"Panel {panel} needs exactly one {state} tree, found {found}."
            )
        resolved.append(found[0])
    return resolved


def comparison_specification(
    view: str, name: str, comparison: Mapping[str, Any]
) -> dict[str, Any]:
    """What a comparison hash covers besides its inputs: the declaration, not looks."""
    return {"view": view, "comparison": name, **comparison}


def comparison_layouts(name: str) -> list[tuple[str, str]]:
    """``(layout, file name stem)`` of each figure of one comparison.

    The two-row layout is the default under ``name``; the combined single-chart extra
    is named ``<name>_combined`` (comparison names are lowercase letters and digits).
    """
    return [("rows", name), ("combined", f"{name}_combined")]


def layout_outputs(view: str, style: str) -> list[tuple[str, str]]:
    """``(view name, layout)`` of each figure one style of a single-result view writes.

    The annotated style is written as two rows (stop-codon peaks, then synonymous
    leaves) under ``view`` and as one chart under ``<view>_combined``;
    ``bootstrap_band`` instead writes ``bootstrap_band_stops`` and
    ``bootstrap_band_synonymous``, two rows each (nominal, band). The vanilla style
    marks neither leaf kind, so it has one figure and no layout of its own.
    """
    if view == "bootstrap_band" and style == "annotated":
        # Two files of two rows each, one per leaf kind, instead of one four-row file.
        outputs = [
            ("bootstrap_band_stops", "stops"),
            ("bootstrap_band_synonymous", "synonymous"),
        ]
    else:
        outputs = [(view, "rows")]
    if style == "annotated":
        # The combined file is the one-row-per-tree encoded layout (teal synonymous,
        # red cross for a stop); the older single-chart "combined" layout is slides'.
        outputs.append((f"{view}_combined", "encoded"))
    return outputs


def plot_branch_count(inputs: Any) -> int:
    """Number of drawn branches of a saved layout."""
    return int(inputs.tree_analysis.plot_data.edge_table.height)


def _single_stem(view: str, params: Mapping[str, Any], style: str) -> str:
    return stable_stem(view, single_result_figure_name(view, params, style, "png"))


# -- tree_persistence ---------------------------------------------------------------------


def tree_persistence_outputs(
    payload: Mapping[str, Any], settings: dict[str, Any], style: str
) -> Iterator[FigureOutput]:
    """Each nominal tree's ``tree_persistence`` files in one style."""
    fdr_q = settings_fdr_q(settings)
    for label, inputs in payload["nominal"].items():
        params = dict(parse_label(label, validate=False).params)
        subtitle = _with_fdr(
            f"Papkou et al. (2023) · {count_label_text(inputs.vertex_count)} sequences · "
            f"{count_label_text(inputs.persistence_table.height)} peaks",
            [params],
            fdr_q,
        )
        for out_view, layout in layout_outputs(TREE_PERSISTENCE_ID, style):
            yield FigureOutput(
                _single_stem(out_view, params, style),
                lambda inputs=inputs, params=params, subtitle=subtitle, layout=layout: (
                    build_tree_persistence_figure(
                        inputs,
                        settings,
                        style,
                        title=population_title(params, fdr_q=fdr_q),
                        subtitle=subtitle,
                        layout=layout,
                    )
                ),
            )


# -- bootstrap_band -----------------------------------------------------------------------


def bootstrap_band_outputs(
    payload: Mapping[str, Any], settings: dict[str, Any], style: str
) -> Iterator[FigureOutput]:
    """Each band tree's ``bootstrap_band`` files in one style."""
    fdr_q = settings_fdr_q(settings)
    for label, item in payload["band"].items():
        params = dict(parse_label(label, validate=False).params)
        subtitle = _with_fdr(
            f"Papkou et al. (2023) · {count_label_text(item.nominal.vertex_count)} sequences · "
            f"bootstrap band quantile {params['bandq']}, {params['draws']} draws",
            [params],
            fdr_q,
        )
        for out_view, layout in layout_outputs(BOOTSTRAP_BAND_ID, style):
            yield FigureOutput(
                _single_stem(out_view, params, style),
                lambda item=item, params=params, subtitle=subtitle, layout=layout: (
                    build_bootstrap_band_figure(
                        item.nominal,
                        item.band,
                        item.survival,
                        settings,
                        style,
                        title=population_title(params, fdr_q=fdr_q),
                        subtitle=subtitle,
                        layout=layout,
                    )
                ),
            )


# -- tree_comparison_views ----------------------------------------------------------------


def tree_comparison_outputs(
    payload: Mapping[str, Any], settings: dict[str, Any], variant: str
) -> Iterator[FigureOutput]:
    """Each declared comparison's two-row and combined files, with their JSON sidecars."""
    del variant  # comparisons have no style: one set of files
    fdr_q = settings_fdr_q(settings)
    figures = settings["figures"]
    bands = payload["band"]
    trees = {
        **payload["nominal"],
        **{label: item.band for label, item in bands.items()},
    }
    peak_labels_wanted = bool(
        figures["d10"]["tree_comparison"].get("peak_identity_labels", True)
    )
    for comparison_id, comparison in figures["comparisons"].items():
        view, name = split_comparison_id(comparison_id)
        labels = resolve_panel_labels(comparison, list(trees))
        loaded = [trees[label] for label in labels]
        digest = comparison_hash(
            [item.tree_id for item in loaded],
            comparison_specification(view, name, comparison),
        )
        panels = []
        comparison_params = [
            dict(parse_label(label, validate=False).params) for label in labels
        ]
        for label, item in zip(labels, loaded, strict=True):
            params = dict(parse_label(label, validate=False).params)
            peaks = item.persistence_table.height
            band = params.get("tree") == "band"
            panels.append(
                ComparisonPanel(
                    title=panel_title(params, fdr_q=fdr_q),
                    subtitle=(
                        f"{count_label_text(item.vertex_count)} sequences · {count_label_text(peaks)} "
                        f"{'surviving ' if band else ''}"
                        f"peaks{' (incl. global maximum)' if band else ''}"
                        f" · {count_label_text(plot_branch_count(item))} branches"
                    ),
                    inputs=item,
                    peak_labels=bands[label].peaks
                    if band and peak_labels_wanted
                    else None,
                )
            )
        for layout, shown in comparison_layouts(name):
            yield FigureOutput(
                stable_stem(view, comparison_figure_name(view, shown, digest, "png")),
                lambda layout=layout, panels=panels, view=view: (
                    build_tree_comparison_chart(
                        view,
                        panels,
                        settings,
                        title=_COMPARISON_TITLES[view],
                        subtitle=_with_fdr(
                            "Papkou et al. (2023) · common fitness axis",
                            comparison_params,
                            fdr_q,
                        ),
                        layout=layout,
                    )
                ),
                comparison_sidecar_document(
                    view=view,
                    comparison=shown,
                    inputs=[(item.tree_label, item.tree_id) for item in loaded],
                    commit=payload.get("commit"),
                ),
            )


# -- ridge_entropy ------------------------------------------------------------------------


def ridge_entropy_outputs(
    payload: Mapping[str, Any], settings: dict[str, Any], style: str
) -> Iterator[FigureOutput]:
    """Each saved ``ridge_entropy_sweeps`` product's ``ridge_entropy`` file in one style."""
    fdr_q = settings_fdr_q(settings)
    for label, item in payload.items():
        params = dict(parse_label(label, validate=False).params)
        entropy = item["entropy"]
        subtitle = _with_fdr(
            f"Papkou et al. (2023) · {count_label_text(entropy.sidecar['counts']['core'])} sequences · "
            f"bootstrap band quantile {params['bandq']}, {params['draws']} draws",
            [params],
            fdr_q,
        )
        yield FigureOutput(
            _single_stem(RIDGE_ENTROPY_ID, params, style),
            lambda item=item, params=params, subtitle=subtitle: (
                build_ridge_entropy_figure(
                    item["entropy"],
                    item["functional"],
                    settings,
                    style,
                    title=state_label(_state(params, band=True, fdr_q=fdr_q)),
                    subtitle=subtitle,
                )
            ),
        )


# -- graph_codon_entropy ------------------------------------------------------------------


def graph_codon_entropy_outputs(
    payload: Mapping[str, Any], settings: dict[str, Any], style: str
) -> Iterator[FigureOutput]:
    """Supplementary S14: the Full and Papkou graphs' entropy, with its JSON sidecar."""
    entropy = payload["entropy"]
    digests = entropy.sidecar["population_sha256"]
    digest = comparison_hash(
        [digests["full"], digests["published"]],
        {"sweep_points": payload["sweep_points"], "populations": ["full", "published"]},
    )
    subtitle = (
        f"{graph_label('full')} {count_label_text(entropy.sidecar['counts']['full'])} sequences · "
        f"{graph_label('published')} {count_label_text(entropy.sidecar['counts']['published'])} sequences"
    )
    yield FigureOutput(
        stable_stem(
            GRAPH_CODON_ENTROPY_ID,
            comparison_figure_name(
                GRAPH_CODON_ENTROPY_ID, FULL_PUBLISHED_COMPARISON, digest, "png"
            ),
        ),
        lambda: build_graph_codon_entropy_figure(
            entropy,
            settings,
            style,
            title="Entropy against fitness threshold",
            subtitle=subtitle,
        ),
        comparison_sidecar_document(
            view=GRAPH_CODON_ENTROPY_ID,
            comparison=FULL_PUBLISHED_COMPARISON,
            inputs=[
                (payload["labels"]["full"], digests["full"]),
                (payload["labels"]["published"], digests["published"]),
            ],
            commit=None,
            notes={
                "entropy": (
                    "Shannon entropy in bits of the amino acid at each codon "
                    "position, sequences with raw fitness at or above the threshold"
                )
            },
        ),
    )
