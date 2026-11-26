"""Frameless slide trees: file names, the shared fitness extent and the tree-only panel."""

from __future__ import annotations

import os
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from dubanevics2026topological.figures import slide_tree
from dubanevics2026topological.figures._charts import stop_codon_nodes


def test_coloured_files_keep_the_d10_name() -> None:
    name = "tree_persistence__thr=published_fdr=none_tree=nominal_style=annotated.png"
    assert slide_tree.slide_figure_name(name, "count") == name


@pytest.mark.parametrize("style", ["annotated", "vanilla"])
def test_uncoloured_files_insert_the_color_axis_before_style(style: str) -> None:
    name = f"tree_persistence__graph=full_tree=nominal_style={style}.png"
    assert (
        slide_tree.slide_figure_name(name, "none")
        == f"tree_persistence__graph=full_tree=nominal_color=none_style={style}.png"
    )


def test_slide_figure_name_refuses_unknown_coloring_and_unstyled_names() -> None:
    with pytest.raises(ValueError, match="coloring must be one of"):
        slide_tree.slide_figure_name("x_style=vanilla.png", "hue")
    with pytest.raises(ValueError, match="no _style="):
        slide_tree.slide_figure_name("x.png", "none")


def _png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    return struct.unpack(">II", header[16:24])


@pytest.mark.parametrize("group", ["tree_only", "full"])
def test_rendered_slides_of_a_group_share_a_pixel_size(group: str) -> None:
    """Checks the PNGs of a slide run (``HPC_OUTPUT_DIR`` or ``output``); skipped without one.

    The tree-only files of every tree, style and coloring have one size; the full files
    have one size per view (``tree_persistence`` and ``bootstrap_band`` differ in layout).
    """
    root = Path(os.environ.get("HPC_OUTPUT_DIR", "output")) / "figures" / "slides"
    files = sorted((root / group).glob("*.png"))
    if not files:
        pytest.skip(f"no rendered slides under {root / group}")
    sizes: dict[str, set[tuple[int, int]]] = {}
    for path in files:
        view = "bootstrap_band" if path.name.startswith("bootstrap_band") else "other"
        key = view if group == "full" else "all"
        sizes.setdefault(key, set()).add(_png_size(path))
    assert all(len(found) == 1 for found in sizes.values()), sizes


def test_shared_domain_must_contain_the_tree_extent() -> None:
    settings = {"figures": {"fitness_domain": [-1.2, 1.4]}}
    inside = SimpleNamespace(scalar_domain=(-0.9, 0.8))
    assert slide_tree.slide_fitness_domain(settings, inside) == (-1.2, 1.4)
    outside = SimpleNamespace(scalar_domain=(-1.3, 0.8))
    with pytest.raises(ValueError, match="outside the shared slide domain"):
        slide_tree.slide_fitness_domain(settings, outside)


def test_slide_stop_table_keeps_every_stop_node_with_its_kind() -> None:
    import polars as pl

    nodes = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.9, 0.8, 0.5, -0.5],
            "display_event": ["maximum", "maximum", "split_saddle", "minimum"],
            "has_stop_codon": [True, False, True, True],
            "dna_sequence": ["TAA", "AAA", "TAG", "TGA"],
            "aa_sequence": ["*", "K", "*", "*"],
        }
    )
    tree = SimpleNamespace(plot_data=SimpleNamespace(node_table=nodes))
    prepared, annotated = slide_tree.slide_tree_analysis(tree, "annotated")
    assert annotated
    table = prepared.all_stop_marker_table
    assert table["dna_sequence"].to_list() == ["TAA", "TAG", "TGA"]
    # The figure layer then drops the saddle under the default node kinds.
    kept = stop_codon_nodes(table, {}, stop_only=True)
    assert kept["dna_sequence"].to_list() == ["TAA", "TGA"]
    assert not hasattr(slide_tree, "slide_stop_identities")


def test_registry_slide_names_and_group_sizes(tmp_path) -> None:
    import struct as _struct

    from dubanevics2026topological.build import registry_slides

    target = registry_slides.slide_target(
        tmp_path, "full", "candidate_tree_histogram", "vanilla", "none"
    )
    assert target == (
        tmp_path
        / "slides"
        / "full"
        / "candidate_tree_histogram_color=none_style=vanilla.png"
    )

    def png(path: Path, size: tuple[int, int]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 8 + _struct.pack(">II", *size))
        return path

    same = [
        png(tmp_path / "slides/tree_only/a_style=vanilla.png", (10, 5)),
        png(tmp_path / "slides/tree_only/b_color=none_style=vanilla.png", (10, 5)),
    ]
    registry_slides.check_sizes(same)
    odd = [*same, png(tmp_path / "slides/tree_only/c_style=vanilla.png", (11, 5))]
    with pytest.raises(ValueError, match="differ in pixel size"):
        registry_slides.check_sizes(odd)
