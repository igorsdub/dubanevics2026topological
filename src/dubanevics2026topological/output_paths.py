"""Shelf paths for figures and tables.

Tracked renders live under ``output/figures/canonical/`` (tables:
``output/tables/canonical/``). A shelf file is ``<figure_id>[__params].ext``.
The ``papkou2023_`` prefix is not part of a shelf name, and the band-figure
prefix ``bootstrap_band__`` is ``bootstrap__``, and likewise ``bootstrap_band_stops__``,
``bootstrap_band_synonymous__`` and ``bootstrap_band_combined__`` become ``bootstrap_stops__``,
``bootstrap_synonymous__`` and ``bootstrap_combined__``.
"""

from __future__ import annotations

from pathlib import Path

CANONICAL = "canonical"


def shelf_stem(name: str) -> str:
    """Shelf file name for a figure id or a flat render name."""
    stem = name.removeprefix("papkou2023_")
    stem = stem.replace(
        "full_graph_tree_persistence", "full_tree_persistence"
    )
    stem = stem.replace("article_tree_histogram", "published_tree_histogram")
    for kind in ("stops", "synonymous", "combined"):
        if stem.startswith(f"bootstrap_band_{kind}__"):
            stem = f"bootstrap_{kind}__" + stem.removeprefix(f"bootstrap_band_{kind}__")
    if stem.startswith("bootstrap_band__"):
        stem = "bootstrap__" + stem.removeprefix("bootstrap_band__")
    return stem


def figure_path(
    root: Path,
    figure_id: str,
    variant: str | None = None,
    *,
    titled: bool = False,
    ext: str = "png",
) -> Path:
    """Path of one figure on the identity shelf.

    ``root`` is the output root (``output/``) or a directory that already
    contains ``figures/``. A variant is appended as ``__variant`` when the
    stem does not already name it. ``titled`` selects ``canonical/titled/``.
    """
    figures = root / "figures" if (root / "figures").exists() or root.name != "figures" else root
    if root.name == "figures":
        figures = root
    stem = shelf_stem(figure_id)
    if variant and f"__{variant}" not in stem and not stem.endswith(f"_{variant}"):
        stem = f"{stem}__{variant}"
    folder = figures / CANONICAL
    if titled:
        folder = folder / "titled"
    return folder / f"{stem}.{ext.lstrip('.')}"


def table_path(root: Path, table_id: str, ext: str = "csv") -> Path:
    """Path of one table on the identity shelf."""
    tables = root / "tables" if root.name != "tables" else root
    return tables / CANONICAL / f"{shelf_stem(table_id)}.{ext.lstrip('.')}"


def migrate_pairs(figures: Path = Path("output/figures")) -> list[tuple[Path, Path]]:
    """Flat tracked renders that still need a move onto the shelf."""
    pairs = []
    if not figures.is_dir():
        return pairs
    for path in sorted(figures.iterdir()):
        if not path.is_file():
            continue
        if path.suffix not in {".png", ".json"}:
            continue
        target = figures / CANONICAL / shelf_stem(path.name)
        if path.resolve() != target.resolve():
            pairs.append((path, target))
    return pairs


def main() -> None:
    """Move flat renders onto the shelf. ``--print`` lists the pairs and stops."""
    import subprocess
    import sys

    pairs = migrate_pairs()
    if "--print" in sys.argv:
        for source, target in pairs:
            print(f"{source.as_posix()} {target.as_posix()}")
        return
    for source, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "mv", source.as_posix(), target.as_posix()])
    print(f"moved {len(pairs)} files")


def resolve_figure(figures: Path, name: str) -> Path | None:
    """Find a render under ``canonical/`` or, for one release, the old flat path."""
    shelf = figures / CANONICAL / shelf_stem(name)
    if shelf.is_file():
        return shelf
    flat = figures / name
    if flat.is_file():
        return flat
    return None


if __name__ == "__main__":
    main()
