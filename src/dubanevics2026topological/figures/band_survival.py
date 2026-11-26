"""Surviving and absorbed peaks of a bootstrap band, shared by the D10 band views.

A nominal peak survives iff its ordinary H0 persistence exceeds 2c (Spec 2,
Section 6, ``bootstrap_band``); the band-simplified tree keeps exactly those peaks.
Peaks are identified by their sequence (the pair's ``birth_label``, the top leaf of
its branch), the same identity in the nominal and the band tree, so no matching
between trees is needed: a nominal peak is absorbed iff it is not a birth of the
band tree's pairs. The two rules are checked against each other and a disagreement
fails the build.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ._count_labels import general_label_text


@dataclass(frozen=True)
class BandSurvival:
    """The surviving and absorbed peak identities and the band's 2c."""

    surviving: frozenset[str]
    absorbed: frozenset[str]
    two_c: float

    @property
    def nominal_peaks(self) -> int:
        return len(self.surviving) + len(self.absorbed)


def classify_peaks(
    nominal_pairs: pl.DataFrame, band_pairs: pl.DataFrame, two_c: float
) -> BandSurvival:
    """Split the nominal peaks by the band tree's pairs and check against 2c."""
    nominal = set(nominal_pairs["birth_label"])
    surviving = set(band_pairs["birth_label"])
    if len(surviving) != band_pairs.height:
        raise ValueError("Band persistence pairs repeat a birth.")
    if not surviving <= nominal:
        raise ValueError("The band tree has peaks that the nominal tree lacks.")
    by_rule = set(nominal_pairs.filter(pl.col("persistence") > two_c)["birth_label"])
    if by_rule != surviving:
        raise ValueError(
            f"The band tree keeps {len(surviving)} peaks but {len(by_rule)} nominal "
            f"peaks have persistence > 2c = {general_label_text(two_c)}."
        )
    return BandSurvival(
        frozenset(surviving), frozenset(nominal - surviving), float(two_c)
    )


def peak_nodes(node_table: pl.DataFrame, peaks: frozenset[str]) -> pl.DataFrame:
    """Layout position and sequences (``x``, ``y``, ``dna_sequence``,
    ``aa_sequence``) of each peak's top leaf, one row per peak."""
    nodes = node_table.filter(pl.col("dna_sequence").is_in(sorted(peaks))).select(
        "x", "y", "dna_sequence", "aa_sequence"
    )
    if nodes.height != len(peaks) or nodes["dna_sequence"].n_unique() != len(peaks):
        raise ValueError("Every peak needs exactly one node in the saved layout.")
    return nodes
