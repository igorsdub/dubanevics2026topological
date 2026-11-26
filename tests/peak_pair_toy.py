"""Shared toy tree for the peak-pair tests: a hand-built split tree (6 nt = 2 codons).

Edges point from a node toward the root, as ``save_band_tree`` writes them
(``source_tree_node_identity`` is the upper node). Tree, raw fitness in brackets::

        A [5]   B [4]
          \\     /
           S2 [3.5]     C [3]
              \\         /
               S1 [2]        D [2.5]
                  \\         /
                   R [1] (root)

Sequences (codon 1 | codon 2) and amino acids::

    A  GCT|AAA  AK      S2 GCG|AAA  AK
    B  GCC|AAA  AK      S1 TCT|AAA  SK
    C  GCT|GAA  AE      R  TGT|AAC  CN
    D  TCT|GAC  SD

Leaf order is A, B, C, D: at every fork the child reaching the highest peak
goes first (A 5 > B 4 under S2; S2's subtree 5 > C 3 under S1; S1's 5 > D 2.5).
Every expected value below is derived by hand from the sequences above.
"""

from __future__ import annotations

import polars as pl

A, B, C, D = "GCTAAA", "GCCAAA", "GCTGAA", "TCTGAC"
S2, S1, R = "GCGAAA", "TCTAAA", "TGTAAC"
FITNESS = {A: 5.0, B: 4.0, C: 3.0, D: 2.5, S2: 3.5, S1: 2.0, R: 1.0}
EDGES = [(A, S2), (B, S2), (S2, S1), (C, S1), (S1, R), (D, R)]


def toy_tree() -> tuple[pl.DataFrame, pl.DataFrame]:
    # Rows are deliberately not in leaf order.
    order = [D, S1, B, R, C, A, S2]
    critical = pl.DataFrame(
        {
            "tree_node_identity": order,
            "scalar": [FITNESS[n] for n in order],
        }
    )
    edges = list(reversed(EDGES))
    branches = pl.DataFrame(
        {
            "source_tree_node_identity": [s for s, _ in edges],
            "target_tree_node_identity": [t for _, t in edges],
            "branch_vertex_count": [0] * len(edges),
            "source_scalar": [FITNESS[s] for s, _ in edges],
            "target_scalar": [FITNESS[t] for _, t in edges],
        }
    )
    return critical, branches
