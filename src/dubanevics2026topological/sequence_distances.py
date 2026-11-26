"""Sequence distances shared across analyses and tables."""

from __future__ import annotations


def hamming_distance(left: str, right: str) -> int:
    """Count mismatched positions between two equal-length sequences.

    Unequal lengths raise ``ValueError``. This is a plain mismatch count, the same
    value as Topographer's own Hamming distance, which is not part of its public
    interface and so is not imported here.
    """
    if len(left) != len(right):
        raise ValueError(
            "Hamming distance needs two sequences of the same length, "
            f"got {len(left)} and {len(right)}."
        )
    return sum(a != b for a, b in zip(left, right, strict=True))
