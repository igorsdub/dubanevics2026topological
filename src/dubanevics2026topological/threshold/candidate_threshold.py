"""Compatibility alias for :mod:`threshold.functional_threshold`."""

import sys

from dubanevics2026topological.threshold import functional_threshold as _implementation

sys.modules[__name__] = _implementation
