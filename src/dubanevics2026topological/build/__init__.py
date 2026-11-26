"""Incremental analysis, table, and publication figure build commands.

The commands live in modules by group: ``common`` (constants and helpers),
``analysis_stages`` (fingerprints and the analysis stages), ``tables``, ``products``,
``bootstrap`` (bootstrap and Candidate-graph stages) and ``figures``. ``cli``
holds the argument parser and the command table behind
``python -m dubanevics2026topological.build``.

This package re-exports every name those modules define or import, so
``from dubanevics2026topological.build import X`` and ``build.X`` keep working.
Assigning to a name on the package (``monkeypatch.setattr(build, "OUTPUT", ...)``)
also rebinds it in every group module that holds it, so a patch through the
package reaches the code that runs, as it did when this was a single module.
"""

from __future__ import annotations

import sys
import types

from . import analysis_stages, bootstrap, cli, common, figures, products, tables

_GROUPS = (common, analysis_stages, tables, products, bootstrap, figures, cli)

for _group in _GROUPS:
    for _name, _value in vars(_group).items():
        if not _name.startswith("__") and _name != "annotations":
            globals().setdefault(_name, _value)


class _BuildPackage(types.ModuleType):
    """The package module, forwarding attribute writes to the group modules."""

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        for group in _GROUPS:
            if name in vars(group):
                setattr(group, name, value)


sys.modules[__name__].__class__ = _BuildPackage
del _group, _name, _value
