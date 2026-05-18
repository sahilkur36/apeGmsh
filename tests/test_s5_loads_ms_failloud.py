"""S5.2 regression — loads/masses ``__ms__`` consumer fails loud.

The ``__ms__`` mesh-selection sentinel consumer in
``LoadsComposite._target_nodes`` / ``MassesComposite._target_nodes``
used to ``return set()`` when the named set was missing from
``g.mesh_selection._sets`` — silently binding the load/mass to **zero**
nodes. It now raises ``KeyError``. This locks that fail-loud behavior.

Scope: S5.2 only. S5.1 (results ``selection=`` keep-loud on
import-origin FEMData) and S5.3 (``_element_centroids`` fail-loud) are
already on ``main`` (the latter via the ``_element_centroids``
dangling-node fix) and are deliberately not retested here.

Unit-level driver: ``LoadsComposite``/``MassesComposite`` resolve
against the live session's ``_parent.mesh_selection``, not a FEMData.
The stub parent matches the name to a sentinel via ``_resolve_target``
(which iterates ``ms._sets.items()``), but ``_sets.get((dim, tag))``
misses — exercising the literal ``if info is None:`` arm.
"""

from __future__ import annotations

import types

import pytest

from apeGmsh.core.LoadsComposite import LoadsComposite
from apeGmsh.core.MassesComposite import MassesComposite


class _MissingSets(dict):
    """``.items()`` yields the sentinel match; ``.get`` always misses."""

    def get(self, _key, _default=None):
        return None


def _stub_parent():
    ms = types.SimpleNamespace()
    ms._sets = _MissingSets()
    ms._sets[(0, 99)] = {"name": "ghost", "node_ids": [1, 2, 3]}
    return types.SimpleNamespace(mesh_selection=ms, parts=None)


def test_loads_ms_missing_set_raises_keyerror():
    lc = LoadsComposite.__new__(LoadsComposite)
    lc._parent = _stub_parent()
    with pytest.raises(KeyError, match="Refusing to silently bind this load"):
        lc._target_nodes(
            "ghost", node_map=None, all_nodes=None, source="auto"
        )


def test_masses_ms_missing_set_raises_keyerror():
    mc = MassesComposite.__new__(MassesComposite)
    mc._parent = _stub_parent()
    with pytest.raises(KeyError, match="Refusing to silently bind this mass"):
        mc._target_nodes(
            "ghost", node_map=None, all_nodes=None, source="auto"
        )
