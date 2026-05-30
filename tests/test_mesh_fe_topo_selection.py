"""ADR 0045 S3b — mesh FE element/node picks on the SelectionState
contract as MESH_TOPO targets.

The viewer's pick path is VTK-observer-bound, but the new contract is
headless: FE picks are ``SelectionTarget(MESH_TOPO, ...)`` (nodes dim=0,
elements dim>=1) flowing through a dedicated ``SelectionState`` (so they
gain undo/redo and never collide with the BREP ``.picks`` shim). These
tests pin the target shapes, the count split, and the mode-routed
undo/clear helpers — without constructing a real (Qt/VTK) viewer.
"""
from __future__ import annotations

from types import SimpleNamespace

from apeGmsh.viewers.core.selection import SelectionState
from apeGmsh.viewers.mesh_viewer import MeshViewer
from apeGmsh.viewers.scene_ir import SelectionTarget, Substrate


def _elem(tag: int, dim: int = 3) -> SelectionTarget:
    return SelectionTarget(Substrate.MESH_TOPO, dim, tag)


def _node(tag: int) -> SelectionTarget:
    return SelectionTarget(Substrate.MESH_TOPO, 0, tag)


# ── FE picks are MESH_TOPO targets in a SelectionState ──────────────

def test_fe_picks_store_mesh_topo_targets() -> None:
    sel = SelectionState()
    sel.toggle(_elem(5))
    sel.toggle(_node(5))            # node tag 5 distinct from element tag 5
    assert len(sel.targets) == 2
    assert all(t.substrate is Substrate.MESH_TOPO for t in sel.targets)
    dims = {t.dim for t in sel.targets}
    assert dims == {0, 3}           # node dim 0, solid element dim 3


def test_fe_pick_toggle_and_undo_redo() -> None:
    sel = SelectionState()
    sel.toggle(_elem(1))
    sel.toggle(_elem(2))
    sel.toggle(_elem(1))            # un-pick element 1
    assert {t.key for t in sel.targets} == {2}
    assert sel.undo()               # restore element 1
    assert {t.key for t in sel.targets} == {1, 2}
    assert sel.redo()               # re-remove element 1
    assert {t.key for t in sel.targets} == {2}


def test_picks_shim_never_sees_mesh_topo() -> None:
    # The BREP .picks shim would crash on a MESH_TOPO target; S3b keeps
    # FE picks in a SEPARATE state so the BREP state stays shim-safe.
    brep = SelectionState()
    brep.pick((2, 7))
    assert brep.picks == [(2, 7)]   # no MESH_TOPO contamination


# ── viewer helper logic (unbound, on a light stub) ──────────────────

def test_fe_pick_counts_splits_elements_and_nodes() -> None:
    fe = SelectionState()
    fe.toggle(_elem(10, dim=2))
    fe.toggle(_elem(11, dim=3))
    fe.toggle(_node(1))
    stub = SimpleNamespace(_fe_sel=fe)
    assert MeshViewer._fe_pick_counts(stub) == (2, 1)


def test_active_pick_sel_routes_by_mode() -> None:
    brep, fe = SelectionState(), SelectionState()
    stub = SimpleNamespace(_sel=brep, _fe_sel=fe, _pick_mode=["brep"])
    assert MeshViewer._active_pick_sel(stub) is brep
    stub._pick_mode[0] = "element"
    assert MeshViewer._active_pick_sel(stub) is fe
    stub._pick_mode[0] = "node"
    assert MeshViewer._active_pick_sel(stub) is fe


def test_handle_undo_routes_to_active_set() -> None:
    brep, fe = SelectionState(), SelectionState()
    brep.pick((3, 1))
    fe.toggle(_elem(9))
    stub = SimpleNamespace(_sel=brep, _fe_sel=fe, _pick_mode=["element"])
    # _handle_undo delegates to _active_pick_sel — bind it on the stub.
    stub._active_pick_sel = lambda: MeshViewer._active_pick_sel(stub)
    MeshViewer._handle_undo(stub)
    assert fe.targets == []          # FE pick undone
    assert brep.picks == [(3, 1)]    # BREP set untouched
