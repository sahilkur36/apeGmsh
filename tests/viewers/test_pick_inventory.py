"""PickInventory actor-inventory handshake (ADR 0047 R-D.2b).

The inventory is a thin dict keyed by ``id(actor)`` → ``(kind,
reverse_map_fn, actor)``. Diagrams register on attach and unregister on
detach so the results click controller can route a vtkCellPicker hit
directly without walking every active diagram.

R-D.2b removed the dead per-mode allow-list (``set_pick_mode`` /
``_MODE_ALLOW``) and the ``Alt`` pick-through context manager. The
inventory no longer touches actor pickability — registered overlay
actors stay pickable and the controller routes by mode at resolve time.
These tests pin the surviving inventory contract + that register no
longer disables an actor (the GP-picking-without-Alt fix).
"""
from __future__ import annotations

from typing import Optional

import pytest

from apeGmsh.viewers.core.results_pick_engine import PickInventory

from tests.conftest import _open_model_from_h5


class _StubActor:
    """Minimal stand-in for a vtkProp / vtkActor."""
    def __init__(self) -> None:
        self._pickable = True

    def SetPickable(self, v: bool) -> None:    # noqa: N802 — VTK API name
        self._pickable = bool(v)

    def GetPickable(self) -> bool:             # noqa: N802 — VTK API name
        return self._pickable


# ---------------------------------------------------------------------
# Core inventory contract
# ---------------------------------------------------------------------

def test_register_actor_records_kind_and_reverse_map():
    pe = PickInventory()
    actor = _StubActor()

    def _rev(cell_id: int) -> Optional[tuple]:
        return (100 + cell_id, 0, (0.0, 0.0, 0.0))

    pe.register_actor(actor, "gp", _rev)
    assert pe.is_registered(actor) is True
    assert pe.kind_for_actor(actor) == "gp"
    assert len(pe) == 1


def test_register_actor_leaves_actor_pickable():
    """The R-D.2b fix: register no longer disables the actor. A GP marker
    registered while in (default) NODE intent stays pickable — the old
    allow-list flipped it to False, which is why GP picking needed Alt."""
    pe = PickInventory()
    actor = _StubActor()
    assert actor.GetPickable() is True
    pe.register_actor(actor, "gp", lambda _c: None)
    assert actor.GetPickable() is True  # untouched


def test_resolve_pick_dispatches_to_reverse_map():
    pe = PickInventory()
    actor = _StubActor()

    def _rev(cell_id: int) -> Optional[tuple]:
        return (200, cell_id, (1.0, 2.0, 3.0))

    pe.register_actor(actor, "gp", _rev)
    assert pe.resolve_pick(actor, 7) == (200, 7, (1.0, 2.0, 3.0))


def test_resolve_pick_returns_none_for_unregistered_actor():
    pe = PickInventory()
    assert pe.resolve_pick(_StubActor(), 0) is None


def test_resolve_pick_returns_none_when_reverse_map_returns_none():
    pe = PickInventory()
    actor = _StubActor()
    pe.register_actor(actor, "gp", lambda _cell: None)
    assert pe.resolve_pick(actor, 5) is None


def test_resolve_pick_swallows_reverse_map_exception():
    pe = PickInventory()
    actor = _StubActor()

    def _raise(_cell: int) -> Optional[tuple]:
        raise RuntimeError("boom")

    pe.register_actor(actor, "gp", _raise)
    assert pe.resolve_pick(actor, 0) is None


def test_unregister_drops_actor():
    pe = PickInventory()
    actor = _StubActor()
    pe.register_actor(actor, "gp", lambda _c: None)
    pe.unregister_actor(actor)
    assert pe.is_registered(actor) is False
    assert len(pe) == 0


def test_unregister_unknown_actor_is_noop():
    pe = PickInventory()
    pe.unregister_actor(_StubActor())
    assert len(pe) == 0


def test_register_overwrites_prior_entry():
    """Diagrams that re-attach should not duplicate entries."""
    pe = PickInventory()
    actor = _StubActor()
    pe.register_actor(actor, "gp", lambda _c: (1, 0))
    pe.register_actor(actor, "gp", lambda _c: (2, 0))
    assert pe.resolve_pick(actor, 0) == (2, 0)
    assert len(pe) == 1


def test_register_actor_with_none_is_noop():
    pe = PickInventory()
    pe.register_actor(None, "gp", lambda _c: None)
    assert len(pe) == 0


def test_registered_actors_snapshot():
    pe = PickInventory()
    a1, a2 = _StubActor(), _StubActor()
    pe.register_actor(a1, "gp", lambda _c: None)
    pe.register_actor(a2, "fiber", lambda _c: None)
    pairs = pe.registered_actors()
    assert sorted(k for (k, _a) in pairs) == ["fiber", "gp"]
    assert len(pairs) == 2


# ---------------------------------------------------------------------
# Integration — GaussPointDiagram attach/detach round-trip
# ---------------------------------------------------------------------

@pytest.fixture
def _gp_setup(g, tmp_path):
    """Solid hex mesh + 1-GP gauss results — mirrors the pattern in
    test_gauss_marker.py:gauss_results, kept local so this file doesn't
    cross-import test-module fixtures."""
    import numpy as np
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter

    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    eids: list[int] = []
    for group in fem.elements:
        if group.element_type.dim == 3:
            eids.extend(int(x) for x in group.ids)
    eids = sorted(eids)
    assert eids, "no 3-D elements meshed"

    n_elem = len(eids)
    values = np.zeros((1, n_elem, 1), dtype=np.float64)
    for ei in range(n_elem):
        values[0, ei, 0] = float(ei)
    natural = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    path = tmp_path / "pe_handshake.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="s", kind="transient",
            time=np.zeros(1, dtype=np.float64),
        )
        w.write_gauss_group(
            sid, "partition_0", group_id="g0",
            class_tag=10, int_rule=0,
            element_index=np.asarray(eids, dtype=np.int64),
            natural_coords=natural,
            local_axes_quaternion=None,
            components={"stress_xx": values},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path)), fem, eids


def test_gauss_diagram_registers_on_attach_unregisters_on_detach(_gp_setup):
    """Attach a GaussPointDiagram against a headless plotter and verify it
    lands in (and leaves) the scene's PickInventory — and that the GP
    actor is left pickable (the R-D.2b fix)."""
    import pyvista as pv

    from apeGmsh.viewers.backends import PyVistaQtBackend
    from apeGmsh.viewers.diagrams import (
        DiagramSpec,
        GaussPointDiagram,
        GaussMarkerStyle,
        SlabSelector,
    )
    from apeGmsh.viewers.scene.fem_scene import build_fem_scene

    results, fem, eids = _gp_setup
    scene = build_fem_scene(fem)
    pe = PickInventory()
    scene.pick_engine = pe

    plotter = pv.Plotter(off_screen=True)
    diagram = GaussPointDiagram(
        DiagramSpec(
            kind="gauss_marker",
            selector=SlabSelector(component="stress_xx"),
            style=GaussMarkerStyle(),
        ),
        results,
    )
    try:
        diagram.attach(PyVistaQtBackend(plotter), fem, scene)
        assert len(pe) == 1
        pairs = pe.registered_actors()
        assert [k for (k, _a) in pairs] == ["gp"]
        actor = pairs[0][1]
        # The fix: the GP actor is pickable after registration (no Alt
        # needed for GP picking).
        assert bool(actor.GetPickable()) is True
        result = pe.resolve_pick(actor, 0)
        assert result is not None
        eid_resolved, gp_idx, world = result
        assert eid_resolved in eids
        assert gp_idx == 0
        assert len(world) == 3

        diagram.detach()
        assert len(pe) == 0
    finally:
        plotter.close()


def test_gauss_diagram_attach_with_no_pick_engine_is_noop(_gp_setup):
    """Diagrams must tolerate ``scene.pick_engine is None`` (headless test
    contexts, mesh-only flows). The attach should still succeed."""
    import pyvista as pv

    from apeGmsh.viewers.backends import PyVistaQtBackend
    from apeGmsh.viewers.diagrams import (
        DiagramSpec,
        GaussPointDiagram,
        GaussMarkerStyle,
        SlabSelector,
    )
    from apeGmsh.viewers.scene.fem_scene import build_fem_scene

    results, fem, _eids = _gp_setup
    scene = build_fem_scene(fem)
    assert getattr(scene, "pick_engine", None) is None

    plotter = pv.Plotter(off_screen=True)
    diagram = GaussPointDiagram(
        DiagramSpec(
            kind="gauss_marker",
            selector=SlabSelector(component="stress_xx"),
            style=GaussMarkerStyle(),
        ),
        results,
    )
    try:
        diagram.attach(PyVistaQtBackend(plotter), fem, scene)
        diagram.detach()
    finally:
        plotter.close()
