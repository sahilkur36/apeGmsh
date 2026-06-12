"""ADR 0058 S2c — picking disambiguation across concurrent geometries.

Under S2b every geometry with ``visible=True`` renders its own
substrate actor pair over its own ``FEMSceneData`` — but pick
resolution still assumed ONE scene. S2c makes picking geometry-aware:

* the viewer keeps ``_actor_scenes`` (``id(substrate actor) ->
  (geometry_id, scene)``) in lockstep with ``_scene_actors``;
* ``install_results_pick`` takes a ``scene_resolver`` so cell→element
  / node / box-candidate reads come from the HIT geometry's scene
  (its deformed grid), not the install-time boot scene;
* the pick IR (:class:`PickResult` / :class:`BoxPickResult` /
  :class:`PointProbeResult`) widens additively with ``geometry_id``
  (ADR 0047 precedent — old constructors keep working);
* :class:`ProbeOverlay` / :class:`LocalAxesOverlay` resolve their
  scene at use time instead of holding the boot scene forever
  (the S2a known gap).

The qt-marked test drives a real viewer with two visible geometries
(B deformed) and asserts a pick landing on B's actor resolves
against B's grid and reports B's geometry id.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.viewers.core.results_pick import (
    BoxPickResult,
    PickResult,
    install_results_pick,
)
from apeGmsh.viewers.core.results_pick_engine import PickInventory
from apeGmsh.viewers.diagrams._geometries import GeometryManager
from apeGmsh.viewers.scene_ir import BoxGesture, PickHit, PickModifiers


# =====================================================================
# Stubs (mirror tests/viewers/test_results_pick.py)
# =====================================================================

class _StubBackend:
    def __init__(self, project=None) -> None:
        self._project = project
        self.on_pick = None
        self.on_box = None

    def install(self, *, on_pick, on_hover=None, on_box=None) -> None:
        self.on_pick = on_pick
        self.on_box = on_box

    def project_points(self, pts):
        if self._project is not None:
            return self._project(pts)
        return np.asarray(pts, dtype=np.float64)[:, :2]

    def uninstall(self) -> None:
        pass

    def fire_pick(self, hit, mods=None) -> None:
        self.on_pick(hit, mods or PickModifiers())

    def fire_box(self, box) -> None:
        self.on_box(BoxGesture(box=box, crossing=box[2] < box[0]))


class _Grid:
    def __init__(self, points, centers=()):
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self._centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        self.cell_data = {}

    def cell_centers(self):
        return type("C", (), {"points": self._centers})()


class _Scene:
    """Minimal FEMSceneData stand-in."""

    def __init__(self, *, cell_to_element_id=None, node_ids=None,
                 grid=None, cell_dim=None, element_id_to_cell=None,
                 inventory=None) -> None:
        self.cell_to_element_id = np.asarray(
            cell_to_element_id if cell_to_element_id is not None
            else [1001, 1002, 1003], dtype=np.int64,
        )
        self.node_ids = np.asarray(
            node_ids if node_ids is not None else [10, 20, 30],
            dtype=np.int64,
        )
        self.grid = grid
        self.cell_dim = np.asarray(
            cell_dim if cell_dim is not None else [], dtype=np.int8,
        )
        self.element_id_to_cell = element_id_to_cell or {}
        self.pick_engine = inventory


def _install(scene, *, scene_resolver=None, gp_candidates=None,
             backend=None):
    seen, boxes = [], []
    backend = backend or _StubBackend()
    ctrl = install_results_pick(
        None, seen.append, scene=scene, on_box_pick=boxes.append,
        gp_candidates=gp_candidates, pick_backend=backend,
        scene_resolver=scene_resolver,
    )
    return ctrl, backend, seen, boxes


def _hit(prop_id=None, cell_id=0, world=(0.0, 0.0, 0.0)):
    return PickHit(world=world, cell_id=cell_id, prop_id=prop_id)


# =====================================================================
# Pick IR — additive geometry_id widening (ADR 0047 precedent)
# =====================================================================

def test_pick_result_geometry_id_is_additive():
    # Old construction (no geometry_id) keeps working → defaults None.
    old = PickResult(kind="node", world=(0.0, 0.0, 0.0))
    assert old.geometry_id is None
    new = PickResult(
        kind="element", world=(0.0, 0.0, 0.0),
        element_id=7, cell_id=0, geometry_id="g1",
    )
    assert new.geometry_id == "g1"


def test_box_pick_result_geometry_id_is_additive():
    empty = np.zeros(0, dtype=np.int64)
    old = BoxPickResult(
        kind="node", ids=empty, cell_ids=empty, gp_indices=empty,
        box=(0, 0, 1, 1), crossing=False,
    )
    assert old.geometry_id is None
    new = BoxPickResult(
        kind="node", ids=empty, cell_ids=empty, gp_indices=empty,
        box=(0, 0, 1, 1), crossing=False, geometry_id="g2",
    )
    assert new.geometry_id == "g2"


def test_point_probe_result_geometry_id_is_additive():
    from apeGmsh.viewers.overlays.probe_overlay import PointProbeResult

    old = PointProbeResult(
        position=np.zeros(3),
        closest_node_id=1,
        closest_coord=np.zeros(3),
        distance=0.0,
        step_index=0,
        field_values={},
    )
    assert old.geometry_id is None
    assert "geometry_id" not in old.summary()    # display unchanged


# =====================================================================
# Controller — hit-scene resolution through scene_resolver
# =====================================================================

def _two_scene_resolver(prop_to_entry):
    def resolver(prop_id):
        return prop_to_entry.get(prop_id)
    return resolver


def test_element_pick_resolves_against_hit_scene():
    scene_a = _Scene(cell_to_element_id=[1001, 1002, 1003])
    scene_b = _Scene(cell_to_element_id=[2001, 2002, 2003])
    ctrl, backend, seen, _ = _install(
        scene_a,
        scene_resolver=_two_scene_resolver({555: ("gB", scene_b)}),
    )
    ctrl.set_mode("element")
    backend.fire_pick(_hit(prop_id=555, cell_id=1))
    assert seen[0].element_id == 2002          # B's array, not A's
    assert seen[0].geometry_id == "gB"


def test_node_pick_carries_hit_geometry_id():
    scene_a = _Scene()
    scene_b = _Scene()
    ctrl, backend, seen, _ = _install(
        scene_a,
        scene_resolver=_two_scene_resolver({555: ("gB", scene_b)}),
    )
    backend.fire_pick(_hit(prop_id=555, cell_id=0, world=(1.0, 2.0, 3.0)))
    assert seen[0].kind == "node"
    assert seen[0].geometry_id == "gB"
    # An unresolved prop keeps geometry_id None.
    backend.fire_pick(_hit(prop_id=999, cell_id=0))
    assert seen[1].geometry_id is None


def test_unresolved_prop_falls_back_to_install_scene():
    scene_a = _Scene(cell_to_element_id=[1001, 1002, 1003])
    ctrl, backend, seen, _ = _install(
        scene_a, scene_resolver=lambda prop_id: None,
    )
    ctrl.set_mode("element")
    backend.fire_pick(_hit(prop_id=999, cell_id=1))
    assert seen[0].element_id == 1002
    assert seen[0].geometry_id is None


def test_raising_resolver_falls_back_to_install_scene():
    def _boom(_prop_id):
        raise RuntimeError("resolver exploded")

    scene_a = _Scene(cell_to_element_id=[1001, 1002, 1003])
    ctrl, backend, seen, _ = _install(scene_a, scene_resolver=_boom)
    ctrl.set_mode("element")
    backend.fire_pick(_hit(prop_id=1, cell_id=0))
    assert seen[0].element_id == 1001
    assert seen[0].geometry_id is None


def test_no_resolver_keeps_single_scene_behaviour():
    ctrl, backend, seen, _ = _install(
        _Scene(cell_to_element_id=[1001, 1002, 1003]),
    )
    ctrl.set_mode("element")
    backend.fire_pick(_hit(prop_id=999, cell_id=2))
    assert seen[0].element_id == 1003
    assert seen[0].geometry_id is None


def test_dim_gate_reads_hit_scene_cell_dim():
    # Install scene has no dims (gate inert); the HIT scene's dims must
    # be the ones consulted.
    scene_a = _Scene(cell_to_element_id=[1001, 1002])
    scene_b = _Scene(cell_to_element_id=[2001, 2002], cell_dim=[1, 2])
    ctrl, backend, seen, _ = _install(
        scene_a,
        scene_resolver=_two_scene_resolver({555: ("gB", scene_b)}),
    )
    ctrl.set_mode("element")
    ctrl.active_dims = frozenset({2})
    backend.fire_pick(_hit(prop_id=555, cell_id=0))   # B cell 0 is dim 1
    assert seen == []
    backend.fire_pick(_hit(prop_id=555, cell_id=1))   # B cell 1 is dim 2
    assert seen[0].element_id == 2002


def test_gp_marker_element_pick_keeps_geometry_id_none():
    inv = PickInventory()
    gp_actor = object()
    inv.register_actor(gp_actor, "gp", lambda c: (1002, 4, (9.0, 9.0, 9.0)))
    scene_a = _Scene(
        cell_to_element_id=[1001, 1002, 1003],
        element_id_to_cell={1002: 1},
        inventory=inv,
    )
    ctrl, backend, seen, _ = _install(
        scene_a, scene_resolver=lambda prop_id: None,
    )
    ctrl.set_mode("element")
    backend.fire_pick(_hit(prop_id=id(gp_actor), cell_id=0))
    assert seen[0].element_id == 1002
    assert seen[0].geometry_id is None   # overlay actor — geometry unknown


def test_box_pick_resolves_active_scene_via_resolver_none():
    """A box has no hit actor — the resolver is consulted with ``None``
    and must hand back the ACTIVE geometry's scene (its deformed
    points are what get projected)."""
    proj = lambda pts: np.asarray(pts, dtype=np.float64)[:, :2]  # noqa: E731
    grid_a = _Grid(points=[[0, 0, 0], [100, 100, 0]])
    grid_b = _Grid(points=[[5, 5, 0], [200, 200, 0]])
    scene_a = _Scene(node_ids=[10, 20], grid=grid_a)
    scene_b = _Scene(node_ids=[30, 40], grid=grid_b)
    ctrl, backend, _, boxes = _install(
        scene_a,
        scene_resolver=lambda prop_id: (
            ("gB", scene_b) if prop_id is None else None
        ),
        backend=_StubBackend(project=proj),
    )
    backend.fire_box((-1, -1, 10, 10))
    assert boxes[0].ids.tolist() == [30]     # B's node inside the box
    assert boxes[0].geometry_id == "gB"


# =====================================================================
# Viewer — _resolve_pick_scene / _scene_for_geometry_id /
# _pick_geometry_label (bound onto a stub namespace, no Qt)
# =====================================================================

class _NS:
    pass


def _viewer_ns():
    """Stub viewer namespace: two geometries, per-geometry scenes."""
    from apeGmsh.viewers.results_viewer import ResultsViewer

    gm = GeometryManager()
    geom_a = gm.active
    geom_b = gm.add("Geometry B", make_active=False)
    scene_a, scene_b = _Scene(), _Scene()
    scenes = {geom_a.id: scene_a, geom_b.id: scene_b}
    director = SimpleNamespace(
        geometries=gm,
        scene_for=lambda geom: scenes.get(getattr(geom, "id", None)),
    )
    ns = _NS()
    ns._director = director
    ns._scene = scene_a                     # active scene (A is active)
    fill_b, wf_b = object(), object()
    ns._actor_scenes = {
        id(fill_b): (geom_b.id, scene_b),
        id(wf_b): (geom_b.id, scene_b),
    }
    ns._resolve_pick_scene = ResultsViewer._resolve_pick_scene.__get__(ns)
    ns._scene_for_geometry_id = (
        ResultsViewer._scene_for_geometry_id.__get__(ns)
    )
    ns._pick_geometry_label = (
        ResultsViewer._pick_geometry_label.__get__(ns)
    )
    return ns, gm, geom_a, geom_b, scene_a, scene_b, fill_b


def test_resolve_pick_scene_registered_actor_maps_to_its_geometry():
    ns, _, _, geom_b, _, scene_b, fill_b = _viewer_ns()
    assert ns._resolve_pick_scene(id(fill_b)) == (geom_b.id, scene_b)


def test_resolve_pick_scene_none_prop_maps_to_active_geometry():
    ns, _, geom_a, _, scene_a, _, _ = _viewer_ns()
    assert ns._resolve_pick_scene(None) == (geom_a.id, scene_a)


def test_resolve_pick_scene_unknown_actor_keeps_geometry_unknown():
    ns, _, _, _, scene_a, _, _ = _viewer_ns()
    # Coordinates still read off the active scene, but the geometry is
    # NOT claimed (the actor could belong to any overlay).
    assert ns._resolve_pick_scene(123456789) == (None, scene_a)


def test_scene_for_geometry_id_resolves_and_falls_back():
    ns, _, _, geom_b, scene_a, scene_b, _ = _viewer_ns()
    assert ns._scene_for_geometry_id(geom_b.id) is scene_b
    assert ns._scene_for_geometry_id(None) is scene_a
    assert ns._scene_for_geometry_id("no-such-id") is scene_a


def test_pick_geometry_label_only_when_multiple_visible():
    ns, gm, geom_a, geom_b, _, _, _ = _viewer_ns()
    # Two visible geometries → label.
    assert ns._pick_geometry_label(geom_b.id) == "Geometry B"
    # One visible → suppressed (mirrors the scalar-bar prefix rule).
    gm.set_visible(geom_a.id, False)
    assert ns._pick_geometry_label(geom_b.id) is None
    gm.set_visible(geom_a.id, True)
    assert ns._pick_geometry_label(None) is None
    assert ns._pick_geometry_label("no-such-id") is None


# =====================================================================
# ProbeOverlay — use-time scene resolution + per-pick scene override
# =====================================================================

class _SnapScene:
    """Stand-in scene for snap reads: grid.points + node_ids only."""

    def __init__(self, points, node_ids):
        self.grid = SimpleNamespace(
            points=np.asarray(points, dtype=np.float64),
        )
        self.node_ids = np.asarray(node_ids, dtype=np.int64)
        self.model_diagonal = 1.0
        self.node_tree = None

    def ensure_node_tree(self):
        return None    # force the brute-force snap path


def _probe_overlay(boot_scene, director=None):
    from apeGmsh.viewers.overlays.probe_overlay import ProbeOverlay

    director = director or SimpleNamespace(
        registry=SimpleNamespace(diagrams=lambda: []),
        read_at_pick=lambda nid, comps: {},
        step_index=0,
    )
    return ProbeOverlay(None, boot_scene, director)


def test_snap_to_nearest_node_honours_scene_override():
    boot = _SnapScene([[0.0, 0.0, 0.0]], [10])
    hit = _SnapScene([[5.0, 0.0, 0.0]], [99])
    overlay = _probe_overlay(boot)
    nid, snapped, _ = overlay._snap_to_nearest_node(
        np.array([5.1, 0.0, 0.0]), scene=hit,
    )
    assert nid == 99
    np.testing.assert_allclose(snapped, [5.0, 0.0, 0.0])
    # Default (no override, stub director) falls back to the boot scene.
    nid, snapped, _ = overlay._snap_to_nearest_node(
        np.array([5.1, 0.0, 0.0]),
    )
    assert nid == 10


def test_probe_overlay_scene_property_tracks_active_geometry():
    boot = _SnapScene([[0.0, 0.0, 0.0]], [10])
    active_scene = _SnapScene([[1.0, 0.0, 0.0]], [20])
    gm = GeometryManager()
    director = SimpleNamespace(
        geometries=gm,
        scene_for=lambda geom: active_scene,
        registry=SimpleNamespace(diagrams=lambda: []),
        read_at_pick=lambda nid, comps: {},
        step_index=0,
    )
    overlay = _probe_overlay(boot, director=director)
    assert overlay._scene is active_scene
    # Stub director without geometries → boot fallback.
    assert _probe_overlay(boot)._scene is boot


def test_probe_at_point_carries_geometry_id_and_hit_scene_coord():
    boot = _SnapScene([[0.0, 0.0, 0.0]], [10])
    hit = _SnapScene([[7.0, 0.0, 0.0]], [42])
    overlay = _probe_overlay(boot)
    result = overlay.probe_at_point(
        (7.2, 0.0, 0.0), scene=hit, geometry_id="gB",
    )
    assert result.geometry_id == "gB"
    assert result.closest_node_id == 42
    np.testing.assert_allclose(result.closest_coord, [7.0, 0.0, 0.0])


# =====================================================================
# LocalAxesOverlay — use-time scene resolution
# =====================================================================

def test_local_axes_overlay_scene_resolves_at_use_time():
    from apeGmsh.viewers.overlays.local_axes_overlay import LocalAxesOverlay

    boot = _SnapScene([[0.0, 0.0, 0.0]], [10])
    active_scene = _SnapScene([[1.0, 0.0, 0.0]], [10])
    gm = GeometryManager()
    director = SimpleNamespace(
        geometries=gm, scene_for=lambda geom: active_scene,
    )
    overlay = LocalAxesOverlay(None, boot, director)
    assert overlay._scene is active_scene
    # Node coords read the resolved scene's grid.
    np.testing.assert_allclose(overlay._node_coord(10), [1.0, 0.0, 0.0])
    # Stub director without geometries → boot fallback.
    bare = LocalAxesOverlay(
        None, boot, SimpleNamespace(view=None, results=None, stage_id=None),
    )
    assert bare._scene is boot


# =====================================================================
# PickReadoutHUD — geometry label gating (no Qt: bind the helper)
# =====================================================================

def test_hud_geometry_label_only_when_multiple_visible():
    from apeGmsh.viewers.ui._pick_readout_hud import PickReadoutHUD

    gm = GeometryManager()
    geom_a = gm.active
    geom_b = gm.add("Geometry B", make_active=False)
    ns = _NS()
    ns._director = SimpleNamespace(geometries=gm)
    label = PickReadoutHUD._geometry_label.__get__(ns)

    assert label(geom_b.id) == "Geometry B"
    gm.set_visible(geom_a.id, False)
    assert label(geom_b.id) is None          # single visible → suppressed
    gm.set_visible(geom_a.id, True)
    assert label(None) is None
    assert label("no-such-id") is None


# =====================================================================
# Qt — pick on a second (deformed) geometry resolves the hit scene
# (local-only; -m qt)
# =====================================================================

@pytest.fixture
def deforming_results(g, tmp_path: Path):
    """Tiny native Results with a non-zero displacement field."""
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter
    from tests.conftest import _open_model_from_h5

    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    n_nodes = len(fem.nodes.ids)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    path = tmp_path / "s2c.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static",
            time=np.array([0.0, 0.5, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={
                "displacement_x": np.ones((3, n_nodes)),
            },
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


@pytest.mark.qt
def test_pick_on_second_geometry_resolves_hit_scene(deforming_results):
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    pytest.importorskip("qtpy.QtWidgets").QApplication.instance() \
        or pytest.importorskip("qtpy.QtWidgets").QApplication([])
    from qtpy import QtCore

    from apeGmsh.viewers.results_viewer import ResultsViewer

    viewer = ResultsViewer(
        deforming_results, title="s2c-picking",
        restore_session=False, save_session=False,
    )
    seen: dict = {}

    def _drive_then_close():
        try:
            director = viewer._director
            geoms = director.geometries
            geom_a = geoms.active
            geom_b = geoms.add("Geometry B", make_active=False)
            # B deformed (+3 in x), A at reference — both visible.
            geoms.set_deformation(
                geom_b.id, enabled=True,
                field="displacement", scale=3.0,
            )
            scene_b = director.scene_for(geom_b)
            fill_b, wf_b = viewer._scene_actors[geom_b.id]

            # Registration: both of B's actors map to (geom_b, scene_b).
            seen["map_fill"] = (
                viewer._actor_scenes.get(id(fill_b))
                == (geom_b.id, scene_b)
            )
            seen["map_wf"] = (
                viewer._actor_scenes.get(id(wf_b))
                == (geom_b.id, scene_b)
            )

            backend = viewer._pick_controller._backend
            # ── Node pick on B's actor: snap reads B's DEFORMED grid
            # and the result carries B's geometry id. ──
            captured: list = []
            viewer._probe_overlay.on_point_result = captured.append
            b_node0 = np.asarray(scene_b.grid.points)[0]
            seen["b_actually_deformed"] = bool(
                abs(b_node0[0] - scene_b.reference_points[0][0]) > 2.9
            )
            viewer._pick_controller.set_mode("node")
            backend._on_pick(
                PickHit(
                    world=tuple(b_node0 + 0.01),
                    cell_id=0,
                    prop_id=id(fill_b),
                ),
                PickModifiers(),
            )
            result = captured[-1] if captured else None
            seen["node_geometry_id"] = (
                result is not None
                and result.geometry_id == geom_b.id
            )
            # Snapped coordinate is B's deformed position, NOT the
            # reference (= what the boot scene would have returned).
            seen["node_snap_deformed"] = (
                result is not None
                and float(
                    np.linalg.norm(
                        np.asarray(result.closest_coord) - b_node0,
                    )
                ) < 1e-9
            )

            # ── Element pick on B's actor: highlight extracts from
            # B's grid (its deformed bounds, x shifted by +3). ──
            viewer._pick_controller.set_mode("element")
            backend._on_pick(
                PickHit(
                    world=tuple(b_node0),
                    cell_id=0,
                    prop_id=id(fill_b),
                ),
                PickModifiers(),
            )
            hl = viewer._element_pick_highlight_actor
            seen["highlight_on_b_grid"] = (
                hl is not None and hl.GetBounds()[0] >= 2.5
            )

            # ── Removal drops the actor→scene entries. ──
            geoms.remove(geom_b.id)
            seen["map_dropped"] = (
                id(fill_b) not in viewer._actor_scenes
                and id(wf_b) not in viewer._actor_scenes
            )
            seen["boot_entries_remain"] = any(
                gid == geom_a.id
                for (gid, _s) in viewer._actor_scenes.values()
            )
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(400, _drive_then_close)
    viewer.show()

    assert seen.get("map_fill") is True
    assert seen.get("map_wf") is True
    assert seen.get("b_actually_deformed") is True
    assert seen.get("node_geometry_id") is True
    assert seen.get("node_snap_deformed") is True
    assert seen.get("highlight_on_b_grid") is True
    assert seen.get("map_dropped") is True
    assert seen.get("boot_entries_remain") is True
