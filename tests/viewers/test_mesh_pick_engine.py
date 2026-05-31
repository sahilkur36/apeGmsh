"""ADR 0047 R-D.2a — mesh ``PickEngine`` domain-resolution adapters.

The box-select path is covered by ``tests/test_box_select_*``; this pins
the *new* domain glue introduced when the VTK gesture machine moved
behind ``PyVistaPickBackend``: ``_resolve_hit`` gating (prop→DimTag,
pickable-dim, hidden), hover dedup-by-entity, and install/uninstall
delegation. Fully headless — a stub registry + an injected stub backend
(the engine's geometry is the backend's job, mocked here).
"""
from __future__ import annotations

from apeGmsh.viewers.core.pick_engine import PickEngine
from apeGmsh.viewers.scene_ir import PickHit, PickModifiers


class _StubRegistry:
    dims = [0, 1, 2, 3]

    def __init__(self, mapping: dict) -> None:
        self._map = mapping  # (prop_id, cell_id) -> DimTag

    def resolve_pick(self, prop_id, cell_id):
        return self._map.get((prop_id, cell_id))


class _StubBackend:
    """Records install/uninstall; no geometry."""

    def __init__(self) -> None:
        self.installed = None
        self.uninstalled = 0
        self._drag_threshold = 8
        self._click_picker = "CLICK"
        self._hover_picker = "HOVER"

    def install(self, *, on_pick, on_hover=None, on_box=None):
        self.installed = {"on_pick": on_pick, "on_hover": on_hover, "on_box": on_box}

    def uninstall(self):
        self.uninstalled += 1


def _engine(mapping, *, pickable=None, hidden=None, backend=None):
    reg = _StubRegistry(mapping)
    e = PickEngine(object(), reg, pick_backend=backend or _StubBackend())
    if pickable is not None:
        e.set_pickable_dims(pickable)
    if hidden is not None:
        e.set_hidden_check(hidden)
    return e


def _hit(prop_id, cell_id):
    return PickHit(world=(0.0, 0.0, 0.0), cell_id=cell_id, prop_id=prop_id)


# ── click resolution + gating ───────────────────────────────────────

def test_pick_resolves_and_fires_with_ctrl():
    e = _engine({(7, 3): (2, 5)})
    picks = []
    e.on_pick = lambda dt, ctrl: picks.append((dt, ctrl))
    e._on_geom_pick(_hit(7, 3), PickModifiers(ctrl=True))
    assert picks == [((2, 5), True)]


def test_pick_miss_none_hit_does_not_fire():
    e = _engine({})
    picks = []
    e.on_pick = lambda dt, ctrl: picks.append(dt)
    e._on_geom_pick(None, PickModifiers())
    assert picks == []


def test_pick_miss_no_prop_does_not_fire():
    e = _engine({})
    picks = []
    e.on_pick = lambda dt, ctrl: picks.append(dt)
    e._on_geom_pick(_hit(None, 0), PickModifiers())
    assert picks == []


def test_pick_unresolved_registry_does_not_fire():
    e = _engine({(7, 3): None})  # registry returns None
    picks = []
    e.on_pick = lambda dt, ctrl: picks.append(dt)
    e._on_geom_pick(_hit(7, 3), PickModifiers())
    assert picks == []


def test_pick_outside_pickable_dims_does_not_fire():
    e = _engine({(7, 3): (3, 5)}, pickable={2})  # volume hit, only surfaces pickable
    picks = []
    e.on_pick = lambda dt, ctrl: picks.append(dt)
    e._on_geom_pick(_hit(7, 3), PickModifiers())
    assert picks == []


def test_pick_hidden_entity_does_not_fire():
    e = _engine({(7, 3): (2, 5)}, hidden=lambda dt: dt == (2, 5))
    picks = []
    e.on_pick = lambda dt, ctrl: picks.append(dt)
    e._on_geom_pick(_hit(7, 3), PickModifiers())
    assert picks == []


# ── hover dedup-by-entity ───────────────────────────────────────────

def test_hover_fires_once_per_entity_change():
    e = _engine({(7, 3): (2, 5), (7, 9): (2, 8)})
    hovers = []
    e.on_hover = hovers.append
    e._on_geom_hover(_hit(7, 3))          # -> (2,5)
    e._on_geom_hover(_hit(7, 3))          # same entity, no re-fire
    e._on_geom_hover(_hit(7, 9))          # -> (2,8)
    e._on_geom_hover(None)                # miss -> None
    assert hovers == [(2, 5), (2, 8), None]
    assert e.hover_entity is None


def test_hover_miss_after_hit_fires_none():
    e = _engine({(7, 3): (2, 5)})
    hovers = []
    e.on_hover = hovers.append
    e._on_geom_hover(_hit(7, 3))
    e._on_geom_hover(_hit(1, 1))  # unresolved -> None
    assert hovers == [(2, 5), None]


# ── install / uninstall delegation + escape hatch ───────────────────

def test_install_delegates_geometric_callbacks_to_backend():
    b = _StubBackend()
    e = _engine({}, backend=b)
    e.install()
    assert b.installed is not None
    assert b.installed["on_pick"] == e._on_geom_pick
    assert b.installed["on_hover"] == e._on_geom_hover
    assert b.installed["on_box"] == e._on_geom_box


def test_uninstall_delegates_to_backend():
    b = _StubBackend()
    e = _engine({}, backend=b)
    e.uninstall()
    assert b.uninstalled == 1


def test_click_hover_picker_proxy_to_backend():
    b = _StubBackend()
    e = _engine({}, backend=b)
    assert e._click_picker == "CLICK"
    assert e._hover_picker == "HOVER"


def test_drag_threshold_setter_updates_backend():
    b = _StubBackend()
    e = _engine({}, backend=b)
    e.drag_threshold = 15
    assert e.drag_threshold == 15
    assert b._drag_threshold == 15
