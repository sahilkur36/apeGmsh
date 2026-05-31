"""
PickEngine — mesh/BREP domain pick controller (ADR 0047 R-D.2a).

The *domain* half of the picking seam: it resolves a geometric hit to a
BREP :class:`DimTag` via the :class:`EntityRegistry`, applies the
pickable-dim and hidden-entity gates, dedups hover, and runs box-select
candidate sourcing — then fires the caller's callbacks.  It owns **no
VTK**: the ``vtkCellPicker``, the LMB press/move/release gesture machine,
the rubber-band overlay, and the screen↔world geometry all live behind
:class:`~apeGmsh.viewers.backends._pyvista_pick.PyVistaPickBackend`
(ADR 0042/0047 ``PickBackend``).  This is the same domain/backend split
the render seam drew for drawing, now drawn for picking (INV-3).

Shared verbatim by ``mesh_viewer`` and ``model_viewer`` (both resolve to
BREP ``(dim, tag)``); the public API is unchanged from the pre-seam
engine so neither viewer nor the box-select tests needed to change.

Usage::

    engine = PickEngine(plotter, registry)
    engine.on_pick = lambda dt, ctrl: selection.toggle(dt)
    engine.on_hover = lambda dt: color_mgr.set_entity_state(...)
    engine.on_box_select = lambda dts, ctrl: ...
    engine.install()
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv
    from apeGmsh._types import DimTag
    from ..scene_ir._pick import BoxGesture, PickHit, PickModifiers
    from .entity_registry import EntityRegistry


def _entity_in_box(
    sx: np.ndarray,
    sy: np.ndarray,
    bx0: float,
    bx1: float,
    by0: float,
    by1: float,
    crossing: bool,
) -> bool:
    """Box-select containment predicate for one entity's projected points.

    Window mode:   entity's projected 2D AABB is fully inside the box.
    Crossing mode: any sampled point is inside the box (classic).

    Crossing uses sample-point-inside rather than AABB overlap because
    ``bbox()`` for curves/surfaces returns the 3D AABB's 8 corners,
    whose projected 2D AABB can be much larger than the visible
    silhouette at angled views — pure AABB overlap over-selected.
    The sample-density workaround (64 points per volume) makes the
    classic test tight enough in practice.
    """
    if crossing:
        inside = (bx0 <= sx) & (sx <= bx1) & (by0 <= sy) & (sy <= by1)
        return bool(np.any(inside))
    return bool(
        bx0 <= sx.min() and sx.max() <= bx1
        and by0 <= sy.min() and sy.max() <= by1
    )


class PickEngine:
    """BREP/mesh pick controller over a :class:`PyVistaPickBackend`."""

    # Type declarations for __slots__ (consumed by mypy / pyright).
    _plotter: Any
    _registry: EntityRegistry
    _backend: Any
    _pickable_dims: set[int]
    _hidden_check: Callable[["DimTag"], bool]
    _drag_threshold: int
    on_pick: Callable[["DimTag", bool], None] | None
    on_hover: Callable[["DimTag | None"], None] | None
    on_box_select: Callable[[list["DimTag"], bool], None] | None
    _hover_id: "DimTag | None"

    __slots__ = (
        "_plotter",
        "_registry",
        "_backend",
        "_pickable_dims",
        "_hidden_check",
        "_drag_threshold",
        # Callbacks
        "on_pick",
        "on_hover",
        "on_box_select",
        # Internal state
        "_hover_id",
    )

    def __init__(
        self,
        plotter: "pv.Plotter",
        registry: "EntityRegistry",
        *,
        drag_threshold: int = 8,
        pick_backend: Any = None,
    ) -> None:
        self._plotter = plotter
        self._registry = registry
        self._pickable_dims: set[int] = set(registry.dims)
        self._hidden_check: Callable[["DimTag"], bool] = lambda _: False
        self._drag_threshold = drag_threshold

        # The geometry/gesture backend (ADR 0047). Injectable for headless
        # tests; otherwise a PyVistaPickBackend over the plotter. Built
        # eagerly so the box-select methods (driven directly in tests,
        # without install()) have it.
        if pick_backend is None:
            from ..backends._pyvista_pick import PyVistaPickBackend

            pick_backend = PyVistaPickBackend(
                plotter, drag_threshold=drag_threshold
            )
        self._backend = pick_backend

        self.on_pick: Callable[["DimTag", bool], None] | None = None
        self.on_hover: Callable[["DimTag | None"], None] | None = None
        self.on_box_select: Callable[[list["DimTag"], bool], None] | None = None

        self._hover_id: "DimTag | None" = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_pickable_dims(self, dims: set[int]) -> None:
        self._pickable_dims = set(dims)

    def set_hidden_check(self, fn: Callable[["DimTag"], bool]) -> None:
        """Set a function that returns True if an entity should be skipped."""
        self._hidden_check = fn

    @property
    def drag_threshold(self) -> int:
        return self._drag_threshold

    @drag_threshold.setter
    def drag_threshold(self, value: int) -> None:
        self._drag_threshold = max(2, value)
        self._backend._drag_threshold = self._drag_threshold

    # ------------------------------------------------------------------
    # Install / teardown (delegate the desktop event face to the backend)
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Install LMB press/move/release picking via the backend."""
        self._backend.install(
            on_pick=self._on_geom_pick,
            on_hover=self._on_geom_hover,
            on_box=self._on_geom_box,
        )

    def uninstall(self) -> None:
        """Remove the backend's observers + overlay. Idempotent.

        Closes the observer leak the pre-seam engine carried (it had no
        teardown path)."""
        self._backend.uninstall()

    # ------------------------------------------------------------------
    # Geometric-callback adapters (backend → domain resolution)
    # ------------------------------------------------------------------

    def _resolve_hit(self, hit: "PickHit | None") -> "DimTag | None":
        """Resolve a geometric hit to a pickable, non-hidden BREP DimTag."""
        if hit is None or hit.prop_id is None:
            return None
        dt = self._registry.resolve_pick(hit.prop_id, hit.cell_id)
        if dt is None:
            return None
        if dt[0] not in self._pickable_dims:
            return None
        if self._hidden_check(dt):
            return None
        return dt

    def _on_geom_pick(self, hit: "PickHit | None", mods: "PickModifiers") -> None:
        dt = self._resolve_hit(hit)
        if dt is not None and self.on_pick is not None:
            self.on_pick(dt, mods.ctrl)

    def _on_geom_hover(self, hit: "PickHit | None") -> None:
        new_dt = self._resolve_hit(hit)
        if new_dt == self._hover_id:
            return
        self._hover_id = new_dt
        if self.on_hover is not None:
            self.on_hover(new_dt)

    def _on_geom_box(self, gesture: "BoxGesture") -> None:
        x0, y0, x1, y1 = gesture.box
        self._do_box(x0, y0, x1, y1, gesture.modifiers.ctrl)

    # ------------------------------------------------------------------
    # Box-select (domain candidate sourcing over the backend's geometry)
    # ------------------------------------------------------------------

    def _do_box(self, x0: int, y0: int, x1: int, y1: int, ctrl: bool) -> None:
        """Box-select with proper window vs crossing modes.

        Candidate sourcing (which entities, which representative points)
        is domain logic; the world→display projection and the exact 3D
        frustum test are the backend's geometry (ADR 0047 INV-3).

        Set env var ``APEGMSH_DEBUG_BOX=1`` to log per-entity projection
        and hit results for diagnosing crossing-mode misses.
        """
        import os
        _debug = bool(os.environ.get("APEGMSH_DEBUG_BOX"))

        crossing = x1 < x0
        bx0 = min(x0, x1)
        bx1 = max(x0, x1)
        by0 = min(y0, y1)
        by1 = max(y0, y1)

        if _debug:
            print(
                f"[box] mode={'crossing' if crossing else 'window'} "
                f"event=({x0},{y0})->({x1},{y1}) "
                f"box=[{bx0}..{bx1}]x[{by0}..{by1}]",
                flush=True,
            )

        hits: list["DimTag"] = []

        # Collect all pickable entities and their representative points
        entities: list["DimTag"] = []
        all_corners: list[np.ndarray] = []
        corner_counts: list[int] = []

        for dt in self._registry.all_entities():
            if dt[0] not in self._pickable_dims:
                continue
            if self._hidden_check(dt):
                continue
            # Prefer actual mesh vertices (tight to silhouette) over
            # 3D AABB corners (loose — projects to a rectangle much
            # wider than curves/surfaces at angled views, causing
            # phantom hits on nearby boxes).
            if dt[0] >= 1:
                pts = self._registry.entity_points(dt)
                if pts is not None and len(pts) > 0:
                    entities.append(dt)
                    all_corners.append(pts)
                    corner_counts.append(len(pts))
                    continue
            bbox = self._registry.bbox(dt)
            if bbox is not None:
                corners = bbox.corners8   # canonical BBox → (8, 3) (ADR 0045)
                entities.append(dt)
                all_corners.append(corners)
                corner_counts.append(len(corners))
            else:
                c = self._registry.centroid(dt)
                if c is not None:
                    entities.append(dt)
                    all_corners.append(c.reshape(1, 3))
                    corner_counts.append(1)

        if entities:
            pts_all = np.vstack(all_corners)
            planes = self._box_frustum_planes(bx0, by0, bx1, by1)
            if planes is not None:
                # ADR 0045 S5-box: exact 3D frustum half-space test. A
                # point is inside the screen box's frustum iff it is on
                # the inner side of all six planes — tighter than the 2D
                # projection at angled views and respects near/far clip.
                # One matmul over all points preserves the ~20x perf.
                dist = pts_all @ planes[:, :3].T + planes[:, 3]   # (N, 6)
                inside_all = np.all(dist >= -1e-9, axis=1)        # (N,)
                offset = 0
                for i, dt in enumerate(entities):
                    n = corner_counts[i]
                    ins = inside_all[offset:offset + n]
                    if crossing:
                        hit = bool(np.any(ins))
                    else:
                        hit = bool(np.all(ins)) if n else False
                    if _debug:
                        print(
                            f"[box]   dt={dt} n={n} inside={int(ins.sum())}"
                            f"/{n} -> {'HIT' if hit else 'miss'} (frustum)",
                            flush=True,
                        )
                    if hit:
                        hits.append(dt)
                    offset += n
            else:
                # Fallback (no camera / un-projection unavailable): the
                # legacy 2D screen-box projection — the parity oracle.
                xy = self._backend.project_points(pts_all)
                screen_x = xy[:, 0]
                screen_y = xy[:, 1]
                offset = 0
                for i, dt in enumerate(entities):
                    n = corner_counts[i]
                    sx = screen_x[offset:offset + n]
                    sy = screen_y[offset:offset + n]
                    hit = _entity_in_box(sx, sy, bx0, bx1, by0, by1, crossing)
                    if _debug:
                        print(
                            f"[box]   dt={dt} n={n} "
                            f"sx=[{sx.min():.1f}..{sx.max():.1f}] "
                            f"sy=[{sy.min():.1f}..{sy.max():.1f}] "
                            f"-> {'HIT' if hit else 'miss'} (2d)",
                            flush=True,
                        )
                    if hit:
                        hits.append(dt)
                    offset += n

        if _debug:
            print(f"[box] total hits={len(hits)}: {hits}", flush=True)

        if self.on_box_select is not None:
            self.on_box_select(hits, ctrl)

    def _box_frustum_planes(self, bx0: int, by0: int, bx1: int, by1: int):
        """Six inward frustum planes for the screen box, or ``None``.

        Delegates the un-projection to the backend (ADR 0047); kept as a
        method for the box-select smoke test and the ``_do_box`` caller.
        ``APEGMSH_BOX_2D=1`` forces the 2D-projection parity path."""
        return self._backend.frustum_planes((bx0, by0, bx1, by1))

    # ------------------------------------------------------------------
    # Raw-picker escape hatch for FE element/node modes
    # ------------------------------------------------------------------
    # ``mesh_viewer``/``model_viewer`` read ``GetCellId()`` /
    # ``GetPickPosition()`` off these to resolve FE element / node tags
    # in element/node pick mode (the engine itself only ever resolves a
    # BREP DimTag). The picker state is fresh: the backend ran ``Pick``
    # for the same event just before firing the pick/hover callback.

    @property
    def _click_picker(self) -> Any:
        return self._backend._click_picker

    @property
    def _hover_picker(self) -> Any:
        return self._backend._hover_picker

    @property
    def hover_entity(self) -> "DimTag | None":
        return self._hover_id
