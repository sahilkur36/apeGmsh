"""
Navigation — Custom camera-control interactor for pyGmshViewer.

Replicates the mouse/keyboard scheme from pyGmsh's BaseViewer so all
viewers feel identical:

    LMB click/drag   : reserved for PyVista picking (probes, etc.)
    MIDDLE drag       : pan camera
    Shift+MIDDLE drag : quaternion orbit around scene centre
    RIGHT drag        : pan camera
    SCROLL            : zoom (default VTK behaviour, not intercepted)

The implementation uses high-priority VTK observers (priority 10.0)
that compose reliably with pyvistaqt's QtInteractor without subclassing
the interactor style.
"""
from __future__ import annotations

import math

import pyvista as pv


def install_navigation(plotter: pv.Plotter) -> None:
    """Install BaseViewer-compatible camera controls on *plotter*.

    Call once after the plotter/QtInteractor is created and before
    data is loaded.  LMB events are **not** intercepted so that
    PyVista's ``enable_point_picking`` / ``enable_cell_picking`` work
    normally for the probe system.
    """
    import vtk

    iren_wrap = plotter.iren
    iren = iren_wrap.interactor
    renderer = plotter.renderer

    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)

    # ── orbit state ─────────────────────────────────────────────────
    _orbit_pivot: list[tuple | None] = [None]
    _orbit_last:  list[tuple | None] = [None]

    def _scene_center():
        try:
            b = plotter.renderer.ComputeVisiblePropBounds()
            if b[1] > b[0]:
                return (
                    (b[0] + b[1]) * 0.5,
                    (b[2] + b[3]) * 0.5,
                    (b[4] + b[5]) * 0.5,
                )
        except Exception:
            pass
        return (0.0, 0.0, 0.0)

    def _orbit_around(pivot, dx_px, dy_px):
        """Orbit camera around *pivot* using unit-quaternion rotation."""
        cam = renderer.GetActiveCamera()
        pos = cam.GetPosition()
        fp  = cam.GetFocalPoint()
        up  = cam.GetViewUp()

        az_rad = -dx_px * 0.005
        el_rad =  dy_px * 0.005

        def quat_from_axis_angle(ax, ang):
            ha = ang * 0.5
            s = math.sin(ha)
            return (math.cos(ha), ax[0] * s, ax[1] * s, ax[2] * s)

        def quat_mul(a, b):
            return (
                a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
                a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
                a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
                a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
            )

        def quat_conj(q):
            return (q[0], -q[1], -q[2], -q[3])

        def quat_rotate(q, v):
            vq = (0.0, v[0], v[1], v[2])
            r = quat_mul(quat_mul(q, vq), quat_conj(q))
            return (r[1], r[2], r[3])

        vd = [fp[i] - pos[i] for i in range(3)]
        d = math.sqrt(sum(v * v for v in vd))
        if d < 1e-12:
            return
        vd = [v / d for v in vd]

        right = [
            vd[1] * up[2] - vd[2] * up[1],
            vd[2] * up[0] - vd[0] * up[2],
            vd[0] * up[1] - vd[1] * up[0],
        ]
        rn = math.sqrt(sum(v * v for v in right))
        if rn < 1e-12:
            return
        right = [v / rn for v in right]

        un = math.sqrt(sum(v * v for v in up))
        up_n = [v / un for v in up] if un > 1e-12 else [0, 0, 1]

        q_az = quat_from_axis_angle(up_n, az_rad)
        q_el = quat_from_axis_angle(right, el_rad)
        q = quat_mul(q_el, q_az)

        pv_ = pivot
        p_rel  = (pos[0] - pv_[0], pos[1] - pv_[1], pos[2] - pv_[2])
        fp_rel = (fp[0] - pv_[0],  fp[1] - pv_[1],  fp[2] - pv_[2])

        rp  = quat_rotate(q, p_rel)
        rfp = quat_rotate(q, fp_rel)

        cam.SetPosition(pv_[0] + rp[0],  pv_[1] + rp[1],  pv_[2] + rp[2])
        cam.SetFocalPoint(pv_[0] + rfp[0], pv_[1] + rfp[1], pv_[2] + rfp[2])
        cam.OrthogonalizeViewUp()
        renderer.ResetCameraClippingRange()

    # ── abort helper ────────────────────────────────────────────────
    _tag_lmb_press:   list[int] = [0]
    _tag_move:        list[int] = [0]
    _tag_lmb_release: list[int] = [0]
    _tag_mmb_press:   list[int] = [0]
    _tag_mmb_release: list[int] = [0]
    _tag_rmb_press:   list[int] = [0]
    _tag_rmb_release: list[int] = [0]

    def _abort(caller, tag: int) -> None:
        cmd = caller.GetCommand(tag)
        if cmd is not None:
            cmd.SetAbortFlag(1)

    # ── LMB: block trackball rotation, let picking through ─────────
    # We absorb LMB so the trackball style doesn't rotate the scene.
    # PyVista's enable_point_picking installs its OWN observer at a
    # lower priority, so it still receives the event after us.
    _picking_active: list[bool] = [False]

    def on_lmb_press(caller, _event):
        if _picking_active[0]:
            # Let PyVista's picker handle it — don't abort
            return
        _abort(caller, _tag_lmb_press[0])

    def on_lmb_release(caller, _event):
        if _picking_active[0]:
            return
        _abort(caller, _tag_lmb_release[0])

    # ── mouse move: orbit or pass through ──────────────────────────
    def on_mouse_move(caller, _event):
        if _orbit_pivot[0] is not None and _orbit_last[0] is not None:
            px, py = caller.GetEventPosition()
            lx, ly = _orbit_last[0]
            _orbit_last[0] = (px, py)
            _orbit_around(_orbit_pivot[0], px - lx, py - ly)
            plotter.render()
            _abort(caller, _tag_move[0])
            return
        # Don't abort — let trackball style handle pan moves,
        # and PyVista handle pick hover

    # ── MMB: pan / Shift+orbit ─────────────────────────────────────
    def on_mmb_press(caller, _event):
        if caller.GetShiftKey():
            _orbit_pivot[0] = _scene_center()
            _orbit_last[0] = caller.GetEventPosition()
        else:
            _orbit_pivot[0] = None
            style.StartPan()
        _abort(caller, _tag_mmb_press[0])

    def on_mmb_release(caller, _event):
        if _orbit_pivot[0] is not None:
            _orbit_pivot[0] = None
            _orbit_last[0] = None
            _abort(caller, _tag_mmb_release[0])
            return
        state = style.GetState()
        if state == 2:  # VTKIS_PAN
            style.EndPan()
        _abort(caller, _tag_mmb_release[0])

    # ── RMB: pan ───────────────────────────────────────────────────
    def on_rmb_press(caller, _event):
        style.StartPan()
        _abort(caller, _tag_rmb_press[0])

    def on_rmb_release(caller, _event):
        state = style.GetState()
        if state == 2:
            style.EndPan()
        _abort(caller, _tag_rmb_release[0])

    # ── register observers ─────────────────────────────────────────
    _tag_lmb_press[0] = iren.AddObserver(
        "LeftButtonPressEvent", on_lmb_press, 10.0,
    )
    _tag_move[0] = iren.AddObserver(
        "MouseMoveEvent", on_mouse_move, 10.0,
    )
    _tag_lmb_release[0] = iren.AddObserver(
        "LeftButtonReleaseEvent", on_lmb_release, 10.0,
    )
    _tag_mmb_press[0] = iren.AddObserver(
        "MiddleButtonPressEvent", on_mmb_press, 10.0,
    )
    _tag_mmb_release[0] = iren.AddObserver(
        "MiddleButtonReleaseEvent", on_mmb_release, 10.0,
    )
    _tag_rmb_press[0] = iren.AddObserver(
        "RightButtonPressEvent", on_rmb_press, 10.0,
    )
    _tag_rmb_release[0] = iren.AddObserver(
        "RightButtonReleaseEvent", on_rmb_release, 10.0,
    )

    # ── public hook for probe system ───────────────────────────────
    def set_picking_active(active: bool) -> None:
        """Tell the navigation layer whether picking is active.

        When True, LMB events pass through to PyVista's picker.
        When False (default), LMB is absorbed to prevent trackball
        rotation.
        """
        _picking_active[0] = active

    # Attach to the plotter so the probe engine can toggle it
    plotter._nav_set_picking = set_picking_active
