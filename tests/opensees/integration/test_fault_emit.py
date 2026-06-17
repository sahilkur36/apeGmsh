"""MT-4 (ADR 0062) end-to-end — ops.fault.* → one pattern per subfault.

A finite fault becomes N ``Plain`` patterns (one per subfault), each a
``Yoffe`` moment function at the subfault's rupture time carrying a single
``moment_tensor``. Verifies the per-pattern emit recovers each subfault's
moment tensor (Σ_a x_a⊗F_a = M_k) and that distinct subfaults get distinct
time series.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.geometry._moment_tensor import unit_moment_tensor
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.ns.fault import WarnFaultSubfaultSkipped
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.nd import ElasticIsotropic


@pytest.fixture(scope="module")
def box_fem():
    """A 2×2×2 structured hex box (deck metres)."""
    g = apeGmsh(model_name="fault_emit", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0, label="soil")
        g.physical.add(3, "soil", name="soil")
        g.mesh.structured.set_transfinite("soil", n=3)
        g.mesh.generation.generate(dim=3)
        yield g.mesh.queries.get_fem_data()
    finally:
        g.end()


CRUST = dict(thickness=np.array([1000.0]), vs=np.array([1000.0]),
             rho=np.array([2000.0]))          # μ = 2e9 Pa
AREA = 1.0e6
LSCALE = 1000.0                                # km → m
MSCALE = 1.0e-3                                # N·m → kN·m
MU = 2.0e9


def _patterns_loads(rec):
    """Group emitted load lines by enclosing pattern_open/close block."""
    blocks, cur = [], None
    for name, args, _ in rec.calls:
        if name == "pattern_open":
            cur = []
        elif name == "pattern_close":
            if cur is not None:
                blocks.append(cur)
            cur = None
        elif name == "load" and cur is not None:
            cur.append(args)
    return blocks


def _first_moment(fem, load_args):
    coord_of = {int(t): np.asarray(c, float)
                for t, c in zip(fem.nodes.ids, fem.nodes.coords)}
    nodes = [int(a[0]) for a in load_args]
    forces = np.array([[float(v) for v in a[1:4]] for a in load_args])
    xs = np.array([coord_of[n] for n in nodes])
    xs = xs - xs.mean(axis=0)
    return forces, xs.T @ forces


def _emit(ops):
    rec = RecordingEmitter()
    ops.build().emit(rec)
    return rec


def test_from_ffsp_one_pattern_per_subfault(box_fem):
    subfaults = dict(
        x=np.array([0.0005, 0.0015]),     # km → 0.5, 1.5 m
        y=np.array([0.0005, 0.0015]),
        z=np.array([0.0005, 0.0015]),     # depth km → +0.5, +1.5 m (z-down)
        slip=np.array([0.5, 1.0]),
        strike=np.array([350.0, 30.0]),
        dip=np.array([40.0, 45.0]),
        rake=np.array([113.0, 90.0]),
        rupture_time=np.array([0.0, 0.3]),
        rise_time=np.array([1.0, 1.0]),
        peak_time=np.array([0.2, 0.2]),
    )
    ops = apeSees(box_fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.register(ElasticIsotropic(E=1.0e7, nu=0.25, rho=2000.0))
    ops.element.stdBrick(pg="soil", material=mat)
    patterns = ops.fault.from_ffsp(
        subfaults, CRUST, frame="z-down", area_m2=AREA, dt=0.02, t_total=4.0,
        length_scale=LSCALE, moment_scale=MSCALE, method="consistent",
    )
    assert len(patterns) == 2

    rec = _emit(ops)
    blocks = _patterns_loads(rec)
    assert len(blocks) == 2                       # one pattern per subfault
    # distinct time series — two Path emissions
    ts = [c for c in rec.calls if c[0] == "timeSeries"]
    assert len(ts) == 2

    # each pattern's load lines recover that subfault's moment tensor
    M0 = [MU * AREA * 0.5 * MSCALE, MU * AREA * 1.0 * MSCALE]
    mech = [(350.0, 40.0, 113.0), (30.0, 45.0, 90.0)]
    # blocks are emitted in pattern-registration order (subfault order)
    for blk, m0, (s, d, r) in zip(blocks, M0, mech):
        M = m0 * unit_moment_tensor(strike=s, dip=d, rake=r)  # z-down, no flip
        forces, fm = _first_moment(box_fem, blk)
        assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-6 * np.abs(forces).max())
        assert np.allclose(fm, M, rtol=1e-6, atol=1.0)


def test_from_ffsp_skips_zero_slip_subfault(box_fem):
    subfaults = dict(
        x=np.array([0.001, 0.001]), y=np.array([0.001, 0.001]),
        z=np.array([0.001, 0.001]),
        slip=np.array([0.5, 0.0]),                # second has no slip
        strike=np.array([0.0, 0.0]), dip=np.array([90.0, 90.0]),
        rake=np.array([0.0, 0.0]),
        rupture_time=np.array([0.0, 0.0]),
        rise_time=np.array([1.0, 1.0]), peak_time=np.array([0.2, 0.2]),
    )
    ops = apeSees(box_fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.register(ElasticIsotropic(E=1.0e7, nu=0.25, rho=2000.0))
    ops.element.stdBrick(pg="soil", material=mat)
    with pytest.warns(WarnFaultSubfaultSkipped, match="skipped 1"):
        patterns = ops.fault.from_ffsp(
            subfaults, CRUST, frame="z-down", area_m2=AREA, dt=0.02,
            t_total=4.0, length_scale=LSCALE, moment_scale=MSCALE,
        )
    assert len(patterns) == 1


def test_from_shakermaker_duck_typed_radians(box_fem):
    """from_shakermaker reads .x/.angles(RAD)/.tt without importing SM."""
    class _PS:
        def __init__(self, x, angles_deg, tt):
            self.x = np.asarray(x)
            self.angles = np.radians(angles_deg)   # ShakerMaker stores radians
            self.tt = tt

    fault = [
        _PS([0.0005, 0.0005, 0.001], [350.0, 40.0, 113.0], 0.0),
        _PS([0.0015, 0.0015, 0.001], [30.0, 45.0, 90.0], 0.5),
    ]
    ops = apeSees(box_fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.register(ElasticIsotropic(E=1.0e7, nu=0.25, rho=2000.0))
    ops.element.stdBrick(pg="soil", material=mat)
    patterns = ops.fault.from_shakermaker(
        fault, frame="z-down", M0=1.0e12, rise_time=1.0, peak_time=0.2,
        dt=0.02, t_total=4.0, length_scale=LSCALE, method="consistent",
    )
    assert len(patterns) == 2

    rec = _emit(ops)
    blocks = _patterns_loads(rec)
    assert len(blocks) == 2
    # first source: angles recovered as degrees → its moment tensor
    M = 1.0e12 * unit_moment_tensor(strike=350.0, dip=40.0, rake=113.0)
    forces, fm = _first_moment(box_fem, blocks[0])
    assert np.allclose(fm, M, rtol=1e-6, atol=1.0)
