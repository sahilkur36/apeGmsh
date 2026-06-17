"""Live MT-5b — moment-tensor source double-couple radiation pattern.

Builds a homogeneous box, embeds a vertical strike-slip double-couple at
the centre, runs a live transient, and reads the P-wave first-motion radial
velocity on a horizontal ring of receivers. The azimuthal pattern must
match the analytic double-couple radiation ``A_P(γ) = γ·M·γ`` (∝ sin 2θ —
four lobes, nodal planes along x and y), confirming the equivalent nodal
forces radiate the correct mechanism through a real OpenSees run.

Gated by the ``live`` marker + openseespy (skips in CI).
"""
from __future__ import annotations

import numpy as np
import pytest

openseespy = pytest.importorskip("openseespy.opensees")

from apeGmsh import apeGmsh  # noqa: E402
from apeGmsh._kernel.geometry._moment_tensor import unit_moment_tensor  # noqa: E402
from apeGmsh.opensees import apeSees  # noqa: E402
from apeGmsh.opensees.emitter.live import LiveOpsEmitter  # noqa: E402
from apeGmsh.opensees.material.nd import ElasticIsotropic  # noqa: E402


@pytest.mark.live
def test_double_couple_radiation_pattern():
    L = 40.0
    res = 10                                   # elements per side
    ctr = np.array([L / 2, L / 2, L / 2])
    E, nu, rho = 2.0e8, 0.25, 2000.0
    dt, t_total = 5e-4, 0.12                   # through the direct P arrival
    nsteps = int(round(t_total / dt))

    g = apeGmsh(model_name="mt_rad_live", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0, 0, 0, L, L, L, label="soil")
        g.physical.add(3, "soil", name="soil")
        g.mesh.structured.set_transfinite("soil", n=res + 1)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data()
    finally:
        g.end()

    ids = np.asarray(fem.nodes.ids)
    coords = np.asarray(fem.nodes.coords, dtype=float)

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.register(ElasticIsotropic(E=E, nu=nu, rho=rho))
    ops.element.stdBrick(pg="soil", material=mat)
    src = ops.timeSeries.MomentStep(
        half_duration=0.008, t_total=t_total, dt=dt, t0=0.02,
    )
    with ops.pattern.Plain(series=src) as p:
        p.moment_tensor(
            position=tuple(ctr), frame="z-down", M0=1.0e8,
            mech=dict(strike=0, dip=90, rake=0),
            method="consistent", region="soil",
        )
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-8, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.Newmark(gamma=0.5, beta=0.25)
    ops.analysis.Transient()

    # ring of receivers at the source depth
    r = 10.0
    azis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring_nodes, ring_dirs = [], []
    for a in azis:
        target = ctr + np.array([r * np.cos(a), r * np.sin(a), 0.0])
        j = int(np.argmin(np.linalg.norm(coords - target, axis=1)))
        d = coords[j] - ctr
        ring_nodes.append(int(ids[j]))
        ring_dirs.append(d / np.linalg.norm(d))

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    o = emitter.ops
    vr = {n: [] for n in ring_nodes}
    for _ in range(nsteps):
        assert emitter.analyze(steps=1, dt=dt) == 0
        for n, d in zip(ring_nodes, ring_dirs):
            v = np.array([o.nodeVel(n, k) for k in (1, 2, 3)])
            vr[n].append(float(v @ d))

    fem_amp = np.array([max(np.asarray(vr[n]), key=abs) for n in ring_nodes])
    M = unit_moment_tensor(strike=0, dip=90, rake=0)
    analytic = np.array([d @ (M @ d) for d in ring_dirs])     # γ·M·γ = sin 2θ

    fa = fem_amp / np.abs(fem_amp).max()
    an = analytic / np.abs(analytic).max()
    corr = float(np.corrcoef(fa, an)[0, 1])
    # The decisive gate: the FEM first-motion azimuthal pattern matches the
    # analytic double-couple radiation γ·M·γ — computed from each receiver's
    # *actual* direction, so robust to grid-snapping. The four lobes, the
    # nodal planes, and the quadrant sign-alternation are all encoded in this
    # single correlation (≈0.90 at this resolution, ≈0.97 refined).
    assert corr > 0.85, f"radiation-pattern correlation too low: {corr:.3f}"
