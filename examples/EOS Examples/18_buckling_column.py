# %% [markdown]
# # 18 — Column Buckling via Imperfection + Nonlinear Geometry
#
# **Curriculum slot:** Tier 5, slot 18.
# **Prerequisite:** 02 — Cantilever Beam, 17 — Modal Analysis.
#
# ## Purpose
#
# Bifurcation buckling is hard to resolve with a pure linear
# analysis — a perfectly straight column under pure axial load
# has no lateral force to activate bending, so the static
# solution converges at every load and misses the critical point.
# The standard workaround used in every FEM textbook:
#
# 1. **Add a small initial imperfection** (mid-height lateral
#    offset) that couples axial and bending.
# 2. **Use a geometrically nonlinear transformation**
#    (``Corotational`` here) so the geometric stiffness captures
#    P-Δ amplification.
# 3. **Incrementally load** and watch the lateral deflection blow
#    up as $P \to P_{\text{cr}}$.
# 4. **Southwell plot** — a linear fit of $\delta$ vs $\delta/P$
#    gives $P_{\text{cr}}$ as the slope, extrapolating cleanly
#    even from data well below buckling.
#
# ## Problem
#
# A pin-pin slender column of length $L$, square cross-section,
# loaded axially in compression. The classical Euler critical load
# is
#
# $$
# P_{\text{cr}} \;=\; \dfrac{\pi^{2}\,E\,I}{L^{2}}.
# $$
#
# We include a mid-height imperfection $\delta_{0}$ (sinusoidal
# shape) and ramp $P$ from 0 up to 90% of the analytical $P_{\text{cr}}$
# in 30 load steps. At each step we record the mid-height lateral
# deflection. The Southwell plot of $\delta$ vs $\delta / P$ is
# a straight line whose slope is $P_{\text{cr}}$ — that's the
# verification.

# %% [markdown]
# ## 1. Imports and parameters

# %%
import numpy as np
import openseespy.opensees as ops

from apeGmsh import apeGmsh

# Geometry + material
L = 3.0
E = 2.1e11
nu = 0.3
G = E / (2 * (1 + nu))
A  = 1.0e-3
I  = 1.0e-5                 # same for both bending axes
J  = 2.0e-5

# Analytical Euler
P_cr_analytical = np.pi**2 * E * I / L**2

# Imperfection — a small lateral offset at mid-height (sinusoidal)
# so we activate the first bending mode. delta_0 / L ~ 1/500 is
# a typical "built-in" imperfection magnitude.
delta_0 = L / 500.0

N_ELEM = 30
LC = L / N_ELEM


# %% [markdown]
# ## 2. Geometry + mesh (perfect straight line; imperfection is added at the nodal coordinates during ingest)

# %%
g_ctx = apeGmsh(model_name="18_buckling", verbose=False)
g = g_ctx.__enter__()

# Column along +z so "axial" = z and "lateral" = x/y. Pin at z=0,
# roller at z=L.
p_bot = g.model.geometry.add_point(0.0, 0.0, 0.0, lc=LC)
p_top = g.model.geometry.add_point(0.0, 0.0, L,   lc=LC)
ln    = g.model.geometry.add_line(p_bot, p_top)
g.model.sync()

g.physical.add(0, [p_bot], name="bottom")
g.physical.add(0, [p_top], name="top")
g.physical.add(1, [ln],    name="column")

g.mesh.structured.set_transfinite_curve(ln, N_ELEM + 1)
g.mesh.generation.generate(1)
fem = g.mesh.queries.get_fem_data()
print(f"mesh: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")


# %% [markdown]
# ## 3. OpenSees ingest with a nodal-coordinate imperfection
#
# For each non-end node at height $z$, we shift its $x$-coordinate
# by $\delta_{0} \sin(\pi z / L)$ — the first buckling eigenmode
# of a pin-pin column. End nodes stay on the axis so the
# boundary conditions remain clean.

# %%
ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 6)

tag_to_idx = {int(t): i for i, t in enumerate(fem.nodes.ids)}
bot_id = int(next(iter(fem.nodes.get(target="bottom").ids)))
top_id = int(next(iter(fem.nodes.get(target="top").ids)))
end_ids = {bot_id, top_id}

for nid, xyz in fem.nodes.get():
    nid = int(nid)
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    if nid not in end_ids:
        x += delta_0 * np.sin(np.pi * z / L)
    ops.node(nid, x, y, z)

# Corotational geomTransf — the key ingredient. Linear would give
# no P-Δ amplification.
ops.geomTransf("Corotational", 1, 1.0, 0.0, 0.0)        # vecxz = +x

for group in fem.elements.get(target="column"):
    for eid, nodes in zip(group.ids, group.connectivity):
        ops.element("elasticBeamColumn", int(eid),
                    int(nodes[0]), int(nodes[1]),
                    A, E, G, J, I, I, 1)

# Pin at bottom (fix ux, uy, uz; allow all rotations),
# roller at top (fix ux, uy; allow uz, rotations).
ops.fix(bot_id, 1, 1, 1, 0, 0, 0)
ops.fix(top_id, 1, 1, 0, 0, 0, 0)

# Suppress out-of-plane motion (uy, rx, rz) on all internal nodes so
# bending stays in xz plane. This matches the 1D imperfection shape.
for nid in fem.nodes.ids:
    nid = int(nid)
    if nid in end_ids:
        continue
    ops.fix(nid, 0, 1, 0, 1, 0, 1)


# %% [markdown]
# ## 4. Incremental compression with nonlinear static

# %%
# Load path: apply a reference compression at the top node; use
# LoadControl with delta_lambda to step from 0 to lambda_max, where
# lambda corresponds to axial load P = -lambda * P_unit.
P_unit = 1.0                   # unit axial load [N]
lambda_max = 0.90 * P_cr_analytical
N_STEPS    = 30

ops.timeSeries("Linear", 1)
ops.pattern("Plain", 1, 1)
ops.load(top_id, 0.0, 0.0, -P_unit, 0.0, 0.0, 0.0)    # downward (-z) = compression

# Pick a mid-height node for deflection tracking — find node closest to z = L/2.
mid_id = None
for nid in fem.nodes.ids:
    idx = tag_to_idx[int(nid)]
    z = fem.nodes.coords[idx, 2]
    if abs(z - L/2) < LC * 0.51:
        mid_id = int(nid)
        break
assert mid_id is not None
print(f"mid-height tracking node: {mid_id}")

ops.system("BandGeneral")
ops.numberer("Plain")
ops.constraints("Plain")
ops.test("NormDispIncr", 1e-8, 20)
ops.algorithm("Newton")
d_lambda = lambda_max / N_STEPS
ops.integrator("LoadControl", d_lambda)
ops.analysis("Static")

loads:    list[float] = []
defls_x:  list[float] = []
for step in range(N_STEPS):
    status = ops.analyze(1)
    if status != 0:
        print(f"  diverged at step {step + 1}")
        break
    P_now = (step + 1) * d_lambda
    dx = ops.nodeDisp(mid_id, 1)   # mid-height lateral in x
    # Total deflection = initial imperfection + FEM dx
    total_x = delta_0 + dx
    loads.append(P_now)
    defls_x.append(total_x)

loads   = np.asarray(loads)
defls_x = np.asarray(defls_x)
print(f"converged {len(loads)}/{N_STEPS} steps")
print(f"max lateral @ last step : {defls_x[-1]:.6e} m")
print(f"amplification           : {defls_x[-1] / delta_0:.2f} × the imperfection")


# %% [markdown]
# ## 5. Extract $P_{\text{cr}}$ from the amplification
#
# For a column with imperfection $\delta_{0}$, small-deflection
# theory gives $\delta(P) = \delta_{0} / (1 - P/P_{\text{cr}})$.
# Solving for $P_{\text{cr}}$:
#
# $$
# P_{\text{cr}} \;=\; \dfrac{P}{1 - \delta_{0}/\delta(P)}.
# $$
#
# Evaluated at each load step, the estimate converges to the
# true $P_{\text{cr}}$ as $P$ approaches it (amplification
# grows, so the division becomes numerically well-conditioned).
# We report the average of the last 5 steps.

# %%
est_tail = loads[-5:] / (1.0 - delta_0 / defls_x[-5:])
P_cr_fit = float(np.mean(est_tail))

err = abs(P_cr_fit - P_cr_analytical) / P_cr_analytical * 100.0

print(f"P_cr analytical  : {P_cr_analytical:.6e}  N   (pi^2 E I / L^2)")
print(f"P_cr from fit    : {P_cr_fit:.6e}  N")
print(f"Error            : {err:.4f} %")


# %% [markdown]
# ## 6. (Optional) plot the load-deflection + Southwell line

# %%
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# axes[0].plot(defls_x, loads, 'o-')
# axes[0].axhline(P_cr_analytical, ls='--', color='r', label='P_cr analytical')
# axes[0].set(xlabel='mid-height lateral [m]', ylabel='P [N]',
#             title='Load-deflection')
# axes[0].legend(); axes[0].grid()
# axes[1].plot(x_southwell, y_southwell, 'o')
# xs = np.linspace(x_southwell.min(), x_southwell.max(), 10)
# axes[1].plot(xs, slope * xs + intercept, '--', label=f'fit slope 1/P_cr_fit')
# axes[1].set(xlabel='delta [m]', ylabel='delta / P [m/N]',
#             title='Southwell plot')
# axes[1].legend(); axes[1].grid()
# plt.tight_layout(); plt.show()


# %% [markdown]
# ## What this unlocks
#
# * **Imperfection + Corotational** as the standard recipe for
#   bifurcation problems in OpenSees. A perfectly straight column
#   under a pure axial load won't buckle in the solver; an
#   imperfection + geomTransf = "Corotational" produces the
#   amplification required for the classical $P_{\text{cr}}$.
# * **Southwell plot.** Even without running all the way to $P_{\text{cr}}$
#   (which is numerically fragile), the linear fit of $\delta/P$ vs
#   $\delta$ recovers $P_{\text{cr}}$ to within a percent.
# * **Pattern for nodal imperfections.** Modify nodal coords at
#   ingest time with a sinusoidal perturbation in the expected
#   mode shape. For LTB of an I-beam the perturbation would be
#   a combined lateral + twist pattern; the bookkeeping is the
#   same.

# %%
g_ctx.__exit__(None, None, None)
