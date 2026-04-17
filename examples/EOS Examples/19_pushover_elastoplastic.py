# %% [markdown]
# # 19 — Elastoplastic Pushover (Uniaxial Bar)
#
# **Curriculum slot:** Tier 5, slot 19.
# **Prerequisite:** 06 — Sections Catalog, 18 — Buckling.
#
# ## Purpose
#
# Pushover is displacement-controlled nonlinear static analysis —
# the standard tool for estimating the capacity curve (base shear
# vs roof displacement) of a structure up to collapse. Two new
# ingredients compared to earlier slots:
#
# 1. A **nonlinear material law**. Slot 06 used ``uniaxialMaterial
#    Elastic``; here we switch to ``Steel01`` for an
#    elastic-perfectly-plastic response.
# 2. **Displacement control** as the OpenSees integrator.
#    Instead of ``LoadControl(d_lambda)``, the integrator
#    ``DisplacementControl(node, dof, dU)`` increments a target
#    DOF's displacement and computes the reaction force.
#
# ## Problem
#
# A one-dimensional bar of length $L$ and cross section $A$, made
# of Steel01 material with Young's modulus $E$, yield stress
# $f_y$, and zero post-yield hardening ($b = 0$). The left end is
# fixed; the right end is pulled by a monotonically increasing
# displacement $u$. The reaction at the fixed end traces the
# capacity curve:
#
# $$
# F(u) \;=\; \begin{cases}
#   E\,A\,u/L, & u < u_y \\
#   f_y\,A = F_y, & u \geq u_y
# \end{cases}
# \quad\text{where}\quad u_y = \dfrac{f_y\,L}{E}.
# $$
#
# Two checks:
#
# 1. **Yield onset** — bar yields when $u = u_y$.
# 2. **Yield plateau** — force stays exactly $F_y$ for $u > u_y$.

# %% [markdown]
# ## 1. Imports and parameters

# %%
import numpy as np
import openseespy.opensees as ops

from apeGmsh import apeGmsh

L = 1.0                     # bar length [m]
A = 1.0e-4                  # cross section [m^2]
E = 2.0e11                  # Young's modulus [Pa]
fy = 300.0e6                # yield stress [Pa]
Fy = fy * A                 # force at yield [N]
uy = fy * L / E             # displacement at yield [m]

# Displacement control: ramp up to 4 * uy in small steps
N_STEPS = 80
u_max = 4 * uy
dU = u_max / N_STEPS

N_ELEM = 10
LC = L / N_ELEM


# %% [markdown]
# ## 2. Geometry

# %%
g_ctx = apeGmsh(model_name="19_pushover", verbose=False)
g = g_ctx.__enter__()

p_L = g.model.geometry.add_point(0.0, 0.0, 0.0, lc=LC)
p_R = g.model.geometry.add_point(L,   0.0, 0.0, lc=LC)
ln  = g.model.geometry.add_line(p_L, p_R)
g.model.sync()

g.physical.add(0, [p_L], name="left")
g.physical.add(0, [p_R], name="right")
g.physical.add(1, [ln],  name="bar")

g.mesh.structured.set_transfinite_curve(ln, N_ELEM + 1)
g.mesh.generation.generate(1)
fem = g.mesh.queries.get_fem_data()
print(f"mesh: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")


# %% [markdown]
# ## 3. OpenSees build
#
# Bar modelled as ``truss`` elements (1 DOF = axial). The truss's
# material is ``Steel01(E=E, Fy=Fy, b=0.0)`` — elastic up to
# $F_y$, perfectly plastic beyond. Using a 1D model (``-ndm 1
# -ndf 1``) keeps the problem clean.

# %%
ops.wipe()
ops.model("basic", "-ndm", 1, "-ndf", 1)

# Nodes (use only x coordinate in 1D)
for nid, xyz in fem.nodes.get():
    ops.node(int(nid), float(xyz[0]))

# Material
mat_tag = 1
ops.uniaxialMaterial("Steel01", mat_tag, fy, E, 0.0)    # b = 0 (no hardening)

# Truss elements
for group in fem.elements.get(target="bar"):
    for eid, nodes in zip(group.ids, group.connectivity):
        ops.element("truss", int(eid),
                    int(nodes[0]), int(nodes[1]),
                    A, mat_tag)

# BCs
left_id  = int(next(iter(fem.nodes.get(target="left").ids)))
right_id = int(next(iter(fem.nodes.get(target="right").ids)))
ops.fix(left_id, 1)


# %% [markdown]
# ## 4. Pushover driver — displacement control
#
# ``DisplacementControl(node, dof, dU)`` increments the chosen
# DOF's displacement by ``dU`` every step. We record the reaction
# at the fixed end (= the internal force in the bar) at each
# step — that's the capacity curve.

# %%
# Reference load at the right end: a tiny pull. DisplacementControl
# uses the load pattern's *shape*, not its magnitude, so we apply
# a unit force just to define the controlled DOF.
ops.timeSeries("Linear", 1)
ops.pattern("Plain", 1, 1)
ops.load(right_id, 1.0)        # unit pull in +x

ops.system("BandGeneral")
ops.numberer("Plain")
ops.constraints("Plain")
ops.test("NormDispIncr", 1e-10, 20)
ops.algorithm("Newton")
ops.integrator("DisplacementControl", right_id, 1, dU)
ops.analysis("Static")

u_hist:  list[float] = []
F_hist:  list[float] = []
for k in range(N_STEPS):
    status = ops.analyze(1)
    if status != 0:
        print(f"  diverged at step {k + 1}")
        break
    ops.reactions()
    u = ops.nodeDisp(right_id, 1)
    F = -ops.nodeReaction(left_id, 1)     # force ON the bar at the left
    u_hist.append(u)
    F_hist.append(F)

u_hist = np.asarray(u_hist)
F_hist = np.asarray(F_hist)
print(f"converged {len(u_hist)}/{N_STEPS} steps, u_max = {u_hist[-1]:.4e} m")


# %% [markdown]
# ## 5. Verification
#
# Three printed checks: elastic modulus recovered from the
# pre-yield slope; yield point matches $u_y$; plateau stays at $F_y$.

# %%
# --- elastic modulus from the pre-yield linear slope ---
# Points safely in the elastic regime: u < 0.8 * u_y
elastic_mask = u_hist < 0.8 * uy
slope = np.polyfit(u_hist[elastic_mask], F_hist[elastic_mask], 1)[0]
E_recovered = slope * L / A
err_E = abs(E_recovered - E) / E * 100.0

# --- yield plateau force: mean of last 20% of steps (safely post-yield) ---
tail = F_hist[int(0.8 * len(F_hist)):]
F_plateau = float(np.mean(tail))
err_F = abs(F_plateau - Fy) / Fy * 100.0

# --- yield onset: find first step where F >= 99% * Fy ---
idx_yield = next(i for i, f in enumerate(F_hist) if f >= 0.99 * Fy)
u_at_yield = u_hist[idx_yield]
err_uy = abs(u_at_yield - uy) / uy * 100.0

print("Pre-yield elastic modulus")
print(f"  Analytical  : E    = {E:.4e}  Pa")
print(f"  FEM-recovered: {E_recovered:.4e}  Pa")
print(f"  Error       : {err_E:.4f} %")
print()
print("Yield plateau force (last 20% of steps)")
print(f"  Analytical  : F_y  = {Fy:.4e}  N")
print(f"  FEM plateau : {F_plateau:.4e}  N")
print(f"  Error       : {err_F:.4f} %")
print()
print(f"Yield onset: first step with F >= 0.99 * F_y -> u = {u_at_yield:.4e} m")
print(f"  analytical u_y = {uy:.4e} m")
print(f"  relative    : {err_uy:.2f} %")


# %% [markdown]
# ## 6. (Optional) plot the capacity curve

# %%
# import matplotlib.pyplot as plt
# plt.figure(figsize=(7, 5))
# plt.plot(u_hist, F_hist, 'o-', label='FEM')
# plt.axhline(Fy, ls='--', color='r', label=f'F_y = {Fy:.1e}')
# plt.axvline(uy, ls=':', color='g', label=f'u_y = {uy:.1e}')
# plt.xlabel('u [m]'); plt.ylabel('F [N]'); plt.grid(True, alpha=0.3)
# plt.title('Elastoplastic bar capacity curve')
# plt.legend(); plt.show()


# %% [markdown]
# ## What this unlocks
#
# * **Displacement-controlled pushover** via
#   ``ops.integrator("DisplacementControl", node, dof, dU)``. Any
#   later pushover notebook (fiber-section RC column, ductile
#   steel frame, shear wall) uses the same control structure:
#   apply a reference load whose *shape* defines the load
#   distribution, then increment a target DOF's displacement.
# * **Steel01** — the simplest useful nonlinear uniaxial law in
#   OpenSees. Parameters ``(Fy, E, b)`` map directly to yield
#   force, elastic modulus, and post-yield hardening ratio.
#   ``b = 0`` gives the textbook elastic-perfectly-plastic
#   response used here; real steel has ``b`` ≈ 0.01-0.02.
# * **Capacity curve extraction** via ``ops.nodeReaction`` step by
#   step. For a multi-DOF frame the "base shear" is the sum of
#   horizontal reactions at the fixed base nodes.

# %%
g_ctx.__exit__(None, None, None)
