# %% [markdown]
# # 02 — 2D Cantilever Beam
#
# **Curriculum slot:** Tier 1, slot 02.
# **Prerequisite:** 01 — Hello Plate.
#
# ## Problem statement
#
# A prismatic cantilever of length $L$ is fixed at $x=0$ and loaded at
# its free tip with a vertical point force $P$ (downward).
#
# ```
#    P↓
#    ●────────────────●        +z
#   (tip)  L = 3 m   (base)    │
#                              └── +x
# ```
#
# For linear elastic small-displacement theory the **tip deflection**
# is
#
# $$
# \delta_{\text{tip}} \;=\; \dfrac{P\,L^{3}}{3\,E\,I}
# $$
#
# Using $P = 10{,}000 \text{ N}$, $L = 3 \text{ m}$,
# $E = 2.1 \times 10^{11} \text{ Pa}$ and $I = 10^{-5} \text{ m}^{4}$,
# the analytical tip deflection is $\delta = -0.042857 \text{ m}$
# (negative because the load is downward in $+z$).
#
# This notebook establishes the **template** every subsequent
# curriculum notebook follows. Eight sections:
#
# 1. imports + parameters
# 2. geometry (with apeGmsh)
# 3. physical groups / labels
# 4. mesh
# 5. FEM build (sections + loads + BCs)
# 6. OpenSees ingest + analysis
# 7. result extraction + printed error vs analytical
# 8. optional viewer hook

# %% [markdown]
# ## 1. Imports and parameters

# %%
import openseespy.opensees as ops

from apeGmsh import apeGmsh

# --- Geometry + loading ---
L = 3.0          # beam length [m]
P = 10_000.0     # tip force [N], positive magnitude

# --- Elastic material / cross-section ---
E  = 2.1e11      # Young's modulus [Pa]
nu = 0.3         # Poisson's ratio
A  = 1.0e-3      # cross-section area [m^2]
Iy = 1.0e-5      # weak-axis moment of inertia [m^4]
Iz = 1.0e-5      # strong-axis moment of inertia [m^4] — controls bending
J  = 2.0e-5      # torsional constant [m^4]
G  = E / (2.0 * (1.0 + nu))

# --- Mesh density ---
LC = L / 10.0    # target element size [m]


# %% [markdown]
# ## 2. Geometry
#
# A single line from the fixed base at the origin to the free tip at
# $(L, 0, 0)$.

# %%
g_ctx = apeGmsh(model_name="02_cantilever_beam_2D", verbose=False)
g = g_ctx.__enter__()

p_base = g.model.geometry.add_point(0.0, 0.0, 0.0, lc=LC)
p_tip  = g.model.geometry.add_point(L,   0.0, 0.0, lc=LC)
line   = g.model.geometry.add_line(p_base, p_tip)

g.model.sync()


# %% [markdown]
# ## 3. Physical groups
#
# Three physical groups name the things we will load and restrain:
#
# - ``base`` — the fixed support point.
# - ``tip``  — the loaded free point.
# - ``beam`` — the line whose elements we assign a section to.

# %%
g.physical.add(0, [p_base], name="base")
g.physical.add(0, [p_tip],  name="tip")
g.physical.add(1, [line],   name="beam")


# %% [markdown]
# ## 4. Mesh

# %%
g.mesh.generation.generate(1)
fem = g.mesh.queries.get_fem_data()
print(f"mesh built: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")


# %% [markdown]
# ## 5. FEM build
#
# The elastic beam has an analytical counterpart in OpenSees
# (``elasticBeamColumn`` with ``Linear`` transformation) so no
# sections or materials need to be assigned through apeGmsh here —
# Section 6 wires those directly through the ingest.

# %%
# Nothing extra to declare for this simple problem.


# %% [markdown]
# ## 6. OpenSees ingest + analysis
#
# Direct ingest: emit the nodes, add a linear geometric transformation
# on the global $+y$ axis, emit ``elasticBeamColumn`` elements along
# the ``beam`` physical group, fix the ``base`` node with 6 DOFs, and
# apply a downward point load to the ``tip`` node.

# %%
ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 6)

# -- nodes --
for nid, xyz in fem.nodes.get():
    ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))

# -- geometric transformation (local y-axis = global +y is arbitrary here
#    because the cross-section is isotropic bending-wise) --
transf_tag = 1
ops.geomTransf("Linear", transf_tag, 0.0, 1.0, 0.0)

# -- elements --
for group in fem.elements.get(target="beam"):
    for eid, nodes in zip(group.ids, group.connectivity):
        ops.element(
            "elasticBeamColumn", int(eid),
            int(nodes[0]), int(nodes[1]),
            A, E, G, J, Iy, Iz, transf_tag,
        )

# -- boundary condition: fully fix the base point --
for n in fem.nodes.get(target="base").ids:
    ops.fix(int(n), 1, 1, 1, 1, 1, 1)

# -- load pattern: downward point force at the tip --
ops.timeSeries("Constant", 1)
ops.pattern("Plain", 1, 1)
for n in fem.nodes.get(target="tip").ids:
    ops.load(int(n), 0.0, 0.0, -P, 0.0, 0.0, 0.0)

# -- analysis recipe --
ops.system("BandGeneral")
ops.numberer("Plain")
ops.constraints("Plain")
ops.test("NormUnbalance", 1e-10, 10)
ops.algorithm("Linear")
ops.integrator("LoadControl", 1.0)
ops.analysis("Static")

status = ops.analyze(1)
assert status == 0, f"ops.analyze returned {status}"
print("analysis converged")


# %% [markdown]
# ## 7. Result extraction and verification
#
# Pull the tip node's vertical displacement (DOF 3 = $u_z$) and compare
# to $\delta = P L^{3} / (3 E I)$ (signed — downward load gives
# negative displacement in $+z$).

# %%
tip_node = int(next(iter(fem.nodes.get(target="tip").ids)))
fem_disp    = ops.nodeDisp(tip_node, 3)
analytical  = -P * L**3 / (3.0 * E * Iz)
err_pct     = abs(fem_disp - analytical) / abs(analytical) * 100.0

print(f"FEM tip disp :  {fem_disp:.6e}  m")
print(f"Analytical   :  {analytical:.6e}  m")
print(f"Error        :  {err_pct:.4f} %")


# %% [markdown]
# ## 8. (Optional) viewer check
#
# Uncomment the next cell in Jupyter to open the apeGmsh results
# viewer on the deformed shape. Skipped in headless/CI runs.

# %%
# g.mesh.results_viewer()


# %% [markdown]
# ## What this unlocks
#
# * The ``g.model`` → ``g.mesh`` → ``FEMData`` → ``ops.*`` pipeline,
#   which every later curriculum notebook reuses.
# * The **eight-section template** for result-verified teaching
#   notebooks: problem statement → geometry → PGs → mesh → FEM build →
#   OpenSees ingest+analyze → printed error → viewer hook.
# * ``elasticBeamColumn`` with a ``Linear`` geometric transformation
#   as the **verification yardstick** for later nonlinear or
#   large-displacement beam examples — deviations from $PL^{3}/3EI$
#   there tell you something about the added modelling physics, not
#   about the software.

# %%
g_ctx.__exit__(None, None, None)
