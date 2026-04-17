# %% [markdown]
# # 03 — Simply-Supported Beam with Uniform Load
#
# **Curriculum slot:** Tier 1, slot 03.
# **Prerequisite:** 02 — 2D Cantilever Beam.
#
# ## Problem statement
#
# A prismatic beam of length $L$ rests on two supports:
#
# * a **pin** at $x = 0$ (restrains $u_x$, $u_y$, $u_z$, free to rotate),
# * a **roller** at $x = L$ (restrains $u_y$, $u_z$ only, free to rotate
#   and to slide along $x$ — needed so the supports are statically
#   determinate).
#
# A uniform distributed load $w$ (force/length) acts along the beam
# in the $-z$ direction.
#
# ```
#    │↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓│   w [N/m]
#    ●───────────────────────────●        +z
#   pin         L = 4 m        roller      │
#    (x=0)                     (x=L)       └── +x
# ```
#
# For Bernoulli-beam linear elastic theory the three things we check
# are:
#
# * **Midspan deflection**  $\;\delta_{\text{mid}} = \dfrac{5\,w\,L^{4}}{384\,E\,I}$
# * **Midspan bending moment**  $\;M_{\text{mid}} = \dfrac{w\,L^{2}}{8}$
# * **End reactions**  $\;R_{A} = R_{B} = \dfrac{w\,L}{2}$
#
# All three have closed-form answers and an elastic-beam FEM model
# should reproduce them **exactly** (because uniform load on a
# Bernoulli beam is interpolated by the element's exact shape
# functions when we use OpenSees ``eleLoad -beamUniform``).

# %% [markdown]
# ## 1. Imports and parameters

# %%
import openseespy.opensees as ops

from apeGmsh import apeGmsh

# --- Geometry + loading ---
L = 4.0          # span [m]
w = 5_000.0      # uniform distributed load [N/m]  (positive magnitude)

# --- Elastic material / cross-section ---
E  = 2.1e11      # Young's modulus [Pa]
nu = 0.3         # Poisson's ratio
A  = 1.0e-3      # cross-section area  [m^2]
Iy = 1.0e-5
Iz = 1.0e-5      # controls bending about the load direction (see slot 02)
J  = 2.0e-5
G  = E / (2.0 * (1.0 + nu))

# Mesh density — the beam must have a node at midspan so we can read
# the midspan deflection directly. An even number of elements places
# a node there.
N_ELEM = 10
LC = L / N_ELEM


# %% [markdown]
# ## 2. Geometry

# %%
g_ctx = apeGmsh(model_name="03_simply_supported_beam", verbose=False)
g = g_ctx.__enter__()

p_pin    = g.model.geometry.add_point(0.0,   0.0, 0.0, lc=LC)
p_mid    = g.model.geometry.add_point(L/2,   0.0, 0.0, lc=LC)   # explicit midspan
p_roller = g.model.geometry.add_point(L,     0.0, 0.0, lc=LC)

line_l = g.model.geometry.add_line(p_pin, p_mid)
line_r = g.model.geometry.add_line(p_mid, p_roller)

g.model.sync()


# %% [markdown]
# ## 3. Physical groups

# %%
g.physical.add(0, [p_pin],            name="pin")
g.physical.add(0, [p_roller],         name="roller")
g.physical.add(0, [p_mid],            name="midspan")
g.physical.add(1, [line_l, line_r],   name="beam")


# %% [markdown]
# ## 4. Mesh

# %%
# Force a transfinite line so each half gets exactly N_ELEM/2 elements.
g.mesh.structured.set_transfinite_curve(line_l, N_ELEM // 2 + 1)
g.mesh.structured.set_transfinite_curve(line_r, N_ELEM // 2 + 1)

g.mesh.generation.generate(1)
fem = g.mesh.queries.get_fem_data()
print(f"mesh built: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")


# %% [markdown]
# ## 5. FEM build
#
# Nothing to declare at the apeGmsh level for this elastic-only
# example; materials/sections are wired directly in the ingest.


# %% [markdown]
# ## 6. OpenSees ingest + analysis
#
# Uniform load is applied through ``eleLoad -beamUniform`` per element
# so the consistent nodal forces are exact. With the local $y$-axis
# set via ``vecxz = (0, 1, 0)`` (same convention as slot 02), a
# global $-z$ distributed load projects onto $+W_y$ in local coords:
#
# * local $x$: along beam (from i to j)
# * local $y = \text{cross}(\text{vecxz}, \text{local }x) = -\hat z$
# * so global $-z$ load $\equiv$ local $+W_y$ load.

# %%
ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 6)

# -- nodes --
for nid, xyz in fem.nodes.get():
    ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))

# -- geometric transformation --
transf_tag = 1
ops.geomTransf("Linear", transf_tag, 0.0, 1.0, 0.0)

# -- elements --
elem_tags: list[int] = []
for group in fem.elements.get(target="beam"):
    for eid, nodes in zip(group.ids, group.connectivity):
        ops.element(
            "elasticBeamColumn", int(eid),
            int(nodes[0]), int(nodes[1]),
            A, E, G, J, Iy, Iz, transf_tag,
        )
        elem_tags.append(int(eid))

# -- boundary conditions --
for n in fem.nodes.get(target="pin").ids:
    # pin: fix 3 translations; free 3 rotations
    ops.fix(int(n), 1, 1, 1, 0, 0, 0)
for n in fem.nodes.get(target="roller").ids:
    # roller: fix uy and uz only; free ux (slides) and rotations
    ops.fix(int(n), 0, 1, 1, 0, 0, 0)

# -- load: uniform -w in global -z == +Wy in local coords --
ops.timeSeries("Constant", 1)
ops.pattern("Plain", 1, 1)
for eid in elem_tags:
    ops.eleLoad("-ele", eid, "-type", "-beamUniform", w, 0.0)
    #                                                    ^^^ ^^^
    #                                                    Wy   Wz

# Need reactions -> turn them on before analyze.
ops.reactions()

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

# Reactions are only updated after an analysis step. Ask for them now.
ops.reactions()
print("analysis converged")


# %% [markdown]
# ## 7. Result extraction and verification
#
# Three printed checks: midspan deflection, midspan moment, end
# reactions. Each prints the FEM value, the analytical value, and
# the percentage error.

# %%
mid_node    = int(next(iter(fem.nodes.get(target="midspan").ids)))
pin_node    = int(next(iter(fem.nodes.get(target="pin").ids)))
roller_node = int(next(iter(fem.nodes.get(target="roller").ids)))

# --- midspan deflection ---
fem_disp_mid    = ops.nodeDisp(mid_node, 3)
analytical_disp = -5.0 * w * L**4 / (384.0 * E * Iz)
err_disp = abs(fem_disp_mid - analytical_disp) / abs(analytical_disp) * 100.0

# --- midspan moment ---
# Section force of the element whose j-node IS the midspan node.
# For elasticBeamColumn the section forces at the ends are returned by
# eleForce: [N_i, Vy_i, Vz_i, T_i, My_i, Mz_i, N_j, Vy_j, Vz_j, T_j, My_j, Mz_j]
# in local coords. We pick that element and read its j-end Mz.
elem_at_mid = None
for group in fem.elements.get(target="beam"):
    for eid, nodes in zip(group.ids, group.connectivity):
        if int(nodes[1]) == mid_node:
            elem_at_mid = int(eid)
            break
    if elem_at_mid is not None:
        break

forces = ops.eleForce(elem_at_mid)
fem_M_mid = forces[10]                  # Mz at node j (local)
analytical_M_mid = w * L**2 / 8.0
err_M = abs(abs(fem_M_mid) - analytical_M_mid) / analytical_M_mid * 100.0

# --- end reactions ---
# ``ops.nodeReaction(n)`` returns all 6 DOF reactions at node n.
# For a uniform downward load, each support carries w*L/2 upward in +z.
fem_R_pin    = ops.nodeReaction(pin_node,    3)    # Fz
fem_R_roller = ops.nodeReaction(roller_node, 3)    # Fz
analytical_R = w * L / 2.0
err_R_pin    = abs(fem_R_pin    - analytical_R) / analytical_R * 100.0
err_R_roller = abs(fem_R_roller - analytical_R) / analytical_R * 100.0

print("Midspan deflection")
print(f"  FEM        :  {fem_disp_mid:.6e}  m")
print(f"  Analytical :  {analytical_disp:.6e}  m")
print(f"  Error      :  {err_disp:.4f} %")
print()
print("Midspan moment")
print(f"  FEM        :  {abs(fem_M_mid):.6e}  N*m")
print(f"  Analytical :  {analytical_M_mid:.6e}  N*m")
print(f"  Error      :  {err_M:.4f} %")
print()
print("End reactions (Fz)")
print(f"  pin    FEM :  {fem_R_pin:.6e}  N     analytical:  {analytical_R:.6e}     err {err_R_pin:.4f} %")
print(f"  roller FEM :  {fem_R_roller:.6e}  N     analytical:  {analytical_R:.6e}     err {err_R_roller:.4f} %")


# %% [markdown]
# ## 8. (Optional) viewer check
#
# Uncomment in Jupyter to open the results viewer with the deflected
# shape and the moment diagram overlay.

# %%
# g.mesh.results_viewer()


# %% [markdown]
# ## What this unlocks
#
# * Two-support boundary-condition assembly (pin + roller) as a
#   baseline pattern. Any later simply-supported / propped-cantilever
#   example reuses this exact BC pair.
# * Distributed element loading via ``eleLoad -beamUniform``, with the
#   convention that global $-z$ lines up with local $+W_y$ under the
#   ``vecxz = (0, 1, 0)`` transformation. Slot 07 (load patterns)
#   replaces this with the ``g.loads.line(...)`` composite which emits
#   nodal equivalents instead.
# * Section-force extraction via ``ops.eleForce`` and reaction
#   extraction via ``ops.nodeReaction`` after ``ops.reactions()``.
#   Every later verification notebook uses the same triad:
#   displacement + internal force + reaction.

# %%
g_ctx.__exit__(None, None, None)
