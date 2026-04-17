# %% [markdown]
# # 01 — Hello Plate (Plane-Stress Uniaxial Tension)
#
# **Curriculum slot:** Tier 1, slot 01.
# **Prerequisite:** none — this is the entry point.
#
# ## Problem statement
#
# A square plate of side $L$ and unit thickness is:
#
# * fixed in $x$ (left-edge symmetry) along the left edge,
# * anchored in $y$ at a single corner (to kill rigid-body
#   translation in $y$),
# * pulled by a uniform horizontal traction $\sigma$ on the right
#   edge.
#
# ```
#                  σ [MPa]  →
#           ┌──────────────────→┐        +y
#           │                   │         │
#       ux=0│                   │         └── +x
#           │      plate        │
#           │      L × L        │
#           │    (plane stress) │
#           ●───────────────────┘    ← uy=0 at one corner only
# ```
#
# For a plane-stress plate in pure uniaxial tension the exact
# elastic displacement of the right edge is
#
# $$
# u_{x,\text{right}} \;=\; \dfrac{\sigma\,L}{E}.
# $$
#
# No mesh-dependent error in the limit (plane-stress uniaxial
# tension is a constant-stress field, which linear triangles
# represent **exactly**).
#
# ## What this notebook demonstrates
#
# The **full apeGmsh pipeline** on a 2D continuum problem:
#
# 1. ``g.model.geometry`` — build the plate.
# 2. ``g.physical`` — tag edges for loads and BCs.
# 3. ``g.mesh`` — triangulate.
# 4. ``g.mesh.queries.get_fem_data()`` — broker between Gmsh and the
#    solver.
# 5. Direct ``openseespy`` ingest: ``node`` → ``element tri31`` →
#    ``fix`` → ``load`` → ``analyze``.
# 6. Displacement extraction + printed error vs the analytical
#    answer.

# %% [markdown]
# ## 1. Imports and parameters

# %%
import numpy as np
import openseespy.opensees as ops

from apeGmsh import apeGmsh

# --- Geometry + loading ---
L      = 1.0            # plate edge length [m]
sigma  = 1.0e6          # traction on right edge [Pa]

# --- Material ---
E  = 2.1e11             # Young's modulus [Pa]
nu = 0.3                # Poisson ratio (used by material; unused by analytical)
thk = 1.0               # plane-stress thickness [m]

# --- Mesh ---
LC = L / 20.0           # target triangle edge [m]


# %% [markdown]
# ## 2. Geometry
#
# The plate is built from four corner points and a plane surface
# through a curve loop. ``sync()`` at the end is mandatory before
# any physical group or mesh operation can see the geometry.

# %%
g_ctx = apeGmsh(model_name="01_hello_plate", verbose=False)
g = g_ctx.__enter__()

p_BL = g.model.geometry.add_point(0.0, 0.0, 0.0, lc=LC)
p_BR = g.model.geometry.add_point(L,   0.0, 0.0, lc=LC)
p_TR = g.model.geometry.add_point(L,   L,   0.0, lc=LC)
p_TL = g.model.geometry.add_point(0.0, L,   0.0, lc=LC)

l_B = g.model.geometry.add_line(p_BL, p_BR)    # bottom
l_R = g.model.geometry.add_line(p_BR, p_TR)    # right (loaded)
l_T = g.model.geometry.add_line(p_TR, p_TL)    # top
l_L = g.model.geometry.add_line(p_TL, p_BL)    # left (fixed in x)

loop    = g.model.geometry.add_curve_loop([l_B, l_R, l_T, l_L])
surface = g.model.geometry.add_plane_surface([loop])

g.model.sync()


# %% [markdown]
# ## 3. Physical groups
#
# Four groups give us everything we need to touch:
#
# * ``plate``   — the surface (assigns elements in bulk).
# * ``left``    — left edge, for the symmetry BC.
# * ``right``   — right edge, for the applied traction.
# * ``corner``  — one node, for the y-anchor.

# %%
g.physical.add(0, [p_BL],    name="corner")
g.physical.add(1, [l_L],     name="left")
g.physical.add(1, [l_R],     name="right")
g.physical.add(2, [surface], name="plate")


# %% [markdown]
# ## 4. Mesh
#
# 2D triangulation at the global ``lc`` we set on every corner.

# %%
g.mesh.generation.generate(2)
fem = g.mesh.queries.get_fem_data()
print(f"mesh built: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")


# %% [markdown]
# ## 5. FEM build
#
# Nothing to do here for this linear-elastic plane-stress example —
# the material and element type are declared directly in the ingest
# (section 6). Slot 06 "Section catalog" introduces the
# ``g.opensees.sections.*`` composite that centralises this.


# %% [markdown]
# ## 6. OpenSees ingest + analysis
#
# This is **plane-stress 2-D**, so the OpenSees model is
# ``-ndm 2 -ndf 2`` (two spatial dimensions, two translational DOFs
# per node).  ``tri31`` elements take a plane-stress or plane-strain
# flag and a 2-D material; we use ``ElasticIsotropic`` as the
# material and ``"PlaneStress"`` as the element type.
#
# The right-edge traction is applied as per-node horizontal point
# loads, each carrying its tributary share of $\sigma \cdot L$.

# %%
ops.wipe()
ops.model("basic", "-ndm", 2, "-ndf", 2)

# -- nodes (2D: only x,y used) --
for nid, xyz in fem.nodes.get():
    ops.node(int(nid), float(xyz[0]), float(xyz[1]))

# -- material: ElasticIsotropic --
mat_tag = 1
ops.nDMaterial("ElasticIsotropic", mat_tag, E, nu)

# -- elements: tri31 (linear triangle) in plane stress --
for group in fem.elements.get(target="plate"):
    for eid, nodes in zip(group.ids, group.connectivity):
        ops.element(
            "tri31", int(eid),
            int(nodes[0]), int(nodes[1]), int(nodes[2]),
            thk, "PlaneStress", mat_tag,
        )

# -- BCs --
# Left edge: symmetry in x (ux = 0).
for n in fem.nodes.get(target="left").ids:
    ops.fix(int(n), 1, 0)
# Bottom-left corner: anchor in y (uy = 0) so the plate is not a
# rigid body in the +y direction.
for n in fem.nodes.get(target="corner").ids:
    ops.fix(int(n), 0, 1)

# -- traction on right edge --
# Distribute the total horizontal force sigma * L * thickness over the
# right-edge nodes by tributary length. Sort the edge nodes by y so
# the tributary-length computation is well defined.
right_nodes = list(fem.nodes.get(target="right").ids)
right_coords = np.array([
    [int(n), float(fem.nodes.coords[i, 1])]   # (node_id, y coordinate)
    for i, n in enumerate(fem.nodes.ids) if int(n) in set(int(r) for r in right_nodes)
])
# Sort by y
order = np.argsort(right_coords[:, 1])
right_sorted_ids = right_coords[order, 0].astype(int)
right_sorted_ys  = right_coords[order, 1]

# Tributary length for each right-edge node:
# end nodes get half the adjacent segment, interior nodes get half
# of each side.
N_right = len(right_sorted_ids)
seg_lens = np.diff(right_sorted_ys)
trib = np.zeros(N_right)
trib[0]  = seg_lens[0] / 2.0
trib[-1] = seg_lens[-1] / 2.0
for i in range(1, N_right - 1):
    trib[i] = (seg_lens[i-1] + seg_lens[i]) / 2.0
# Total should equal L exactly.
assert abs(trib.sum() - L) < 1e-9, f"tributary sum {trib.sum()} != {L}"

ops.timeSeries("Constant", 1)
ops.pattern("Plain", 1, 1)
for nid, t in zip(right_sorted_ids, trib):
    Fx = sigma * float(t) * thk
    ops.load(int(nid), Fx, 0.0)

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
# Mean horizontal displacement of the right-edge nodes compared to
# the analytical $u_{x} = \sigma L / E$.

# %%
u_x_right = np.mean([ops.nodeDisp(int(n), 1) for n in right_sorted_ids])
analytical = sigma * L / E
err_pct = abs(u_x_right - analytical) / abs(analytical) * 100.0

print(f"FEM ux (mean)  :  {u_x_right:.6e}  m")
print(f"Analytical     :  {analytical:.6e}  m")
print(f"Error          :  {err_pct:.4f} %")


# %% [markdown]
# ## 8. (Optional) viewer check
#
# Uncomment in Jupyter to open the results viewer with the deformed
# plate.

# %%
# g.mesh.results_viewer()


# %% [markdown]
# ## What this unlocks
#
# * **The 8-section template** used by every subsequent curriculum
#   notebook.
# * **2-D plane-stress continuum** setup in OpenSees
#   (``-ndm 2 -ndf 2``, ``nDMaterial ElasticIsotropic``, ``tri31``).
#   Slot 09 "Mesh refinement" reuses this exact problem at multiple
#   element sizes to produce a convergence curve.
# * **Tributary-length edge loading** — the manual pattern for
#   translating a distributed traction into nodal forces. Slot 07
#   "Load patterns" replaces the hand-coded loop with
#   ``g.loads.line(target='right', magnitude=..., direction=...)``.

# %%
g_ctx.__exit__(None, None, None)
