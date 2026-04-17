# %% [markdown]
# # 14 — Moment-Loaded Master via ``node_to_surface_spring``
#
# **Curriculum slot:** Tier 4, slot 14.
# **Prerequisite:** 13 — Beam-to-Solid Coupling.
#
# ## Purpose
#
# Slot 13 used ``g.constraints.node_to_surface`` to couple a 6-DOF
# beam master to a 3-DOF solid surface through phantom nodes and
# ``rigidLink`` constraints. That pattern works cleanly when the
# master has *element stiffness attached to all 6 DOFs* — in slot
# 13 the master was the beam's base node, and the beam element
# itself supplied the rotational stiffness.
#
# The pattern breaks down when **all three** of the following hold:
#
# 1. The master node has **free rotational DOFs** (no element
#    directly supplies rotational stiffness at the master).
# 2. A **moment** is applied directly to those free rotation DOFs.
# 3. The slave is a **solid** (``ndf = 3``), so the rigid-link
#    constraint is the *only* path through which the master's
#    rotation DOFs connect to any element.
#
# Under those conditions, the reduced stiffness matrix is
# ill-conditioned and OpenSees's solver typically fails with
# ``numeric analysis returns 1 -- UmfpackGenLinSolver::solve``.
#
# ``g.constraints.node_to_surface_spring`` is the fix. It
# generates the *same* phantom-node topology, but the master →
# phantom link is tagged ``RIGID_BEAM_STIFF`` so downstream
# emission sends it through ``stiff_beam_groups()`` — which
# produces **stiff ``elasticBeamColumn`` elements** instead of
# ``rigidLink`` constraints. Each stiff beam element's 6×6
# stiffness contributes terms on the master's rotation diagonal
# directly, so conditioning stays good regardless of which
# constraint handler OpenSees uses.
#
# ## Problem — moment-loaded reference point on a rigid solid
#
# The same geometry as slot 13: a $0.2 \times 0.2 \times 0.2$ m
# solid footing with a 1 m beam stub extending vertically. This
# time the **tip moment** $M$ replaces the tip force. With a very
# stiff solid acting as a rigid base, the classical Bernoulli
# answers for a cantilever under a pure tip moment are
#
# $$
# \theta_{\text{tip}} \;=\; \dfrac{M\,L}{E\,I_y},
# \qquad
# u_{\text{tip}} \;=\; \dfrac{M\,L^{2}}{2\,E\,I_y}.
# $$

# %% [markdown]
# ## 1. Imports and parameters

# %%
import gmsh
import openseespy.opensees as ops

from apeGmsh import apeGmsh

# Geometry / mesh (same as slot 13)
S, L = 0.2, 1.0
LC = S / 2.5

# Loading
M = 1_000.0            # tip moment about +y  [N·m]

# Beam material
E_beam  = 2.1e11
nu_beam = 0.3
G_beam  = E_beam / (2 * (1 + nu_beam))
A, Iy, Iz, Jb = 1.0e-3, 1.0e-5, 1.0e-5, 2.0e-5

# Solid material (effectively rigid)
E_solid, nu_solid = 1.0e14, 0.3


# %% [markdown]
# ## 2. Geometry

# %%
g_ctx = apeGmsh(model_name="14_contact_springs", verbose=False)
g = g_ctx.__enter__()

vol_tag = g.model.geometry.add_box(0.0, 0.0, 0.0, S, S, S)

bx, by = S / 2.0, S / 2.0
p_base_beam = g.model.geometry.add_point(bx, by, S,     lc=LC)
p_tip_beam  = g.model.geometry.add_point(bx, by, S + L, lc=LC)
ln_beam     = g.model.geometry.add_line(p_base_beam, p_tip_beam)

g.model.sync()

# Locate top + base faces of the box by bounding-box query.
TOL = 1e-6
top_face_tag = base_face_tag = None
for (dim, tag) in gmsh.model.getEntities(dim=2):
    bb = gmsh.model.getBoundingBox(dim, tag)
    if abs(bb[2] - S)   < TOL and abs(bb[5] - S)   < TOL:
        top_face_tag = tag
    elif abs(bb[2] - 0) < TOL and abs(bb[5] - 0)   < TOL:
        base_face_tag = tag
assert top_face_tag is not None and base_face_tag is not None


# %% [markdown]
# ## 3. Tagging

# %%
g.physical.add(3, [vol_tag],      name="solid")
g.physical.add(2, [top_face_tag], name="top_face")
g.physical.add(1, [ln_beam],      name="beam")
g.physical.add(0, [p_base_beam],  name="beam_base")
g.physical.add(0, [p_tip_beam],   name="beam_tip")


# %% [markdown]
# ## 4. The spring variant
#
# ``node_to_surface_spring`` produces the same structure as
# ``node_to_surface`` (phantom nodes + master→phantom links +
# phantom→slave equalDOFs), but the master→phantom records are
# tagged ``RIGID_BEAM_STIFF`` instead of ``RIGID_BEAM``. Downstream,
# ``fem.nodes.constraints.stiff_beam_groups()`` yields them for
# emission as stiff ``elasticBeamColumn`` elements, while
# ``rigid_link_groups()`` skips them.

# %%
g.constraints.node_to_surface_spring(
    master=p_base_beam,
    slave=top_face_tag,
)


# %% [markdown]
# ## 5. Mesh

# %%
g.mesh.sizing.set_global_size(LC)
g.mesh.generation.generate(3)
fem = g.mesh.queries.get_fem_data()
print(f"mesh: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")

# Inspect: the resolved record is a NodeToSurfaceRecord whose
# rigid_link_records are all RIGID_BEAM_STIFF, so
# stiff_beam_groups() has content and rigid_link_groups() is empty.
n_stiff = sum(len(slaves) for _, slaves in
              fem.nodes.constraints.stiff_beam_groups())
n_rigid = sum(len(slaves) for _, slaves in
              fem.nodes.constraints.rigid_link_groups())
n_edofs = sum(1 for _ in fem.nodes.constraints.equal_dofs())
print(f"stiff_beam pairs : {n_stiff}   (master -> phantom stiff beams)")
print(f"rigid_link pairs : {n_rigid}   (should be 0 for the spring variant)")
print(f"equal_dof pairs  : {n_edofs}  (phantom -> solid slave)")


# %% [markdown]
# ## 6. OpenSees ingest + analysis
#
# Compared to slot 13, the only difference is in the constraint
# emission. Instead of ``ops.rigidLink("beam", master, phantom)``,
# we iterate ``stiff_beam_groups()`` and emit an
# ``elasticBeamColumn`` between the master and each phantom with
# an **intentionally enormous** section ($A, I \gg$ the downstream
# beam) so the stiff beam is effectively a rigid link in practice
# but contributes real element stiffness to the master's rotation
# DOFs.

# %%
ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 6)

# Solid nodes (ndf=3)
solid_node_ids = set(int(n) for n in fem.nodes.get(target="solid").ids)
for nid, xyz in fem.nodes.get(target="solid"):
    ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]),
             "-ndf", 3)

# Beam reference nodes (ndf=6)
emitted = set(solid_node_ids)
for tag_name in ("beam_base", "beam_tip"):
    for nid, xyz in fem.nodes.get(target=tag_name):
        nid = int(nid)
        if nid in emitted:
            continue
        ops.node(nid, float(xyz[0]), float(xyz[1]), float(xyz[2]))
        emitted.add(nid)

# Beam internal mesh nodes
for nid, xyz in fem.nodes.get(target="beam"):
    nid = int(nid)
    if nid in emitted:
        continue
    ops.node(nid, float(xyz[0]), float(xyz[1]), float(xyz[2]))
    emitted.add(nid)

# Phantom nodes (ndf=6)
for nid, xyz in fem.nodes.constraints.phantom_nodes():
    ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))

# Solid material + tet4 elements
ops.nDMaterial("ElasticIsotropic", 1, E_solid, nu_solid)
for group in fem.elements.get(element_type="tet4"):
    for eid, nodes in zip(group.ids, group.connectivity):
        ops.element("FourNodeTetrahedron", int(eid),
                    *[int(n) for n in nodes], 1)

# --- Beam (elasticBeamColumn) on the line mesh ---
ops.geomTransf("Linear", 1, 1.0, 0.0, 0.0)
for group in fem.elements.get(target="beam"):
    for eid, nodes in zip(group.ids, group.connectivity):
        ops.element("elasticBeamColumn", int(eid),
                    int(nodes[0]), int(nodes[1]),
                    A, E_beam, G_beam, Jb, Iy, Iz, 1)

# --- Stiff beams (master -> phantom) FROM stiff_beam_groups() ---
# Assign each stiff beam a unique element tag starting above any
# existing tag so we don't collide with the beam or solid tets.
max_existing_eid = 0
for group in fem.elements:
    if group.ids is not None and len(group.ids) > 0:
        max_existing_eid = max(max_existing_eid, int(max(group.ids)))

STIFF_E = E_beam * 1e4        # 10,000× stiffer than the beam
STIFF_A = A * 1e4
STIFF_I = Iy * 1e4
STIFF_J = Jb * 1e4
STIFF_G = STIFF_E / (2 * (1 + nu_beam))

# Use a fresh geomTransf for the stiff beams so their local axes
# are defined irrespective of the main beam's direction.
ops.geomTransf("Linear", 2, 0.0, 1.0, 0.0)

next_eid = max_existing_eid + 1
for master, slaves in fem.nodes.constraints.stiff_beam_groups():
    for slave in slaves:
        ops.element("elasticBeamColumn", next_eid,
                    int(master), int(slave),
                    STIFF_A, STIFF_E, STIFF_G, STIFF_J, STIFF_I, STIFF_I, 2)
        next_eid += 1

# --- EqualDOF (phantom -> solid slave, DOFs 1..3) ---
for pair in fem.nodes.constraints.equal_dofs():
    ops.equalDOF(int(pair.master_node), int(pair.slave_node), *pair.dofs)

# --- Fix the base face of the solid ---
bnt, _, _ = gmsh.model.mesh.getNodes(
    dim=2, tag=base_face_tag, includeBoundary=True,
    returnParametricCoord=False,
)
for nid in bnt:
    ops.fix(int(nid), 1, 1, 1)

# --- Tip moment about +y ---
ops.timeSeries("Constant", 1)
ops.pattern("Plain", 1, 1)
for n in fem.nodes.get(target="beam_tip").ids:
    ops.load(int(n), 0.0, 0.0, 0.0, 0.0, M, 0.0)   # My = M

# Penalty handler (needed for the ndf-mixed equalDOF phantoms → solid)
ops.constraints("Penalty", 1e15, 1e15)
ops.numberer("RCM")
ops.system("UmfPack")
ops.test("NormDispIncr", 1e-8, 20)
ops.algorithm("Linear")
ops.integrator("LoadControl", 1.0)
ops.analysis("Static")
status = ops.analyze(1)
assert status == 0, f"analyze() returned {status}"
print("analysis converged")


# %% [markdown]
# ## 7. Verification
#
# With a rigid solid the tip of the 1 m beam should rotate and
# translate according to the classical cantilever-with-tip-moment
# formulas.

# %%
tip_node = int(next(iter(fem.nodes.get(target="beam_tip").ids)))
tip_ux    = ops.nodeDisp(tip_node, 1)
tip_theta = ops.nodeDisp(tip_node, 5)

analytical_theta = M * L       / (E_beam * Iy)
analytical_ux    = M * L**2 / (2.0 * E_beam * Iy)

err_theta = abs(tip_theta - analytical_theta) / abs(analytical_theta) * 100.0
err_ux    = abs(tip_ux    - analytical_ux)    / abs(analytical_ux)    * 100.0

print("Tip rotation (r_y)")
print(f"  FEM        : {tip_theta:.6e}  rad")
print(f"  Analytical : {analytical_theta:.6e}  rad   (ML/EI)")
print(f"  Error      : {err_theta:.4f} %")
print()
print("Tip deflection (u_x)")
print(f"  FEM        : {tip_ux:.6e}  m")
print(f"  Analytical : {analytical_ux:.6e}  m   (ML^2/2EI)")
print(f"  Error      : {err_ux:.4f} %")


# %% [markdown]
# ## What this unlocks
#
# * **The spring variant of node-to-surface.** Whenever you have a
#   6-DOF master that doesn't have an element directly attached to
#   all 6 of its DOFs and you're applying a moment there, reach for
#   ``g.constraints.node_to_surface_spring`` instead of
#   ``node_to_surface``. The stiff-beam pattern gives the master's
#   rotation DOFs real element-level stiffness.
# * **``stiff_beam_groups()`` iterator.** Pairs are emitted as
#   ``elasticBeamColumn`` elements between master and phantom.
#   Tag allocation must avoid colliding with existing element tags
#   from the beam and solid meshes — the pattern ``next_eid =
#   max_existing_eid + 1`` is reliable.
# * **Stiff-beam section sizing.** The idea is "orders of magnitude
#   stiffer than the downstream beam", not "infinite". A factor of
#   $10^4$ is typical. Higher values buy you less (the downstream
#   beam's compliance dominates) at the cost of marginal
#   conditioning degradation.

# %%
g_ctx.__exit__(None, None, None)
