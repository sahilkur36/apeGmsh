# %% [markdown]
# # 13 — Beam-to-Solid Coupling via ``node_to_surface``
#
# **Curriculum slot:** Tier 4, slot 13.
# **Prerequisite:** 04 — 2D Portal Frame, 12 — Interface Tie.
#
# ## Purpose
#
# This is the canonical **mixed-dimension** coupling problem. A
# 6-DOF beam element (``elasticBeamColumn``) cannot attach directly
# to a 3-DOF solid (``FourNodeTetrahedron``) because their DOF
# spaces don't match. Attempting ``ops.equalDOF(beam_node,
# solid_node, 1..6)`` raises an error — the solid node has only 3
# DOFs.
#
# apeGmsh's ``g.constraints.node_to_surface(master, slave)`` solves
# this with the **phantom-node pattern**:
#
# 1. For each slave node on the solid surface, create a
#    *phantom* 6-DOF node at the same coordinates.
# 2. ``rigidLink('beam', master, phantom)`` — 6-DOF rigid link
#    from the beam master to every phantom.
# 3. ``equalDOF(phantom, slave, 1, 2, 3)`` — translational equality
#    between each phantom and its original solid node.
#
# The beam master's translations and rotations propagate rigidly
# through the phantoms, and the phantoms push only translations
# into the 3-DOF solid — the rotational DOFs of the beam effectively
# see a rigid base made of all the solid-face nodes.
#
# ## Problem — stub beam on a stiff solid footing
#
# A $0.2 \times 0.2 \times 0.2$ m cubic solid footing, meshed with
# tet4, with a 1 m beam stub extending vertically from the top
# face. The solid is declared with $E_{\text{solid}} = 10^{14}$ Pa
# so it behaves as a rigid base for the beam; the beam has the
# usual slot-02 elastic parameters.
#
# Load: a horizontal tip force $P$ at the top of the beam.
#
# Verification: with the solid effectively rigid, the tip
# deflection reduces to the classical cantilever
#
# $$
# \delta_{\text{tip}} = \dfrac{P\,L^{3}}{3\,E\,I_y}
# $$
#
# where $L = 1$ m is the beam length and $I_y$ is the beam's
# in-plane moment of inertia. Because the solid has finite
# compliance, we expect a small ($\ll 1\%$) residual above the
# classical formula.

# %% [markdown]
# ## 1. Imports and parameters

# %%
import numpy as np
import gmsh
import openseespy.opensees as ops

from apeGmsh import apeGmsh

# --- Geometry ---
S  = 0.2                   # side of the cubic solid footing [m]
L  = 1.0                   # beam length above the footing   [m]
H  = S + L                 # overall height (base to tip)    [m]

# --- Loading ---
P = 10_000.0               # horizontal tip load  [N]

# --- Material: beam ---
E_beam  = 2.1e11
nu_beam = 0.3
G_beam  = E_beam / (2 * (1 + nu_beam))
A  = 1.0e-3
Iy = 1.0e-5                # controls in-plane bending under horizontal load
Iz = 1.0e-5
Jb = 2.0e-5

# --- Material: solid (effectively rigid) ---
E_solid  = 1.0e14          # 3 orders of magnitude stiffer than the beam
nu_solid = 0.3

# --- Mesh ---
LC = S / 2.5               # coarse on solid; beam is a single line


# %% [markdown]
# ## 2. Geometry — solid cube + beam stub

# %%
g_ctx = apeGmsh(model_name="13_beam_to_solid", verbose=False)
g = g_ctx.__enter__()

# Build the solid cube via box primitive
vol_tag = g.model.geometry.add_box(0.0, 0.0, 0.0, S, S, S)

# Beam stub: a vertical line from the top-center of the cube to z = S + L
bx, by = S / 2.0, S / 2.0
p_base_beam = g.model.geometry.add_point(bx, by, S,     lc=LC)   # master reference
p_tip_beam  = g.model.geometry.add_point(bx, by, S + L, lc=LC)
ln_beam     = g.model.geometry.add_line(p_base_beam, p_tip_beam)

g.model.sync()


# %% [markdown]
# ## 3. Identify entities
#
# * ``solid`` — volume entity of the cube.
# * ``top_face`` — the $z = S$ face of the cube, which is the
#   coupling target for ``node_to_surface``.
# * ``beam`` — the line element along which elasticBeamColumn runs.
# * ``beam_base`` / ``beam_tip`` — the two endpoints of the beam.

# %%
# The box's surfaces have predictable tags after sync. We find the
# top face ($z = S$) by bounding-box lookup so the notebook is
# robust against tag renumbering.
TOL = 1e-6
top_face_tag = None
for (dim, tag) in gmsh.model.getEntities(dim=2):
    bb = gmsh.model.getBoundingBox(dim, tag)
    # z-range collapsed to z = S
    if abs(bb[2] - S) < TOL and abs(bb[5] - S) < TOL:
        top_face_tag = tag
        break
assert top_face_tag is not None, "could not locate top face of the cube"

g.physical.add(3, [vol_tag],        name="solid")
g.physical.add(2, [top_face_tag],   name="top_face")
g.physical.add(1, [ln_beam],        name="beam")
g.physical.add(0, [p_base_beam],    name="beam_base")
g.physical.add(0, [p_tip_beam],     name="beam_tip")


# %% [markdown]
# ## 4. Declare the mixed-dimension coupling

# %%
# master is the beam's bottom node (dim=0, tag=p_base_beam); slave is
# the solid's top face (dim=2, tag=top_face_tag). The resolver will
# create one phantom per mesh node on the top face at resolve time.
g.constraints.node_to_surface(
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


# %% [markdown]
# ## 6. Inspect the resolved constraint records
#
# ``fem.nodes.constraints.node_to_surfaces()`` yields one compound
# record per ``node_to_surface`` declaration; that record contains
# the phantom node tags + the rigid-beam / equalDOF pair records
# that will feed OpenSees.

# %%
for rec in fem.nodes.constraints.node_to_surfaces():
    print(f"node_to_surface record:")
    print(f"  master         : {rec.master_node}")
    print(f"  slave_nodes    : {len(rec.slave_nodes)}  (top-face mesh nodes)")
    print(f"  phantom_nodes  : {len(rec.phantom_nodes)}")
    print(f"  rigid_link     : {len(rec.rigid_link_records)} pair records")
    print(f"  equal_dof      : {len(rec.equal_dof_records)} pair records")


# %% [markdown]
# ## 7. OpenSees ingest + analysis
#
# Four distinct node kinds in the scene:
#
# | Kind | ndf | How created |
# |---|---|---|
# | solid | 3 | ``ops.node(nid, *xyz, '-ndf', 3)`` per-node override |
# | beam  | 6 | ``ops.node(nid, *xyz)`` (uses model default ndf=6) |
# | beam reference pts | 6 | same |
# | phantom | 6 | ``ops.node(pid, *xyz)`` (default ndf) |
#
# Because the slave-equalDOF links a 6-DOF phantom to a 3-DOF solid
# on a subset of DOFs, the ``Plain`` constraint handler cannot
# process it — we need ``Penalty`` (large factor) or
# ``Transformation``.

# %%
ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 6)

# -- solid nodes (per-node ndf=3) --
solid_node_ids = set(int(n) for n in fem.nodes.get(target="solid").ids)
for nid, xyz in fem.nodes.get(target="solid"):
    ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]),
             "-ndf", 3)

# -- beam reference nodes (default ndf=6) --
beam_ref_ids = set()
for tag_name in ("beam_base", "beam_tip"):
    for nid, xyz in fem.nodes.get(target=tag_name):
        nid = int(nid)
        if nid not in solid_node_ids and nid not in beam_ref_ids:
            ops.node(nid, float(xyz[0]), float(xyz[1]), float(xyz[2]))
            beam_ref_ids.add(nid)

# -- any remaining beam-line internal mesh nodes (default ndf=6) --
emitted = solid_node_ids | beam_ref_ids
for nid, xyz in fem.nodes.get(target="beam"):
    nid = int(nid)
    if nid in emitted:
        continue
    ops.node(nid, float(xyz[0]), float(xyz[1]), float(xyz[2]))
    emitted.add(nid)

# -- phantom nodes (6-DOF) --
for nid, xyz in fem.nodes.constraints.phantom_nodes():
    ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))

# -- solid material + tet4 elements --
ops.nDMaterial("ElasticIsotropic", 1, E_solid, nu_solid)
for group in fem.elements.get(element_type="tet4"):
    for eid, nodes in zip(group.ids, group.connectivity):
        ops.element("FourNodeTetrahedron", int(eid),
                    *[int(n) for n in nodes], 1)

# -- beam: elasticBeamColumn on line elements --
ops.geomTransf("Linear", 1, 1.0, 0.0, 0.0)      # vecxz = +x, column along +z
for group in fem.elements.get(target="beam"):
    for eid, nodes in zip(group.ids, group.connectivity):
        ops.element("elasticBeamColumn", int(eid),
                    int(nodes[0]), int(nodes[1]),
                    A, E_beam, G_beam, Jb, Iy, Iz, 1)

# -- mixed-dim constraints --
# rigid beam: master (beam_base) -> each phantom
for master, slaves in fem.nodes.constraints.rigid_link_groups():
    for slave in slaves:
        ops.rigidLink("beam", int(master), int(slave))

# equalDOF: each phantom -> solid slave (DOFs 1,2,3)
for pair in fem.nodes.constraints.equal_dofs():
    ops.equalDOF(int(pair.master_node), int(pair.slave_node), *pair.dofs)

# -- fix the bottom face of the solid to ground --
base_face_tag = None
for (dim, tag) in gmsh.model.getEntities(dim=2):
    bb = gmsh.model.getBoundingBox(dim, tag)
    if abs(bb[2] - 0.0) < TOL and abs(bb[5] - 0.0) < TOL:
        base_face_tag = tag
        break
assert base_face_tag is not None, "could not locate base face of the cube"

# The base face isn't in a PG yet; resolve its nodes through gmsh directly.
base_node_tags = set()
bnt, _, _ = gmsh.model.mesh.getNodes(
    dim=2, tag=base_face_tag, includeBoundary=True,
    returnParametricCoord=False,
)
for t in bnt:
    base_node_tags.add(int(t))
for nid in base_node_tags:
    ops.fix(nid, 1, 1, 1)           # 3-DOF fix — matches solid nodes

# -- horizontal tip load at beam_tip --
ops.timeSeries("Constant", 1)
ops.pattern("Plain", 1, 1)
for n in fem.nodes.get(target="beam_tip").ids:
    ops.load(int(n), P, 0.0, 0.0, 0.0, 0.0, 0.0)

# -- analysis (Penalty handler required for mixed-ndf equalDOF) --
ops.constraints("Penalty", 1e15, 1e15)
ops.numberer("RCM")
ops.system("UmfPack")
ops.test("NormDispIncr", 1e-8, 20)
ops.algorithm("Linear")
ops.integrator("LoadControl", 1.0)
ops.analysis("Static")
status = ops.analyze(1)
assert status == 0
print("analysis converged")


# %% [markdown]
# ## 8. Verification

# %%
tip_node = int(next(iter(fem.nodes.get(target="beam_tip").ids)))
d_tip = ops.nodeDisp(tip_node, 1)          # u_x

# Classical rigid-base cantilever: beam only, length L, about I_y
analytical = P * L**3 / (3.0 * E_beam * Iy)
err = abs(d_tip - analytical) / abs(analytical) * 100.0

print(f"FEM tip u_x       :  {d_tip:.6e}  m")
print(f"Classical (rigid) :  {analytical:.6e}  m")
print(f"Error             :  {err:.4f} %")


# %% [markdown]
# ## What this unlocks
#
# * **Mixed-ndf OpenSees models.** Once you need a beam attached
#   to a solid, per-node ``-ndf`` overrides + Penalty / Transformation
#   constraint handlers become standard equipment. Every later
#   mixed-dim notebook (slots 14, 15) uses this exact pattern.
# * **Phantom-node routing.** ``fem.nodes.constraints.rigid_link_groups()``
#   and ``fem.nodes.constraints.equal_dofs()`` replay the resolved
#   ``NodeToSurfaceRecord`` as a pair of iterables that drop
#   straight into ``ops.rigidLink`` and ``ops.equalDOF``.
# * **Face-targeted ``node_to_surface``.** The master can be any
#   6-DOF reference node; the slave can be any dim=2 PG or
#   bounding-box-located surface entity. This is the primitive
#   behind stub-to-floor, beam-to-shell, and reference-point-to-
#   face coupling patterns.

# %%
g_ctx.__exit__(None, None, None)
