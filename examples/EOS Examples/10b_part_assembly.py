# %% [markdown]
# # 10b — Part Assembly with Transforms
#
# **Curriculum slot:** Tier 3, slot 10.5.
# **Prerequisite:** 10 — Parts Basics.
#
# ## Purpose
#
# Slot 10 showed how to *tag* existing geometry with ``g.parts.register``.
# That's the simplest part workflow, but it doesn't demonstrate the
# most useful application of the Parts composite: **instancing a
# template geometry multiple times with transforms.**
#
# The canonical apeGmsh idiom for placing a pre-built part multiple
# times is
#
# ```python
# col = Part(name="column").begin()
# # ... build the column geometry ...
# col.end()
#
# with apeGmsh(model_name="assembly") as g:
#     g.parts.add(col, label="col_0", translate=(0.0, 0, 0))
#     g.parts.add(col, label="col_1", translate=(4.0, 0, 0))
#     g.parts.add(col, label="col_2", translate=(8.0, 0, 0))
# ```
#
# That path currently hits a library bug
# (``_parts_registry._import_cad`` references ``labels_comp`` before
# assignment when no labels composite is registered on the session —
# tracked as a follow-up task). Until that's fixed, the **equivalent
# workflow using in-scene copy + translate + register** gives the
# same end result: one template, three placed copies, three
# independent parts. That's what this slot demonstrates.
#
# ## Problem — three identical columns in a row
#
# A 3 m vertical cantilever beam, meshed at ``lc = 0.3``, replicated
# at $x = 0$, $x = 4$, $x = 8$. Each column gets its own fixed base
# and tip point load $P$. Each column's tip deflection must match
# the classical $P L^{3} / (3\,E\,I)$ — that the transform (pure
# translation) doesn't alter the stiffness is the verification.

# %% [markdown]
# ## 1. Imports and parameters

# %%
import numpy as np
import gmsh
import openseespy.opensees as ops

from apeGmsh import apeGmsh

# --- Geometry ---
L = 3.0                     # column height
N_COLS = 3                  # how many instances to place
DX = 4.0                    # spacing between columns in +x
COLUMN_X_OFFSETS = [i * DX for i in range(N_COLS)]

# --- Loading ---
P = 10_000.0                # tip horizontal load (pushed in +x) [N]

# --- Material ---
E  = 2.1e11
nu = 0.3
G  = E / (2 * (1 + nu))
A, Iy, Iz, J = 1e-3, 1e-5, 1e-5, 2e-5

LC = L / 10.0


# %% [markdown]
# ## 2. Build the template column at origin

# %%
g_ctx = apeGmsh(model_name="10b_part_assembly", verbose=False)
g = g_ctx.__enter__()

p_base = g.model.geometry.add_point(0.0, 0.0, 0.0, lc=LC)
p_tip  = g.model.geometry.add_point(0.0, 0.0, L,   lc=LC)
ln     = g.model.geometry.add_line(p_base, p_tip)
g.model.sync()


# %% [markdown]
# ## 3. Instance the template three times
#
# Column 0 uses the original geometry directly — no copy needed.
# Columns 1..N-1 are built by ``g.model.transforms.copy`` of the
# template line followed by ``.translate`` to shift the copy in
# place, then registered as a new part.

# %%
# Column 0 = the template itself
g.parts.register("col_0", [(1, ln), (0, p_base), (0, p_tip)])

# Columns 1..N-1 = copies translated along +x
for i, x_off in enumerate(COLUMN_X_OFFSETS[1:], start=1):
    # copy the line (endpoints come along as its boundary)
    new_line_tags = g.model.transforms.copy([ln], dim=1)
    new_line = new_line_tags[0]
    # translate in place
    g.model.transforms.translate([new_line], dx=x_off, dy=0.0, dz=0.0, dim=1)
    g.model.sync()
    # find the new line's endpoints via boundary query
    bnd = gmsh.model.getBoundary([(1, new_line)], oriented=False)
    endpoint_tags = [abs(int(t)) for _, t in bnd]
    g.parts.register(
        f"col_{i}",
        [(1, new_line)] + [(0, t) for t in endpoint_tags],
    )

print(f"parts registered: {g.parts.labels()}")


# %% [markdown]
# ## 4. Mesh

# %%
g.mesh.generation.generate(1)
fem = g.mesh.queries.get_fem_data()
print(f"mesh: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")

# Per-part node maps
node_map = g.parts.build_node_map(fem.nodes.ids, fem.nodes.coords)
for lbl in sorted(node_map):
    print(f"  {lbl}: {len(node_map[lbl])} nodes")


# %% [markdown]
# ## 5. Per-part base / tip node identification
#
# Each column has 11 mesh nodes; base is the one at $z = 0$ and tip
# is the one at $z = L$. We compute them per part label by looking
# up each candidate's coordinates from ``fem.nodes.coords``.

# %%
# Build a tag→index lookup for fast coord access
tag_to_idx = {int(t): i for i, t in enumerate(fem.nodes.ids)}

def base_and_tip(label: str) -> tuple[int, int]:
    """Return (base_node_id, tip_node_id) for part ``label``."""
    base = tip = None
    for nid in node_map[label]:
        z = float(fem.nodes.coords[tag_to_idx[int(nid)], 2])
        if abs(z - 0.0) < 1e-9:
            base = int(nid)
        elif abs(z - L)   < 1e-9:
            tip = int(nid)
    assert base is not None and tip is not None, \
        f"could not find base/tip nodes for part '{label}'"
    return base, tip

column_nodes = {lbl: base_and_tip(lbl) for lbl in sorted(node_map)}
for lbl, (b, t) in column_nodes.items():
    print(f"  {lbl}: base={b}, tip={t}")


# %% [markdown]
# ## 6. OpenSees ingest + analysis
#
# Standard pattern; the only thing specific to this slot is that
# the element emission iterates per part label via ``Instance.entities``.

# %%
ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 6)

for nid, xyz in fem.nodes.get():
    ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))

ops.geomTransf("Linear", 1, 1.0, 0.0, 0.0)      # vecxz = +x so the columns' local y ≠ local x

# Elements per part (reuse pattern from slot 10)
def line_elements_of(inst):
    out = []
    for tag in inst.entities.get(1, []):
        etypes, etags, enodes = gmsh.model.mesh.getElements(1, int(tag))
        for etype, elist, nlist in zip(etypes, etags, enodes):
            if int(etype) != 1:
                continue
            arr = np.asarray(nlist, dtype=np.int64).reshape(-1, 2)
            for eid, row in zip(elist, arr):
                out.append((int(eid), (int(row[0]), int(row[1]))))
    return out

for lbl in sorted(node_map):
    inst = g.parts.get(lbl)
    for eid, (ni, nj) in line_elements_of(inst):
        ops.element("elasticBeamColumn", eid, ni, nj,
                    A, E, G, J, Iy, Iz, 1)

# Fix each base; apply horizontal load at each tip
for lbl, (base, tip) in column_nodes.items():
    ops.fix(int(base), 1, 1, 1, 1, 1, 1)

ops.timeSeries("Constant", 1)
ops.pattern("Plain", 1, 1)
for lbl, (base, tip) in column_nodes.items():
    ops.load(int(tip), P, 0.0, 0.0, 0.0, 0.0, 0.0)     # +x tip load

ops.system("BandGeneral"); ops.numberer("Plain"); ops.constraints("Plain")
ops.test("NormUnbalance", 1e-10, 10); ops.algorithm("Linear")
ops.integrator("LoadControl", 1.0); ops.analysis("Static")
ops.analyze(1)
print("analysis converged")


# %% [markdown]
# ## 7. Verification — all three columns must give identical δ_tip
#
# Each vertical cantilever loaded horizontally at its tip has the
# classical $\delta_{\text{tip}} = P L^{3} / (3 E I)$. With
# vecxz = ($+$x, 0, 0) and the column along $+z$, in-plane bending
# happens about local $+y$ — controlled by $I_{y}$ in our
# parameters.

# %%
analytical = P * L**3 / (3.0 * E * Iy)
print(f"{'part':>6s}  {'tip dx':>14s}  {'error %':>10s}")
worst = 0.0
for lbl, (base, tip) in column_nodes.items():
    d_tip = ops.nodeDisp(int(tip), 1)        # u_x (horizontal)
    err = abs(d_tip - analytical) / abs(analytical) * 100.0
    worst = max(worst, err)
    print(f"{lbl:>6s}  {d_tip:>14.6e}  {err:>9.4f} %")

print(f"\nanalytical reference : {analytical:.6e}  m")
print(f"worst-case error     : {worst:.4f} %")


# %% [markdown]
# ## What this unlocks
#
# * **Template-plus-transforms assembly workflow.** The whole point
#   of the ``Part`` class + ``Parts.add(translate=..., rotate=...)``
#   idiom: build the geometry once, place it many times.
# * **A CAD-import analogue.** ``g.parts.import_step(file_path,
#   translate=..., rotate=...)`` does the same thing for a pre-built
#   STEP file. Once the ``_import_cad`` bug is fixed, both paths will
#   produce the same ``Instance`` objects this notebook builds via
#   ``copy`` + ``translate`` + ``register``.
# * **Per-part role extraction after meshing.** The "iterate
#   ``Instance.entities[1]`` for elements, look up coordinates via
#   ``fem.nodes.coords`` to classify endpoints" pattern scales to any
#   assembly geometry.

# %%
g_ctx.__exit__(None, None, None)
