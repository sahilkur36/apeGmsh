---
name: apegmsh-helper
description: Use whenever the user is working with apeGmsh — the structural-FEM wrapper around Gmsh with OpenSees integration. Triggers on building FEM models from CAD/STEP imports, Part-based assembly workflows, composite-based geometry/mesh/constraint APIs (g.model, g.mesh, g.physical, g.constraints, g.opensees, etc.), non-matching-mesh ties via ASDEmbeddedNodeElement, loads/masses/constraints ingestion into the OpenSees bridge, and exporting models to OpenSees Tcl or openseespy scripts. Covers apeGmsh's own abstractions on top of Gmsh and OpenSees. For raw gmsh API questions see the gmsh-structural skill; for raw OpenSees analysis commands see opensees-expert; for FEM theory first principles see fem-mechanics-expert.
---

# apeGmsh helper

apeGmsh is a structural-FEM wrapper around [Gmsh](https://gmsh.info) with a
composition-based API and an OpenSees integration.  The library's goal is
to make it cheap to describe a structural model **once** — geometry,
physical groups, loads, masses, constraints — and feed it to any solver
through a snapshot broker (`FEMData`).  OpenSees is the first-class
target; other solvers plug into the same contract.

This skill teaches **apeGmsh's own vocabulary and idioms**.  It does not
re-teach Gmsh or OpenSees — for those, follow the "See also" links at
the end of each section and the cross-reference section at the bottom.

---

## 1. Mental model

Three ideas carry most of the weight.  Once these click, the rest of
apeGmsh follows.

### 1.1  The session owns a Gmsh kernel and a family of composites

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="my_model", verbose=True) as g:
    # ... everything inside this block shares one Gmsh kernel
    ...
```

`g` is an **`apeGmsh` session**.  It starts one `gmsh.initialize()` call,
adds a Gmsh model, and attaches a dozen composite objects as attributes.
Each composite owns a **focused slice** of the Gmsh/OpenSees API with an
opinionated, consistent shape.  You never call `gmsh.*` directly — you
call `g.model.*`, `g.mesh.*`, `g.constraints.*`, `g.opensees.*`.

The session is a Python context manager.  Leaving the `with` block
calls `gmsh.finalize()` and cleans everything up.  This is the only
supported lifecycle for a normal session — don't try to call
`g.begin()` / `g.end()` by hand unless you have a specific reason.

Every composite is just an attribute on `g`.  The full session-level
map is:

| Access | Purpose |
|---|---|
| `g.model`          | OCC geometry — points, curves, surfaces, solids, boolean ops, transforms, I/O |
| `g.parts`          | Part instances + assembly-level registry (labels, fragment_all, fuse_group) |
| `g.physical`       | Named physical groups — pre-mesh, entity-driven |
| `g.mesh`           | Meshing — generation, sizing, structured, editing, queries, partitioning |
| `g.mesh_selection` | Post-mesh node/element selection sets |
| `g.constraints`    | Solver-agnostic constraint definitions + resolver |
| `g.loads`          | Load patterns + definitions (resolved into `fem.loads`) |
| `g.masses`         | Mass definitions (resolved into `fem.masses`) |
| `g.opensees`       | OpenSees bridge (materials, elements, ingest, inspect, export) |
| `g.g2o`            | Legacy one-liner `gmsh2opensees` transfer |
| `g.inspect`        | Session-level diagnostics |
| `g.plot`           | Matplotlib visualisations (optional dep) |
| `g.view`           | Gmsh post-processing scalar/vector views |

### 1.2  `g.model`, `g.mesh`, and `g.opensees` are split into sub-composites

The three biggest composites are further subdivided into focused
sub-namespaces.  This is why apeGmsh code reads like
`g.model.geometry.add_point(...)` not `g.model.add_point(...)`.  The
sub-composite prefix is required — there are **no shortcut methods
on the parent composites**.

**`g.model.*`** — five sub-composites for OCC geometry:

| Sub-composite | Methods |
|---|---|
| `g.model.geometry`   | `add_point`, `add_line`, `add_arc`, `add_circle`, `add_plane_surface`, `add_box`, `add_sphere`, `add_cylinder`, `add_cutting_plane`, `add_axis_cutting_plane`, `cut_by_surface`, `cut_by_plane`, … |
| `g.model.boolean`    | `fuse`, `cut`, `intersect`, `fragment` |
| `g.model.transforms` | `translate`, `rotate`, `scale`, `mirror`, `copy`, `extrude`, `revolve`, `sweep`, `thru_sections` |
| `g.model.io`         | `load_step`, `load_iges`, `load_dxf`, `save_step`, `save_msh`, `heal_shapes` |
| `g.model.queries`    | `bounding_box`, `center_of_mass`, `mass`, `boundary`, `adjacencies`, `entities_in_bounding_box`, `remove`, `remove_duplicates`, `make_conformal`, `fragment_all`, `registry` |

Plus `g.model.selection` (spatial entity queries) and three flat
top-level methods: `g.model.sync()`, `g.model.viewer()`, `g.model.gui()`.

**`g.mesh.*`** — seven sub-composites for meshing:

| Sub-composite | Methods |
|---|---|
| `g.mesh.generation`   | `generate`, `set_order`, `refine`, `optimize`, `set_algorithm`, `set_algorithm_by_physical` |
| `g.mesh.sizing`       | `set_global_size`, `set_size`, `set_size_sources`, `set_size_callback`, `set_size_by_physical` |
| `g.mesh.field`        | `distance`, `threshold`, `box`, `math_eval`, `boundary_layer`, `minimum`, `set_background` |
| `g.mesh.structured`   | `set_transfinite_{curve,surface,volume,automatic}`, `set_recombine`, `recombine`, `set_smoothing`, `set_compound`, `remove_constraints` |
| `g.mesh.editing`      | `embed`, `set_periodic`, `clear`, `reverse`, `relocate_nodes`, `remove_duplicate_{nodes,elements}`, `affine_transform`, `import_stl`, `classify_surfaces`, `create_geometry` |
| `g.mesh.queries`      | `get_nodes`, `get_elements`, `get_fem_data`, `get_element_properties`, `get_element_qualities`, `quality_report` |
| `g.mesh.partitioning` | `partition`, `unpartition`, `compute_renumbering`, `renumber_{nodes,elements,mesh}` |

Plus flat `g.mesh.viewer()` and `g.mesh.results_viewer()` for
interactive windows.

**`g.opensees.*`** — five sub-composites for the OpenSees bridge:

| Sub-composite | Methods |
|---|---|
| `g.opensees.materials` | `add_nd_material`, `add_uni_material`, `add_section` |
| `g.opensees.elements`  | `add_geom_transf`, `assign`, `fix` |
| `g.opensees.ingest`    | `loads(fem)`, `masses(fem)`, `constraints(fem, *, tie_penalty=None)` |
| `g.opensees.inspect`   | `node_table`, `element_table`, `summary` |
| `g.opensees.export`    | `tcl(path)`, `py(path)` |

Plus two flat lifecycle verbs: `g.opensees.set_model(ndm, ndf)` and
`g.opensees.build()`.

### 1.3  `FEMData` is the broker — the single contract between the library and any solver

The pattern every solver consumes is:

```python
g.mesh.partitioning.renumber_mesh(method="rcm", base=1)    # optional
fem = g.mesh.queries.get_fem_data(dim=3)
```

`fem` is a **`FEMData` snapshot** — a self-contained container with
everything a solver needs:

- `fem.node_ids`, `fem.node_coords`, `fem.element_ids`, `fem.connectivity`
- `fem.info` — mesh statistics (`n_nodes`, `n_elems`, `bandwidth`)
- `fem.physical` — physical group lookup (works offline, no live Gmsh)
- `fem.mesh_selection` — post-mesh selection sets
- `fem.loads` — resolved `NodalLoadRecord` / `ElementLoadRecord` objects
- `fem.masses` — resolved `MassRecord` objects
- `fem.constraints` — resolved `ConstraintRecord` objects (node pairs,
  groups, interpolations, surface couplings, node-to-surface)

The resolver runs automatically inside `get_fem_data()` — when you
declared loads/masses/constraints on the session before meshing, they
get resolved against the mesh nodes at snapshot time.  The resulting
`FEMData` is **pure data** — no live Gmsh session required.  You can
pickle it, ship it to another process, load it in a different Python
interpreter, and it still works.

The OpenSees bridge consumes `fem` through the ingest composite:

```python
(g.opensees.ingest
    .loads(fem)
    .masses(fem)
    .constraints(fem, tie_penalty=1e12))
g.opensees.build()
g.opensees.export.tcl("model.tcl")
```

That's the full solver pipeline for apeGmsh.  Load/mass/constraint
definitions are solver-agnostic; the bridge-side `ingest` layer is
where OpenSees-specific translation happens.

**See also**: raw Gmsh I/O and primitives — `gmsh-structural` skill.

---

## 2. Parts and the session — when to use which

apeGmsh separates two different uses of "a Gmsh session":

### 2.1  A `Part` is an isolated geometry container

```python
from apeGmsh import Part

beam = Part("W18x50_beam")
beam.begin()
# ... build geometry with beam.model.geometry.add_point(), add_line(), ...
beam.save("beam.step")
beam.end()
```

A `Part` owns **its own** isolated Gmsh session.  It only exists to
build a shape and export it to STEP/IGES.  Parts know **nothing**
about meshing, physical groups, constraints, loads, or masses — only
`model` and `inspect` composites are attached.

Use Parts when:
- You want to build a geometry once and import it into multiple
  assemblies later
- You want to compose several separate bodies (beams, columns,
  gussets) into one assembly
- You're scripting a geometry that deserves to live as a STEP file
  so other people (or other tools) can use it

### 2.2  The session-level `g.parts` registry imports Parts into one assembly

```python
with apeGmsh(model_name="I_beam") as g:
    g.parts.import_step("web.step",    label="web")
    g.parts.import_step("flange.step", label="top_flange", translate=(0, 0, 200))
    g.parts.import_step("flange.step", label="bot_flange")

    g.parts.fragment_all()       # conformal interfaces
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
```

The session-level `g.parts` is a **registry** — it tracks which
entities belong to which logical label.  After `fragment_all()`, the
volumes from different Parts become topologically connected at their
shared interfaces, and `g.parts.instances` still lets you address
them by label.

Use the session directly (no Parts) when:
- Your geometry is small enough to build inline in one notebook
- You don't need to re-import the same shape in multiple models
- You're prototyping — Parts are a productivity lever that kicks in
  once your model is reused

### 2.3  Rule of thumb

If you can say "this is a *part*" in plain English (an I-beam, a gusset,
a column, a rebar cage), it's probably worth being a `Part`.  If you're
describing *sites* in a model ("the tunnel lining", "the soil block"),
it's probably just geometry on the session.

**See also**: raw `gmsh.model.occ.*` primitives, import tolerances,
and OCC kernel quirks — `gmsh-structural` skill.

---

## 3. Constraints are a two-stage pipeline

This is the most commonly misunderstood part of apeGmsh, so it gets
its own section.

### 3.1  Stage 1 — declare constraints **before** meshing

```python
g.constraints.equal_dof(
    master_label="col", slave_label="bm",
    dofs=[1, 2, 3], tolerance=1e-3,
)
g.constraints.tie(
    master_label="col", slave_label="bm",
    master_entities=[(2, flange_tag)],
    slave_entities=[(2, end_face_tag)],
    dofs=[1, 2, 3, 4, 5, 6],
    tolerance=5.0,
)
```

Each method returns a lightweight `ConstraintDef` that references
**labels** (not raw tags).  These definitions live on `g.constraints`
until the mesh exists.  Labels resolve against `g.parts.instances`.

The full constraint catalogue:

| Level | Method | Produces record type |
|---|---|---|
| Node-to-node | `equal_dof`, `rigid_link`, `penalty` | `NodePairRecord` |
| Node-to-group | `rigid_diaphragm`, `rigid_body`, `kinematic_coupling` | `NodeGroupRecord` |
| Node-to-surface | `tie`, `distributing_coupling`, `embedded` | `InterpolationRecord` |
| Node-to-surface (compound) | `node_to_surface` | `NodeToSurfaceRecord` |
| Surface-to-surface | `tied_contact`, `mortar` | `SurfaceCouplingRecord` |

### 3.2  Stage 2 — resolution happens automatically in `get_fem_data`

When you call `g.mesh.queries.get_fem_data(dim=...)`, the resolver:

1. Walks every `ConstraintDef` on `g.constraints`
2. Looks up the master/slave node sets from physical groups or
   Part instances
3. For each definition, calls the appropriate `ConstraintResolver`
   method (`resolve_equal_dof`, `resolve_tie`, etc.) with numpy arrays
4. Accumulates the resulting records into a `ConstraintSet`
5. Attaches it to `fem.constraints`

You **don't** call the resolver yourself.  It's wired through
`get_fem_data`.  The records on `fem.constraints` are the authoritative
post-mesh form — this is what solvers consume.

### 3.3  Ingest into the OpenSees bridge

```python
(g.opensees.ingest
    .loads(fem)
    .masses(fem)
    .constraints(fem, tie_penalty=1e12))   # ← the constraint path
g.opensees.build()
```

`ingest.constraints(fem, tie_penalty=...)` stores `fem.constraints` on
the broker and sets the optional penalty stiffness.  During
`g.opensees.build()`, `emit_tie_elements()` walks the interpolation
records and populates `ops._tie_elements` — one entry per tie that
will become an `element ASDEmbeddedNodeElement` in the exported script.

### 3.4  The tie penalty (`ASDEmbeddedNodeElement`)

apeGmsh emits ties as `ASDEmbeddedNodeElement` — a **penalty element**
from OpenSees, not a classical `equalDOF`.  Key facts:

- The element assembles into `K`, not as an MP_Constraint.  The user's
  constraint handler is free — Plain, Transformation,
  AutoConstraintHandler all work.
- Default stiffness is `1e18`.  Pass `tie_penalty=1e10` to `1e12` if you
  see conditioning issues.  The element is a penalty, so you only need
  the stiffness several orders of magnitude above the parent element
  stiffness, not infinite.
- Retained nodes must be **3 (triangle)** or **4 (tetrahedron)**.  A
  Quad4 master face is split at its isoparametric diagonal into two
  triangles; the tie attaches to whichever triangle contains the
  slave's projection.
- Tri6 and Quad8 master faces are downgraded to their corner nodes
  (logged as a warning) — first-order tie kinematics only.
- If the slave node has rotational DOFs and the user declared them in
  `dofs=[1..6]`, the element gets `-rot` automatically.

**See also**: OpenSees `ASDEmbeddedNodeElement` internals, constraint
handler choices, and classical `equalDOF` vs penalty semantics —
`opensees-expert` skill.

---

## 4. Recipe chooser — common workflows

### 4.1  Cantilever beam with a tip load (smallest non-trivial example)

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="cantilever") as g:
    # Geometry: two points, one line
    p0 = g.model.geometry.add_point(0, 0, 0)
    p1 = g.model.geometry.add_point(5, 0, 0)
    l  = g.model.geometry.add_line(p0, p1)

    # Physical groups
    g.physical.add_point([p0],  name="base")
    g.physical.add_point([p1],  name="tip")
    g.physical.add_curve([l],   name="beam")

    # Mesh the line
    g.mesh.sizing.set_global_size(0.25)
    g.mesh.generation.generate(dim=1)
    g.mesh.partitioning.renumber_mesh(dim=1, method="simple", base=1)

    # Loads (pre-mesh declaration, resolved by get_fem_data)
    with g.loads.pattern("Tip"):
        g.loads.point("tip", force_xyz=(0, 0, -10e3))

    # Snapshot for the solver
    fem = g.mesh.queries.get_fem_data(dim=1)

    # OpenSees pipeline
    g.opensees.set_model(ndm=3, ndf=6)
    g.opensees.materials.add_uni_material("Steel", "Elastic", E=200e9)
    g.opensees.elements.add_geom_transf(
        "BeamTransf", "Linear", vecxz=[0, 0, 1],
    )
    g.opensees.elements.assign(
        "beam", "elasticBeamColumn",
        geom_transf="BeamTransf",
        A=0.01, E=200e9, G=77e9, Jx=1e-5, Iy=1e-5, Iz=1e-5,
    )
    g.opensees.elements.fix("base", dofs=[1, 1, 1, 1, 1, 1])
    g.opensees.ingest.loads(fem)
    g.opensees.build()
    g.opensees.export.tcl("cantilever.tcl")
```

### 4.2  Import a STEP solid and mesh it conformally

```python
with apeGmsh(model_name="frame") as g:
    g.model.io.load_step("frame.step", highest_dim_only=False)

    # Heal CAD roundoff + make conformal.  Tolerance depends on units:
    # 1e-3 for mm models, 1e-6 for m models, 1e-9 for cm models.
    g.model.queries.remove_duplicates(tolerance=1e-3)
    g.model.queries.make_conformal(tolerance=1e-3)

    # Sanity check: what did we import?
    vols = g.model.selection.select_volumes()
    print(f"Volumes after heal: {len(vols)}  dimtags: {vols.dimtags}")

    g.mesh.sizing.set_global_size(100)
    g.mesh.generation.generate(dim=3)

    # Conformality check — 0 removed nodes means every shared interface
    # is truly shared topology, not just coincident points.
    fem_before = g.mesh.queries.get_fem_data(dim=3)
    g.mesh.editing.remove_duplicate_nodes()
    fem_after = g.mesh.queries.get_fem_data(dim=3)
    duplicates = fem_before.info.n_nodes - fem_after.info.n_nodes
    print(f"Duplicate nodes removed: {duplicates}  (0 = fully conformal)")
```

If the conformality check shows duplicates, the `remove_duplicates`
tolerance was too tight — bump it up (remember the unit system).  If
increasing tolerance doesn't help, the imported geometry has a real
gap that CAD healing can't close.

### 4.3  Build a model with Parts and fragment them

```python
from apeGmsh import apeGmsh, Part

# Build each Part once, save to STEP
col = Part("column")
col.begin()
col.model.geometry.add_box(-0.2, -0.2, 0, 0.4, 0.4, 3.0)
col.save("column.step")
col.end()

beam = Part("beam")
beam.begin()
beam.model.geometry.add_box(0, -0.1, 0, 5.0, 0.2, 0.4)
beam.save("beam.step")
beam.end()

# Assembly
with apeGmsh(model_name="frame_assembly") as g:
    g.parts.import_step("column.step", label="col")
    g.parts.import_step("beam.step",   label="bm",
                        translate=(0.2, 0, 1.5))

    g.parts.fragment_all()     # ← shared interface becomes shared topology
    g.parts.add_physical_groups()

    g.mesh.sizing.set_global_size(0.1)
    g.mesh.generation.generate(dim=3)
```

### 4.4  Non-matching-mesh tie (the ASDEmbeddedNodeElement workflow)

Use this when two parts should be structurally connected but their
meshes don't match at the interface (different mesh sizes, different
CAD sources, etc.).

```python
with apeGmsh(model_name="beam_on_shell") as g:
    # ... build shell (master) and beam (slave) geometry ...

    # Declare the tie before meshing.  Labels reference parts.
    g.constraints.tie(
        master_label="shell",
        slave_label="beam",
        master_entities=[(2, shell_face_tag)],   # master surface
        slave_entities=[(1, beam_edge_tag)],      # slave edge
        dofs=[1, 2, 3, 4, 5, 6],                  # full 6-DOF tie
        tolerance=5.0,                             # projection tolerance
    )

    # Mesh each part independently (no fragment_all — we want
    # non-conformal meshes so the tie is actually doing work).
    g.mesh.sizing.set_size_by_physical("shell", 10)
    g.mesh.sizing.set_size_by_physical("beam",  50)
    g.mesh.generation.generate(dim=2)

    # Resolve ties against the mesh
    fem = g.mesh.queries.get_fem_data(dim=2)

    # Ingest into OpenSees with a custom penalty
    g.opensees.set_model(ndm=3, ndf=6)
    # ... materials / element assignments ...
    g.opensees.ingest.constraints(fem, tie_penalty=1e12)
    g.opensees.build()
    g.opensees.export.tcl("model.tcl")
```

The exported Tcl will contain `element ASDEmbeddedNodeElement` lines
with tags starting at `max(user_element_tag) + 1`.

### 4.5  Cut a solid with a plane and address the two halves

```python
with apeGmsh(model_name="cut_demo") as g:
    col = g.model.geometry.add_box(0, 0, 0, 1, 1, 3)

    # Build a horizontal cutting plane at z = 1.5
    pl = g.model.geometry.add_axis_cutting_plane('z', offset=1.5)

    # Split the column — classification by plane normal (above = +z)
    top, bot = g.model.geometry.cut_by_plane(
        col, pl,
        label_above="col_upper",
        label_below="col_lower",
    )
    print(f"Top pieces: {top}, bottom pieces: {bot}")
```

The returned lists contain **solid tags** you can address separately
— assign different materials, different mesh sizes, different physical
groups.  For an arbitrary non-planar cutting surface use
`cut_by_surface(solid, surface)` — same mechanics but returns a flat
list (no above/below classification).

### 4.6  Build a mesh field for local refinement

```python
# Refine near a crack tip
d = g.mesh.field.distance(curves=[crack_tip_curve])
t = g.mesh.field.threshold(
    d,
    size_min=0.1, size_max=5.0,
    dist_min=0.5, dist_max=10.0,
)
g.mesh.field.set_background(t)
g.mesh.generation.generate(dim=3)
```

Mesh fields are evaluated at every mesh point during generation and
take the minimum over all active size sources.  Use `set_size_sources`
to control which sources apply — for example, after importing CAD, set
`from_points=False` to ignore the per-point `lc` values baked into the
STEP file.

### 4.7  Export to OpenSees (Tcl **or** Python)

```python
g.opensees.set_model(ndm=3, ndf=3)
g.opensees.materials.add_nd_material(
    "Concrete", "ElasticIsotropic", E=30e9, nu=0.2, rho=2400,
)
g.opensees.elements.assign("Body", "FourNodeTetrahedron", material="Concrete")
g.opensees.elements.fix("Base", dofs=[1, 1, 1])

fem = g.mesh.queries.get_fem_data(dim=3)
(g.opensees.ingest
    .loads(fem)
    .masses(fem)
    .constraints(fem, tie_penalty=1e12))
g.opensees.build()

# Export as Tcl (for opensees CLI) or openseespy
g.opensees.export.tcl("model.tcl")
g.opensees.export.py("model.py")
```

Both exporters return the broker instance for chaining:

```python
g.opensees.export.tcl("model.tcl").py("model.py")
```

---

## 5. Pitfalls & gotchas

These are apeGmsh-specific traps we've learned the hard way.  Most
come up when the user's workflow straddles the Gmsh/apeGmsh boundary
or the apeGmsh/OpenSees boundary.

### 5.1  `remove_duplicates` tolerance is unit-dependent

```python
# For a model in millimetres (typical CAD export), use:
g.model.queries.remove_duplicates(tolerance=1e-3)   # 1 micrometre

# For a model in metres:
g.model.queries.remove_duplicates(tolerance=1e-6)   # 1 micrometre

# For a model in centimetres:
g.model.queries.remove_duplicates(tolerance=1e-4)   # 1 micrometre
```

The right tolerance is "much smaller than your smallest real feature
but bigger than CAD export roundoff".  Start strict (1 μm equivalent)
and loosen only if the conformality check still shows duplicates.
**Too aggressive a tolerance silently merges features.**

### 5.2  `generate(dim=2)` on a solid body leaves volumes unmeshed

```python
g.model.io.load_step("column.step", highest_dim_only=False)
g.mesh.generation.generate(dim=2)   # only surfaces meshed
g.mesh.viewer()                      # ← works correctly
```

This is a legitimate workflow — you import a solid for BRep robustness
but only want the surface mesh.  The solid volumes sit unmeshed in the
BRep registry.  The mesh viewer auto-skips dimensions with zero
elements (fix landed in commit `25e17b0`), so this just works.  Don't
try to delete the unmeshed volumes with `g.model.queries.remove` —
you lose the BRep identity for later operations.

### 5.3  The default `tie_penalty` is infinite

```python
g.opensees.ingest.constraints(fem)                 # no penalty → K=1e18
g.opensees.ingest.constraints(fem, tie_penalty=1e12)  # explicit softer
```

OpenSees' `ASDEmbeddedNodeElement` defaults to `K = 1e18`, which is
intentionally huge.  If you see conditioning issues (Newton failing to
converge, solver complaining about the stiffness matrix), drop it to
`1e10`–`1e12`.  The element is a penalty — you only need the
stiffness several orders of magnitude above the parent stiffness, not
infinite.

### 5.4  Tie element tags count from `max(user_element_tag) + 1`

User-declared elements via `g.opensees.elements.assign(...)` get tags
starting from 1 during `build()`.  Tie elements from
`ASDEmbeddedNodeElement` get tags starting one past the highest user
element tag — so a model with 1125 tets and 5 ties numbers the ties
1126, 1127, 1128, 1129, 1130.  When the user table is empty (rare),
tie tags start at `1_000_000`.  If you need to inject additional
user elements later, do it **before** `ingest.constraints(fem)` runs.

### 5.5  Don't modify `fem.constraints` after `get_fem_data`

`FEMData` is an immutable snapshot.  If you need to add or change
constraints, re-declare on `g.constraints`, then call
`get_fem_data(...)` again.  Modifying `fem.constraints` directly will
either silently fail or give you a stale picture on the next ingest.

### 5.6  `set_size_sources(from_points=False)` after CAD import

CAD files often bake a tiny per-point characteristic length (`lc`)
into every vertex.  `set_global_size(6000)` does **not** override these
per-point sources — Gmsh takes the minimum over all sources at every
node.  To make global sizing authoritative after IGES/STEP import:

```python
(g.mesh.sizing
    .set_size_sources(from_points=False, from_curvature=False)
    .set_global_size(6000))
```

### 5.7  `renumber_mesh(base=1)` gives OpenSees-ready IDs

Gmsh assigns non-contiguous node/element tags that jump around after
fragmenting.  OpenSees expects dense 1-based IDs.  Always call
`g.mesh.partitioning.renumber_mesh(method="rcm", base=1)` (or `method="simple"`)
before `get_fem_data` if you're going to OpenSees.  The `rcm` method
additionally minimises the bandwidth, which helps banded solvers.

### 5.8  `ASDEmbeddedNodeElement` needs 3 or 4 retained nodes — but 4 means **tetrahedron**, not Quad4

This is the most subtle rule.  A 4-retained-node tie is a **volumetric
embedding** (slave node inside a tet), not a Quad4 surface tie.  apeGmsh
handles Quad4 master surfaces by splitting at the isoparametric diagonal
and picking the triangle containing the slave's projection — the user
never sees this, but if you're debugging at the Tcl level, expect each
Quad4 face to emit **one** ASDEmbeddedNodeElement with 3 retained
nodes, not one with 4.

### 5.9  Test module purging breaks `isinstance` checks with lazy imports

This is an authoring-side gotcha, not a user-facing one.  If you write
pytest tests that hand-build `InterpolationRecord` / `NodePairRecord`
objects and feed them to a `ConstraintSet`, and another test in the
suite purges `apeGmsh.*` modules in tearDown, your hand-built records
will have a different class object than the one `ConstraintSet`
re-imports lazily.  `isinstance(rec, InterpolationRecord)` returns
False and the emitter sees zero records.  Fix: purge modules **before**
importing in each test's `setUp`, re-import fresh, and store the fresh
classes on `self`.  See `tests/test_opensees_tie_ingest.py` for the
pattern.

**See also**: Gmsh tolerance model, OCC boolean kernel quirks —
`gmsh-structural`.  OpenSees constraint handler semantics —
`opensees-expert`.

---

## 6. Anti-patterns

Things users try that don't work because apeGmsh has opinions.

### 6.1  ❌ Don't use `equal_dof` for a non-matching-mesh tie

```python
# WRONG: equal_dof requires master/slave nodes to be co-located
g.constraints.equal_dof(master_label="shell", slave_label="beam", ...)

# CORRECT: tie projects the slave onto the master face with shape
#          function interpolation
g.constraints.tie(
    master_label="shell", slave_label="beam",
    master_entities=[(2, shell_face)],
    slave_entities=[(1, beam_edge)],
    dofs=[1, 2, 3, 4, 5, 6],
    tolerance=5.0,
)
```

`equal_dof` requires a 1:1 node match within `tolerance`.  If your
meshes are non-matching (different sizes, different CAD sources), you
need shape-function interpolation via `tie`.

### 6.2  ❌ Don't call `g.mesh.generate()` — use `g.mesh.generation.generate()`

```python
# WRONG — AttributeError, no such method
g.mesh.generate(dim=3)

# CORRECT
g.mesh.generation.generate(dim=3)
```

The apeGmsh v1.0 API has **no shortcut methods** on parent composites.
Every action method lives in a sub-composite.  This is an intentional
break with older releases — it keeps the API discoverable and the
composites focused.

Same pattern for `g.model.add_point` (use `g.model.geometry.add_point`),
`g.opensees.add_nd_material` (use `g.opensees.materials.add_nd_material`),
etc.

### 6.3  ❌ Don't skip `make_conformal` after a CAD import if volumes should share interfaces

```python
# WRONG: imports 3 touching volumes but they remain topologically separate
g.model.io.load_step("frame.step", highest_dim_only=False)
g.mesh.generation.generate(dim=3)    # ← non-conformal mesh at the joints

# CORRECT
g.model.io.load_step("frame.step", highest_dim_only=False)
g.model.queries.remove_duplicates(tolerance=1e-3)
g.model.queries.make_conformal(tolerance=1e-3)
g.mesh.generation.generate(dim=3)
```

STEP/IGES files often export touching bodies as independent solids
with duplicate-but-separate faces at the interfaces.  Without
`make_conformal`, each body gets its own mesh and the interface has
two sets of coincident-but-disconnected nodes — exactly what you
don't want.

### 6.4  ❌ Don't bypass `g.physical` and store tags in a local dict

```python
# WRONG: local dict dies with the function scope, can't be addressed
#        by ingest.constraints / solver bridges / post-processing
tags = {"base": [p0], "tip": [p1]}

# CORRECT
g.physical.add_point([p0], name="base")
g.physical.add_point([p1], name="tip")
```

Physical groups are the **only** way to address mesh subsets by name
across the apeGmsh pipeline.  The resolver, the OpenSees bridge, and
every other consumer look up nodes by physical-group name.  Local
tag dicts don't survive meshing (Gmsh renumbers tags during
`fragment_all`) and they don't propagate to `fem.physical`.

### 6.5  ❌ Don't try to call `get_fem_data` before `generate`

```python
# WRONG
fem = g.mesh.queries.get_fem_data(dim=3)
g.mesh.generation.generate(dim=3)

# CORRECT
g.mesh.generation.generate(dim=3)
fem = g.mesh.queries.get_fem_data(dim=3)
```

`get_fem_data` reads from the live Gmsh mesh.  Before `generate`
there's nothing to read.  This is obvious in a script but easy to
miss in a notebook when you re-run cells out of order.

### 6.6  ❌ Don't assume `fragment_all` keeps volume tags stable

```python
# WRONG: box_tag is invalid after fragment_all if the box got split
box_tag = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
g.parts.fragment_all()
g.mesh.sizing.set_size(box_tag, 0.1)   # ← may refer to a stale tag
```

OCC's `fragment` renumbers entities.  After `fragment_all`, re-query
volumes via `g.model.selection.select_volumes()` or `g.parts.instances`
rather than holding on to tags from before the fragment.

---

## 7. When to reach for the lower-level skills

apeGmsh wraps Gmsh and OpenSees with opinionated abstractions.  Some
questions are really about the wrapped layers — in those cases, the
answer lives in one of these lower-level skills:

### Use `gmsh-structural` when you need to:
- Understand Gmsh's meshing algorithm choices (MeshAdapt vs Delaunay
  vs Frontal vs Quads vs HXT) beyond the string-name catalogue in
  `g.mesh.generation.set_algorithm`
- Debug OCC kernel errors (boolean failures, transfinite inconsistencies,
  STEP import warnings)
- Use raw `gmsh.model.occ.*` or `gmsh.model.mesh.*` primitives that
  apeGmsh doesn't wrap
- Understand transfinite meshing internals beyond the apeGmsh wrapper
- Set up complex mesh field combinations (nested `Min`, `MathEval`
  with expressions) beyond the convenience builders on `g.mesh.field`

### Use `opensees-expert` when you need to:
- Choose a constraint handler (Plain vs Transformation vs
  AutoConstraintHandler) for an analysis
- Write custom analysis commands beyond apeGmsh's export (integrators,
  solvers, recorders, patterns with nonlinear time series)
- Understand element tangent stiffness matrices or custom material
  development
- Extend apeGmsh's `_ELEM_REGISTRY` with a new OpenSees element type
  (requires both the registry entry and the render templates)
- Debug convergence failures, nonlinear solver parameters, or element
  validation errors in the exported script

### Use `fem-mechanics-expert` when you need to:
- Understand the FEM theory behind a constraint choice (why
  `ASDEmbeddedNodeElement` is a penalty element and what that
  implies for accuracy)
- Validate modelling decisions against first principles (shell vs
  solid, mesh density, shape function order, integration rules)
- Understand why a tie with Tri6 master faces is downgraded to Tri3
  corners for first-order kinematics

### Use `stko-to-python` when you need to:
- Load OpenSees MPCO/STKO HDF5 recorder output (different from
  apeGmsh's own results — STKO is the post-processing side)
- Query nodal or element results from a completed analysis

---

## 8. Commit-level reference

When in doubt about recent behaviour changes, these commits on
`nmb_WIP` are the authoritative source:

- `226ac80` — Phase 5: split `Mesh.py` into 7 sub-composites.  The flat
  `g.mesh.generate()` / `g.mesh.set_global_size()` methods removed.
- `393e9de` — Phase 6: split `OpenSees.py` into 5 sub-composites.
  `assign_element` → `assign`, `consume_*_from_fem` → `ingest.loads/masses`,
  `export_tcl/py` → `export.tcl/py`.
- `4133a1f` — Phase 7: lint & type cleanup, generic `_add_def` TypeVars.
- `25e17b0` — Phase 10: mesh viewer auto-skips empty dimensions when a
  solid is imported but meshed only up to dim=2.
- `50d750b` — Phase 11a: tie constraints via `ASDEmbeddedNodeElement`.
  The `g.opensees.ingest.constraints(fem, tie_penalty=...)` path.
- `c6f432e` — Cutting planes: `add_cutting_plane`, `add_axis_cutting_plane`,
  `cut_by_surface`, `cut_by_plane`.

If a user reports behaviour that contradicts this skill, first check
whether they're on a branch that predates the relevant phase — the
pre-refactor API is genuinely different and the v1.0 migration guide
at `docs/MIGRATION_v1.md` is the definitive checklist.

---

## 9. Quick reference — one-page cheat sheet

```python
from apeGmsh import apeGmsh, Part

# ── Session ─────────────────────────────────────────────────────
with apeGmsh(model_name="my_model", verbose=True) as g:

    # ── Geometry (g.model.*) ────────────────────────────────────
    p  = g.model.geometry.add_point(x, y, z, lc=1.0)
    ln = g.model.geometry.add_line(p1, p2)
    bx = g.model.geometry.add_box(x, y, z, dx, dy, dz)
    g.model.io.load_step("file.step", highest_dim_only=False)
    g.model.boolean.fuse(a, b)
    g.model.boolean.fragment(objects, tools)
    g.model.queries.remove_duplicates(tolerance=1e-3)
    g.model.queries.make_conformal(tolerance=1e-3)
    g.model.transforms.translate(tags, dx, dy, dz)

    # ── Parts (g.parts.*) ───────────────────────────────────────
    g.parts.import_step("col.step", label="col")
    g.parts.fragment_all()
    g.parts.add_physical_groups()

    # ── Physical groups (g.physical.*) ──────────────────────────
    g.physical.add_point([p], name="base")
    g.physical.add_curve([ln], name="beam")
    g.physical.add_surface([s], name="face")
    g.physical.add_volume([v], name="body")

    # ── Constraints (declare pre-mesh) ──────────────────────────
    g.constraints.equal_dof(master_label="a", slave_label="b", dofs=[1,2,3])
    g.constraints.tie(
        master_label="shell", slave_label="beam",
        master_entities=[(2, fx)], slave_entities=[(1, ex)],
        dofs=[1,2,3,4,5,6], tolerance=5.0,
    )

    # ── Loads & masses (declare pre-mesh) ───────────────────────
    with g.loads.pattern("gravity"):
        g.loads.point("tip", force_xyz=(0, 0, -10e3))
        g.loads.gravity("body", g=(0, 0, -9.81), density=2400)
    g.masses.volume("body", density=2400)

    # ── Mesh (g.mesh.*) ─────────────────────────────────────────
    g.mesh.sizing.set_global_size(100)
    g.mesh.sizing.set_size_sources(from_points=False)
    g.mesh.generation.set_algorithm(0, "hxt", dim=3)
    g.mesh.generation.generate(dim=3)
    g.mesh.partitioning.renumber_mesh(method="rcm", base=1)

    # ── FEM broker ──────────────────────────────────────────────
    fem = g.mesh.queries.get_fem_data(dim=3)

    # ── OpenSees pipeline ───────────────────────────────────────
    g.opensees.set_model(ndm=3, ndf=6)
    g.opensees.materials.add_nd_material(
        "Concrete", "ElasticIsotropic",
        E=30e9, nu=0.2, rho=2400,
    )
    g.opensees.elements.assign("body", "FourNodeTetrahedron", material="Concrete")
    g.opensees.elements.fix("base", dofs=[1, 1, 1])
    (g.opensees.ingest
        .loads(fem)
        .masses(fem)
        .constraints(fem, tie_penalty=1e12))
    g.opensees.build()
    g.opensees.export.tcl("model.tcl").py("model.py")

    # ── Viewers ─────────────────────────────────────────────────
    g.model.viewer()    # BRep picker
    g.mesh.viewer()     # mesh picker (auto-skips unmeshed dimensions)
```

This cheat sheet is the "90% of apeGmsh" surface — if a user's task
fits here, no deeper context is needed.  If their task is outside
this surface, start with the relevant section above.
