# Example: 3D Frame with Slab — `apeGmsh` Walkthrough

This notebook (`example_frame3D_slab_opensees.ipynb`) is a pedagogical walkthrough of **`apeGmsh`**: how to take a CAD file, build a conformal mesh, attach physical meaning to the geometry, and hand off a clean, solver-agnostic data package to a FEM solver. OpenSees is used downstream only as a consumer — someone else will teach you OpenSees. Here, OpenSees is just the "output port" that proves the mesh is clean and addressable.

Everything in this document is about **the Gmsh side**: the `apeGmsh` API, the conventions that make it work, and the small but important choices that keep your model honest before the solver ever sees it.

---

## 1. Why `apeGmsh`

Gmsh is a powerful mesher, but its native Python API is low level: raw integer tags, entity dimensions, manual physical groups, manual duplicate cleanup, explicit sync calls. When you drive a FEM solver from a CAD file, you end up writing the same boilerplate every single time.

`apeGmsh` is a thin, opinionated wrapper around that workflow. It gives you:

1. **A single session object** (`apeGmsh(...)`) with clear `initialize()` / `finalize()` lifecycle.
2. **Chainable managers** for the geometry (`m.model.*`) and the mesh (`m.mesh.*`), with a strict boundary between the two.
3. **Predicate-based selections** (`on_plane`, `aligned`, ...) that turn spatial intent into physical groups without manual tag bookkeeping.
4. **A `FEMData` handoff object** — a solver-agnostic snapshot (node ids, coordinates, connectivity, physical-group lookups) that you can use long after Gmsh has been shut down.
5. **Viewers** at every stage (`model.viewer`, `mesh.viewer`, `GeomTransfViewer`, `Results.viewer`) so you can catch mistakes the moment they happen instead of a hundred lines later.

The mental model to internalize before reading the notebook line by line:

- **Everything is addressed through physical groups**, never raw Gmsh tags.
- **Geometry and mesh are separate worlds.** `m.model.*` mutates CAD entities; `m.mesh.*` mutates the discretization. Duplicates, tolerances, and conformal stitching live on the geometry side; element generation, renumbering, and duplicate-element cleanup live on the mesh side.
- **`FEMData` is the contract with the solver.** Once it exists, `apeGmsh` can be finalized and the rest is pure Python.

---

## 2. The `apeGmsh` API, block by block

### 2.1 Creating the session

```python
from apeGmsh import apeGmsh

m1 = apeGmsh(model_name="Frame3D_story", verbose=True)
m1.initialize()
```

`initialize()` boots the underlying Gmsh context. Every `apeGmsh` session must be paired with `m1.finalize()` later to release Gmsh — exactly like `gmsh.initialize()` / `gmsh.finalize()`, but lifted to the wrapper so you cannot forget model naming or verbosity flags.

### 2.2 Importing CAD and making it FEM-ready

```python
m1.model.io.load_iges(
    file_path=model_iges_path,
    highest_dim_only=False,
)

m1.model.queries.remove_duplicates(tolerance=1)
m1.make_conformal(tolerance=1.0)
m1.model.queries.remove_duplicates(tolerance=1)
m1.model.sync()
```

Four things are happening here, each of which prevents a classic Gmsh bug:

- **`highest_dim_only=False`** keeps lower-dimensional entities in the model. If you leave it at the default `True`, Gmsh throws away anything below the top dimension — meaning the column lines in `Frame3D.iges` disappear silently and you end up with a slab-only model. This is the most common "where did my beams go?" bug with IGES/STEP import.
- **`remove_duplicates(tolerance=1)`** merges geometric points within 1 mm of each other. CAD files commonly export coincident endpoints that are a few microns apart; without this, Gmsh treats them as independent vertices and column tops refuse to share nodes with the slab. The tolerance is in model units (mm here).
- **`make_conformal(tolerance=1.0)`** is the step that stitches the column curves into the slab surface. It finds where a 1D entity touches a 2D entity and *imprints* the intersection into the surface's topology, so that when the surface is meshed, there is guaranteed to be a mesh node exactly at each column top. Without conformal stitching, the beam/shell connection is a coincidence of mesh sizes and usually fails.
- **`m1.model.sync()`** pushes all the OCC kernel changes into Gmsh's internal model. Forgetting to sync is another silent-failure source: your geometry looks right in `model.viewer()` but physical groups can't find anything because the internal representation hasn't been updated.

Notice the ordering: remove duplicates, *then* make conformal, *then* remove duplicates again. The second pass catches the new stitch points that `make_conformal` introduced. This is a recipe worth memorizing.

### 2.3 Selections → physical groups

```python
m1.model.selection.select_points(on_plane=("z", 0, 1e-3)).to_physical("base_supports")
m1.model.selection.select_curves(aligned="z").to_physical("columns")
m1.model.selection.select_surfaces(on_plane=("z", 3000, 1e-3)).to_physical("slab")
```

This is the single most important idiom in `apeGmsh`. Instead of writing raw Gmsh like:

```python
# The kind of code apeGmsh is built to replace
pts = [tag for dim, tag in gmsh.model.getEntities(0)
       if abs(gmsh.model.getValue(0, tag, [])[2]) < 1e-3]
gmsh.model.addPhysicalGroup(0, pts, tag=1)
gmsh.model.setPhysicalName(0, 1, "base_supports")
```

you write a *predicate* and let the library resolve the entities:

- `select_points(on_plane=("z", 0, 1e-3))` — every point with `|z - 0| < 1e-3`.
- `select_curves(aligned="z")` — every curve whose axis is aligned with the global `z` direction.
- `select_surfaces(on_plane=("z", 3000, 1e-3))` — every surface lying on the plane `z = 3000`.

`select_*` returns a selection object that you can either:

- Chain with `.to_physical("name")` to create a named physical group (and still get the selection back for further use), or
- Consume directly with `.to_tags()` to get raw integer tags for something like a mesh-size override.

That distinction matters: **selections and physical groups are different things**. A selection is a set of entities you just grabbed. A physical group is that set given a name so the solver side can find it again. Every physical group was a selection first, but not every selection needs to become one — for example, the column-top points in the next block are used only as a mesh sizing source, so they stay a selection.

### 2.4 Visualizing the geometry first

```python
m1.model.viewer()
```

This pops the Gmsh GUI on the current geometry, colored by physical groups. Use it *immediately after* creating your groups — if `base_supports`, `columns`, or `slab` look empty or wrong in the GUI, your tolerances were off or you forgot a `sync()`. Catching this here saves enormous amounts of downstream debugging. The rule of thumb: never mesh something you have not visually inspected at the geometry level.

### 2.5 Mesh sizing and generation

```python
m1.mesh.editing.clear()

col_tops = m1.model.selection.select_points(on_plane=("z", 3000, 1e-3))
print(f"Column-top nodes: {col_tops.to_tags()}")

(m1.mesh
     .set_size_sources(from_points=True, extend_from_boundary=True)
     .set_global_size(3000)
     .set_size(col_tops.to_tags(), 1000)
     .generate(dim=2)
     .remove_duplicate_nodes()
     .remove_duplicate_elements())

m1.mesh.viewer()
```

Walking through each call:

- **`m1.mesh.editing.clear()`** wipes any previous mesh. If you re-run cells in a notebook without clearing, you can stack old and new meshes on top of each other, and `remove_duplicate_elements` becomes load-bearing in a bad way. Start clean.
- **`set_size_sources(from_points=True, extend_from_boundary=True)`** tells Gmsh that mesh sizes should be read from points in the model and diffused inward from the boundary. This is the mode that pairs naturally with `set_size(tags, h)`.
- **`set_global_size(3000)`** is the fallback size — everything far from a sized point uses this.
- **`set_size(col_tops.to_tags(), 1000)`** refines the mesh around the column tops. This is how you get a finer load path near the columns without meshing the whole slab at 1000 mm.
- **`generate(dim=2)`** meshes *surfaces only*. This is deliberate: we want the columns to remain single line entities (one element per column), and only the slab should be triangulated. `generate(dim=3)` would try to build volumes that don't exist; `generate(dim=1)` would shatter the columns into tiny segments.
- **`remove_duplicate_nodes()`** and **`remove_duplicate_elements()`** are defensive. After conformal stitching you can occasionally end up with two nodes at a stitch point or two copies of a boundary element. Calling these at the end of the pipeline costs nothing and guarantees a clean mesh.

The chained fluent interface is intentional: it reads top-to-bottom as a recipe, and reordering becomes obvious. Try moving `.generate(dim=2)` before `.set_size(...)` and watch the refinement get ignored — that's a useful exercise.

`m1.mesh.viewer()` pops the GUI again, this time on the mesh. Check the refinement at the column tops and make sure nothing looks stretched or sliver-thin.

### 2.6 Renumbering — why and when

```python
m1.mesh.partitioning.renumber_mesh(base=1)
```

Right after meshing, Gmsh's node and element tags can be sparse and non-contiguous (especially after duplicate removal). Renumbering with `base=1` gives you `1, 2, 3, ...` node ids and elements, which is what you want when debugging — error messages become human-readable.

**Timing matters:** renumber *after* physical groups exist and *before* you extract `FEMData`. The notebook deliberately calls `m1.physical.get_nodes(dim=0, tag=1)` both before and after renumbering to show that physical groups get rewired transparently. That's a useful exercise: run those two cells in isolation and compare the tag arrays.

### 2.7 Extracting `FEMData` — the handoff

```python
fem_data = m1.mesh.queries.get_fem_data()
```

`FEMData` is an immutable snapshot that exposes:

- `fem_data.node_ids` — sorted numpy array of node tags.
- `fem_data.node_coords` — `(N, 3)` numpy array of coordinates, row-aligned with `node_ids`.
- `fem_data.element_ids` and `fem_data.connectivity` — all elements and their vertex lists.
- `fem_data.physical` — the same physical-group lookup the live model had, but detached from Gmsh.

The physical API on `fem_data` is the one you'll use everywhere downstream:

```python
fem_data.physical.get_tag(2, "slab")              # name → integer tag
fem_data.physical.get_nodes(dim=0, tag=1)         # {'tags': ndarray}
fem_data.physical.get_nodes(dim=2, tag=3)         # all nodes of the slab
fem_data.physical.get_elements(dim=1, tag=2)      # {'element_ids', 'connectivity'}
fem_data.physical.get_elements(dim=2, tag=3)      # slab triangles
```

Two shapes to recognize:

- **Nodes** come back as `{'tags': ndarray}`.
- **Elements** come back as `{'element_ids': ndarray, 'connectivity': ndarray}`.

This is exactly the shape a FEM solver wants: one loop, zero reshaping. Whether the solver is OpenSees, FEniCS, or something you wrote yourself, the consumption pattern is identical.

### 2.8 Finalizing

```python
m1.finalize()
```

Once `fem_data` exists, the Gmsh context is no longer needed. `finalize()` releases it. Everything below this line in the notebook — solver assembly, load computation, analysis, visualization — runs on `fem_data` alone. This is by design: the mesher and the solver are decoupled, and you can pickle, serialize, or pass `fem_data` around freely.

---

## 3. How `FEMData` is consumed downstream

The notebook hands `fem_data` to OpenSees, but the patterns you see there are generic to any solver. Three idioms repeat.

**Pattern A — create all nodes from `FEMData`:**

```python
for node_id, coords in zip(fem_data.node_ids, fem_data.node_coords):
    ops.node(int(node_id), *coords)
```

The `int(...)` cast is because `FEMData` uses numpy integers and OpenSees' Python bindings are strict about native `int`. This is a good thing to warn students about — it's the kind of bug that only shows up in an error message three functions deep.

**Pattern B — apply a boundary condition to a named group of points:**

```python
base_nodes = fem_data.physical.get_nodes(dim=0, tag=1)['tags']
for node_id in base_nodes:
    ops.fix(int(node_id), 1, 1, 1, 1, 1, 1)
```

The `dim=0, tag=1` pair is the integer handle for `"base_supports"`. You can also look it up by name with `fem_data.physical.get_tag(0, "base_supports")` if you want the code to survive reordering.

**Pattern C — create all elements of a physical group:**

```python
cols = fem_data.physical.get_elements(dim=1, tag=2)
for eid, (ni, nj) in zip(cols['element_ids'], cols['connectivity']):
    ops.element('elasticBeamColumn', int(eid), int(ni), int(nj), ...)

slab = fem_data.physical.get_elements(dim=2, tag=3)
for eid, (ni, nj, nk) in zip(slab['element_ids'], slab['connectivity']):
    ops.element('ShellDKGT', int(eid), int(ni), int(nj), int(nk), ...)
```

One physical group per element family: columns → beam elements, slab → shell elements. Adding a second slab, a brace group, or a floor-beam group is just one more selection in the geometry step plus one more loop here. No global counters, no manual tag bookkeeping.

---

## 4. Turning a distributed pressure into nodal loads — a Gmsh-side exercise

Many solvers (OpenSees included) only accept nodal forces through their load API. That means any distributed load you want to apply has to be **lumped onto the mesh** before the solver sees it, and the mesh is exactly what `apeGmsh` gives you. This makes load lumping a natural Gmsh-side exercise, not a solver-side one.

There are two common ways to lump a uniform pressure `q` applied over a triangular mesh.

### 4.1 Tributary-area approach

For each node, compute the area of mesh that "belongs" to it — typically one third of every adjacent triangle, or a Voronoi-style cell — then assign `F_node = q · A_tributary`. Easy to explain, a bit fiddly to implement, and dependent on your definition of "tributary".

### 4.2 Consistent (work-equivalent) lumping — what the notebook uses

For a constant-strain triangle (CST) loaded by a uniform pressure `q`, the work-equivalent nodal forces are exactly equal on all three nodes:

```
F_i = F_j = F_k = q · A_tri / 3
```

This falls straight out of integrating the linear shape functions over the triangle. It is "consistent" because it preserves virtual work between the continuous pressure and the discrete nodal forces, and for a uniform load on CSTs it coincides numerically with the tributary-area answer (each node owns one-third of each adjacent triangle).

The notebook computes this entirely from `fem_data` — no gmsh, no solver:

```python
slab_tag   = fem_data.physical.get_tag(2, "slab")
slab_elems = fem_data.physical.get_elements(2, slab_tag)

nodal_fz = {}
for conn in slab_elems['connectivity']:
    ni, nj, nk = int(conn[0]), int(conn[1]), int(conn[2])

    pi = fem_data.node_coords[np.searchsorted(fem_data.node_ids, ni)]
    pj = fem_data.node_coords[np.searchsorted(fem_data.node_ids, nj)]
    pk = fem_data.node_coords[np.searchsorted(fem_data.node_ids, nk)]

    # Triangle area via the cross product — works for any orientation in 3D
    area = 0.5 * np.linalg.norm(np.cross(pj - pi, pk - pi))
    fz   = q * area / 3.0

    for n in (ni, nj, nk):
        nodal_fz[n] = nodal_fz.get(n, 0.0) + fz
```

Things to point out to students, all of them `apeGmsh`-flavored:

- **`fem_data.physical.get_elements(2, slab_tag)`** is the entire secret. You do not care which Gmsh tags the slab triangles happened to get, you do not care about their order, you just ask for "everything in the slab group" and iterate.
- **`np.searchsorted(fem_data.node_ids, ni)`** works because `fem_data.node_ids` is sorted — that is guaranteed by the extraction step. If you were using raw Gmsh arrays you'd need to build your own id-to-index map; here the sortedness is part of the contract.
- **Accumulation across elements is essential.** A shared interior node receives a contribution from every triangle it touches; the `nodal_fz` dictionary accumulates them before you push anything to the solver. If the solver's `load` call overwrote, you'd lose every contribution but the last — so finish the lumping in Python first.
- **The cross-product area formula works in 3D.** The slab happens to be flat here, but the same code handles a tilted roof, a folded plate, or any non-planar triangulated surface. That robustness comes from `fem_data.node_coords` being true 3D coordinates straight out of the mesh.
- **Sanity check:** sum `nodal_fz.values()` and compare with `q · A_total`, where `A_total` is the sum of triangle areas from the same loop. They must be equal to machine precision. This is the cleanest proof that your lumping is correct, and it depends on nothing but `fem_data`.
- **For higher-order elements** (T6, Q4, Q8), the consistent nodal forces are *not* equal at every node — corner and mid-side nodes get different fractions. The `A/3` trick is a CST result. This is a nice opportunity to show students why consistent lumping is the safer mental model: the recipe is "integrate the shape functions", which scales to any element type without new bookkeeping.

---

## 5. Visualizations in `apeGmsh`

`apeGmsh` gives you several viewers, each aimed at catching a specific class of bug.

### 5.1 `m.model.viewer()` — geometry sanity

Pops the Gmsh GUI on the current CAD geometry. Use it right after `to_physical(...)` calls, before meshing. Confirm that every physical group contains what you think it contains (Gmsh colors entities by group). If a group is empty, your tolerance was too tight or you forgot a `sync()`. This is cheap, fast, and catches 90% of selection bugs.

### 5.2 `m.mesh.viewer()` — mesh sanity

Same GUI, now on the mesh. Check that refinement zones appear where you expect them (around the column tops here) and that there are no slivers, overlaps, or disconnected islands. If the columns look detached from the slab, your `make_conformal` tolerance was wrong.

### 5.3 `GeomTransfViewer` — local axis sanity (for beam elements)

```python
from apeGmsh.viewers import GeomTransfViewer

viewer = GeomTransfViewer()
viewer.show(node_i=[0, 0, 0], node_j=[0, 0, 3], vecxz=[1, 0, 0])
```

This tiny standalone viewer draws a single beam element and its local `x`, `y`, `z` axes given a `vecxz` vector. It exists because the most common bug in any 3D frame model is choosing a `vecxz` that is parallel to the element axis (degenerate) or that flips the local `y`/`z` in an unintended way. Seeing the local triad in 3D *before* you commit it to thousands of column elements removes an entire class of mistake — and because it runs on pure coordinates, you can use it without ever touching the solver.

### 5.4 Matplotlib overlay on `fem_data` — explicit and inspectable

The notebook also builds a deformed-shape figure entirely by hand using `matplotlib` (`Poly3DCollection` for the slab, `Line3DCollection` for the columns), plus a plan-view contour using `matplotlib.tri.Triangulation`. Pedagogically this is valuable because *nothing is hidden*: students can see:

- How `nid_to_idx = {int(nid): i for i, nid in enumerate(fem_data.node_ids)}` maps node tags into row indices.
- How `fem_data.physical.get_elements(2, slab_tag)['connectivity']` becomes a `(n_tri, 3)` index array.
- How that index array is used directly by `matplotlib.tri.Triangulation` (plan view) and by `Poly3DCollection` (3D view).

The point is not the matplotlib code — it is that `fem_data` gives you exactly the arrays matplotlib already wants. No reshaping, no lookup tables, no extra data structures.

### 5.5 `Results` + `results.viewer()` — the production view

```python
from apeGmsh import Results

results = Results.from_fem(
    fem_data,
    point_data={"Displacement": disp},
    cell_data={
        "VonMises":  sig_vm,
        "Stress_xx": sig_xx,
        "Stress_yy": sig_yy,
        "Stress_xy": sig_xy,
    },
)

results.viewer(blocking=False)
# results.to_vtu("output.vtu")
# results.to_pvd("modes")
```

`Results.from_fem` is the second important handoff in the library, after `FEMData`. It re-attaches solver outputs onto the *same mesh topology* that `fem_data` carries, split into:

- **`point_data`** — fields defined at nodes (displacements, reactions, nodal stresses).
- **`cell_data`** — fields defined per element (von Mises, stress components, element forces).

Once a `Results` object exists, the viewer gives you an interactive window with deformed shape warping, field contours, and (for time series) a time slider — all in memory, no temporary files. The same object can also be exported with `to_vtu` for a single step or `to_pvd` for a time series, so ParaView users can continue from there.

The symmetry is worth pointing out: `FEMData` is "mesh in", `Results` is "solution back on mesh out". Together they let `apeGmsh` round-trip through any solver without the solver ever knowing the library exists.

---

## 6. Suggested `apeGmsh`-focused exercises

These are the exercises that teach the library itself, independent of the solver:

1. **Tolerance sensitivity.** Change `remove_duplicates(tolerance=1)` to `1e-6`. Observe that the columns and slab no longer share nodes. This is the single most-debugged issue in CAD-to-mesh workflows.
2. **Skip `make_conformal`.** Comment it out and re-run. The column tops will land inside slab triangles but will not share vertices with them. Use `mesh.viewer()` to see the problem clearly.
3. **Rename a physical group.** Add a new slab region with a different name and switch the load lumping to use `get_tag(2, "new_name")`. This shows that the solver-side code depends on names, not tags.
4. **Swap the selection predicate.** Replace `on_plane=("z", 3000, 1e-3)` with `inside_box(...)` or another predicate and confirm the slab still gets picked up.
5. **Renumber before creating physical groups.** Intentionally call `renumber_mesh` before `to_physical` to observe what happens, then fix it by restoring the correct order.
6. **Export and reload `FEMData`.** Pickle `fem_data` to disk after `finalize()`, then load it in a fresh Python session and rerun the solver step. This proves the decoupling: once you have `fem_data`, you never need Gmsh again.
7. **Sanity-check the load lumping.** Sum `nodal_fz.values()` and compare to `q * total_slab_area`. They must agree to machine precision — a one-line test that guards the entire load path.

---

## 7. Mental model in one sentence

`apeGmsh` turns a CAD file into a **named, physical-group-addressable `FEMData`** that any FEM solver can consume with a handful of loops — and the library's job ends the moment that handoff happens.
