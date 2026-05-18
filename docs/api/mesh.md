# Mesh — `g.mesh`

Meshing composite. Seven focused sub-composites.

## `g.mesh`

::: apeGmsh.mesh.Mesh.Mesh

## Sub-composites

### `g.mesh.generation`

::: apeGmsh.mesh._mesh_generation._Generation

### `g.mesh.sizing`

::: apeGmsh.mesh._mesh_sizing._Sizing

### `g.mesh.field`

::: apeGmsh.mesh._mesh_field.FieldHelper

### `g.mesh.options`

::: apeGmsh.mesh._mesh_options._Options

#### Common recipes

```python
# All-hex output for an unstructured tet model
g.mesh.options.set_subdivision_algorithm("all_hex")

# Quadratic elements (8-node hex → 27-node hex, 4-node tet → 10-node tet)
g.mesh.options.set_element_order(2)

# HXT algorithm — much faster than Delaunay for large unstructured tet meshes
g.mesh.options.set_algorithm_3d("hxt")

# Highest-quality all-quad surface mesh
g.mesh.options.set_recombination_algorithm("blossom_full")
g.mesh.options.set_algorithm_2d("frontal_quads")

# Fluent chaining — every setter returns self
(g.mesh.options
    .set_algorithm_3d("hxt")
    .set_element_order(2)
    .set_smoothing(iterations=3))
```

For options not wrapped here, drop to `gmsh.option.setNumber("Mesh.X", v)`
directly. The wrapper covers the ~6 options users actually tune; Gmsh has
~50+ `Mesh.*` options total.

### `g.mesh.structured`

::: apeGmsh.mesh._mesh_structured._Structured

#### One-call recipe — `set_transfinite()`

For the common case of "apply transfinite + recombine to one entity, many,
or the whole model," use the unified
[`set_transfinite()`](#apeGmsh.mesh._mesh_structured._Structured.set_transfinite)
method. It infers the dim from the target and runs the appropriate cascade
(curves → faces → recombine → volume):

```python
# Whole model, uniform — typical "just give me hexes"
g.mesh.structured.set_transfinite(n=11)

# Axis-aligned hex, per-axis sizing (most readable)
g.mesh.structured.set_transfinite("layer_top",
                                   n={"x": 101, "y": 101, "z": 6})

# Rotated hex, per principal axis (works for any orientation)
g.mesh.structured.set_transfinite("rotated_box", n=(11, 11, 21))

# Length-based sizing (per axis)
g.mesh.structured.set_transfinite("layer_top",
                                   size={"x": 100, "y": 100, "z": 10})
```

**Sizing forms** — all three coexist; pick by readability:

| Form | Use when | Example |
|---|---|---|
| **scalar** `n=11` | uniform on every edge; any orientation | `n=11` |
| **dict** `n={"x":..., "y":..., "z":...}` | axis-aligned hex, want per-axis counts | `n={"x":11, "y":11, "z":21}` |
| **tuple** `n=(n1, n2, n3)` | any rotation; counts in `(X-closest, Y-closest, Z-closest)` order | `n=(11, 11, 21)` |

**Behavior on incompatible geometry** — entities whose edges don't cluster
into the expected principal-axis count (e.g. a face split by a boolean op
into a 5-sided patch) are **warned and skipped** rather than failing the
whole call. Use
[`set_transfinite_automatic()`](#apeGmsh.mesh._mesh_structured._Structured.set_transfinite_automatic)
if you want silent skipping with no warning.

For per-edge bias (Progression / Bump / Beta with `coef=`), explicit
corner lists, or custom triangle arrangement, drop to the granular methods:
`set_transfinite_curve()`, `set_transfinite_surface()`,
`set_transfinite_volume()`.

### `g.mesh.editing`

::: apeGmsh.mesh._mesh_editing._Editing

### `g.mesh.queries`

::: apeGmsh.mesh._mesh_queries._Queries

### `g.mesh.partitioning`

::: apeGmsh.mesh._mesh_partitioning._Partitioning

## Fluent selection — `g.mesh_selection.select()`

`g.mesh_selection` (a session attribute,
[`MeshSelectionSet`][apeGmsh.mesh.MeshSelectionSet.MeshSelectionSet])
is the live-mesh entry of the unified, daisy-chainable
[selection idiom](selection.md). It is **additive** —
`add_nodes` / `add_elements` / `filter_set` (including their `name=`
named-persistence path) are unchanged. `select()` returns a
[`MeshSelectionChain`][apeGmsh.mesh._mesh_selection_chain.MeshSelectionChain]
(point family, bi-level); `.result()` is the same-shape `dict`
`get_nodes` / `get_elements` return.

```python
node_set = (g.mesh_selection.select()              # node level
    .in_box((0, 0, 0), (1, 1, 1))                  # half-open [lo, hi)
    .on_plane((0, 0, 0), (0, 0, 1), tol=1e-9)
    .result())

hexes = g.mesh_selection.select(level="element", dim=3).in_box(
    lo, hi, inclusive=True).ids
```

!!! warning "S2 — box default flipped to half-open"
    `add_nodes(in_box=)` / `add_elements(in_box=)` /
    `filter_set(in_box=)` changed from a **closed** `[lo, hi]` default
    to **half-open** `[lo, hi)` (to match `results`). Pass
    `inclusive=True` to restore the old closed box. See the
    [changelog](../changelog.md) for the migration note.

Name-based seeding for `select()` and chain persistence are tracked
follow-ups — see [Selection](selection.md) for the full idiom and the
planned/not-yet items.

## Supporting types

::: apeGmsh.mesh.PhysicalGroups.PhysicalGroups

::: apeGmsh.mesh.MeshSelectionSet.MeshSelectionSet

::: apeGmsh.mesh.MeshSelectionSet.MeshSelectionStore

::: apeGmsh.mesh.MshLoader.MshLoader

!!! warning "Legacy"
    `Partition` below is the standalone, pre-composite class. New code
    should use the live `g.mesh.partitioning` composite documented above.

::: apeGmsh.mesh.Partition.Partition

::: apeGmsh.mesh.View.View

## Algorithms & enums

::: apeGmsh.mesh._mesh_algorithms

::: apeGmsh.mesh._mesh_partitioning.RenumberResult

::: apeGmsh.mesh._mesh_partitioning.PartitionInfo

## Group sets

::: apeGmsh.mesh._group_set.PhysicalGroupSet

::: apeGmsh.mesh._group_set.LabelSet
