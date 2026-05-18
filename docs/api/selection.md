# Selection — the unified `.select()` idiom

One fluent, daisy-chainable selection idiom shared across all four
levels — geometry, the live mesh, the FEM broker, and results. Every
`.select()` returns a chain whose spatial verbs (`in_box`, `in_sphere`,
`on_plane`, `nearest_to`, `where`) and set algebra (`|`, `&`, `-`, `^`)
compose, ending in a level-appropriate terminal.

!!! note "Additive — the old methods still work"
    `.select()` is a **new canonical idiom that works alongside** the
    existing selectors. Nothing was removed, deprecated, or facaded.
    `fem.nodes.get`, `fem.elements.get`/`.resolve`,
    `results.nodes`/`elements.get`/`.in_box`/`.nearest_to`/`.on_plane`,
    `g.mesh_selection.add_nodes`/`add_elements`/`filter_set`,
    `g.model.queries.select`, `g.model.selection.select_*`, and the two
    legacy `Selection` classes all behave exactly as before. See
    [Old methods still work](#old-methods-still-work) below.

## Entry points

`.select()` is available at six entry points across the four levels.
The first call seeds the chain (by name, ids, or the full universe);
subsequent verbs refine it.

| Entry point | Level | Returns | Atoms |
|---|---|---|---|
| [`g.model.select(...)`][apeGmsh.core.Model.Model.select] | Geometry | [`GeometryChain`][apeGmsh.core._selection.GeometryChain] | `(dim, tag)` dimtags |
| [`g.mesh_selection.select(...)`][apeGmsh.mesh.MeshSelectionSet.MeshSelectionSet.select] | Live mesh | [`MeshSelectionChain`][apeGmsh.mesh._mesh_selection_chain.MeshSelectionChain] | node / element ids |
| `fem.nodes.select(...)` | FEM broker | [`NodeChain`][apeGmsh.mesh._node_chain.NodeChain] | node ids |
| `fem.elements.select(...)` | FEM broker | [`ElementChain`][apeGmsh.mesh._elem_chain.ElementChain] | element ids |
| `results.nodes.select(...)` | Results | [`ResultChain`][apeGmsh.results._result_chain.ResultChain] | node ids |
| `results.elements.select(...)` | Results | [`ResultChain`][apeGmsh.results._result_chain.ResultChain] | element ids |

`fem` is the [`FEMData`](fem.md) snapshot from
`g.mesh.queries.get_fem_data(dim)`; `results` is a
[`Results`](results.md) container.

## The verb surface

Every chain — regardless of level — exposes the **same verb names**
with the **same signatures** (the lone exception is `GeometryChain`'s
`in_box`, see [Two families](#two-families-point-vs-entity)). Each
refining verb returns a new chain of the *same* concrete type, so calls
daisy-chain:

| Verb | Signature | Keeps atoms that… |
|---|---|---|
| `in_box` | `in_box(lo, hi, *, inclusive=False)` | lie in the box `[lo, hi)` (half-open; point family) |
| `in_sphere` | `in_sphere(center, radius)` | lie within the closed ball of `radius` about `center` |
| `on_plane` | `on_plane(point, normal, *, tol)` | lie within `tol` of the plane (`tol` is **required**, keyword-only) |
| `nearest_to` | `nearest_to(point, *, count=1)` | are the `count` nearest to `point` (deterministic, lowest-index tie-break) |
| `where` | `where(predicate)` | satisfy `predicate(xyz)` on their coordinate row |

`lo` / `hi` / `center` / `point` / `normal` are 3-sequences.

!!! warning "`on_plane` has no default tolerance"
    `tol` is keyword-only with **no default** — always pass it
    explicitly, e.g. `.on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)`.

### Set algebra

Two chains of the **same type bound to the same engine** combine with
set operators or their named-method equivalents. The dedup law is
insertion-order-preserving: `a`'s order first, then `b`'s new atoms,
each atom once.

| Operator | Method | Meaning |
|---|---|---|
| `a \| b` | `a.union(b)` | atoms in either |
| `a & b` | `a.intersect(b)` | atoms in both |
| `a - b` | `a.difference(b)` | in `a`, not in `b` |
| `a ^ b` | `a.symmetric_difference(b)` | in exactly one |

Combining two **different** chain types (e.g. a `NodeChain` with a
`GeometryChain`), or two chains bound to **different** engines (two
different `FEMData` / `Results`), raises `TypeError` — set algebra is
loud across incompatible spaces. Pair selections from one engine.

### Terminals

The terminal differs per level — each `.select()` chain materialises
into the **same type the level's existing query already returns**, so a
chain is a drop-in for the old call:

| Level | Terminal | Returns |
|---|---|---|
| `g.model.select(...)` | `.result()` | the legacy [`Selection`][apeGmsh.core._selection.Selection] (`.to_label` / `.to_physical` / `.tags()` keep working) |
| `g.mesh_selection.select(...)` | `.result()` | the same-shape `dict` `MeshSelectionSet.get_nodes`/`get_elements` return; `.ids` lists the raw ids |
| `fem.nodes.select(...)` | `.result()` | a `NodeResult` — the **same type** `fem.nodes.get(...)` returns |
| `fem.elements.select(...)` | `.result()` | a `GroupResult` — the **same type** `fem.elements.get(...)` returns |
| `results.<nodes\|elements>.select(...)` | `.get(*, component, time=None, stage=None)` | the existing `NodeSlab` / `ElementSlab` — id/value parity with `results.<level>.get(...)` |

!!! note "Results selections need a component"
    A results chain identifies *where* to read; a slab read still needs
    *what*. `results.<level>.select(...).get(component=...)` is the
    terminal. Calling `.result()` on a results chain raises
    `RuntimeError` (a bare results selection is meaningless without a
    component).

## Two families: point vs entity

The chains split into two families with **honestly different** spatial
contracts. The CI contract test asserts per-family laws; it never
claims cross-family identical behavior.

### Point family

`NodeChain`, `ElementChain`, `ResultChain`, `MeshSelectionChain`
(`fem.*`, `results.*`, `g.mesh_selection`). Spatial verbs test node
**coordinates** (or element **centroids** for element-level chains):

- `in_box` is **half-open `[lo, hi)`** by default — an atom exactly on
  an upper face is excluded.
- `in_box(..., inclusive=True)` switches to the **closed box
  `[lo, hi]`** — upper-face atoms are kept.
- `in_sphere` is a closed ball; `on_plane` keeps atoms within `tol` of
  the plane.

### Entity family

`GeometryChain` (`g.model.select()`). Atoms are `(dim, tag)` CAD
dimtags; there is no single coordinate per entity, so the contract is
entity-typed:

- `in_box` delegates to `gmsh.model.getEntitiesInBoundingBox` — BRep
  bounding-box **CONTAINMENT** (the *whole* entity bbox must lie inside
  the query box, the box expanded by `Geometry.Tolerance` ≈ 1e-8). It
  is closed-ish, **not** an intersect and **not** half-open: a query
  box exactly equal to an entity's own extent will *not* contain it —
  enclose it comfortably.
- `in_box` therefore **cannot** honor the half-open / `inclusive=`
  knob. Passing `inclusive=` (or any keyword) raises `TypeError` — the
  knob is inexpressible here and is rejected loudly, never silently
  ignored.
- `in_sphere` / `nearest_to` / `where` use the entity bbox **centre**;
  `on_plane` keeps an entity iff *all 8* bbox corners are within `tol`.
  These are coarse entity proxies — for exact geometric on/crossing
  semantics use the unchanged
  [`g.model.queries.select(on=/crossing=)`](model.md) predicate.

## Examples per level

### Geometry — `g.model.select()`

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="frame") as g:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.model.sync()
    faces = g.model.queries.boundary("box", dim=3, oriented=False)
    g.physical.add_surface([int(t) for _d, t in faces], name="Faces")

    # Seed by PG name (tiered name resolution, contract-locked),
    # refine with entity-family spatial verbs, materialise the
    # legacy Selection and register a physical group.
    (g.model.select("Faces")                       # -> GeometryChain
        .in_box((-0.1, -0.1, -0.1), (1.1, 1.1, 1.1))  # gmsh BRep containment
        .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
        .result()                                  # -> legacy Selection
        .to_physical("Base"))
```

`select(target=None, *, dim=None)` accepts anything the locked geometry
resolver accepts — a label / PG / part name, a bare int tag, a
`(dim, tag)` pair, or a list. `dim` is the resolver's `default_dim`
(used for bare ints and `target=None`), **not** a post-filter.

### Live mesh — `g.mesh_selection.select()`

```python
# After meshing, still in the live session
node_set = (g.mesh_selection.select()                  # node level
    .in_box((0, 0, 0), (1, 1, 1))                      # half-open [lo, hi)
    .on_plane((0, 0, 0), (0, 0, 1), tol=1e-9)
    .result())                                         # {'tags', 'coords'}

# Element level — atoms are element ids of `dim`, verbs use centroids
hexes = (g.mesh_selection.select(level="element", dim=3)
    .in_box((0, 0, 0), (1, 1, 1), inclusive=True)      # closed box
    .ids)
```

`select(*, level="node", dim=2, ids=None)` — `level="element"` uses
`dim`; `ids=` seeds an explicit list, otherwise the full live-mesh
universe. This is the fluent equivalent of
`g.mesh_selection.add_nodes(in_box=..., on_plane=...)`.

### FEM broker — `fem.nodes.select()` / `fem.elements.select()`

```python
fem = g.mesh.queries.get_fem_data(dim=3)

# Same selectors as fem.nodes.get (target / pg / label / tag /
# partition / dim) plus ids=; .result() is the same NodeResult
# fem.nodes.get returns.
top = (fem.nodes.select(pg="Body")
    .in_box((0, 0, 0), (1, 1, 1))                      # half-open
    .on_plane((0, 0, 1), (0, 0, 1), tol=1e-6)
    .result())                                         # -> NodeResult

# Set algebra across two node chains on the same FEMData
a = fem.nodes.select(ids=[1, 2, 3])
b = fem.nodes.select(ids=[2, 3, 4])
both = (a | b).result()                                # union, deduped

# Elements — atoms are element ids; spatial verbs use centroids
core = (fem.elements.select(pg="Body")
    .in_box((0.0, 0.0, 0.0), (0.75, 0.75, 0.75))
    .result())                                         # -> GroupResult
```

### Results — `results.nodes.select()` / `results.elements.select()`

```python
# results bound to a fem (Results(..., fem=...) or results.bind(fem))
slab = (results.nodes.select(pg="Base")
    .in_box(lo, hi)                                    # half-open
    .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
    .get(component="displacement_x"))                  # -> NodeSlab

# Element results — spatial verbs operate on element centroids
forces = (results.elements.select(pg="Beams")
    .in_box(lo, hi)
    .get(component="globalForce"))                     # -> ElementSlab
```

`results.nodes.select` accepts `pg` / `label` / `selection` / `ids`;
`results.elements.select` adds `element_type=`. The `.get(...)` terminal
delegates to the existing reader, so it is id/value parity with
`results.<level>.get(...)`.

## Old methods still work

`.select()` is purely additive. None of the following changed — use
whichever idiom fits:

- `fem.nodes.get(...)`, `fem.elements.get(...)` / `.resolve(...)`
- `results.nodes`/`results.elements` `.get` / `.in_box` /
  `.nearest_to` / `.on_plane`
- `g.mesh_selection.add_nodes` / `add_elements` / `filter_set`
  (including their `name=` named-persistence path)
- `g.model.queries.select(...)` (the `on=`/`crossing=` predicate
  selector) and `g.model.selection.select_*`
- the legacy [`Selection`][apeGmsh.core._selection.Selection] (the
  `g.model.select(...).result()` terminal) and the `viz` `Selection`

Frame `.select()` as the new canonical idiom that composes spatial
refinement and set algebra; reach for the legacy methods when you need
their specific behavior (e.g. the exact `on=`/`crossing=` geometric
predicates, or named, round-tripping mesh-selection persistence).

## Behavior changes

Two pre-existing, intentionally-pinned behavior changes ride alongside
this work. Brief notes here; the full migration text is in the
[changelog](../changelog.md).

!!! warning "S2 — `g.mesh_selection` box default flipped"
    `g.mesh_selection`'s box filters
    (`add_nodes(in_box=)` / `add_elements(in_box=)` /
    `filter_set(in_box=)`) changed from a **closed** `[lo, hi]` default
    to **half-open** `[lo, hi)`, to match `results` (which was already
    half-open). Pass `inclusive=True` to restore the old closed box.
    `results`' box was already half-open and is unchanged. See the
    [changelog](../changelog.md) for the migration note.

!!! warning "S5 — three formerly-silent paths now raise"
    Three paths that previously bound to an empty/wrong set now fail
    loud: `results` with `selection=` on an import-origin
    (`from_msh`/MPCO/native) `FEMData` raises `RuntimeError`; a
    loads/masses `__ms__` target with no info raises `KeyError`
    (instead of silently binding to zero nodes); and `results`
    element-centroid computation raises `KeyError` on an unknown
    connectivity node — which also makes the legacy
    `results.elements.in_box`/`nearest_to`/`on_plane` helpers fail
    loud. See the [changelog](../changelog.md).

## Planned / not yet available

Tracked follow-ups — **not** yet shipped; do not rely on them:

- **Results sub-composite `.select()`** — `results.nodes` and
  `results.elements` have `.select()`, but the five element
  sub-composites (`gauss`, `fibers`, `layers`, `line_stations`,
  `springs`) do **not** yet. Planned as a uniform per-terminal
  kwarg-forwarding follow-on.
- **`g.mesh_selection.select()` name-seed** — today only `ids=` (or
  the full universe) seeds a live-mesh chain. Seeding by an existing
  set name / gmsh PG / label is planned (it must reuse the locked
  resolver, never re-implement it).

!!! note "Chained selections are query-only by design"
    There is **no** `.save_as(name)` on any chain. Chained selections
    are ephemeral query objects. Named, round-tripping persistence
    stays the pre-mesh author-time path
    (`g.mesh_selection.add_nodes(..., name=...)` → `FEMData` snapshot →
    `results(selection=...)`), which already works and is unchanged.

## Reference

The shared base mixin (chaining + set-algebra + definition-time
verb-name enforcement). The per-family spatial behavior lives on the
concrete subclasses listed under [Entry points](#entry-points).

::: apeGmsh._chain.SelectionChain

### `GeometryChain` — entity family

::: apeGmsh.core._selection.GeometryChain

### `NodeChain` — point family (FEM nodes)

::: apeGmsh.mesh._node_chain.NodeChain

### `ElementChain` — point family (FEM elements, centroid-based)

::: apeGmsh.mesh._elem_chain.ElementChain

### `ResultChain` — point family (results, bi-level)

::: apeGmsh.results._result_chain.ResultChain

### `MeshSelectionChain` — point family (live mesh, bi-level)

::: apeGmsh.mesh._mesh_selection_chain.MeshSelectionChain
