# FEM Broker — `FEMData`

Solver-agnostic snapshot returned by `g.mesh.queries.get_fem_data(dim)`.
Composite of `.nodes`, `.elements`, `.info`, `.inspect`, `.mesh_selection`.
The `.physical`, `.loads`, `.masses`, `.constraints`, and `.sp` views
are reached through `.nodes.*` / `.elements.*` rather than living
directly on `FEMData`.

Also surfaces `snapshot_id` plus `to_native_h5` / `from_native_h5` /
`from_mpco_model` for binding and round-tripping with Results.

## Fluent selection — `.nodes.select()` / `.elements.select()`

`fem.nodes.select(...)` and `fem.elements.select(...)` are the FEM
broker entries of the unified, daisy-chainable
[selection idiom](selection.md). They are **additive** — `fem.nodes.get`
and `fem.elements.get` / `.resolve` are unchanged. `.select()` accepts
the **same selectors** as `.get` (plus `ids=`), and `.result()` returns
the **same type** `.get` returns (`NodeResult` / `GroupResult`).

```python
top = (fem.nodes.select(pg="Body")
    .in_box((0, 0, 0), (1, 1, 1))                  # half-open [lo, hi)
    .on_plane((0, 0, 1), (0, 0, 1), tol=1e-6)
    .result())                                     # -> NodeResult

both = (fem.nodes.select(ids=a) | fem.nodes.select(ids=b)).result()
```

Node/element chains are the **point family** — `in_box` is half-open
`[lo, hi)` by default; `inclusive=True` restores the closed box.
Element spatial verbs operate on element centroids. See
[Selection](selection.md) for the full idiom.

::: apeGmsh.mesh.FEMData
