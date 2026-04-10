# Migrating to pyGmsh → apeGmsh v1.0

v1.0 bundles two breaking changes into a single release:

1. **Package rename**: `pyGmsh` → `apeGmsh`
2. **API restructure**: `g.model.*` methods split into five focused
   sub-composites (`geometry`, `boolean`, `transforms`, `io`, `queries`),
   plus several smaller cleanups.

This guide is a drop-in find/replace checklist. Most codebases can
migrate in one pass with a single search across the project.

**If you were on v0.3.0 (last `pyGmsh` release)**, apply all sections below.
**If you were on v0.3.1 (the transitional `apeGmsh` rename)**, skip section 1.

---

## 1. Package rename: `pyGmsh` → `apeGmsh`

```diff
-from pyGmsh import pyGmsh
+from apeGmsh import apeGmsh

-from pyGmsh import Part, PartsRegistry, Instance
+from apeGmsh import Part, PartsRegistry, Instance

-from pyGmsh.viewers.ui.theme import STYLESHEET
+from apeGmsh.viewers.ui.theme import STYLESHEET

-g = pyGmsh(model_name="bridge")
+g = apeGmsh(model_name="bridge")
```

The class name stays lowercase to match the package (`from apeGmsh import apeGmsh`).

**Install:** uninstall the old wheel (`pip uninstall pyGmsh`) and install the
new one (`pip install -e .` from the repo root).

**What's NOT renamed:** the companion app `pyGmshViewer` keeps its own name.
Its internal imports now read `from apeGmsh.viewers.ui.theme import ...` but
the entry point `pygmsh-viewer` and the package `pyGmshViewer` are unchanged.

---

## 2. Model methods → sub-composites

`Model` is now a composition of five focused sub-composites. Every geometry
method moved behind a categorical namespace.

### Geometry primitives (`g.model.geometry.*`)

```diff
-p = g.model.add_point(0, 0, 0)
+p = g.model.geometry.add_point(0, 0, 0)

-l = g.model.add_line(p1, p2)
+l = g.model.geometry.add_line(p1, p2)

-box = g.model.add_box(0, 0, 0, 1, 1, 1)
+box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)

-cyl = g.model.add_cylinder(0, 0, 0, 0, 0, 10, 0.5)
+cyl = g.model.geometry.add_cylinder(0, 0, 0, 0, 0, 10, 0.5)
```

All 19 methods move: `add_point`, `add_line`, `add_arc`, `add_circle`,
`add_ellipse`, `add_spline`, `add_bspline`, `add_bezier`, `add_wire`,
`add_curve_loop`, `add_plane_surface`, `add_surface_filling`,
`add_rectangle`, `add_box`, `add_sphere`, `add_cylinder`, `add_cone`,
`add_torus`, `add_wedge`.

### Boolean operations (`g.model.boolean.*`)

```diff
-result = g.model.fuse(a, b)
+result = g.model.boolean.fuse(a, b)

-part = g.model.cut(box, hole)
+part = g.model.boolean.cut(box, hole)

-common = g.model.intersect(a, b)
+common = g.model.boolean.intersect(a, b)

-pieces = g.model.fragment(objects=[1], tools=[2])
+pieces = g.model.boolean.fragment(objects=[1], tools=[2])
```

### Transforms & sweeps (`g.model.transforms.*`)

```diff
-g.model.translate(v, 0, 0, 5)
+g.model.transforms.translate(v, 0, 0, 5)

-g.model.rotate(v, angle=np.pi/4, ax=0, ay=0, az=1)
+g.model.transforms.rotate(v, angle=np.pi/4, ax=0, ay=0, az=1)

-g.model.scale(v, 2, 2, 2)
+g.model.transforms.scale(v, 2, 2, 2)

-g.model.mirror(v, 1, 0, 0, 0)
+g.model.transforms.mirror(v, 1, 0, 0, 0)

-copies = g.model.copy(v)
+copies = g.model.transforms.copy(v)

-ext = g.model.extrude(surf, 0, 0, 10)
+ext = g.model.transforms.extrude(surf, 0, 0, 10)

-rev = g.model.revolve(surf, np.pi, ax=0, ay=1, az=0)
+rev = g.model.transforms.revolve(surf, np.pi, ax=0, ay=1, az=0)

-swept = g.model.sweep(profile, path)
+swept = g.model.transforms.sweep(profile, path)

-loft = g.model.thru_sections([w1, w2])
+loft = g.model.transforms.thru_sections([w1, w2])
```

**Fluent chaining still works** on sub-composites:

```python
(g.model.transforms
    .translate(v, 5, 0, 0)
    .rotate(v, angle=np.pi/2, ax=0, ay=0, az=1))
```

### Import / export (`g.model.io.*`)

```diff
-g.model.load_step("bracket.step")
+g.model.io.load_step("bracket.step")

-g.model.save_step("output.step")
+g.model.io.save_step("output.step")

-g.model.load_iges("file.iges")
+g.model.io.load_iges("file.iges")

-g.model.load_dxf("plan.dxf")
+g.model.io.load_dxf("plan.dxf")

-g.model.save_msh("mesh.msh")
+g.model.io.save_msh("mesh.msh")

-g.model.heal_shapes(tolerance=1e-6)
+g.model.io.heal_shapes(tolerance=1e-6)
```

### Queries & cleanup (`g.model.queries.*`)

```diff
-bb = g.model.bounding_box(tag)
+bb = g.model.queries.bounding_box(tag)

-cm = g.model.center_of_mass(tag)
+cm = g.model.queries.center_of_mass(tag)

-vol = g.model.mass(tag)
+vol = g.model.queries.mass(tag)

-bdry = g.model.boundary(tags)
+bdry = g.model.queries.boundary(tags)

-up, down = g.model.adjacencies(tag)
+up, down = g.model.queries.adjacencies(tag)

-ents = g.model.entities_in_bounding_box(0, 0, 0, 1, 1, 1)
+ents = g.model.queries.entities_in_bounding_box(0, 0, 0, 1, 1, 1)

-g.model.remove([(3, 1)])
+g.model.queries.remove([(3, 1)])

-g.model.remove_duplicates()
+g.model.queries.remove_duplicates()

-g.model.make_conformal()
+g.model.queries.make_conformal()

-df = g.model.registry()
+df = g.model.queries.registry()
```

### What stays on `Model` directly

These methods remain flat on `g.model`:

- `g.model.sync()`
- `g.model.viewer(fem=...)`
- `g.model.gui()`
- `g.model.launch_picker()`
- `g.model.selection` (the entity selection sub-composite)

And the existing `g.parts`, `g.physical`, `g.constraints`, `g.loads`,
`g.mesh_selection` composites are unchanged in location.

---

## 3. Rename: `g.mass` → `g.masses`

Every other composite is plural. The last outlier is fixed.

```diff
-g.mass.volume("concrete", density=2400)
+g.masses.volume("concrete", density=2400)

-g.mass.point("tip", mass=500)
+g.masses.point("tip", mass=500)

-g.mass.line("beam", linear_density=80)
+g.masses.line("beam", linear_density=80)

-g.mass.surface("slab", areal_density=300)
+g.masses.surface("slab", areal_density=300)
```

The FEMData field is also renamed:

```diff
 fem = g.mesh.get_fem_data(dim=3)
-print(fem.mass.total_mass())
+print(fem.masses.total_mass())
```

The class names (`MassesComposite`, `MassDef`, `MassRecord`, `MassSet`)
are unchanged — only the session attribute and FEMData field renamed.

**Not renamed** (intentionally): `r.mass` on `MassRecord` is the value field,
`d.mass` on `PointMassDef`, `g.model.queries.mass(tag)` (geometric mass),
and `ops.mass(...)` (OpenSees solver command).

---

## 4. Removed legacy aliases and deprecated methods

These had lived as shims or `DeprecationWarning` for one or more releases.
v1.0 deletes them entirely.

### Session lifecycle

```diff
-g.initialize()   # old alias
+g.begin()

-g.finalize()     # old alias
+g.end()

-g._initialized   # old property
+g.is_active

-g.model_name     # old property alias for g.name
+g.name
```

### Fast/slow viewer split

```diff
-g.model.viewer_fast()   # always fast now
+g.model.viewer()

-g.mesh.viewer_fast()
+g.mesh.viewer()
```

### Parts / PhysicalGroups separation

```diff
-# Old: auto 1:1 mapping from parts to physical groups
-g.parts.add_physical_groups()

+# New: be explicit about which entities join which group
+inst = g.parts.get("i_beam")
+g.physical.add_volume(inst.entities[3], name="steel")
```

Parts track assembly identity ("which geometry belongs to which part").
Physical groups tag regions for solver purposes ("which entities get
this material / BC"). A part can have many groups; a group can span
parts. The auto-mapping collapsed this richness.

### OpenSees bridge

```diff
-g.opensees.add_nodal_load(
-    "Wind", "WindwardFace", force=[1e4, 0, 0]
-)

+with g.loads.pattern("Wind"):
+    g.loads.point("WindwardFace", force_xyz=(1e4, 0, 0))

 # Then `fem.loads` is auto-populated by get_fem_data(),
 # and the OpenSees bridge consumes it via consume_loads_from_fem()
 # (auto-called from build_from_fem()).
```

### Mesh selection parameter alias

```diff
 g.mesh_selection.add_nodes(
-    nearest_to=(1.0, 0.5, 0.0),     # deprecated alias
+    closest_to=(1.0, 0.5, 0.0),
     count=3,
 )
```

### Convenience delegates on the session

```diff
-g.remove_duplicates()        # apeGmsh convenience delegate
+g.model.queries.remove_duplicates()

-g.make_conformal()
+g.model.queries.make_conformal()
```

---

## 5. Automated migration

For a mechanical find-replace across your own codebase, this Python
script handles every transform in this guide:

```python
#!/usr/bin/env python3
"""Migrate pyGmsh v0.x → apeGmsh v1.0 across a project."""
import re, os, sys

GEOMETRY = {
    'add_point', 'add_line', 'add_arc', 'add_circle', 'add_ellipse',
    'add_spline', 'add_bspline', 'add_bezier', 'add_wire',
    'add_curve_loop', 'add_plane_surface', 'add_surface_filling',
    'add_rectangle', 'add_box', 'add_sphere', 'add_cylinder',
    'add_cone', 'add_torus', 'add_wedge',
}
BOOLEAN = {'fuse', 'cut', 'intersect', 'fragment'}
TRANSFORMS = {
    'translate', 'rotate', 'scale', 'mirror', 'copy',
    'extrude', 'revolve', 'sweep', 'thru_sections',
}
IO = {
    'load_iges', 'load_step', 'load_dxf', 'load_msh',
    'save_iges', 'save_step', 'save_dxf', 'save_msh', 'heal_shapes',
}
QUERIES = {
    'remove', 'remove_duplicates', 'make_conformal', 'fragment_all',
    'bounding_box', 'center_of_mass', 'mass', 'boundary',
    'adjacencies', 'entities_in_bounding_box', 'registry',
}

def sub_for(m):
    if m in GEOMETRY: return 'geometry'
    if m in BOOLEAN: return 'boolean'
    if m in TRANSFORMS: return 'transforms'
    if m in IO: return 'io'
    if m in QUERIES: return 'queries'
    return None

pat_model = re.compile(r'(\.model\.)([a-zA-Z_][a-zA-Z_0-9]*)')

def transform(src):
    def repl(m):
        method = m.group(2)
        sub = sub_for(method)
        return f'.model.{sub}.{method}' if sub else m.group(0)
    out = pat_model.sub(repl, src)
    out = out.replace('from pyGmsh', 'from apeGmsh')
    out = out.replace('import pyGmsh', 'import apeGmsh')
    out = re.sub(r'\bpyGmsh\(', 'apeGmsh(', out)
    out = re.sub(r'\bg\.mass\.', 'g.masses.', out)
    out = re.sub(r'\bfem\.mass\b', 'fem.masses', out)
    out = re.sub(r'\bg\.initialize\(\)', 'g.begin()', out)
    out = re.sub(r'\bg\.finalize\(\)', 'g.end()', out)
    out = re.sub(r'\bg\.model_name\b', 'g.name', out)
    return out

SKIP = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}

for dirpath, dirnames, filenames in os.walk(sys.argv[1] if len(sys.argv) > 1 else '.'):
    dirnames[:] = [d for d in dirnames if d not in SKIP]
    for fn in filenames:
        if not (fn.endswith('.py') or fn.endswith('.ipynb') or fn.endswith('.md')):
            continue
        path = os.path.join(dirpath, fn)
        with open(path, 'r', encoding='utf-8') as fh:
            src = fh.read()
        new = transform(src)
        if new != src:
            with open(path, 'w', encoding='utf-8', newline='') as fh:
                fh.write(new)
            print(f'  updated: {path}')
```

Save as `migrate_v1.py` and run: `python migrate_v1.py /path/to/your/project`

**Caveats:**
- Backup first. Run it on a clean git branch.
- The regex `\bg\.mass\.` requires a dot after `mass` — it will catch
  `g.mass.point(...)` but not a bare `g.mass` reference. Grep manually
  for any leftover bare references.
- `gmsh.initialize()` / `gmsh.finalize()` at the raw Gmsh level are
  preserved — the script only rewrites `g.initialize` (our wrapper).
- If your project has custom subclasses of `Model` that referenced
  the old `_GeometryMixin` etc. classes directly, those need manual
  updates (the mixin classes are gone — replaced with `_Geometry`
  etc. that take a `model` arg in `__init__`).

---

## 6. What didn't change

- The session lifecycle contract (`g.begin()` / `g.end()` / context manager)
- `g.parts`, `g.physical`, `g.constraints`, `g.loads`, `g.mesh_selection`
  composite APIs
- `fem.node_ids`, `fem.node_coords`, `fem.element_ids`, `fem.connectivity`,
  `fem.physical`, `fem.constraints`, `fem.loads` — all unchanged
- The OpenSees bridge's `consume_*_from_fem` / `build` / `export_tcl` /
  `export_py` pipeline
- `Part` and `PartsRegistry` APIs (only docstring examples were swept)
- `g.mesh.generate()`, `g.mesh.set_global_size()`, `g.mesh.field.*`, etc.

---

## 7. Full example — old vs new

### Old (v0.x pyGmsh)

```python
from pyGmsh import pyGmsh

g = pyGmsh(model_name="cantilever")
g.initialize()

v = g.model.add_box(0, 0, 0, 1, 0.2, 5)
g.model.translate(v, 0.5, 0, 0)
bb = g.model.bounding_box(v)

g.physical.add_volume([v], name="concrete")

with g.loads.pattern("dead"):
    g.loads.gravity("concrete", g=(0, 0, -9.81), density=2400)
g.mass.volume("concrete", density=2400)

g.mesh.set_global_size(0.5)
g.mesh.generate(3)
fem = g.mesh.get_fem_data(3)

print(f"total mass: {fem.mass.total_mass():.0f} kg")
g.finalize()
```

### New (v1.0 apeGmsh)

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="cantilever") as g:
    v = g.model.geometry.add_box(0, 0, 0, 1, 0.2, 5)
    g.model.transforms.translate(v, 0.5, 0, 0)
    bb = g.model.queries.bounding_box(v)

    g.physical.add_volume([v], name="concrete")

    with g.loads.pattern("dead"):
        g.loads.gravity("concrete", g=(0, 0, -9.81), density=2400)
    g.masses.volume("concrete", density=2400)

    g.mesh.set_global_size(0.5)
    g.mesh.generate(3)
    fem = g.mesh.get_fem_data(3)

    print(f"total mass: {fem.masses.total_mass():.0f} kg")
```

Note: this example also uses the context manager (`with`) — that's
not new in v1.0 but it replaces the old `g.initialize()` / `g.finalize()`
pattern shown in the "old" version.
