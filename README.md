# pyGmsh

Gmsh wrapper for structural FEM workflows with an Abaqus-style Part/Assembly architecture and OpenSees integration.

## Installation

```bash
cd pyGmsh
pip install -e .

# With all optional dependencies
pip install -e ".[all]"
```

Requires Gmsh (with Python bindings), NumPy, and Pandas. Optional extras: matplotlib (plotting), openseespy (analysis), gmsh2opensees (one-liner mesh transfer).

## Quick start

### Standalone (single model)

```python
from pyGmsh import pyGmsh

g = pyGmsh(model_name="plate", verbose=True)
g.initialize()

p1 = g.model.add_point(0, 0, 0, lc=10)
p2 = g.model.add_point(100, 0, 0, lc=10)
p3 = g.model.add_point(100, 50, 0, lc=10)
p4 = g.model.add_point(0, 50, 0, lc=10)

l1 = g.model.add_line(p1, p2)
l2 = g.model.add_line(p2, p3)
l3 = g.model.add_line(p3, p4)
l4 = g.model.add_line(p4, p1)

loop = g.model.add_curve_loop([l1, l2, l3, l4])
surf = g.model.add_plane_surface(loop)

g.mesh.generate(2)
mesh = g.mesh.get_numbered_mesh(dim=2)
print(mesh.summary())

g.finalize()
```

### Part / Assembly (multi-part)

```python
from pyGmsh import Part, Assembly

# Build geometry in isolated sessions
web = Part("web")
web.begin()
# ... build geometry with web.model.add_point(), add_line(), etc.
web.save("web.step")
web.end()

flange = Part("flange")
flange.begin()
# ... build geometry
flange.save("flange.step")
flange.end()

# Assemble, fragment, mesh
asm = Assembly("I_beam")
asm.begin()
asm.add_part(web)
asm.add_part(flange, translate=(0, 0, 200), label="top_flange")
asm.add_part(flange, label="bot_flange")

asm.fragment_all()
asm.mesh.generate(dim=2)
mesh = asm.mesh.get_numbered_mesh(dim=2)

# mesh.node_ids, mesh.node_coords, mesh.connectivity → feed to any solver
asm.end()
```

## Architecture

Two main classes with a clean separation:

- **Part** -- geometry modeler. Creates points, lines, surfaces, volumes in an isolated Gmsh session. Exports to STEP. Has only `model` and `inspect` composites.

- **Assembly** -- full Gmsh wrapper. Imports Parts, positions them, fragments for conformal interfaces, meshes, assigns physical groups, extracts FEM data, resolves constraints. Has all composites: `mesh`, `model`, `physical`, `view`, `plot`, `inspect`, `partition`, `g2o`.

The STEP file is the only contract between Part and Assembly -- they are fully decoupled.

### Composites

| Composite        | Access           | Purpose                                  |
|:-----------------|:-----------------|:-----------------------------------------|
| `Model`          | `.model`         | Geometry building, sync, export          |
| `Mesh`           | `.mesh`          | Generation, size control, FEM extraction |
| `PhysicalGroups` | `.physical`      | Named entity groups for BCs/loads        |
| `View`           | `.view`          | Gmsh post-processing scalar/vector views |
| `Plot`           | `.plot`          | Matplotlib visualisations                |
| `Inspect`        | `.inspect`       | Geometry and mesh queries                |
| `Partition`      | `.partition`     | Mesh partitioning                        |
| `Gmsh2OpenSees`  | `.g2o`           | One-liner mesh transfer to OpenSees      |

### Numberer

The `Numberer` remaps non-contiguous Gmsh tags to contiguous 1-based solver IDs with bidirectional maps (`gmsh_to_solver_node`, `solver_to_gmsh_node`, etc.). Supports simple sequential numbering and RCM bandwidth optimisation.

```python
mesh = g.mesh.get_numbered_mesh(dim=2, method="rcm")
# mesh.node_ids, mesh.elem_ids, mesh.connectivity
# mesh.gmsh_to_solver_node, mesh.solver_to_gmsh_node
```

### Constraints

Solver-agnostic two-stage constraint pipeline on the Assembly:

1. **Define** (pre-mesh): `asm.equal_dof(...)`, `asm.tie(...)`, `asm.rigid_link(...)`, etc.
2. **Resolve** (post-mesh): `asm.resolve_constraints(...)` produces concrete node pairs, weights, and constraint matrices for any solver.

Four levels: node-to-node, node-to-group, node-to-surface, surface-to-surface.

## Project layout

```
pyGmsh/
  pyproject.toml
  README.md
  src/
    pyGmsh/
      __init__.py
      _core.py           # standalone pyGmsh class
      Part.py
      Assembly.py
      Mesh.py
      Model.py
      View.py
      Plot.py
      Inspect.py
      PhysicalGroups.py
      Partition.py
      Gmsh2OpenSees.py
      VTKExport.py
      Constraints.py
      Numberer.py
      OpenSees.py
  examples/
    example_plate_pyGmsh.ipynb
    example_assembly.ipynb
    ...
```

## Examples

See the `examples/` directory. After `pip install -e .`, all notebooks can be run directly without sys.path manipulation.
