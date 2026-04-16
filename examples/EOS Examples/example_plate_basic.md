# Thick-Walled Cylinder Under Internal Pressure (Lame Problem)

## Gmsh Mesh Generation + OpenSeesPy Finite Element Analysis

---

## 1. Problem Description

We analyze a **thick-walled cylinder** subjected to uniform internal pressure -- a classical benchmark in solid mechanics known as the **Lame problem**. This problem has a closed-form analytical solution, making it ideal for verifying finite element implementations.

### Geometry

| Parameter      | Symbol | Value    |
|----------------|--------|----------|
| Inner radius   | r_i    | 100 mm   |
| Outer radius   | r_o    | 200 mm   |
| Thickness      | t      | 1.0 mm   |

### Material (Linear Elastic)

| Parameter        | Symbol | Value     |
|------------------|--------|-----------|
| Young's modulus  | E      | 210 GPa   |
| Poisson's ratio  | nu     | 0.3       |

### Loading

| Parameter           | Symbol | Value     |
|---------------------|--------|-----------|
| Internal pressure   | p      | 100 MPa   |

### Symmetry

Because the geometry, material, and loading are all symmetric, we only model a **quarter of the cross-section** (first quadrant). This reduces the computational cost by 4x while capturing the full solution.

**Boundary conditions (symmetry):**

- **Bottom edge** (x-axis): u_y = 0 (roller in y)
- **Left edge** (y-axis): u_x = 0 (roller in x)
- **Inner arc**: internal pressure applied radially outward

---

## 2. Analytical Solution (Lame Equations)

The closed-form stresses in **polar coordinates** are:

```
sigma_rr(r) = A - B / r^2
sigma_tt(r) = A + B / r^2
sigma_rt    = 0
```

where:

```
A = p * r_i^2 / (r_o^2 - r_i^2)
B = p * r_i^2 * r_o^2 / (r_o^2 - r_i^2)
```

To compare with FEM results in Cartesian coordinates, transform using:

```
sigma_xx = sigma_rr * cos^2(theta) + sigma_tt * sin^2(theta)
sigma_yy = sigma_rr * sin^2(theta) + sigma_tt * cos^2(theta)
sigma_xy = (sigma_rr - sigma_tt) * sin(theta) * cos(theta)
```

This analytical solution serves as the reference for validating the finite element results.

---

## 3. Mesh Generation with Gmsh

### 3.1 Initialization

```python
import gmsh

gmsh.initialize()
gmsh.model.add("Plate2D")
```

Every Gmsh session starts by calling `gmsh.initialize()`. The model name `"Plate2D"` is important -- it must be referenced later when attaching post-processing data.

### 3.2 Geometry Definition (Bottom-Up Approach)

Gmsh uses a **bottom-up** approach: you create points, then lines/arcs from those points, then surfaces from those lines.

```python
lc = 10.0  # characteristic mesh size [mm]

# Center point (needed for circular arcs)
pc = gmsh.model.geo.addPoint(0, 0, 0, lc)

# 4 corner points of the quarter annulus
p1 = gmsh.model.geo.addPoint(inner_radius, 0, 0, lc)   # inner, x-axis
p2 = gmsh.model.geo.addPoint(outer_radius, 0, 0, lc)   # outer, x-axis
p3 = gmsh.model.geo.addPoint(0, outer_radius, 0, lc)    # outer, y-axis
p4 = gmsh.model.geo.addPoint(0, inner_radius, 0, lc)    # inner, y-axis
```

**Key concept -- `lc` at `addPoint`:** The fourth argument is a *local mesh size* attached to each geometric vertex. Gmsh interpolates between vertex sizes, so you can control local refinement (e.g., finer mesh near the inner radius where stresses are higher).

Then create the boundary curves:

```python
l1 = gmsh.model.geo.addLine(p1, p2)              # bottom radial line (x-axis)
l2 = gmsh.model.geo.addCircleArc(p2, pc, p3)     # outer arc (CCW)
l3 = gmsh.model.geo.addLine(p3, p4)              # left radial line (y-axis)
l4 = gmsh.model.geo.addCircleArc(p4, pc, p1)     # inner arc (CCW)
```

`addCircleArc(start, center, end)` creates a circular arc. The center point `pc` at the origin defines the circle.

Finally, create the surface:

```python
s1_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
s1      = gmsh.model.geo.addPlaneSurface([s1_loop])
```

### 3.3 Synchronization

```python
gmsh.model.geo.synchronize()
```

This call is **mandatory** before defining physical groups or meshing. It transfers the geometry from the CAD kernel to the Gmsh model. Call it **once** after all geometry is defined.

### 3.4 Physical Groups

Physical groups label geometric entities so you can identify boundary conditions and domains after meshing:

```python
pg_symY     = gmsh.model.addPhysicalGroup(1, [l1], name="Sym_Y")      # bottom: uy = 0
pg_symX     = gmsh.model.addPhysicalGroup(1, [l3], name="Sym_X")      # left:   ux = 0
pg_pressure = gmsh.model.addPhysicalGroup(1, [l4], name="Pressure")   # inner arc: pressure
pg_plate    = gmsh.model.addPhysicalGroup(2, [s1], name="Plate")      # 2D domain
```

- First argument: **dimension** (1 = curves, 2 = surfaces)
- Second argument: list of entity tags belonging to the group
- The returned tag is used later with `getNodesForPhysicalGroup()` to retrieve boundary nodes

### 3.5 Mesh Settings and Generation

```python
gmsh.option.setNumber("Mesh.Algorithm", 6)         # Frontal-Delaunay
gmsh.option.setNumber("Mesh.ElementOrder", 1)       # linear elements
gmsh.option.setNumber("Mesh.MeshSizeMin", lc * 0.5)
gmsh.option.setNumber("Mesh.MeshSizeMax", lc)

gmsh.model.mesh.generate(2)  # generate 2D mesh
```

**`MeshSizeMin` / `MeshSizeMax`** are global clamps applied *after* local sizes are computed. They act as hard bounds: even if `lc=10` at all points, `MeshSizeMin=5` prevents elements smaller than 5 mm. This is independent from the `lc` value at each point.

### 3.6 Extracting Mesh Data

**Nodes:**

```python
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
node_coords = node_coords.reshape(-1, 3)   # (N, 3) array
tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
```

Gmsh node tags are **not necessarily contiguous** (they can have gaps). The `tag_to_idx` dictionary maps Gmsh tags to array indices.

**Elements (triangles):**

```python
elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)

tri_conn = []
for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
    name, dim, order, nnodes, _, _ = gmsh.model.mesh.getElementProperties(etype)
    tri_conn.append(enodes.reshape(-1, nnodes).astype(int))

connectivity = np.vstack(tri_conn)
```

**Filtering orphan nodes:**

```python
used_tags = set(connectivity.flatten())
```

The geometric center point `pc` at `(0,0)` gets a mesh node from Gmsh but no triangle connects to it. Including it in OpenSees would create a zero-stiffness DOF and a **singular stiffness matrix**. We must filter these orphan nodes.

**Boundary nodes:**

```python
bottom_node_tags = gmsh.model.mesh.getNodesForPhysicalGroup(1, pg_symY)[0]
left_node_tags   = gmsh.model.mesh.getNodesForPhysicalGroup(1, pg_symX)[0]
inner_node_tags  = gmsh.model.mesh.getNodesForPhysicalGroup(1, pg_pressure)[0]
```

**Inner arc edges** (needed for pressure load calculation):

```python
ie_types, ie_tags, ie_nodes = gmsh.model.mesh.getElements(dim=1, tag=l4)
```

Passing `tag=l4` retrieves only 1D elements on the inner arc curve.

---

## 4. Finite Element Model in OpenSeesPy

### 4.1 Model Initialization

```python
import openseespy.opensees as ops

ops.wipe()
ops.model("basic", "-ndm", 2, "-ndf", 2)
```

- `-ndm 2`: 2D model
- `-ndf 2`: 2 degrees of freedom per node (u_x, u_y) -- appropriate for plane strain

### 4.2 Node Creation (with Tag Remapping)

```python
gmsh_to_ops = {}
new_id = 0
for gtag, coords in zip(node_tags.astype(int), node_coords):
    if int(gtag) not in used_tags:
        continue  # skip orphan nodes
    new_id += 1
    gmsh_to_ops[int(gtag)] = new_id
    ops.node(new_id, float(coords[0]), float(coords[1]))
```

The `gmsh_to_ops` dictionary maps Gmsh node tags to sequential OpenSees node IDs. This remapping is essential because:
1. Gmsh tags may have gaps
2. Orphan nodes are skipped

### 4.3 Material

```python
ops.nDMaterial("ElasticIsotropic", 1, E, nu)
```

`ElasticIsotropic` is a 2D/3D isotropic elastic material defined by Young's modulus and Poisson's ratio.

### 4.4 Elements

```python
for eid, row in enumerate(connectivity, start=1):
    n1 = gmsh_to_ops[row[0]]
    n2 = gmsh_to_ops[row[1]]
    n3 = gmsh_to_ops[row[2]]
    ops.element("tri31", eid, n1, n2, n3, thk, "PlaneStrain", 1)
```

**`tri31`** is a 3-node **constant-strain triangle** (CST). Each element has:
- 3 nodes, 2 DOFs each = 6 DOFs per element
- Constant stress/strain throughout the element (single integration point)
- `"PlaneStrain"` assumes out-of-plane strain = 0

### 4.5 Boundary Conditions

```python
# Bottom edge (x-axis symmetry): fix u_y
for gtag in bottom_node_tags.astype(int):
    ops.fix(gmsh_to_ops[int(gtag)], 0, 1)   # free x, fixed y

# Left edge (y-axis symmetry): fix u_x
for gtag in left_node_tags.astype(int):
    ops.fix(gmsh_to_ops[int(gtag)], 1, 0)   # fixed x, free y
```

`ops.fix(nodeTag, fixX, fixY)` where 1 = fixed, 0 = free.

### 4.6 Pressure Loading (Consistent Nodal Forces)

Internal pressure on a curved boundary cannot be applied directly as a uniform load. Instead, we compute **equivalent nodal forces** by integrating the traction vector over each boundary edge.

For a 2-node linear edge of length L with traction **t** = p * **n_hat** (outward radial):

```
F_i = (L/6) * (2*t_i + t_j)
F_j = (L/6) * (t_i + 2*t_j)
```

This is the **consistent load vector** derived from the shape functions. Using `L/2 * t` at each node (lumped) would be less accurate.

```python
ops.timeSeries("Linear", 1)
ops.pattern("Plain", 1, 1)

nodal_forces = {}
for edge in inner_edges:
    n1g, n2g = int(edge[0]), int(edge[1])
    # ... compute traction at each node ...
    # ... integrate with linear shape functions ...
    # ... accumulate forces (shared nodes get contributions from both edges) ...

for nid, (fx, fy) in nodal_forces.items():
    ops.load(nid, fx, fy)
```

### 4.7 Analysis

```python
ops.constraints("Transformation")
ops.numberer("RCM")
ops.system("BandGeneral")
ops.test("NormDispIncr", 1.0e-8, 10)
ops.algorithm("Newton")
ops.integrator("LoadControl", 1.0)
ops.analysis("Static")
ok = ops.analyze(1)
```

| Command | Purpose |
|---------|---------|
| `constraints("Transformation")` | Handles multi-point constraints via coordinate transformation |
| `numberer("RCM")` | Reverse Cuthill-McKee reordering to reduce bandwidth |
| `system("BandGeneral")` | Banded solver (efficient for FEM with local connectivity) |
| `test("NormDispIncr", 1e-8, 10)` | Convergence test: displacement norm < 1e-8, max 10 iterations |
| `algorithm("Newton")` | Newton-Raphson solver (converges in 1 iteration for linear problems) |
| `integrator("LoadControl", 1.0)` | Apply full load in one step (lambda = 1.0) |
| `analysis("Static")` | Static analysis type |

---

## 5. Post-Processing

### 5.1 Extracting Results from OpenSees

**Displacements:**

```python
disp = np.zeros((len(node_tags), 2))
for gtag, ops_id in gmsh_to_ops.items():
    idx = tag_to_idx[gtag]
    disp[idx, 0] = ops.nodeDisp(ops_id, 1)  # u_x
    disp[idx, 1] = ops.nodeDisp(ops_id, 2)  # u_y
```

**Radial displacement** (derived from Cartesian):

```
u_r = (x * u_x + y * u_y) / r
```

**Element stresses:**

```python
for eid in range(1, nElem + 1):
    stress = ops.eleResponse(eid, "stresses")  # returns [sigma_xx, sigma_yy, sigma_xy]
```

For `tri31` (CST), stress is constant per element.

### 5.2 Nodal Averaging for Smooth Contours

Since CST elements have piecewise-constant stress, neighboring elements generally give different stress values at shared nodes. **Nodal averaging** smooths the field:

```python
sig_xx_nodal = np.zeros(len(node_tags))
node_count   = np.zeros(len(node_tags))

for e in range(nElem):
    for local_n in range(3):
        nidx = conn_idx[e, local_n]
        sig_xx_nodal[nidx] += sig_xx_elem[e]
        node_count[nidx]   += 1.0

sig_xx_nodal /= node_count
```

Each node accumulates stress from all surrounding elements, then divides by the count.

### 5.3 Visualization with Matplotlib

The code produces four plots:

1. **Deformed vs. undeformed mesh** -- using `PolyCollection` with a magnification factor
2. **Radial displacement contour** -- `tripcolor` with Gouraud shading
3. **sigma_xx contour (nodal-averaged)** -- smooth, Gouraud-shaded
4. **sigma_xx contour (element-wise)** -- piecewise constant, shows the true CST behavior

### 5.4 Passing Results Back to Gmsh

OpenSees results can be injected into Gmsh as **post-processing views** for visualization in the Gmsh GUI:

```python
v1 = gmsh.view.add("sigma_xx [MPa]")
gmsh.view.addModelData(
    v1,                # view tag
    0,                 # time step (0 for static)
    "Plate2D",         # model name (must match gmsh.model.add)
    "ElementData",     # one value per element
    gmsh_elem_tags,    # Gmsh element tags (not OpenSees IDs!)
    [[float(s)] for s in sig_xx_elem],  # data
    0.0,               # time
    1                  # numComponents (1 = scalar)
)
```

**Data types:**
- `"ElementData"` -- one value per element (constant per element, e.g. CST stress)
- `"NodeData"` -- one value per node (interpolated, e.g. displacement)

**Number of components:**
- 1 = scalar field (stress component, displacement magnitude)
- 3 = vector field (displacement vector)
- 6 = symmetric tensor, 9 = full tensor

Launch the GUI with:

```python
gmsh.fltk.run()
gmsh.finalize()
```

---

## 6. Mesh Convergence Study

### 6.1 Motivation

The CST element (`tri31`) is the simplest 2D element. Its accuracy depends heavily on mesh density. A **convergence study** verifies that the numerical solution approaches the analytical solution as the mesh is refined.

### 6.2 L2 Error Norm

The relative L2 error norm measures the global accuracy of the stress field:

```
         || sigma^FEM - sigma^exact ||_L2
  e  =  --------------------------------
              || sigma^exact ||_L2

         sqrt( sum_e  (sigma_xx^e - sigma_xx^exact(x_c))^2 * A_e )
     =  ------------------------------------------------------------
              sqrt( sum_e  (sigma_xx^exact(x_c))^2 * A_e )
```

For CST elements, stress is constant per element, so the integral reduces to a weighted sum over element areas. The analytical stress is evaluated at each element's centroid.

### 6.3 Expected Convergence Rate

For linear triangles (CST), the theoretical convergence rate for stresses is **O(h^1)** -- halving the mesh size halves the error. The convergence rate is computed between successive refinements:

```
rate = log(e_{i-1} / e_i) / log(h_{i-1} / h_i)
```

### 6.4 Implementation

The convergence loop runs the full pipeline (mesh + solve + error computation) for multiple mesh sizes:

```python
lc_values = [40.0, 20.0, 10.0, 5.0, 2.5]
```

Each iteration:
1. Generates a fresh Gmsh mesh at the given `lc`
2. Builds and solves the OpenSees model
3. Computes the L2 error norm against the Lame solution
4. Records h, number of elements/nodes, and error

The results are plotted on a **log-log** scale. The slope of the error curve should match the O(h^1) reference line, confirming the expected first-order convergence of the CST element.

---

## 7. Summary of the Gmsh-OpenSees Workflow

```
1. gmsh.initialize()              -- start Gmsh
2. Define geometry (points, lines, arcs, surfaces)
3. gmsh.model.geo.synchronize()   -- transfer to model
4. Define physical groups          -- label boundaries and domains
5. Set mesh options and generate   -- gmsh.model.mesh.generate(2)
6. Extract nodes, elements, boundary info
7. ops.wipe() / ops.model()       -- start OpenSees
8. Create nodes (with tag remapping, skip orphans)
9. Define material and elements
10. Apply BCs and loads
11. Solve
12. Extract displacements and stresses
13. (Optional) Pass results back to Gmsh views
14. gmsh.finalize()                -- clean up
```

This workflow is general and can be adapted to any 2D problem: change the geometry in steps 2-4, swap the element type in step 9, and adjust the loading in step 10.
