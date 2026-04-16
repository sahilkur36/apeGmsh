"""
Mesh Convergence Study — L2 Error Norm of σ_xx
================================================
Runs the thick-walled cylinder model at multiple mesh sizes,
computes the relative L2 error norm against the Lamé analytical
solution, and plots error vs h on a log-log scale.

L2 error norm (relative):

         ‖σ^FEM - σ^exact‖_L2       √( Σ_e  (σ_xx^e - σ_xx^exact(x_c))² · A_e )
  e  =  ────────────────────── =  ──────────────────────────────────────────────────
            ‖σ^exact‖_L2             √( Σ_e  (σ_xx^exact(x_c))² · A_e )

For CST (tri31): stress is constant per element, so the integral
reduces to a weighted sum over element areas.

Expected convergence: O(h^1) for stresses with linear triangles.
"""

import gmsh
import numpy as np
import openseespy.opensees as ops
from matplotlib import pyplot as plt

# ============================================================
# Parameters
# ============================================================
inner_radius = 100.0    # mm
outer_radius = 200.0    # mm
E   = 210.0e3           # MPa
nu  = 0.3
p   = 100.0             # MPa
thk = 1.0               # mm

# Mesh sizes to test (coarse → fine)
lc_values = [40.0, 20.0, 10.0, 5.0, 2.5]


# ============================================================
# Lamé analytical solution (plane strain)
# ============================================================
def lame_stress_cartesian(x, y, ri, ro, p):
    """
    Returns (σ_xx, σ_yy, σ_xy) at point (x, y) from the
    Lamé closed-form solution for internal pressure.

    In polar coordinates:
        σ_rr(r) = A - B/r²
        σ_θθ(r) = A + B/r²
        σ_rθ    = 0

    where A = p·ri² / (ro² - ri²)
          B = p·ri²·ro² / (ro² - ri²)

    Transform to Cartesian:
        σ_xx = σ_rr·cos²θ + σ_θθ·sin²θ
        σ_yy = σ_rr·sin²θ + σ_θθ·cos²θ
        σ_xy = (σ_rr - σ_θθ)·sinθ·cosθ
    """
    r2 = x**2 + y**2
    r  = np.sqrt(r2)

    A = p * ri**2 / (ro**2 - ri**2)
    B = p * ri**2 * ro**2 / (ro**2 - ri**2)

    sig_rr = A - B / r2
    sig_tt = A + B / r2

    cos_t = x / r
    sin_t = y / r

    sig_xx = sig_rr * cos_t**2 + sig_tt * sin_t**2
    sig_yy = sig_rr * sin_t**2 + sig_tt * cos_t**2
    sig_xy = (sig_rr - sig_tt) * sin_t * cos_t

    return sig_xx, sig_yy, sig_xy


def triangle_area(x1, y1, x2, y2, x3, y3):
    """Area of a triangle from vertex coordinates."""
    return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


# ============================================================
# Convergence loop
# ============================================================
errors_L2  = []
h_values   = []
num_elems  = []
num_nodes  = []

for lc in lc_values:
    print(f"\n{'='*50}")
    print(f"  Mesh size lc = {lc} mm")
    print(f"{'='*50}")

    # ----------------------------------------------------------
    # GMSH: mesh generation
    # ----------------------------------------------------------
    gmsh.initialize()
    gmsh.model.add("Plate2D")

    pc = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p1 = gmsh.model.geo.addPoint(inner_radius, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(outer_radius, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(0, outer_radius, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, inner_radius, 0, lc)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addCircleArc(p2, pc, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addCircleArc(p4, pc, p1)

    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s1   = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()

    pg_symY     = gmsh.model.addPhysicalGroup(1, [l1], name="Sym_Y")
    pg_symX     = gmsh.model.addPhysicalGroup(1, [l3], name="Sym_X")
    pg_pressure = gmsh.model.addPhysicalGroup(1, [l4], name="Pressure")
    pg_plate    = gmsh.model.addPhysicalGroup(2, [s1], name="Plate")

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc * 0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)

    gmsh.model.mesh.generate(2)

    # Extract nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = node_coords.reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    # Extract triangles
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    tri_conn = []
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        _, _, _, nnodes, _, _ = gmsh.model.mesh.getElementProperties(etype)
        tri_conn.append(enodes.reshape(-1, nnodes).astype(int))
    connectivity = np.vstack(tri_conn)

    # Filter used nodes
    used_tags = set(connectivity.flatten())

    # Boundary nodes
    bottom_nodes = gmsh.model.mesh.getNodesForPhysicalGroup(1, pg_symY)[0]
    left_nodes   = gmsh.model.mesh.getNodesForPhysicalGroup(1, pg_symX)[0]
    inner_nodes  = gmsh.model.mesh.getNodesForPhysicalGroup(1, pg_pressure)[0]

    # Inner arc edges
    ie_types, ie_tags, ie_nodes = gmsh.model.mesh.getElements(dim=1, tag=l4)
    inner_edges = []
    for etype, etags, enodes in zip(ie_types, ie_tags, ie_nodes):
        _, _, _, nnodes, _, _ = gmsh.model.mesh.getElementProperties(etype)
        inner_edges = enodes.reshape(-1, nnodes).astype(int)

    gmsh.finalize()

    # ----------------------------------------------------------
    # OpenSees: build and solve
    # ----------------------------------------------------------
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)

    # Nodes (skip orphans)
    gmsh_to_ops = {}
    new_id = 0
    for gtag, coords in zip(node_tags.astype(int), node_coords):
        if int(gtag) not in used_tags:
            continue
        new_id += 1
        gmsh_to_ops[int(gtag)] = new_id
        ops.node(new_id, float(coords[0]), float(coords[1]))

    # Material
    ops.nDMaterial("ElasticIsotropic", 1, E, nu)

    # Elements
    for eid, row in enumerate(connectivity, start=1):
        n1 = gmsh_to_ops[row[0]]
        n2 = gmsh_to_ops[row[1]]
        n3 = gmsh_to_ops[row[2]]
        ops.element("tri31", eid, n1, n2, n3, thk, "PlaneStrain", 1)

    # BCs
    for gtag in bottom_nodes.astype(int):
        ops.fix(gmsh_to_ops[int(gtag)], 0, 1)
    for gtag in left_nodes.astype(int):
        ops.fix(gmsh_to_ops[int(gtag)], 1, 0)

    # Pressure (consistent nodal forces)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)

    nodal_forces = {}
    for edge in inner_edges:
        n1g, n2g = int(edge[0]), int(edge[1])
        i1, i2 = tag_to_idx[n1g], tag_to_idx[n2g]
        x1, y1 = node_coords[i1, 0], node_coords[i1, 1]
        x2, y2 = node_coords[i2, 0], node_coords[i2, 1]

        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        r1 = np.sqrt(x1**2 + y1**2)
        r2 = np.sqrt(x2**2 + y2**2)

        tx1, ty1 = p * x1 / r1, p * y1 / r1
        tx2, ty2 = p * x2 / r2, p * y2 / r2

        Fx1 = (L / 6.0) * (2.0 * tx1 + tx2)
        Fy1 = (L / 6.0) * (2.0 * ty1 + ty2)
        Fx2 = (L / 6.0) * (tx1 + 2.0 * tx2)
        Fy2 = (L / 6.0) * (ty1 + 2.0 * ty2)

        o1, o2 = gmsh_to_ops[n1g], gmsh_to_ops[n2g]
        nodal_forces.setdefault(o1, [0.0, 0.0])
        nodal_forces.setdefault(o2, [0.0, 0.0])
        nodal_forces[o1][0] += Fx1;  nodal_forces[o1][1] += Fy1
        nodal_forces[o2][0] += Fx2;  nodal_forces[o2][1] += Fy2

    for nid, (fx, fy) in nodal_forces.items():
        ops.load(nid, fx, fy)

    # Solve
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-8, 10)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    ok = ops.analyze(1)

    if ok != 0:
        print(f"  *** FAILED for lc = {lc} ***")
        errors_L2.append(np.nan)
        h_values.append(lc)
        num_elems.append(connectivity.shape[0])
        num_nodes.append(len(gmsh_to_ops))
        ops.wipe()
        continue

    # ----------------------------------------------------------
    # L2 error norm of σ_xx
    # ----------------------------------------------------------
    nElem = connectivity.shape[0]
    numerator   = 0.0   # Σ (σ^FEM - σ^exact)² · A_e
    denominator = 0.0   # Σ (σ^exact)² · A_e

    for eid in range(1, nElem + 1):
        stress_fem = ops.eleResponse(eid, "stresses")   # [sxx, syy, sxy]
        sxx_fem = stress_fem[0]

        # Element centroid (evaluation point for analytical solution)
        row = connectivity[eid - 1]
        idx = [tag_to_idx[int(n)] for n in row]
        xc = np.mean([node_coords[i, 0] for i in idx])
        yc = np.mean([node_coords[i, 1] for i in idx])

        # Element area
        x1, y1 = node_coords[idx[0], 0], node_coords[idx[0], 1]
        x2, y2 = node_coords[idx[1], 0], node_coords[idx[1], 1]
        x3, y3 = node_coords[idx[2], 0], node_coords[idx[2], 1]
        Ae = triangle_area(x1, y1, x2, y2, x3, y3)

        # Analytical σ_xx at centroid
        sxx_exact, _, _ = lame_stress_cartesian(xc, yc, inner_radius, outer_radius, p)

        numerator   += (sxx_fem - sxx_exact)**2 * Ae
        denominator += sxx_exact**2 * Ae

    e_L2 = np.sqrt(numerator / denominator)

    errors_L2.append(e_L2)
    h_values.append(lc)
    num_elems.append(nElem)
    num_nodes.append(len(gmsh_to_ops))

    print(f"  Nodes: {len(gmsh_to_ops)},  Elements: {nElem},  L2 error: {e_L2:.6e}")

    ops.wipe()

# ============================================================
# Results table
# ============================================================
print(f"\n{'='*65}")
print(f"  {'h [mm]':>8}  {'Nodes':>7}  {'Elements':>9}  {'L2 error':>12}  {'Rate':>6}")
print(f"{'='*65}")
for i in range(len(h_values)):
    if i == 0 or np.isnan(errors_L2[i]) or np.isnan(errors_L2[i-1]):
        rate_str = "  —"
    else:
        rate = np.log(errors_L2[i-1] / errors_L2[i]) / np.log(h_values[i-1] / h_values[i])
        rate_str = f"{rate:5.2f}"
    print(f"  {h_values[i]:8.1f}  {num_nodes[i]:7d}  {num_elems[i]:9d}  {errors_L2[i]:12.6e}  {rate_str}")
print(f"{'='*65}")

# ============================================================
# Plot: L2 error vs mesh size (log-log)
# ============================================================
h_arr = np.array(h_values)
e_arr = np.array(errors_L2)

fig, ax = plt.subplots(figsize=(8, 6))

# FEM data
ax.loglog(h_arr, e_arr, "bo-", lw=2, ms=8, label=r"$\| \sigma_{xx}^{FEM} - \sigma_{xx}^{exact} \|_{L_2}$")

# Reference slope: O(h^1)
# Anchor to the finest-mesh data point
h_ref = h_arr
e_ref = e_arr[-1] * (h_ref / h_ref[-1])**1
ax.loglog(h_ref, e_ref, "k--", lw=1.2, label=r"$O(h^1)$ reference")

# Reference slope: O(h^2) for comparison
e_ref2 = e_arr[-1] * (h_ref / h_ref[-1])**2
ax.loglog(h_ref, e_ref2, "k:", lw=1.2, label=r"$O(h^2)$ reference")

ax.set_xlabel("Characteristic mesh size $h$ [mm]", fontsize=12)
ax.set_ylabel(r"Relative $L_2$ error norm", fontsize=12)
ax.set_title(r"Mesh Convergence — $\sigma_{xx}$ (tri31, plane strain)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, which="both", ls=":", alpha=0.5)
plt.tight_layout()
plt.savefig("convergence_L2.png", dpi=150)
plt.show()

print("\nDone.")
