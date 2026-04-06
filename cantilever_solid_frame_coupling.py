"""
Cantilever Beam: Solid-Frame Coupling Example
==============================================

A cantilever beam of length L under a point load P at the free end.
The left half is modeled with shell (solid) elements and the right
half with a beam-column (frame) element.

Because OpenSees 2D quad elements do not support ndf=3, the model is
built in 3D (ndm=3, ndf=6) and constrained to 2D behavior by fixing
the out-of-plane DOFs (uz, rx, ry) for all nodes.

Coupling strategy at the interface (x = L/2):
  1. Master node at the frame end (cross-section centroid)
  2. Duplicated nodes co-located with the solid interface nodes
  3. rigidLink('beam') from master to each duplicated node
     — enforces plane-sections-remain-plane (rigid body kinematics)
  4. equalDOF between each duplicated node and its matching solid
     node for translational DOFs 1 and 2 (displacement compatibility)

     Fixed                                                Point load P
     |||||                                                      |
     |||||  SOLID (ShellMITC4)    |   FRAME (beam-column 3D)    v
     |||||========================|=============================*
     |||||  nx_s x ny_s elements  |  master         tip
     |||||                        |
     |||||________________________|

Author: Claude / OpenSees Expert Skill
"""

import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt

# ================================================================
#  1. PARAMETERS
# ================================================================
L      = 10.0        # Total beam length [m]
h      = 1.0         # Beam depth [m]
b      = 0.5         # Beam width (out-of-plane thickness) [m]
E      = 30.0e6      # Young's modulus [kPa]
nu     = 0.2         # Poisson's ratio
P      = -100.0      # Tip point load [kN] (negative = downward)

L_solid = L / 2.0
L_frame = L / 2.0

# Solid mesh density
nx_s = 20            # elements along x
ny_s = 8             # elements through depth

dx = L_solid / nx_s
dy = h / ny_s

# Frame cross-section properties
A  = b * h
Iz = b * h**3 / 12.0
Iy = h * b**3 / 12.0
G  = E / (2.0 * (1.0 + nu))
J  = 0.5 * (Iy + Iz)     # approximate torsional constant

# ================================================================
#  2. MODEL INITIALIZATION (3D)
# ================================================================
ops.wipe()
ops.model('basic', '-ndm', 3, '-ndf', 6)

# ================================================================
#  3. SOLID REGION NODES
# ================================================================
solid_nodes = {}   # (i, j) -> node_tag
nid = 0

for j in range(ny_s + 1):
    for i in range(nx_s + 1):
        nid += 1
        x = i * dx
        y = -h / 2.0 + j * dy
        ops.node(nid, x, y, 0.0)
        solid_nodes[(i, j)] = nid

n_solid_nodes = nid

# Identify interface solid nodes at x = L_solid (i = nx_s)
interface_solid_tags = [solid_nodes[(nx_s, j)] for j in range(ny_s + 1)]

# Constrain to 2D: fix uz(3), rx(4), ry(5) for non-interface, non-fixed nodes
# Interface nodes: DOFs handled by equalDOF coupling — fix out-of-plane here too
for (i, j), tag in solid_nodes.items():
    if i == 0:
        continue   # left edge — will get full fixity below
    # Fix out-of-plane DOFs; leave ux(1), uy(2), rz(6) free
    ops.fix(tag, 0, 0, 1, 1, 1, 0)

# Fixed support at x=0 (full fixity)
for j in range(ny_s + 1):
    ops.fix(solid_nodes[(0, j)], 1, 1, 1, 1, 1, 1)

# ================================================================
#  4. FRAME NODES
# ================================================================
master_tag = n_solid_nodes + 1
tip_tag    = n_solid_nodes + 2

ops.node(master_tag, L_solid, 0.0, 0.0)
ops.node(tip_tag,    L,       0.0, 0.0)

# Constrain to 2D
ops.fix(master_tag, 0, 0, 1, 1, 1, 0)
ops.fix(tip_tag,    0, 0, 1, 1, 1, 0)

# ================================================================
#  5. DUPLICATED INTERFACE NODES
# ================================================================
# Co-located with solid interface nodes.
# Connected to master via rigidLink; tied to solid nodes via equalDOF.
dup_tags = []
for j in range(ny_s + 1):
    dn = tip_tag + 1 + j
    y  = -h / 2.0 + j * dy
    ops.node(dn, L_solid, y, 0.0)
    # NO fixity on dup nodes — rigidLink constrains all 6 DOFs
    dup_tags.append(dn)

total_nodes = dup_tags[-1]
print(f"Nodes: {total_nodes} total")
print(f"  Solid: 1..{n_solid_nodes}  Master: {master_tag}  Tip: {tip_tag}")
print(f"  Duplicated: {dup_tags[0]}..{dup_tags[-1]}")

# ================================================================
#  6. MATERIAL & SECTION
# ================================================================
# Shell section: membrane (in-plane) + plate (bending)
# thickness = b (beam width = out-of-plane dimension of the shell)
ops.section('ElasticMembranePlateSection', 1, E, nu, b, 0.0)

# ================================================================
#  7. SOLID ELEMENTS (ShellMITC4)
# ================================================================
eid = 0
quad_elements = []

for j in range(ny_s):
    for i in range(nx_s):
        eid += 1
        n1 = solid_nodes[(i,   j)]
        n2 = solid_nodes[(i+1, j)]
        n3 = solid_nodes[(i+1, j+1)]
        n4 = solid_nodes[(i,   j+1)]
        ops.element('ShellMITC4', eid, n1, n2, n3, n4, 1)
        quad_elements.append((eid, n1, n2, n3, n4))

n_solid_ele = eid

# ================================================================
#  8. FRAME ELEMENT (3D elastic beam-column)
# ================================================================
# vecxz = (0,0,1): local xz plane contains the Z-axis
# → local z ~ Z direction, local y ~ Y direction
# → Iz controls bending in XY plane (our plane of interest)
ops.geomTransf('Linear', 1, 0.0, 0.0, 1.0)
eid += 1
frame_eid = eid
ops.element('elasticBeamColumn', eid, master_tag, tip_tag,
            A, E, G, J, Iy, Iz, 1)

print(f"Elements: {eid} ({n_solid_ele} ShellMITC4 + 1 frame)")

# ================================================================
#  9. COUPLING CONSTRAINTS
# ================================================================
# Step A: rigidLink('beam') from master to each duplicated node.
#   Constrains all 6 DOFs of the dup node based on rigid body
#   kinematics from the master → plane sections remain plane.
for dn in dup_tags:
    ops.rigidLink('beam', master_tag, dn)

# Step B: equalDOF — tie duplicated nodes to solid interface nodes.
#   Only translational DOFs 1 (ux) and 2 (uy).
#   dup node = retained, solid node = constrained.
for dn, sn in zip(dup_tags, interface_solid_tags):
    ops.equalDOF(dn, sn, 1, 2)

print(f"Coupling: {len(dup_tags)} rigid links + {len(dup_tags)} equalDOF")

# ================================================================
#  10. LOADING
# ================================================================
ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)
ops.load(tip_tag, 0.0, P, 0.0, 0.0, 0.0, 0.0)

# ================================================================
#  11. ANALYSIS
# ================================================================
# Penalty handler: required for nested MP constraints
# (dup nodes are slaves of rigidLink AND retained in equalDOF)
ops.constraints('Penalty', 1.0e14, 1.0e14)
ops.numberer('RCM')
ops.system('UmfPack')
ops.test('NormDispIncr', 1.0e-6, 50)
ops.algorithm('Newton')
ops.integrator('LoadControl', 0.1)
ops.analysis('Static')

print("\nRunning analysis (10 load steps)...")
ok = ops.analyze(10)
if ok != 0:
    print("*** Analysis FAILED ***")
else:
    print("Analysis completed successfully!\n")

# ================================================================
#  12. RESULTS
# ================================================================
# DOF mapping in 3D: ux=1, uy=2, uz=3, rx=4, ry=5, rz=6
tip_d    = ops.nodeDisp(tip_tag)
master_d = ops.nodeDisp(master_tag)

print("=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"\nTip ({tip_tag}, x={L}):    ux={tip_d[0]:+.6e}  uy={tip_d[1]:+.6e}  rz={tip_d[5]:+.6e}")
print(f"Master ({master_tag}, x={L_solid}): ux={master_d[0]:+.6e}  uy={master_d[1]:+.6e}  rz={master_d[5]:+.6e}")

# Analytical reference
delta_EB    = P * L**3 / (3.0 * E * Iz)
theta_EB    = P * L**2 / (2.0 * E * Iz)
delta_shear = P * L / (5.0/6.0 * A * G)
delta_Timo  = delta_EB + delta_shear

print(f"\n--- Analytical Comparison ---")
print(f"Euler-Bernoulli tip deflection: {delta_EB:+.6e} m")
print(f"Timoshenko tip deflection:      {delta_Timo:+.6e} m")
print(f"Numerical tip deflection:       {tip_d[1]:+.6e} m")
if abs(delta_EB) > 0:
    print(f"Ratio (num / EB):               {tip_d[1]/delta_EB:.4f}")
    print(f"Ratio (num / Timoshenko):       {tip_d[1]/delta_Timo:.4f}")

# Interface displacements
print(f"\n--- Interface displacements (x = {L_solid} m) ---")
print(f"{'Node':>6}  {'y':>7}  {'ux':>14}  {'uy':>14}")
print("-" * 50)
for j in range(ny_s + 1):
    sn = interface_solid_tags[j]
    d  = ops.nodeDisp(sn)
    y  = -h / 2.0 + j * dy
    print(f"{sn:6d}  {y:7.3f}  {d[0]:+14.6e}  {d[1]:+14.6e}")

# ================================================================
#  13. VISUALIZATION
# ================================================================

def get_data():
    """Collect coordinates and displacements for all relevant nodes."""
    coords, disps = {}, {}
    for (i, j), tag in solid_nodes.items():
        coords[tag] = (i * dx, -h/2.0 + j * dy)
        d = ops.nodeDisp(tag)
        disps[tag] = (d[0], d[1])
    for tag in [master_tag, tip_tag]:
        c = ops.nodeCoord(tag)
        d = ops.nodeDisp(tag)
        coords[tag] = (c[0], c[1])
        disps[tag]  = (d[0], d[1])
    return coords, disps


coords, disps = get_data()
max_d = max(abs(d[1]) for d in disps.values() if abs(d[1]) > 0)
scale = 0.5 * L / max_d if max_d > 0 else 1.0

fig, axes = plt.subplots(2, 1, figsize=(16, 11))
ax1, ax2 = axes

# ---- Plot 1: Deformed mesh ----
# Undeformed solid (light gray)
for (_, n1, n2, n3, n4) in quad_elements:
    xs = [coords[n][0] for n in [n1, n2, n3, n4, n1]]
    ys = [coords[n][1] for n in [n1, n2, n3, n4, n1]]
    ax1.plot(xs, ys, color='#d4d4d4', lw=0.3)

# Undeformed frame (light gray dashed)
ax1.plot([coords[master_tag][0], coords[tip_tag][0]],
         [coords[master_tag][1], coords[tip_tag][1]],
         '--', color='#d4d4d4', lw=2)

# Deformed solid (blue filled)
for (_, n1, n2, n3, n4) in quad_elements:
    xs_d = [coords[n][0] + scale*disps[n][0] for n in [n1, n2, n3, n4, n1]]
    ys_d = [coords[n][1] + scale*disps[n][1] for n in [n1, n2, n3, n4, n1]]
    ax1.fill(xs_d[:-1], ys_d[:-1], alpha=0.10, color='#2563EB')
    ax1.plot(xs_d, ys_d, color='#2563EB', lw=0.5)

# Deformed frame (red thick line)
mx = coords[master_tag][0] + scale*disps[master_tag][0]
my = coords[master_tag][1] + scale*disps[master_tag][1]
tx = coords[tip_tag][0] + scale*disps[tip_tag][0]
ty = coords[tip_tag][1] + scale*disps[tip_tag][1]
ax1.plot([mx, tx], [my, ty], '-', color='#DC2626', lw=3.5, label='Frame (deformed)')

# Rigid links (orange dashed)
for sn in interface_solid_tags:
    sx = coords[sn][0] + scale*disps[sn][0]
    sy = coords[sn][1] + scale*disps[sn][1]
    ax1.plot([mx, sx], [my, sy], '--', color='#F59E0B', lw=1.0)
ax1.plot([], [], '--', color='#F59E0B', lw=1.0, label='Rigid links (coupling)')

# Special markers
ax1.plot(mx, my, 'o', color='#DC2626', ms=8, zorder=5, label='Master node')
ax1.plot(tx, ty, 'v', color='#7C3AED', ms=10, zorder=5, label=f'Tip (P = {P:.0f} kN)')
for sn in interface_solid_tags:
    ax1.plot(coords[sn][0] + scale*disps[sn][0],
             coords[sn][1] + scale*disps[sn][1],
             's', color='#F59E0B', ms=3, zorder=5)

# Fixed support symbols
for j in range(ny_s + 1):
    n = solid_nodes[(0, j)]
    ax1.plot(coords[n][0], coords[n][1], '>', color='#059669', ms=5, zorder=5)

# Annotations
ax1.annotate('SOLID\n(ShellMITC4)', xy=(L_solid/2, -h/2 - 0.15),
             ha='center', fontsize=9, color='#2563EB', weight='bold')
ax1.annotate('FRAME\n(elasticBeamColumn)', xy=(L_solid + L_frame/2, -h/2 - 0.15),
             ha='center', fontsize=9, color='#DC2626', weight='bold')

ax1.set_title(f'Deformed shape (scale factor = {scale:.0f}x)', fontsize=12)
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.25)
ax1.legend(loc='lower left', fontsize=8, framealpha=0.9)

# ---- Plot 2: Deflection curve ----
# Solid centroid row (j = ny_s // 2)
j_mid = ny_s // 2
x_solid  = [coords[solid_nodes[(i, j_mid)]][0] for i in range(nx_s + 1)]
uy_solid = [disps[solid_nodes[(i, j_mid)]][1]   for i in range(nx_s + 1)]

# Frame nodes
x_frame  = [coords[master_tag][0], coords[tip_tag][0]]
uy_frame = [disps[master_tag][1],  disps[tip_tag][1]]

# Analytical E-B curve
x_an  = np.linspace(0, L, 300)
uy_EB = P / (6.0 * E * Iz) * (3.0 * L * x_an**2 - x_an**3)

ax2.plot(x_an, uy_EB * 1e3, 'k--', lw=1.5, label='Euler-Bernoulli (analytical)')
ax2.plot(x_solid, np.array(uy_solid)*1e3, 'o-', color='#2563EB',
         ms=3, lw=1.5, label='Solid (centroid)')
ax2.plot(x_frame, np.array(uy_frame)*1e3, 's-', color='#DC2626',
         ms=8, lw=2, label='Frame')

# Interface line
ax2.axvline(L_solid, color='#F59E0B', lw=1, ls=':', alpha=0.7, label='Interface')

ax2.set_xlabel('x [m]')
ax2.set_ylabel('Vertical deflection [mm]')
ax2.set_title('Deflection along beam centreline', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.25)

plt.tight_layout()
out_png = 'cantilever_solid_frame_results.png'
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {out_png}")
plt.close()

ops.wipe()
print("Done.")
