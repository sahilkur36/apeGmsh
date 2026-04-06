"""
example_column.py
=================
Full pyGmsh workflow for a 3-D cantilever reinforced-concrete column.

Pipeline
--------
    Geometry  →  Mesh  →  Physical Groups  →  Inspect
    →  Quality report  →  Plots  →  OpenSees export

Geometry
    Square cross-section column: 0.30 m × 0.30 m × 3.00 m (W × D × H)
    Built entirely from OCC primitives — no external CAD file needed.

Physical groups
    "Concrete"  — volume  (FourNodeTetrahedron, ElasticIsotropic)
    "Fixed"     — bottom face  (fully fixed: dofs = [1,1,1])
    "TopFace"   — top face     (lateral wind load + self-weight reaction)

Outputs
    column.msh  — Gmsh native mesh
    column.tcl  — OpenSees Tcl input script
    column.py   — openseespy Python script

Two usage styles are shown:

    Style A — context manager (recommended)
        ``with pyGmsh(...) as g:``
        Gmsh is initialised on entry and finalised automatically on exit,
        even if an exception occurs.

    Style B — explicit initialize / finalize
        ``g = pyGmsh(...)``
        ``g.initialize()``
        ``...``
        ``g.finalize()``
        Useful when the session spans multiple functions or when you need
        the pyGmsh object to outlive a single block.  Use a try/finally
        to guarantee ``finalize()`` is always called.

Both styles produce identical results; choose whichever fits your
application structure.
"""

from pyGmsh import pyGmsh, Algorithm3D, OptimizeMethod

# ── geometry ──────────────────────────────────────────────────────────────
W, D, H = 0.30, 0.30, 3.00   # column width, depth, height  [m]
LC      = 0.07                # target element size           [m]

# ── material ──────────────────────────────────────────────────────────────
E_C  = 30.0e9   # Young's modulus  [Pa]
NU   = 0.20     # Poisson's ratio
RHO  = 2400.0   # density          [kg/m³]
G    = 9.81     # gravity          [m/s²]

# ── loads ─────────────────────────────────────────────────────────────────
WIND_FORCE = 5.0e3   # lateral wind force applied per top-face node [N]


# ═══════════════════════════════════════════════════════════════════════════
# Shared pipeline — called by both usage styles below
# ═══════════════════════════════════════════════════════════════════════════

def run_column_pipeline(g: pyGmsh) -> None:
    """Execute the full column workflow on an already-initialised g."""

    # ── 1. Geometry ───────────────────────────────────────────────────────
    col = g.model.add_box(0.0, 0.0, 0.0, W, D, H)
    g.model.sync()

    # ── 2. Identify surfaces before meshing ───────────────────────────────
    # Inspect queries OCC analytically — no mesh needed at this stage.
    mapping, global_summary = g.inspect.get_geometry_info()
    surf_df = mapping['surfaces']['df']

    # Filter by centre-of-mass Z coordinate
    tol     = 1e-6
    bot_tag = int(surf_df.loc[surf_df['cz'].abs()       < tol, 'tag'].iloc[0])
    top_tag = int(surf_df.loc[(surf_df['cz'] - H).abs() < tol, 'tag'].iloc[0])

    print(f"\nBottom face tag : {bot_tag}")
    print(f"Top    face tag : {top_tag}")
    print("\n── Global geometry summary ──")
    print(global_summary.to_string())

    # ── 3. Mesh ───────────────────────────────────────────────────────────
    (g.mesh
       .set_global_size(LC)
       .set_algorithm(0, Algorithm3D.HXT, dim=3)   # fast, high-quality tets
       .generate(3)
       .optimize(OptimizeMethod.NETGEN, niter=3))   # smooth with Netgen

    # ── 4. Physical groups ────────────────────────────────────────────────
    (g.physical
       .add_volume( [col],      name="Concrete")
       .add_surface([bot_tag],  name="Fixed")
       .add_surface([top_tag],  name="TopFace"))

    print("\n── Physical groups ──")
    print(g.physical.summary().to_string())

    # ── 5. Mesh quality report ────────────────────────────────────────────
    print("\n── Mesh quality (minSICN) ──")
    print(g.mesh.quality_report().to_string())

    # ── 6. Plots ──────────────────────────────────────────────────────────
    print("\nRendering geometry plot ...")
    (g.plot
       .geometry(
           show_surfaces=True,
           surface_alpha=0.15,
           label_tags=True,
           show=False,
       )
       .label_entities(dims=[2])
       .show())

    print("Rendering quality plot ...")
    g.plot.clear()
    g.plot.quality(quality_name="minSICN", cmap="RdYlGn", show=True)

    # ── 7. Save mesh ──────────────────────────────────────────────────────
    g.mesh.save("column.msh")
    print("\nMesh written → column.msh")

    # ── 8. OpenSees model ─────────────────────────────────────────────────
    (g.opensees
       .set_model(ndm=3, ndf=3)
       .add_nd_material(
           "Concrete", "ElasticIsotropic",
           E=E_C, nu=NU, rho=RHO,
       )
       .assign_element(
           "Concrete", "FourNodeTetrahedron",
           material="Concrete",
           bodyForce=[0.0, 0.0, -RHO * G],
       )
       .fix("Fixed", dofs=[1, 1, 1])
       .add_nodal_load("Wind", "TopFace", force=[WIND_FORCE, 0.0, 0.0])
       .build()
    )

    print("\n── OpenSees model summary ──")
    print(g.opensees.summary())

    df_nodes = g.opensees.node_table()
    df_elems = g.opensees.element_table()
    print(f"\nNodes    : {len(df_nodes)}")
    print(f"Elements : {len(df_elems)}")
    print("\nElement types:\n",
          df_elems.groupby('ops_type').size().to_string())

    (g.opensees
       .export_tcl("column.tcl")
       .export_py ("column.py"))

    print("\nOpenSees scripts written → column.tcl  column.py")


# ═══════════════════════════════════════════════════════════════════════════
# Style A — context manager  (recommended)
#
# ``with pyGmsh(...) as g:`` calls gmsh.initialize() on entry and
# gmsh.finalize() on exit — guaranteed even if an exception is raised.
# ═══════════════════════════════════════════════════════════════════════════
#
# Style B — explicit initialize / finalize
#
# ``g = pyGmsh(...)`` / ``g.initialize()`` / ``g.finalize()``
# Useful when the session spans multiple functions or when you need
# the pyGmsh object to outlive a single block.  Wrap in try/finally to
# guarantee ``finalize()`` is always called even on error.
# ═══════════════════════════════════════════════════════════════════════════

USE_CONTEXT_MANAGER = True   # flip to False to run Style B instead

if USE_CONTEXT_MANAGER:
    print("── Style A: context manager ──")
    with pyGmsh(model_name="CantileverColumn", verbose=True) as g:
        run_column_pipeline(g)
    print("\n── Done ──")

else:
    print("── Style B: explicit initialize / finalize ──")
    g = pyGmsh(model_name="CantileverColumn", verbose=True)
    g.initialize()
    try:
        run_column_pipeline(g)
    finally:
        g.finalize()
    print("\n── Done ──")
