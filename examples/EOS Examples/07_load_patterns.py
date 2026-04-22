# %% [markdown]
# # 07 — Load Patterns and Linear Superposition
#
# **Curriculum slot:** Tier 2, slot 07.
# **Prerequisite:** 02 — 2D Cantilever Beam.
#
# ## Purpose
#
# The ``g.loads.pattern()`` context manager groups load definitions
# under a **named pattern**. apeGmsh attaches the pattern name to
# every ``LoadDef`` you register inside the ``with`` block, and the
# resolved records (available as ``fem.nodes.loads.by_kind('nodal')``
# and ``fem.elements.loads.by_kind('element')``) carry the pattern
# name forward. At the OpenSees emission stage we group records by
# pattern and emit one ``ops.timeSeries`` + ``ops.pattern "Plain"``
# block per group.
#
# Distributed loads (``g.loads.line``, ``g.loads.surface``) offer a
# **choice of reduction**:
#
# | ``target_form=`` | Record kind | OpenSees call | Accuracy for beams |
# |---|---|---|---|
# | ``"nodal"`` (default) | ``fem.nodes.loads`` | ``ops.load(...)`` | tributary-length equivalent — small discretization error |
# | ``"element"`` | ``fem.elements.loads`` | ``ops.eleLoad('-type', '-beamUniform', ...)`` | exact at nodes for linear elastic beams |
#
# The two are meant to be interchangeable at the user level —
# switching ``target_form`` is a one-kwarg change, and the
# ``run()`` helper below handles both kinds uniformly.
#
# This notebook shows:
#
# 1. How to register independent patterns.
# 2. How the same distributed load can be expressed as **either**
#    nodal or element records by flipping ``target_form=``.
# 3. How to emit one OpenSees pattern per group, handling both
#    kinds in the same dispatch loop.
# 4. **Linear superposition** — combined tip deflection = sum of
#    individual pattern deflections.
# 5. **Fidelity** — element-load dead weight matches the analytical
#    result; nodal-load dead weight has a small tributary error that
#    shrinks with refinement.
#
# ## Problem
#
# The same cantilever from slot 02 (length $L = 3$ m, the usual
# elastic beam section), loaded with:
#
# * **Pattern "dead_nodal"** — uniform distributed $w$ in $-z$,
#   reduced to nodal forces (``target_form="nodal"``).
# * **Pattern "dead_element"** — same $w$, kept as an element-level
#   ``beamUniform`` record (``target_form="element"``).
# * **Pattern "live"** — tip point load $P$ in $-z$. Point loads
#   are inherently nodal; there is no ``target_form=`` for them.
#
# Analytical results for the cantilever:
#
# $$
# \delta_{w} = -\dfrac{w\,L^{4}}{8\,E\,I},
# \qquad
# \delta_{P} = -\dfrac{P\,L^{3}}{3\,E\,I}.
# $$
#
# Verification targets:
#
# * Both dead-load paths must agree with the analytical distributed
#   result — the element path to machine precision, the nodal path
#   to tributary-discretisation accuracy.
# * Combined run ``["dead_element", "live"]`` must equal
#   ``d_dead_element + d_live`` exactly (linearity).

# %% [markdown]
# ## 1. Imports and parameters

# %%
import openseespy.opensees as ops

from apeGmsh import apeGmsh

L = 3.0
w = 2_000.0            # distributed "dead" load [N/m]
P = 10_000.0           # tip "live" point load   [N]

E  = 2.1e11
nu = 0.3
G  = E / (2 * (1 + nu))
A, Iy, Iz, J = 1e-3, 1e-5, 1e-5, 2e-5
LC = L / 10.0

# %% [markdown]
# ## 2. Geometry

# %%
g = apeGmsh(model_name="07_load_patterns", verbose=False)
g.begin()

p_base = g.model.geometry.add_point(0.0, 0.0, 0.0, lc=LC)
p_tip  = g.model.geometry.add_point(L,   0.0, 0.0, lc=LC)
ln     = g.model.geometry.add_line(p_base, p_tip)
g.model.sync()

g.physical.add(0, [p_base], name="base")
g.physical.add(0, [p_tip],  name="tip")
g.physical.add(1, [ln],     name="beam")

# %% [markdown]
# ## 3. Declare loads **inside named patterns**
#
# Every ``LoadDef`` created inside ``with g.loads.pattern(name):``
# carries ``pattern = name``.  We register the same distributed
# load twice — once as nodal-reduced, once as element-level — so
# that later we can pick which one to activate per run and compare.

# %%
with g.loads.pattern("dead_nodal"):
    # target_form="nodal" (the default) reduces the line load to
    # tributary-length nodal forces.  Records end up in
    # fem.nodes.loads and are emitted via ops.load(...).
    g.loads.line(target="beam", magnitude=w, direction=(0.0, 0.0, -1.0),
                 target_form="nodal")

with g.loads.pattern("dead_element"):
    # target_form="element" keeps the load as a beamUniform record
    # on each element.  Records end up in fem.elements.loads and
    # are emitted via ops.eleLoad('-type', '-beamUniform', ...).
    g.loads.line(target="beam", magnitude=w, direction=(0.0, 0.0, -1.0),
                 target_form="element")

with g.loads.pattern("live"):
    g.loads.point(target="tip", force_xyz=(0.0, 0.0, -P))

print(f"patterns registered: {g.loads.patterns()}")

# %% [markdown]
# ## 4. Mesh + resolve
#
# After meshing, the resolved records split across two brokers:
# nodal-reduced records on ``fem.nodes.loads`` and element-level
# records on ``fem.elements.loads``.  Each one still carries the
# pattern name we gave it, so filtering by ``rec.pattern`` picks
# the right subset in the analysis driver below.

# %%
g.mesh.generation.generate(1)
fem = g.mesh.queries.get_fem_data()

nodal_records = list(fem.nodes.loads.by_kind("nodal"))
elem_records  = list(fem.elements.loads.by_kind("element"))

print(f"mesh: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")
print(f"nodal-load records   : {len(nodal_records)}")
print(f"element-load records : {len(elem_records)}")

# Show one of each so the structure is visible
print()
print("first nodal record  :", nodal_records[0])
print("first element record:", elem_records[0])

# %% [markdown]
# ## 5. Analysis driver
#
# The ``run`` function below builds a fresh OpenSees model on each
# call, applies **only the patterns named in** ``active_patterns``,
# and returns the tip deflection.
#
# Each OpenSees pattern ingests **both** kinds in the same block:
#
# * ``fem.nodes.loads.by_kind('nodal')``   → ``ops.load(node, fx,...)``
# * ``fem.elements.loads.by_kind('element')`` → ``ops.eleLoad(...)``
#
# This is the same dispatch structure used by
# ``g.opensees.ingest.loads(fem)`` internally — here we do it by
# hand to make the translation visible.

# %%
def run(active_patterns: list[str]) -> float:
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    # -- nodes + elements --
    for nid, xyz in fem.nodes.get():
        ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))
    # vecxz = (0, 0, 1) → local-z aligned with global +Z, so a
    # record with wz = -2000 (global -Z) is applied correctly.
    ops.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
    for group in fem.elements.get(target="beam"):
        for eid, nodes in zip(group.ids, group.connectivity):
            ops.element("elasticBeamColumn", int(eid),
                        int(nodes[0]), int(nodes[1]),
                        A, E, G, J, Iy, Iz, 1)

    # -- fix the base --
    for n in fem.nodes.get(target="base").ids:
        ops.fix(int(n), 1, 1, 1, 1, 1, 1)

    # -- one OpenSees pattern per apeGmsh pattern --
    nodal_all = list(fem.nodes.loads.by_kind("nodal"))
    elem_all  = list(fem.elements.loads.by_kind("element"))

    for pat_tag, pat_name in enumerate(active_patterns, start=1):
        nodals = [r for r in nodal_all if r.pattern == pat_name]
        elems  = [r for r in elem_all  if r.pattern == pat_name]
        if not nodals and not elems:
            continue

        ops.timeSeries("Constant", pat_tag)
        ops.pattern("Plain", pat_tag, pat_tag)

        # -- nodal records: ops.load(node, fx, fy, fz, mx, my, mz) --
        for rec in nodals:
            fx, fy, fz = rec.force_xyz  if rec.force_xyz  else (0, 0, 0)
            mx, my, mz = rec.moment_xyz if rec.moment_xyz else (0, 0, 0)
            ops.load(int(rec.node_id), fx, fy, fz, mx, my, mz)

        # -- element records: ops.eleLoad(...) --
        # beamUniform order in OpenSees 3D: (Wy, Wz, Wx) in local axes
        for rec in elems:
            p = rec.params
            if rec.load_type == "beamUniform":
                ops.eleLoad('-ele', int(rec.element_id),
                            '-type', '-beamUniform',
                            p.get("wy", 0.0),
                            p.get("wz", 0.0),
                            p.get("wx", 0.0))
            else:
                raise NotImplementedError(
                    f"eleLoad type {rec.load_type!r} not handled in this demo"
                )

    ops.system("BandGeneral"); ops.numberer("Plain"); ops.constraints("Plain")
    ops.test("NormUnbalance", 1e-10, 10); ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0); ops.analysis("Static")
    status = ops.analyze(1)
    assert status == 0, f"analyze failed for patterns={active_patterns}"

    tip_node = int(next(iter(fem.nodes.get(target="tip").ids)))
    return ops.nodeDisp(tip_node, 3)

# %% [markdown]
# ## 6. Four runs: each dead path on its own, live, and combined

# %%
d_dead_nodal   = run(["dead_nodal"])
d_dead_element = run(["dead_element"])
d_live         = run(["live"])
d_both         = run(["dead_element", "live"])

print(f"dead_nodal   tip disp  : {d_dead_nodal:.6e}  m")
print(f"dead_element tip disp  : {d_dead_element:.6e}  m")
print(f"live         tip disp  : {d_live:.6e}  m")
print(f"combined     tip disp  : {d_both:.6e}  m  "
      f"(patterns=['dead_element', 'live'])")

# %% [markdown]
# ## 7. Verification — analytical values + linearity check

# %%
analytical_dead = -w * L**4 / (8.0 * E * Iz)
analytical_live = -P * L**3 / (3.0 * E * Iz)
analytical_both = analytical_dead + analytical_live

err_nodal = abs(d_dead_nodal   - analytical_dead) / abs(analytical_dead) * 100.0
err_elem  = abs(d_dead_element - analytical_dead) / abs(analytical_dead) * 100.0
err_live  = abs(d_live         - analytical_live) / abs(analytical_live) * 100.0
err_both  = abs(d_both         - analytical_both) / abs(analytical_both) * 100.0

# Linearity: d_both should equal d_dead_element + d_live
super_sum = d_dead_element + d_live
err_linearity = abs(d_both - super_sum) / abs(d_both) * 100.0

print("Dead-load case (wL^4 / 8EI)")
print(f"  analytical                : {analytical_dead:.6e}  m")
print(f"  FEM nodal   (tributary)   : {d_dead_nodal:.6e}  m   err {err_nodal:.4f} %")
print(f"  FEM element (beamUniform) : {d_dead_element:.6e}  m   err {err_elem:.4f} %")
print()
print("Live-load case (PL^3 / 3EI)")
print(f"  analytical : {analytical_live:.6e}  m")
print(f"  FEM        : {d_live:.6e}  m   err {err_live:.4f} %")
print()
print("Combined case (element-dead + live, analytical superposition)")
print(f"  analytical : {analytical_both:.6e}  m")
print(f"  FEM        : {d_both:.6e}  m   err {err_both:.4f} %")
print()
print("Numerical linearity: d_both vs (d_dead_element + d_live)")
print(f"  d_dead_element + d_live : {super_sum:.6e}  m")
print(f"  d_both                  : {d_both:.6e}  m")
print(f"  Error                   : {err_linearity:.6e} %")

# %% [markdown]
# ## 8. Bonus — computing ``vecxz`` with the ``elements.vecxz`` helper
#
# The ``run()`` function above hand-codes
# ``ops.geomTransf("Linear", 1, 0.0, 0.0, 1.0)`` — the ``vecxz``
# argument ``(0, 0, 1)`` picks the global-Z direction as the section's
# local-z axis.  Getting that vector wrong is a classic silent bug:
# a wrong ``vecxz`` produces a consistent (but *physically incorrect*)
# answer.  For example, ``vecxz=(0, 1, 0)`` on a beam along +X puts
# the section's local-z on global +Y, so a ``-beamUniform`` in
# ``wz`` applies a *horizontal* load and the tip deflection in
# global -Z comes out zero.
#
# ``g.opensees.elements.vecxz(axis, local_z=(0,0,1), roll_deg=0)``
# computes ``vecxz`` from intent:
#
# * ``axis`` — the beam direction (need not be unit length).
# * ``local_z`` — where the section's local-z should point when
#   ``roll_deg = 0``.  Default global +Z covers most horizontal beams.
# * ``roll_deg`` — section rotation about the beam axis
#   (right-hand rule).  ``roll_deg = 90`` on a horizontal beam rotates
#   a W-shape from strong-axis to weak-axis bending.
#
# The helper raises if ``axis`` and ``local_z`` are collinear (no
# unique x-z plane exists) — catching the silent-footgun case early.

# %%
beam_axis = (1.0, 0.0, 0.0)

vxz_default = g.opensees.elements.vecxz(axis=beam_axis)
vxz_weak    = g.opensees.elements.vecxz(axis=beam_axis, roll_deg=90)
vxz_local_y = g.opensees.elements.vecxz(axis=beam_axis, local_z=(0, 1, 0))

print(f"vecxz (default, local_z=+Z)        : {vxz_default}")
print(f"vecxz (roll_deg=90, weak axis up)  : {vxz_weak}")
print(f"vecxz (local_z=+Y explicit)        : {vxz_local_y}")

# %% [markdown]
# ### Re-run the element-load case using the helper
#
# Proof of equivalence: swap the hand-written ``(0, 0, 1)`` for the
# helper output inside a fresh ``run_with_helper()`` and confirm the
# tip deflection is unchanged.

# %%
def run_with_helper(active_patterns: list[str]) -> float:
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    for nid, xyz in fem.nodes.get():
        ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))

    # Use the helper instead of a hand-written vecxz.
    vxz = g.opensees.elements.vecxz(axis=beam_axis)
    ops.geomTransf("Linear", 1, *vxz)

    for group in fem.elements.get(target="beam"):
        for eid, nodes in zip(group.ids, group.connectivity):
            ops.element("elasticBeamColumn", int(eid),
                        int(nodes[0]), int(nodes[1]),
                        A, E, G, J, Iy, Iz, 1)

    for n in fem.nodes.get(target="base").ids:
        ops.fix(int(n), 1, 1, 1, 1, 1, 1)

    elem_all = list(fem.elements.loads.by_kind("element"))
    for pat_tag, pat_name in enumerate(active_patterns, start=1):
        elems = [r for r in elem_all if r.pattern == pat_name]
        if not elems:
            continue
        ops.timeSeries("Constant", pat_tag)
        ops.pattern("Plain", pat_tag, pat_tag)
        for rec in elems:
            p = rec.params
            if rec.load_type == "beamUniform":
                ops.eleLoad('-ele', int(rec.element_id),
                            '-type', '-beamUniform',
                            p.get("wy", 0.0),
                            p.get("wz", 0.0),
                            p.get("wx", 0.0))

    ops.system("BandGeneral"); ops.numberer("Plain"); ops.constraints("Plain")
    ops.test("NormUnbalance", 1e-10, 10); ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0); ops.analysis("Static")
    status = ops.analyze(1)
    assert status == 0

    tip_node = int(next(iter(fem.nodes.get(target="tip").ids)))
    return ops.nodeDisp(tip_node, 3)

d_helper = run_with_helper(["dead_element"])
print(f"dead_element via helper vecxz : {d_helper:.6e}  m")
print(f"dead_element hand-written     : {d_dead_element:.6e}  m")
print(f"difference                    : {abs(d_helper - d_dead_element):.2e} m")

# %% [markdown]
# ## 9. (Optional) viewer check
#
# Uncomment in Jupyter to open the results viewer on the combined
# run.

# %%
# g.mesh.results_viewer()

# %% [markdown]
# ## What this unlocks
#
# * **Named patterns** — ``g.loads.pattern("foo")`` groups any mix
#   of ``point``, ``line``, ``surface``, ``gravity``, etc.  Every
#   later notebook that separates dead / live / seismic reuses
#   this structure.
# * **Pattern-filtered emission** — filter by ``rec.pattern`` to
#   decide what enters each ``ops.pattern "Plain"`` block.  One
#   OpenSees pattern per apeGmsh pattern is the standard mapping.
# * **Two reduction paths, one API** — ``target_form="nodal"``
#   (default) reduces distributed loads to nodal forces via
#   tributary length; ``target_form="element"`` keeps them on the
#   elements as ``beamUniform`` records.  The choice is a one-kwarg
#   switch and the downstream FEM data brokers separate the two
#   cleanly (``fem.nodes.loads`` vs ``fem.elements.loads``).
# * **When to prefer which?**  For beam/shell elements, the element
#   path is *exact* at nodes for linear elastic problems — no
#   discretisation error from load reduction.  The nodal path is
#   simpler (one ``ops.load`` line per node, no ``eleLoad``) and
#   its error vanishes with refinement, so it is a fine default for
#   dense meshes or when you plan to post-process nodal forces.
# * **``g.opensees.ingest.loads(fem)``** — in production code you
#   typically let the ingest adapter translate both kinds
#   automatically; the manual dispatch in ``run()`` above is shown
#   for pedagogy.
# * **``g.opensees.elements.vecxz(axis, local_z, roll_deg)``** —
#   derives the ``geomTransf`` ``vecxz`` from intent instead of
#   raw numbers, guarding against the collinear-axis footgun and
#   making section rotation a one-kwarg change.

# %%
g.end()
