# %% [markdown]
# # 06 — Sections Catalog (elastic, elastic-section, fiber)
#
# **Curriculum slot:** Tier 2, slot 06.
# **Prerequisite:** 02 — 2D Cantilever Beam.
#
# ## Purpose
#
# OpenSees offers three escalating ways to describe a beam cross
# section:
#
# | Path | What you declare | What's stored |
# |---|---|---|
# | 1. **Inline** — ``ops.element("elasticBeamColumn", eid, i, j, A, E, G, J, Iy, Iz, transf)`` | axial / flexural / torsional stiffnesses per element | 6×6 closed-form tangent |
# | 2. **Elastic section** — ``ops.section("Elastic", tag, E, A, Iz, Iy, G, J)`` used by a ``forceBeamColumn`` / ``dispBeamColumn`` | same stiffnesses but in a named section | section object referenced by tag, integrated along the element |
# | 3. **Fiber section** — ``ops.section("Fiber", tag)`` + ``ops.fiber`` / ``ops.patch`` / ``ops.layer`` | a grid of (y, z, area, matTag) fibers | stress resultants computed by integrating over fibers each step |
#
# This notebook builds **the same cantilever three times** — one per
# path — and confirms the three tip deflections agree to machine
# precision. That's the sanity check the more sophisticated nonlinear
# fiber-section applications (embedded rebar, moment-curvature,
# pushover) rely on: if the elastic baseline doesn't match, no
# nonlinear result from the same section can be trusted.

# %% [markdown]
# ## 1. Imports and parameters

# %%
import openseespy.opensees as ops

from apeGmsh import apeGmsh

# Geometry + load (same numbers as slot 02)
L = 3.0
P = 10_000.0
E  = 2.1e11
nu = 0.3
G  = E / (2 * (1 + nu))

# Cross section: rectangle b x h
b, h = 0.05, 0.10                        # m
A   = b * h
Iz  = b * h**3 / 12.0                    # bending about +y (in-plane)
Iy  = h * b**3 / 12.0                    # bending about +z
J   = Iy + Iz                            # approx for thin rect
# Torsion constant for a rectangle is more subtle, but for this
# bending-only problem the torsion stiffness doesn't enter the
# verification.

LC = L / 10.0


# %% [markdown]
# ## 2. Geometry (apeGmsh, shared across all three runs)

# %%
g_ctx = apeGmsh(model_name="06_sections_catalog", verbose=False)
g = g_ctx.__enter__()

p_base = g.model.geometry.add_point(0.0, 0.0, 0.0, lc=LC)
p_tip  = g.model.geometry.add_point(L,   0.0, 0.0, lc=LC)
ln     = g.model.geometry.add_line(p_base, p_tip)
g.model.sync()

g.physical.add(0, [p_base], name="base")
g.physical.add(0, [p_tip],  name="tip")
g.physical.add(1, [ln],     name="beam")

g.mesh.generation.generate(1)
fem = g.mesh.queries.get_fem_data()
print(f"mesh: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")


# %% [markdown]
# ## 3. Analysis driver
#
# ``run(element_factory)`` builds a fresh OpenSees model, lets the
# caller declare whatever materials + sections it wants, wires up
# one element per meshed segment via the given ``element_factory``,
# fixes the base, applies the tip load, and returns the tip $u_z$.
# The three paths below differ **only** in what ``element_factory``
# does.

# %%
def run(setup, element_factory) -> float:
    """Driver. ``setup()`` declares materials / sections / integrations
    ONCE before the element loop. ``element_factory(eid, ni, nj)`` then
    emits a single ops.element per meshed segment, reusing whatever
    setup() created."""
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    for nid, xyz in fem.nodes.get():
        ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))

    ops.geomTransf("Linear", 1, 0.0, 1.0, 0.0)
    setup()

    for group in fem.elements.get(target="beam"):
        for eid, nodes in zip(group.ids, group.connectivity):
            element_factory(int(eid), int(nodes[0]), int(nodes[1]))

    for n in fem.nodes.get(target="base").ids:
        ops.fix(int(n), 1, 1, 1, 1, 1, 1)

    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    for n in fem.nodes.get(target="tip").ids:
        ops.load(int(n), 0.0, 0.0, -P, 0.0, 0.0, 0.0)

    ops.system("BandGeneral"); ops.numberer("Plain"); ops.constraints("Plain")
    ops.test("NormUnbalance", 1e-10, 10); ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0); ops.analysis("Static")
    ops.analyze(1)
    return ops.nodeDisp(int(next(iter(fem.nodes.get(target="tip").ids))), 3)


# %% [markdown]
# ## 4. Path 1 — inline ``elasticBeamColumn``
#
# The baseline. No section concept — every stiffness is a direct
# argument. This is what slots 02-05 used.

# %%
def setup_1():
    # Nothing to set up — elasticBeamColumn takes stiffness inline.
    pass

def make_elastic_beam_col(eid, ni, nj):
    ops.element("elasticBeamColumn", eid, ni, nj,
                A, E, G, J, Iy, Iz, 1)

d1 = run(setup_1, make_elastic_beam_col)


# %% [markdown]
# ## 5. Path 2 — ``forceBeamColumn`` with an Elastic section
#
# Declare the same stiffnesses as a **named section**, then use
# a ``forceBeamColumn`` element with Lobatto integration (5 points
# per element). Mathematically identical for linear elastic.

# %%
SEC_ELASTIC_TAG = 10
INT_ELASTIC_TAG = 20

def setup_2():
    ops.section("Elastic", SEC_ELASTIC_TAG, E, A, Iz, Iy, G, J)
    ops.beamIntegration("Lobatto", INT_ELASTIC_TAG, SEC_ELASTIC_TAG, 5)

def make_fbc_elastic_section(eid, ni, nj):
    ops.element("forceBeamColumn", eid, ni, nj, 1, INT_ELASTIC_TAG)

d2 = run(setup_2, make_fbc_elastic_section)


# %% [markdown]
# ## 6. Path 3 — ``forceBeamColumn`` with a Fiber section
#
# Build a **fiber section** with one elastic material and
# ``ops.patch("rect", ...)`` to discretise the $b \times h$ rectangle
# into a small grid of fibers (12×12 here — 144 fibers total).
#
# For elastic fibers this is **exactly equivalent** to the elastic
# section above in the limit of a refined fiber mesh. Any residual
# error here comes from the finite fiber count.

# %%
MAT_ELASTIC_TAG = 30
SEC_FIBER_TAG = 31
INT_FIBER_TAG = 40

def setup_3():
    ops.uniaxialMaterial("Elastic", MAT_ELASTIC_TAG, E)

    # Fiber sections in 3D need an explicit torsional stiffness — the
    # fiber patch only covers axial and flexural resultants. Declare
    # it via "-GJ"; any value large compared to the problem's torsion
    # demand works (we're not loading in torsion here).
    ops.section("Fiber", SEC_FIBER_TAG, "-GJ", G * J)

    # patch("rect", matTag, nSubdivY, nSubdivZ, yI, zI, yJ, zJ) —
    # rectangle from (yI, zI) to (yJ, zJ), subdivided into an n × m
    # grid of fibers, each carrying the given uniaxial material.
    NY, NZ = 12, 12
    ops.patch("rect", MAT_ELASTIC_TAG, NY, NZ, -h/2, -b/2, h/2, b/2)

    ops.beamIntegration("Lobatto", INT_FIBER_TAG, SEC_FIBER_TAG, 5)

def make_fbc_fiber_section(eid, ni, nj):
    ops.element("forceBeamColumn", eid, ni, nj, 1, INT_FIBER_TAG)

d3 = run(setup_3, make_fbc_fiber_section)


# %% [markdown]
# ## 7. Result extraction and verification
#
# Tip $u_z$ for each path compared to the classical cantilever
# formula $\delta = PL^{3}/(3EI_z)$.

# %%
analytical = -P * L**3 / (3.0 * E * Iz)
err_1 = abs(d1 - analytical) / abs(analytical) * 100.0
err_2 = abs(d2 - analytical) / abs(analytical) * 100.0
err_3 = abs(d3 - analytical) / abs(analytical) * 100.0

print(f"analytical PL^3 / 3EI    :  {analytical:.6e}  m")
print()
print(f"path 1 elasticBeamColumn :  {d1:.6e}  m   err {err_1:.4f} %")
print(f"path 2 forceBC + Elastic :  {d2:.6e}  m   err {err_2:.4f} %")
print(f"path 3 forceBC + Fiber   :  {d3:.6e}  m   err {err_3:.4f} %")
print()
print(f"path 2 vs path 1  (delta):  {abs(d2 - d1):.3e}  m")
print(f"path 3 vs path 1  (delta):  {abs(d3 - d1):.3e}  m")


# %% [markdown]
# ### Why the fiber-section path has ~0.7% residual
#
# Paths 1 and 2 agree to machine precision (they both encode the
# closed-form elastic tangent). Path 3 is visibly off by ~0.7%. That
# is **not** a bug — it's the fiber-discretization error of the
# rectangular patch.
#
# For an $n_z$-subdivision grid of uniform fibers spanning the
# depth $h$, the integrated second moment about the bending axis is
#
# $$
# I_{\text{fib}}
#   \;=\;
#   \dfrac{b\,h^{3}}{12}\,\left(1 - \dfrac{1}{n_{z}^{2}}\right).
# $$
#
# With ``NZ = 12`` we get $I_{\text{fib}}/I_{\text{exact}} =
# 1 - 1/144 \approx 0.9931$. A cantilever tip deflection is
# inversely proportional to $I$, so the FEM tip displacement
# over-estimates the analytical one by $(I_{\text{exact}} /
# I_{\text{fib}} - 1) \approx 1/143 \approx 0.70\%$ — exactly
# what path 3 prints.
#
# Bumping ``NY, NZ`` to 50 would drop the residual to $1/2500
# \approx 0.04\%$. Bumping to 100 gives $0.01\%$. The fiber count
# is the **only** knob; it's worth knowing the cost of refinement
# is just memory per section.


# %% [markdown]
# ## 8. (Optional) viewer check
#
# Uncomment in Jupyter to open the results viewer on path 3.

# %%
# g.mesh.results_viewer()


# %% [markdown]
# ## What this unlocks
#
# * **Three-way section equivalence.** If the elastic baseline
#   doesn't match across inline / elastic-section / fiber-section,
#   any nonlinear fiber result from the same section is suspect.
#   This notebook's verification is the sanity gate every later
#   section-based notebook builds on.
# * **``forceBeamColumn`` + ``beamIntegration`` idiom.** This is the
#   portal to every nonlinear beam problem — fiber sections,
#   plasticity, pushover, moment-curvature. The element takes a
#   ``beamIntegration`` tag (Lobatto, Legendre, etc.) and the
#   integration references a section tag.
# * **Fibre-section construction via ``ops.patch``.** The rectangular
#   patch shorthand is the standard for prismatic cross sections —
#   later RC examples swap the ``Elastic`` material for ``Concrete02``
#   and add a ``ops.layer("straight", …)`` for rebar.

# %%
g_ctx.__exit__(None, None, None)
