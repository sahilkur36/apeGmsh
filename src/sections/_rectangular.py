"""
Rectangular RC column fiber section generator.

Uses Gmsh to mesh the cross-section into quad/tri fibers with separate
core (confined) and cover (unconfined) concrete regions, then injects
materials and fibers directly into an active OpenSeesPy model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy import ndarray


# ── Fiber data container ────────────────────────────────────────────
@dataclass
class FiberData:
    """Raw fiber arrays extracted from the Gmsh mesh."""

    y:        ndarray   # horizontal centroid of each fiber [mm]
    z:        ndarray   # vertical centroid of each fiber   [mm]
    area:     ndarray   # area of each fiber                [mm²]
    mat_tag:  ndarray   # material tag assigned to each fiber (int)
    region:   ndarray   # label per fiber: "core", "cover", or "steel"

    @property
    def n_fibers(self) -> int:
        return len(self.y)


# ── Rebar layer descriptor ─────────────────────────────────────────
@dataclass
class _RebarLayer:
    """Internal: a single row/column of bars."""

    n_bars: int
    bar_diam: float
    y_coords: ndarray
    z_coords: ndarray

    @property
    def bar_area(self) -> float:
        return np.pi * self.bar_diam ** 2 / 4.0


# ── Main class ──────────────────────────────────────────────────────
class RectangularColumnSection:
    """
    Rectangular reinforced-concrete column fiber section.

    Creates a Gmsh mesh of the cross-section (core + cover), extracts
    fiber data, and can inject materials + fibers into an active
    OpenSeesPy model.

    Parameters
    ----------
    b : float
        Section width [mm].
    h : float
        Section depth [mm].
    cover : float
        Clear cover to the *outside* face of the stirrup [mm].
    top_bars : tuple[int, float]
        ``(number_of_bars, bar_diameter_mm)`` along the top face.
    bot_bars : tuple[int, float]
        ``(number_of_bars, bar_diameter_mm)`` along the bottom face.
    side_bars : tuple[int, float] | None
        ``(number_of_bars_per_side, bar_diameter_mm)`` along each
        vertical face (intermediate bars between top and bottom layers).
        ``None`` means no side bars.
    fc : float
        Concrete compressive strength f'c [MPa] (positive value).
    fy : float
        Steel yield strength [MPa].
    Es : float
        Steel elastic modulus [MPa].
    confinement_factor : float
        Ratio of confined-to-unconfined peak stress (default 1.3).
        Applied as ``f'cc = confinement_factor × f'c``.
    mesh_size : float
        Target fiber (element) size [mm].
    mesh_size_tol : float
        Fractional tolerance around *mesh_size* for Gmsh's min/max
        bounds.  E.g. 0.2 → min = 0.8 × mesh_size, max = 1.2 × mesh_size.
    concrete_material : str
        OpenSees uniaxial material model for concrete.
        Currently ``"Concrete02"`` is supported.
    steel_material : str
        OpenSees uniaxial material model for steel.
        Currently ``"Steel02"`` is supported.

    Example
    -------
    ::

        import openseespy.opensees as ops
        from sections import RectangularColumnSection

        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 3)

        sec = RectangularColumnSection(
            b=400, h=600, cover=40,
            top_bars=(3, 25), bot_bars=(3, 25),
            side_bars=(2, 16),
            fc=30, fy=420,
        )

        # Build materials (tags 1-3) and fiber section (tag 1)
        sec.build(sec_tag=1, start_mat_tag=1)

        # Now use sec_tag=1 in any beam/column or zero-length element
    """

    # ── construction ────────────────────────────────────────────────
    def __init__(
        self,
        b: float,
        h: float,
        cover: float,
        top_bars: tuple[int, float],
        bot_bars: tuple[int, float],
        *,
        side_bars: tuple[int, float] | None = None,
        fc: float = 30.0,
        fy: float = 420.0,
        Es: float = 200_000.0,
        confinement_factor: float = 1.3,
        mesh_size: float = 20.0,
        mesh_size_tol: float = 0.2,
        concrete_material: str = "Concrete02",
        steel_material: str = "Steel02",
    ) -> None:
        # geometry
        self.b     = float(b)
        self.h     = float(h)
        self.cover = float(cover)

        # rebar
        self.top_bars  = top_bars
        self.bot_bars  = bot_bars
        self.side_bars = side_bars

        # materials
        self.fc  = float(fc)
        self.fy  = float(fy)
        self.Es  = float(Es)
        self.confinement_factor = float(confinement_factor)
        self.concrete_material  = concrete_material
        self.steel_material     = steel_material

        # mesh
        self.mesh_size     = float(mesh_size)
        self.mesh_size_tol = float(mesh_size_tol)

        # derived
        self.Ec     = 4700.0 * np.sqrt(self.fc)
        self.eps_c0 = 2.0 * self.fc / self.Ec
        self.ft     = 0.62 * np.sqrt(self.fc)

        # cover-to-bar-centroid distance
        self._ci = self.cover + max(self.top_bars[1], self.bot_bars[1]) / 2.0

        # state
        self._fibers: FiberData | None = None
        self._rebar_layers: list[_RebarLayer] = []
        self._built = False

    # ── public properties ───────────────────────────────────────────
    @property
    def Ag(self) -> float:
        """Gross cross-section area [mm²]."""
        return self.b * self.h

    @property
    def core_dimensions(self) -> tuple[float, float]:
        """(width, height) of the confined core [mm]."""
        ci = self._ci
        return (self.b - 2 * ci, self.h - 2 * ci)

    @property
    def rho(self) -> float:
        """Longitudinal reinforcement ratio As/Ag."""
        As = self._total_steel_area()
        return As / self.Ag

    # ── meshing (Gmsh) ──────────────────────────────────────────────
    def mesh(self, *, verbose: bool = False) -> FiberData:
        """
        Mesh the cross-section with Gmsh and extract fiber data.

        Returns the :class:`FiberData` and caches it internally so that
        :meth:`build` can use it without re-meshing.
        """
        import gmsh

        gmsh.initialize()
        gmsh.model.add("_rc_section")

        if not verbose:
            gmsh.option.setNumber("General.Verbosity", 0)

        ci = self._ci
        b, h = self.b, self.h

        # ── outer rectangle (full section, origin at centroid) ──
        p1 = gmsh.model.occ.addPoint(-b/2,      -h/2,      0)
        p2 = gmsh.model.occ.addPoint( b/2,      -h/2,      0)
        p3 = gmsh.model.occ.addPoint( b/2,       h/2,      0)
        p4 = gmsh.model.occ.addPoint(-b/2,       h/2,      0)
        outer_loop = gmsh.model.occ.addCurveLoop([
            gmsh.model.occ.addLine(p1, p2),
            gmsh.model.occ.addLine(p2, p3),
            gmsh.model.occ.addLine(p3, p4),
            gmsh.model.occ.addLine(p4, p1),
        ])

        # ── inner rectangle (confined core) ──
        p5 = gmsh.model.occ.addPoint(-b/2 + ci, -h/2 + ci, 0)
        p6 = gmsh.model.occ.addPoint( b/2 - ci, -h/2 + ci, 0)
        p7 = gmsh.model.occ.addPoint( b/2 - ci,  h/2 - ci, 0)
        p8 = gmsh.model.occ.addPoint(-b/2 + ci,  h/2 - ci, 0)
        inner_loop = gmsh.model.occ.addCurveLoop([
            gmsh.model.occ.addLine(p5, p6),
            gmsh.model.occ.addLine(p6, p7),
            gmsh.model.occ.addLine(p7, p8),
            gmsh.model.occ.addLine(p8, p5),
        ])

        # surfaces
        core_surf  = gmsh.model.occ.addPlaneSurface([inner_loop])
        cover_surf = gmsh.model.occ.addPlaneSurface([outer_loop, inner_loop])
        gmsh.model.occ.synchronize()

        # physical groups
        gmsh.model.addPhysicalGroup(2, [core_surf],  name="Core")
        gmsh.model.addPhysicalGroup(2, [cover_surf], name="Cover")

        # mesh sizing
        tol = self.mesh_size_tol
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_size * (1.0 - tol))
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_size * (1.0 + tol))
        gmsh.option.setNumber("Mesh.Algorithm", 8)           # Frontal-Delaunay quads
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)

        gmsh.model.mesh.generate(2)

        # ── extract fibers ──
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = node_coords.reshape(-1, 3)
        node_map = {int(t): node_coords[i, :2]
                    for i, t in enumerate(node_tags)}

        core_y,  core_z,  core_A  = self._extract_pg_fibers("Core",  node_map, gmsh)
        cover_y, cover_z, cover_A = self._extract_pg_fibers("Cover", node_map, gmsh)

        gmsh.finalize()

        # ── rebar ──
        self._build_rebar_layers()
        rebar_y, rebar_z, rebar_A = self._rebar_arrays()

        # ── assemble FiberData (mat_tag filled during build) ──
        y    = np.concatenate([core_y,  cover_y,  rebar_y])
        z    = np.concatenate([core_z,  cover_z,  rebar_z])
        area = np.concatenate([core_A,  cover_A,  rebar_A])

        # placeholder tags (0 = unassigned until build)
        mat  = np.zeros(len(y), dtype=int)

        region = np.array(
            ["core"]  * len(core_A)
            + ["cover"] * len(cover_A)
            + ["steel"] * len(rebar_A)
        )

        self._fibers = FiberData(y=y, z=z, area=area, mat_tag=mat, region=region)

        if verbose:
            self._print_mesh_summary()

        return self._fibers

    # ── OpenSeesPy injection ────────────────────────────────────────
    def build(
        self,
        sec_tag: int = 1,
        *,
        start_mat_tag: int = 1,
        GJ: float | None = None,
    ) -> dict[str, int]:
        """
        Define materials and fiber section in an active OpenSeesPy model.

        Parameters
        ----------
        sec_tag : int
            Section tag for ``ops.section('Fiber', sec_tag)``.
        start_mat_tag : int
            First material tag to use.  Three tags are consumed:
            ``start_mat_tag`` (confined concrete),
            ``start_mat_tag + 1`` (unconfined concrete),
            ``start_mat_tag + 2`` (steel).
        GJ : float | None
            Optional torsional stiffness.  If provided, the section is
            created with ``ops.section('Fiber', sec_tag, '-GJ', GJ)``.

        Returns
        -------
        dict
            ``{'sec_tag': int, 'core_mat': int, 'cover_mat': int,
            'steel_mat': int}``  — the tags that were registered.
        """
        import openseespy.opensees as ops

        if self._fibers is None:
            self.mesh()

        fibers = self._fibers
        assert fibers is not None

        core_tag  = start_mat_tag
        cover_tag = start_mat_tag + 1
        steel_tag = start_mat_tag + 2

        # ── define materials ──
        self._define_materials(ops, core_tag, cover_tag, steel_tag)

        # ── assign material tags to fiber data ──
        mask_core  = fibers.region == "core"
        mask_cover = fibers.region == "cover"
        mask_steel = fibers.region == "steel"

        fibers.mat_tag[mask_core]  = core_tag
        fibers.mat_tag[mask_cover] = cover_tag
        fibers.mat_tag[mask_steel] = steel_tag

        # ── create fiber section ──
        if GJ is not None:
            ops.section('Fiber', sec_tag, '-GJ', GJ)
        else:
            ops.section('Fiber', sec_tag)

        for yi, zi, Ai, mi in zip(fibers.y, fibers.z, fibers.area, fibers.mat_tag):
            # ops.fiber(yLoc, zLoc, A, matTag)
            # OpenSees convention: first arg is along the section depth (z),
            # second is along the width (y)
            ops.fiber(float(zi), float(yi), float(Ai), int(mi))

        self._built = True

        return {
            "sec_tag":   sec_tag,
            "core_mat":  core_tag,
            "cover_mat": cover_tag,
            "steel_mat": steel_tag,
        }

    # ── accessors ───────────────────────────────────────────────────
    def get_fibers(self) -> FiberData:
        """Return the cached fiber data (meshes if needed)."""
        if self._fibers is None:
            self.mesh()
        assert self._fibers is not None
        return self._fibers

    # ── visualisation ───────────────────────────────────────────────
    def plot(self, *, ax=None, show: bool = True):
        """
        Plot the fiber section layout.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            If ``None``, a new figure is created.
        show : bool
            Call ``plt.show()`` at the end.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        fibers = self.get_fibers()

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 9))

        b, h, ci = self.b, self.h, self._ci

        mask_core  = fibers.region == "core"
        mask_cover = fibers.region == "cover"
        mask_steel = fibers.region == "steel"

        # concrete fibers (size proportional to area)
        if mask_cover.any():
            ax.scatter(
                fibers.y[mask_cover], fibers.z[mask_cover],
                s=fibers.area[mask_cover] * 0.03,
                c="lightblue", edgecolors="steelblue",
                linewidths=0.5, alpha=0.7, label="Cover concrete",
            )
        if mask_core.any():
            ax.scatter(
                fibers.y[mask_core], fibers.z[mask_core],
                s=fibers.area[mask_core] * 0.03,
                c="lightcoral", edgecolors="firebrick",
                linewidths=0.5, alpha=0.7, label="Core concrete (confined)",
            )
        if mask_steel.any():
            ax.scatter(
                fibers.y[mask_steel], fibers.z[mask_steel],
                s=fibers.area[mask_steel] * 0.5,
                c="black", marker="o", zorder=5, label="Steel rebar",
            )

        # section outline
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle(
            (-b/2, -h/2), b, h,
            fill=False, edgecolor="black", linewidth=2,
        ))
        ax.add_patch(Rectangle(
            (-b/2 + ci, -h/2 + ci), b - 2*ci, h - 2*ci,
            fill=False, edgecolor="firebrick", linewidth=1.5, linestyle="--",
        ))

        ax.set_xlabel("y [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_title(f"RC Section {self.b:.0f}×{self.h:.0f} mm  "
                     f"({fibers.n_fibers} fibers, ρ={self.rho*100:.2f}%)")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        if show:
            plt.tight_layout()
            plt.show()

        return ax

    def summary(self) -> str:
        """Return a human-readable summary string."""
        fibers = self.get_fibers()
        mask_core  = fibers.region == "core"
        mask_cover = fibers.region == "cover"
        mask_steel = fibers.region == "steel"

        lines = [
            f"RectangularColumnSection {self.b:.0f} × {self.h:.0f} mm",
            f"  Cover: {self.cover:.0f} mm",
            f"  Top bars:    {self.top_bars[0]} ø {self.top_bars[1]:.0f}",
            f"  Bottom bars: {self.bot_bars[0]} ø {self.bot_bars[1]:.0f}",
        ]
        if self.side_bars:
            lines.append(
                f"  Side bars:   {self.side_bars[0]} ø {self.side_bars[1]:.0f} (each side)"
            )
        lines += [
            f"  f'c = {self.fc:.1f} MPa,  fy = {self.fy:.1f} MPa",
            f"  Confinement factor: {self.confinement_factor:.2f}",
            f"  Ag = {self.Ag:.0f} mm²,  ρ = {self.rho*100:.2f}%",
            f"  Total fibers: {fibers.n_fibers}"
            f"  (core: {mask_core.sum()}, cover: {mask_cover.sum()}, "
            f"steel: {mask_steel.sum()})",
            f"  Mesh area: {fibers.area[mask_core | mask_cover].sum():.0f} mm²"
            f"  (analytical: {self.Ag:.0f})",
        ]
        return "\n".join(lines)

    # ── private: Gmsh fiber extraction ──────────────────────────────
    @staticmethod
    def _extract_pg_fibers(
        pg_name: str,
        node_map: dict[int, ndarray],
        gmsh_mod,
    ) -> tuple[ndarray, ndarray, ndarray]:
        """Extract (y, z, area) arrays from a named Gmsh physical group."""
        pg_tag = None
        for dim, tag in gmsh_mod.model.getPhysicalGroups(dim=2):
            if gmsh_mod.model.getPhysicalName(dim, tag) == pg_name:
                pg_tag = tag
                break
        if pg_tag is None:
            raise RuntimeError(f"Physical group '{pg_name}' not found in Gmsh model")

        entities = gmsh_mod.model.getEntitiesForPhysicalGroup(2, pg_tag)

        ys: list[float] = []
        zs: list[float] = []
        As: list[float] = []

        for ent in entities:
            elem_types, elem_tags, elem_node_tags = (
                gmsh_mod.model.mesh.getElements(2, ent)
            )
            for etype, _, enodes in zip(elem_types, elem_tags, elem_node_tags):
                # getElementProperties → (name, dim, order, numNodes, paramCoord)
                npe = gmsh_mod.model.mesh.getElementProperties(etype)[3]
                enodes = enodes.reshape(-1, npe).astype(int)

                for elem_nodes in enodes:
                    coords = np.array([node_map[n] for n in elem_nodes])
                    ys.append(float(np.mean(coords[:, 0])))
                    zs.append(float(np.mean(coords[:, 1])))
                    # shoelace formula
                    nn = len(coords)
                    area = abs(sum(
                        coords[i, 0] * coords[(i + 1) % nn, 1]
                        - coords[(i + 1) % nn, 0] * coords[i, 1]
                        for i in range(nn)
                    )) / 2.0
                    As.append(area)

        return np.array(ys), np.array(zs), np.array(As)

    # ── private: rebar geometry ─────────────────────────────────────
    def _build_rebar_layers(self) -> None:
        """Compute bar positions for all rebar layers."""
        ci = self._ci
        b, h = self.b, self.h
        layers: list[_RebarLayer] = []

        # bottom bars
        n, d = self.bot_bars
        y_coords = np.linspace(-b/2 + ci, b/2 - ci, n)
        z_coords = np.full(n, -h/2 + ci)
        layers.append(_RebarLayer(n, d, y_coords, z_coords))

        # top bars
        n, d = self.top_bars
        y_coords = np.linspace(-b/2 + ci, b/2 - ci, n)
        z_coords = np.full(n, h/2 - ci)
        layers.append(_RebarLayer(n, d, y_coords, z_coords))

        # side bars (intermediate between top and bottom layers)
        if self.side_bars is not None:
            n_side, d_side = self.side_bars
            z_positions = np.linspace(-h/2 + ci, h/2 - ci, n_side + 2)[1:-1]
            for z in z_positions:
                # left side
                layers.append(_RebarLayer(1, d_side,
                                          np.array([-b/2 + ci]),
                                          np.array([z])))
                # right side
                layers.append(_RebarLayer(1, d_side,
                                          np.array([b/2 - ci]),
                                          np.array([z])))

        self._rebar_layers = layers

    def _rebar_arrays(self) -> tuple[ndarray, ndarray, ndarray]:
        """Flatten all rebar layers into (y, z, A) arrays."""
        if not self._rebar_layers:
            self._build_rebar_layers()

        ys, zs, As = [], [], []
        for layer in self._rebar_layers:
            for y, z in zip(layer.y_coords, layer.z_coords):
                ys.append(y)
                zs.append(z)
                As.append(layer.bar_area)

        return np.array(ys), np.array(zs), np.array(As)

    def _total_steel_area(self) -> float:
        """Total reinforcement area [mm²]."""
        if not self._rebar_layers:
            self._build_rebar_layers()
        return sum(
            layer.n_bars * layer.bar_area for layer in self._rebar_layers
        )

    # ── private: OpenSees material definitions ──────────────────────
    def _define_materials(self, ops, core_tag: int, cover_tag: int, steel_tag: int) -> None:
        """Register uniaxial materials in the active OpenSeesPy model."""
        fc  = self.fc
        ft  = self.ft
        Ec  = self.Ec
        fy  = self.fy
        Es  = self.Es
        cf  = self.confinement_factor
        eps_c0 = self.eps_c0

        if self.concrete_material == "Concrete02":
            # Confined concrete (core)
            fpc_conf   = -cf * fc
            epsc0_conf = -0.002 * (1.0 + 5.0 * (cf - 1.0))   # Scott et al.
            fpcu_conf  = -0.2 * fc
            epsU_conf  = -0.014
            lam        = 0.1
            Ets        = ft / 0.002

            ops.uniaxialMaterial(
                'Concrete02', core_tag,
                fpc_conf, epsc0_conf, fpcu_conf, epsU_conf,
                lam, ft, Ets,
            )

            # Unconfined concrete (cover)
            fpc_unconf   = -fc
            epsc0_unconf = -eps_c0
            fpcu_unconf  = 0.0      # cover spalls completely
            epsU_unconf  = -0.006

            ops.uniaxialMaterial(
                'Concrete02', cover_tag,
                fpc_unconf, epsc0_unconf, fpcu_unconf, epsU_unconf,
                lam, ft, Ets,
            )
        else:
            raise ValueError(
                f"Unsupported concrete material: {self.concrete_material!r}. "
                f"Currently only 'Concrete02' is implemented."
            )

        if self.steel_material == "Steel02":
            # Steel02: tag, Fy, E0, b, R0, cR1, cR2
            ops.uniaxialMaterial(
                'Steel02', steel_tag,
                fy, Es, 0.01,
                18.5, 0.925, 0.15,
            )
        else:
            raise ValueError(
                f"Unsupported steel material: {self.steel_material!r}. "
                f"Currently only 'Steel02' is implemented."
            )

    # ── private: helpers ────────────────────────────────────────────
    def _print_mesh_summary(self) -> None:
        fibers = self._fibers
        assert fibers is not None
        print(self.summary())

    def __repr__(self) -> str:
        return (
            f"RectangularColumnSection("
            f"b={self.b:.0f}, h={self.h:.0f}, "
            f"cover={self.cover:.0f}, "
            f"fc={self.fc:.0f}, fy={self.fy:.0f})"
        )
