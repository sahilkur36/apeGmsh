"""
``g.rebar`` — the L2 reinforcement-cage authoring composite (ADR 0066).

Sits **above** the shipped ``g.reinforce`` binding composite: it owns the
L1 spec objects (:mod:`apeGmsh._kernel.defs.rebar`) + geometry generation
+ standardized-member generators, and **delegates** coupling —
*conformal* via ``g.mesh.editing.embed`` (this module, P1) and *embedded*
via ``g.reinforce`` (P2). It never emits an OpenSees element itself.

P1 scope: ``bar`` / ``stirrup`` / ``stirrup_rect`` spec emitters, eager
**polyline** geometry emission (``true_arc`` is deferred to P3), and
``place(cage, into, coupling="conformal")`` which embeds the bar curves
into the host solid before meshing so the host mesh conforms and the bars
share its nodes (perfect bond — the ``ladruno_rc.py`` behaviour
generalised off the grid).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterable

import gmsh

from .._kernel.defs.rebar import METADATA, Bar, Cage, Hook, Path, Stirrup, Vec3
from ._compose_errors import chain_phase_guard
from ._helpers import resolve_to_tags

if TYPE_CHECKING:
    from .._core import _ApeGmshSession


# ── resolution-side records (not L1 specs) ───────────────────────────

@dataclass(frozen=True)
class RebarMember:
    """A placed bar/stirrup: the curve physical group + everything the
    bridge needs to realise a Truss/CorotTruss/DispBeamColumn on it
    (diameter + area resolved for BOTH couplings)."""
    pg: str
    role: str
    db: float | str
    diameter: float
    area: float
    material: str
    element: str
    coupling: str
    line_tags: tuple[int, ...]


@dataclass(frozen=True)
class RebarPlacement:
    """The record of one ``place()`` call."""
    name: str
    host: str
    coupling: str
    members: tuple[RebarMember, ...]


# ── the composite ────────────────────────────────────────────────────

class RebarComposite:
    """``g.rebar`` — reinforcement-cage authoring (ADR 0066)."""

    def __init__(self, parent: "_ApeGmshSession") -> None:
        self._parent = parent
        self._standard: Any = None
        self._place_seq = 0          # per-session counter → unique default PG base
        self.placements: list[RebarPlacement] = []

    # ---- detailing standard (used at resolve time, P3) --------------
    def use_standard(self, standard: Any) -> None:
        """Set the default :class:`DetailingStandard` for this session's
        cages (resolves ``"<k>db"`` tokens + hook factories at bind)."""
        self._standard = standard

    # ---- L1 spec emitters (thin) ------------------------------------
    def bar(self, points: Iterable[Vec3], *, db, material,
            role: str = "longitudinal", element: str = "truss",
            start_hook: Hook | None = None, end_hook: Hook | None = None,
            corner_radius=METADATA, name: str | None = None) -> Bar:
        return Bar(path=Path(tuple(points), corner_radius=corner_radius),
                   db=db, material=material, role=role, element=element,
                   start_hook=start_hook, end_hook=end_hook, name=name)

    def stirrup(self, points: Iterable[Vec3], *, db, material,
                closure_hook: Hook | None = None, role: str = "tie",
                corner_radius=METADATA, name: str | None = None) -> Stirrup:
        return Stirrup(path=Path(tuple(points), corner_radius=corner_radius),
                       db=db, material=material, role=role,
                       closure_hook=closure_hook or Hook.seismic_135(),
                       name=name)

    def stirrup_rect(self, bx: float, by: float, cover: float, *,
                     db, material, **kw) -> Stirrup:
        return Stirrup.rect(bx, by, cover, db=db, material=material, **kw)

    # ---- placement / coupling router --------------------------------
    def place(self, cage: Cage, into: str, *, coupling: str = "conformal",
              per_member_coupling: dict[str, str] | None = None,
              bond: str | None = None, perfect: float | None = None,
              kt=None, kt_alpha=None, enforce: str = "penalty",
              bipenalty: bool = False, dtcr=None, tolerance: float = 1.0e-6,
              snap: bool = False, host_dim: int | None = None,
              true_arc: bool = False, on_conformal_infeasible: str = "fail",
              name: str | None = None) -> RebarPlacement:
        """Emit the cage geometry and couple each member to host ``into``.

        ``coupling="conformal"`` embeds the bar curves into the host so the
        mesh conforms (shared nodes, perfect bond). ``coupling="embedded"``
        meshes the bars independently and forwards to ``g.reinforce`` (→
        ``LadrunoEmbeddedRebar``); it needs ``bond=`` (a ``LadrunoBondSlip``
        material name) **or** ``perfect=`` (a perfect-bond axial penalty).
        ``per_member_coupling={role: coupling}`` overrides per role for
        **mixed** cages (e.g. longitudinal conformal + ties embedded).
        """
        chain_phase_guard(self._parent, "g.rebar.place")
        if not isinstance(cage, Cage):
            raise TypeError(
                f"g.rebar.place: cage must be a Cage, got {type(cage).__name__}."
            )
        if true_arc:
            raise NotImplementedError(
                "g.rebar.place: true_arc fillet geometry is deferred to P3; "
                "use the polyline default (true_arc=False) for now."
            )
        if coupling not in ("conformal", "embedded"):
            raise ValueError(
                f"g.rebar.place: coupling must be 'conformal' or 'embedded', "
                f"got {coupling!r}."
            )
        if on_conformal_infeasible not in ("fail", "embedded"):
            raise ValueError(
                f"g.rebar.place: on_conformal_infeasible must be 'fail' or "
                f"'embedded', got {on_conformal_infeasible!r}."
            )
        pmc = per_member_coupling or {}
        std = cage.standard if cage.standard is not None else self._standard
        rein_kw = dict(bond=bond, perfect=perfect, kt=kt, kt_alpha=kt_alpha,
                       enforce=enforce, bipenalty=bipenalty, dtcr=dtcr,
                       tolerance=tolerance, snap=snap)
        # Pass 0 — validate EVERYTHING (cage + host) before mutating gmsh, so a
        # bad cage never leaves the model half-emitted.
        plan = self._plan(cage, into, default_coupling=coupling,
                          per_member_coupling=pmc, std=std, rein_kw=rein_kw,
                          on_conformal_infeasible=on_conformal_infeasible,
                          host_dim=host_dim, name=name)
        return self._emit_plan(plan, into, rein_kw=rein_kw,
                               on_conformal_infeasible=on_conformal_infeasible)

    # ---- Pass 0: validation + planning (no gmsh mutation) -----------
    def _plan(self, cage: Cage, into: str, *, default_coupling: str,
              per_member_coupling: dict[str, str], std, rein_kw: dict,
              on_conformal_infeasible: str, host_dim: int | None,
              name: str | None) -> dict:
        in_dim = host_dim if host_dim is not None else self._detect_host_dim(into)
        host_tags = resolve_to_tags(into, dim=in_dim, session=self._parent)
        base = name or f"rebar{self._place_seq}"

        planned: list = []
        roles_seen: set[str] = set()
        names_seen: set[str] = set()
        has_conf = has_emb = False
        idx = 0
        for default_role, items, is_stirrup in (
                ("longitudinal", cage.bars, False),
                ("tie", cage.stirrups, True)):
            for m in items:
                role = getattr(m, "role", default_role)
                roles_seen.add(role)
                eff = per_member_coupling.get(role, default_coupling)
                if eff not in ("conformal", "embedded"):
                    raise ValueError(
                        f"g.rebar.place: per_member_coupling[{role!r}]={eff!r} "
                        f"must be 'conformal' or 'embedded'."
                    )
                key = m.name or f"{role}_{idx}"
                if key in names_seen:
                    raise ValueError(
                        f"g.rebar.place: duplicate member identity {key!r}; "
                        f"member names must be unique within a cage."
                    )
                names_seen.add(key)
                pg = f"{base}.{key}"
                if self._is_physical_group(pg):
                    raise ValueError(
                        f"g.rebar.place: physical group {pg!r} already exists "
                        f"(name collision across placements); pass a distinct "
                        f"name= or member name."
                    )
                elem = getattr(m, "element", "truss")
                if elem == "beam" and (
                        len(m.path.points) > 2 or m.start_hook is not None
                        or m.end_hook is not None
                        or getattr(m, "closure_hook", None) is not None):
                    raise NotImplementedError(
                        "g.rebar: element='beam' on a curved/hooked bar needs "
                        "the ADR-0010 Phase-4 orientation fan-out (not yet "
                        "wired); use element='truss' or a straight bar."
                    )
                if is_stirrup:
                    pts = m.path.points
                    distinct = pts[:-1] if pts[0] == pts[-1] else pts
                    if len(set(distinct)) < 3:
                        raise ValueError(
                            f"g.rebar: stirrup {key!r} closed loop needs ≥3 "
                            f"distinct corners, got {len(set(distinct))}."
                        )
                if eff == "embedded":
                    self._check_embedded_args(rein_kw["bond"], rein_kw["perfect"],
                                              member=key)
                    has_emb = True
                else:
                    has_conf = True
                planned.append((role, eff, m, pg, elem,
                                self._dia(std, m.db), self._area(std, m.db)))
                idx += 1

        for k in per_member_coupling:
            if k not in roles_seen:
                warnings.warn(
                    f"g.rebar.place: per_member_coupling key {k!r} matches no "
                    f"member role {sorted(roles_seen)}; ignored.", stacklevel=3)

        host_tag = host_tags[0] if host_tags else None
        if has_conf:
            if len(host_tags) != 1:
                raise ValueError(
                    f"g.rebar.place: conformal coupling needs a single host "
                    f"volume; {into!r} resolved to {len(host_tags)} entities. "
                    f"Name one volume or use coupling='embedded'."
                )
            if self._host_is_meshed(in_dim, host_tag):
                raise RuntimeError(
                    "g.rebar.place: conformal coupling must run BEFORE "
                    "g.mesh.generation.generate() — embedding into an already-"
                    "meshed host is a silent no-op."
                )
            self._reject_foreign_part(into)
            if on_conformal_infeasible == "embedded":
                self._check_embedded_args(rein_kw["bond"], rein_kw["perfect"],
                                          member="conformal-fallback")
        if has_emb:
            if not self._is_physical_group(into):
                raise ValueError(
                    f"g.rebar.place: embedded coupling needs host {into!r} to "
                    f"be a physical group (e.g. g.physical.add_volume(...)); a "
                    f"bare geometry label is not resolvable by g.reinforce."
                )
            warnings.warn(
                "g.rebar.place: embedded coupling uses LadrunoEmbeddedRebar, "
                "which is single-process today; partitioned/MPI models must "
                "use coupling='conformal'.", stacklevel=3)

        self._place_seq += 1
        return dict(base=base, in_dim=in_dim, host_tag=host_tag, planned=planned)

    # ---- emit (mutates gmsh; all inputs pre-validated) --------------
    def _emit_plan(self, plan: dict, into: str, *, rein_kw: dict,
                   on_conformal_infeasible: str) -> RebarPlacement:
        g = self._parent
        geom = g.model.geometry
        base, in_dim, host_tag = plan["base"], plan["in_dim"], plan["host_tag"]

        # Pass 1 — emit all curve geometry (no PGs yet).
        emitted: list = []
        for role, eff, m, pg, elem, dia, area in plan["planned"]:
            lts = self._emit_polyline(geom, m.path.points)
            emitted.append((role, eff, m, pg, elem, dia, area, lts))
        # Sync once so the curve entities exist before we wrap them in PGs.
        g.model.sync()

        # Pass 2 — physical groups + coupling registration.
        members: list[RebarMember] = []
        conformal_tags: list[int] = []
        conformal_specs: list = []
        for role, eff, m, pg, elem, dia, area, lts in emitted:
            g.physical.add_curve(lts, name=pg)
            member = RebarMember(
                pg=pg, role=role, db=m.db, diameter=dia, area=area,
                material=m.material, element=elem, coupling=eff,
                line_tags=tuple(lts),
            )
            members.append(member)
            if eff == "conformal":
                conformal_tags.extend(lts)
                conformal_specs.append((member, dia, area))
            else:
                self._register_embedded(into, pg, dia, area, **rein_kw)

        if conformal_tags:
            try:
                g.mesh.editing.embed(conformal_tags, host_tag, dim=1, in_dim=in_dim)
            except Exception as exc:                       # embed-time failure
                if on_conformal_infeasible != "embedded":
                    raise
                warnings.warn(
                    f"g.rebar.place: conformal embed failed ({exc}); falling "
                    f"back to embedded coupling for {len(conformal_specs)} "
                    f"member(s).", stacklevel=2,
                )
                members = [mm if mm.coupling == "embedded"
                           else replace(mm, coupling="embedded")
                           for mm in members]
                for member, dia, area in conformal_specs:
                    self._register_embedded(into, member.pg, dia, area, **rein_kw)

        couplings = {mm.coupling for mm in members}
        placement = RebarPlacement(
            name=base, host=into,
            coupling=next(iter(couplings)) if len(couplings) == 1 else "mixed",
            members=tuple(members),
        )
        self.placements.append(placement)
        return placement

    def _register_embedded(self, into: str, pg: str, diameter: float,
                           area: float, *, bond, perfect, kt, kt_alpha,
                           enforce, bipenalty, dtcr, tolerance, snap) -> None:
        """Forward one embedded member to the shipped ``g.reinforce`` binding
        composite (→ ``LadrunoEmbeddedRebar``), then invalidate the FEMData
        cache (ADR §9: a def-append is a broker mutation)."""
        self._parent.reinforce.reinforce(
            host=into, bars=pg, bond=bond, perfect=perfect,
            bar_diameter=diameter, bar_area=area,
            kt=kt, kt_alpha=kt_alpha, enforce=enforce, bipenalty=bipenalty,
            dtcr=dtcr, tolerance=tolerance, snap=snap, name=pg,
        )
        bump = getattr(self._parent, "_bump_fem_counter", None)
        if bump is not None:
            bump()

    # ---- small resolvers / host checks ------------------------------
    @staticmethod
    def _check_embedded_args(bond, perfect, *, member: str) -> None:
        if (bond is None) == (perfect is None):
            raise ValueError(
                f"g.rebar.place: embedded coupling for {member!r} needs "
                f"exactly one of bond=<LadrunoBondSlip name> or "
                f"perfect=<axial penalty>."
            )

    @staticmethod
    def _dia(std, db) -> float:
        if isinstance(db, (int, float)) and not isinstance(db, bool):
            return float(db)
        if std is not None:
            return float(std.bar_diameter(db))
        raise ValueError(
            f"g.rebar: db {db!r} is a designation but no DetailingStandard is "
            f"set; pass a numeric db, a Cage(standard=...), or call "
            f"g.rebar.use_standard(ACI318())."
        )

    @staticmethod
    def _area(std, db) -> float:
        if isinstance(db, (int, float)) and not isinstance(db, bool):
            return math.pi * float(db) ** 2 / 4.0
        if std is not None:
            return float(std.bar_area(db))
        raise ValueError(
            f"g.rebar: db {db!r} is a designation but no DetailingStandard is "
            f"set; pass a numeric db, a Cage(standard=...), or call "
            f"g.rebar.use_standard(ACI318())."
        )

    @staticmethod
    def _is_physical_group(name: str) -> bool:
        for d, t in gmsh.model.getPhysicalGroups():
            try:
                if gmsh.model.getPhysicalName(int(d), int(t)) == name:
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _host_is_meshed(in_dim: int, host_tag) -> bool:
        if host_tag is None:
            return False
        try:
            _types, etags, _ = gmsh.model.mesh.getElements(in_dim, host_tag)
            return any(len(t) > 0 for t in etags)
        except Exception:
            return False

    def _reject_foreign_part(self, into: str) -> None:
        parts = getattr(self._parent, "parts", None)
        try:
            labels = parts.labels() if parts is not None else []
        except Exception:
            labels = []
        if into in labels:
            raise ValueError(
                f"g.rebar.place: conformal coupling requires same-session "
                f"authoring, but host {into!r} is a composed Part — use "
                f"coupling='embedded' (ADR 0066 §6.4)."
            )

    # ---- geometry helpers -------------------------------------------
    def _emit_polyline(self, geom, points: tuple[Vec3, ...]) -> list[int]:
        """Emit a polyline as gmsh points + line segments, returning the
        line tags. A closed loop (first == last) reuses the first point so
        the loop welds into one node ring."""
        closed = len(points) >= 2 and points[0] == points[-1]
        pt_tags: list[int] = []
        first_tag: int | None = None
        n = len(points)
        for i, p in enumerate(points):
            if closed and i == n - 1 and first_tag is not None:
                pt_tags.append(first_tag)
            else:
                t = geom.add_point(p[0], p[1], p[2], sync=False)
                if i == 0:
                    first_tag = t
                pt_tags.append(t)
        return [geom.add_line(pt_tags[i], pt_tags[i + 1], sync=False)
                for i in range(len(pt_tags) - 1)]

    def _detect_host_dim(self, into: str) -> int:
        """Resolve the host's dimension (3D solid preferred, then 2D)."""
        for d in (3, 2):
            try:
                if resolve_to_tags(into, dim=d, session=self._parent):
                    return d
            except Exception:
                continue
        raise ValueError(
            f"g.rebar.place: cannot resolve host {into!r} as a 3-D or 2-D "
            f"entity. Pass host_dim= explicitly or check the label."
        )

    # validate hook — resolution at get_fem_data (P3); nothing pre-mesh yet
    def validate_pre_mesh(self) -> None:
        return None
