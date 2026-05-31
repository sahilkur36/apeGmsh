"""
``_BeamIntegrationNS`` — backs ``ops.beamIntegration.<Type>(...)``.

Constructs and registers a typed :class:`BeamIntegration` primitive
for each OpenSees ``beamIntegration <Type>`` rule. The bridge
allocates a tag, the rule is referenced by tag from force-/disp-based
beam-column elements (see ADR 0011 / Phase 4.5).
"""
from __future__ import annotations

from ...integration import (
    HingeEndpoint,
    HingeMidpoint,
    HingeRadau,
    HingeRadauTwo,
    Legendre,
    Lobatto,
    NewtonCotes,
    Radau,
    Trapezoidal,
)
from ..types import Section
from ._base import _BridgeNamespace


__all__ = ["_BeamIntegrationNS"]


class _BeamIntegrationNS(_BridgeNamespace):
    """``ops.beamIntegration.<Type>(...)``.

    Each method constructs a typed :class:`BeamIntegration` rule,
    registers it with the bridge (allocating its tag), and returns
    the typed instance.
    """

    # -- Uniform-section quadrature rules -------------------------------

    def Lobatto(
        self, *, section: Section | str, n_ip: int, name: str | None = None
    ) -> Lobatto:
        """``beamIntegration Lobatto`` — Gauss-Lobatto, IPs include both ends."""
        section = self._bridge._resolve(section, base=Section)
        return self._bridge._register(
            Lobatto(section=section, n_ip=n_ip), name=name
        )

    def Legendre(
        self, *, section: Section | str, n_ip: int, name: str | None = None
    ) -> Legendre:
        """``beamIntegration Legendre`` — Gauss-Legendre, interior IPs only."""
        section = self._bridge._resolve(section, base=Section)
        return self._bridge._register(
            Legendre(section=section, n_ip=n_ip), name=name
        )

    def NewtonCotes(
        self, *, section: Section | str, n_ip: int, name: str | None = None
    ) -> NewtonCotes:
        """``beamIntegration NewtonCotes`` — closed Newton-Cotes."""
        section = self._bridge._resolve(section, base=Section)
        return self._bridge._register(
            NewtonCotes(section=section, n_ip=n_ip), name=name
        )

    def Radau(
        self, *, section: Section | str, n_ip: int, name: str | None = None
    ) -> Radau:
        """``beamIntegration Radau`` — Gauss-Radau."""
        section = self._bridge._resolve(section, base=Section)
        return self._bridge._register(
            Radau(section=section, n_ip=n_ip), name=name
        )

    def Trapezoidal(
        self, *, section: Section | str, n_ip: int, name: str | None = None
    ) -> Trapezoidal:
        """``beamIntegration Trapezoidal`` — composite trapezoidal."""
        section = self._bridge._resolve(section, base=Section)
        return self._bridge._register(
            Trapezoidal(section=section, n_ip=n_ip), name=name
        )

    # -- Concentrated-plasticity hinge rules ----------------------------

    def _resolve_hinge_sections(
        self,
        section_i: Section | str,
        section_j: Section | str,
        section_interior: Section | str,
    ) -> tuple[Section, Section, Section]:
        """Resolve the three hinge-section references (handle or name)."""
        r = self._bridge._resolve
        return (
            r(section_i, base=Section),
            r(section_j, base=Section),
            r(section_interior, base=Section),
        )

    def HingeRadau(
        self, *,
        section_i: Section | str, lp_i: float,
        section_j: Section | str, lp_j: float,
        section_interior: Section | str,
        name: str | None = None,
    ) -> HingeRadau:
        """``beamIntegration HingeRadau`` — 2-point Radau in each hinge."""
        section_i, section_j, section_interior = self._resolve_hinge_sections(
            section_i, section_j, section_interior
        )
        return self._bridge._register(
            HingeRadau(
                section_i=section_i, lp_i=lp_i,
                section_j=section_j, lp_j=lp_j,
                section_interior=section_interior,
            ),
            name=name,
        )

    def HingeRadauTwo(
        self, *,
        section_i: Section | str, lp_i: float,
        section_j: Section | str, lp_j: float,
        section_interior: Section | str,
        name: str | None = None,
    ) -> HingeRadauTwo:
        """``beamIntegration HingeRadauTwo`` — endpoint-anchored Radau."""
        section_i, section_j, section_interior = self._resolve_hinge_sections(
            section_i, section_j, section_interior
        )
        return self._bridge._register(
            HingeRadauTwo(
                section_i=section_i, lp_i=lp_i,
                section_j=section_j, lp_j=lp_j,
                section_interior=section_interior,
            ),
            name=name,
        )

    def HingeMidpoint(
        self, *,
        section_i: Section | str, lp_i: float,
        section_j: Section | str, lp_j: float,
        section_interior: Section | str,
        name: str | None = None,
    ) -> HingeMidpoint:
        """``beamIntegration HingeMidpoint`` — 1-point at hinge midpoint."""
        section_i, section_j, section_interior = self._resolve_hinge_sections(
            section_i, section_j, section_interior
        )
        return self._bridge._register(
            HingeMidpoint(
                section_i=section_i, lp_i=lp_i,
                section_j=section_j, lp_j=lp_j,
                section_interior=section_interior,
            ),
            name=name,
        )

    def HingeEndpoint(
        self, *,
        section_i: Section | str, lp_i: float,
        section_j: Section | str, lp_j: float,
        section_interior: Section | str,
        name: str | None = None,
    ) -> HingeEndpoint:
        """``beamIntegration HingeEndpoint`` — 1-point at element end."""
        section_i, section_j, section_interior = self._resolve_hinge_sections(
            section_i, section_j, section_interior
        )
        return self._bridge._register(
            HingeEndpoint(
                section_i=section_i, lp_i=lp_i,
                section_j=section_j, lp_j=lp_j,
                section_interior=section_interior,
            ),
            name=name,
        )
