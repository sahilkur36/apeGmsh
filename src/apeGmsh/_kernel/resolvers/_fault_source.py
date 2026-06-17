"""Finite-fault → per-subfault moment-tensor source specs (MT-4, ADR 0062).

A finite fault is a loop of point moment-tensor sources, each with its
own location, scalar moment ``M0 = μ·A·D̄``, mechanism, and **rupture
onset** ``t0``. This module is the pure converter: it turns the aligned
arrays a ShakerMaker ``FFSPSource.get_subfaults()`` produces (plus a plain
crust description) into a list of :class:`FaultSubsource` specs the bridge
``ops.fault`` namespace replays as one ``Plain`` pattern per subfault.

Per ADR 0062 decision 8 this imports **nothing** from ShakerMaker — the
interchange is plain numpy arrays + a dict crust. Per decision 6 the unit
conversions are **explicit and required** (no silent defaults): the caller
declares ``length_scale`` (source-length → deck length) and
``moment_scale`` (N·m → deck moment); the crust is given in SI
(``vs`` m/s, ``rho`` kg/m³, ``thickness`` m), and ``area``/``slip`` in SI
(m², m), so ``μ = ρ·Vs²`` [Pa] and ``M0 = μ·A·D̄`` [N·m] before scaling.

Source coordinates are assumed **km** (the ShakerMaker/FFSP convention)
with ``z`` the depth *below the free surface* (positive down). The mesh
``frame`` (``"z-up"`` / ``"z-down"``) sets the sign of the deck ``z``.

Pure NumPy — no Gmsh, no OpenSees, no ShakerMaker imports.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import ndarray


__all__ = [
    "FaultSubsource",
    "shear_modulus_at_depth",
    "ffsp_subfaults_to_sources",
]

#: km → m (source coords are km; the crust is SI, so depths convert here).
_KM_TO_M = 1000.0


@dataclass(frozen=True, slots=True)
class FaultSubsource:
    """One point moment-tensor source of a finite fault.

    ``position`` is in deck length units; ``M0`` in deck moment units;
    ``strike``/``dip``/``rake`` in degrees; ``t0`` the rupture onset (s);
    ``rise_time``/``peak_time`` the moment-function shape (s).
    """

    position: tuple[float, float, float]
    M0: float
    strike: float
    dip: float
    rake: float
    t0: float
    rise_time: float
    peak_time: float


def shear_modulus_at_depth(
    depth_m: float,
    *,
    thickness_m: ndarray,
    vs_ms: ndarray,
    rho_kgm3: ndarray,
) -> float:
    """Layered shear modulus ``μ = ρ·Vs²`` [Pa] at ``depth_m`` (metres).

    The crust is a stack of layers (top → bottom) with per-layer
    ``thickness_m`` / ``vs_ms`` / ``rho_kgm3``; the deepest layer is the
    half-space (its thickness is ignored — any depth at or below the last
    interface uses it). ``depth_m`` is taken as a non-negative depth below
    the free surface.
    """
    thickness = np.asarray(thickness_m, dtype=float)
    vs = np.asarray(vs_ms, dtype=float)
    rho = np.asarray(rho_kgm3, dtype=float)
    n = len(vs)
    if not (len(thickness) == len(rho) == n) or n == 0:
        raise ValueError(
            "shear_modulus_at_depth: thickness/vs/rho must be equal-length "
            f"non-empty arrays (got {len(thickness)}, {n}, {len(rho)})."
        )
    d = abs(float(depth_m))
    top = 0.0
    for i in range(n):
        bottom = top + thickness[i]
        if i == n - 1 or d < bottom:
            return float(rho[i] * vs[i] * vs[i])
        top = bottom
    # Unreachable (the i==n-1 branch returns), but keep mypy + the linter happy.
    return float(rho[-1] * vs[-1] * vs[-1])


def _deck_position(
    x_km: float, y_km: float, z_km: float, *, length_scale: float, frame: str,
) -> tuple[float, float, float]:
    """Map a (km, depth-down) source point to deck coordinates.

    ``length_scale`` is deck-length per km. For a ``"z-up"`` mesh (free
    surface at ``z = 0``, depth positive downward) the depth becomes a
    negative deck ``z``; ``"z-down"`` keeps it positive.
    """
    if frame not in ("z-up", "z-down"):
        raise ValueError(
            f"_deck_position: frame must be 'z-up' or 'z-down', got {frame!r}."
        )
    zx = float(z_km) * length_scale
    z_deck = -zx if frame == "z-up" else zx
    return (float(x_km) * length_scale, float(y_km) * length_scale, z_deck)


def ffsp_subfaults_to_sources(
    subfaults: dict,
    *,
    crust: dict,
    area_m2: float,
    length_scale: float,
    moment_scale: float,
    frame: str,
) -> list[FaultSubsource]:
    """Convert ``FFSPSource.get_subfaults()`` arrays into source specs.

    ``subfaults`` carries the aligned arrays ``x, y, z, slip, strike, dip,
    rake, rupture_time, rise_time, peak_time`` (each length ``npts``); ``x,
    y, z`` in km (``z`` = depth, positive down), ``slip`` in **metres**.
    ``crust`` is ``{"thickness": [...m], "vs": [...m/s], "rho": [...kg/m³]}``.
    ``area_m2`` is the per-subfault rupture area (m²). The local shear
    modulus ``μ(z)`` gives ``M0 = μ·area·slip`` [N·m], then ``× moment_scale``
    for the deck. Positions scale by ``length_scale`` with the ``frame``
    z-sign.
    """
    required = (
        "x", "y", "z", "slip", "strike", "dip", "rake",
        "rupture_time", "rise_time", "peak_time",
    )
    missing = [k for k in required if k not in subfaults]
    if missing:
        raise ValueError(
            f"ffsp_subfaults_to_sources: subfaults dict missing keys "
            f"{missing} (need {list(required)})."
        )
    if area_m2 <= 0:
        raise ValueError(
            f"ffsp_subfaults_to_sources: area_m2 must be > 0, got {area_m2!r}."
        )
    thickness = np.asarray(crust["thickness"], dtype=float)
    vs = np.asarray(crust["vs"], dtype=float)
    rho = np.asarray(crust["rho"], dtype=float)

    arrs = {k: np.asarray(subfaults[k], dtype=float).ravel() for k in required}
    n = len(arrs["x"])
    if any(len(arrs[k]) != n for k in required):
        raise ValueError(
            "ffsp_subfaults_to_sources: subfault arrays are not aligned "
            f"(lengths {{k: len}} = {{{', '.join(f'{k}: {len(arrs[k])}' for k in required)}}})."
        )

    out: list[FaultSubsource] = []
    for i in range(n):
        depth_m = abs(float(arrs["z"][i])) * _KM_TO_M
        mu = shear_modulus_at_depth(
            depth_m, thickness_m=thickness, vs_ms=vs, rho_kgm3=rho,
        )
        m0_si = mu * area_m2 * float(arrs["slip"][i])
        pos = _deck_position(
            arrs["x"][i], arrs["y"][i], arrs["z"][i],
            length_scale=length_scale, frame=frame,
        )
        out.append(FaultSubsource(
            position=pos,
            M0=m0_si * moment_scale,
            strike=float(arrs["strike"][i]),
            dip=float(arrs["dip"][i]),
            rake=float(arrs["rake"][i]),
            t0=float(arrs["rupture_time"][i]),
            rise_time=float(arrs["rise_time"][i]),
            peak_time=float(arrs["peak_time"][i]),
        ))
    return out
