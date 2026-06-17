"""Seismic moment-tensor math — the pure core of ADR 0062 (MT-1).

A seismic point source is a **moment tensor** ``M_ij``. By the
representation theorem (Aki & Richards 2002, Ch. 3) its radiated field
equals that of an equivalent body force; projected onto the FE basis the
consistent nodal force on host node ``a`` is ``F^a_i = M_ij ∂N_a/∂x_j``.
This module owns the *tensor* half of that: building the unit
double-couple ``m_ij`` from a fault mechanism, scaling it by the scalar
moment ``M0``, and mapping the geophysics frame to the mesh frame. The
*force* half (``∂N/∂x`` projection) lives in
:mod:`apeGmsh._kernel.resolvers._moment_tensor`.

Frame convention
----------------
The unit tensor is built in the **Aki & Richards geographic frame**:
``x = North``, ``y = East``, ``z = Down`` — the same convention as
ShakerMaker's ``core/radiats.c`` (``nmtensor``), verified term-for-term.
The unit double-couple (A&R Eq. 4.83–4.88; ``φ`` strike, ``δ`` dip,
``λ`` rake, all degrees):

.. math::

    m_{xx} &= -(\\sin\\delta\\cos\\lambda\\sin2\\phi
               + \\sin2\\delta\\sin\\lambda\\sin^2\\phi)\\\\
    m_{yy} &=   \\sin\\delta\\cos\\lambda\\sin2\\phi
               - \\sin2\\delta\\sin\\lambda\\cos^2\\phi\\\\
    m_{zz} &=   \\sin2\\delta\\sin\\lambda = -(m_{xx}+m_{yy})\\\\
    m_{xy} &=   \\sin\\delta\\cos\\lambda\\cos2\\phi
               + \\tfrac12\\sin2\\delta\\sin\\lambda\\sin2\\phi\\\\
    m_{xz} &= -(\\cos\\delta\\cos\\lambda\\cos\\phi
               + \\cos2\\delta\\sin\\lambda\\sin\\phi)\\\\
    m_{yz} &= -(\\cos\\delta\\cos\\lambda\\sin\\phi
               - \\cos2\\delta\\sin\\lambda\\cos\\phi)

apeGmsh meshes are usually **z-up** (depth is a positive number *below*
the free surface), so :func:`to_mesh_frame` flips the ``z`` axis —
negating ``m_xz`` and ``m_yz`` while leaving the in-plane and ``m_zz``
components unchanged. Getting this wrong silently mirrors the radiation
pattern, so the high-level :func:`moment_tensor` requires an explicit
``frame``.

Units
-----
This module is **unit-neutral**. :func:`scalar_moment` returns
``mu * area * slip`` in whatever (consistent) units the caller passes.
:func:`magnitude_to_scalar_moment` follows the Hanks & Kanamori (1979)
SI convention ``M0 [N·m] = 10**(1.5*Mw + 9.1)`` — converting to the deck's
``kN·m`` is the ShakerMaker adapter's job (MT-4), not this layer's.

Pure NumPy — no Gmsh, no OpenSees imports.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray


__all__ = [
    "unit_moment_tensor",
    "scalar_moment",
    "magnitude_to_scalar_moment",
    "scalar_moment_to_magnitude",
    "to_mesh_frame",
    "moment_tensor",
    "MESH_FRAMES",
]

#: Accepted ``frame`` values for :func:`moment_tensor` /
#: :func:`to_mesh_frame`. ``"z-up"`` is the typical apeGmsh continuum
#: convention (free surface at the top, depth positive downward) and
#: flips the A&R ``z``-down tensor; ``"z-down"`` keeps the A&R frame.
MESH_FRAMES = ("z-up", "z-down")


def unit_moment_tensor(*, strike: float, dip: float, rake: float) -> ndarray:
    """Unit double-couple ``m_ij`` (3×3) in the A&R z-**down** frame.

    ``strike`` (``φ``), ``dip`` (``δ``) and ``rake`` (``λ``) are in
    **degrees**. The returned tensor is symmetric, trace-free (a pure
    deviatoric double-couple), and **not** scaled by the scalar moment —
    multiply by ``M0`` (see :func:`moment_tensor`).

    The component layout is the standard ``(x=N, y=E, z=Down)`` Cartesian
    grid; ``m[0,2]`` is ``m_xz`` etc.
    """
    phi = np.radians(strike)
    delta = np.radians(dip)
    lam = np.radians(rake)

    sphi, cphi = np.sin(phi), np.cos(phi)
    s2phi, c2phi = np.sin(2 * phi), np.cos(2 * phi)
    sdel, cdel = np.sin(delta), np.cos(delta)
    s2del, c2del = np.sin(2 * delta), np.cos(2 * delta)
    slam, clam = np.sin(lam), np.cos(lam)

    m_xx = -(sdel * clam * s2phi + s2del * slam * sphi * sphi)
    m_yy = sdel * clam * s2phi - s2del * slam * cphi * cphi
    m_zz = s2del * slam
    m_xy = sdel * clam * c2phi + 0.5 * s2del * slam * s2phi
    m_xz = -(cdel * clam * cphi + c2del * slam * sphi)
    m_yz = -(cdel * clam * sphi - c2del * slam * cphi)

    return np.array(
        [
            [m_xx, m_xy, m_xz],
            [m_xy, m_yy, m_yz],
            [m_xz, m_yz, m_zz],
        ],
        dtype=float,
    )


def scalar_moment(*, mu: float, area: float, slip: float) -> float:
    """Scalar seismic moment ``M0 = μ·A·D̄``.

    ``mu`` is the (local) shear modulus, ``area`` the ruptured area, and
    ``slip`` the average slip. Unit-neutral: the result is in whatever
    consistent units the three inputs share (e.g. Pa·m²·m = N·m).
    """
    if mu <= 0 or area <= 0 or slip <= 0:
        raise ValueError(
            "scalar_moment: mu, area and slip must all be > 0 "
            f"(got mu={mu!r}, area={area!r}, slip={slip!r})."
        )
    return float(mu) * float(area) * float(slip)


def magnitude_to_scalar_moment(mw: float) -> float:
    """Moment magnitude ``Mw`` → scalar moment ``M0`` in **N·m**.

    Hanks & Kanamori (1979), SI form: ``M0 = 10**(1.5*Mw + 9.1)``.
    """
    return float(10.0 ** (1.5 * float(mw) + 9.1))


def scalar_moment_to_magnitude(m0: float) -> float:
    """Scalar moment ``M0`` (**N·m**) → moment magnitude ``Mw``.

    Inverse of :func:`magnitude_to_scalar_moment`:
    ``Mw = (log10(M0) - 9.1) / 1.5``.
    """
    if m0 <= 0:
        raise ValueError(
            f"scalar_moment_to_magnitude: M0 must be > 0 (got {m0!r})."
        )
    return float((np.log10(m0) - 9.1) / 1.5)


def to_mesh_frame(m: ndarray, *, frame: str) -> ndarray:
    """Map an A&R z-**down** tensor onto the mesh frame.

    ``frame="z-up"`` flips the vertical axis (negating ``m_xz`` and
    ``m_yz``; ``m_zz`` and the in-plane components are invariant under
    ``z → -z``). ``frame="z-down"`` returns a copy unchanged. Any other
    value fails loud — the flip is the #1 footgun (it silently mirrors
    the radiation pattern), so there is no default.
    """
    if frame not in MESH_FRAMES:
        raise ValueError(
            f"to_mesh_frame: frame must be one of {MESH_FRAMES}, got "
            f"{frame!r}. 'z-up' (typical apeGmsh: free surface on top, "
            f"depth positive downward) flips the A&R z-down tensor; "
            f"'z-down' keeps it. There is no default — the flip mirrors "
            f"the radiation pattern if wrong."
        )
    out = np.array(m, dtype=float, copy=True)
    if frame == "z-up":
        # z -> -z : negate the single-z components, leave m_zz / in-plane.
        out[0, 2] = -out[0, 2]
        out[2, 0] = -out[2, 0]
        out[1, 2] = -out[1, 2]
        out[2, 1] = -out[2, 1]
    return out


def moment_tensor(
    *,
    frame: str,
    M0: float = 1.0,
    strike: float | None = None,
    dip: float | None = None,
    rake: float | None = None,
    m_ij: ndarray | None = None,
) -> ndarray:
    """Full moment tensor ``M = M0 · m`` (3×3) in the **mesh** frame.

    Supply *either* a fault mechanism (``strike`` + ``dip`` + ``rake``,
    degrees) *or* a pre-built unit tensor ``m_ij`` (3×3, A&R z-down) —
    never both. ``M0`` is the scalar moment (default 1.0 → the scaled
    unit tensor). ``frame`` (required) selects the mesh vertical
    convention; see :func:`to_mesh_frame`.

    The result is symmetric and, for a double-couple, trace-free.
    """
    has_mech = strike is not None or dip is not None or rake is not None
    if has_mech == (m_ij is not None):
        raise ValueError(
            "moment_tensor: supply exactly one of a mechanism "
            "(strike+dip+rake) or m_ij (a unit tensor), not both / neither."
        )
    if has_mech:
        if strike is None or dip is None or rake is None:
            raise ValueError(
                "moment_tensor: a mechanism needs all of strike, dip, rake "
                f"(got strike={strike!r}, dip={dip!r}, rake={rake!r})."
            )
        m = unit_moment_tensor(strike=strike, dip=dip, rake=rake)
    else:
        assert m_ij is not None  # validated above
        m = np.asarray(m_ij, dtype=float)
        if m.shape != (3, 3):
            raise ValueError(
                f"moment_tensor: m_ij must be a 3x3 array, got shape {m.shape}."
            )
        if not np.allclose(m, m.T, atol=1e-12):
            raise ValueError(
                "moment_tensor: m_ij must be symmetric (a physical moment "
                "tensor has no net torque)."
            )
    return to_mesh_frame(float(M0) * m, frame=frame)
