"""``_FaultNS`` — backs ``ops.fault.from_ffsp`` / ``from_shakermaker`` (MT-4).

A finite fault is a loop of point moment-tensor sources, each with its own
location, scalar moment, mechanism, and rupture onset ``t0``. Because
OpenSees binds one ``timeSeries`` per pattern, each subfault becomes **one
``Plain`` pattern**: a ``Yoffe`` moment function shifted to the subfault's
rupture time (``t0``) carrying a single ``moment_tensor`` source (whose own
``t0`` stays 0 — the delay lives in the series). Both methods return the
list of created patterns.

Per ADR 0062 decision 8 this imports nothing from ShakerMaker: ``from_ffsp``
consumes the plain ``get_subfaults()`` dict + a dict crust; ``from_shakermaker``
duck-types the ``FaultSource`` (reads ``.x`` / ``.angles`` / ``.tt`` — note
``.angles`` is in **radians**). Per decision 6 the unit scales are required.
"""
from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from apeGmsh._kernel.resolvers._fault_source import (
    FaultSubsource,
    _deck_position,
    ffsp_subfaults_to_sources,
)

from ._base import _BridgeNamespace

if TYPE_CHECKING:
    from ...pattern.pattern import Plain


__all__ = ["_FaultNS", "WarnFaultSubfaultSkipped"]


class WarnFaultSubfaultSkipped(UserWarning):
    """One or more subfaults were skipped (non-positive moment / rise /
    peak time) when building moment-tensor patterns from a fault."""


def _per_source(value: "float | Sequence[float]", i: int, n: int) -> float:
    """Broadcast a scalar, or index a length-``n`` sequence, per subfault."""
    if np.isscalar(value):
        return float(value)  # type: ignore[arg-type]
    seq = np.asarray(value, dtype=float).ravel()
    if len(seq) != n:
        raise ValueError(
            f"from_shakermaker: per-source value has length {len(seq)} but "
            f"the fault has {n} sources."
        )
    return float(seq[i])


class _FaultNS(_BridgeNamespace):
    """``ops.fault.<from_*>(...)`` — finite-fault moment-tensor sources."""

    def from_ffsp(
        self,
        subfaults: "dict[str, Any]",
        crust: "dict[str, Any]",
        *,
        frame: str,
        area_m2: float,
        dt: float,
        t_total: float,
        length_scale: float,
        moment_scale: float,
        method: str = "consistent",
        f_max: float | None = None,
    ) -> "list[Plain]":
        """Build one ``Plain`` pattern per subfault from FFSP arrays.

        ``subfaults`` is ``FFSPSource.get_subfaults()`` (aligned arrays;
        coords km, ``z`` depth-down, ``slip`` m). ``crust`` is
        ``{"thickness": [m], "vs": [m/s], "rho": [kg/m³]}`` → ``μ = ρVs²``.
        ``area_m2`` is the per-subfault rupture area. ``length_scale``
        (deck-length per km) and ``moment_scale`` (deck-moment per N·m) are
        the **required** unit conversions (ADR 0062 decision 6). Each
        subfault's ``M0 = μ·area·slip`` drives a ``Yoffe`` moment function
        (``rise_time``/``peak_time``, onset ``t0 = rupture_time``).

        Subfaults with non-positive moment / rise / peak (taper edges) are
        skipped with a :class:`WarnFaultSubfaultSkipped`.
        """
        specs = ffsp_subfaults_to_sources(
            subfaults, crust=crust, area_m2=area_m2,
            length_scale=length_scale, moment_scale=moment_scale, frame=frame,
        )
        return self._build_patterns(
            specs, frame=frame, dt=dt, t_total=t_total,
            method=method, f_max=f_max,
        )

    def from_shakermaker(
        self,
        fault: Any,
        *,
        frame: str,
        M0: "float | Sequence[float]",
        rise_time: "float | Sequence[float]",
        peak_time: "float | Sequence[float]",
        dt: float,
        t_total: float,
        length_scale: float,
        method: str = "consistent",
        f_max: float | None = None,
    ) -> "list[Plain]":
        """Build one ``Plain`` pattern per ``PointSource`` of a ``FaultSource``.

        Duck-types the fault (iterates it; reads each source's ``.x``
        [km], ``.angles`` [**radians** — converted to degrees here], and
        ``.tt`` [rupture onset]). A ``PointSource`` carries no scalar
        moment or rise/peak time, so ``M0`` / ``rise_time`` / ``peak_time``
        are supplied here — a scalar (shared) or a per-source sequence.
        ``length_scale`` (deck-length per km) is required.
        """
        sources = list(fault)
        n = len(sources)
        specs: list[FaultSubsource] = []
        for i, ps in enumerate(sources):
            x = np.asarray(ps.x, dtype=float).ravel()
            ang = np.asarray(ps.angles, dtype=float).ravel()  # RADIANS
            if x.shape[0] != 3 or ang.shape[0] < 3:
                raise ValueError(
                    f"from_shakermaker: source {i} needs x (len 3) and "
                    f"angles (len 3); got x={x.tolist()}, angles={ang.tolist()}."
                )
            pos = _deck_position(
                x[0], x[1], x[2], length_scale=length_scale, frame=frame,
            )
            specs.append(FaultSubsource(
                position=pos,
                M0=_per_source(M0, i, n),
                strike=math.degrees(float(ang[0])),
                dip=math.degrees(float(ang[1])),
                rake=math.degrees(float(ang[2])),
                t0=float(getattr(ps, "tt", 0.0)),
                rise_time=_per_source(rise_time, i, n),
                peak_time=_per_source(peak_time, i, n),
            ))
        return self._build_patterns(
            specs, frame=frame, dt=dt, t_total=t_total,
            method=method, f_max=f_max,
        )

    # -- shared pattern builder ----------------------------------------

    def _build_patterns(
        self,
        specs: "list[FaultSubsource]",
        *,
        frame: str,
        dt: float,
        t_total: float,
        method: str,
        f_max: float | None,
    ) -> "list[Plain]":
        patterns: list[Plain] = []
        skipped = 0
        for s in specs:
            if s.M0 <= 0 or s.rise_time <= 0 or s.peak_time <= 0:
                skipped += 1
                continue
            # Yoffe requires peak_time < rise_time/2 (the regularization
            # half-width); FFSP's peak_time can exceed that, so clamp.
            peak = min(s.peak_time, 0.49 * s.rise_time)
            series = self._bridge.timeSeries.Yoffe(
                rise_time=s.rise_time, peak_time=peak, t0=s.t0,
                t_total=t_total, dt=dt, f_max=f_max,
            )
            p = self._bridge.pattern.Plain(series=series)
            p.moment_tensor(
                position=s.position, frame=frame, M0=s.M0,
                mech=dict(strike=s.strike, dip=s.dip, rake=s.rake),
                method=method,
            )
            patterns.append(p)
        if skipped:
            warnings.warn(
                f"ops.fault: skipped {skipped} of {len(specs)} subfaults with "
                f"non-positive moment / rise / peak time (taper edges carry no "
                f"radiated moment).",
                WarnFaultSubfaultSkipped, stacklevel=3,
            )
        return patterns
