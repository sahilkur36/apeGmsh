"""Deprecation shim for the orientation-class relocation (Phase 8.2).

Canonical home is :mod:`apeGmsh.opensees._orientation`; the public
:class:`Cartesian` / :class:`Cylindrical` / :class:`Spherical` types
are also re-exported from :mod:`apeGmsh.opensees`.
"""
from __future__ import annotations

import warnings

from apeGmsh.opensees._orientation import (
    Cartesian,
    Cylindrical,
    Spherical,
    resolve_vecxz,
)

warnings.warn(
    "apeGmsh.solvers._opensees_csys is deprecated; import "
    "Cartesian / Cylindrical / Spherical from apeGmsh.opensees "
    "(or resolve_vecxz from apeGmsh.opensees._orientation) instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "Cartesian",
    "Cylindrical",
    "Spherical",
    "resolve_vecxz",
]
