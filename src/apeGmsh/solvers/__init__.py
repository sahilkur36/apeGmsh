"""apeGmsh.solvers — Legacy compatibility shim.

This package is being retired.  Phase 8.1 moved the broker-side
content (record dataclasses, the constraint / load / mass resolvers,
and the Numberer) out of ``apeGmsh.solvers`` into the mesh and core
layers.  Phase 8.2 moves the coordinate-system helpers
(``Cartesian`` / ``Cylindrical`` / ``Spherical`` / ``resolve_vecxz``)
into ``apeGmsh.opensees``.  The legacy ``OpenSees`` bridge class and
the response / recorder catalog move in subsequent sub-phases.

Until those finish, this module keeps the old import paths alive.
The per-module shim files (``Numberer.py``, ``Constraints.py``,
``Loads.py``, ``Masses.py``, ``_kinds.py``, ``_opensees_csys.py`` …)
warn on import; the package-level ``__getattr__`` below catches the
``from apeGmsh.solvers import X`` shape so that path also emits a
one-shot :class:`DeprecationWarning`.
"""

from __future__ import annotations

import warnings
from typing import Any

from .OpenSees import OpenSees

# Names that moved out of apeGmsh.solvers during the Phase-8 untangle.
# Listed as ``{attr: (canonical_module, canonical_attr)}`` so
# __getattr__ can point users at the new home.
#
# Phase 8.1 — broker-side records & resolvers (mesh/ + core/).
# Phase 8.2 — bridge-side coordinate-system helpers (opensees/).
_RELOCATED: dict[str, tuple[str, str]] = {
    # Phase 8.1
    "Numberer":     ("apeGmsh.mesh._numberer", "Numberer"),
    "NumberedMesh": ("apeGmsh.mesh._numberer", "NumberedMesh"),
    # Phase 8.2
    "Cartesian":    ("apeGmsh.opensees", "Cartesian"),
    "Cylindrical":  ("apeGmsh.opensees", "Cylindrical"),
    "Spherical":    ("apeGmsh.opensees", "Spherical"),
}


def __getattr__(name: str) -> Any:
    """Lazy attribute hook for Phase 8.1-relocated names.

    Fires a one-shot :class:`DeprecationWarning` and returns the
    canonical object so existing ``from apeGmsh.solvers import X``
    code keeps working for one release cycle.
    """
    target = _RELOCATED.get(name)
    if target is not None:
        mod_path, attr = target
        warnings.warn(
            f"apeGmsh.solvers.{name} is deprecated; import {attr} from "
            f"{mod_path} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from importlib import import_module
        return getattr(import_module(mod_path), attr)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = [
    "OpenSees",
    # Accessible via __getattr__ (one-shot DeprecationWarning):
    "Numberer",
    "NumberedMesh",
    "Cartesian",
    "Cylindrical",
    "Spherical",
]
