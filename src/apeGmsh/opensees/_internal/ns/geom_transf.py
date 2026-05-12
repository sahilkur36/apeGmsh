"""
``_GeomTransfNS`` — backs ``ops.geomTransf.<Type>(...)``.

Phase 1D populates the three concrete transforms (Linear, PDelta,
Corotational). Each method registers a typed primitive on the bridge
and returns the typed instance for downstream use as
``transf=trans`` on element creation calls.

Per ADR 0010, the bridge accepts an orientation (``orientation=``) OR
an explicit ``vecxz=`` vector — not both. The orientation path defers
per-element vecxz derivation to the build pipeline (Phase 4); the
vecxz path emits one ``geomTransf`` line as-is.
"""
from __future__ import annotations

from ...transform import Corotational, Linear, Orientation, PDelta
from ._base import _BridgeNamespace


__all__ = ["_GeomTransfNS"]


class _GeomTransfNS(_BridgeNamespace):
    """``ops.geomTransf.<Type>(...)`` — typed geomTransf primitives."""

    def Linear(
        self,
        *,
        orientation: Orientation | None = None,
        vecxz: tuple[float, float, float] | None = None,
        roll_deg: float = 0.0,
    ) -> Linear:
        """Register a ``geomTransf Linear``. See :class:`Linear`."""
        return self._bridge._register(
            Linear(orientation=orientation, vecxz=vecxz, roll_deg=roll_deg)
        )

    def PDelta(
        self,
        *,
        orientation: Orientation | None = None,
        vecxz: tuple[float, float, float] | None = None,
        roll_deg: float = 0.0,
    ) -> PDelta:
        """Register a ``geomTransf PDelta``. See :class:`PDelta`."""
        return self._bridge._register(
            PDelta(orientation=orientation, vecxz=vecxz, roll_deg=roll_deg)
        )

    def Corotational(
        self,
        *,
        orientation: Orientation | None = None,
        vecxz: tuple[float, float, float] | None = None,
        roll_deg: float = 0.0,
    ) -> Corotational:
        """Register a ``geomTransf Corotational``. See :class:`Corotational`."""
        return self._bridge._register(
            Corotational(orientation=orientation, vecxz=vecxz, roll_deg=roll_deg)
        )
