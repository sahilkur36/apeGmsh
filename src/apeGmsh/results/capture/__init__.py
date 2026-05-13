"""In-process domain capture (Strategy B).

Drives the openseespy domain via ``ops.nodeDisp(...)``,
``ops.nodeEigenvector(...)`` etc., writing native HDF5 directly.

Phase 9 commit 5 introduces a sibling declarative spec
(:class:`DomainCaptureSpec`) and two construction paths on
:class:`DomainCapture` — ``ops.domain_capture(spec, path=...)`` for
live bridge-driven capture, and :meth:`DomainCapture.from_h5` for
loading ``ndm`` / ``ndf`` from a saved ``model.h5``.

See :class:`DomainCapture` and :class:`DomainCaptureSpec` for usage.
"""
from ._domain import DomainCapture
from .spec import (
    DomainCaptureRecord,
    DomainCaptureSpec,
    ResolvedDomainCaptureRecord,
    ResolvedDomainCaptureSpec,
)

__all__ = [
    "DomainCapture",
    "DomainCaptureRecord",
    "DomainCaptureSpec",
    "ResolvedDomainCaptureRecord",
    "ResolvedDomainCaptureSpec",
]
