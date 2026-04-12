"""Shared logging mixin for composites that own a ``_parent`` session reference."""

from __future__ import annotations


class _HasLogging:
    """Mixin providing a ``_log()`` helper gated by ``_parent._verbose``.

    Subclasses must:

    * Store the owning session as ``self._parent``.
    * Set a ``_log_prefix`` class variable (used in the ``[Prefix]``
      tag printed before each message).

    Example::

        class Labels(_HasLogging):
            _log_prefix = "Labels"

            def __init__(self, parent):
                self._parent = parent
    """

    _log_prefix: str = ""

    def _log(self, msg: str) -> None:
        if getattr(getattr(self, '_parent', None), '_verbose', False):
            print(f"[{self._log_prefix}] {msg}")
