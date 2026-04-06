from __future__ import annotations

from ._session import _SessionBase


class pyGmsh(_SessionBase):
    """Standalone single-model Gmsh session with all composites.

    Parameters
    ----------
    model_name : str
        Name passed to ``gmsh.model.add()``.
    verbose : bool
        If True, composites print diagnostic messages.
    """

    _COMPOSITES = (
        ("inspect",   ".viz.Inspect",           "Inspect",        False),
        ("model",     ".core.Model",            "Model",          False),
        ("mesh",      ".mesh.Mesh",             "Mesh",           False),
        ("physical",  ".mesh.PhysicalGroups",   "PhysicalGroups", False),
        ("partition", ".mesh.Partition",        "Partition",      False),
        ("view",      ".mesh.View",             "View",           False),
        ("g2o",       ".solvers.Gmsh2OpenSees", "Gmsh2OpenSees",  False),
        ("opensees",  ".solvers.OpenSees",      "OpenSees",       False),
        ("plot",      ".viz.Plot",              "Plot",           True),
    )

    def __init__(
        self,
        *,
        model_name: str = "ModelName",
        verbose: bool = False,
    ) -> None:
        super().__init__(name=model_name, verbose=verbose)

    # ------------------------------------------------------------------
    # model_name setter (backward compat: g.model_name = "foo")
    # ------------------------------------------------------------------

    @_SessionBase.model_name.getter
    def model_name(self) -> str:
        return self.name

    @model_name.setter
    def model_name(self, value: str) -> None:
        self.name = value

    # ------------------------------------------------------------------
    # Backward-compatible aliases
    # ------------------------------------------------------------------

    @property
    def _initialized(self) -> bool:
        """Legacy alias — use ``is_active`` instead."""
        return self._active

    @_initialized.setter
    def _initialized(self, value: bool) -> None:
        self._active = value

    def initialize(self) -> "pyGmsh":
        """Alias for ``begin()``."""
        return self.begin()

    def finalize(self) -> None:
        """Alias for ``end()``."""
        self.end()

    # ------------------------------------------------------------------
    # Convenience delegates
    # ------------------------------------------------------------------

    def remove_duplicates(
        self,
        *,
        tolerance: float | None = None,
        sync: bool = True,
    ) -> None:
        """Delegate to ``self.model.remove_duplicates()``."""
        self.model.remove_duplicates(tolerance=tolerance, sync=sync)

    def make_conformal(
        self,
        *,
        dims: list[int] | None = None,
        tolerance: float | None = None,
        sync: bool = True,
    ) -> None:
        """Delegate to ``self.model.make_conformal()``."""
        self.model.make_conformal(dims=dims, tolerance=tolerance, sync=sync)
