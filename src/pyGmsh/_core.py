from __future__ import annotations
from pathlib import Path

import gmsh
from ._optional      import MissingOptionalDependency


class pyGmsh:
    def __init__(self,
                 *,
                 model_name: str = "ModelName",
                 verbose: bool = False) -> None:
        from .Inspect        import Inspect
        from .Model          import Model
        from .Mesh           import Mesh
        from .PhysicalGroups import PhysicalGroups
        from .Partition      import Partition
        from .View           import View
        from .Gmsh2OpenSees  import Gmsh2OpenSees
        from .OpenSees       import OpenSees

        self.model_name = model_name
        self._verbose = verbose
        self._initialized: bool = False
        self.inspect   = Inspect(self)
        self.model     = Model(self)
        self.mesh      = Mesh(self)
        self.physical  = PhysicalGroups(self)
        self.partition = Partition(self)
        self.view      = View(self)
        self.g2o       = Gmsh2OpenSees(self)
        self.opensees  = OpenSees(self)

        try:
            from .Plot import Plot
            self.plot = Plot(self)
        except ImportError as exc:
            self.plot = MissingOptionalDependency(
                "Plotting support",
                "matplotlib",
                extra="plot",
                cause=exc,
            )

    def __enter__(self) -> pyGmsh:
        if self._initialized:
            return self
        gmsh.initialize()
        gmsh.model.add(self.model_name)
        self._initialized = True
        if self._verbose:
            print(f"Gmsh version: {gmsh.__version__}")
        return self

    def initialize(self) -> pyGmsh:
        return self.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._initialized:
            gmsh.finalize()
            self._initialized = False
        return False  # Do not suppress exceptions

    def finalize(self) -> None:
        """Manual finalize, in case context manager is not used."""
        if self._initialized:
            gmsh.finalize()
            self._initialized = False

    @property
    def is_active(self) -> bool:
        """Return True when the wrapped Gmsh session is open."""
        return self._initialized

    def remove_duplicates(
        self,
        *,
        tolerance: float | None = None,
        sync     : bool         = True,
    ) -> None:
        """Delegate to ``self.model.remove_duplicates()``."""
        self.model.remove_duplicates(tolerance=tolerance, sync=sync)

    def make_conformal(
        self,
        *,
        dims     : list[int] | None = None,
        tolerance: float | None     = None,
        sync     : bool             = True,
    ) -> None:
        """Delegate to ``self.model.make_conformal()``."""
        self.model.make_conformal(dims=dims, tolerance=tolerance, sync=sync)



