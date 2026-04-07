"""
pyGmsh — Gmsh wrapper for structural FEM workflows.
====================================================

Two usage modes:

1. **Standalone** (single-model, quick prototyping)::

       from pyGmsh import pyGmsh

       g = pyGmsh(model_name="plate", verbose=True)
       g.initialize()
       g.model.add_point(0, 0, 0)
       ...
       g.finalize()

2. **Part / Assembly** (multi-part, Abaqus-style)::

       from pyGmsh import Part, Assembly

       web = Part("web")
       web.begin()
       web.model.add_point(0, 0, 0)
       web.save("web.step")
       web.end()

       asm = Assembly("bridge")
       asm.begin()
       asm.add_part(web)
       asm.mesh.generate(dim=2)
       asm.end()
"""

from pyGmsh._session import _SessionBase
from pyGmsh._core import pyGmsh
from pyGmsh.core.Part import Part
from pyGmsh.core.Assembly import Assembly
from pyGmsh.mesh.FEMData import FEMData, MeshInfo, PhysicalGroupSet
from pyGmsh.solvers.Numberer import Numberer, NumberedMesh
from pyGmsh.viz.Selection import Selection, SelectionComposite
from pyGmsh.viewers.BaseViewer import BaseViewer
from pyGmsh.viewers.SelectionPicker import SelectionPicker
from pyGmsh.viewers.SelectionPickerUI import SelectionPickerWindow
from pyGmsh.viewers.MeshViewer import MeshViewer
from pyGmsh.viewers.MeshViewerUI import MeshViewerWindow
import pyGmsh.solvers.Constraints as Constraints

__all__ = [
    "_SessionBase",
    "pyGmsh",
    "Part",
    "Assembly",
    "FEMData",
    "MeshInfo",
    "PhysicalGroupSet",
    "Numberer",
    "NumberedMesh",
    "Selection",
    "SelectionComposite",
    "BaseViewer",
    "SelectionPicker",
    "SelectionPickerWindow",
    "MeshViewer",
    "MeshViewerWindow",
    "Constraints",
]
