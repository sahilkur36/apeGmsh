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
from pyGmsh.mesh.MshLoader import MshLoader
from pyGmsh.solvers.Numberer import Numberer, NumberedMesh
from pyGmsh.viz.Selection import Selection, SelectionComposite
from pyGmsh.viewers.model_viewer import ModelViewer
from pyGmsh.viewers._mesh_viewer import MeshViewer as MeshViewerV2

# Backward-compatible aliases
SelectionPicker = ModelViewer
MeshViewer = MeshViewerV2
from pyGmsh.results.Results import Results
import pyGmsh.solvers.Constraints as Constraints

__all__ = [
    "_SessionBase",
    "pyGmsh",
    "Part",
    "Assembly",
    "FEMData",
    "MeshInfo",
    "PhysicalGroupSet",
    "MshLoader",
    "Results",
    "Numberer",
    "NumberedMesh",
    "Selection",
    "SelectionComposite",
    "ModelViewer",
    "MeshViewerV2",
    "SelectionPicker",
    "MeshViewer",
    "Results",
    "Constraints",
]