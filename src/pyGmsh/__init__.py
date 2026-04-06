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

from pyGmsh._core import pyGmsh
from pyGmsh.Part import Part
from pyGmsh.Assembly import Assembly
from pyGmsh.Numberer import Numberer, NumberedMesh
from pyGmsh.Selection import Selection, SelectionComposite
from pyGmsh.SelectionPicker import SelectionPicker
from pyGmsh.SelectionPickerUI import SelectionPickerWindow
import pyGmsh.Constraints as Constraints

__all__ = [
    "pyGmsh",
    "Part",
    "Assembly",
    "Numberer",
    "NumberedMesh",
    "Selection",
    "SelectionComposite",
    "SelectionPicker",
    "SelectionPickerWindow",
    "Constraints",
]