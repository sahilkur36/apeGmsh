from .model_viewer import ModelViewer
from ._mesh_viewer import MeshViewer as MeshViewerV2
from .GeomTransfViewer import GeomTransfViewer

# Backward-compatible aliases
SelectionPicker = ModelViewer
MeshViewer = MeshViewerV2

__all__ = [
    "ModelViewer",
    "MeshViewerV2",
    "GeomTransfViewer",
    # Aliases
    "SelectionPicker",
    "MeshViewer",
]
