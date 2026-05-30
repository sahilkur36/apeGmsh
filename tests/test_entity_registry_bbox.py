"""S1 acceptance — EntityRegistry returns canonical BBox; no degenerate tile.

Headless (EntityRegistry imports only numpy + the scene_ir BBox; no
Qt/VTK): register dims with and without real boxes on a golden registry
and assert the canonical-bbox contract + that the deleted
``np.tile(centroid, (8, 1))`` fallback no longer synthesises a fake box.
"""
from __future__ import annotations

import numpy as np

from apeGmsh.viewers.core.entity_registry import EntityRegistry
from apeGmsh.viewers.scene_ir import BBox


def _reg_with(dim, dt, *, bbox=None, centroid=None):
    reg = EntityRegistry()
    reg.register_dim(
        dim,
        mesh=object(),
        actor=object(),
        cell_to_dt={0: dt},
        centroids={dt: np.asarray(centroid)} if centroid is not None else None,
        bboxes={dt: bbox} if bbox is not None else None,
    )
    return reg


def test_bbox_returns_canonical_bbox() -> None:
    bb = BBox([0.0, 0.0, 0.0], [2.0, 4.0, 6.0])
    reg = _reg_with(2, (2, 5), bbox=bb, centroid=[1, 2, 3])
    got = reg.bbox((2, 5))
    assert isinstance(got, BBox)
    assert got.min.tolist() == [0.0, 0.0, 0.0]
    assert got.max.tolist() == [2.0, 4.0, 6.0]
    assert got.corners8.shape == (8, 3)


def test_no_degenerate_tile_when_only_centroid() -> None:
    # Centroid given but NO bbox — the old code tiled the centroid into a
    # zero-size 8-corner box; S1 deletes that, so bbox() is None.
    reg = _reg_with(1, (1, 7), centroid=[3.0, 3.0, 3.0])
    assert reg.bbox((1, 7)) is None
    assert (1, 7) not in reg._bboxes


def test_unknown_entity_bbox_is_none() -> None:
    reg = _reg_with(2, (2, 5), bbox=BBox([0, 0, 0], [1, 1, 1]))
    assert reg.bbox((3, 99)) is None
