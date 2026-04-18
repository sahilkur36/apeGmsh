"""B1 regression: surface constraints with face_map=None must raise."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.core.ConstraintsComposite import ConstraintsComposite


def _mock_parent_with_parts(*labels: str) -> SimpleNamespace:
    parts = SimpleNamespace(_instances={lbl: object() for lbl in labels})
    return SimpleNamespace(parts=parts)


def test_face_map_none_raises_for_tie():
    comp = ConstraintsComposite(_mock_parent_with_parts("master", "slave"))
    comp.tie("master", "slave",
             master_entities=[(2, 1)], slave_entities=[(2, 2)])

    node_tags = np.array([1, 2, 3], dtype=int)
    node_coords = np.zeros((3, 3), dtype=float)

    with pytest.raises(TypeError, match="face_map"):
        comp.resolve(node_tags, node_coords, face_map=None)


def test_face_map_none_ok_when_no_face_constraints():
    comp = ConstraintsComposite(_mock_parent_with_parts("master", "slave"))
    comp.equal_dof("master", "slave")

    node_tags = np.array([1, 2], dtype=int)
    node_coords = np.array([[0., 0., 0.], [0., 0., 0.]], dtype=float)

    node_map = {"master": {1}, "slave": {2}}
    result = comp.resolve(node_tags, node_coords,
                          node_map=node_map, face_map=None)
    assert result is not None
