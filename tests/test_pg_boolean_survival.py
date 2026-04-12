"""
Tests — Physical group survival across OCC boolean operations.

Verifies that both label PGs (Tier 1, ``_label:*`` prefix) and user-
facing PGs (Tier 2) survive every boolean path in the codebase:

1. ``gmsh.model.occ.fragment`` via ``g.parts.fragment_all()``
2. ``gmsh.model.occ.fragment`` via ``g.parts.fragment_pair()``
3. ``gmsh.model.occ.fuse``     via ``g.parts.fuse_group()``
4. ``gmsh.model.occ.fragment`` via ``g.model.boolean.fragment()``
5. ``gmsh.model.occ.fuse``     via ``g.model.boolean.fuse()``
6. ``gmsh.model.occ.cut``      via ``g.model.boolean.cut()``
7. ``gmsh.model.occ.fragment`` via ``g.model.queries.make_conformal()``
8. ``gmsh.model.occ.fragment`` via ``g.model.geometry.cut_by_surface()``

The core mechanism under test is :func:`snapshot_physical_groups` /
:func:`remap_physical_groups` in ``Labels.py``.
"""
from __future__ import annotations

import gmsh
import pytest

from apeGmsh.core.Labels import (
    LABEL_PREFIX,
    is_label_pg,
    snapshot_physical_groups,
    remap_physical_groups,
)


# =====================================================================
# Helpers
# =====================================================================

def _pg_names() -> dict[str, list[int]]:
    """Return ``{name: [entity_tags]}`` for ALL PGs in the session."""
    out: dict[str, list[int]] = {}
    for dim, pg_tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, pg_tag)
        ents = list(gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag))
        out[name] = [int(t) for t in ents]
    return out


def _label_names() -> list[str]:
    """Return bare label names (prefix stripped)."""
    return [
        n[len(LABEL_PREFIX):]
        for n in _pg_names()
        if n.startswith(LABEL_PREFIX)
    ]


def _user_pg_names() -> list[str]:
    """Return non-label PG names."""
    return [n for n in _pg_names() if not n.startswith(LABEL_PREFIX)]


# =====================================================================
# Low-level: snapshot + remap directly on Gmsh API
# =====================================================================

class TestSnapshotRemap:
    """Test the raw snapshot/remap utilities against bare Gmsh calls."""

    def setup_method(self):
        gmsh.initialize()
        gmsh.model.add("test")

    def teardown_method(self):
        gmsh.finalize()

    def test_label_pgs_survive_fragment(self):
        """_label:* PGs must survive occ.fragment + synchronize."""
        box_a = gmsh.model.occ.addBox(0, 0, 0, 2, 1, 1)
        box_b = gmsh.model.occ.addBox(1, 0, 0, 2, 1, 1)
        gmsh.model.occ.synchronize()

        pg_a = gmsh.model.addPhysicalGroup(3, [box_a])
        gmsh.model.setPhysicalName(3, pg_a, "_label:part_A")
        pg_b = gmsh.model.addPhysicalGroup(3, [box_b])
        gmsh.model.setPhysicalName(3, pg_b, "_label:part_B")

        obj = [(3, box_a)]
        tool = [(3, box_b)]
        input_dimtags = obj + tool

        snap = snapshot_physical_groups()
        result, result_map = gmsh.model.occ.fragment(
            obj, tool, removeObject=True, removeTool=True,
        )
        gmsh.model.occ.synchronize()
        remap_physical_groups(snap, input_dimtags, result_map)

        names = _pg_names()
        assert "_label:part_A" in names, f"part_A missing. Got: {names}"
        assert "_label:part_B" in names, f"part_B missing. Got: {names}"

        # Fragment should produce 3 volumes: A-only, overlap, B-only
        all_vols = [t for _, t in gmsh.model.getEntities(3)]
        assert len(all_vols) == 3

        # Each label should reference at least 1 volume, ≤ 2
        assert 1 <= len(names["_label:part_A"]) <= 2
        assert 1 <= len(names["_label:part_B"]) <= 2

        # The overlap volume should belong to BOTH labels
        a_set = set(names["_label:part_A"])
        b_set = set(names["_label:part_B"])
        assert a_set & b_set, "Overlap volume should be in both labels"

    def test_user_pgs_survive_fragment(self):
        """Tier-2 (user-facing) PGs must also survive."""
        box_a = gmsh.model.occ.addBox(0, 0, 0, 2, 1, 1)
        box_b = gmsh.model.occ.addBox(1, 0, 0, 2, 1, 1)
        gmsh.model.occ.synchronize()

        pg = gmsh.model.addPhysicalGroup(3, [box_a])
        gmsh.model.setPhysicalName(3, pg, "Concrete")

        obj = [(3, box_a)]
        tool = [(3, box_b)]
        input_dimtags = obj + tool

        snap = snapshot_physical_groups()
        result, result_map = gmsh.model.occ.fragment(
            obj, tool, removeObject=True, removeTool=True,
        )
        gmsh.model.occ.synchronize()
        remap_physical_groups(snap, input_dimtags, result_map)

        names = _pg_names()
        assert "Concrete" in names
        # box_a split into 2 pieces → PG should have both
        assert len(names["Concrete"]) == 2

    def test_pgs_survive_fuse(self):
        """PGs must survive occ.fuse."""
        box_a = gmsh.model.occ.addBox(0, 0, 0, 2, 1, 1)
        box_b = gmsh.model.occ.addBox(1, 0, 0, 2, 1, 1)
        gmsh.model.occ.synchronize()

        pg_a = gmsh.model.addPhysicalGroup(3, [box_a])
        gmsh.model.setPhysicalName(3, pg_a, "_label:part_A")
        pg_b = gmsh.model.addPhysicalGroup(3, [box_b])
        gmsh.model.setPhysicalName(3, pg_b, "_label:part_B")

        obj = [(3, box_a)]
        tool = [(3, box_b)]
        input_dimtags = obj + tool

        snap = snapshot_physical_groups()
        result, result_map = gmsh.model.occ.fuse(
            obj, tool, removeObject=True, removeTool=True,
        )
        gmsh.model.occ.synchronize()
        remap_physical_groups(snap, input_dimtags, result_map, absorbed_into_result=True)

        names = _pg_names()
        # Fuse → single volume; both labels should point to it
        all_vols = [t for _, t in gmsh.model.getEntities(3)]
        assert len(all_vols) == 1

        assert "_label:part_A" in names
        assert "_label:part_B" in names
        assert names["_label:part_A"] == all_vols
        assert names["_label:part_B"] == all_vols

    def test_pgs_survive_cut(self):
        """PGs on the object survive cut; tool PG warns and vanishes."""
        box_a = gmsh.model.occ.addBox(0, 0, 0, 2, 1, 1)
        box_b = gmsh.model.occ.addBox(1, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()

        pg_a = gmsh.model.addPhysicalGroup(3, [box_a])
        gmsh.model.setPhysicalName(3, pg_a, "Object")
        pg_b = gmsh.model.addPhysicalGroup(3, [box_b])
        gmsh.model.setPhysicalName(3, pg_b, "Tool")

        obj = [(3, box_a)]
        tool = [(3, box_b)]
        input_dimtags = obj + tool

        snap = snapshot_physical_groups()
        result, result_map = gmsh.model.occ.cut(
            obj, tool, removeObject=True, removeTool=True,
        )
        gmsh.model.occ.synchronize()

        with pytest.warns(UserWarning, match="consumed"):
            remap_physical_groups(snap, input_dimtags, result_map)

        names = _pg_names()
        assert "Object" in names
        # Tool was subtracted — its PG should be gone (empty)
        assert "Tool" not in names

    def test_uninvolved_pg_survives(self):
        """A PG on an entity NOT involved in the boolean stays intact."""
        box_a = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        box_b = gmsh.model.occ.addBox(2, 0, 0, 1, 1, 1)
        box_c = gmsh.model.occ.addBox(5, 5, 5, 1, 1, 1)  # uninvolved
        gmsh.model.occ.synchronize()

        pg_c = gmsh.model.addPhysicalGroup(3, [box_c])
        gmsh.model.setPhysicalName(3, pg_c, "Bystander")

        # Only fragment A and B
        obj = [(3, box_a)]
        tool = [(3, box_b)]
        input_dimtags = obj + tool

        snap = snapshot_physical_groups()
        result, result_map = gmsh.model.occ.fragment(
            obj, tool, removeObject=True, removeTool=True,
        )
        gmsh.model.occ.synchronize()
        remap_physical_groups(snap, input_dimtags, result_map)

        names = _pg_names()
        assert "Bystander" in names
        assert names["Bystander"] == [box_c]

    def test_surface_labels_on_2d_fragment(self):
        """Surface-level label PGs survive a 2D fragment."""
        r_a = gmsh.model.occ.addRectangle(0, 0, 0, 2, 1)
        r_b = gmsh.model.occ.addRectangle(1, 0, 0, 2, 1)
        gmsh.model.occ.synchronize()

        pg_a = gmsh.model.addPhysicalGroup(2, [r_a])
        gmsh.model.setPhysicalName(2, pg_a, "_label:rect_A")
        pg_b = gmsh.model.addPhysicalGroup(2, [r_b])
        gmsh.model.setPhysicalName(2, pg_b, "_label:rect_B")

        obj = [(2, r_a)]
        tool = [(2, r_b)]
        input_dimtags = obj + tool

        snap = snapshot_physical_groups()
        result, result_map = gmsh.model.occ.fragment(
            obj, tool, removeObject=True, removeTool=True,
        )
        gmsh.model.occ.synchronize()
        remap_physical_groups(snap, input_dimtags, result_map)

        names = _pg_names()
        assert "_label:rect_A" in names
        assert "_label:rect_B" in names

        # Should have 3 surfaces total (A-only, overlap, B-only)
        all_surfs = [t for _, t in gmsh.model.getEntities(2)]
        assert len(all_surfs) == 3
