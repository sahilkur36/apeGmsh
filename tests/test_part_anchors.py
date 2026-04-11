"""
Phase 2 regression tests — Part label anchors + Instance lookup.

These tests exercise the full anchor round-trip against a live
Gmsh kernel:

1. Anchor sidecar is written next to the auto-persisted STEP.
2. Only user-named entities are anchored (auto-generated labels
   are filtered out).
3. ``parts.add(part)`` rebinds labels into ``Instance.label_to_tag``
   under each of: no transform, translate-only, rotate-only, and
   combined translate+rotate.
4. ``Instance.__getitem__`` accepts string labels, int tags, and
   ``(dim, tag)`` tuples — returning ``(dim, tag)`` in every case.
5. Named escape hatches (``by_label``, ``by_tag``) work.
6. Error paths: unknown label → KeyError; tag not in instance →
   KeyError; unsupported key type → TypeError.
7. Opt-out: ``save(..., write_anchors=False)`` suppresses the
   sidecar.
8. Missing sidecar falls back cleanly (empty ``label_to_tag``,
   instance still works).
9. Multiple instances of the same Part each rebind independently
   — useful for the "reuse this column 20 times" workflow.
"""
from __future__ import annotations

import math

import pytest

from apeGmsh import Part, apeGmsh
from apeGmsh.core._part_anchors import read_sidecar, sidecar_path


# =====================================================================
# Helpers
# =====================================================================

def _make_labeled_column() -> Part:
    """Build a Part with two user-named boxes and one unnamed
    helper point (the point must NOT show up in the sidecar)."""
    col = Part("column")
    with col:
        col.model.geometry.add_box(0, 0, 0, 1, 1, 3, label="shaft")
        col.model.geometry.add_box(
            0, 0, 2.5, 0.3, 0.3, 0.5, label="top_region",
        )
        # Unlabeled helper point — falls through the auto-label
        # filter in collect_anchors
        col.model.geometry.add_point(5, 5, 5)
    return col


# =====================================================================
# Sidecar file tests
# =====================================================================

class TestSidecar:

    def test_sidecar_is_written_next_to_cad(self):
        col = _make_labeled_column()
        try:
            side = sidecar_path(col.file_path)
            assert side.exists(), f"expected sidecar at {side}"
        finally:
            col.cleanup()

    def test_sidecar_contains_only_user_named_entities(self):
        col = _make_labeled_column()
        try:
            payload = read_sidecar(col.file_path)
            assert payload is not None
            labels = {a["label"] for a in payload["anchors"]}
            # User-named entities are present
            assert "shaft" in labels
            assert "top_region" in labels
            # Auto-generated labels are NOT present
            for a in payload["anchors"]:
                assert not a["label"].startswith("point_")
                assert not a["label"].startswith("box_")
        finally:
            col.cleanup()

    def test_sidecar_records_dim_and_kind(self):
        col = _make_labeled_column()
        try:
            payload = read_sidecar(col.file_path)
            by_label = {a["label"]: a for a in payload["anchors"]}
            assert by_label["shaft"]["dim"] == 3
            assert by_label["shaft"]["kind"] == "box"
            assert by_label["top_region"]["dim"] == 3
            # COM of shaft (0,0,0)→(1,1,3) is (0.5, 0.5, 1.5)
            com = by_label["shaft"]["com"]
            assert abs(com[0] - 0.5) < 1e-9
            assert abs(com[1] - 0.5) < 1e-9
            assert abs(com[2] - 1.5) < 1e-9
        finally:
            col.cleanup()

    def test_empty_part_writes_no_sidecar(self):
        """A Part with no user-named entities produces NO sidecar."""
        part = Part("nameless")
        with part:
            part.model.geometry.add_box(0, 0, 0, 1, 1, 1)  # no label
        try:
            # Auto-persist created the STEP, but the sidecar should
            # be absent because no user labels exist.
            side = sidecar_path(part.file_path)
            assert not side.exists()
        finally:
            part.cleanup()

    def test_write_anchors_false_suppresses_sidecar(self, tmp_path):
        part = _make_labeled_column()
        target = tmp_path / "explicit.step"
        # Re-open the session to call save() explicitly
        with part:
            part.model.geometry.add_box(0, 0, 0, 1, 1, 3, label="shaft")
            part.save(target, write_anchors=False)

        try:
            assert target.exists()
            assert not sidecar_path(target).exists()
        finally:
            part.cleanup()
            if target.exists():
                target.unlink()


# =====================================================================
# Rebinding — transform matrix
# =====================================================================

class TestRebindingTransforms:

    def test_rebind_no_transform(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(col)
                assert "shaft" in inst.label_to_tag
                assert "top_region" in inst.label_to_tag
                shaft_dim, shaft_tag = inst.label_to_tag["shaft"]
                assert shaft_dim == 3
                assert shaft_tag in inst.entities[3]
        finally:
            col.cleanup()

    def test_rebind_with_translate(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(col, translate=(100, 200, 300))
                assert "shaft" in inst.label_to_tag
                assert "top_region" in inst.label_to_tag
                # Shaft volume's COM should now be around (100.5, 200.5, 301.5)
                import gmsh
                _, shaft_tag = inst.label_to_tag["shaft"]
                com = gmsh.model.occ.getCenterOfMass(3, shaft_tag)
                assert abs(com[0] - 100.5) < 1e-6
                assert abs(com[1] - 200.5) < 1e-6
                assert abs(com[2] - 301.5) < 1e-6
        finally:
            col.cleanup()

    def test_rebind_with_rotate_90deg_about_z(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(col, rotate=(math.pi / 2, 0, 0, 1))
                assert "shaft" in inst.label_to_tag
                assert "top_region" in inst.label_to_tag
        finally:
            col.cleanup()

    def test_rebind_with_translate_and_rotate(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(
                    col,
                    translate=(50, 0, 0),
                    rotate=(math.pi, 0, 0, 1),
                )
                assert "shaft" in inst.label_to_tag
                assert "top_region" in inst.label_to_tag
        finally:
            col.cleanup()


# =====================================================================
# Instance lookup — the three input modes
# =====================================================================

class TestInstanceLookup:

    def test_lookup_by_label_returns_dimtag(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(col)
                result = inst["shaft"]
                assert isinstance(result, tuple)
                assert len(result) == 2
                dim, tag = result
                assert dim == 3
                assert tag in inst.entities[3]
        finally:
            col.cleanup()

    def test_lookup_by_tag_int_passthrough(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(col)
                # Pick a tag that's actually in the instance
                _, shaft_tag = inst["shaft"]
                result = inst[shaft_tag]
                assert result == (3, shaft_tag)
        finally:
            col.cleanup()

    def test_lookup_by_dimtag_tuple_passthrough(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(col)
                dimtag = inst["shaft"]
                # Round-trip through tuple input
                assert inst[dimtag] == dimtag
        finally:
            col.cleanup()

    def test_by_label_and_by_tag_methods(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(col)
                via_label = inst.by_label("shaft")
                via_getitem = inst["shaft"]
                assert via_label == via_getitem

                _, shaft_tag = via_label
                via_by_tag = inst.by_tag(shaft_tag)
                assert via_by_tag == via_label

                via_by_tag_with_dim = inst.by_tag(shaft_tag, dim=3)
                assert via_by_tag_with_dim == via_label
        finally:
            col.cleanup()


# =====================================================================
# Error paths
# =====================================================================

class TestInstanceLookupErrors:

    def test_unknown_label_raises_keyerror(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(col)
                with pytest.raises(KeyError, match="no entity labeled 'nope'"):
                    inst["nope"]
        finally:
            col.cleanup()

    def test_tag_not_in_instance_raises_keyerror(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(col)
                with pytest.raises(KeyError, match="no entity with tag 99999"):
                    inst[99999]
        finally:
            col.cleanup()

    def test_dimtag_not_in_instance_raises_keyerror(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(col)
                with pytest.raises(KeyError, match=r"no entity \(0, 99999\)"):
                    inst[(0, 99999)]
        finally:
            col.cleanup()

    def test_unsupported_key_type_raises_typeerror(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(col)
                with pytest.raises(TypeError):
                    inst[3.14]  # float
        finally:
            col.cleanup()

    def test_bool_key_rejected(self):
        """bool is an int subclass — make sure the __getitem__
        dispatcher rejects it cleanly so ``inst[True]`` doesn't
        resolve to tag 1."""
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(col)
                with pytest.raises(TypeError, match="bool"):
                    inst[True]
        finally:
            col.cleanup()


# =====================================================================
# Graceful fallback — missing sidecar
# =====================================================================

class TestMissingSidecar:

    def test_import_without_sidecar_gives_empty_label_map(self, tmp_path):
        """A Part saved with write_anchors=False still imports
        cleanly, but ``inst.label_to_tag`` is empty."""
        col = _make_labeled_column()
        target = tmp_path / "no_sidecar.step"
        # Reopen and save explicitly with anchors disabled
        with col:
            col.model.geometry.add_box(0, 0, 0, 1, 1, 3, label="shaft")
            col.save(target, write_anchors=False)

        try:
            assert target.exists()
            assert not sidecar_path(target).exists()

            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(col)
                assert inst.label_to_tag == {}
                # Instance is still usable via tag / dimtag lookup
                assert inst.entities[3], "entities should still be populated"
        finally:
            col.cleanup()
            if target.exists():
                target.unlink()


# =====================================================================
# Multi-instance — the "reuse this Part 20 times" workflow
# =====================================================================

class TestMultipleInstances:

    def test_two_instances_have_independent_label_maps(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                a = g.parts.add(col, translate=(0, 0, 0))
                b = g.parts.add(col, translate=(10, 0, 0))

                assert a.label_to_tag["shaft"] != b.label_to_tag["shaft"]

                # Looking up the same label in each instance gives
                # different tags
                a_shaft = a["shaft"]
                b_shaft = b["shaft"]
                assert a_shaft != b_shaft

                # Each instance correctly rejects the OTHER's tag
                _, a_tag = a_shaft
                _, b_tag = b_shaft
                assert a[a_tag] == a_shaft
                with pytest.raises(KeyError):
                    a[b_tag]
        finally:
            col.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
