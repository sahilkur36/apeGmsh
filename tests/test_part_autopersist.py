"""
Phase-1 regression tests for ``Part`` auto-persist.

These tests exercise the full auto-persist lifecycle end-to-end
against the real Gmsh kernel — unlike the unit tests under
``tests/test_opensees_tie_*.py``, which stub Gmsh out because they
only need the record shapes, this file needs a live OCC kernel to
exercise ``save()`` / ``importShapes()``.

Coverage
--------

1. Auto-persist happy path: Part built inside a ``with`` block
   exits, auto-persists to an OS tempfile, ``parts.add(part)``
   can consume it, and the file lives under ``tempfile.gettempdir()``.
2. ``parts.add`` consumes the auto-persisted Part and registers
   the instance in a session.
3. Explicit ``save()`` is never auto-cleaned, even after the Part
   is garbage-collected.
4. ``auto_persist=False`` opts out: ``parts.add`` raises
   ``FileNotFoundError`` with a message that hints at the opt-out.
5. ``cleanup()`` is idempotent and re-entry into a fresh ``with``
   block works after cleanup.
6a. Exception raised inside ``with part:`` still triggers auto-persist
    on exit — the partial geometry is a debug artifact.  The user's
    original exception propagates unchanged.
6b. If auto-persist itself fails (e.g. the temp save raises), the
    user's original exception is preserved and a warning is emitted.
7. GC-during-mesh safety: ``parts.add(part); del part; gc.collect();
   g.mesh.generation.generate(2)`` must succeed because ``gmsh.merge``
   reads the file synchronously.
"""
from __future__ import annotations

import gc
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from apeGmsh import Part, apeGmsh


def _build_plate(part: Part) -> None:
    """Helper: build a small closed plate surface inside *part*."""
    p1 = part.model.geometry.add_point(0, 0, 0)
    p2 = part.model.geometry.add_point(1, 0, 0)
    p3 = part.model.geometry.add_point(1, 1, 0)
    p4 = part.model.geometry.add_point(0, 1, 0)
    l1 = part.model.geometry.add_line(p1, p2)
    l2 = part.model.geometry.add_line(p2, p3)
    l3 = part.model.geometry.add_line(p3, p4)
    l4 = part.model.geometry.add_line(p4, p1)
    loop = part.model.geometry.add_curve_loop([l1, l2, l3, l4])
    part.model.geometry.add_plane_surface(loop)


# =====================================================================
# Test 1 — auto-persist happy path
# =====================================================================

class TestAutoPersistHappyPath:

    def test_auto_persist_writes_tempfile_on_exit(self):
        part = Part("plate")
        with part:
            _build_plate(part)
        try:
            assert part.has_file
            assert part._owns_file
            assert part.file_path is not None
            assert part.file_path.exists()
            # Tempfile must live under the OS tempdir (spec contract)
            assert str(part.file_path).startswith(tempfile.gettempdir())
            # Non-empty STEP file
            assert part.file_path.stat().st_size > 0
        finally:
            part.cleanup()

    def test_empty_part_does_not_create_tempdir(self):
        """A ``with part: pass`` must NOT leave an empty tempdir
        behind — the auto-persist guard skips empty Parts."""
        part = Part("empty")
        with part:
            pass
        assert not part.has_file
        assert not part._owns_file
        assert part._temp_dir is None
        assert part.file_path is None


# =====================================================================
# Test 2 — consume auto-persisted Part in an assembly
# =====================================================================

class TestPartsAddConsumesAutoPersisted:

    def test_parts_add_imports_autopersisted_part(self):
        col = Part("column")
        with col:
            col.model.geometry.add_box(0, 0, 0, 1, 1, 3)

        try:
            with apeGmsh(model_name="bridge") as g:
                inst = g.parts.add(col)
                assert inst.label == "column_1"
                # One volume was imported
                assert 3 in inst.entities
                assert len(inst.entities[3]) == 1
        finally:
            col.cleanup()


# =====================================================================
# Test 3 — explicit save is never auto-cleaned
# =====================================================================

class TestExplicitSaveIsNotAutoCleaned:

    def test_explicit_save_survives_cleanup(self, tmp_path):
        target = tmp_path / "explicit.step"

        part = Part("explicit")
        with part:
            _build_plate(part)
            part.save(target)

        # Explicit save wins — _owns_file is False, file lives at
        # the user's path
        assert not part._owns_file
        assert part.file_path == target.resolve()
        assert target.exists()

        # Calling cleanup() is a no-op for user-owned files
        part.cleanup()
        assert target.exists()

        # Dropping the Part and forcing GC also does not touch it
        del part
        gc.collect()
        assert target.exists()


# =====================================================================
# Test 4 — auto_persist=False opt-out
# =====================================================================

class TestAutoPersistOptOut:

    def test_opt_out_leaves_no_file(self):
        part = Part("manual", auto_persist=False)
        with part:
            _build_plate(part)

        assert not part.has_file
        assert part.file_path is None
        assert not part._owns_file

    def test_opt_out_parts_add_raises_clear_error(self):
        part = Part("manual", auto_persist=False)
        with part:
            _build_plate(part)

        try:
            with apeGmsh(model_name="asm") as g:
                with pytest.raises(FileNotFoundError, match="no file to import"):
                    g.parts.add(part)
        finally:
            part.cleanup()

    def test_opt_out_explicit_save_is_user_owned(self, tmp_path):
        target = tmp_path / "mine.step"
        part = Part("manual", auto_persist=False)
        with part:
            _build_plate(part)
            part.save(target)

        assert target.exists()
        assert part.file_path == target.resolve()
        assert not part._owns_file
        part.cleanup()
        assert target.exists()  # user file is sacred


# =====================================================================
# Test 5 — cleanup() is idempotent + re-entry works
# =====================================================================

class TestCleanupIdempotent:

    def test_cleanup_is_idempotent(self):
        part = Part("cycler")
        with part:
            _build_plate(part)

        file_was = part.file_path
        assert file_was is not None
        assert file_was.exists()

        part.cleanup()
        assert not part.has_file
        assert part.file_path is None
        assert not file_was.exists()

        # Second cleanup is a no-op
        part.cleanup()
        assert part.file_path is None

    def test_reentry_after_cleanup(self):
        """A Part can be re-used after cleanup — entering a new
        ``with`` block rebuilds the session from scratch and the
        second auto-persist writes a fresh tempfile."""
        part = Part("cycler")
        with part:
            _build_plate(part)
        first_path = part.file_path
        assert first_path is not None and first_path.exists()

        part.cleanup()
        assert not first_path.exists()

        with part:
            _build_plate(part)
        second_path = part.file_path
        try:
            assert second_path is not None
            assert second_path != first_path
            assert second_path.exists()
        finally:
            part.cleanup()

    def test_reentry_without_explicit_cleanup(self):
        """Re-entering ``with part:`` without calling cleanup()
        first must not break auto-persist — ``begin()`` auto-cleans
        any stale tempfile before the new session starts."""
        part = Part("auto_cycler")
        with part:
            _build_plate(part)
        first_path = part.file_path
        assert first_path is not None and first_path.exists()

        # NO explicit cleanup — just re-enter.
        with part:
            _build_plate(part)
        second_path = part.file_path
        try:
            assert second_path is not None
            assert second_path != first_path
            assert second_path.exists()
            # First path was cleaned up by Part.begin()
            assert not first_path.exists()
        finally:
            part.cleanup()


# =====================================================================
# Test 6 — exception during build
# =====================================================================

class TestExceptionDuringBuild:

    def test_user_exception_propagates_and_artifact_is_persisted(self):
        """Geometry built before the exception survives as a
        debug artifact.  The user's original exception propagates
        untouched."""
        part = Part("crashy")

        with pytest.raises(RuntimeError, match="boom"):
            with part:
                _build_plate(part)           # real geometry
                raise RuntimeError("boom")   # user bug

        # Auto-persist ran despite the exception — this is the
        # documented debug-artifact behaviour.
        try:
            assert part.has_file, (
                "auto-persist should have written the partial "
                "geometry as a debug artifact"
            )
        finally:
            part.cleanup()

    def test_auto_persist_failure_warns_without_shadowing_user_exception(self):
        """If auto-persist itself raises, the user's exception
        still propagates unchanged and a warning is emitted."""
        part = Part("doubly_crashy")

        def _boom(self):
            raise OSError("tempdir exploded")

        with patch.object(Part, "_auto_persist_to_temp", _boom):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with pytest.raises(RuntimeError, match="user bug"):
                    with part:
                        _build_plate(part)
                        raise RuntimeError("user bug")

            # User exception propagated.  Auto-persist failure
            # surfaced as a warning mentioning the Part's name.
            messages = [str(w.message) for w in caught]
            assert any(
                "doubly_crashy" in m and "auto-persist failed" in m
                for m in messages
            ), f"expected auto-persist warning in {messages}"

        assert not part.has_file


# =====================================================================
# Test 7 — GC-during-mesh safety
# =====================================================================

class TestGCDuringMesh:

    def test_gc_after_parts_add_does_not_break_mesh(self):
        """After ``parts.add(part)`` returns, gmsh.merge has already
        read the tempfile synchronously and copied the geometry
        into the current model — so dropping the Part and forcing
        GC mid-workflow must be safe."""
        part = Part("ephemeral")
        with part:
            part.model.geometry.add_box(0, 0, 0, 1, 1, 1)

        try:
            with apeGmsh(model_name="mesh_gc") as g:
                g.parts.add(part)
                # Drop every Python-side reference to the part and
                # force collection BEFORE generating the mesh.
                path_was = part.file_path
                del part
                gc.collect()

                # With aggressive GC, the tempfile may have already
                # been reclaimed — that's fine, the geometry is
                # already inside the session's OCC kernel.
                g.mesh.sizing.set_global_size(0.5)
                g.mesh.generation.generate(dim=3)
                fem = g.mesh.queries.get_fem_data(dim=3)
                assert fem.info.n_nodes > 0
                assert fem.info.n_elems > 0
        finally:
            # If GC didn't clean up the tempfile, rmtree the parent
            # dir ourselves to keep the test hermetic.
            if path_was is not None and path_was.parent.exists():
                import shutil
                shutil.rmtree(path_was.parent, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
