"""
Regression tests for window-mode box-select containment predicate.

Bug covered:
  - np.all(inside) over sampled points killed entities whose 2D AABB
    was fully inside the box but one sampled vertex projected just
    outside. _entity_in_box replaces that test.

Note: window mode uses the projected 2D AABB of sample points — still
looser than the true silhouette for rotated geometry. This matches
standard CAD conventions (Rhino, FreeCAD).
"""
import numpy as np

from apeGmsh.viewers.core.pick_engine import _entity_in_box


# ── Window mode ─────────────────────────────────────────────────────

def test_window_fully_contained():
    sx = np.array([10.0, 20.0, 30.0, 40.0])
    sy = np.array([10.0, 20.0, 30.0, 40.0])
    assert _entity_in_box(sx, sy, 0, 50, 0, 50, crossing=False)


def test_window_one_vertex_outside():
    # 2D AABB max-x = 60 is outside the box (xmax=50) — must miss
    sx = np.array([10.0, 20.0, 30.0, 60.0])
    sy = np.array([10.0, 20.0, 30.0, 40.0])
    assert not _entity_in_box(sx, sy, 0, 50, 0, 50, crossing=False)


def test_window_aabb_accepts_where_old_all_predicate_rejected():
    """Old np.all(inside) would accept this. New AABB predicate also does."""
    sx = np.array([1.0, 1.0, 49.0, 49.0])
    sy = np.array([1.0, 49.0, 1.0, 49.0])
    assert _entity_in_box(sx, sy, 0, 50, 0, 50, crossing=False)


def test_window_fully_outside():
    sx = np.array([60.0, 70.0, 80.0, 90.0])
    sy = np.array([10.0, 20.0, 30.0, 40.0])
    assert not _entity_in_box(sx, sy, 0, 50, 0, 50, crossing=False)


def test_window_touching_boundary_accepted():
    """AABB touching the box boundary counts as inside (<= semantics)."""
    sx = np.array([0.0, 50.0])
    sy = np.array([0.0, 50.0])
    assert _entity_in_box(sx, sy, 0, 50, 0, 50, crossing=False)


# ── Crossing mode ───────────────────────────────────────────────────

def test_crossing_partial_overlap():
    sx = np.array([10.0, 20.0, 60.0, 70.0])  # half outside
    sy = np.array([10.0, 20.0, 30.0, 40.0])
    assert _entity_in_box(sx, sy, 0, 50, 0, 50, crossing=True)


def test_crossing_fully_contained():
    sx = np.array([10.0, 20.0, 30.0, 40.0])
    sy = np.array([10.0, 20.0, 30.0, 40.0])
    assert _entity_in_box(sx, sy, 0, 50, 0, 50, crossing=True)


def test_crossing_fully_outside():
    sx = np.array([60.0, 70.0, 80.0, 90.0])
    sy = np.array([10.0, 20.0, 30.0, 40.0])
    assert not _entity_in_box(sx, sy, 0, 50, 0, 50, crossing=True)


def test_crossing_requires_sample_in_box():
    """Crossing is sample-point-inside (classic). A small box drawn
    inside a big entity with samples only at the far corners misses
    by design — loosening the predicate over-selected adjacent
    entities whose 2D AABB was wider than the visible silhouette."""
    sx = np.array([-100.0, 100.0, -100.0, 100.0])
    sy = np.array([-100.0, -100.0, 100.0, 100.0])
    assert not _entity_in_box(sx, sy, -10, 10, -10, 10, crossing=True)


def test_crossing_touching_boundary():
    sx = np.array([-10.0, 0.0])  # sample at x=0 is on the left edge
    sy = np.array([25.0, 25.0])
    assert _entity_in_box(sx, sy, 0, 50, 0, 50, crossing=True)
