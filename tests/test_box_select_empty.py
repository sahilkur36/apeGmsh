"""
Phase 3 regression: an empty box-select release fires on_box_select
with an empty list. Previously the callback was gated behind
`if hits`, leaving the user with no feedback on an unsuccessful drag.
"""
import numpy as np

from apeGmsh.viewers.core.pick_engine import PickEngine


class _StubRegistry:
    """Reports one entity far outside any reasonable box."""
    dims: list[int] = [3]

    def all_entities(self):
        return [(3, 1)]

    def bbox(self, dt):
        # 8 corners of an AABB way off-screen in world coords
        return np.array(
            [[1000, 1000, 1000], [1001, 1000, 1000],
             [1000, 1001, 1000], [1001, 1001, 1000],
             [1000, 1000, 1001], [1001, 1000, 1001],
             [1000, 1001, 1001], [1001, 1001, 1001]],
            dtype=float,
        )

    def centroid(self, dt):
        return np.array([1000.5, 1000.5, 1000.5])

    def entity_points(self, dt):
        return None


class _StubRenderer:
    """Projects every world point to a fixed display coord far from
    any (small) test box."""
    def __init__(self, display=(9999.0, 9999.0, 0.0)):
        self._display = display
        self._wp = (0, 0, 0, 1)

    def SetWorldPoint(self, x, y, z, w):
        self._wp = (x, y, z, w)

    def WorldToDisplay(self):
        pass

    def GetDisplayPoint(self):
        return self._display


class _StubPlotter:
    def __init__(self):
        self.renderer = _StubRenderer()
        self.render_window = None  # unused after Phase 1 Branch A


def test_empty_box_fires_callback_with_empty_hits():
    engine = PickEngine(_StubPlotter(), _StubRegistry())
    calls: list[tuple[list, bool]] = []
    engine.on_box_select = lambda hits, ctrl: calls.append((list(hits), ctrl))

    engine._do_box(10, 10, 100, 100, ctrl=False)

    assert len(calls) == 1, f"expected exactly one callback, got {len(calls)}"
    hits, ctrl = calls[0]
    assert hits == []
    assert ctrl is False


def test_empty_box_with_ctrl_flag_propagates():
    engine = PickEngine(_StubPlotter(), _StubRegistry())
    calls: list[tuple[list, bool]] = []
    engine.on_box_select = lambda hits, ctrl: calls.append((list(hits), ctrl))

    engine._do_box(10, 10, 100, 100, ctrl=True)

    assert calls == [([], True)]


def test_no_entities_still_fires_callback():
    """Even with no pickable entities at all, the drag release reports."""
    class _EmptyRegistry(_StubRegistry):
        def all_entities(self):
            return []

    engine = PickEngine(_StubPlotter(), _EmptyRegistry())
    calls: list[tuple[list, bool]] = []
    engine.on_box_select = lambda hits, ctrl: calls.append((list(hits), ctrl))

    engine._do_box(10, 10, 100, 100, ctrl=False)
    assert calls == [([], False)]
