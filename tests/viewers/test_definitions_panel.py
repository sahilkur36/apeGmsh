"""DefinitionsPanel — populates from ViewerData.names (headless Qt).

Offscreen-Qt smoke: build the panel, feed it a fake snapshot exposing
``.names``, assert the tree groups by kind and the idle hint toggles.
No VTK / GL — the panel is a pure Qt widget over the read seam.
"""
from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


class _FakeViewerData:
    def __init__(self, names):
        self.names = tuple(names)


def test_empty_shows_hint_hides_tree(qapp):
    from apeGmsh.viewers.ui._definitions_panel import DefinitionsPanel

    p = DefinitionsPanel()
    p.set_data(_FakeViewerData(()))
    # isVisibleTo(ancestor) reflects setVisible() without a shown window.
    assert p._hint.isVisibleTo(p.widget) is True
    assert p._tree.isVisibleTo(p.widget) is False
    assert p._tree.topLevelItemCount() == 0


def test_groups_by_kind_and_lists_names(qapp):
    from apeGmsh.viewers.ui._definitions_panel import DefinitionsPanel

    p = DefinitionsPanel()
    p.set_data(_FakeViewerData([
        ("rebar", "uniaxialMaterial", 1),
        ("cover", "uniaxialMaterial", 2),
        ("col_sec", "section", 1),
        ("ramp", "timeSeries", 1),
    ]))
    assert p._tree.isVisibleTo(p.widget) is True
    assert p._hint.isVisibleTo(p.widget) is False

    # Top-level groups, in canonical kind order: uniaxial, section, timeSeries.
    headers = [
        p._tree.topLevelItem(i).text(0)
        for i in range(p._tree.topLevelItemCount())
    ]
    assert headers[0].startswith("Uniaxial materials (2)")
    assert any(h.startswith("Sections (1)") for h in headers)
    assert any(h.startswith("Time series (1)") for h in headers)

    # Names sorted under their kind, tag in the second column.
    uniaxial = p._tree.topLevelItem(0)
    child_names = [uniaxial.child(i).text(0) for i in range(uniaxial.childCount())]
    assert child_names == ["cover", "rebar"]
    assert uniaxial.child(1).text(1) == "1"  # rebar -> tag 1


def test_tolerates_snapshot_without_names(qapp):
    from apeGmsh.viewers.ui._definitions_panel import DefinitionsPanel

    p = DefinitionsPanel()
    p.set_data(object())  # no .names attr
    assert p._hint.isVisibleTo(p.widget) is True
