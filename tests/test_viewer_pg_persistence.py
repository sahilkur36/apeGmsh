"""
Regression tests for ModelViewer physical-group persistence.

Bugs covered:
  - empty active group not propagated (commit + flush + set_active)
  - rename leaves stale PG in gmsh
  - apply_group missing from _group_order
"""
import gmsh
import pytest

from apeGmsh.viewers.core.selection import (
    SelectionState,
    _load_group_members,
)


@pytest.fixture
def pg_gmsh_session():
    gmsh.initialize()
    gmsh.model.add("test_pg_persistence")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)   # vol 1
    gmsh.model.occ.addBox(2, 0, 0, 1, 1, 1)   # vol 2
    gmsh.model.occ.synchronize()
    yield
    gmsh.finalize()


def _pg_names() -> set[str]:
    names = set()
    for d, t in gmsh.model.getPhysicalGroups():
        try:
            names.add(gmsh.model.getPhysicalName(d, t))
        except Exception:
            pass
    return names


# ── Phase 1 ─────────────────────────────────────────────────────────

def test_empty_active_group_deleted_on_commit(pg_gmsh_session):
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.commit_active_group()
    assert "Foo" in _pg_names()

    sel.unpick((3, 1))
    sel.commit_active_group()
    assert "Foo" not in _pg_names()


def test_empty_group_deleted_on_flush(pg_gmsh_session):
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.commit_active_group()
    assert "Foo" in _pg_names()

    sel.unpick((3, 1))
    sel.flush_to_gmsh()
    assert "Foo" not in _pg_names()


def test_switching_from_empty_group_deletes_it(pg_gmsh_session):
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.commit_active_group()
    assert "Foo" in _pg_names()

    sel.unpick((3, 1))
    sel.set_active_group("Bar")  # switching away while Foo is empty
    assert "Foo" not in _pg_names()


# ── Phase 2 ─────────────────────────────────────────────────────────

def test_rename_removes_old_pg(pg_gmsh_session):
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.commit_active_group()
    assert "Foo" in _pg_names()

    sel.rename_group("Foo", "Bar")
    names = _pg_names()
    assert "Bar" in names
    assert "Foo" not in names


def test_rename_updates_group_order(pg_gmsh_session):
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.commit_active_group()
    sel.rename_group("Foo", "Bar")
    assert "Foo" not in sel.group_order
    assert "Bar" in sel.group_order


def test_rename_active_group_preserves_pointer(pg_gmsh_session):
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.commit_active_group()
    sel.rename_group("Foo", "Bar")
    assert sel.active_group == "Bar"


def test_apply_group_registers_order(pg_gmsh_session):
    sel = SelectionState()
    sel.pick((3, 1))
    sel.apply_group("X")
    sel.unpick((3, 1))
    sel.pick((3, 2))
    sel.apply_group("Y")
    assert sel.group_order == ["X", "Y"]


# ── Reported repro ──────────────────────────────────────────────────

def test_reported_repro_create_then_query(pg_gmsh_session):
    """Reproduction: create a group via _on_new_group flow, query it after."""
    sel = SelectionState()
    sel.pick((3, 1))
    sel.pick((3, 2))
    # Simulates _on_new_group in model_viewer.py
    sel._staged_groups["Foo"] = list(sel._picks)
    sel.set_active_group("Foo")
    # Simulates close
    sel.flush_to_gmsh()

    assert "Foo" in _pg_names()
    members = _load_group_members("Foo")
    assert set(members) == {(3, 1), (3, 2)}
