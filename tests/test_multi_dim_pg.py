"""
Multi-dim physical-group handling — BRep layer (g.physical) and mesh
layer (fem.*.physical).

Context: the ModelViewer now creates multi-dim PGs when the user picks
entities across dims and assigns them to one name. The public read APIs
must either expose that (dim_tags / union) or fail loud rather than
silently returning a single dim.
"""
import numpy as np
import pytest

from apeGmsh import apeGmsh


# =====================================================================
# g.physical (BRep entity layer)
# =====================================================================

@pytest.fixture
def g_multi_pg():
    """Session with two boxes and a multi-dim PG 'Mixed' covering
    volume 1 and surfaces 1, 2."""
    with apeGmsh(model_name="multi_pg", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)  # vol 1
        g.model.geometry.add_box(2, 0, 0, 1, 1, 1)  # vol 2
        g.model.sync()
        g.physical.add_volume([1], name="Mixed")
        g.physical.add_surface([1, 2], name="Mixed")
        yield g


def test_dim_tags_returns_all_dims(g_multi_pg):
    dts = g_multi_pg.physical.dim_tags("Mixed")
    assert (3, 1) in dts
    assert (2, 1) in dts
    assert (2, 2) in dts
    assert len(dts) == 3


def test_dim_tags_missing_raises(g_multi_pg):
    with pytest.raises(KeyError):
        g_multi_pg.physical.dim_tags("NonExistent")


def test_entities_no_dim_multi_dim_raises(g_multi_pg):
    with pytest.raises(ValueError, match="spans multiple dimensions"):
        g_multi_pg.physical.entities("Mixed")


def test_entities_with_dim_still_works_when_multi(g_multi_pg):
    vol = g_multi_pg.physical.entities("Mixed", dim=3)
    assert vol == [1]
    surfs = g_multi_pg.physical.entities("Mixed", dim=2)
    assert set(surfs) == {1, 2}


def test_entities_no_dim_single_dim_still_works():
    with apeGmsh(model_name="single_pg", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        g.physical.add_volume([1], name="Only3D")
        assert g.physical.entities("Only3D") == [1]


# =====================================================================
# fem.*.physical (mesh layer) — union across dims
# =====================================================================

@pytest.fixture
def fem_multi_pg():
    """Mesh-generated session with a multi-dim PG 'Mixed'."""
    with apeGmsh(model_name="fem_multi_pg", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)  # vol 1
        g.model.sync()
        g.physical.add_volume([1], name="Mixed")
        g.physical.add_surface([1], name="Mixed")  # bottom face
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        fem = g.mesh.queries.get_fem_data()
        yield fem


def _mixed_keys(fem):
    """Return the (dim, pg_tag) keys that share the name 'Mixed'."""
    keys = [k for k in fem.nodes.physical.get_all()
            if fem.nodes.physical.get_name(*k) == "Mixed"]
    assert len(keys) == 2, f"expected two keys, got {keys}"
    return keys


def test_fem_physical_node_ids_union(fem_multi_pg):
    """Node IDs should be the union of dim=3 and dim=2 PG nodes,
    deduplicated. The surface nodes are a subset of the volume nodes,
    so the union equals the volume's node set."""
    keys = _mixed_keys(fem_multi_pg)
    ids_per_dim = [
        set(int(n) for n in fem_multi_pg.nodes.physical.node_ids(k))
        for k in keys
    ]
    merged = set(int(n) for n in fem_multi_pg.nodes.physical.node_ids("Mixed"))
    assert merged == set().union(*ids_per_dim)
    for s in ids_per_dim:
        assert s.issubset(merged)


def test_fem_physical_getitem_returns_merged_dict(fem_multi_pg):
    info = fem_multi_pg.nodes.physical["Mixed"]
    assert info["name"] == "Mixed"
    vol_key = next(k for k in _mixed_keys(fem_multi_pg) if k[0] == 3)
    vol_n = len(fem_multi_pg.nodes.physical.node_ids(vol_key))
    # Merged must have >= volume's node count (union of vol + surface).
    assert len(info["node_ids"]) >= vol_n


def test_fem_physical_node_coords_match_ids(fem_multi_pg):
    info = fem_multi_pg.nodes.physical["Mixed"]
    assert len(info["node_ids"]) == len(info["node_coords"])


def test_fem_physical_element_ids_union_when_multi(fem_multi_pg):
    keys = _mixed_keys(fem_multi_pg)
    per_dim = [
        set(int(e) for e in fem_multi_pg.nodes.physical.element_ids(k))
        for k in keys
    ]
    merged = set(int(e) for e in fem_multi_pg.nodes.physical.element_ids("Mixed"))
    assert merged == set().union(*per_dim)


def test_fem_nodes_get_pg_union(fem_multi_pg):
    """fem.nodes.get(pg=name) routes through physical.node_ids — must
    return the union when the PG is multi-dim."""
    result = fem_multi_pg.nodes.get(pg="Mixed")
    direct = set(int(n) for n in fem_multi_pg.nodes.physical.node_ids("Mixed"))
    assert set(int(n) for n in result.ids) == direct


def test_fem_physical_dim_tag_tuple_path_still_single(fem_multi_pg):
    """Explicit (dim, tag) access returns only that dim's data — no union."""
    vol_key = next(k for k in _mixed_keys(fem_multi_pg) if k[0] == 3)
    info = fem_multi_pg.nodes.physical._resolve(vol_key)
    vol_ids = fem_multi_pg.nodes.physical.node_ids(vol_key)
    assert len(info["node_ids"]) == len(vol_ids)


def test_fem_nodes_get_pg_with_dim_filter(fem_multi_pg):
    """`fem.nodes.get(pg='Mixed', dim=2)` returns only the dim=2 PG's
    nodes, not the union."""
    keys = _mixed_keys(fem_multi_pg)
    surf_key = next(k for k in keys if k[0] == 2)
    vol_key = next(k for k in keys if k[0] == 3)
    surf_only_ids = set(
        int(n) for n in fem_multi_pg.nodes.physical.node_ids(surf_key))
    vol_only_ids = set(
        int(n) for n in fem_multi_pg.nodes.physical.node_ids(vol_key))

    surf_dim_ids = set(int(n) for n in
                       fem_multi_pg.nodes.get(pg="Mixed", dim=2).ids)
    vol_dim_ids = set(int(n) for n in
                      fem_multi_pg.nodes.get(pg="Mixed", dim=3).ids)

    assert surf_dim_ids == surf_only_ids
    assert vol_dim_ids == vol_only_ids
    assert surf_dim_ids != vol_dim_ids  # sanity: they really differ


def test_fem_physical_node_ids_with_dim_kwarg(fem_multi_pg):
    keys = _mixed_keys(fem_multi_pg)
    surf_key = next(k for k in keys if k[0] == 2)
    by_dim = fem_multi_pg.nodes.physical.node_ids("Mixed", dim=2)
    by_tuple = fem_multi_pg.nodes.physical.node_ids(surf_key)
    assert np.array_equal(
        np.asarray(by_dim, dtype=np.int64),
        np.asarray(by_tuple, dtype=np.int64),
    )


def test_fem_nodes_get_pg_with_bad_dim_raises(fem_multi_pg):
    """Asking for a dim the PG doesn't exist at raises."""
    with pytest.raises(KeyError, match="dim=1"):
        fem_multi_pg.nodes.get(pg="Mixed", dim=1)


def test_fem_elements_get_pg_with_dim_filter(fem_multi_pg):
    """ElementComposite's existing `dim=` filter already restricts a
    multi-dim PG to the requested dim slice."""
    keys = _mixed_keys(fem_multi_pg)
    surf_key = next(k for k in keys if k[0] == 2)

    # Elements of the dim=2 PG of 'Mixed' alone
    expected = set(int(e) for e in fem_multi_pg.nodes.physical.element_ids(surf_key))
    # Via the get(pg=, dim=) path
    res = fem_multi_pg.elements.get(pg="Mixed", dim=2)
    got = set()
    for group in res:
        got.update(int(x) for x in group.ids)
    assert got == expected


def test_fem_physical_single_dim_name_unchanged():
    """Sanity: when a name has only one dim, the old behavior holds —
    no merged-view overhead, no change in return shape."""
    with apeGmsh(model_name="single", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        g.physical.add_volume([1], name="Only")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        fem = g.mesh.queries.get_fem_data()
        ids_by_name = fem.nodes.physical.node_ids("Only")
        ids_by_tuple = fem.nodes.physical.node_ids((3, 1))
        assert np.array_equal(
            np.asarray(ids_by_name, dtype=np.int64),
            np.asarray(ids_by_tuple, dtype=np.int64),
        )
