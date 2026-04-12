"""
Standalone test for PG survival across OCC boolean operations.

Run: python tests/run_pg_survival_test.py
"""
import sys
import os

# Add src to path so we can import apeGmsh.core.Labels
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import gmsh
from apeGmsh.core.Labels import (
    LABEL_PREFIX,
    snapshot_physical_groups,
    remap_physical_groups,
)

PASSED = 0
FAILED = 0


def pg_names() -> dict[str, list[int]]:
    out = {}
    for dim, pg_tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, pg_tag)
        ents = list(gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag))
        out[name] = [int(t) for t in ents]
    return out


def check(condition, msg):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS: {msg}")
    else:
        FAILED += 1
        print(f"  FAIL: {msg}")


# =====================================================================
# Test 1: Label PGs survive fragment
# =====================================================================
def test_label_pgs_survive_fragment():
    print("\n[Test 1] Label PGs survive fragment")
    gmsh.initialize()
    gmsh.model.add("test1")

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

    names = pg_names()
    all_vols = [t for _, t in gmsh.model.getEntities(3)]

    check("_label:part_A" in names, "part_A label exists")
    check("_label:part_B" in names, "part_B label exists")
    check(len(all_vols) == 3, f"3 volumes after fragment (got {len(all_vols)})")

    if "_label:part_A" in names and "_label:part_B" in names:
        a_set = set(names["_label:part_A"])
        b_set = set(names["_label:part_B"])
        check(len(a_set) >= 1, f"part_A has entities: {a_set}")
        check(len(b_set) >= 1, f"part_B has entities: {b_set}")
        check(bool(a_set & b_set), f"overlap volume in both labels: A={a_set}, B={b_set}")

    gmsh.finalize()


# =====================================================================
# Test 2: User PGs survive fragment
# =====================================================================
def test_user_pgs_survive_fragment():
    print("\n[Test 2] User PGs survive fragment")
    gmsh.initialize()
    gmsh.model.add("test2")

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

    names = pg_names()
    check("Concrete" in names, "Concrete PG exists")
    if "Concrete" in names:
        check(len(names["Concrete"]) == 2,
              f"Concrete has 2 entities after split (got {len(names['Concrete'])})")

    gmsh.finalize()


# =====================================================================
# Test 3: PGs survive fuse
# =====================================================================
def test_pgs_survive_fuse():
    print("\n[Test 3] PGs survive fuse")
    gmsh.initialize()
    gmsh.model.add("test3")

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

    names = pg_names()
    all_vols = [t for _, t in gmsh.model.getEntities(3)]

    check(len(all_vols) == 1, f"1 volume after fuse (got {len(all_vols)})")
    check("_label:part_A" in names, "part_A label exists after fuse")
    check("_label:part_B" in names, "part_B label exists after fuse")

    if "_label:part_A" in names and "_label:part_B" in names:
        check(names["_label:part_A"] == all_vols,
              f"part_A points to fused volume")
        check(names["_label:part_B"] == all_vols,
              f"part_B points to fused volume")

    gmsh.finalize()


# =====================================================================
# Test 4: Cut — object survives, tool PG warns
# =====================================================================
def test_pgs_survive_cut():
    print("\n[Test 4] PGs survive cut (object kept, tool consumed)")
    gmsh.initialize()
    gmsh.model.add("test4")

    box_a = gmsh.model.occ.addBox(0, 0, 0, 2, 1, 1)
    box_b = gmsh.model.occ.addBox(0.5, 0.25, 0.25, 1, 0.5, 0.5)
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

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        remap_physical_groups(snap, input_dimtags, result_map)
        warn_msgs = [str(x.message) for x in w]

    names = pg_names()
    check("Object" in names, "Object PG survives cut")
    check("Tool" not in names, "Tool PG correctly removed (consumed)")

    consumed_warned = any("consumed" in m for m in warn_msgs)
    check(consumed_warned, f"Warning emitted for consumed tool PG")

    gmsh.finalize()


# =====================================================================
# Test 5: Uninvolved entity's PG stays intact
# =====================================================================
def test_uninvolved_pg_survives():
    print("\n[Test 5] Uninvolved PG stays intact")
    gmsh.initialize()
    gmsh.model.add("test5")

    box_a = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    box_b = gmsh.model.occ.addBox(2, 0, 0, 1, 1, 1)
    box_c = gmsh.model.occ.addBox(5, 5, 5, 1, 1, 1)
    gmsh.model.occ.synchronize()

    pg_c = gmsh.model.addPhysicalGroup(3, [box_c])
    gmsh.model.setPhysicalName(3, pg_c, "Bystander")

    obj = [(3, box_a)]
    tool = [(3, box_b)]
    input_dimtags = obj + tool

    snap = snapshot_physical_groups()
    result, result_map = gmsh.model.occ.fragment(
        obj, tool, removeObject=True, removeTool=True,
    )
    gmsh.model.occ.synchronize()
    remap_physical_groups(snap, input_dimtags, result_map)

    names = pg_names()
    check("Bystander" in names, "Bystander PG exists")
    if "Bystander" in names:
        check(names["Bystander"] == [box_c],
              f"Bystander still points to original entity (got {names['Bystander']})")

    gmsh.finalize()


# =====================================================================
# Test 6: Surface labels survive 2D fragment
# =====================================================================
def test_surface_labels_survive_2d_fragment():
    print("\n[Test 6] Surface labels survive 2D fragment")
    gmsh.initialize()
    gmsh.model.add("test6")

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

    names = pg_names()
    all_surfs = [t for _, t in gmsh.model.getEntities(2)]

    check("_label:rect_A" in names, "rect_A label exists")
    check("_label:rect_B" in names, "rect_B label exists")
    check(len(all_surfs) == 3, f"3 surfaces after fragment (got {len(all_surfs)})")

    gmsh.finalize()


# =====================================================================
# Test 7: Multi-PG on same entity both survive
# =====================================================================
def test_multiple_pgs_on_same_entity():
    print("\n[Test 7] Multiple PGs referencing same entity both survive")
    gmsh.initialize()
    gmsh.model.add("test7")

    box_a = gmsh.model.occ.addBox(0, 0, 0, 2, 1, 1)
    box_b = gmsh.model.occ.addBox(1, 0, 0, 2, 1, 1)
    gmsh.model.occ.synchronize()

    # Two PGs on the same entity
    pg1 = gmsh.model.addPhysicalGroup(3, [box_a])
    gmsh.model.setPhysicalName(3, pg1, "_label:col_A.shaft")
    pg2 = gmsh.model.addPhysicalGroup(3, [box_a])
    gmsh.model.setPhysicalName(3, pg2, "ConcreteColumn")

    obj = [(3, box_a)]
    tool = [(3, box_b)]
    input_dimtags = obj + tool

    snap = snapshot_physical_groups()
    result, result_map = gmsh.model.occ.fragment(
        obj, tool, removeObject=True, removeTool=True,
    )
    gmsh.model.occ.synchronize()
    remap_physical_groups(snap, input_dimtags, result_map)

    names = pg_names()
    check("_label:col_A.shaft" in names, "Label PG survived")
    check("ConcreteColumn" in names, "User PG survived")

    if "_label:col_A.shaft" in names and "ConcreteColumn" in names:
        check(
            set(names["_label:col_A.shaft"]) == set(names["ConcreteColumn"]),
            "Both PGs point to the same remapped entities",
        )

    gmsh.finalize()


# =====================================================================
# Run all
# =====================================================================

if __name__ == "__main__":
    test_label_pgs_survive_fragment()
    test_user_pgs_survive_fragment()
    test_pgs_survive_fuse()
    test_pgs_survive_cut()
    test_uninvolved_pg_survives()
    test_surface_labels_survive_2d_fragment()
    test_multiple_pgs_on_same_entity()

    print(f"\n{'='*60}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    if FAILED:
        sys.exit(1)
    else:
        print("All tests passed!")
