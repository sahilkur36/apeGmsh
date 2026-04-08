from __future__ import annotations

import importlib
import sys
import types
import unittest

import numpy as np


def _purge_pygmsh_modules() -> None:
    for name in list(sys.modules):
        if name == "pyGmsh" or name.startswith("pyGmsh."):
            del sys.modules[name]


def _install_fake_gmsh() -> None:
    fake_gmsh = types.ModuleType("gmsh")
    fake_gmsh.__version__ = "fake"
    fake_gmsh.initialize = lambda: None
    fake_gmsh.finalize = lambda: None
    fake_gmsh.model = types.SimpleNamespace(mesh=types.SimpleNamespace())
    sys.modules["gmsh"] = fake_gmsh


def _install_fake_pandas() -> None:
    fake_pandas = types.ModuleType("pandas")

    class _FakeDataFrame:
        def __init__(self, *args, **kwargs) -> None:
            pass

    fake_pandas.DataFrame = _FakeDataFrame
    sys.modules["pandas"] = fake_pandas


def _install_fake_pyvista() -> None:
    sys.modules["pyvista"] = types.ModuleType("pyvista")


class LibraryContractTests(unittest.TestCase):
    def setUp(self) -> None:
        _purge_pygmsh_modules()
        self._saved_gmsh = sys.modules.get("gmsh")
        self._saved_g2o = sys.modules.get("gmsh2opensees")
        self._saved_pandas = sys.modules.get("pandas")
        self._saved_pyvista = sys.modules.get("pyvista")
        _install_fake_gmsh()
        _install_fake_pandas()
        _install_fake_pyvista()

    def tearDown(self) -> None:
        _purge_pygmsh_modules()
        if self._saved_gmsh is None:
            sys.modules.pop("gmsh", None)
        else:
            sys.modules["gmsh"] = self._saved_gmsh

        if self._saved_g2o is None:
            sys.modules.pop("gmsh2opensees", None)
        else:
            sys.modules["gmsh2opensees"] = self._saved_g2o

        if self._saved_pandas is None:
            sys.modules.pop("pandas", None)
        else:
            sys.modules["pandas"] = self._saved_pandas

        if self._saved_pyvista is None:
            sys.modules.pop("pyvista", None)
        else:
            sys.modules["pyvista"] = self._saved_pyvista

    def test_top_level_import_only_requires_core_dependencies(self) -> None:
        pkg = importlib.import_module("pyGmsh")
        self.assertTrue(hasattr(pkg, "pyGmsh"))
        self.assertTrue(hasattr(pkg, "Assembly"))

    def test_g2o_accepts_assembly_style_active_context(self) -> None:
        calls: list[str] = []
        fake_g2o = types.ModuleType("gmsh2opensees")
        fake_g2o.gmsh2ops = lambda: calls.append("gmsh2ops")
        fake_g2o.msh2ops = lambda path: calls.append(path)
        sys.modules["gmsh2opensees"] = fake_g2o

        mod = importlib.import_module("pyGmsh.solvers.Gmsh2OpenSees")
        wrapper = mod.Gmsh2OpenSees(types.SimpleNamespace(is_active=True, _verbose=False))
        wrapper.transfer()

        self.assertEqual(calls, ["gmsh2ops"])

    def test_equal_dof_uses_instance_scope_without_manual_maps(self) -> None:
        mod = importlib.import_module("pyGmsh.core.Assembly")
        asm = mod.Assembly("demo")
        asm.instances = {
            "left": mod.Instance(
                label="left",
                part_name="left",
                file_path=types.SimpleNamespace(),
                bbox=(-1.0, -1.0, -0.1, 1.0, 1.0, 0.0),
            ),
            "right": mod.Instance(
                label="right",
                part_name="right",
                file_path=types.SimpleNamespace(),
                bbox=(-1.0, -1.0, 1e-4, 1.0, 1.0, 0.1),
            ),
        }
        asm.equal_dof("left", "right", tolerance=1e-3)

        node_tags = np.array([1, 2, 3, 4], dtype=int)
        node_coords = np.array(
            [
                [0.0, 0.0, -0.05],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 5e-4],
                [0.0, 0.0, 0.05],
            ],
            dtype=float,
        )

        records = asm.resolve_constraints(node_tags, node_coords)
        pairs = [(rec.master_node, rec.slave_node) for rec in records]
        self.assertEqual(pairs, [(2, 3)])

    def test_rigid_body_master_is_chosen_from_master_instance(self) -> None:
        mod = importlib.import_module("pyGmsh.core.Assembly")
        asm = mod.Assembly("demo")
        asm.instances = {
            "master": mod.Instance(
                label="master",
                part_name="master",
                file_path=types.SimpleNamespace(),
                bbox=(0.1, -1.0, -1.0, 0.3, 1.0, 1.0),
            ),
            "slave": mod.Instance(
                label="slave",
                part_name="slave",
                file_path=types.SimpleNamespace(),
                bbox=(-0.3, -1.0, -1.0, 0.05, 1.0, 1.0),
            ),
        }
        asm.rigid_body("master", "slave", master_point=(0.0, 0.0, 0.0))

        node_tags = np.array([1, 2, 3], dtype=int)
        node_coords = np.array(
            [
                [0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-0.2, 0.0, 0.0],
            ],
            dtype=float,
        )

        records = asm.resolve_constraints(node_tags, node_coords)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].master_node, 1)

    def test_public_constraint_api_accepts_entity_filters(self) -> None:
        mod = importlib.import_module("pyGmsh.core.Assembly")
        asm = mod.Assembly("demo")
        asm.instances = {
            "solid": mod.Instance(
                label="solid",
                part_name="solid",
                file_path=types.SimpleNamespace(),
            ),
            "frame": mod.Instance(
                label="frame",
                part_name="frame",
                file_path=types.SimpleNamespace(),
            ),
        }

        eq = asm.equal_dof(
            "solid",
            "frame",
            slave_entities=[(0, 10), (0, 11)],
            dofs=[1, 2, 3],
        )
        rl = asm.rigid_link(
            "frame",
            "frame",
            master_point=(1.0, 0.0, 0.0),
            slave_entities=[(0, 10), (0, 11)],
        )

        self.assertEqual(eq.slave_entities, [(0, 10), (0, 11)])
        self.assertEqual(eq.dofs, [1, 2, 3])
        self.assertEqual(rl.slave_entities, [(0, 10), (0, 11)])

    def test_fem_data_dim3_keeps_volume_physical_elements(self) -> None:
        fake_gmsh = sys.modules["gmsh"]

        coords_by_tag = {
            1: [0.0, 0.0, 0.0],
            2: [1.0, 0.0, 0.0],
            3: [0.0, 1.0, 0.0],
            4: [0.0, 0.0, 1.0],
        }

        def _flatten(node_tags: list[int]) -> list[float]:
            flat: list[float] = []
            for tag in node_tags:
                flat.extend(coords_by_tag[tag])
            return flat

        class _FakeMesh:
            def getNodes(self):
                tags = [1, 2, 3, 4]
                return tags, _flatten(tags), []

            def getElements(self, dim=-1, tag=-1):
                data = {
                    (1, -1): ([1], [[301]], [[1, 2]]),
                    (2, -1): ([2], [[201]], [[1, 2, 3]]),
                    (3, -1): ([4], [[101]], [[1, 2, 3, 4]]),
                    (1, 30): ([1], [[301]], [[1, 2]]),
                    (2, 20): ([2], [[201]], [[1, 2, 3]]),
                    (3, 10): ([4], [[101]], [[1, 2, 3, 4]]),
                }
                types, elem_tags, node_tags = data.get((dim, tag), ([], [], []))
                return (
                    types,
                    [np.array(tags, dtype=np.int64) for tags in elem_tags],
                    [np.array(nodes, dtype=np.int64) for nodes in node_tags],
                )

            def getElementProperties(self, etype):
                return {
                    1: ("Line 2", 1, 1, 2, [], 2),
                    2: ("Triangle 3", 2, 1, 3, [], 3),
                    4: ("Tetrahedron 4", 3, 1, 4, [], 4),
                }[int(etype)]

            def getNodesForPhysicalGroup(self, dim, pg_tag):
                node_tags = {
                    (1, 3): [1, 2],
                    (2, 2): [1, 2, 3],
                    (3, 1): [1, 2, 3, 4],
                }[(dim, pg_tag)]
                return node_tags, _flatten(node_tags)

        class _FakeModel:
            def __init__(self):
                self.mesh = _FakeMesh()

            def getPhysicalGroups(self):
                return [(1, 3), (2, 2), (3, 1)]

            def getPhysicalName(self, dim, pg_tag):
                return {
                    (1, 3): "Edge",
                    (2, 2): "Face",
                    (3, 1): "Body",
                }[(dim, pg_tag)]

            def getEntitiesForPhysicalGroup(self, dim, pg_tag):
                return {
                    (1, 3): [30],
                    (2, 2): [20],
                    (3, 1): [10],
                }[(dim, pg_tag)]

        fake_gmsh.model = _FakeModel()

        mod = importlib.import_module("pyGmsh.mesh._fem_extract")
        fem = mod.build_fem_data(dim=3)

        self.assertEqual(fem.info.n_nodes, 4)
        self.assertEqual(fem.info.n_elems, 1)
        np.testing.assert_array_equal(fem.node_ids, np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(
            fem.connectivity,
            np.array([[1, 2, 3, 4]], dtype=int),
        )

        body_tag = fem.physical.get_tag(3, "Body")
        self.assertEqual(body_tag, 1)

        body = fem.physical.get_elements(3, body_tag)
        face = fem.physical.get_elements(2, 2)
        edge = fem.physical.get_elements(1, 3)

        self.assertEqual(body["connectivity"].shape, (1, 4))
        self.assertEqual(face["connectivity"].shape, (1, 3))
        self.assertEqual(edge["connectivity"].shape, (1, 2))
        self.assertEqual(list(map(int, body["element_ids"])), [101])

    def test_fast_surface_helper_keeps_elements_with_embedded_nodes(self) -> None:
        mod = importlib.import_module("pyGmsh.viewers.SelectionPicker")

        node_tags = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        node_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],
            ],
            dtype=float,
        )
        tag_to_idx = np.full(51, -1, dtype=np.int64)
        tag_to_idx[node_tags] = np.arange(len(node_tags), dtype=np.int64)

        pts, faces_parts, n_cells = mod._surface_polydata_from_global_mesh(
            node_coords,
            tag_to_idx,
            [2],
            [
                np.array(
                    [
                        10, 20, 50,
                        20, 30, 50,
                        30, 40, 50,
                        40, 10, 50,
                    ],
                    dtype=np.int64,
                )
            ],
        )

        self.assertEqual(n_cells, 4)
        self.assertEqual(len(pts), 5)
        self.assertEqual(len(faces_parts), 1)
        self.assertEqual(len(faces_parts[0]), 16)
        self.assertEqual(int(faces_parts[0].max()), 4)
        self.assertTrue(any(np.allclose(pt, [0.5, 0.5, 0.0]) for pt in pts))


if __name__ == "__main__":
    unittest.main()
