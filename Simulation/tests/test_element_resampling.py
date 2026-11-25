import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pyNastran.bdf.bdf import BDF

from Simulation.utils.element_resampling import _interpolate_element
from Simulation.utils.study_pipeline import evaluate_study, load_study_config
from Simulation.utils.surface_resampling import SurfaceGrid, resample_npz


class ElementResamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "simulation.npz"
        self.model = BDF(debug=False)
        self.coordinates = np.array([
            [0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2],
            [1, 0, 0], [2, 0, 1], [1, 0, 2], [0, 0, 1],
            [3, 0, 0], [5, 0, 0], [3, 0, 2],
            [4, 0, 0], [4, 0, 1], [3, 0, 1],
        ], dtype=float)
        for node_id, coordinate in enumerate(self.coordinates, 1):
            self.model.add_grid(node_id, coordinate)
        self.model.add_pshell(1, mid1=1, t=1.)
        self.model.add_mat1(1, E=1., G=None, nu=0.3)
        self.model.add_cquad8(1, 1, list(range(1, 9)))
        self.model.add_ctria6(2, 1, list(range(9, 15)))

    def write_files(self, node_count: int = 14) -> None:
        self.model.write_bdf(str(self.path.with_suffix(".nas")))
        xyz = self.coordinates[:node_count]
        x, z = xyz[:, 0], xyz[:, 2]
        displacement = np.column_stack((x**2 + z**2, x * z, 2 * x - z))
        np.savez(
            self.path,
            node_ids=np.arange(1, node_count + 1),
            original_coordinates=xyz,
            displacement=displacement,
        )

    def test_quadratic_fields_and_outside_mesh(self) -> None:
        self.write_files()
        points = np.array([[0.5, 0.5], [3.5, 0.5], [2.5, 0.5], [2, 0], [4.5, 1.5]])
        grid = SurfaceGrid((0, 2), 1, 0., points, (5, 1))
        result = resample_npz(self.path, grid, method="elements")
        np.testing.assert_array_equal(result.valid_mask, [True, True, False, True, False])
        np.testing.assert_allclose(
            result.displacement[[0, 1, 3]],
            [[0.5, 0.25, 0.5], [12.5, 1.75, 6.5], [4., 0., 4.]],
            atol=1e-10,
        )
        self.assertTrue(np.isnan(result.displacement[2]).all())

    def test_missing_node_result_is_an_error(self) -> None:
        self.write_files(node_count=13)
        grid = SurfaceGrid((0, 2), 1, 0., np.array([[0.5, 0.5]]), (1, 1))
        with self.assertRaisesRegex(ValueError, "nodes missing"):
            resample_npz(self.path, grid, method="elements")

    def test_missing_nas_is_an_error(self) -> None:
        np.savez(self.path, node_ids=[1], original_coordinates=[[0, 0, 0]], displacement=[[0, 0, 0]])
        grid = SurfaceGrid((0, 2), 1, 0., np.array([[0., 0.]]), (1, 1))
        with self.assertRaises(FileNotFoundError):
            resample_npz(self.path, grid, method="elements")

    def test_curved_quadrilateral_uses_inverse_isoparametric_mapping(self) -> None:
        self.coordinates[4, 2] = -0.4
        self.coordinates[5, 0] = 2.3
        self.coordinates[6, 2] = 2.1
        self.coordinates[7, 0] = -0.2
        for node_id in range(5, 9):
            self.model.nodes[node_id].xyz = self.coordinates[node_id - 1].copy()
        self.model.write_bdf(str(self.path.with_suffix(".nas")))
        displacement = np.zeros((14, 3))
        displacement[4, 0] = 1.
        np.savez(self.path, node_ids=np.arange(1, 15), original_coordinates=self.coordinates, displacement=displacement)
        grid = SurfaceGrid((0, 2), 1, 0., np.array([[1.284, 0.36]]), (1, 1))
        result = resample_npz(self.path, grid, method="elements")
        self.assertTrue(result.valid_mask[0])
        np.testing.assert_allclose(result.displacement[0], [0.672, 0, 0], atol=1e-9)

    def test_curved_triangle_uses_inverse_isoparametric_mapping(self) -> None:
        self.coordinates[11] = [4., 0., -0.4]
        self.coordinates[12] = [4.2, 0., 1.1]
        self.coordinates[13] = [2.7, 0., 1.]
        for node_id in range(12, 15):
            self.model.nodes[node_id].xyz = self.coordinates[node_id - 1].copy()
        self.model.write_bdf(str(self.path.with_suffix(".nas")))
        displacement = np.zeros((14, 3))
        displacement[11, 0] = 1.
        np.savez(
            self.path, node_ids=np.arange(1, 15),
            original_coordinates=self.coordinates, displacement=displacement,
        )
        grid = SurfaceGrid((0, 2), 1, 0., np.array([[3.268, 0.464]]), (1, 1))
        result = resample_npz(self.path, grid, method="elements")
        self.assertTrue(result.valid_mask[0])
        np.testing.assert_allclose(result.displacement[0], [0.4, 0, 0], atol=1e-9)

    def test_inverse_mapping_converges_near_distorted_element_corner(self) -> None:
        coordinates = np.array([
            [0., 0.], [2., 0.], [2., 2.], [0., 2.],
            [1.095669458373444, 0.3949248187863345],
            [1.860176797860456, 1.0513016036933052],
            [0.8166696674081366, 1.7028199763954368],
            [-0.2743330231632099, 0.6021508100843169],
        ])
        point = np.array([[1.969257909344647, 0.09318705111677612]])
        displacement = np.column_stack((coordinates, np.zeros(8)))
        inside, result = _interpolate_element("CQUAD8", coordinates, displacement, point, 2e-9)
        self.assertTrue(inside[0])
        np.testing.assert_allclose(result[0], [point[0, 0], point[0, 1], 0.], atol=2e-9)

    def test_unsupported_element_is_an_error(self) -> None:
        self.model.add_ctria3(3, 1, [1, 2, 3])
        self.write_files()
        grid = SurfaceGrid((0, 2), 1, 0., np.array([[0.5, 0.5]]), (1, 1))
        with self.assertRaisesRegex(ValueError, "CTRIA3"):
            resample_npz(self.path, grid, method="elements")

    def test_configured_study_uses_element_resampling(self) -> None:
        root = Path(self.directory.name)
        cases = []
        x, z = self.coordinates[:, 0], self.coordinates[:, 2]
        displacement = np.column_stack((x**2 + z**2, x * z, 2 * x - z))
        for i in (1, 2):
            folder = root / f"case_{i}"
            folder.mkdir()
            self.model.write_bdf(str(folder / "case.nas"))
            np.savez(
                folder / "case.npz",
                node_ids=np.arange(1, 15),
                original_coordinates=self.coordinates,
                displacement=i * displacement,
            )
            cases.append({"status": "success", "npz_file": f"case_{i}/case.npz"})
        (root / "simulation_status.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
        config_path = root / "config.json"
        config_path.write_text(json.dumps({
            "name": "elements",
            "surface_resampling": "elements",
            "interpolation_grid": {"nx": 11, "nz": 5},
            "rank": 1,
            "landmarks": [{"x": 0.5, "z": 0.5}, {"x": 3.5, "z": 0.5}],
        }), encoding="utf-8")
        config = load_study_config(config_path, results_dir=root)
        metrics = evaluate_study(config)
        self.assertEqual(len(metrics), 2)
        self.assertLess(max(row["relative_l2_error"] for row in metrics), 1e-12)
        manifest = json.loads((config.output_dir / "study_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["surface_resampling"], "elements")


if __name__ == "__main__":
    unittest.main()
