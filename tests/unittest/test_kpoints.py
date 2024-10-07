import unittest
import numpy as np
from src.kpoints import KPoints
import sys

class TestKPoints(unittest.TestCase):

    def test_generate_kpoints_2d(self):
        lattice_vectors = np.array([[1,0,0],[0,1,0],[0,0,1]])
        num_kpoints = 4
        kpoints = KPoints(num_kpoints, lattice_vectors, dim=2)
        self.assertEqual(len(kpoints.k_points), num_kpoints**2)
        # 检查k点是否在二维平面上
        for k in kpoints.k_points:
            self.assertEqual(k[2], 0.0)

    def test_generate_kpoints_3d(self):
        lattice_vectors = np.array([[1,0,0],[0,1,0],[0,0,1]])
        num_kpoints = 4
        kpoints = KPoints(num_kpoints, lattice_vectors, dim=3)
        self.assertEqual(len(kpoints.k_points), num_kpoints**3)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)