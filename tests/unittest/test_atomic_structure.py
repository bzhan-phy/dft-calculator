import unittest
import sys
import numpy as np
from src.atomic_structure import AtomicStructure

class TestAtomicStructure(unittest.TestCase):

    def test_num_atoms(self):
        lattice_vectors = np.array([[1,0,0],[0,1,0],[0,0,1]])
        atomic_positions = [[0,0,0], [0.5,0.5,0.5]]
        atomic_symbols = ['Si', 'Si']
        structure = AtomicStructure(lattice_vectors, atomic_positions, atomic_symbols)
        self.assertEqual(structure.get_num_atoms(), 2)

    def test_lattice_vectors(self):
        lattice_vectors = np.array([[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,2.0]])
        atomic_positions = [[0,0,0]]
        atomic_symbols = ['C']
        structure = AtomicStructure(lattice_vectors, atomic_positions, atomic_symbols)
        np.testing.assert_array_equal(structure.lattice_vectors, lattice_vectors)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
