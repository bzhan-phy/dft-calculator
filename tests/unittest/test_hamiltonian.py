import unittest
import numpy as np
from src.atomic_structure import AtomicStructure
from src.kpoints import KPoints
from src.exchange_correlation import ExchangeCorrelation
from src.hamiltonian import Hamiltonian

class TestHamiltonian(unittest.TestCase):

    def setUp(self):
        lattice_vectors = np.array([[1,0,0],[0,1,0],[0,0,1]])
        atomic_positions = [[0,0,0]]
        atomic_symbols = ['C']
        self.atomic_structure = AtomicStructure(lattice_vectors, atomic_positions, atomic_symbols)
        self.kpoints = KPoints(1, lattice_vectors, dim=3)
        self.exchange_correlation = ExchangeCorrelation(functional='LDA')
        self.pseudopotentials = {'C': {'V_pp': -10.0}}
        self.hamiltonian = Hamiltonian(
            atomic_structure=self.atomic_structure,
            k_points=self.kpoints,
            pseudopotentials=self.pseudopotentials,
            exchange_correlation=self.exchange_correlation,
            max_G=1,
            num_eigen=1,
            dim=3
        )

    def test_generate_G_vectors(self):
        G_vectors = self.hamiltonian.generate_G_vectors()
        expected_num_G = (2 * self.hamiltonian.max_G + 1) ** 3
        self.assertEqual(len(G_vectors), expected_num_G)

    def test_kinetic_energy(self):
        T = self.hamiltonian.kinetic_energy()
        # 检查动能矩阵是否为对角矩阵
        self.assertTrue(np.allclose(T, np.diag(np.diag(T))))

    def test_total_hamiltonian(self):
        # 正确初始化 rho_real，使其大小与 num_G 匹配
        rho_real = np.ones(self.hamiltonian.num_G)  # 使用全1数组作为电子密度
        rho_G = rho_real.copy()  # 假设rho_G也为全1
        H = self.hamiltonian.total_hamiltonian(rho_real, rho_G)
        
        # 检查哈密顿量矩阵是否为正确大小的矩阵
        self.assertEqual(H.shape, (self.hamiltonian.num_G, self.hamiltonian.num_G))
        
        # 进一步检查哈密顿量的对称性
        self.assertTrue(np.allclose(H, H.T, atol=1e-8))

    def test_hartree_potential(self):
        rho_G = np.ones(self.hamiltonian.num_G)
        V_H_matrix = self.hamiltonian.hartree_potential(rho_G)
        
        # 检查对角元素是否正确计算
        for i, G in enumerate(self.hamiltonian.G_vectors):
            G_norm = np.linalg.norm(G)
            if G_norm != 0:
                expected_V_H = 4 * np.pi / (G_norm ** 2) * rho_G[i]
            else:
                expected_V_H = 0.0  # 对于G=0，Hartree势能设为0
            self.assertAlmostEqual(V_H_matrix[i, i], expected_V_H, places=6)

    def test_pseudopotential(self):
        rho_G = np.ones(self.hamiltonian.num_G)
        V_pp_matrix = self.hamiltonian.pseudopotential(rho_G)
        V_pp_total = sum([params['V_pp'] for params in self.pseudopotentials.values()])
        # 检查赝势势能矩阵是否为对角矩阵，并且对角元素正确
        self.assertTrue(np.allclose(V_pp_matrix, np.diag([V_pp_total]*self.hamiltonian.num_G)))

    def test_exchange_correlation_potential(self):
        rho_real = np.ones(self.hamiltonian.num_G)
        V_xc_matrix = self.hamiltonian.exchange_correlation_potential(rho_real)
        avg_density = np.mean(rho_real)
        expected_V_xc = self.exchange_correlation.calculate_potential([avg_density])
        self.assertTrue(np.allclose(V_xc_matrix, np.diag([expected_V_xc]*self.hamiltonian.num_G)))

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
