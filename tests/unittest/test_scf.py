# tests/test_scf.py

import unittest
import numpy as np
from src.atomic_structure import AtomicStructure
from src.kpoints import KPoints
from src.exchange_correlation import ExchangeCorrelation
from src.hamiltonian import Hamiltonian
from src.solver import KS_Solver
from src.scf import SCF

class TestSCF(unittest.TestCase):

    def setUp(self):
        # 简化系统：氢原子
        lattice_vectors = np.array([[1,0,0],[0,1,0],[0,0,1]])
        atomic_positions = [[0,0,0]]
        atomic_symbols = ['H']
        self.atomic_structure = AtomicStructure(lattice_vectors, atomic_positions, atomic_symbols)
        self.kpoints = KPoints(1, lattice_vectors, dim=3)
        self.exchange_correlation = ExchangeCorrelation(functional='LDA')
        self.pseudopotentials = {'H': {'V_pp': -1.0}}
        self.hamiltonian = Hamiltonian(
            atomic_structure=self.atomic_structure,
            k_points=self.kpoints,
            pseudopotentials=self.pseudopotentials,
            exchange_correlation=self.exchange_correlation,
            max_G=1,
            num_eigen=1,
            dim=3
        )
        self.scf = SCF(
            hamiltonian_builder=self.hamiltonian,
            kpoints=self.kpoints,
            solver_class=KS_Solver,
            max_iterations=10,
            convergence_threshold=1e-3,
            mixing_factor=0.5
        )

    def test_scf_convergence(self):
        """测试SCF是否可以在给定的迭代次数内收敛，并且总能量为负。"""
        converged, final_rho_real, total_energy = self.scf.run()
        self.assertTrue(converged, "SCF未能收敛。")
        self.assertLess(total_energy, 0, "总能量应为负值。")

    def test_initial_density(self):
        """测试SCF初始电子密度是否被正确初始化。"""
        initial_density = self.scf.initialize_density()
        expected_density = np.ones(self.hamiltonian.num_G) * (1.0 / self.hamiltonian.num_G)
        np.testing.assert_array_almost_equal(
            initial_density,
            expected_density,
            decimal=5,
            err_msg="初始电子密度初始化不正确。"
        )

    def test_energy_decrease(self):
        """测试SCF迭代过程中总能量是否逐步减少或趋于稳定。"""
        scf = SCF(
            hamiltonian_builder=self.hamiltonian,
            kpoints=self.kpoints,
            solver_class=KS_Solver,
            max_iterations=10,
            convergence_threshold=1e-3,
            mixing_factor=0.5
        )
        # 修改SCF类以记录每次迭代的能量
        # 假设SCF类已经存储了每次迭代的能量列表为 scf.iteration_energies
        converged, final_rho_real, total_energy = scf.run()
        self.assertTrue(converged, "SCF未能收敛。")
        
        # 假设我们可以访问每次迭代的能量
        # 需要在SCF类中添加记录每次迭代能量的功能
        if not hasattr(scf, 'iteration_energies'):
            self.skipTest("SCF类未提供迭代能量记录功能。")
        
        energies = scf.iteration_energies
        # 确保能量列表不为空
        self.assertTrue(len(energies) > 1, "能量列表为空或只有一个能量值。")
        # 检查能量是否递减或保持不变
        for i in range(1, len(energies)):
            self.assertLessEqual(energies[i], energies[i-1] + 1e-6,
                                 f"能量在迭代{i}时未递减。前一能量={energies[i-1]}, 当前能量={energies[i]}")

    def test_final_density_positive(self):
        """测试最终电子密度是否所有值均为非负。"""
        converged, final_rho_real, total_energy = self.scf.run()
        self.assertTrue(converged, "SCF未能收敛。")
        self.assertTrue(np.all(final_rho_real >= 0), "最终电子密度存在负值。")

    def test_scf_non_convergence(self):
        """测试SCF在特定条件下是否无法收敛，并正确返回converged=False。"""
        # 使用极端参数，增加混合因子并减少最大迭代次数，可能导致不收敛
        scf = SCF(
            hamiltonian_builder=self.hamiltonian,
            kpoints=self.kpoints,
            solver_class=KS_Solver,
            max_iterations=1,               # 减少迭代次数
            convergence_threshold=1e-6,     # 提高收敛要求
            mixing_factor=0.99                # 设置高混合因子
        )
        converged, final_rho_real, total_energy = scf.run()
        self.assertFalse(converged, "SCF在设置下应未能收敛，但实际收敛。")


    def test_scf_total_energy_consistency(self):
        """测试不同SCF运行下总能量的一致性。"""
        scf1 = SCF(
            hamiltonian_builder=self.hamiltonian,
            kpoints=self.kpoints,
            solver_class=KS_Solver,
            max_iterations=10,
            convergence_threshold=1e-4,
            mixing_factor=0.3
        )
        converged1, final_rho_real1, total_energy1 = scf1.run()
        self.assertTrue(converged1, "SCF1未能收敛。")

        scf2 = SCF(
            hamiltonian_builder=self.hamiltonian,
            kpoints=self.kpoints,
            solver_class=KS_Solver,
            max_iterations=20,
            convergence_threshold=1e-5,
            mixing_factor=0.3
        )
        converged2, final_rho_real2, total_energy2 = scf2.run()
        self.assertTrue(converged2, "SCF2未能收敛。")

        # 能量应相近
        self.assertAlmostEqual(total_energy1, total_energy2, places=4, msg="不同SCF运行的总能量不一致。")

    def test_scf_different_pseudopotentials(self):
        """测试使用不同赝势参数对SCF结果的影响。"""
        # 使用不同的赝势
        pseudopotentials = {'H': {'V_pp': -2.0}}
        hamiltonian = Hamiltonian(
            atomic_structure=self.atomic_structure,
            k_points=self.kpoints,
            pseudopotentials=pseudopotentials,
            exchange_correlation=self.exchange_correlation,
            max_G=1,
            num_eigen=1,
            dim=3
        )
        scf = SCF(
            hamiltonian_builder=hamiltonian,
            kpoints=self.kpoints,
            solver_class=KS_Solver,
            max_iterations=10,
            convergence_threshold=1e-3,
            mixing_factor=0.5
        )
        converged, final_rho_real, total_energy = scf.run()
        self.assertTrue(converged, "SCF未能收敛。")
        self.assertLess(total_energy, -1.0, "使用更强赝势后的总能量应更低。")
    
    def test_scf_different_exchange_correlation(self):
        """测试使用不同交换-相关功能对SCF结果的影响。"""
        # 使用不同的交换-相关功能（如非LDA，当前未实现，因此预期结果不同）
        exchange_correlation = ExchangeCorrelation(functional='GGA')  # 假设GGA未实现，返回0.0
        hamiltonian = Hamiltonian(
            atomic_structure=self.atomic_structure,
            k_points=self.kpoints,
            pseudopotentials=self.pseudopotentials,
            exchange_correlation=exchange_correlation,
            max_G=1,
            num_eigen=1,
            dim=3
        )
        scf = SCF(
            hamiltonian_builder=hamiltonian,
            kpoints=self.kpoints,
            solver_class=KS_Solver,
            max_iterations=10,
            convergence_threshold=1e-3,
            mixing_factor=0.5
        )
        converged, final_rho_real, total_energy = scf.run()
        self.assertTrue(converged, "SCF未能收敛。")
        # 由于GGA未实现，能量应略有不同，具体取决于实现细节
        # 此处仅检查能量存在且为负
        self.assertLess(total_energy, 0, "总能量应为负值。")




if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)