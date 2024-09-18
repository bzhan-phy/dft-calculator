# src/scf.py

import numpy as np
import logging

from src.solver import KS_Solver


class SCF:
    """
    自洽场（SCF）循环控制类，用于迭代计算电子密度和总能量。
    """
    
    def __init__(self, hamiltonian_builder, kpoints, solver, max_iterations=100, convergence_threshold=1e-4, mixing_factor=0.3):
        """
        初始化SCF循环。
        
        参数:
        --------
        hamiltonian_builder : Hamiltonian
            哈密顿量构建器对象。
        kpoints : KPoints
            k点对象。
        solver : KS_Solver
            Kohn-Sham方程求解器类。
        max_iterations : int
            最大迭代次数。
        convergence_threshold : float
            收敛阈值。
        mixing_factor : float
            密度混合因子。
        """
        self.hamiltonian_builder = hamiltonian_builder
        self.kpoints = kpoints
        self.solver = solver
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.mixing_factor = mixing_factor
        self.logger = logging.getLogger("SCF")
        
    def initialize_density(self):
        """
        初始化电子密度。
        这里采用均匀分布的初始密度。
        
        返回:
        -------
        electron_density : numpy.ndarray
            初始电子密度。
        """
        num_G = self.hamiltonian_builder.num_G
        electron_density = np.ones(num_G) * 0.5  # 简化为常数密度
        return electron_density
    
    def build_hamiltonian(self, electron_density):
        """
        使用当前电子密度构建哈密顿量矩阵。
        
        参数:
        --------
        electron_density : numpy.ndarray
            当前电子密度。
        
        返回:
        -------
        H : numpy.ndarray
            总哈密顿量矩阵。
        """
        H = self.hamiltonian_builder.total_hamiltonian(electron_density)
        return H
    
    def compute_total_energy(self, H, eigenvalues):
        """
        计算总能量（简化版）。
        
        参数:
        --------
        H : numpy.ndarray
            哈密顿量矩阵。
        eigenvalues : numpy.ndarray
            本征值数组。
        
        返回:
        -------
        total_energy : float
            总能量。
        """
        # 简化的总能量计算：所有占据态能量之和
        total_energy = np.sum(eigenvalues)
        return total_energy
    
    def update_density(self, eigenvectors):
        """
        根据本征向量更新电子密度。
        这里简化为电子密度作为本征向量的模平方的和。
        
        参数:
        --------
        eigenvectors : numpy.ndarray
            本征向量矩阵。
        
        返回:
        -------
        new_density : numpy.ndarray
            更新后的电子密度。
        """
        # 简化的电子密度更新
        new_density = np.sum(np.abs(eigenvectors)**2, axis=1)
        return new_density
    
    def run(self):
        """
        执行SCF循环。
        
        返回:
        -------
        converged : bool
            是否收敛。
        final_density : numpy.ndarray
            最终的电子密度。
        total_energy : float
            最终的总能量。
        """
        self.logger.info("开始SCF循环...")
        electron_density = self.initialize_density()
        for iteration in range(1, self.max_iterations + 1):
            self.logger.info(f"SCF迭代 {iteration}")
            # 构建哈密顿量
            H = self.build_hamiltonian(electron_density)
            # 求解本征问题
            num_eigen = self.hamiltonian_builder.num_eigen

            # 初始化 KS_Solver 实例，传递 H 和 num_eigen
            self.solver = KS_Solver(H, num_eigen)

            eigenvalues, eigenvectors = self.solver.solve(H, num_eigen)
            #eigenvalues, eigenvectors = self.solver.solve()
            # 计算总能量
            total_energy = self.compute_total_energy(H, eigenvalues)
            self.logger.info(f"总能量: {total_energy} eV")
            # 更新电子密度
            new_density = self.update_density(eigenvectors)
            # 计算密度变化
            delta_density = np.linalg.norm(new_density - electron_density)
            self.logger.info(f"密度变化: {delta_density}")
            # 判断收敛
            if delta_density < self.convergence_threshold:
                self.logger.info("SCF循环收敛。")
                return True, new_density, total_energy
            # 密度混合
            electron_density = (1 - self.mixing_factor) * electron_density + self.mixing_factor * new_density
        self.logger.warning("SCF循环未能收敛。")
        return False, electron_density, total_energy
    
    def calculate_band_structure(self):
        """
        计算能带结构（简化示例）。
        
        返回:
        -------
        bandstructure : numpy.ndarray
            能带结构数组，形状为 (num_bands, num_kpoints)。
        """
        self.logger.info("计算能带结构...")
        num_bands = self.hamiltonian_builder.num_eigen
        num_kpoints = len(self.kpoints.k_points)
        bandstructure = np.zeros((num_bands, num_kpoints))
        
        for idx, k in enumerate(self.kpoints.k_points):
            self.logger.info(f"计算k点 {idx+1}/{num_kpoints}: {k}")
            # 更新k点
            self.hamiltonian_builder.k_points.current_k_point = k
            # 构建哈密顿量
            H = self.build_hamiltonian(self.initialize_density())
            # 求解本征问题
            eigenvalues, _ = self.solver.solve(H, num_bands)
            #eigenvalues, _ = self.solver.solve()

            # 存储能带
            bandstructure[:, idx] = eigenvalues
        self.logger.info("能带结构计算完成。")
        return bandstructure
