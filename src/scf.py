# src/scf.py

import numpy as np
import logging
from scipy.fft import fftn, ifftn, fftshift
from src.solver import KS_Solver
import math

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SCF:
    """
    自洽场（SCF）循环控制类，用于迭代计算电子密度和总能量。
    """

    def __init__(self, hamiltonian_builder, kpoints, solver_class, max_iterations=100, convergence_threshold=1e-4, mixing_factor=0.3):
        """
        初始化SCF循环。

        参数:
        --------
        hamiltonian_builder : Hamiltonian
            哈密顿量构建器对象。
        kpoints : KPoints
            k点对象。
        solver_class : class
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
        self.solver_class = solver_class
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.mixing_factor = mixing_factor
        self.logger = logging.getLogger("SCF")
        self.final_rho_real = None  # 用于存储收敛后的密度


    def initialize_density(self):
        """
        初始化电子密度，使用原子密度叠加的近似。

        返回:
        -------
        rho_real : numpy.ndarray
            初始化的实空间电子密度，形状为 (num_real_grid,).
        """
        # 假设每个原子贡献一个均匀电子密度
        # 真实实现中应该基于原子赝势进行初始化
        self.logger.info("初始化电子密度...")
        num_G = self.hamiltonian_builder.num_G
        rho_G = np.zeros(num_G, dtype=np.complex128)
        # 假设初始密度为高斯分布或其他合理猜测
        # 这里简化为仅G=0分量非零
        # rho_G[0] = 1.0  # 总电子数的傅里叶零分量
        rho_G[0] = self.hamiltonian_builder.atomic_structure.get_num_atoms()
        # 反傅里叶变换到实空间
        rho_real = np.real(ifftn(rho_G))
        self.logger.info("电子密度初始化完成。")
        return rho_real

    def rho_G_from_real(self, rho_real):
        """
        将实空间电子密度转换到G空间。

        参数:
        --------
        rho_real : numpy.ndarray
            实空间电子密度，形状为 (num_real_grid,).

        返回:
        -------
        rho_G : numpy.ndarray
            密度的傅里叶变换，形状为 (num_G,).
        """
        rho_G_full = fftn(rho_real)
        # 由于简化假设，只有特定的G向量
        # 需要取对应的G点的傅里叶系数
        # 假设rho_G与G_vectors中各G点一一对应
        # 在实际实现中，需要映射实空间网格到G向量
        # 这里简化为取平均
        rho_G = rho_G_full.flatten()
        return rho_G


    def compute_total_energy(self, H, eigenvalues, rho_G, rho_real):
        """
        计算总能量，包括动能、Hartree能量、赝势能量、
        交换-相关能量和核能量。

        参数:
        --------
        H : numpy.ndarray
            总哈密顿量矩阵，形状为 (num_G, num_G)。
        eigenvalues : numpy.ndarray
            本征值数组，形状为 (num_eigen,).
        rho_G : numpy.ndarray
            电子密度的傅里叶变换，形状为 (num_G,).
        rho_real : numpy.ndarray
            实空间电子密度，形状为 (num_real_grid,).

        返回:
        -------
        E_total : float
            总能量。
        """
        # Hartree 能量
        E_H = 0.5 * np.sum(np.real(rho_G) * np.real(self.hamiltonian_builder.hartree_potential(rho_G).diagonal()))
        
        # 潜能能量 (V_pp + V_xc)
        V_potential = self.hamiltonian_builder.pseudopotential(rho_G).diagonal() + self.hamiltonian_builder.exchange_correlation_potential(rho_real).diagonal()
        E_V = np.sum(np.real(rho_G) * np.real(V_potential))
        
        # 交换-相关能量
        # 已经包括在 V_xc 中，可能需要减少重复计算
        # 假设 E_xc 单独计算
        E_xc = self.exchange_correlation_energy(np.mean(rho_real))
        
        # 总能量
        E_total = E_H + E_V + E_xc
        return E_total

    def exchange_correlation_energy(self, rho_avg):
        """
        计算交换-相关能量。

        参数:
        --------
        rho_avg : float
            平均电子密度。

        返回:
        -------
        E_xc : float
            交换-相关能量。
        """
        # 使用LDA交换能量密度
        # E_x = - (3/4) * (3/pi)**(1/3) * rho^(4/3)
        # 使用LDA相关能量密度 (Perdew-Zunger)
        # 为了简化，这里只实现交换部分
        E_x = - (3/4) * (3/math.pi)** (1/3) * rho_avg**(4/3)
        # 相关部分需要实现，这里简化为0
        E_c = 0.0
        E_xc = E_x + E_c
        self.logger.info(f"交换-相关能量: E_xc = {E_xc}")
        return E_xc

    def update_density(self, eigenvectors, num_eigen):
        """
        根据本征向量更新电子密度。

        参数:
        --------
        eigenvectors : numpy.ndarray
            本征向量矩阵，形状为 (num_G, num_eigen)。
        num_eigen : int
            占据态数量。

        返回:
        -------
        new_rho_G : numpy.ndarray
            更新后的电子密度的傅里叶变换，形状为 (num_G,).
        """
        # 电子密度 rho(r) = sum_{n=1}^{N} |C_n(r)|^2
        # 在G空间，rho_G = sum_n C_n(G') * C_n^*(G' - G)
        # 简化处理：仅考虑对角部分
        self.logger.info("更新电子密度...")
        new_rho_G = np.sum(np.abs(eigenvectors[:, :num_eigen])** 2, axis=1)
        self.logger.info("电子密度更新完成。")
        return new_rho_G

    def run(self):
        """
        执行SCF循环。

        返回:
        -------
        converged : bool
            是否收敛。
        final_rho_real : numpy.ndarray
            最终的实空间电子密度，形状为 (num_real_grid,).
        total_energy : float
            最终的总能量。
        """
        self.logger.info("开始SCF循环...")
        # 初始化电子密度
        rho_real = self.initialize_density()
        rho_G = self.rho_G_from_real(rho_real)

        # 初始化混合电子密度用于混合
        rho_G_old = rho_G.copy()

        # 记录每次能量变化，单元测试
        self.iteration_energies = []

        for iteration in range(1, self.max_iterations + 1):
            self.logger.info(f"SCF迭代 {iteration}")
            # 构建哈密顿量
            H = self.hamiltonian_builder.total_hamiltonian(rho_real,rho_G)
            # 求解本征问题
            # num_eigen = self.hamiltonian_builder.num_eigen

            # 求解Kohn-Sham方程
            solver = self.solver_class(H,self.hamiltonian_builder.num_eigen)
            eigenvalues, eigenvectors = solver.solve(H, self.hamiltonian_builder.num_eigen)

            # 计算总能量
            E_total = self.compute_total_energy(H, eigenvalues, rho_G, rho_real)

            # 记录每次能量变化，单元测试
            self.iteration_energies.append(E_total)

            self.logger.info(f"总能量: {E_total:.6f} eV")

            # 更新电子密度
            new_rho_G = self.update_density(eigenvectors, self.hamiltonian_builder.num_eigen)

            # 混合电子密度
            mixed_rho_G = (1 - self.mixing_factor) * rho_G_old + self.mixing_factor * new_rho_G

            # 计算密度变化
            delta_rho = np.linalg.norm(mixed_rho_G - rho_G)
            self.logger.info(f"电子密度变化: {delta_rho:.6e}")

            # 判断收敛
            if delta_rho < self.convergence_threshold:
                self.logger.info("SCF循环收敛。")
                # 反傅里叶变换得到实空间密度
                final_rho_real = np.real(ifftn(mixed_rho_G))
                self.final_rho_real = np.real(ifftn(mixed_rho_G))
                return True, final_rho_real, E_total

            # 准备下一次迭代
            rho_G_old = rho_G.copy()
            rho_G = mixed_rho_G.copy()

        self.logger.warning("SCF循环未能在最大迭代次数内收敛。")
        final_rho_real = np.real(ifftn(rho_G))
        self.final_rho_real = np.real(ifftn(mixed_rho_G))
        return False, final_rho_real, E_total

    def calculate_band_structure(self):
        """
        计算能带结构。

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
            # 更新当前k点
            self.hamiltonian_builder.k_points.current_k_point = k
            # 使用收敛后的电子密度
            rho_real = self.final_rho_real  # 使用收敛后的电子密度
            rho_G = self.rho_G_from_real(rho_real)
            # 构建哈密顿量
            # print(rho_G)
            H = self.hamiltonian_builder.total_hamiltonian(rho_real, rho_G)
            # 求解本征值
            solver = self.solver_class(H, num_bands)
            eigenvalues, _ = solver.solve(H, self.hamiltonian_builder.num_eigen)
            # 存储能带
            bandstructure[:, idx] = eigenvalues

        self.logger.info("能带结构计算完成。")
        return bandstructure
