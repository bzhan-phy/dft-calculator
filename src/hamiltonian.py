# MIT License
# Copyright (c) 2024 [Bangzheng Han]
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy.constants import pi, epsilon_0
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Hamiltonian:
    """
    哈密顿量构建类，用于构建总的哈密顿量矩阵，包括动能、Hartree 势能、
    赝势和交换-相关势能。
    """

    def __init__(self, atomic_structure, k_points, pseudopotentials, exchange_correlation, max_G=10, num_eigen=10, dim=3):
        """
        初始化哈密顿量构建器。

        参数:
        --------
        atomic_structure : AtomicStructure
            原子结构对象，包含晶格矢量和原子位置。
        k_points : KPoints
            k点对象，包含网格信息。
        pseudopotentials : dict
            赝势字典，键为元素符号，值为赝势参数（例如V_pp(G)）。
        exchange_correlation : ExchangeCorrelation
            交换-相关势对象，包含相关势计算方法。
        max_G : int
            最大G矢量的范围。
        num_eigen : int
            要求解的本征值和本征向量的数量。
        dim : int
            系统维度，2表示二维材料，3表示三维材料。
        """
        self.atomic_structure = atomic_structure
        self.k_points = k_points
        self.pseudopotentials = pseudopotentials
        self.exchange_correlation = exchange_correlation
        self.max_G = max_G
        self.dim = dim
        self.G_vectors = self.generate_G_vectors()
        self.num_G = len(self.G_vectors)
        self.num_eigen = num_eigen
        logger.info(f"生成的平面波基组数量: {self.num_G}")

    def generate_G_vectors(self):
        """
        生成平面波基组中的G矢量。

        返回:
        -------
        G_vectors : numpy.ndarray
            G矢量数组，形状为 (num_G, 3)。
        """
        logger.info(f"生成G矢量，系统维度: {self.dim}D")
        Gx = np.arange(-self.max_G, self.max_G + 1)
        Gy = np.arange(-self.max_G, self.max_G + 1)

        if self.dim == 2:
            #Gz = np.zeros_like(Gx)  # 二维材料
            Gx_mesh, Gy_mesh = np.meshgrid(Gx, Gy)
            Gz = np.zeros_like(Gx_mesh.flatten())
            G_vectors = np.vstack((Gx_mesh.flatten(), Gy_mesh.flatten(), Gz.flatten())).T
        elif self.dim == 3:
            Gz = np.arange(-self.max_G, self.max_G + 1)
            Gx_mesh, Gy_mesh, Gz_mesh = np.meshgrid(Gx, Gy, Gz)
            G_vectors = np.vstack((Gx_mesh.flatten(), Gy_mesh.flatten(), Gz_mesh.flatten())).T
        else:
            raise ValueError("维度参数dim必须为2或3。")
        
        logger.info(f"生成的G矢量数量: {G_vectors.shape[0]}")
        return G_vectors

    def kinetic_energy(self):
        """
        计算动能部分的哈密顿量矩阵。

        返回:
        -------
        T : numpy.ndarray
            动能矩阵，形状为 (num_G, num_G)，对角矩阵。
        """
        logger.info("计算动能部分...")
        # 计算 (k + G)^2
        k = self.k_points.current_k_point  # 当前k点
        G_plus_k = self.G_vectors + k  # |G + k|
        kinetic = np.sum(G_plus_k**2, axis=1)
        T = np.diag(kinetic)
        logger.info("动能部分计算完成。")
        return T

    def hartree_potential(self, rho_G):
        """
        计算Hartree势能。

        参数:
        --------
        rho_G : numpy.ndarray
            电子密度的傅里叶变换，形状为 (num_G,)。

        返回:
        -------
        V_H : numpy.ndarray
            Hartree势能矩阵，形状为 (num_G, num_G)，对角矩阵。
        """
        logger.info("计算Hartree势能...")
        V_H = np.zeros_like(rho_G)
        if rho_G is None:
            logger.error("exchange_correlation_potential 被调用时，rho_G 为 None。")
            raise ValueError("rho_G cannot be None in exchange_correlation_potential.")
        
        for i, G in enumerate(self.G_vectors):
            G_norm = np.linalg.norm(G)
            if G_norm != 0:
                V_H[i] = 4 * pi / (G_norm**2) * rho_G[i]
            else:
                V_H[i] = 0.0  # 对于G=0，Hartree势能通常与核电荷中和
        V_H_matrix = np.diag(V_H)
        logger.info("Hartree势能计算完成。")
        return V_H_matrix

    def pseudopotential(self, rho_G):
        """
        计算赝势部分的势能。

        参数:
        --------
        rho_G : numpy.ndarray
            电子密度的傅里叶变换，形状为 (num_G,)。

        返回:
        -------
        V_pp : numpy.ndarray
            赝势势能矩阵，形状为 (num_G, num_G)，对角矩阵。
        """
        logger.info("计算赝势势能...")
        V_pp = np.zeros_like(rho_G)
        for element, params in self.pseudopotentials.items():
            V_pp += params['V_pp']  # 假设V_pp已经是G空间的赝势
        V_pp_matrix = np.diag(V_pp)
        logger.info("赝势势能计算完成。")
        return V_pp_matrix

    def exchange_correlation_potential(self, rho_real):
        """
        计算交换-相关势能。

        参数:
        --------
        rho_real : numpy.ndarray
            实空间中的电子密度，形状为 (num_real_grid,).

        返回:
        -------
        V_xc : numpy.ndarray
            交换-相关势能矩阵，形状为 (num_G, num_G)，对角矩阵。
        """
        logger.info("计算交换-相关势能...")
        if rho_real is None:
            logger.error("exchange_correlation_potential 被调用时，rho_real 为 None。")
            raise ValueError("rho_real cannot be None in exchange_correlation_potential.")
        
        rho_avg = np.mean(rho_real)
        V_xc_val = self.exchange_correlation.calculate_potential(rho_avg)
        V_xc = np.full(self.num_G, V_xc_val)
        V_xc_matrix = np.diag(V_xc)
        logger.info(f"交换-相关势能计算完成: V_xc = {V_xc_val}")
        return V_xc_matrix

    def total_hamiltonian(self, rho_real, rho_G):
        """
        构建总的哈密顿量矩阵。

        参数:
        --------
        rho_real : numpy.ndarray
            实空间中的电子密度，形状为 (num_real_grid,).
        rho_G : numpy.ndarray
            电子密度的傅里叶变换，形状为 (num_G,).

        返回:
        -------
        H : numpy.ndarray
            总哈密顿量矩阵，形状为 (num_G, num_G)。
        """
        logger.info("构建总哈密顿量...")
        T = self.kinetic_energy()
        V_H = self.hartree_potential(rho_G)
        V_pp = self.pseudopotential(rho_G)
        V_xc = self.exchange_correlation_potential(rho_real)
        H = T + V_H + V_pp + V_xc
        logger.info("总哈密顿量构建完成。")
        return H
