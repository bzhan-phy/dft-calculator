# src/hamiltonian.py

import numpy as np
from scipy.constants import electron_volt, pi
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Hamiltonian:
    """
    哈密顿量构建类，用于构建总的哈密顿量矩阵。
    """

    def __init__(self, atomic_structure, k_points, pseudopotentials, exchange_correlation, max_G=10, num_eigen=10):
        """
        初始化哈密顿量构建器。

        参数:
        --------
        atomic_structure : AtomicStructure
            原子结构对象，包含晶格常数和原子位置。
        k_points : KPoints
            k点对象，包含网格信息。
        pseudopotentials : dict
            赝势字典，键为元素符号，值为赝势参数。
        exchange_correlation : ExchangeCorrelation
            交换-相关势对象，包含相关势计算方法。
        max_G : int
            最大G矢量的范围。
        num_eigen : int
            要求解的本征值和本征向量的数量。
        """
        self.atomic_structure = atomic_structure
        self.k_points = k_points
        self.pseudopotentials = pseudopotentials
        self.exchange_correlation = exchange_correlation
        self.max_G = max_G
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
        # 对于二维材料，我们简化为Gz = 0
        Gx = np.arange(-self.max_G, self.max_G + 1)
        Gy = np.arange(-self.max_G, self.max_G + 1)
        Gx, Gy = np.meshgrid(Gx, Gy)
        Gx = Gx.flatten()
        Gy = Gy.flatten()
        Gz = np.zeros_like(Gx)  # 二维材料
        G_vectors = np.vstack((Gx, Gy, Gz)).T
        return G_vectors

    def kinetic_energy(self):
        """
        计算动能部分的哈密顿量矩阵。

        返回:
        -------
        T : numpy.ndarray
            动能矩阵，形状为 (num_G, num_G)。
        """
        logger.info("计算动能部分...")
        # 动能算符在平面波基组下为 (ħ²|G+k|²)/(2m)
        # 这里我们使用简化的单位，使得 ħ²/(2m) = 1
        # 真实计算中需要考虑实际的单位和缩放因子
        k = self.k_points.current_k_point  # 当前k点
        G_plus_k = self.G_vectors + k  # |G + k|
        kinetic = np.sum(G_plus_k**2, axis=1)
        T = np.diag(kinetic)
        logger.info("动能部分计算完成。")
        return T

    def potential_energy(self, electron_density):
        """
        计算势能部分的哈密顿量矩阵，包括赝势和交换-相关势。

        参数:
        --------
        electron_density : numpy.ndarray
            电子密度数组，形状为 (num_G,)。

        返回:
        -------
        V : numpy.ndarray
            势能矩阵，形状为 (num_G, num_G)。
        """
        logger.info("计算势能部分...")
        # 简化处理，假设势能为对角矩阵
        # 真实计算中，赝势和交换-相关势可能有复杂的G依赖
        # 此处使用赝势和交换-相关势的G=0分量
        V_ps = 0.0
        for element, params in self.pseudopotentials.items():
            # 简化赝势为常数，这里仅为示例
            V_ps += params['V0']

        V_xc = self.exchange_correlation.calculate_potential(electron_density)

        V_total = V_ps + V_xc  # 总势能
        V = np.diag([V_total] * self.num_G)
        logger.info("势能部分计算完成。")
        return V

    def total_hamiltonian(self, electron_density):
        """
        构建总的哈密顿量矩阵。

        参数:
        --------
        electron_density : numpy.ndarray
            电子密度数组，形状为 (num_G,)。

        返回:
        -------
        H : numpy.ndarray
            总哈密顿量矩阵，形状为 (num_G, num_G)。
        """
        logger.info("构建总哈密顿量...")
        T = self.kinetic_energy()
        V = self.potential_energy(electron_density)
        H = T + V
        logger.info("总哈密顿量构建完成。")
        return H
