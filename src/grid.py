# src/grid.py

import numpy as np
from scipy.constants import pi
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Grid:
    """
    网格类，用于处理实空间和G空间的转换，定义FFT相关参数。
    """

    def __init__(self, real_space_grid, lattice_vectors):
        """
        初始化网格。

        参数:
        --------
        real_space_grid : tuple of ints
            实空间格点的数量，例如 (N_x, N_y, N_z)。
        lattice_vectors : numpy.ndarray
            晶格矢量，形状为 (3, 3)。
        """
        self.real_space_grid = real_space_grid
        self.lattice_vectors = lattice_vectors
        self.volume = np.linalg.det(lattice_vectors[:3, :3])
        self.G_vectors = self.generate_G_vectors()
        logger.info(f"生成的G矢量数量: {len(self.G_vectors)}")

    def generate_G_vectors(self):
        """
        生成G向量。

        返回:
        -------
        G_vectors : numpy.ndarray
            G向量数组，形状为 (num_G, 3)。
        """
        # 使用FFT频率生成G向量
        N_x, N_y, N_z = self.real_space_grid
        Gx = fftfreq(N_x, d=1.0) * 2 * pi
        Gy = fftfreq(N_y, d=1.0) * 2 * pi
        Gz = fftfreq(N_z, d=1.0) * 2 * pi
        Gx, Gy, Gz = np.meshgrid(Gx, Gy, Gz, indexing='ij')
        Gx = Gx.flatten()
        Gy = Gy.flatten()
        Gz = Gz.flatten()
        G_vectors = np.vstack((Gx, Gy, Gz)).T
        # 过滤掉过大的G向量，如果需要
        # G_vectors = G_vectors[np.linalg.norm(G_vectors, axis=1) <= G_cutoff]
        return G_vectors
