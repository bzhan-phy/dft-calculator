# src/kpoints.py

import numpy as np

class KPoints:
    """
    k点生成类，生成布里渊区中的k点。
    """

    def __init__(self, num_kpoints, lattice_vectors):
        """
        初始化k点生成器。

        参数:
        --------
        num_kpoints : int
            每个维度上的k点数量。
        lattice_vectors : numpy.ndarray
            晶格矢量，形状为 (3, 3)。
        """
        self.num_kpoints = num_kpoints
        self.lattice_vectors = lattice_vectors
        self.k_points = self.generate_k_points()
        self.current_k_point = self.k_points[0]

    def generate_k_points(self):
        """
        生成均匀k点网格。

        返回:
        -------
        k_points : numpy.ndarray
            生成的k点，形状为 (num_total_kpoints, 3)。
        """
        nk = self.num_kpoints
        kx = np.linspace(-0.5, 0.5, nk, endpoint=False)
        ky = np.linspace(-0.5, 0.5, nk, endpoint=False)
        kx, ky = np.meshgrid(kx, ky)
        kz = np.zeros_like(kx)  # 二维材料
        k_points = np.vstack((kx.flatten(), ky.flatten(), kz.flatten())).T
        return k_points
