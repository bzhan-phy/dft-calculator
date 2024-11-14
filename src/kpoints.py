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

class KPoints:
    """
    k点生成类，生成布里渊区中的k点。
    """

    def __init__(self, num_kpoints, lattice_vectors, dim=3):
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
        self.dim = dim
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
        if self.dim == 2:
            kx = np.linspace(-0.5, 0.5, nk, endpoint=False)
            ky = np.linspace(-0.5, 0.5, nk, endpoint=False)
            kx, ky = np.meshgrid(kx, ky)
            kz = np.zeros_like(kx)  # 二维材料
            k_points = np.vstack((kx.flatten(), ky.flatten(), kz.flatten())).T
        elif self.dim == 3:
            kx = np.linspace(-0.5, 0.5, nk, endpoint=False)
            ky = np.linspace(-0.5, 0.5, nk, endpoint=False)
            kz = np.linspace(-0.5, 0.5, nk, endpoint=False)
            kx, ky, kz = np.meshgrid(kx, ky, kz)
            k_points = np.vstack((kx.flatten(), ky.flatten(), kz.flatten())).T
        else:
            raise ValueError("维度参数dim必须为2或3。")
        return k_points
