
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

class AtomicStructure:
    """
    原子结构类，包含晶格常数和原子位置。
    """

    def __init__(self, lattice_vectors, atomic_positions, atomic_symbols):
        """
        初始化原子结构。

        参数:
        --------
        lattice_vectors : numpy.ndarray
            晶格矢量，形状为 (3, 3)。
        atomic_positions : list of lists or tuples
            每个原子的坐标，单位为晶格常数。
        atomic_symbols : list of str
            每个原子的元素符号。
        """
        self.lattice_vectors = lattice_vectors
        self.atomic_positions = np.array(atomic_positions)
        self.atomic_symbols = atomic_symbols

    def get_num_atoms(self):
        """
        获取原子数量。

        返回:
        -------
        num_atoms : int
            原子数量。
        """
        return len(self.atomic_positions)
