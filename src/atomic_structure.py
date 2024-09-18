# src/atomic_structure.py

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
