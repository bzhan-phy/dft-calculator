# src/solver.py

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def solve_eigenproblem(H, num_eigen, which='SM'):
    """
    求解给定哈密顿量矩阵 H 的前 num_eigen 个本征值和本征向量。

    参数:
    --------
    H : numpy.ndarray 或 scipy.sparse.spmatrix
        哈密顿量矩阵，可以是稠密矩阵或稀疏矩阵。
    num_eigen : int
        要求解的本征值和本征向量的数量。
    which : str
        指定要计算的本征值类型，默认为 'SM' (Smallest Magnitude)。

        常用选项包括：
        - 'SM' : 最小模长
        - 'LM' : 最大模长
        - 'SR' : 最接近右侧指定值
        - 'LR' : 最接近左侧指定值
        - 'SI' : 最小虚部
        - 'LI' : 最大虚部

    返回:
    -------
    eigenvalues : numpy.ndarray
        计算得到的本征值，按升序排列。
    eigenvectors : numpy.ndarray
        对应的本征向量，每一列对应一个本征值。
    """
    logger.info("开始求解本征问题...")
    if scipy.sparse.issparse(H):
        logger.info("哈密顿量为稀疏矩阵，使用稀疏求解器。")
        # 使用稀疏求解器 (适用于对称或厄米矩阵)
        #eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(H, k=num_eigen, which=which)
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(H, k=num_eigen, which=which)
    else:
        logger.info("哈密顿量为稠密矩阵，使用稠密求解器。")
        # 使用稠密求解器
        eigenvalues, eigenvectors = scipy.linalg.eigh(H)
        # 选取前 num_eigen 个本征值和本征向量
        eigenvalues = eigenvalues[:num_eigen]
        eigenvectors = eigenvectors[:, :num_eigen]
    
    logger.info("本征问题求解完成。")
    return eigenvalues, eigenvectors

class KS_Solver:
    """
    Kohn-Sham 方程求解器类，用于求解给定哈密顿量的本征值问题。
    """

    def __init__(self, H, num_eigen, which='SM'):
        """
        初始化求解器。

        参数:
        --------
        H : numpy.ndarray 或 scipy.sparse.spmatrix
            哈密顿量矩阵。
        num_eigen : int
            要求解的本征值和本征向量的数量。
        which : str
            指定要计算的本征值类型，参见 solve_eigenproblem 函数。
        """
        self.H = H
        self.num_eigen = num_eigen
        self.which = which

    def solve(self, H, num_eigen):
        """
        执行本征值求解。

        返回:
        -------
        eigenvalues : numpy.ndarray
            计算得到的本征值。
        eigenvectors : numpy.ndarray
            对应的本征向量。
        """
        logger = logging.getLogger("KS_Solver")
        logger.info(f"使用 KS_Solver 求解前 {self.num_eigen} 个本征值和本征向量。")
        eigenvalues, eigenvectors = solve_eigenproblem(H, num_eigen, which=self.which)
        #eigenvalues, eigenvectors = solve_eigenproblem(H, num_eigen, which=self.which)
        return eigenvalues, eigenvectors
