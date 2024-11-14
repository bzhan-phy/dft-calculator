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
    N = H.shape[0]

    if scipy.sparse.issparse(H):
        if num_eigen > N:
            raise ValueError(f"请求的本征值数量 k={num_eigen} 超过矩阵维度 N={N}。")
        elif num_eigen == N:
            logger.info("请求的本征值数量 k = 矩阵维度 N，转换为稠密矩阵并使用 scipy.linalg.eigh。")
            H_dense = H.toarray()
            eigenvalues, eigenvectors = scipy.linalg.eigh(H_dense)
        else:
            logger.info("哈密顿量为稀疏矩阵，使用稀疏求解器。")
            try:
                eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(H, k=num_eigen, which=which)
            except (RuntimeWarning, TypeError) as e:
                logger.warning(f"遇到异常 {e}，尝试使用稠密求解器。")
                H_dense = H.toarray()
                eigenvalues, eigenvectors = scipy.linalg.eigh(H_dense)
    else:
        logger.info("哈密顿量为稠密矩阵，使用稠密求解器。")
        eigenvalues, eigenvectors = scipy.linalg.eigh(H)
        if num_eigen > N:
            raise ValueError(f"请求的本征值数量 k={num_eigen} 超过矩阵维度 N={N}。")

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
