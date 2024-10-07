# tests/test_solver.py

import unittest
import numpy as np
from scipy.sparse import csr_matrix, diags
from src.solver import KS_Solver

class TestSolver(unittest.TestCase):

    def test_solver_dense(self):
        H = np.array([[1, 0], [0, 2]])
        num_eigen = 2
        solver = KS_Solver(H, num_eigen)
        eigenvalues, eigenvectors = solver.solve(H, num_eigen)
        np.testing.assert_array_almost_equal(eigenvalues, [1, 2])
        # 检查本征向量是否正交
        self.assertTrue(np.allclose(np.dot(eigenvectors.T, eigenvectors), np.eye(num_eigen)))

    def test_solver_sparse(self):
        # 使用更大规模的稀疏矩阵
        size = 100
        diagonal = np.arange(1, size + 1)
        H_sparse = diags(diagonal, 0, format='csr')
        num_eigen = 10  # 确保 k < N

        solver = KS_Solver(H_sparse, num_eigen)
        eigenvalues, eigenvectors = solver.solve(H_sparse, num_eigen)
        expected_eigenvalues = np.arange(1, num_eigen + 1)
        np.testing.assert_array_almost_equal(eigenvalues, expected_eigenvalues)
        # 检查本征向量是否正交
        self.assertTrue(np.allclose(np.dot(eigenvectors.T, eigenvectors), np.eye(num_eigen)))

    def test_solver_sparse_k_equals_n(self):
        # 测试当 k = N 时，是否正确切换到稠密求解器
        size = 2
        H_sparse = csr_matrix(np.diag([1, 2]))
        num_eigen = 2  # k = N

        solver = KS_Solver(H_sparse, num_eigen)
        eigenvalues, eigenvectors = solver.solve(H_sparse, num_eigen)
        expected_eigenvalues = np.array([1, 2])
        np.testing.assert_array_almost_equal(eigenvalues, expected_eigenvalues)
        # 检查本征向量是否正交
        self.assertTrue(np.allclose(np.dot(eigenvectors.T, eigenvectors), np.eye(num_eigen)))

    def test_solver_sparse_k_greater_than_n(self):
        # 测试当 k > N 时，是否抛出适当的错误或处理
        size = 2
        H_sparse = csr_matrix(np.diag([1, 2]))
        num_eigen = 3  # k > N，应触发异常或被正确处理

        solver = KS_Solver(H_sparse, num_eigen)
        with self.assertRaises(ValueError):
            eigenvalues, eigenvectors = solver.solve(H_sparse, num_eigen)



if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)