# tests/benchmark_solver.py

import time
import numpy as np
from src.solver import KS_Solver

def benchmark_solver():
    matrix_sizes = [100, 500, 1000, 2000]
    for size in matrix_sizes:
        H = np.random.rand(size, size)
        H = (H + H.T) / 2  # 保证矩阵是对称的
        num_eigen = 10  # 要求解的本征值数量
        
        solver = KS_Solver(H, num_eigen)
        
        start_time = time.time()
        eigenvalues, eigenvectors = solver.solve(H, num_eigen)
        end_time = time.time()
        
        print(f"Matrix size: {size}, Time: {end_time - start_time:.4f} s")

if __name__ == '__main__':
    benchmark_solver()
