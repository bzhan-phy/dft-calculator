# tests/benchmark_bandstructure.py

import numpy as np
from src.atomic_structure import AtomicStructure
from src.kpoints import KPoints
from src.exchange_correlation import ExchangeCorrelation
from src.hamiltonian import Hamiltonian
from src.solver import KS_Solver
from src.scf import SCF
import matplotlib.pyplot as plt
# 验证能带计算正确性
def benchmark_bandstructure():
    # 定义简单的二维晶格
    lattice_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 10]])  # 2D
    atomic_positions = [[0, 0, 0]]
    atomic_symbols = ['C']
    atomic_structure = AtomicStructure(lattice_vectors, atomic_positions, atomic_symbols)
    exchange_correlation = ExchangeCorrelation(functional='LDA')
    pseudopotentials = {'C': {'V_pp': 0.0}}
    
    kpoints = KPoints(20, lattice_vectors, dim=2)
    hamiltonian = Hamiltonian(
        atomic_structure=atomic_structure,
        k_points=kpoints,
        pseudopotentials=pseudopotentials,
        exchange_correlation=exchange_correlation,
        max_G=1,
        num_eigen=5,
        dim=2
    )
    scf = SCF(
        hamiltonian_builder=hamiltonian,
        kpoints=kpoints,
        solver_class=KS_Solver,
        max_iterations=50,
        convergence_threshold=1e-4,
        mixing_factor=0.3
    )
    converged, final_rho_real, total_energy = scf.run()
    
    bandstructure = scf.calculate_band_structure()
    num_bands = bandstructure.shape[0]
    
    # 绘制能带结构
    plt.figure(figsize=(8,6))
    k_point_indices = np.arange(len(kpoints.k_points))
    for i in range(num_bands):
        plt.plot(k_point_indices, bandstructure[i], label=f'Band {i+1}')
    plt.xlabel('k Point Index')
    plt.ylabel('Energy (eV)')
    plt.title('Band Structure')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    benchmark_bandstructure()
