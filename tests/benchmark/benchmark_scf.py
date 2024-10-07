# tests/benchmark_scf.py

import time
import numpy as np
from src.atomic_structure import AtomicStructure
from src.kpoints import KPoints
from src.exchange_correlation import ExchangeCorrelation
from src.hamiltonian import Hamiltonian
from src.solver import KS_Solver
from src.scf import SCF

def benchmark_scf():
    lattice_vectors = np.eye(3)
    atomic_positions = [[0, 0, 0]]
    atomic_symbols = ['H']
    atomic_structure = AtomicStructure(lattice_vectors, atomic_positions, atomic_symbols)
    exchange_correlation = ExchangeCorrelation(functional='LDA')
    pseudopotentials = {'H': {'V_pp': 0.0}}
    
    max_G_values = [1, 2, 5]
    num_kpoints_values = [1, 4, 8]
    mixing_factors = [0.1, 0.3, 0.5]
    
    for max_G in max_G_values:
        for num_kpoints in num_kpoints_values:
            for mixing_factor in mixing_factors:
                kpoints = KPoints(num_kpoints, lattice_vectors, dim=3)
                hamiltonian = Hamiltonian(
                    atomic_structure=atomic_structure,
                    k_points=kpoints,
                    pseudopotentials=pseudopotentials,
                    exchange_correlation=exchange_correlation,
                    max_G=max_G,
                    num_eigen=1,
                    dim=3
                )
                scf = SCF(
                    hamiltonian_builder=hamiltonian,
                    kpoints=kpoints,
                    solver_class=KS_Solver,
                    max_iterations=100,
                    convergence_threshold=1e-4,
                    mixing_factor=mixing_factor
                )
                
                start_time = time.time()
                converged, final_rho_real, total_energy = scf.run()
                end_time = time.time()
                
                print(f"max_G: {max_G}, num_kpoints: {num_kpoints}, mixing_factor: {mixing_factor}")
                print(f"Converged: {converged}, Total Energy: {total_energy:.6f}, Time: {end_time - start_time:.4f} s\n")

if __name__ == '__main__':
    benchmark_scf()
