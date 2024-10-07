# tests/benchmark_hamiltonian.py

import time
import numpy as np
from src.atomic_structure import AtomicStructure
from src.kpoints import KPoints
from src.exchange_correlation import ExchangeCorrelation
from src.hamiltonian import Hamiltonian

def benchmark_hamiltonian_construction():
    lattice_vectors = np.eye(3)
    atomic_positions = [[0, 0, 0]]
    atomic_symbols = ['H']
    atomic_structure = AtomicStructure(lattice_vectors, atomic_positions, atomic_symbols)
    kpoints = KPoints(1, lattice_vectors, dim=3)
    exchange_correlation = ExchangeCorrelation(functional='LDA')
    pseudopotentials = {'H': {'V_pp': 0.0}}

    max_G_values = [1, 2, 5, 10, 15]
    for max_G in max_G_values:
        hamiltonian = Hamiltonian(
            atomic_structure=atomic_structure,
            k_points=kpoints,
            pseudopotentials=pseudopotentials,
            exchange_correlation=exchange_correlation,
            max_G=max_G,
            num_eigen=1,
            dim=3
        )
        start_time = time.time()
        rho_real = np.ones(hamiltonian.num_G)
        rho_G = rho_real.copy()
        H = hamiltonian.total_hamiltonian(rho_real, rho_G)
        end_time = time.time()
        print(f"max_G: {max_G}, num_G: {hamiltonian.num_G}, Time: {end_time - start_time:.4f} s")

if __name__ == '__main__':
    benchmark_hamiltonian_construction()
