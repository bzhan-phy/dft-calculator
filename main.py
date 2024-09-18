
import json
import logging
import numpy as np
import os

from src.atomic_structure import AtomicStructure
from src.kpoints import KPoints
from src.exchange_correlation import ExchangeCorrelation
from src.hamiltonian import Hamiltonian
from src.solver import KS_Solver
from src.scf import SCF


def setup_logging():
    """
    设置日志记录格式和级别。
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
        
    )


def load_input(file_path):
    """
    从JSON文件加载输入参数。
    
    参数:
    --------
    file_path : str
        JSON文件的路径。
        
    返回:
    -------
    data : dict
        加载的数据字典。
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def main():
    # 设置日志
    setup_logging()
    logger = logging.getLogger("Main")
    
    # 定义输入和输出路径
    structure_input_path = os.path.join('input', 'structure_input.json')
    parameters_input_path = os.path.join('input', 'parameters.json')
    
    # 读取输入文件
    logger.info("读取输入文件...")
    structure_data = load_input(structure_input_path)
    parameters_data = load_input(parameters_input_path)
    logger.info("输入文件读取完成。")
    
    # 初始化原子结构
    logger.info("初始化原子结构...")
    lattice_vectors = np.array(structure_data['lattice_vectors'])
    atomic_positions = structure_data['atomic_positions']
    atomic_symbols = structure_data['atomic_symbols']
    atomic_structure = AtomicStructure(lattice_vectors, atomic_positions, atomic_symbols)
    logger.info(f"原子数量: {atomic_structure.get_num_atoms()}")
    
    # 初始化k点网格
    logger.info("初始化k点网格...")
    num_kpoints = parameters_data['kpoints']['num_kpoints']
    kpoints = KPoints(num_kpoints, lattice_vectors)
    logger.info(f"总k点数: {len(kpoints.k_points)}")
    
    # 初始化赝势
    logger.info("初始化赝势...")
    pseudopotentials = parameters_data['pseudopotentials']  # 期望为字典格式，键为元素符号
    logger.info(f"使用的赝势元素: {list(pseudopotentials.keys())}")
    
    # 初始化交换-相关势
    logger.info("初始化交换-相关势...")
    xc_functional = parameters_data['exchange_correlation']['functional']
    exchange_correlation = ExchangeCorrelation(functional=xc_functional)
    
    # 初始化哈密顿量构建器
    logger.info("初始化哈密顿量构建器...")
    hamiltonian_builder = Hamiltonian(
        atomic_structure=atomic_structure,
        k_points=kpoints,
        pseudopotentials=pseudopotentials,
        exchange_correlation=exchange_correlation
    )
    
    # 读取SCF参数
    scf_params = parameters_data['scf']
    max_iterations = scf_params.get('max_iterations', 100)
    convergence_threshold = scf_params.get('convergence_threshold', 1e-4)
    mixing_factor = scf_params.get('mixing_factor', 0.3)
    
    # 初始化自洽场（SCF）循环
    logger.info("初始化自洽场（SCF）循环...")
    scf = SCF(
        hamiltonian_builder=hamiltonian_builder,
        kpoints=kpoints,
        solver=KS_Solver,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        mixing_factor=mixing_factor
    )
    
    # 执行SCF循环
    logger.info("开始执行SCF循环...")
    converged, electron_density, total_energy = scf.run()
    
    if converged:
        logger.info("SCF循环收敛成功。")
        logger.info(f"总能量: {total_energy} eV")
    else:
        logger.warning("SCF循环未能在最大迭代次数内收敛。")
    
    # 保存结果
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    charge_density_path = os.path.join(output_dir, 'charge_density.dat')
    bandstructure_path = os.path.join(output_dir, 'bandstructure.dat')
    
    logger.info("保存结果...")
    np.savetxt(charge_density_path, electron_density)
    # 这里假设scf.run()返回了电子密度和总能量，你可以根据实际情况调整
    # 能带结构计算和保存需要在SCF收敛后进行，可能需要额外的步骤
    
    # 计算能带结构
    logger.info("计算能带结构...")
    bandstructure = scf.calculate_band_structure()
    np.savetxt(bandstructure_path, bandstructure)
    logger.info("能带结构计算完成。")
    
    # 绘制能带图（可选）
    logger.info("绘制能带图...")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    for band in bandstructure.T:
        plt.plot(band, color='b')
    plt.xlabel('k Point')
    plt.ylabel('Energy (eV)')
    plt.title('Band Structure')
    plt.savefig(os.path.join('output', 'bandstructure.png'))
    plt.close()
    logger.info("能带结构图已保存。")
    
    logger.info("计算完成。")

if __name__ == "__main__":
    main()