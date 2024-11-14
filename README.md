# 二维材料异质结能带结构模拟器 | 2D Material Heterojunction Bandstructure Simulator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/)
[![GitHub issues](https://img.shields.io/github/issues/bzhan-phy/dft-calculator)](https://github.com/bzhan-phy/dft-calculator/issues)

## 🌟 项目简介 | Project Overview

本项目是一个开源的密度泛函理论（DFT）计算工具，专注于模拟和计算二维材料异质结的电子结构和能带。

This open-source Density Functional Theory (DFT) computational tool is designed for simulating electronic structures and band structures of two-dimensional material heterojunctions.

## ✨ 主要特性 | Key Features

- 🔬 二维材料结构模拟
- 🧮 自洽场（SCF）迭代计算
- 📊 能带结构计算与可视化
- 🧩 模块化设计，易于扩展
- 💻 支持局域密度近似（LDA）交换-相关势

## 🛠 环境依赖 | Dependencies

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

## 📦 安装步骤 | Installation

1. 克隆仓库 | Clone the repository
```bash
git clone https://github.com/bzhan-phy/dft-calculator.git
cd dft-calculator
```

2. 安装依赖 | Install dependencies
```bash
pip install -r requirements.txt
```

## 🚀 快速开始 | Quick Start

### 配置输入文件 | Configure Input Files

#### 结构参数文件 | Structure Parameters (`input/structure.json`)

```json
{
    "dim": 2,
    "lattice_vectors": [
        [2.46, 0.0, 0.0],
        [1.23, 2.13, 0.0],
        [0.0, 0.0, 10.0]
    ],
    "atomic_positions": [
        [0.0, 0.0, 0.0],
        [0.82, 1.42, 0.0]
    ],
    "atomic_symbols": [
        "C",
        "C"
    ]
}
```

#### 计算参数文件 | Calculation Parameters (`input/parameters.json`)

```json
{
    "max_G": 10,
    "kpoints": {
        "num_kpoints": 8
    },
    "pseudopotentials": {
        "C": {
            "V_pp": -10.0  
        }
    },
    "exchange_correlation": {
        "functional": "LDA"
    },
    "scf": {
        "max_iterations": 100,
        "convergence_threshold": 1e-4,
        "mixing_factor": 0.3
    }
}
```

### 运行程序 | Run the Program

```bash
python main.py
```

## 📂 输出文件 | Output Files

- `output/charge_density.dat`：电子密度数据 | Electron Density Data
- `output/bandstructure.dat`：能带结构数据 | Bandstructure Data
- `output/bandstructure.png`：能带结构图像 | Bandstructure Visualization

## 🧠 理论背景 | Theoretical Background

本项目基于密度泛函理论（DFT）的核心计算步骤：
1. 初始化电子密度
2. 构建哈密顿量
3. 自洽场（SCF）迭代
4. 能带结构计算

## 🗂 功能模块 | Functional Modules

- `atomic_structure.py`：原子结构处理
- `exchange_correlation.py`：交换-相关势计算
- `hamiltonian.py`：哈密顿量构建
- `kpoints.py`：k点网格生成
- `scf.py`：自洽场迭代控制
- `solver.py`：本征值问题求解

## ⚠️ 局限性 | Limitations

当前版本是一个简化的DFT模拟器，存在以下局限：
- 仅支持局域密度近似（LDA）
- 使用简单的平面波基组
- 赝势模型较为简单
- 仅适用于理想的二维材料结构

## 🔬 未来工作 | Future Work

- 实现更先进的交换-相关泛函
- 支持更复杂的赝势模型
- 添加更多材料体系的支持
- 优化计算性能

## 🤝 贡献指南 | Contributing

欢迎通过以下方式参与项目：
- 报告Bug
- 提交改进建议
- 发起Pull Request

### 开发流程 | Development Workflow

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证 | License

本项目采用MIT许可证。详细信息请参见 `LICENSE` 文件。

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 📚 引用 | Citation

如果您在学术工作中使用本项目，请引用：

```bibtex
@software{dft_calculator,
  title = {{2D Material Heterojunction Bandstructure Simulator}},
  author = {Your Name},
  year = {2024},
  version = {1.0.0},
  doi = {10.5281/zenodo.XXXXXX},
  url = {https://github.com/bzhan-phy/dft-calculator},
  license = {MIT}
}
```

## 🙏 致谢 | Acknowledgements

感谢所有为项目贡献的开发者和研究者。

Thanks to all developers and researchers who contribute to this project.

## 📞 联系方式 | Contact

- 项目负责人：[Your Name]
- 电子邮件：[your.email@example.com]
- GitHub: [https://github.com/bzhan-phy](https://github.com/bzhan-phy)

---

**注意**：本项目仅供学术研究和教育目的。对于精确的材料性质计算，建议使用成熟的商业或开源DFT软件包。

**Note**: This project is for academic research and educational purposes only. For precise material property calculations, it is recommended to use mature commercial or open-source DFT software packages.