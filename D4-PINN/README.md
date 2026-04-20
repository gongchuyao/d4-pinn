# D4-PINN: Dihedral Group Symmetry Preserving Physics-Informed Neural Networks

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## 📖 简介

这是一个**完全零门槛、保姆级**的 D4-PINN 开源项目，专门用于求解具有正方形对称性的偏微分方程(PDE)。

D4-PINN 通过利用 D4 二面体群的对称性，能够：
- 🚀 **精度提升 10 倍以上**：相比传统 PINN，误差降低一个数量级
- ⚡ **收敛速度更快**：利用对称性约束，训练更快收敛
- 🎯 **零依赖基础版**：最简单的版本只需要 PyTorch，不需要任何复杂的库
- 📊 **论文级绘图**：一键生成 20+ 张 SCI 一区期刊标准的高清配图

---

## 🎯 新手入门：5 分钟快速上手

> 📝 **如果你是完全的编程小白，跟着这一步一步来就好！**
> 不需要任何前置知识，我们会从 0 开始教你！

### 第一步：安装 Python

如果你还没有安装 Python：
1. 去 [Python 官网](https://www.python.org/downloads/) 下载 Python 3.8 或更高版本
2. 安装的时候**一定要勾选 "Add Python to PATH"**！
3. 安装完成后，打开命令提示符(CMD)，输入 `python --version`，如果能显示版本号就说明成功了！

### 第二步：下载代码

你可以：
- 点击 GitHub 页面右上角的 **Code** -> **Download ZIP**，然后解压到你电脑上
- 或者如果你会用 git 的话：`git clone https://github.com/yourusername/d4-pinn.git`

### 第三步：安装依赖

打开命令提示符，进入到你解压的文件夹：
```bash
# 比如你把代码解压到了 D 盘的 d4-pinn 文件夹
cd D:\d4-pinn
```

然后安装依赖：
```bash
# 升级 pip
python -m pip install --upgrade pip

# 安装所有需要的包
pip install -r requirements.txt
```

> 💡 如果安装慢的话，可以用清华源：
> ```bash
> pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
> ```

### 第四步：跑第一个例子！

现在你可以跑最简单的 baseline 例子了！
```bash
python examples/baseline_d4_pinn.py
```

🎉 搞定！这个脚本会：
1. 自动训练 D4-PINN 模型
2. 训练完成后自动生成收敛曲线和结果对比图
3. 输出最终的精度指标，直接可以抄进论文里！

---

## 📁 项目结构

```
D4-PINN/
├── 📂 src/                    # 核心源码
│   ├── 📂 models/             # 各种模型实现
│   │   ├── base_d4_pinn.py           # 基础版：最简单的 group averaging
│   │   ├── optimized_d4_pinn.py      # 优化版：快 2-3 倍的 batched 版本
│   │   ├── d4_pinn_equivariant.py    # 等变层版：理论上最严谨的等变层
│   │   ├── d4_pinn_escnn_full.py     # 官方 ESCNN 版：用官方 escnn 库
│   │   └── d4_pinn_escnn_standalone.py # 独立 ESCNN 版：自己实现的，不需要依赖
│   ├── 📂 core/               # 核心功能
│   │   ├── d4_transforms.py          # D4 群变换
│   │   └── pde_problems.py           # 各种 PDE 问题定义
│   └── 📂 utils/              # 工具函数
│       ├── metrics.py                # 误差计算
│       └── visualization.py          # 绘图工具
│
├── 📂 examples/               # 示例脚本（你可以直接跑这些！）
│   ├── baseline_d4_pinn.py               # 【推荐新手】最基础的例子，最快上手
│   ├── run_equivariant_pinn.py           # 等变层版本的实验
│   ├── 3d_poisson_experiment.py          # 3D 泊松方程实验
│   ├── inverse_problem_experiment.py     # 逆问题实验：同时求解解和未知参数
│   ├── extended_experiments.py            # 扩展实验：多个问题、鲁棒性、采样效率
│   └── speed_optimization_test.py        # 速度优化测试：对比基础版和优化版的速度
│
├── 📂 scripts/                # 脚本
│   └── 📂 plotting/           # 绘图脚本（一键生成论文图！）
│       ├── generate_main_figure.py           # 生成论文主图 Figure 1
│       ├── generate_20_cool_figures.py       # 生成 20 张炫酷的顶刊级配图
│       ├── generate_20_figures_basic.py      # 基础版 20 张图
│       ├── generate_20_figures_seaborn.py    # Seaborn 风格的 20 张图
│       ├── generate_supplementary_figures.py # 补充材料的图
│       └── generate_comparison_radar.py      # 综合性能雷达图
│
├── 📂 docs/                  # 文档
│   ├── 📂 tutorial/           # 保姆级教程
│   ├── 📂 manuscript/         # 论文草稿
│   └── 📂 figures/            # 生成的图会存在这里
│
├── requirements.txt          # 依赖列表
├── setup.py                  # 安装脚本
└── README.md                 # 你现在看的这个文件
```

---

## 🚀 所有示例脚本怎么跑？

### 1. 基础版 D4-PINN（推荐新手第一个跑）
```bash
python examples/baseline_d4_pinn.py
```
- 这是最简单的版本，纯 PyTorch，没有任何额外依赖
- 跑 2D 泊松方程，2-3 分钟就能跑完
- 自动生成 `results/` 文件夹，里面有所有的图

### 2. 等变层版本
```bash
python examples/run_equivariant_pinn.py
```
- 用真正的等变层，而不是输出平均
- 理论上更严谨的实现

### 3. 3D 问题
```bash
python examples/3d_poisson_experiment.py
```
- 展示怎么把 D4-PINN 扩展到 3D 问题
- 在 xy 平面上应用 D4 对称

### 4. 逆问题
```bash
python examples/inverse_problem_experiment.py
```
- 逆问题！同时求解 PDE 的解，还有未知的参数 λ
- 支持带噪声的观测数据！

### 5. 扩展实验
```bash
python examples/extended_experiments.py
```
- 一次性跑多个基准问题：泊松、Ginzburg-Landau、Allen-Cahn
- 测试鲁棒性：加噪声的情况
- 测试采样效率：不同采样点数量的影响

### 6. 速度优化测试
```bash
python examples/speed_optimization_test.py
```
- 对比基础版和优化版的速度
- 优化版能快 2-3 倍！

---

## 🎨 一键生成论文配图！

所有的绘图脚本都在 `scripts/plotting/` 文件夹里，你只需要：
```bash
# 先进入 plotting 文件夹
cd scripts/plotting
```

然后跑你想要的脚本：

### 生成论文主图（Figure 1）
```bash
python generate_main_figure.py
```
- 生成论文的主图，包含 4 张子图
- 直接可以放到论文里当 Figure 1！

### 生成 20 张炫酷的顶刊级配图
```bash
python generate_20_cool_figures.py
```
- 一键生成 20 张 300 DPI 的高清图！
- 包含 3D 曲面、热力图、雷达图、小提琴图等等
- 所有字体都是 Times New Roman，完全符合 SCI 期刊标准！

### 生成综合性能雷达图
```bash
python generate_comparison_radar.py
```
- 对比 D4-PINN、ESCNN、Soft Constraint、传统 PINN 的性能
- 5 个维度：精度、速度、部署、鲁棒性、通用性

---

## 📊 实验结果

我们的 D4-PINN 相比传统 PINN 有显著的提升：

| 指标 | 传统 PINN | D4-PINN (Ours) | 提升 |
|------|-----------|-----------------|------|
| L2 相对误差 | 9.98e-3 | **8.55e-4** | **11.7x** |
| L∞ 相对误差 | 1.32e-2 | **1.24e-3** | **10.6x** |
| 收敛速度 | 慢 | 快 | **2-3x** |

---

## 🔧 模型选择指南

不知道用哪个模型？看这个：

| 模型 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| **BaseD4PINN** | 最简单，纯 PyTorch，好理解 | 速度一般 | **新手入门，快速验证** |
| **OptimizedD4PINN** | 速度快 2-3 倍，还是纯 PyTorch | 稍微复杂一点 | **正式训练，大数据集** |
| **D4PINNEquivariant** | 理论严谨，真正的等变层 | 训练慢一点 | **理论研究** |
| **D4PINNESCNNFull** | 标准 ESCNN 实现 | 需要安装 escnn 库 | **和 ESCNN 对比** |
| **D4PINNESCNNStandalone** | 不需要 escnn 库，自己实现 | 代码长一点 | **想理解 ESCNN 原理** |

---

## 📝 常见问题

### 1. 跑的时候报错说找不到模块？
如果你跑的时候出现 `ModuleNotFoundError: No module named 'src'`，
这是因为 Python 找不到我们的包，解决方法：
```bash
# 在项目根目录下运行这个，安装这个包
pip install -e .
```
这样就好了！

### 2. 跑的时候显存不够？
没关系！我们的代码默认用 CPU 就可以跑！
不需要 GPU！CPU 也能在几分钟内跑完！

### 3. 我想改参数怎么办？
打开对应的脚本，你可以改：
- `hidden_dim`: 隐藏层维度
- `epochs`: 训练轮数
- `learning_rate`: 学习率
- 等等...都有注释的！

---

## 📄 引用

如果这个项目对你的研究有帮助，请引用我们的论文：
```bibtex
@article{yourpaper2024,
  title={D4-PINN: Dihedral Group Symmetry Preserving Physics-Informed Neural Networks},
  author={Your Name and Co-authors},
  journal={Your Journal},
  year={2024}
}
```

---

## 📬 联系我们

如果有任何问题，欢迎提 Issue 或者联系我们！

---

## 🎉 许可证

MIT License - 随便用，随便改！
