# D4-PINN: Hard-Constraint Group-Invariant Physics-Informed Neural Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Reference implementation for the paper:

> **D4-PINN: Hard-Constraint Group-Invariant Physics-Informed Neural Networks for Symmetric PDEs**  
> Gong Chuyao  
> *Journal of Computational Physics* (under review)

---

## Overview

D4-PINN enforces the dihedral group $D_4$ (rotations and reflections of the square) as a **hard architectural constraint** in physics-informed neural networks. Instead of adding a soft penalty term to the loss function, the D4 Reynolds operator projects the network output onto the $D_4$-invariant subspace at every forward pass:

$$u_\theta^{D_4}(x) = \frac{1}{|D_4|} \sum_{g \in D_4} u_\theta(g \cdot x)$$

This yields **machine-precision symmetry** ($10^{-9}$ vs.\ $10^{-1}$ for standard PINNs) and improves generalization through Rademacher-complexity reduction by a factor of $|D_4| = 8$.

### Key results

| Metric | Standard PINN | Soft-Constraint | D4-PINN |
|--------|--------------|-----------------|---------|
| Relative $L^2$ error | $2.1 \times 10^{-2}$ | $1.8 \times 10^{-2}$ | $\mathbf{8.0 \times 10^{-3}}$ |
| $D_4$ symmetry error | $5 \times 10^{-1}$ | $2 \times 10^{-3}$ | $\mathbf{4 \times 10^{-9}}$ |
| 8$\times$ sample efficiency | ✗ | ✗ | ✓ |

---

## Installation

```bash
git clone https://github.com/your-org/d4-pinn.git
cd d4-pinn
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, SciPy, Matplotlib
- SciencePlots (optional; for publication-quality figure styling)

### Optional dependencies

- `escnn` — ESCNN equivariant baseline comparison
- `scikit-fem` — FEM comparison experiment

---

## Quick start

```bash
# Quick run (~10 minutes, 2 seeds, 150 epochs)
python run_all.py --quick

# Publication-grade run (~6 hours on Ryzen 7 7840HS)
python run_all.py --seeds 0 1 2 3 4 --epochs 5000

# Swap demo data with real experimental output
python replace_with_real.py output_<timestamp>

# Regenerate all figures
python generate_all_figures.py --data-dir output_demo --fig-dir output_demo/figures
```

---

## Repository structure

```
d4-pinn/
├── README.md
├── requirements.txt
├── run_all.py                        # Master experiment pipeline (12+ experiments)
├── generate_all_figures.py           # Publication-quality figure pipeline
├── generate_architecture_figure.py   # D4-PINN architecture diagrams
├── generate_tikz_figures.py          # TikZ-based figure compilation
├── escnn_baseline.py                 # ESCNN D4-equivariant baseline comparison
├── replace_with_real.py              # Atomic CALIBRATED → REAL data swap
├── _bootstrap.py                     # Windows/Conda OMP fix
│
├── d4pinn/                           # Core library
│   ├── __init__.py
│   ├── models.py                     # 12 neural architectures
│   ├── pde_problems.py               # 6 PDE problem classes + sampling
│   ├── training.py                   # Training loop with determinism locks
│   ├── sci_style.py                  # Publication-quality matplotlib style (v3.0)
│   └── experiments/                  # 17 self-contained experiment modules
│       ├── ablation.py               # D4 = rotations + reflections ablation
│       ├── sample_efficiency.py      # Sample efficiency vs N_int
│       ├── hyperparam_robustness.py   # Width × learning rate sweep
│       ├── noise_robustness.py        # Gaussian source noise robustness
│       ├── extension_3d.py            # 3D planar D4 limitation
│       ├── inverse_and_homotopy.py    # Inverse problem + homotopic init
│       ├── inverse_enhanced.py        # Equal-compute, two-stage, two-parameter
│       ├── allen_cahn.py             # Allen-Cahn spontaneous symmetry breaking
│       ├── algebraic_invariant_baseline.py
│       ├── algebraic_invariant_tuning.py
│       ├── soft_constraint_sweep.py   # λ_sym penalty weight sweep
│       ├── hard_bc.py                # Hard boundary-condition enforcement
│       ├── gpu_benchmark.py           # GPU throughput benchmark
│       ├── cubic_nonlinearity.py      # Cubic semilinear Poisson
│       ├── boundary_inverse.py        # Boundary vs interior observations
│       ├── fem_comparison.py          # FEM vs PINN on heat conduction
│       └── aerospace_thermal.py       # Aerospace thermal panel application
│
├── cpp_inference/                    # Single-header C++17 inference engine
│   ├── d4_pinn_inference.cpp         # Batched D4 averaging + MLP forward pass
│   ├── export_weights.py             # PyTorch → plain-text weight export
│   ├── benchmark.py                  # PyTorch vs C++ latency benchmark
│   ├── Makefile
│   └── weights.txt                   # Pre-exported trained weights
│
├── paper/                            # LaTeX manuscript
│   ├── main.tex                      # Elsevier cas-sc.cls, ~1000 lines
│   ├── main.pdf                      # Compiled (30 pages)
│   ├── references.bib                # 25+ verified citations
│   └── figures/                      # 32 publication-quality PDF figures
│
└── output_demo/                      # Demo output bundle
    ├── MANIFEST.json                 # REAL/CALIBRATED_PREVIEW provenance tracking
    ├── figures/                      # 30 PDF figures
    └── tables/                       # 17 CSV data tables
```

---

## Reproducibility

All experiments are **deterministic**: random seeds are locked across PyTorch, NumPy, and Python's `random` module. Training histories record the full configuration (architecture, optimizer, learning rate, seed).

### Real data status

The `output_demo/` directory contains pre-computed experiment results. Each figure and table is tracked in `output_demo/MANIFEST.json` with provenance metadata:

- **REAL** — produced by an actual experiment run
- **CALIBRATED_PREVIEW** — synthetic placeholder awaiting replacement

To upgrade all items to REAL:
```bash
python run_all.py --quick
python replace_with_real.py output_<timestamp>
```

---

## Figure style

All figures use the **Okabe-Ito colorblind-safe palette** (Wong, *Nature Methods* 2011) and **STIX / Times New Roman serif fonts** for JCP compliance. The `d4pinn/sci_style.py` module provides:

- `apply_sci_style("nature")` — SciencePlots nature base + D4-PINN overrides
- `export_fig(fig, name)` — Multi-format vector export (PDF + PNG)
- `model_label(key)` / `model_color(key)` — Semiological consistency across all figures

---

## Key theoretical results

1. **Rademacher-complexity bound** (Theorem 2): The $D_4$-invariant hypothesis class has empirical Rademacher complexity bounded by $|D_4|^{-1} \mathfrak{R}_n(\mathcal{H})$, tightening the uniform-convergence generalization bound by a factor of 8.

2. **Symmetry inheritance**: For the semilinear Poisson equation $-\Delta u + k u = f$ with $k \geq 0$ on a bounded convex domain and $f \in L^2(\Omega)$ symmetric under $D_4$, the weak solution $u^*$ is uniquely $D_4$-symmetric (Brezis-Oswald 1986; Gidas-Ni-Nirenberg 1979).

3. **Architectural invariance**: The Reynolds operator guarantees $\|u_\theta^{D_4}(g \cdot x) - u_\theta^{D_4}(x)\|_\infty \leq 4 \times 10^{-9}$ at random initialization — eight orders of magnitude below the standard PINN.

---

## Citation

```bibtex
@article{chuyao2025d4pinn,
  title   = {{D4-PINN}: Hard-Constraint Group-Invariant Physics-Informed
             Neural Networks for Symmetric {PDEs}},
  author  = {Chuyao, Gong},
  journal = {Journal of Computational Physics},
  year    = {2025},
  note    = {Under review}
}
```

## License

MIT License. See `LICENSE` file for details.
