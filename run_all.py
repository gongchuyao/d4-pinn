"""
run_all.py — master experiment + figure pipeline for D4-PINN
=============================================================

Single entry point that:
  1. Trains Standard / Soft / D4 PINN over multiple seeds at epsilon = 0
     (forward benchmark).
  2. Sweeps the symmetry-breaking parameter epsilon over a controlled grid.
  3. Verifies architectural D4-invariance across all 7 non-identity D4
     transforms at machine precision.
  4. Benchmarks batched vs sequential D4 group averaging at five batch sizes.
  5. Exports every numerical result as a CSV file in `tables/`.
  6. Produces seven publication-grade vector PDF figures in `figures/`.

Usage on Windows (cmd.exe):
    python run_all.py --quick                  (~10 min on AMD 7840HS)
    python run_all.py                          (~60 min, default settings)
    python run_all.py --epochs 5000 --seeds 0 1 2 3 4   (full publication run)

Output directory layout (created next to this script):

    output_<timestamp>/
        figures/
            fig1_convergence.pdf
            fig2_l2_vs_seeds.pdf
            fig3_symmetry_error_bar.pdf
            fig4_solution_field.pdf
            fig5_symmetry_breaking_curve.pdf
            fig6_batched_speedup.pdf
            fig7_loss_landscape.pdf
        tables/
            T1_main_benchmark.csv
            T2_symmetry_at_init.csv
            T3_symmetry_breaking_sweep.csv
            T4_batched_speedup.csv
            T5_loss_landscape.csv
        run_config.json
        run_log.txt
"""

from __future__ import annotations

# IMPORTANT: bootstrap must be the FIRST import (before numpy/torch).
import _bootstrap  # noqa: F401

import argparse
import csv
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

from d4pinn import (
    D4PINN, StandardPINN, SoftConstraintPINN,
    ReactionDiffusionBVP, sample_interior, sample_boundary,
    train_one_run,
    apply_sci_style, COLORS, model_label, model_color, export_fig, figsize,
)
from d4pinn.experiments import (
    ablation as exp_ablation,
    sample_efficiency as exp_sample,
    hyperparam_robustness as exp_hp,
    noise_robustness as exp_noise,
    extension_3d as exp_3d,
    inverse_and_homotopy as exp_inv_homo,
    allen_cahn as exp_allen_cahn,
    algebraic_invariant_baseline as exp_alg_inv,
    inverse_enhanced as exp_inv_enhanced,
    soft_constraint_sweep as exp_soft_sweep,
    hard_bc as exp_hard_bc,
    gpu_benchmark as exp_gpu,
    cubic_nonlinearity as exp_cubic,
)

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

LOG = logging.getLogger("d4pinn")


def _setup_logging(log_file: Path) -> None:
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    fmt = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    LOG.addHandler(sh)
    LOG.addHandler(fh)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    """Write rows to CSV with consistent precision for floating-point columns."""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                f"{v:.6e}" if isinstance(v, float) else v
                for v in r
            ])
    LOG.info(f"  CSV  -> {path.relative_to(path.parent.parent)}")


# ---------------------------------------------------------------------------
# Statistical hypothesis testing
# ---------------------------------------------------------------------------

def welch_t_test(sample_a: list[float], sample_b: list[float]) -> dict:
    """Welch's t-test (unequal variance) for two independent samples.

    Returns {t_stat, p_value, significant_at_5pct, mean_diff, cohens_d}.
    """
    import math
    a = np.array(sample_a, dtype=np.float64)
    b = np.array(sample_b, dtype=np.float64)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return {"t_stat": float("nan"), "p_value": float("nan"),
                "significant_at_5pct": False, "mean_diff": float("nan"),
                "cohens_d": float("nan")}
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-30:
        return {"t_stat": 0.0, "p_value": 1.0,
                "significant_at_5pct": False,
                "mean_diff": float(a.mean() - b.mean()), "cohens_d": 0.0}
    t_stat = float((a.mean() - b.mean()) / se)
    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / den if den > 0 else 1.0
    # p-value from Student's t CDF (uses torch which is always available)
    try:
        from torch.distributions import StudentT
        t_dist = StudentT(df)
        p_val = float(2.0 * (1.0 - t_dist.cdf(torch.tensor(abs(t_stat))).item()))
    except Exception:
        # Fallback: normal approximation for large df
        import math
        p_val = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0)))))
    # Cohen's d (pooled std)
    pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    cohens_d = float((a.mean() - b.mean()) / pooled) if pooled > 1e-30 else 0.0
    return {
        "t_stat": t_stat, "p_value": p_val,
        "significant_at_5pct": p_val < 0.05,
        "mean_diff": float(a.mean() - b.mean()),
        "cohens_d": cohens_d,
    }


def nonparametric_stats(sample: list[float]) -> dict:
    """Compute median, IQR, and MAD for non-parametric reporting.

    Reviewer 3: "应使用中位数、四分位数等非参数统计量"
    """
    a = np.array(sample, dtype=np.float64)
    median = float(np.median(a))
    q1 = float(np.percentile(a, 25))
    q3 = float(np.percentile(a, 75))
    iqr = q3 - q1
    mad = float(np.median(np.abs(a - median)))
    return {"median": median, "q1": q1, "q3": q3, "iqr": iqr, "mad": mad,
            "n": len(a), "min": float(a.min()), "max": float(a.max())}


# ---------------------------------------------------------------------------
# Experiment 1 -- main benchmark, epsilon = 0
# ---------------------------------------------------------------------------

def exp1_main_benchmark(
    *, seeds: list[int], epochs: int, n_int: int, n_bc: int,
    layers: list[int], lr: float, lambda_bc: float, lambda_sym: float,
    eval_every: int, tables_dir: Path,
) -> dict:
    LOG.info("=" * 70)
    LOG.info(f"Experiment 1: main benchmark at epsilon=0  (epochs={epochs})")
    LOG.info("=" * 70)

    pde = ReactionDiffusionBVP(k=1.0, epsilon=0.0)
    methods: dict[str, tuple[Callable, float | None]] = {
        "standard": (lambda: StandardPINN(layers=layers), None),
        "soft":     (lambda: SoftConstraintPINN(layers=layers), lambda_sym),
        "d4":       (lambda: D4PINN(layers=layers), None),
    }

    results: dict[str, dict] = {m: {"runs": [], "histories": []} for m in methods}
    csv_rows: list[list] = []

    for m, (factory, lam_sym) in methods.items():
        LOG.info(f"-- {model_label(m)}")
        for s in seeds:
            t0 = time.time()
            _, h = train_one_run(
                pde=pde, model=factory(),
                n_int=n_int, n_bc=n_bc, epochs=epochs,
                learning_rate=lr, lambda_bc=lambda_bc, lambda_sym=lam_sym,
                seed=s, eval_every=eval_every, verbose=False,
            )
            wall = time.time() - t0
            results[m]["runs"].append({
                "seed": s, "L2": h.l2_error[-1], "sym": h.sym_error[-1],
                "loss": h.loss_total[-1], "wall_s": wall,
            })
            results[m]["histories"].append(h)
            LOG.info(f"   seed={s:>2}  L2={h.l2_error[-1]:.3e}  "
                     f"sym={h.sym_error[-1]:.2e}  wall={wall:5.1f}s")
            csv_rows.append([m, s, h.l2_error[-1], h.sym_error[-1],
                              h.loss_total[-1], wall])

    # Stats summary
    for m in methods:
        a = np.array([r["L2"] for r in results[m]["runs"]])
        b = np.array([r["sym"] for r in results[m]["runs"]])
        t = np.array([r["wall_s"] for r in results[m]["runs"]])
        results[m]["summary"] = {
            "L2_mean": float(a.mean()), "L2_std": float(a.std(ddof=1) if len(a)>1 else 0.0),
            "sym_mean": float(b.mean()), "sym_std": float(b.std(ddof=1) if len(b)>1 else 0.0),
            "wall_mean": float(t.mean()), "wall_std": float(t.std(ddof=1) if len(t)>1 else 0.0),
        }

    write_csv(
        tables_dir / "T1_main_benchmark.csv",
        ["method", "seed", "L2_rel_error", "symmetry_error", "final_loss", "wall_time_s"],
        csv_rows,
    )

    # Aggregated summary CSV (one row per method)
    summary_rows = [
        [m,
         results[m]["summary"]["L2_mean"], results[m]["summary"]["L2_std"],
         results[m]["summary"]["sym_mean"], results[m]["summary"]["sym_std"],
         results[m]["summary"]["wall_mean"], results[m]["summary"]["wall_std"]]
        for m in methods
    ]
    write_csv(
        tables_dir / "T1_main_benchmark_summary.csv",
        ["method", "L2_mean", "L2_std", "sym_mean", "sym_std", "wall_mean_s", "wall_std_s"],
        summary_rows,
    )

    LOG.info("Summary:")
    for m in methods:
        s = results[m]["summary"]
        LOG.info(f"  {m:<10} L2 = {s['L2_mean']:.3e} +/- {s['L2_std']:.1e}    "
                 f"sym = {s['sym_mean']:.2e} +/- {s['sym_std']:.1e}    "
                 f"wall = {s['wall_mean']:.1f} +/- {s['wall_std']:.1f} s")

    return results


# ---------------------------------------------------------------------------
# Experiment 2 -- D4 invariance verification at random init
# ---------------------------------------------------------------------------

def exp2_invariance_verification(
    *, layers: list[int], n_test: int, seeds: list[int],
    tables_dir: Path,
) -> dict:
    LOG.info("=" * 70)
    LOG.info(f"Experiment 2: architectural D4-invariance check at random init")
    LOG.info("=" * 70)

    pde = ReactionDiffusionBVP()
    transforms = [
        ("identity",  torch.eye(2)),
        ("rot90",     torch.tensor([[0., -1.], [1., 0.]])),
        ("rot180",    torch.tensor([[-1., 0.], [0., -1.]])),
        ("rot270",    torch.tensor([[0., 1.], [-1., 0.]])),
        ("flipx",     torch.tensor([[1., 0.], [0., -1.]])),
        ("flipy",     torch.tensor([[-1., 0.], [0., 1.]])),
        ("diag",      torch.tensor([[0., 1.], [1., 0.]])),
        ("antidiag",  torch.tensor([[0., -1.], [-1., 0.]])),
    ]

    rows: list[list] = []
    summary: dict[str, dict] = {}
    for model_key, factory in [
        ("standard", lambda: StandardPINN(layers=layers)),
        ("d4",       lambda: D4PINN(layers=layers)),
    ]:
        LOG.info(f"-- {model_label(model_key)}")
        per_seed = []  # (seed, transform_name) -> max abs deviation
        for s in seeds:
            torch.manual_seed(s)
            m = factory()
            m.eval()
            x = torch.rand(n_test, 2) * 2 - 1
            with torch.no_grad():
                y_id = m(x)
                if y_id.dim() == 1:
                    y_id = y_id.unsqueeze(-1)
                seed_errs = {}
                for name, M in transforms:
                    if name == "identity":
                        seed_errs[name] = 0.0
                        continue
                    x_g = x @ M.T
                    y_g = m(x_g)
                    if y_g.dim() == 1:
                        y_g = y_g.unsqueeze(-1)
                    err = float(torch.max(torch.abs(y_id - y_g)))
                    seed_errs[name] = err
                per_seed.append(seed_errs)
                for name, _ in transforms:
                    rows.append([model_key, s, name, seed_errs.get(name, 0.0)])
            for name, _ in transforms:
                LOG.info(f"   seed={s} {name:<10} max|y(x)-y(g.x)| = {seed_errs.get(name, 0.0):.3e}")
        # aggregate per transform
        agg = {}
        for name, _ in transforms:
            arr = np.array([e[name] for e in per_seed])
            agg[name] = {"mean": float(arr.mean()), "max": float(arr.max())}
        summary[model_key] = agg

    write_csv(
        tables_dir / "T2_symmetry_at_init.csv",
        ["method", "seed", "transform", "max_abs_deviation"],
        rows,
    )

    return summary


# ---------------------------------------------------------------------------
# Experiment 3 -- symmetry-breaking sweep
# ---------------------------------------------------------------------------

def exp3_symmetry_breaking_sweep(
    *, epsilons: list[float], seeds: list[int], epochs: int,
    n_int: int, n_bc: int, lr: float, lambda_bc: float,
    layers: list[int], eval_every: int, tables_dir: Path,
) -> dict:
    LOG.info("=" * 70)
    LOG.info(f"Experiment 3: symmetry-breaking sweep "
             f"(epsilons={epsilons}, seeds={seeds}, epochs={epochs})")
    LOG.info("=" * 70)

    rows: list[list] = []
    by_eps: dict[float, dict[str, list[dict]]] = {
        eps: {"standard": [], "d4": []} for eps in epsilons
    }
    for eps in epsilons:
        LOG.info(f"-- epsilon = {eps}")
        pde = ReactionDiffusionBVP(k=1.0, epsilon=eps)
        for model_key, factory in [
            ("standard", lambda: StandardPINN(layers=layers)),
            ("d4",       lambda: D4PINN(layers=layers)),
        ]:
            for s in seeds:
                t0 = time.time()
                _, h = train_one_run(
                    pde=pde, model=factory(),
                    n_int=n_int, n_bc=n_bc, epochs=epochs,
                    learning_rate=lr, lambda_bc=lambda_bc, lambda_sym=None,
                    seed=s, eval_every=eval_every, verbose=False,
                )
                wall = time.time() - t0
                rec = {"seed": s, "L2": h.l2_error[-1],
                        "sym": h.sym_error[-1],
                        "loss": h.loss_total[-1], "wall_s": wall}
                by_eps[eps][model_key].append(rec)
                rows.append([eps, model_key, s, h.l2_error[-1],
                              h.sym_error[-1], h.loss_total[-1], wall])
                LOG.info(f"   eps={eps:>6.3f}  {model_key:<8} seed={s}  "
                         f"L2={h.l2_error[-1]:.3e}  sym={h.sym_error[-1]:.2e}  "
                         f"t={wall:.1f}s")

    write_csv(
        tables_dir / "T3_symmetry_breaking_sweep.csv",
        ["epsilon", "method", "seed", "L2_rel_error", "symmetry_error",
         "final_loss", "wall_time_s"],
        rows,
    )

    # aggregate
    agg: dict[float, dict[str, dict[str, float]]] = {}
    for eps in epsilons:
        agg[eps] = {}
        for model_key in ("standard", "d4"):
            arr_l2 = np.array([r["L2"] for r in by_eps[eps][model_key]])
            arr_s = np.array([r["sym"] for r in by_eps[eps][model_key]])
            agg[eps][model_key] = {
                "L2_mean": float(arr_l2.mean()),
                "L2_std": float(arr_l2.std(ddof=1) if len(arr_l2) > 1 else 0.0),
                "sym_mean": float(arr_s.mean()),
                "sym_std": float(arr_s.std(ddof=1) if len(arr_s) > 1 else 0.0),
            }
    return agg


# ---------------------------------------------------------------------------
# Experiment 4 -- batched vs sequential speedup
# ---------------------------------------------------------------------------

def exp4_batched_speedup(
    *, batch_sizes: list[int], n_repeats: int, layers: list[int],
    tables_dir: Path,
) -> list[dict]:
    LOG.info("=" * 70)
    LOG.info(f"Experiment 4: batched vs sequential D4 group averaging")
    LOG.info("=" * 70)

    rows: list[list] = []
    summary: list[dict] = []

    # Build a sequential reference D4-PINN (loop over the 8 transforms).
    from d4pinn.models import _D4_MATRICES

    class SequentialD4PINN(torch.nn.Module):
        def __init__(self, layers):
            super().__init__()
            from d4pinn.models import MLP
            self.mlp = MLP(layers)
            self.register_buffer("_D4", _D4_MATRICES.clone())

        def forward(self, x):
            acc = None
            for i in range(8):
                xg = x @ self._D4[i].T
                y = self.mlp(xg)
                acc = y if acc is None else acc + y
            return acc / 8.0

    torch.manual_seed(0)
    m_seq = SequentialD4PINN(layers)
    torch.manual_seed(0)
    m_bat = D4PINN(layers=layers)
    # Equalise weights so timings compare like-for-like.
    m_seq.mlp.load_state_dict(m_bat.mlp.state_dict())

    for n in batch_sizes:
        LOG.info(f"-- batch size = {n}")
        seq_times = []
        bat_times = []
        max_diff = 0.0
        for r in range(n_repeats):
            x = torch.randn(n, 2)
            x.requires_grad_(True)
            # warmup
            _ = m_seq(x).sum()
            _ = m_bat(x).sum()

            # sequential
            t0 = time.perf_counter()
            y_seq = m_seq(x).sum()
            y_seq.backward()
            seq_times.append(time.perf_counter() - t0)

            x2 = x.detach().clone().requires_grad_(True)
            t0 = time.perf_counter()
            y_bat = m_bat(x2).sum()
            y_bat.backward()
            bat_times.append(time.perf_counter() - t0)

            with torch.no_grad():
                y_a = m_seq(x.detach())
                y_b = m_bat(x.detach())
                max_diff = max(max_diff, float(torch.max(torch.abs(y_a - y_b))))

        seq_a = np.array(seq_times)
        bat_a = np.array(bat_times)
        speedup = seq_a.mean() / bat_a.mean()
        summary.append({
            "batch_size": n,
            "seq_mean": float(seq_a.mean()), "seq_std": float(seq_a.std(ddof=1) if len(seq_a)>1 else 0.0),
            "bat_mean": float(bat_a.mean()), "bat_std": float(bat_a.std(ddof=1) if len(bat_a)>1 else 0.0),
            "speedup": float(speedup),
            "numerical_max_diff": max_diff,
        })
        for r in range(n_repeats):
            rows.append([n, r, seq_times[r], bat_times[r],
                          seq_times[r] / bat_times[r]])
        LOG.info(f"   N={n:>6}  seq={seq_a.mean()*1e3:7.2f} ms   "
                 f"bat={bat_a.mean()*1e3:7.2f} ms   "
                 f"speedup={speedup:5.2f}x   max|seq-bat|={max_diff:.2e}")

    write_csv(
        tables_dir / "T4_batched_speedup.csv",
        ["batch_size", "repeat", "sequential_time_s", "batched_time_s",
         "per_run_speedup"],
        rows,
    )
    write_csv(
        tables_dir / "T4_batched_speedup_summary.csv",
        ["batch_size", "seq_mean_s", "seq_std_s", "bat_mean_s", "bat_std_s",
         "speedup", "numerical_max_diff"],
        [[r["batch_size"], r["seq_mean"], r["seq_std"], r["bat_mean"],
          r["bat_std"], r["speedup"], r["numerical_max_diff"]]
         for r in summary],
    )

    geo = math.exp(np.mean([math.log(r["speedup"]) for r in summary]))
    LOG.info(f"  Geometric mean speedup: {geo:.2f}x")
    return summary


# ---------------------------------------------------------------------------
# Experiment 5 -- loss landscape (filter-normalised, real)
# ---------------------------------------------------------------------------

def exp5_loss_landscape(
    *, layers: list[int], train_epochs: int, n_int: int, n_bc: int,
    lr: float, lambda_bc: float, grid_n: int, span: float,
    tables_dir: Path,
) -> dict:
    LOG.info("=" * 70)
    LOG.info(f"Experiment 5: filter-normalised loss landscape")
    LOG.info("=" * 70)

    pde = ReactionDiffusionBVP(k=1.0, epsilon=0.0)
    landscapes: dict[str, dict] = {}

    for model_key, factory in [
        ("standard", lambda: StandardPINN(layers=layers)),
        ("d4",       lambda: D4PINN(layers=layers)),
    ]:
        LOG.info(f"-- training {model_label(model_key)} for landscape")
        torch.manual_seed(0)
        model = factory()
        _, _ = train_one_run(
            pde=pde, model=model,
            n_int=n_int, n_bc=n_bc, epochs=train_epochs,
            learning_rate=lr, lambda_bc=lambda_bc, lambda_sym=None,
            seed=0, eval_every=max(train_epochs // 2, 1), verbose=False,
        )

        # Filter-normalised directions (Li et al. NeurIPS 2018).
        params = [p.detach().clone() for p in model.parameters()]

        def _filter_normalised(direction: list[torch.Tensor],
                                params: list[torch.Tensor]) -> list[torch.Tensor]:
            out = []
            for d, p in zip(direction, params):
                if d.dim() <= 1:  # bias / 1-D parameter
                    out.append(torch.zeros_like(d))
                    continue
                # Reshape per-filter (output dim x rest)
                d_flat = d.reshape(d.shape[0], -1)
                p_flat = p.reshape(p.shape[0], -1)
                d_norm = d_flat.norm(dim=1, keepdim=True) + 1e-12
                p_norm = p_flat.norm(dim=1, keepdim=True) + 1e-12
                d_scaled = (d_flat / d_norm) * p_norm
                out.append(d_scaled.reshape(d.shape))
            return out

        torch.manual_seed(123)
        d1 = _filter_normalised([torch.randn_like(p) for p in params], params)
        torch.manual_seed(456)
        d2 = _filter_normalised([torch.randn_like(p) for p in params], params)

        # Fixed evaluation set (no requires_grad to save memory)
        torch.manual_seed(0)
        xy_int = sample_interior(800).requires_grad_(True)
        xy_bc = sample_boundary(160)

        alphas = np.linspace(-span, span, grid_n)
        betas = np.linspace(-span, span, grid_n)
        loss_grid = np.zeros((grid_n, grid_n), dtype=np.float64)

        original = [p.detach().clone() for p in model.parameters()]
        try:
            for i, a in enumerate(alphas):
                for j, b in enumerate(betas):
                    with torch.no_grad():
                        for p, p0, e1, e2 in zip(model.parameters(), original, d1, d2):
                            p.copy_(p0 + a * e1 + b * e2)
                    # Loss requires grad through input only (params static here).
                    loss_total, _, _ = pde.loss(model, xy_int, xy_bc,
                                                  lambda_bc=lambda_bc)
                    loss_grid[i, j] = float(loss_total.detach())
                LOG.info(f"   {model_label(model_key)}  "
                         f"row {i+1}/{grid_n}  loss range "
                         f"[{loss_grid[i].min():.3e}, {loss_grid[i].max():.3e}]")
        finally:
            with torch.no_grad():
                for p, p0 in zip(model.parameters(), original):
                    p.copy_(p0)

        landscapes[model_key] = {
            "alphas": alphas.tolist(),
            "betas": betas.tolist(),
            "loss_grid": loss_grid.tolist(),
        }

        # CSV: long-form
        rows = []
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                rows.append([model_key, float(a), float(b), float(loss_grid[i, j])])
        write_csv(
            tables_dir / f"T5_loss_landscape_{model_key}.csv",
            ["method", "alpha", "beta", "loss"],
            rows,
        )

    return landscapes


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def fig1_convergence(results: dict, fig_path: Path) -> None:
    """Convergence curves: L2 error vs epoch with mean line and 1-sigma band."""
    apply_sci_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True)
    for ax, key in zip(axes, ("l2_error", "loss_total")):
        for m in ("standard", "soft", "d4"):
            histories = results[m]["histories"]
            epochs = np.array(histories[0].epochs)
            data = np.array([getattr(h, key) for h in histories])
            mean = data.mean(axis=0)
            std = data.std(axis=0, ddof=1) if len(histories) > 1 else None
            ax.plot(epochs, mean, color=model_color(m),
                    label=model_label(m), linewidth=1.5)
            if std is not None:
                ax.fill_between(epochs, np.maximum(mean - std, 1e-30),
                                 mean + std, color=model_color(m), alpha=0.18,
                                 linewidth=0)
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
    axes[0].set_ylabel(r"Relative $L_2$ error")
    axes[0].set_title("Test accuracy")
    axes[1].set_ylabel("Total training loss")
    axes[1].set_title("Training loss")
    axes[1].legend(loc="upper right", fontsize=8)
    for ax, label in zip(axes, "ab"):
        ax.text(-0.12, 1.02, f"({label})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


def fig2_l2_vs_seeds(results: dict, fig_path: Path) -> None:
    """Per-seed L2 errors with summary box."""
    apply_sci_style()
    methods = ["standard", "soft", "d4"]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for i, m in enumerate(methods):
        l2s = [r["L2"] for r in results[m]["runs"]]
        x = np.full_like(l2s, i, dtype=float) + np.random.uniform(-0.08, 0.08, len(l2s))
        ax.scatter(x, l2s, color=model_color(m), s=22, zorder=3,
                    edgecolors="white", linewidths=0.6)
        mean = np.mean(l2s)
        ax.hlines(mean, i - 0.22, i + 0.22, colors=model_color(m), linewidth=2)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([model_label(m) for m in methods], rotation=15, ha="right")
    ax.set_ylabel(r"Relative $L_2$ error")
    ax.set_yscale("log")
    ax.set_title(r"Final $L_2$ error per seed (line = mean)")
    fig.tight_layout()
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


def fig3_symmetry_error_bar(invariance_summary: dict, fig_path: Path) -> None:
    """Per-transform symmetry errors at random init: shows that D4-PINN is exact
    while StandardPINN is O(0.1)."""
    apply_sci_style()
    transforms = ["rot90", "rot180", "rot270", "flipx", "flipy", "diag", "antidiag"]
    pretty = [r"$R_{90}$", r"$R_{180}$", r"$R_{270}$",
              r"$S_x$", r"$S_y$", r"$S_{d_1}$", r"$S_{d_2}$"]

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    width = 0.38
    pos = np.arange(len(transforms))
    std_means = [invariance_summary["standard"][t]["mean"] for t in transforms]
    d4_means = [invariance_summary["d4"][t]["mean"] for t in transforms]
    # Avoid log(0) -- D4 errors can be ~0; use floor at 1e-12.
    d4_means = [max(v, 1e-12) for v in d4_means]
    std_means = [max(v, 1e-12) for v in std_means]
    ax.bar(pos - width / 2, std_means, width,
            color=model_color("standard"), label=model_label("standard"),
            edgecolor="white", linewidth=0.5)
    ax.bar(pos + width / 2, d4_means, width,
            color=model_color("d4"), label=model_label("d4"),
            edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(pos)
    ax.set_xticklabels(pretty)
    ax.set_xlabel(r"Non-identity element of $D_4$")
    ax.set_ylabel(r"max $|u_\theta(x) - u_\theta(g\!\cdot\! x)|$")
    ax.set_title(r"Architectural $D_4$-invariance at random initialisation")
    ax.legend(loc="lower left", ncol=2)
    ax.axhline(1e-7, linestyle=":", color=COLORS["text"], linewidth=0.7,
                alpha=0.6)
    ax.text(len(transforms) - 0.5, 1.5e-7, r"machine precision", ha="right",
             va="bottom", fontsize=7, color=COLORS["text"], alpha=0.7)
    fig.tight_layout()
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


def fig4_solution_field(results: dict, fig_path: Path,
                         layers: list[int],
                         n_int: int = 2000, n_bc: int = 400,
                         epochs: int = 2000) -> None:
    """Solution heatmap, reference, and pointwise error for a single trained
    D4-PINN. Uses a deterministic seed and the same training budget as the
    main benchmark."""
    apply_sci_style()
    pde = ReactionDiffusionBVP(k=1.0, epsilon=0.0)
    torch.manual_seed(0)
    model = D4PINN(layers=layers)
    LOG.info("  Training a single D4-PINN for solution-field figure ...")
    _, _h = train_one_run(
        pde=pde, model=model,
        n_int=n_int, n_bc=n_bc, epochs=epochs,
        learning_rate=1e-3, lambda_bc=10.0, lambda_sym=None,
        seed=0, eval_every=max(epochs // 2, 1), verbose=False,
    )

    # Evaluate on a 200x200 grid.
    s = torch.linspace(-1, 1, 200)
    xx, yy = torch.meshgrid(s, s, indexing="xy")
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    with torch.no_grad():
        u_pred = model(grid).reshape(200, 200).numpy()
        u_ref = pde.reference_solution(grid).reshape(200, 200).numpy()
    err = np.abs(u_pred - u_ref)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    extent = (-1, 1, -1, 1)
    im0 = axes[0].imshow(u_ref, origin="lower", extent=extent, cmap="viridis")
    axes[0].set_title(r"Reference $u^*$")
    im1 = axes[1].imshow(u_pred, origin="lower", extent=extent, cmap="viridis")
    axes[1].set_title(r"$D_4$-PINN prediction")
    im2 = axes[2].imshow(err, origin="lower", extent=extent, cmap="magma")
    axes[2].set_title(r"Pointwise $|u_\theta - u^*|$")
    for ax, label in zip(axes, "abc"):
        ax.text(-0.08, 1.04, f"({label})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")
    for ax in axes:
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_aspect("equal")
        ax.grid(False)
    fig.colorbar(im0, ax=axes[0], shrink=0.85, pad=0.03)
    fig.colorbar(im1, ax=axes[1], shrink=0.85, pad=0.03)
    fig.colorbar(im2, ax=axes[2], shrink=0.85, pad=0.03)
    fig.tight_layout()
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


def fig5_symmetry_breaking(agg: dict, fig_path: Path) -> None:
    """Symmetry-breaking sweep: L2 vs eps, with crossover band."""
    apply_sci_style()
    epsilons = sorted(agg.keys())
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    # --- (a) L2 error vs eps ---
    for key, label, color in (
        ("standard", model_label("standard"), model_color("standard")),
        ("d4",        model_label("d4"),        model_color("d4")),
    ):
        means = [agg[e][key]["L2_mean"] for e in epsilons]
        stds = [agg[e][key]["L2_std"] for e in epsilons]
        means = np.array(means); stds = np.array(stds)
        axes[0].errorbar(epsilons, means, yerr=stds, label=label,
                          color=color, capsize=2.5, linewidth=1.4,
                          marker="o", markersize=4)
    axes[0].set_xlabel(r"Asymmetric perturbation $\varepsilon$")
    axes[0].set_ylabel(r"Relative $L_2$ error vs $u^*$")
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Forward-problem accuracy")
    axes[0].legend(loc="best")

    # --- (b) Symmetry error vs eps ---
    for key, label, color in (
        ("standard", model_label("standard"), model_color("standard")),
        ("d4",        model_label("d4"),        model_color("d4")),
    ):
        means = [max(agg[e][key]["sym_mean"], 1e-12) for e in epsilons]
        axes[1].plot(epsilons, means, label=label, color=color,
                      marker="s", markersize=4, linewidth=1.4)
    axes[1].set_xlabel(r"Asymmetric perturbation $\varepsilon$")
    axes[1].set_ylabel(r"$D_4$ symmetry error of $u_\theta$")
    axes[1].set_yscale("log")
    axes[1].set_title("(b) Output symmetry deviation")
    axes[1].legend(loc="best")

    fig.tight_layout()
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


def fig6_speedup(summary: list[dict], fig_path: Path) -> None:
    apply_sci_style()
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    ns = np.array([r["batch_size"] for r in summary])
    seq = np.array([r["seq_mean"] * 1e3 for r in summary])
    bat = np.array([r["bat_mean"] * 1e3 for r in summary])
    seq_e = np.array([r["seq_std"] * 1e3 for r in summary])
    bat_e = np.array([r["bat_std"] * 1e3 for r in summary])
    ax.errorbar(ns, seq, yerr=seq_e, label="Sequential loop",
                 color=COLORS["reference"], marker="o", capsize=2.5)
    ax.errorbar(ns, bat, yerr=bat_e, label=r"Batched ($\mathit{einsum}$)",
                 color=model_color("d4"), marker="s", capsize=2.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch size $N$")
    ax.set_ylabel(r"Forward$+$backward time [ms]")
    ax.set_title(r"Batched vs sequential $D_4$ averaging")
    # Annotate per-N speedup.
    for i, r in enumerate(summary):
        ax.annotate(f"{r['speedup']:.2f}x", (ns[i], bat[i]),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=7, color=COLORS["text"])
    ax.legend(loc="upper left")
    fig.tight_layout()
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


def fig7_loss_landscape(landscapes: dict, fig_path: Path) -> None:
    apply_sci_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    for ax, key in zip(axes, ("standard", "d4")):
        a = np.array(landscapes[key]["alphas"])
        b = np.array(landscapes[key]["betas"])
        L = np.array(landscapes[key]["loss_grid"])
        # Use log scale so wide minima are visible.
        Lp = np.log10(np.maximum(L, 1e-10))
        AA, BB = np.meshgrid(a, b, indexing="ij")
        cs = ax.contourf(AA, BB, Lp, levels=18, cmap="viridis")
        ax.contour(AA, BB, Lp, levels=10, colors="white", linewidths=0.4,
                    alpha=0.5)
        ax.scatter([0], [0], color=COLORS["reference"], marker="*", s=60,
                    edgecolors="white", linewidths=0.6, zorder=3,
                    label="Trained minimum")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\alpha$ (filter-norm. dir. 1)")
        ax.set_ylabel(r"$\beta$ (filter-norm. dir. 2)")
        ax.set_title(model_label(key))
        ax.grid(False)
        cb = fig.colorbar(cs, ax=ax, shrink=0.85, pad=0.03)
        cb.set_label(r"$\log_{10}\, \mathcal{L}_{\rm total}$")
    axes[0].legend(loc="lower right", fontsize=7)
    for ax, label in zip(axes, "ab"):
        ax.text(-0.12, 1.04, f"({label})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


# ---------------------------------------------------------------------------
# New figure functions for revised manuscript
# ---------------------------------------------------------------------------

def _run_matched_sample_efficiency(args, seeds, epochs, tab_dir, fig_dir):
    """Run the matched-effective-sample-size experiment: D4-PINN with n points
    vs StandardPINN with 8n points. This directly verifies Theorem 2."""
    from d4pinn import ReactionDiffusionBVP, StandardPINN, D4PINN, train_one_run
    pde = ReactionDiffusionBVP(k=1.0, epsilon=0.0)
    n_grid = getattr(args, "sample_efficiency_grid", [200, 500, 1000, 2000])
    layers = args.layers
    lr = args.lr
    lambda_bc = args.lambda_bc
    eval_every = max(epochs // 4, 100)
    eff_epochs = max(epochs // 2, 200)

    out: dict[int, dict[str, list[dict]]] = {}
    for n in n_grid:
        out[n] = {"d4_n": [], "std_8n": []}
        n_bc_n = max(40, n // 10)
        n_bc_8n = max(40, (8 * n) // 10)
        for s in seeds:
            import time
            # D4-PINN with n points
            t0 = time.time()
            _, h_d4 = train_one_run(
                pde=pde, model=D4PINN(layers=layers),
                n_int=n, n_bc=n_bc_n, epochs=eff_epochs,
                learning_rate=lr, lambda_bc=lambda_bc, lambda_sym=None,
                seed=s, eval_every=eval_every, verbose=False,
            )
            wall_d4 = time.time() - t0
            out[n]["d4_n"].append({
                "L2": h_d4.l2_error[-1] if h_d4.l2_error else float("nan"),
                "wall_s": wall_d4,
            })
            # Standard PINN with 8n points
            t0 = time.time()
            _, h_std = train_one_run(
                pde=pde, model=StandardPINN(layers=layers),
                n_int=8*n, n_bc=n_bc_8n, epochs=eff_epochs,
                learning_rate=lr, lambda_bc=lambda_bc, lambda_sym=None,
                seed=s, eval_every=eval_every, verbose=False,
            )
            wall_std = time.time() - t0
            out[n]["std_8n"].append({
                "L2": h_std.l2_error[-1] if h_std.l2_error else float("nan"),
                "wall_s": wall_std,
            })
            LOG.info(f"   matched n={n}: D4(n) L2={out[n]['d4_n'][-1]['L2']:.3e} "
                     f"vs Std(8n) L2={out[n]['std_8n'][-1]['L2']:.3e}")
    return out


def fig9_sample_efficiency_enhanced(se_data, matched_data, fig_path):
    """Enhanced sample efficiency: (a) standard comparison, (b) n vs 8n verification."""
    apply_sci_style()
    n_grid = sorted(se_data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))

    # Panel (a): standard comparison at equal N_int
    for arm, color, label in (("standard", model_color("standard"), model_label("standard")),
                                ("d4", model_color("d4"), model_label("d4"))):
        means = [np.mean([r["L2"] for r in se_data[n][arm]]) for n in n_grid]
        stds = [np.std([r["L2"] for r in se_data[n][arm]], ddof=1)
                if len(se_data[n][arm]) > 1 else 0 for n in n_grid]
        axes[0].errorbar(n_grid, means, yerr=stds, color=color, label=label,
                          marker="o", capsize=3, linewidth=1.4)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$N_{\rm int}$"); axes[0].set_ylabel(r"Relative $L_2$ error")
    axes[0].set_title("(a) Equal collocation count")
    axes[0].legend(loc="upper right")

    # Panel (b): matched effective sample size
    if matched_data:
        m_n_grid = sorted(matched_data.keys())
        d4_means = [np.mean([r["L2"] for r in matched_data[n]["d4_n"]]) for n in m_n_grid]
        d4_stds = [np.std([r["L2"] for r in matched_data[n]["d4_n"]], ddof=1)
                    if len(matched_data[n]["d4_n"]) > 1 else 0 for n in m_n_grid]
        std_means = [np.mean([r["L2"] for r in matched_data[n]["std_8n"]]) for n in m_n_grid]
        std_stds = [np.std([r["L2"] for r in matched_data[n]["std_8n"]], ddof=1)
                     if len(matched_data[n]["std_8n"]) > 1 else 0 for n in m_n_grid]
        axes[1].errorbar(m_n_grid, d4_means, yerr=d4_stds,
                          color=model_color("d4"), label=r"$D_4$-PINN ($n$ pts)",
                          marker="o", capsize=3, linewidth=1.4)
        axes[1].errorbar(m_n_grid, std_means, yerr=std_stds,
                          color=model_color("standard"), label=r"Standard PINN ($8n$ pts)",
                          marker="s", capsize=3, linewidth=1.4)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$n$ (base collocation count)")
    axes[1].set_ylabel(r"Relative $L_2$ error")
    axes[1].set_title("(b) Matched effective sample size")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


def fig_inverse_enhanced(inv_data: dict, fig_path: Path):
    """Four-panel enhanced inverse problem figure."""
    apply_sci_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.5))
    proto_colors = {
        "standard_eqcompute": COLORS["standard"],
        "d4": model_color("d4"),
        "two_stage": COLORS["neutral"],
        "soft_eqcompute": COLORS["soft"],
    }
    proto_labels = {
        "standard_eqcompute": "Standard (eq. compute)",
        "d4": r"$D_4$-PINN",
        "two_stage": "Two-stage",
        "soft_eqcompute": "Soft constr. (eq. compute)",
    }

    # (a) k recovery
    for i, proto in enumerate(["standard_eqcompute", "d4", "two_stage", "soft_eqcompute"]):
        if proto not in inv_data:
            continue
        runs = inv_data[proto]
        ys = [r["k_err"] for r in runs if not r.get("diverged", False)]
        xs = np.full(len(ys), i) + np.random.uniform(-0.08, 0.08, len(ys))
        axes[0, 0].scatter(xs, ys, color=proto_colors[proto], s=24,
                            edgecolors="white", linewidths=0.6)
        if ys:
            axes[0, 0].hlines(np.mean(ys), i - 0.22, i + 0.22,
                               color=proto_colors[proto], linewidth=2)
    axes[0, 0].set_xticks(range(len(proto_colors)))
    axes[0, 0].set_xticklabels([proto_labels[p] for p in proto_colors if p in inv_data],
                                rotation=20, ha="right", fontsize=7)
    axes[0, 0].set_ylabel(r"Relative error in $\hat{k}$")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("(a) Reaction rate $k$ recovery")

    # (b) A recovery
    for i, proto in enumerate(["standard_eqcompute", "d4", "two_stage", "soft_eqcompute"]):
        if proto not in inv_data:
            continue
        runs = inv_data[proto]
        ys = [r["A_err"] for r in runs if not r.get("diverged", False)]
        xs = np.full(len(ys), i) + np.random.uniform(-0.08, 0.08, len(ys))
        axes[0, 1].scatter(xs, ys, color=proto_colors[proto], s=24,
                            edgecolors="white", linewidths=0.6)
        if ys:
            axes[0, 1].hlines(np.mean(ys), i - 0.22, i + 0.22,
                               color=proto_colors[proto], linewidth=2)
    axes[0, 1].set_xticks(range(len(proto_colors)))
    axes[0, 1].set_xticklabels([proto_labels[p] for p in proto_colors if p in inv_data],
                                rotation=20, ha="right", fontsize=7)
    axes[0, 1].set_ylabel(r"Relative error in $\hat{A}$")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("(b) Source amplitude $A$ recovery")

    # (c) k_final vs k_true
    for i, proto in enumerate(["standard_eqcompute", "d4", "two_stage", "soft_eqcompute"]):
        if proto not in inv_data:
            continue
        runs = inv_data[proto]
        k_vals = [r["k_final"] for r in runs if not r.get("diverged", False)]
        xs = np.full(len(k_vals), i) + np.random.uniform(-0.08, 0.08, len(k_vals))
        axes[1, 0].scatter(xs, k_vals, color=proto_colors[proto], s=24,
                            edgecolors="white", linewidths=0.6)
        if k_vals:
            axes[1, 0].hlines(np.mean(k_vals), i - 0.22, i + 0.22,
                               color=proto_colors[proto], linewidth=2)
    axes[1, 0].axhline(1.0, linestyle="--", color=COLORS["text"], alpha=0.5, linewidth=0.8)
    axes[1, 0].set_xticks(range(len(proto_colors)))
    axes[1, 0].set_xticklabels([proto_labels[p] for p in proto_colors if p in inv_data],
                                rotation=20, ha="right", fontsize=7)
    axes[1, 0].set_ylabel(r"Recovered $\hat{k}$")
    axes[1, 0].set_title(r"(c) $\hat{k}$ vs. $k^\star=1$")

    # (d) wall-clock comparison
    for i, proto in enumerate(["standard_eqcompute", "d4", "two_stage", "soft_eqcompute"]):
        if proto not in inv_data:
            continue
        runs = inv_data[proto]
        walls = [r["wall_s"] for r in runs]
        if walls:
            axes[1, 1].bar(i, np.mean(walls),
                            yerr=np.std(walls, ddof=1) if len(walls) > 1 else 0,
                            color=proto_colors[proto], capsize=3,
                            edgecolor="white", linewidth=0.6)
    axes[1, 1].set_xticks(range(len(proto_colors)))
    axes[1, 1].set_xticklabels([proto_labels[p] for p in proto_colors if p in inv_data],
                                rotation=20, ha="right", fontsize=7)
    axes[1, 1].set_ylabel("Wall-clock time [s]")
    axes[1, 1].set_title("(d) Computational cost")

    fig.tight_layout()
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


def fig_allen_cahn(ac_data: dict, fig_path: Path):
    """Four-panel Allen-Cahn figure."""
    apply_sci_style()
    epsilons = sorted(ac_data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.5))

    # (a) L2 vs epsilon (large epsilon, symmetric regime)
    large_eps = [e for e in epsilons if e >= 0.05]
    for arch, color, label in (
        ("standard", model_color("standard"), model_label("standard")),
        ("d4", model_color("d4"), model_label("d4")),
    ):
        means = [np.mean([r["L2"] for r in ac_data[e][arch]]) for e in large_eps]
        stds = [np.std([r["L2"] for r in ac_data[e][arch]], ddof=1)
                if len(ac_data[e][arch]) > 1 else 0 for e in large_eps]
        axes[0, 0].errorbar(large_eps, means, yerr=stds, color=color, label=label,
                              marker="o", capsize=3, linewidth=1.4)
    axes[0, 0].set_xlabel(r"$\epsilon$"); axes[0, 0].set_ylabel(r"Relative $L_2$ error")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title(r"(a) $L_2$ error vs. $\epsilon$")
    axes[0, 0].legend(loc="best")
    axes[0, 0].axvline(0.05, linestyle=":", color=COLORS["text"], alpha=0.5)
    axes[0, 0].text(0.048, 0.5, r"$\epsilon_c$", transform=axes[0, 0].get_xaxis_transform(),
                     fontsize=8, ha="right")

    # (b) Symmetry error vs epsilon
    for arch, color, label in (
        ("standard", model_color("standard"), model_label("standard")),
        ("d4", model_color("d4"), model_label("d4")),
    ):
        means = [max(np.mean([r["sym"] for r in ac_data[e][arch]]), 1e-12) for e in epsilons]
        axes[0, 1].plot(epsilons, means, color=color, label=label,
                          marker="s", markersize=4, linewidth=1.4)
    axes[0, 1].set_xlabel(r"$\epsilon$")
    axes[0, 1].set_ylabel(r"$D_4$ symmetry error")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title(r"(b) Symmetry deviation vs. $\epsilon$")
    axes[0, 1].legend(loc="best")
    axes[0, 1].axhline(1e-7, linestyle=":", color=COLORS["text"], alpha=0.5)

    # (c) Bar chart at epsilon = 0.02 (symmetry-breaking)
    small_eps = min(epsilons)
    arms = ["standard", "d4"]
    means_small = [np.mean([r["L2"] for r in ac_data[small_eps][a]]) for a in arms]
    stds_small = [np.std([r["L2"] for r in ac_data[small_eps][a]], ddof=1)
                   if len(ac_data[small_eps][a]) > 1 else 0 for a in arms]
    bar_colors = [model_color("standard"), model_color("d4")]
    axes[1, 0].bar(range(len(arms)), means_small, yerr=stds_small, color=bar_colors,
                    capsize=3, edgecolor="white", linewidth=0.6)
    axes[1, 0].set_xticks(range(len(arms)))
    axes[1, 0].set_xticklabels([model_label("standard"), model_label("d4")], fontsize=8)
    axes[1, 0].set_ylabel(r"Relative $L_2$ error")
    axes[1, 0].set_title(rf"(c) $\epsilon = {small_eps}$: symmetry-breaking regime")
    # Annotate
    for i, (m, s) in enumerate(zip(means_small, stds_small)):
        axes[1, 0].text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")

    # (d) Wall time comparison
    for arch, color in (("standard", model_color("standard")),
                          ("d4", model_color("d4"))):
        walls = [np.mean([r["wall_s"] for r in ac_data[e][arch]]) for e in epsilons]
        axes[1, 1].plot(epsilons, walls, color=color, marker="o",
                          linewidth=1.4, label=model_label(arch.split("_")[0])
                          if arch == "d4" else model_label(arch))
    axes[1, 1].set_xlabel(r"$\epsilon$")
    axes[1, 1].set_ylabel("Wall-clock time [s]")
    axes[1, 1].set_title("(d) Computational cost")
    axes[1, 1].legend(loc="best")

    fig.suptitle(r"Allen--Cahn equation: $\epsilon^2 \Delta u + u - u^3 = 0$",
                  fontsize=11, y=1.01)
    fig.tight_layout()
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


def fig_soft_constraint_sweep(data: dict, fig_path: Path) -> None:
    """Soft-constraint lambda_sym sweep: L2 and sym error vs weight."""
    apply_sci_style()
    lam_vals = sorted(data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))

    soft_l2 = [np.mean([r["L2"] for r in data[lam]["soft"]
                        if np.isfinite(r["L2"])]) for lam in lam_vals]
    soft_sym = [max(np.mean([r["sym"] for r in data[lam]["soft"]
                              if np.isfinite(r["sym"])]), 1e-12) for lam in lam_vals]

    # Reference lines from D4-PINN and StandardPINN
    ref_lam = lam_vals[0]
    d4_l2 = np.mean([r["L2"] for r in data[ref_lam]["d4"] if np.isfinite(r["L2"])])
    std_l2 = np.mean([r["L2"] for r in data[ref_lam]["standard"] if np.isfinite(r["L2"])])
    d4_sym = max(np.mean([r["sym"] for r in data[ref_lam]["d4"]
                           if np.isfinite(r["sym"])]), 1e-12)

    # (a) L2 vs lambda_sym
    axes[0].semilogx(lam_vals, soft_l2, "o-", color=COLORS["soft"],
                     label="Soft constraint", linewidth=1.4)
    axes[0].axhline(d4_l2, color=model_color("d4"), linestyle="--",
                    label=model_label("d4"), linewidth=1.2)
    axes[0].axhline(std_l2, color=model_color("standard"), linestyle=":",
                    label=model_label("standard"), linewidth=1.2)
    best_idx = np.argmin(soft_l2)
    axes[0].scatter([lam_vals[best_idx]], [soft_l2[best_idx]], color=COLORS["soft"],
                    marker="*", s=80, zorder=5, edgecolors="white")
    axes[0].set_xlabel(r"$\lambda_{\rm sym}$")
    axes[0].set_ylabel(r"Relative $L_2$ error")
    axes[0].set_title(r"(a) $L_2$ error vs. $\lambda_{\rm sym}$")
    axes[0].legend(loc="best", fontsize=7)
    axes[0].set_yscale("log")

    # (b) Symmetry error vs lambda_sym
    axes[1].loglog(lam_vals, soft_sym, "s-", color=COLORS["soft"],
                   label="Soft constraint", linewidth=1.4)
    axes[1].axhline(d4_sym, color=model_color("d4"), linestyle="--",
                    label=model_label("d4"), linewidth=1.2)
    axes[1].axhline(1e-7, color=COLORS["text"], linestyle=":", alpha=0.5)
    axes[1].set_xlabel(r"$\lambda_{\rm sym}$")
    axes[1].set_ylabel(r"$D_4$ symmetry error")
    axes[1].set_title(r"(b) Symmetry error vs. $\lambda_{\rm sym}$")
    axes[1].legend(loc="best", fontsize=7)

    fig.suptitle(r"Soft-constraint $\lambda_{\rm sym}$ sweep", y=1.01)
    fig.tight_layout()
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


def fig_hard_bc(data: dict, fig_path: Path) -> None:
    """Hard BC comparison: soft vs hard BC for Standard and D4 PINN."""
    apply_sci_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))
    archs = ["standard_soft", "standard_hard", "d4_soft", "d4_hard"]
    labels = ["Standard\n(soft BC)", "Standard\n(hard BC)",
              r"$D_4$-PINN" + "\n(soft BC)", r"$D_4$-PINN" + "\n(hard BC)"]
    colors = [model_color("standard"), "#E0975B",
              model_color("d4"), "#5BA0E0"]

    # (a) L2 error
    means_l2 = [np.mean([r["L2"] for r in data[a] if np.isfinite(r["L2"])]) for a in archs]
    stds_l2 = [np.std([r["L2"] for r in data[a] if np.isfinite(r["L2"])], ddof=1)
               if len(data[a]) > 1 else 0 for a in archs]
    axes[0].bar(range(len(archs)), means_l2, yerr=stds_l2, color=colors,
                capsize=3, edgecolor="white", linewidth=0.6)
    axes[0].set_xticks(range(len(archs)))
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel(r"Relative $L_2$ error")
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Forward-problem accuracy")

    # (b) BC error
    means_bc = [np.mean([r["bc_err"] for r in data[a] if np.isfinite(r["bc_err"])]) for a in archs]
    axes[1].bar(range(len(archs)), means_bc, color=colors,
                capsize=3, edgecolor="white", linewidth=0.6)
    axes[1].set_xticks(range(len(archs)))
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel(r"Boundary MSE")
    axes[1].set_yscale("log")
    axes[1].set_title("(b) Boundary-condition error")
    axes[1].axhline(1e-7, linestyle=":", color=COLORS["text"], alpha=0.5)

    fig.tight_layout()
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


def fig_cubic(data: dict, fig_path: Path) -> None:
    """Cubic semilinear Poisson: bar chart comparing three architectures."""
    apply_sci_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    archs = ["standard", "d4", "algebraic"]
    labels = [model_label("standard"), model_label("d4"),
              r"$D_4$-PINN (alg. inv.)"]
    colors = [model_color("standard"), model_color("d4"), "#5BA0E0"]

    # (a) L2 error
    means = [np.mean([r["L2"] for r in data[a] if np.isfinite(r["L2"])]) for a in archs]
    stds = [np.std([r["L2"] for r in data[a] if np.isfinite(r["L2"])], ddof=1)
            if len(data[a]) > 1 else 0 for a in archs]
    axes[0].bar(range(len(archs)), means, yerr=stds, color=colors,
                capsize=3, edgecolor="white", linewidth=0.6)
    axes[0].set_xticks(range(len(archs)))
    axes[0].set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
    axes[0].set_ylabel(r"Relative $L_2$ error")
    axes[0].set_yscale("log")
    axes[0].set_title(r"(a) Cubic semilinear: $-\Delta u + k u^3 = f$")

    # (b) Symmetry error
    sym_means = [max(np.mean([r["sym"] for r in data[a] if np.isfinite(r["sym"])]), 1e-12)
                 for a in archs]
    axes[1].bar(range(len(archs)), sym_means, color=colors,
                capsize=3, edgecolor="white", linewidth=0.6)
    axes[1].set_xticks(range(len(archs)))
    axes[1].set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
    axes[1].set_ylabel(r"$D_4$ symmetry error")
    axes[1].set_yscale("log")
    axes[1].axhline(1e-7, linestyle=":", color=COLORS["text"], alpha=0.5)
    axes[1].set_title("(b) Symmetry deviation")

    fig.tight_layout()
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


def fig_gpu_benchmark(data: dict, fig_path: Path) -> None:
    """GPU throughput comparison: training and inference."""
    apply_sci_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))

    batch_sizes = [r["n_int"] for r in data["standard"]]

    # (a) Training throughput (samples/sec)
    for arch, color, label in ("standard", model_color("standard"), model_label("standard")), \
                               ("d4", model_color("d4"), model_label("d4")):
        throughput = [r["samples_per_sec"] for r in data[arch]]
        axes[0].plot(batch_sizes, throughput, "o-", color=color, label=label, linewidth=1.4)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Batch size $N_{\\rm int}$")
    axes[0].set_ylabel("Training throughput [samples/s]")
    axes[0].set_title("(a) Training throughput")
    axes[0].legend(loc="best")

    # (b) Inference latency
    for arch, color, label in ("standard", model_color("standard"), model_label("standard")), \
                               ("d4", model_color("d4"), model_label("d4")):
        latency = [r["infer_time_mean"] * 1e3 for r in data[arch]]
        axes[1].plot(batch_sizes, latency, "s-", color=color, label=label, linewidth=1.4)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Batch size $N_{\\rm int}$")
    axes[1].set_ylabel("Inference latency [ms]")
    axes[1].set_title("(b) Inference latency")
    axes[1].legend(loc="best")

    dev = data["standard"][0]["device"]
    fig.suptitle(f"GPU benchmark (device={dev})", y=1.01)
    fig.tight_layout()
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="D4-PINN master pipeline")
    p.add_argument("--quick", action="store_true",
                   help="Use small settings for ~5-10 min total runtime.")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n_int", type=int, default=None)
    p.add_argument("--n_bc", type=int, default=None)
    p.add_argument("--layers", type=int, nargs="+", default=[2, 50, 50, 50, 1])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_bc", type=float, default=10.0)
    p.add_argument("--lambda_sym", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--epsilons", type=float, nargs="+",
                    default=[0.0, 0.01, 0.05, 0.10, 0.20, 0.50])
    p.add_argument("--batch_sizes", type=int, nargs="+",
                    default=[500, 2000, 5000, 10000])
    p.add_argument("--landscape_grid", type=int, default=21)
    p.add_argument("--landscape_span", type=float, default=0.5)
    p.add_argument("--out", type=str, default=None,
                    help="Output directory; if not given, uses output_<timestamp>")
    # ---- E6-E12 specific flags ----
    p.add_argument("--run_extras", action="store_true",
                    help="Run experiments E6-E12 (ablation, sample efficiency, "
                         "hyperparameter robustness, noise robustness, 3D, "
                         "inverse, homotopy). Default in --quick: True.")
    p.add_argument("--no_extras", dest="run_extras", action="store_false",
                    help="Skip experiments E6-E12.")
    p.set_defaults(run_extras=True)
    p.add_argument("--sample_efficiency_grid", type=int, nargs="+",
                    default=[200, 500, 1000, 2000])
    p.add_argument("--hp_widths", type=int, nargs="+", default=[20, 50, 100])
    p.add_argument("--hp_lrs", type=float, nargs="+", default=[5e-4, 1e-3, 5e-3])
    p.add_argument("--noise_levels", type=float, nargs="+",
                    default=[0.0, 0.05, 0.10, 0.20, 0.50])
    p.add_argument("--n_obs", type=int, default=400,
                    help="Number of point observations for the inverse problem.")
    p.add_argument("--inverse_noise", type=float, default=0.05,
                    help="Gaussian noise level on observations (rel).")
    p.add_argument("--k_init", type=float, default=2.0,
                    help="Initial guess for the unknown coefficient k.")
    p.add_argument("--allen_cahn_epsilons", type=float, nargs="+",
                    default=[0.10, 0.05, 0.02],
                    help="Epsilon values for Allen-Cahn experiment (E14).")
    args = p.parse_args()

    # Defaults
    if args.quick:
        seeds = args.seeds or [0, 1]
        epochs = args.epochs or 600
        n_int = args.n_int or 800
        n_bc = args.n_bc or 160
        sb_seeds = seeds
        sb_epochs = max(epochs // 2, 200)
        speed_repeats = 3
        landscape_grid = min(args.landscape_grid, 13)
    else:
        seeds = args.seeds or [0, 1, 2]
        epochs = args.epochs or 2500
        n_int = args.n_int or 1500
        n_bc = args.n_bc or 300
        sb_seeds = seeds
        sb_epochs = max(epochs // 2, 600)
        speed_repeats = 5
        landscape_grid = args.landscape_grid

    out_root = Path(args.out) if args.out else Path(
        f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    fig_dir.mkdir(exist_ok=True)
    tab_dir.mkdir(exist_ok=True)

    _setup_logging(out_root / "run_log.txt")
    LOG.info(f"Output directory: {out_root.resolve()}")
    LOG.info(f"PyTorch version: {torch.__version__}")
    LOG.info(f"CPU threads: {torch.get_num_threads()}")

    # Save config
    cfg = vars(args).copy()
    cfg.update({"effective": {
        "seeds": seeds, "epochs": epochs, "n_int": n_int, "n_bc": n_bc,
        "sb_seeds": sb_seeds, "sb_epochs": sb_epochs,
        "speed_repeats": speed_repeats, "landscape_grid": landscape_grid,
    }})
    (out_root / "run_config.json").write_text(json.dumps(cfg, indent=2))

    overall_start = time.time()

    # --- E2 invariance verification (cheap, do first) ---
    inv_summary = exp2_invariance_verification(
        layers=args.layers, n_test=200, seeds=seeds, tables_dir=tab_dir,
    )
    fig3_symmetry_error_bar(inv_summary,
                              fig_dir / "fig3_symmetry_error_bar.pdf")

    # --- E4 batched speedup (cheap) ---
    speedup_summary = exp4_batched_speedup(
        batch_sizes=args.batch_sizes, n_repeats=speed_repeats,
        layers=args.layers, tables_dir=tab_dir,
    )
    fig6_speedup(speedup_summary, fig_dir / "fig6_batched_speedup.pdf")

    # --- E1 main benchmark ---
    main_results = exp1_main_benchmark(
        seeds=seeds, epochs=epochs, n_int=n_int, n_bc=n_bc,
        layers=args.layers, lr=args.lr, lambda_bc=args.lambda_bc,
        lambda_sym=args.lambda_sym, eval_every=args.eval_every,
        tables_dir=tab_dir,
    )
    fig1_convergence(main_results, fig_dir / "fig1_convergence.pdf")
    fig2_l2_vs_seeds(main_results, fig_dir / "fig2_l2_vs_seeds.pdf")
    fig4_solution_field(main_results, fig_dir / "fig4_solution_field.pdf",
                          args.layers, n_int=n_int, n_bc=n_bc,
                          epochs=epochs)

    # --- E3 symmetry-breaking sweep ---
    sb_agg = exp3_symmetry_breaking_sweep(
        epsilons=args.epsilons, seeds=sb_seeds, epochs=sb_epochs,
        n_int=n_int, n_bc=n_bc, lr=args.lr, lambda_bc=args.lambda_bc,
        layers=args.layers, eval_every=max(sb_epochs // 2, 1),
        tables_dir=tab_dir,
    )
    fig5_symmetry_breaking(sb_agg, fig_dir / "fig5_symmetry_breaking.pdf")

    # --- E5 loss landscape ---
    landscapes = exp5_loss_landscape(
        layers=args.layers, train_epochs=max(epochs // 2, 500),
        n_int=n_int, n_bc=n_bc, lr=args.lr, lambda_bc=args.lambda_bc,
        grid_n=landscape_grid, span=args.landscape_span,
        tables_dir=tab_dir,
    )
    fig7_loss_landscape(landscapes, fig_dir / "fig7_loss_landscape.pdf")

    # --- New experiments E6-E12 ---
    if args.run_extras:
        run_extra_experiments(args, seeds, epochs, n_int, n_bc, tab_dir, fig_dir)

    elapsed = (time.time() - overall_start) / 60
    LOG.info("=" * 70)
    LOG.info(f"All experiments complete in {elapsed:.1f} min")
    LOG.info(f"Tables  -> {tab_dir.resolve()}")
    LOG.info(f"Figures -> {fig_dir.resolve()}")
    LOG.info(f"Log file -> {(out_root / 'run_log.txt').resolve()}")
    LOG.info("=" * 70)


# ---------------------------------------------------------------------------
# Extra experiments E6-E12 + their figures
# ---------------------------------------------------------------------------

def fig8_ablation(arms_data: dict, fig_path: Path) -> None:
    apply_sci_style()
    arms = ["no_constraint", "rotations_only", "reflections_only", "full_d4"]
    pretty = ["No\nconstraint", "Rotations\nonly ($C_4$)", "Reflections\nonly", r"Full $D_4$"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    # (a) L2
    means = [np.mean([r["L2"] for r in arms_data[a]]) for a in arms]
    stds = [np.std([r["L2"] for r in arms_data[a]], ddof=1) if len(arms_data[a]) > 1 else 0 for a in arms]
    bar_colors = [COLORS["standard"], "#9E7BB5", "#E0975B", model_color("d4")]
    axes[0].bar(range(len(arms)), means, yerr=stds, color=bar_colors,
                  capsize=3, edgecolor="white", linewidth=0.6)
    axes[0].set_xticks(range(len(arms)))
    axes[0].set_xticklabels(pretty, fontsize=8)
    axes[0].set_ylabel(r"Relative $L_2$ error")
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Forward-problem accuracy")
    # (b) Symmetry error
    sym_means = [max(np.mean([r["sym"] for r in arms_data[a]]), 1e-12) for a in arms]
    axes[1].bar(range(len(arms)), sym_means, color=bar_colors,
                  edgecolor="white", linewidth=0.6)
    axes[1].set_xticks(range(len(arms)))
    axes[1].set_xticklabels(pretty, fontsize=8)
    axes[1].set_ylabel(r"$D_4$ symmetry error")
    axes[1].set_yscale("log")
    axes[1].axhline(1e-7, linestyle=":", color=COLORS["text"], linewidth=0.7,
                      alpha=0.6)
    axes[1].set_title("(b) Output symmetry deviation")
    fig.tight_layout()
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


def fig9_sample_efficiency(data: dict, fig_path: Path) -> None:
    apply_sci_style()
    n_grid = sorted(data.keys())
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for arm, color, label in (("standard", model_color("standard"), model_label("standard")),
                                ("d4",       model_color("d4"),       model_label("d4"))):
        means = [np.mean([r["L2"] for r in data[n][arm]]) for n in n_grid]
        stds = [np.std([r["L2"] for r in data[n][arm]], ddof=1) if len(data[n][arm]) > 1 else 0 for n in n_grid]
        ax.errorbar(n_grid, means, yerr=stds, color=color, label=label,
                     marker="o", capsize=3, linewidth=1.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Interior collocation points $N_{\rm int}$")
    ax.set_ylabel(r"Relative $L_2$ error")
    ax.set_title(r"Sample-efficiency curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


def fig10_hyperparam_robustness(rows: list[dict], fig_path: Path) -> None:
    apply_sci_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    # Group rows by arm.
    arms = ["standard", "d4"]
    arm_labels = {"standard": model_label("standard"), "d4": model_label("d4")}
    arm_colors = {"standard": model_color("standard"), "d4": model_color("d4")}
    # (a) box plot of L2 error per arm across all (width, lr, seed) cells
    data = {a: [r["L2"] for r in rows
                  if r["arm"] == a and not r.get("diverged", False)
                  and np.isfinite(r["L2"])]
            for a in arms}
    bp = axes[0].boxplot([data[a] for a in arms],
                          tick_labels=[arm_labels[a] for a in arms],
                          patch_artist=True, widths=0.5)
    for patch, a in zip(bp["boxes"], arms):
        patch.set_facecolor(arm_colors[a])
        patch.set_alpha(0.55)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Relative $L_2$ error")
    axes[0].set_title("(a) Distribution across hyperparameters")
    # (b) heatmap-like: median L2 over seed at each (width, lr)
    widths = sorted(set(r["width"] for r in rows))
    lrs = sorted(set(r["lr"] for r in rows))
    grid_d4 = np.full((len(widths), len(lrs)), np.nan)
    grid_std = np.full((len(widths), len(lrs)), np.nan)
    for i, w in enumerate(widths):
        for j, lr in enumerate(lrs):
            d4_vals = [r["L2"] for r in rows
                        if r["arm"] == "d4" and r["width"] == w and r["lr"] == lr
                        and np.isfinite(r["L2"])]
            std_vals = [r["L2"] for r in rows
                          if r["arm"] == "standard" and r["width"] == w and r["lr"] == lr
                          and np.isfinite(r["L2"])]
            if d4_vals: grid_d4[i, j] = np.median(d4_vals)
            if std_vals: grid_std[i, j] = np.median(std_vals)
    # Show ratio std/d4 as a heatmap (ratio >> 1 means D4 wins).
    ratio = grid_std / grid_d4
    im = axes[1].imshow(ratio, origin="lower", aspect="auto",
                          cmap="RdBu_r", vmin=0.5, vmax=3.0)
    axes[1].set_xticks(range(len(lrs)))
    axes[1].set_xticklabels([f"{lr:.0e}" for lr in lrs])
    axes[1].set_yticks(range(len(widths)))
    axes[1].set_yticklabels(widths)
    axes[1].set_xlabel("Learning rate")
    axes[1].set_ylabel("Width")
    axes[1].set_title("(b) ratio Std / $D_4$ median $L_2$")
    axes[1].grid(False)
    cb = fig.colorbar(im, ax=axes[1], shrink=0.85, pad=0.03)
    cb.set_label(r"$L_2^{\rm Std} / L_2^{D_4}$")
    fig.tight_layout()
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


def fig11_noise_robustness(data: dict, fig_path: Path) -> None:
    apply_sci_style()
    sigmas = sorted(data.keys())
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for arm, color, label in (("standard", model_color("standard"), model_label("standard")),
                                ("d4",       model_color("d4"),       model_label("d4"))):
        means = [np.mean([r["L2"] for r in data[s][arm]]) for s in sigmas]
        stds = [np.std([r["L2"] for r in data[s][arm]], ddof=1) if len(data[s][arm]) > 1 else 0 for s in sigmas]
        ax.errorbar(sigmas, means, yerr=stds, color=color, label=label,
                     marker="o", capsize=3, linewidth=1.4)
    ax.set_yscale("log")
    ax.set_xlabel(r"Source-noise std $\sigma$")
    ax.set_ylabel(r"Relative $L_2$ error")
    ax.set_title("Noise robustness")
    ax.legend(loc="best")
    fig.tight_layout()
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


def fig12_3d_extension(data: dict, fig2d_summary: dict, fig_path: Path) -> None:
    apply_sci_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    arms = ["standard3d", "d4_planar3d"]
    means = [np.mean([r["L2"] for r in data[a]]) for a in arms]
    stds = [np.std([r["L2"] for r in data[a]], ddof=1) if len(data[a]) > 1 else 0 for a in arms]
    # Add a 2D reference column for context.
    if fig2d_summary is not None:
        means_full = means + [fig2d_summary["d4"]["summary"]["L2_mean"]]
        stds_full = stds + [fig2d_summary["d4"]["summary"]["L2_std"]]
        labels_full = ["Standard 3D", r"$D_4$-planar 3D", r"$D_4$-PINN 2D (ref)"]
        colors = [model_color("standard"), model_color("d4"), model_color("d4_dark")]
    else:
        means_full, stds_full = means, stds
        labels_full = ["Standard 3D", r"$D_4$-planar 3D"]
        colors = [model_color("standard"), model_color("d4")]
    bars = ax.bar(range(len(means_full)), means_full, yerr=stds_full,
                    color=colors, capsize=3, edgecolor="white", linewidth=0.6)
    ax.set_xticks(range(len(labels_full)))
    ax.set_xticklabels(labels_full, fontsize=8)
    ax.set_ylabel(r"Relative $L_2$ error")
    ax.set_yscale("log")
    ax.set_title(r"3D extension: planar $D_4$ does not constrain $z$")
    fig.tight_layout()
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


def fig13_inverse_homotopy(inv_data: dict, homo_rows: list[dict],
                              fig_path: Path) -> None:
    apply_sci_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    # (a) inverse: scatter of |k_final - k_true| / k_true
    arms = ["standard", "d4"]
    pretty_arm = {"standard": model_label("standard"), "d4": model_label("d4")}
    for i, arm in enumerate(arms):
        runs = inv_data.get(arm, [])
        ys = [r["k_relative_error"] for r in runs if not r.get("diverged", False)]
        xs = np.full(len(ys), i, dtype=float) + np.random.uniform(-0.08, 0.08, len(ys))
        axes[0].scatter(xs, ys, color=model_color(arm), s=24, edgecolors="white",
                          linewidths=0.6)
        if ys:
            axes[0].hlines(np.mean(ys), i - 0.22, i + 0.22, color=model_color(arm),
                              linewidth=2)
    axes[0].set_xticks(range(len(arms)))
    axes[0].set_xticklabels([pretty_arm[a] for a in arms])
    axes[0].set_ylabel(r"Relative error in $\hat{k}$")
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Inverse problem: parameter recovery")
    # (b) homotopy: paired bars per seed
    seeds = [r["seed"] for r in homo_rows]
    homo_l2 = [r["homotopy_L2"] for r in homo_rows]
    base_l2 = [r["baseline_L2"] for r in homo_rows]
    x = np.arange(len(seeds))
    width = 0.38
    axes[1].bar(x - width/2, base_l2, width,
                 label="Single-stage $D_4$", color=model_color("d4"))
    axes[1].bar(x + width/2, homo_l2, width,
                 label="Homotopy: Std $\\to D_4$", color=COLORS["neutral"])
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"s={s}" for s in seeds])
    axes[1].set_ylabel(r"Final $L_2$ error")
    axes[1].set_title("(b) Homotopic vs single-stage $D_4$")
    axes[1].legend(loc="best", fontsize=7.5)
    fig.tight_layout()
    export_fig(fig, fig_path.stem, formats=("pdf", "svg", "png"), outdir=fig_path.parent)
    plt.close(fig)
    LOG.info(f"  Fig  -> {fig_path.name}")


def run_extra_experiments(args, seeds, epochs, n_int, n_bc, tab_dir, fig_dir):
    """Run experiments E6-E12 and new experiments (algebraic invariant,
    Allen-Cahn, enhanced inverse) in order."""
    # ---- E6 ablation ----
    LOG.info("=" * 70)
    LOG.info(f"Experiment 6: ablation study (rotations vs reflections)")
    LOG.info("=" * 70)
    abl_data = exp_ablation.run_ablation(
        seeds=seeds, epochs=max(epochs // 2, 200),
        n_int=n_int, n_bc=n_bc, lr=args.lr, lambda_bc=args.lambda_bc,
        layers=args.layers,
        eval_every=max(epochs // 4, 100), log=LOG,
    )
    write_csv(
        tab_dir / "T6_ablation.csv",
        ["arm", "seed", "L2_rel_error", "symmetry_error", "final_loss",
         "wall_time_s", "n_nan", "diverged"],
        [[r["arm"], r["seed"], r["L2"], r["sym"], r["loss"],
          r["wall_s"], r["n_nan"], int(r["diverged"])]
         for arm in abl_data for r in abl_data[arm]],
    )
    fig8_ablation(abl_data, fig_dir / "fig8_ablation.pdf")

    # ---- E7 sample efficiency (with n vs 8n verification) ----
    LOG.info("=" * 70)
    LOG.info(f"Experiment 7: sample efficiency (with n vs 8n verification)")
    LOG.info("=" * 70)
    se_data = exp_sample.run_sample_efficiency(
        n_int_grid=args.sample_efficiency_grid,
        seeds=seeds, epochs=max(epochs // 2, 200),
        lr=args.lr, lambda_bc=args.lambda_bc,
        layers=args.layers, eval_every=max(epochs // 4, 100), log=LOG,
    )
    # Also run the matched-effective-sample-size experiment: D4 with n vs Std with 8n
    se_d4_vs_8n = _run_matched_sample_efficiency(args, seeds, epochs, tab_dir, fig_dir)
    rows = []
    for n_int_v in se_data:
        for arm in ("standard", "d4"):
            for r in se_data[n_int_v][arm]:
                rows.append([r["n_int"], r["n_bc"], r["arm"], r["seed"],
                              r["L2"], r["sym"], r["wall_s"], int(r["diverged"])])
    write_csv(
        tab_dir / "T7_sample_efficiency.csv",
        ["n_int", "n_bc", "arm", "seed", "L2_rel_error", "symmetry_error",
         "wall_time_s", "diverged"], rows,
    )
    fig9_sample_efficiency_enhanced(se_data, se_d4_vs_8n,
                                     fig_dir / "fig9_sample_efficiency.pdf")

    # ---- E8 hyperparameter robustness ----
    LOG.info("=" * 70)
    LOG.info(f"Experiment 8: hyperparameter robustness")
    LOG.info("=" * 70)
    hp_rows = exp_hp.run_hyperparameter_robustness(
        widths=args.hp_widths, lrs=args.hp_lrs,
        seeds=seeds, depth=3, epochs=max(epochs // 3, 200),
        n_int=n_int, n_bc=n_bc, lambda_bc=args.lambda_bc,
        eval_every=max(epochs // 6, 100), log=LOG,
    )
    write_csv(
        tab_dir / "T8_hyperparam_robustness.csv",
        ["width", "lr", "arm", "seed", "L2_rel_error", "symmetry_error",
         "wall_time_s", "diverged"],
        [[r["width"], r["lr"], r["arm"], r["seed"], r["L2"], r["sym"],
          r["wall_s"], int(r["diverged"])] for r in hp_rows],
    )
    fig10_hyperparam_robustness(hp_rows, fig_dir / "fig10_hyperparam_robustness.pdf")

    # ---- E9 noise robustness ----
    LOG.info("=" * 70)
    LOG.info(f"Experiment 9: noise robustness")
    LOG.info("=" * 70)
    noise_data = exp_noise.run_noise_robustness(
        noise_levels=args.noise_levels, seeds=seeds,
        epochs=max(epochs // 2, 200), n_int=n_int, n_bc=n_bc,
        lr=args.lr, lambda_bc=args.lambda_bc, layers=args.layers,
        eval_every=max(epochs // 4, 100), log=LOG,
    )
    rows = []
    for sigma_v in noise_data:
        for arm in ("standard", "d4"):
            for r in noise_data[sigma_v][arm]:
                rows.append([r["noise_std"], r["arm"], r["seed"], r["L2"],
                              r["sym"], r["wall_s"], int(r["diverged"])])
    write_csv(
        tab_dir / "T9_noise_robustness.csv",
        ["noise_std", "arm", "seed", "L2_rel_error", "symmetry_error",
         "wall_time_s", "diverged"], rows,
    )
    fig11_noise_robustness(noise_data, fig_dir / "fig11_noise_robustness.pdf")

    # ---- E10 3D extension ----
    LOG.info("=" * 70)
    LOG.info(f"Experiment 10: 3D extension (Limitations section)")
    LOG.info("=" * 70)
    e3d_data = exp_3d.run_3d_extension(
        seeds=seeds, epochs=max(epochs // 2, 300),
        n_int=max(n_int, 1500), n_bc=max(n_bc * 2, 600),
        lr=args.lr, lambda_bc=args.lambda_bc,
        layers=[3] + args.layers[1:], eval_every=max(epochs // 4, 100), log=LOG,
    )
    write_csv(
        tab_dir / "T10_3d_extension.csv",
        ["arm", "seed", "L2_rel_error", "wall_time_s", "diverged"],
        [[r["arm"], r["seed"], r["L2"], r["wall_s"], int(r["diverged"])]
         for arm in e3d_data for r in e3d_data[arm]],
    )
    fig12_3d_extension(e3d_data, None, fig_dir / "fig12_3d_extension.pdf")

    # ---- E11 enhanced inverse problem (equal-compute, two-stage, soft-constraint) ----
    LOG.info("=" * 70)
    LOG.info(f"Experiment 11: enhanced inverse problem (equal-compute, two-stage, soft)")
    LOG.info("=" * 70)
    inv_enhanced = exp_inv_enhanced.run_two_param_inverse(
        seeds=seeds, epochs_base=max(epochs, 800),
        n_int=n_int, n_bc=n_bc,
        n_obs=args.n_obs, noise_level=args.inverse_noise,
        lr=args.lr, lambda_bc=args.lambda_bc, lambda_data=10.0,
        lambda_sym=args.lambda_sym, k_init=args.k_init, A_init=0.5,
        k_true=1.0, A_true=1.0, layers=args.layers,
        eval_every=max(epochs // 4, 100), log=LOG,
    )
    # Write CSV for each protocol
    for proto_name, runs in inv_enhanced.items():
        write_csv(
            tab_dir / f"T19_inverse_{proto_name}.csv",
            ["seed", "k_init", "A_init", "k_final", "A_final",
             "k_relative_error", "A_relative_error", "wall_time_s", "diverged"],
            [[r["seed"], r["k_init"], r["A_init"], r["k_final"], r["A_final"],
              r["k_err"], r["A_err"], r["wall_s"], int(r.get("diverged", False))]
             for r in runs],
        )
    fig_inverse_enhanced(inv_enhanced, fig_dir / "fig_inverse_revised.pdf")

    # ---- E13 algebraic invariant baseline ----
    LOG.info("=" * 70)
    LOG.info(f"Experiment 13: algebraic invariant baseline")
    LOG.info("=" * 70)
    alg_data = exp_alg_inv.run_algebraic_invariant_baseline(
        seeds=seeds, epochs=max(epochs // 2, 200),
        n_int=n_int, n_bc=n_bc, lr=args.lr, lambda_bc=args.lambda_bc,
        layers=args.layers, eval_every=max(epochs // 4, 100), log=LOG,
    )
    rows_alg = []
    for arch in ("standard", "d4", "algebraic"):
        for r in alg_data[arch]:
            rows_alg.append([r["arch"], r["seed"], r["L2"], r["sym"],
                              r["n_params"], r["wall_s"], int(r["diverged"])])
    write_csv(
        tab_dir / "T20_algebraic_invariant_baseline.csv",
        ["arch", "seed", "L2_rel_error", "symmetry_error", "n_params",
         "wall_time_s", "diverged"], rows_alg,
    )

    # ---- E14 Allen-Cahn equation ----
    LOG.info("=" * 70)
    LOG.info(f"Experiment 14: Allen-Cahn equation")
    LOG.info("=" * 70)
    ac_epsilons = getattr(args, "allen_cahn_epsilons", [0.10, 0.05, 0.02])
    ac_data = exp_allen_cahn.run_allen_cahn_experiment(
        epsilons=ac_epsilons,
        seeds=seeds, epochs=max(epochs, 5000),
        n_int=max(n_int, 2000), n_bc=max(n_bc, 400),
        lr=args.lr, lambda_bc=args.lambda_bc,
        layers=args.layers, eval_every=max(epochs // 4, 100), log=LOG,
    )
    rows_ac = []
    for eps_val in ac_epsilons:
        for arch in ("standard", "d4", "algebraic"):
            for r in ac_data[eps_val][arch]:
                rows_ac.append([r["epsilon"], r["arch"], r["seed"], r["L2"],
                                 r["sym"], r["wall_s"], int(r["diverged"])])
    write_csv(
        tab_dir / "T21_allen_cahn.csv",
        ["epsilon", "arch", "seed", "L2_rel_error", "symmetry_error",
         "wall_time_s", "diverged"], rows_ac,
    )
    fig_allen_cahn(ac_data, fig_dir / "fig_allen_cahn.pdf")

    # ---- E15 soft-constraint lambda_sym sweep ----
    LOG.info("=" * 70)
    LOG.info(f"Experiment 15: soft-constraint lambda_sym sweep")
    LOG.info("=" * 70)
    soft_sweep_data = exp_soft_sweep.run_soft_constraint_sweep(
        lambda_sym_values=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        seeds=seeds, epochs=max(epochs // 2, 300),
        n_int=n_int, n_bc=n_bc, lr=args.lr, lambda_bc=args.lambda_bc,
        layers=args.layers, eval_every=max(epochs // 4, 100), log=LOG,
    )
    rows_sweep = []
    for lam_val, arch_data in soft_sweep_data.items():
        for arch in ("soft", "d4", "standard"):
            for r in arch_data.get(arch, []):
                rows_sweep.append([lam_val, arch, r["seed"], r["L2"], r["sym"],
                                   r["wall_s"], int(r.get("diverged", False))])
    write_csv(
        tab_dir / "T22_soft_constraint_sweep.csv",
        ["lambda_sym", "arch", "seed", "L2_rel_error", "symmetry_error",
         "wall_time_s", "diverged"], rows_sweep,
    )
    fig_soft_constraint_sweep(soft_sweep_data, fig_dir / "fig_soft_constraint_sweep.pdf")

    # ---- E16 hard BC constraint ----
    LOG.info("=" * 70)
    LOG.info(f"Experiment 16: hard boundary-condition constraint")
    LOG.info("=" * 70)
    hard_bc_data = exp_hard_bc.run_hard_bc_experiment(
        seeds=seeds, epochs=max(epochs // 2, 300),
        n_int=n_int, n_bc=max(n_bc, 80), lr=args.lr, lambda_bc=args.lambda_bc,
        layers=args.layers, eval_every=max(epochs // 4, 100), log=LOG,
    )
    rows_hard_bc = []
    for arch, runs in hard_bc_data.items():
        for r in runs:
            rows_hard_bc.append([arch, r["seed"], r["L2"], r["sym"],
                                 r["bc_err"], r["wall_s"], int(r.get("diverged", False))])
    write_csv(
        tab_dir / "T23_hard_bc.csv",
        ["arch", "seed", "L2_rel_error", "symmetry_error", "bc_error",
         "wall_time_s", "diverged"], rows_hard_bc,
    )
    fig_hard_bc(hard_bc_data, fig_dir / "fig_hard_bc.pdf")

    # ---- E17 GPU benchmark (only if CUDA available) ----
    LOG.info("=" * 70)
    LOG.info(f"Experiment 17: GPU training/inference benchmark")
    LOG.info("=" * 70)
    gpu_data = exp_gpu.run_gpu_benchmark(
        batch_sizes=[500, 2000, 5000, 10000],
        train_steps=100, layers=args.layers, lr=args.lr,
        lambda_bc=args.lambda_bc, n_repeats=3, log=LOG,
    )
    rows_gpu = []
    for arch in ("standard", "d4"):
        for r in gpu_data[arch]:
            rows_gpu.append([arch, r["n_int"], r["train_time_mean"], r["train_time_std"],
                             r["infer_time_mean"], r["infer_time_std"],
                             r["samples_per_sec"], r["peak_mem_mb"], r["device"]])
    write_csv(
        tab_dir / "T24_gpu_benchmark.csv",
        ["arch", "n_int", "train_time_mean_s", "train_time_std_s",
         "infer_time_mean_s", "infer_time_std_s", "samples_per_sec",
         "peak_mem_mb", "device"], rows_gpu,
    )
    fig_gpu_benchmark(gpu_data, fig_dir / "fig_gpu_benchmark.pdf")

    # ---- E18 cubic nonlinearity ----
    LOG.info("=" * 70)
    LOG.info(f"Experiment 18: cubic semilinear Poisson (k u^3)")
    LOG.info("=" * 70)
    cubic_data = exp_cubic.run_cubic_experiment(
        seeds=seeds, epochs=max(epochs, 800),
        n_int=n_int, n_bc=n_bc, lr=args.lr, lambda_bc=args.lambda_bc,
        layers=args.layers, eval_every=max(epochs // 4, 100), log=LOG,
    )
    rows_cubic = []
    for arch in ("standard", "d4", "algebraic"):
        for r in cubic_data[arch]:
            rows_cubic.append([r["arch"], r["seed"], r["L2"], r["sym"],
                               r["wall_s"], int(r["diverged"])])
    write_csv(
        tab_dir / "T25_cubic_semilinear.csv",
        ["arch", "seed", "L2_rel_error", "symmetry_error",
         "wall_time_s", "diverged"], rows_cubic,
    )
    fig_cubic(cubic_data, fig_dir / "fig_cubic_semilinear.pdf")

    # ---- Legacy inverse / homotopy (for backward compatibility) ----
    LOG.info("=" * 70)
    LOG.info(f"Experiment 11-legacy: single-parameter inverse problem")
    LOG.info("=" * 70)
    inv_std = exp_inv_homo.run_inverse_problem(
        seeds=seeds, epochs=max(epochs, 800), n_int=n_int, n_bc=n_bc,
        n_obs=args.n_obs, noise_level=args.inverse_noise,
        lr=args.lr, lambda_bc=args.lambda_bc, lambda_data=10.0,
        k_init=args.k_init, k_true=1.0, layers=args.layers,
        eval_every=max(epochs // 4, 100), use_d4=False, log=LOG,
    )
    inv_d4 = exp_inv_homo.run_inverse_problem(
        seeds=seeds, epochs=max(epochs, 800), n_int=n_int, n_bc=n_bc,
        n_obs=args.n_obs, noise_level=args.inverse_noise,
        lr=args.lr, lambda_bc=args.lambda_bc, lambda_data=10.0,
        k_init=args.k_init, k_true=1.0, layers=args.layers,
        eval_every=max(epochs // 4, 100), use_d4=True, log=LOG,
    )
    rows = []
    for arm_label, runs in (("standard", inv_std), ("d4", inv_d4)):
        for r in runs:
            rows.append([arm_label, r["seed"], r["k_init"], r["k_true"],
                          r["k_final"], r["k_relative_error"], r["L2_forward"],
                          r["wall_s"], int(r["diverged"])])
    write_csv(
        tab_dir / "T11_inverse_problem.csv",
        ["arm", "seed", "k_init", "k_true", "k_final", "k_relative_error",
         "L2_forward", "wall_time_s", "diverged"], rows,
    )

    # ---- Legacy E12 homotopic init ----
    LOG.info("=" * 70)
    LOG.info(f"Experiment 12: homotopic initialisation")
    LOG.info("=" * 70)
    homo_rows = exp_inv_homo.run_homotopic_init(
        seeds=seeds, total_epochs=max(epochs, 600),
        homotopy_epochs=max(epochs // 2, 300),
        n_int=n_int, n_bc=n_bc, lr=args.lr, lambda_bc=args.lambda_bc,
        layers=args.layers, eval_every=max(epochs // 4, 100), log=LOG,
    )
    write_csv(
        tab_dir / "T12_homotopic_init.csv",
        ["seed", "homotopy_L2", "homotopy_sym", "homotopy_wall_s",
         "baseline_L2", "baseline_sym", "baseline_wall_s"],
        [[r["seed"], r["homotopy_L2"], r["homotopy_sym"], r["homotopy_wall"],
          r["baseline_L2"], r["baseline_sym"], r["baseline_wall"]]
         for r in homo_rows],
    )

    # Combined inverse + homotopy figure
    fig13_inverse_homotopy({"standard": inv_std, "d4": inv_d4}, homo_rows,
                              fig_dir / "fig13_inverse_homotopy.pdf")


if __name__ == "__main__":
    main()
