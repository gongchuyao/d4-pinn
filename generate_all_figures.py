"""
generate_all_figures.py --- D4-PINN publication-quality figure pipeline (v2.0)
===============================================================================

Every figure reads from CSV data tables and exports vector PDF + raster PNG.
Design principles:

- Okabe-Ito colorblind-safe palette (Wong, Nature Methods 2011).
- Semiological consistency: same colour = same model across all figures.
- STIX / Times fonts for JCP compliance.
- No chartjunk, high data--ink ratio (Tufte-style).
- Panel labels (a), (b), ... in consistent position.
- All LaTeX math-mode labels --- zero Unicode subscript/superscript glyphs.

Usage:
  python generate_all_figures.py [--data-dir output_demo] [--fig-dir output_demo/figures]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

try:
    import _bootstrap  # noqa: F401
except ImportError:
    pass

from d4pinn.sci_style import (
    apply_sci_style, export_fig, COLORS, model_label, model_color,
    OKABE_ITO, panel_label,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _read_csv(path: Path) -> list[dict]:
    """Read CSV, converting numeric fields and normalizing column names."""
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                try:
                    r[k] = float(v)
                except (ValueError, TypeError):
                    pass
            # Normalize column names
            if "method" in r and "arch" not in r:
                r["arch"] = r["method"]
            if "arm" in r and "arch" not in r:
                r["arch"] = r["arm"]
            if "L2_rel_error" in r and "L2" not in r:
                r["L2"] = r["L2_rel_error"]
            if "wall_time_s" in r and "wall_s" not in r:
                r["wall_s"] = r["wall_time_s"]
            if "L2_rel_error" in r and "l2_error" not in r:
                r["l2_error"] = r["L2_rel_error"]
            if "symmetry_error" in r and "sym" not in r:
                r["sym"] = r["symmetry_error"]
            if "symmetry_error" in r and "sym_error" not in r:
                r["sym_error"] = r["symmetry_error"]
            if "sequential_time_s" in r and "seq_ms" not in r:
                r["seq_ms"] = float(r.get("sequential_time_s", 0)) * 1000.0
            if "batched_time_s" in r and "bat_ms" not in r:
                r["bat_ms"] = float(r.get("batched_time_s", 0)) * 1000.0
            rows.append(r)
    return rows


def _model_plot(ax, x, y, yerr, color, label, marker="o", ls="-", ms=4.5):
    """Consistent errorbar style for model comparison plots."""
    ax.errorbar(x, y, yerr=yerr, color=color, label=label,
                marker=marker, markersize=ms, linewidth=1.2,
                capsize=2.5, capthick=0.6, markeredgewidth=0.4)


def _machine_precision_line(ax, xmax: float, y: float = 1e-7):
    """Subtle machine-precision reference line."""
    ax.axhline(y=y, color="gray", linestyle=":", linewidth=0.6, alpha=0.45)
    ax.text(xmax, y * 1.8, "machine precision", fontsize=6, color="gray",
            alpha=0.6, ha="right", va="bottom")


def _sci_axes(ax, which: str = "both"):
    """Configure log-scale axes with subtle grid."""
    if which in ("x", "both"):
        ax.set_xscale("log")
    if which in ("y", "both"):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.15, linewidth=0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Convergence curves
# ═══════════════════════════════════════════════════════════════════════════════

def fig_convergence(data_dir: Path, fig_dir: Path):
    """Log-scale L2 error and training loss vs epoch, per-seed traces + mean +/- 1 std."""
    apply_sci_style("nature")
    path = data_dir / "tables" / "T1_main_benchmark.csv"
    rows = _read_csv(path)
    if not rows:
        print("  SKIP fig1: no data")
        return

    if "epoch" not in rows[0]:
        print("  SKIP fig1: summary data only (no per-epoch logs)")
        return

    arch_data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        arch_data[r["arch"]][r["seed"]].append(r)

    archs = ["standard", "soft", "d4"]
    colors = {a: model_color(a) for a in archs}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.6))

    for a in archs:
        seed_dict = arch_data[a]
        all_epochs, all_l2, all_loss = [], [], []
        for seed, recs in sorted(seed_dict.items()):
            recs_sorted = sorted(recs, key=lambda r: r.get("epoch", 0))
            eps = [r["epoch"] for r in recs_sorted]
            l2s = [r.get("l2_error", float("nan")) for r in recs_sorted]
            losses = [r.get("loss_total", float("nan")) for r in recs_sorted]
            if eps:
                all_epochs.append(np.array(eps))
                all_l2.append(np.array(l2s))
                all_loss.append(np.array(losses))

        if not all_epochs:
            continue

        min_len = min(len(e) for e in all_epochs)
        epochs = all_epochs[0][:min_len]
        l2_stack = np.array([x[:min_len] for x in all_l2])
        loss_stack = np.array([x[:min_len] for x in all_loss])

        mean_l2 = np.nanmean(l2_stack, axis=0)
        std_l2 = np.nanstd(l2_stack, axis=0, ddof=1) if len(all_l2) > 1 else np.zeros_like(mean_l2)
        mean_loss = np.nanmean(loss_stack, axis=0)
        std_loss = np.nanstd(loss_stack, axis=0, ddof=1) if len(all_loss) > 1 else np.zeros_like(mean_loss)

        # Faint per-seed traces
        for l2, loss in zip(l2_stack, loss_stack):
            ax1.plot(epochs, l2, color=colors[a], alpha=0.12, linewidth=0.3)
            ax2.plot(epochs, loss, color=colors[a], alpha=0.12, linewidth=0.3)

        # Mean line + 1-sigma band
        ax1.plot(epochs, mean_l2, color=colors[a], label=model_label(a), linewidth=1.4)
        ax1.fill_between(epochs, np.maximum(mean_l2 - std_l2, 1e-30), mean_l2 + std_l2,
                         color=colors[a], alpha=0.12, linewidth=0)
        ax2.plot(epochs, mean_loss, color=colors[a], linewidth=1.4)
        ax2.fill_between(epochs, np.maximum(mean_loss - std_loss, 1e-30), mean_loss + std_loss,
                         color=colors[a], alpha=0.12, linewidth=0)

    ax1.set_yscale("log"); ax2.set_yscale("log")
    ax1.set_ylabel(r"Relative $L^2$ error")
    ax2.set_ylabel("Total training loss")
    ax1.set_xlabel("Epoch"); ax2.set_xlabel("Epoch")
    ax2.legend(loc="upper right", fontsize=7, frameon=True,
               facecolor="white", edgecolor="#ddd")
    panel_label(ax1, "(a)"); panel_label(ax2, "(b)")
    fig.tight_layout(pad=0.8)
    export_fig(fig, "fig1_convergence", outdir=fig_dir)
    plt.close(fig)
    print("  fig1_convergence")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Per-seed L2 error strip plot
# ═══════════════════════════════════════════════════════════════════════════════

def fig_l2_per_seed(data_dir: Path, fig_dir: Path):
    apply_sci_style("nature")
    path = data_dir / "tables" / "T1_main_benchmark.csv"
    rows = _read_csv(path)
    if not rows:
        print("  SKIP fig2: no data")
        return

    arch_data = defaultdict(list)
    for r in rows:
        arch_data[r["arch"]].append(r)

    methods = ["standard", "soft", "d4"]
    final_l2 = {m: [] for m in methods}
    for m in methods:
        by_seed = defaultdict(list)
        for r in arch_data[m]:
            by_seed[r["seed"]].append(r.get("l2_error", float("nan")))
        for seed, vals in by_seed.items():
            final_l2[m].append(vals[-1] if vals else float("nan"))

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    rng = np.random.default_rng(42)

    for i, m in enumerate(methods):
        vals = [v for v in final_l2[m] if np.isfinite(v)]
        if not vals:
            continue
        x = np.full(len(vals), i) + rng.uniform(-0.1, 0.1, len(vals))
        ax.scatter(x, vals, color=model_color(m), s=32, zorder=3,
                   edgecolors="white", linewidths=0.6, alpha=0.85)
        mu = np.mean(vals)
        ax.plot([i - 0.28, i + 0.28], [mu, mu], color=model_color(m),
                linewidth=2.0, zorder=2)
        ax.text(i + 0.30, mu, f"{mu:.2e}", fontsize=6.5, va="center",
                color=model_color(m), fontweight="bold")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([model_label(m) for m in methods], fontsize=9)
    ax.set_ylabel(r"Relative $L^2$ error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.12, axis="y")
    fig.tight_layout()
    export_fig(fig, "fig2_l2_vs_seeds", outdir=fig_dir)
    plt.close(fig)
    print("  fig2_l2_vs_seeds")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Architectural D4 invariance bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def fig_symmetry_verification(data_dir: Path, fig_dir: Path):
    apply_sci_style("nature")
    path = data_dir / "tables" / "T2_symmetry_at_init.csv"
    rows = _read_csv(path)
    if not rows:
        print("  SKIP fig3: no data")
        return

    transforms = ["rot90", "rot180", "rot270", "flipx", "flipy", "diag", "antidiag"]
    labels = [r"$R_{90}$", r"$R_{180}$", r"$R_{270}$",
              r"$S_x$", r"$S_y$", r"$S_{d_1}$", r"$S_{d_2}$"]

    std_vals, d4_vals = [], []
    for t in transforms:
        std_v = [r["max_abs_deviation"] for r in rows
                 if r["arch"] == "standard" and r.get("transform") == t]
        d4_v = [r["max_abs_deviation"] for r in rows
                if r["arch"] == "d4" and r.get("transform") == t]
        std_vals.append(np.mean(std_v) if std_v else 0)
        d4_vals.append(max(np.mean(d4_v) if d4_v else 0, 1e-14))

    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    x = np.arange(len(transforms))
    w = 0.35

    ax.bar(x - w/2, [max(v, 1e-12) for v in std_vals], w,
           color=model_color("standard"), label=model_label("standard"),
           edgecolor="white", linewidth=0.4, alpha=0.9)
    ax.bar(x + w/2, [max(v, 1e-14) for v in d4_vals], w,
           color=model_color("d4"), label=model_label("d4"),
           edgecolor="white", linewidth=0.4, alpha=0.9)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"$\max\,|u_\theta(x) - u_\theta(g{\cdot}x)|$" +
                  "\n(random init.)")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    _machine_precision_line(ax, len(transforms) - 0.5)
    ax.grid(True, alpha=0.12, axis="y")
    fig.tight_layout()
    export_fig(fig, "fig3_symmetry_error_bar", outdir=fig_dir)
    plt.close(fig)
    print("  fig3_symmetry_error_bar")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Solution field
# ═══════════════════════════════════════════════════════════════════════════════

def fig_solution_field(data_dir: Path, fig_dir: Path):
    """Needs a trained model; falls back to existing PDF if torch unavailable."""
    try:
        import torch
        from d4pinn import D4PINN, ReactionDiffusionBVP, train_one_run
    except ImportError:
        print("  SKIP fig4: torch not available")
        return

    apply_sci_style("nature")
    pde = ReactionDiffusionBVP(k=1.0, epsilon=0.0)
    torch.manual_seed(0)
    model = D4PINN(layers=[2, 50, 50, 50, 1])
    _, _h = train_one_run(
        pde=pde, model=model, n_int=2000, n_bc=400, epochs=2000,
        learning_rate=1e-3, lambda_bc=10.0, lambda_sym=None,
        seed=0, eval_every=1000, verbose=False,
    )

    s = torch.linspace(-1, 1, 220)
    xx, yy = torch.meshgrid(s, s, indexing="xy")
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    with torch.no_grad():
        u_pred = model(grid).reshape(220, 220).numpy()
        u_ref = pde.reference_solution(grid).reshape(220, 220).numpy()
    err = np.abs(u_pred - u_ref)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    extent = (-1, 1, -1, 1)
    kw = dict(origin="lower", extent=extent, aspect="equal")

    im0 = axes[0].imshow(u_ref, cmap="viridis", **kw)
    axes[0].set_xlabel("$x$"); axes[0].set_ylabel("$y$")
    c0 = fig.colorbar(im0, ax=axes[0], shrink=0.80, pad=0.03)
    c0.ax.tick_params(labelsize=6.5)

    im1 = axes[1].imshow(u_pred, cmap="viridis",
                         vmin=u_ref.min(), vmax=u_ref.max(), **kw)
    axes[1].set_xlabel("$x$")
    c1 = fig.colorbar(im1, ax=axes[1], shrink=0.80, pad=0.03)
    c1.ax.tick_params(labelsize=6.5)

    im2 = axes[2].imshow(err, cmap="magma",
                         norm=plt.matplotlib.colors.LogNorm(
                             vmin=err[err > 0].min(), vmax=err.max()), **kw)
    axes[2].set_xlabel("$x$")
    c2 = fig.colorbar(im2, ax=axes[2], shrink=0.80, pad=0.03)
    c2.ax.tick_params(labelsize=6.5)

    panel_label(axes[0], r"(a) Reference $u^*$")
    panel_label(axes[1], r"(b) $D_4$-PINN")
    panel_label(axes[2], r"(c) $|u_\theta - u^*|$")

    fig.tight_layout(pad=0.5)
    export_fig(fig, "fig4_solution_field", outdir=fig_dir)
    plt.close(fig)
    print("  fig4_solution_field")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Symmetry-breaking sweep
# ═══════════════════════════════════════════════════════════════════════════════

def fig_symmetry_breaking(data_dir: Path, fig_dir: Path):
    apply_sci_style("nature")
    path = data_dir / "tables" / "T3_symmetry_breaking_sweep.csv"
    rows = _read_csv(path)
    if not rows:
        print("  SKIP fig5: no data")
        return

    epsilons = sorted(set(r["epsilon"] for r in rows))
    archs = ["standard", "d4"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.6))

    for a in archs:
        l2_means, l2_stds, sym_means = [], [], []
        for e in epsilons:
            l2s = [r["L2_mean"] if "L2_mean" in r else r.get("l2_error", 0)
                   for r in rows if r["arch"] == a and r["epsilon"] == e]
            syms = [r["sym_mean"] if "sym_mean" in r else r.get("sym_error", 0)
                    for r in rows if r["arch"] == a and r["epsilon"] == e]
            l2_means.append(np.mean(l2s) if l2s else 0)
            l2_stds.append(np.std(l2s, ddof=1) if len(l2s) > 1 else 0)
            sym_means.append(max(np.mean(syms) if syms else 0, 1e-14))
        _model_plot(ax1, epsilons, l2_means, l2_stds,
                    model_color(a), model_label(a))
        ax2.plot(epsilons, sym_means, color=model_color(a), marker="s",
                 markersize=4, linewidth=1.2, label=model_label(a))

    ax1.set_xlabel(r"Asymmetric perturbation $\varepsilon$")
    ax1.set_ylabel(r"Relative $L^2$ error")
    ax1.set_yscale("log"); ax1.legend(fontsize=7)
    _sci_axes(ax1, "y")

    ax2.set_xlabel(r"Asymmetric perturbation $\varepsilon$")
    ax2.set_ylabel(r"$D_4$ symmetry error")
    ax2.set_yscale("log"); ax2.legend(fontsize=7)
    _machine_precision_line(ax2, max(epsilons))
    _sci_axes(ax2, "y")

    panel_label(ax1, "(a)"); panel_label(ax2, "(b)")
    fig.tight_layout()
    export_fig(fig, "fig5_symmetry_breaking", outdir=fig_dir)
    plt.close(fig)
    print("  fig5_symmetry_breaking")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: Batched vs sequential speedup
# ═══════════════════════════════════════════════════════════════════════════════

def fig_batched_speedup(data_dir: Path, fig_dir: Path):
    apply_sci_style("nature")
    path = data_dir / "tables" / "T4_batched_speedup.csv"
    rows = _read_csv(path)
    if not rows:
        print("  SKIP fig6: no data")
        return

    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    ns = sorted(set(r["batch_size"] for r in rows))
    seq_mean = [np.mean([r["seq_ms"] for r in rows if r["batch_size"] == n])
                for n in ns]
    bat_mean = [np.mean([r["bat_ms"] for r in rows if r["batch_size"] == n])
                for n in ns]
    seq_std = [np.std([r["seq_ms"] for r in rows if r["batch_size"] == n],
                      ddof=1) for n in ns]
    bat_std = [np.std([r["bat_ms"] for r in rows if r["batch_size"] == n],
                      ddof=1) for n in ns]

    ax.errorbar(ns, seq_mean, yerr=seq_std, color=COLORS["reference"],
                marker="o", capsize=3, linewidth=1.2, label="Sequential loop")
    ax.errorbar(ns, bat_mean, yerr=bat_std, color=model_color("d4"),
                marker="s", capsize=3, linewidth=1.2,
                label=r"Batched ($\mathtt{einsum}$)")

    for i, n in enumerate(ns):
        if bat_mean[i] > 0:
            s = seq_mean[i] / bat_mean[i]
            ax.annotate(
                f"{s:.1f}$\\times$", (n, bat_mean[i]),
                xytext=(0, 9), textcoords="offset points", ha="center",
                fontsize=6.5, color=COLORS["text"],
            )

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Batch size $N$")
    ax.set_ylabel("Forward + backward [ms]")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(True, alpha=0.12)
    fig.tight_layout()
    export_fig(fig, "fig6_batched_speedup", outdir=fig_dir)
    plt.close(fig)
    print("  fig6_batched_speedup")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7: Filter-normalised loss landscape
# ═══════════════════════════════════════════════════════════════════════════════

def fig_loss_landscape(data_dir: Path, fig_dir: Path):
    apply_sci_style("nature")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))

    for ax, arch in zip(axes, ["standard", "d4"]):
        path = data_dir / "tables" / f"T5_loss_landscape_{arch}.csv"
        rows = _read_csv(path)
        if not rows:
            continue
        alphas = sorted(set(r["alpha"] for r in rows))
        betas = sorted(set(r["beta"] for r in rows))
        na, nb = len(alphas), len(betas)
        L = np.zeros((na, nb))
        for r in rows:
            i = alphas.index(r["alpha"]); j = betas.index(r["beta"])
            L[i, j] = r["loss"]
        Lp = np.log10(np.maximum(L, 1e-12))
        AA, BB = np.meshgrid(alphas, betas, indexing="ij")
        cs = ax.contourf(AA, BB, Lp, levels=20, cmap="viridis")
        ax.contour(AA, BB, Lp, levels=8, colors="white",
                   linewidths=0.25, alpha=0.45)
        ax.scatter([0], [0], marker="*", color=COLORS["reference"], s=60,
                   edgecolors="white", linewidths=0.5, zorder=3)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\alpha$ (direction 1)")
        ax.set_ylabel(r"$\beta$ (direction 2)")
        ax.set_title(model_label(arch), fontsize=9, loc="left", pad=4)
        ax.grid(False)
        cb = fig.colorbar(cs, ax=ax, shrink=0.82, pad=0.03)
        cb.set_label(r"$\log_{10}\,\mathcal{L}$", fontsize=7)
        cb.ax.tick_params(labelsize=6)

    panel_label(axes[0], "(a)"); panel_label(axes[1], "(b)")
    fig.tight_layout()
    export_fig(fig, "fig7_loss_landscape", outdir=fig_dir)
    plt.close(fig)
    print("  fig7_loss_landscape")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8: Ablation --- rotations / reflections / full D4
# ═══════════════════════════════════════════════════════════════════════════════

def fig_ablation(data_dir: Path, fig_dir: Path):
    apply_sci_style("nature")
    path = data_dir / "tables" / "T6_ablation.csv"
    rows = _read_csv(path)
    if not rows:
        print("  SKIP fig8: no data")
        return

    groups = sorted(set(r.get("arm", r.get("arch", "")) for r in rows))
    labels_map = {
        "C4": r"$C_4$ (rotations)",
        "V4": r"$V_4$ (reflections)",
        "D4": r"$D_4$ (full)",
        "no_constraint": "No constraint",
    }
    labels = [labels_map.get(g, g) for g in groups]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.6))

    # (a) L2 violin + strip
    data_l2 = {
        g: [r["L2"] for r in rows
            if r.get("arm", r.get("arch", "")) == g
            and np.isfinite(r.get("L2", float("nan")))]
        for g in groups
    }
    positions = range(len(groups))
    vp = ax1.violinplot(
        [data_l2[g] for g in groups], positions=positions,
        showmeans=False, showmedians=True, widths=0.55,
    )
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(OKABE_ITO[i % len(OKABE_ITO)])
        body.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        vals = data_l2[g]
        x = np.full(len(vals), i) + rng.uniform(-0.12, 0.12, len(vals))
        ax1.scatter(x, vals, color=OKABE_ITO[i % len(OKABE_ITO)], s=16,
                    edgecolors="white", linewidths=0.4, zorder=3)
    ax1.set_xticks(list(positions))
    ax1.set_xticklabels(labels, fontsize=7.5, rotation=12, ha="right")
    ax1.set_ylabel(r"Relative $L^2$ error")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.12, axis="y")

    # (b) Symmetry error
    data_sym = {
        g: [max(r.get("sym", 0), 1e-14) for r in rows
            if r.get("arm", r.get("arch", "")) == g]
        for g in groups
    }
    for i, g in enumerate(groups):
        vals = data_sym[g]
        ax2.bar(i, np.mean(vals),
                yerr=np.std(vals, ddof=1) if len(vals) > 1 else 0,
                color=OKABE_ITO[i % len(OKABE_ITO)], capsize=3,
                edgecolor="white", linewidth=0.4)
    ax2.set_xticks(list(positions))
    ax2.set_xticklabels(labels, fontsize=7.5, rotation=12, ha="right")
    ax2.set_ylabel(r"$D_4$ symmetry error")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.12, axis="y")
    _machine_precision_line(ax2, len(groups) - 0.5)

    panel_label(ax1, "(a)"); panel_label(ax2, "(b)")
    fig.tight_layout()
    export_fig(fig, "fig8_ablation", outdir=fig_dir)
    plt.close(fig)
    print("  fig8_ablation")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 9: Sample efficiency
# ═══════════════════════════════════════════════════════════════════════════════

def fig_sample_efficiency(data_dir: Path, fig_dir: Path):
    apply_sci_style("nature")
    path = data_dir / "tables" / "T7_sample_efficiency.csv"
    rows = _read_csv(path)
    if not rows:
        print("  SKIP fig9: no data")
        return

    n_vals = sorted(set(r["n_int"] for r in rows))
    archs = ["standard", "d4"]

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for a in archs:
        means = [np.mean([r["L2"] for r in rows
                          if r["arch"] == a and r["n_int"] == n])
                 for n in n_vals]
        stds = [np.std([r["L2"] for r in rows
                        if r["arch"] == a and r["n_int"] == n], ddof=1)
                for n in n_vals]
        _model_plot(ax, n_vals, means, stds, model_color(a), model_label(a))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$N_{\rm int}$ (collocation points)")
    ax.set_ylabel(r"Relative $L^2$ error")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.12)
    fig.tight_layout()
    export_fig(fig, "fig9_sample_efficiency", outdir=fig_dir)
    plt.close(fig)
    print("  fig9_sample_efficiency")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure: ESCNN baseline comparison
# ═══════════════════════════════════════════════════════════════════════════════

def fig_escnn_baseline(data_dir: Path, fig_dir: Path):
    apply_sci_style("nature")
    path = data_dir / "tables" / "T13_escnn_baseline.csv"
    rows = _read_csv(path)
    if not rows:
        print("  SKIP fig14 (escnn): no data")
        return

    archs = sorted(set(r["arch"] for r in rows))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.6))

    for i, a in enumerate(archs):
        arows = [r for r in rows if r["arch"] == a]
        l2s = [r["L2"] if "L2" in r else r.get("l2_error", float("nan"))
               for r in arows]
        syms = [r["sym"] if "sym" in r else r.get("sym_error", float("nan"))
                for r in arows]
        l2s = [v for v in l2s if np.isfinite(v)]
        syms = [v for v in syms if np.isfinite(v)]
        c = OKABE_ITO[i % len(OKABE_ITO)]
        if l2s:
            ax1.bar(i, np.mean(l2s),
                    yerr=np.std(l2s, ddof=1) if len(l2s) > 1 else 0,
                    color=c, capsize=3, edgecolor="white", linewidth=0.4)
        if syms:
            ax2.bar(i, max(np.mean(syms), 1e-14), color=c,
                    capsize=3, edgecolor="white", linewidth=0.4)

    ax1.set_xticks(range(len(archs)))
    ax1.set_xticklabels([model_label(a) for a in archs],
                        fontsize=7, rotation=15, ha="right")
    ax1.set_ylabel(r"Relative $L^2$ error")
    ax1.set_yscale("log"); ax1.grid(True, alpha=0.12, axis="y")

    ax2.set_xticks(range(len(archs)))
    ax2.set_xticklabels([model_label(a) for a in archs],
                        fontsize=7, rotation=15, ha="right")
    ax2.set_ylabel(r"$D_4$ symmetry error")
    ax2.set_yscale("log"); ax2.grid(True, alpha=0.12, axis="y")
    _machine_precision_line(ax2, len(archs) - 0.5)

    panel_label(ax1, "(a)"); panel_label(ax2, "(b)")
    fig.tight_layout()
    export_fig(fig, "fig14_escnn_baseline", outdir=fig_dir)
    plt.close(fig)
    print("  fig14_escnn_baseline")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure: C++ vs Python inference latency
# ═══════════════════════════════════════════════════════════════════════════════

def fig_cpp_latency(data_dir: Path, fig_dir: Path):
    apply_sci_style("nature")
    path = data_dir / "tables" / "T14_cpp_latency.csv"
    rows = _read_csv(path)
    if not rows:
        print("  SKIP fig15: no data")
        return

    bs = sorted(set(r["batch_size"] for r in rows))
    py_means = [np.mean([r["python_mean_ms"] for r in rows
                         if r["batch_size"] == b]) * 1000 for b in bs]
    cpp_means = [np.mean([r["cpp_mean_ms"] for r in rows
                          if r["batch_size"] == b]) * 1000 for b in bs]

    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    ax.plot(bs, py_means, "o-", color=OKABE_ITO[0],
            label="PyTorch (MKL BLAS)", markersize=4.5, linewidth=1.2)
    ax.plot(bs, cpp_means, "s-", color=OKABE_ITO[1],
            label="C++17 (naive loops)", markersize=4.5, linewidth=1.2)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Batch size")
    ax.set_ylabel(r"Latency [$\mu$s]")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.12)
    fig.tight_layout()
    export_fig(fig, "fig15_cpp_latency", outdir=fig_dir)
    plt.close(fig)
    print("  fig15_cpp_latency")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 16: Wall-clock-equalised Pareto frontier
# ═══════════════════════════════════════════════════════════════════════════════

def fig_wallclock_equalised(data_dir: Path, fig_dir: Path):
    apply_sci_style("nature")
    path = data_dir / "tables" / "T1_main_benchmark.csv"
    rows = _read_csv(path)
    if not rows:
        print("  SKIP fig16: no data")
        return

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    archs = ["standard", "soft", "d4"]
    markers = ["o", "D", "s"]
    for a, mk in zip(archs, markers):
        pts = [(r.get("wall_s", 0), r.get("l2_error", float("nan")))
               for r in rows if r["arch"] == a]
        pts = [(w, l2) for w, l2 in pts if w > 0 and np.isfinite(l2)]
        if not pts:
            continue
        pts.sort()
        ws, l2s = zip(*pts)
        ax.scatter(ws, l2s, color=model_color(a), marker=mk, s=20,
                   edgecolors="white", linewidths=0.4,
                   label=model_label(a), alpha=0.7)
        # Pareto frontier
        frontier_w, frontier_l2 = [ws[0]], [l2s[0]]
        best = l2s[0]
        for w, l in zip(ws[1:], l2s[1:]):
            if l < best:
                best = l
                frontier_w.append(w); frontier_l2.append(l)
        if len(frontier_w) > 1:
            ax.plot(frontier_w, frontier_l2, "--", color=model_color(a),
                    alpha=0.30, linewidth=0.7)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Wall-clock time [s]")
    ax.set_ylabel(r"Relative $L^2$ error")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.12)
    fig.tight_layout()
    export_fig(fig, "fig16_wallclock_equalised", outdir=fig_dir)
    plt.close(fig)
    print("  fig16_wallclock_equalised")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure: Allen-Cahn
# ═══════════════════════════════════════════════════════════════════════════════

def fig_allen_cahn(data_dir: Path, fig_dir: Path):
    apply_sci_style("nature")
    path = data_dir / "tables" / "T21_allen_cahn.csv"
    rows = _read_csv(path)
    if not rows:
        print("  SKIP allen_cahn: no data")
        return

    epsilons = sorted(set(r["epsilon"] for r in rows))
    archs = ["standard", "d4"]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))

    # (a) L2 vs epsilon
    for a in archs:
        means = [np.mean([r["L2"] for r in rows
                          if r["arch"] == a and r["epsilon"] == e])
                 for e in epsilons]
        stds = [np.std([r["L2"] for r in rows
                        if r["arch"] == a and r["epsilon"] == e], ddof=1)
                for e in epsilons]
        _model_plot(axes[0, 0], epsilons, means, stds,
                    model_color(a), model_label(a))
    axes[0, 0].set_xlabel(r"$\epsilon$")
    axes[0, 0].set_ylabel(r"Relative $L^2$ error")
    axes[0, 0].set_yscale("log"); axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.12)
    panel_label(axes[0, 0], r"(a) $L^2$ vs.\ $\epsilon$")

    # (b) Symmetry vs epsilon
    for a in archs:
        means = [max(np.mean([r.get("sym", r.get("sym_error", 0))
                              for r in rows
                              if r["arch"] == a and r["epsilon"] == e]),
                     1e-14) for e in epsilons]
        axes[0, 1].plot(epsilons, means, color=model_color(a), marker="s",
                        markersize=4, linewidth=1.2, label=model_label(a))
    axes[0, 1].set_xlabel(r"$\epsilon$")
    axes[0, 1].set_ylabel(r"$D_4$ symmetry error")
    axes[0, 1].set_yscale("log"); axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.12)
    _machine_precision_line(axes[0, 1], max(epsilons))
    panel_label(axes[0, 1], r"(b) Symmetry vs.\ $\epsilon$")

    # (c) Smallest epsilon
    e_min = min(epsilons)
    for i, a in enumerate(archs):
        l2s = [r["L2"] for r in rows
               if r["arch"] == a and r["epsilon"] == e_min]
        axes[1, 0].bar(i, np.mean(l2s),
                       yerr=np.std(l2s, ddof=1) if len(l2s) > 1 else 0,
                       color=model_color(a), capsize=3,
                       edgecolor="white", linewidth=0.4)
    axes[1, 0].set_xticks(range(len(archs)))
    axes[1, 0].set_xticklabels([model_label(a) for a in archs], fontsize=8)
    axes[1, 0].set_ylabel(r"Relative $L^2$ error")
    axes[1, 0].grid(True, alpha=0.12, axis="y")
    panel_label(axes[1, 0], rf"(c) $\epsilon = {e_min}$")

    # (d) Wall time
    for a in archs:
        walls = [np.mean([r.get("wall_s", 0) for r in rows
                          if r["arch"] == a and r["epsilon"] == e])
                 for e in epsilons]
        axes[1, 1].plot(epsilons, walls, color=model_color(a), marker="o",
                        markersize=4, linewidth=1.2, label=model_label(a))
    axes[1, 1].set_xlabel(r"$\epsilon$")
    axes[1, 1].set_ylabel("Wall time [s]")
    axes[1, 1].legend(fontsize=7); axes[1, 1].grid(True, alpha=0.12)
    panel_label(axes[1, 1], "(d) Cost")

    fig.tight_layout(pad=1.0)
    export_fig(fig, "fig_allen_cahn", outdir=fig_dir)
    plt.close(fig)
    print("  fig_allen_cahn")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure: Soft-constraint lambda_sym sweep
# ═══════════════════════════════════════════════════════════════════════════════

def fig_soft_constraint_sweep(data_dir: Path, fig_dir: Path):
    apply_sci_style("nature")
    path = data_dir / "tables" / "T22_soft_constraint_sweep.csv"
    rows = _read_csv(path)
    if not rows:
        print("  SKIP soft_constraint_sweep: no data")
        return

    lam_vals = sorted(set(r["lambda_sym"] for r in rows))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.6))

    soft_l2 = [np.mean([r["L2"] for r in rows
                        if r["lambda_sym"] == lam and r["arch"] == "soft"
                        and np.isfinite(r.get("L2", float("nan")))])
               for lam in lam_vals]
    soft_sym = [max(np.mean([r.get("sym", r.get("sym_error", 0))
                             for r in rows
                             if r["lambda_sym"] == lam
                             and r["arch"] == "soft"]), 1e-14)
                for lam in lam_vals]

    d4_l2 = np.mean([r["L2"] for r in rows
                     if r["arch"] == "d4"
                     and np.isfinite(r.get("L2", float("nan")))]) \
        if any(r["arch"] == "d4" for r in rows) else None
    std_l2 = np.mean([r["L2"] for r in rows
                      if r["arch"] == "standard"
                      and np.isfinite(r.get("L2", float("nan")))]) \
        if any(r["arch"] == "standard" for r in rows) else None

    ax1.semilogx(lam_vals, soft_l2, "o-", color=COLORS["soft"],
                 label="Soft constraint", linewidth=1.2, markersize=4.5)
    if d4_l2:
        ax1.axhline(d4_l2, color=model_color("d4"), linestyle="--",
                    label=model_label("d4"), linewidth=1.0)
    if std_l2:
        ax1.axhline(std_l2, color=model_color("standard"), linestyle=":",
                    label=model_label("standard"), linewidth=1.0)
    best_i = np.argmin(soft_l2)
    ax1.scatter([lam_vals[best_i]], [soft_l2[best_i]], color=COLORS["soft"],
                marker="*", s=90, zorder=5, edgecolors="white")
    ax1.set_xlabel(r"$\lambda_{\rm sym}$")
    ax1.set_ylabel(r"Relative $L^2$ error")
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.12)

    ax2.loglog(lam_vals, soft_sym, "s-", color=COLORS["soft"],
               label="Soft constraint", linewidth=1.2, markersize=4.5)
    ax2.axhline(1e-7, color="gray", linestyle=":", alpha=0.45)
    ax2.set_xlabel(r"$\lambda_{\rm sym}$")
    ax2.set_ylabel(r"$D_4$ symmetry error")
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.12)

    panel_label(ax1, "(a)"); panel_label(ax2, "(b)")
    fig.tight_layout()
    export_fig(fig, "fig_soft_constraint_sweep", outdir=fig_dir)
    plt.close(fig)
    print("  fig_soft_constraint_sweep")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure: Hard BC comparison
# ═══════════════════════════════════════════════════════════════════════════════

def fig_hard_bc(data_dir: Path, fig_dir: Path):
    apply_sci_style("nature")
    path = data_dir / "tables" / "T23_hard_bc.csv"
    rows = _read_csv(path)
    if not rows:
        print("  SKIP hard_bc: no data")
        return

    configs = sorted(set(r["arch"] for r in rows))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.6))

    for i, cfg in enumerate(configs):
        cfg_rows = [r for r in rows if r["arch"] == cfg]
        l2s = [r["L2_rel_error"] for r in cfg_rows
               if np.isfinite(r.get("L2_rel_error", float("nan")))]
        bcs = [r.get("bc_err", r.get("bc_error", float("nan")))
               for r in cfg_rows]
        bcs = [v for v in bcs if np.isfinite(v)]
        c = OKABE_ITO[i % len(OKABE_ITO)]
        if l2s:
            ax1.bar(i, np.mean(l2s),
                    yerr=np.std(l2s, ddof=1) if len(l2s) > 1 else 0,
                    color=c, capsize=3, edgecolor="white", linewidth=0.4)
        if bcs:
            ax2.bar(i, max(np.mean(bcs), 1e-14), color=c,
                    capsize=3, edgecolor="white", linewidth=0.4)

    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, fontsize=7, rotation=15, ha="right")
    ax1.set_ylabel(r"Relative $L^2$ error")
    ax1.set_yscale("log"); ax1.grid(True, alpha=0.12, axis="y")

    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, fontsize=7, rotation=15, ha="right")
    ax2.set_ylabel("Boundary MSE")
    ax2.set_yscale("log"); ax2.grid(True, alpha=0.12, axis="y")
    _machine_precision_line(ax2, len(configs) - 0.5)

    panel_label(ax1, "(a)"); panel_label(ax2, "(b)")
    fig.tight_layout()
    export_fig(fig, "fig_hard_bc", outdir=fig_dir)
    plt.close(fig)
    print("  fig_hard_bc")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure: Cubic semilinear Poisson
# ═══════════════════════════════════════════════════════════════════════════════

def fig_cubic_semilinear(data_dir: Path, fig_dir: Path):
    apply_sci_style("nature")
    path = data_dir / "tables" / "T25_cubic_semilinear.csv"
    rows = _read_csv(path)
    if not rows:
        print("  SKIP cubic: no data")
        return

    archs = sorted(set(r["arch"] for r in rows))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.6))

    for i, a in enumerate(archs):
        arows = [r for r in rows if r["arch"] == a]
        l2s = [r["L2"] for r in arows
               if np.isfinite(r.get("L2", float("nan")))]
        syms = [r.get("sym", r.get("sym_error", 0)) for r in arows]
        c = OKABE_ITO[i % len(OKABE_ITO)]
        if l2s:
            ax1.bar(i, np.mean(l2s),
                    yerr=np.std(l2s, ddof=1) if len(l2s) > 1 else 0,
                    color=c, capsize=3, edgecolor="white", linewidth=0.4)
        if syms:
            ax2.bar(i, max(np.mean(syms), 1e-14), color=c,
                    capsize=3, edgecolor="white", linewidth=0.4)

    ax1.set_xticks(range(len(archs)))
    ax1.set_xticklabels([model_label(a) for a in archs],
                        fontsize=8, rotation=15, ha="right")
    ax1.set_ylabel(r"Relative $L^2$ error")
    ax1.set_yscale("log"); ax1.grid(True, alpha=0.12, axis="y")

    ax2.set_xticks(range(len(archs)))
    ax2.set_xticklabels([model_label(a) for a in archs],
                        fontsize=8, rotation=15, ha="right")
    ax2.set_ylabel(r"$D_4$ symmetry error")
    ax2.set_yscale("log"); ax2.grid(True, alpha=0.12, axis="y")
    _machine_precision_line(ax2, len(archs) - 0.5)

    panel_label(ax1, "(a)"); panel_label(ax2, "(b)")
    fig.tight_layout()
    export_fig(fig, "fig_cubic_semilinear", outdir=fig_dir)
    plt.close(fig)
    print("  fig_cubic_semilinear")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="D4-PINN unified figure pipeline")
    p.add_argument("--data-dir", default="output_demo",
                   help="Directory with tables/ subdir")
    p.add_argument("--fig-dir", default=None,
                   help="Output directory for figures")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    fig_dir = Path(args.fig_dir) if args.fig_dir else data_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data: {data_dir.resolve()}")
    print(f"Figures: {fig_dir.resolve()}")
    print()

    figures = [
        ("fig1_convergence",              fig_convergence),
        ("fig2_l2_vs_seeds",              fig_l2_per_seed),
        ("fig3_symmetry_error_bar",       fig_symmetry_verification),
        ("fig4_solution_field",           fig_solution_field),
        ("fig5_symmetry_breaking",        fig_symmetry_breaking),
        ("fig6_batched_speedup",          fig_batched_speedup),
        ("fig7_loss_landscape",           fig_loss_landscape),
        ("fig8_ablation",                 fig_ablation),
        ("fig9_sample_efficiency",        fig_sample_efficiency),
        ("fig14_escnn_baseline",          fig_escnn_baseline),
        ("fig15_cpp_latency",             fig_cpp_latency),
        ("fig16_wallclock_equalised",     fig_wallclock_equalised),
        ("fig_allen_cahn",                fig_allen_cahn),
        ("fig_soft_constraint_sweep",     fig_soft_constraint_sweep),
        ("fig_hard_bc",                   fig_hard_bc),
        ("fig_cubic_semilinear",          fig_cubic_semilinear),
    ]

    for name, func in figures:
        try:
            func(data_dir, fig_dir)
        except Exception as exc:
            print(f"  FAIL {name}: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
