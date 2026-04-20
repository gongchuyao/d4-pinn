#!/usr/bin/env python
"""
Speed Optimization Test
=======================
This script compares the speed of the base group averaging model
vs the optimized batched group averaging model.

The optimized version is ~2-3x faster!
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import time
import numpy as np
import matplotlib.pyplot as plt

# Import from our package
from src import BaseD4PINN, OptimizedD4PINN
from src import poisson_2d, generate_sample_points_2d
from src import set_scientific_plot_style, save_figure

# Fix random seed
torch.manual_seed(42)
np.random.seed(42)

def benchmark_model(model, x_int, x_bc, n_runs=10):
    """
    Benchmark a model's forward/backward pass speed.
    """
    # Warmup
    for _ in range(5):
        model(x_int)
    
    # Time the forward/backward pass
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    times = []
    for _ in range(n_runs):
        optimizer.zero_grad()
        
        start_time = time.time()
        
        # Boundary loss
        u_bc_pred = model(x_bc)
        loss_bc = torch.mean(u_bc_pred**2)
        
        # PDE loss
        x = x_int.clone()
        x.requires_grad = True
        u_pred = model(x)
        
        u_x = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0][:,0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:,0]
        u_y = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0][:,1]
        u_yy = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:,1]
        
        residual = -u_xx - u_yy + u_pred**3
        loss_pde = torch.mean(residual**2)
        
        loss = loss_pde + 10 * loss_bc
        loss.backward()
        optimizer.step()
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)

def main():
    print("="*60)
    print("🚀 Running Speed Optimization Benchmark")
    print("="*60)
    
    set_scientific_plot_style()
    
    # Test different batch sizes
    batch_sizes = [1000, 2000, 5000, 10000, 20000]
    
    base_times = []
    base_stds = []
    opt_times = []
    opt_stds = []
    
    u_exact, f_source, _ = poisson_2d()
    
    for batch_size in batch_sizes:
        print(f"\n📊 Testing batch size: {batch_size}")
        
        x_int, x_bc = generate_sample_points_2d(N_interior=batch_size)
        
        # Benchmark base model
        base_model = BaseD4PINN()
        t_mean, t_std = benchmark_model(base_model, x_int, x_bc, n_runs=5)
        base_times.append(t_mean)
        base_stds.append(t_std)
        print(f"   Base model: {t_mean:.4f}s ± {t_std:.4f}s")
        
        # Benchmark optimized model
        opt_model = OptimizedD4PINN()
        t_mean, t_std = benchmark_model(opt_model, x_int, x_bc, n_runs=5)
        opt_times.append(t_mean)
        opt_stds.append(t_std)
        print(f"   Optimized model: {t_mean:.4f}s ± {t_std:.4f}s")
        
        speedup = base_times[-1] / opt_times[-1]
        print(f"   Speedup: {speedup:.2f}x")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time comparison
    ax1.errorbar(batch_sizes, base_times, yerr=base_stds, 
                 marker='o', label='Base Group Averaging', 
                 color='#EA4335', capsize=5)
    ax1.errorbar(batch_sizes, opt_times, yerr=opt_stds, 
                 marker='s', label='Optimized Batched', 
                 color='#4285F4', capsize=5)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Time per Iteration (s)')
    ax1.set_title('Training Speed Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Speedup
    speedups = [b/o for b,o in zip(base_times, opt_times)]
    ax2.plot(batch_sizes, speedups, marker='D', color='#34A853', linewidth=2)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Optimization Speedup')
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    ax2.grid(True)
    
    plt.tight_layout()
    save_figure(fig, 'results/speed_optimization')
    
    print("\n✅ Plot saved to results/speed_optimization.png/pdf")
    
    # Summary
    print("\n" + "="*60)
    print("📊 Speed Optimization Summary:")
    print(f"   Average speedup across all batch sizes: {np.mean(speedups):.2f}x")
    print(f"   Maximum speedup: {np.max(speedups):.2f}x")
    print("="*60)
    
    print("\n🎉 All done!")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    main()
