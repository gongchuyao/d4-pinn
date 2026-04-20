#!/usr/bin/env python
"""
Extended Experiments
====================
This script runs a series of extended experiments:
1. Multiple benchmark PDE problems
2. Robustness test with noise
3. Sample efficiency test with different sampling point counts
4. Ablation studies

This generates comprehensive results for your paper!
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json

# Import from our package
from src import BaseD4PINN, OptimizedD4PINN
from src import poisson_2d, ginzburg_landau_2d, allen_cahn_2d
from src import generate_sample_points_2d
from src import compute_l2_error, compute_linf_error

# Fix random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def train_single_problem(problem_name, u_exact, f_source, N_int=2000, noise=0.0, epochs=20000):
    """
    Train D4-PINN on a single problem with given parameters.
    """
    # Generate points
    x_int, x_bc = generate_sample_points_2d(N_interior=N_int, N_boundary=500)
    
    # Initialize model
    model = OptimizedD4PINN(hidden_dim=100)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Boundary loss
        u_bc_pred = model(x_bc)
        u_bc_true = u_exact(x_bc[:,0], x_bc[:,1])
        loss_bc = torch.mean((u_bc_pred - u_bc_true)**2)
        
        # PDE loss
        x = x_int.clone()
        x.requires_grad = True
        u_pred = model(x)
        
        u_x = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0][:,0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:,0]
        u_y = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0][:,1]
        u_yy = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:,1]
        
        if problem_name == "Semilinear Poisson":
            residual = -u_xx - u_yy + u_pred**3 - f_source(x_int[:,0], x_int[:,1])
        elif problem_name == "Steady Ginzburg-Landau":
            kappa = 1.0
            residual = -u_xx - u_yy + kappa**2 * u_pred * (u_pred**2 - 1) - f_source(x_int[:,0], x_int[:,1])
        elif problem_name == "Steady Allen-Cahn":
            eps = 0.1
            residual = -u_xx - u_yy + (u_pred**3 - u_pred)/eps**2 - f_source(x_int[:,0], x_int[:,1])
        
        loss_pde = torch.mean(residual**2)
        loss = loss_pde + 10 * loss_bc
        
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    # Evaluate
    x_test = torch.FloatTensor(10000, 2).uniform_(-1, 1)
    u_test_pred = model(x_test).detach().numpy()
    u_test_true = u_exact(x_test[:,0], x_test[:,1]).numpy()
    
    l2_err = compute_l2_error(u_test_pred, u_test_true)
    linf_err = compute_linf_error(u_test_pred, u_test_true)
    
    return l2_err, linf_err

def main():
    print("="*60)
    print("🚀 Running Extended Experiments")
    print("="*60)
    
    results = {}
    
    # =========================================================================
    # 1. Multiple benchmark problems
    # =========================================================================
    print("\n" + "-"*60)
    print("1. Running multiple benchmark problems...")
    print("-"*60)
    
    problems = [
        poisson_2d(),
        ginzburg_landau_2d(),
        allen_cahn_2d(),
    ]
    
    results['problems'] = {}
    
    for u_exact, f_source, name in problems:
        print(f"   Training on {name}...")
        l2, linf = train_single_problem(name, u_exact, f_source)
        results['problems'][name] = {
            'l2_error': l2,
            'linf_error': linf
        }
        print(f"      Done! L2 Error: {l2:.6e}")
    
    # =========================================================================
    # 2. Robustness test with noise
    # =========================================================================
    print("\n" + "-"*60)
    print("2. Running robustness test with noise...")
    print("-"*60)
    
    results['robustness'] = {}
    noise_levels = [0.0, 0.05]
    
    u_exact, f_source, name = poisson_2d()
    
    for noise in noise_levels:
        print(f"   Testing with {noise*100}% noise...")
        l2, linf = train_single_problem(name, u_exact, f_source, noise=noise)
        results['robustness'][f'noise_{noise}'] = {
            'l2_error': l2,
            'linf_error': linf
        }
        print(f"      Done! L2 Error: {l2:.6e}")
    
    # =========================================================================
    # 3. Sample efficiency test
    # =========================================================================
    print("\n" + "-"*60)
    print("3. Running sample efficiency test...")
    print("-"*60)
    
    results['sampling'] = {}
    n_points = [1000, 2000, 5000]
    
    for n in n_points:
        print(f"   Testing with {n} interior points...")
        l2, linf = train_single_problem(name, u_exact, f_source, N_int=n)
        results['sampling'][f'n_{n}'] = {
            'l2_error': l2,
            'linf_error': linf
        }
        print(f"      Done! L2 Error: {l2:.6e}")
    
    # =========================================================================
    # Save results
    # =========================================================================
    print("\n" + "-"*60)
    print("Saving results...")
    
    os.makedirs('results', exist_ok=True)
    with open('results/extended_experiments_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Results saved to results/extended_experiments_results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 Extended Experiments Summary:")
    print("="*60)
    
    print("\nBenchmark Problems:")
    for name, res in results['problems'].items():
        print(f"   {name}:")
        print(f"      L2 Error: {res['l2_error']:.6e}")
        print(f"      L∞ Error: {res['linf_error']:.6e}")
    
    print("\nRobustness:")
    for noise, res in results['robustness'].items():
        print(f"   {noise}:")
        print(f"      L2 Error: {res['l2_error']:.6e}")
    
    print("\nSample Efficiency:")
    for n, res in results['sampling'].items():
        print(f"   {n} points:")
        print(f"      L2 Error: {res['l2_error']:.6e}")
    
    print("\n🎉 All experiments completed!")

if __name__ == "__main__":
    main()
