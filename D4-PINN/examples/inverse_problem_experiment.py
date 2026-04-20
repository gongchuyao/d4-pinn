#!/usr/bin/env python
"""
Inverse Problem Experiment
==========================
This example shows how to use D4-PINN to solve inverse problems.
In an inverse problem, we don't know some parameters of the PDE,
but we have some observation data. D4-PINN can simultaneously:
1. Solve for the solution u(x,y)
2. Infer the unknown parameter λ

This also works with noisy observation data!
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import from our package
from src import BaseD4PINN
from src import compute_l2_error, compute_linf_error
from src import plot_solution_comparison, plot_convergence_curves

# Fix random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def main():
    print("="*60)
    print("🚀 Running Inverse Problem Experiment")
    print("="*60)
    
    # Inverse problem: We want to find λ in:
    # -Δu + λ u³ = f
    # We have some observation data of u
    
    # True parameter we want to recover
    true_lambda = 1.0
    print(f"🎯 True unknown parameter λ = {true_lambda}")
    
    # Exact solution (same as Poisson problem)
    def u_exact(x, y):
        return (1 - x**2) * (1 - y**2) * (1 + 0.5 * (x**2 + y**2))
    
    # Source term with true λ
    def f_source(x, y):
        x.requires_grad = True
        y.requires_grad = True
        u = u_exact(x, y)
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        return (-u_xx - u_yy + true_lambda * u**3).detach()
    
    # Generate observation data (with optional noise)
    N_obs = 1000
    x_obs = torch.FloatTensor(N_obs, 2).uniform_(-1, 1)
    u_obs_true = u_exact(x_obs[:,0], x_obs[:,1])
    
    # Add 5% noise to observations
    noise_level = 0.05
    u_obs = u_obs_true + torch.randn_like(u_obs_true) * noise_level * torch.std(u_obs_true)
    print(f"📊 Observation data: {N_obs} points with {noise_level*100}% noise")
    
    # Generate interior points for PDE residual
    x_int = torch.FloatTensor(2000, 2).uniform_(-1, 1)
    
    # Initialize model
    model = BaseD4PINN(hidden_dim=100)
    
    # Initialize the unknown parameter as a learnable parameter!
    lambda_hat = nn.Parameter(torch.tensor(0.5))  # Start with initial guess
    print(f"📊 Initial guess for λ = {lambda_hat.item():.4f}")
    
    # Optimizer: optimize both model parameters AND the unknown λ!
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [lambda_hat], 
        lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    
    # Training parameters
    epochs = 20000
    loss_hist, err_hist, lambda_hist = [], [], []
    
    print("\n🏋️ Starting inverse problem training...")
    # Training loop
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        
        # Data loss: fit to observation data
        u_pred_obs = model(x_obs)
        loss_data = torch.mean((u_pred_obs - u_obs)**2)
        
        # PDE residual loss
        x = x_int.clone()
        x.requires_grad = True
        u_pred = model(x)
        
        u_x = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0][:,0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:,0]
        u_y = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0][:,1]
        u_yy = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:,1]
        
        residual = -u_xx - u_yy + lambda_hat * u_pred**3 - f_source(x_int[:,0], x_int[:,1])
        loss_pde = torch.mean(residual**2)
        
        # Total loss
        loss = loss_pde + loss_data
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Logging
        if epoch % 100 == 0:
            x_test = torch.FloatTensor(10000, 2).uniform_(-1, 1)
            u_test_pred = model(x_test).detach()
            u_test_true = u_exact(x_test[:,0], x_test[:,1])
            l2_err = torch.norm(u_test_pred - u_test_true) / torch.norm(u_test_true)
            
            loss_hist.append(loss.item())
            err_hist.append(l2_err.item())
            lambda_hist.append(lambda_hat.item())
    
    print("\n✅ Training completed!")
    
    # Final parameter recovery
    print(f"\n📊 Recovered λ = {lambda_hat.item():.6f}")
    print(f"📊 True λ = {true_lambda}")
    print(f"📊 Parameter error: {abs(lambda_hat.item() - true_lambda):.6e}")
    
    # Generate plots
    print("\n📊 Generating plots...")
    
    # Convergence curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].semilogy(loss_hist)
    axes[0].set_xlabel('Iteration (×100)')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)
    
    axes[1].semilogy(err_hist)
    axes[1].set_xlabel('Iteration (×100)')
    axes[1].set_ylabel('L2 Relative Error')
    axes[1].set_title('Solution Error')
    axes[1].grid(True)
    
    axes[2].plot(lambda_hist)
    axes[2].axhline(y=true_lambda, color='r', linestyle='--', label='True λ')
    axes[2].set_xlabel('Iteration (×100)')
    axes[2].set_ylabel('λ')
    axes[2].set_title('Parameter Recovery')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    from src.utils.visualization import save_figure
    save_figure(fig, 'results/inverse_convergence')
    print("   ✅ Convergence curves saved to results/inverse_convergence.png/pdf")
    
    # Solution comparison
    x = np.linspace(-1,1,200); y = np.linspace(-1,1,200)
    X,Y = np.meshgrid(x,y)
    x_grid = torch.FloatTensor(np.stack([X.flatten(), Y.flatten()], axis=1))
    
    u_pred = model(x_grid).detach().numpy().reshape(X.shape)
    u_true = u_exact(torch.FloatTensor(X.flatten()), torch.FloatTensor(Y.flatten())).numpy().reshape(X.shape)
    err = np.abs(u_pred - u_true)
    
    plot_solution_comparison(X, Y, u_true, u_pred, err, save_path='results/inverse_solution')
    print("   ✅ Solution comparison saved to results/inverse_solution.png/pdf")
    
    # Final metrics
    final_l2 = compute_l2_error(u_pred, u_true)
    final_linf = compute_linf_error(u_pred, u_true)
    
    print("\n" + "="*60)
    print("✅ Final Results:")
    print(f"   Recovered λ: {lambda_hat.item():.6f} (True: {true_lambda})")
    print(f"   L2 Relative Error: {final_l2:.6e}")
    print(f"   L∞ Relative Error: {final_linf:.6e}")
    print("="*60)
    
    print("\n🎉 All done! Check the 'results' folder for the generated figures.")

if __name__ == "__main__":
    # Create results folder if it doesn't exist
    os.makedirs('results', exist_ok=True)
    main()
