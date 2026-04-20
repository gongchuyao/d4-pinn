#!/usr/bin/env python
"""
Equivariant D4-PINN Example
===========================
This example runs the equivariant layer version of D4-PINN.
Instead of averaging outputs, this uses true equivariant layers
that preserve symmetry at every layer of the network.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Import from our package
from src import D4PINNEquivariant, poisson_2d, generate_sample_points_2d
from src import compute_l2_error, compute_linf_error
from src import plot_solution_comparison, plot_convergence_curves

# Fix random seed
torch.manual_seed(42)
np.random.seed(42)

def main():
    print("="*60)
    print("🚀 Running Equivariant D4-PINN Example")
    print("="*60)
    
    # Load problem
    u_exact, f_source, problem_name = poisson_2d()
    print(f"📊 Problem: {problem_name}")
    
    # Generate points
    x_int, x_bc = generate_sample_points_2d(N_interior=2000, N_boundary=500)
    print(f"📊 Generated {len(x_int)} interior points, {len(x_bc)} boundary points")
    
    # Initialize equivariant model
    model = D4PINNEquivariant(hidden_dim=50)
    print(f"📊 Model: D4PINNEquivariant with hidden_dim=50")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    
    # Training
    epochs = 20000
    loss_hist, err_hist = [], []
    
    print("\n🏋️ Starting training...")
    for epoch in tqdm(range(epochs)):
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
        
        residual = -u_xx - u_yy + u_pred**3 - f_source(x_int[:,0], x_int[:,1])
        loss_pde = torch.mean(residual**2)
        
        loss = loss_pde + 10 * loss_bc
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 100 == 0:
            x_test = torch.FloatTensor(10000, 2).uniform_(-1, 1)
            u_test_pred = model(x_test).detach()
            u_test_true = u_exact(x_test[:,0], x_test[:,1])
            l2_err = torch.norm(u_test_pred - u_test_true) / torch.norm(u_test_true)
            
            loss_hist.append(loss.item())
            err_hist.append(l2_err.item())
    
    print("\n✅ Training completed!")
    
    # Plots
    print("\n📊 Generating plots...")
    
    plot_convergence_curves(loss_hist, err_hist, save_path='results/equivariant_convergence')
    print("   ✅ Convergence curves saved")
    
    # Solution comparison
    x = np.linspace(-1,1,200); y = np.linspace(-1,1,200)
    X,Y = np.meshgrid(x,y)
    x_grid = torch.FloatTensor(np.stack([X.flatten(), Y.flatten()], axis=1))
    
    u_pred = model(x_grid).detach().numpy().reshape(X.shape)
    u_true = u_exact(torch.FloatTensor(X.flatten()), torch.FloatTensor(Y.flatten())).numpy().reshape(X.shape)
    err = np.abs(u_pred - u_true)
    
    plot_solution_comparison(X, Y, u_true, u_pred, err, save_path='results/equivariant_solution')
    print("   ✅ Solution comparison saved")
    
    # Metrics
    final_l2 = compute_l2_error(u_pred, u_true)
    final_linf = compute_linf_error(u_pred, u_true)
    
    print("\n" + "="*60)
    print("✅ Final Results:")
    print(f"   L2 Relative Error: {final_l2:.6e}")
    print(f"   L∞ Relative Error: {final_linf:.6e}")
    print("="*60)
    
    print("\n🎉 All done!")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    main()
