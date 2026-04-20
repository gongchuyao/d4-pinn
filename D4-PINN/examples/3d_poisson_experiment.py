#!/usr/bin/env python
"""
3D Poisson Experiment
=====================
This example shows how to extend D4-PINN to 3D problems.
We solve the 3D Poisson equation, applying D4 symmetry on the xy plane.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import from our package
from src import BaseD4PINN, poisson_3d, generate_sample_points_3d
from src import compute_l2_error, compute_linf_error
from src import plot_convergence_curves

# Fix random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# We need to define the 3D version of D4-PINN here, since it's slightly different
class D4PINN3D(nn.Module):
    """
    3D version of D4-PINN.
    Applies D4 transformations on the xy plane, z remains unchanged.
    """
    def __init__(self, hidden_dim=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        from src.core.d4_transforms import d4_transform_3d
        outs = []
        for op in range(8):
            x_trans = d4_transform_3d(x, op)
            outs.append(self.net(x_trans))
        return torch.stack(outs).mean(dim=0).squeeze()

def main():
    print("="*60)
    print("🚀 Running 3D Poisson Experiment")
    print("="*60)
    
    # Load the 3D PDE problem
    u_exact, f_source, problem_name = poisson_3d()
    print(f"📊 Problem: {problem_name}")
    
    # Generate sampling points
    x_int, x_bc = generate_sample_points_3d(N_interior=2000, N_boundary=600)
    print(f"📊 Generated {len(x_int)} interior points, {len(x_bc)} boundary points")
    
    # Initialize 3D model
    model = D4PINN3D(hidden_dim=50)
    print(f"📊 Model: 3D D4-PINN with hidden_dim=50")
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    
    # Training parameters
    epochs = 5000
    loss_hist, err_hist = [], []
    
    print("\n🏋️ Starting training...")
    # Training loop
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        
        # Boundary loss
        u_bc_pred = model(x_bc)
        loss_bc = torch.mean(u_bc_pred**2)  # Dirichlet BC: u=0 on boundary
        
        # PDE residual loss
        x = x_int.clone()
        x.requires_grad = True
        u_pred = model(x)
        
        # Compute derivatives
        u_x = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0][:,0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:,0]
        u_y = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0][:,1]
        u_yy = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:,1]
        u_z = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0][:,2]
        u_zz = torch.autograd.grad(u_z.sum(), x, create_graph=True)[0][:,2]
        
        # Compute residual
        residual = -u_xx - u_yy - u_zz - f_source(x_int[:,0], x_int[:,1], x_int[:,2])
        loss_pde = torch.mean(residual**2)
        
        # Total loss
        loss = loss_pde + 10 * loss_bc
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Logging
        if epoch % 100 == 0:
            # Compute test error
            x_test = torch.FloatTensor(5000, 3).uniform_(-1, 1)
            u_test_pred = model(x_test).detach()
            u_test_true = u_exact(x_test[:,0], x_test[:,1], x_test[:,2])
            l2_err = torch.norm(u_test_pred - u_test_true) / torch.norm(u_test_true)
            
            loss_hist.append(loss.item())
            err_hist.append(l2_err.item())
    
    print("\n✅ Training completed!")
    
    # Generate plots
    print("\n📊 Generating plots...")
    
    # Convergence curves
    plot_convergence_curves(loss_hist, err_hist, save_path='results/3d_convergence')
    print("   ✅ Convergence curves saved to results/3d_convergence.png/pdf")
    
    # Slice comparison (z=0 plane)
    x = np.linspace(-1,1,100); y = np.linspace(-1,1,100)
    X,Y = np.meshgrid(x,y)
    x_grid = torch.FloatTensor(np.stack([X.flatten(), Y.flatten(), np.zeros_like(X.flatten())], axis=1))
    
    u_pred = model(x_grid).detach().numpy().reshape(X.shape)
    u_true = u_exact(x_grid[:,0], x_grid[:,1], x_grid[:,2]).numpy().reshape(X.shape)
    err = np.abs(u_pred - u_true)
    
    # Plot slice comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im1 = axes[0].contourf(X, Y, u_true, 50, cmap='viridis')
    fig.colorbar(im1, ax=axes[0])
    axes[0].set_title('Exact Solution (z=0 slice)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    
    im2 = axes[1].contourf(X, Y, u_pred, 50, cmap='viridis')
    fig.colorbar(im2, ax=axes[1])
    axes[1].set_title('D4-PINN Prediction (z=0 slice)')
    axes[1].set_xlabel('x')
    
    im3 = axes[2].contourf(X, Y, err, 50, cmap='jet')
    fig.colorbar(im3, ax=axes[2])
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x')
    
    plt.tight_layout()
    from src.utils.visualization import save_figure
    save_figure(fig, 'results/3d_slice_comparison')
    print("   ✅ Slice comparison saved to results/3d_slice_comparison.png/pdf")
    
    # Final metrics
    x_test = torch.FloatTensor(10000, 3).uniform_(-1, 1)
    u_pred_test = model(x_test).detach().numpy()
    u_true_test = u_exact(x_test[:,0], x_test[:,1], x_test[:,2]).numpy()
    
    final_l2 = compute_l2_error(u_pred_test, u_true_test)
    final_linf = compute_linf_error(u_pred_test, u_true_test)
    
    print("\n" + "="*60)
    print("✅ Final Results:")
    print(f"   L2 Relative Error: {final_l2:.6e}")
    print(f"   L∞ Relative Error: {final_linf:.6e}")
    print("="*60)
    
    print("\n🎉 All done! Check the 'results' folder for the generated figures.")

if __name__ == "__main__":
    # Create results folder if it doesn't exist
    os.makedirs('results', exist_ok=True)
    main()
