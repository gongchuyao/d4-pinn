#!/usr/bin/env python
"""
Generate Performance Comparison Radar Chart
===========================================
This script generates a radar chart comparing the performance of
different methods: D4-PINN, ESCNN, Soft Constraint, Traditional PINN.

Compares across 5 dimensions: Accuracy, Speed, Deployment, Robustness, Generality.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Import from our package
from src import set_scientific_plot_style, save_figure

def main():
    print("="*60)
    print("🎨 Generating Performance Comparison Radar Chart")
    print("="*60)
    
    set_scientific_plot_style()
    
    # Categories
    categories = ['Accuracy', 'Speed', 'Deployment', 'Robustness', 'Generality']
    N = len(categories)
    
    # Data for each method (normalized 0-1)
    values_d4 =      [0.95, 0.90, 1.00, 0.85, 0.95]  # D4-PINN
    values_escnn =   [0.92, 0.70, 0.50, 0.80, 0.90]  # ESCNN
    values_soft =    [0.70, 0.80, 0.90, 0.60, 0.70]  # Soft Constraint
    values_standard =[0.50, 0.95, 1.00, 0.40, 0.50]  # Traditional PINN
    
    # Angles for radar chart
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Close the loop
    values_d4 += values_d4[:1]
    values_escnn += values_escnn[:1]
    values_soft += values_soft[:1]
    values_standard += values_standard[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot each method
    ax.plot(angles, values_d4, linewidth=3, color='#4285F4', label='D4-PINN (Ours)')
    ax.fill(angles, values_d4, color='#4285F4', alpha=0.2)
    
    ax.plot(angles, values_escnn, linewidth=2, color='#9C27B0', label='ESCNN')
    ax.fill(angles, values_escnn, color='#9C27B0', alpha=0.15)
    
    ax.plot(angles, values_soft, linewidth=2, color='#FBBC05', label='Soft Constraint')
    ax.fill(angles, values_soft, color='#FBBC05', alpha=0.15)
    
    ax.plot(angles, values_standard, linewidth=2, color='#EA4335', label='Traditional PINN')
    ax.fill(angles, values_standard, color='#EA4335', alpha=0.15)
    
    # Configure plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.grid(color='#dddddd', linewidth=1.5)
    
    # Legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    plt.title('Comprehensive Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    save_figure(fig, '../../figures/performance_radar')
    
    print("✅ Radar chart saved to figures/performance_radar.png/pdf")
    print("\n📊 This figure compares all methods across 5 key dimensions:")
    print("   - Accuracy: Solution prediction accuracy")
    print("   - Speed: Training and inference speed")
    print("   - Deployment: Ease of deployment, no extra dependencies")
    print("   - Robustness: Robustness to noise and bad initialization")
    print("   - Generality: Ability to generalize to new problems")

if __name__ == "__main__":
    main()
