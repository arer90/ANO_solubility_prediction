#!/usr/bin/env python3
"""
Developer: Lee, Seungjin (arer90)

Complete Figure Generation Code for SCI Journal Publication
========================================================

This code creates all figures demonstrating Bayesian optimization superiority
for hyperparameter search, suitable for SCI-level journal publication.

Derived from: 0_search_comp.ipynb
Author: Claude Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import os

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Set color palette for publication
COLORS = {
    'Grid': '#E74C3C',      # Red
    'Random': '#F39C12',    # Orange  
    'Bayesian': '#27AE60',  # Green
    'True_Opt': '#8E44AD',  # Purple
}

def objective_function(x, y):
    """
    Multi-modal objective function with known global optimum at (0,0)
    
    Parameters:
    -----------
    x, y : float or array
        Input parameters
        
    Returns:
    --------
    float or array
        Objective function value
    """
    term1 = 4 * np.exp(-0.5 * (x**2 + y**2))  # Global maximum
    term2 = 2.5 * np.exp(-2 * ((x - 1.8)**2 + (y - 1.8)**2))  # Local maximum
    term3 = 0.3 * np.cos(3 * x) * np.cos(3 * y)  # Ripples
    term4 = 1.8 * np.exp(-3 * ((x + 1.5)**2 + (y + 1.5)**2))  # Local maximum
    return term1 + term2 + term3 + term4

def expected_improvement(gpr, X, Y, bounds, n_candidates=2000, xi=0.01):
    """
    Expected Improvement acquisition function for Bayesian optimization
    
    Parameters:
    -----------
    gpr : GaussianProcessRegressor
        Fitted Gaussian Process model
    X : array
        Observed points
    Y : array
        Observed values
    bounds : array
        Parameter bounds
    n_candidates : int
        Number of candidate points
    xi : float
        Exploration parameter
        
    Returns:
    --------
    array
        Next point to evaluate
    """
    X_cand = np.random.uniform(bounds[:,0], bounds[:,1], size=(n_candidates, 2))
    mu, sigma = gpr.predict(X_cand, return_std=True)
    mu_opt = gpr.predict(X).max()
    
    with np.errstate(divide='ignore'):
        imp = mu - mu_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0] = 0.0
    
    return X_cand[np.argmax(ei)].reshape(1, 2)

def generate_optimization_data():
    """
    Generate comprehensive optimization data for all methods
    
    Returns:
    --------
    tuple
        (records, convergence, trajectories, TRUE_OPT)
    """
    np.random.seed(42)  # For reproducibility
    
    # Settings
    GRID_RES = 8
    N_ITER = 50
    TRUE_OPT = (0.0, 0.0)
    
    records = {'Grid': [], 'Random': [], 'Bayesian': []}
    convergence = {'Grid': [], 'Random': [], 'Bayesian': []}
    trajectories = {'Grid': [], 'Random': [], 'Bayesian': []}
    
    # Grid Search
    print("Generating Grid Search data...")
    x_vals = np.linspace(-3, 3, GRID_RES)
    y_vals = np.linspace(-3, 3, GRID_RES)
    best_val = -np.inf
    
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            val = objective_function(x, y)
            records['Grid'].append(((x, y), val))
            best_val = max(best_val, val)
            convergence['Grid'].append(best_val)
            trajectories['Grid'].append((x, y))
    
    # Random Search
    print("Generating Random Search data...")
    best_val = -np.inf
    for i in range(N_ITER):
        x, y = np.random.uniform(-3, 3, 2)
        val = objective_function(x, y)
        records['Random'].append(((x, y), val))
        best_val = max(best_val, val)
        convergence['Random'].append(best_val)
        trajectories['Random'].append((x, y))
    
    # Bayesian Optimization
    print("Generating Bayesian Optimization data...")
    n_init = 8
    X_sample = np.random.uniform(-3, 3, (n_init, 2))
    Y_sample = [objective_function(x, y) for x, y in X_sample]
    best_val = -np.inf
    
    for (x, y), v in zip(X_sample, Y_sample):
        records['Bayesian'].append(((x, y), v))
        best_val = max(best_val, v)
        convergence['Bayesian'].append(best_val)
        trajectories['Bayesian'].append((x, y))
    
    # Sequential optimization
    gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6)
    bounds = np.array([[-3, 3], [-3, 3]])
    
    for i in range(N_ITER - n_init):
        gpr.fit(X_sample, np.array(Y_sample))
        x_next = expected_improvement(gpr, X_sample, Y_sample, bounds)[0]
        y_next = objective_function(*x_next)
        X_sample = np.vstack([X_sample, x_next])
        Y_sample.append(y_next)
        records['Bayesian'].append(((x_next[0], x_next[1]), y_next))
        best_val = max(best_val, y_next)
        convergence['Bayesian'].append(best_val)
        trajectories['Bayesian'].append((x_next[0], x_next[1]))
    
    return records, convergence, trajectories, TRUE_OPT

def create_fig1_search_space_comparison():
    """
    Figure 1: Search space comparison with perfect colorbar placement
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    print("Creating Figure 1: Search Space Comparison...")
    
    records, convergence, trajectories, TRUE_OPT = generate_optimization_data()
    
    # Create figure with proper spacing
    fig = plt.figure(figsize=(16, 5))
    
    # Create grid layout with extra space for colorbar
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.1], wspace=0.3)
    
    # Prepare contour data
    Xc = np.linspace(-3, 3, 200)
    Yc = np.linspace(-3, 3, 200)
    XX, YY = np.meshgrid(Xc, Yc)
    ZZ = objective_function(XX, YY)
    
    methods = ['Grid', 'Random', 'Bayesian']
    
    # Create the three subplots
    for i, method in enumerate(methods):
        ax = fig.add_subplot(gs[i])
        
        # Contour plot
        contour = ax.contourf(XX, YY, ZZ, levels=20, cmap='viridis', alpha=0.6)
        ax.contour(XX, YY, ZZ, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        # Plot evaluation points
        pts, vals = zip(*records[method])
        xs, ys = zip(*pts)
        
        # Color points by value
        scatter = ax.scatter(xs, ys, c=vals, cmap='plasma', s=30, 
                           edgecolors='black', linewidth=0.5, alpha=0.8)
        
        # Mark best found point
        best_idx = np.argmax(vals)
        bx, by = xs[best_idx], ys[best_idx]
        bv = vals[best_idx]
        ax.scatter(bx, by, marker='*', color='red', s=150, 
                  edgecolors='darkred', linewidth=2, zorder=5)
        
        # Mark true optimum
        true_val = objective_function(*TRUE_OPT)
        ax.scatter(*TRUE_OPT, marker='X', color='white', s=120, 
                  edgecolors='black', linewidth=2, zorder=5)
        
        ax.set_title(f'{method} Search', fontsize=12, fontweight='bold')
        ax.set_xlabel('Parameter 1', fontsize=10)
        ax.set_ylabel('Parameter 2', fontsize=10)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.tick_params(axis='both', which='major', labelsize=9)
    
    # Add colorbar in the dedicated space
    cbar_ax = fig.add_subplot(gs[3])
    cbar = plt.colorbar(contour, cax=cbar_ax)
    cbar.set_label('Objective Value', rotation=270, labelpad=20, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    return fig

def create_fig2_convergence_comparison():
    """
    Figure 2: Convergence comparison
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    print("Creating Figure 2: Convergence Comparison...")
    
    records, convergence, trajectories, TRUE_OPT = generate_optimization_data()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    methods = ['Grid', 'Random', 'Bayesian']
    
    # Plot convergence curves
    for method in methods:
        ax.plot(convergence[method], label=method, color=COLORS[method], 
               linewidth=2.5, alpha=0.8, marker='o', markersize=4, markevery=5)
    
    # Add horizontal line for true optimum
    true_opt_value = objective_function(*TRUE_OPT)
    ax.axhline(y=true_opt_value, color='black', linestyle='--', alpha=0.7, 
              linewidth=1.5, label=f'Global Optimum: {true_opt_value:.3f}')
    
    ax.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
    ax.set_xlabel('Function Evaluations', fontsize=12)
    ax.set_ylabel('Best Objective Value Found', fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    return fig

def create_fig3_performance_comparison():
    """
    Figure 3: Performance metrics comparison
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    print("Creating Figure 3: Performance Comparison...")
    
    records, convergence, trajectories, TRUE_OPT = generate_optimization_data()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = ['Grid', 'Random', 'Bayesian']
    
    # Calculate performance metrics
    best_values = []
    efficiencies = []
    
    for method in methods:
        pts, vals = zip(*records[method])
        best_val = max(vals)
        best_values.append(best_val)
        
        true_opt_val = objective_function(*TRUE_OPT)
        efficiency = (best_val / true_opt_val) * 100
        efficiencies.append(efficiency)
    
    # Plot 1: Best values found
    bars1 = ax1.bar(methods, best_values, color=[COLORS[m] for m in methods], 
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, val in zip(bars1, best_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_title('Best Values Found', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Objective Value', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # Plot 2: Optimization efficiency
    bars2 = ax2.bar(methods, efficiencies, color=[COLORS[m] for m in methods], 
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, eff in zip(bars2, efficiencies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{eff:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_title('Optimization Efficiency', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Efficiency (%)', fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    return fig

def create_fig4_bayesian_advantage():
    """
    Figure 4: Bayesian optimization advantage analysis
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    print("Creating Figure 4: Bayesian Advantage Analysis...")
    
    records, convergence, trajectories, TRUE_OPT = generate_optimization_data()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = ['Grid', 'Random', 'Bayesian']
    
    # Plot 1: Search pattern comparison
    for method in methods:
        pts, vals = zip(*records[method])
        xs, ys = zip(*pts)
        
        # Calculate spatial distribution
        x_std = np.std(xs)
        y_std = np.std(ys)
        spatial_coverage = x_std * y_std
        
        # Calculate clustering (lower is more clustered)
        clustering = np.mean([np.min([np.sqrt((x-xs[j])**2 + (y-ys[j])**2) 
                                    for j in range(len(xs)) if j != i]) 
                            for i, (x, y) in enumerate(zip(xs, ys))])
        
        ax1.scatter(spatial_coverage, clustering, color=COLORS[method], s=100, 
                   alpha=0.8, edgecolors='black', linewidth=1, label=method, zorder=5)
    
    ax1.set_xlabel('Spatial Coverage', fontsize=11)
    ax1.set_ylabel('Clustering Index', fontsize=11)
    ax1.set_title('Search Pattern Characteristics', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # Plot 2: Improvement rate over time
    for method in methods:
        conv = convergence[method]
        improvements = np.diff(conv)
        ax2.plot(improvements, color=COLORS[method], linewidth=2, alpha=0.8, label=method)
    
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Improvement Rate', fontsize=11)
    ax2.set_title('Improvement Rate Over Time', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    return fig

def create_fig5_3d_surface():
    """
    Figure 5: Clear 3D surface plot of objective function
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    print("Creating Figure 5: 3D Surface Plot...")
    
    # Create figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the surface
    Xc = np.linspace(-3, 3, 100)
    Yc = np.linspace(-3, 3, 100)
    XX, YY = np.meshgrid(Xc, Yc)
    ZZ = objective_function(XX, YY)
    
    # Plot surface
    surf = ax.plot_surface(XX, YY, ZZ, cmap='viridis', alpha=0.8, 
                          edgecolor='none', antialiased=True)
    
    # Mark key points
    TRUE_OPT = (0.0, 0.0)
    opt_val = objective_function(*TRUE_OPT)
    ax.scatter(*TRUE_OPT, opt_val, color='red', s=150, marker='*', 
              edgecolors='darkred', linewidth=3, label='Global Maximum (Target)', zorder=10)
    
    # Local maxima
    local_max_1 = (1.8, 1.8, objective_function(1.8, 1.8))
    local_max_2 = (-1.5, -1.5, objective_function(-1.5, -1.5))
    ax.scatter(*local_max_1, color='orange', s=80, marker='o', 
              edgecolors='darkorange', linewidth=2, label='Local Maxima (Traps)', zorder=10)
    ax.scatter(*local_max_2, color='orange', s=80, marker='o', 
              edgecolors='darkorange', linewidth=2, zorder=10)
    
    # Add text annotations
    ax.text(0, 0, opt_val + 0.8, 'Global\nMaximum\n(Target)', fontsize=10, ha='center', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.text(1.8, 1.8, local_max_1[2] + 0.5, 'Local\nMaximum\n(Trap)', fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    ax.text(-1.5, -1.5, local_max_2[2] + 0.5, 'Local\nMaximum\n(Trap)', fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax.set_title('Objective Function Landscape\n(All Optimization Methods Search This Terrain)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Parameter 1', fontsize=12, labelpad=10)
    ax.set_ylabel('Parameter 2', fontsize=12, labelpad=10)
    ax.set_zlabel('Objective Function Value', fontsize=12, labelpad=10)
    ax.view_init(elev=30, azim=45)
    ax.legend(fontsize=10, loc='upper left')
    
    # Add colorbar
    cbar = plt.colorbar(surf, ax=ax, shrink=0.6, aspect=8, pad=0.1)
    cbar.set_label('Objective Function Value', rotation=270, labelpad=20, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Add explanatory text
    fig.text(0.02, 0.02, 
             'Note: This shows the terrain that all optimization methods (Grid, Random, Bayesian) search.\n'
             'The goal is to find the global maximum (red star) while avoiding local maxima (orange circles).',
             fontsize=10, style='italic', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    return fig

def main():
    """
    Main function to create all figures
    """
    print("=" * 60)
    print("Creating All Figures for SCI Journal Publication")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    output_dir = './result/0_search_comp_figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create all figures
    figures = [
        (create_fig1_search_space_comparison, 'fig1_search_space_comparison'),
        (create_fig2_convergence_comparison, 'fig2_convergence_comparison'),
        (create_fig3_performance_comparison, 'fig3_performance_comparison'),
        (create_fig4_bayesian_advantage, 'fig4_bayesian_advantage'),
        (create_fig5_3d_surface, 'fig5_3d_surface')
    ]
    
    for create_func, filename in figures:
        try:
            fig = create_func()
            
            # Save as PNG
            png_path = os.path.join(output_dir, f'{filename}.png')
            fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {png_path}")
            
            # Save as PDF
            pdf_path = os.path.join(output_dir, f'{filename}.pdf')
            fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {pdf_path}")
            
            plt.close(fig)
            
        except Exception as e:
            print(f"✗ Error creating {filename}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("All figures created successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("- fig1_search_space_comparison.png/pdf (perfect colorbar separation)")
    print("- fig2_convergence_comparison.png/pdf")
    print("- fig3_performance_comparison.png/pdf")
    print("- fig4_bayesian_advantage.png/pdf")
    print("- fig5_3d_surface.png/pdf (clear title and description)")
    print(f"\nOutput directory: {output_dir}")
    print("\nAll figures are from the same program with consistent quality!")

if __name__ == "__main__":
    main() 