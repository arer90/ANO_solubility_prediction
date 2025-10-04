"""
QSAR Metrics Visualization Module - 2025 Update
Focuses on RMSE differences and uses clean plots without grid
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Import config
try:
    from ..config import PLOT_SETTINGS
except ImportError:
    PLOT_SETTINGS = {
        'figure_dpi': 300,
        'figure_dpi_high': 400,
        'style': 'seaborn-v0_8-white',  # No grid
        'palette': 'husl'
    }


class MetricVisualizer:
    """Visualize QSAR metrics with focus on differences"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style without grid
        try:
            plt.style.use(PLOT_SETTINGS['style'])
        except:
            plt.style.use('seaborn-white')  # Fallback to white (no grid)
        
        # Color palette
        self.colors = sns.color_palette(PLOT_SETTINGS.get('palette', 'husl'))
    
    def plot_rmse_differences(self, split_metrics: Dict, save_path: Optional[Path] = None):
        """Plot RMSE differences between inner and outer sets"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Prepare data
        splits = []
        rmse_diffs = []
        rmse_relative_diffs = []
        qualities = []
        
        for split_name, metrics in split_metrics.items():
            if 'difference_evaluation' in metrics:
                diff = metrics['difference_evaluation']
                splits.append(split_name)
                rmse_diffs.append(diff['rmse_difference'])
                rmse_relative_diffs.append(diff['rmse_relative_difference'])
                qualities.append(diff['quality'])
        
        if not splits:
            plt.close()
            return
        
        # Plot absolute differences
        bars1 = ax1.bar(range(len(splits)), rmse_diffs, color=self.colors[0])
        ax1.set_xlabel('Split Type')
        ax1.set_ylabel('RMSE Difference (Outer - Inner)')
        ax1.set_title('RMSE Differences by Split Type')
        ax1.set_xticks(range(len(splits)))
        ax1.set_xticklabels(splits, rotation=45, ha='right')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Poor threshold')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, rmse_diffs)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height > 0 else height - 0.01,
                    f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # Plot relative differences
        bars2 = ax2.bar(range(len(splits)), rmse_relative_diffs, color=self.colors[1])
        ax2.set_xlabel('Split Type')
        ax2.set_ylabel('Relative RMSE Difference (%)')
        ax2.set_title('Relative RMSE Differences by Split Type')
        ax2.set_xticks(range(len(splits)))
        ax2.set_xticklabels(splits, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Poor threshold')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, rmse_relative_diffs)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5 if height > 0 else height - 0.5,
                    f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # Remove grid (explicitly)
        ax1.grid(False)
        ax2.grid(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=PLOT_SETTINGS['figure_dpi'], bbox_inches='tight')
        else:
            plt.savefig(self.metrics_dir / 'rmse_differences.png', 
                       dpi=PLOT_SETTINGS['figure_dpi'], bbox_inches='tight')
        plt.close()
    
    def plot_inner_outer_comparison(self, split_metrics: Dict, save_path: Optional[Path] = None):
        """Plot inner vs outer RMSE comparison"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data
        inner_rmses = []
        outer_rmses = []
        labels = []
        
        for split_name, metrics in split_metrics.items():
            if 'inner' in metrics and 'outer' in metrics:
                inner_rmses.append(metrics['inner']['rmse'])
                outer_rmses.append(metrics['outer']['rmse'])
                labels.append(split_name)
        
        if not inner_rmses:
            plt.close()
            return
        
        # Create scatter plot
        scatter = ax.scatter(inner_rmses, outer_rmses, s=100, alpha=0.7, c=range(len(labels)))
        
        # Add diagonal line (perfect agreement)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, label='Perfect agreement')
        
        # Add ±0.3 RMSE difference lines
        ax.plot(lims, [x + 0.3 for x in lims], 'r--', alpha=0.3, label='±0.3 RMSE')
        ax.plot([x + 0.3 for x in lims], lims, 'r--', alpha=0.3)
        
        # Labels and title
        ax.set_xlabel('Inner Set RMSE')
        ax.set_ylabel('Outer Set RMSE')
        ax.set_title('Inner vs Outer Set RMSE Comparison')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        
        # Add labels for each point
        for i, label in enumerate(labels):
            ax.annotate(label, (inner_rmses[i], outer_rmses[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Remove grid
        ax.grid(False)
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=PLOT_SETTINGS['figure_dpi'], bbox_inches='tight')
        else:
            plt.savefig(self.metrics_dir / 'inner_outer_comparison.png', 
                       dpi=PLOT_SETTINGS['figure_dpi'], bbox_inches='tight')
        plt.close()
    
    def plot_metrics_heatmap(self, split_metrics: Dict, save_path: Optional[Path] = None):
        """Create a heatmap of all metrics"""
        # Prepare data for heatmap
        metric_names = ['RMSE_inner', 'RMSE_outer', 'RMSE_diff', 'MAE_inner', 
                       'MAE_outer', 'MAE_diff', 'R2_inner', 'R2_outer', 'R2_diff']
        
        data = []
        split_names = []
        
        for split_name, metrics in split_metrics.items():
            if 'inner' in metrics and 'outer' in metrics and 'difference_evaluation' in metrics:
                row = [
                    metrics['inner']['rmse'],
                    metrics['outer']['rmse'],
                    metrics['difference_evaluation']['rmse_difference'],
                    metrics['inner']['mae'],
                    metrics['outer']['mae'],
                    metrics['difference_evaluation']['mae_difference'],
                    metrics['inner']['r2'],
                    metrics['outer']['r2'],
                    metrics['difference_evaluation']['r2_difference']
                ]
                data.append(row)
                split_names.append(split_name)
        
        if not data:
            return
        
        # Create DataFrame
        df = pd.DataFrame(data, index=split_names, columns=metric_names)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, len(split_names) * 0.5 + 2))
        
        # Normalize data for better visualization
        df_norm = df.copy()
        for col in df.columns:
            if 'diff' in col:
                # Center differences around 0
                max_abs = max(abs(df[col].min()), abs(df[col].max()))
                df_norm[col] = df[col] / max_abs if max_abs > 0 else df[col]
            else:
                # Normalize other metrics to [0, 1]
                col_min = df[col].min()
                col_max = df[col].max()
                if col_max > col_min:
                    df_norm[col] = (df[col] - col_min) / (col_max - col_min)
        
        # Create heatmap
        sns.heatmap(df_norm, annot=df.values, fmt='.3f', cmap='RdBu_r', 
                   center=0, cbar_kws={'label': 'Normalized Value'},
                   ax=ax, linewidths=0.5)
        
        ax.set_title('Metrics Comparison Across Split Types')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Split Type')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=PLOT_SETTINGS['figure_dpi'], bbox_inches='tight')
        else:
            plt.savefig(self.metrics_dir / 'metrics_heatmap.png', 
                       dpi=PLOT_SETTINGS['figure_dpi'], bbox_inches='tight')
        plt.close()
    
    def create_all_metric_plots(self, split_metrics: Dict):
        """Create all metric visualizations"""
        print("    Creating metric visualizations...")
        
        self.plot_rmse_differences(split_metrics)
        print("      - RMSE differences plot created")
        
        self.plot_inner_outer_comparison(split_metrics)
        print("      - Inner vs Outer comparison plot created")
        
        self.plot_metrics_heatmap(split_metrics)
        print("      - Metrics heatmap created")
        
        print(f"    All metric plots saved to: {self.metrics_dir}")