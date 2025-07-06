"""
QSAR Summary Visualization Module - COMPLETE VERSION

This module contains summary visualization functions for QSAR analysis.
Includes mode-specific summary tables and comprehensive visualizations.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
import warnings
import gc


from .visualization_utils import safe_savefig, safe_to_excel, safe_excel_writer
try:
    from ..config import AD_METHODS, PLOT_SETTINGS, AD_COVERAGE_STANDARDS, AD_COVERAGE_MODES
except ImportError:
    from config import AD_METHODS, PLOT_SETTINGS, AD_COVERAGE_STANDARDS, AD_COVERAGE_MODES

warnings.filterwarnings('ignore')


class SummaryVisualizer:
    """Handles summary visualizations"""
    
    def __init__(self, output_dir: Path):
        import matplotlib
        if matplotlib.get_backend() != 'Agg':
            matplotlib.use('Agg')
        self.output_dir = Path(output_dir)
        self.summary_path = self.output_dir / 'summary'
        self.summary_path.mkdir(parents=True, exist_ok=True)
        print(f"  Summary path created: {self.summary_path}")
        
        self.individual_path = self.summary_path / 'individual_plots'
        self.individual_path.mkdir(parents=True, exist_ok=True)
        print(f"  Individual summary path created: {self.individual_path}")
        
        # 메모리 관리를 위한 플래그
        self._low_memory_mode = False
        self._check_memory_status()
        
    def _check_memory_status(self):
        """Check available memory and set mode"""
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            if available_gb < 2:  # Less than 2GB available
                self._low_memory_mode = True
                print("  ⚠️ Low memory mode enabled for visualizations")
        except ImportError:
            pass
        
    def create_all_summary_visualizations(self, datasets: Dict, splits: Dict,
                                        ad_analysis: Dict, statistical_results: Dict):
        """Create summary visualizations"""
        print("  Creating summary visualizations...")
        
        # Main summary plot
        self._create_main_summary_plot(datasets, splits, ad_analysis)
        
        # Individual summary plots
        self.individual_path.mkdir(exist_ok=True)
        
        # Save individual plots
        self._save_individual_summary_plots(datasets, splits, ad_analysis, statistical_results)
        
        # Save comprehensive plots as individual plots
        self._save_comprehensive_individual_plots(datasets, splits, ad_analysis, statistical_results)
        
        # Create mode-specific summary tables
        self._create_mode_specific_summary_tables(datasets, ad_analysis)
        
        print("  ✓ Saved all individual plots from comprehensive view")
        
    def _create_mode_specific_summary_tables(self, datasets: Dict, ad_analysis: Dict):
        """Create summary tables for each AD mode"""
        print("  Creating mode-specific summary tables...")
        
        # Define modes and their coverage standards
        modes = ['strict', 'flexible', 'adaptive']
        
        for mode in modes:
            try:
                fig, ax = plt.subplots(figsize=(14, 10))
                self._plot_mode_summary_table(ax, datasets, ad_analysis, mode)
                plt.tight_layout()
                
                # Save the mode-specific table
                save_path = self.individual_path / f'summary_table_{mode}_mode.png'
                safe_savefig(save_path, dpi=PLOT_SETTINGS['figure_dpi'])
                
                print(f"    ✓ Created {mode} mode summary table")
                
            except Exception as e:
                print(f"    ❌ Failed to create {mode} mode summary table: {str(e)}")
                plt.close()
                gc.collect()
    
    def _plot_mode_summary_table(self, ax, datasets: Dict, ad_analysis: Dict, mode: str):
        """Create mode-specific summary table"""
        # Get mode-specific coverage standards
        if mode in AD_COVERAGE_MODES:
            if mode == 'adaptive':
                coverage_standards = AD_COVERAGE_MODES['adaptive']['coverage_standards']['research']
            else:
                coverage_standards = AD_COVERAGE_MODES[mode]['coverage_standards']
            mode_info = AD_COVERAGE_MODES[mode]
        else:
            # Fallback to strict mode
            coverage_standards = AD_COVERAGE_MODES['strict']['coverage_standards']
            mode_info = AD_COVERAGE_MODES['strict']
        
        table_data = []
        headers = ['Dataset', 'Type', 'Samples', 'Best Split', 'Mean Coverage', f'{mode.title()} Quality']
        
        for name, dataset_info in datasets.items():
            row = [name, 
                   'Test-Only' if dataset_info['is_test_only'] else 'Train/Test',
                   f"{dataset_info['analysis_size']:,}"]
            
            # Find best split and coverage
            if name in ad_analysis:
                best_split = None
                best_coverage = 0
                
                for split_name, split_data in ad_analysis[name].items():
                    coverages = [r['coverage'] for r in split_data['ad_results'].values()
                                if r and 'coverage' in r]
                    if coverages:
                        mean_cov = np.mean(coverages)
                        if mean_cov > best_coverage:
                            best_coverage = mean_cov
                            best_split = split_name
                
                if best_split:
                    # Assess quality with mode-specific standards
                    quality = self._assess_quality_by_mode(best_coverage, coverage_standards)
                    
                    row.extend([best_split, f"{best_coverage:.3f}", quality])
                else:
                    row.extend(['N/A', 'N/A', 'N/A'])
            else:
                row.extend(['N/A', 'N/A', 'N/A'])
            
            table_data.append(row)
        
        if table_data:
            table = ax.table(cellText=table_data, colLabels=headers,
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2)
            
            # Mode-specific color coding
            quality_colors = self._get_mode_quality_colors(mode)
            
            for i, row in enumerate(table_data):
                if len(row) > 5:
                    quality = row[5]
                    if quality != 'N/A' and quality in quality_colors:
                        table[(i+1, 5)].set_facecolor(quality_colors[quality])
        
        ax.axis('off')
        ax.set_title(f'Dataset Summary Table - {mode_info["name"]} Mode\n{mode_info["reference"]}', 
                    fontweight='bold', pad=20)
        
        # Add mode-specific information
        info_text = f"Coverage Standards:\n"
        for quality, bounds in coverage_standards.items():
            if isinstance(bounds, tuple) and len(bounds) == 2:
                min_val, max_val = bounds
                info_text += f"• {quality.title()}: {min_val:.1%} - {max_val:.1%}\n"
        
        ax.text(0.02, -0.1, info_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
               verticalalignment='top', fontsize=9)
    
    def _assess_quality_by_mode(self, coverage: float, coverage_standards: Dict) -> str:
        """Assess quality based on mode-specific coverage standards"""
        for quality, bounds in coverage_standards.items():
            if isinstance(bounds, tuple) and len(bounds) == 2:
                min_val, max_val = bounds
                if min_val <= coverage <= max_val:
                    return quality.title()
        return 'Unknown'
    
    def _get_mode_quality_colors(self, mode: str) -> Dict[str, str]:
        """Get quality colors based on mode"""
        if mode == 'strict':
            return {
                'Excellent': '#006400',     # dark green
                'Good': '#32CD32',          # lime green
                'Acceptable': '#FFD700',    # gold
                'Risky': '#FF8C00',         # dark orange
                'Poor': '#DC143C',          # crimson
                'Overfitted': '#8B008B'     # dark magenta
            }
        elif mode == 'flexible':
            return {
                'Excellent': '#006400',     # dark green
                'Good': '#32CD32',          # lime green
                'Acceptable': '#FFD700',    # gold
                'Moderate': '#FFA500',      # orange
                'Limited': '#FF6347',       # tomato
                'Poor': '#DC143C',          # crimson
                'Overfitted': '#8B008B'     # dark magenta
            }
        else:  # adaptive
            return {
                'Excellent': '#006400',     # dark green
                'Good': '#32CD32',          # lime green
                'Acceptable': '#FFD700',    # gold
                'Limited': '#FF6347',       # tomato
                'Poor': '#DC143C'           # crimson
            }
    

    def _create_main_summary_plot(self, datasets, splits, ad_analysis):
        """Create main summary plot with multiple panels"""
        try:
            # 메모리 최적화를 위한 설정
            n_datasets = len(datasets)
            if n_datasets > 5:
                figsize = (20, 14)
                dpi = 150
                max_subplots = 6
            else:
                figsize = (24, 16)
                dpi = 300
                max_subplots = 9
            
            # fig를 먼저 생성! (이 부분이 중요)
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Dataset overview
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_dataset_overview(ax1, datasets)
            
            # 2. Split summary
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_split_summary(ax2, splits)
            
            # 3. AD coverage summary  
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_ad_coverage_summary(ax3, ad_analysis)
            
            # 4. Sample distribution
            ax4 = fig.add_subplot(gs[1, 0])
            self._plot_sample_distribution(ax4, datasets)
            
            # 5. Feature dimension summary
            ax5 = fig.add_subplot(gs[1, 1])
            self._plot_feature_summary(ax5, splits)
            
            # 6. AD method performance
            ax6 = fig.add_subplot(gs[1, 2])
            self._plot_ad_method_performance(ax6, ad_analysis)
            
            # 7. Dataset quality metrics
            ax7 = fig.add_subplot(gs[2, 0])
            self._plot_dataset_quality(ax7, datasets, splits)
            
            # 8. Split type comparison
            ax8 = fig.add_subplot(gs[2, 1])
            self._plot_split_comparison(ax8, splits)
            
            # 9. Overall summary
            ax9 = fig.add_subplot(gs[2, 2])
            self._plot_overall_summary(ax9, datasets, splits, ad_analysis)
            
            plt.suptitle('QSAR Analysis Summary', fontsize=20, fontweight='bold')
            
            # 저장
            save_path = self.summary_path / 'comprehensive_summary.png'
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            gc.collect()
            
            print(f"    ✓ Saved comprehensive_summary.png")
            
        except Exception as e:
            print(f"    ❌ Failed to create main summary plot: {str(e)}")
            import traceback
            traceback.print_exc()
            plt.close('all')
            gc.collect()

    
    def _save_mode_specific_summary_tables(self, datasets: Dict, ad_analysis: Dict):
        """Save mode-specific tables"""
        modes = ['strict', 'flexible', 'adaptive']
        
        print("  Creating mode-specific summary tables...")
        
        for mode in modes:
            try:
                # Create summary data
                summary_data = self._create_mode_summary_data(datasets, ad_analysis, mode)
                
                if summary_data:
                    # Save as Excel
                    excel_path = self.summary_path / f'summary_table_{mode}_mode.xlsx'
                    df = pd.DataFrame(summary_data)
                    
                    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Summary', index=False)
                        
                        # Add mode info sheet
                        mode_info = pd.DataFrame({
                            'Mode': [mode],
                            'Description': [AD_COVERAGE_MODES[mode]['name']],
                            'Reference': [AD_COVERAGE_MODES[mode]['reference']]
                        })
                        mode_info.to_excel(writer, sheet_name='Mode_Info', index=False)
                    
                    print(f"    ✓ Created {mode} mode summary table (Excel)")
                
                # Also create image version
                fig, ax = plt.subplots(figsize=(14, 10))
                self._plot_mode_summary_table(ax, datasets, ad_analysis, mode)
                plt.tight_layout()
                
                save_path = self.individual_path / f'summary_table_{mode}_mode.png'
                safe_savefig(save_path, dpi=PLOT_SETTINGS['figure_dpi'])
                
            except Exception as e:
                print(f"    ❌ Failed to create {mode} mode summary table: {str(e)}")
    
    def _create_mode_summary_data(self, datasets: Dict, ad_analysis: Dict, mode: str) -> List[Dict]:
        """Create mode summary data for Excel"""
        summary_data = []
        
        # Get mode-specific coverage standards
        if mode in AD_COVERAGE_MODES:
            if mode == 'adaptive':
                coverage_standards = AD_COVERAGE_MODES['adaptive']['coverage_standards']['research']
            else:
                coverage_standards = AD_COVERAGE_MODES[mode]['coverage_standards']
        else:
            coverage_standards = AD_COVERAGE_MODES['strict']['coverage_standards']
        
        for name, dataset_info in datasets.items():
            row = {
                'Dataset': name,
                'Type': 'Test-Only' if dataset_info['is_test_only'] else 'Train/Test',
                'Samples': dataset_info['analysis_size']
            }
            
            # Find best split and coverage
            if name in ad_analysis:
                best_split = None
                best_coverage = 0
                
                for split_name, split_data in ad_analysis[name].items():
                    if 'ad_results' in split_data:
                        coverages = [r['coverage'] for r in split_data['ad_results'].values()
                                    if r and 'coverage' in r]
                        if coverages:
                            mean_cov = np.mean(coverages)
                            if mean_cov > best_coverage:
                                best_coverage = mean_cov
                                best_split = split_name
                
                if best_split:
                    quality = self._assess_quality_by_mode(best_coverage, coverage_standards)
                    
                    row.update({
                        'Best Split': best_split,
                        'Mean Coverage': best_coverage,
                        f'{mode.title()} Quality': quality
                    })
                else:
                    row.update({
                        'Best Split': 'N/A',
                        'Mean Coverage': np.nan,
                        f'{mode.title()} Quality': 'N/A'
                    })
            else:
                row.update({
                    'Best Split': 'N/A',
                    'Mean Coverage': np.nan,
                    f'{mode.title()} Quality': 'N/A'
                })
            
            summary_data.append(row)
        
        return summary_data
    
    def _plot_overall_ad_performance(self, ax, ad_analysis: Dict):
        """Plot overall AD performance across all datasets"""
        methods = list(AD_METHODS.keys())[:6]
        
        all_coverages = {method: [] for method in methods}
        
        for dataset_ad in ad_analysis.values():
            for split_data in dataset_ad.values():
                if 'ad_results' in split_data:
                    for method in methods:
                        if method in split_data['ad_results']:
                            result = split_data['ad_results'][method]
                            if result and 'coverage' in result:
                                all_coverages[method].append(result['coverage'])
        
        # Calculate statistics
        means = []
        stds = []
        method_labels = []
        
        for method in methods:
            if all_coverages[method]:
                means.append(np.mean(all_coverages[method]))
                stds.append(np.std(all_coverages[method]))
                method_labels.append(f"{method}\n(n={len(all_coverages[method])})")
        
        if means:
            x = np.arange(len(means))
            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                          color='skyblue', edgecolor='black')
            
            # Add value labels
            for bar, mean, std in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                       f'{mean:.3f}', ha='center', va='bottom')
            
            # Add ultra-strict quality zones
            ax.axhspan(0.90, 0.95, alpha=0.15, color='darkgreen', zorder=0)
            ax.axhspan(0.80, 0.90, alpha=0.15, color='green', zorder=0)
            ax.axhspan(0.70, 0.80, alpha=0.15, color='yellow', zorder=0)
            ax.axhspan(0.60, 0.70, alpha=0.15, color='orange', zorder=0)
            ax.axhspan(0.0, 0.60, alpha=0.15, color='red', zorder=0)
            ax.axhspan(0.95, 1.0, alpha=0.15, color='purple', zorder=0)
            
            ax.set_xlabel('AD Methods')
            ax.set_ylabel('Mean Coverage ± Std')
            ax.set_title('Overall AD Method Performance Across All Datasets\n(Ultra-strict standards: 90-95% optimal)')
            ax.set_xticks(x)
            ax.set_xticklabels(method_labels)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
    
    def _plot_split_method_comparison(self, ax, ad_analysis: Dict):
        """Compare different split methods"""
        split_methods = ['random', 'scaffold', 'cluster', 'time']
        
        split_coverages = {split: [] for split in split_methods}
        
        for dataset_ad in ad_analysis.values():
            for split_name, split_data in dataset_ad.items():
                if split_name in split_methods:
                    if 'ad_results' in split_data:
                        coverages = [r['coverage'] for r in split_data['ad_results'].values()
                                    if r and 'coverage' in r]
                        if coverages:
                            split_coverages[split_name].append(np.mean(coverages))
        
        # Box plot
        data_to_plot = []
        labels = []
        for split in split_methods:
            if split_coverages[split]:
                data_to_plot.append(split_coverages[split])
                labels.append(f"{split}\n(n={len(split_coverages[split])})")
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            
            ax.set_ylabel('Mean AD Coverage')
            ax.set_title('AD Coverage by Split Method')
            ax.grid(True, alpha=0.3)
            
            # Add acceptable threshold
            ax.axhline(0.7, color='red', linestyle='--', alpha=0.5, 
                      label='Acceptable threshold')
            ax.legend()
    
    def _plot_dataset_overview_summary(self, ax, datasets: Dict):
        """Plot dataset overview with Greek letters and external legend"""
        dataset_names = list(datasets.keys())
        sizes = [datasets[name]['analysis_size'] for name in dataset_names]
        is_test_only = [datasets[name]['is_test_only'] for name in dataset_names]
        
        # 유니코드 그리스 문자 사용 (더 안전한 방법)
        greek_letters = [
            '\u03B1', '\u03B2', '\u03B3', '\u03B4', '\u03B5', '\u03B6', 
            '\u03B7', '\u03B8', '\u03B9', '\u03BA', '\u03BB', '\u03BC'
        ]
        
        # # Greek letters for x-axis
        # greek_letters = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ']
        
        # 데이터셋이 12개를 초과하는 경우 처리
        if len(dataset_names) > 12:
            # 숫자 사용 또는 확장된 라벨링
            labels = [f"D{i+1}" for i in range(len(dataset_names))]
        else:
            labels = greek_letters[:len(dataset_names)]
            
        
        
        x_pos = np.arange(len(dataset_names))
        colors = ['red' if test_only else 'blue' for test_only in is_test_only]
        
        bars = ax.bar(x_pos, sizes, color=colors, alpha=0.6, edgecolor='black')
        
        # Value labels on bars
        for bar, size in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                   f'{size:,}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('Dataset Sizes', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        display_labels = labels
        ax.set_xticklabels(display_labels)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Create external legend with dataset mapping
        legend_elements = []
        
        # Dataset name mapping
        for i, (name, letter) in enumerate(zip(dataset_names, greek_letters[:len(dataset_names)])):
            color = 'red' if is_test_only[i] else 'blue'
            legend_elements.append(Patch(facecolor=color, alpha=0.6, 
                                       label=f'{letter}: {name}'))
        
        # Add category legend
        legend_elements.extend([
            Patch(facecolor='white', alpha=0, label=''),  # Separator
            Patch(facecolor='blue', alpha=0.6, label='Train/Test Dataset'),
            Patch(facecolor='red', alpha=0.6, label='Test-Only Dataset')
        ])
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Interpretation outside
        interpretation = "Dataset Size Guidelines:\n" \
                        "• >1000 samples: Ideal for QSAR\n" \
                        "• 500-1000: Acceptable\n" \
                        "• <500: Limited reliability"
        ax.text(1.05, 0.02, interpretation, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
               verticalalignment='bottom', fontsize=9)
    
    def _plot_quality_distribution_summary(self, ax, ad_analysis: Dict):
        """Plot quality distribution with improved labeling"""
        quality_counts = {'Excellent': 0, 'Good': 0, 'Acceptable': 0, 'Poor': 0, 'Overfitted': 0}
        
        for dataset_ad in ad_analysis.values():
            for split_data in dataset_ad.values():
                if 'ad_results' in split_data:
                    for result in split_data['ad_results'].values():
                        if result and 'quality' in result:
                            quality = result['quality']
                            if quality in quality_counts:
                                quality_counts[quality] += 1
        
        total = sum(quality_counts.values())
        if total > 0:
            # Define colors and create numeric labels
            quality_info = {
                'Excellent': {'color': '#006400', 'desc': '90-95%', 'symbol': 'A'},
                'Good': {'color': '#32CD32', 'desc': '80-90%', 'symbol': 'B'},
                'Acceptable': {'color': '#FFA500', 'desc': '70-80%', 'symbol': 'C'},
                'Poor': {'color': '#DC143C', 'desc': '<60%', 'symbol': 'D'},
                'Overfitted': {'color': '#8B0000', 'desc': '>95%', 'symbol': 'E'}
            }
            
            # Filter out zero counts
            labels = []
            sizes = []
            colors = []
            symbols = []
            
            for quality, count in quality_counts.items():
                if count > 0:
                    percentage = count / total * 100
                    symbols.append(quality_info[quality]['symbol'])
                    sizes.append(count)
                    colors.append(quality_info[quality]['color'])
            
            # Create pie chart without percentage labels
            wedges, texts = ax.pie(sizes, labels=[''] * len(sizes), 
                                  colors=colors,
                                  startangle=90, wedgeprops=dict(width=0.7))
            
            # Add symbols to wedges
            for i, (wedge, symbol) in enumerate(zip(wedges, symbols)):
                angle = (wedge.theta2 + wedge.theta1) / 2
                x = 0.5 * np.cos(np.deg2rad(angle))
                y = 0.5 * np.sin(np.deg2rad(angle))
                ax.text(x, y, symbol, ha='center', va='center', fontsize=16, 
                       fontweight='bold', color='white')
            
            ax.set_title('AD Quality Distribution\n(Based on coverage percentage)', 
                        fontsize=14, weight='bold')
            
            # Create legend outside
            legend_elements = []
            for quality, count in quality_counts.items():
                if count > 0:
                    info = quality_info[quality]
                    percentage = count / total * 100
                    label = f"{info['symbol']}: {quality} ({info['desc']}) - {count} ({percentage:.1f}%)"
                    legend_elements.append(Patch(facecolor=info['color'], label=label))
            
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add total at bottom
            ax.text(0.5, -0.15, f'Total Analyses: {total}', ha='center', 
                   transform=ax.transAxes, fontsize=12, fontweight='bold')
    
    def _save_individual_summary_plots(self, datasets: Dict, splits: Dict,
                                     ad_analysis: Dict, statistical_results: Dict):
        """Save individual summary plots"""
        # 1. Overall AD performance
        fig, ax = plt.subplots(figsize=(12, 8))
        self._plot_overall_ad_performance(ax, ad_analysis)
        plt.tight_layout()
        safe_savefig(self.individual_path / 'overall_ad_performance.png', 
                   dpi=PLOT_SETTINGS['figure_dpi'])
                
        # 2. Dataset comparison
        fig, ax = plt.subplots(figsize=(10, 8))
        self._plot_dataset_comparison_summary(ax, datasets, ad_analysis)
        plt.tight_layout()
        safe_savefig(self.individual_path / 'dataset_comparison.png', 
                   dpi=PLOT_SETTINGS['figure_dpi'])
        
        # 3. Method reliability
        fig, ax = plt.subplots(figsize=(10, 8))
        self._plot_method_reliability_summary(ax, ad_analysis)
        plt.tight_layout()
        safe_savefig(self.individual_path / 'method_reliability.png', 
                   dpi=PLOT_SETTINGS['figure_dpi'])
        
        # 4. Split strategy comparison
        fig, ax = plt.subplots(figsize=(10, 8))
        self._plot_split_strategy_comparison(ax, ad_analysis)
        plt.tight_layout()
        safe_savefig(self.individual_path / 'split_strategy_comparison.png', 
                   dpi=PLOT_SETTINGS['figure_dpi'])
        
        # 5. Decision matrix
        fig, ax = plt.subplots(figsize=(12, 10))
        self._plot_decision_matrix(ax, datasets, ad_analysis)
        plt.tight_layout()
        safe_savefig(self.individual_path / 'decision_matrix.png', 
                   dpi=PLOT_SETTINGS['figure_dpi'])
    
    def _plot_dataset_comparison_summary(self, ax, datasets: Dict, ad_analysis: Dict):
        """Plot comprehensive dataset comparison"""
        # Create comparison matrix
        dataset_names = list(datasets.keys())
        n_datasets = len(dataset_names)
        
        # Metrics to compare
        metrics = ['Size', 'AD Coverage', 'Quality Score']
        comparison_matrix = np.zeros((n_datasets, len(metrics)))
        
        for i, name in enumerate(dataset_names):
            # Size (normalized)
            size = datasets[name]['analysis_size']
            max_size = max(d['analysis_size'] for d in datasets.values())
            comparison_matrix[i, 0] = size / max_size
            
            # AD Coverage
            if name in ad_analysis:
                coverages = []
                for split_data in ad_analysis[name].values():
                    if 'ad_results' in split_data:
                        for result in split_data['ad_results'].values():
                            if result and 'coverage' in result:
                                coverages.append(result['coverage'])
                if coverages:
                    comparison_matrix[i, 1] = np.mean(coverages)
            
            # Quality score (composite)
            quality_score = (comparison_matrix[i, 0] + comparison_matrix[i, 1]) / 2
            comparison_matrix[i, 2] = quality_score
        
        im = ax.imshow(comparison_matrix.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Turn off grid lines that appear over the heatmap
        ax.grid(False)
        
        # Labels
        ax.set_xticks(np.arange(n_datasets))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax.set_yticklabels(metrics)
        
        # Add text annotations
        for i in range(n_datasets):
            for j in range(len(metrics)):
                text = ax.text(i, j, f'{comparison_matrix[i, j]:.2f}',
                             ha='center', va='center', color='black')
        
        ax.set_title('Dataset Comparison Matrix')
        plt.colorbar(im, ax=ax, label='Normalized Score')
    
    def _plot_method_reliability_summary(self, ax, ad_analysis: Dict):
        """Plot method reliability summary"""
        methods = list(AD_METHODS.keys())
        
        # Calculate reliability scores
        reliability_scores = {}
        
        for method in methods:
            scores = []
            
            for dataset_ad in ad_analysis.values():
                for split_data in dataset_ad.values():
                    if 'ad_results' in split_data:
                        if method in split_data['ad_results']:
                            result = split_data['ad_results'][method]
                            if result and 'coverage' in result:
                                coverage = result['coverage']
                                # Reliability score based on ultra-strict optimal range
                                if 0.9 <= coverage <= 0.95:
                                    score = 1.0
                                elif 0.8 <= coverage < 0.9:
                                    score = 0.75
                                elif 0.7 <= coverage < 0.8:
                                    score = 0.5
                                elif 0.6 <= coverage < 0.7:
                                    score = 0.25
                                else:
                                    score = 0.0
                                scores.append(score)
            
            if scores:
                reliability_scores[method] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'count': len(scores)
                }
        
        # Plot
        if reliability_scores:
            methods_sorted = sorted(reliability_scores.keys(), 
                                  key=lambda x: reliability_scores[x]['mean'], 
                                  reverse=True)
            
            means = [reliability_scores[m]['mean'] for m in methods_sorted]
            stds = [reliability_scores[m]['std'] for m in methods_sorted]
            
            x = np.arange(len(methods_sorted))
            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
            
            # Color by reliability
            for bar, mean in zip(bars, means):
                if mean >= 0.8:
                    bar.set_color('green')
                elif mean >= 0.5:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            ax.set_xlabel('AD Methods')
            ax.set_ylabel('Reliability Score')
            ax.set_title('AD Method Reliability Assessment')
            ax.set_xticks(x)
            ax.set_xticklabels(methods_sorted, rotation=45, ha='right')
            
            # Set y-axis limit with proper margin for error bars
            max_height = max([m + s for m, s in zip(means, stds)])
            ax.set_ylim(0, max(1.2, max_height + 0.2))
            ax.grid(True, alpha=0.3)
            
            # Add count labels inside bars or below
            for i, method in enumerate(methods_sorted):
                count = reliability_scores[method]['count']
                # Place label inside bar if there's space, otherwise below
                if means[i] > 0.2:
                    ax.text(i, means[i] / 2, f'n={count}', 
                           ha='center', va='center', fontsize=9, color='white', fontweight='bold')
                else:
                    ax.text(i, -0.05, f'n={count}', 
                           ha='center', va='top', fontsize=9)
    
    def _plot_split_strategy_comparison(self, ax, ad_analysis: Dict):
        """Plot comprehensive split strategy comparison"""
        split_methods = ['random', 'scaffold', 'cluster', 'time']
        
        # Collect metrics for each split method
        split_metrics = {split: {'coverage': [], 'similarity': []} 
                        for split in split_methods}
        
        for dataset_ad in ad_analysis.values():
            for split_name, split_data in dataset_ad.items():
                if split_name in split_methods:
                    # Coverage
                    if 'ad_results' in split_data:
                        coverages = [r['coverage'] for r in split_data['ad_results'].values()
                                    if r and 'coverage' in r]
                        if coverages:
                            split_metrics[split_name]['coverage'].append(np.mean(coverages))
                    
                    # Similarity
                    if 'similarity_results' in split_data and split_data['similarity_results']:
                        if 'combined' in split_data['similarity_results']:
                            sim = split_data['similarity_results']['combined']['combined_similarity']
                            split_metrics[split_name]['similarity'].append(sim)
        
        # Create comparison bars
        n_methods = len(split_methods)
        x = np.arange(n_methods)
        width = 0.35
        
        coverage_means = [np.mean(split_metrics[s]['coverage']) if split_metrics[s]['coverage'] else 0 
                         for s in split_methods]
        similarity_means = [np.mean(split_metrics[s]['similarity']) if split_metrics[s]['similarity'] else 0 
                           for s in split_methods]
        
        bars1 = ax.bar(x - width/2, coverage_means, width, label='Coverage', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, similarity_means, width, label='Similarity', alpha=0.8, color='lightcoral')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom')
        
        ax.set_xlabel('Split Method')
        ax.set_ylabel('Score')
        ax.set_title('Split Strategy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(split_methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_decision_matrix(self, ax, datasets: Dict, ad_analysis: Dict):
        """Plot decision matrix for all datasets"""
        dataset_names = list(datasets.keys())
        criteria = ['Size', 'AD Coverage', 'Similarity', 'Overall']
        
        decision_matrix = np.zeros((len(dataset_names), len(criteria)))
        
        for i, name in enumerate(dataset_names):
            # Size score
            size = datasets[name]['analysis_size']
            if size >= 1000:
                decision_matrix[i, 0] = 1.0
            elif size >= 500:
                decision_matrix[i, 0] = 0.5
            else:
                decision_matrix[i, 0] = 0.0
            
            # AD Coverage score (ultra-strict)
            if name in ad_analysis:
                coverages = []
                for split_data in ad_analysis[name].values():
                    if 'ad_results' in split_data:
                        for result in split_data['ad_results'].values():
                            if result and 'coverage' in result:
                                coverages.append(result['coverage'])
                if coverages:
                    mean_cov = np.mean(coverages)
                    if 0.9 <= mean_cov <= 0.95:
                        decision_matrix[i, 1] = 1.0
                    elif 0.8 <= mean_cov < 0.9:
                        decision_matrix[i, 1] = 0.75
                    elif 0.7 <= mean_cov < 0.8:
                        decision_matrix[i, 1] = 0.5
                    else:
                        decision_matrix[i, 1] = 0.0
            
            # Similarity score (ultra-strict)
            min_sim = 1.0
            for split_data in ad_analysis.get(name, {}).values():
                if 'similarity_results' in split_data and split_data['similarity_results']:
                    if 'combined' in split_data['similarity_results']:
                        sim = split_data['similarity_results']['combined']['combined_similarity']
                        min_sim = min(min_sim, sim)
                    elif 'tanimoto' in split_data['similarity_results']:
                        sim = split_data['similarity_results']['tanimoto']['stats']['mean']
                        min_sim = min(min_sim, sim)
            
            if min_sim < 0.2:
                decision_matrix[i, 2] = 1.0
            elif min_sim < 0.4:
                decision_matrix[i, 2] = 0.75
            elif min_sim < 0.6:
                decision_matrix[i, 2] = 0.5
            else:
                decision_matrix[i, 2] = 0.0
            
            # Overall quality
            decision_matrix[i, 3] = np.mean(decision_matrix[i, :3])
        
        # Create heatmap
        try:
            cmap = plt.cm.get_cmap('RdYlGn')
        except:
            # Fallback to basic colormap
            cmap = plt.cm.get_cmap('viridis')
            print("  ⚠️ Using fallback colormap")
        
        im = ax.imshow(decision_matrix.T, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        
        # Turn off grid lines
        ax.grid(False)
        
        # Labels
        ax.set_xticks(np.arange(len(dataset_names)))
        ax.set_yticks(np.arange(len(criteria)))
        ax.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax.set_yticklabels(criteria)
        
        # Add text annotations
        for i in range(len(dataset_names)):
            for j in range(len(criteria)):
                score = decision_matrix[i, j]
                if score >= 0.8:
                    text = '✓'
                elif score >= 0.5:
                    text = '-'
                else:
                    text = '✗'
                ax.text(i, j, text, ha='center', va='center', 
                       color='white' if score < 0.5 else 'black', fontsize=16, fontweight='bold')
        
        ax.set_title('Dataset Decision Matrix', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Score')
        
        # Add recommendation
        recommendations = []
        for i, name in enumerate(dataset_names):
            if decision_matrix[i, 3] >= 0.7:
                recommendations.append(f"{name}: ✓ Recommended")
            elif decision_matrix[i, 3] >= 0.5:
                recommendations.append(f"{name}: ~ Use with caution")
            else:
                recommendations.append(f"{name}: ✗ Not recommended")
        
        rec_text = '\n'.join(recommendations)
        ax.text(1.3, 0.5, rec_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
               verticalalignment='center', fontfamily='monospace')
    
    def _save_comprehensive_individual_plots(self, datasets: Dict, splits: Dict,
                                           ad_analysis: Dict, statistical_results: Dict):
        """Save individual plots from comprehensive summary view"""
        plots = [
            ('overall_ad_performance', 
             lambda ax: self._plot_overall_ad_performance(ax, ad_analysis)),
            ('split_method_comparison', 
             lambda ax: self._plot_split_method_comparison(ax, ad_analysis)),
            ('dataset_overview_summary', 
             lambda ax: self._plot_dataset_overview_summary(ax, datasets)),
            ('quality_distribution_summary', 
             lambda ax: self._plot_quality_distribution_summary(ax, ad_analysis)),
            ('summary_table_strict', 
             lambda ax: self._plot_mode_summary_table(ax, datasets, ad_analysis, 'strict'))
        ]
        
        for plot_name, plot_func in plots:
            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                plot_func(ax)
                plt.tight_layout()
                safe_savefig(self.individual_path / f'{plot_name}.png',
                           dpi=PLOT_SETTINGS['figure_dpi'])
                print(f"    ✓ Saved {plot_name}.png")
            except Exception as e:
                print(f"    ❌ Failed to save {plot_name}.png: {str(e)}")
                plt.close()
                gc.collect()
                
    def _plot_dataset_overview(self, ax, datasets: Dict):
        """Plot dataset overview"""
        dataset_names = list(datasets.keys())
        sizes = [datasets[name]['analysis_size'] for name in dataset_names]
        is_test_only = [datasets[name]['is_test_only'] for name in dataset_names]
        
        # Create bar plot
        x_pos = np.arange(len(dataset_names))
        colors = ['red' if test_only else 'blue' for test_only in is_test_only]
        
        bars = ax.bar(x_pos, sizes, color=colors, alpha=0.6, edgecolor='black')
        
        # Add value labels
        for bar, size in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                   f'{size:,}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Datasets')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Dataset Overview')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([name[:10] for name in dataset_names], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.6, label='Train/Test'),
            Patch(facecolor='red', alpha=0.6, label='Test-Only')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _plot_split_summary(self, ax, splits: Dict):
        """Plot split summary"""
        split_types = ['random', 'scaffold', 'cluster', 'time_series']
        split_counts = {st: 0 for st in split_types}
        
        # Count splits by type
        for dataset_splits in splits.values():
            for split_name in dataset_splits.keys():
                if split_name != 'smiles' and split_name != 'targets':
                    for st in split_types:
                        if st in split_name:
                            split_counts[st] += 1
                            break
        
        # Create bar plot
        types = list(split_counts.keys())
        counts = list(split_counts.values())
        
        bars = ax.bar(types, counts, alpha=0.8, color='skyblue', edgecolor='black')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom')
        
        ax.set_xlabel('Split Type')
        ax.set_ylabel('Count')
        ax.set_title('Split Methods Summary')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_ad_coverage_summary(self, ax, ad_analysis: Dict):
        """Plot AD coverage summary"""
        all_coverages = []
        
        for dataset_ad in ad_analysis.values():
            for split_data in dataset_ad.values():
                if 'ad_results' in split_data:
                    for result in split_data['ad_results'].values():
                        if result and 'coverage' in result:
                            all_coverages.append(result['coverage'])
        
        if not all_coverages:
            ax.text(0.5, 0.5, 'No AD coverage data', ha='center', va='center')
            ax.set_title('AD Coverage Summary')
            return
        
        # Create histogram
        ax.hist(all_coverages, bins=20, alpha=0.7, color='green', edgecolor='black')
        
        # Add statistics
        mean_cov = np.mean(all_coverages)
        median_cov = np.median(all_coverages)
        
        ax.axvline(mean_cov, color='red', linestyle='--', label=f'Mean: {mean_cov:.3f}')
        ax.axvline(median_cov, color='blue', linestyle='--', label=f'Median: {median_cov:.3f}')
        
        ax.set_xlabel('Coverage')
        ax.set_ylabel('Frequency')
        ax.set_title('AD Coverage Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_sample_distribution(self, ax, datasets: Dict):
        """Plot sample distribution"""
        # Prepare data for pie chart
        sizes = []
        labels = []
        colors = []
        
        for name, info in datasets.items():
            sizes.append(info['analysis_size'])
            labels.append(name[:15])
            colors.append('red' if info['is_test_only'] else 'blue')
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                         autopct='%1.1f%%', startangle=90)
        
        # Make percentage text smaller
        for autotext in autotexts:
            autotext.set_fontsize(8)
        
        ax.set_title('Sample Distribution Across Datasets')
    
    def _plot_feature_summary(self, ax, splits: Dict):
        """Plot feature dimension summary"""
        # Get feature dimensions if available
        feature_dims = []
        dataset_names = []
        
        for name, split_data in splits.items():
            if 'smiles' in split_data:
                # Estimate feature dimension (this is a placeholder)
                feature_dims.append(2048)  # Typical fingerprint size
                dataset_names.append(name[:10])
        
        if not feature_dims:
            ax.text(0.5, 0.5, 'No feature data available', ha='center', va='center')
            ax.set_title('Feature Dimensions')
            return
        
        # Create bar plot
        x = np.arange(len(dataset_names))
        bars = ax.bar(x, feature_dims, alpha=0.8, color='purple')
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Feature Dimensions')
        ax.set_title('Feature Space Dimensions')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_ad_method_performance(self, ax, ad_analysis: Dict):
        """Plot AD method performance summary"""
        methods = ['leverage', 'euclidean_distance', 'knn_distance', 'dmodx', 'descriptor_range']
        method_coverages = {m: [] for m in methods}
        
        # Collect coverages by method
        for dataset_ad in ad_analysis.values():
            for split_data in dataset_ad.values():
                if 'ad_results' in split_data:
                    for method in methods:
                        if method in split_data['ad_results']:
                            result = split_data['ad_results'][method]
                            if result and 'coverage' in result:
                                method_coverages[method].append(result['coverage'])
        
        # Calculate statistics
        means = []
        stds = []
        method_labels = []
        
        for method in methods:
            if method_coverages[method]:
                means.append(np.mean(method_coverages[method]))
                stds.append(np.std(method_coverages[method]))
                method_labels.append(method.replace('_', ' ').title())
        
        if not means:
            ax.text(0.5, 0.5, 'No method performance data', ha='center', va='center')
            ax.set_title('AD Method Performance')
            return
        
        # Create bar plot with error bars
        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color='orange')
        
        ax.set_xlabel('AD Method')
        ax.set_ylabel('Mean Coverage')
        ax.set_title('AD Method Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
    
    def _plot_dataset_quality(self, ax, datasets: Dict, splits: Dict):
        """Plot dataset quality metrics"""
        quality_data = []
        
        for name, info in datasets.items():
            size = info['analysis_size']
            is_test_only = info['is_test_only']
            
            # Quality scoring
            quality_score = 0
            if size >= 1000:
                quality_score += 3
            elif size >= 500:
                quality_score += 2
            else:
                quality_score += 1
                
            if not is_test_only:
                quality_score += 2
            
            quality_data.append({
                'name': name[:15],
                'score': quality_score,
                'size': size,
                'type': 'Test-Only' if is_test_only else 'Train/Test'
            })
        
        # Sort by score
        quality_data.sort(key=lambda x: x['score'], reverse=True)
        
        # Create bar plot
        names = [d['name'] for d in quality_data]
        scores = [d['score'] for d in quality_data]
        
        colors = []
        for score in scores:
            if score >= 4:
                colors.append('green')
            elif score >= 3:
                colors.append('orange')
            else:
                colors.append('red')
        
        bars = ax.bar(range(len(names)), scores, color=colors, alpha=0.8)
        
        # Add value labels
        for i, (bar, d) in enumerate(zip(bars, quality_data)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f"{d['score']}/5", ha='center', va='bottom')
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Quality Score')
        ax.set_title('Dataset Quality Assessment')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylim(0, 6)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_split_comparison(self, ax, splits: Dict):
        """Plot split comparison"""
        split_methods = ['random', 'scaffold', 'cluster', 'time_series']
        split_sizes = {method: {'train': [], 'test': []} for method in split_methods}
        
        # Collect split sizes
        for dataset_splits in splits.values():
            for split_name, split_data in dataset_splits.items():
                if isinstance(split_data, dict) and 'train_idx' in split_data:
                    train_size = len(split_data['train_idx'])
                    test_size = len(split_data['test_idx'])
                    
                    for method in split_methods:
                        if method in split_name:
                            split_sizes[method]['train'].append(train_size)
                            split_sizes[method]['test'].append(test_size)
                            break
        
        # Calculate average sizes
        avg_sizes = []
        labels = []
        
        for method in split_methods:
            if split_sizes[method]['train']:
                avg_train = np.mean(split_sizes[method]['train'])
                avg_test = np.mean(split_sizes[method]['test'])
                avg_sizes.append([avg_train, avg_test])
                labels.append(method.title())
        
        if not avg_sizes:
            ax.text(0.5, 0.5, 'No split data available', ha='center', va='center')
            ax.set_title('Split Comparison')
            return
        
        # Create grouped bar plot
        x = np.arange(len(labels))
        width = 0.35
        
        train_sizes = [s[0] for s in avg_sizes]
        test_sizes = [s[1] for s in avg_sizes]
        
        bars1 = ax.bar(x - width/2, train_sizes, width, label='Train', alpha=0.8)
        bars2 = ax.bar(x + width/2, test_sizes, width, label='Test', alpha=0.8)
        
        ax.set_xlabel('Split Method')
        ax.set_ylabel('Average Sample Count')
        ax.set_title('Average Split Sizes by Method')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_overall_summary(self, ax, datasets: Dict, splits: Dict, ad_analysis: Dict):
        """Plot overall summary"""
        # Collect overall statistics
        n_datasets = len(datasets)
        n_train_test = sum(1 for d in datasets.values() if not d['is_test_only'])
        n_test_only = sum(1 for d in datasets.values() if d['is_test_only'])
        total_samples = sum(d['analysis_size'] for d in datasets.values())
        
        # AD statistics
        all_coverages = []
        for dataset_ad in ad_analysis.values():
            for split_data in dataset_ad.values():
                if 'ad_results' in split_data:
                    for result in split_data['ad_results'].values():
                        if result and 'coverage' in result:
                            all_coverages.append(result['coverage'])
        
        # Create summary text
        summary_text = f"""QSAR ANALYSIS SUMMARY
{'='*50}

DATASETS:
• Total datasets: {n_datasets}
• Train/Test datasets: {n_train_test}
• Test-only datasets: {n_test_only}
• Total samples: {total_samples:,}

AD ANALYSIS:
• Total AD analyses: {len(all_coverages)}"""
        
        if all_coverages:
            summary_text += f"""
• Mean AD coverage: {np.mean(all_coverages):.3f}
• Coverage range: [{np.min(all_coverages):.3f}, {np.max(all_coverages):.3f}]

QUALITY ASSESSMENT:
• {'✓' if np.mean(all_coverages) > 0.7 else '✗'} Coverage > 70%
• {'✓' if total_samples > 5000 else '✗'} Sufficient data (>5000 samples)
• {'✓' if n_train_test >= 3 else '✗'} Multiple train/test datasets"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               verticalalignment='top', fontfamily='monospace', fontsize=11,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        ax.axis('off')