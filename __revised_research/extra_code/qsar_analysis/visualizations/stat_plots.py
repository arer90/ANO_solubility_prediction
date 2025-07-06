"""
QSAR Statistical Visualization Module

This module contains statistical visualization functions for QSAR analysis.
"""
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import gc
import numpy as np
import pandas as pd
import seaborn as sns
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy not available. Some statistical functions may be limited.")

from .visualization_utils import safe_savefig, safe_to_excel, safe_excel_writer
from ..config import PLOT_SETTINGS, DESCRIPTOR_NAMES

class StatisticalVisualizer:
    """Handles statistical visualizations"""
    
    def __init__(self, output_dir: Path):
        import matplotlib
        if matplotlib.get_backend() != 'Agg':
            matplotlib.use('Agg')
        self.output_dir = Path(output_dir)
        self.stats_path = self.output_dir / 'statistics'
        self.stats_path.mkdir(parents=True, exist_ok=True)
        print(f"  Statistics path created: {self.stats_path}")
        print(f"  Statistics path absolute: {self.stats_path.absolute()}") 
        
        self.individual_path = self.stats_path / 'individual_plots'
        self.individual_path.mkdir(parents=True, exist_ok=True)
        print(f"  Individual stats path created: {self.individual_path}")
        print(f"  Individual stats path absolute: {self.individual_path.absolute()}")
        
    def create_all_statistical_visualizations(self, splits: Dict, features: Dict,
                                            statistical_results: Dict):
        """Create comprehensive statistical visualizations"""
        print("  Creating statistical visualizations...")
        
        # Main statistical figure
        self._create_comprehensive_statistics_plot(splits, features, statistical_results)
        
        # Save individual statistical plots
        self._save_all_statistical_individual_plots(splits, features, statistical_results)
        
        # Save comprehensive plots as individual plots
        self._save_comprehensive_individual_plots(splits, features, statistical_results)
        print("  ✓ Saved all individual plots from comprehensive view")
            
    def _create_comprehensive_statistics_plot(self, splits: Dict, features: Dict,
                                            statistical_results: Dict):
        """Create main comprehensive statistics plot with better layout"""
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(4, 3, hspace=0.6, wspace=0.4,  # Increased hspace from 0.5 to 0.6
                             left=0.05, right=0.95, top=0.93, bottom=0.05)  # Reduced top from 0.95 to 0.93
        
        # 1. Comprehensive statistics for all datasets
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_comprehensive_statistics(ax1, splits, statistical_results)
        
        # 2. Distribution analysis - moved to second row
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_distribution_analysis(ax2, splits)
        
        # 3. Outlier analysis
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_outlier_analysis(ax3, statistical_results)
        
        # 4. Pharmaceutical relevance
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_pharma_relevance(ax4, splits)
        
        # 5. Drug development insights
        ax5 = fig.add_subplot(gs[2, 2])
        self._plot_drug_development_insights(ax5, splits)
        
        # 6. Clinical interpretation - full width bottom
        ax6 = fig.add_subplot(gs[3, :])
        self._plot_clinical_interpretation(ax6, splits, statistical_results)
        
        plt.suptitle('Comprehensive Statistical Analysis for QSAR/Drug Development', 
                    fontsize=18, fontweight='bold', y=0.98)
        safe_savefig(self.stats_path / 'comprehensive_statistics.png', 
                   dpi=PLOT_SETTINGS['figure_dpi_high'])
    
    def _plot_comprehensive_statistics(self, ax, splits: Dict, statistical_results: Dict):
        """Plot comprehensive statistics without normality tests"""
        # Create comprehensive table
        rows = []
        headers = ['Dataset', 'N', 'Mean±SEM', 'Median[IQR]', 'Range', 'CV%', 'Skewness']
        
        for name in splits:
            if name in statistical_results and 'basic_stats' in statistical_results[name]:
                s = statistical_results[name]['basic_stats']
                
                rows.append([
                    name[:15],
                    f"{s['n']}",
                    f"{s['mean']:.3f}±{s['sem']:.3f}",
                    f"{s['median']:.3f}[{s['iqr']:.3f}]",
                    f"{s['min']:.2f}-{s['max']:.2f}",
                    f"{s['cv']:.1f}",
                    f"{s['skewness']:.2f}"
                ])
        
        if rows:
            table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.8)
            
            # Color coding for CV%
            for i, row in enumerate(rows):
                cv_val = float(row[5])
                if cv_val < 30:
                    table[(i+1, 5)].set_facecolor('#90EE90')
                elif cv_val < 50:
                    table[(i+1, 5)].set_facecolor('#FFFFE0')
                else:
                    table[(i+1, 5)].set_facecolor('#FFB6C1')
        
        ax.axis('off')
        ax.set_title('Comprehensive Statistical Summary\n(Medical/Pharmaceutical Standard)', 
                    fontsize=12, fontweight='bold', pad=20)
        
        # Add interpretation below the table with adjusted position
        interpretation = "Statistical Guidelines:\n" \
                        "• CV% <30%: Low variability (excellent for QSAR)\n" \
                        "• CV% 30-50%: Moderate variability\n" \
                        "• CV% >50%: High variability (caution needed)\n" \
                        "• Skewness ≈0: Symmetric distribution"
        ax.text(0.5, -0.2, interpretation, transform=ax.transAxes,
               bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
               ha='center', fontsize=9)
    
    def _plot_distribution_analysis(self, ax, splits: Dict):
        """Plot distribution analysis with medical relevance"""
        # Combine all targets
        all_targets = []
        for name in splits:
            all_targets.extend(splits[name]['targets'])
        
        all_targets = np.array(all_targets)
        
        # Create distribution plot
        n, bins, patches = ax.hist(all_targets, bins=50, density=True, alpha=0.7, 
                                  color='skyblue', edgecolor='black')
        
        # Fit normal distribution
        from scipy.stats import norm
        mu, sigma = norm.fit(all_targets)
        x = np.linspace(all_targets.min(), all_targets.max(), 100)
        ax.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal fit')
        
        # Add reference lines
        ax.axvline(mu, color='red', linestyle='--', label=f'Mean: {mu:.3f}')
        ax.axvline(mu - 2*sigma, color='orange', linestyle='--', label='±2σ (95%)')
        ax.axvline(mu + 2*sigma, color='orange', linestyle='--')
        
        # Pharmaceutical thresholds (example: solubility)
        if -5 <= mu <= 2:  # Typical logS range
            ax.axvspan(-5, -3, alpha=0.2, color='red', label='Poor solubility')
            ax.axvspan(-3, -1, alpha=0.2, color='yellow', label='Moderate solubility')
            ax.axvspan(-1, 1, alpha=0.2, color='green', label='Good solubility')
        
        ax.set_xlabel('Target Values', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Target Distribution with Pharmaceutical Relevance', fontsize=14)
        
        # Place legend outside to avoid overlap
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.grid(True, alpha=0.3)
    
    def _plot_outlier_analysis(self, ax, statistical_results: Dict):
        """Plot outlier analysis"""
        outlier_stats = []
        
        for name in statistical_results:
            if 'outlier_stats' in statistical_results[name]:
                outliers = statistical_results[name]['outlier_stats']
                
                outlier_stats.append([
                    name[:10],
                    f"{outliers['iqr_method']['n_outliers']} ({outliers['iqr_method']['percentage']:.1f}%)",
                    f"{outliers['z_score_3sigma']['n_outliers']} ({outliers['z_score_3sigma']['percentage']:.1f}%)",
                    f"{outliers['mad_method']['n_outliers']} ({outliers['mad_method']['percentage']:.1f}%)"
                ])
        
        if outlier_stats:
            headers = ['Dataset', 'IQR Method', 'Z-Score (>3σ)', 'MAD Method']
            table = ax.table(cellText=outlier_stats, colLabels=headers,
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # Color code cells with high outlier percentages
            for i, row in enumerate(outlier_stats):
                for j in range(1, 4):
                    # Extract percentage from string
                    percentage = float(row[j].split('(')[1].split('%')[0])
                    if percentage > 10:
                        table[(i+1, j)].set_facecolor('#FFB6C1')
                    elif percentage > 5:
                        table[(i+1, j)].set_facecolor('#FFFFE0')
                    else:
                        table[(i+1, j)].set_facecolor('#90EE90')
        
        ax.axis('off')
        ax.set_title('Outlier Detection Analysis', fontsize=12, fontweight='bold')
    
    def _plot_statistical_tests_explanation(self, ax, statistical_results: Dict):
        """Explain statistical tests and show results"""
        explanation_text = """Statistical Tests Explanation:

Purpose: Compare datasets to identify significant differences

1. T-test:
   • Tests if means of two datasets differ significantly
   • Assumes normal distribution
   • p < 0.05 indicates significant difference

2. Mann-Whitney U test:
   • Non-parametric alternative to t-test
   • Does not assume normal distribution
   • Better for skewed data

3. Cohen's d (Effect Size):
   • Measures practical significance
   • Small: 0.2-0.5
   • Medium: 0.5-0.8
   • Large: >0.8

4. Levene's test:
   • Tests equality of variances
   • Important for selecting correct test
"""
        
        ax.text(0.05, 0.95, explanation_text, transform=ax.transAxes,
               verticalalignment='top', fontfamily='monospace', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        # Show actual test results if available
        if 'dataset_comparison' in statistical_results:
            comparisons = statistical_results['dataset_comparison']
            
            if comparisons:
                result_text = "\nActual Results:\n"
                for comp_name, comp_data in list(comparisons.items())[:2]:  # Show first 2
                    datasets = comp_name.replace('_vs_', ' vs ')
                    result_text += f"\n{datasets}:\n"
                    result_text += f"  T-test p-value: {comp_data['t_test']['p_value']:.4f}\n"
                    result_text += f"  Effect size: {comp_data['effect_size']['magnitude']}\n"
                    
                ax.text(0.05, 0.4, result_text, transform=ax.transAxes,
                       verticalalignment='top', fontfamily='monospace', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        ax.axis('off')
        ax.set_title('Statistical Tests Explanation', fontsize=12, fontweight='bold')
    
    def _plot_pharma_relevance(self, ax, splits: Dict):
        """Plot pharmaceutical relevance metrics"""
        # Calculate pharmaceutical metrics
        pharma_metrics = []
        
        for name in splits:
            targets = splits[name]['targets']
            
            # Categorize by pharmaceutical relevance (assuming logS values)
            highly_soluble = np.sum(targets > 0) / len(targets) * 100
            soluble = np.sum((targets >= -2) & (targets <= 0)) / len(targets) * 100
            moderate = np.sum((targets >= -4) & (targets < -2)) / len(targets) * 100
            poor = np.sum(targets < -4) / len(targets) * 100
            
            pharma_metrics.append([
                name[:10],
                f'{highly_soluble:.1f}%',
                f'{soluble:.1f}%',
                f'{moderate:.1f}%',
                f'{poor:.1f}%'
            ])
        
        headers = ['Dataset', 'Highly\nSoluble', 'Soluble', 'Moderate', 'Poor']
        table = ax.table(cellText=pharma_metrics, colLabels=headers,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Color coding
        for i, row in enumerate(pharma_metrics):
            for j in range(1, 5):
                val = float(row[j].rstrip('%'))
                if j == 1 or j == 2:  # Good categories
                    if val > 30:
                        table[(i+1, j)].set_facecolor('#90EE90')
                else:  # Poor categories
                    if val > 30:
                        table[(i+1, j)].set_facecolor('#FFB6C1')
        
        ax.axis('off')
        ax.set_title('Pharmaceutical Relevance\n(Solubility Categories)', 
                    fontsize=12, fontweight='bold')
    
    def _plot_clinical_interpretation(self, ax, splits: Dict = None, 
                                    statistical_results: Dict = None):
        """Plot clinical interpretation with actual data-based insights"""
        # Calculate actual metrics from data if available
        if splits and statistical_results:
            # Gather all targets for analysis
            all_targets = []
            for name in splits:
                all_targets.extend(splits[name]['targets'])
            all_targets = np.array(all_targets)
            
            # Calculate key pharmaceutical metrics
            mean_val = np.mean(all_targets)
            std_val = np.std(all_targets)
            cv_percent = (std_val / mean_val * 100) if mean_val != 0 else 0
            
            # Count outliers
            outlier_count = 0
            total_count = 0
            for name in statistical_results:
                if 'outlier_stats' in statistical_results[name]:
                    outliers = statistical_results[name]['outlier_stats']
                    if 'z_score_3sigma' in outliers:
                        outlier_count += outliers['z_score_3sigma']['n_outliers']
                        total_count += statistical_results[name]['basic_stats']['n']
            
            outlier_percent = (outlier_count / total_count * 100) if total_count > 0 else 0
            
            # Solubility analysis (assuming logS values)
            if -10 <= mean_val <= 2:  # Typical logS range
                highly_soluble = np.sum(all_targets > 0) / len(all_targets) * 100
                soluble = np.sum((all_targets >= -2) & (all_targets <= 0)) / len(all_targets) * 100
                moderate = np.sum((all_targets >= -4) & (all_targets < -2)) / len(all_targets) * 100
                poor = np.sum(all_targets < -4) / len(all_targets) * 100
                
                interpretation_text = f"""CLINICAL & PHARMACEUTICAL INTERPRETATION (Data-Based):

DATASET CHARACTERISTICS:
• Total Compounds: {len(all_targets):,}
• Mean ± SD: {mean_val:.3f} ± {std_val:.3f}
• CV%: {cv_percent:.1f}% {'(Excellent <30%)' if cv_percent < 30 else '(High variability)' if cv_percent > 50 else '(Moderate)'}
• Outliers: {outlier_percent:.1f}% {'(Good <5%)' if outlier_percent < 5 else '(Concerning >10%)' if outlier_percent > 10 else '(Acceptable)'}

SOLUBILITY DISTRIBUTION:
• Highly Soluble (>0): {highly_soluble:.1f}%
• Soluble (-2 to 0): {soluble:.1f}%
• Moderate (-4 to -2): {moderate:.1f}%
• Poor (<-4): {poor:.1f}%

DRUG DEVELOPMENT FEASIBILITY:
• Developable compounds: {highly_soluble + soluble:.1f}%
• Challenge compounds: {moderate + poor:.1f}%

STATISTICAL QUALITY FOR REGULATORY SUBMISSION:
• {'✓' if cv_percent < 30 else '✗'} CV% < 30% (FDA/EMA bioanalytical requirement)
• {'✓' if outlier_percent < 5 else '✗'} Outliers < 5% (data quality indicator)
• Overall Assessment: {'Ready for modeling' if cv_percent < 30 and outlier_percent < 5 else 'Needs data refinement'}

RECOMMENDATIONS:
"""
                if cv_percent > 30:
                    interpretation_text += "• Reduce variability through better experimental controls\n"
                if outlier_percent > 5:
                    interpretation_text += "• Investigate and potentially remove outliers\n"
                if poor > 30:
                    interpretation_text += "• Consider formulation strategies for poorly soluble compounds\n"
                if highly_soluble + soluble < 50:
                    interpretation_text += "• Expand chemical space toward more soluble compounds\n"
            else:
                # Generic interpretation if not logS
                interpretation_text = f"""CLINICAL & PHARMACEUTICAL INTERPRETATION (Data-Based):

DATASET CHARACTERISTICS:
• Total Compounds: {len(all_targets):,}
• Mean ± SD: {mean_val:.3f} ± {std_val:.3f}
• CV%: {cv_percent:.1f}% {'(Excellent)' if cv_percent < 30 else '(High)' if cv_percent > 50 else '(Moderate)'}
• Outliers: {outlier_percent:.1f}% {'(Good)' if outlier_percent < 5 else '(High)' if outlier_percent > 10 else '(Acceptable)'}

STATISTICAL QUALITY:
• {'✓' if cv_percent < 30 else '✗'} Low variability (CV% < 30%)
• {'✓' if outlier_percent < 5 else '✗'} Few outliers (<5%)
• {'✓' if total_count > 500 else '✗'} Sufficient sample size (>500)

QSAR MODEL SUITABILITY:
• Data consistency: {'Excellent' if cv_percent < 20 else 'Good' if cv_percent < 30 else 'Fair' if cv_percent < 50 else 'Poor'}
• Prediction reliability: {'High' if cv_percent < 30 and outlier_percent < 5 else 'Moderate' if cv_percent < 50 else 'Low'}

RECOMMENDATIONS:
"""
                if cv_percent > 30:
                    interpretation_text += "• Standardize experimental protocols\n"
                if outlier_percent > 5:
                    interpretation_text += "• Review data quality and remove errors\n"
                if total_count < 500:
                    interpretation_text += "• Increase dataset size for better coverage\n"
        
        else:
            # Fallback to generic text if no data
            interpretation_text = """CLINICAL & PHARMACEUTICAL INTERPRETATION:

1. SOLUBILITY RANGES (Typical for logS):
   • > 0: Highly soluble (optimal for oral drugs)
   • -2 to 0: Soluble (good bioavailability)
   • -4 to -2: Slightly soluble (may need formulation)
   • < -4: Poorly soluble (challenging for development)

2. STATISTICAL REQUIREMENTS:
   • FDA/EMA guidelines: CV% < 30% for bioanalytical methods
   • Normal distribution: Enables parametric statistical tests
   • Outliers < 5%: Indicates consistent data quality

3. QSAR MODEL RELIABILITY:
   • R² > 0.6: Acceptable for biological data
   • RMSE < 1 log unit: Reasonable prediction error
   • AD coverage 90-95%: Optimal applicability (ultra-strict)

4. DRUG DEVELOPMENT IMPLICATIONS:
   • High chemical diversity: Better domain coverage
   • Balanced dataset: Reduces prediction bias
   • Quality metrics guide go/no-go decisions"""
        
        ax.text(0.05, 0.95, interpretation_text, transform=ax.transAxes,
               verticalalignment='top', fontfamily='monospace', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        ax.axis('off')
        ax.set_title('Clinical & Pharmaceutical Interpretation', fontsize=12, fontweight='bold')
            
    def _plot_drug_development_insights(self, ax, splits: Dict):
        """Plot drug development insights with improved layout"""
        # Calculate key metrics
        all_targets = []
        for name in splits:
            all_targets.extend(splits[name]['targets'])
        
        all_targets = np.array(all_targets)
        
        # Categorize by solubility
        highly_soluble = np.sum(all_targets > 0)
        soluble = np.sum((all_targets >= -2) & (all_targets <= 0))
        slightly_soluble = np.sum((all_targets >= -4) & (all_targets < -2))
        poorly_soluble = np.sum(all_targets < -4)
        
        total = len(all_targets)
        
        # Create donut chart
        sizes = [highly_soluble, soluble, slightly_soluble, poorly_soluble]
        labels = ['Highly Soluble', 'Soluble', 'Slightly Soluble', 'Poorly Soluble']
        colors = ['darkgreen', 'green', 'orange', 'red']
        
        # Filter out zero values
        non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
        if non_zero:
            sizes, labels, colors = zip(*non_zero)
            
            # Calculate developable percentage
            developable = (highly_soluble + soluble) / total * 100
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                              autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total)})',
                                              startangle=90, wedgeprops=dict(width=0.6),
                                              textprops={'fontsize': 10})
            
            # Make percentage text black and bold for better visibility
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(11)
            
            # Make labels smaller to avoid overlap
            for text in texts:
                text.set_fontsize(9)
            
            # Add developable percentage to title
            ax.set_title(f'Drug Development Potential\n(Developable: {developable:.1f}%)', 
                        fontsize=12, fontweight='bold')
        
        # Add recommendation below plot
        if developable > 70:
            recommendation = "✓ High potential for drug development"
            color = "lightgreen"
        elif developable > 50:
            recommendation = "⚠ Moderate potential, formulation needed"
            color = "lightyellow"
        else:
            recommendation = "✗ Low potential, significant challenges"
            color = "lightcoral"
        
        ax.text(0.5, -0.2, recommendation, ha='center', transform=ax.transAxes,
               bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
               fontsize=10, fontweight='bold')
    
    def _plot_detailed_statistics_table(self, ax, splits: Dict, statistical_results: Dict):
        """Plot detailed statistics table"""
        all_stats = []
        
        for name in splits:
            if name in statistical_results and 'basic_stats' in statistical_results[name]:
                stats = statistical_results[name]['basic_stats']
                
                stats_row = [
                    name[:15],
                    stats['n'],
                    f"{stats['mean']:.3f}",
                    f"{stats['std']:.3f}",
                    f"{stats['median']:.3f}",
                    f"{stats['skewness']:.3f}",
                    f"{stats['kurtosis']:.3f}",
                    f"{stats['q25']:.3f}",
                    f"{stats['q75']:.3f}"
                ]
                all_stats.append(stats_row)
        
        headers = ['Dataset', 'N', 'Mean', 'Std', 'Median', 'Skew', 'Kurtosis', '25th %ile', '75th %ile']
        
        if all_stats:
            table = ax.table(cellText=all_stats, colLabels=headers,
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
        
        ax.axis('off')
        ax.set_title('Detailed Statistical Summary', fontsize=12, fontweight='bold')
    
    def _plot_correlation_summary(self, ax, features: Dict, splits: Dict, 
                                 statistical_results: Dict):
        """Plot correlation summary if available"""
        if not features:
            ax.text(0.5, 0.5, 'No correlation data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Descriptor-Target Correlations')
            ax.axis('off')
            return
        
        # Get correlation data
        all_correlations = []
        dataset_names = []
        
        for name in statistical_results:
            if 'correlation_stats' in statistical_results[name]:
                corr_stats = statistical_results[name]['correlation_stats']
                if corr_stats:
                    dataset_names.append(name)
                    all_correlations.append(corr_stats['correlations'])
        
        if all_correlations:
            # Average correlations across datasets
            mean_correlations = np.mean(all_correlations, axis=0)
            
            # Get top 10 correlations
            top_indices = np.argsort(np.abs(mean_correlations))[-10:][::-1]
            
            y_pos = np.arange(len(top_indices))
            values = [mean_correlations[i] for i in top_indices]
            colors = ['green' if v > 0 else 'red' for v in values]
            
            bars = ax.barh(y_pos, values, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([DESCRIPTOR_NAMES[i] for i in top_indices])
            ax.set_xlabel('Correlation with Target')
            ax.set_title('Top 10 Descriptor-Target Correlations')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-1, 1)
        else:
            ax.text(0.5, 0.5, 'No correlation data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    def _plot_dataset_quality_assessment(self, ax, splits: Dict, 
                                        statistical_results: Dict):
        """Plot dataset quality assessment"""
        quality_metrics = []
        
        for name in splits:
            if name in statistical_results and 'basic_stats' in statistical_results[name]:
                stats = statistical_results[name]['basic_stats']
                outliers = statistical_results[name].get('outlier_stats', {})
                
                # Calculate quality score
                cv = stats['cv']
                outlier_pct = outliers.get('iqr_method', {}).get('percentage', 0)
                n_samples = stats['n']
                
                # Quality scoring
                quality_score = 0
                if cv < 30:
                    quality_score += 3
                elif cv < 50:
                    quality_score += 2
                else:
                    quality_score += 1
                
                if outlier_pct < 5:
                    quality_score += 3
                elif outlier_pct < 10:
                    quality_score += 2
                else:
                    quality_score += 1
                
                if n_samples > 1000:
                    quality_score += 3
                elif n_samples > 500:
                    quality_score += 2
                else:
                    quality_score += 1
                
                quality_metrics.append({
                    'Dataset': name[:15],
                    'CV%': cv,
                    'Outliers%': outlier_pct,
                    'N': n_samples,
                    'Quality': quality_score,
                    'Grade': self._get_quality_grade(quality_score)
                })
        
        if quality_metrics:
            # Create bar plot
            datasets = [m['Dataset'] for m in quality_metrics]
            scores = [m['Quality'] for m in quality_metrics]
            grades = [m['Grade'] for m in quality_metrics]
            
            colors = []
            for grade in grades:
                if grade == 'Excellent':
                    colors.append('darkgreen')
                elif grade == 'Good':
                    colors.append('green')
                elif grade == 'Fair':
                    colors.append('orange')
                else:
                    colors.append('red')
            
            bars = ax.bar(range(len(datasets)), scores, color=colors, alpha=0.8)
            
            # Add value labels
            for i, (bar, score, grade) in enumerate(zip(bars, scores, grades)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{score}/9\n{grade}', ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Datasets')
            ax.set_ylabel('Quality Score')
            ax.set_title('Dataset Quality Assessment\n(Based on CV%, Outliers, Sample Size)')
            ax.set_xticks(range(len(datasets)))
            ax.set_xticklabels(datasets, rotation=45, ha='right')
            ax.set_ylim(0, 10)
            ax.grid(True, alpha=0.3)
    
    def _get_quality_grade(self, score: int) -> str:
        """Get quality grade from score"""
        if score >= 8:
            return 'Excellent'
        elif score >= 6:
            return 'Good'
        elif score >= 4:
            return 'Fair'
        else:
            return 'Poor'
    
    def _save_all_statistical_individual_plots(self, splits: Dict, features: Dict,
                                             statistical_results: Dict):
        """Save all individual statistical plots"""
        individual_path = self.individual_path
        individual_path.mkdir(exist_ok=True)
        
        # 1. Box plots for each dataset
        self._save_dataset_boxplots(splits, individual_path)
        
        # 2. Q-Q plots
        self._save_qq_plots(splits, individual_path)
        
        # 3. Violin plots
        self._save_violin_plots(splits, individual_path)
        
        # 4. Density plots
        self._save_density_plots(splits, individual_path)
        
        # 5. Cumulative distribution
        self._save_cumulative_distribution(splits, individual_path)
        
        # 6. Detailed statistics table
        self._save_detailed_statistics_table(splits, statistical_results, individual_path)
        
        # 7. Correlation summary
        self._save_correlation_summary(features, splits, statistical_results, individual_path)
        
        # 8. Dataset quality assessment
        self._save_dataset_quality_assessment(splits, statistical_results, individual_path)
    
    def _save_dataset_boxplots(self, splits: Dict, path: Path):
        """Save box plots for each dataset"""
        print(f"      Creating dataset boxplots...")
        
        fig, axes = plt.subplots(1, len(splits), figsize=(4*len(splits), 6))
        
        if len(splits) == 1:
            axes = [axes]
        
        for idx, (name, data) in enumerate(splits.items()):
            targets = data['targets']
            axes[idx].boxplot(targets, vert=True)
            axes[idx].set_title(f'{name}\nTarget Distribution')
            axes[idx].set_ylabel('Target Values')
            axes[idx].grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f"Mean: {np.mean(targets):.3f}\n" \
                       f"Median: {np.median(targets):.3f}\n" \
                       f"Std: {np.std(targets):.3f}"
            axes[idx].text(0.05, 0.95, stats_text, transform=axes[idx].transAxes,
                          bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                          verticalalignment='top')
        
        plt.tight_layout()
        save_path = path / 'dataset_boxplots.png'
        safe_savefig(save_path, dpi=PLOT_SETTINGS['figure_dpi'])

    
    def _save_qq_plots(self, splits: Dict, path: Path):
        """Save Q-Q plots for normality assessment"""
        n_datasets = len(splits)
        fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 5))
        
        if n_datasets == 1:
            axes = [axes]
        
        for idx, (name, data) in enumerate(splits.items()):
            targets = data['targets']
            stats.probplot(targets, dist="norm", plot=axes[idx])
            axes[idx].set_title(f'{name}\nQ-Q Plot')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        safe_savefig(path / 'qq_plots.png', dpi=PLOT_SETTINGS['figure_dpi'])
    
    def _save_violin_plots(self, splits: Dict, path: Path):
        """Save violin plots"""
        if len(splits) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            data_to_plot = []
            labels = []
            for name, data in splits.items():
                data_to_plot.append(data['targets'])
                labels.append(name)
            
            parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)), 
                                 showmeans=True, showextrema=True)
            
            for pc in parts['bodies']:
                pc.set_facecolor('skyblue')
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylabel('Target Values')
            ax.set_title('Target Distribution Comparison (Violin Plot)')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            safe_savefig(path / 'violin_plots.png', dpi=PLOT_SETTINGS['figure_dpi'])
    
    def _save_density_plots(self, splits: Dict, path: Path):
        """Save density plots"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for name, data in splits.items():
            targets = data['targets']
            density = stats.gaussian_kde(targets)
            x = np.linspace(targets.min(), targets.max(), 200)
            ax.plot(x, density(x), label=name, linewidth=2)
        
        ax.set_xlabel('Target Values')
        ax.set_ylabel('Density')
        ax.set_title('Target Distribution Comparison (Density Plot)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        safe_savefig(path / 'density_plots.png', dpi=PLOT_SETTINGS['figure_dpi'])
    
    def _save_cumulative_distribution(self, splits: Dict, path: Path):
        """Save cumulative distribution plots"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for name, data in splits.items():
            targets = np.sort(data['targets'])
            cumulative = np.arange(1, len(targets) + 1) / len(targets)
            ax.plot(targets, cumulative, label=name, linewidth=2)
        
        ax.set_xlabel('Target Values')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        safe_savefig(path / 'cumulative_distribution.png', dpi=PLOT_SETTINGS['figure_dpi'])
        
    def _save_detailed_statistics_table(self, splits: Dict, statistical_results: Dict, path: Path):
        """Save detailed statistics table as image"""
        fig, ax = plt.subplots(figsize=(14, 8))
        self._plot_detailed_statistics_table(ax, splits, statistical_results)
        plt.tight_layout()
        safe_savefig(path / 'detailed_statistics_table.png', dpi=PLOT_SETTINGS['figure_dpi'])
        
    def _save_correlation_summary(self, features: Dict, splits: Dict, 
                                 statistical_results: Dict, path: Path):
        """Save correlation summary plot"""
        fig, ax = plt.subplots(figsize=(10, 8))
        self._plot_correlation_summary(ax, features, splits, statistical_results)
        plt.tight_layout()
        safe_savefig(path / 'correlation_summary.png', dpi=PLOT_SETTINGS['figure_dpi'])
        
    def _save_dataset_quality_assessment(self, splits: Dict, 
                                       statistical_results: Dict, path: Path):
        """Save dataset quality assessment plot"""
        fig, ax = plt.subplots(figsize=(10, 8))
        self._plot_dataset_quality_assessment(ax, splits, statistical_results)
        plt.tight_layout()
        safe_savefig(path / 'dataset_quality_assessment.png', dpi=PLOT_SETTINGS['figure_dpi'])
        
    def _save_comprehensive_individual_plots(self, splits: Dict, features: Dict,
                                           statistical_results: Dict):
        """Save individual plots from comprehensive statistics view"""
        plots = [
            ('comprehensive_statistics', 
             lambda ax: self._plot_comprehensive_statistics(ax, splits, statistical_results)),
            ('distribution_analysis', 
             lambda ax: self._plot_distribution_analysis(ax, splits)),
            ('outlier_analysis', 
             lambda ax: self._plot_outlier_analysis(ax, statistical_results)),
            ('pharma_relevance', 
             lambda ax: self._plot_pharma_relevance(ax, splits)),
            ('drug_development_insights', 
             lambda ax: self._plot_drug_development_insights(ax, splits)),
            ('clinical_interpretation', 
             lambda ax: self._plot_clinical_interpretation(ax, splits, statistical_results))
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