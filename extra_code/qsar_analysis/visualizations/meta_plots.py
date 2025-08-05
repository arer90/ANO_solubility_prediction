"""
QSAR Meta/Pharmaceutical Visualization Module - COMPLETE VERSION

This module contains visualization functions focused on pharmaceutical and drug development insights.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from typing import Dict, List, Optional
from math import pi

import gc
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from .visualization_utils import safe_savefig, safe_to_excel, safe_excel_writer
from ..base import RDKIT_AVAILABLE
from ..config import PLOT_SETTINGS, DESCRIPTOR_NAMES


if RDKIT_AVAILABLE:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold


class MetaVisualizer:
    """Handles pharmaceutical and drug development focused visualizations"""
    
    def __init__(self, output_dir: Path):
        import matplotlib
        if matplotlib.get_backend() != 'Agg':
            matplotlib.use('Agg')
        self.output_dir = Path(output_dir)
        self.meta_path = self.output_dir / 'meta'
        self.meta_path.mkdir(parents=True, exist_ok=True)
        print(f"  Meta path created: {self.meta_path}")
        print(f"  Meta path absolute: {self.meta_path.absolute()}")
        
        self.descriptor_names = DESCRIPTOR_NAMES
        
    def create_all_meta_visualizations(self, datasets: Dict, splits: Dict, 
                                     features: Dict, ad_analysis: Dict):
        """Create all meta visualizations with pharmaceutical focus"""
        print("  Creating meta visualizations (medical/pharmaceutical focus)...")
        
        for name, dataset_info in datasets.items():
            print(f"    Creating meta analysis for {name}...")
            meta_path = self.meta_path / name
            meta_path.mkdir(parents=True, exist_ok=True)
            print(f"      Dataset meta path: {meta_path}")
            
            # Create comprehensive analysis
            self._create_comprehensive_pharma_analysis(
                name, dataset_info, splits, features, ad_analysis, meta_path
            )
            
            # Save individual meta plots
            self._save_meta_individual_plots(name, splits, features, meta_path)
            
            # Save comprehensive plots as individual plots
            self._save_comprehensive_individual_plots(name, dataset_info, splits, features, meta_path)
            print(f"      [OK] Saved all individual plots from comprehensive view")
    
    def _create_comprehensive_pharma_analysis(self, name: str, dataset_info: Dict,
                                           splits: Dict, features: Dict, 
                                           ad_analysis: Dict, meta_path: Path):
        """Create comprehensive pharmaceutical analysis plot"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4,
                             left=0.06, right=0.94, top=0.94, bottom=0.06)
        
        # 1. Chemical space analysis
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_chemical_space(ax1, name, features, splits)
        
        # 2. Descriptor importance
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_descriptor_importance(ax2, name, features, splits)
        
        # 3. Target distribution
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_target_distribution(ax3, name, splits)
        
        # 4. Lipinski's Rule of Five
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_lipinski_analysis(ax4, name, features)
        
        # 5. Scaffold analysis
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_scaffold_analysis(ax5, name, splits)
        
        # 6. Drug-likeness assessment
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_drug_likeness(ax6, name, features)
        
        # 7. Feature correlations
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_feature_correlations(ax7, name, features)
        
        # 8. Pharmaceutical property radar
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_pharma_property_radar(ax8, name, features)
        
        # 9. Comprehensive summary
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_meta_summary(ax9, name, dataset_info, features, splits)
        
        plt.suptitle(f'{name}: Comprehensive Chemical & Pharmaceutical Analysis', 
                    fontsize=16, fontweight='bold')
        safe_savefig(meta_path / 'comprehensive_pharma_analysis.png', 
                   dpi=PLOT_SETTINGS['figure_dpi_high'])
    
    def _plot_chemical_space(self, ax, name: str, features: Dict, splits: Dict):
        """Plot chemical space using PCA"""
        if name not in features:
            ax.text(0.5, 0.5, 'No feature data', ha='center', va='center')
            ax.set_title('Chemical Space')
            return
        
        feature_data = features[name]['features']
        targets = splits[name]['targets']
        
        # PCA on features
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(feature_data[:min(1000, len(feature_data))])
        
        # Scatter plot colored by target
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=targets[:len(X_pca)], 
                            cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('Chemical Space (PCA)')
        plt.colorbar(scatter, ax=ax, label='Target Value')
    
    def _plot_descriptor_importance(self, ax, name: str, features: Dict, splits: Dict):
        """Plot descriptor importance with names"""
        if name not in features:
            ax.text(0.5, 0.5, 'No descriptor data', ha='center', va='center')
            ax.set_title('Descriptor Importance')
            return
        
        # Calculate feature importance (correlation with target)
        descriptors = features[name]['descriptors']
        targets = splits[name]['targets']
        
        importances = []
        for i in range(min(49, descriptors.shape[1])):
            corr = np.corrcoef(descriptors[:, i], targets)[0, 1]
            importances.append(abs(corr))
        
        # Sort by importance
        sorted_idx = np.argsort(importances)[-10:]  # Top 10
        
        # Plot
        y_pos = np.arange(len(sorted_idx))
        ax.barh(y_pos, [importances[i] for i in sorted_idx])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.descriptor_names[i] for i in sorted_idx])
        ax.set_xlabel('Absolute Correlation with Target')
        ax.set_title('Top 10 Important Descriptors')
        ax.grid(True, alpha=0.3)
    
    def _plot_target_distribution(self, ax, name: str, splits: Dict):
        """Plot target distribution with statistics"""
        if name not in splits:
            ax.text(0.5, 0.5, 'No target data', ha='center', va='center')
            ax.set_title('Target Distribution')
            return
        
        targets = splits[name]['targets']
        ax.hist(targets, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Target Values')
        ax.set_ylabel('Frequency')
        ax.set_title('Target Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        from scipy import stats
        stats_text = f"Mean: {np.mean(targets):.3f}\n" \
                   f"Std: {np.std(targets):.3f}\n" \
                   f"Range: [{np.min(targets):.3f}, {np.max(targets):.3f}]\n" \
                   f"Skewness: {stats.skew(targets):.3f}"
        ax.text(0.7, 0.95, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
               verticalalignment='top')
    
    def _plot_lipinski_analysis(self, ax, name: str, features: Dict):
        """Analyze Lipinski's Rule of Five"""
        if name not in features:
            ax.text(0.5, 0.5, 'No descriptor data', ha='center', va='center')
            ax.set_title("Lipinski's Rule of Five")
            return
        
        descriptors = features[name]['descriptors']
        
        # Extract relevant descriptors (indices based on descriptor list)
        mw = descriptors[:, 0]  # MolWt
        logp = descriptors[:, 1]  # LogP
        hbd = descriptors[:, 5]  # NumHDonors
        hba = descriptors[:, 4]  # NumHAcceptors
        
        # Calculate violations
        violations = (
            (mw > 500).astype(int) +
            (logp > 5).astype(int) +
            (hbd > 5).astype(int) +
            (hba > 10).astype(int)
        )
        
        # Plot distribution
        unique, counts = np.unique(violations, return_counts=True)
        colors = ['green', 'yellow', 'orange', 'red', 'darkred']
        
        bars = ax.bar(unique, counts, color=[colors[i] for i in unique])
        ax.set_xlabel('Number of Violations')
        ax.set_ylabel('Count')
        ax.set_title("Lipinski's Rule of Five Analysis")
        ax.set_xticks(unique)
        
        # Set y-axis limit to prevent overlap
        max_count = max(counts) if len(counts) > 0 else 1
        ax.set_ylim(0, max_count * 1.2)
        
        # Add percentage labels
        total = len(violations)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_count * 0.01,
                   f'{count}\n({count/total*100:.1f}%)', ha='center', va='bottom')
        
        # Add interpretation
        drug_like = np.sum(violations <= 1) / total * 100
        ax.text(0.98, 0.98, f'Drug-like: {drug_like:.1f}%\n(≤1 violation)',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle="round", facecolor="lightgreen" if drug_like > 80 else "lightyellow"))
    
    def _plot_scaffold_analysis(self, ax, name: str, splits: Dict):
        """Plot scaffold diversity analysis"""
        if not RDKIT_AVAILABLE or name not in splits:
            ax.text(0.5, 0.5, 'Scaffold analysis not available', ha='center', va='center')
            ax.set_title('Scaffold Analysis')
            return
        
        smiles = splits[name]['smiles'][:1000]  # Limit for performance
        
        # Calculate scaffolds
        scaffolds = {}
        for smi in smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                    scaffolds[scaffold] = scaffolds.get(scaffold, 0) + 1
            except:
                continue
        
        # Get top scaffolds
        top_scaffolds = sorted(scaffolds.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if top_scaffolds:
            scaffold_names = [f"Scaffold {i+1}" for i in range(len(top_scaffolds))]
            counts = [count for _, count in top_scaffolds]
            
            bars = ax.bar(scaffold_names, counts, alpha=0.8, color='skyblue')
            ax.set_xlabel('Scaffold')
            ax.set_ylabel('Count')
            ax.set_title('Top 10 Scaffolds')
            ax.tick_params(axis='x', rotation=45)
            
            # Add diversity metric
            total_mols = len(smiles)
            unique_scaffolds = len(scaffolds)
            diversity = unique_scaffolds / total_mols
            
            ax.text(0.98, 0.98, f'Scaffold Diversity: {diversity:.3f}\n({unique_scaffolds} unique / {total_mols} total)',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle="round", facecolor="lightgreen" if diversity > 0.5 else "lightyellow"))
    
    def _plot_drug_likeness(self, ax, name: str, features: Dict):
        """Plot drug-likeness assessment"""
        if name not in features:
            ax.text(0.5, 0.5, 'No descriptor data', ha='center', va='center')
            ax.set_title('Drug-likeness Assessment')
            return
        
        descriptors = features[name]['descriptors']
        
        # Calculate multiple drug-likeness criteria
        criteria = {
            "Lipinski's Ro5": 0,
            "Veber's Rules": 0,
            "Ghose Filter": 0,
            "Egan Filter": 0,
            "Muegge Filter": 0
        }
        
        n_mols = len(descriptors)
        
        # Lipinski's Rule of Five
        mw = descriptors[:, 0]
        logp = descriptors[:, 1]
        hbd = descriptors[:, 5]
        hba = descriptors[:, 4]
        criteria["Lipinski's Ro5"] = np.sum(
            (mw <= 500) & (logp <= 5) & (hbd <= 5) & (hba <= 10)
        ) / n_mols * 100
        
        # Veber's Rules
        rotatable = descriptors[:, 7]
        tpsa = descriptors[:, 13]
        criteria["Veber's Rules"] = np.sum(
            (rotatable <= 10) & (tpsa <= 140)
        ) / n_mols * 100
        
        # Simplified versions of other filters
        criteria["Ghose Filter"] = np.sum(
            (mw >= 160) & (mw <= 480) & (logp >= -0.4) & (logp <= 5.6)
        ) / n_mols * 100
        
        criteria["Egan Filter"] = np.sum(
            (tpsa <= 132) & (logp <= 5.88)
        ) / n_mols * 100
        
        criteria["Muegge Filter"] = np.sum(
            (mw >= 200) & (mw <= 600) & (logp >= -2) & (logp <= 5)
        ) / n_mols * 100
        
        # Plot
        names = list(criteria.keys())
        values = list(criteria.values())
        colors = ['green' if v > 80 else 'orange' if v > 60 else 'red' for v in values]
        
        bars = ax.bar(names, values, color=colors, alpha=0.8)
        ax.set_ylabel('Percentage Passing (%)')
        ax.set_title('Drug-likeness Assessment')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 110)  # Extended to 110 to prevent overlap
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Add horizontal lines
        ax.axhline(80, color='green', linestyle='--', alpha=0.5, label='Good (>80%)')
        ax.axhline(60, color='orange', linestyle='--', alpha=0.5, label='Acceptable (>60%)')
        
        # Place legend outside the plot area
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    def _plot_feature_correlations(self, ax, name: str, features: Dict):
        """Plot feature correlation heatmap"""
        if name not in features:
            ax.text(0.5, 0.5, 'No feature data', ha='center', va='center')
            ax.set_title('Feature Correlations')
            return
        
        descriptors = features[name]['descriptors']
        
        # Select top 15 descriptors by variance
        variances = np.var(descriptors, axis=0)
        top_indices = np.argsort(variances)[-15:]
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(descriptors[:, top_indices].T)
        
        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Labels
        labels = [self.descriptor_names[i] for i in top_indices]
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, ha='right')
        ax.set_yticklabels(labels)
        
        ax.set_title('Feature Correlations (Top 15 by Variance)')
        plt.colorbar(im, ax=ax)
        
        # Add grid
        ax.set_xticks(np.arange(len(labels)+1)-.5, minor=True)
        ax.set_yticks(np.arange(len(labels)+1)-.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    
    def _plot_pharma_property_radar(self, ax, name: str, features: Dict):
        """Create radar plot of pharmaceutical properties"""
        if name not in features:
            ax.text(0.5, 0.5, 'No descriptor data', ha='center', va='center')
            ax.set_title('Pharmaceutical Properties')
            return
        
        descriptors = features[name]['descriptors']
        
        # Calculate normalized scores (0-1)
        properties = {
            'Lipophilicity': np.clip((5 - np.mean(descriptors[:, 1])) / 5, 0, 1),  # LogP
            'Size': np.clip((500 - np.mean(descriptors[:, 0])) / 500, 0, 1),  # MW
            'Polarity': np.clip(np.mean(descriptors[:, 13]) / 140, 0, 1),  # TPSA
            'Flexibility': np.clip((10 - np.mean(descriptors[:, 7])) / 10, 0, 1),  # RotBonds
            'Complexity': np.clip((1000 - np.mean(descriptors[:, 16])) / 1000, 0, 1),  # BertzCT
            'Aromaticity': np.clip(np.mean(descriptors[:, 9]) / 5, 0, 1)  # NumAromaticRings
        }
        
        # Radar plot
        categories = list(properties.keys())
        values = list(properties.values())
        
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        values += values[:1]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Pharmaceutical Property Profile')
        ax.grid(True)
    
    def _plot_meta_summary(self, ax, name: str, dataset_info: Dict, 
                          features: Dict, splits: Dict):
        """Plot meta analysis summary"""
        summary_text = f"DATASET ANALYSIS SUMMARY\n"
        summary_text += f"{'='*50}\n\n"
        summary_text += f"Dataset: {name}\n"
        summary_text += f"Type: {'Test-Only' if dataset_info['is_test_only'] else 'Train/Test'}\n"
        summary_text += f"Original Size: {dataset_info['original_size']:,}\n"
        summary_text += f"Analysis Size: {dataset_info['analysis_size']:,}\n"
        
        if dataset_info['original_size'] != dataset_info['analysis_size']:
            summary_text += f"Sampling: {dataset_info['analysis_size']/dataset_info['original_size']*100:.1f}%\n"
        
        # Add key findings
        summary_text += f"\nKEY FINDINGS:\n"
        
        if name in features:
            descriptors = features[name]['descriptors']
            
            # Drug-likeness
            mw = descriptors[:, 0]
            logp = descriptors[:, 1]
            hbd = descriptors[:, 5]
            hba = descriptors[:, 4]
            
            ro5_compliant = np.sum(
                (mw <= 500) & (logp <= 5) & (hbd <= 5) & (hba <= 10)
            ) / len(descriptors) * 100
            
            summary_text += f"• Lipinski Ro5 compliant: {ro5_compliant:.1f}%\n"
            
            # Chemical diversity
            if 'fingerprints' in features[name]:
                fps = features[name]['fingerprints']
                diversity = np.mean(np.std(fps, axis=0))
                summary_text += f"• Chemical diversity score: {diversity:.3f}\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               verticalalignment='top', fontfamily='monospace', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9))
        ax.axis('off')
    
    def _save_meta_individual_plots(self, name: str, splits: Dict, 
                                   features: Dict, meta_path: Path):
        """Save individual meta analysis plots"""
        individual_path = meta_path / "individual_plots"
        individual_path.mkdir(exist_ok=True)
        
        # Create ADMET property distribution plot
        if name in features:
            self._create_admet_plots(name, features, individual_path)
        
        # Create drug development stage assessment
        if name in features:
            self._create_development_stage_plot(name, features, individual_path)
        
        # Create bioavailability prediction
        if name in features:
            self._create_bioavailability_plot(name, features, individual_path)
    
    def _create_admet_plots(self, name: str, features: Dict, path: Path):
        """Create ADMET property distribution plots"""
        descriptors = features[name]['descriptors']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Absorption (LogP vs MW)
        axes[0,0].scatter(descriptors[:, 0], descriptors[:, 1], alpha=0.5)
        axes[0,0].set_xlabel('Molecular Weight')
        axes[0,0].set_ylabel('LogP')
        axes[0,0].set_title('Absorption Properties')
        axes[0,0].axvline(500, color='red', linestyle='--', alpha=0.5)
        axes[0,0].axhline(5, color='red', linestyle='--', alpha=0.5)
        
        # Distribution (LogP vs TPSA)
        axes[0,1].scatter(descriptors[:, 1], descriptors[:, 13], alpha=0.5)
        axes[0,1].set_xlabel('LogP')
        axes[0,1].set_ylabel('TPSA')
        axes[0,1].set_title('Distribution Properties')
        axes[0,1].axvline(5, color='red', linestyle='--', alpha=0.5)
        axes[0,1].axhline(140, color='red', linestyle='--', alpha=0.5)
        
        # Metabolism (RotBonds vs MW)
        axes[1,0].scatter(descriptors[:, 0], descriptors[:, 7], alpha=0.5)
        axes[1,0].set_xlabel('Molecular Weight')
        axes[1,0].set_ylabel('Rotatable Bonds')
        axes[1,0].set_title('Metabolism Properties')
        axes[1,0].axvline(500, color='red', linestyle='--', alpha=0.5)
        axes[1,0].axhline(10, color='red', linestyle='--', alpha=0.5)
        
        # Excretion (HBD vs HBA)
        axes[1,1].scatter(descriptors[:, 5], descriptors[:, 4], alpha=0.5)
        axes[1,1].set_xlabel('H-Bond Donors')
        axes[1,1].set_ylabel('H-Bond Acceptors')
        axes[1,1].set_title('Excretion Properties')
        axes[1,1].axvline(5, color='red', linestyle='--', alpha=0.5)
        axes[1,1].axhline(10, color='red', linestyle='--', alpha=0.5)
        
        plt.suptitle('ADMET Property Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        safe_savefig(path / 'admet_properties.png', dpi=PLOT_SETTINGS['figure_dpi'])
    
    def _create_development_stage_plot(self, name: str, features: Dict, path: Path):
        """Create drug development stage assessment plot"""
        descriptors = features[name]['descriptors']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Categorize compounds by development potential
        lead_like = np.sum(
            (descriptors[:, 0] <= 350) & 
            (descriptors[:, 1] <= 3.5) & 
            (descriptors[:, 7] <= 7)
        )
        drug_like = np.sum(
            (descriptors[:, 0] > 350) & (descriptors[:, 0] <= 500) &
            (descriptors[:, 1] <= 5) & 
            (descriptors[:, 5] <= 5) & 
            (descriptors[:, 4] <= 10)
        )
        beyond_ro5 = len(descriptors) - lead_like - drug_like
        
        categories = ['Lead-like', 'Drug-like', 'Beyond Rule of 5']
        counts = [lead_like, drug_like, beyond_ro5]
        colors = ['green', 'orange', 'red']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.8)
        ax.set_ylabel('Number of Compounds')
        ax.set_title('Drug Development Stage Assessment')
        
        # Add percentage labels
        total = len(descriptors)
        for bar, count in zip(bars, counts):
            percentage = count / total * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        plt.tight_layout()
        safe_savefig(path / 'development_stage_assessment.png', 
                   dpi=PLOT_SETTINGS['figure_dpi'])
    
    def _create_bioavailability_plot(self, name: str, features: Dict, path: Path):
        """Create bioavailability score prediction plot"""
        descriptors = features[name]['descriptors']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Simple bioavailability score based on multiple criteria
        bioavailability_scores = []
        for i in range(len(descriptors)):
            score = 0
            if descriptors[i, 0] <= 500: score += 1  # MW
            if descriptors[i, 1] <= 5: score += 1    # LogP
            if descriptors[i, 5] <= 5: score += 1    # HBD
            if descriptors[i, 4] <= 10: score += 1   # HBA
            if descriptors[i, 7] <= 10: score += 1   # RotBonds
            if descriptors[i, 13] <= 140: score += 1 # TPSA
            bioavailability_scores.append(score / 6 * 100)
        
        ax.hist(bioavailability_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Bioavailability Score (%)')
        ax.set_ylabel('Number of Compounds')
        ax.set_title('Predicted Oral Bioavailability Distribution')
        ax.axvline(np.mean(bioavailability_scores), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(bioavailability_scores):.1f}%')
        ax.legend()
        
        plt.tight_layout()
        safe_savefig(path / 'bioavailability_prediction.png', 
                   dpi=PLOT_SETTINGS['figure_dpi'])
    
    def _save_comprehensive_individual_plots(self, name: str, dataset_info: Dict,
                                            splits: Dict, features: Dict, meta_path: Path):
        """Save individual plots from comprehensive pharma analysis"""
        individual_path = meta_path / "individual_plots"
        individual_path.mkdir(parents=True, exist_ok=True)
        
        print(f"      Saving individual plots to: {individual_path}")
        
        # Define all plots to save
        plots = [
            ('chemical_space', lambda ax: self._plot_chemical_space(ax, name, features, splits)),
            ('descriptor_importance', lambda ax: self._plot_descriptor_importance(ax, name, features, splits)),
            ('target_distribution', lambda ax: self._plot_target_distribution(ax, name, splits)),
            ('lipinski_analysis', lambda ax: self._plot_lipinski_analysis(ax, name, features)),
            ('scaffold_analysis', lambda ax: self._plot_scaffold_analysis(ax, name, splits)),
            ('drug_likeness', lambda ax: self._plot_drug_likeness(ax, name, features)),
            ('feature_correlations', lambda ax: self._plot_feature_correlations(ax, name, features)),
            ('pharma_property_radar', lambda ax: self._plot_pharma_property_radar(ax, name, features)),
            ('meta_summary', lambda ax: self._plot_meta_summary(ax, name, dataset_info, features, splits))
        ]
        
        saved_count = 0
        for plot_name, plot_func in plots:
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                plot_func(ax)
                plt.tight_layout()
                
                save_path = individual_path / f'{plot_name}.png'
                if safe_savefig(save_path, dpi=PLOT_SETTINGS['figure_dpi']):
                    saved_count += 1
            except Exception as e:
                print(f"        [ERROR] Failed to create {plot_name}.png: {str(e)}")
                plt.close()
                gc.collect()
        
        print(f"      Saved {saved_count}/{len(plots)} individual plots")
