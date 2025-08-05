"""
QSAR Utility Functions Module

This module contains utility functions used across the QSAR analyzer.
"""
import logging
logger = logging.getLogger(__name__)
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

from .config import AD_COVERAGE_MODES


def save_results_json(data: Dict, filepath: Path):
    """Save results as JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    try: 
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            else:
                return obj
        
        serializable_data = convert_to_serializable(data)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save results to {filepath}: {e}")
        raise


def create_summary_excel(output_dir: Path, datasets: Dict, splits: Dict, 
                        ad_analysis: Dict, statistical_results: Dict,
                        ad_mode: str = 'strict'):
    """Create comprehensive summary Excel file with AD mode information"""
    excel_path = output_dir / 'summary' / f'comprehensive_summary_{ad_mode}.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Dataset Overview
        dataset_overview = []
        for name, info in datasets.items():
            dataset_overview.append({
                'Dataset': name,
                'Type': 'Test-Only' if info['is_test_only'] else 'Train/Test',
                'Original_Size': info['original_size'],
                'Analysis_Size': info['analysis_size'],
                'Sampling_Rate': info['analysis_size']/info['original_size']*100,
                'AD_Mode': ad_mode
            })
        
        pd.DataFrame(dataset_overview).to_excel(
            writer, sheet_name='Dataset_Overview', index=False
        )
        
        # Sheet 2: AD Method Performance
        ad_performance = []
        for dataset, data in ad_analysis.items():
            for split, split_data in data.items():
                if split_data is not None and 'ad_results' in split_data:
                    for method, result in split_data['ad_results'].items():
                        if result and isinstance(result, dict) and 'coverage' in result:
                            ad_performance.append({
                                'Dataset': dataset,
                                'Split': split,
                                'Method': method,
                                'Coverage': result['coverage'],
                                'Quality': result['quality'],
                                'AD_Mode': split_data.get('ad_mode', ad_mode)
                            })
        
        if ad_performance:
            pd.DataFrame(ad_performance).to_excel(
                writer, sheet_name='AD_Performance', index=False
            )
        
        # Sheet 3: AD Mode Information
        mode_info = []
        if ad_mode in AD_COVERAGE_MODES:
            mode_data = AD_COVERAGE_MODES[ad_mode]
            mode_info.append({
                'Mode': ad_mode,
                'Name': mode_data['name'],
                'Reference': mode_data['reference']
            })
            
            # Add coverage standards
            coverage_df = pd.DataFrame([
                {'Quality': k, 'Min': v[0], 'Max': v[1]} 
                for k, v in mode_data['coverage_standards'].items()
                if isinstance(v, tuple)
            ])
            coverage_df.to_excel(writer, sheet_name='AD_Coverage_Standards', index=False)
        
        pd.DataFrame(mode_info).to_excel(writer, sheet_name='AD_Mode_Info', index=False)
        
        # Sheet 4: Split Statistics
        split_stats = []
        for name, split_data in splits.items():
            for split_name, split_info in split_data['splits'].items():
                if split_info:
                    train_size = len(split_info['train_idx']) if 'train_idx' in split_info else 0
                    test_size = len(split_info['test_idx']) if 'test_idx' in split_info else 0
                    total_size = train_size + test_size
                    
                    split_stats.append({
                        'Dataset': name,
                        'Split_Method': split_name,
                        'Train_Size': train_size,
                        'Test_Size': test_size,
                        'Train_Ratio': train_size/total_size*100 if total_size > 0 else 0
                    })
        
        pd.DataFrame(split_stats).to_excel(
            writer, sheet_name='Split_Statistics', index=False
        )
        
        # Sheet 5: Statistical Summary
        stat_summary = []
        for name, stats in statistical_results.items():
            if 'basic_stats' in stats:
                basic = stats['basic_stats']
                stat_summary.append({
                    'Dataset': name,
                    'Mean': basic['mean'],
                    'Std': basic['std'],
                    'Min': basic['min'],
                    'Max': basic['max'],
                    'Q25': basic['q25'],
                    'Median': basic['median'],
                    'Q75': basic['q75'],
                    'Skewness': basic['skewness'],
                    'Kurtosis': basic['kurtosis']
                })
        
        if stat_summary:
            pd.DataFrame(stat_summary).to_excel(
                writer, sheet_name='Statistical_Summary', index=False
            )


def generate_decision_report(output_dir: Path, ad_analysis: Dict, 
                           ad_mode: str = 'strict'):
    """Generate AD decision report with mode-specific standards"""
    report_path = output_dir / 'ad_analysis' / f'DECISION_REPORT_{ad_mode}.md'
    
    # Get mode-specific standards
    if ad_mode in AD_COVERAGE_MODES:
        mode_info = AD_COVERAGE_MODES[ad_mode]
        coverage_standards = mode_info['coverage_standards']
    else:
        # Fallback to strict mode
        mode_info = AD_COVERAGE_MODES['strict']
        coverage_standards = mode_info['coverage_standards']
    
    with open(report_path, 'w') as f:
        f.write(f"# Applicability Domain Analysis Decision Report ({mode_info['name']})\n\n")
        f.write("## Executive Summary\n\n")
        
        decisions = {'recommended': [], 'caution': [], 'not_recommended': []}
        
        for name, ad_data in ad_analysis.items():
            # Calculate metrics
            all_coverages = []
            for split_data in ad_data.values():
                if split_data is not None and 'ad_results' in split_data:
                    for result in split_data['ad_results'].values():
                        if result and isinstance(result, dict) and 'coverage' in result:
                            all_coverages.append(result['coverage'])
            
            mean_coverage = np.mean(all_coverages) if all_coverages else 0
            
            # Get similarity
            min_similarity = 1.0
            for split_data in ad_data.values():
                if split_data is not None and 'similarity_results' in split_data and split_data['similarity_results']:
                    if 'combined' in split_data['similarity_results']:
                        sim = split_data['similarity_results']['combined']['combined_similarity']
                        min_similarity = min(min_similarity, sim)
                    elif 'tanimoto' in split_data['similarity_results']:
                        sim = split_data['similarity_results']['tanimoto']['stats']['mean']
                        min_similarity = min(min_similarity, sim)
            
            # Decision logic based on mode
            decision, symbol = assess_dataset_quality(
                mean_coverage, min_similarity, coverage_standards, ad_mode
            )
            
            if 'EXCELLENT' in decision or 'RECOMMENDED' in decision:
                decisions['recommended'].append(name)
            elif 'CAUTION' in decision:
                decisions['caution'].append(name)
            else:
                decisions['not_recommended'].append(name)
            
            f.write(f"### {symbol} {name}: {decision}\n")
            f.write(f"- Mean AD Coverage: {mean_coverage:.3f}\n")
            f.write(f"- Train-Test Similarity: {min_similarity:.3f}\n")
            f.write(f"- Best Split Method: {find_best_split(ad_data)}\n\n")
        
        # Summary table
        f.write("\n## Summary Table\n\n")
        f.write("| Category | Count | Datasets |\n")
        f.write("|----------|-------|----------|\n")
        f.write(f"| [CHECK] Recommended | {len(decisions['recommended'])} | {', '.join(decisions['recommended'])} |\n")
        f.write(f"| [WARNING] Caution | {len(decisions['caution'])} | {', '.join(decisions['caution'])} |\n")
        f.write(f"| [ERROR] Not Recommended | {len(decisions['not_recommended'])} | {', '.join(decisions['not_recommended'])} |\n")
        
        f.write(f"\n## {mode_info['name']} Guidelines\n\n")
        f.write(f"Reference: {mode_info['reference']}\n\n")
        
        f.write("### Coverage Standards:\n")
        for quality, bounds in coverage_standards.items():
            if isinstance(bounds, tuple):
                f.write(f"- **{quality.title()}**: {bounds[0]*100:.0f}-{bounds[1]*100:.0f}%\n")
        
        f.write("\n### Similarity Standards:\n")
        f.write("- **Excellent**: 0-20%\n")
        f.write("- **Good**: 20-40%\n")
        f.write("- **Acceptable**: 40-60%\n")
        f.write("- **Risky**: 60-75%\n")
        f.write("- **Dangerous**: >75%\n\n")
        
        # Mode-specific recommendations
        if ad_mode == 'strict':
            f.write("### For Regulatory Submission:\n")
            f.write("- Maintain AD coverage between 90-95%\n")
            f.write("- Document all predictions outside AD\n")
            f.write("- Use consensus AD methods\n")
            f.write("- Validate with external test sets\n\n")
        elif ad_mode == 'flexible':
            f.write("### For Research Applications:\n")
            f.write("- Target AD coverage of 70-80%\n")
            f.write("- Balance coverage with accuracy\n")
            f.write("- Consider ensemble approaches\n")
            f.write("- Use reliability scoring for confidence\n\n")
        else:  # adaptive
            f.write("### For Context-Specific Use:\n")
            f.write("- Adjust thresholds based on application\n")
            f.write("- Consider risk tolerance\n")
            f.write("- Document decision criteria\n")
            f.write("- Use tiered confidence levels\n\n")
        
        # Add citations
        f.write("### Key References:\n")
        for citation in mode_info.get('citations', [])[:3]:
            f.write(f"- {citation}\n")


def assess_dataset_quality(mean_coverage: float, min_similarity: float, 
                          coverage_standards: Dict, ad_mode: str) -> tuple:
    """Assess dataset quality based on mode-specific standards"""
    # Find coverage quality
    coverage_quality = "Unknown"
    for quality, bounds in coverage_standards.items():
        if isinstance(bounds, tuple) and len(bounds) == 2:
            if bounds[0] <= mean_coverage <= bounds[1]:
                coverage_quality = quality
                break
    
    # Similarity assessment (same for all modes)
    if min_similarity < 0.2:
        similarity_quality = "excellent"
    elif min_similarity < 0.4:
        similarity_quality = "good"
    elif min_similarity < 0.6:
        similarity_quality = "acceptable"
    elif min_similarity < 0.75:
        similarity_quality = "risky"
    else:
        similarity_quality = "dangerous"
    
    # Combined assessment based on mode
    if ad_mode == 'strict':
        if coverage_quality.lower() == 'excellent' and similarity_quality in ['excellent', 'good']:
            return 'EXCELLENT - READY FOR REGULATORY', '[CHECK]'
        elif coverage_quality.lower() == 'good' and similarity_quality in ['excellent', 'good']:
            return 'GOOD - RECOMMENDED', '[CHECK]'
        elif coverage_quality.lower() in ['acceptable', 'good'] and similarity_quality == 'acceptable':
            return 'USE WITH CAUTION', '[WARNING]'
        else:
            return 'NOT RECOMMENDED', '[ERROR]'
    
    elif ad_mode == 'flexible':
        if coverage_quality.lower() in ['excellent', 'good'] and similarity_quality in ['excellent', 'good']:
            return 'EXCELLENT FOR RESEARCH', '[CHECK]'
        elif coverage_quality.lower() in ['acceptable', 'good'] and similarity_quality == 'acceptable':
            return 'GOOD - RECOMMENDED', '[CHECK]'
        elif coverage_quality.lower() in ['moderate', 'acceptable']:
            return 'USE WITH CAUTION', '[WARNING]'
        else:
            return 'LIMITED USE', '[ERROR]'
    
    else:  # adaptive
        if coverage_quality.lower() in ['excellent', 'good'] and similarity_quality in ['excellent', 'good', 'acceptable']:
            return 'SUITABLE FOR APPLICATION', '[CHECK]'
        elif coverage_quality.lower() == 'acceptable':
            return 'ACCEPTABLE WITH LIMITATIONS', '[WARNING]'
        else:
            return 'NOT SUITABLE', '[ERROR]'


def find_best_split(ad_data: Dict) -> str:
    """Find best performing split method"""
    best_split = None
    best_score = 0
    
    for split_name, split_data in ad_data.items():
        if split_data is not None and 'consensus_ad' in split_data and split_data['consensus_ad']:
            if 'weighted' in split_data['consensus_ad']:
                score = split_data['consensus_ad']['weighted']['coverage']
                if score > best_score:
                    best_score = score
                    best_split = split_name
    
    return best_split if best_split else 'unknown'


def print_analysis_summary(start_time: float, datasets: Dict, 
                          memory_usage, output_dir: Path,
                          ad_mode: str = 'flexible',
                          enable_reliability: bool = False):
    """Print analysis summary with proper memory usage handling"""
    elapsed = time.time() - start_time
    
    # Get mode info
    if ad_mode in AD_COVERAGE_MODES:
        mode_info = AD_COVERAGE_MODES[ad_mode]
    else:
        mode_info = {
            'name': 'Custom',
            'reference': 'User-defined standards'
        }
    
    print("\n" + "="*60)
    print("[PARTY] ENHANCED QSAR ANALYSIS COMPLETE!")
    print("="*60)
    
    print(f"\n[CHART] DATASETS ANALYZED:")
    for name, info in datasets.items():
        print(f"  • {name}: {info['analysis_size']:,} samples", end="")
        if info['is_test_only']:
            print(" (test-only)", end="")
        print()
    
    print(f"\n⏰ Total analysis time: {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
    
    # Handle memory_usage whether it's float or list
    if memory_usage:
        if isinstance(memory_usage, (int, float)):
            # Single value case
            print(f"[SAVE] Peak memory usage: {memory_usage:.2f} MB")
        elif isinstance(memory_usage, (list, tuple)):
            # List case
            if len(memory_usage) > 0:
                max_memory = max(memory_usage)
                avg_memory = sum(memory_usage) / len(memory_usage)
                print(f"[SAVE] Peak memory usage: {max_memory:.2f} MB")
                print(f"[SAVE] Average memory usage: {avg_memory:.2f} MB")
                if len(memory_usage) > 1:
                    memory_increase = memory_usage[-1] - memory_usage[0]
                    print(f"[SAVE] Memory increase: {memory_increase:.2f} MB")
        else:
            print(f"[SAVE] Memory usage: {memory_usage}")
    
    print(f"\n[CHECK] KEY FEATURES IMPLEMENTED:")
    print(f"  • {mode_info['name']} AD standards")
    print(f"  • Reference: {mode_info['reference']}")
    print(f"  • Applicability Domain Analysis (Mode: {ad_mode})")
    print(f"  • Reliability Scoring: {'Enabled' if enable_reliability else 'Disabled'}")
    print(f"  • Ultra-strict similarity thresholds (<40% optimal)")
    print(f"  • Comprehensive AD analysis with 7 methods")
    print(f"  • Medical/pharmaceutical focused visualizations")
    print(f"  • Complete statistical analysis with individual plots")
    print(f"  • Test-only dataset handling")
    print(f"  • Memory optimization with chunking")
    
    print(f"\n[FOLDER] OUTPUT STRUCTURE:")
    print(f"  {output_dir}/")
    print(f"  ├── train/         # Training data (saved immediately)")
    print(f"  ├── test/          # Test data (saved immediately)")
    print(f"  ├── ad_analysis/   # AD analysis results & plots")
    print(f"  │   ├── [dataset]/ # Individual dataset AD results")
    print(f"  │   │   ├── comprehensive_ad_analysis_{ad_mode}.png")
    print(f"  │   │   ├── ad_results_{ad_mode}.xlsx")
    print(f"  │   │   ├── interpretation_{ad_mode}.txt")
    print(f"  │   │   └── individual_plots/")
    print(f"  │   └── overall_ad_summary_{ad_mode}.png")
    print(f"  ├── meta/          # Medical/pharmaceutical analysis")
    print(f"  │   └── [dataset]/")
    print(f"  │       ├── comprehensive_pharma_analysis.png")
    print(f"  │       └── individual_plots/")
    print(f"  ├── statistics/    # Statistical analysis")
    print(f"  │   ├── comprehensive_statistics.png")
    print(f"  │   └── individual_plots/")
    print(f"  └── summary/       # Overall summary")
    print(f"      ├── complete_analysis_summary.png")
    print(f"      ├── comprehensive_summary_{ad_mode}.xlsx")
    print(f"      └── individual_plots/")
    
    print("\n[LAB] NEXT STEPS:")
    print("  1. Review AD coverage in summary plots")
    print("  2. Check dataset quality in statistics")
    print("  3. Examine pharmaceutical properties in meta analysis")
    print("  4. Use decision matrices for dataset selection")
    
    # Add mode-specific recommendations
    if ad_mode == 'strict':
        print("\n[PIN] STRICT MODE RECOMMENDATIONS:")
        print("  • Coverage should be 90-95% (ultra-strict standards)")
        print("  • Lower values indicate extrapolation risk")
        print("  • Higher values (>95%) may indicate overfitting")
        print("  • Reference: Roy et al., Chemom. Intell. Lab. Syst. 145, 22-29 (2015)")
    elif ad_mode == 'flexible':
        print("\n[PIN] FLEXIBLE MODE RECOMMENDATIONS:")
        print("  • Coverage of 70-90% is acceptable")
        print("  • Balance between coverage and reliability")
        print("  • Good for diverse chemical spaces")
        print("  • Reference: Sahigara et al., J. Cheminform. 4, 1 (2012)")
    elif ad_mode == 'adaptive':
        print("\n[PIN] ADAPTIVE MODE RECOMMENDATIONS:")
        print("  • Context-dependent coverage targets")
        print("  • Research: 85-95%, Production: 80-90%")
        print("  • Regulatory: 90-95%")
        print("  • Reference: Hanser et al., J. Cheminform. 8, 69 (2016)")
    
    print("\n" + "="*60)


def save_analysis_summary(output_dir: Path, datasets: Dict, 
                         start_time: float, memory_usage, 
                         ad_mode: str = 'flexible',
                         enable_reliability: bool = False):
    """Save analysis summary"""
    # Get mode info
    if ad_mode in AD_COVERAGE_MODES:
        mode_info = AD_COVERAGE_MODES[ad_mode]
    else:
        mode_info = AD_COVERAGE_MODES['flexible']
    
    # Create summary
    summary_path = output_dir / 'analysis_summary.txt'
    
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("QSAR ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"AD Mode: {mode_info['name']}\n")
        f.write(f"Reliability Scoring: {'Enabled' if enable_reliability else 'Disabled'}\n\n")
        
        f.write("Datasets Analyzed:\n")
        for name, info in datasets.items():
            f.write(f"  - {name}: {info['analysis_size']:,} samples")
            if info['is_test_only']:
                f.write(" (test-only)")
            f.write("\n")
        
        elapsed = time.time() - start_time
        f.write(f"\nTotal Analysis Time: {elapsed:.2f} seconds\n")
        
        if memory_usage:
            # memory_usage가 숫자인지 리스트인지 확인
            if isinstance(memory_usage, (int, float)):
                # 단일 값인 경우
                f.write(f"Peak Memory Usage: {memory_usage:.2f} MB\n")
            elif isinstance(memory_usage, (list, tuple)):
                # 리스트인 경우
                max_memory = max(memory_usage)
                f.write(f"Peak Memory Usage: {max_memory:.2f} MB\n")
            else:
                # 기타 경우
                f.write(f"Memory Usage: {memory_usage}\n")
        
        f.write(f"\n{mode_info['name']} Standards Applied:\n")
        f.write(f"Reference: {mode_info['reference']}\n")
        
        # Coverage standards
        if ad_mode == 'adaptive':
            coverage_standards = mode_info['coverage_standards']['research']
        else:
            coverage_standards = mode_info['coverage_standards']
        
        f.write("\nCoverage Quality Thresholds:\n")
        for quality, bounds in coverage_standards.items():
            if isinstance(bounds, tuple) and len(bounds) == 2:
                min_val, max_val = bounds
                f.write(f"  - {quality.title()}: {min_val*100:.0f}% - {max_val*100:.0f}%\n")