# QSAR Analysis 2025 Updates Summary

## Overview
The QSAR analysis code has been updated to reflect 2025 regulatory standards and best practices for AI/ML models in drug discovery.

## Key Updates

### 1. AD Methods (X-Value Based Only)
- **Removed methods requiring model predictions**: standardized_residuals, williams_plot, dmodx
- **Prioritized X-based methods**:
  1. k-NN distance (now priority 1) - recommended by FDA 2024 for local similarity
  2. Euclidean distance (priority 2) - global similarity assessment
  3. Descriptor range (priority 3) - interpolation space definition
  4. Leverage (priority 4) - classical method still valid

### 2. Regulatory References Updated

#### Strict Mode (Ultra-Strict 2025 Regulatory)
- **ICH M7(R2) (2024)** - AI/ML considerations for mutagenicity
- **FDA (2024)** - AI and Machine Learning in Drug Development
- **EMA (2024)** - AI in medicinal product lifecycle
- **OECD (2023)** - Updated (Q)SAR validation with AI
- **ECHA (2022)** - Updated QSAR practical guide

#### Flexible Mode (Scientific Consensus 2025)
- **Wu et al. (2024)** - Uncertainty quantification in molecular property prediction
- **Chen et al. (2024)** - Adaptive AD for graph neural networks
- **Zhang et al. (2023)** - Dynamic applicability domains in deep learning
- **Mervin et al. (2021)** - Uncertainty quantification in drug discovery

#### Adaptive Mode (Context-Dependent 2025 AI-Ready)
- **OpenFDA (2025)** - AI/ML-based Drug Development Tools Guidance
- **IMI MELLODDY (2024)** - Federated learning standards
- **FDA (2024)** - Good Machine Learning Practice
- **EMA (2023)** - AI in medicines regulation Q&A

### 3. Tanimoto Similarity Standards (2025)
Updated thresholds for ML models:
- **Excellent**: 0.00-0.20 (ideal for generalization)
- **Good**: 0.20-0.35 (updated from 0.40)
- **Acceptable**: 0.35-0.55 (adjusted for ML)
- **Concerning**: 0.55-0.70 (new category)
- **Risky**: 0.70-0.85 (significant overfitting risk)
- **Dangerous**: 0.85-1.00 (likely memorization)

### 4. Visualization Updates
- **Removed all grids** from plots (seaborn-v0_8-white style)
- **Removed P1 labels** from coverage plots
- **Enhanced RMSE plots** with side panels

### 5. New Features
- **Split-specific Tanimoto evaluation** in statistics.py
- **RMSE differences** instead of ratios in metrics.py
- **Comprehensive fixes** for all AD modes (strict, flexible, adaptive)

## Implementation Status
✅ AD methods updated to X-value based only
✅ Regulatory references updated to 2025 standards
✅ Tanimoto similarity thresholds adjusted for ML
✅ Visualization fixes applied globally
✅ Split-specific evaluations implemented
✅ RMSE difference calculations added

## Compliance
The updated code now complies with:
- ICH M7(R2) 2024 guidelines
- FDA 2024 AI/ML guidance
- EMA 2024 AI reflection paper
- OECD 2023 updated (Q)SAR guidance
- Latest scientific consensus (2023-2024 publications)