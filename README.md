# Automated Network Optimizer (ANO) for Enhanced Prediction of Intrinsic Solubility in Drug-like Organic Compounds: A Comprehensive Machine Learning Approach

## Overview
This repository presents a novel approach to predicting aqueous solubility of drug-like organic compounds using our Automated Network Optimizer (ANO) framework. By integrating advanced machine learning techniques with automated feature selection and hyperparameter optimization, we achieve state-of-the-art prediction accuracy for intrinsic solubility (logS).

<div align="center">
    <a href="./result_prior/res1.png" target="_blank">
        <img src="./result_prior/res1.png" alt="Result 1" width="400" style="cursor: pointer;"/>
    </a>
</div>

<div align="center">
    <a href="./result_prior/res2.png" target="_blank">
        <img src="./result_prior/res2.png" alt="Result 2" width="400" style="cursor: pointer;"/>
    </a>
</div>

<div align="center">
    <a href="./result_prior/res3.png" target="_blank">
        <img src="./result_prior/res3.png" alt="Result 3" width="400" style="cursor: pointer;"/>
    </a>
</div>

## System Requirements
### Dependencies
- Python 3.12 or later
- TensorFlow 2.15.0 (Linux/MacOS/WSL)
- TensorFlow 2.15.0-GPU (Windows)
- RDKit 2024.3.1
- pandas 2.2.1
- scikit-learn 1.4.1.post1
- seaborn 0.13.2
- matplotlib 3.8.3
- optuna 3.5.0

## Repository Structure

### Jupyter Notebooks
1. **1_standard_ML.ipynb**
   - Comprehensive evaluation of traditional ML approaches
   - Random Forest, XGBoost, and SVM implementations
   - Baseline performance metrics and comparative analysis

2. **2_solubility_fingerprint_comparison.ipynb**
   - Detailed analysis of molecular fingerprint methods
   - Evaluation of ECFP, MACCS, and custom fingerprints
   - Performance comparison across fingerprint types

3. **3_ANO_with_feature_checker.ipynb**
   - Implementation of ANO framework
   - Automated feature importance analysis
   - Real-time feature selection optimization

4. **4_ANO_feature.ipynb**
   - Optimal physicochemical feature search using ANO

5. **5_ANO_structure.ipynb**
   - Hyperparameter optimization using ANO

6. **6_ANO_network_[fea_struc].ipynb**
   - Network architecture optimization based on optimal physicochemical features

7. **7_ANO_network_[struc_fea].ipynb**
   - Network architecture optimization based on optimal hyperparameters

8. **7_Solubility_final_HPO_proving.ipynb** (Bug fixing...)
   - Performance validation of final ANO model

9. **8_solubility_xai.ipynb**
   - Model explainability analysis
   - Permutation importance and SHAP evaluation
   - Correlation analysis between physicochemical features and logS
   - Implementation of Lipinski's Rule of 5


## Key Innovations
- 49 carefully selected chemical descriptors for target dataset
- Fast and efficient selections of chemical descriptors and hyperparameters in machine learning models

<div align="center">
    <img src="result_prior/descriptors_list.png" alt="Chemical Descriptors List" width="400" style="cursor: pointer;" onclick="window.open(this.src)"/>
</div>

- These are the prediction results of deep learning models using individual solubility features to verify the improvement of the basic model.

<div align="center">
    <img src="result/3_solubility_descriptor_deeplearning/r2_score_ws_individual.png" alt="Water Solubility Score" width="400" style="cursor: pointer;" onclick="window.open(this.src)"/>
</div>

<div align="center">
    <img src="result/3_solubility_descriptor_deeplearning/r2_score_de_individual.png" alt="Density Score" width="400" style="cursor: pointer;" onclick="window.open(this.src)"/>
</div>

<div align="center">
    <img src="result/3_solubility_descriptor_deeplearning/r2_score_lo_individual.png" alt="LogP Score" width="400" style="cursor: pointer;" onclick="window.open(this.src)"/>
</div>

<div align="center">
    <img src="result/3_solubility_descriptor_deeplearning/r2_score_hu_individual.png" alt="Human Score" width="400" style="cursor: pointer;" onclick="window.open(this.src)"/>
</div>


## Obstacles in the Development Process 🚧
---
### ⭐ Folder "**__revised_research**" status

> **Status notice (July 2025)**  
> 🐞 Major coding bugs and lengthy cross-validation cycles have pushed back the final fixes.  
> **New ETA:** **August 2025** (all notebooks fully integrated).

#### ✅ Completed & Reproducible
- Data search & collection 📗
- Data preprocessing 🔄
- Standard compound curation ⚙️
- Fingerprint-based solubility benchmarking 🧬
- Applied cross-validation🛰

#### 🚧 Work in Progress
- Deep-learning solubility model optimization 🧠  
- Advanced ANO + CV upgrade optimization 🚀  

Stable versions for every step are available in **`these codes`**, and live updates are tracked in **`__revised_research/`**.  
All notebooks and documentation will be committed when the remaining bugs are fixed—**target: August 2025 +**.

---

## Upgrade Highlights

- **Applicability-Domain Toolkit** 🛡️  
  • Six regulatory AD checks (Leverage, k-NN distance, Euclidean centroid, descriptor range, DModX, Std. residual)  
  • Consensus modes (Conservative / Majority / Weighted)  
  • Multithreading, λ-regularised SVD, Williams-plot auto-export

- **Data-Splitting Engine** ✂️  
  • Ten split strategies (random, scaffold, chemical-space, cluster, physchem, activity-cliff, solubility-aware, time-series, ensemble, test-only)  
  • RDKit integration, coverage metrics, JSON reports, CI-friendly folders

- **Concurrent QSAR Pipeline** 🚀  
  • Thread/process pools, CPU-core auto-detection  
  • Memory monitoring, reliability scoring, adaptive sampling for big data

- **Feature Calculation Engine** 🧬  
  • 49 classical descriptors + 2048-bit Morgan fingerprints  
  • On-disk caching, RDKit fall-backs, performance-mode down-sampling

- **Statistical Analysis Suite** 📊  
  • Basic stats, Shapiro / KS / JB / AD tests  
  • IQR, Z-score, MAD outliers; descriptor–target correlations

- **Model Performance Analyzer** ⚡  
  • Clean-data guard rails (NaN/Inf, constant-feature fixes)  
  • Sequential RF, XGBoost, LightGBM benchmarking with RMSE/MAE/R² tables & plots :contentReference

- **Visualization Suite** 🖼️  
  • High-resolution PNGs for AD coverage, meta-pharma insights, statistical & summary dashboards  
  • Memory-aware plotting, consistent pathing across modules :contentReference

> **Compatibility:** All upgraded components are drop-in; just update the import paths and rerun the pipeline.




## Version
Current Version: 7.0.00 (2025.07)

## License
This project is licensed under the MIT License - see the LICENSE file for details.