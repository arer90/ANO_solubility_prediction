# Automated Network Optimizer (ANO) for Enhanced Prediction of Intrinsic Solubility in Drug-like Organic Compounds: A Comprehensive Machine Learning Approach

## Overview
This repository presents a novel approach to predicting aqueous solubility of drug-like organic compounds using our Automated Network Optimizer (ANO) framework. By integrating advanced machine learning techniques with automated feature selection and hyperparameter optimization, we achieve state-of-the-art prediction accuracy for intrinsic solubility (logS).


## Obstacles in the development process.

Completed and reproducible:

0_search_comp.ipynb
1_preprocess.ipynb
2_standard_comp.ipynb
3_solubility_fp_comp.ipynb

Work in progress:
4_solubility_deeplearning.ipynb
6~7. Upgraded ANO implmentation.

For steps 4-7, please refer to the prior_research/ directory, which contains the last stable implementations. These versions reproduce the key results cited in the manuscript.

We will push updated notebooks and revised documentation as soon as the remaining bugs are resolved.


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

## Version
Current Version: 1.0.2 (2024.11)

## License
This project is licensed under the MIT License - see the LICENSE file for details.
