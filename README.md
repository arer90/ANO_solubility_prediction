# Automated Network Optimizer (ANO) for Enhanced Prediction of Intrinsic Solubility in Drug-like Organic Compounds: A Comprehensive Machine Learning Approach

## Overview

ANO is a comprehensive framework for molecular solubility prediction using deep neural networks with automated hyperparameter optimization. The framework implements multiple optimization strategies including feature selection, model architecture search, and sequential optimization approaches to achieve state-of-the-art prediction accuracy for aqueous solubility.

### Framework Versions

**ANO_pytorch2025** (Main/Recommended):
- Latest implementation using PyTorch
- **5-fold cross-validation** throughout all optimization processes
- Comprehensive **applicability domain (AD)** analysis with multiple methods:
  - Leverage-based AD
  - Distance-based AD (Euclidean, Mahalanobis)
  - Density-based AD (LOF, Isolation Forest)
  - Three AD modes: strict, flexible, adaptive
- Centralized configuration management via `config.py`
- Advanced QSAR analysis package with visualizations
- Memory-efficient training with TorchScript support
- Performance monitoring and logging
- Device-agnostic (CUDA/MPS/CPU)

**ANO_tensorflow** (Legacy):
- Original TensorFlow implementation
- Focuses on finding the best-performing model configuration
- Maintained for backward compatibility and reference
- Historical baseline comparisons

<div align="center">
    <a href="./md_sources/res2.png" target="_blank">
        <img src="./md_sources/res2.png" alt="ANO Framework Overview" width="1200" style="cursor: pointer;"/>
    </a>
</div>

## System Requirements

### Core Dependencies (ANO_pytorch2025)
- **Python 3.8 or later** (3.9+ recommended)
- **PyTorch 1.9.0 or later** (2.0+ recommended for best performance)
  - CUDA support for GPU acceleration (optional)
  - MPS support for Apple Silicon (optional)
- **RDKit 2021.09.1 or later** - molecular descriptors and fingerprints
- **Optuna 2.10.0 or later** - Bayesian hyperparameter optimization
- **NumPy 1.21.0 or later**
- **pandas 1.3.0 or later**
- **scikit-learn 0.24.0 or later** - baseline models and metrics
- **matplotlib 3.4.0 or later** - visualizations
- **seaborn 0.11.0 or later** - statistical plots
- **joblib 1.0.0 or later** - parallel processing
- **tqdm 4.62.0 or later** - progress bars
- **openpyxl 3.0.0 or later** - Excel export

### Additional Dependencies
- **scipy** - statistical analysis
- **plotly** - interactive visualizations
- **tensorboard** - training monitoring (optional)

## Datasets

### Main Training Datasets
- **ws496_logS.csv**(ws): Water solubility dataset (496 compounds)
- **delaney-processed.csv**(de): ESOL dataset (Delaney, 2004)
- **huusk.csv**(hu): Huuskonen aqueous solubility dataset (2000)
- **Lovric2020_logS0.csv**(lo): Intrinsic aqueous solubility dataset (2020)

### Test-Only Datasets
- **FreeSolv.csv**: Hydration free energy database
- **Lipophilicity.csv**: ChEMBL lipophilicity dataset
- **AqSolDB.csv**: Curated aqueous solubility database (9,982 compounds)
- **BigSolDB.csv**: Large-scale solubility dataset

For detailed dataset information and citations, see `data/DATASET_SOURCES.md`.

## ANO Modules

### 1. Data Preprocessing (`1_preprocess.py`)
- Standardizes SMILES strings using RDKit
- Generates molecular fingerprints (Morgan, MACCS, Avalon)
- Creates multiple data splitting strategies:
  - Random split (rm)
  - Scaffold split (sc)
  - Chemical space split (cs)
  - Cluster split (cl)
  - Physchem property split (pc)
  - Activity cliff split (ac)
  - Solubility-aware split (sa)
  - Time series split (ti)
  - Ensemble split (en)
- Performs comprehensive applicability domain (AD) analysis:
  - Multiple AD methods: leverage, distance-based, density-based
  - Three AD modes: strict, flexible, adaptive
  - Generates detailed AD reports and visualizations
  - Based on results in `result/1_preprocess/`

### 2. Standard Model Comparison (`2_standard_comp_pytorch_optimized.py`)
- Benchmark comparison with traditional ML models
- Random Forest, XGBoost, SVM implementations
- Deep neural networks using PyTorch
- Performance evaluation across different datasets and splits
- **5-fold cross-validation** for all experiments

### 3. Feature Deep Learning (`3_solubility_feature_deeplearning.py`)
- Deep learning approaches for solubility prediction
- Feature engineering and selection strategies
- Comparison of different neural network architectures
- Automated feature importance analysis
- **5-fold cross-validation** throughout

### 4. Feature Optimization (`4_ANO_FeatureOptimization_FO.py`)
- Automated feature selection from 49 molecular descriptor categories
- Uses Optuna for Bayesian hyperparameter optimization
- Fixed neural network architecture: 2727 (final fingerprint, morgan+maccs+avalon)→1024→496→1
- **5-fold cross-validation** for robust evaluation
- Outputs best feature subset for each dataset/split combination
- **Performance**: ~28 seconds per trial, ~200 minutes for 200 trials (50 epochs each)
- Typical configuration: 200 trials × 4 datasets × 9 split types = 7,200 optimizations

### 5. Model Optimization (`5_ANO_ModelOptimization_MO.py`)
- Optimizes neural network architecture with all features
- Searches over:
  - Number of layers (1-5)
  - Hidden units per layer (up to 9,999 units)
  - Dropout rates
  - Activation functions (ReLU, GELU, SiLU, Mish, SELU)
  - Optimizers (Adam, AdamW, NAdam, RAdam)
  - Learning rates (1e-5 to 1e-3)
  - Batch sizes (32, 64, 128)
  - Batch normalization
  - Learning rate schedulers (Cosine, Exponential, ReduceOnPlateau, Step)
- **5-fold cross-validation** for all evaluations
- **Performance**: ~30-190 seconds per trial (varies by architecture complexity)
  - Simple architectures (1-2 layers): ~30 seconds
  - Complex architectures (4-5 layers): ~190 seconds
- Expected duration: ~300 minutes for 200 trials

### 6. Network Optimization FOMO - FO→MO (`6_ANO_NetworkOptimization_FOMO.py`)
- Feature optimization followed by model structure optimization
- First selects best features, then optimizes architecture
- **5-fold cross-validation** throughout the process
- Combines advantages of both approaches
- Sequential optimization strategy

### 7. Network Optimization MOFO - MO→FO (`7_ANO_NetworkOptimization_MOFO.py`)
- Model structure optimization followed by feature optimization
- First finds best architecture, then selects features
- **5-fold cross-validation** for robust results
- Alternative sequential strategy

### 8. Final Model Training (`8_ANO_final_model_training.py`)
- Trains final models using best configurations from previous steps
- Comprehensive model evaluation and comparison
- Generates ensemble predictions
- Model persistence and checkpointing
- Full cross-validation and test set evaluation

### 9. Test-Only Evaluation (`9_ANO_testonly_evaluation.py`)
- Evaluates models on external test-only datasets
- Tests model generalization capability
- Applicability domain analysis on test sets
- Performance comparison across different test sets
- Generates comprehensive evaluation reports


## Key Features

### Molecular Representations
- **Base Fingerprints** (2727 bits total):
  - **Morgan Fingerprints**: 2048-bit circular fingerprints (radius=2)
  - **MACCS Keys**: 167 structural keys
  - **Avalon Fingerprints**: 512-bit fingerprints

- **RDKit Descriptors**: 49 categories (~882 features) - selectable via Optuna:

  **2D Descriptors (27 categories):**
  1. **MolWt** - Molecular Weight
  2. **MolLogP** - Molecular LogP
  3. **MolMR** - Molecular Refractivity
  4. **TPSA** - Topological Polar Surface Area
  5. **NumRotatableBonds** - Number of Rotatable Bonds
  6. **HeavyAtomCount** - Heavy Atom Count
  7. **NumHAcceptors** - Number of H Acceptors
  8. **NumHDonors** - Number of H Donors
  9. **NumHeteroatoms** - Number of Heteroatoms
  10. **NumValenceElectrons** - Number of Valence Electrons
  11. **NHOHCount** - NHOH Count
  12. **NOCount** - NO Count
  13. **RingCount** - Ring Count
  14. **NumAromaticRings** - Number of Aromatic Rings
  15. **NumSaturatedRings** - Number of Saturated Rings
  16. **NumAliphaticRings** - Number of Aliphatic Rings
  17. **LabuteASA** - Labute ASA
  18. **BalabanJ** - Balaban J Index
  19. **BertzCT** - Bertz Complexity Index
  20. **Ipc** - Information Content
  21. **kappa_Series[1-3]_ind** - Kappa Shape Indices (3 descriptors)
  22. **Chi_Series[13]_ind** - Chi Connectivity Indices (13 descriptors)
  23. **Phi** - Flexibility Index
  24. **HallKierAlpha** - Hall-Kier Alpha
  25. **NumAmideBonds** - Number of Amide Bonds
  26. **NumSpiroAtoms** - Number of Spiro Atoms
  27. **NumBridgeheadAtoms** - Number of Bridgehead Atoms

  **VSA Descriptors (5 series):**
  28. **PEOE_VSA_Series[1-14]_ind** - PEOE VSA (14 individual descriptors)
  29. **SMR_VSA_Series[1-10]_ind** - SMR VSA (10 individual descriptors)
  30. **SlogP_VSA_Series[1-12]_ind** - SlogP VSA (12 individual descriptors)
  31. **EState_VSA_Series[1-11]_ind** - EState VSA (11 individual descriptors)
  32. **VSA_EState_Series[1-10]** - VSA EState (10 individual descriptors)

  **rdMolDescriptors (3 categories):**
  33. **MQNs** - Molecular Quantum Numbers
  34. **AUTOCORR2D** - 2D Autocorrelation
  35. **BCUT2D** - 2D BCUT Descriptors

  **3D Descriptors (14 categories):**
  36. **FractionCSP3** - Fraction of sp3 Carbon Atoms
  37. **Asphericity** - Asphericity
  38. **PBF** - Plane of Best Fit
  39. **RadiusOfGyration** - Radius of Gyration
  40. **InertialShapeFactor** - Inertial Shape Factor
  41. **Eccentricity** - Eccentricity
  42. **SpherocityIndex** - Spherocity Index
  43. **PMI_series[1-3]_ind** - Principal Moments of Inertia (3 descriptors)
  44. **NPR_series[1-2]_ind** - Normalized Principal Moments Ratio (2 descriptors)
  45. **AUTOCORR3D** - 3D Autocorrelation
  46. **RDF** - Radial Distribution Function
  47. **MORSE** - Morse Descriptors
  48. **WHIM** - WHIM Descriptors
  49. **GETAWAY** - GETAWAY Descriptors

<div align="center">
    <a href="./md_sources/descriptors_list.png" target="_blank">
        <img src="./md_sources/descriptors_list.png" alt="Molecular Descriptors" width="600" style="cursor: pointer;"/>
    </a>
    <p><i>Complete list of 49 molecular descriptor categories used in ANO</i></p>
</div>

### Data Caching System

The framework implements an efficient multi-level caching system to minimize redundant computations:

**Fingerprint Caching** (`result/fingerprint/`):
- Molecular fingerprints saved as NPZ files
- Format: `{dataset}_{split}_train.npz` and `{dataset}_{split}_test.npz`
- Contains: Morgan (2048-bit) + MACCS (167-bit) + Avalon (512-bit) = 2727 features
- Automatically loaded if available, regenerated only if missing

**Chemical Descriptor Caching** (`result/chemical_descriptors/`):
- 3D conformers and molecular descriptors cached as NPZ files
- Format: `{dataset}_{split}_train_descriptors.npz` and `{dataset}_{split}_test_descriptors.npz`
- Contains: 49 descriptor categories (~882 individual features)
- 3D conformers generated once and reused across all modules

**Model Checkpoints** (`result/{module_name}/`):
- Best models saved as `.pth` files
- Optuna study databases (`.db` files) for resumable optimization
- Performance metrics and hyperparameters in JSON/CSV format

**Cache Validation**:
- Automatic dimension checking before loading
- Graceful fallback to regeneration if cache is corrupted
- Debug logging for cache hit/miss events

**Benefits**:
- ~10-100x faster data loading for repeated experiments
- Consistent feature representations across all modules
- Reduced memory usage through on-demand loading
- Easy cache invalidation by deleting specific NPZ files

### Optimization Strategies

#### Bayesian Optimization with Optuna
The framework employs sophisticated Bayesian optimization for efficient hyperparameter search:

**Sampler (Search Algorithm)**:
- **TPESampler** (Tree-structured Parzen Estimator) - Default and recommended
  - Multivariate optimization with dependency modeling
  - Adaptive exploration-exploitation balance
  - Best performance in 2025 real data tests
- Alternative samplers: AutoSampler, RandomSampler, CmaEsSampler, NSGAIISampler

**Pruner (Early Stopping)**:
- **HyperbandPruner** - Default for TPESampler
  - Aggressive early termination of unpromising trials
  - Resource-efficient optimization
  - Adaptive bracket-based scheduling
- **MedianPruner** - Alternative for balanced pruning
- **WilcoxonPruner** - Statistical significance-based pruning
- Configurable via `config.py` OPTUNA_CONFIG

**Random Sampling Strategy**:
- Initial 30% of trials use random sampling
- Ensures diverse exploration of search space
- Remaining 70% use Bayesian optimization

**Startup Trials (Warm-up)**:
- Module 4 (FO): 5 trials (10% of 200) before pruning
- Module 5 (MO): 10 trials (20% of 200) before pruning
- Prevents premature convergence

#### Cross-Validation & Training
- **5-fold Cross-Validation**: Used throughout all optimization processes
- **Dual CV Approaches**: Research and Production pipelines
- **Memory Management**: Subprocess-based training to prevent memory leaks
- **Parallel Processing**: Multi-dataset/split processing support

### Evaluation Metrics

#### Cross-Validation Strategy
All ANO modules employ **rigorous 5-fold cross-validation** with two complementary approaches:

**Type 1 - Research Pipeline:**
- 5-fold CV on training set with test set prediction per fold
- Provides average performance across folds
- Used as primary optimization metric in Optuna

**Type 2 - Production Pipeline:**
- 5-fold CV on training set + final model on full training data
- Final test set prediction using the best model
- Simulates real-world deployment scenario

#### Performance Metrics
- **R²** (coefficient of determination) - primary optimization target
- **RMSE** (root mean squared error)
- **MSE** (mean squared error)
- **MAE** (mean absolute error)

#### Applicability Domain Analysis
- **Leverage-based AD**: Williams plot analysis
- **Distance-based AD**: Euclidean and Mahalanobis distance
- **Density-based AD**: Local Outlier Factor (LOF), Isolation Forest
- **Coverage analysis**: Percentage of predictions within AD
- **Three AD modes**:
  - Strict: Conservative threshold (higher confidence)
  - Flexible: Balanced threshold (moderate confidence)
  - Adaptive: Data-driven threshold (optimized per dataset)

## Quick Start

### 1. Installation
```bash
cd ANO_pytorch2025
pip install -r requirements.txt  # Install dependencies
```

### 2. Data Preprocessing
```bash
# Run preprocessing pipeline
python 1_preprocess.py
```

### 3. Run ANO Modules

All modules support command-line arguments for flexible execution:

#### Module 1: Data Preprocessing
```bash
# Preprocess all datasets and splits
python 1_preprocess.py

# Preprocess specific dataset
python 1_preprocess.py --dataset ws

# Preprocess specific split type
python 1_preprocess.py --split rm

# Combine both
python 1_preprocess.py --dataset ws --split rm
```

#### Module 2: Standard Model Comparison
```bash
# Run all baseline comparisons
python 2_standard_comp_pytorch_optimized.py

# Specific dataset and split
python 2_standard_comp_pytorch_optimized.py --dataset de --split ac

# Custom epochs
python 2_standard_comp_pytorch_optimized.py --epochs 100
```

#### Module 3: Feature Deep Learning
```bash
# Analyze features with deep learning
python 3_solubility_feature_deeplearning.py

# Specific configuration
python 3_solubility_feature_deeplearning.py --dataset ws --epochs 50
```

#### Module 4: Feature Optimization (FO)
```bash
# Run feature optimization
python 4_ANO_FeatureOptimization_FO.py

# Specific dataset with custom trials
python 4_ANO_FeatureOptimization_FO.py --dataset ws --trials 200

# Specific split with custom epochs
python 4_ANO_FeatureOptimization_FO.py --split rm --epochs 50

# Start fresh optimization (ignore cache)
python 4_ANO_FeatureOptimization_FO.py --renew

# Full example
python 4_ANO_FeatureOptimization_FO.py --dataset ws --split rm --trials 200 --epochs 50
```

#### Module 5: Model Optimization (MO)
```bash
# Run model architecture optimization
python 5_ANO_ModelOptimization_MO.py

# Custom configuration
python 5_ANO_ModelOptimization_MO.py --dataset de --split ac --trials 200 --epochs 50

# Restart optimization
python 5_ANO_ModelOptimization_MO.py --renew
```

#### Module 6: Network Optimization FOMO (FO→MO)
```bash
# Run sequential feature then model optimization
python 6_ANO_NetworkOptimization_FOMO.py

# Custom setup
python 6_ANO_NetworkOptimization_FOMO.py --dataset ws --trials 100 --renew
```

#### Module 7: Network Optimization MOFO (MO→FO)
```bash
# Run sequential model then feature optimization
python 7_ANO_NetworkOptimization_MOFO.py

# Custom setup
python 7_ANO_NetworkOptimization_MOFO.py --dataset hu --split sa --trials 150
```

#### Module 8: Final Model Training
```bash
# Train final models with best configurations
python 8_ANO_final_model_training.py

# Specific dataset
python 8_ANO_final_model_training.py --dataset ws --epochs 100
```

#### Module 9: Test-Only Evaluation
```bash
# Evaluate on external test sets
python 9_ANO_testonly_evaluation.py

# Custom configuration
python 9_ANO_testonly_evaluation.py --dataset ws
```

#### Common Arguments

All modules support these common arguments:

```bash
--dataset {ws,de,lo,hu}    # Specific dataset (default: all)
--split {rm,ac,cl,cs,en,pc,sa,sc,ti}  # Specific split type (default: all active)
--trials N                  # Number of Optuna trials (Modules 4-7)
--epochs N                  # Training epochs per trial
--renew                     # Start fresh optimization (ignore previous results)
--help                      # Show help message
```

**Available Datasets**:
- `ws`: WS496 (Water Solubility, 496 compounds)
- `de`: Delaney-processed (ESOL dataset)
- `lo`: Lovric2020_logS0 (Intrinsic aqueous solubility)
- `hu`: Huuskonen (Aqueous solubility, 2000)

**Available Split Types**:
- `rm`: Random split
- `ac`: Activity cliff split
- `cl`: Cluster split
- `cs`: Chemical space split
- `en`: Ensemble split
- `pc`: Physchem property split
- `sa`: Solubility-aware split
- `sc`: Scaffold split
- `ti`: Time series split

### 4. Configuration
All modules support configuration through `config.py`:
- `N_TRIALS`: Number of Optuna trials (default: 100)
- `N_EPOCHS`: Training epochs (default: 100)
- `BATCH_SIZE`: Batch size for training (default: 32)
- `LEARNING_RATE`: Initial learning rate (default: 0.001)
- `SPLIT_TYPES`: Data splitting strategies to use
- `DATASETS`: Training datasets to process
- `DEVICE`: Computing device (cuda/mps/cpu)
- `N_JOBS`: Number of parallel jobs
- `RANDOM_SEED`: Random seed for reproducibility

## Results

Results are saved in the `result/` directory with the following structure:
- Model checkpoints (`.pth` files)
- Optuna study databases (`.db` files)
- Performance metrics (CSV/Excel)
- Visualizations (PNG/PDF)
- LaTeX tables for publication
- Applicability domain reports
- QSAR analysis outputs

### Computational Performance

**Training Time per Module**:

- **Module 1 (Preprocessing)**:
  - Fingerprint generation: ~1-5 seconds per molecule
  - 3D conformer generation: ~0.5-2 seconds per molecule
  - Descriptor calculation: ~0.1-1 second per molecule
  - Full preprocessing (500 molecules): ~5-10 minutes
  - Large datasets (10,000 molecules): ~1-2 hours
  - **Note**: Results are cached, subsequent runs are instant

- **Module 2 (Standard ML Comparison)**: (200 epochs, 5-fold CV)
  - Random Forest: ~10-30 seconds per dataset-split
  - XGBoost: ~20-60 seconds per dataset-split
  - SVM: ~30-120 seconds per dataset-split
  - DNN (PyTorch): ~100-300 seconds per dataset-split
  - Full comparison (all models): ~3-10 minutes per dataset-split
  - Complete benchmark (4 datasets × 9 splits × all models): ~2-6 hours per run
  - **Full optimization (all fingerprints + all splits)**: ~2-3 days
  - **Note**: Tests multiple fingerprints (Morgan, MACCS, Avalon, combinations)

- **Module 3 (Feature Deep Learning)**: (100 epochs, 5-fold CV)
  - Single descriptor category: ~30-60 seconds per dataset-split
  - Full 49 descriptor analysis: ~25-50 minutes per dataset-split
  - Single dataset × 9 splits: ~4-8 hours
  - All datasets (4) × 9 splits × 49 descriptors: ~2-3 days
  - **Total feature evaluations**: 4 datasets × 9 splits × 49 descriptors = 1,764 experiments
  - **Note**: Evaluates each descriptor category independently for comprehensive analysis

- **Module 4 (Feature Optimization)**: (~28 seconds/trial, 50 epochs, 5-fold CV)
  - Per trial: ~28 seconds
  - 200 trials: ~93 minutes per dataset-split combination
  - Full optimization (4 datasets × 9 splits): ~93 × 36 = ~56 hours
  - **Optimization**: Bayesian search for optimal descriptor combinations

- **Module 5 (Model Optimization)**: (~30-190 seconds/trial, 50 epochs, 5-fold CV)
  - Simple models (1-2 layers): ~30 seconds/trial
  - Complex models (4-5 layers): ~190 seconds/trial
  - Average: ~60 seconds/trial
  - 200 trials: ~200 minutes per dataset-split combination
  - Full optimization (4 datasets × 9 splits): ~200 × 36 = ~120 hours
  - **Optimization**: Architecture search (layers, units, activation, dropout)

- **Module 6 & 7 (Network Optimization)**: Sequential combination of FO and MO
  - FOMO (FO→MO): ~(93 + 200) = ~293 minutes per combination
  - MOFO (MO→FO): ~(200 + 93) = ~293 minutes per combination
  - Full optimization: ~176-352 hours (7-15 days)
  - **Note**: Most comprehensive optimization strategy

- **Module 8 (Final Model Training)**: (1000 epochs default)
  - Per dataset-split: ~10-30 minutes
  - All configurations: ~6-12 hours
  - Includes ensemble model training

- **Module 9 (Test-Only Evaluation)**:
  - Per test dataset: ~5-15 minutes
  - Includes AD analysis and visualization
  - All test datasets: ~30-60 minutes

**Overall Time Estimates** (Complete Framework):

| Module | Quick Test (1 dataset, 1 split) | Single Dataset (9 splits) | Full Framework (4 datasets × 9 splits) |
|--------|----------------------------------|---------------------------|----------------------------------------|
| Module 1 | ~5-10 min | ~15-30 min | ~30-60 min (cached after first run) |
| Module 2 | ~10-30 min | ~2-6 hours | **~2-3 days** (all fingerprints) |
| Module 3 | ~30-60 min | ~4-8 hours | **~2-3 days** (49 descriptors) |
| Module 4 | ~1.5 hours (200 trials) | ~14 hours | **~2-3 days** (7,200 trials total) |
| Module 5 | ~3 hours (200 trials) | ~27 hours | **~5-6 days** (7,200 trials total) |
| Module 6 | ~4.5 hours (200 trials) | ~41 hours | **~7-8 days** (sequential FO+MO) |
| Module 7 | ~4.5 hours (200 trials) | ~41 hours | **~7-8 days** (sequential MO+FO) |
| Module 8 | ~20-40 min | ~3-6 hours | ~12-24 hours |
| Module 9 | ~10-20 min | N/A (test only) | ~30-60 min (external tests) |

**Total Time for Complete Framework Run**: ~3-4 weeks (if running all modules sequentially)

**Recommended Workflow**:
- **Quick validation**: Modules 1, 2, 4, 5 with 1 dataset, 1 split, 10 trials (~2-3 hours)
- **Standard research**: Modules 1-5 with all splits, 100 trials (~1-2 weeks)
- **Publication-ready**: All modules with 200 trials (~3-4 weeks)
- **Parallel execution**: Run different datasets/splits in parallel to reduce time by 50-75%

**Hardware Considerations**:
- CPU: Multi-core processors recommended (4+ cores)
- GPU: CUDA or MPS acceleration supported for faster training (2-5x speedup)
- Memory: 8GB+ RAM recommended (16GB+ for large datasets)
- Storage: ~10GB for full results and models (cached data + model checkpoints)

### Performance Examples

<div align="center">
    <a href="./md_sources/r2_score_ws_individual.png" target="_blank">
        <img src="./md_sources/r2_score_ws_individual.png" alt="WS Dataset Performance" width="800" style="cursor: pointer;"/>
    </a>
    <p><i>R² scores for WS dataset with DNN</i></p>
</div>

<div align="center">
    <a href="./md_sources/r2_score_de_individual.png" target="_blank">
        <img src="./md_sources/r2_score_de_individual.png" alt="DE Dataset Performance" width="800" style="cursor: pointer;"/>
    </a>
    <p><i>R² scores for DE dataset with DNN</i></p>
</div>

<div align="center">
    <a href="./md_sources/r2_score_lo_individual.png" target="_blank">
        <img src="./md_sources/r2_score_lo_individual.png" alt="LO Dataset Performance" width="800" style="cursor: pointer;"/>
    </a>
    <p><i>R² scores for LO dataset with DNN</i></p>
</div>

<div align="center">
    <a href="./md_sources/r2_score_hu_individual.png" target="_blank">
        <img src="./md_sources/r2_score_hu_individual.png" alt="HU Dataset Performance" width="800" style="cursor: pointer;"/>
    </a>
    <p><i>R² scores for HU dataset with DNN</i></p>
</div>

<div align="center">
    <a href="./md_sources/test_predictions_vs_actual.png" target="_blank">
        <img src="./md_sources/test_predictions_vs_actual.png" alt="Test Predictions vs Actual" width="1200" style="cursor: pointer;"/>
    </a>
    <p><i>Test set predictions vs actual values across all ANO modules and datasets</i></p>
</div>

<div align="center">
    <a href="./md_sources/cv_test_comparison.png" target="_blank">
        <img src="./md_sources/cv_test_comparison.png" alt="CV vs Test Comparison" width="1200" style="cursor: pointer;"/>
    </a>
    <p><i>Cross-validation vs test set performance comparison</i></p>
</div>

## Project Structure

```
ANO_solubility_prediction/
├── ANO_pytorch2025/                # Main PyTorch implementation (2025)
│   ├── data/                       # Dataset directory
│   │   ├── train/                  # Training data splits
│   │   ├── test/                   # Test data splits
│   │   └── DATASET_SOURCES.md      # Dataset documentation
│   ├── extra_code/                 # Utility modules
│   │   ├── ano_feature_search.py   # Feature selection with Optuna
│   │   ├── ano_feature_selection.py # Molecular descriptor calculation
│   │   ├── mol_fps_maker.py        # Fingerprint generation
│   │   ├── chem_descriptor_maker.py # Chemical descriptor calculation
│   │   ├── preprocess.py           # Data preprocessing utilities
│   │   ├── device_utils.py         # Device management utilities
│   │   ├── performance_monitor.py  # Performance monitoring
│   │   ├── learning_process_pytorch_torchscript.py # TorchScript training
│   │   └── qsar_analysis/          # QSAR analysis package
│   │       ├── ad_methods.py       # Applicability domain methods
│   │       ├── advanced_ad_methods.py # Advanced AD methods
│   │       ├── density_ad_methods.py  # Density-based AD methods
│   │       ├── analyzer.py         # Main analyzer
│   │       ├── data_loader.py      # Data loading utilities
│   │       ├── features.py         # Feature engineering
│   │       ├── metrics.py          # Evaluation metrics
│   │       ├── splitters.py        # Data splitting strategies
│   │       ├── statistics.py       # Statistical analysis
│   │       ├── utils.py            # Utility functions
│   │       └── visualizations/     # Visualization modules
│   │           ├── ad_plots.py     # AD visualizations
│   │           ├── advanced_ad_plots.py # Advanced AD plots
│   │           ├── metric_plots.py # Metric visualizations
│   │           ├── stat_plots.py   # Statistical plots
│   │           ├── meta_plots.py   # Meta analysis plots
│   │           ├── summary_plots.py # Summary visualizations
│   │           ├── ad_performance_analysis.py # AD performance
│   │           ├── visualization_manager.py # Visualization manager
│   │           └── visualization_utils.py # Visualization utilities
│   ├── result/                     # Output directory
│   │   ├── fingerprint/            # Cached fingerprints (NPZ)
│   │   ├── chemical_descriptors/   # Cached descriptors (NPZ)
│   │   ├── 1_preprocess/           # Preprocessing results
│   │   ├── 4_ANO_FeatureOptimization_FO/
│   │   ├── 5_ANO_ModelOptimization_MO/
│   │   ├── 6_ANO_NetworkOptimization_FOMO/
│   │   ├── 7_ANO_NetworkOptimization_MOFO/
│   │   ├── 8_ANO_final_model_training/
│   │   └── 9_ANO_testonly_evaluation/
│   ├── logs/                       # Training logs
│   ├── 0_search_comp_figures_sci.py # Search method comparison
│   ├── 0_search_comp_figures_sci.ipynb # Search analysis notebook
│   ├── 1_preprocess.py             # Data preprocessing pipeline
│   ├── 2_standard_comp_pytorch_optimized.py # Standard model comparison
│   ├── 3_solubility_feature_deeplearning.py # Feature DL analysis
│   ├── 4_ANO_FeatureOptimization_FO.py # Feature optimization module
│   ├── 5_ANO_ModelOptimization_MO.py # Model architecture optimization
│   ├── 6_ANO_NetworkOptimization_FOMO.py # Feature→Model optimization
│   ├── 7_ANO_NetworkOptimization_MOFO.py # Model→Feature optimization
│   ├── 8_ANO_final_model_training.py # Final model training
│   ├── 9_ANO_testonly_evaluation.py # Test-only evaluation
│   ├── config.py                   # Configuration file
│   ├── requirements.txt            # Python dependencies
│   └── ano_final.db                # Optuna study database
├── ANO_tensorflow/                 # TensorFlow implementation (legacy)
│   └── extra_code/                 # TensorFlow utilities
├── previous_version/               # Previous implementation versions
├── md_sources/                     # Documentation and figures
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## Additional Analysis

### Search Strategy Comparison
**0_search_comp_figures_sci.py / 0_search_comp_figures_sci.ipynb**
   - Comparison of optimization algorithms
   - Bayesian vs. Grid Search vs. Random Search
   - Demonstrates superior efficiency of Bayesian optimization
   - Scientific journal-quality figures

<div align="center">
    <a href="./md_sources/0_search_comp_figures/fig1_search_space_comparison.png" target="_blank">
        <img src="./md_sources/0_search_comp_figures/fig1_search_space_comparison.png" alt="Search Space Comparison" width="800" style="cursor: pointer;"/>
    </a>
    <p><i>Comparison of search strategies: Grid Search vs Random Search vs Bayesian Search</i></p>
</div>

### Model Comparison

<div align="center">
    <a href="./md_sources/plots/1_standard_model_compare_sa_maccs.png" target="_blank">
        <img src="./md_sources/plots/1_standard_model_compare_sa_maccs.png" alt="Model Comparison" width="1200" style="cursor: pointer;"/>
    </a>
    <p><i>Comparison of different machine learning models</i></p>
</div>

### Legacy Implementations

#### ANO_tensorflow/
- Original TensorFlow-based implementation
- Includes legacy model architectures
- Maintained for backward compatibility

#### previous_version/
- Previous implementation versions
- Historical experimental notebooks
- Archive of earlier approaches and results

## Citation

If you use the ANO framework in your research, please cite:

```bibtex
@software{ano_framework_2025,
  title={ANO: Automated Network Optimizer for Enhanced Prediction of Intrinsic Solubility in Drug-like Organic Compounds},
  author={Lee, Seungjin},
  year={2025},
  url={https://github.com/arer90/ANO_solubility_prediction},
  note={A Comprehensive Machine Learning Approach with Bayesian Optimization}
}
```

## License

This project is licensed under the **MIT License**.

Copyright (c) 2025 Lee, Seungjin (arer90)

See the [LICENSE](LICENSE) file for full details.

**Key Points**:
- ✅ Free to use for commercial and non-commercial purposes
- ✅ Modification and distribution permitted
- ✅ Private use allowed
- ⚠️ No warranty provided
- 📋 Must include original license and copyright notice

## Acknowledgments

- RDKit development team for the cheminformatics toolkit
- Optuna team for the hyperparameter optimization framework
- PyTorch team for the deep learning framework
- scikit-learn team for machine learning utilities
- Contributors to open-source solubility datasets

## Contact

For questions or collaborations, please open an issue on the GitHub repository.