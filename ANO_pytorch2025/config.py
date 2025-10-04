"""
Configuration File for ANO Solubility Prediction Framework
==========================================================

This comprehensive configuration module serves as the central control hub for the ANO framework,
defining all global settings, dataset configurations, model parameters, and system optimizations
across all modules (1-9). It ensures consistent behavior and reproducible results throughout
the entire machine learning pipeline.

Core Configuration Areas:
------------------------
1. **System Optimization**: OS-specific settings for multiprocessing and memory management
2. **Dataset Management**: Paths, metadata, and preprocessing parameters for all molecular datasets
3. **Model Architecture**: Neural network configurations, hyperparameters, and training settings
4. **Fingerprint Configuration**: Molecular representation settings (Morgan, RDKit, MACCS, etc.)
5. **Splitting Strategies**: Random, scaffold-based, and temporal splitting configurations
6. **Performance Tuning**: Memory limits, parallelization, and resource allocation
7. **Result Management**: Output directories, logging, and visualization settings

Key Features:
------------
- **Cross-Platform Compatibility**: Automatic OS detection and optimization
- **Modular Design**: Separate configurations for each framework module
- **Reproducible Science**: Fixed seeds and deterministic settings
- **Scalable Architecture**: Configurable resource allocation based on system capabilities
- **Flexible Dataset Handling**: Support for custom datasets and experimental configurations

Supported Datasets:
------------------
- **WS (Water Solubility)**: AqSolDB dataset with ~9,000 experimental measurements
- **DE (Density)**: Density prediction dataset with ~1,100 compounds
- **LO (LogS)**: Solubility in various solvents with ~1,300 compounds
- **HU (Human)**: Human-relevant endpoints with ~600+ bioactive compounds

Configuration Examples:
----------------------
# Access dataset information
dataset_info = DATASETS['ws']
print(f"Dataset: {dataset_info['name']}")
print(f"Size: {dataset_info['size']} compounds")

# Get active datasets for a module
active_datasets = get_code_datasets(3)  # For module 3
print(f"Module 3 uses: {active_datasets}")

# Configure fingerprint settings
fingerprint_config = FINGERPRINTS['morgan']
fp_params = fingerprint_config['params']

# Set up model architecture
model_config = MODEL_CONFIG['simple_dnn']
hidden_dims = model_config['hidden_dims']

System Requirements:
-------------------
- Python 3.8+ with scientific computing stack
- PyTorch 1.10+ for neural network implementations
- RDKit 2022+ for molecular descriptor calculations
- Scikit-learn 1.0+ for baseline models and metrics
- Memory: 8GB+ RAM recommended for large datasets
- Storage: 10GB+ for datasets, models, and results

Environment Variables:
---------------------
The configuration automatically sets optimal environment variables based on the detected OS:
- OMP_NUM_THREADS: OpenMP thread count for CPU parallelization
- MKL_NUM_THREADS: Intel MKL thread count for linear algebra operations
- KMP_DUPLICATE_LIB_OK: Prevents OpenMP library conflicts
- MAX_WORKERS: Maximum parallel processes for data processing
"""

import os
import platform
from pathlib import Path

# ===== OS Detection and Optimization =====
OS_TYPE = platform.system()  # 'Darwin' for macOS, 'Linux', 'Windows'

# OS-specific multiprocessing settings
if OS_TYPE == "Windows":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'
    # MAX_WORKERS: Conservative limit (4 workers max, reserve 1 CPU for system)
    MAX_WORKERS = min(4, os.cpu_count() - 1) if os.cpu_count() else 2
elif OS_TYPE == "Darwin":  # macOS
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'
    # MAX_WORKERS: Moderate limit (4 workers max, reserve 1 CPU for system)
    MAX_WORKERS = min(4, os.cpu_count() - 1) if os.cpu_count() else 2
else:  # Linux
    os.environ['OMP_NUM_THREADS'] = str(min(4, os.cpu_count())) if os.cpu_count() else '4'
    os.environ['MKL_NUM_THREADS'] = str(min(4, os.cpu_count())) if os.cpu_count() else '4'
    # MAX_WORKERS: Aggressive limit (8 workers max, reserve 1 CPU for system)
    MAX_WORKERS = min(8, os.cpu_count() - 1) if os.cpu_count() else 4

# Matplotlib settings for different OS
try:
    import matplotlib
    # Use Agg backend for all OS to prevent GUI popups
    matplotlib.use('Agg')  # Non-interactive backend - works on all platforms

    # Font settings for different OS
    if OS_TYPE == "Darwin":
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    elif OS_TYPE == "Windows":
        matplotlib.rcParams['font.family'] = 'Arial'
    else:
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
except ImportError:
    pass

try:
    import torch
except ImportError:
    torch = None

# ====================================================================
# USER CONFIGURATION SECTION - MODIFY THESE SETTINGS FOR YOUR NEEDS
# ====================================================================
#
# User-configurable settings that you can modify:
#   1. USER_DATASET_FILE_MAPPING - CSV files to use for training/testing
#   2. USER_DISPLAY_NAMES - Names that appear in plots and reports
#   3. CODE_SPECIFIC_DATASETS - Which datasets each code module should use
#   4. ACTIVE_SPLIT_TYPES - Data splitting methods to apply
#   5. CODE_SPECIFIC_FINGERPRINTS - Molecular fingerprints for each code module
#      (CODE_SPECIFIC settings override default ACTIVE settings)
#   6. MODEL_CONFIG - Deep learning training parameters
#   7. TEST_ONLY_DATASETS - Datasets used only for final testing (not training)
#   8. AD_MODE_CONFIG - Applicability Domain settings for model validation
#   9. PREPROCESS_CONFIG - Data preprocessing options
#   10. CACHE_CONFIG - Cache regeneration control settings
#
# WARNING: Do not modify sections below this area unless you know what you're doing!
# ====================================================================

# 10. Cache Configuration - Control cache regeneration
CACHE_CONFIG = {
    'remake_fingerprint': False,  # True: always regenerate fingerprints, False: use cache if exists
    'remake_descriptors': False,  # True: always regenerate descriptors, False: use cache if exists
}

# 11. Restart/Resume Configuration - Control experiment restart behavior
RESTART_CONFIG = {
    'mode': 'resume',  # 'restart': start from beginning, 'resume': skip completed, 'smart': check conditions and decide
    'check_input_changes': True,  # True: compare input conditions and restart if changed, False: trust existing results
    'force_restart_modules': [3],  # Force restart Module 3 to test improved code (L2, dropout, early stopping)
    'resume_from_partial': True,  # True: resume even from partial results, False: only from complete results
}

# 1. Dataset File Mapping - MUST BE DEFINED BY USER
# Map your actual CSV file names (without path) to abbreviations
USER_DATASET_FILE_MAPPING = {
    # Training datasets (files in data/ folder)
    'ws496_logS.csv': 'ws',
    'delaney-processed.csv': 'de',
    'huusk.csv': 'hu', 
    'Lovric2020_logS0.csv': 'lo',
    
    # Test-only datasets (files in data/ folder)
    'FreeSolv.csv': 'fs',
    'Lipophilicity.csv': 'lp',
    'AqSolDB.csv': 'aq',  # Note: Some papers call this "curated-solubility-dataset"
    'BigSolDB.csv': 'bs',
    
    # Add more datasets as needed:
    # 'curated-solubility-dataset.csv': 'cs',
    # 'SAMPL.csv': 'sa',
}

# 2. Display Names for Plots and Reports
# These names will appear in visualizations. If not defined, abbreviation will be used
USER_DISPLAY_NAMES = {
    'ws': 'WS496',
    'de': 'Delaney', 
    'hu': 'Huuskonen',
    'lo': 'Lovric2020',
    'fs': 'FreeSolv',
    'lp': 'Lipophilicity',
    'aq': 'AqSolDB',
    'bs': 'BigSolDB',
    # Add custom display names as needed
}

# 3. Code-Specific Dataset Configuration [TARGET]
# Specify which datasets each code module should use (use abbreviations from USER_DATASET_FILE_MAPPING)
CODE_SPECIFIC_DATASETS = {
    '2': ['ws', 'de', 'lo', 'hu'],  # Code 2: Standard ML models comparison (Random Forest, SVM, XGBoost)
    '3': ['ws', 'de', 'lo', 'hu'],  # Code 3: Feature-based deep learning
    '4': ['ws', 'de', 'lo', 'hu'],  # Code 4: ANO-FO (Feature Optimization)
    '5': ['ws', 'de', 'lo', 'hu'],  # Code 5: ANO-MO (Model Architecture Optimization)
    '6': ['ws', 'de', 'lo', 'hu'],  # Code 6: ANO-FOMO (Feature→Model Optimization)
    '7': ['ws', 'de', 'lo', 'hu'],  # Code 7: ANO-MOFO (Model→Feature Optimization)
    '8': ['ws', 'de', 'lo', 'hu'],  # Code 8: Compare all ANO models and select best
    '9': ['ws', 'de', 'lo', 'hu'],  # Code 9: Final predictions using best ANO model
}

# 4. Active Split Types [REFRESH]
# Choose which data splitting methods to use (for quick testing, use only ['rm'])
# 'rm': Random splitting (molecules randomly assigned to train/test sets)
# 'ac': Activity Cliff (split based on similar molecules with different activities)
# 'cl': Cluster splitting (molecules grouped by similarity, then split)
# 'cs': Chemical Space splitting (split based on chemical diversity)
# 'en': Ensemble splitting (multiple random splits combined)
# 'pc': PhysChem splitting (split based on physicochemical properties)
# 'sa': Solubility Aware splitting (split considering solubility ranges)
# 'sc': Scaffold splitting (split based on molecular scaffold/framework)
# 'ti': Time-based splitting (split based on chronological order)
# ACTIVE_SPLIT_TYPES = ['rm','ac','cl','cs','en','pc','sa','sc','ti']  # Use all split types
ACTIVE_SPLIT_TYPES = ['rm'] # Use only random splitting for faster execution

# 5. Code-Specific Fingerprint Configuration [SCIENCE]
# Choose which molecular fingerprints each code module should use
# Options: 'morgan', 'maccs', 'avalon', 'all' (uses all available fingerprints)
# Combinations possible: 'morgan+maccs', 'morgan+avalon', 'maccs+avalon'
CODE_SPECIFIC_FINGERPRINTS = {
    # '1': Code 1 doesn't use fingerprints (preprocessing only)
    '2': ['morgan','maccs','avalon','morgan+maccs','morgan+avalon','maccs+avalon','all'],  # Code 2: Use all fingerprints for baseline comparison
    '3': ['morgan','maccs','avalon','morgan+maccs','morgan+avalon','maccs+avalon','all'],  # Code 3: Use all fingerprints for feature-based deep learning
    '4': ['all'],     # Code 4: ANO-FO (Feature Optimization)
    '5': ['all'],     # Code 5: ANO-MO (Model Architecture Optimization)
    '6': ['all'],     # Code 6: ANO-FOMO (Feature→Model Optimization)
    '7': ['all'],     # Code 7: ANO-MOFO (Model→Feature Optimization)
    '8': ['all'],     # Code 8: Compare all ANO models and select best
    '9': ['all'],     # Code 9: Final predictions using best ANO model
}

# [WARNING] Important notes about CODE_SPECIFIC_DATASETS and CODE_SPECIFIC_FINGERPRINTS:
# - These define which datasets and fingerprints each module should use
# - All modules (1-9) must be defined here
# - If a module is not defined, the system will raise an error

# 6. Model Configuration ⚙️ (Deep Learning Training Parameters)
MODEL_CONFIG = {
    'epochs': None,                  # Default number of training epochs (increased for better convergence)
    'early_stopping_patience': 200,  # Early stopping patience (epochs to wait for improvement)
    'batch_size': 32,               # Reduced batch size to prevent CUDA OOM
    'learning_rate': 0.001,         # Initial learning rate
    'cv_folds': 5,                  # Number of cross-validation folds
    'val_size': 0.2,                # Validation set size
    'random_state': 42,             # Random seed for reproducibility
    'regularizer': 1e-5,            # L2 regularization strength
    'dropout_rate': 0.3,            # Dropout rate
    'architecture': [1024, 469, 1], # DNN architecture [hidden1, hidden2, output]
    'optuna_trials': 10,             # Number of Optuna optimization trials (increased for better results)

    # Optuna Sampler Configuration (matched with reference implementation successful strategy)
    # Available Samplers (2024-2025):
    #   - 'tpe': TPESampler - Best for deep learning, handles categorical parameters well (current choice)
    #   - 'auto': AutoSampler - Latest (2024), automatically selects best algorithm, better than TPE
    #   - 'cmaes': CmaEsSampler - Best for continuous variables, doesn't support categorical
    #   - 'gp': GPSampler - Gaussian Process, good for integer variables
    #   - 'random': RandomSampler - Baseline for comparison
    'optuna_sampler': {
        'type': 'tpe',                # TPESampler - most proven for deep learning
        'n_startup_trials_ratio': 0.2, # 20% random exploration before TPE
        'n_ei_candidates': 50,        # Number of EI candidates to evaluate
        'multivariate': False,        # Disable for dynamic search space compatibility
    },

    # Optuna Pruner Configuration (matched with reference implementation successful strategy)
    # Available Pruners (2024-2025):
    #   - 'hyperband': HyperbandPruner - Best with TPESampler, efficient for long epochs (current choice)
    #   - 'successive_halving': SuccessiveHalvingPruner - Similar to Hyperband but simpler
    #   - 'median': MedianPruner - Best with RandomSampler, most widely used
    #   - 'wilcoxon': WilcoxonPruner - Latest (2024), most robust to noise
    #   - 'percentile': PercentilePruner - Keeps top N% trials
    'optuna_pruner': {
        'type': 'hyperband',          # HyperbandPruner - most efficient for 1000 epochs
        'min_resource': 100,          # Minimum epochs per trial
        'max_resource': 1000,         # Maximum epochs per trial (same as EPOCHS)
        'reduction_factor': 3         # Keep top 33% at each stage (reference implementation used this)
    },
    'cuda_memory_fraction': 0.6,    # Limit GPU memory usage to 60% (conservative setting)
    'gradient_accumulation': 2,     # Accumulate gradients over 2 steps
    'renew': False,                 # True: Delete existing studies and restart, False: Continue from saved studies

    # Unified Pruning Configuration for Modules 4-7
    'pruner_type': 'MedianPruner',  # Options: MedianPruner, HyperbandPruner, PercentilePruner
    'pruner_params': {
        'MedianPruner': {
            'n_startup_trials': 3,    # Start pruning after 3 trials (reduced from 10)
            'n_warmup_steps': 1,      # Allow 1 fold before pruning (aggressive)
            'interval_steps': 1       # Check after each fold
        },
        'HyperbandPruner': {
            'min_resource': 1,        # Minimum 1 fold
            'max_resource': 5,        # Maximum 5 folds (CV-5)
            'reduction_factor': 2     # Halving
        },
        'PercentilePruner': {
            'percentile': 25.0,       # Prune bottom 25%
            'n_startup_trials': 3,
            'n_warmup_steps': 1,
            'interval_steps': 1
        }
    },

    # Module-specific epochs (used when epochs is None)
    'module_epochs': {
        '2': 100,    # Module 2: Standard comparison
        '3': 100,    # Module 3: Feature analysis with DL
        '4': 10,   # Module 4: Feature optimization (matched with reference implementation)
        '5': 10,   # Module 5: Model architecture optimization (matched with reference implementation)
        '6': 10,   # Module 6: Network optimization with fixed features
        '7': 10,   # Module 7: Network optimization with fixed architecture
        '8': 10000,   # Module 8: Final model training
    }

    # Training Method Configuration
    # 'training_method': 'subprocess', # 'subprocess' (default) or 'direct'
    # - 'subprocess': Use learning_process_pytorch_torchscript.py (memory isolation, model saving, TorchScript)
    # - 'direct': Direct SimpleDNN training in main process (faster, easier debugging)
}

# 7. Database Configuration 💾 (Optuna Storage Settings)
DATABASE_CONFIG = {
    # Database backend selection: 'basic', 'parallel', 'postgresql'
    'backend': 'basic',  # Change this to switch database backends

    # Basic configuration (shared SQLite database)
    'basic': {
        'storage': 'sqlite:///ano_final.db?timeout=60',
        'description': 'Single shared SQLite database - simple but can have lock issues with parallel execution'
    },

    # Parallel configuration (separate SQLite databases per module)
    'parallel': {
        'storage_pattern': 'sqlite:///ano_module{module_id}_{module_name}.db?timeout=60',
        'databases': {
            '4': 'sqlite:///ano_module4_feature_optimization.db?timeout=60',
            '5': 'sqlite:///ano_module5_model_optimization.db?timeout=60',
            '6': 'sqlite:///ano_module6_network_fomo.db?timeout=60',
            '7': 'sqlite:///ano_module7_network_mofo.db?timeout=60',
            '8': 'sqlite:///ano_module8_final_training.db?timeout=60'
        },
        'description': 'Separate SQLite databases per module - enables true parallel execution without locks'
    },

    # PostgreSQL configuration (production-grade concurrent database)
    'postgresql': {
        'storage': 'postgresql://optuna_user:optuna_pass@localhost:5432/optuna_db',
        'description': 'PostgreSQL database - best for high-performance concurrent access'
    }
}

# Database Helper Functions
def get_storage_url(module_id=None):
    """
    Get the appropriate storage URL based on current database configuration.

    Args:
        module_id (str): Module identifier ('4', '5', '6', '7', '8')

    Returns:
        str: Storage URL for Optuna
    """
    backend = DATABASE_CONFIG['backend']

    if backend == 'basic':
        return DATABASE_CONFIG['basic']['storage']
    elif backend == 'parallel':
        if module_id and module_id in DATABASE_CONFIG['parallel']['databases']:
            return DATABASE_CONFIG['parallel']['databases'][module_id]
        else:
            # Fallback to pattern-based generation
            module_names = {
                '4': 'feature_optimization',
                '5': 'model_optimization',
                '6': 'network_fomo',
                '7': 'network_mofo',
                '8': 'final_training'
            }
            module_name = module_names.get(module_id, 'unknown')
            return DATABASE_CONFIG['parallel']['storage_pattern'].format(
                module_id=module_id, module_name=module_name)
    elif backend == 'postgresql':
        return DATABASE_CONFIG['postgresql']['storage']
    else:
        raise ValueError(f"Unknown database backend: {backend}")

def get_database_info():
    """Get current database configuration info."""
    backend = DATABASE_CONFIG['backend']
    config = DATABASE_CONFIG[backend]
    return {
        'backend': backend,
        'description': config['description'],
        'storage': config.get('storage', 'Multiple databases')
    }

def get_epochs_for_module(module_id, args=None):
    """
    Get epochs for a specific module with priority:
    1. Command line arguments (if provided)
    2. MODEL_CONFIG['epochs'] (if not None)
    3. MODEL_CONFIG['module_epochs'][module_id] (module-specific)
    4. Default value of 30

    Args:
        module_id (str): Module identifier ('2', '3', '4', '5', '6', '7', '8')
        args: Argparse arguments object with optional 'epochs' attribute

    Returns:
        int: Number of epochs to use
    """
    # Priority 1: Command line arguments
    if args and hasattr(args, 'epochs') and args.epochs is not None:
        return args.epochs

    # Priority 2: Global epochs setting
    if MODEL_CONFIG.get('epochs') is not None:
        return MODEL_CONFIG['epochs']

    # Priority 3: Module-specific epochs
    if module_id in MODEL_CONFIG.get('module_epochs', {}):
        return MODEL_CONFIG['module_epochs'][module_id]

    # Priority 4: Default
    return 30

# DNN Hyperparameter Search Spaces (for Modules 5 and 7)
DNN_HYPERPARAMETERS = {
    'search_space': {  # DNN Hyperparameter Search Space (for Modules 5 and 7)
        # Network Architecture
        'n_layers': [1, 5],  # Number of hidden layers
        'hidden_dims': [2, 9999],  # Hidden layer size range for all layers (trial.suggest_int)

        # Regularization (Fast & Effective)
        'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],  # Dropout rate discrete choices
        'use_batch_norm': [True, False],  # Batch Normalization - speeds up convergence
        'gradient_clip': [None, 1.0, 5.0],  # Gradient clipping for stability

        # Optimization
        'learning_rate': [0.001, 0.0001, 0.00001],  # Learning rate options (removed 0.01 for stability)
        # 'learning_rate': [0.001, 0.0001, 0.00001],  # Learning rate options (removed 0.01 for stability)
        'batch_size': [32, 64, 128],
        'weight_decay': [1e-5, 1e-4, 1e-3],  # L2 regularization, narrower range for regression
        'optimizer': ['adamw', 'adam', 'radam', 'nadam'],  # Stable optimizers for regression (rmsprop commented out)
        # 'optimizer': ['adamw', 'adam', 'radam', 'nadam','rmsprop'],  # Include rmsprop
        # 'optimizer': ['adamw', 'adam', 'radam', 'nadam', 'rmsprop', 'lbfgs'],  # Full optimizer list

        # Activation Functions
        'activation': ['gelu', 'silu', 'relu', 'leaky_relu', 'elu', 'mish', 'selu'],  # Stable activations for regression
        # 'activation': ['gelu', 'silu', 'relu', 'leaky_relu', 'elu', 'mish', 'selu'],  # Full activation list
        'final_activation': None,  # None for regression (no activation on output layer)


        # Weight Initialization (Fast Convergence)
        'weight_init': ['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'],
        'bias_init': ['zeros'],  # zeros is standard and fastest

        # Learning Rate Scheduler (Simple & Fast)
        'scheduler': {
            'none': {},  # No scheduler
            'step_lr': {'step_size': [10, 30], 'gamma': [0.5, 0.9]},
            'exponential': {'gamma': [0.95, 0.99]},
        },

        # ===== Advanced Options (Currently Disabled for Speed) =====
        # Uncomment these for future use (classification, complex tasks, etc.)

        # # Advanced Regularization
        # 'dropout_strategy': ['standard', 'alpha_dropout', 'dropout2d'],  # Different dropout types
        # 'gradient_accumulation': [1, 2, 4],  # For memory-limited situations

        # # Output Layer Options (for Classification)
        # 'final_activation': ['softmax', 'sigmoid'],  # For classification tasks
        # # 'final_activation': ['tanh'],  # For bounded regression [-1, 1]

        # # Advanced Schedulers (Slower but potentially better)
        # 'scheduler': {
        #     'cosine': {'T_max': [50, 100]},  # Cosine annealing
        #     'reduce_on_plateau': {'patience': [5, 10], 'factor': [0.5]},  # Adaptive LR
        #     'cyclic': {'base_lr': [1e-5, 1e-4], 'max_lr': [1e-3, 1e-2]},  # Cyclic LR
        # },
        # 'warmup_steps': [0, 100, 500],  # Learning rate warmup

        # # Data Augmentation & Ensemble
        # 'label_smoothing': [0.0, 0.1, 0.2],  # For classification robustness
        # 'mixup_alpha': [0.0, 0.2, 0.4],  # Data augmentation

        # # Architecture Enhancements
        # 'skip_connection': [True, False],  # ResNet-style connections
        # 'attention_layers': [True, False],  # Self-attention mechanism
    },
    'default': {  # Default settings when optimization fails
        'n_layers': 2,
        'hidden_dims': [1024, 496],  # Fixed architecture as specified
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'weight_decay': 0.0,
        'optimizer': 'adam',
        'activation': 'relu',
        'use_batch_norm': False,
        'gradient_clip': None,
        'weight_init': 'xavier_uniform',
        'bias_init': 'zeros',
        'scheduler': 'none',
        'final_activation': None  # None for regression
    }
}

# ===== OPTUNA HYPERPARAMETER OPTIMIZATION CONFIGURATION =====
# Centralized Optuna configuration for all modules (4-7)
# Based on 2025 benchmarks and real data testing results

OPTUNA_CONFIG = {
    # Default configuration (2025 proven best combination)
    'default': {
        'sampler': 'TPESampler',      # Winner in 2025 real data tests
        'pruner': 'HyperbandPruner', # Best pruner for TPESampler
        'sampler_kwargs': {},         # Use default parameters (optimal)
        'pruner_kwargs': {}           # Use default parameters (optimal)
    },

    # Available sampler algorithms with descriptions
    'available_samplers': {
        # === RECOMMENDED SAMPLERS (2025) ===
        'TPESampler': {
            'description': 'Tree-structured Parzen Estimator - Best overall performance, handles categorical variables well',
            'use_case': 'General purpose, molecular property prediction, default choice',
            'performance_rank': 1,  # Based on 2025 real data tests
            'kwargs': {
                # 'n_startup_trials': 10,      # Number of random trials before TPE starts
                # 'n_ei_candidates': 24,       # Number of candidate samples for EI
                # 'multivariate': False,       # Whether to use multivariate TPE
                # 'warn_independent_sampling': True  # Warn when falling back to independent sampling
            }
        },

        'AutoSampler': {
            'description': 'Automatic sampler selection - 2025 cutting-edge, requires OptunaHub',
            'use_case': 'When unsure about sampler choice, experimental/research use',
            'performance_rank': 2,  # Promising but requires additional setup
            'kwargs': {}  # No parameters needed
        },

        # === ALTERNATIVE SAMPLERS ===
        'RandomSampler': {
            'description': 'Random sampling - Simple baseline, good for quick exploration',
            'use_case': 'Baseline comparison, debugging, very early exploration',
            'performance_rank': 4,
            'kwargs': {
                # 'seed': None  # Random seed for reproducibility (None = random)
            }
        },

        'CmaEsSampler': {
            'description': 'Covariance Matrix Adaptation Evolution Strategy - Good for continuous variables',
            'use_case': 'Large evaluation budgets (>100x parameters), continuous-only problems',
            'performance_rank': 3,
            'limitations': 'Does not support categorical parameters, requires large budgets',
            'kwargs': {
                # 'n_startup_trials': 1,       # Number of trials before CMA-ES starts
                # 'independent_sampler': None, # Sampler for initial trials
                # 'warn_independent_sampling': True
            }
        },

        'SkoptSampler': {
            'description': 'Scikit-optimize based Gaussian Process - Alternative to TPE',
            'use_case': 'When TPE performance is poor, small-medium search spaces',
            'performance_rank': 5,
            'kwargs': {
                # 'independent_sampler': None,
                # 'warn_independent_sampling': True
            }
        }
    },

    # Available pruner algorithms with descriptions
    'available_pruners': {
        # === RECOMMENDED PRUNERS (2025) ===
        'HyperbandPruner': {
            'description': 'Hyperband algorithm - Best overall performance, especially with TPESampler',
            'use_case': 'Default choice, deep learning, any iterative training',
            'performance_rank': 1,  # Best for TPESampler combination
            'kwargs': {
                # 'min_resource': 1,           # Minimum resource (e.g., epochs)
                # 'max_resource': 'auto',      # Maximum resource allocation
                # 'reduction_factor': 3        # Resource reduction factor
            }
        },

        'WilcoxonPruner': {
            'description': 'Statistical test-based pruning - 2025 new feature, specialized for CV',
            'use_case': 'Cross-validation tasks, statistical significance testing',
            'performance_rank': 2,  # Specialized for CV tasks
            'kwargs': {
                # 'p_threshold': 0.1  # P-value threshold for pruning
            }
        },

        # === ALTERNATIVE PRUNERS ===
        'MedianPruner': {
            'description': 'Median-based pruning - Simple and conservative',
            'use_case': 'Conservative pruning, when unsure about aggressive pruning',
            'performance_rank': 3,  # Outperformed by Hyperband
            'kwargs': {
                # 'n_startup_trials': 5,      # Trials before pruning starts
                # 'n_warmup_steps': 0,        # Steps before pruning in each trial
                # 'interval_steps': 1         # Interval between pruning checks
            }
        },

        'SuccessiveHalvingPruner': {
            'description': 'Successive halving algorithm - Aggressive early stopping',
            'use_case': 'When computational budget is very limited',
            'performance_rank': 4,
            'kwargs': {
                # 'min_resource': 1,          # Minimum resource allocation
                # 'reduction_factor': 4,      # Resource reduction factor
                # 'min_early_stopping_rate': 0  # Minimum early stopping rate
            }
        },

        'PercentilePruner': {
            'description': 'Percentile-based pruning - Prune bottom X% of trials',
            'use_case': 'When you want to prune a fixed percentage of trials',
            'performance_rank': 5,
            'kwargs': {
                # 'percentile': 25.0,         # Percentile threshold (0-100)
                # 'n_startup_trials': 5,      # Trials before pruning starts
                # 'n_warmup_steps': 0,        # Steps before pruning in each trial
                # 'interval_steps': 1         # Interval between pruning checks
            }
        },

        'NopPruner': {
            'description': 'No pruning - All trials run to completion',
            'use_case': 'Debugging, when pruning causes issues, small trial counts',
            'performance_rank': 6,
            'kwargs': {}  # No parameters
        }
    },

    # Recommended combinations for different scenarios
    'recommended_combinations': {
        'best_overall': {
            'sampler': 'TPESampler',
            'pruner': 'HyperbandPruner',
            'description': '2025 proven best combination - winner in real data tests'
        },
        'experimental_2025': {
            'sampler': 'AutoSampler',
            'pruner': 'HyperbandPruner',
            'description': '2025 cutting-edge automatic selection (requires OptunaHub)'
        },
        'cv_specialized': {
            'sampler': 'TPESampler',
            'pruner': 'WilcoxonPruner',
            'description': 'Specialized for cross-validation tasks (2025 feature)'
        },
        'conservative': {
            'sampler': 'TPESampler',
            'pruner': 'MedianPruner',
            'description': 'Conservative approach with less aggressive pruning'
        },
        'baseline': {
            'sampler': 'RandomSampler',
            'pruner': 'MedianPruner',
            'description': 'Simple baseline for comparison purposes'
        },
        'debug': {
            'sampler': 'RandomSampler',
            'pruner': 'NopPruner',
            'description': 'For debugging - no intelligence, no pruning'
        }
    }
}

# 7. Test-Only Datasets [STATS] (External validation datasets - not used for training)
# These datasets are used only for final testing to evaluate model generalization
# They are never seen during training or optimization phases
TEST_ONLY_DATASETS = ['FreeSolv', 'Lipophilicity', 'AqSolDB', 'BigSolDB']

# 8. Applicability Domain (AD) Mode Configuration [TARGET]
AD_MODE_CONFIG = {
    'default_mode': 'flexible',  # Options: 'strict', 'flexible', 'adaptive'
    'strict_threshold': 0.95,    # For strict mode
    'flexible_threshold': 0.85,  # For flexible mode
    'adaptive_min': 0.80,        # For adaptive mode
    'adaptive_max': 0.95,        # For adaptive mode
}

# 9. Preprocessing Configuration [CONFIG]
PREPROCESS_CONFIG = {
    'test_size': 0.2,           # Test set size for data splitting
    'random_state': 42,         # Random seed for splitting
    'min_heavy_atoms': 3,       # Minimum heavy atoms in molecule
    'max_heavy_atoms': 100,     # Maximum heavy atoms in molecule
    'remove_salts': True,       # Remove salt forms
    'neutralize': True,         # Neutralize charged molecules
    'standardize': True,        # Standardize tautomers
}

# 10. Module Names for Log Organization [LOGS]
# These names will be used for log and result folder organization
MODULE_NAMES = {
    '2': '2_ANO_StandardML_Baseline',
    '3': '3_ANO_FeatureDeepLearning',
    '4': '4_ANO_FeatureOptimization_FO',
    '5': '5_ANO_ModelOptimization_MO',
    '6': '6_ANO_NetworkOptimization_FOMO',
    '7': '7_ANO_NetworkOptimization_MOFO',
    '8': '8_ANO_final_model_training',
    '9': '9_ANO_testonly_evaluation'
}

# ====================================================================
# ⛔ AUTOMATIC SYSTEM CONFIGURATION - DO NOT MODIFY BELOW THIS LINE ⛔
# ====================================================================
#
# 🚨 WARNING: The settings below are automatically generated - DO NOT MODIFY!
# The system automatically configures these based on your user settings above.
# Changing anything below may break the system functionality.
# ====================================================================

# Build internal dataset mappings from user configuration
DATASETS = {}
for filename, abbrev in USER_DATASET_FILE_MAPPING.items():
    dataset_name = filename.replace('.csv', '')
    DATASETS[abbrev] = dataset_name

# Reverse mapping for lookup
DATASET_NAME_TO_KEY = {v: k for k, v in DATASETS.items()}

# Build display names with fallback to abbreviation
DATASET_DISPLAY_NAMES = {}
for abbrev in DATASETS.keys():
    if abbrev in USER_DISPLAY_NAMES:
        DATASET_DISPLAY_NAMES[abbrev] = USER_DISPLAY_NAMES[abbrev]
    else:
        DATASET_DISPLAY_NAMES[abbrev] = abbrev.upper()

# Note: DATASET_ALIASES removed - no longer used in the system

# Split type definitions
SPLIT_TYPES = {
    'rm': 'random',
    'ac': 'activity_cliff', 
    'cl': 'cluster',
    'cs': 'chemical_space',
    'en': 'ensemble',
    'pc': 'physchem',
    'sa': 'solubility_aware',
    'sc': 'scaffold',
    'ti': 'time'
}

# Fingerprint configurations
FINGERPRINTS = {
    'morgan': {'radius': 2, 'nBits': 2048},
    'maccs': {'nBits': 167},
    'avalon': {'nBits': 512}
}

# 49 Chemical descriptors
CHEMICAL_DESCRIPTORS = [
        'MolWeight',                    # 0
        'MolLogP',                      # 1
        'MolMR',                        # 2
        'TPSA',                         # 3
        'NumRotatableBonds',            # 4
        'HeavyAtomCount',               # 5
        'NumHAcceptors',                # 6
        'NumHDonors',                   # 7
        'NumHeteroatoms',               # 8
        'NumValenceElectrons',          # 9
        'NHOHCount',                    # 10
        'NOCount',                      # 11
        'RingCount',                    # 12
        'NumAromaticRings',             # 13
        'NumSaturatedRings',            # 14
        'NumAliphaticRings',            # 15
        'LabuteASA',                    # 16
        'BalabanJ',                     # 17
        'BertzCT',                      # 18
        'Ipc',                          # 19
        'kappa_Series[1-3]_ind',        # 20
        'Chi_Series[1-98]_ind',         # 21
        'Phi',                          # 22
        'HallKierAlpha',                # 23
        'NumAmideBonds',                # 24
        'FractionCSP3',                 # 25
        'NumSpiroAtoms',                # 26
        'NumBridgeheadAtoms',           # 27
        'PEOE_VSA_Series[1-14]_ind',    # 28
        'SMR_VSA_Series[1-10]_ind',     # 29
        'SlogP_VSA_Series[1-12]_ind',   # 30
        'EState_VSA_Series[1-11]_ind',  # 31
        'VSA_EState_Series[1-10]_ind',  # 32
        'MQNs',                         # 33
        'AUTOCORR2D',                   # 34
        'BCUT2D',                       # 35
        'Asphericity',                  # 36
        'PBF',                          # 37
        'RadiusOfGyration',             # 38
        'InertialShapeFactor',          # 39
        'Eccentricity',                 # 40
        'SpherocityIndex',              # 41
        'PMI_series[1-3]_ind',          # 42
        'NPR_series[1-2]_ind',          # 43
        'AUTOCORR3D',                   # 44
        'RDF',                          # 45
        'MORSE',                        # 46
        'WHIM',                         # 47
        'GETAWAY'                       # 48
]

# Helper function to format descriptor names for display (remove _ind suffix)
def format_descriptor_name_for_display(descriptor_name):
    """
    Format descriptor name for plot display by removing '_ind' suffix.

    Args:
        descriptor_name (str): Original descriptor name

    Returns:
        str: Formatted name for display (without _ind)

    Examples:
        'kappa_Series[1-3]_ind' -> 'kappa_Series[1-3]'
        'TPSA' -> 'TPSA' (unchanged)
    """
    return descriptor_name.replace('_ind', '')

# Descriptors that need normalization
DESCRIPTORS_NEED_NORMALIZATION = [
    'Ipc', 'PMI1', 'PMI2', 'PMI3', 'RDF', 'MORSE', 'WHIM', 'GETAWAY'
]

# Path configuration
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'data')
RESULT_PATH = os.path.join(BASE_PATH, 'result')
MODEL_PATH = os.path.join(BASE_PATH, 'save_model')

# Create necessary directories
for path in [RESULT_PATH, MODEL_PATH]:
    os.makedirs(path, exist_ok=True)

# OS-specific settings
if platform.system() == "Windows":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    MAX_WORKERS = min(4, os.cpu_count() - 1) if os.cpu_count() else 1
else:
    MAX_WORKERS = min(8, os.cpu_count()) if os.cpu_count() else 4

# Parallel processing configuration with OS-specific optimizations
PARALLEL_CONFIG = {
    'max_workers': MAX_WORKERS,
    'chunk_size': 100,
    'enable_multiprocessing': True,
    'use_gpu': torch.cuda.is_available() if torch else False,
    'use_mps': torch.backends.mps.is_available() if torch and hasattr(torch.backends, 'mps') else False,
    'mixed_precision': True if OS_TYPE in ["Windows", "Linux"] else False,  # Limited MPS support
    'pin_memory': True if OS_TYPE in ["Windows", "Linux"] else False,  # Not supported on MPS
    'num_workers_dataloader': 0 if OS_TYPE == "Windows" else min(4, MAX_WORKERS),  # Windows multiprocessing issues
    'device_optimization': True  # Enable device_utils.py optimizations
}

# Note: SPEED_OPTIMIZATIONS removed - no longer used in the system

# Logging configuration
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def get_dataset_display_name(dataset_key):
    """Get display name for plots and UI from dataset key"""
    if dataset_key in DATASET_DISPLAY_NAMES:
        return DATASET_DISPLAY_NAMES[dataset_key]
    else:
        return dataset_key.upper()

def get_dataset_filename(dataset_key):
    """Get actual filename from dataset key"""
    if dataset_key in DATASETS:
        return DATASETS[dataset_key]
    else:
        return dataset_key

def get_dataset_abbreviation(dataset_name):
    """Get abbreviation from dataset name or filename"""
    if dataset_name in DATASETS:
        return dataset_name
    if dataset_name in DATASET_NAME_TO_KEY:
        return DATASET_NAME_TO_KEY[dataset_name]
    return dataset_name

def get_split_type_name(split_key):
    """Get full split type name from abbreviation"""
    return SPLIT_TYPES.get(split_key, split_key)

def get_code_datasets(code_number):
    """
    Get datasets for specific code number
    
    Returns CODE_SPECIFIC_DATASETS if defined for the code,
    otherwise raises error
    """
    code_str = str(code_number)
    if code_str in CODE_SPECIFIC_DATASETS:
        return CODE_SPECIFIC_DATASETS[code_str]
    else:
        raise ValueError(f"Code {code_number} not defined in CODE_SPECIFIC_DATASETS")

def get_code_fingerprints(code_number):
    """
    Get fingerprints for specific code number
    
    Returns CODE_SPECIFIC_FINGERPRINTS if defined for the code,
    otherwise raises error
    """
    code_str = str(code_number)
    if code_str in CODE_SPECIFIC_FINGERPRINTS:
        return CODE_SPECIFIC_FINGERPRINTS[code_str]
    else:
        raise ValueError(f"Code {code_number} not defined in CODE_SPECIFIC_FINGERPRINTS")

def generate_dataset_abbreviation(dataset_name: str) -> str:
    """
    Generate a short abbreviation for dataset name.

    Args:
        dataset_name: Full dataset name

    Returns:
        Short abbreviation (typically 3-6 characters)
    """
    # Handle common dataset names
    abbreviation_map = {
        'FreeSolv': 'FS',
        'Lipophilicity': 'Lipo',
        'AqSolDB': 'AqSol',
        'BigSolDB': 'BigSol',
        'ESOL': 'ESOL',
        'Solubility': 'Sol',
        'ws496': 'WS496',
        'SAMPL': 'SAMPL'
    }

    # Return mapped abbreviation if available
    if dataset_name in abbreviation_map:
        return abbreviation_map[dataset_name]

    # Generate abbreviation for unknown datasets
    # Take first letter + consonants, limit to 6 characters
    name_clean = dataset_name.replace('_', '').replace('-', '')
    if len(name_clean) <= 3:
        return name_clean.upper()

    # Take first letter plus consonants
    abbrev = name_clean[0]
    for char in name_clean[1:]:
        if char.lower() not in 'aeiou':
            abbrev += char
        if len(abbrev) >= 6:
            break

    return abbrev.upper()[:6]

# Note: get_active_dataset_filenames function removed (unnecessary due to ACTIVE_DATASETS removal)

# ====================================================================
# BEST MODEL CONFIGURATIONS FROM ANO OPTIMIZATION
# ====================================================================
# This section stores the best configurations found by ANO modules 4-7
# These values are automatically updated when optimization completes

BEST_ANO_CONFIGS = {
    # Module 4 (Feature Optimization) best features for each dataset/split
    'module_4_features': {
        # Format: {dataset}_{split}: [selected_feature_indices]
        # Example: 'ws_rm': [0, 1, 2, 3, 15, 22, ...],
    },
    
    # Module 5 (Model Optimization) best structures for each dataset/split
    'module_5_structures': {
        # Format: {dataset}_{split}: {'n_layers': int, 'hidden_dims': [list], 'dropout': float, ...}
        # Example: 'ws_rm': {'n_layers': 3, 'hidden_dims': [1024, 512, 256], 'dropout': 0.2},
    },
    
    # Module 6 (Feature->Structure) best configs
    'module_6_configs': {
        # Format: {dataset}_{split}: {'features': [...], 'structure': {...}}
    },
    
    # Module 7 (Structure->Feature) best configs
    'module_7_configs': {
        # Format: {dataset}_{split}: {'structure': {...}, 'features': [...]}
    },
    
    # Global best configuration (updated after module 8 comparison)
    'global_best': {
        'module': None,  # Which module performed best overall
        'config': {},    # Best configuration details
        'metrics': {}    # Performance metrics
    }
}

def update_best_config(module_num, dataset, split_type, config_data):
    """
    Update best configuration for a specific module/dataset/split combination

    Args:
        module_num: Module number (4, 5, 6, or 7)
        dataset: Dataset abbreviation (e.g., 'ws', 'de')
        split_type: Split type (e.g., 'rm', 'ac')
        config_data: Dictionary containing the configuration to save
    """
    import json
    
    key = f"{dataset}_{split_type}"
    
    if module_num == 4:
        BEST_ANO_CONFIGS['module_4_features'][key] = config_data
    elif module_num == 5:
        BEST_ANO_CONFIGS['module_5_structures'][key] = config_data
    elif module_num == 6:
        BEST_ANO_CONFIGS['module_6_configs'][key] = config_data
    elif module_num == 7:
        BEST_ANO_CONFIGS['module_7_configs'][key] = config_data
    
    # Save to JSON file for persistence
    config_file = Path(RESULT_PATH) / "best_ano_configs.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_file, 'w') as f:
            json.dump(BEST_ANO_CONFIGS, f, indent=2, default=str)
        print(f"Updated best config for module {module_num}, {dataset}-{split_type}")
    except Exception as e:
        print(f"Warning: Could not save best configs to file: {e}")

def load_best_configs():
    """Load saved best configurations from file"""
    import json
    from pathlib import Path
    
    config_file = Path(RESULT_PATH) / "best_ano_configs.json"
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                loaded_configs = json.load(f)
                # Update the global BEST_ANO_CONFIGS
                for key in loaded_configs:
                    if key in BEST_ANO_CONFIGS:
                        BEST_ANO_CONFIGS[key] = loaded_configs[key]
                print(f"Loaded best configs from {config_file}")
                return True
        except Exception as e:
            print(f"Warning: Could not load best configs from file: {e}")
    return False

def load_best_from_db(module_num, dataset, split_type):
    """Load best configuration directly from Optuna DB"""
    try:
        import optuna
        storage = "sqlite:///ano_final.db"
        
        # Determine study name based on module
        if module_num == 4:
            study_name = f"ano_feature_{dataset}_{split_type}"
        elif module_num == 5:
            study_name = f"ano_structure_{dataset}_{split_type}"
        elif module_num == 6:
            study_name = f"ano_network_FOMO_{dataset}_{split_type}"
        elif module_num == 7:
            study_name = f"ano_network_type2_MOFO_{dataset}_{split_type}"
        else:
            return None
        
        # Load study from DB
        study = optuna.load_study(study_name=study_name, storage=storage)
        
        if not study.trials:
            print(f"No trials found in DB for {study_name}")
            return None
            
        best_trial = study.best_trial
        
        # Extract configuration based on module type
        if module_num == 4:
            # Feature optimization - get selected descriptors
            return best_trial.user_attrs.get('selected_descriptors', [])
            
        elif module_num == 5:
            # Structure optimization
            return {
                'n_layers': best_trial.params.get('n_layers'),
                'hidden_dims': best_trial.user_attrs.get('hidden_dims', []),
                'dropout_rate': best_trial.params.get('dropout_rate'),
                'activation': best_trial.params.get('activation', 'relu'),
                'learning_rate': best_trial.params.get('learning_rate'),
                'batch_size': best_trial.params.get('batch_size')
            }
            
        elif module_num == 6:
            # Feature->Structure
            return {
                'features': best_trial.user_attrs.get('selected_descriptors', []),
                'structure': {
                    'n_layers': best_trial.params.get('n_layers'),
                    'hidden_dims': best_trial.user_attrs.get('hidden_dims', []),
                    'dropout_rate': best_trial.params.get('dropout_rate'),
                    'learning_rate': best_trial.params.get('learning_rate'),
                    'batch_size': best_trial.params.get('batch_size')
                }
            }
            
        elif module_num == 7:
            # Structure->Feature
            return {
                'structure': {
                    'n_layers': best_trial.user_attrs.get('n_layers'),
                    'hidden_dims': best_trial.user_attrs.get('hidden_dims', []),
                    'dropout_rate': best_trial.user_attrs.get('dropout_rate'),
                    'activation': best_trial.user_attrs.get('activation', 'relu')
                },
                'features': best_trial.user_attrs.get('selected_descriptors', [])
            }
            
    except Exception as e:
        print(f"Could not load from DB: {e}")
        return None

def get_module_epochs(module_id, command_line_epochs=None):
    """
    Get epochs for specific module with priority system

    Priority: command_line > global_epochs > module_specific > default

    Args:
        module_id (str or int): Module identifier ('2', '3', '4', '5', '6', '7', '8')
        command_line_epochs (int, optional): Epochs from command-line argument

    Returns:
        int: Number of epochs for the module
    """
    module_id = str(module_id)

    # Priority 1: Command-line argument (highest priority)
    if command_line_epochs is not None:
        return command_line_epochs

    # Priority 2: Global epochs override
    global_epochs = MODEL_CONFIG.get('global_epochs')
    if global_epochs is not None:
        return global_epochs

    # Priority 3: Module-specific epochs
    module_epochs = MODEL_CONFIG.get('module_epochs', {}).get(module_id)
    if module_epochs is not None:
        return module_epochs

    # Priority 4: Default epochs (lowest priority)
    return MODEL_CONFIG.get('epochs', 30)

def get_best_structure(dataset, split_type, module=5):
    """
    Get best structure for a dataset/split from specified module
    Priority: 1) DB, 2) Config file, 3) Default
    
    Args:
        dataset: Dataset abbreviation
        split_type: Split type
        module: Module number (default 5 for structure optimization)
    
    Returns:
        Dictionary with structure configuration or default structure
    """
    key = f"{dataset}_{split_type}"
    
    # 1. Try to load from DB first
    db_config = load_best_from_db(module, dataset, split_type)
    if db_config:
        if module == 5:
            return db_config
        elif module in [6, 7]:
            return db_config.get('structure') if isinstance(db_config, dict) else None
    
    # 2. Try to load from config file
    load_best_configs()  # Ensure configs are loaded
    
    if module == 5:
        config = BEST_ANO_CONFIGS['module_5_structures'].get(key)
        if config:
            return config
    elif module == 6:
        config = BEST_ANO_CONFIGS['module_6_configs'].get(key)
        if config:
            return config.get('structure')
    elif module == 7:
        config = BEST_ANO_CONFIGS['module_7_configs'].get(key)
        if config:
            return config.get('structure')
    
    # 3. Return default structure
    print(f"Warning: No best structure found for {dataset}-{split_type} module {module}, using default")
    return {
        'n_layers': 2,
        'hidden_dims': [1024, 496],
        'dropout_rate': 0.2,
        'activation': 'relu',
        'learning_rate': 0.001,
        'batch_size': 32
    }

def get_best_features(dataset, split_type, module=4):
    """
    Get best features for a dataset/split from specified module
    Priority: 1) DB, 2) Config file, 3) Default (all features)
    
    Args:
        dataset: Dataset abbreviation
        split_type: Split type
        module: Module number (default 4 for feature optimization)
    
    Returns:
        List of feature indices or empty list for all features
    """
    key = f"{dataset}_{split_type}"
    
    # 1. Try to load from DB first
    db_config = load_best_from_db(module, dataset, split_type)
    if db_config:
        if module == 4:
            return db_config if isinstance(db_config, list) else []
        elif module in [6, 7]:
            if isinstance(db_config, dict):
                return db_config.get('features', [])
    
    # 2. Try to load from config file
    load_best_configs()  # Ensure configs are loaded
    
    if module == 4:
        features = BEST_ANO_CONFIGS['module_4_features'].get(key)
        if features:
            return features
    elif module == 6:
        config = BEST_ANO_CONFIGS['module_6_configs'].get(key)
        if config:
            return config.get('features', [])
    elif module == 7:
        config = BEST_ANO_CONFIGS['module_7_configs'].get(key)
        if config:
            return config.get('features', [])
    
    # 3. Return empty list (meaning use all features)
    print(f"Warning: No best features found for {dataset}-{split_type} module {module}, using all features")
    return []

# ============================================================================
# GLOBAL BEST CONFIGURATION MANAGEMENT
# ============================================================================

def load_best_configurations(config_file="best_configurations.json"):
    """Load best configurations from JSON file"""
    import json
    config_path = Path(__file__).parent / config_file

    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                return data
        except Exception as e:
            print(f"Error loading configurations: {e}")
    return {}

# Note: save_best_configurations function removed - no longer used

# Note: get_best_config function removed - no longer used

# ===== OPTUNA HELPER FUNCTIONS =====

def get_optuna_sampler_and_pruner(combination='best_overall'):
    """
    Get Optuna sampler and pruner instances based on configuration

    Args:
        combination (str): Configuration combination name from OPTUNA_CONFIG['recommended_combinations']
                          Options: 'best_overall', 'experimental_2025', 'cv_specialized',
                                  'conservative', 'baseline', 'debug'

    Returns:
        tuple: (sampler_instance, pruner_instance)
    """
    import optuna

    # Get the configuration
    if combination in OPTUNA_CONFIG['recommended_combinations']:
        config = OPTUNA_CONFIG['recommended_combinations'][combination]
    else:
        print(f"Unknown combination '{combination}', using 'best_overall'")
        config = OPTUNA_CONFIG['recommended_combinations']['best_overall']

    sampler_name = config['sampler']
    pruner_name = config['pruner']

    # Get sampler instance
    sampler_info = OPTUNA_CONFIG['available_samplers'].get(sampler_name, OPTUNA_CONFIG['available_samplers']['TPESampler'])
    sampler_kwargs = sampler_info.get('kwargs', {})

    # Create sampler
    if sampler_name == 'TPESampler':
        sampler = optuna.samplers.TPESampler(**sampler_kwargs)
    elif sampler_name == 'RandomSampler':
        sampler = optuna.samplers.RandomSampler(**sampler_kwargs)
    elif sampler_name == 'CmaEsSampler':
        sampler = optuna.samplers.CmaEsSampler(**sampler_kwargs)
    elif sampler_name == 'SkoptSampler':
        sampler = optuna.samplers.SkoptSampler(**sampler_kwargs)
    elif sampler_name == 'AutoSampler':
        try:
            # AutoSampler requires OptunaHub
            import optuna_hub
            sampler = optuna_hub.samplers.AutoSampler(**sampler_kwargs)
        except ImportError:
            print("AutoSampler requires OptunaHub. Falling back to TPESampler.")
            sampler = optuna.samplers.TPESampler()
    else:
        print(f"Unknown sampler '{sampler_name}', using TPESampler")
        sampler = optuna.samplers.TPESampler()

    # Get pruner instance
    pruner_info = OPTUNA_CONFIG['available_pruners'].get(pruner_name, OPTUNA_CONFIG['available_pruners']['HyperbandPruner'])
    pruner_kwargs = pruner_info.get('kwargs', {})

    # Create pruner
    if pruner_name == 'HyperbandPruner':
        pruner = optuna.pruners.HyperbandPruner(**pruner_kwargs)
    elif pruner_name == 'MedianPruner':
        pruner = optuna.pruners.MedianPruner(**pruner_kwargs)
    elif pruner_name == 'SuccessiveHalvingPruner':
        pruner = optuna.pruners.SuccessiveHalvingPruner(**pruner_kwargs)
    elif pruner_name == 'PercentilePruner':
        pruner = optuna.pruners.PercentilePruner(**pruner_kwargs)
    elif pruner_name == 'WilcoxonPruner':
        try:
            pruner = optuna.pruners.WilcoxonPruner(**pruner_kwargs)
        except AttributeError:
            print("WilcoxonPruner not available in this Optuna version. Falling back to HyperbandPruner.")
            pruner = optuna.pruners.HyperbandPruner()
    elif pruner_name == 'NopPruner':
        pruner = optuna.pruners.NopPruner(**pruner_kwargs)
    else:
        print(f"Unknown pruner '{pruner_name}', using HyperbandPruner")
        pruner = optuna.pruners.HyperbandPruner()

    return sampler, pruner

def get_unified_pruner():
    """
    Get unified pruner for modules 4-7 based on MODEL_CONFIG settings
    Returns pruner object that can be used consistently across all modules
    Note: Each module should override n_startup_trials as needed
    """
    import optuna

    pruner_type = MODEL_CONFIG['pruner_type']
    params = MODEL_CONFIG['pruner_params'][pruner_type]

    if pruner_type == 'MedianPruner':
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=params['n_startup_trials'],
            n_warmup_steps=params['n_warmup_steps'],
            interval_steps=params['interval_steps']
        )
    elif pruner_type == 'HyperbandPruner':
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=params['min_resource'],
            max_resource=params['max_resource'],
            reduction_factor=params['reduction_factor']
        )
    elif pruner_type == 'PercentilePruner':
        pruner = optuna.pruners.PercentilePruner(
            percentile=params['percentile'],
            n_startup_trials=params['n_startup_trials'],
            n_warmup_steps=params['n_warmup_steps'],
            interval_steps=params['interval_steps']
        )
    else:
        # Default to MedianPruner if unknown type
        print(f"Unknown pruner type: {pruner_type}, using MedianPruner")
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=1,
            interval_steps=1
        )

    print(f"Using {pruner_type} with params: {params}")
    return pruner

def print_optuna_info(combination='best_overall'):
    """Print information about the selected Optuna configuration"""
    if combination in OPTUNA_CONFIG['recommended_combinations']:
        config = OPTUNA_CONFIG['recommended_combinations'][combination]
        print(f"\n=== OPTUNA CONFIGURATION: {combination.upper()} ===")
        print(f"Description: {config['description']}")
        print(f"Sampler: {config['sampler']}")
        print(f"Pruner: {config['pruner']}")

        # Print sampler details
        sampler_info = OPTUNA_CONFIG['available_samplers'][config['sampler']]
        print(f"\nSampler Details:")
        print(f"  - Description: {sampler_info['description']}")
        print(f"  - Use Case: {sampler_info['use_case']}")
        print(f"  - Performance Rank: {sampler_info['performance_rank']}")

        # Print pruner details
        pruner_info = OPTUNA_CONFIG['available_pruners'][config['pruner']]
        print(f"\nPruner Details:")
        print(f"  - Description: {pruner_info['description']}")
        print(f"  - Use Case: {pruner_info['use_case']}")
        print(f"  - Performance Rank: {pruner_info['performance_rank']}")
        print("=" * 50)
    else:
        print(f"Unknown combination: {combination}")
        print(f"Available combinations: {list(OPTUNA_CONFIG['recommended_combinations'].keys())}")

# ============================================================================
# MODULE CONFIGURATION SETTINGS
# ============================================================================

MODULE_CONFIG = {
    "module_6": {
        "force_feature_update": False,  # Set to True to always use latest Module 4 results
        "enable_versioning": True,      # Enable versioning logic for descriptor selection
        "consistency_mode": "existing", # "existing" or "latest" - default behavior
    },
    "module_7": {
        "force_feature_update": False,
        "enable_versioning": True,
        "consistency_mode": "existing",
    }
}

def list_optuna_options():
    """List all available Optuna sampler and pruner options"""
    print("\n=== AVAILABLE OPTUNA SAMPLERS ===")
    for name, info in OPTUNA_CONFIG['available_samplers'].items():
        rank = info['performance_rank']
        desc = info['description']
        print(f"{rank}. {name}: {desc}")

    print("\n=== AVAILABLE OPTUNA PRUNERS ===")
    for name, info in OPTUNA_CONFIG['available_pruners'].items():
        rank = info['performance_rank']
        desc = info['description']
        print(f"{rank}. {name}: {desc}")

    print("\n=== RECOMMENDED COMBINATIONS ===")
    for name, info in OPTUNA_CONFIG['recommended_combinations'].items():
        print(f"- {name}: {info['description']}")
        print(f"  Sampler: {info['sampler']}, Pruner: {info['pruner']}")
    print("=" * 50)