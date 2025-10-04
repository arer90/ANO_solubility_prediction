"""
QSAR Configuration Module with System Optimization
"""
import os
import platform
import psutil
import multiprocessing as mp

import polars as pl

# 1. Add missing POLARS_TYPE_MAPPING
POLARS_TYPE_MAPPING = {
    'float32': pl.Float32,
    'float64': pl.Float64,
    'int32': pl.Int32,
    'int64': pl.Int64,
    'string': pl.Utf8,
    'utf8': pl.Utf8
}


# ===== System Detection and Optimization =====
def get_system_info():
    """Detect system information and suggest optimal settings with enhanced error handling"""
    try:
        system_info = {
            'os': platform.system(),
            'cpu_count': mp.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': platform.python_version()
        }
        
        # OS-specific optimization settings - calculate recommendations only (not actually used)
        if system_info['os'] == 'Linux':
            system_info['recommended_backend'] = 'threading'
            system_info['recommended_workers'] = min(system_info['cpu_count'] - 1, 8)
        elif system_info['os'] == 'Darwin':  # macOS
            system_info['recommended_backend'] = 'threading'
            system_info['recommended_workers'] = min(system_info['cpu_count'] - 1, 8)
        elif system_info['os'] == 'Windows':
            system_info['recommended_backend'] = 'threading'
            system_info['recommended_workers'] = min(system_info['cpu_count'] - 1, 4)
        else:
            # Unknown OS
            system_info['recommended_backend'] = 'threading'
            system_info['recommended_workers'] = 2
        
        # Memory-based sample size recommendations - calculate recommendations only
        memory_gb = system_info['memory_gb']
        if memory_gb >= 32:
            system_info['recommended_max_samples'] = 100000
            system_info['recommended_max_ad_samples'] = 50000
        elif memory_gb >= 16:
            system_info['recommended_max_samples'] = 50000
            system_info['recommended_max_ad_samples'] = 20000
        elif memory_gb >= 8:
            system_info['recommended_max_samples'] = 20000
            system_info['recommended_max_ad_samples'] = 10000
        else:
            system_info['recommended_max_samples'] = 10000
            system_info['recommended_max_ad_samples'] = 5000
        
        # Safe default values to use
        system_info['max_workers'] = 2  # Safe default
        system_info['max_samples_analysis'] = 10000  # Safe default
        system_info['max_samples_ad'] = 5000  # Safe default
        
        return system_info
        
    except Exception as e:
        print(f"Warning: Could not detect system info: {e}")
        # Return safe default values
        return {
            'os': 'Unknown',
            'cpu_count': 2,
            'memory_gb': 4.0,
            'python_version': '3.8',
            'recommended_backend': 'threading',
            'recommended_workers': 2,
            'recommended_max_samples': 10000,
            'recommended_max_ad_samples': 5000,
            'max_workers': 2,
            'max_samples_analysis': 10000,
            'max_samples_ad': 5000
        }

# 3. DEFAULT_PARAMS with enhanced error handling
try:
    SYSTEM_INFO = get_system_info()
except:
    SYSTEM_INFO = {
        'max_workers': 2,
        'memory_gb': 4.0,
        'max_samples_analysis': 10000,
        'max_samples_ad': 5000
    }

def print_system_recommendations():
    """Print system optimization recommendations - display information only"""
    print("\n" + "="*60)
    print("[SYSTEM] ANALYSIS AND RECOMMENDATIONS")
    print("="*60)
    print(f"OS: {SYSTEM_INFO.get('os', 'Unknown')}")
    print(f"CPU Cores: {SYSTEM_INFO.get('cpu_count', 'Unknown')}")
    print(f"Memory: {SYSTEM_INFO.get('memory_gb', 4.0):.1f} GB")
    print(f"Python: {SYSTEM_INFO.get('python_version', 'Unknown')}")
    print("\n[CHART] RECOMMENDED SETTINGS (for reference only):")
    print(f"  * Parallel Backend: {SYSTEM_INFO.get('recommended_backend', 'threading')}")
    print(f"  * Recommended Workers: {SYSTEM_INFO.get('recommended_workers', 2)}")
    print(f"  * Recommended Max Samples (Analysis): {SYSTEM_INFO.get('recommended_max_samples', 10000):,}")
    print(f"  * Recommended Max Samples (AD): {SYSTEM_INFO.get('recommended_max_ad_samples', 5000):,}")
    print("\n[CONFIG] ACTUAL SETTINGS (safe defaults):")
    print(f"  * Max Workers: {SYSTEM_INFO.get('max_workers', 2)}")
    print(f"  * Max Samples (Analysis): {SYSTEM_INFO.get('max_samples_analysis', 10000):,}")
    print(f"  * Max Samples (AD): {SYSTEM_INFO.get('max_samples_ad', 5000):,}")
    print("\nNote: Using conservative defaults for stability. Adjust in config.py if needed.")
    print("="*60 + "\n")
        
        
# ── AD Method Definitions (X-VALUE BASED ONLY - 2025 UPDATE) ─────────────────────────────────────────
# Only includes methods that use descriptor values (X) without requiring model predictions
AD_METHODS = {
    'knn_distance': {
        'priority': 1, 
        'type': 'distance', 
        'description': 'k-Nearest Neighbors Distance',
        'regulatory_approved': True,
        'x_based_only': True,
        'references': ['OECD (2007)', 'ECHA (2016)']
    },
    'euclidean_distance': {
        'priority': 2,
        'type': 'distance',
        'description': 'Euclidean Distance to Centroid',
        'regulatory_approved': True,
        'x_based_only': True,
        'references': ['OECD (2007)']
    },
    'descriptor_range': {
        'priority': 3,
        'type': 'range',
        'description': 'Descriptor Range Check',
        'regulatory_approved': True,
        'x_based_only': True,
        'references': ['OECD (2007)']
    }
}

# Methods removed (require model predictions):
# - standardized_residuals: needs y_pred
# - williams_plot: needs y_pred for residuals
# - dmodx: for PCA models specifically

# ── Non-Regulatory AD Methods (RESEARCH USE ONLY) ─────────────────────────────────────────
# These methods are not mentioned in regulatory guidance documents
NON_REGULATORY_AD_METHODS = {
    'mahalanobis': {
        'type': 'statistical',
        'description': 'Mahalanobis Distance',
        'regulatory_approved': False,
        'note': 'Not in FDA/OECD/ECHA guidance'
    },
    'pca_hotelling': {
        'type': 'statistical',
        'description': 'PCA Hotelling T²',
        'regulatory_approved': False,
        'note': 'Similar to DModX but not explicitly mentioned'
    },
    'isolation_forest': {
        'type': 'ml',
        'description': 'Isolation Forest',
        'regulatory_approved': False,
        'note': 'Machine learning method - not in regulatory guidance'
    },
    'local_outlier_factor': {
        'type': 'density',
        'description': 'Local Outlier Factor',
        'regulatory_approved': False,
        'note': 'Density-based method - not in regulatory guidance'
    },
    'kernel_density': {
        'type': 'density',
        'description': 'Kernel Density Estimation',
        'regulatory_approved': False,
        'note': 'Statistical method - not in regulatory guidance'
    }
}

# ── AD Coverage Modes ─────────────────────────────────────────
# Based on comprehensive literature review and regulatory guidelines
AD_COVERAGE_MODES = {
    'strict': {  # Keep existing content
        'name': 'Ultra-Strict (Regulatory)',
        'coverage_standards': {
            'excellent': (0.90, 0.95),
            'good': (0.80, 0.90),
            'acceptable': (0.70, 0.80),
            'risky': (0.60, 0.70),
            'poor': (0.00, 0.60),
            'overfitted': (0.95, 1.01)
        },
        'reference': 'ICH M7 guideline (>90% target) - Regulatory requirement for mutagenicity assessment',
        'citations': [
            'ICH M7(R1) (2017) - Assessment and control of DNA reactive (mutagenic) impurities',
            'FDA Guidance (2018) - M7(R1) Assessment and Control of DNA Reactive Impurities'
        ]
    },
    
    'flexible': {  # Keep existing content
        'name': 'Scientific Consensus',
        'coverage_standards': {
            'excellent': (0.80, 0.90),
            'good': (0.70, 0.80),
            'acceptable': (0.60, 0.70),
            'moderate': (0.50, 0.60),
            'limited': (0.40, 0.50),
            'poor': (0.00, 0.40),
            'overfitted': (0.90, 1.01)
        },
        'reference': 'Sahigara et al. (2012), Roy et al. (2015) - Practical AD implementation',
        'citations': [
            'Sahigara et al. (2012) Molecules 17(5):4791-4810 - 60-80% coverage as practical range',
            'Roy et al. (2015) Chemometr Intell Lab Syst 145:22-29 - Standardized approach with 3sigma',
            'Sheridan (2012) J Chem Inf Model 52(3):814-823 - 75% coverage as optimal trade-off',
            'Klingspohn et al. (2017) J Cheminform 9:44 - 70-80% balances accuracy and coverage'
        ]
    },
    
    'adaptive': {  # Modified content - added specific references
        'name': 'Context-Dependent',
        'coverage_standards': {
            'research': {
                'excellent': (0.70, 0.85),
                'good': (0.60, 0.70),
                'acceptable': (0.50, 0.60),
                'limited': (0.00, 0.50)
            },
            'regulatory': {
                'excellent': (0.85, 0.95),
                'good': (0.75, 0.85),
                'acceptable': (0.65, 0.75),
                'poor': (0.00, 0.65)
            },
            'screening': {
                'excellent': (0.60, 0.80),
                'good': (0.50, 0.60),
                'acceptable': (0.40, 0.50),
                'limited': (0.00, 0.40)
            }
        },
        'reference': 'Application-specific thresholds based on literature consensus',
        'citations': [
            # Research mode thresholds (70-85%)
            'Weaver & Gleeson (2008) J Cheminform 41:1-7 - 70% coverage for lead optimization',
            'Tetko et al. (2008) Drug Discov Today 13:157-163 - Context-dependent AD thresholds',
            'Dragos et al. (2009) J Chem Inf Model 49:1762-1776 - 75% for research applications',
            
            # Regulatory mode thresholds (65-95%)
            'OECD (2014) Guidance Document No. 69 - Regulatory flexibility principle',
            'ECHA (2016) Practical Guide 5 - Read-across assessment framework',
            'Jaworska et al. (2005) Altern Lab Anim 33:445-459 - Regulatory AD requirements',
            
            # Screening mode thresholds (40-80%)
            'Tropsha & Golbraikh (2007) Curr Pharm Des 13:3494-3504 - VS applications 50-60%',
            'Hanser et al. (2016) J Cheminform 8:27 - Formal framework for context-dependent AD',
            'Sushko et al. (2010) J Comput Aided Mol Des 24:251-264 - Screening AD criteria'
        ]
    },

    'modern_2025': {  # 2025 latest standards
        'name': '2025 Enhanced Coverage Standards',
        'coverage_standards': {
            'excellent': (0.75, 0.90),    # CATMoS model coverage standards
            'good': (0.65, 0.75),         # Industrial chemical coverage
            'acceptable': (0.55, 0.65),   # Military chemical coverage
            'moderate': (0.45, 0.55),     # Specialized domain coverage
            'limited': (0.35, 0.45),      # Novel chemical space
            'poor': (0.00, 0.35),         # Insufficient coverage
            'overfitted': (0.90, 1.01)    # Over-training detection
        },
        'reference': '2025 Chemical Space Coverage Studies - Enhanced AD Methods',
        'citations': [
            'Puhl et al. (2024) - Evaluating applicability domain of acute toxicity QSAR models',
            'EPA (2024) - QSAR models for military and industrial chemical risk assessment',
            'OECD QSAR Toolbox (2024) - Enhanced applicability domain methods',
            'CATMoS (2024) - Collaborative Acute Toxicity Modeling Suite broad coverage',
            'Chemical Space Coverage Study (2024) - Half of organic chemicals predictable',
            'Automated QSAR Framework (2024) - Integrated AD assessment with uncertainty'
        ],
        'features': [
            'Enhanced chemical space coverage for industrial/military chemicals',
            'Tissue-specific QSAR model AD assessment',
            'CATMoS integration for acute toxicity models',
            'Automated uncertainty quantification',
            'REACH-compliant standardization',
            'Real-time regulatory compliance checking'
        ]
    }
}

# Default to 'strict' for backward compatibility
AD_COVERAGE_STANDARDS = AD_COVERAGE_MODES['strict']['coverage_standards']

# ── Tanimoto Similarity (ultra-strict) ─────────────────────────────────────────
# Based on chemical similarity principles and empirical validation
TANIMOTO_STANDARDS = {
    'excellent': (0.00, 0.20),  # Very diverse chemical space
    'good': (0.20, 0.40),       # Good diversity
    'acceptable': (0.40, 0.60), # Moderate similarity
    'risky': (0.60, 0.75),      # High similarity - potential bias
    'dangerous': (0.75, 1.01)   # Very high similarity - overfitting risk
}
# References:
# - Maggiora et al. (2014) J Med Chem 57(8):3186-3204 - Activity cliffs and similarity paradox
# - Martin et al. (2002) J Med Chem 45(19):4350-4358 - Tanimoto coefficient in drug discovery

# ── Reliability Scoring Settings ─────────────────────────────────────────
# Enable/disable AD-based reliability scoring system
RELIABILITY_SCORING_CONFIG = {
    'enabled': False,  # Default: disabled for performance
    'mode': 'adaptive',  # Default mode when enabled
    'save_reports': True,  # Save reliability reports when enabled
    'methods_weights': {
        # Weights based on method reliability for X-value based methods only
        'strict': {
            'knn_distance': 0.45,           # OECD approved distance method
            'euclidean_distance': 0.35,     # Simple but effective
            'descriptor_range': 0.20        # Most conservative
        },
        'flexible': {
            'knn_distance': 0.45,
            'euclidean_distance': 0.35,
            'descriptor_range': 0.20
        },
        'adaptive': {
            'knn_distance': 0.50,
            'euclidean_distance': 0.35,
            'descriptor_range': 0.15
        }
    }
}

# Memory Optimization Constants
CHUNK_SIZE = 1000
MAX_MEMORY_MB = 4096

# Standalone descriptor definitions for qsar_analysis module
# These match the main config.py but allow independent usage

# Full list of 49 chemical descriptors
CHEMICAL_DESCRIPTORS_STANDALONE = [
        'MolWeight',                    # 0 - Molecular weight
        'MolLogP',                      # 1 - Octanol-water partition coefficient
        'MolMR',                        # 2 - Molecular refractivity
        'TPSA',                         # 3 - Topological polar surface area
        'NumRotatableBonds',            # 4 - Number of rotatable bonds
        'HeavyAtomCount',               # 5 - Number of heavy atoms
        'NumHAcceptors',                # 6 - Number of H-bond acceptors
        'NumHDonors',                   # 7 - Number of H-bond donors
        'NumHeteroatoms',               # 8 - Number of heteroatoms
        'NumValenceElectrons',          # 9 - Number of valence electrons
        'NHOHCount',                    # 10 - Number of NH and OH groups
        'NOCount',                      # 11 - Number of N and O atoms
        'RingCount',                    # 12 - Number of rings
        'NumAromaticRings',             # 13 - Number of aromatic rings
        'NumSaturatedRings',            # 14 - Number of saturated rings
        'NumAliphaticRings',            # 15 - Number of aliphatic rings
        'LabuteASA',                    # 16 - Labute's Approximate Surface Area
        'BalabanJ',                     # 17 - Balaban's J index
        'BertzCT',                      # 18 - Bertz complexity index
        'Ipc',                          # 19 - Information content index
        'kappa_Series[1-3]_ind',        # 20 - Kappa shape indices
        'Chi_Series[1-98]_ind',           # 21 - Chi connectivity indices
        'Phi',                          # 22 - Flexibility index
        'HallKierAlpha',                # 23 - Hall-Kier alpha value
        'NumAmideBonds',                # 24 - Number of amide bonds
        'FractionCSP3',                 # 25 - Fraction of sp3 carbons
        'NumSpiroAtoms',                # 26 - Number of spiro atoms
        'NumBridgeheadAtoms',           # 27 - Number of bridgehead atoms
        'PEOE_VSA_Series[1-14]_ind',    # 28 - PEOE VSA descriptors
        'SMR_VSA_Series[1-10]_ind',     # 29 - SMR VSA descriptors
        'SlogP_VSA_Series[1-12]_ind',   # 30 - SlogP VSA descriptors
        'EState_VSA_Series[1-11]_ind',  # 31 - EState VSA descriptors
        'VSA_EState_Series[1-10]_ind',  # 32 - VSA EState descriptors
        'MQNs',                         # 33 - Molecular Quantum Numbers
        'AUTOCORR2D',                   # 34 - 2D autocorrelation
        'BCUT2D',                       # 35 - 2D BCUT descriptors
        'Asphericity',                  # 36 - Molecular asphericity
        'PBF',                          # 37 - Plane of best fit
        'RadiusOfGyration',             # 38 - Radius of gyration
        'InertialShapeFactor',          # 39 - Inertial shape factor
        'Eccentricity',                 # 40 - Molecular eccentricity
        'SpherocityIndex',              # 41 - Spherocity index
        'PMI_series[1-3]_ind',          # 42 - Principal moments of inertia
        'NPR_series[1-2]_ind',          # 43 - Normalized principal moments
        'AUTOCORR3D',                   # 44 - 3D autocorrelation
        'RDF',                          # 45 - Radial distribution function
        'MORSE',                        # 46 - MoRSE descriptors
        'WHIM',                         # 47 - WHIM descriptors
        'GETAWAY'                       # 48 - GETAWAY descriptors
]

# Try to import from main config, but use standalone if not available
try:
    import sys
    import os
    main_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if main_config_path not in sys.path:
        sys.path.insert(0, main_config_path)
    from config import CHEMICAL_DESCRIPTORS
    # Use the main config descriptors if available
    if isinstance(CHEMICAL_DESCRIPTORS, dict) and 'selection' in CHEMICAL_DESCRIPTORS:
        DESCRIPTOR_NAMES_FULL = CHEMICAL_DESCRIPTORS['selection']
    else:
        DESCRIPTOR_NAMES_FULL = CHEMICAL_DESCRIPTORS
    print("[INFO] Using descriptor list from main config.py")
except ImportError:
    # Use standalone descriptors
    DESCRIPTOR_NAMES_FULL = CHEMICAL_DESCRIPTORS_STANDALONE
    print("[INFO] Using standalone descriptor list from qsar_analysis/config.py")

# Fast subset (7 descriptors) - Most important for solubility prediction
DESCRIPTOR_NAMES_FAST = [
    'MolWeight', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 
    'NumRotatableBonds', 'TPSA', 'NumAromaticRings'
]

# Descriptors that need normalization
DESCRIPTORS_NEED_NORMALIZATION = [
    'Ipc', 'PMI_series[1-3]_ind', 'RDF', 'MORSE', 'WHIM', 'GETAWAY'
]

# Backward compatibility
DESCRIPTOR_NAMES = DESCRIPTOR_NAMES_FULL

# Store currently active descriptors (can be modified at runtime)
ACTIVE_DESCRIPTORS = DESCRIPTOR_NAMES_FAST  # Default to fast mode

# Plot Settings (2025 update - no grid)
PLOT_SETTINGS = {
    'figure_dpi': 300,
    'figure_dpi_high': 400,
    'style': 'seaborn-v0_8-white',  # Changed from whitegrid to white (no grid)
    'palette': 'husl'
}

DEFAULT_PARAMS = {
    # Core parameters
    'random_state': 42,
    'test_size': 0.2,
    'n_components_pca': 50,
    
    # Performance and sampling - use safe defaults
    'performance_mode': False,
    'max_samples_analysis': 10000,  # Safe default
    'max_samples_ad': 5000,  # Safe default
    'min_samples_ad': 1000,
    'enable_sampling': True,
    'sampling_ratio': 0.1,
    
    # AD analysis
    'ad_mode': 'flexible',
    'ad_analysis_mode': 'all',
    'enable_ad_performance': True,
    'enable_reliability_scoring': False,
    'use_regulatory_methods_only': True,  # Use only regulatory approved methods
    
    # Algorithm parameters
    'knn_neighbors': 10,
    'isolation_forest_estimators': 50,  # Non-regulatory method
    'lof_neighbors': 20,  # Non-regulatory method
    'kde_bandwidth': 'scott',  # Non-regulatory method
    
    # Technical parameters
    'float_precision': 'float32',
    'fp_bits': 2048,  # Fingerprint size - consistent across all methods
    'num_descriptors': 7,  # Number of descriptors to use (7 for fast, 49 for full)
    'n_jobs': 2,  # Fixed safe default
    'parallel_timeout': 30,
    'chunk_size': 100,
    'cache_enabled': True,
    
    # API parameters
    'api_batch_size': 50,
    'api_max_workers': 10,
}

# AD Method-specific maximum samples (X-value based methods only)
AD_METHOD_MAX_SAMPLES = {
    'knn_distance': 5000,  # Computationally intensive
    'euclidean_distance': 50000,  # Fast calculation
    'descriptor_range': 50000,  # Fast calculation
}

# Non-regulatory methods sample limits (for research)
NON_REGULATORY_METHOD_MAX_SAMPLES = {
    'mahalanobis': 10000,
    'pca_hotelling': 8000,
    'isolation_forest': 5000,
    'local_outlier_factor': 3000,
    'kernel_density': 5000
}

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

# Memory limit configuration
MAX_MEMORY_MB = 4096  # Limited to 4GB