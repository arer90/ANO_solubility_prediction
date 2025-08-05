"""
QSAR Data Splitters Module - Complete Implementation (v10.0)
===========================================================

Comprehensive framework implementing ten distinct data splitting strategies 
specifically optimized for aqueous solubility prediction in QSAR modeling.

This module provides scientifically rigorous splitting methods based on 
peer-reviewed literature, addressing different aspects of chemical space 
representation, molecular diversity, and experimental challenges inherent 
in solubility measurements.

Core References:
----------------
1. Avdeef, A. (2020) Multi-lab intrinsic solubility measurement reproducibility. 
   ADMET & DMPK 8:29-77
2. Bemis, G.W. & Murcko, M.A. (1996) The properties of known drugs. 1. Molecular frameworks. 
   J Med Chem 39:2887-2893
3. Butina, D. (1999) Unsupervised data base clustering based on daylight's fingerprint. 
   J Chem Inf Comput Sci 39:747-750
4. Delaney, J.S. (2004) ESOL: Estimating aqueous solubility directly from molecular structure. 
   J Chem Inf Comput Sci 44:1000-1005
5. Guha, R. & Van Drie, J.H. (2008) Structure-activity landscape index. 
   J Chem Inf Model 48:646-658
6. Kennard, R.W. & Stone, L.A. (1969) Computer aided design of experiments. 
   Technometrics 11:137-148
7. Llinas, A. et al. (2008) Solubility challenge. 
   J Chem Inf Model 48:1289-1303
8. Palmer, D.S. & Mitchell, J.B. (2014) Is experimental data quality the limiting factor. 
   Mol Pharm 11:2962-2972
9. Sheridan, R.P. (2013) Time-split cross-validation. 
   J Chem Inf Model 53:783-790
10. Tropsha, A. (2010) Best practices for QSAR model development. 
    Mol Inform 29:476-488

Author: [Your name]
Version: 10.0
Date: 2024
License: MIT
"""

import os
import platform
import psutil
import multiprocessing
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr, spearmanr, ks_2samp, entropy, gaussian_kde
from scipy.optimize import differential_evolution
from typing import Dict, List, Tuple, Optional, Union, Set, Any
import logging
from collections import defaultdict, Counter
import json
import warnings
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
import time
from datetime import datetime

# ===== RDKit Import Section =====
print("=== splitters.py loading ===")

# config.py import 시도, 실패시 기본값 사용
try:
    from .config import SYSTEM_INFO, print_system_recommendations
except ImportError:
    print("[WARNING] config.py not found, using default values")
    
    # 기본 SYSTEM_INFO
    SYSTEM_INFO = {
        'os': 'Unknown',
        'cpu_count': 2,
        'memory_gb': 4.0,
        'python_version': '3.8',
        'max_workers': 2,
        'max_samples_analysis': 10000,
        'max_samples_ad': 5000
    }
    
    # 기본 print_system_recommendations 함수
    def print_system_recommendations():
        print("\n" + "="*60)
        print("🖥️  SYSTEM INFO (Default)")
        print("="*60)
        print("Using default system configuration")
        print(f"Max Workers: {SYSTEM_INFO['max_workers']}")
        print(f"Max Samples: {SYSTEM_INFO['max_samples_analysis']:,}")
        print("="*60 + "\n")

# ===== RDKit Import Section =====
print("=== splitters.py loading ===")
print(f"Version: 10.0 - Complete Scientific Implementation")
print(f"Date: {datetime.now().isoformat()}")

# Initialize flags
RDKIT_AVAILABLE = False
RDKIT_ADVANCED = False
FILTER_CATALOG_AVAILABLE = False

# Default functions for when RDKit is not available
def BertzCT(mol): return 0
def BalabanJ(mol): return 0
FilterCatalog = None
FilterCatalogParams = None
rdMolTransforms = None
EState = None
EState_VSA = None

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, Crippen, Lipinski
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import Descriptors, rdMolDescriptors, MolSurf
    from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
    
    # Test RDKit functionality
    mol = Chem.MolFromSmiles("CC")
    if mol is not None:
        RDKIT_AVAILABLE = True
        print("RDKit successfully imported and working")
        
        # Set BertzCT and BalabanJ
        if hasattr(Descriptors, 'BertzCT'):
            BertzCT = Descriptors.BertzCT
        if hasattr(Descriptors, 'BalabanJ'):
            BalabanJ = Descriptors.BalabanJ
        
        # Try FilterCatalog
        try:
            from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
            FILTER_CATALOG_AVAILABLE = True
            print("FilterCatalog available")
        except ImportError:
            print("FilterCatalog not available")
        
        # Try advanced features
        try:
            from rdkit.Chem import rdMolTransforms
            from rdkit.Chem.EState import EState, EState_VSA
            RDKIT_ADVANCED = True
            print("Advanced RDKit features available")
        except ImportError:
            print("Some advanced RDKit features not available")
    else:
        print("RDKit imported but MolFromSmiles failed")
        RDKIT_AVAILABLE = False
        
except ImportError as e:
    print(f"RDKit import failed: {e}")
    print("Advanced splitting methods will be disabled")
    print("Please install RDKit: pip install rdkit-pypi")
    RDKIT_AVAILABLE = False

# Final status
print(f"\n=== RDKit Status ===")
print(f"RDKIT_AVAILABLE: {RDKIT_AVAILABLE}")
print(f"RDKIT_ADVANCED: {RDKIT_ADVANCED}")
print(f"FILTER_CATALOG_AVAILABLE: {FILTER_CATALOG_AVAILABLE}")
print("=== splitters.py loaded ===\n")

# ===== Constants and Default Parameters =====

DEFAULT_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': True
}

# Splitting method abbreviations
SPLIT_ABBREVIATIONS = {
    'random': 'rm',
    'scaffold': 'sc',
    'chemical_space_coverage': 'cs',
    'cluster': 'cl',
    'physchem': 'pc',
    'activity_cliff': 'ac',
    'solubility_aware': 'sa',
    'time_series': 'ti',
    'ensemble': 'en',
    'test_only': 'to'
}

# ===== Data Classes =====

@dataclass
class SolubilityContext:
    """
    Context for solubility measurements following Avdeef (2020) guidelines.
    
    Attributes:
        temperature: Temperature in Celsius (default: 25°C)
        pH: pH value (default: 7.4, physiological)
        ionic_strength: Ionic strength in M (default: 0.15 M)
        buffer: Buffer system used
        measurement_method: Method used for measurement
        equilibration_time: Time in hours for equilibration
    """
    temperature: float = 25.0
    pH: float = 7.4
    ionic_strength: float = 0.15
    buffer: str = "phosphate"
    measurement_method: str = "shake-flask"
    equilibration_time: int = 24
    
    def get_uncertainty_factor(self) -> float:
        """
        Calculate measurement uncertainty based on experimental conditions.
        
        Following Palmer & Mitchell (2014) and Avdeef (2020).
        
        Returns:
            float: Uncertainty factor between 0 and 1
        """
        base_uncertainty = 0.1
        
        # Temperature deviation from standard
        temp_factor = abs(self.temperature - 25.0) * 0.01
        
        # pH deviation from neutral
        ph_factor = abs(self.pH - 7.0) * 0.05
        
        # Method-specific uncertainties
        method_uncertainties = {
            "shake-flask": 0.1,
            "turbidimetric": 0.15,
            "potentiometric": 0.08,
            "HPLC": 0.12,
            "CheqSol": 0.06
        }
        method_factor = method_uncertainties.get(self.measurement_method, 0.2)
        
        total_uncertainty = base_uncertainty + temp_factor + ph_factor + method_factor
        
        return min(total_uncertainty, 1.0)


@dataclass
class SplitReport:
    """Container for split method report data"""
    method: str
    references: List[str]
    criteria: Dict[str, Any]
    success: bool
    statistics: Dict[str, Any]
    failed_smiles: List[Tuple[int, str, str]]
    error: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class AdvancedDataSplitter:
    """
    Advanced data splitting strategies for QSAR solubility prediction.
    
    Implements ten complementary splitting strategies, each addressing specific
    aspects of model evaluation and generalization.
    """
    
    # Split evaluation criteria with scientific justification
    SPLIT_CRITERIA = {
        'random': {
            'min_train_size': 10,
            'min_test_size': 5,
            'max_class_imbalance': 0.9,
            'stratification_threshold': 10,
            'solubility_bins': [-np.inf, -6, -4, -2, 0, np.inf],  # Log S units
            'bin_labels': ['very_poor', 'poor', 'moderate', 'good', 'excellent'],
            'reference': 'Sheridan (2013) J Chem Inf Model 53:783-790'
        },
        'chemical_space_coverage': {
            'min_train_size': 10,
            'min_test_size': 5,
            'min_coverage_ratio': 0.8,
            'distance_metric': 'tanimoto',
            'fingerprint_type': 'morgan',
            'fingerprint_size': 2048,
            'radius': 3,
            'structure_weight': 0.7,  # α in distance metric
            'property_weight': 0.3,   # 1-α
            'reference': 'Kennard & Stone (1969) Technometrics 11:137-148'
        },
        'cluster': {
            'min_train_size': 10,
            'min_test_size': 5,
            'min_inter_distance': 0.5,
            'optimal_clusters': (5, 20),
            'clustering_method': 'ward',
            'distance_threshold': 0.7,
            'reference': 'Butina (1999) J Chem Inf Comput Sci 39:747-750'
        },
        'physchem': {
            'min_train_size': 10,
            'min_test_size': 5,
            'property_coverage_threshold': 0.75,
            'correlation_threshold': 0.7,
            'optimal_clusters': (5, 20),
            'key_properties': [
                'MolLogP', 'MolWt', 'TPSA', 'NumHBD', 'NumHBA',
                'NumRotatableBonds', 'NumAromaticRings', 'FractionCsp3', 'BertzCT'
            ],
            'property_ranges': {
                'MolWt': (0, 500),      # Lipinski Ro5
                'MolLogP': (-0.4, 5.6), # Lipinski Ro5
                'NumHBD': (0, 5),       # Lipinski Ro5
                'NumHBA': (0, 10)       # Lipinski Ro5
            },
            'reference': 'Delaney (2004) J Chem Inf Comput Sci 44:1000-1005'
        },
        'activity_cliff': {
            'min_train_size': 10,
            'min_test_size': 5,
            'similarity_threshold': 0.7,
            'activity_difference_threshold': 1.0,  # 10-fold change
            'max_cliff_ratio': 0.3,
            'cliff_detection_method': 'SALI',
            'smoothing_factor': 0.001,  # ε in SALI formula
            'reference': 'Guha & Van Drie (2008) J Chem Inf Model 48:646-658'
        },
        'solubility_aware': {
            'min_train_size': 10,
            'min_test_size': 5,
            'measurement_uncertainty_threshold': 0.5,
            'polymorphism_risk_threshold': 0.7,
            'aggregation_risk_threshold': 0.6,
            'ph_sensitivity_threshold': 1.0,
            'temperature_sensitivity_threshold': 0.1,
            'difficulty_weights': {
                'measurement_uncertainty': 0.25,
                'polymorphism_risk': 0.20,
                'aggregation_tendency': 0.15,
                'ph_sensitivity': 0.20,
                'temperature_sensitivity': 0.20
            },
            'reference': 'Avdeef (2020) ADMET & DMPK 8:29-77'
        },
        'ensemble': {
            'min_train_size': 10,
            'min_test_size': 5,
            'min_agreement_ratio': 0.6,
            'weight_bounds': (0.1, 0.4),
            'weight_distribution': {
                'chemical_space': 0.20,
                'physchem': 0.25,
                'activity_cliff': 0.20,
                'uncertainty': 0.20,
                'solubility_aware': 0.15
            },
            'optimization_method': 'differential_evolution',
            'de_params': {
                'popsize': 10,
                'maxiter': 50,
                'seed': 42
            },
            'cv_folds': 5,
            'reference': 'Sheridan (2019) J Chem Inf Model 59:1645-1649'
        },
        'time_series': {
            'min_train_size': 10,
            'min_test_size': 5,
            'preserve_order': True,
            'reference': 'Sheridan (2013) J Chem Inf Model 53:783-790'
        },
        'test_only': {
            'reference': 'Tropsha (2010) Mol Inform 29:476-488'
        }
    }
    
    def __init__(self, 
                 output_dir: Union[str, Path], 
                 random_state: int = DEFAULT_PARAMS['random_state'], 
                 solubility_context: Optional[SolubilityContext] = None,
                 verbose: bool = DEFAULT_PARAMS['verbose']):
        """
        Initialize the Advanced Data Splitter.
        
        Args:
            output_dir: Directory for saving split results
            random_state: Random seed for reproducibility
            solubility_context: Experimental context for measurements
            verbose: Enable detailed logging
        """
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.solubility_context = solubility_context or SolubilityContext()
        self.verbose = verbose
        
        # Split method abbreviations
        self.split_abbrev = SPLIT_ABBREVIATIONS
        
        # Validate output directory
        if not self.output_dir.parent.exists():
            raise ValueError(f"Parent directory does not exist: {self.output_dir.parent}")
        
        # Setup logging
        self._setup_logging()
        
        # Storage for split reports
        self.split_reports = {}
        self.version = "10.0"
        
        # Create subdirectories
        self._create_split_directories()
        
        # Initialize components if RDKit available
        if RDKIT_AVAILABLE:
            self._initialize_rdkit_components()
        
        # Cache for expensive calculations
        self._cache = {}
        
        # Performance tracking
        self.timing_info = {}
    
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    
    def _initialize_rdkit_components(self):
        """Initialize RDKit-dependent components"""
        try:
            self._init_descriptor_calculator()
        except Exception as e:
            self.logger.warning(f"Descriptor calculator init failed: {e}")
            self.descriptor_calc = None
        
        try:
            self._init_solubility_models()
        except Exception as e:
            self.logger.warning(f"Solubility models init failed: {e}")
        
        try:
            self._init_quality_filters()
        except Exception as e:
            self.logger.warning(f"Quality filters init failed: {e}")
            self.filter_catalog = None
    
    def _init_descriptor_calculator(self):
        """
        Initialize RDKit descriptor calculator.
        
        Based on descriptors identified as important for solubility prediction
        in Delaney (2004) and subsequent studies.
        """
        # Core descriptors for solubility
        core_descriptors = [
            'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors',
            'NumRotatableBonds', 'TPSA', 'LabuteASA', 'BertzCT',
            'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings',
            'RingCount', 'FractionCsp3', 'NumHeteroatoms',
            'Chi0', 'Chi1', 'Kappa1', 'Kappa2',
            'HallKierAlpha', 'BalabanJ'
        ]
        
        # Additional descriptors if available
        if RDKIT_ADVANCED:
            advanced_descriptors = [
                'MaxEStateIndex', 'MinEStateIndex',
                'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3',
                'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3',
                'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3'
            ]
        else:
            advanced_descriptors = []
        
        self.descriptor_names = core_descriptors + advanced_descriptors
        
        # Filter to only available descriptors
        available_descriptors = []
        for desc in self.descriptor_names:
            if hasattr(Descriptors, desc):
                available_descriptors.append(desc)
        
        try:
            self.descriptor_calc = MolecularDescriptorCalculator(available_descriptors)
            self.logger.info(f"Initialized descriptor calculator with {len(available_descriptors)} descriptors")
        except Exception:
            self.logger.warning("MolecularDescriptorCalculator not available")
            self.descriptor_calc = None
    
    def _init_solubility_models(self):
        """
        Initialize solubility prediction model coefficients.
        
        General Solubility Equation (GSE) from Delaney (2004).
        """
        self.gse_coefficients = {
            'logp_coeff': -1.05,
            'mp_coeff': -0.0095,
            'intercept': 1.22
        }
        
        # WATERNT-like atom contributions
        self.waternt_weights = {
            'aromatic_carbon': -0.3,
            'aliphatic_carbon': -0.5,
            'alcohol_oxygen': 1.5,
            'ether_oxygen': 0.8,
            'nitrogen': 1.0
        }
    
    def _init_quality_filters(self):
        """
        Initialize chemical quality filters.
        
        PAINS: Pan-Assay Interference Compounds
        BRENK: Unwanted functionality filters
        """
        if FILTER_CATALOG_AVAILABLE:
            try:
                params = FilterCatalogParams()
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
                self.filter_catalog = FilterCatalog(params)
                self.logger.info("Initialized chemical quality filters")
            except Exception as e:
                self.logger.warning(f"FilterCatalog initialization failed: {e}")
                self.filter_catalog = None
        else:
            self.filter_catalog = None
    
    def _create_split_directories(self):
        """Create directory structure for all split methods"""
        try:
            for split_type in ['train', 'test']:
                for method, abbrev in self.split_abbrev.items():
                    dir_path = self.output_dir / split_type / abbrev
                    dir_path.mkdir(parents=True, exist_ok=True)
                    
                    # Check permissions
                    if not os.access(dir_path, os.W_OK):
                        raise PermissionError(f"No write permission for {dir_path}")
            
            # Create reports directory
            self.reports_dir = self.output_dir / 'split_reports'
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Created directory structure in {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to create directories: {e}")
            raise
    
    def create_all_splits(self, 
                         name: str, 
                         smiles: List[str], 
                         targets: np.ndarray, 
                         is_test_only: bool = False,
                         measurement_metadata: Optional[Dict] = None) -> Dict:
        """
        Create all ten splitting strategies for the dataset.
        
        Args:
            name: Dataset name
            smiles: List of SMILES strings
            targets: Array of log S values
            is_test_only: If True, create only test set (no training)
            measurement_metadata: Additional experimental metadata
            
        Returns:
            Dictionary containing all split results
        """
        self.logger.info(f"\nCreating advanced splits for {name} (v{self.version})...")
        self.logger.info(f"Dataset size: {len(smiles)} molecules")
        
        # Validate input data
        validation_results = self._validate_input_data(smiles, targets)
        if not validation_results['is_valid']:
            raise ValueError(f"Data validation failed: {validation_results['errors']}")
        
        # Initialize report
        self.split_reports[name] = {
            'dataset_info': {
                'name': name,
                'total_samples': len(smiles),
                'is_test_only': is_test_only,
                'timestamp': datetime.now().isoformat(),
                'version': self.version,
                'solubility_context': asdict(self.solubility_context),
                'validation_results': validation_results,
                'solubility_statistics': self._calculate_solubility_statistics(targets),
                'measurement_metadata': measurement_metadata or {}
            },
            'splits': {}
        }
        
        splits = {}
        
        if is_test_only:
            # Only create test set
            splits['test_only'] = self._create_test_only_split(name, smiles, targets)
        else:
            # Create all splits
            self.logger.info("\n=== Creating all data splits ===")
            
            # 1. Basic splits (always available)
            self.logger.info("\n--- Basic splits (Always available) ---")
            
            # Random split
            start_time = time.time()
            splits['random'] = self._create_random_split(name, smiles, targets)
            self.timing_info['random'] = time.time() - start_time
            
            # Time series split
            start_time = time.time()
            splits['time_series'] = self._create_time_series_split(name, smiles, targets)
            self.timing_info['time_series'] = time.time() - start_time
            
            # 2. RDKit-dependent splits
            if RDKIT_AVAILABLE:
                self.logger.info("\n--- Advanced splits (RDKit-based) ---")
                
                # Scaffold split
                start_time = time.time()
                try:
                    splits['scaffold'] = self._create_scaffold_split(name, smiles, targets)
                    self.timing_info['scaffold'] = time.time() - start_time
                except Exception as e:
                    self.logger.error(f"Scaffold split failed: {e}")
                    splits['scaffold'] = None
                
                # Chemical space coverage split
                start_time = time.time()
                try:
                    splits['chemical_space_coverage'] = self._create_chemical_space_coverage_split(
                        name, smiles, targets
                    )
                    self.timing_info['chemical_space_coverage'] = time.time() - start_time
                except Exception as e:
                    self.logger.error(f"Chemical space coverage split failed: {e}")
                    splits['chemical_space_coverage'] = None
                
                # Cluster split
                start_time = time.time()
                try:
                    splits['cluster'] = self._create_cluster_split(name, smiles, targets)
                    self.timing_info['cluster'] = time.time() - start_time
                except Exception as e:
                    self.logger.error(f"Cluster split failed: {e}")
                    splits['cluster'] = None
                
                # Physicochemical split
                start_time = time.time()
                try:
                    splits['physchem'] = self._create_physchem_split(name, smiles, targets)
                    self.timing_info['physchem'] = time.time() - start_time
                except Exception as e:
                    self.logger.error(f"Physchem split failed: {e}")
                    splits['physchem'] = None
                
                # Activity cliff split
                start_time = time.time()
                try:
                    splits['activity_cliff'] = self._create_activity_cliff_split(
                        name, smiles, targets
                    )
                    self.timing_info['activity_cliff'] = time.time() - start_time
                except Exception as e:
                    self.logger.error(f"Activity cliff split failed: {e}")
                    splits['activity_cliff'] = None
                
                # Solubility-aware split
                start_time = time.time()
                try:
                    splits['solubility_aware'] = self._create_solubility_aware_split(
                        name, smiles, targets, measurement_metadata
                    )
                    self.timing_info['solubility_aware'] = time.time() - start_time
                except Exception as e:
                    self.logger.error(f"Solubility-aware split failed: {e}")
                    splits['solubility_aware'] = None
                
                # Ensemble split (needs other splits)
                start_time = time.time()
                try:
                    splits['ensemble'] = self._create_ensemble_split(
                        name, smiles, targets, splits
                    )
                    self.timing_info['ensemble'] = time.time() - start_time
                except Exception as e:
                    self.logger.error(f"Ensemble split failed: {e}")
                    splits['ensemble'] = None
                    
            else:
                self.logger.warning("\n--- RDKit not available ---")
                self.logger.info("To enable all splits, install RDKit: pip install rdkit-pypi")
                
                # Alternative splits without RDKit
                self.logger.info("\n--- Alternative splits (No RDKit required) ---")
                start_time = time.time()
                try:
                    splits['cluster'] = self._create_simple_cluster_split(name, smiles, targets)
                    self.timing_info['cluster'] = time.time() - start_time
                except Exception as e:
                    self.logger.error(f"Simple cluster split failed: {e}")
                    splits['cluster'] = None
        
        # Validate all splits
        self._validate_all_splits(splits, smiles, targets)
        
        # Save comprehensive report
        self._save_split_report(name)
        
        # Summary
        self._print_split_summary(splits)
        
        return splits
    
    def _validate_input_data(self, smiles: List[str], targets: np.ndarray) -> Dict:
        """
        Comprehensive validation of input data.
        
        Based on data quality considerations from Palmer & Mitchell (2014).
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Basic checks
        if len(smiles) != len(targets):
            validation_results['is_valid'] = False
            validation_results['errors'].append("SMILES and targets length mismatch")
        
        if len(smiles) < 20:
            validation_results['warnings'].append(
                "Small dataset size may lead to unreliable splits"
            )
        
        # Validate SMILES if RDKit available
        if RDKIT_AVAILABLE:
            invalid_smiles = []
            duplicates = []
            seen_smiles = {}
            
            for i, smi in enumerate(smiles):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    invalid_smiles.append((i, smi))
                else:
                    # Canonical SMILES for duplicate detection
                    canonical_smi = Chem.MolToSmiles(mol)
                    if canonical_smi in seen_smiles:
                        duplicates.append((i, smi, seen_smiles[canonical_smi]))
                    else:
                        seen_smiles[canonical_smi] = i
            
            if invalid_smiles:
                validation_results['warnings'].append(
                    f"{len(invalid_smiles)} invalid SMILES found"
                )
                validation_results['statistics']['invalid_smiles'] = invalid_smiles
            
            if duplicates:
                validation_results['warnings'].append(
                    f"{len(duplicates)} duplicate structures found"
                )
                validation_results['statistics']['duplicates'] = duplicates
        
        # Validate targets (solubility values)
        target_stats = {
            'min': float(np.min(targets)),
            'max': float(np.max(targets)),
            'mean': float(np.mean(targets)),
            'std': float(np.std(targets)),
            'outliers': []
        }
        
        # Check for outliers (typical log S range: -10 to 2)
        typical_range = (-10, 2)
        outliers = np.where((targets < typical_range[0]) | (targets > typical_range[1]))[0]
        if len(outliers) > 0:
            validation_results['warnings'].append(
                f"{len(outliers)} solubility values outside typical range {typical_range}"
            )
            target_stats['outliers'] = outliers.tolist()
        
        # Check for suspicious patterns
        if np.std(targets) < 0.1:
            validation_results['warnings'].append(
                "Very low variance in target values"
            )
        
        validation_results['statistics']['target_stats'] = target_stats
        
        return validation_results
    
    def _calculate_solubility_statistics(self, targets: np.ndarray) -> Dict:
        """
        Calculate comprehensive solubility statistics.
        
        Following solubility classification from Delaney (2004) and
        pharmaceutical standards.
        """
        molar_solubility = 10 ** targets  # Convert log S to molar
        
        stats = {
            'log_s': {
                'mean': float(np.mean(targets)),
                'median': float(np.median(targets)),
                'std': float(np.std(targets)),
                'min': float(np.min(targets)),
                'max': float(np.max(targets)),
                'q1': float(np.percentile(targets, 25)),
                'q3': float(np.percentile(targets, 75)),
                'iqr': float(np.percentile(targets, 75) - np.percentile(targets, 25))
            },
            'molar': {
                'mean': float(np.mean(molar_solubility)),
                'median': float(np.median(molar_solubility)),
                'geometric_mean': float(np.exp(np.mean(np.log(molar_solubility + 1e-10))))
            },
            'distribution': {
                'very_poor': int(np.sum(targets < -6)),      # < 1 μM
                'poor': int(np.sum((targets >= -6) & (targets < -4))),       # 1 μM - 100 μM
                'moderate': int(np.sum((targets >= -4) & (targets < -2))),   # 100 μM - 10 mM
                'good': int(np.sum((targets >= -2) & (targets < 0))),        # 10 mM - 1 M
                'excellent': int(np.sum(targets >= 0))       # > 1 M
            },
            'pharmaceutical_relevance': {
                'below_1uM': int(np.sum(targets < -6)),
                'drug_like': int(np.sum((targets >= -5) & (targets <= -1))),
                'highly_soluble': int(np.sum(targets > -1))
            },
            'bimodality_test': self._test_bimodality(targets)
        }
        
        return stats
    
    def _test_bimodality(self, data: np.ndarray) -> Dict:
        """
        Test for bimodal distribution using multiple methods.
        
        Bimodality coefficient (BC) from SAS/STAT documentation.
        """
        n = len(data)
        
        # Calculate moments
        m3 = np.sum((data - np.mean(data))**3) / n
        m4 = np.sum((data - np.mean(data))**4) / n
        m2 = np.var(data)
        
        # Skewness and kurtosis
        g1 = m3 / (m2 ** 1.5) if m2 > 0 else 0  # Skewness
        g2 = m4 / (m2 ** 2) - 3 if m2 > 0 else 0  # Excess kurtosis
        
        # Bimodality coefficient
        bc = (g1**2 + 1) / (g2 + 3) if g2 > -3 else 0
        
        # Find peaks using KDE
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        density = kde(x_range)
        
        # Find local maxima
        peaks = []
        for i in range(1, len(density)-1):
            if density[i] > density[i-1] and density[i] > density[i+1]:
                peaks.append(float(x_range[i]))
        
        return {
            'bimodality_coefficient': float(bc),
            'is_bimodal': bc > 0.555,  # Threshold for bimodality
            'skewness': float(g1),
            'kurtosis': float(g2),
            'n_peaks': len(peaks),
            'peak_locations': peaks[:5]  # Top 5 peaks
        }
    
    # ===== Split Method Implementations =====
    
    def _create_random_split(self, name: str, smiles: List[str], 
                            targets: np.ndarray) -> Optional[Dict]:
        """
        Create stratified random split.
        
        Implementation follows Sheridan (2013) recommendations for
        time-split cross-validation as baseline.
        
        Mathematical formulation:
        P(molecule i ∈ test | bin(log S_i) = b) = test_size for all bins b
        """
        method = 'random'
        criteria = self.SPLIT_CRITERIA[method]
        
        # Initialize report
        report = SplitReport(
            method=method,
            references=[criteria['reference']],
            criteria=criteria,
            success=True,
            statistics={},
            failed_smiles=[]
        )
        
        try:
            self.logger.info(f"Creating random split with stratification...")
            
            indices = np.arange(len(smiles))
            
            # Stratification by solubility bins
            solubility_bins = pd.cut(
                targets,
                bins=criteria['solubility_bins'],
                labels=criteria['bin_labels']
            )
            
            # Check if stratification is possible
            bin_counts = solubility_bins.value_counts()
            min_bin_count = bin_counts.min()
            
            if min_bin_count >= 2:
                # Stratified split
                train_idx, test_idx = train_test_split(
                    indices, 
                    test_size=DEFAULT_PARAMS['test_size'],
                    random_state=self.random_state,
                    stratify=solubility_bins
                )
                stratified = True
                self.logger.info("  [OK] Stratification successful")
            else:
                # Simple random split
                train_idx, test_idx = train_test_split(
                    indices, 
                    test_size=DEFAULT_PARAMS['test_size'],
                    random_state=self.random_state
                )
                stratified = False
                self.logger.warning("  ⚠ Stratification not possible, using simple random split")
            
            # Analyze distribution
            train_bins = solubility_bins[train_idx].value_counts()
            test_bins = solubility_bins[test_idx].value_counts()
            
            # Create split data
            split_data = {
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_smiles': [smiles[i] for i in train_idx],
                'test_smiles': [smiles[i] for i in test_idx],
                'train_targets': targets[train_idx],
                'test_targets': targets[test_idx],
                'stratified': stratified,
                'solubility_distribution': {
                    'train': train_bins.to_dict(),
                    'test': test_bins.to_dict()
                }
            }
            
            # Calculate quality metrics
            quality_metrics = self._calculate_split_quality_metrics(
                train_idx, test_idx, smiles, targets
            )
            split_data.update(quality_metrics)
            
            # Save split
            self._save_split(name, 'random', split_data)
            
            self.logger.info(f"  [OK] Random split: {len(train_idx)} train, {len(test_idx)} test")
            self.logger.info(f"    Stratified: {stratified}")
            self.logger.info(f"    KS statistic: {quality_metrics['ks_statistic']:.3f}")
            
            # Update report
            report.statistics = {
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'is_stratified': stratified,
                'solubility_balance': self._calculate_distribution_similarity(
                    train_bins, test_bins
                ),
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Random split failed: {str(e)}")
            report.success = False
            report.error = str(e)
            split_data = None
        
        self.split_reports[name]['splits'][method] = asdict(report)
        return split_data
    
    def _create_scaffold_split(self, name: str, smiles: List[str], 
                              targets: np.ndarray) -> Optional[Dict]:
        """
        Create scaffold-based split using Bemis-Murcko frameworks.
        
        Reference: Bemis & Murcko (1996) J Med Chem 39:2887-2893
        
        Algorithm:
        1. Extract Bemis-Murcko scaffold for each molecule
        2. Group molecules by scaffold
        3. Sort groups by size (descending)
        4. Assign groups to test until target size reached
        """
        method = 'scaffold'
        criteria = self.SPLIT_CRITERIA.get(method, {})
        
        report = SplitReport(
            method=method,
            references=['Bemis & Murcko (1996) J Med Chem 39:2887-2893'],
            criteria=criteria,
            success=True,
            statistics={},
            failed_smiles=[]
        )
        
        if not RDKIT_AVAILABLE:
            self.logger.error("Scaffold split requires RDKit")
            report.success = False
            report.error = 'RDKit not available'
            self.split_reports[name]['splits'][method] = asdict(report)
            return None
        
        try:
            self.logger.info("Creating scaffold split...")
            
            # Calculate scaffolds
            scaffolds = defaultdict(list)
            failed_molecules = []
            
            for i, smi in enumerate(smiles):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                        scaffold_smi = Chem.MolToSmiles(scaffold)
                        scaffolds[scaffold_smi].append(i)
                    else:
                        failed_molecules.append((i, smi, "Invalid SMILES"))
                        scaffolds['INVALID'].append(i)
                except Exception as e:
                    failed_molecules.append((i, smi, str(e)))
                    scaffolds['ERROR'].append(i)
            
            report.failed_smiles = failed_molecules
            
            # Sort scaffolds by size (number of molecules)
            scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)
            
            self.logger.info(f"  Found {len(scaffolds)} unique scaffolds")
            self.logger.info(f"  Largest scaffold contains {len(scaffold_sets[0])} molecules")
            
            # Distribute scaffolds using greedy algorithm
            train_idx = []
            test_idx = []
            test_size = DEFAULT_PARAMS['test_size']
            
            for scaffold_set in scaffold_sets:
                if len(test_idx) / len(smiles) < test_size:
                    test_idx.extend(scaffold_set)
                else:
                    train_idx.extend(scaffold_set)
            
            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)
            
            # Analyze scaffold distribution
            scaffold_analysis = self._analyze_scaffold_distribution(
                scaffolds, train_idx, test_idx
            )
            
            split_data = {
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_smiles': [smiles[i] for i in train_idx],
                'test_smiles': [smiles[i] for i in test_idx],
                'train_targets': targets[train_idx],
                'test_targets': targets[test_idx],
                'n_scaffolds': len(scaffolds),
                'scaffold_analysis': scaffold_analysis
            }
            
            # Calculate quality metrics
            quality_metrics = self._calculate_split_quality_metrics(
                train_idx, test_idx, smiles, targets
            )
            split_data.update(quality_metrics)
            
            self._save_split(name, 'scaffold', split_data)
            
            self.logger.info(f"  [OK] Scaffold split: {len(train_idx)} train, {len(test_idx)} test")
            self.logger.info(f"    Number of scaffolds: {len(scaffolds)}")
            self.logger.info(f"    Test scaffolds: {scaffold_analysis['n_test_scaffolds']}")
            
            report.statistics = {
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'n_scaffolds': len(scaffolds),
                'scaffold_analysis': scaffold_analysis,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Scaffold split failed: {str(e)}")
            report.success = False
            report.error = str(e)
            split_data = None
        
        self.split_reports[name]['splits'][method] = asdict(report)
        return split_data
    
    def _create_chemical_space_coverage_split(self, name: str, smiles: List[str], 
                                            targets: np.ndarray) -> Optional[Dict]:
        """
        Create chemical space coverage split using MaxMin algorithm.
        
        Reference: Kennard & Stone (1969) Technometrics 11:137-148
        
        Distance metric:
        D(i,j) = α·D_tanimoto(FP_i, FP_j) + (1-α)·D_euclidean(P_i, P_j)
        where α = 0.7 (structure weight)
        
        Algorithm:
        1. Initialize with molecule closest to property space median
        2. Iteratively add molecule with maximum minimum distance to test set
        3. Continue until target test size reached
        """
        method = 'chemical_space_coverage'
        criteria = self.SPLIT_CRITERIA[method]
        
        report = SplitReport(
            method=method,
            references=[criteria['reference']],
            criteria=criteria,
            success=True,
            statistics={},
            failed_smiles=[]
        )
        
        if not RDKIT_AVAILABLE:
            self.logger.error("Chemical space coverage split requires RDKit")
            report.success = False
            report.error = 'RDKit not available'
            self.split_reports[name]['splits'][method] = asdict(report)
            return None
        
        try:
            self.logger.info("Creating chemical space coverage split...")
            
            # Calculate fingerprints
            fingerprints = self._calculate_multiple_fingerprints(smiles)
            fps = fingerprints['morgan']
            report.failed_smiles = fingerprints['failed']
            
            # Calculate property matrix
            property_matrix = self._calculate_key_properties_batch(smiles)
            
            # Normalize properties
            scaler = StandardScaler()
            property_matrix_norm = scaler.fit_transform(property_matrix)
            
            # Initialize with molecule closest to median
            median_props = np.median(property_matrix_norm, axis=0)
            distances_to_median = cdist([median_props], property_matrix_norm)[0]
            start_idx = np.argmin(distances_to_median)
            
            test_idx = [start_idx]
            remaining_idx = list(range(len(smiles)))
            remaining_idx.remove(start_idx)
            
            # MaxMin selection
            target_test_size = int(DEFAULT_PARAMS['test_size'] * len(smiles))
            
            self.logger.info(f"  MaxMin selection targeting {target_test_size} test molecules...")
            
            while len(test_idx) < target_test_size and remaining_idx:
                if len(test_idx) % 50 == 0:
                    self.logger.info(f"    Progress: {len(test_idx)}/{target_test_size}")
                
                min_distances = []
                
                for idx in remaining_idx:
                    # Calculate distances to all test molecules
                    tanimoto_dists = []
                    prop_dists = []
                    
                    for t_idx in test_idx:
                        # Tanimoto distance
                        sim = DataStructs.TanimotoSimilarity(fps[idx], fps[t_idx])
                        tanimoto_dists.append(1 - sim)
                        
                        # Property distance
                        prop_dist = np.linalg.norm(
                            property_matrix_norm[idx] - property_matrix_norm[t_idx]
                        )
                        prop_dists.append(prop_dist)
                    
                    # Normalize property distances
                    prop_dists = np.array(prop_dists)
                    if np.max(prop_dists) > 0:
                        prop_dists = prop_dists / np.max(prop_dists)
                    
                    # Combined distance (weighted)
                    alpha = criteria['structure_weight']
                    combined_dists = [
                        alpha * t_dist + (1 - alpha) * p_dist
                        for t_dist, p_dist in zip(tanimoto_dists, prop_dists)
                    ]
                    
                    # Minimum distance to test set
                    min_distances.append(np.min(combined_dists))
                
                # Select molecule with maximum minimum distance
                max_idx = np.argmax(min_distances)
                selected_idx = remaining_idx[max_idx]
                
                test_idx.append(selected_idx)
                remaining_idx.remove(selected_idx)
            
            train_idx = np.array(remaining_idx)
            test_idx = np.array(test_idx)
            
            # Calculate coverage metrics
            coverage_metrics = self._calculate_comprehensive_coverage(
                fps, property_matrix_norm, train_idx, test_idx
            )
            
            split_data = {
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_smiles': [smiles[i] for i in train_idx],
                'test_smiles': [smiles[i] for i in test_idx],
                'train_targets': targets[train_idx],
                'test_targets': targets[test_idx],
                'coverage_metrics': coverage_metrics,
                'maxmin_params': {
                    'alpha': criteria['structure_weight'],
                    'start_molecule': int(start_idx)
                }
            }
            
            # Calculate quality metrics
            quality_metrics = self._calculate_split_quality_metrics(
                train_idx, test_idx, smiles, targets
            )
            split_data.update(quality_metrics)
            
            self._save_split(name, 'chemical_space_coverage', split_data)
            
            self.logger.info(f"  [OK] Chemical space coverage split: {len(train_idx)} train, {len(test_idx)} test")
            self.logger.info(f"    Overall coverage: {coverage_metrics['overall_coverage']:.2%}")
            
            report.statistics = {
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'coverage_metrics': coverage_metrics,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Chemical space coverage split failed: {str(e)}")
            report.success = False
            report.error = str(e)
            split_data = None
        
        self.split_reports[name]['splits'][method] = asdict(report)
        return split_data
    
    def _create_cluster_split(self, name: str, smiles: List[str], 
                             targets: np.ndarray, features: np.ndarray = None) -> Optional[Dict]:
        """
        Create cluster-based split using hierarchical clustering.
        
        Reference: Butina (1999) J Chem Inf Comput Sci 39:747-750
        
        Algorithm:
        1. Calculate molecular features (fingerprints or properties)
        2. Apply PCA if high-dimensional
        3. Determine optimal number of clusters using silhouette score
        4. Perform hierarchical clustering
        5. Assign maximally distant clusters to train/test
        """
        method = 'cluster'
        criteria = self.SPLIT_CRITERIA[method]
        
        report = SplitReport(
            method=method,
            references=[criteria['reference']],
            criteria=criteria,
            success=True,
            statistics={},
            failed_smiles=[]
        )
        
        try:
            self.logger.info("Creating cluster-based split...")
            
            # Get features
            if features is None:
                if RDKIT_AVAILABLE:
                    # Use fingerprints
                    fps, failed_fps = self._calculate_fingerprints_with_errors(smiles)
                    features = np.array([self._fp_to_numpy(fp) for fp in fps])
                    report.failed_smiles = failed_fps
                    self.logger.info(f"  Using Morgan fingerprints ({features.shape[1]} bits)")
                else:
                    # Use targets as 1D feature
                    features = targets.reshape(-1, 1)
                    self.logger.warning("  Using target values for clustering (RDKit not available)")
            
            # Dimensionality reduction if needed
            if features.shape[1] > 50:
                pca = PCA(n_components=50, random_state=self.random_state)
                features_reduced = pca.fit_transform(features)
                explained_var = sum(pca.explained_variance_ratio_)
                self.logger.info(f"  PCA: {features.shape[1]} → 50 dimensions ({explained_var:.1%} variance)")
            else:
                features_reduced = features
            
            # Determine optimal number of clusters
            n_clusters = self._determine_optimal_clusters(
                features_reduced, criteria['optimal_clusters']
            )
            self.logger.info(f"  Optimal clusters: {n_clusters}")
            
            # Hierarchical clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, 
                linkage=criteria['clustering_method']
            )
            cluster_labels = clustering.fit_predict(features_reduced)
            
            # Analyze clusters
            cluster_stats = self._analyze_clusters(
                features_reduced, cluster_labels, targets
            )
            
            # Create split by assigning distant clusters
            train_idx, test_idx = self._create_distant_cluster_split(
                features_reduced, cluster_labels, n_clusters, targets
            )
            
            # Calculate cluster distances
            distance_metrics = self._calculate_cluster_distances(
                features_reduced, train_idx, test_idx
            )
            
            split_data = {
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_smiles': [smiles[i] for i in train_idx],
                'test_smiles': [smiles[i] for i in test_idx],
                'train_targets': targets[train_idx],
                'test_targets': targets[test_idx],
                'n_clusters': n_clusters,
                'cluster_stats': cluster_stats,
                'distance_metrics': distance_metrics
            }
            
            # Calculate quality metrics
            quality_metrics = self._calculate_split_quality_metrics(
                train_idx, test_idx, smiles, targets
            )
            split_data.update(quality_metrics)
            
            self._save_split(name, 'cluster', split_data)
            
            self.logger.info(f"  [OK] Cluster split: {len(train_idx)} train, {len(test_idx)} test")
            self.logger.info(f"    {n_clusters} clusters, mean distance: {distance_metrics['mean_inter_distance']:.3f}")
            
            report.statistics = {
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'n_clusters': n_clusters,
                'cluster_balance': cluster_stats['balance_score'],
                'distance_metrics': distance_metrics,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Cluster split failed: {str(e)}")
            report.success = False
            report.error = str(e)
            split_data = None
        
        self.split_reports[name]['splits'][method] = asdict(report)
        return split_data
    
    def _create_physchem_split(self, name: str, smiles: List[str], 
                              targets: np.ndarray) -> Optional[Dict]:
        """
        Create physicochemical property-based split.
        
        References:
        - Delaney (2004) J Chem Inf Comput Sci 44:1000-1005
        - Lipinski et al. (2001) Adv Drug Deliv Rev 46:3-26
        
        Algorithm:
        1. Calculate comprehensive molecular properties
        2. Remove correlated features (|r| > 0.7)
        3. Apply PCA for dimensionality reduction
        4. Cluster in property space
        5. Stratified sampling from clusters
        """
        method = 'physchem'
        criteria = self.SPLIT_CRITERIA[method]
        
        report = SplitReport(
            method=method,
            references=[criteria['reference']],
            criteria=criteria,
            success=True,
            statistics={},
            failed_smiles=[]
        )
        
        if not RDKIT_AVAILABLE:
            self.logger.error("Physicochemical split requires RDKit")
            report.success = False
            report.error = 'RDKit not available'
            self.split_reports[name]['splits'][method] = asdict(report)
            return None
        
        try:
            self.logger.info("Creating physicochemical property split...")
            
            # Calculate properties
            property_matrix, property_names, failed = self._calculate_solubility_properties(smiles)
            report.failed_smiles = failed
            
            self.logger.info(f"  Calculated {len(property_names)} properties")
            
            # Remove correlated features
            property_matrix_cleaned, kept_features = self._remove_correlated_features_enhanced(
                property_matrix, property_names, criteria['correlation_threshold']
            )
            
            self.logger.info(f"  Features after correlation filtering: {len(property_names)} → {len(kept_features)}")
            
            # Scale features
            scaler = RobustScaler()
            property_matrix_scaled = scaler.fit_transform(property_matrix_cleaned)
            
            # PCA
            optimal_components = self._determine_optimal_pca_components(
                property_matrix_scaled, variance_threshold=0.95
            )
            
            pca = PCA(n_components=optimal_components, random_state=self.random_state)
            property_pca = pca.fit_transform(property_matrix_scaled)
            
            explained_var = sum(pca.explained_variance_ratio_)
            self.logger.info(f"  PCA: {property_matrix_cleaned.shape[1]} → {optimal_components} ({explained_var:.1%} variance)")
            
            # Clustering in property space
            n_clusters = self._determine_optimal_clusters(
                property_pca, criteria.get('optimal_clusters', (5, 20))
            )
            
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, 
                linkage='complete'  # Complete linkage for property-based clustering
            )
            cluster_labels = clustering.fit_predict(property_pca)
            
            # Analyze property clusters
            cluster_analysis = self._analyze_property_clusters(
                property_matrix_cleaned, kept_features, cluster_labels, targets
            )
            
            # Create stratified split from clusters
            train_idx, test_idx = self._stratified_property_split(
                cluster_labels, targets, property_matrix_cleaned,
                test_size=DEFAULT_PARAMS['test_size'],
                random_state=self.random_state
            )
            
            # Calculate property coverage
            coverage_metrics = self._calculate_property_coverage_enhanced(
                property_matrix_scaled, train_idx, test_idx, kept_features
            )
            
            split_data = {
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_smiles': [smiles[i] for i in train_idx],
                'test_smiles': [smiles[i] for i in test_idx],
                'train_targets': targets[train_idx],
                'test_targets': targets[test_idx],
                'n_properties': len(kept_features),
                'kept_features': kept_features,
                'pca_components': optimal_components,
                'pca_variance_explained': float(explained_var),
                'n_clusters': n_clusters,
                'cluster_analysis': cluster_analysis,
                'coverage_metrics': coverage_metrics
            }
            
            # Calculate quality metrics
            quality_metrics = self._calculate_split_quality_metrics(
                train_idx, test_idx, smiles, targets
            )
            split_data.update(quality_metrics)
            
            self._save_split(name, 'physchem', split_data)
            
            self.logger.info(f"  [OK] Physicochemical split: {len(train_idx)} train, {len(test_idx)} test")
            self.logger.info(f"    Property coverage: {coverage_metrics['overall_coverage']:.2%}")
            
            report.statistics = {
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'n_properties_used': len(kept_features),
                'n_clusters': n_clusters,
                'coverage_metrics': coverage_metrics,
                'cluster_balance': cluster_analysis['balance_score'],
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Physicochemical split failed: {str(e)}")
            report.success = False
            report.error = str(e)
            split_data = None
        
        self.split_reports[name]['splits'][method] = asdict(report)
        return split_data
    
    def _create_activity_cliff_split(self, name: str, smiles: List[str], 
                                   targets: np.ndarray) -> Optional[Dict]:
        """
        Create activity cliff-aware split using SALI.
        
        References:
        - Guha & Van Drie (2008) J Chem Inf Model 48:646-658
        - Maggiora (2006) J Chem Inf Model 46:1535
        
        SALI (Structure-Activity Landscape Index):
        SALI(i,j) = |A_i - A_j| / (1 - sim(i,j) + ε)
        
        where:
        - A_i, A_j: Activities (log S values)
        - sim(i,j): Tanimoto similarity
        - ε: Small smoothing factor (0.001)
        
        Cliff criteria:
        - sim(i,j) ≥ 0.7 (high structural similarity)
        - |A_i - A_j| ≥ 1.0 (10-fold activity difference)
        """
        method = 'activity_cliff'
        criteria = self.SPLIT_CRITERIA[method]
        
        report = SplitReport(
            method=method,
            references=[
                criteria['reference'],
                'Maggiora (2006) J Chem Inf Model 46:1535'
            ],
            criteria=criteria,
            success=True,
            statistics={},
            failed_smiles=[]
        )
        
        if not RDKIT_AVAILABLE:
            self.logger.error("Activity cliff split requires RDKit")
            report.success = False
            report.error = 'RDKit not available'
            self.split_reports[name]['splits'][method] = asdict(report)
            return None
        
        try:
            self.logger.info("Creating activity cliff split...")
            
            # Calculate fingerprints
            fps, failed_fps = self._calculate_fingerprints_with_errors(smiles)
            report.failed_smiles = failed_fps
            
            # Calculate SALI matrix
            if criteria['cliff_detection_method'] == 'SALI':
                self.logger.info("  Calculating SALI matrix...")
                sali_matrix, cliff_pairs = self._calculate_sali_matrix(
                    fps, targets, criteria['smoothing_factor']
                )
                
                # Find significant cliffs
                cliff_molecules = set()
                significant_cliffs = []
                
                for i, j, sim, act_diff, sali in cliff_pairs:
                    if (sim >= criteria['similarity_threshold'] and 
                        act_diff >= criteria['activity_difference_threshold']):
                        cliff_molecules.add(i)
                        cliff_molecules.add(j)
                        significant_cliffs.append((i, j, sim, act_diff, sali))
                
                self.logger.info(f"  Found {len(significant_cliffs)} activity cliffs")
                self.logger.info(f"  Molecules involved in cliffs: {len(cliff_molecules)}")
            else:
                # Simple threshold-based detection
                self.logger.info("  Using threshold-based cliff detection...")
                cliff_molecules = set()
                significant_cliffs = []
                
                for i in range(len(smiles)):
                    for j in range(i+1, len(smiles)):
                        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                        act_diff = abs(targets[i] - targets[j])
                        
                        if (sim >= criteria['similarity_threshold'] and 
                            act_diff >= criteria['activity_difference_threshold']):
                            cliff_molecules.add(i)
                            cliff_molecules.add(j)
                            significant_cliffs.append((i, j, sim, act_diff, 0))
            
            # Sort cliffs by importance (SALI score or activity difference)
            significant_cliffs.sort(key=lambda x: x[4] if x[4] > 0 else x[3], reverse=True)
            
            # Select test molecules
            test_idx_set = set()
            max_cliff_molecules = int(len(smiles) * criteria['max_cliff_ratio'])
            
            # Add cliff molecules to test set (up to max ratio)
            for i, j, sim, diff, sali in significant_cliffs:
                if len(test_idx_set) >= max_cliff_molecules:
                    break
                test_idx_set.add(i)
                test_idx_set.add(j)
            
            self.logger.info(f"  Added {len(test_idx_set)} cliff molecules to test set")
            
            # Add diverse non-cliff molecules to reach target test size
            remaining_needed = int(len(smiles) * DEFAULT_PARAMS['test_size']) - len(test_idx_set)
            
            if remaining_needed > 0:
                non_cliff_idx = list(set(range(len(smiles))) - cliff_molecules)
                
                if non_cliff_idx:
                    self.logger.info(f"  Selecting {remaining_needed} diverse non-cliff molecules...")
                    additional_test = self._select_diverse_subset_enhanced(
                        fps, non_cliff_idx, remaining_needed, 
                        targets, self.random_state
                    )
                    test_idx_set.update(additional_test)
            
            test_idx = np.array(sorted(list(test_idx_set)))
            train_idx = np.array([i for i in range(len(smiles)) if i not in test_idx_set])
            
            # Analyze activity cliffs in the split
            cliff_analysis = self._analyze_activity_cliffs(
                fps, targets, train_idx, test_idx, significant_cliffs, criteria
            )
            
            split_data = {
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_smiles': [smiles[i] for i in train_idx],
                'test_smiles': [smiles[i] for i in test_idx],
                'train_targets': targets[train_idx],
                'test_targets': targets[test_idx],
                'n_cliff_pairs': len(significant_cliffs),
                'cliff_analysis': cliff_analysis,
                'top_cliffs': [
                    {
                        'mol1_idx': int(i),
                        'mol2_idx': int(j),
                        'similarity': float(sim),
                        'activity_diff': float(diff),
                        'sali': float(sali)
                    }
                    for i, j, sim, diff, sali in significant_cliffs[:10]
                ]
            }
            
            # Calculate quality metrics
            quality_metrics = self._calculate_split_quality_metrics(
                train_idx, test_idx, smiles, targets
            )
            split_data.update(quality_metrics)
            
            self._save_split(name, 'activity_cliff', split_data)
            
            self.logger.info(f"  [OK] Activity cliff split: {len(train_idx)} train, {len(test_idx)} test")
            self.logger.info(f"    Cliff molecules in test: {cliff_analysis['cliff_molecules_in_test']}")
            
            report.statistics = {
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'n_cliff_pairs': len(significant_cliffs),
                'cliff_analysis': cliff_analysis,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Activity cliff split failed: {str(e)}")
            report.success = False
            report.error = str(e)
            split_data = None
        
        self.split_reports[name]['splits'][method] = asdict(report)
        return split_data
    
    def _create_solubility_aware_split(self, name: str, smiles: List[str], 
                                      targets: np.ndarray,
                                      measurement_metadata: Optional[Dict] = None) -> Optional[Dict]:
        """
        Create solubility measurement challenge-aware split.
        
        Reference: Avdeef (2020) ADMET & DMPK 8:29-77
        
        Difficulty score:
        D_i = Σ_k w_k · C_k(mol_i)
        
        Challenges considered:
        1. Measurement uncertainty (LogP>5, aggregation, ionization)
        2. Polymorphism risk (flexibility, H-bonding)
        3. Aggregation tendency (LogP, TPSA, MW)
        4. pH sensitivity (ionizable groups)
        5. Temperature sensitivity (estimated from structure)
        """
        method = 'solubility_aware'
        criteria = self.SPLIT_CRITERIA[method]
        
        report = SplitReport(
            method=method,
            references=[criteria['reference']],
            criteria=criteria,
            success=True,
            statistics={},
            failed_smiles=[]
        )
        
        if not RDKIT_AVAILABLE:
            self.logger.error("Solubility-aware split requires RDKit")
            report.success = False
            report.error = 'RDKit not available'
            self.split_reports[name]['splits'][method] = asdict(report)
            return None
        
        try:
            self.logger.info("Creating solubility-aware split...")
            
            # Calculate solubility-specific challenges
            challenges = self._calculate_solubility_challenges(smiles, targets)
            
            # Create difficulty scores
            difficulty_scores = np.zeros(len(smiles))
            weights = criteria['difficulty_weights']
            
            for challenge, weight in weights.items():
                if challenge in challenges:
                    difficulty_scores += weight * challenges[challenge]
            
            # Normalize difficulty scores
            if np.ptp(difficulty_scores) > 0:
                difficulty_scores = (difficulty_scores - difficulty_scores.min()) / \
                                  (difficulty_scores.max() - difficulty_scores.min())
            
            # Create difficulty quartiles
            difficulty_quartiles = pd.qcut(
                difficulty_scores, 
                q=4, 
                labels=['easy', 'moderate', 'hard', 'very_hard']
            )
            
            self.logger.info("  Difficulty distribution:")
            for level in ['easy', 'moderate', 'hard', 'very_hard']:
                count = sum(difficulty_quartiles == level)
                self.logger.info(f"    {level}: {count} molecules")
            
            # Stratified sampling with higher test representation for difficult molecules
            train_idx = []
            test_idx = []
            
            for difficulty in ['easy', 'moderate', 'hard', 'very_hard']:
                difficulty_mask = difficulty_quartiles == difficulty
                difficulty_indices = np.where(difficulty_mask)[0]
                
                if len(difficulty_indices) > 0:
                    # Higher test fraction for difficult molecules
                    if difficulty in ['hard', 'very_hard']:
                        test_fraction = min(0.3, DEFAULT_PARAMS['test_size'] * 1.5)
                    else:
                        test_fraction = DEFAULT_PARAMS['test_size'] * 0.7
                    
                    n_test = max(1, int(len(difficulty_indices) * test_fraction))
                    
                    # Shuffle indices
                    np.random.seed(self.random_state)
                    np.random.shuffle(difficulty_indices)
                    
                    test_idx.extend(difficulty_indices[:n_test])
                    train_idx.extend(difficulty_indices[n_test:])
            
            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)
            
            # Analyze challenge distribution
            challenge_analysis = self._analyze_challenge_distribution(
                challenges, difficulty_scores, train_idx, test_idx
            )
            
            split_data = {
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_smiles': [smiles[i] for i in train_idx],
                'test_smiles': [smiles[i] for i in test_idx],
                'train_targets': targets[train_idx],
                'test_targets': targets[test_idx],
                'difficulty_scores': difficulty_scores.tolist(),
                'challenge_analysis': challenge_analysis,
                'measurement_metadata': measurement_metadata or {},
                'solubility_context': asdict(self.solubility_context)
            }
            
            # Calculate quality metrics
            quality_metrics = self._calculate_split_quality_metrics(
                train_idx, test_idx, smiles, targets
            )
            split_data.update(quality_metrics)
            
            self._save_split(name, 'solubility_aware', split_data)
            
            self.logger.info(f"  [OK] Solubility-aware split: {len(train_idx)} train, {len(test_idx)} test")
            self.logger.info(f"    Mean difficulty - Train: {difficulty_scores[train_idx].mean():.3f}, "
                           f"Test: {difficulty_scores[test_idx].mean():.3f}")
            
            report.statistics = {
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'challenge_analysis': challenge_analysis,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Solubility-aware split failed: {str(e)}")
            report.success = False
            report.error = str(e)
            split_data = None
        
        self.split_reports[name]['splits'][method] = asdict(report)
        return split_data
    
    def _create_time_series_split(self, name: str, smiles: List[str], 
                                 targets: np.ndarray) -> Optional[Dict]:
        """
        Create time-based split preserving temporal order.
        
        Reference: Sheridan (2013) J Chem Inf Model 53:783-790
        
        This mimics real-world scenarios where models are trained on
        historical data and used to predict future compounds.
        """
        method = 'time_series'
        criteria = self.SPLIT_CRITERIA[method]
        
        report = SplitReport(
            method=method,
            references=[criteria['reference']],
            criteria=criteria,
            success=True,
            statistics={},
            failed_smiles=[]
        )
        
        try:
            self.logger.info("Creating time series split...")
            
            # Preserve order - no shuffling
            indices = np.arange(len(smiles))
            split_point = int(len(indices) * (1 - DEFAULT_PARAMS['test_size']))
            
            train_idx = indices[:split_point]
            test_idx = indices[split_point:]
            
            # Analyze temporal trends
            temporal_analysis = {
                'train_period': f"Molecules 0-{split_point-1}",
                'test_period': f"Molecules {split_point}-{len(indices)-1}",
                'train_mean_target': float(targets[train_idx].mean()),
                'test_mean_target': float(targets[test_idx].mean()),
                'temporal_drift': float(abs(targets[train_idx].mean() - targets[test_idx].mean()))
            }
            
            split_data = {
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_smiles': [smiles[i] for i in train_idx],
                'test_smiles': [smiles[i] for i in test_idx],
                'train_targets': targets[train_idx],
                'test_targets': targets[test_idx],
                'split_point': split_point,
                'temporal_analysis': temporal_analysis
            }
            
            # Calculate quality metrics
            quality_metrics = self._calculate_split_quality_metrics(
                train_idx, test_idx, smiles, targets
            )
            split_data.update(quality_metrics)
            
            self._save_split(name, 'time_series', split_data)
            
            self.logger.info(f"  [OK] Time series split: {len(train_idx)} train, {len(test_idx)} test")
            self.logger.info(f"    Split point: {split_point}")
            self.logger.info(f"    Temporal drift: {temporal_analysis['temporal_drift']:.3f}")
            
            report.statistics = {
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'split_point': split_point,
                'temporal_analysis': temporal_analysis,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Time series split failed: {str(e)}")
            report.success = False
            report.error = str(e)
            split_data = None
        
        self.split_reports[name]['splits'][method] = asdict(report)
        return split_data
    
    def _create_ensemble_split(self, name: str, smiles: List[str], 
                             targets: np.ndarray, previous_splits: Dict) -> Optional[Dict]:
        """
        Create optimized ensemble split combining multiple methods.
        
        Reference: Sheridan (2019) J Chem Inf Model 59:1645-1649
        
        Optimization objective:
        maximize Quality = α·DistQuality + β·RangeQuality + γ·SizeQuality
        
        Uses differential evolution to find optimal weights.
        """
        method = 'ensemble'
        criteria = self.SPLIT_CRITERIA[method]
        
        report = SplitReport(
            method=method,
            references=[criteria['reference']],
            criteria=criteria,
            success=True,
            statistics={},
            failed_smiles=[]
        )
        
        try:
            self.logger.info("Creating optimized ensemble split...")
            
            # Filter valid splits
            valid_splits = {k: v for k, v in previous_splits.items() 
                           if v is not None and 'test_idx' in v}
            
            if len(valid_splits) < 3:
                self.logger.warning("Not enough valid splits for ensemble")
                report.success = False
                report.error = 'Insufficient valid splits'
                self.split_reports[name]['splits'][method] = asdict(report)
                return None
            
            # Calculate features if RDKit available
            if RDKIT_AVAILABLE:
                fps, _ = self._calculate_fingerprints_with_errors(smiles)
                property_matrix = self._calculate_key_properties_batch(smiles)
                uncertainty_scores = self._calculate_comprehensive_uncertainty(
                    smiles, targets, fps, property_matrix
                )
            else:
                # Simple uncertainty based on target distribution
                kde = gaussian_kde(targets)
                densities = kde(targets)
                uncertainty_scores = 1 - (densities / np.max(densities))
            
            # Optimize weights using differential evolution
            if len(smiles) > 100 and criteria['optimization_method'] == 'differential_evolution':
                self.logger.info("  Optimizing ensemble weights using differential evolution...")
                optimal_weights = self._optimize_ensemble_weights(
                    valid_splits, targets, uncertainty_scores
                )
            else:
                # Use default weights
                optimal_weights = criteria['weight_distribution']
            
            # Create ensemble scores
            ensemble_scores = np.zeros(len(smiles))
            method_contributions = {}
            
            # Weight contributions from each method
            for split_method, weight in optimal_weights.items():
                if split_method == 'uncertainty':
                    # Add uncertainty contribution
                    uncertainty_ranks = np.argsort(uncertainty_scores)[::-1]
                    top_uncertain = uncertainty_ranks[:int(len(smiles) * DEFAULT_PARAMS['test_size'])]
                    ensemble_scores[top_uncertain] += weight
                elif split_method in valid_splits:
                    test_indices = valid_splits[split_method]['test_idx']
                    ensemble_scores[test_indices] += weight
                    method_contributions[split_method] = len(test_indices)
            
            # Add small random noise for tie-breaking
            ensemble_scores += np.random.RandomState(self.random_state).normal(
                0, 0.01, len(smiles)
            )
            
            # Select test set based on highest ensemble scores
            test_size = int(len(smiles) * DEFAULT_PARAMS['test_size'])
            test_idx = np.argsort(ensemble_scores)[-test_size:]
            train_idx = np.array([i for i in range(len(smiles)) if i not in test_idx])
            
            # Analyze ensemble composition
            ensemble_analysis = self._analyze_ensemble_split(
                valid_splits, train_idx, test_idx, optimal_weights
            )
            
            split_data = {
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_smiles': [smiles[i] for i in train_idx],
                'test_smiles': [smiles[i] for i in test_idx],
                'train_targets': targets[train_idx],
                'test_targets': targets[test_idx],
                'optimal_weights': optimal_weights,
                'ensemble_analysis': ensemble_analysis,
                'method_contributions': method_contributions
            }
            
            # Calculate quality metrics
            quality_metrics = self._calculate_split_quality_metrics(
                train_idx, test_idx, smiles, targets
            )
            split_data.update(quality_metrics)
            
            self._save_split(name, 'ensemble', split_data)
            
            self.logger.info(f"  [OK] Ensemble split: {len(train_idx)} train, {len(test_idx)} test")
            self.logger.info(f"    Agreement: {ensemble_analysis['overall_agreement']:.2%}")
            self.logger.info("    Optimal weights:")
            for method, weight in optimal_weights.items():
                self.logger.info(f"      {method}: {weight:.3f}")
            
            report.statistics = {
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'optimal_weights': optimal_weights,
                'ensemble_analysis': ensemble_analysis,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble split failed: {str(e)}")
            report.success = False
            report.error = str(e)
            split_data = None
        
        self.split_reports[name]['splits'][method] = asdict(report)
        return split_data
    
    def _create_test_only_split(self, name: str, smiles: List[str], 
                               targets: np.ndarray) -> Optional[Dict]:
        """
        Create test-only split for external validation.
        
        Reference: Tropsha (2010) Mol Inform 29:476-488
        
        All data assigned to test set, no training data.
        Used for independent validation of pre-trained models.
        """
        method = 'test_only'
        criteria = self.SPLIT_CRITERIA.get(method, {})
        
        report = SplitReport(
            method=method,
            references=[criteria.get('reference', 'Tropsha (2010) Mol Inform 29:476-488')],
            criteria={'description': 'All data assigned to test set'},
            success=True,
            statistics={},
            failed_smiles=[]
        )
        
        try:
            self.logger.info("Creating test-only split...")
            
            test_idx = np.arange(len(smiles))
            
            split_data = {
                'train_idx': np.array([]),
                'test_idx': test_idx,
                'train_smiles': [],
                'test_smiles': smiles,
                'train_targets': np.array([]),
                'test_targets': targets,
                'purpose': 'external_validation'
            }
            
            # Save only test set
            test_df = pl.DataFrame({
                'smiles': smiles,
                'target': targets
            })
            
            test_path = self.output_dir / 'test' / self.split_abbrev['test_only'] / f'{name}_test.csv'
            test_df.write_csv(test_path)
            
            self.logger.info(f"  [OK] Test-only split: 0 train, {len(test_idx)} test")
            
            report.statistics = {
                'train_size': 0,
                'test_size': len(test_idx),
                'purpose': 'external_validation'
            }
            
        except Exception as e:
            self.logger.error(f"Test-only split failed: {str(e)}")
            report.success = False
            report.error = str(e)
            split_data = None
        
        self.split_reports[name]['splits'][method] = asdict(report)
        return split_data
    
    def _create_simple_cluster_split(self, name: str, smiles: List[str], 
                                   targets: np.ndarray) -> Optional[Dict]:
        """
        Create simple cluster split without RDKit.
        
        Fallback method using target values for clustering when
        molecular descriptors are not available.
        """
        method = 'cluster'
        report = SplitReport(
            method=method,
            references=['Simple target-based clustering'],
            criteria={},
            success=True,
            statistics={},
            failed_smiles=[]
        )
        
        try:
            self.logger.info("Creating simple cluster split (target-based)...")
            
            # Cluster based on targets
            targets_reshaped = targets.reshape(-1, 1)
            
            # Number of clusters
            n_clusters = min(5, len(targets) // 10)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(targets_reshaped)
            
            # Create split
            train_idx = []
            test_idx = []
            
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                if len(cluster_indices) > 0:
                    n_test = max(1, int(len(cluster_indices) * DEFAULT_PARAMS['test_size']))
                    np.random.shuffle(cluster_indices)
                    test_idx.extend(cluster_indices[:n_test])
                    train_idx.extend(cluster_indices[n_test:])
            
            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)
            
            split_data = {
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_smiles': [smiles[i] for i in train_idx],
                'test_smiles': [smiles[i] for i in test_idx],
                'train_targets': targets[train_idx],
                'test_targets': targets[test_idx],
                'n_clusters': n_clusters,
                'clustering_method': 'kmeans_on_targets'
            }
            
            # Calculate quality metrics
            quality_metrics = self._calculate_split_quality_metrics(
                train_idx, test_idx, smiles, targets
            )
            split_data.update(quality_metrics)
            
            self._save_split(name, 'cluster', split_data)
            
            self.logger.info(f"  [OK] Simple cluster split: {len(train_idx)} train, {len(test_idx)} test")
            
            report.statistics = {
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'n_clusters': n_clusters,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Simple cluster split failed: {str(e)}")
            report.success = False
            report.error = str(e)
            split_data = None
        
        self.split_reports[name]['splits'][method] = asdict(report)
        return split_data
    
    # ===== Helper Methods =====
    
    def _calculate_solubility_properties(self, smiles: List[str]) -> Tuple[np.ndarray, List[str], List]:
        """
        Calculate comprehensive solubility-relevant properties.
        
        Based on descriptors identified in:
        - Delaney (2004) - ESOL model
        - Hou et al. (2004) - ADME properties
        - Mitchell & Jurs (1998) - Solubility prediction
        """
        property_matrix = []
        failed_molecules = []
        
        # Property names
        standard_props = [
            'MolLogP', 'MolWt', 'NumRotatableBonds', 'TPSA',
            'NumHBD', 'NumHBA', 'NumHeteroatoms',
            'NumAromaticRings', 'FractionCsp3', 'BertzCT',
            'RingCount', 'NumSaturatedRings',
            'LabuteASA', 'Chi0', 'Chi1', 'Kappa1', 'Kappa2'
        ]
        
        electronic_props = ['MaxPartialCharge', 'MinPartialCharge']
        
        custom_props = [
            'AromaticProportion', 'Flexibility', 'Polarity',
            'MeltingPointEstimate', 'CrystalPackingScore',
            'AggregationScore', 'IonizableGroups'
        ]
        
        property_names = standard_props + electronic_props + custom_props
        
        for idx, smi in enumerate(smiles):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    props = []
                    
                    # Standard descriptors
                    props.extend([
                        Crippen.MolLogP(mol),
                        Descriptors.MolWt(mol),
                        Descriptors.NumRotatableBonds(mol),
                        Descriptors.TPSA(mol),
                        Descriptors.NumHDonors(mol),
                        Descriptors.NumHAcceptors(mol),
                        Descriptors.NumHeteroatoms(mol),
                        Descriptors.NumAromaticRings(mol),
                        rdMolDescriptors.CalcFractionCsp3(mol) if hasattr(rdMolDescriptors, 'CalcFractionCsp3') else 0,
                        BertzCT(mol),
                        Descriptors.RingCount(mol),
                        Descriptors.NumSaturatedRings(mol),
                        Descriptors.LabuteASA(mol),
                        Descriptors.Chi0(mol),
                        Descriptors.Chi1(mol),
                        Descriptors.Kappa1(mol),
                        Descriptors.Kappa2(mol),
                    ])
                    
                    # Electronic properties
                    props.extend(self._calculate_electronic_properties(mol))
                    
                    # Custom features
                    props.extend([
                        self._calculate_aromatic_proportion(mol),
                        self._calculate_flexibility_index(mol),
                        self._calculate_polarity_index(mol),
                        self._estimate_melting_point(mol),
                        self._calculate_crystal_packing_score(mol),
                        self._calculate_aggregation_score(mol),
                        self._count_ionizable_groups(mol)
                    ])
                    
                    property_matrix.append(props)
                else:
                    property_matrix.append(np.zeros(len(property_names)))
                    failed_molecules.append((idx, smi, "Invalid SMILES"))
                    
            except Exception as e:
                property_matrix.append(np.zeros(len(property_names)))
                failed_molecules.append((idx, smi, str(e)))
        
        return np.array(property_matrix), property_names, failed_molecules
    
    def _calculate_electronic_properties(self, mol) -> List[float]:
        """Calculate partial charge properties"""
        try:
            AllChem.ComputeGasteigerCharges(mol)
            charges = []
            for atom in mol.GetAtoms():
                charge = atom.GetProp('_GasteigerCharge')
                if isinstance(charge, str):
                    charge = float(charge)
                if not np.isnan(charge) and not np.isinf(charge):
                    charges.append(charge)
            
            if charges:
                return [max(charges), min(charges)]
            else:
                return [0.0, 0.0]
        except:
            return [0.0, 0.0]
    
    def _calculate_aromatic_proportion(self, mol) -> float:
        """Calculate proportion of aromatic atoms"""
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        total_atoms = mol.GetNumAtoms()
        return aromatic_atoms / total_atoms if total_atoms > 0 else 0
    
    def _calculate_flexibility_index(self, mol) -> float:
        """
        Calculate molecular flexibility index.
        
        Based on Kier flexibility index:
        Flexibility = n_rot / sqrt(n_atoms)
        """
        n_rot = Descriptors.NumRotatableBonds(mol)
        n_atoms = mol.GetNumAtoms()
        return n_rot / (n_atoms ** 0.5) if n_atoms > 0 else 0
    
    def _calculate_polarity_index(self, mol) -> float:
        """Calculate polarity as TPSA/TSA ratio"""
        tpsa = Descriptors.TPSA(mol)
        tsa = Descriptors.LabuteASA(mol)
        return tpsa / tsa if tsa > 0 else 0
    
    def _estimate_melting_point(self, mol) -> float:
        """
        Estimate melting point using simple structural features.
        
        Based on correlations from Yalkowsky & Valvani (1980).
        """
        n_rot = Descriptors.NumRotatableBonds(mol)
        symmetry = self._calculate_symmetry_score(mol)
        n_aromatic = Descriptors.NumAromaticRings(mol)
        mw = Descriptors.MolWt(mol)
        
        # Simple estimation model
        mp_estimate = 25.0  # Base temperature
        mp_estimate += n_aromatic * 40  # Aromatic rings increase MP
        mp_estimate += symmetry * 20     # Symmetry increases MP
        mp_estimate -= n_rot * 20        # Flexibility decreases MP
        mp_estimate += np.log(mw) * 10   # Size effect
        
        return mp_estimate
    
    def _calculate_symmetry_score(self, mol) -> float:
        """Estimate molecular symmetry from atom type distribution"""
        atoms = mol.GetAtoms()
        atom_types = [atom.GetAtomicNum() for atom in atoms]
        
        type_counts = Counter(atom_types)
        
        if len(atoms) > 0:
            # Higher score for more symmetric distribution
            max_count = max(type_counts.values())
            return max_count / len(atoms)
        return 0
    
    def _calculate_crystal_packing_score(self, mol) -> float:
        """
        Estimate crystal packing efficiency.
        
        Factors:
        - Planarity (low Fsp3)
        - H-bonding capability
        - Aromatic stacking
        """
        planarity = 1 - Descriptors.FractionCsp3(mol)
        n_hbonds = Descriptors.NumHDonors(mol) + Descriptors.NumHAcceptors(mol)
        n_aromatic = Descriptors.NumAromaticRings(mol)
        
        # Weighted combination
        packing_score = (
            planarity * 0.4 +
            min(n_hbonds / 10, 1) * 0.3 +
            min(n_aromatic / 5, 1) * 0.3
        )
        
        return packing_score
    
    def _calculate_aggregation_score(self, mol) -> float:
        """
        Calculate aggregation tendency.
        
        Based on:
        - High LogP (hydrophobicity)
        - Low TPSA (poor solvation)
        - High MW (size)
        """
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        mw = Descriptors.MolWt(mol)
        
        aggregation = 0
        
        # High LogP increases aggregation
        if logp > 3:
            aggregation += (logp - 3) * 0.2
        
        # Low TPSA increases aggregation
        if tpsa < 60:
            aggregation += (60 - tpsa) / 60 * 0.3
        
        # High MW increases aggregation
        if mw > 400:
            aggregation += (mw - 400) / 400 * 0.2
        
        return min(aggregation, 1.0)
    
    def _count_ionizable_groups(self, mol) -> int:
        """Count potentially ionizable groups"""
        # SMARTS patterns for common ionizable groups
        acidic_smarts = [
            '[CX3](=O)[OX2H1]',          # Carboxylic acid
            '[SX4](=O)(=O)[OX2H1]',      # Sulfonic acid
            '[PX4](=O)([OX2H1])[OX2H1]', # Phosphonic acid
        ]
        
        basic_smarts = [
            '[NX3;H2,H1;!$(NC=O)]',      # Primary/secondary amine
            '[NX3;H0;!$(NC=O)]',         # Tertiary amine
            '[nX3;H1]',                  # Aromatic NH
        ]
        
        count = 0
        for smarts in acidic_smarts + basic_smarts:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                count += len(mol.GetSubstructMatches(pattern))
        
        return count
    
    def _calculate_solubility_challenges(self, smiles: List[str], 
                                       targets: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate various solubility measurement challenges.
        
        Based on Avdeef (2020) multi-lab reproducibility study.
        """
        n_samples = len(smiles)
        
        challenges = {
            'measurement_uncertainty': np.zeros(n_samples),
            'polymorphism_risk': np.zeros(n_samples),
            'aggregation_tendency': np.zeros(n_samples),
            'ph_sensitivity': np.zeros(n_samples),
            'temperature_sensitivity': np.zeros(n_samples)
        }
        
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                # Each challenge scored 0-1
                challenges['measurement_uncertainty'][i] = self._calculate_measurement_uncertainty(mol)
                challenges['polymorphism_risk'][i] = self._calculate_polymorphism_risk(mol)
                challenges['aggregation_tendency'][i] = self._calculate_aggregation_score(mol)
                
                # pH sensitivity based on ionizable groups
                n_ionizable = self._count_ionizable_groups(mol)
                challenges['ph_sensitivity'][i] = min(n_ionizable / 3, 1.0)
                
                # Temperature sensitivity
                challenges['temperature_sensitivity'][i] = self._calculate_temperature_sensitivity(mol)
        
        return challenges
    
    def _calculate_measurement_uncertainty(self, mol) -> float:
        """
        Estimate measurement uncertainty based on molecular properties.
        
        Factors from Avdeef (2020):
        - High LogP (>5): precipitation issues
        - Aggregation tendency
        - Multiple ionizable groups
        - High MW
        """
        uncertainty = 0
        
        # High LogP compounds difficult to measure
        logp = Crippen.MolLogP(mol)
        if logp > 5:
            uncertainty += 0.3
        
        # Aggregation issues
        agg_score = self._calculate_aggregation_score(mol)
        uncertainty += agg_score * 0.2
        
        # Ionization complexity
        n_ionizable = self._count_ionizable_groups(mol)
        uncertainty += min(n_ionizable * 0.1, 0.3)
        
        # Size issues
        mw = Descriptors.MolWt(mol)
        if mw > 500:
            uncertainty += 0.2
        
        return min(uncertainty, 1.0)
    
    def _calculate_polymorphism_risk(self, mol) -> float:
        """
        Estimate risk of polymorphism.
        
        Based on:
        - Conformational flexibility
        - H-bonding patterns
        - Molecular symmetry
        """
        risk = 0
        
        # Flexibility increases polymorphism risk
        n_rot = Descriptors.NumRotatableBonds(mol)
        risk += min(n_rot / 10, 0.3)
        
        # Multiple H-bond donors/acceptors
        n_hb = Descriptors.NumHDonors(mol) + Descriptors.NumHAcceptors(mol)
        risk += min(n_hb / 10, 0.3)
        
        # Low symmetry increases risk
        flexibility = self._calculate_flexibility_index(mol)
        risk += flexibility * 0.2
        
        # Asymmetry
        symmetry = self._calculate_symmetry_score(mol)
        risk += (1 - symmetry) * 0.2
        
        return min(risk, 1.0)
    
    def _calculate_temperature_sensitivity(self, mol) -> float:
        """
        Estimate temperature sensitivity of solubility.
        
        Based on enthalpy of fusion and mixing.
        """
        # Simplified estimation
        h_fusion = self._estimate_melting_point(mol) * 0.01  # Rough correlation
        h_mixing = abs(Crippen.MolLogP(mol)) * 0.1           # Hydrophobic effect
        
        sensitivity = (h_fusion + h_mixing) / 10
        
        return min(sensitivity, 1.0)
    
    def _calculate_comprehensive_uncertainty(self, smiles: List[str], 
                                           targets: np.ndarray,
                                           fps: List, 
                                           property_matrix: np.ndarray) -> np.ndarray:
        """Calculate comprehensive uncertainty scores for ensemble method"""
        n_samples = len(smiles)
        uncertainty_scores = np.zeros(n_samples)
        
        # 1. Structural complexity (BertzCT)
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                complexity = BertzCT(mol) / 1000  # Normalize
                uncertainty_scores[i] += complexity * 0.2
        
        # 2. Activity distribution (low density = high uncertainty)
        kde = gaussian_kde(targets)
        densities = kde(targets)
        density_uncertainty = 1 - (densities / np.max(densities))
        uncertainty_scores += density_uncertainty * 0.2
        
        # 3. Chemical space density
        if property_matrix is not None and len(property_matrix) > 10:
            n_neighbors = min(10, len(smiles) - 1)
            for i in range(len(smiles)):
                distances = cdist([property_matrix[i]], property_matrix)[0]
                nearest_distances = np.sort(distances)[1:n_neighbors+1]
                local_density = 1 / (np.mean(nearest_distances) + 1e-6)
                sparsity = 1 / (local_density + 1)
                uncertainty_scores[i] += sparsity * 0.2
        
        # 4. Measurement uncertainty
        measurement_uncertainties = self._calculate_solubility_challenges(
            smiles, targets
        )['measurement_uncertainty']
        uncertainty_scores += measurement_uncertainties * 0.2
        
        # 5. Property variance
        if property_matrix is not None:
            property_std = np.std(property_matrix, axis=0)
            property_variance = np.sum(
                (property_matrix - np.mean(property_matrix, axis=0)) ** 2 * property_std,
                axis=1
            )
            normalized_variance = property_variance / (np.max(property_variance) + 1e-8)
            uncertainty_scores += normalized_variance * 0.2
        
        return uncertainty_scores
    
    def _calculate_multiple_fingerprints(self, smiles: List[str]) -> Dict:
        """Calculate multiple fingerprint types for diversity"""
        results = {
            'morgan': [],
            'topological': [],
            'failed': []
        }
        
        for idx, smi in enumerate(smiles):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    # Morgan fingerprint (circular)
                    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, radius=3, nBits=2048
                    )
                    results['morgan'].append(morgan_fp)
                    
                    # Topological fingerprint
                    topo_fp = Chem.RDKFingerprint(mol, maxPath=7)
                    results['topological'].append(topo_fp)
                else:
                    # Create empty fingerprints for failed molecules
                    results['morgan'].append(DataStructs.ExplicitBitVect(2048))
                    results['topological'].append(DataStructs.ExplicitBitVect(2048))
                    results['failed'].append((idx, smi, "Invalid SMILES"))
                    
            except Exception as e:
                results['morgan'].append(DataStructs.ExplicitBitVect(2048))
                results['topological'].append(DataStructs.ExplicitBitVect(2048))
                results['failed'].append((idx, smi, str(e)))
        
        return results
    
    def _calculate_key_properties_batch(self, smiles: List[str]) -> np.ndarray:
        """Calculate key properties for all molecules"""
        properties = []
        
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                props = [
                    Descriptors.MolWt(mol),
                    Crippen.MolLogP(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.NumAromaticRings(mol),
                    rdMolDescriptors.CalcFractionCsp3(mol) if hasattr(rdMolDescriptors, 'CalcFractionCsp3') else 0
                ]
                properties.append(props)
            else:
                properties.append([0] * 8)
        
        return np.array(properties)
    
    def _calculate_comprehensive_coverage(self, fps: List, property_matrix: np.ndarray,
                                        train_idx: np.ndarray, test_idx: np.ndarray) -> Dict:
        """Calculate comprehensive coverage metrics"""
        metrics = {}
        
        # Fingerprint coverage
        fp_coverage = self._calculate_coverage_ratio(fps, train_idx, test_idx)
        metrics['fingerprint_coverage'] = fp_coverage
        
        # Property space coverage
        if property_matrix is not None:
            prop_coverage = self._calculate_property_coverage_enhanced(
                property_matrix, train_idx, test_idx, 
                ['prop_' + str(i) for i in range(property_matrix.shape[1])]
            )
            metrics['property_coverage'] = prop_coverage['overall_coverage']
            metrics['critical_balance'] = prop_coverage['critical_balance']
        
        # Overall coverage
        metrics['overall_coverage'] = np.mean([
            metrics.get('fingerprint_coverage', 0),
            metrics.get('property_coverage', 0)
        ])
        
        return metrics
    
    def _remove_correlated_features_enhanced(self, feature_matrix: np.ndarray,
                                           feature_names: List[str],
                                           threshold: float) -> Tuple[np.ndarray, List[str]]:
        """
        Remove correlated features with priority preservation.
        
        Keeps important features for solubility prediction.
        """
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(feature_matrix.T)
        
        # Priority features (from ESOL and other models)
        priority_features = ['MolLogP', 'TPSA', 'MolWt', 'NumHBD', 'NumHBA']
        priority_indices = [i for i, name in enumerate(feature_names) if name in priority_features]
        
        to_remove = set()
        n_features = len(feature_names)
        
        for i in range(n_features):
            if i in to_remove:
                continue
                
            for j in range(i+1, n_features):
                if j in to_remove:
                    continue
                    
                if abs(corr_matrix[i, j]) > threshold:
                    # Keep priority features
                    if i in priority_indices and j not in priority_indices:
                        to_remove.add(j)
                    elif j in priority_indices and i not in priority_indices:
                        to_remove.add(i)
                    else:
                        # Remove feature with lower variance
                        if np.var(feature_matrix[:, i]) < np.var(feature_matrix[:, j]):
                            to_remove.add(i)
                        else:
                            to_remove.add(j)
        
        keep_indices = [i for i in range(n_features) if i not in to_remove]
        kept_features = [feature_names[i] for i in keep_indices]
        
        return feature_matrix[:, keep_indices], kept_features
    
    def _determine_optimal_pca_components(self, data: np.ndarray, 
                                        variance_threshold: float = 0.95) -> int:
        """Determine optimal number of PCA components"""
        pca = PCA()
        pca.fit(data)
        
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= variance_threshold) + 1
        
        # Constraints
        n_components = max(2, min(n_components, data.shape[1], 50))
        
        return n_components
    
    def _determine_optimal_clusters(self, data: np.ndarray, 
                                  cluster_range: Tuple[int, int]) -> int:
        """
        Determine optimal number of clusters using silhouette score.
        
        Silhouette score measures how similar a point is to its own
        cluster compared to other clusters.
        """
        min_clusters, max_clusters = cluster_range
        n_samples = len(data)
        
        # Adjust range based on dataset size
        max_clusters = min(max_clusters, n_samples // 10)
        min_clusters = max(2, min_clusters)
        
        if max_clusters <= min_clusters:
            return min_clusters
        
        silhouette_scores = []
        
        for n_clusters in range(min_clusters, max_clusters + 1):
            clusterer = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = clusterer.fit_predict(data)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(data, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)
        
        # Find optimal
        optimal_idx = np.argmax(silhouette_scores)
        optimal_clusters = min_clusters + optimal_idx
        
        return optimal_clusters
    
    def _analyze_scaffold_distribution(self, scaffolds: Dict, 
                                     train_idx: np.ndarray, 
                                     test_idx: np.ndarray) -> Dict:
        """Analyze scaffold distribution in train/test sets"""
        train_scaffolds = set()
        test_scaffolds = set()
        
        # Reverse mapping: index -> scaffold
        idx_to_scaffold = {}
        for scaffold, indices in scaffolds.items():
            for idx in indices:
                idx_to_scaffold[idx] = scaffold
        
        # Count scaffolds in each set
        for idx in train_idx:
            if idx in idx_to_scaffold:
                train_scaffolds.add(idx_to_scaffold[idx])
        
        for idx in test_idx:
            if idx in idx_to_scaffold:
                test_scaffolds.add(idx_to_scaffold[idx])
        
        # Calculate metrics
        shared_scaffolds = train_scaffolds & test_scaffolds
        
        return {
            'n_train_scaffolds': len(train_scaffolds),
            'n_test_scaffolds': len(test_scaffolds),
            'n_shared_scaffolds': len(shared_scaffolds),
            'scaffold_overlap_ratio': len(shared_scaffolds) / len(test_scaffolds) if test_scaffolds else 0
        }
    
    def _analyze_property_clusters(self, property_matrix: np.ndarray,
                                 feature_names: List[str],
                                 cluster_labels: np.ndarray,
                                 targets: np.ndarray) -> Dict:
        """Analyze clusters in property space"""
        n_clusters = len(np.unique(cluster_labels))
        
        analysis = {
            'cluster_sizes': [],
            'cluster_properties': [],
            'cluster_targets': [],
            'balance_score': 0
        }
        
        for i in range(n_clusters):
            mask = cluster_labels == i
            size = np.sum(mask)
            analysis['cluster_sizes'].append(int(size))
            
            if size > 0:
                # Mean properties per cluster
                mean_props = np.mean(property_matrix[mask], axis=0)
                analysis['cluster_properties'].append({
                    name: float(value) for name, value in zip(feature_names, mean_props)
                })
                
                # Target statistics per cluster
                cluster_targets = targets[mask]
                analysis['cluster_targets'].append({
                    'mean': float(np.mean(cluster_targets)),
                    'std': float(np.std(cluster_targets)) if size > 1 else 0,
                    'min': float(np.min(cluster_targets)),
                    'max': float(np.max(cluster_targets))
                })
        
        # Balance score (how evenly distributed)
        size_variance = np.var(analysis['cluster_sizes'])
        mean_size = np.mean(analysis['cluster_sizes'])
        analysis['balance_score'] = 1 / (1 + size_variance / (mean_size ** 2)) if mean_size > 0 else 0
        
        return analysis
    
    def _stratified_property_split(self, cluster_labels: np.ndarray,
                                 targets: np.ndarray,
                                 property_matrix: np.ndarray,
                                 test_size: float,
                                 random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create stratified split from property clusters"""
        unique_clusters = np.unique(cluster_labels)
        train_idx = []
        test_idx = []
        
        np.random.seed(random_state)
        
        for cluster in unique_clusters:
            cluster_mask = cluster_labels == cluster
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) >= 2:
                # Sort by target value for better distribution
                cluster_targets = targets[cluster_indices]
                sorted_idx = cluster_indices[np.argsort(cluster_targets)]
                
                # Take evenly spaced samples for test
                n_test = max(1, int(len(sorted_idx) * test_size))
                
                # Select test indices evenly across the range
                test_selections = np.linspace(0, len(sorted_idx)-1, n_test, dtype=int)
                cluster_test = sorted_idx[test_selections]
                cluster_train = np.setdiff1d(sorted_idx, cluster_test)
                
                test_idx.extend(cluster_test)
                train_idx.extend(cluster_train)
            elif len(cluster_indices) == 1:
                # Single molecule clusters go to training
                train_idx.extend(cluster_indices)
        
        return np.array(train_idx), np.array(test_idx)
    
    def _calculate_property_coverage_enhanced(self, property_matrix: np.ndarray,
                                            train_idx: np.ndarray,
                                            test_idx: np.ndarray,
                                            feature_names: List[str]) -> Dict:
        """Calculate enhanced property coverage metrics"""
        train_props = property_matrix[train_idx]
        test_props = property_matrix[test_idx]
        
        coverage_metrics = {
            'overall_coverage': 0,
            'feature_coverages': {},
            'critical_balance': 0
        }
        
        # Critical features for solubility
        critical_features = ['MolLogP', 'TPSA', 'MolWt', 'NumHBD', 'NumHBA']
        critical_indices = [i for i, name in enumerate(feature_names) if name in critical_features]
        
        feature_coverages = []
        critical_coverages = []
        
        for i, feature_name in enumerate(feature_names):
            # Calculate range coverage
            train_range = np.ptp(train_props[:, i])
            test_range = np.ptp(test_props[:, i])
            
            if train_range > 0:
                coverage = min(test_range / train_range, 1.0)
            else:
                coverage = 1.0 if test_range == 0 else 0.0
            
            coverage_metrics['feature_coverages'][feature_name] = float(coverage)
            feature_coverages.append(coverage)
            
            if i in critical_indices:
                critical_coverages.append(coverage)
        
        coverage_metrics['overall_coverage'] = float(np.mean(feature_coverages))
        coverage_metrics['critical_balance'] = float(np.mean(critical_coverages)) if critical_coverages else 0
        
        return coverage_metrics
    
    def _calculate_sali_matrix(self, fps: List, targets: np.ndarray, 
                             smoothing: float) -> Tuple[np.ndarray, List]:
        """
        Calculate Structure-Activity Landscape Index matrix.
        
        SALI(i,j) = |activity_i - activity_j| / (1 - similarity_ij + smoothing)
        """
        n = len(fps)
        sali_matrix = np.zeros((n, n))
        cliff_pairs = []
        
        # Calculate pairwise SALI
        for i in range(n):
            for j in range(i+1, n):
                # Tanimoto similarity
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                
                # Activity difference
                act_diff = abs(targets[i] - targets[j])
                
                # SALI calculation
                if sim < 1.0:
                    sali = act_diff / (1 - sim + smoothing)
                else:
                    sali = 0  # Identical structures
                
                sali_matrix[i, j] = sali
                sali_matrix[j, i] = sali
                
                # Store cliff pairs
                if sali > 0:
                    cliff_pairs.append((i, j, sim, act_diff, sali))
        
        # Sort by SALI score
        cliff_pairs.sort(key=lambda x: x[4], reverse=True)
        
        return sali_matrix, cliff_pairs
    
    def _analyze_activity_cliffs(self, fps: List, targets: np.ndarray,
                               train_idx: np.ndarray, test_idx: np.ndarray,
                               cliff_pairs: List, criteria: Dict) -> Dict:
        """Analyze activity cliffs in the split"""
        analysis = {
            'cliff_molecules_in_test': 0,
            'cliff_molecules_in_train': 0,
            'mean_sali_score': 0,
            'max_sali_score': 0,
            'cliff_coverage': 0,
            'mean_cliff_similarity': 0,
            'mean_cliff_activity_diff': 0
        }
        
        # Identify cliff molecules
        cliff_molecules = set()
        sali_scores = []
        similarities = []
        activity_diffs = []
        
        for i, j, sim, diff, sali in cliff_pairs:
            if (sim >= criteria['similarity_threshold'] and 
                diff >= criteria['activity_difference_threshold']):
                cliff_molecules.add(i)
                cliff_molecules.add(j)
                sali_scores.append(sali)
                similarities.append(sim)
                activity_diffs.append(diff)
        
        # Count cliff molecules in each set
        test_set = set(test_idx)
        train_set = set(train_idx)
        
        analysis['cliff_molecules_in_test'] = len(cliff_molecules & test_set)
        analysis['cliff_molecules_in_train'] = len(cliff_molecules & train_set)
        
        # Statistics
        if sali_scores:
            analysis['mean_sali_score'] = float(np.mean(sali_scores))
            analysis['max_sali_score'] = float(np.max(sali_scores))
        
        if similarities:
            analysis['mean_cliff_similarity'] = float(np.mean(similarities))
            
        if activity_diffs:
            analysis['mean_cliff_activity_diff'] = float(np.mean(activity_diffs))
        
        # Cliff coverage (how many cliffs have at least one molecule in test)
        covered_cliffs = 0
        for i, j, _, _, _ in cliff_pairs[:100]:  # Top 100 cliffs
            if i in test_set or j in test_set:
                covered_cliffs += 1
        
        analysis['cliff_coverage'] = covered_cliffs / min(100, len(cliff_pairs)) if cliff_pairs else 0
        
        return analysis
    
    def _select_diverse_subset_enhanced(self, fps: List, candidates: List[int],
                                      n_select: int, targets: np.ndarray,
                                      random_state: int) -> List[int]:
        """Select diverse subset using MaxMin algorithm with activity consideration"""
        np.random.seed(random_state)
        
        if len(candidates) <= n_select:
            return candidates
        
        # Start with molecule closest to median activity
        candidate_targets = targets[candidates]
        median_idx = candidates[np.argmin(np.abs(candidate_targets - np.median(candidate_targets)))]
        selected = [median_idx]
        candidates = [c for c in candidates if c != median_idx]
        
        # MaxMin selection
        while len(selected) < n_select and candidates:
            max_min_dist = -1
            best_candidate = None
            
            for candidate in candidates:
                # Calculate distances to selected molecules
                struct_dists = [
                    1 - DataStructs.TanimotoSimilarity(fps[candidate], fps[s])
                    for s in selected
                ]
                
                # Also consider activity distance
                act_dists = [
                    abs(targets[candidate] - targets[s]) / (np.std(targets) + 1e-8)
                    for s in selected
                ]
                
                # Combined distance (weighted)
                combined_dists = [
                    0.7 * s_dist + 0.3 * a_dist 
                    for s_dist, a_dist in zip(struct_dists, act_dists)
                ]
                
                # Minimum distance to selected set
                min_dist = min(combined_dists)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected.append(best_candidate)
                candidates.remove(best_candidate)
        
        return selected
    
    def _analyze_challenge_distribution(self, challenges: Dict[str, np.ndarray],
                                      difficulty_scores: np.ndarray,
                                      train_idx: np.ndarray,
                                      test_idx: np.ndarray) -> Dict:
        """Analyze distribution of solubility challenges"""
        analysis = {}
        
        # Overall difficulty statistics
        analysis['difficulty_stats'] = {
            'train_mean': float(np.mean(difficulty_scores[train_idx])),
            'train_std': float(np.std(difficulty_scores[train_idx])),
            'test_mean': float(np.mean(difficulty_scores[test_idx])),
            'test_std': float(np.std(difficulty_scores[test_idx])),
            'ks_statistic': float(ks_2samp(
                difficulty_scores[train_idx], 
                difficulty_scores[test_idx]
            ).statistic)
        }
        
        # Individual challenge analysis
        for challenge_name, challenge_scores in challenges.items():
            analysis[challenge_name] = {
                'train_mean': float(np.mean(challenge_scores[train_idx])),
                'test_mean': float(np.mean(challenge_scores[test_idx])),
                'train_high_risk': int(np.sum(challenge_scores[train_idx] > 0.7)),
                'test_high_risk': int(np.sum(challenge_scores[test_idx] > 0.7))
            }
        
        return analysis
    
    def _optimize_ensemble_weights(self, previous_splits: Dict,
                                 targets: np.ndarray,
                                 uncertainty_scores: np.ndarray) -> Dict:
        """
        Optimize ensemble weights using differential evolution.
        
        Based on Storn & Price (1997) differential evolution algorithm.
        """
        # Define methods to optimize
        methods = ['chemical_space', 'physchem', 'activity_cliff', 'solubility_aware', 'uncertainty']
        available_methods = []
        
        # Check which methods are available
        method_mapping = {
            'chemical_space': 'chemical_space_coverage',
            'physchem': 'physchem',
            'activity_cliff': 'activity_cliff',
            'solubility_aware': 'solubility_aware'
        }
        
        for method, split_name in method_mapping.items():
            if split_name in previous_splits and previous_splits[split_name]:
                available_methods.append(method)
        
        # Always include uncertainty
        available_methods.append('uncertainty')
        
        def objective(weights):
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Create ensemble scores
            ensemble_scores = np.zeros(len(targets))
            
            for i, method in enumerate(available_methods[:-1]):
                split_name = method_mapping.get(method, method)
                if split_name in previous_splits and previous_splits[split_name]:
                    test_indices = previous_splits[split_name].get('test_idx', [])
                    ensemble_scores[test_indices] += weights[i]
            
            # Add uncertainty contribution
            if 'uncertainty' in available_methods:
                uncertainty_idx = available_methods.index('uncertainty')
                uncertainty_ranks = np.argsort(uncertainty_scores)[::-1]
                top_uncertain = uncertainty_ranks[:int(len(targets) * 0.2)]
                ensemble_scores[top_uncertain] += weights[uncertainty_idx]
            
            # Select test set
            test_size = int(len(targets) * 0.2)
            test_idx = np.argsort(ensemble_scores)[-test_size:]
            train_idx = np.setdiff1d(np.arange(len(targets)), test_idx)
            
            # Evaluate quality
            quality = self._evaluate_split_quality(train_idx, test_idx, targets)
            
            return -quality  # Minimize negative quality
        
        # Optimization bounds
        bounds = [(0.1, 0.4)] * len(available_methods)
        
        # Run differential evolution
        result = differential_evolution(
            objective, 
            bounds, 
            seed=self.random_state,
            maxiter=50,
            popsize=10
        )
        
        # Create weight dictionary
        optimal_weights = result.x / np.sum(result.x)
        weight_dict = {}
        
        for i, method in enumerate(available_methods):
            if method == 'uncertainty':
                weight_dict['uncertainty'] = float(optimal_weights[i])
            else:
                weight_dict[method] = float(optimal_weights[i])
        
        return weight_dict
    
    def _evaluate_split_quality(self, train_idx: np.ndarray, 
                              test_idx: np.ndarray,
                              targets: np.ndarray) -> float:
        """
        Evaluate split quality using multiple criteria.
        
        Quality = (distribution_quality + range_quality + size_quality) / 3
        """
        train_targets = targets[train_idx]
        test_targets = targets[test_idx]
        
        # 1. Distribution similarity (KS statistic)
        ks_stat = ks_2samp(train_targets, test_targets).statistic
        distribution_quality = 1 - ks_stat  # Lower KS is better
        
        # 2. Range coverage
        train_range = np.ptp(train_targets)
        test_range = np.ptp(test_targets)
        range_quality = min(test_range / (train_range + 1e-8), 1.0)
        
        # 3. Size balance
        size_ratio = len(test_idx) / (len(train_idx) + len(test_idx))
        size_quality = 1 - abs(size_ratio - 0.2) / 0.2  # Target 20% test
        
        # Combined quality
        quality = (distribution_quality + range_quality + size_quality) / 3
        
        return quality
    
    def _analyze_ensemble_split(self, previous_splits: Dict,
                              train_idx: np.ndarray,
                              test_idx: np.ndarray,
                              weights: Dict) -> Dict:
        """Analyze ensemble split composition"""
        analysis = {
            'overall_agreement': 0,
            'pairwise_agreements': {},
            'diversity_score': 0,
            'method_contributions': {}
        }
        
        test_set = set(test_idx)
        
        # Get valid methods
        methods = [m for m in previous_splits if previous_splits[m] and 'test_idx' in previous_splits[m]]
        
        # Calculate pairwise agreements
        agreements = []
        for i, method1 in enumerate(methods):
            test_set1 = set(previous_splits[method1]['test_idx'])
            
            for method2 in methods[i+1:]:
                test_set2 = set(previous_splits[method2]['test_idx'])
                
                # Jaccard similarity
                intersection = len(test_set1 & test_set2)
                union = len(test_set1 | test_set2)
                
                if union > 0:
                    jaccard = intersection / union
                    agreements.append(jaccard)
                    analysis['pairwise_agreements'][f"{method1}_{method2}"] = float(jaccard)
        
        if agreements:
            analysis['overall_agreement'] = float(np.mean(agreements))
        
        # Diversity score (inverse of agreement)
        analysis['diversity_score'] = 1 - analysis['overall_agreement']
        
        # Method contributions
        for method in methods:
            method_test = set(previous_splits[method]['test_idx'])
            contribution = len(method_test & test_set) / len(test_set)
            analysis['method_contributions'][method] = float(contribution)
        
        return analysis
    
    def _analyze_clusters(self, features: np.ndarray, 
                         cluster_labels: np.ndarray,
                         targets: np.ndarray) -> Dict:
        """Analyze cluster properties"""
        unique_clusters = np.unique(cluster_labels)
        
        stats = {
            'n_clusters': len(unique_clusters),
            'cluster_sizes': [],
            'cluster_target_stats': [],
            'balance_score': 0,
            'separation_score': 0
        }
        
        # Analyze each cluster
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            size = np.sum(mask)
            stats['cluster_sizes'].append(int(size))
            
            if size > 0:
                cluster_targets = targets[mask]
                stats['cluster_target_stats'].append({
                    'mean': float(np.mean(cluster_targets)),
                    'std': float(np.std(cluster_targets)) if size > 1 else 0,
                    'size': int(size)
                })
        
        # Balance score (how evenly sized clusters are)
        size_std = np.std(stats['cluster_sizes'])
        size_mean = np.mean(stats['cluster_sizes'])
        stats['balance_score'] = float(1 / (1 + size_std / size_mean)) if size_mean > 0 else 0
        
        # Separation score (silhouette coefficient)
        if len(unique_clusters) > 1:
            # Sample for efficiency
            sample_size = min(1000, len(features))
            sample_idx = np.random.choice(len(features), sample_size, replace=False)
            stats['separation_score'] = float(silhouette_score(
                features[sample_idx], 
                cluster_labels[sample_idx]
            ))
        
        return stats
    
    def _create_distant_cluster_split(self, features: np.ndarray,
                                    cluster_labels: np.ndarray,
                                    n_clusters: int,
                                    targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create split by assigning maximally distant clusters"""
        # Calculate cluster information
        cluster_info = []
        
        for i in range(n_clusters):
            mask = cluster_labels == i
            if np.any(mask):
                indices = np.where(mask)[0]
                center = np.mean(features[mask], axis=0)
                
                cluster_info.append({
                    'id': i,
                    'center': center,
                    'size': len(indices),
                    'indices': indices,
                    'mean_target': float(np.mean(targets[indices]))
                })
        
        # Handle edge case
        if len(cluster_info) < 2:
            indices = np.arange(len(features))
            np.random.RandomState(self.random_state).shuffle(indices)
            split_point = int(0.8 * len(indices))
            return indices[:split_point], indices[split_point:]
        
        # Calculate pairwise cluster distances
        n_actual = len(cluster_info)
        distances = np.zeros((n_actual, n_actual))
        
        for i in range(n_actual):
            for j in range(i+1, n_actual):
                dist = np.linalg.norm(cluster_info[i]['center'] - cluster_info[j]['center'])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Find two most distant clusters
        i, j = np.unravel_index(np.argmax(distances), distances.shape)
        train_clusters = [i]
        test_clusters = [j]
        
        # Assign remaining clusters
        unassigned = list(range(n_actual))
        unassigned.remove(i)
        unassigned.remove(j)
        
        train_size = cluster_info[i]['size']
        test_size = cluster_info[j]['size']
        target_train_size = int(0.8 * sum(c['size'] for c in cluster_info))
        
        while unassigned:
            best_cluster = None
            best_assignment = None
            best_score = -np.inf
            
            for cluster_id in unassigned:
                cluster_size = cluster_info[cluster_id]['size']
                
                # Calculate average distance to train and test clusters
                train_dist = np.mean([distances[cluster_id, t] for t in train_clusters])
                test_dist = np.mean([distances[cluster_id, t] for t in test_clusters])
                
                # Decide assignment based on size constraints and distances
                if train_size + cluster_size <= target_train_size:
                    # Can go to train, prefer if far from test
                    if test_dist > best_score:
                        best_score = test_dist
                        best_cluster = cluster_id
                        best_assignment = 'train'
                else:
                    # Must go to test, prefer if far from train
                    if train_dist > best_score:
                        best_score = train_dist
                        best_cluster = cluster_id
                        best_assignment = 'test'
            
            # Fallback assignment
            if best_cluster is None:
                best_cluster = unassigned[0]
                best_assignment = 'train' if train_size < test_size else 'test'
            
            # Make assignment
            if best_assignment == 'train':
                train_clusters.append(best_cluster)
                train_size += cluster_info[best_cluster]['size']
            else:
                test_clusters.append(best_cluster)
                test_size += cluster_info[best_cluster]['size']
            
            unassigned.remove(best_cluster)
        
        # Collect indices
        train_idx = []
        test_idx = []
        
        for cluster_id in train_clusters:
            train_idx.extend(cluster_info[cluster_id]['indices'])
        
        for cluster_id in test_clusters:
            test_idx.extend(cluster_info[cluster_id]['indices'])
        
        return np.array(train_idx), np.array(test_idx)
    
    def _calculate_cluster_distances(self, features: np.ndarray,
                                   train_idx: np.ndarray,
                                   test_idx: np.ndarray) -> Dict:
        """Calculate inter-cluster distances"""
        train_features = features[train_idx]
        test_features = features[test_idx]
        
        # Sample for efficiency
        n_sample = min(100, len(train_idx), len(test_idx))
        train_sample = train_features[np.random.choice(len(train_idx), n_sample, replace=False)]
        test_sample = test_features[np.random.choice(len(test_idx), n_sample, replace=False)]
        
        # Calculate distances
        inter_distances = cdist(test_sample, train_sample)
        
        metrics = {
            'mean_inter_distance': float(np.mean(inter_distances)),
            'min_inter_distance': float(np.min(inter_distances)),
            'max_inter_distance': float(np.max(inter_distances)),
            'median_inter_distance': float(np.median(inter_distances))
        }
        
        # Normalized distance
        max_possible = np.max(cdist(features[:n_sample], features[:n_sample]))
        metrics['normalized_mean_distance'] = metrics['mean_inter_distance'] / max_possible if max_possible > 0 else 0
        
        return metrics
    
    def _calculate_split_quality_metrics(self, train_idx: np.ndarray,
                                       test_idx: np.ndarray,
                                       smiles: List[str],
                                       targets: np.ndarray) -> Dict:
        """Calculate comprehensive split quality metrics"""
        metrics = {}
        
        train_targets = targets[train_idx]
        test_targets = targets[test_idx]
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = ks_2samp(train_targets, test_targets)
        metrics['ks_statistic'] = float(ks_stat)
        metrics['ks_pvalue'] = float(ks_pval)
        
        # Range coverage
        train_range = (float(np.min(train_targets)), float(np.max(train_targets)))
        test_range = (float(np.min(test_targets)), float(np.max(test_targets)))
        
        overlap = min(train_range[1], test_range[1]) - max(train_range[0], test_range[0])
        union = max(train_range[1], test_range[1]) - min(train_range[0], test_range[0])
        
        metrics['range_overlap'] = float(overlap / union) if union > 0 else 0
        
        # Entropy ratio
        train_entropy = entropy(np.histogram(train_targets, bins=10)[0] + 1)
        test_entropy = entropy(np.histogram(test_targets, bins=10)[0] + 1)
        
        metrics['entropy_ratio'] = float(test_entropy / train_entropy) if train_entropy > 0 else 1
        
        # Additional statistics
        metrics['train_target_mean'] = float(np.mean(train_targets))
        metrics['test_target_mean'] = float(np.mean(test_targets))
        metrics['train_target_std'] = float(np.std(train_targets))
        metrics['test_target_std'] = float(np.std(test_targets))
        
        return metrics
    
    def _calculate_distribution_similarity(self, dist1: pd.Series, 
                                         dist2: pd.Series) -> float:
        """
        Calculate distribution similarity using Jensen-Shannon divergence.
        
        JS divergence is symmetric and bounded [0, 1].
        """
        # Get all categories
        all_categories = set(dist1.index) | set(dist2.index)
        
        # Normalize distributions
        norm1 = np.array([dist1.get(cat, 0) for cat in all_categories])
        norm2 = np.array([dist2.get(cat, 0) for cat in all_categories])
        
        norm1 = norm1 / (np.sum(norm1) + 1e-8)
        norm2 = norm2 / (np.sum(norm2) + 1e-8)
        
        # Jensen-Shannon divergence
        m = 0.5 * (norm1 + norm2)
        js_div = 0.5 * entropy(norm1, m) + 0.5 * entropy(norm2, m)
        
        # Convert to similarity (1 - sqrt(JS))
        similarity = 1 - np.sqrt(js_div)
        
        return float(similarity)
    
    def _calculate_coverage_ratio(self, fps: List, train_idx: np.ndarray, 
                                test_idx: np.ndarray) -> float:
        """Calculate chemical space coverage ratio"""
        # Sample for efficiency
        n_sample = min(100, len(test_idx))
        test_sample = np.random.choice(test_idx, n_sample, replace=False)
        
        covered = 0
        coverage_threshold = 0.3  # Similarity threshold for coverage
        
        for t_idx in test_sample:
            # Check if any training molecule is similar enough
            for tr_idx in train_idx[:500]:  # Limit checks for efficiency
                sim = DataStructs.TanimotoSimilarity(fps[t_idx], fps[tr_idx])
                if sim > coverage_threshold:
                    covered += 1
                    break
        
        return covered / n_sample
    
    def _calculate_fingerprints_with_errors(self, smiles_list: List[str]) -> Tuple[List, List]:
        """Calculate fingerprints with error tracking"""
        fps = []
        failed_molecules = []
        
        for idx, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
                    fps.append(fp)
                else:
                    # Create empty fingerprint for failed molecules
                    fp = DataStructs.ExplicitBitVect(2048)
                    fps.append(fp)
                    failed_molecules.append((idx, smi, "Invalid SMILES"))
            except Exception as e:
                fp = DataStructs.ExplicitBitVect(2048)
                fps.append(fp)
                failed_molecules.append((idx, smi, str(e)))
        
        return fps, failed_molecules
    
    def _fp_to_numpy(self, fp) -> np.ndarray:
        """Convert RDKit fingerprint to numpy array"""
        arr = np.zeros(2048, dtype=np.uint8)
        if fp is not None:
            DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    
    def _validate_all_splits(self, splits: Dict, smiles: List[str], 
                           targets: np.ndarray):
        """Validate all splits for correctness"""
        self.logger.info("\n=== Validating all splits ===")
        
        for split_name, split_data in splits.items():
            if split_data is None:
                continue
                
            train_idx = split_data.get('train_idx', [])
            test_idx = split_data.get('test_idx', [])
            
            # Basic checks
            assert len(set(train_idx) & set(test_idx)) == 0, f"{split_name}: Train/test overlap detected"
            assert len(train_idx) + len(test_idx) <= len(smiles), f"{split_name}: Index out of bounds"
            
            # Size checks
            if split_name != 'test_only':
                if len(train_idx) < 10 or len(test_idx) < 5:
                    self.logger.warning(f"  ⚠ {split_name}: Very small split sizes")
            
            # Distribution check
            if len(train_idx) > 0 and len(test_idx) > 0:
                train_targets = targets[train_idx]
                test_targets = targets[test_idx]
                
                ks_stat, ks_pval = ks_2samp(train_targets, test_targets)
                
                if ks_pval < 0.01:
                    self.logger.info(f"  ℹ {split_name}: Significantly different distributions (p={ks_pval:.3f})")
                    self.logger.info(f"    This is expected for some methods (e.g., scaffold, cluster)")
        
        self.logger.info("[OK] All splits validated successfully")
    
    def _save_split(self, name: str, split_type: str, split_data: Dict):
        """Save split data to disk"""
        abbrev = self.split_abbrev.get(split_type, split_type[:2])
        
        # Validate data
        assert 'train_idx' in split_data and 'test_idx' in split_data
        assert 'train_smiles' in split_data and 'test_smiles' in split_data
        assert 'train_targets' in split_data and 'test_targets' in split_data
        
        # Save train data
        if len(split_data['train_idx']) > 0:
            train_df = pl.DataFrame({
                'smiles': split_data['train_smiles'],
                'target': split_data['train_targets']
            })
            train_path = self.output_dir / 'train' / abbrev / f'{abbrev}_{name}_train.csv'
            train_df.write_csv(train_path)
        
        # Save test data
        if len(split_data['test_idx']) > 0:
            test_df = pl.DataFrame({
                'smiles': split_data['test_smiles'],
                'target': split_data['test_targets']
            })
            test_path = self.output_dir / 'test' / abbrev / f'{abbrev}_{name}_test.csv'
            test_df.write_csv(test_path)
        
        # Save metadata
        metadata = {k: v for k, v in split_data.items() 
                   if k not in ['train_idx', 'test_idx', 'train_smiles', 'test_smiles', 'train_targets', 'test_targets']}
        
        if metadata:
            metadata_path = self.output_dir / 'split_reports' / f'{name}_{split_type}_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
    
    def _save_split_report(self, name: str):
        """Save comprehensive split report"""
        report = self.split_reports[name]
        
        # Add summary
        report['summary'] = self._generate_split_summary(report)
        
        # Add timing information
        report['timing_info'] = self.timing_info
        
        # Save JSON report
        json_path = self.reports_dir / f'{name}_split_report_v{self.version}.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save Excel report
        excel_path = self.reports_dir / f'{name}_split_report_v{self.version}.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Dataset info sheet
            info_df = pd.DataFrame([report['dataset_info']])
            info_df.to_excel(writer, sheet_name='Dataset_Info', index=False)
            
            # Summary sheet
            summary_df = pd.DataFrame(report['summary'])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Split details sheets
            for split_method, split_report in report['splits'].items():
                if split_report['success'] and 'statistics' in split_report:
                    stats_df = pd.DataFrame([split_report['statistics']])
                    sheet_name = f'{split_method[:20]}_stats'
                    stats_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # References sheet
            ref_data = []
            for method, split_report in report['splits'].items():
                for ref in split_report.get('references', []):
                    ref_data.append({
                        'Method': method,
                        'Reference': ref
                    })
            
            if ref_data:
                ref_df = pd.DataFrame(ref_data)
                ref_df.to_excel(writer, sheet_name='References', index=False)
            
            # Solubility statistics sheet
            if 'solubility_statistics' in report['dataset_info']:
                sol_stats = report['dataset_info']['solubility_statistics']
                sol_df = pd.DataFrame([sol_stats['log_s']])
                sol_df.to_excel(writer, sheet_name='Solubility_Stats', index=False)
                
                dist_df = pd.DataFrame([sol_stats['distribution']])
                dist_df.to_excel(writer, sheet_name='Solubility_Distribution', index=False)
            
            # Timing information
            if self.timing_info:
                timing_df = pd.DataFrame(list(self.timing_info.items()), 
                                        columns=['Method', 'Time_seconds'])
                timing_df.to_excel(writer, sheet_name='Timing', index=False)
        
        self.logger.info(f"[OK] Saved comprehensive report: {json_path.name} and {excel_path.name}")
    
    def _generate_split_summary(self, report: Dict) -> List[Dict]:
        """Generate summary of all splits"""
        summary = []
        
        for method, split_report in report['splits'].items():
            if split_report['success']:
                stats = split_report.get('statistics', {})
                
                summary_entry = {
                    'Method': method,
                    'Train_Size': stats.get('train_size', 0),
                    'Test_Size': stats.get('test_size', 0),
                    'Test_Ratio': stats.get('test_size', 0) / (stats.get('train_size', 0) + stats.get('test_size', 0)) if (stats.get('train_size', 0) + stats.get('test_size', 0)) > 0 else 0,
                    'Success': split_report['success'],
                    'Time_s': self.timing_info.get(method, 0)
                }
                
                # Add method-specific metrics
                if method == 'random':
                    summary_entry['Stratified'] = stats.get('is_stratified', False)
                    summary_entry['Balance'] = stats.get('solubility_balance', 0)
                elif method == 'scaffold':
                    summary_entry['N_Scaffolds'] = stats.get('n_scaffolds', 0)
                elif method == 'chemical_space_coverage':
                    summary_entry['Coverage'] = stats.get('coverage_metrics', {}).get('overall_coverage', 0)
                elif method == 'cluster':
                    summary_entry['N_Clusters'] = stats.get('n_clusters', 0)
                    summary_entry['Balance_Score'] = stats.get('cluster_balance', 0)
                elif method == 'physchem':
                    summary_entry['Property_Coverage'] = stats.get('coverage_metrics', {}).get('overall_coverage', 0)
                    summary_entry['N_Properties'] = stats.get('n_properties_used', 0)
                elif method == 'activity_cliff':
                    summary_entry['N_Cliff_Pairs'] = stats.get('n_cliff_pairs', 0)
                    summary_entry['Cliff_Coverage'] = stats.get('cliff_analysis', {}).get('cliff_coverage', 0)
                elif method == 'solubility_aware':
                    summary_entry['Mean_Test_Difficulty'] = stats.get('challenge_analysis', {}).get('difficulty_stats', {}).get('test_mean', 0)
                elif method == 'time_series':
                    summary_entry['Temporal_Drift'] = stats.get('temporal_analysis', {}).get('temporal_drift', 0)
                elif method == 'ensemble':
                    summary_entry['Agreement'] = stats.get('ensemble_analysis', {}).get('overall_agreement', 0)
                    summary_entry['Diversity'] = stats.get('ensemble_analysis', {}).get('diversity_score', 0)
                
                summary.append(summary_entry)
        
        return summary
    
    def _print_split_summary(self, splits: Dict):
        """Print summary of all splits"""
        self.logger.info("\n" + "="*80)
        self.logger.info("SPLIT SUMMARY")
        self.logger.info("="*80)
        
        successful_splits = sum(1 for s in splits.values() if s is not None)
        self.logger.info(f"Total splits created: {successful_splits}/{len(self.split_abbrev)}")
        
        for split_name, split_data in splits.items():
            if split_data:
                train_size = len(split_data['train_idx'])
                test_size = len(split_data['test_idx'])
                test_ratio = test_size / (train_size + test_size) if (train_size + test_size) > 0 else 0
                
                self.logger.info(f"\n{split_name.upper()}:")
                self.logger.info(f"  [OK] Train: {train_size}, Test: {test_size} (Test ratio: {test_ratio:.2%})")
                
                # Method-specific information
                if split_name == 'chemical_space_coverage' and 'coverage_metrics' in split_data:
                    self.logger.info(f"    Overall coverage: {split_data['coverage_metrics']['overall_coverage']:.2%}")
                elif split_name == 'physchem' and 'n_clusters' in split_data:
                    self.logger.info(f"    Number of clusters: {split_data['n_clusters']}")
                    self.logger.info(f"    PCA variance explained: {split_data.get('pca_variance_explained', 0):.2%}")
                elif split_name == 'activity_cliff' and 'cliff_analysis' in split_data:
                    self.logger.info(f"    Cliff pairs found: {split_data['n_cliff_pairs']}")
                    self.logger.info(f"    Cliff molecules in test: {split_data['cliff_analysis']['cliff_molecules_in_test']}")
                elif split_name == 'solubility_aware' and 'challenge_analysis' in split_data:
                    analysis = split_data['challenge_analysis']
                    self.logger.info(f"    Test set mean difficulty: {analysis['difficulty_stats']['test_mean']:.3f}")
                elif split_name == 'ensemble' and 'optimal_weights' in split_data:
                    self.logger.info(f"    Optimal weights:")
                    for component, weight in split_data['optimal_weights'].items():
                        self.logger.info(f"      - {component}: {weight:.3f}")
                
                # Timing
                if split_name in self.timing_info:
                    self.logger.info(f"    Time: {self.timing_info[split_name]:.2f}s")
            else:
                self.logger.info(f"\n{split_name.upper()}: [X] Failed")
        
        self.logger.info("\n" + "="*80)
        self.logger.info(f"All splits completed! Results saved in: {self.output_dir}")
        self.logger.info("="*80)


# Main splitter class for backward compatibility
class DataSplitter(AdvancedDataSplitter):
    """Alias for AdvancedDataSplitter maintaining backward compatibility"""
    pass


# ===== Module Testing =====
if __name__ == "__main__":
    # Test the complete implementation
    from pathlib import Path
    
    print("\n" + "="*80)
    print("QSAR DATA SPLITTERS MODULE TEST")
    print("Version 10.0 - Complete Scientific Implementation")
    print("="*80 + "\n")
    
    # Test data directory
    output_dir = Path("./test_splits")
    
    # Create solubility context
    solubility_context = SolubilityContext(
        temperature=25.0,
        pH=7.4,
        measurement_method="shake-flask"
    )
    
    # Initialize splitter
    print("Initializing splitter...")
    splitter = AdvancedDataSplitter(
        output_dir, 
        solubility_context=solubility_context,
        verbose=True
    )
    
    # Generate test dataset
    print("\nGenerating test dataset...")
    
    # Example SMILES (diverse structures)
    smiles = [
        # Small molecules
        "CCO", "CC(C)O", "CCCO", "CCCCO", "CC(C)(C)O",
        # Aromatics
        "c1ccccc1", "c1ccc(O)cc1", "c1ccc(C)cc1", "c1ccc(Cl)cc1",
        # Acids
        "CC(=O)O", "CCC(=O)O", "CCCC(=O)O", "c1ccc(C(=O)O)cc1",
        # Amines
        "CN", "CCN", "CCCN", "c1ccc(N)cc1",
        # Complex structures
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC1=C(C=C(C=C1)C(C)C)C(=O)O",  # Similar to ibuprofen
        "O=C(O)c1ccccc1O",  # Salicylic acid
        "CC(=O)NC1=CC=C(O)C=C1",  # Acetaminophen
        # More diverse structures
        "C1CCC(CC1)O", "C1CCCCC1", "C1=CC=CC=C1C=O",
        "CCOC(=O)C", "CC(C)=O", "CCC=O",
        "CCCCCCO", "CCCCCCCO", "CCCCCCCCO",
    ] * 4  # Replicate to get 100+ molecules
    
    # Add some variations
    import random
    random.seed(42)
    for i in range(20):
        base = random.choice(smiles[:20])
        # Simple modifications
        if "C" in base:
            modified = base.replace("C", "CC", 1)
            smiles.append(modified)
    
    # Simulated solubility values (log S) with realistic distribution
    np.random.seed(42)
    n_molecules = len(smiles)
    
    # Create bimodal distribution (common in solubility data)
    targets1 = np.random.normal(-3.0, 1.2, n_molecules // 2)  # Moderate solubility
    targets2 = np.random.normal(-5.5, 0.8, n_molecules // 2)  # Poor solubility
    targets = np.concatenate([targets1, targets2])
    np.random.shuffle(targets)
    
    # Add some outliers
    targets[0] = -8.5  # Very poor
    targets[1] = 0.5   # Very good
    
    print(f"Dataset size: {len(smiles)} molecules")
    print(f"Solubility range: [{targets.min():.2f}, {targets.max():.2f}] log S")
    print(f"Solubility mean: {targets.mean():.2f} ± {targets.std():.2f} log S")
    
    # Create all splits
    print("\nCreating all splits...")
    print("-" * 80)
    
    splits = splitter.create_all_splits(
        name="test_dataset",
        smiles=smiles,
        targets=targets,
        is_test_only=False,
        measurement_metadata={
            'source': 'synthetic_test_data',
            'measurement_date': '2024-01',
            'lab': 'computational_test'
        }
    )
    
    # Print final summary
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults saved in: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - Split data: train/<method>/ and test/<method>/")
    print("  - Reports: split_reports/")
    print("  - Comprehensive report: test_dataset_split_report_v10.0.json/xlsx")
    
    # Print method summary
    print("\nMethods implemented:")
    for i, (method, abbrev) in enumerate(SPLIT_ABBREVIATIONS.items(), 1):
        status = "[OK]" if method in splits and splits[method] is not None else "[X]"
        print(f"  {i:2d}. {method:<25} ({abbrev}) [{status}]")
    
    print("\n" + "="*80)
    print("Module test completed!")
    print("="*80)