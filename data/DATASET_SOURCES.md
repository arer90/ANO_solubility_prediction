# Dataset Sources and Attribution

## Main Training Datasets

- **ws496_logS.csv**: Water solubility dataset with 496 compounds

- **delaney-processed.csv**: Delaney ESOL Dataset
  - **Description**: ESOL (Estimated SOLubility) dataset
  - **Reference**: Delaney, J. S. (2004). ESOL: estimating aqueous solubility directly from molecular structure. Journal of Chemical Information and Computer Sciences, 44(3), 1000-1005.

- **huusk.csv**: Huuskonen Dataset
  - **Description**: Aqueous solubility data
  - **Reference**: Huuskonen, J. (2000). Estimation of aqueous solubility for a diverse set of organic compounds based on molecular topology. Journal of Chemical Information and Computer Sciences, 40(3), 773-777.

- **Lovric2020_logS0.csv**: Lovric et al. 2020
  - **Description**: Intrinsic aqueous solubility dataset
  - **Reference**: Lovric, M., et al. (2020). Machine learning in prediction of intrinsic aqueous solubility of drug-like compounds.

## Test-Only Datasets

### FreeSolv.csv

- **Source**: FreeSolv Database
- **Description**: Experimental and calculated hydration free energy database
- **Reference**: Mobley, D. L., & Guthrie, J. P. (2014). FreeSolv: a database of experimental and calculated hydration free energies, with input files. Journal of Computer-Aided Molecular Design, 28(7), 711-720.
- **URL**: https://github.com/MobleyLab/FreeSolv

### Lipophilicity.csv

- **Source**: ChEMBL Lipophilicity Dataset
- **Description**: Experimental lipophilicity (octanol/water partition coefficient) data
- **Reference**: ChEMBL database - experimental lipophilicity measurements
- **URL**: https://www.ebi.ac.uk/chembl/

### AqSolDB.csv

- **Source**: Aqueous Solubility Database (AqSolDB)
- **Description**: Curated aqueous solubility dataset with 9,982 unique compounds
- **Reference**: Sorkun, M. C., Khetan, A., & Er, S. (2019). AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds. Scientific Data, 6(1), 143.
- **URL**: https://www.nature.com/articles/s41597-019-0151-1

### BigSolDB.csv

- **Source**: Big Solubility Database
- **Description**: Large-scale solubility dataset compiled from multiple sources
- **Reference**: Collection of solubility data from various literature sources
- **Note**: Comprehensive database for solubility prediction benchmarking

## Usage Notes

- All solubility values are in log10(mol/L) units unless otherwise specified
- SMILES strings have been standardized using RDKit
- Test-only datasets are used exclusively for external validation
