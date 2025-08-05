# Applicability Domain Analysis Decision Report (Scientific Consensus)

## Executive Summary

### [ERROR] Lovric2020_logS0: LIMITED USE
- Mean AD Coverage: 0.758
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] SAMPL: LIMITED USE
- Mean AD Coverage: 0.783
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] ws496_logS: LIMITED USE
- Mean AD Coverage: 0.900
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] delaney-processed: LIMITED USE
- Mean AD Coverage: 0.755
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] huusk: LIMITED USE
- Mean AD Coverage: 0.772
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] Lipophilicity: LIMITED USE
- Mean AD Coverage: 0.944
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] curated-solubility-dataset: LIMITED USE
- Mean AD Coverage: 0.961
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] BigSolDB: LIMITED USE
- Mean AD Coverage: 0.716
- Train-Test Similarity: 1.000
- Best Split Method: unknown


## Summary Table

| Category | Count | Datasets |
|----------|-------|----------|
| [CHECK] Recommended | 0 |  |
| [WARNING] Caution | 0 |  |
| [ERROR] Not Recommended | 8 | Lovric2020_logS0, SAMPL, ws496_logS, delaney-processed, huusk, Lipophilicity, curated-solubility-dataset, BigSolDB |

## Scientific Consensus Guidelines

Reference: Sahigara et al. (2012), Roy et al. (2015) - Practical AD implementation

### Coverage Standards:
- **Excellent**: 80-90%
- **Good**: 70-80%
- **Acceptable**: 60-70%
- **Moderate**: 50-60%
- **Limited**: 40-50%
- **Poor**: 0-40%
- **Overfitted**: 90-101%

### Similarity Standards:
- **Excellent**: 0-20%
- **Good**: 20-40%
- **Acceptable**: 40-60%
- **Risky**: 60-75%
- **Dangerous**: >75%

### For Research Applications:
- Target AD coverage of 70-80%
- Balance coverage with accuracy
- Consider ensemble approaches
- Use reliability scoring for confidence

### Key References:
- Sahigara et al. (2012) Molecules 17(5):4791-4810 - 60-80% coverage as practical range
- Roy et al. (2015) Chemometr Intell Lab Syst 145:22-29 - Standardized approach with 3σ
- Sheridan (2012) J Chem Inf Model 52(3):814-823 - 75% coverage as optimal trade-off
