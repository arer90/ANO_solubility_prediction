# Applicability Domain Analysis Decision Report (Scientific Consensus)

## Executive Summary

### ❌ Lovric2020_logS0: LIMITED USE
- Mean AD Coverage: 0.820
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### ❌ SAMPL: LIMITED USE
- Mean AD Coverage: 0.837
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### ❌ ws496_logS: LIMITED USE
- Mean AD Coverage: 0.928
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### ❌ delaney-processed: LIMITED USE
- Mean AD Coverage: 0.824
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### ❌ huusk: LIMITED USE
- Mean AD Coverage: 0.828
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### ❌ Lipophilicity: LIMITED USE
- Mean AD Coverage: 0.866
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### ❌ curated-solubility-dataset: LIMITED USE
- Mean AD Coverage: 0.952
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### ❌ BigSolDB: LIMITED USE
- Mean AD Coverage: 0.830
- Train-Test Similarity: 1.000
- Best Split Method: unknown


## Summary Table

| Category | Count | Datasets |
|----------|-------|----------|
| ✅ Recommended | 0 |  |
| ⚠️ Caution | 0 |  |
| ❌ Not Recommended | 8 | Lovric2020_logS0, SAMPL, ws496_logS, delaney-processed, huusk, Lipophilicity, curated-solubility-dataset, BigSolDB |

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
