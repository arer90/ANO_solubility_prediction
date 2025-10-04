# Applicability Domain Analysis Decision Report (Ultra-Strict (Regulatory))

## Executive Summary

### [ERROR] Lovric2020_logS0: NOT RECOMMENDED
- Mean AD Coverage: 0.837
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] FreeSolv: NOT RECOMMENDED
- Mean AD Coverage: 0.804
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] ws496_logS: NOT RECOMMENDED
- Mean AD Coverage: 0.772
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] delaney-processed: NOT RECOMMENDED
- Mean AD Coverage: 0.847
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] huusk: NOT RECOMMENDED
- Mean AD Coverage: 0.868
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] Lipophilicity: NOT RECOMMENDED
- Mean AD Coverage: 0.978
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] AqSolDB: NOT RECOMMENDED
- Mean AD Coverage: 0.978
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] BigSolDB: NOT RECOMMENDED
- Mean AD Coverage: 0.719
- Train-Test Similarity: 1.000
- Best Split Method: unknown


## Summary Table

| Category | Count | Datasets |
|----------|-------|----------|
| [CHECK] Recommended | 8 | Lovric2020_logS0, FreeSolv, ws496_logS, delaney-processed, huusk, Lipophilicity, AqSolDB, BigSolDB |
| [WARNING] Caution | 0 |  |
| [ERROR] Not Recommended | 0 |  |

## Ultra-Strict (Regulatory) Guidelines

Reference: ICH M7 guideline (>90% target) - Regulatory requirement for mutagenicity assessment

### Coverage Standards:
- **Excellent**: 90-95%
- **Good**: 80-90%
- **Acceptable**: 70-80%
- **Risky**: 60-70%
- **Poor**: 0-60%
- **Overfitted**: 95-101%

### Similarity Standards:
- **Excellent**: 0-20%
- **Good**: 20-40%
- **Acceptable**: 40-60%
- **Risky**: 60-75%
- **Dangerous**: >75%

### For Regulatory Submission:
- Maintain AD coverage between 90-95%
- Document all predictions outside AD
- Use consensus AD methods
- Validate with external test sets

### Key References:
- ICH M7(R1) (2017) - Assessment and control of DNA reactive (mutagenic) impurities
- FDA Guidance (2018) - M7(R1) Assessment and Control of DNA Reactive Impurities
