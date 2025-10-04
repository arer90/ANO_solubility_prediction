# Applicability Domain Analysis Decision Report (Context-Dependent)

## Executive Summary

### [ERROR] Lovric2020_logS0: NOT SUITABLE
- Mean AD Coverage: 0.837
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] FreeSolv: NOT SUITABLE
- Mean AD Coverage: 0.804
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] ws496_logS: NOT SUITABLE
- Mean AD Coverage: 0.772
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] delaney-processed: NOT SUITABLE
- Mean AD Coverage: 0.847
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] huusk: NOT SUITABLE
- Mean AD Coverage: 0.868
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] Lipophilicity: NOT SUITABLE
- Mean AD Coverage: 0.978
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] AqSolDB: NOT SUITABLE
- Mean AD Coverage: 0.978
- Train-Test Similarity: 1.000
- Best Split Method: unknown

### [ERROR] BigSolDB: NOT SUITABLE
- Mean AD Coverage: 0.719
- Train-Test Similarity: 1.000
- Best Split Method: unknown


## Summary Table

| Category | Count | Datasets |
|----------|-------|----------|
| [CHECK] Recommended | 0 |  |
| [WARNING] Caution | 0 |  |
| [ERROR] Not Recommended | 8 | Lovric2020_logS0, FreeSolv, ws496_logS, delaney-processed, huusk, Lipophilicity, AqSolDB, BigSolDB |

## Context-Dependent Guidelines

Reference: Application-specific thresholds based on literature consensus

### Coverage Standards:

### Similarity Standards:
- **Excellent**: 0-20%
- **Good**: 20-40%
- **Acceptable**: 40-60%
- **Risky**: 60-75%
- **Dangerous**: >75%

### For Context-Specific Use:
- Adjust thresholds based on application
- Consider risk tolerance
- Document decision criteria
- Use tiered confidence levels

### Key References:
- Weaver & Gleeson (2008) J Cheminform 41:1-7 - 70% coverage for lead optimization
- Tetko et al. (2008) Drug Discov Today 13:157-163 - Context-dependent AD thresholds
- Dragos et al. (2009) J Chem Inf Model 49:1762-1776 - 75% for research applications
