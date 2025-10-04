# AD Application Guide: huusk

## Quick Decision Framework

### Step 1: AD Assessment
```python
# Check if compound is inside AD
ad_inside = model.predict_ad(smiles)
confidence = model.get_confidence(smiles)
```

### Step 2: Decision Matrix

| AD Status | Confidence | Action |
|-----------|------------|--------|
| Inside | High (>0.8) | Accept prediction |
| Inside | Medium (0.5-0.8) | Accept with caution |
| Inside | Low (<0.5) | Consider validation |
| Outside | Any | Require experimental data |

### Step 3: Documentation
- Record AD assessment results
- Document decision rationale
- Track prediction performance

