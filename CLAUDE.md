# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Machine learning competition project for predicting melting point (Tm) of chemical compounds based on molecular structure. This is a regression task in the computational chemistry domain.

## Data Structure

```
data/
├── train.csv           # 2,662 samples with target Tm
├── test.csv            # 666 samples for prediction
└── sample_submission.csv
```

### Data Format
- **id**: Molecule identifier
- **SMILES**: Chemical structure notation (text representation of molecules)
- **Group 1-424**: 424 pre-engineered molecular features (likely fingerprints/functional groups)
- **Tm**: Target variable - melting point (only in train.csv)

## Development Workflow

### Environment Setup
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install pandas numpy scikit-learn xgboost lightgbm rdkit matplotlib jupyter
```

### Typical Data Loading
```python
import pandas as pd

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Features: drop id, SMILES, and target
X_train = train_df.drop(['id', 'SMILES', 'Tm'], axis=1)
y_train = train_df['Tm']
X_test = test_df.drop(['id', 'SMILES'], axis=1)
```

### Submission Format
```python
submission = pd.DataFrame({
    'id': test_df['id'],
    'Tm': predictions
})
submission.to_csv('submission.csv', index=False)
```

## Architecture Notes

- **SMILES column**: Can be parsed with RDKit for additional molecular descriptors if the 424 group features are insufficient
- **Group features**: Integer counts representing molecular substructures - can be used directly for tree-based models
- **Recommended models**: XGBoost, LightGBM, or ensemble methods for this tabular regression task
- **Validation**: Use k-fold cross-validation on training data; track RMSE/MAE

- **Running Notebooks**:
Do not make a seperate _executed notebook
I will personaly run any notebooks myself