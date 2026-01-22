"""
GNN Training Script for Melting Point Prediction
Uses Chemprop D-MPNN with 5-fold cross-validation
"""
import warnings
warnings.filterwarnings('ignore')

import os
os.chdir(r'C:/Users/tkasiror/Desktop/Thermophysical Property Melting Point')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from chemprop.models import MPNN

import lightning.pytorch as pl

print("=" * 60)
print("  GNN TRAINING FOR MELTING POINT PREDICTION")
print("=" * 60)

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Load data
print("\n[1/6] Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
print(f"  Training samples: {len(train_df)}")
print(f"  Test samples: {len(test_df)}")

# Prepare data
train_smiles = train_df['SMILES'].tolist()
train_targets = train_df['Tm'].tolist()
test_smiles = test_df['SMILES'].tolist()

# Create datapoints
print("\n[2/6] Creating molecule datapoints...")
featurizer = SimpleMoleculeMolGraphFeaturizer()

train_datapoints = [
    MoleculeDatapoint.from_smi(smi, y=np.array([target]))
    for smi, target in zip(train_smiles, train_targets)
]

test_datapoints = [
    MoleculeDatapoint.from_smi(smi)
    for smi in test_smiles
]
print(f"  Created {len(train_datapoints)} training datapoints")
print(f"  Created {len(test_datapoints)} test datapoints")

# Model config (CPU-optimized)
CONFIG = {
    'hidden_size': 300,
    'depth': 3,
    'ffn_hidden_size': 300,
    'ffn_num_layers': 2,
    'dropout': 0.1,
    'batch_size': 50,
    'epochs': 50,
    'patience': 10,
    'lr': 1e-3,
}

def create_mpnn_model(config):
    mp = BondMessagePassing(
        d_h=config['hidden_size'],
        depth=config['depth'],
        dropout=config['dropout']
    )
    agg = MeanAggregation()
    predictor = RegressionFFN(
        input_dim=config['hidden_size'],
        hidden_dim=config['ffn_hidden_size'],
        n_layers=config['ffn_num_layers'],
        dropout=config['dropout'],
        n_tasks=1
    )
    model = MPNN(
        message_passing=mp,
        agg=agg,
        predictor=predictor,
        batch_norm=True
    )
    return model

# Test model creation
print("\n[3/6] Testing model creation...")
test_model = create_mpnn_model(CONFIG)
n_params = sum(p.numel() for p in test_model.parameters())
print(f"  Model parameters: {n_params:,}")

# 5-Fold Cross-Validation
print("\n[4/6] Starting 5-fold cross-validation...")
print("  (This will take a while on CPU)")

n_folds = 5
kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

fold_results = []
fold_models = []
all_val_preds = np.zeros(len(train_datapoints))

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_datapoints), 1):
    print(f"\n  --- Fold {fold}/{n_folds} ---")

    train_data = [train_datapoints[i] for i in train_idx]
    val_data = [train_datapoints[i] for i in val_idx]

    # Create datasets
    train_dataset = MoleculeDataset(train_data, featurizer=featurizer)
    val_dataset = MoleculeDataset(val_data, featurizer=featurizer)

    # Create dataloaders
    train_loader = build_dataloader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    model = create_mpnn_model(CONFIG)

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG['epochs'],
        accelerator='cpu',
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=CONFIG['patience'],
                mode='min'
            )
        ]
    )

    # Train
    print(f"  Training...")
    trainer.fit(model, train_loader, val_loader)

    # Get predictions
    model.eval()
    val_preds = []
    val_targets_fold = []

    with torch.no_grad():
        for batch in val_loader:
            bmg, _, _, targets, *_ = batch
            preds = model(bmg)
            val_preds.extend(preds.squeeze().tolist())
            val_targets_fold.extend(targets.squeeze().tolist())

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(val_targets_fold, val_preds))
    mae = mean_absolute_error(val_targets_fold, val_preds)
    r2 = r2_score(val_targets_fold, val_preds)

    print(f"  Results: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    fold_results.append({'rmse': rmse, 'mae': mae, 'r2': r2})
    fold_models.append(model)

    # Store OOF predictions
    for i, idx in enumerate(val_idx):
        all_val_preds[idx] = val_preds[i]

# Summary
print("\n" + "=" * 60)
print("  CROSS-VALIDATION SUMMARY")
print("=" * 60)

cv_results = pd.DataFrame(fold_results)
print("\nPer-Fold Results:")
print(cv_results.to_string())

print(f"\nMean ± Std:")
print(f"  RMSE: {cv_results['rmse'].mean():.4f} ± {cv_results['rmse'].std():.4f}")
print(f"  MAE:  {cv_results['mae'].mean():.4f} ± {cv_results['mae'].std():.4f}")
print(f"  R²:   {cv_results['r2'].mean():.4f} ± {cv_results['r2'].std():.4f}")

# Overall OOF metrics
y_true = np.array(train_targets)
oof_rmse = np.sqrt(mean_squared_error(y_true, all_val_preds))
oof_mae = mean_absolute_error(y_true, all_val_preds)
oof_r2 = r2_score(y_true, all_val_preds)

print(f"\nOverall Out-of-Fold Metrics:")
print(f"  RMSE: {oof_rmse:.4f}")
print(f"  MAE:  {oof_mae:.4f}")
print(f"  R²:   {oof_r2:.4f}")

print(f"\nComparison with Baseline (XGBoost):")
print(f"  XGBoost: RMSE=43.50, R²=0.739")
print(f"  D-MPNN:  RMSE={oof_rmse:.2f}, R²={oof_r2:.3f}")

# Visualization
print("\n[5/6] Creating visualization...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.scatter(y_true, all_val_preds, alpha=0.4, s=20, c='steelblue')
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
ax.set_xlabel('Actual Melting Point (K)', fontsize=12)
ax.set_ylabel('Predicted Melting Point (K)', fontsize=12)
ax.set_title(f'D-MPNN (GNN)\nR² = {oof_r2:.4f}, RMSE = {oof_rmse:.2f}', fontsize=13)

ax = axes[1]
residuals = all_val_preds - y_true
ax.scatter(y_true, residuals, alpha=0.4, s=20, c='steelblue')
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Actual Melting Point (K)', fontsize=12)
ax.set_ylabel('Residual (Predicted - Actual)', fontsize=12)
ax.set_title('Residual Plot', fontsize=13)

plt.tight_layout()
plt.savefig('gnn_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
print("  Saved: gnn_actual_vs_predicted.png")

# Generate test predictions
print("\n[6/6] Generating test predictions...")
test_dataset = MoleculeDataset(test_datapoints, featurizer=featurizer)
test_loader = build_dataloader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

all_test_preds = []

for fold_num, model in enumerate(fold_models, 1):
    model.eval()
    fold_preds = []

    with torch.no_grad():
        for batch in test_loader:
            bmg, _, _, targets, *_ = batch
            preds = model(bmg)
            fold_preds.extend(preds.squeeze().tolist())

    all_test_preds.append(fold_preds)
    print(f"  Fold {fold_num}: {len(fold_preds)} predictions")

# Average predictions
ensemble_preds = np.mean(all_test_preds, axis=0)

print(f"\nEnsemble predictions statistics:")
print(f"  Min: {ensemble_preds.min():.2f}")
print(f"  Max: {ensemble_preds.max():.2f}")
print(f"  Mean: {ensemble_preds.mean():.2f}")

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Tm': ensemble_preds
})
submission.to_csv('submission_gnn.csv', index=False)
print(f"\nSaved: submission_gnn.csv")

print("\n" + "=" * 60)
print("  TRAINING COMPLETE!")
print("=" * 60)
print(f"\nFinal Results:")
print(f"  D-MPNN RMSE: {oof_rmse:.2f} (baseline XGBoost: 43.50)")
print(f"  D-MPNN R²:   {oof_r2:.3f} (baseline XGBoost: 0.739)")
improvement = ((43.50 - oof_rmse) / 43.50) * 100
if improvement > 0:
    print(f"  Improvement: {improvement:.1f}% better RMSE")
else:
    print(f"  Result: {-improvement:.1f}% higher RMSE than baseline")
