import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Number of features: {len(train_df.columns) - 3}")  # Exclude id, SMILES, Tm

# Prepare features and target
X_train = train_df.drop(['id', 'SMILES', 'Tm'], axis=1)
y_train = train_df['Tm']
X_test = test_df.drop(['id', 'SMILES'], axis=1)

print(f"\nTarget (Tm) statistics:")
print(f"  Min: {y_train.min():.2f}")
print(f"  Max: {y_train.max():.2f}")
print(f"  Mean: {y_train.mean():.2f}")
print(f"  Std: {y_train.std():.2f}")

# Define models
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

lgbm_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Cross-validation setup
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model, X, y, model_name):
    """Evaluate model using 5-fold cross-validation"""
    print(f"\n{'='*50}")
    print(f"Training and Evaluating: {model_name}")
    print('='*50)

    # Get cross-validated predictions
    y_pred_cv = cross_val_predict(model, X, y, cv=kfold)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred_cv))
    mae = mean_absolute_error(y, y_pred_cv)
    r2 = r2_score(y, y_pred_cv)

    print(f"\nCross-Validation Results (5-Fold):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

    # Train on full data for test predictions
    print(f"\nTraining on full dataset...")
    model.fit(X, y)

    return model, {'RMSE': rmse, 'MAE': mae, 'R2': r2}

# Evaluate both models
print("\n" + "="*60)
print("MELTING POINT PREDICTION - MODEL EVALUATION")
print("="*60)

xgb_trained, xgb_metrics = evaluate_model(xgb_model, X_train, y_train, "XGBoost")
lgbm_trained, lgbm_metrics = evaluate_model(lgbm_model, X_train, y_train, "LightGBM")

# Summary comparison
print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)
print(f"\n{'Metric':<10} {'XGBoost':>15} {'LightGBM':>15} {'Best':>15}")
print("-"*55)
for metric in ['RMSE', 'MAE', 'R2']:
    xgb_val = xgb_metrics[metric]
    lgbm_val = lgbm_metrics[metric]
    if metric == 'R2':
        best = 'XGBoost' if xgb_val > lgbm_val else 'LightGBM'
    else:
        best = 'XGBoost' if xgb_val < lgbm_val else 'LightGBM'
    print(f"{metric:<10} {xgb_val:>15.4f} {lgbm_val:>15.4f} {best:>15}")

# Generate predictions on test set
print("\n" + "="*60)
print("GENERATING TEST PREDICTIONS")
print("="*60)

xgb_predictions = xgb_trained.predict(X_test)
lgbm_predictions = lgbm_trained.predict(X_test)

# Save predictions
xgb_submission = pd.DataFrame({'id': test_df['id'], 'Tm': xgb_predictions})
lgbm_submission = pd.DataFrame({'id': test_df['id'], 'Tm': lgbm_predictions})

xgb_submission.to_csv('submission_xgboost.csv', index=False)
lgbm_submission.to_csv('submission_lightgbm.csv', index=False)

print(f"\nXGBoost predictions saved to: submission_xgboost.csv")
print(f"LightGBM predictions saved to: submission_lightgbm.csv")

print(f"\nTest predictions statistics:")
print(f"  XGBoost  - Min: {xgb_predictions.min():.2f}, Max: {xgb_predictions.max():.2f}, Mean: {xgb_predictions.mean():.2f}")
print(f"  LightGBM - Min: {lgbm_predictions.min():.2f}, Max: {lgbm_predictions.max():.2f}, Mean: {lgbm_predictions.mean():.2f}")

print("\nDone!")
