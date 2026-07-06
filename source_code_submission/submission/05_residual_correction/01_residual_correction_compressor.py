"""
Day 5 (continued) - Residual correction for compressor health.
Same leave-one-engine-out method as turbine.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

ukf_results = pd.read_csv("ukf_results_full.csv")
train = pd.read_csv("train.csv")
gt = pd.read_csv("ground_truth.csv")

df = ukf_results.merge(train, on=['EngineID', 'Cycle']).merge(gt, on=['EngineID', 'Cycle'])
df['residual_c'] = df['CompressorHealth'] - df['est_eta_c']

print("=== Residual stats (before correction) ===")
print(df['residual_c'].describe())

feature_cols = ['RPM_rev_min', 'Mach', 'Altitude_m', 'Pamb_Pa', 'Tamb_K',
                 'FuelFlow_kg_s', 'Cycle', 'est_eta_c']

engines = sorted(df['EngineID'].unique())

mae_before_list, mae_after_list = [], []
all_true, all_before, all_after = [], [], []

for test_eng in engines:
    train_mask = df['EngineID'] != test_eng
    test_mask = df['EngineID'] == test_eng

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, 'residual_c']
    X_test = df.loc[test_mask, feature_cols]
    y_test_true = df.loc[test_mask, 'CompressorHealth']
    y_test_before = df.loc[test_mask, 'est_eta_c']

    model = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    residual_pred = model.predict(X_test)
    y_test_after = y_test_before + residual_pred

    mae_before_list.append(mean_absolute_error(y_test_true, y_test_before))
    mae_after_list.append(mean_absolute_error(y_test_true, y_test_after))

    all_true.extend(y_test_true)
    all_before.extend(y_test_before)
    all_after.extend(y_test_after)

print("\n=== Leave-one-engine-out results (Compressor) ===")
print(f"Mean MAE BEFORE correction (per-engine avg): {np.mean(mae_before_list):.4f}")
print(f"Mean MAE AFTER correction (per-engine avg):  {np.mean(mae_after_list):.4f}")

print(f"\nOverall MAE BEFORE (pooled): {mean_absolute_error(all_true, all_before):.4f}")
print(f"Overall MAE AFTER (pooled):  {mean_absolute_error(all_true, all_after):.4f}")
print(f"(Baseline MAE from Day 3 evaluation was 0.0615 - compare against that too)")

print("\n=== Linear regression sanity check ===")
mae_before_lin, mae_after_lin = [], []
for test_eng in engines:
    train_mask = df['EngineID'] != test_eng
    test_mask = df['EngineID'] == test_eng
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, 'residual_c']
    X_test = df.loc[test_mask, feature_cols]
    y_test_true = df.loc[test_mask, 'CompressorHealth']
    y_test_before = df.loc[test_mask, 'est_eta_c']

    lin = LinearRegression().fit(X_train, y_train)
    residual_pred = lin.predict(X_test)
    y_test_after = y_test_before + residual_pred

    mae_before_lin.append(mean_absolute_error(y_test_true, y_test_before))
    mae_after_lin.append(mean_absolute_error(y_test_true, y_test_after))

print(f"Linear model - Mean MAE BEFORE: {np.mean(mae_before_lin):.4f}")
print(f"Linear model - Mean MAE AFTER:  {np.mean(mae_after_lin):.4f}")
