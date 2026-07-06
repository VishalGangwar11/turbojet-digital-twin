"""
Day 5 - Residual correction layer for turbine health.
Learns TurbineHealth - est_eta_t as a function of flight condition,
using leave-one-engine-out cross-validation (honest evaluation given
only 10 engines - avoids overfitting to specific engines).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

ukf_results = pd.read_csv("ukf_results_full.csv")
train = pd.read_csv("train.csv")

# Merge in flight condition features (RPM, Mach, Altitude, ambient) that
# the UKF's hx() didn't have full access to when estimating turbine health
df = ukf_results.merge(train, on=['EngineID', 'Cycle'])

# Residual = what the UKF got wrong
df['residual_t'] = df['TurbineHealth'] - df['est_eta_t']

print("=== Residual stats (before correction) ===")
print(df['residual_t'].describe())

feature_cols = ['RPM_rev_min', 'Mach', 'Altitude_m', 'Pamb_Pa', 'Tamb_K',
                 'FuelFlow_kg_s', 'Cycle', 'est_eta_t']

engines = sorted(df['EngineID'].unique())

# ---- Leave-one-engine-out cross validation ----
mae_before_list, mae_after_list = [], []
all_true, all_before, all_after = [], [], []

for test_eng in engines:
    train_mask = df['EngineID'] != test_eng
    test_mask = df['EngineID'] == test_eng

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, 'residual_t']
    X_test = df.loc[test_mask, feature_cols]
    y_test_true = df.loc[test_mask, 'TurbineHealth']
    y_test_before = df.loc[test_mask, 'est_eta_t']

    model = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    residual_pred = model.predict(X_test)
    y_test_after = y_test_before + residual_pred

    mae_before = mean_absolute_error(y_test_true, y_test_before)
    mae_after = mean_absolute_error(y_test_true, y_test_after)
    mae_before_list.append(mae_before)
    mae_after_list.append(mae_after)

    all_true.extend(y_test_true)
    all_before.extend(y_test_before)
    all_after.extend(y_test_after)

print("\n=== Leave-one-engine-out results ===")
print(f"Mean MAE BEFORE correction (per-engine avg): {np.mean(mae_before_list):.4f}")
print(f"Mean MAE AFTER correction (per-engine avg):  {np.mean(mae_after_list):.4f}")

print(f"\nOverall MAE BEFORE (pooled): {mean_absolute_error(all_true, all_before):.4f}")
print(f"Overall MAE AFTER (pooled):  {mean_absolute_error(all_true, all_after):.4f}")
print(f"(Baseline MAE from Day 3 evaluation was 0.0440 - compare against that too)")

# ---- Also check simple linear model as a sanity check (less prone to overfitting) ----
print("\n=== Same test with simple Linear Regression (sanity check vs GBM) ===")
mae_before_lin, mae_after_lin = [], []
for test_eng in engines:
    train_mask = df['EngineID'] != test_eng
    test_mask = df['EngineID'] == test_eng
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, 'residual_t']
    X_test = df.loc[test_mask, feature_cols]
    y_test_true = df.loc[test_mask, 'TurbineHealth']
    y_test_before = df.loc[test_mask, 'est_eta_t']

    lin = LinearRegression().fit(X_train, y_train)
    residual_pred = lin.predict(X_test)
    y_test_after = y_test_before + residual_pred

    mae_before_lin.append(mean_absolute_error(y_test_true, y_test_before))
    mae_after_lin.append(mean_absolute_error(y_test_true, y_test_after))

print(f"Linear model - Mean MAE BEFORE: {np.mean(mae_before_lin):.4f}")
print(f"Linear model - Mean MAE AFTER:  {np.mean(mae_after_lin):.4f}")
