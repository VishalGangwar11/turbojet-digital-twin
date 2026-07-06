"""
Day 6 - Black-box ML baseline: predict health directly from raw sensors,
NO physics at all. This is the comparison judges will ask for.
Uses leave-one-engine-out CV, same as our residual-correction evaluation,
so the comparison is apples-to-apples.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

train = pd.read_csv("train.csv")
gt = pd.read_csv("ground_truth.csv")
df = train.merge(gt, on=['EngineID', 'Cycle']).sort_values(['EngineID', 'Cycle']).reset_index(drop=True)

# Raw sensor + flight condition features - NO physics equations, NO UKF
raw_features = ['P2_Pa', 'T2_K', 'P3_Pa', 'T3_K', 'P4_Pa', 'T4_K',
                 'RPM_rev_min', 'FuelFlow_kg_s', 'Pamb_Pa', 'Tamb_K',
                 'Mach', 'Altitude_m', 'Cycle']

engines = sorted(df['EngineID'].unique())
targets = {
    'Compressor': 'CompressorHealth',
    'Combustor': 'CombustorHealth',
    'Turbine': 'TurbineHealth',
}

print("=== BLACK-BOX ML BASELINE (raw sensors -> health, no physics) ===")
print("Leave-one-engine-out cross-validation, same protocol as hybrid model\n")

results_summary = {}

for label, target_col in targets.items():
    mae_list = []
    for test_eng in engines:
        train_mask = df['EngineID'] != test_eng
        test_mask = df['EngineID'] == test_eng

        X_train = df.loc[train_mask, raw_features]
        y_train = df.loc[train_mask, target_col]
        X_test = df.loc[test_mask, raw_features]
        y_test = df.loc[test_mask, target_col]

        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae_list.append(mean_absolute_error(y_test, pred))

    avg_mae = np.mean(mae_list)
    results_summary[label] = avg_mae
    print(f"{label}: Black-box ML MAE = {avg_mae:.4f}")

print("\n=== COMPARISON TABLE (fill in your hybrid numbers from Day 3/5) ===")
print(f"{'Subsystem':<12} {'Naive Baseline':<16} {'Black-box ML':<16} {'Hybrid (UKF+Residual)':<22}")
print(f"{'Compressor':<12} {'0.0615':<16} {results_summary['Compressor']:<16.4f} {'0.0195':<22}")
print(f"{'Combustor':<12} {'0.0219':<16} {results_summary['Combustor']:<16.4f} {'0.0147':<22}")
print(f"{'Turbine':<12} {'0.0440':<16} {results_summary['Turbine']:<16.4f} {'0.0236':<22}")
