"""
Day 5 (test validation) - Run the complete pipeline on test.csv, which has
NEVER been touched during physics calibration, UKF tuning, or residual
correction training. This is genuine held-out validation.
"""

import pandas as pd
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

GAMMA_AIR = 1.4
GAMMA_GAS = 1.33
CP_AIR = 1005.0
CP_GAS = 1148.0
LHV_FUEL = 43e6
R_AIR = 287.0

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
gt = pd.read_csv("ground_truth.csv")

train_df = train.merge(gt, on=['EngineID', 'Cycle']).sort_values(['EngineID', 'Cycle']).reset_index(drop=True)
test_df = test.merge(gt, on=['EngineID', 'Cycle']).sort_values(['EngineID', 'Cycle']).reset_index(drop=True)

print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
print(f"Test engines/cycles overlap check - test Cycles present: {sorted(test_df['Cycle'].unique())}")


def fx(state, dt, **kwargs):
    return state


def hx(state, Pamb, Tamb, RPM, FuelFlow, Mach=0.4, Altitude=6000):
    eta_c, eta_comb, eta_t = state
    PR_c_assumed = max(0.0001286487 * RPM + 2.624160 * Mach + 0.0000033082 * Altitude - 3.931887, 1.05)
    rho_amb = Pamb / (R_AIR * Tamb)
    air_flow = max(0.0006162388 * RPM - 0.424435 * rho_amb - 8.280180, 1.0)

    P2 = Pamb * PR_c_assumed
    T2_ideal = Tamb * (PR_c_assumed) ** ((GAMMA_AIR - 1) / GAMMA_AIR)
    T2 = Tamb + (T2_ideal - Tamb) / max(eta_c, 1e-3)

    P3 = P2 * 0.96
    total_flow = air_flow + FuelFlow
    T3 = (air_flow * CP_AIR * T2 + eta_comb * FuelFlow * LHV_FUEL) / (total_flow * CP_GAS + 1e-9)

    PR_t_assumed = min(max(-0.0000118963 * RPM + 1.334673, 0.1), 0.99)
    P4 = P3 * PR_t_assumed
    T4_ideal = T3 * (PR_t_assumed) ** ((GAMMA_GAS - 1) / GAMMA_GAS)
    T4 = T3 - eta_t * (T3 - T4_ideal)

    return np.array([P2, T2, P3, T3, P4, T4])


def run_ukf_for_engine(engine_df, Q_diag=(0.0001, 0.001, 0.0005)):
    points = MerweScaledSigmaPoints(n=3, alpha=0.5, beta=2.0, kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=3, dim_z=6, dt=1.0, fx=fx, hx=hx, points=points)
    ukf.x = np.array([0.99, 0.999, 0.995])
    ukf.P *= 0.005
    ukf.Q = np.diag(Q_diag)
    ukf.R = np.diag([857714883.88, 378.63, 790911236.82, 4529.15, 1681026493.53, 9676.53])

    estimates = []
    for _, row in engine_df.iterrows():
        z = np.array([row['P2_Pa'], row['T2_K'], row['P3_Pa'], row['T3_K'], row['P4_Pa'], row['T4_K']])
        ukf.predict()
        ukf.update(z, Pamb=row['Pamb_Pa'], Tamb=row['Tamb_K'], RPM=row['RPM_rev_min'],
                   FuelFlow=row['FuelFlow_kg_s'], Mach=row['Mach'], Altitude=row['Altitude_m'])
        estimates.append(ukf.x.copy())
    return np.array(estimates)


# ---- Step 1: Run UKF on TRAIN (to build training data for residual models) ----
train_rows = []
for eng_id in sorted(train_df['EngineID'].unique()):
    eng = train_df[train_df['EngineID'] == eng_id].sort_values('Cycle').reset_index(drop=True)
    estimates = run_ukf_for_engine(eng)
    eng = eng.copy()
    eng['est_eta_c'] = estimates[:, 0]
    eng['est_eta_comb'] = estimates[:, 1]
    eng['est_eta_t'] = estimates[:, 2]
    train_rows.append(eng)
train_full = pd.concat(train_rows, ignore_index=True)

# ---- Step 2: Run UKF on TEST (genuinely held out, never seen before) ----
test_rows = []
for eng_id in sorted(test_df['EngineID'].unique()):
    eng = test_df[test_df['EngineID'] == eng_id].sort_values('Cycle').reset_index(drop=True)
    estimates = run_ukf_for_engine(eng)
    eng = eng.copy()
    eng['est_eta_c'] = estimates[:, 0]
    eng['est_eta_comb'] = estimates[:, 1]
    eng['est_eta_t'] = estimates[:, 2]
    test_rows.append(eng)
test_full = pd.concat(test_rows, ignore_index=True)

# ---- Step 3: Train residual correction models on TRAIN ONLY ----
feature_cols = ['RPM_rev_min', 'Mach', 'Altitude_m', 'Pamb_Pa', 'Tamb_K', 'FuelFlow_kg_s', 'Cycle']

train_full['residual_c'] = train_full['CompressorHealth'] - train_full['est_eta_c']
train_full['residual_t'] = train_full['TurbineHealth'] - train_full['est_eta_t']

model_c = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42)
model_c.fit(train_full[feature_cols + ['est_eta_c']], train_full['residual_c'])

model_t = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42)
model_t.fit(train_full[feature_cols + ['est_eta_t']], train_full['residual_t'])

# ---- Step 4: Apply correction to TEST (never used in training anything) ----
test_full['est_eta_c_corrected'] = test_full['est_eta_c'] + model_c.predict(test_full[feature_cols + ['est_eta_c']])
test_full['est_eta_t_corrected'] = test_full['est_eta_t'] + model_t.predict(test_full[feature_cols + ['est_eta_t']])

# ---- Step 5: Evaluate on TEST ----
print("\n=== HELD-OUT TEST SET RESULTS ===")
print(f"Test set size: {len(test_full)} rows across {test_full['EngineID'].nunique()} engines\n")

for label, before_col, after_col, true_col in [
    ('Compressor', 'est_eta_c', 'est_eta_c_corrected', 'CompressorHealth'),
    ('Turbine', 'est_eta_t', 'est_eta_t_corrected', 'TurbineHealth'),
]:
    mae_before = mean_absolute_error(test_full[true_col], test_full[before_col])
    mae_after = mean_absolute_error(test_full[true_col], test_full[after_col])
    baseline_mae = mean_absolute_error(test_full[true_col], [test_full[true_col].mean()] * len(test_full))
    print(f"{label}: UKF-only MAE={mae_before:.4f}, Corrected MAE={mae_after:.4f}, Naive-baseline MAE={baseline_mae:.4f}")

mae_comb = mean_absolute_error(test_full['CombustorHealth'], test_full['est_eta_comb'])
baseline_comb = mean_absolute_error(test_full['CombustorHealth'], [test_full['CombustorHealth'].mean()] * len(test_full))
print(f"Combustor: UKF-only MAE={mae_comb:.4f}, Naive-baseline MAE={baseline_comb:.4f}")

test_full.to_csv("test_set_validation_results.csv", index=False)
print("\nSaved test_set_validation_results.csv")
