"""
Day 5 (final) - Apply residual corrections (trained on full data) to produce
the final corrected results CSV for the dashboard.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from scipy.spatial.distance import mahalanobis

GAMMA_AIR = 1.4
GAMMA_GAS = 1.33
CP_AIR = 1005.0
CP_GAS = 1148.0
LHV_FUEL = 43e6
R_AIR = 287.0

ukf_results = pd.read_csv("ukf_results_full.csv")
train = pd.read_csv("train.csv")
gt = pd.read_csv("ground_truth.csv")

df = ukf_results.merge(train, on=['EngineID', 'Cycle']).merge(gt, on=['EngineID', 'Cycle'])

feature_cols = ['RPM_rev_min', 'Mach', 'Altitude_m', 'Pamb_Pa', 'Tamb_K', 'FuelFlow_kg_s', 'Cycle']

# ---- Train FINAL production correction models on ALL data ----
df['residual_c'] = df['CompressorHealth'] - df['est_eta_c']
df['residual_t'] = df['TurbineHealth'] - df['est_eta_t']

model_c = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42)
model_c.fit(df[feature_cols + ['est_eta_c']], df['residual_c'])

model_t = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42)
model_t.fit(df[feature_cols + ['est_eta_t']], df['residual_t'])

df['est_eta_c_corrected'] = df['est_eta_c'] + model_c.predict(df[feature_cols + ['est_eta_c']])
df['est_eta_t_corrected'] = df['est_eta_t'] + model_t.predict(df[feature_cols + ['est_eta_t']])
df['est_eta_comb_corrected'] = df['est_eta_comb']  # combustor already strong, no correction needed


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
    return P2, T2, P3, T3, P4, T4


def estimate_thrust_tsfc(eta_c, eta_comb, eta_t, Pamb, Tamb, RPM, FuelFlow, Mach, Altitude):
    P2, T2, P3, T3, P4, T4 = hx([eta_c, eta_comb, eta_t], Pamb, Tamb, RPM, FuelFlow, Mach, Altitude)
    rho_amb = Pamb / (R_AIR * Tamb)
    air_flow = max(0.0006162388 * RPM - 0.424435 * rho_amb - 8.280180, 1.0)
    total_flow = air_flow + FuelFlow
    V_exit = np.sqrt(max(2 * CP_GAS * (T4 - Tamb), 0))
    thrust = total_flow * V_exit
    tsfc = (FuelFlow * 1000) / max(thrust, 1e-6)
    return thrust, tsfc


thrust_preds, tsfc_preds = [], []
for _, row in df.iterrows():
    t, s = estimate_thrust_tsfc(row['est_eta_c_corrected'], row['est_eta_comb_corrected'],
                                  row['est_eta_t_corrected'], row['Pamb_Pa'], row['Tamb_K'],
                                  row['RPM_rev_min'], row['FuelFlow_kg_s'], row['Mach'], row['Altitude_m'])
    thrust_preds.append(t)
    tsfc_preds.append(s)
df['pred_Thrust_corrected'] = thrust_preds
df['pred_TSFC_corrected'] = tsfc_preds

# ---- Recompute unified health index using corrected estimates ----
healthy_baseline = df[df['Cycle'] == 1][['est_eta_c_corrected', 'est_eta_comb_corrected', 'est_eta_t_corrected']].values
mean_baseline = healthy_baseline.mean(axis=0)
cov_baseline = np.cov(healthy_baseline.T) + np.eye(3) * 1e-6
inv_cov = np.linalg.inv(cov_baseline)

def health_index(row):
    v = np.array([row['est_eta_c_corrected'], row['est_eta_comb_corrected'], row['est_eta_t_corrected']])
    return mahalanobis(v, mean_baseline, inv_cov)

df['health_index_raw_corrected'] = df.apply(health_index, axis=1)
max_d = df['health_index_raw_corrected'].max()
df['unified_health_index_corrected'] = 1 - (df['health_index_raw_corrected'] / max_d)

# ---- Save final CSV with both original and corrected columns for the dashboard ----
output_cols = ['EngineID', 'Cycle',
               'est_eta_c', 'est_eta_c_corrected', 'CompressorHealth',
               'est_eta_comb', 'est_eta_comb_corrected', 'CombustorHealth',
               'est_eta_t', 'est_eta_t_corrected', 'TurbineHealth',
               'var_eta_c', 'var_eta_comb', 'var_eta_t',
               'true_Thrust', 'pred_Thrust_corrected',
               'true_TSFC', 'pred_TSFC_corrected',
               'true_OverallHealth', 'unified_health_index_corrected']

df[output_cols].to_csv("ukf_results_corrected.csv", index=False)
print("Saved ukf_results_corrected.csv")
print(df[output_cols].head(10).to_string())
