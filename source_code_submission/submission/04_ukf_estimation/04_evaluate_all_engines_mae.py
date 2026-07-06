"""
Day 3 (proper evaluation) - MAE across ALL 10 engines, not just R^2 on Engine 1.
"""

import pandas as pd
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from sklearn.metrics import mean_absolute_error, r2_score

GAMMA_AIR = 1.4
GAMMA_GAS = 1.33
CP_AIR = 1005.0
CP_GAS = 1148.0
LHV_FUEL = 43e6
R_AIR = 287.0

train = pd.read_csv("train.csv")
gt = pd.read_csv("ground_truth.csv")
df = train.merge(gt, on=['EngineID', 'Cycle']).sort_values(['EngineID', 'Cycle']).reset_index(drop=True)


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


def run_ukf_for_engine(engine_df, Q_scale=0.001):
    points = MerweScaledSigmaPoints(n=3, alpha=0.5, beta=2.0, kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=3, dim_z=6, dt=1.0, fx=fx, hx=hx, points=points)
    ukf.x = np.array([0.9, 0.98, 0.9])
    ukf.P *= 0.01
    ukf.Q = np.eye(3) * Q_scale
    ukf.R = np.diag([857714883.88, 378.63, 790911236.82, 4529.15, 1681026493.53, 9676.53])

    estimates = []
    for _, row in engine_df.iterrows():
        z = np.array([row['P2_Pa'], row['T2_K'], row['P3_Pa'], row['T3_K'], row['P4_Pa'], row['T4_K']])
        ukf.predict()
        ukf.update(z, Pamb=row['Pamb_Pa'], Tamb=row['Tamb_K'], RPM=row['RPM_rev_min'],
                   FuelFlow=row['FuelFlow_kg_s'], Mach=row['Mach'], Altitude=row['Altitude_m'])
        estimates.append(ukf.x.copy())
    return np.array(estimates)


all_true_c, all_true_comb, all_true_t = [], [], []
all_est_c, all_est_comb, all_est_t = [], [], []

for eng_id in sorted(df['EngineID'].unique()):
    eng = df[df['EngineID'] == eng_id].sort_values('Cycle').reset_index(drop=True)
    estimates = run_ukf_for_engine(eng, Q_scale=0.001)
    all_true_c.extend(eng['CompressorHealth'])
    all_true_comb.extend(eng['CombustorHealth'])
    all_true_t.extend(eng['TurbineHealth'])
    all_est_c.extend(estimates[:, 0])
    all_est_comb.extend(estimates[:, 1])
    all_est_t.extend(estimates[:, 2])

print("=== MAE across ALL 10 engines ===")
print(f"Compressor MAE: {mean_absolute_error(all_true_c, all_est_c):.4f}")
print(f"Combustor MAE:  {mean_absolute_error(all_true_comb, all_est_comb):.4f}")
print(f"Turbine MAE:    {mean_absolute_error(all_true_t, all_est_t):.4f}")

print("\n=== R^2 across ALL 10 engines (for reference) ===")
print(f"Compressor R^2: {r2_score(all_true_c, all_est_c):.4f}")
print(f"Combustor R^2:  {r2_score(all_true_comb, all_est_comb):.4f}")
print(f"Turbine R^2:    {r2_score(all_true_t, all_est_t):.4f}")

print("\n=== Baseline comparison: MAE if we just always predicted the mean ===")
print(f"Compressor baseline MAE: {mean_absolute_error(all_true_c, [np.mean(all_true_c)]*len(all_true_c)):.4f}")
print(f"Combustor baseline MAE:  {mean_absolute_error(all_true_comb, [np.mean(all_true_comb)]*len(all_true_comb)):.4f}")
print(f"Turbine baseline MAE:    {mean_absolute_error(all_true_t, [np.mean(all_true_t)]*len(all_true_t)):.4f}")
