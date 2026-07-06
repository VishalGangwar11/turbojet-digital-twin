"""
Day 6 (continued) - Extrapolation test: train on EARLY cycles only (healthy
engine, narrow health range), test on LATE cycles (degraded, health range
never seen in training). This is where physics-grounded models classically
outperform black-box ML, since the physics equations don't need to have
"seen" low-health examples to still compute correctly - they just plug in
the (extrapolated) efficiency value. A black-box model has no such
guarantee outside its training distribution.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from sklearn.metrics import mean_absolute_error

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


def run_ukf_for_engine(engine_df):
    points = MerweScaledSigmaPoints(n=3, alpha=0.5, beta=2.0, kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=3, dim_z=6, dt=1.0, fx=fx, hx=hx, points=points)
    ukf.x = np.array([0.99, 0.999, 0.995])
    ukf.P *= 0.005
    ukf.Q = np.diag([0.0001, 0.001, 0.0005])
    ukf.R = np.diag([857714883.88, 378.63, 790911236.82, 4529.15, 1681026493.53, 9676.53])
    estimates = []
    for _, row in engine_df.iterrows():
        z = np.array([row['P2_Pa'], row['T2_K'], row['P3_Pa'], row['T3_K'], row['P4_Pa'], row['T4_K']])
        ukf.predict()
        ukf.update(z, Pamb=row['Pamb_Pa'], Tamb=row['Tamb_K'], RPM=row['RPM_rev_min'],
                   FuelFlow=row['FuelFlow_kg_s'], Mach=row['Mach'], Altitude=row['Altitude_m'])
        estimates.append(ukf.x.copy())
    return np.array(estimates)


ukf_estimates_all = {}
for eng_id in sorted(df['EngineID'].unique()):
    eng = df[df['EngineID'] == eng_id].sort_values('Cycle').reset_index(drop=True)
    est = run_ukf_for_engine(eng)
    ukf_estimates_all[eng_id] = est

df['est_eta_t'] = np.nan
for eng_id, est in ukf_estimates_all.items():
    df.loc[df['EngineID'] == eng_id, 'est_eta_t'] = est[:, 2]

raw_features = ['P2_Pa', 'T2_K', 'P3_Pa', 'T3_K', 'P4_Pa', 'T4_K',
                 'RPM_rev_min', 'FuelFlow_kg_s', 'Pamb_Pa', 'Tamb_K',
                 'Mach', 'Altitude_m', 'Cycle']
hybrid_features = ['RPM_rev_min', 'Mach', 'Altitude_m', 'Pamb_Pa', 'Tamb_K',
                    'FuelFlow_kg_s', 'Cycle', 'est_eta_t']

# ---- Extrapolation split: train on cycles 1-15, test on cycles 16-30 ----
CYCLE_SPLIT = 15
train_mask = df['Cycle'] <= CYCLE_SPLIT
test_mask = df['Cycle'] > CYCLE_SPLIT

print(f"=== EXTRAPOLATION TEST: train on cycles <= {CYCLE_SPLIT}, test on cycles > {CYCLE_SPLIT} ===")
print(f"Train rows: {train_mask.sum()}, Test rows: {test_mask.sum()}")
print(f"Train health range: {df.loc[train_mask,'TurbineHealth'].min():.3f} to {df.loc[train_mask,'TurbineHealth'].max():.3f}")
print(f"Test health range:  {df.loc[test_mask,'TurbineHealth'].min():.3f} to {df.loc[test_mask,'TurbineHealth'].max():.3f}")
print("(Test set health values are LOWER than anything seen in training - genuine extrapolation)\n")

# Black-box
bb_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
bb_model.fit(df.loc[train_mask, raw_features], df.loc[train_mask, 'TurbineHealth'])
bb_pred = bb_model.predict(df.loc[test_mask, raw_features])
bb_mae = mean_absolute_error(df.loc[test_mask, 'TurbineHealth'], bb_pred)

# Hybrid
df['residual_t'] = df['TurbineHealth'] - df['est_eta_t']
hy_model = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42)
hy_model.fit(df.loc[train_mask, hybrid_features], df.loc[train_mask, 'residual_t'])
hy_residual_pred = hy_model.predict(df.loc[test_mask, hybrid_features])
hy_pred = df.loc[test_mask, 'est_eta_t'] + hy_residual_pred
hy_mae = mean_absolute_error(df.loc[test_mask, 'TurbineHealth'], hy_pred)

# Physics/UKF alone, no correction at all - the "pure physics" floor
phys_mae = mean_absolute_error(df.loc[test_mask, 'TurbineHealth'], df.loc[test_mask, 'est_eta_t'])

print(f"Black-box ML MAE (extrapolating):        {bb_mae:.4f}")
print(f"Hybrid (Physics+Residual) MAE (extrap.):  {hy_mae:.4f}")
print(f"Pure Physics/UKF alone MAE (extrap.):      {phys_mae:.4f}")
print("\n(If black-box MAE is notably worse here than in the earlier in-distribution")
print(" test, that demonstrates the classic extrapolation weakness of black-box ML)")
