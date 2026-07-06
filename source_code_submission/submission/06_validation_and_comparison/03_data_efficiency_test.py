"""
Day 6 (continued) - Data efficiency test: train both the black-box model
and the hybrid residual-correction model on progressively FEWER engines,
see which degrades faster. Physics-informed models should need less data
since they're only learning the residual, not the whole physics from scratch.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from sklearn.metrics import mean_absolute_error
import random

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


# Precompute UKF estimates for ALL engines once (physics part doesn't need retraining -
# it's calibrated from equations, not data-hungry ML)
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

engines = sorted(df['EngineID'].unique())
test_engines = engines[-2:]  # fix 2 engines as the constant test set
pool_engines = engines[:-2]  # 8 engines to draw increasing training subsets from

print("=== DATA EFFICIENCY TEST: Turbine Health ===")
print(f"Fixed test engines: {test_engines}\n")
print(f"{'#Training Engines':<20}{'Black-box ML MAE':<20}{'Hybrid (Physics+Residual) MAE':<30}")

random.seed(42)
for n_train in [1, 2, 3, 4, 6, 8]:
    train_engines = pool_engines[:n_train]  # deterministic subset for reproducibility

    train_mask = df['EngineID'].isin(train_engines)
    test_mask = df['EngineID'].isin(test_engines)

    # Black-box: raw sensors -> TurbineHealth directly
    bb_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    bb_model.fit(df.loc[train_mask, raw_features], df.loc[train_mask, 'TurbineHealth'])
    bb_pred = bb_model.predict(df.loc[test_mask, raw_features])
    bb_mae = mean_absolute_error(df.loc[test_mask, 'TurbineHealth'], bb_pred)

    # Hybrid: physics/UKF already gives est_eta_t (no retraining needed - it's equations,
    # not learned), only the residual correction layer needs training data
    df['residual_t'] = df['TurbineHealth'] - df['est_eta_t']
    hy_model = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42)
    hy_model.fit(df.loc[train_mask, hybrid_features], df.loc[train_mask, 'residual_t'])
    hy_residual_pred = hy_model.predict(df.loc[test_mask, hybrid_features])
    hy_pred = df.loc[test_mask, 'est_eta_t'] + hy_residual_pred
    hy_mae = mean_absolute_error(df.loc[test_mask, 'TurbineHealth'], hy_pred)

    print(f"{n_train:<20}{bb_mae:<20.4f}{hy_mae:<30.4f}")

print("\n(If hybrid MAE stays lower/flatter as training engines shrink, that's your")
print(" data-efficiency evidence - physics-grounding needs less data to generalize)")
