"""
Day 3 (tune v2) - Per-state Q values + realistic initial state (~1.0, healthy engine).
"""

import pandas as pd
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from sklearn.metrics import mean_absolute_error, r2_score
import itertools

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


def run_ukf_for_engine(engine_df, Q_diag):
    points = MerweScaledSigmaPoints(n=3, alpha=0.5, beta=2.0, kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=3, dim_z=6, dt=1.0, fx=fx, hx=hx, points=points)
    # Realistic starting point - healthy engine, matches actual Cycle-1 values
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


def evaluate(Q_diag):
    all_true_c, all_true_comb, all_true_t = [], [], []
    all_est_c, all_est_comb, all_est_t = [], [], []
    for eng_id in sorted(df['EngineID'].unique()):
        eng = df[df['EngineID'] == eng_id].sort_values('Cycle').reset_index(drop=True)
        estimates = run_ukf_for_engine(eng, Q_diag)
        all_true_c.extend(eng['CompressorHealth']); all_est_c.extend(estimates[:, 0])
        all_true_comb.extend(eng['CombustorHealth']); all_est_comb.extend(estimates[:, 1])
        all_true_t.extend(eng['TurbineHealth']); all_est_t.extend(estimates[:, 2])
    mae_c = mean_absolute_error(all_true_c, all_est_c)
    mae_comb = mean_absolute_error(all_true_comb, all_est_comb)
    mae_t = mean_absolute_error(all_true_t, all_est_t)
    return mae_c, mae_comb, mae_t


print("Baselines: Compressor=0.0615, Combustor=0.0219, Turbine=0.0440\n")

# Grid search over per-state Q values
q_options = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]
best = None
for qc, qcomb, qt in itertools.product(q_options, [0.0001, 0.0005, 0.001], q_options):
    mae_c, mae_comb, mae_t = evaluate([qc, qcomb, qt])
    total = mae_c + mae_comb + mae_t
    if best is None or total < best[0]:
        best = (total, qc, qcomb, qt, mae_c, mae_comb, mae_t)

print(f"Best combo: Q=[{best[1]}, {best[2]}, {best[3]}]")
print(f"MAE: Compressor={best[4]:.4f}, Combustor={best[5]:.4f}, Turbine={best[6]:.4f}")
print("(Compare each to baseline above - lower than baseline = UKF is genuinely helping)")
