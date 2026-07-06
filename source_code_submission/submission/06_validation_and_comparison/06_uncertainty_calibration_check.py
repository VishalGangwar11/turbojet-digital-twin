"""
Day 6 - Uncertainty calibration check. A well-calibrated UKF should have
the TRUE health value fall inside its +-1 sigma band roughly 68% of the
time, and inside +-2 sigma roughly 95% of the time. If the actual coverage
is far off from these numbers, the reported confidence is misleading -
either overconfident (too narrow) or underconfident (too wide).
"""

import pandas as pd
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

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
    estimates, variances = [], []
    for _, row in engine_df.iterrows():
        z = np.array([row['P2_Pa'], row['T2_K'], row['P3_Pa'], row['T3_K'], row['P4_Pa'], row['T4_K']])
        ukf.predict()
        ukf.update(z, Pamb=row['Pamb_Pa'], Tamb=row['Tamb_K'], RPM=row['RPM_rev_min'],
                   FuelFlow=row['FuelFlow_kg_s'], Mach=row['Mach'], Altitude=row['Altitude_m'])
        estimates.append(ukf.x.copy())
        variances.append(np.diag(ukf.P).copy())
    return np.array(estimates), np.array(variances)


all_true = {'c': [], 'comb': [], 't': []}
all_est = {'c': [], 'comb': [], 't': []}
all_std = {'c': [], 'comb': [], 't': []}

for eng_id in sorted(df['EngineID'].unique()):
    eng = df[df['EngineID'] == eng_id].sort_values('Cycle').reset_index(drop=True)
    estimates, variances = run_ukf_for_engine(eng)
    all_true['c'].extend(eng['CompressorHealth']); all_est['c'].extend(estimates[:, 0]); all_std['c'].extend(np.sqrt(variances[:, 0]))
    all_true['comb'].extend(eng['CombustorHealth']); all_est['comb'].extend(estimates[:, 1]); all_std['comb'].extend(np.sqrt(variances[:, 1]))
    all_true['t'].extend(eng['TurbineHealth']); all_est['t'].extend(estimates[:, 2]); all_std['t'].extend(np.sqrt(variances[:, 2]))

print("=== UNCERTAINTY CALIBRATION CHECK ===")
print("A well-calibrated filter should have ~68% of true values inside +-1 sigma,")
print("and ~95% inside +-2 sigma. Far off from these = miscalibrated confidence.\n")

for label, key in [('Compressor', 'c'), ('Combustor', 'comb'), ('Turbine', 't')]:
    true_vals = np.array(all_true[key])
    est_vals = np.array(all_est[key])
    std_vals = np.array(all_std[key])

    within_1sigma = np.abs(true_vals - est_vals) <= std_vals
    within_2sigma = np.abs(true_vals - est_vals) <= 2 * std_vals

    coverage_1sigma = within_1sigma.mean() * 100
    coverage_2sigma = within_2sigma.mean() * 100

    print(f"{label}:")
    print(f"  Actual coverage within +-1 sigma: {coverage_1sigma:.1f}% (target: ~68%)")
    print(f"  Actual coverage within +-2 sigma: {coverage_2sigma:.1f}% (target: ~95%)")
    if coverage_1sigma < 50:
        print(f"  --> OVERCONFIDENT: reported uncertainty is too narrow, true values fall outside band too often")
    elif coverage_1sigma > 85:
        print(f"  --> UNDERCONFIDENT: reported uncertainty is too wide, band is overly conservative")
    else:
        print(f"  --> Reasonably calibrated")
    print()
