"""
Day 3 (final tune) - UKF with R matrix set from real measured residual
variances instead of guesses.
"""

import pandas as pd
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from sklearn.metrics import r2_score

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

    PR_c_assumed = (0.0001286487 * RPM + 2.624160 * Mach
                    + 0.0000033082 * Altitude - 3.931887)
    PR_c_assumed = max(PR_c_assumed, 1.05)

    rho_amb = Pamb / (R_AIR * Tamb)
    air_flow = 0.0006162388 * RPM - 0.424435 * rho_amb - 8.280180
    air_flow = max(air_flow, 1.0)

    P2 = Pamb * PR_c_assumed
    T2_ideal = Tamb * (PR_c_assumed) ** ((GAMMA_AIR - 1) / GAMMA_AIR)
    T2 = Tamb + (T2_ideal - Tamb) / max(eta_c, 1e-3)

    PRESSURE_LOSS = 0.96
    P3 = P2 * PRESSURE_LOSS
    total_flow = air_flow + FuelFlow
    T3 = (air_flow * CP_AIR * T2 + eta_comb * FuelFlow * LHV_FUEL) / (total_flow * CP_GAS + 1e-9)

    PR_t_assumed = -0.0000118963 * RPM + 1.334673
    PR_t_assumed = min(max(PR_t_assumed, 0.1), 0.99)

    P4 = P3 * PR_t_assumed
    T4_ideal = T3 * (PR_t_assumed) ** ((GAMMA_GAS - 1) / GAMMA_GAS)
    T4 = T3 - eta_t * (T3 - T4_ideal)

    return np.array([P2, T2, P3, T3, P4, T4])


def run_ukf_for_engine(engine_df, Q_scale=0.0005):
    dim_x = 3
    dim_z = 6

    points = MerweScaledSigmaPoints(n=dim_x, alpha=0.5, beta=2.0, kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=1.0, fx=fx, hx=hx, points=points)

    ukf.x = np.array([0.9, 0.98, 0.9])
    ukf.P *= 0.01
    ukf.Q = np.eye(dim_x) * Q_scale
    # Calibrated from actual residual variances (Day 3 tuning step)
    ukf.R = np.diag([857714883.88, 378.63, 790911236.82, 4529.15, 1681026493.53, 9676.53])

    estimates = []
    for _, row in engine_df.iterrows():
        z = np.array([row['P2_Pa'], row['T2_K'], row['P3_Pa'], row['T3_K'], row['P4_Pa'], row['T4_K']])

        ukf.predict()
        ukf.update(z, Pamb=row['Pamb_Pa'], Tamb=row['Tamb_K'],
                   RPM=row['RPM_rev_min'], FuelFlow=row['FuelFlow_kg_s'],
                   Mach=row['Mach'], Altitude=row['Altitude_m'])

        estimates.append(ukf.x.copy())

    return np.array(estimates)


if __name__ == "__main__":
    eng1 = df[df['EngineID'] == 1].sort_values('Cycle').reset_index(drop=True)

    # Try a few Q scales to see which tracks the true trend best
    for Q_scale in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
        estimates = run_ukf_for_engine(eng1, Q_scale=Q_scale)
        r2_c = r2_score(eng1['CompressorHealth'], estimates[:, 0])
        r2_comb = r2_score(eng1['CombustorHealth'], estimates[:, 1])
        r2_t = r2_score(eng1['TurbineHealth'], estimates[:, 2])
        print(f"Q_scale={Q_scale}: R2_compressor={r2_c:.4f}, R2_combustor={r2_comb:.4f}, R2_turbine={r2_t:.4f}")

    print("\n=== Detailed output for best-looking Q_scale (edit below to match) ===")
    best_estimates = run_ukf_for_engine(eng1, Q_scale=0.001)
    eng1['est_eta_c'] = best_estimates[:, 0]
    eng1['est_eta_comb'] = best_estimates[:, 1]
    eng1['est_eta_t'] = best_estimates[:, 2]
    print(eng1[['Cycle', 'est_eta_c', 'CompressorHealth',
                'est_eta_comb', 'CombustorHealth',
                'est_eta_t', 'TurbineHealth']].to_string())
