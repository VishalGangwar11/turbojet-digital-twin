"""
Day 4 - Unified health index (Mahalanobis distance from healthy baseline)
and thrust/TSFC prediction from UKF-estimated efficiencies.
"""

import pandas as pd
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.spatial.distance import mahalanobis

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


def run_ukf_for_engine(engine_df, Q_diag=(0.0001, 0.001, 0.0005)):
    points = MerweScaledSigmaPoints(n=3, alpha=0.5, beta=2.0, kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=3, dim_z=6, dt=1.0, fx=fx, hx=hx, points=points)
    ukf.x = np.array([0.99, 0.999, 0.995])
    ukf.P *= 0.005
    ukf.Q = np.diag(Q_diag)
    ukf.R = np.diag([857714883.88, 378.63, 790911236.82, 4529.15, 1681026493.53, 9676.53])

    estimates, covariances = [], []
    for _, row in engine_df.iterrows():
        z = np.array([row['P2_Pa'], row['T2_K'], row['P3_Pa'], row['T3_K'], row['P4_Pa'], row['T4_K']])
        ukf.predict()
        ukf.update(z, Pamb=row['Pamb_Pa'], Tamb=row['Tamb_K'], RPM=row['RPM_rev_min'],
                   FuelFlow=row['FuelFlow_kg_s'], Mach=row['Mach'], Altitude=row['Altitude_m'])
        estimates.append(ukf.x.copy())
        covariances.append(np.diag(ukf.P).copy())
    return np.array(estimates), np.array(covariances)


def estimate_thrust_tsfc(eta_c, eta_comb, eta_t, Pamb, Tamb, RPM, FuelFlow, Mach, Altitude):
    """Rough thrust/TSFC estimate from UKF-estimated efficiencies + known flight condition."""
    P2, T2, P3, T3, P4, T4 = hx([eta_c, eta_comb, eta_t], Pamb, Tamb, RPM, FuelFlow, Mach, Altitude)
    rho_amb = Pamb / (R_AIR * Tamb)
    air_flow = max(0.0006162388 * RPM - 0.424435 * rho_amb - 8.280180, 1.0)
    total_flow = air_flow + FuelFlow

    # Simplified momentum thrust estimate (ignoring detailed nozzle geometry,
    # per problem statement note: detailed CFD/nozzle design not required)
    V_exit = np.sqrt(max(2 * CP_GAS * (T4 - Tamb), 0))
    thrust = total_flow * V_exit  # N, simplified momentum thrust
    tsfc = (FuelFlow * 1000) / max(thrust, 1e-6)  # g/(N.s) - matches TSFC_g_N_s units roughly
    return thrust, tsfc


# ---- Run UKF for all engines, collect estimates + thrust/TSFC predictions ----
all_rows = []
healthy_baseline = []  # collect cycle-1 (or early) estimates to define "healthy" reference

for eng_id in sorted(df['EngineID'].unique()):
    eng = df[df['EngineID'] == eng_id].sort_values('Cycle').reset_index(drop=True)
    estimates, covariances = run_ukf_for_engine(eng)

    for i, row in eng.iterrows():
        eta_c, eta_comb, eta_t = estimates[i]
        thrust_pred, tsfc_pred = estimate_thrust_tsfc(
            eta_c, eta_comb, eta_t, row['Pamb_Pa'], row['Tamb_K'],
            row['RPM_rev_min'], row['FuelFlow_kg_s'], row['Mach'], row['Altitude_m']
        )
        all_rows.append({
            'EngineID': eng_id, 'Cycle': row['Cycle'],
            'est_eta_c': eta_c, 'est_eta_comb': eta_comb, 'est_eta_t': eta_t,
            'var_eta_c': covariances[i][0], 'var_eta_comb': covariances[i][1], 'var_eta_t': covariances[i][2],
            'true_Thrust': row['Thrust_N'], 'pred_Thrust': thrust_pred,
            'true_TSFC': row['TSFC_g_N_s'], 'pred_TSFC': tsfc_pred,
            'true_OverallHealth': row['OverallHealth'],
        })
        if row['Cycle'] == 1:
            healthy_baseline.append([eta_c, eta_comb, eta_t])

results = pd.DataFrame(all_rows)

# ---- Unified health index via Mahalanobis distance from healthy baseline ----
baseline = np.array(healthy_baseline)
mean_baseline = baseline.mean(axis=0)
cov_baseline = np.cov(baseline.T) + np.eye(3) * 1e-6  # regularize to avoid singular matrix
inv_cov = np.linalg.inv(cov_baseline)

def health_index(row):
    v = np.array([row['est_eta_c'], row['est_eta_comb'], row['est_eta_t']])
    d = mahalanobis(v, mean_baseline, inv_cov)
    return d

results['health_index_raw'] = results.apply(health_index, axis=1)
# Normalize so healthy=1, degraded approaches 0 (invert and scale)
max_d = results['health_index_raw'].max()
results['unified_health_index'] = 1 - (results['health_index_raw'] / max_d)

print("=== Unified health index vs true OverallHealth (correlation check) ===")
print(f"Correlation: {results['unified_health_index'].corr(results['true_OverallHealth']):.4f}")

print("\n=== Thrust prediction accuracy ===")
print(f"MAE: {mean_absolute_error(results['true_Thrust'], results['pred_Thrust']):.1f} N")
print(f"R^2: {r2_score(results['true_Thrust'], results['pred_Thrust']):.4f}")
print(f"True thrust range: {results['true_Thrust'].min():.0f} to {results['true_Thrust'].max():.0f} N")

print("\n=== TSFC prediction accuracy ===")
print(f"MAE: {mean_absolute_error(results['true_TSFC'], results['pred_TSFC']):.5f}")
print(f"R^2: {r2_score(results['true_TSFC'], results['pred_TSFC']):.4f}")

results.to_csv("ukf_results_full.csv", index=False)
print("\nSaved full results to ukf_results_full.csv (use this for dashboard later)")
