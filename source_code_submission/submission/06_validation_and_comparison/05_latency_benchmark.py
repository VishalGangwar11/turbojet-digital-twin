"""
Day 6 - Latency/compute benchmark. Measures per-cycle inference time for:
1. Pure physics/UKF prediction step
2. Residual correction (GBM) prediction step
3. Full hybrid pipeline (UKF + residual)
4. Black-box GBM model (for comparison)
"""

import pandas as pd
import numpy as np
import time
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from sklearn.ensemble import GradientBoostingRegressor

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


# ---- Setup: build a fitted GBM model to benchmark prediction speed ----
raw_features = ['P2_Pa', 'T2_K', 'P3_Pa', 'T3_K', 'P4_Pa', 'T4_K',
                 'RPM_rev_min', 'FuelFlow_kg_s', 'Pamb_Pa', 'Tamb_K',
                 'Mach', 'Altitude_m', 'Cycle']
bb_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
bb_model.fit(df[raw_features], df['TurbineHealth'])

hybrid_features = ['RPM_rev_min', 'Mach', 'Altitude_m', 'Pamb_Pa', 'Tamb_K', 'FuelFlow_kg_s', 'Cycle', 'est_eta_t_dummy']
df['est_eta_t_dummy'] = 0.9
gbm_residual = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42)
gbm_residual.fit(df[hybrid_features], df['TurbineHealth'] - df['est_eta_t_dummy'])

row = df.iloc[0]
N_TRIALS = 500

# ---- Benchmark 1: UKF single predict+update step ----
points = MerweScaledSigmaPoints(n=3, alpha=0.5, beta=2.0, kappa=0)
ukf = UnscentedKalmanFilter(dim_x=3, dim_z=6, dt=1.0, fx=fx, hx=hx, points=points)
ukf.x = np.array([0.99, 0.999, 0.995])
ukf.P *= 0.005
ukf.Q = np.diag([0.0001, 0.001, 0.0005])
ukf.R = np.diag([857714883.88, 378.63, 790911236.82, 4529.15, 1681026493.53, 9676.53])

z = np.array([row['P2_Pa'], row['T2_K'], row['P3_Pa'], row['T3_K'], row['P4_Pa'], row['T4_K']])

start = time.perf_counter()
for _ in range(N_TRIALS):
    ukf.predict()
    ukf.update(z, Pamb=row['Pamb_Pa'], Tamb=row['Tamb_K'], RPM=row['RPM_rev_min'],
               FuelFlow=row['FuelFlow_kg_s'], Mach=row['Mach'], Altitude=row['Altitude_m'])
ukf_time = (time.perf_counter() - start) / N_TRIALS * 1000  # ms

# ---- Benchmark 2: GBM residual correction single prediction ----
X_single = df[hybrid_features].iloc[[0]]
start = time.perf_counter()
for _ in range(N_TRIALS):
    gbm_residual.predict(X_single)
gbm_residual_time = (time.perf_counter() - start) / N_TRIALS * 1000  # ms

# ---- Benchmark 3: Black-box GBM single prediction ----
X_single_bb = df[raw_features].iloc[[0]]
start = time.perf_counter()
for _ in range(N_TRIALS):
    bb_model.predict(X_single_bb)
bb_time = (time.perf_counter() - start) / N_TRIALS * 1000  # ms

print("=== INFERENCE LATENCY BENCHMARK (per cycle, single prediction) ===")
print(f"(Averaged over {N_TRIALS} trials, on this machine's CPU)\n")
print(f"UKF predict+update step:          {ukf_time:.4f} ms")
print(f"GBM residual correction predict:  {gbm_residual_time:.4f} ms")
print(f"Full hybrid pipeline (UKF+GBM):    {ukf_time + gbm_residual_time:.4f} ms")
print(f"Black-box GBM alone:               {bb_time:.4f} ms")

print(f"\n=== Practical interpretation ===")
print(f"Hybrid pipeline: ~{1000/(ukf_time+gbm_residual_time):.0f} predictions/second on a single CPU core")
print(f"This is well within real-time requirements for cycle-by-cycle engine monitoring")
print(f"(typical engine monitoring cycle rates are on the order of 1 Hz or slower)")
