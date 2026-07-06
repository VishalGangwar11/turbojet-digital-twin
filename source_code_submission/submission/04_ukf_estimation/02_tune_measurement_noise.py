"""
Day 3 (tune) - Compute the ACTUAL residual variance of hx() predictions
(using true health as input) vs real sensors, across ALL engines/cycles.
This gives us a properly calibrated R matrix instead of guessed values.
"""

import pandas as pd
import numpy as np

GAMMA_AIR = 1.4
GAMMA_GAS = 1.33
CP_AIR = 1005.0
CP_GAS = 1148.0
LHV_FUEL = 43e6
R_AIR = 287.0

train = pd.read_csv("train.csv")
gt = pd.read_csv("ground_truth.csv")
df = train.merge(gt, on=['EngineID', 'Cycle']).sort_values(['EngineID', 'Cycle']).reset_index(drop=True)


def hx(state, Pamb, Tamb, RPM, FuelFlow, Mach, Altitude):
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


residuals = []
for _, row in df.iterrows():
    state = [row['CompressorHealth'], row['CombustorHealth'], row['TurbineHealth']]
    pred = hx(state, row['Pamb_Pa'], row['Tamb_K'], row['RPM_rev_min'],
              row['FuelFlow_kg_s'], row['Mach'], row['Altitude_m'])
    actual = np.array([row['P2_Pa'], row['T2_K'], row['P3_Pa'], row['T3_K'], row['P4_Pa'], row['T4_K']])
    residuals.append(pred - actual)

residuals = np.array(residuals)
labels = ['P2', 'T2', 'P3', 'T3', 'P4', 'T4']

print("=== Residual std per sensor (use these^2 as R diagonal) ===")
for i, label in enumerate(labels):
    print(f"{label}: mean={residuals[:,i].mean():.2f}, std={residuals[:,i].std():.2f}")

print("\n=== Suggested R matrix (variance = std^2) ===")
variances = residuals.std(axis=0) ** 2
print(list(variances))
