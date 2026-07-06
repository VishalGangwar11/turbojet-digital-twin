"""
Day 3 (diagnostic) - Feed TRUE health values into hx() and compare predicted
sensors to actual sensors. This isolates whether the forward model itself
is broken, vs. the filter dynamics being the problem.
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


def hx(state, Pamb, Tamb, RPM, FuelFlow, Mach=0.4, Altitude=6000):
    eta_c, eta_comb, eta_t = state

    PR_c_assumed = (0.0001286487 * RPM + 2.624160 * Mach
                    + 0.0000033082 * Altitude - 3.931887)
    PR_c_assumed = max(PR_c_assumed, 1.05)

    air_flow = (Pamb / (R_AIR * Tamb)) * RPM * 1e-4

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

    return np.array([P2, T2, P3, T3, P4, T4]), air_flow


eng1 = df[df['EngineID'] == 1].sort_values('Cycle').reset_index(drop=True)

print("=== Feeding TRUE health into hx(), comparing to ACTUAL sensors ===\n")
for _, row in eng1.head(10).iterrows():
    state = [row['CompressorHealth'], row['CombustorHealth'], row['TurbineHealth']]
    pred, air_flow = hx(state, row['Pamb_Pa'], row['Tamb_K'], row['RPM_rev_min'],
                        row['FuelFlow_kg_s'], row['Mach'], row['Altitude_m'])
    actual = np.array([row['P2_Pa'], row['T2_K'], row['P3_Pa'], row['T3_K'], row['P4_Pa'], row['T4_K']])

    print(f"Cycle {int(row['Cycle'])}: air_flow_estimate={air_flow:.4f} kg/s, FuelFlow={row['FuelFlow_kg_s']:.4f} kg/s")
    print(f"  Predicted: P2={pred[0]:.0f} T2={pred[1]:.1f} P3={pred[2]:.0f} T3={pred[3]:.1f} P4={pred[4]:.0f} T4={pred[5]:.1f}")
    print(f"  Actual:    P2={actual[0]:.0f} T2={actual[1]:.1f} P3={actual[2]:.0f} T3={actual[3]:.1f} P4={actual[4]:.0f} T4={actual[5]:.1f}")
    print(f"  %% error:   P2={100*(pred[0]-actual[0])/actual[0]:.1f}% T2={100*(pred[1]-actual[1])/actual[1]:.1f}% T3={100*(pred[3]-actual[3])/actual[3]:.1f}% T4={100*(pred[5]-actual[5])/actual[5]:.1f}%")
    print()
