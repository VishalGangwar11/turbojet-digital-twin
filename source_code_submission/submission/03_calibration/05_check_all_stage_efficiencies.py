"""
Day 2 - Invert combustor and turbine equations, check against ground truth.
Builds on the compressor check from Day 1.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

train = pd.read_csv("train.csv")
gt = pd.read_csv("ground_truth.csv")
df = train.merge(gt, on=['EngineID', 'Cycle']).sort_values(['EngineID', 'Cycle'])

GAMMA_AIR = 1.4
GAMMA_GAS = 1.33
CP_AIR = 1005.0
CP_GAS = 1148.0
LHV_FUEL = 43e6  # J/kg

# ---- Compressor (from Day 1, kept for completeness) ----
df['PR_c'] = df['P2_Pa'] / df['Pamb_Pa']
df['T2_ideal'] = df['Tamb_K'] * (df['PR_c']) ** ((GAMMA_AIR - 1) / GAMMA_AIR)
df['implied_eta_c'] = (df['T2_ideal'] - df['Tamb_K']) / (df['T2_K'] - df['Tamb_K'])

# ---- Combustor ----
# Energy balance: (air_flow + fuel_flow)*cp_gas*T3 = air_flow*cp_air*T2 + eta_comb*fuel_flow*LHV
# We don't have air_flow directly - estimate it via RPM as a proxy scaled by a rough constant.
# For now, back out an "implied_eta_comb" assuming air_flow >> fuel_flow (air_flow ~ RPM-based proxy)
# implied_eta_comb = (air_flow*cp_gas*T3 - air_flow*cp_air*T2) / (fuel_flow*LHV)   [approx, ignoring fuel mass addition to total flow]

# Since we don't know true air_flow scale, use a normalized proxy: assume air_flow proportional to RPM
# with an unknown constant k. We'll fold k into the calibration regression instead of guessing it.
df['temp_rise_energy'] = CP_GAS * df['T3_K'] - CP_AIR * df['T2_K']  # per unit air_flow
df['fuel_energy_avail'] = df['FuelFlow_kg_s'] * LHV_FUEL / df['RPM_rev_min']  # proxy per unit "RPM-flow"
df['implied_eta_comb_raw'] = df['temp_rise_energy'] / (df['fuel_energy_avail'] + 1e-9)

# ---- Turbine ----
# T4_ideal = T3 * (P4/P3)^((gamma-1)/gamma)   [note: P4/P3 < 1, expansion]
# eta_t = (T3 - T4_actual) / (T3 - T4_ideal)
df['PR_t'] = df['P4_Pa'] / df['P3_Pa']
df['T4_ideal'] = df['T3_K'] * (df['PR_t']) ** ((GAMMA_GAS - 1) / GAMMA_GAS)
df['implied_eta_t'] = (df['T3_K'] - df['T4_K']) / (df['T3_K'] - df['T4_ideal'] + 1e-9)

# ---- Clean obviously invalid values before regression ----
def clean(col, lo=-2, hi=3):
    return df[(df[col] > lo) & (df[col] < hi)]

print("=== COMBUSTOR ===")
df_comb = clean('implied_eta_comb_raw')
X = df_comb[['implied_eta_comb_raw']].values
y = df_comb['CombustorHealth'].values
reg_comb = LinearRegression().fit(X, y)
print(f"Calibration: CombustorHealth = {reg_comb.coef_[0]:.6f} * implied_eta_comb_raw + {reg_comb.intercept_:.4f}")
print(f"R^2 score: {reg_comb.score(X, y):.4f}")
resid = y - reg_comb.predict(X)
print(f"Residual std: {resid.std():.4f}")

print("\n=== TURBINE ===")
df_turb = clean('implied_eta_t')
X = df_turb[['implied_eta_t']].values
y = df_turb['TurbineHealth'].values
reg_turb = LinearRegression().fit(X, y)
print(f"Calibration: TurbineHealth = {reg_turb.coef_[0]:.4f} * implied_eta_t + {reg_turb.intercept_:.4f}")
print(f"R^2 score: {reg_turb.score(X, y):.4f}")
resid = y - reg_turb.predict(X)
print(f"Residual std: {resid.std():.4f}")

print("\n=== Sanity check: raw implied value ranges (before cleaning) ===")
print(df[['implied_eta_c', 'implied_eta_comb_raw', 'implied_eta_t']].describe())

# ---- Also try: does combining ALL implied efficiencies (multi-sensor) predict OverallHealth better? ----
print("\n=== Multi-feature regression: all 3 implied efficiencies -> OverallHealth ===")
df_all = df.copy()
for col in ['implied_eta_c', 'implied_eta_comb_raw', 'implied_eta_t']:
    df_all = df_all[(df_all[col] > -2) & (df_all[col] < 3)]

X_multi = df_all[['implied_eta_c', 'implied_eta_comb_raw', 'implied_eta_t']].values
y_multi = df_all['OverallHealth'].values
reg_multi = LinearRegression().fit(X_multi, y_multi)
print(f"R^2 score (multi-sensor, single-cycle): {reg_multi.score(X_multi, y_multi):.4f}")
print("(Compare this to R^2=0.35 from compressor-only in Day 1 - this is your 'fusion helps' evidence)")
