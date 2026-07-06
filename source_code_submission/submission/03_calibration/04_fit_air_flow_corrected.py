"""
Day 3 (fix #2) - Back out the TRUE air_flow from the combustor energy balance
using actual sensor data + CombustorHealth as eta_comb, then fit air_flow
against RPM/Pamb/Tamb to get a calibrated relationship (replacing the
arbitrary 1e-4 scaling constant that was way off).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

CP_AIR = 1005.0
CP_GAS = 1148.0
LHV_FUEL = 43e6

train = pd.read_csv("train.csv")
gt = pd.read_csv("ground_truth.csv")
df = train.merge(gt, on=['EngineID', 'Cycle']).sort_values(['EngineID', 'Cycle']).reset_index(drop=True)

# Invert: (air+fuel)*cp_gas*T3 = air*cp_air*T2 + eta_comb*fuel*LHV
# => air*(cp_gas*T3 - cp_air*T2) = eta_comb*fuel*LHV - fuel*cp_gas*T3
# => air = fuel*(eta_comb*LHV - cp_gas*T3) / (cp_gas*T3 - cp_air*T2)

eta_comb = df['CombustorHealth']  # use true health as a stand-in for eta_comb
fuel = df['FuelFlow_kg_s']
T2 = df['T2_K']
T3 = df['T3_K']

numerator = fuel * (eta_comb * LHV_FUEL - CP_GAS * T3)
denominator = (CP_GAS * T3 - CP_AIR * T2)
df['true_air_flow'] = numerator / denominator

print("=== Implied TRUE air_flow stats ===")
print(df['true_air_flow'].describe())

# Check air-fuel ratio - should be realistic (typically 30-80 for turbojets)
df['AFR'] = df['true_air_flow'] / df['FuelFlow_kg_s']
print("\n=== Air-Fuel Ratio (sanity check, expect roughly 10-100) ===")
print(df['AFR'].describe())

# Now fit true_air_flow against known inputs (RPM, Pamb, Tamb, Altitude, Mach)
mask = (df['true_air_flow'] > 0) & (df['true_air_flow'] < 200)  # remove any nonsense outliers
dsub = df[mask]

X = dsub[['RPM_rev_min', 'Pamb_Pa', 'Tamb_K', 'Altitude_m', 'Mach']].values
y = dsub['true_air_flow'].values
reg = LinearRegression().fit(X, y)
print(f"\nRows used: {len(dsub)} / {len(df)}")
print(f"R^2: {reg.score(X, y):.4f}")
print("Coefficients [RPM, Pamb, Tamb, Altitude, Mach]:")
print(reg.coef_)
print(f"Intercept: {reg.intercept_}")

# Simpler model: just RPM and Pamb/Tamb (density-like term)
dsub2 = dsub.copy()
dsub2['rho_amb'] = dsub2['Pamb_Pa'] / (287.0 * dsub2['Tamb_K'])
X2 = dsub2[['RPM_rev_min', 'rho_amb']].values
y2 = dsub2['true_air_flow'].values
reg2 = LinearRegression().fit(X2, y2)
print(f"\n=== Simpler model: air_flow = a*RPM + b*rho_amb + intercept ===")
print(f"R^2: {reg2.score(X2, y2):.4f}")
print(f"a (RPM coef) = {reg2.coef_[0]:.10f}")
print(f"b (rho coef) = {reg2.coef_[1]:.6f}")
print(f"intercept = {reg2.intercept_:.6f}")
