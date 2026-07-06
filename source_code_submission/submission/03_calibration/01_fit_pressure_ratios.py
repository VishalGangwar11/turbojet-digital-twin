"""
Day 3 (fix) - Fit PR_compressor and PR_turbine as functions of RPM,
to replace the placeholder constants in the UKF's forward model.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

train = pd.read_csv("train.csv")

train['PR_c'] = train['P2_Pa'] / train['Pamb_Pa']
train['PR_t'] = train['P4_Pa'] / train['P3_Pa']

# Fit PR_c vs RPM (and maybe Mach/Altitude too, but start simple)
X_c = train[['RPM_rev_min']].values
y_c = train['PR_c'].values
reg_c = LinearRegression().fit(X_c, y_c)
print(f"PR_compressor fit: PR_c = {reg_c.coef_[0]:.8f} * RPM + {reg_c.intercept_:.4f}")
print(f"R^2: {reg_c.score(X_c, y_c):.4f}")

X_t = train[['RPM_rev_min']].values
y_t = train['PR_t'].values
reg_t = LinearRegression().fit(X_t, y_t)
print(f"\nPR_turbine fit: PR_t = {reg_t.coef_[0]:.8f} * RPM + {reg_t.intercept_:.4f}")
print(f"R^2: {reg_t.score(X_t, y_t):.4f}")

# Also check if adding Mach/Altitude improves the fit meaningfully
X_c2 = train[['RPM_rev_min', 'Mach', 'Altitude_m']].values
reg_c2 = LinearRegression().fit(X_c2, y_c)
print(f"\nPR_compressor with RPM+Mach+Altitude R^2: {reg_c2.score(X_c2, y_c):.4f}  (compare to RPM-only above)")

X_t2 = train[['RPM_rev_min', 'Mach', 'Altitude_m']].values
reg_t2 = LinearRegression().fit(X_t2, y_t)
print(f"PR_turbine with RPM+Mach+Altitude R^2: {reg_t2.score(X_t2, y_t):.4f}")

print(f"\nPR_c range in data: {train['PR_c'].min():.2f} to {train['PR_c'].max():.2f}")
print(f"PR_t range in data: {train['PR_t'].min():.2f} to {train['PR_t'].max():.2f}")
