"""
Day 3 (fix, continued) - Print full regression coefficients to hardcode into hx().
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

train = pd.read_csv("train.csv")

train['PR_c'] = train['P2_Pa'] / train['Pamb_Pa']
train['PR_t'] = train['P4_Pa'] / train['P3_Pa']

X_c2 = train[['RPM_rev_min', 'Mach', 'Altitude_m']].values
y_c = train['PR_c'].values
reg_c2 = LinearRegression().fit(X_c2, y_c)
print("=== PR_compressor = a*RPM + b*Mach + c*Altitude + intercept ===")
print(f"a (RPM coef)      = {reg_c2.coef_[0]:.10f}")
print(f"b (Mach coef)      = {reg_c2.coef_[1]:.6f}")
print(f"c (Altitude coef) = {reg_c2.coef_[2]:.10f}")
print(f"intercept          = {reg_c2.intercept_:.6f}")
print(f"R^2: {reg_c2.score(X_c2, y_c):.4f}")

X_t = train[['RPM_rev_min']].values
y_t = train['PR_t'].values
reg_t = LinearRegression().fit(X_t, y_t)
print("\n=== PR_turbine = a*RPM + intercept ===")
print(f"a (RPM coef) = {reg_t.coef_[0]:.10f}")
print(f"intercept    = {reg_t.intercept_:.6f}")
print(f"R^2: {reg_t.score(X_t, y_t):.4f}")

# Sanity: predicted PR_c and PR_t should stay in physically sane, positive ranges
pred_prc = reg_c2.predict(X_c2)
pred_prt = reg_t.predict(X_t)
print(f"\nPredicted PR_c range: {pred_prc.min():.2f} to {pred_prc.max():.2f}  (any negative/zero values are a problem)")
print(f"Predicted PR_t range: {pred_prt.min():.2f} to {pred_prt.max():.2f}")
