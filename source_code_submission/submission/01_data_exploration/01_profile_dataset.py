"""
Day 1 - Dataset profiling for HAL x IIT Indore Digital Twin challenge.
Run this FIRST before writing any physics/filter code.
"""

import pandas as pd
import matplotlib.pyplot as plt

# ---- 1. Load ----
# Update path once you've downloaded from the Drive link
df = pd.read_csv("engine_dataset.csv")  # rename to actual filename

print("=== Shape ===")
print(df.shape)

print("\n=== Columns ===")
print(df.columns.tolist())

print("\n=== Engine IDs ===")
print("Number of unique engines:", df["Engine ID"].nunique())
print(df["Engine ID"].value_counts().head())

print("\n=== Cycles per engine ===")
print(df.groupby("Engine ID")["Cycle"].max().describe())

print("\n=== Basic stats ===")
print(df.describe())

print("\n=== Missing values ===")
print(df.isna().sum())

# ---- 2. Visualize degradation trend for a handful of engines ----
sensor_cols = [
    "P2", "T2", "P3", "T3", "P4", "T4",
    "Shaft Speed (RPM)", "Fuel Flow Rate"
]
# adjust exact column names to match your CSV header once you see it

sample_engines = df["Engine ID"].unique()[:5]

fig, axes = plt.subplots(len(sensor_cols), 1, figsize=(10, 3 * len(sensor_cols)), sharex=False)

for col, ax in zip(sensor_cols, axes):
    for eng in sample_engines:
        sub = df[df["Engine ID"] == eng]
        ax.plot(sub["Cycle"], sub[col], alpha=0.7, label=f"Engine {eng}")
    ax.set_title(col)
    ax.legend(fontsize=6)

plt.tight_layout()
plt.savefig("sensor_trends.png", dpi=120)
print("\nSaved sensor_trends.png -- look for monotonic drift = degradation signature")

# ---- 3. Check flight condition spread ----
print("\n=== Altitude / Mach range ===")
print(df[["Altitude", "Mach Number"]].describe())

# ---- 4. Look for any degradation/health label columns ----
possible_label_cols = [c for c in df.columns if "health" in c.lower() or "degrad" in c.lower() or "fault" in c.lower()]
print("\n=== Possible ground-truth label columns found ===")
print(possible_label_cols if possible_label_cols else "None found - likely unsupervised, validate via physical plausibility instead")
