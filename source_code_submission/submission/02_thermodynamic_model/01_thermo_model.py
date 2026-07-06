"""
Day 1 - 0-D thermodynamic cycle model for a single-spool four-stage turbojet.

Stages: Ambient -> Compressor exit (2) -> Combustor exit (3) -> Turbine exit (4)

Health parameters (what the UKF will estimate later):
    eta_c    : compressor isentropic efficiency
    eta_comb : combustor efficiency (fraction of fuel energy actually released)
    eta_t    : turbine isentropic efficiency

Everything else (ambient conditions, RPM, fuel flow) comes from sensor data.
"""

import numpy as np

# ---- Gas properties (reasonable engineering assumptions - document these in report) ----
GAMMA_AIR = 1.4          # ratio of specific heats, cold air (compressor side)
GAMMA_GAS = 1.33         # ratio of specific heats, hot gas (turbine side) - combustion products
CP_AIR = 1005.0          # J/(kg.K)
CP_GAS = 1148.0          # J/(kg.K), post-combustion gas
R_AIR = 287.0            # J/(kg.K)
LHV_FUEL = 43e6          # J/kg, lower heating value of jet fuel (typical Jet-A/Jet-A1)


def compressor_stage(P1, T1, pressure_ratio, eta_c):
    """
    Given inlet conditions and pressure ratio, compute compressor exit P2, T2.
    pressure_ratio = P2 / P1 (you'll back this out from actual sensor P2 if given,
    or treat as a design/control input if you're predicting forward)
    """
    P2 = P1 * pressure_ratio

    # Ideal (isentropic) exit temperature
    T2_ideal = T1 * (pressure_ratio) ** ((GAMMA_AIR - 1) / GAMMA_AIR)

    # Real exit temp using isentropic efficiency definition:
    # eta_c = (T2_ideal - T1) / (T2_actual - T1)
    T2_actual = T1 + (T2_ideal - T1) / eta_c

    return P2, T2_actual


def combustor_stage(P2, T2, fuel_flow, air_flow, eta_comb):
    """
    Energy balance across combustor.
    Assumes P3 ~ P2 * (small pressure loss factor, e.g. 0.95-0.97)
    """
    PRESSURE_LOSS_FACTOR = 0.96
    P3 = P2 * PRESSURE_LOSS_FACTOR

    # Energy balance: (air_flow + fuel_flow) * cp_gas * T3 = air_flow * cp_air * T2 + eta_comb * fuel_flow * LHV_FUEL
    total_flow = air_flow + fuel_flow
    T3 = (air_flow * CP_AIR * T2 + eta_comb * fuel_flow * LHV_FUEL) / (total_flow * CP_GAS)

    return P3, T3


def turbine_stage(P3, T3, pressure_ratio_turbine, eta_t):
    """
    Turbine expansion. pressure_ratio_turbine = P3/P4 (>1, expansion)
    """
    P4 = P3 / pressure_ratio_turbine

    T4_ideal = T3 * (1 / pressure_ratio_turbine) ** ((GAMMA_GAS - 1) / GAMMA_GAS)

    # eta_t = (T3 - T4_actual) / (T3 - T4_ideal)
    T4_actual = T3 - eta_t * (T3 - T4_ideal)

    return P4, T4_actual


def full_cycle(P1, T1, air_flow, fuel_flow, pr_compressor, pr_turbine,
               eta_c, eta_comb, eta_t):
    """
    Run the full 4-stage cycle given health parameters.
    Returns predicted P2, T2, P3, T3, P4, T4 -- directly comparable to sensor columns.
    """
    P2, T2 = compressor_stage(P1, T1, pr_compressor, eta_c)
    P3, T3 = combustor_stage(P2, T2, fuel_flow, air_flow, eta_comb)
    P4, T4 = turbine_stage(P3, T3, pr_turbine, eta_t)

    return {
        "P2": P2, "T2": T2,
        "P3": P3, "T3": T3,
        "P4": P4, "T4": T4,
    }


def estimate_thrust(air_flow, fuel_flow, T4, P4, P_ambient, exit_area=0.05):
    """
    Rough thrust estimate from exit conditions (simplified, ignores nozzle detail --
    fine per their note: 'detailed CFD not required').
    """
    total_flow = air_flow + fuel_flow
    # exit velocity from ideal gas + isentropic nozzle expansion to ambient (simplified)
    T_exit = T4  # approx, ignoring further nozzle expansion losses for now
    V_exit = np.sqrt(2 * CP_GAS * max(T4 - T_exit, 0) + 1e-6) if T4 > T_exit else 0
    # Placeholder - refine once you see real sensor scales from dataset
    thrust = total_flow * V_exit + (P4 - P_ambient) * exit_area
    return thrust


if __name__ == "__main__":
    # Sanity check with rough placeholder numbers - REPLACE with real ambient/RPM
    # values once you've profiled the dataset (Step 1).
    P1, T1 = 101325, 288.15      # sea level standard
    air_flow = 5.0                # kg/s, guess - refine from RPM/geometry once known
    fuel_flow = 0.15               # kg/s, guess

    result = full_cycle(
        P1, T1, air_flow, fuel_flow,
        pr_compressor=8.0, pr_turbine=6.0,
        eta_c=0.85, eta_comb=0.98, eta_t=0.88
    )
    print("Predicted stage conditions (healthy engine, placeholder inputs):")
    for k, v in result.items():
        print(f"  {k}: {v:.2f}")
