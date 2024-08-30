# Gurobi license will be expired on Nov. 17, 2024. Please update the license at https://portal.gurobi.com/iam/licenses/list 

import gurobipy as gp
from gurobipy import GRB
import math

# Define sets (example data)
generators = ['g1', 'g2', 'g3']   # G
time_periods = range(1, 25)       # T
buses = ['b1', 'b2', 'b3', 'b4']  # B
lines = [('b1', 'b2'), ('b2', 'b3'), ('b3', 'b4')]  # L
neighbors = {
    'b1': ['b2'],
    'b2': ['b1', 'b3'],
    'b3': ['b2', 'b4'],
    'b4': ['b3']
}  # δ(i)

# Define parameters (example data)
P_min = {'g1': 10, 'g2': 20, 'g3': 15}
P_max = {'g1': 100, 'g2': 150, 'g3': 120}
Q_min = {'g1': 5, 'g2': 10, 'g3': 7}
Q_max = {'g1': 50, 'g2': 60, 'g3': 55}
V_min = 0.95
V_max = 1.05
theta_max = math.radians(30)
S_max = {('b1', 'b2'): 100, ('b2', 'b3'): 90, ('b3', 'b4'): 85}
C = {'b1': 1.0, 'b2': 1.0, 'b3': 1.0, 'b4': 1.0}
eta_d = 0.95  # Discharge efficiency
eta_c = 0.95  # Charge efficiency

# Create a new model
model = gp.Model("Multi-Period OPF")

# Variables for generation (P_G for active power, Q_G for reactive power)
P_G = model.addVars(generators, time_periods, lb=P_min, ub=P_max, name="P_G")
Q_G = model.addVars(generators, time_periods, lb=Q_min, ub=Q_max, name="Q_G")

# Voltage magnitude (V) and phase angle (theta)
V = model.addVars(buses, time_periods, lb=V_min, ub=V_max, name="V")
theta = model.addVars(lines, time_periods, lb=-theta_max, ub=theta_max, name="theta")

# Power flow variables (P_ij for active power, Q_ij for reactive power)
P_ij = model.addVars(lines, time_periods, name="P_ij")
Q_ij = model.addVars(lines, time_periods, name="Q_ij")

# State of Charge (SOC) variables
Z = model.addVars(buses, time_periods, lb=0, ub=1, name="SOC")

# Charging and discharging power
P_c = model.addVars(buses, time_periods, ub=100, name="Charging")
P_d = model.addVars(buses, time_periods, ub=100, name="Discharging")

# Objective function
# Example quadratic cost function
a = {'g1': 10, 'g2': 12, 'g3': 15}
b = {'g1': 0.1, 'g2': 0.2, 'g3': 0.15}

# Define objective (minimize generation cost)
obj = gp.quicksum(a[i] * P_G[i, t] + b[i] * P_G[i, t] * P_G[i, t] 
                  for i in generators for t in time_periods)

model.setObjective(obj, GRB.MINIMIZE)
# Active power balance
for i in buses:
    for t in time_periods:
        model.addConstr(
            P_G[i, t] - P_c[i, t] + eta_d * P_d[i, t] - P_D[i, t] == gp.quicksum(P_ij[i, j, t] for j in neighbors[i]),
            name=f"Power_balance_P_{i}_{t}"
        )

# Reactive power balance
for i in buses:
    for t in time_periods:
        model.addConstr(
            Q_G[i, t] - Q_D[i, t] == gp.quicksum(Q_ij[i, j, t] for j in neighbors[i]),
            name=f"Power_balance_Q_{i}_{t}"
        )

# Voltage magnitude constraints
for i in buses:
    for t in time_periods:
        model.addConstr(V_min <= V[i, t] <= V_max, name=f"Voltage_limit_{i}_{t}")

# Generator power limits
for i in generators:
    for t in time_periods:
        model.addConstr(P_min[i] <= P_G[i, t] <= P_max[i], name=f"Generator_limit_P_{i}_{t}")
        model.addConstr(Q_min[i] <= Q_G[i, t] <= Q_max[i], name=f"Generator_limit_Q_{i}_{t}")

# SOCP constraints for line flow limits
for (i, j) in lines:
    for t in time_periods:
        model.addQConstr(P_ij[i, j, t] * P_ij[i, j, t] + Q_ij[i, j, t] * Q_ij[i, j, t] <= S_max[i, j] ** 2,
                         name=f"Line_limit_{i}_{j}_{t}")
# SOC balance equation
for i in buses:
    for t in time_periods[:-1]:  # Ensure this is not applied to the last period
        model.addConstr(Z[i, t] * C[i] + eta_c * P_c[i, t] - P_d[i, t] == Z[i, t + 1] * C[i],
                        name=f"SOC_balance_{i}_{t}")

# SOC bounds
for i in buses:
    for t in time_periods:
        model.addConstr(Z[i, t] >= 0, name=f"SOC_min_{i}_{t}")
        model.addConstr(Z[i, t] <= 1, name=f"SOC_max_{i}_{t}")

# Charging and discharging limits
for i in buses:
    for t in time_periods:
        model.addConstr(P_c[i, t] <= 100, name=f"Charging_limit_{i}_{t}")
        model.addConstr(P_d[i, t] <= 100, name=f"Discharging_limit_{i}_{t}")

# Optimize the model
model.optimize()

# Check if the model solved successfully
if model.status == GRB.OPTIMAL:
    print("Optimal solution found")
else:
    print("Model did not solve to optimality")
