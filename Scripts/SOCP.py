import gurobipy as gp
from gurobipy import GRB
import math
# from Matpower.powerNetwork import Input
import grid_loader as gl
import pandas as pd


# Define sets and cost function based on test case
net = gl.load_test_case_grid(14)
generators = net.gen.index
ext_grid = net.ext_grid.index
buses = net.bus.index
lines = net.line.index
# buses, lines, generators, costs = Input("case14.m")

time_periods = range(1, 25) # T

neighbors = {}
for i in buses.index:
    neighbors[i] = lines[lines["from_bus"] == i]["to_bus"].to_list()  # δ(i)

# Define parameters (example data)
# TODO: fetch P_D, Q_D, P_renew from data
PG_min = net.gen.loc[:, "min_p_mw"].to_dict()
PG_max = net.gen.loc[:, "max_p_mw"].to_dict()
Pex_min = net.ext_grid.loc[:, "min_p_mw"].to_dict()
Pex_max = net.ext_grid.loc[:, "max_p_mw"].to_dict()
Qex_min = net.ext_grid.loc[:, "min_q_mvar"].to_dict()
Qex_max = net.ext_grid.loc[:, "max_q_mvar"].to_dict()
QG_min = net.gen.loc[:, "min_q_mvar"].to_dict()
QG_max = net.gen.loc[:, "max_q_mvar"].to_dict()
V_min = 0.94
V_max = 1.06

theta_max = math.radians(30) # maximum phase angle difference

S_max =(net.line.loc[:, "max_i_ka"]*net.line.loc[:, "df"]*net.line.loc[:, "parallel"]).to_dict() # maximum current flow to represent maximum apparent power flow

df_EV_spec = pd.read_csv("Evaluation/Case14_EV/EV_spec.csv")
C = (df_EV_spec.loc[:, "max_e_mwh"]/df_EV_spec.loc[:, "n_car"]).to_dict() # Battery capacity

# TODO: fetch eta from data
eta_d = 0.95  # Discharge efficiency
eta_c = 0.95  # Charge efficiency

# Create a new model
model = gp.Model("Multi-Period OPF")

# Variables for generation (P_G for active power, Q_G for reactive power)
P_G = model.addVars(generators, time_periods, lb=PG_min, ub=PG_max, name="P_G")
Q_G = model.addVars(generators, time_periods, lb=QG_min, ub=QG_max, name="Q_G")

# Variables for external grid (P_ex for active power, Q_ex for reactive power)
P_ex = model.addVars(ext_grid, time_periods, lb=Pex_min, ub=Pex_max, name="P_ex")
Q_ex = model.addVars(ext_grid, time_periods, lb=Qex_min, ub=Qex_max, name="Q_ex")

# Voltage magnitude (V) and phase angle difference (theta)
V = model.addVars(buses, time_periods, lb=V_min, ub=V_max, name="V")
theta = model.addVars(lines, time_periods, lb=0, ub=theta_max, name="theta")

# Power flow variables (P_ij for active power, Q_ij for reactive power)
P_ij = model.addVars(lines, time_periods, name="P_ij")
Q_ij = model.addVars(lines, time_periods, name="Q_ij")

# State of Charge (SOC) variables
Z = model.addVars(buses, time_periods, lb=0, ub=1, name="SOC")

# Charging and discharging power
P_c = model.addVars(buses, time_periods, ub=100, name="Charging")
P_d = model.addVars(buses, time_periods, ub=100, name="Discharging")

# Objective function
costfunction_generator = net.poly_cost[net.poly_cost["et"] == "gen"]
costfunction_extgrid = net.poly_cost[net.poly_cost["et"] == "ext_grid"]

# Define objective (minimize generation cost)
obj = gp.quicksum(costfunction_generator.iat[i,4] * P_G[i, t]**2 + costfunction_generator.iat[i,3] * P_G[i, t] \
                  + costfunction_generator.iat[i,2] for i in generators for t in time_periods) + \
    gp.quicksum(costfunction_extgrid.iat[i,4] * P_ex[i, t]**2 + costfunction_extgrid.iat[i,3] * P_ex[i, t] \
                + costfunction_extgrid.iat[i,2] for i in ext_grid for t in time_periods)

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
