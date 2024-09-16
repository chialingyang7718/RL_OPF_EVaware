# Assumptions: 
# 1. The admittance of the buses is not considered
# 2. The power system frequency is assumed to be 50 Hz
# 3. All test cases comes from https://labs.ece.uw.edu/pstca/

import gurobipy as gp
from gurobipy import GRB
from Matpower.powerNetwork import Input, get_rateA
import grid_loader as gl
import pandas as pd
import numpy as np



"""Set"""
# Define componets of Grid
net = gl.load_test_case_grid(14)
generators = net.gen["bus"].to_list()
ext_grid = net.ext_grid["bus"].to_list()
buses = net.bus.index
loads = net.load["bus"].to_list()
lines = []
for i in net.line.index:
    lines.append((net.line.loc[i, "from_bus"], net.line.loc[i, "to_bus"]))

neighbors = {} # δ(i)
for line in lines:
    if line[0] not in neighbors:
        neighbors[line[0]] = [line[1]]
    else:
        neighbors[line[0]].append(line[1])

# Define time periods
time_periods = range(0, 24) # T
time_periods_plus = range(0, 25) # T+1



"""Parameters"""

# Get line data
net.line["admittance"] = net.line["parallel"]/net.line["length_km"]/(net.line["r_ohm_per_km"] + 1j*net.line["x_ohm_per_km"])
net.line["conductance"] = np.real(net.line["admittance"])
net.line["susceptance"] = np.imag(net.line["admittance"])
buses_data, lines_data, generators_data, costs_data = Input("Scripts/Matpower/case14.m")
S_max = get_rateA(lines_data)


# Get load data
df_load_p = pd.read_csv("Evaluation/Case14_EV/load_p.csv").transpose()
df_load_q = pd.read_csv("Evaluation/Case14_EV/load_q.csv").transpose()
df_load_p["bus"] = loads
df_load_q["bus"] = loads
P_D = {}
Q_D = {}
for i in buses:
    if i in loads:
        P_D[i] = df_load_p[df_load_p["bus"] == 1].drop("bus", axis=1).values[0].tolist()
        Q_D[i] = df_load_q[df_load_q["bus"] == i].drop("bus", axis=1).values[0].tolist()
    else:
        P_D[i] = [0] * len(time_periods)
        Q_D[i] = [0] * len(time_periods)


# Get renewable energy data
df_renewable = pd.read_csv("Evaluation/Case14_EV/renewable.csv").transpose()
df_renewable["bus"] = generators
P_renew = {}
for i in buses:
    if i in generators:
        P_renew[i] = df_renewable[df_renewable["bus"] == i].drop("bus", axis=1).values[0].tolist()
    else:
        P_renew[i] = [0] * len(time_periods)


# Get EV data
df_EV_spec = pd.read_csv("../Evaluation/Case14_EV/EV_spec.csv")
df_EV_SOC = pd.read_csv("../Evaluation/Case14_EV/ev_soc.csv")
C = (df_EV_spec.loc[:, "max_e_mwh"]/df_EV_spec.loc[:, "n_car"]).to_dict() # Battery capacity
eta_d = {} # Discharge efficiency
eta_c = {} # Charge efficiency
for i in buses:
    if i in loads:
        eta_d[i] = df_EV_spec[df_EV_spec.loc[:, "bus"]==i]["eta_d"].values[0]
        eta_c[i] = df_EV_spec[df_EV_spec.loc[:, "bus"]==i]["eta_c"].values[0]
    else:
        eta_d[i] = 0
        eta_c[i] = 0
Z_init = df_EV_SOC.iloc[1].to_dict() # Initial SOC


# Define limits
PG_min = net.gen.loc[:, "min_p_mw"].to_dict()
PG_max = net.gen.loc[:, "max_p_mw"].to_dict()
Pex_min = net.ext_grid.loc[:, "min_p_mw"].to_dict()
Pex_max = net.ext_grid.loc[:, "max_p_mw"].to_dict()
Qex_min = net.ext_grid.loc[:, "min_q_mvar"].to_dict()
Qex_max = net.ext_grid.loc[:, "max_q_mvar"].to_dict()
QG_min = net.gen.loc[:, "min_q_mvar"].to_dict()
QG_max = net.gen.loc[:, "max_q_mvar"].to_dict()
V_min = net.bus.loc[:, "min_vm_pu"].to_dict()
V_max = net.bus.loc[:, "max_vm_pu"].to_dict()
V_max_sqrt = {i: V_max[i]**2 for i in V_max}
V_max_sqrt_neg = {i: -V_max[i]**2 for i in V_max}
theta_max = 30 # maximum phase angle difference


"""Model"""
# Create a new model
model = gp.Model("Multi-Period OPF")


"""Decision Variables"""
# Variables for generation (P_G for active power, Q_G for reactive power)
P_G = model.addVars(buses, time_periods, lb=PG_min, ub=PG_max, name="P_G")
Q_G = model.addVars(buses, time_periods, lb=QG_min, ub=QG_max, name="Q_G")

# Set generation to zero for non-generator buses
for i in buses:
    for t in time_periods:
        if i not in generators:
            P_G[i, t] = 0
            Q_G[i, t] = 0

# Voltage magnitude (V) and phase angle difference (theta)
V = model.addVars(buses, time_periods, lb=V_min, ub=V_max, name="V")
theta = model.addVars(buses, buses, time_periods, lb=0, ub=theta_max, name="theta")

# Cosine and sine related variables
cos = model.addVars(buses, buses, time_periods, name="cosine")
sin = model.addVars(buses, buses, time_periods, name="sine")
c_ijt = model.addVars(buses, buses, time_periods, lb=0, ub=V_max_sqrt, name="c_ijt")
s_ijt = model.addVars(buses, buses, time_periods, lb=V_max_sqrt_neg, ub=0, name="s_ijt")
v_sqrt = model.addVars(buses, buses, time_periods, lb=0, ub=V_max, name="v_sqrt")

# Set non-neighboring and non-self cosine and sine variables to zero
for i in buses:
    for j in buses:
        for t in time_periods:
            if i in neighbors and j not in neighbors[i] and j != i:
                c_ijt[i, j, t] = 0
                s_ijt[i, j, t] = 0


# State of Charge (SOC) variables
Z = model.addVars(buses, time_periods_plus, lb=0, ub=1, name="SOC")
# Set SOC to zero for non-load buses
for i in buses:
    for t in time_periods:
        if i not in loads:
            Z[i, t] = 0

# Charging and discharging power
P_c = model.addVars(buses, time_periods, ub=100, name="Charging")
P_d = model.addVars(buses, time_periods, ub=100, name="Discharging")
# TODO: adjust the ub and lb for charging and discharging power

# Set charging and discharging power to zero for non-EV buses (non-load buses)
for i in buses:
    for t in time_periods:
        if i not in loads:
            P_c[i, t] = 0
            P_d[i, t] = 0

# Variables for external grid (P_ex for active power, Q_ex for reactive power)
P_ex = model.addVars(ext_grid, time_periods, lb=Pex_min, ub=Pex_max, name="P_ex")
Q_ex = model.addVars(ext_grid, time_periods, lb=Qex_min, ub=Qex_max, name="Q_ex")

# Power flow variables
P_ijt = model.addVars(buses, buses, time_periods, name="P_ijt")
Q_ijt = model.addVars(buses, buses, time_periods, name="Q_ijt")


"""Objective Function"""
# Define the cost function
costfunction_generator = net.poly_cost[net.poly_cost["et"] == "gen"]
costfunction_extgrid = net.poly_cost[net.poly_cost["et"] == "ext_grid"]

obj = gp.quicksum(costfunction_generator.iat[i,4] * P_G[generators[i], t]**2 + costfunction_generator.iat[i,3] * P_G[generators[i], t] \
                  + costfunction_generator.iat[i,2] for i in range(len(generators)) for t in time_periods) + \
    gp.quicksum(costfunction_extgrid.iat[i,4] * P_ex[ext_grid[i], t]**2 + costfunction_extgrid.iat[i,3] * P_ex[ext_grid[i], t] \
                + costfunction_extgrid.iat[i,2] for i in range(len(ext_grid)) for t in time_periods)

model.setObjective(obj, GRB.MINIMIZE)


"""Constraints"""
# Active power balance (3a)
for i in buses:
    for t in time_periods:
        if i in neighbors:
            for j in neighbors[i]:
                model.addConstr(
                    P_G[i, t] - P_D[i][t] - P_c[i, t] + eta_d * P_d[i, t] == gp.quicksum(P_ijt[i, j, t] for j in neighbors[i]),
                    name=f"Power_balance_P_{i}_{t}"
                )
        else:
            model.addConstr(
                P_G[i, t] - P_D[i][t] - P_c[i, t] + eta_d * P_d[i, t] == 0,
                name=f"Power_balance_P_{i}_{t}"
            )

# Reactive power balance (3b) 
for i in buses:
    for t in time_periods:
        if i in neighbors:
            for j in neighbors[i]:
                model.addConstr(
                    Q_G[i, t] - Q_D[i][t] == gp.quicksum(Q_ijt[i, j, t] for j in neighbors[i]),
                    name=f"Power_balance_Q_{i}_{t}"
                )
        else:
            model.addConstr(
                Q_G[i, t] - Q_D[i][t] == 0,
                name=f"Power_balance_Q_{i}_{t}"
            )

# Power flow constraints (3c and 3d)
for (i, j) in lines:
    for t in time_periods:

        model.addConstr(
            P_ijt[i, j, t] == net.line.loc[(net.line["from_bus"] == i) & (net.line["to_bus"] == j), "conductance"].values[0] * c_ijt[i, i, t]
                            + net.line.loc[(net.line["from_bus"] == i) & (net.line["to_bus"] == j), "conductance"].values[0] * c_ijt[i, j, t]
                            - net.line.loc[(net.line["from_bus"] == i) & (net.line["to_bus"] == j), "susceptance"].values[0] * s_ijt[i, j, t],
            name=f"Power_flow_P_{i}_{j}_{t}"
        )
        
        model.addConstr(
            Q_ijt[i, j, t] == - net.line.loc[(net.line["from_bus"] == i) & (net.line["to_bus"] == j), "susceptance"].values[0] * c_ijt[i, i, t]
                            - net.line.loc[(net.line["from_bus"] == i) & (net.line["to_bus"] == j), "susceptance"].values[0] * c_ijt[i, j, t]
                            - net.line.loc[(net.line["from_bus"] == i) & (net.line["to_bus"] == j), "conductance"].values[0] * s_ijt[i, j, t],
            name=f"Power_flow_Q_{i}_{j}_{t}"
        )

# Constraints to build up the cosine and sine variables
for i in buses:
    for j in buses:
        for t in time_periods: 
            model.addGenConstrCos(theta[i,j,t], cos[i, j, t])
            model.addGenConstrSin(theta[i,j,t], sin[i, j, t])
            model.addConstr(v_sqrt[i, j, t] == V[i, t] * V[j, t])
            model.addConstr(c_ijt[i, j, t] == v_sqrt[i, j, t] * cos[i, j, t])
            model.addConstr(s_ijt[i, j, t] == -1 * v_sqrt[i, j, t] * sin[i, j, t])

for i in buses:
    for t in time_periods:
        model.addConstr(
            V_min[i]**2 <= c_ijt[i, i, t],
            name=f"cosine_lowerbound_{i}_{t}"
        )
        model.addConstr(
            c_ijt[i, i, t] <= V_max[i]**2,
            name=f"cosine_upperbound_{i}_{t}"
        )
        
# SOCP relaxation constraint (5)
for (i, j) in lines:
    for t in time_periods:
        model.addConstr(
            c_ijt[i, j, t] ** 2 + s_ijt[i, j, t] ** 2 <= c_ijt[i, i, t] * c_ijt[j, j, t],
            name=f"SOCP_{i}_{j}_{t}"
        )

# Generator constraints (1g and 1h)
for i in generators:
    for t in time_periods:
        model.addConstr(
            P_G[i, t] >= PG_min[i],
            name=f"PG_lowerbound_{i}_{t}"
        )
        model.addConstr(
            P_G[i, t] <= PG_max[i],
            name=f"PG_upperbound_{i}_{t}"
        )
        model.addConstr(
            Q_G[i, t] >= QG_min[i],
            name=f"QG_lowerbound_{i}_{t}"
        )
        model.addConstr(
            Q_G[i, t] <= QG_max[i],
            name=f"QG_upperbound_{i}_{t}"
        )

# Line loading constraints (1i)
for (i, j) in lines:
    for t in time_periods:
        model.addConstr(
            P_ijt[i, j, t] ** 2 + Q_ijt[i, j, t] ** 2 <= S_max[(i, j)],
            name=f"Line_loading_{i}_{j}_{t}"
        )

# Phase angle difference constraints (1j)
for i in buses:
    for j in buses:
        for t in time_periods:
            if i in neighbors and j in neighbors[i]:
                # Add the constraints for both positive and negative bounds
                model.addConstr(
                    theta[i, j, t] <= theta_max,
                    name=f"Phase_angle_diff_upper_{i}_{j}_{t}"
                )
                model.addConstr(
                    theta[i, j, t] >= -theta_max,
                    name=f"Phase_angle_diff_lower_{i}_{j}_{t}"
                )

# SOC balance equation (1k)
for i in loads:
    for t in time_periods:
        model.addConstr(
            Z[i, t+1] * C[i] == Z[i, t] * C[i] + eta_c * P_c[i, t] - P_d[i, t],
            name=f"SOC_balance_{i}_{t}"
        )

# Start from here

# # SOC balance equation
# for i in buses:
#     for t in time_periods[:-1]:  # Ensure this is not applied to the last period
#         model.addConstr(Z[i, t] * C[i] + eta_c * P_c[i, t] - P_d[i, t] == Z[i, t + 1] * C[i],
#                         name=f"SOC_balance_{i}_{t}")

# # SOC bounds
# for i in buses:
#     for t in time_periods:
#         model.addConstr(Z[i, t] >= 0, name=f"SOC_min_{i}_{t}")
#         model.addConstr(Z[i, t] <= 1, name=f"SOC_max_{i}_{t}")

# # Charging and discharging limits
# for i in buses:
#     for t in time_periods:
#         model.addConstr(P_c[i, t] <= 100, name=f"Charging_limit_{i}_{t}")
#         model.addConstr(P_d[i, t] <= 100, name=f"Discharging_limit_{i}_{t}")

# # Optimize the model
# model.optimize()

# # Check if the model solved successfully
# if model.status == GRB.OPTIMAL:
#     print("Optimal solution found")
# else:
#     print("Model did not solve to optimality")
