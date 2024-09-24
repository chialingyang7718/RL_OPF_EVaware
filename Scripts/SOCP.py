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
# Define components of Grid
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
    if line[0] not in neighbors and line[1] not in neighbors:
        neighbors[line[0]] = [line[1]]
        neighbors[line[1]] = [line[0]]
    elif line[1] not in neighbors and line[0] in neighbors:
        neighbors[line[0]].append(line[1])
        neighbors[line[1]] = [line[0]]
    elif line[1] in neighbors and line[0] not in neighbors:
        neighbors[line[1]].append(line[0])
        neighbors[line[0]] = [line[1]]
    else:
        neighbors[line[0]].append(line[1])
        neighbors[line[1]].append(line[0])

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
df_EV_spec = pd.read_csv("Evaluation/Case14_EV/EV_spec.csv")
df_EV_SOC = pd.read_csv("Evaluation/Case14_EV/ev_soc.csv").transpose()
df_EV_demand = pd.read_csv('Evaluation/Case14_EV/ev_demand.csv').transpose()
df_EV_spec["bus"] = loads
df_EV_SOC["bus"] = loads
df_EV_demand["bus"] = loads
df_EV_spec["battery_capacity"] = df_EV_spec["max_e_mwh"]/df_EV_spec["n_car"]
C = {} # Battery capacity
eta_d = {} # Discharge efficiency
eta_c = {} # Charge efficiency
Z_init = {} # Initial SOC
EV_demand = {} # EV demand
for i in buses:
    if i in loads:
        C[i] = df_EV_spec.loc[df_EV_spec["bus"]==i, "battery_capacity"].values[0]
        eta_d[i] = df_EV_spec[df_EV_spec.loc[:, "bus"]==i]["eta_d"].values[0]
        eta_c[i] = df_EV_spec[df_EV_spec.loc[:, "bus"]==i]["eta_c"].values[0]
        Z_init[i] = df_EV_SOC[df_EV_SOC.loc[:, "bus"]==i][0].values[0]
        EV_demand[i] = df_EV_demand[df_EV_demand.loc[:, "bus"]==i].drop("bus", axis=1).values[0].tolist()
    else:
        C[i] = 0
        eta_d[i] = 0
        eta_c[i] = 0
        Z_init[i] = 0
        EV_demand[i] = [0] * len(time_periods)

# Generator limits
PG_min = {}
PG_max = {}
QG_min = {}
QG_max = {}
for i in buses:
    if i in generators:
        PG_min[i] = net.gen.loc[net.gen["bus"]==i, "min_p_mw"].values[0]
        PG_max[i] = net.gen.loc[net.gen["bus"]==i, "max_p_mw"].values[0]
        QG_min[i] = net.gen.loc[net.gen["bus"]==i, "min_q_mvar"].values[0]
        QG_max[i] = net.gen.loc[net.gen["bus"]==i, "max_q_mvar"].values[0]
    else:
        PG_min[i] = 0
        PG_max[i] = 0
        QG_min[i] = 0
        QG_max[i] = 0

# External grid limits
Pex_min = {}
Pex_max = {}
Qex_min = {}
Qex_max = {}
for i in buses:
    if i in ext_grid:
        Pex_min[i] = net.ext_grid.loc[net.ext_grid["bus"]==i, "min_p_mw"].values[0]
        Pex_max[i] = net.ext_grid.loc[net.ext_grid["bus"]==i, "max_p_mw"].values[0]
        Qex_min[i] = net.ext_grid.loc[net.ext_grid["bus"]==i, "min_q_mvar"].values[0]
        Qex_max[i] = net.ext_grid.loc[net.ext_grid["bus"]==i, "max_q_mvar"].values[0]
    else:
        Pex_min[i] = 0
        Pex_max[i] = 0
        Qex_min[i] = 0
        Qex_max[i] = 0

# Voltage limits
V_min = (net.bus.loc[:, "min_vm_pu"] * net.bus.loc[:,"vn_kv"]).to_dict()
V_max = (net.bus.loc[:, "max_vm_pu"] * net.bus.loc[:,"vn_kv"]).to_dict()
V_max_sqrt = {i: V_max[i] * V_max[i] for i in range(len(V_max))}
V_max_sqrt_neg = {i: -1 * V_max[i] * V_max[i] for i in range(len(V_max))}

# Phase angle difference limits
theta_max = 30 # Assumption: maximum phase angle difference


"""Model"""
# Create a new model
model = gp.Model("Multi-Period OPF")


"""Decision Variables"""
# GENERATOR
# Variables for generation (P_G for active power, Q_G for reactive power)
P_G = model.addVars(buses, time_periods, lb=PG_min, ub=PG_max, name="P_G")
Q_G = model.addVars(buses, time_periods, lb=QG_min, ub=QG_max, name="Q_G")


# EXTERNAL GRID
# Variables for external grid (P_ex for active power, Q_ex for reactive power)
P_ex = model.addVars(buses, time_periods, lb=Pex_min, ub=Pex_max, name="P_ex")
Q_ex = model.addVars(buses, time_periods, lb=Qex_min, ub=Qex_max, name="Q_ex")

# BUS VOLTAGE
# Voltage magnitude (V)
V = model.addVars(buses, time_periods, lb=V_min, ub=V_max, name="V")

# LINE FLOW
# Phase angle difference (theta)
theta = model.addVars(buses, buses, time_periods, lb=0, ub=theta_max, name="theta")
# Cosine and sine related variables
cos = model.addVars(buses, buses, time_periods, lb=-1, ub=1, name="cosine")
sin = model.addVars(buses, buses, time_periods, lb=-1, ub=1, name="sine")
v_sqrt = model.addVars(buses, buses, time_periods, lb=0, ub=V_max_sqrt, name="v_sqrt")
c_ijt = model.addVars(buses, buses, time_periods, lb=V_max_sqrt_neg, ub=V_max_sqrt, name="c_ijt")
s_ijt = model.addVars(buses, buses, time_periods, lb=V_max_sqrt_neg, ub=V_max_sqrt, name="s_ijt")
# Power flow variables
P_ijt = model.addVars(buses, buses, time_periods, name="P_ijt")
Q_ijt = model.addVars(buses, buses, time_periods, name="Q_ijt")

# EV
# State of Charge (SOC) variables
Z = model.addVars(buses, time_periods_plus, lb=0, ub=1, name="SOC")
# Charging and discharging power
P_c = model.addVars(buses, time_periods, name="Charging")
P_d = model.addVars(buses, time_periods, name="Discharging")


"""Objective Function"""
# Define the cost function
costfunction_generator = net.poly_cost[net.poly_cost["et"] == "gen"]
costfunction_extgrid = net.poly_cost[net.poly_cost["et"] == "ext_grid"]
# Set the objective function
obj = gp.quicksum(costfunction_generator.iat[i,4] * P_G[generators[i], t] * P_G[generators[i], t] + costfunction_generator.iat[i,3] * P_G[generators[i], t] \
                  + costfunction_generator.iat[i,2] for i in range(len(generators)) for t in time_periods) + \
    gp.quicksum(costfunction_extgrid.iat[i,4] * P_ex[ext_grid[i], t]* P_ex[ext_grid[i], t]+ costfunction_extgrid.iat[i,3] * P_ex[ext_grid[i], t] \
                + costfunction_extgrid.iat[i,2] for i in range(len(ext_grid)) for t in time_periods)
model.setObjective(obj, GRB.MINIMIZE)


"""Constraints"""

# Set non-neighboring and non-self cosine, sine and power flow variables to zero
for i in buses:
    for j in buses:
        for t in time_periods:
            if i in neighbors and j not in neighbors[i] and j != i:
                model.addConstr(
                    P_ijt[i, j, t] == 0,
                    name=f"Zero_P_{i}_{j}_{t}"
                )
                model.addConstr(
                    Q_ijt[i, j, t] == 0,
                    name=f"Zero_Q_{i}_{j}_{t}"
                )

# Set SOC to zero for non-load buses
for i in buses:
    for t in time_periods:
        if i not in loads:
            model.addConstr(
                Z[i, t] == 0,
                name=f"Zero_SOC_{i}_{t}"
            )
# Set charging and discharging power to zero for non-EV buses (non-load buses)
for i in buses:
    for t in time_periods:
        if i not in loads:
            model.addConstr(
                P_c[i, t] == 0,
                name=f"Zero_Charging_{i}_{t}"
            )
            model.addConstr(
                P_d[i, t] == 0,
                name=f"Zero_Discharging_{i}_{t}"
            )

# Active power balance 3(a)
for i in buses:
    for t in time_periods:
        if i in neighbors:
            model.addConstr(
                P_G[i, t] + P_ex[i, t] - P_D[i][t] - P_c[i, t] + eta_d[i] * P_d[i, t] == gp.quicksum(P_ijt[i, j, t] for j in neighbors[i]),
                name=f"Power_balance_P_{i}_{t}"
            )
        else:
            model.addConstr(
                P_G[i, t] + P_ex[i, t] - P_D[i][t] - P_c[i, t] + eta_d[i] * P_d[i, t] == 0,
                name=f"Power_balance_P_{i}_{t}"
            )

# Reactive power balance 3(b)
for i in buses:
    for t in time_periods:
        if i in neighbors:
                model.addConstr(
                    Q_G[i, t] + Q_ex[i, t] - Q_D[i][t] == gp.quicksum(Q_ijt[i, j, t] for j in neighbors[i]),
                    name=f"Power_balance_Q_{i}_{t}"
                )
        else:
            model.addConstr(
                Q_G[i, t] + Q_ex[i, t] - Q_D[i][t] == 0,
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
            model.addGenConstrCos(theta[i, j, t], cos[i, j, t])
            model.addGenConstrSin(theta[i, j, t], sin[i, j, t])
            model.addConstr(v_sqrt[i, j, t] == V[i, t] * V[j, t], name=f"v_sqrt_{i}_{j}_{t}")
            model.addConstr(c_ijt[i, j, t] == v_sqrt[i, j, t] * cos[i, j, t], name=f"c_ijt_{i}_{j}_{t}")
            model.addConstr(s_ijt[i, j, t] == -1 * v_sqrt[i, j, t] * sin[i, j, t], name=f"s_ijt_{i}_{j}_{t}")


# Cosine variable constraints (3e)
for i in buses:
    for t in time_periods:
        model.addConstr(
            V_min[i] * V_min[i] <= c_ijt[i, i, t],
            name=f"cosine_lowerbound_{i}_{t}"
        )
        model.addConstr(
            c_ijt[i, i, t] <= V_max[i] * V_max[i],
            name=f"cosine_upperbound_{i}_{t}"
        )
    
# SOCP relaxation constraint (5)
for (i, j) in lines:
    for t in time_periods:
        model.addConstr(
            c_ijt[i, j, t] ** 2 + s_ijt[i, j, t] ** 2 <= c_ijt[i, i, t] * c_ijt[i, j, t],
            name=f"SOCP_{i}_{j}_{t}"
        )

# Generator & external grid constraints (1g and 1h)
for i in buses:
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
        model.addConstr(
            P_ex[i, t] >= Pex_min[i],
            name=f"Pex_lowerbound_{i}_{t}"
        )
        model.addConstr(
            P_ex[i, t] <= Pex_max[i],
            name=f"Pex_upperbound_{i}_{t}"
        )
        model.addConstr(
            Q_ex[i, t] >= Qex_min[i],
            name=f"Qex_lowerbound_{i}_{t}"
        )
        model.addConstr(
            Q_ex[i, t] <= Qex_max[i],
            name=f"Qex_upperbound_{i}_{t}"
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
for i in buses:
    for t in time_periods:
        model.addConstr(
            Z[i, t+1] * C[i] == Z[i, t] * C[i] + eta_c[i] * P_c[i, t] - P_d[i, t] - EV_demand[i][t],
            name=f"SOC_balance_{i}_{t}"
        )
# SOC limit constraints (1l)
for i in buses:
    for t in time_periods:
        model.addConstr(
            Z[i, t] <= 1,
            name=f"SOC_upperbound_{i}_{t}"
        )
        model.addConstr(
            Z[i, t] >= 0,
            name=f"SOC_lowerbound_{i}_{t}"
        )

# Initial SOC (1m)
for i in buses:
    model.addConstr(
        Z[i, 0] == Z_init[i],
        name=f"Initial_SOC_{i}"
    )

# Charging and discharging power constraints (1n and 1o)
for i in buses:
    for t in time_periods:
        model.addConstr(
            P_c[i, t] <= (1 - Z[i, t]) * C[i],
            name=f"Charging_power_{i}_{t}"
        )
        model.addConstr(
            P_d[i, t] <= Z[i, t] * C[i],
            name=f"Discharging_power_{i}_{t}"
        )

# Optimize the model
model.optimize()



# Check if the model solved successfully
if model.status == GRB.OPTIMAL:
    print("Optimal solution found")
else:
    print("Model did not solve to optimality. Status: ", model.status)
    model.computeIIS()
    # Write the model to a file
    model.write("Multi-Period OPF.ilp")

    print('\nThe following constraints and variables are in the IIS:')
    for c in model.getConstrs():
        if c.IISConstr: print(f'\t{c.constrname}: {model.getRow(c)} {c.Sense} {c.RHS}')

    for v in model.getVars():
        if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
        if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')

# TODO: Model infeasible during presolve already. If commented constranint 3a and 3b, model is feasible.
# See https://support.gurobi.com/hc/en-us/articles/4402704428177-How-do-I-resolve-the-error-Model-is-infeasible-or-unbounded
