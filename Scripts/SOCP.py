# Assumptions:
#1. All test cases comes from https://labs.ece.uw.edu/pstca/
#2. Angle difference limit is 30 degrees

import gurobipy as gp
from gurobipy import GRB
from Matpower.powerNetwork import Input, get_rateA
import grid_loader as gl
import pandas as pd
import numpy as np
import math

# Test case
n_case = 9

# MPC base
S_base = 100  # MVA

"""Set"""
# Define components of Grid
net = gl.load_test_case_grid(n_case)
generators = net.gen["bus"].to_list()
ext_grid = net.ext_grid["bus"].to_list()
buses = net.bus.index
loads = net.load["bus"].to_list()
lines = []
for i in net.line.index:
    lines.append((net.line.loc[i, "from_bus"], net.line.loc[i, "to_bus"]))

neighbors = {}  # δ(i)
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
time_periods = range(0, 24)  # T
time_periods_plus = range(0, 25)  # T+1


"""Parameters"""

# Get line data
buses_data, lines_data, generators_data, costs_data = Input("Scripts/Matpower/case%s.m"%n_case)
S_max = get_rateA(lines_data)
S_max = {k: v / S_base for k, v in S_max.items()}

Gij = {}
Bij = {}
for line in lines_data:
    Gij[(line.head, line.tail)] = np.real(1/(line.R + 1j * line.X))
    Bij[(line.head, line.tail)] = np.imag(1/(line.R + 1j * line.X))


# Get shunt conductance and susceptance
g_i = {}
b_i = {}
for bus in buses_data:
    g_i[bus.index-1] = bus.G
    b_i[bus.index-1] = bus.B

# Get load data
df_load_p = pd.read_csv("Evaluation/Case%s/load_p.csv"%n_case).transpose()
df_load_q = pd.read_csv("Evaluation/Case%s/load_q.csv"%n_case).transpose()
df_load_p["bus"] = loads
df_load_q["bus"] = loads
P_D = {}
Q_D = {}
for i in buses:
    if i in loads:
        P_D[i] = df_load_p[df_load_p["bus"] == i].drop("bus", axis=1).values[0].tolist()
        Q_D[i] = df_load_q[df_load_q["bus"] == i].drop("bus", axis=1).values[0].tolist()
    else:
        P_D[i] = [0] * len(time_periods)
        Q_D[i] = [0] * len(time_periods)


# Get renewable energy data
df_renewable = pd.read_csv("Evaluation/Case%s/renewable.csv"%n_case).transpose()
df_renewable["bus"] = generators
P_renew = {}
for i in buses:
    if i in generators:
        P_renew[i] = np.array([x/100 for x in df_renewable[df_renewable["bus"] == i].drop("bus", axis=1).values[0].tolist()])
    else:
        P_renew[i] = np.zeros(len(time_periods))


# Get EV data
df_EV_spec = pd.read_csv("Evaluation/Case%s/EV_spec.csv"%n_case)
df_EV_SOC = pd.read_csv("Evaluation/Case%s/ev_soc.csv"%n_case).transpose()
df_EV_demand = pd.read_csv("Evaluation/Case%s/ev_demand.csv"%n_case).transpose()
df_EV_spec["bus"] = loads
df_EV_SOC["bus"] = loads
df_EV_demand["bus"] = loads
C = {}  # Battery capacity
eta_d = {}  # Discharge efficiency
eta_c = {}  # Charge efficiency
Z_init = {}  # Initial SOC
EV_demand = {}  # EV demand
n_EV = {}  # Number of EVs
for i in buses:
    if i in loads:
        C[i] = df_EV_spec.loc[df_EV_spec["bus"] == i, "max_e_mwh"].values[0]
        eta_d[i] = df_EV_spec[df_EV_spec.loc[:, "bus"] == i]["eta_d"].values[0]
        eta_c[i] = df_EV_spec[df_EV_spec.loc[:, "bus"] == i]["eta_c"].values[0]
        Z_init[i] = df_EV_SOC[df_EV_SOC.loc[:, "bus"] == i][0].values[0]
        EV_demand[i] = (df_EV_demand[df_EV_demand.loc[:, "bus"] == i].drop("bus", axis=1).values[0].tolist())
        n_EV[i] = df_EV_spec.loc[df_EV_spec["bus"] == i, "n_car"].values[0]
    else:
        C[i] = 0
        eta_d[i] = 0
        eta_c[i] = 0
        Z_init[i] = 0
        EV_demand[i] = [0] * len(time_periods)
        n_EV[i] = 0

# Generator limits
PG_min = {}
PG_max = P_renew
QG_min = {}
QG_max = {}
for i in buses:
    if i in generators:
        PG_min[i] = net.gen.loc[net.gen["bus"] == i, "min_p_mw"].values[0]
        QG_min[i] = net.gen.loc[net.gen["bus"] == i, "min_q_mvar"].values[0]
        QG_max[i] = net.gen.loc[net.gen["bus"] == i, "max_q_mvar"].values[0]
    else:
        PG_min[i] = 0
        QG_min[i] = 0
        QG_max[i] = 0
PG_min ={key: np.array(value) / S_base for key, value in PG_min.items()}
QG_min ={key: np.array(value) / S_base for key, value in QG_min.items()}
QG_max ={key: np.array(value) / S_base for key, value in QG_max.items()}

# External grid limits
Pex_min = {}
Pex_max = {}
Qex_min = {}
Qex_max = {}
for i in buses:
    if i in ext_grid:
        Pex_min[i] = net.ext_grid.loc[net.ext_grid["bus"] == i, "min_p_mw"].values[0]
        Pex_max[i] = net.ext_grid.loc[net.ext_grid["bus"] == i, "max_p_mw"].values[0]
        Qex_min[i] = net.ext_grid.loc[net.ext_grid["bus"] == i, "min_q_mvar"].values[0]
        Qex_max[i] = net.ext_grid.loc[net.ext_grid["bus"] == i, "max_q_mvar"].values[0]
    else:
        Pex_min[i] = 0
        Pex_max[i] = 0
        Qex_min[i] = 0
        Qex_max[i] = 0
Pex_min ={key: np.array(value) / S_base for key, value in Pex_min.items()}
Pex_max ={key: np.array(value) / S_base for key, value in Pex_max.items()}
Qex_min ={key: np.array(value) / S_base for key, value in Qex_min.items()}
Qex_max ={key: np.array(value) / S_base for key, value in Qex_max.items()}


# Voltage limits
V_min = {}
V_max = {}
for bus in buses_data:
    V_min[bus.index-1] = bus.Vmin
    V_max[bus.index-1] = bus.Vmax

# Phase angle difference limits
theta_max = 30 * math.pi / 180  # Assumption: maximum phase angle difference 30


"""Model"""
# Create a new model
model = gp.Model("Multi-Period OPF")


"""Decision Variables"""
# GENERATOR
# Variables for generation (P_G for active power, Q_G for reactive power)
P_G = model.addVars(buses, time_periods, vtype=GRB.CONTINUOUS, lb=0, name="P_G")
Q_G = model.addVars(buses, time_periods, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="Q_G")


# EXTERNAL GRID
# Variables for external grid (P_ex for active power, Q_ex for reactive power)
P_ex = model.addVars(buses, time_periods, vtype=GRB.CONTINUOUS, lb=0, name="P_ex")
Q_ex = model.addVars(buses, time_periods, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="Q_ex")

# BUS VOLTAGE
# Voltage magnitude (V)
V = model.addVars(buses, time_periods, vtype=GRB.CONTINUOUS, lb=V_min, ub=V_max, name="V")

# LINE FLOW
# Phase angle difference (theta)
theta = model.addVars(buses, buses, time_periods, vtype=GRB.CONTINUOUS, lb=-2*math.pi, ub=2*math.pi, name="theta")
# Cosine and sine related variables
cos = model.addVars(buses, buses, time_periods, vtype=GRB.CONTINUOUS, lb=-1, ub=1, name="cosine")
sin = model.addVars(buses, buses, time_periods, vtype=GRB.CONTINUOUS, lb=-1, ub=1, name="sine")
v_sqrt = model.addVars(buses, buses, time_periods, vtype=GRB.CONTINUOUS, lb=0, name="v_sqrt")
c_ijt = model.addVars(buses, buses, time_periods, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="c_ijt")
s_ijt = model.addVars(buses, buses, time_periods, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="s_ijt")
c_iit = model.addVars(buses, time_periods, vtype=GRB.CONTINUOUS, lb=0, name="c_iit")
# Power flow variables
P_ijt = model.addVars(buses, buses, time_periods, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="P_ijt")
Q_ijt = model.addVars(buses, buses, time_periods, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="Q_ijt")

# EV
# State of Charge (SOC) variables
Z = model.addVars(buses, time_periods_plus, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="SOC")
# Charging and discharging power
P_c = model.addVars(buses, time_periods, vtype=GRB.CONTINUOUS, lb=0, name="P_c")
P_d = model.addVars(buses, time_periods, vtype=GRB.CONTINUOUS, lb=0, name="P_d")


"""Objective Function"""
# Define the cost function
costfunction_generator = net.poly_cost[net.poly_cost["et"] == "gen"]
costfunction_extgrid = net.poly_cost[net.poly_cost["et"] == "ext_grid"]
# Set the objective function
obj = gp.quicksum(
    costfunction_generator.iat[i, 4] * P_G[generators[i], t] ** 2 * S_base ** 2
    + costfunction_generator.iat[i, 3] * P_G[generators[i], t] * S_base
    + costfunction_generator.iat[i, 2]
    for i in range(len(generators))
    for t in time_periods
) + gp.quicksum(
    costfunction_extgrid.iat[i, 4] * P_ex[ext_grid[i], t] ** 2 * S_base ** 2
    + costfunction_extgrid.iat[i, 3] * P_ex[ext_grid[i], t] * S_base
    + costfunction_extgrid.iat[i, 2]
    for i in range(len(ext_grid))
    for t in time_periods
) 
model.setObjective(obj, GRB.MINIMIZE)


"""Constraints"""

# Active power balance 3(a) & Reactive power balance 3(b)
for i in buses:
    for t in time_periods:
        if i in neighbors:
            model.addConstr(
                P_G[i, t] + P_ex[i, t] - P_D[i][t]/S_base - P_c[i, t] + eta_d[i] * P_d[i, t]
                == c_iit[i, t] * g_i[i] + gp.quicksum(P_ijt[i, j, t] for j in neighbors[i]),
                name=f"Power_balance_P_{i}_{t}",
            )
            model.addConstr(
                Q_G[i, t] + Q_ex[i, t] - Q_D[i][t]/S_base
                ==  - c_iit[i, t] * b_i[i] + gp.quicksum(Q_ijt[i, j, t] for j in neighbors[i]),
                name=f"Power_balance_Q_{i}_{t}",
            )
        else:
            model.addConstr(
                P_G[i, t] + P_ex[i, t] - P_D[i][t]/S_base - P_c[i, t] + eta_d[i] * P_d[i, t]
                == c_iit[i, t] * g_i[i],
                name=f"Power_balance_P_{i}_{t}",
            )
            model.addConstr(
                Q_G[i, t] + Q_ex[i, t] - Q_D[i][t]/S_base
                == - c_iit[i, t] * b_i[i], 
                name=f"Power_balance_Q_{i}_{t}"
            )

# Power flow constraints (3c and 3d)
for i, j in lines:
    for t in time_periods:
        model.addConstr(
            P_ijt[i, j, t] == Gij[i, j] * c_iit[i, t] + Gij[i, j] * c_ijt[i, j, t] - Bij[i, j] * s_ijt[i, j, t], name=f"Power_flow_P_{i}_{j}_{t}"
        )
        model.addConstr(
            Q_ijt[i, j, t] == -Bij[i, j] * c_iit[i, t] - Bij[i, j] * c_ijt[i, j, t] - Gij[i, j] * s_ijt[i, j, t], name=f"Power_flow_Q_{i}_{j}_{t}"
        )

# Constraints to build up the cosine and sine variables
for i in buses:
    model.addConstr(
        c_iit[i, t] == V[i, t] ** 2, name=f"c_iit_{i}_{t}"
    )

for i, j in lines:
    for t in time_periods:
        model.addGenConstrCos(theta[i, j, t], cos[i, j, t])
        model.addGenConstrSin(theta[i, j, t], sin[i, j, t])
        model.addConstr(
            v_sqrt[i, j, t] == V[i, t] * V[j, t], name=f"v_sqrt_{i}_{j}_{t}"
        )
        model.addConstr(
            c_ijt[i, j, t] == v_sqrt[i, j, t] * cos[i, j, t], name=f"c_ijt_{i}_{j}_{t}",
        )
        model.addConstr(
            s_ijt[i, j, t] == -1 * v_sqrt[i, j, t] * sin[i, j, t], name=f"s_ijt_{i}_{j}_{t}",
        )

# Cosine variable constraints (3e)
for i in buses:
    for t in time_periods:
        model.addConstr(
            V_min[i] ** 2 <= c_iit[i, t], name=f"cosine_lowerbound_{i}_{t}"
        )
        model.addConstr(
            c_iit[i, t] <= V_max[i] **2 , name=f"cosine_upperbound_{i}_{t}"
        )

# SOCP relaxation constraint (5)
for i, j in lines: 
    for t in time_periods:
        model.addConstr(c_ijt[i, j, t] ** 2 + s_ijt[i, j, t] ** 2 <= c_iit[i, t] * c_iit[j, t], name=f"SOCP_{i}_{j}_{t}")

# Voltage constraints (1f)
for i in buses:
    for t in time_periods:
        model.addConstr(V[i, t] >= V_min[i], name=f"Voltage_lowerbound_{i}_{t}")
        model.addConstr(V[i, t] <= V_max[i], name=f"Voltage_upperbound_{i}_{t}")



# Generator & external grid constraints (1g and 1h)
for i in buses:
    for t in time_periods:
        model.addConstr(P_G[i, t] >= PG_min[i], name=f"PG_lowerbound_{i}_{t}")
        model.addConstr(P_G[i, t] <= PG_max[i][t], name=f"PG_upperbound_{i}_{t}")
        model.addConstr(Q_G[i, t] >= QG_min[i], name=f"QG_lowerbound_{i}_{t}")
        model.addConstr(Q_G[i, t] <= QG_max[i], name=f"QG_upperbound_{i}_{t}")
        model.addConstr(P_ex[i, t] >= Pex_min[i], name=f"Pex_lowerbound_{i}_{t}")
        model.addConstr(P_ex[i, t] <= Pex_max[i], name=f"Pex_upperbound_{i}_{t}")
        model.addConstr(Q_ex[i, t] >= Qex_min[i], name=f"Qex_lowerbound_{i}_{t}")
        model.addConstr(Q_ex[i, t] <= Qex_max[i], name=f"Qex_upperbound_{i}_{t}")


# Line loading constraints (1i)
for i, j in lines:
    for t in time_periods:
        model.addConstr(
            P_ijt[i, j, t] ** 2 + Q_ijt[i, j, t] ** 2 <= S_max[(i, j)] ** 2,
            name=f"Line_loading_{i}_{j}_{t}",
        )

# Phase angle difference constraints (1j)  #TODO: How about other i, j pairs? Should we set constraints for them?
for i, j in lines:
    # Add the constraints for both positive and negative bounds
    model.addConstr(
        theta[i, j, t] <= theta_max,
        name=f"Phase_angle_diff_upper_{i}_{j}_{t}",
    )
    model.addConstr(
        theta[i, j, t] >= -theta_max,
        name=f"Phase_angle_diff_lower_{i}_{j}_{t}",
    )

# SOC balance equation (1k)
for i in buses:
    for t in time_periods:
        model.addConstr(
            Z[i, t + 1] * C[i]/S_base == Z[i, t] * C[i]/S_base + eta_c[i] * P_c[i, t] - P_d[i, t] - EV_demand[i][t]/S_base,
            name=f"SOC_balance_{i}_{t}")

# SOC limit constraints (1l)
for i in buses:
    for t in time_periods:
        model.addConstr(Z[i, t] <= 1, name=f"SOC_ub_{i}_{t}")
        model.addConstr(Z[i, t] >= 0, name=f"SOC_lb_{i}_{t}")

# Initial SOC (1m)
for i in buses:
    model.addConstr(Z[i, 0] == Z_init[i], name=f"Initial_SOC_{i}")

# Charging and discharging power constraints (1n and 1o)
for i in buses:
    for t in time_periods:
        model.addConstr(P_c[i, t] <= 0.001*50*n_EV[i]/S_base, name=f"Charging_power_ub_{i}_{t}")
        model.addConstr(P_d[i, t] <= 0.001*50*n_EV[i]/S_base, name=f"Discharging_power_ub_{i}_{t}")
        model.addConstr(P_c[i, t] >= 0, name=f"Charging_power_lb_{i}_{t}")
        model.addConstr(P_d[i, t] >= 0, name=f"Discharging_power_lb_{i}_{t}")

# # Set non-neighboring and non-self power flow variables to zero
# for i in buses:
#     for j in buses:
#         for t in time_periods:
#             if i in neighbors and j not in neighbors[i] and j != i:
#                 model.addConstr(P_ijt[i, j, t] == 0, name=f"Zero_P_{i}_{j}_{t}")
#                 model.addConstr(Q_ijt[i, j, t] == 0, name=f"Zero_Q_{i}_{j}_{t}")

# # Set SOC, charging and discharging power to zero for non-load buses
# for i in buses:
#     for t in time_periods:
#         if i not in loads:
#             model.addConstr(Z[i, t] == 0, name=f"Zero_SOC_{i}_{t}")
#             model.addConstr(P_c[i, t] == 0, name=f"Zero_Charging_{i}_{t}")
#             model.addConstr(P_d[i, t] == 0, name=f"Zero_Discharging_{i}_{t}")

"""Optimize"""
# model.write("Multi-Period OPF.lp")
# Optimize the model
model.optimize()

# Check if the model solved successfully
if model.status == GRB.OPTIMAL:
    print("Optimal solution found: %g" % model.objVal)
else:
    print("Model did not solve to optimality. Status: ", model.status)
    model.computeIIS()
    # Write the model to a file
    model.write("Scripts/Multi-Period OPF_Case%s.ilp" % n_case)
