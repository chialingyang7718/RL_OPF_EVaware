import grid_loader as gl
import pandas as pd
import pandapower as pp
import time
import pickle

# Test case
n_case = 9 
feasible_seeds = [0, 2, 5, 12, 17, 19, 20, 21, 25, 26, 28, 29] 

# Define time periods
time_periods = range(0, 24)  # T
time_periods_plus = range(1, 25)  # T+1

# Define parameters
SOC_min = 0.3
SOC_max = 1
charging_rate = 0.0037 # 3.7kWh

total_IPtime = []
total_cost = []

for random_seed in feasible_seeds:
    print(f"Seed: {random_seed}")
    # Load test case
    net = gl.load_test_case_grid(n_case)
    file_path = f"Evaluation/Case{n_case}/{random_seed}"

    # Get load data
    df_load_p = pd.read_csv(f"{file_path}/load_p.csv").transpose()
    df_load_q = pd.read_csv(f"{file_path}/load_q.csv").transpose()

    # Get renewable energy data
    df_renewable = pd.read_csv(f"{file_path}/renewable.csv").transpose()

    # Get EV data
    df_EV_spec = pd.read_csv(f"{file_path}/EV_spec.csv")
    df_EV_SOC = pd.read_csv(f"{file_path}/ev_soc.csv").transpose()
    df_EV_demand = pd.read_csv(f"{file_path}/ev_demand.csv").transpose()

    N_D = len(net.load.index)

    max_e_mwh = []
    # create storage as EV
    for i in net.load.index:
        bus = net.load.loc[i, "bus"]
        n_car = int(net.load.loc[i, "p_mw"])  # number of EVs connected to the bus is assumed to be integer of nominal power of the loads
        max_e_mwh.append(df_EV_spec.loc[df_EV_spec["bus"] == bus, "max_e_mwh"].values[0]/n_car)
        init_soc = df_EV_SOC.loc[:,0].iloc[i]
        pp.create_storage(
            net,
            bus=bus,
            p_mw=0,
            max_e_mwh= max_e_mwh[i],
            soc_percent=init_soc,
            min_e_mwh= 0,
            index=i,
            scaling=n_car,
            in_service=True,
            max_p_mw= min(0.0037, max_e_mwh[i]*(1-init_soc)),
            min_p_mw= max(-0.0037, -max_e_mwh[i]*init_soc),  
            max_q_mvar=0,
            min_q_mvar=0,
            controllable=True,
        )
        df_EV_demand.iloc[i] = df_EV_demand.iloc[i]/net.storage.loc[i, "scaling"]
 
    # initialize dataframes
    df_gen_p = pd.DataFrame(index=time_periods, columns=net.gen.index)
    df_gen_q = pd.DataFrame(index=time_periods, columns=net.gen.index)
    df_charge = pd.DataFrame(index=time_periods, columns=net.storage.index)
    df_soc = pd.DataFrame(index=time_periods, columns=net.storage.index)
    IPtime = []
    Cost = []
    for t in time_periods:
        if t == 0:
            net.storage.loc[:, "soc_percent"] = df_EV_SOC.loc[:, t].values
            for i in range(N_D):
                future_SOC = (net.storage.loc[i, "soc_percent"]* max_e_mwh[i] - df_EV_demand.loc[:,0].iloc[i])/max_e_mwh[i]
                if future_SOC <= SOC_min:
                    net.storage.loc[i, "min_p_mw"] = min(charging_rate, (SOC_min - future_SOC) * max_e_mwh[i])
                    net.storage.loc[i, "max_p_mw"] = min(charging_rate, max_e_mwh[i]*(1-future_SOC))
                else:
                    net.storage.loc[i, "min_p_mw"] = max(-charging_rate, max_e_mwh[i] * (SOC_min - future_SOC))
                    net.storage.loc[i, "max_p_mw"] = min(charging_rate, max_e_mwh[i]*(1-future_SOC))

        for i in range(N_D):
            # Load data
            net.load.loc[i, "p_mw"] = df_load_p.loc[:,t].iloc[i]
            net.load.loc[i, "q_mvar"] = df_load_q.loc[:,t].iloc[i]

            if df_EV_demand.loc[:,t].iloc[i] == 0:
                net.storage.loc[i, "in_service"] = True
            else:
                net.storage.loc[i, "in_service"] = False


        # Renewable energy data
        net.gen.loc[:, "max_p_mw"]= df_renewable.loc[:, t].values


        # Run optimal power flow
        start_time = time.process_time()
        pp.runopp(net, verbose=False)

        # Record results
        IPtime.append(time.process_time() - start_time)
        Cost.append(net.res_cost)
        df_gen_p.loc[t] = net.res_gen.loc[:, "p_mw"]
        df_gen_q.loc[t] = net.res_gen.loc[:, "q_mvar"]
        df_charge.loc[t] = net.res_storage.loc[:, "p_mw"]
        df_soc.loc[t] = net.storage.loc[:, "soc_percent"]

        # Update EV charging limits
        if t != 23:
            # update SOC
            for i in range(N_D):
                # net.storage.loc[i, "p_mw"] = net.res_storage.loc[i, "p_mw"]
                net.storage.loc[i, "soc_percent"] = (net.storage.loc[i, "soc_percent"] * max_e_mwh[i] - df_EV_demand.loc[:,t].iloc[i] + net.res_storage.loc[i, "p_mw"]) / net.storage.loc[i, "max_e_mwh"]
                future_SOC = (net.storage.loc[i, "soc_percent"]* max_e_mwh[i] - df_EV_demand.loc[:,t+1].iloc[i])/max_e_mwh[i]
                if future_SOC <= SOC_min:
                    # no discharging allowed 
                    net.storage.loc[i, "min_p_mw"] = min(charging_rate, (SOC_min - future_SOC) * max_e_mwh[i])
                    net.storage.loc[i, "max_p_mw"] = min(charging_rate, max_e_mwh[i]*(1-future_SOC))
                elif future_SOC >= SOC_max:
                    # no charging allowed 
                    net.storage.loc[i, "min_p_mw"] = max(-charging_rate, max_e_mwh[i] * (SOC_min - future_SOC))
                    net.storage.loc[i, "max_p_mw"] =  max(-charging_rate, max_e_mwh[i]*(SOC_max-future_SOC))
                else:
                    net.storage.loc[i, "min_p_mw"] = max(-charging_rate, max_e_mwh[i] * (SOC_min - future_SOC))
                    net.storage.loc[i, "max_p_mw"] = min(charging_rate, max_e_mwh[i]*(1-future_SOC))
        

    total_IPtime.append(sum(IPtime))
    total_cost.append(sum(Cost))

    # Save results
    df_gen_p.to_csv(f"{file_path}/IPM_gen_p.csv")
    df_gen_q.to_csv(f"{file_path}/IPM_gen_q.csv")
    df_charge.to_csv(f"{file_path}/IPM_charge.csv")
    df_soc.to_csv(f"{file_path}/IPM_soc.csv")

metrics = {
    "IPtime": total_IPtime,
    "Cost": total_cost,
}

with open(f"Evaluation/Case{n_case}/IPM_metrics_Case{n_case}.pickle", "wb") as f:
    pickle.dump(metrics, f)
