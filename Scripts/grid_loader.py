# Import pandapower stuff
import pandapower as pp
import pandapower.networks as pn

# Import simbench stuff
import simbench as sb


def change_gen_into_sgen(grid):
    for i in grid.gen.index:
        bus = grid.gen.loc[i, "bus"]
        p_mw = grid.gen.loc[i, "p_mw"]
        if "max_p_mw" not in grid.gen:
            max_p_mw = (
                grid.gen.loc[i, "p_mw"] * 1.5
            )  # assume the max power output is 1.5 times the current power output if not specified
        else:
            max_p_mw = grid.gen.loc[i, "max_p_mw"]
        if "min_p_mw" not in grid.gen:
            min_p_mw = (
                grid.gen.loc[i, "p_mw"] * 0
            )  # assume the min power output is 0 if not specified
        else:
            min_p_mw = grid.gen.loc[i, "min_p_mw"]
        if "max_q_mvar" not in grid.gen and "q_mvar" in grid.gen:
            max_q_mvar = (
                grid.gen.loc[i, "q_mvar"] * 1.5
            )  # assume the max reactive power output is 1.5 times the current reactive power output if not specified
        elif "q_mvar" not in grid.gen:
            max_q_mvar = 100  # assume the max reactive power output is 100 MVar
        else:
            max_q_mvar = grid.gen.loc[i, "max_q_mvar"]
        if "min_q_mvar" not in grid.gen and "q_mvar" in grid.gen:
            min_q_mvar = abs(grid.gen.loc[i, "q_mvar"]) * -1.5
        elif "q_mvar" not in grid.gen:
            min_q_mvar = -100  # assume the min reactive power output is -100 MVar
        else:
            min_q_mvar = grid.gen.loc[i, "min_q_mvar"]
        grid.gen.drop(i, inplace=True)
        pp.create_sgen(
            grid,
            bus,
            p_mw=p_mw,
            q_mvar=0,
            max_p_mw=max_p_mw,
            min_p_mw=min_p_mw,
            max_q_mvar=max_q_mvar,
            min_q_mvar=min_q_mvar,
        )
    return grid


def change_sgen_into_gen(grid):
    for i in grid.sgen.index:
        bus = grid.sgen.loc[i, "bus"]
        p_mw = grid.sgen.loc[i, "p_mw"]
        if "max_p_mw" not in grid.sgen:
            max_p_mw = (
                grid.sgen.loc[i, "p_mw"] * 1.5
            )  # assume the max power output is 1.5 times the current power output if not specified
        else:
            max_p_mw = grid.sgen.loc[i, "max_p_mw"]
        if "min_p_mw" not in grid.gen:
            min_p_mw = (
                grid.sgen.loc[i, "p_mw"] * 0
            )  # assume the min power output is 0 if not specified
        else:
            min_p_mw = grid.sgen.loc[i, "min_p_mw"]
        if "max_q_mvar" not in grid.sgen and "q_mvar" in grid.sgen:
            max_q_mvar = (
                grid.sgen.loc[i, "q_mvar"] * 1.5
            )  # assume the max reactive power output is 1.5 times the current reactive power output if not specified
        elif "q_mvar" not in grid.sgen:
            max_q_mvar = 100  # assume the max reactive power output is 100 MVar
        else:
            max_q_mvar = grid.sgen.loc[i, "max_q_mvar"]
        if "min_q_mvar" not in grid.sgen and "q_mvar" in grid.sgen:
            min_q_mvar = abs(grid.sgen.loc[i, "q_mvar"]) * -1.5
        elif "q_mvar" not in grid.sgen:
            min_q_mvar = -100  # assume the min reactive power output is -100 MVar
        else:
            min_q_mvar = grid.sgen.loc[i, "min_q_mvar"]
        grid.sgen.drop(i, inplace=True)
        pp.create_gen(
            grid,
            bus,
            p_mw=p_mw,
            vm_pu=1.0,
            max_p_mw=max_p_mw,
            min_p_mw=min_p_mw,
            max_q_mvar=max_q_mvar,
            min_q_mvar=min_q_mvar,
        )
    return grid

def load_test_case_grid(n, str = ""):
    # load the test grid from pandapower
    case = f"case{n}" + str  # n: case number in pandapower
    grid = getattr(pn, case)()
    return grid

def load_simple_grid():
    # load the simple grid from pandapower
    grid = pn.example_simple()
    grid = change_gen_into_sgen(grid)  # change the generators into static generators
    return grid

def load_simbench_grid(grid_code):
    # load the grid from simbench
    grid = sb.get_simbench_net(grid_code)
    return grid
