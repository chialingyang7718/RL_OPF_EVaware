from gymnasium import envs

envs.register(
    id="PowerGrid",
    entry_point="CustEnv.Env.PowerGridEnv:PowerGrid",
    max_episode_steps=1000,
    #  kwargs={}, #default key arguments pass to env
)