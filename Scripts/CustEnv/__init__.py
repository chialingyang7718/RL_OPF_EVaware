from gymnasium import envs

envs.register(
     id='PowerGrid-v0',
     entry_point='CustEnv.Env.PowerGridEnv:PowerGrid',
     max_episode_steps=1000,
    #  kwargs={}, #default key arguments pass to env
)
envs.register(
     id='PowerGrid-v1',
     entry_point='CustEnv.Env.PowerGridEnv_new:PowerGrid',
     max_episode_steps=1000,
    #  kwargs={}, #default key arguments pass to env
)