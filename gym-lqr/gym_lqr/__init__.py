from gym.envs.registration import register

register(
    id='lqr-v0',
    entry_point='gym_lqr.envs:LQR_Env',
)
