from gym.envs.registration import register

register(
    id='robotorydownscale-v0',
    entry_point='gym_robotorydownscale.envs:RobotorydownscaleEnv',
)
