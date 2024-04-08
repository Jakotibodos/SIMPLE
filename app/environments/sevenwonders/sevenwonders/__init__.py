from gym.envs.registration import register

register(
    id='SevenWonders-v0',
    entry_point='sevenwonders.envs:SevenWondersEnv',
)

