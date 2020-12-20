from gym.envs.registration import register

# Multiagent ma_envs
# ----------------------------------------

register(
    id='MultiagentSimple-v0',
    entry_point='multiagent.ma_envs:SimpleEnv',
    # FIXME(cathywu) currently has to be exactly max_path_length parameters in
    # rllab run script
    max_episode_steps=100,
)

register(
    id='MultiagentSimpleSpeakerListener-v0',
    entry_point='multiagent.ma_envs:SimpleSpeakerListenerEnv',
    max_episode_steps=100,
)
