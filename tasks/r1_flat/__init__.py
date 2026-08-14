import gymnasium as gym

from . import flat_env_cfg

gym.register(
    id="Isaac-Velocity-Flat-R1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:R1FlatEnvCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-R1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:R1FlatEnvCfg_PLAY",
    },
)
