# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""rsl_rl PPO runner config for Isaac-Velocity-Flat-R1-v0 (Week03, FR-T4 first training).

Asymmetric actor-critic needs no explicit switch here: rsl_rl's
RslRlVecEnvWrapper auto-detects the env's "critic" observation group
(tasks/r1_flat/flat_env_cfg.py's ObservationsCfg.critic) and sizes the
critic's input from it, independently of the actor's "policy"-sized input.
This config only needs to size the two MLPs -- see scripts/check_asymmetric_ac.py
for a runtime dimension check that they're actually different.

Values mirror isaaclab_tasks' H1FlatPPORunnerCfg (a same-scale bipedal
humanoid task) -- a reasonable starting point for "learn to move", not tuned
for R1 specifically yet.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class R1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "r1_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
