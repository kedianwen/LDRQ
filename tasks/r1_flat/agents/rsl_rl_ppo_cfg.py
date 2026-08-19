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

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from tasks.r1_flat.symmetry import mirror_obs_actions


@configclass
class R1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Week04 (FR-T6): pinned explicitly rather than inherited from rsl_rl's
    # default, so the reproduction command doesn't depend on an upstream default
    # staying put.
    seed = 42
    num_steps_per_env = 24
    # Week04 trains longer than Week03's 1500: domain randomization makes the
    # curves noisier and convergence slower, which the plan budgets for
    # ("用更长训练步数换 sim2real 迁移能力").
    max_iterations = 3000
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
        # Week04 correction: R1's model is provably mirror-symmetric (all 17
        # left/right link pairs in R1.urdf match to 0 in mass, inertia and centre
        # of mass), yet the Week04 policy walked with markedly different air-time
        # fractions per foot (0.416 vs 0.646). Nothing in the reward or the
        # training loop had ever asked for symmetry. Augmenting each minibatch
        # with its mirror image does (Mittal et al. 2024); see
        # tasks/r1_flat/symmetry.py for the mirror map.
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=mirror_obs_actions,
        ),
    )
