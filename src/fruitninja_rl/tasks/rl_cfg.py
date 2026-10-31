"""PPO (rsl_rl) hyperparameters for the slicing task."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def make_slicing_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    # Actions are normalized EE pose deltas; see the action term in slicing_env_cfg.
    clip_actions=1.0,
    experiment_name="iiwa_slice",
    # Switch to "wandb" (after `wandb login`) to share runs across the team.
    logger="tensorboard",
    # One rollout (24 steps = 0.48 s) is roughly one fruit drop.
    num_steps_per_env=24,
    max_iterations=2_000,
    save_interval=100,
  )
