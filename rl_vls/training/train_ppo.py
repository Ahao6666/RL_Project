"""PPO 训练入口（阶段 1）。

用法：
    python -m rl_vls.training.train_ppo --timesteps 500000
    tensorboard --logdir runs/
"""
from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_vls.envs.landing_env import LandingEnv

# 视觉噪声参数：训练时可逐步加大（课程学习），或做 domain randomization
NOISE_CFG = dict(sigma_dist=0.02, sigma_offaxis=0.05, sigma_floor=0.005, dropout_prob=0.02)


def make_env(seed: int = 0):
    def _init():
        env = LandingEnv(noise_cfg=NOISE_CFG, seed=seed)
        return Monitor(env)

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="runs/ppo_landing")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cpu",
                        help="MlpPolicy 建议 cpu；SB3 默认 auto 会占用 GPU 但收益很小")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    train_env = VecNormalize(DummyVecEnv([make_env(args.seed)]), norm_obs=True, norm_reward=True)
    eval_env = VecNormalize(DummyVecEnv([make_env(args.seed + 1000)]), norm_obs=True, norm_reward=False, training=False)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.log_dir,
        device=args.device,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(args.save_dir, "best"),
            log_path=args.log_dir,
            eval_freq=10_000,
            n_eval_episodes=20,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path=args.save_dir,
            name_prefix="ppo_landing",
        ),
    ]

    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save(os.path.join(args.save_dir, "ppo_landing_final"))
    train_env.save(os.path.join(args.save_dir, "vecnormalize.pkl"))
    print(f"模型已保存到 {args.save_dir}/ppo_landing_final.zip")


if __name__ == "__main__":
    main()
