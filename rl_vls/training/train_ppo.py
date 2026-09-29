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

# tau 默认围绕 SITL 实测值（xy≈1.3s, z≈0.95s）收窄采样，两侧留余量覆盖真机差异
DEFAULT_TAU_XY = (0.9, 1.7)
DEFAULT_TAU_Z = (0.7, 1.2)


def make_env(seed: int = 0, tau_xy=DEFAULT_TAU_XY, tau_z=DEFAULT_TAU_Z,
             low_alt_speed_w: float = 2.0, action_smooth_w: float = 0.0,
             action_delay_range: tuple = (0, 0)):
    def _init():
        env = LandingEnv(
            noise_cfg=NOISE_CFG,
            tau_xy_range=tau_xy,
            tau_z_range=tau_z,
            low_alt_speed_w=low_alt_speed_w,
            action_smooth_w=action_smooth_w,
            action_delay_range=action_delay_range,
            seed=seed,
        )
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
    parser.add_argument("--tau-xy", type=float, nargs=2, default=list(DEFAULT_TAU_XY),
                        metavar=("MIN", "MAX"), help="水平 tau 采样范围 (s)")
    parser.add_argument("--tau-z", type=float, nargs=2, default=list(DEFAULT_TAU_Z),
                        metavar=("MIN", "MAX"), help="垂直 tau 采样范围 (s)")
    parser.add_argument("--low-alt-speed-w", type=float, default=2.0,
                        help="低空速度惩罚权重")
    parser.add_argument("--action-smooth-w", type=float, default=0.0,
                        help="动作平滑惩罚权重（抗振荡，建议 0.05~0.2）")
    parser.add_argument("--action-delay", type=int, nargs=2, default=[0, 0],
                        metavar=("MIN", "MAX"), help="控制延迟采样范围（步），建议 0 2")
    parser.add_argument("--init-from", type=str, default=None,
                        help="从已有模型热启动继续训练（.zip 路径）")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    env_kwargs = dict(tau_xy=tuple(args.tau_xy), tau_z=tuple(args.tau_z),
                      low_alt_speed_w=args.low_alt_speed_w,
                      action_smooth_w=args.action_smooth_w,
                      action_delay_range=tuple(args.action_delay))
    train_env = VecNormalize(DummyVecEnv([make_env(args.seed, **env_kwargs)]), norm_obs=True, norm_reward=True)
    eval_env = VecNormalize(DummyVecEnv([make_env(args.seed + 1000, **env_kwargs)]), norm_obs=True, norm_reward=False, training=False)

    if args.init_from:
        model = PPO.load(args.init_from, env=train_env, device=args.device,
                         tensorboard_log=args.log_dir)
        print(f"热启动: {args.init_from}")
    else:
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
