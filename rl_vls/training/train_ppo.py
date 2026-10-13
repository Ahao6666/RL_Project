"""PPO 训练入口（阶段 1）。

任务：对齐—减速—稳定交接（终点为 QLAND 交接高度），与部署端逐项对应。
所有执行器/任务参数来自 rl_vls.contract.CONTRACT，可用命令行覆盖。

用法：
    python -m rl_vls.training.train_ppo --timesteps 1000000
    tensorboard --logdir runs/

训练完成后会在 save-dir 写入 env_config.json（含契约与 PPO 超参），
部署时用 -p contract_path:=checkpoints/env_config.json 加载，避免参数漂移。
"""
from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_vls.contract import CONTRACT
from rl_vls.envs.landing_env import LandingEnv
from rl_vls.training.curriculum import CurriculumCallback

# 视觉噪声参数：训练时可逐步加大（课程学习），或做 domain randomization
NOISE_CFG = dict(sigma_dist=0.02, sigma_offaxis=0.05, sigma_floor=0.005, dropout_prob=0.02)


def make_env(seed: int = 0, env_kwargs: dict | None = None):
    """用契约（或显式覆盖）构造环境；训练与部署参数由此统一。"""
    kwargs = CONTRACT.env_kwargs() if env_kwargs is None else dict(env_kwargs)

    def _init():
        return Monitor(LandingEnv(noise_cfg=NOISE_CFG, seed=seed, **kwargs))

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="runs/ppo_landing")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cpu",
                        help="MlpPolicy 建议 cpu；SB3 默认 auto 会占用 GPU 但收益很小")
    parser.add_argument("--gamma", type=float, default=CONTRACT.gamma,
                        help="折扣因子；20 Hz 下 0.995≈20 s 有效视野，覆盖完整接近段")
    parser.add_argument("--ent-coef", type=float, default=0.005,
                        help="策略熵系数；>0 可缓解策略早期塔缩（如收敛到恒定动作/提前中止）")
    parser.add_argument("--tau-xy", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"), help="水平 tau 采样范围 (s)，默认取契约值")
    parser.add_argument("--tau-z", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"), help="垂直 tau 采样范围 (s)，默认取契约值")
    parser.add_argument("--low-alt-speed-w", type=float, default=None,
                        help="接近段速度惩罚权重，默认取契约值")
    parser.add_argument("--action-smooth-w", type=float, default=None,
                        help="动作平滑惩罚权重（抗振荡，默认取契约值）")
    parser.add_argument("--action-delay", type=int, nargs=2, default=None,
                        metavar=("MIN", "MAX"), help="控制延迟采样范围（步），默认取契约值")
    parser.add_argument("--cmd-filter-alpha", type=float, default=None,
                        help="指令低通滤波系数，默认取契约值（与部署一致）")
    parser.add_argument("--visual-scale", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"), help="视觉尺度残差随机化范围，默认取契约值")
    parser.add_argument("--init-from", type=str, default=None,
                        help="从已有模型热启动继续训练（.zip 路径）")
    parser.add_argument("--init-from-vecnormalize", type=str, default=None,
                        help="热启动时加载对应的 vecnormalize.pkl（必须与模型成套）")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="关闭课程学习（默认开启：按评估成功率从易到难推进任务难度）")
    parser.add_argument("--eval-freq", type=int, default=20_000,
                        help="评估间隔（步）；一次评估同时用于推进课程与记录指标")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="每次评估回合数；评估难度跟随当前训练难度")
    parser.add_argument("--hard-eval-every", type=int, default=5,
                        help="每 N 次评估额外在 HARD 难度上评一次（hard/* 标量，用于选 best_model）")
    parser.add_argument("--advance-at", type=float, default=0.4,
                        help="当前难度下评估成功率 ≥ 该值就推进一档难度（默认 0.4）")
    parser.add_argument("--floor-frac", type=float, default=0.5,
                        help="难度时间下限升到 HARD 所用的预算比例（默认 0.5：半程到达 HARD）")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    env_kwargs = CONTRACT.env_kwargs()
    if args.tau_xy is not None:
        env_kwargs["tau_xy_range"] = tuple(args.tau_xy)
    if args.tau_z is not None:
        env_kwargs["tau_z_range"] = tuple(args.tau_z)
    if args.low_alt_speed_w is not None:
        env_kwargs["low_alt_speed_w"] = args.low_alt_speed_w
    if args.action_smooth_w is not None:
        env_kwargs["action_smooth_w"] = args.action_smooth_w
    if args.action_delay is not None:
        env_kwargs["action_delay_range"] = tuple(args.action_delay)
    if args.cmd_filter_alpha is not None:
        env_kwargs["cmd_filter_alpha"] = args.cmd_filter_alpha
    if args.visual_scale is not None:
        env_kwargs["visual_scale_range"] = tuple(args.visual_scale)

    base_env = DummyVecEnv([make_env(args.seed, env_kwargs)])
    if args.init_from_vecnormalize:
        train_env = VecNormalize.load(args.init_from_vecnormalize, base_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(base_env, norm_obs=True, norm_reward=True)
    eval_env = VecNormalize(DummyVecEnv([make_env(args.seed + 1000, env_kwargs)]),
                            norm_obs=True, norm_reward=False, training=False)

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
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            verbose=1,
            seed=args.seed,
            tensorboard_log=args.log_dir,
            device=args.device,
            policy_kwargs=dict(net_arch=[128, 128]),
        )

    callbacks = [
        # 课程 + 评估一体化：一次评估同时推进课程、记录 eval/* 与 hard/* 指标，
        # 并保存最优模型（best_model 与 VecNormalize 成套）。
        # 评估难度跟随当前训练难度（推进判据）；HARD 难度每 hard_eval_every 次评一次。
        CurriculumCallback(
            eval_env, args.timesteps, eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            advance_at=args.advance_at,
            min_progress_frac=args.floor_frac,
            hard_eval_every=args.hard_eval_every,
            best_model_save_path=os.path.join(args.save_dir, "best"),
            vecnormalize=train_env,
            enabled=not args.no_curriculum),
        CheckpointCallback(
            save_freq=50_000,
            save_path=args.save_dir,
            name_prefix="ppo_landing",
            # 中间检查点必须带归一化统计量，否则无法单独评估/部署
            save_vecnormalize=True,
        ),
    ]

    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save(os.path.join(args.save_dir, "ppo_landing_final"))
    train_env.save(os.path.join(args.save_dir, "vecnormalize.pkl"))
    CONTRACT.save(
        os.path.join(args.save_dir, "env_config.json"),
        ppo=dict(timesteps=args.timesteps, gamma=args.gamma, ent_coef=args.ent_coef,
                 curriculum=not args.no_curriculum,
                 eval_freq=args.eval_freq, eval_episodes=args.eval_episodes,
                 advance_at=args.advance_at, floor_frac=args.floor_frac,
                 hard_eval_every=args.hard_eval_every,
                 n_steps=2048, batch_size=256, learning_rate=3e-4,
                 gae_lambda=0.95, clip_range=0.2, net_arch=[128, 128]))
    print(f"模型已保存到 {args.save_dir}/ppo_landing_final.zip")
    print(f"契约与归一化已保存到 {args.save_dir}/env_config.json 与 vecnormalize.pkl")


if __name__ == "__main__":
    main()
