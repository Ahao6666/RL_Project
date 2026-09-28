"""策略评估：统计着陆成功率、水平偏差、触地速度。

用法：
    python -m rl_vls.training.evaluate --model checkpoints/ppo_landing_final.zip --episodes 50
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_vls.envs.landing_env import LandingEnv
from rl_vls.training.train_ppo import NOISE_CFG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vecnormalize", type=str, default=None,
                        help="vecnormalize.pkl 路径；不传则不做观测归一化")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    env = DummyVecEnv([lambda: LandingEnv(noise_cfg=NOISE_CFG, seed=args.seed)])
    if args.vecnormalize:
        env = VecNormalize.load(args.vecnormalize, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(args.model, env=env)

    results = Counter()
    land_errs, land_speeds = [], []

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            done = dones[0]
        info = infos[0]
        outcome = info.get("landing", "unknown")
        results[outcome] += 1
        if outcome in ("success", "crash"):
            land_errs.append(info["d_xy"])
            land_speeds.append(info["speed"])

    n = args.episodes
    print(f"\n=== 评估结果（{n} 回合）===")
    for k, v in results.items():
        print(f"  {k:8s}: {v:3d} ({100.0 * v / n:.1f}%)")
    if land_errs:
        print(f"  着陆水平偏差: mean={np.mean(land_errs):.3f} m, max={np.max(land_errs):.3f} m")
        print(f"  触地速度:     mean={np.mean(land_speeds):.3f} m/s, max={np.max(land_speeds):.3f} m/s")


if __name__ == "__main__":
    main()
