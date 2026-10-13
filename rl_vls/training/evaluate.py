"""策略评估：交接成功率、水平偏差、水平/下降速度、稳定时间与指令振荡。

评估指标与部署判据一致（对齐—减速—稳定交接）；应固定初始条件、多种子重复，
不能只按平均 reward 选模型。

用法：
    python -m rl_vls.training.evaluate \
        --model checkpoints/ppo_landing_final.zip \
        --vecnormalize checkpoints/vecnormalize.pkl \
        --env-config checkpoints/env_config.json --episodes 100
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_vls.contract import CONTRACT
from rl_vls.envs.landing_env import LandingEnv
from rl_vls.training.train_ppo import NOISE_CFG

AXES = ("right", "forward", "down")


def _env_kwargs(path: str | None) -> dict:
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return dict(data.get("env", CONTRACT.env_kwargs()))
    return CONTRACT.env_kwargs()


def _reversals(series, deadband: float = 0.05) -> int:
    signs = [int(np.sign(v)) if abs(v) > deadband else 0 for v in series]
    signs = [s for s in signs if s != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vecnormalize", type=str, default=None,
                        help="vecnormalize.pkl 路径；不传则不做观测归一化")
    parser.add_argument("--env-config", type=str, default=None,
                        help="训练保存的 env_config.json；不传则用契约默认")
    parser.add_argument("--override", type=str, default=None,
                        help='覆盖环境参数的 JSON，如 \'{"action_delay_range": [20, 20]}\'，'
                             "用于鲁棒性扫描（延迟/尺度等）")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123,
                        help="评估种子；同一 seed 即固定初始条件，便于比较策略")
    args = parser.parse_args()

    env_kwargs = _env_kwargs(args.env_config)
    if args.override:
        env_kwargs.update(json.loads(args.override))
    env = DummyVecEnv([lambda: LandingEnv(noise_cfg=NOISE_CFG, seed=args.seed, **env_kwargs)])
    if args.vecnormalize:
        env = VecNormalize.load(args.vecnormalize, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(args.model, env=env)

    results = Counter()
    errs, hspeeds, vspeeds, holds = [], [], [], []
    diffs = np.zeros(3)
    reversals = np.zeros(3, dtype=int)
    n_pairs = 0

    for _ in range(args.episodes):
        obs = env.reset()
        actions, done = [], False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            actions.append(np.asarray(action[0], dtype=np.float64))
            obs, _, dones, infos = env.step(action)
            done = dones[0]
        info = infos[0]
        outcome = info.get("landing", "unknown")
        results[outcome] += 1
        if outcome in ("success", "crash"):
            errs.append(info["d_xy"])
            hspeeds.append(info["horiz_speed"])
            vspeeds.append(abs(info["v_down"]))
            holds.append(info["align_hold_s"])
        if len(actions) > 1:
            a = np.asarray(actions)
            diffs += np.sum(np.diff(a, axis=0) ** 2, axis=0)
            reversals += np.array([_reversals(a[:, i]) for i in range(3)])
            n_pairs += len(actions) - 1

    n = args.episodes
    print(f"\n=== 评估结果（{n} 回合，seed={args.seed}）===")
    for k, v in sorted(results.items()):
        print(f"  {k:8s}: {v:3d} ({100.0 * v / n:.1f}%)")
    if errs:
        print(f"  交接水平偏差: mean={np.mean(errs):.3f} m, max={np.max(errs):.3f} m")
        print(f"  交接水平速度: mean={np.mean(hspeeds):.3f} m/s, max={np.max(hspeeds):.3f} m/s")
        print(f"  交接下降速度: mean={np.mean(vspeeds):.3f} m/s, max={np.max(vspeeds):.3f} m/s")
        print(f"  交接前稳定保持: mean={np.mean(holds):.3f} s")
    if n_pairs:
        print("  动作差分 RMS: " + ", ".join(
            f"{ax}={np.sqrt(diffs[i] / n_pairs):.3f}" for i, ax in enumerate(AXES)))
    print("  指令变号次数: " + ", ".join(
        f"{ax}={reversals[i]}" for i, ax in enumerate(AXES)))


if __name__ == "__main__":
    main()
