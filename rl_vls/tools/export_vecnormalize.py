"""把 VecNormalize 的观测归一化统计量导出为 .npz（跨 numpy 版本/框架可用）。

用法：
    .venv/bin/python rl_vls/tools/export_vecnormalize.py \
        checkpoints/vecnormalize.pkl checkpoints/obs_norm.npz
"""
from __future__ import annotations

import os
import sys

import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

# 允许直接以文件路径运行（python rl_vls/tools/export_vecnormalize.py ...）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main(src: str, dst: str):
    # VecNormalize.load 需要一个占位 vec env；统计量来自 pickle 本身
    from rl_vls.envs.landing_env import LandingEnv
    vn = VecNormalize.load(src, DummyVecEnv([lambda: LandingEnv(seed=0)]))
    np.savez(
        dst,
        mean=np.asarray(vn.obs_rms.mean, dtype=np.float32),
        var=np.asarray(vn.obs_rms.var, dtype=np.float32),
        epsilon=np.float32(vn.epsilon),
        clip_obs=np.float32(vn.clip_obs), 
    )
    print(f"已导出 obs_norm: mean shape {vn.obs_rms.mean.shape} -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
