"""把训练好的 PPO 策略导出为 ONNX，供无 torch/sb3 的部署环境（如 docker 容器）推理。

导出的是确定性策略（MlpPolicy 的 mean action），与 SB3 的
predict(obs, deterministic=True) 等价。

用法：
    .venv/bin/python rl_vls/tools/export_onnx.py \
        checkpoints/ppo_landing_final.zip checkpoints/ppo_landing_final.onnx
"""
from __future__ import annotations

import os
import sys

# 允许直接以文件路径运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main(src: str, dst: str):
    import torch
    from stable_baselines3 import PPO

    model = PPO.load(src, device="cpu")

    # 只导出确定性动作路径（features -> latent_pi -> action_net），
    # 等价于 SB3 predict(obs, deterministic=True) 的高斯均值输出；
    # 直接导出 policy.forward 会带上分布采样逻辑，ONNX 导出会失败
    class DeterministicPolicy(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.features_extractor = policy.features_extractor
            self.mlp_extractor = policy.mlp_extractor
            self.action_net = policy.action_net

        def forward(self, obs):
            features = self.features_extractor(obs)
            latent_pi, _ = self.mlp_extractor(features)
            return self.action_net(latent_pi)

    policy = DeterministicPolicy(model.policy).eval()

    obs_shape = model.observation_space.shape
    dummy = torch.zeros((1, *obs_shape), dtype=torch.float32)
    torch.onnx.export(
        policy,
        dummy,
        dst,
        input_names=["obs"],
        output_names=["action"],
        opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )
    print(f"已导出 ONNX: obs{obs_shape} -> {dst}")

    # 若环境里有 onnxruntime，顺带校验与 SB3 推理一致
    try:
        import numpy as np
        import onnxruntime as ort

        rng = np.random.default_rng(0)
        obs = rng.normal(size=(4, *obs_shape)).astype(np.float32)
        ref, _ = model.predict(obs, deterministic=True)  # SB3 predict 会 clip 到动作空间
        out = ort.InferenceSession(dst).run(None, {"obs": obs})[0]
        out = np.clip(out, model.action_space.low, model.action_space.high)
        err = float(np.abs(ref - out).max())
        print(f"一致性校验: max|clip(onnx) - sb3| = {err:.2e}")
        assert err < 1e-4, "ONNX 与 SB3 输出不一致！"
    except ImportError:
        print("（未安装 onnxruntime，跳过一致性校验）")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
