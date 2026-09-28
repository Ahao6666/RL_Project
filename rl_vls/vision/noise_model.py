"""视觉检测误差模型：用真值 + 误差模拟 /h_marker/position 的发布结果。

误差特性参考实际视觉检测：
- 噪声标准差随目标距离（深度 z_c）和离轴距离增大而增大（像素误差放大）；
- 支持随机丢帧（valid=False），模拟检测失败/目标被遮挡；
- 可选像素量化误差。

实机验证时该模块被真实视觉估计替换，策略网络不需要改动。
"""
from __future__ import annotations

import numpy as np


class VisionNoiseModel:
    def __init__(
        self,
        sigma_dist: float = 0.02,
        sigma_offaxis: float = 0.05,
        sigma_floor: float = 0.005,
        dropout_prob: float = 0.02,
    ):
        """
        Args:
            sigma_dist: 深度方向比例噪声系数，sigma 贡献 = sigma_dist * z_c
            sigma_offaxis: 离轴比例噪声系数，sigma 贡献 = sigma_offaxis * ||p_xy||
            sigma_floor: 噪声下限 (m)，避免零距离时无噪声
            dropout_prob: 每帧丢帧概率 [0, 1)
        """
        self.sigma_dist = sigma_dist
        self.sigma_offaxis = sigma_offaxis
        self.sigma_floor = sigma_floor
        self.dropout_prob = dropout_prob

    def measure(self, p_cam_true: np.ndarray, rng: np.random.Generator):
        """根据真值生成带噪测量。

        Args:
            p_cam_true: H 标中心在相机光轴系下的真值 [x右, y下, z前] (m)
            rng: 随机数生成器（由 env 管理以保证可复现）

        Returns:
            (p_cam_meas, valid): 带噪测量值与有效标志。valid=False 表示本帧丢帧，
            调用方应沿用上一帧测量。
        """
        p = np.asarray(p_cam_true, dtype=np.float64)
        off_axis = float(np.hypot(p[0], p[1]))
        sigma = self.sigma_floor + self.sigma_dist * abs(p[2]) + self.sigma_offaxis * off_axis
        meas = p + rng.normal(0.0, sigma, size=3)
        valid = bool(rng.random() >= self.dropout_prob)
        return meas.astype(np.float32), valid
