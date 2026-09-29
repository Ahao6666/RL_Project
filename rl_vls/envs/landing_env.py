"""视觉着陆 Gymnasium 环境（阶段 1：纯 Python 质点动力学，不依赖 ROS）。

坐标系约定
----------
世界系 W：原点为 H 靶标中心，X 右、Y 前、Z 上（高度为正），单位米。
相机光轴系 C（机体水平、相机正对下方时）：z 前（指向地面）、x 右、y 下，
即 p_cam = [-X_w, -Y_w, h]，与 /h_marker/position (geometry_msgs/Point) 的定义一致。

动作：期望速度指令 [v_right, v_forward, v_down]（归一化到 [-1, 1] 后按限幅缩放），
对应 MAVROS /mavros/setpoint_velocity/cmd_vel_unstamped 的 NED 速度控制语义。
动力学：一阶惯性环节模拟 ArduPilot guided 模式的速度跟踪响应，
时间常数 tau 每回合随机采样（domain randomization，覆盖真机响应不确定性）。
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_vls.vision.noise_model import VisionNoiseModel


class LandingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.05,
        tau_xy_range: tuple = (0.5, 2.0),
        tau_z_range: tuple = (0.4, 1.5),
        max_episode_time: float = 40.0,
        v_xy_max: float = 2.0,
        v_down_max: float = 1.5,
        v_up_max: float = 0.8,
        fov_half_deg: float = 50.0,
        h_init_range: tuple = (3.0, 20.0),
        xy_init_range: float = 5.0,
        land_h: float = 0.15,
        land_r_max: float = 0.3,
        land_v_max: float = 0.5,
        low_alt_h: float = 2.0,
        low_alt_speed_w: float = 1.0,
        action_smooth_w: float = 0.0,
        action_delay_range: tuple = (0, 0),
        lost_timeout: float = 2.0,
        noise_cfg: dict | None = None,
        seed: int | None = None,
    ):
        """
        Args:
            tau_xy_range: 水平速度响应时间常数每回合随机采样范围 (s)。
                SITL 实测约 1.3s，默认范围取实测值两侧留余量；
                传入 (t, t) 即固定值（用于压力测试）
            tau_z_range: 垂直速度响应时间常数范围 (s)，SITL 实测约 0.95s
            low_alt_h: 低空速度惩罚生效的高度阈值 (m)
            low_alt_speed_w: 低空速度惩罚权重，随高度降低线性增大
            action_smooth_w: 动作平滑惩罚权重，惩罚 ||a_t - a_{t-1}||^2，
                抑制 bang-bang 式振荡输出
            action_delay_range: 控制延迟随机采样范围（步数，含端点），
                模拟视觉+通信链路延迟；实测链路约 50~150ms（1~3 步）
        """
        super().__init__()
        self.dt = dt
        self.tau_xy_range = tau_xy_range
        self.tau_z_range = tau_z_range
        # 占位值，reset 时随机采样（[tau_x, tau_y, tau_z]）
        self.tau_vec = np.array(
            [np.mean(tau_xy_range), np.mean(tau_xy_range), np.mean(tau_z_range)]
        )
        self.max_episode_time = max_episode_time
        self.v_xy_max = v_xy_max
        self.v_down_max = v_down_max
        self.v_up_max = v_up_max
        self.fov_half = np.deg2rad(fov_half_deg)
        self.h_init_range = h_init_range
        self.xy_init_range = xy_init_range
        self.land_h = land_h
        self.land_r_max = land_r_max
        self.land_v_max = land_v_max
        self.low_alt_h = low_alt_h
        self.low_alt_speed_w = low_alt_speed_w
        self.action_smooth_w = action_smooth_w
        self.action_delay_range = action_delay_range
        self.lost_timeout = lost_timeout

        self.noise = VisionNoiseModel(**(noise_cfg or {}))
        self._rng = np.random.default_rng(seed)

        # 动作：[v_right, v_forward, v_down]，归一化 [-1, 1]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        # 观测：p_cam(3, 带噪) + vel_w(3) + prev_action(3) + stale(1)
        # 取有限边界以兼容 VecNormalize 与 gymnasium 检查
        p_cam_bound = np.tan(self.fov_half) * h_init_range[1] * 2.0 + 5.0
        h_bound = h_init_range[1] + v_up_max * max_episode_time + 5.0
        high = np.array(
            [p_cam_bound, p_cam_bound, h_bound,
             v_xy_max, v_xy_max, max(v_down_max, v_up_max),
             1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._reset_state()

    # ------------------------------------------------------------------ #
    def _reset_state(self):
        self.pos = np.zeros(3, dtype=np.float64)   # [X右, Y前, h上]
        self.vel = np.zeros(3, dtype=np.float64)   # [v_x, v_y, v_down]
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.t = 0.0
        self.lost_time = 0.0
        self.p_cam_meas = None
        # 控制延迟缓冲（reset 时按 action_delay_range 采样延迟步数）
        lo, hi = self.action_delay_range
        self._action_delay = int(self._rng.integers(lo, hi + 1))
        self._act_buf = [np.zeros(3)] * self._action_delay

    def _marker_in_fov(self, p_cam: np.ndarray) -> bool:
        off_axis = np.arctan2(np.hypot(p_cam[0], p_cam[1]), max(p_cam[2], 1e-6))
        return bool(off_axis < self.fov_half)

    def _true_p_cam(self) -> np.ndarray:
        """真值 -> 相机光轴系（水平假设）：p_cam = [-X, -Y, h]"""
        return np.array([-self.pos[0], -self.pos[1], self.pos[2]], dtype=np.float64)

    def _get_obs(self) -> np.ndarray:
        meas, valid = self.noise.measure(self._true_p_cam(), self._rng)
        in_fov = self._marker_in_fov(self._true_p_cam())
        if valid and in_fov:
            self.p_cam_meas = meas
            self.lost_time = 0.0
            stale = 0.0
        else:
            self.lost_time += self.dt
            stale = 1.0
            if self.p_cam_meas is None:
                self.p_cam_meas = self._true_p_cam().astype(np.float32)
        return np.concatenate(
            [self.p_cam_meas, self.vel, self.prev_action, [stale]]
        ).astype(np.float32)

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        # 每回合随机化动力学：水平/垂直独立采样（SITL 实测两者不同）
        tau_xy = self._rng.uniform(*self.tau_xy_range)
        tau_z = self._rng.uniform(*self.tau_z_range)
        self.tau_vec = np.array([tau_xy, tau_xy, tau_z])
        h = self._rng.uniform(*self.h_init_range)
        # 保证初始时目标在视场内
        while True:
            xy = self._rng.uniform(-self.xy_init_range, self.xy_init_range, size=2)
            self.pos = np.array([xy[0], xy[1], h])
            if self._marker_in_fov(self._true_p_cam()):
                break
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        v_cmd_new = np.array(
            [
                action[0] * self.v_xy_max,
                action[1] * self.v_xy_max,
                # v_down: [-1, 1] -> [-v_up_max, v_down_max]，正值下降
                (action[2] + 1.0) / 2.0 * (self.v_down_max + self.v_up_max) - self.v_up_max,
            ]
        )
        # 控制延迟：指令进入缓冲，延迟 _action_delay 步后才作用于动力学
        self._act_buf.append(v_cmd_new)
        v_cmd = self._act_buf.pop(0)
        # 动作平滑惩罚（在更新 prev_action 前计算差分）
        smooth_pen = self.action_smooth_w * float(np.sum((action - self.prev_action) ** 2))
        # 一阶惯性速度跟踪（水平/垂直各自的时间常数）+ 位置积分
        self.vel += (v_cmd - self.vel) * (self.dt / self.tau_vec)
        self.pos[0] += self.vel[0] * self.dt
        self.pos[1] += self.vel[1] * self.dt
        self.pos[2] -= self.vel[2] * self.dt  # v_down 为正时高度下降
        self.t += self.dt
        self.prev_action = action

        obs = self._get_obs()
        h = self.pos[2]
        d_xy = float(np.hypot(self.pos[0], self.pos[1]))
        speed = float(np.linalg.norm(self.vel))

        # 过程奖励：偏离中心、高度、水平速度的代价（按 dt 缩放与步长解耦）
        reward = -(1.0 * d_xy + 0.2 * h + 0.1 * float(np.hypot(*self.vel[:2]))) * self.dt
        # 动作平滑惩罚：抑制 bang-bang 式振荡输出（不按 dt 缩放，逐步计）
        reward -= smooth_pen
        # 低空速度惩罚：低于 low_alt_h 后随高度降低线性增大，压低触地速度余量
        if h < self.low_alt_h:
            reward -= self.low_alt_speed_w * speed * (1.0 - h / self.low_alt_h) * self.dt

        terminated = False
        truncated = False
        info = {"d_xy": d_xy, "h": h, "speed": speed}

        if h <= self.land_h:  # 触地判定
            terminated = True
            success = d_xy < self.land_r_max and speed < self.land_v_max
            reward += 10.0 if success else -(5.0 + 2.0 * speed + 2.0 * d_xy)
            info["landing"] = "success" if success else "crash"
        elif self.lost_time >= self.lost_timeout:  # 持续丢失目标
            terminated = True
            reward -= 5.0
            info["landing"] = "lost"
        elif self.t >= self.max_episode_time:
            truncated = True
            info["landing"] = "timeout"

        return obs, reward, terminated, truncated, info


gym.register(id="Landing-v0", entry_point="rl_vls.envs.landing_env:LandingEnv")
