"""视觉着陆 Gymnasium 环境（阶段 1：纯 Python 质点动力学，不依赖 ROS）。

坐标系约定
----------
世界系 W：原点为 H 靶标中心，X 右、Y 前、Z 上（高度为正），单位米。
相机光轴系 C（机体水平、相机正对下方时）：z 前（指向地面）、x 右、y 下，
即 p_cam = [-X_w, -Y_w, h]，与 /h_marker/position (geometry_msgs/Point) 的定义一致。

动作：期望速度指令 [v_right, v_forward, v_down]（归一化到 [-1, 1] 后按限幅缩放），
对应 MAVROS /mavros/setpoint_velocity/cmd_vel_unstamped 的速度控制语义。

任务：把飞行器对齐到靶标上方、减速，并在交接高度（默认 2.5 m，交给 QLAND）时
满足水平偏差与水平速度判据且连续保持一段时间视为成功。终点不再是触地，
因此低空减速/对准的奖励区间一定会被执行，与真实部署流程一致。

训练与部署的一致性由 rl_vls.contract 的 Contract 保证：速度映射、控制频率、
指令整形（加速度限制 + 接近段限速）、控制延迟、视觉尺度、下降门控与交接判据
在两侧逐项对应。
动力学：一阶惯性环节模拟 ArduPilot guided 模式的速度跟踪响应，
时间常数 tau 每回合随机采样（domain randomization，覆盖真机响应不确定性）。
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_vls.contract import CONTRACT
from rl_vls.vision.noise_model import VisionNoiseModel


class LandingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float | None = None,
        tau_xy_range: tuple = CONTRACT.tau_xy_range,
        tau_z_range: tuple = CONTRACT.tau_z_range,
        max_episode_time: float = CONTRACT.max_episode_time,
        v_xy_max: float = CONTRACT.v_xy_max,
        v_down_max: float = CONTRACT.v_down_max,
        v_up_max: float = CONTRACT.v_up_max,
        cmd_filter_alpha: float = CONTRACT.cmd_filter_alpha,
        cmd_rate_max: float = CONTRACT.cmd_rate_max,
        approach_speed_gain: float = CONTRACT.approach_speed_gain,
        approach_speed_floor: float = CONTRACT.approach_speed_floor,
        fov_half_deg: float = 50.0,
        h_init_range: tuple = CONTRACT.h_init_range,
        xy_init_range: float = CONTRACT.xy_init_range,
        handoff_altitude: float = CONTRACT.handoff_altitude,
        handoff_radius: float = CONTRACT.handoff_radius,
        handoff_speed_xy: float = CONTRACT.handoff_speed_xy,
        handoff_hold_s: float = CONTRACT.handoff_hold_s,
        descent_gate_radius: float = CONTRACT.descent_gate_radius,
        descent_gate_speed: float = CONTRACT.descent_gate_speed,
        visual_scale_range: tuple = CONTRACT.visual_scale_range,
        w_pos: float = CONTRACT.w_pos,
        pos_cap: float = CONTRACT.pos_cap,
        w_vxy: float = CONTRACT.w_vxy,
        w_desc: float = CONTRACT.w_desc,
        w_descent_misalign: float = CONTRACT.w_descent_misalign,
        low_alt_h: float = CONTRACT.low_alt_h,
        low_alt_speed_w: float = CONTRACT.low_alt_speed_w,
        action_smooth_w: float = CONTRACT.action_smooth_w,
        success_bonus: float = CONTRACT.success_bonus,
        fail_base: float = CONTRACT.fail_base,
        fail_v: float = CONTRACT.fail_v,
        fail_xy: float = CONTRACT.fail_xy,
        timeout_penalty: float = CONTRACT.timeout_penalty,
        action_delay_range: tuple = CONTRACT.action_delay_range,
        lost_timeout: float = 2.0,
        noise_cfg: dict | None = None,
        seed: int | None = None,
    ):
        """
        所有默认值来自 rl_vls.contract.CONTRACT，与部署端逐项对应。

        Args:
            dt: 控制周期 (s)，默认取契约 1/control_rate_hz（0.05 s = 20 Hz）
            tau_xy_range/tau_z_range: 速度响应时间常数每回合随机采样范围 (s)
            cmd_filter_alpha: 指令一阶低通滤波系数，与部署端 cmd_filter_alpha 一致；
                1.0表示不滤波（默认，改由 cmd_rate_max 限制指令变化率）
            cmd_rate_max: 机体系速度指令最大变化率 (m/s²)，模拟飞控可达加速度上限
            approach_speed_gain/floor: 接近段限速 |v_xy| ≤ gain·d_xy + floor (1/s, m/s)
            handoff_altitude/radius/speed_xy/hold_s: QLAND 交接判据
            descent_gate_radius/speed: 未对齐或横向速度过大时抑制下降的门控
            visual_scale_range: 视觉尺度残差随机化范围（观测 p_cam = s * 真值）
            w_*: 过程奖励权重；success_bonus/fail_*: 终局奖惩
            action_delay_range: 控制延迟随机采样范围（步，含端点）
        """
        super().__init__()
        self.dt = CONTRACT.dt if dt is None else dt
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
        self.cmd_filter_alpha = cmd_filter_alpha
        self.cmd_rate_max = cmd_rate_max
        self.approach_speed_gain = approach_speed_gain
        self.approach_speed_floor = approach_speed_floor
        self.fov_half = np.deg2rad(fov_half_deg)
        self.h_init_range = h_init_range
        self.xy_init_range = xy_init_range
        self.handoff_altitude = handoff_altitude
        self.handoff_radius = handoff_radius
        self.handoff_speed_xy = handoff_speed_xy
        self.handoff_hold_s = handoff_hold_s
        self.descent_gate_radius = descent_gate_radius
        self.descent_gate_speed = descent_gate_speed
        self.visual_scale_range = visual_scale_range
        self.w_pos = w_pos
        self.pos_cap = pos_cap
        self.w_vxy = w_vxy
        self.w_desc = w_desc
        self.w_descent_misalign = w_descent_misalign
        self.low_alt_h = low_alt_h
        self.low_alt_speed_w = low_alt_speed_w
        self.action_smooth_w = action_smooth_w
        self.success_bonus = success_bonus
        self.fail_base = fail_base
        self.fail_v = fail_v
        self.fail_xy = fail_xy
        self.timeout_penalty = timeout_penalty
        self.action_delay_range = action_delay_range
        self.lost_timeout = lost_timeout

        self.noise = VisionNoiseModel(**(noise_cfg or {}))
        self._rng = np.random.default_rng(seed)

        # 动作：[v_right, v_forward, v_down]，归一化 [-1, 1]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        # 观测：p_cam(3, 带噪) + vel_w(3) + prev_action(3) + stale(1)
        # 取有限边界以兼容 VecNormalize 与 gymnasium 检查
        scale_bound = max(abs(s) for s in visual_scale_range)
        p_cam_bound = (np.tan(self.fov_half) * h_init_range[1] * 2.0 + 5.0) * scale_bound
        h_bound = (h_init_range[1] + v_up_max * max_episode_time + 5.0) * scale_bound
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
        self._v_filt = np.zeros(3, dtype=np.float64)  # 指令低通滤波器状态
        self._v_shaped = np.zeros(3, dtype=np.float64)  # 指令整形（速率限制）状态
        self._align_hold_t = 0.0                      # 对齐+低速连续保持时间
        # 视觉尺度残差：观测 p_cam = visual_scale * 真值（部署端已补偿主尺度）
        self.visual_scale = float(self._rng.uniform(*self.visual_scale_range))
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

    def _observed_p_cam(self) -> np.ndarray:
        """视觉管线测量真值（含尺度残差）：p_vis = visual_scale * p_true。"""
        return self.visual_scale * self._true_p_cam()

    def _get_obs(self) -> np.ndarray:
        meas, valid = self.noise.measure(self._observed_p_cam(), self._rng)
        in_fov = self._marker_in_fov(self._true_p_cam())
        if valid and in_fov:
            self.p_cam_meas = meas
            self.lost_time = 0.0
            stale = 0.0
        else:
            self.lost_time += self.dt
            stale = 1.0
            if self.p_cam_meas is None:
                self.p_cam_meas = self._observed_p_cam().astype(np.float32)
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

    def _descent_gate(self, d_xy: float, horiz_speed: float) -> float:
        """下降门控：未对准或横向速度过大时平滑抑制下降速率，上升不受限。

        采用平方（软）因子：轻微偏差/速度几乎不影响下降，接近阈值才迅速收缩。
        早期版本用线性因子过于激进，会把正常横向摆动也压成 0.3~0.5 的下降比例，
        导致高空无法在限定时间内完成下降（实测大量超时）。
        与部署节点 landing_rl_node 的 RL 下降门控逐项一致。
        """
        if self.descent_gate_radius <= 0.0:
            return 1.0
        ratio = min(d_xy / self.descent_gate_radius, 1.0)
        gate = 1.0 - ratio * ratio
        if self.descent_gate_speed > 0.0:
            ratio_v = min(horiz_speed / self.descent_gate_speed, 1.0)
            gate *= 1.0 - ratio_v * ratio_v
        return float(np.clip(gate, 0.0, 1.0))

    def _shape_command(self, v_cmd: np.ndarray, d_xy: float) -> np.ndarray:
        """指令整形：接近段限速 + 加速度（斜率）限制。

        与部署节点 ``landing_rl_node._shape_command`` 逐项一致（参数取契约
        ``cmd_rate_max`` / ``approach_speed_gain`` / ``approach_speed_floor``）。

        动机（2026-10-13 真机低频辨识）：飞控对速度目标可达的加速度约 1~2.5 m/s²。
        策略输出的 ±1 m/s、~2 s 周期继电方波等效要求 4~10 m/s²，被削成慢三角波
        （幅值只剩 0.40~0.53、多出 ~90° 相位 → 拟合成 +0.5 s 假延迟），闭环因此
        变成 0.5 Hz 极限环，越过策略 0.65 s 的可控悬崖。把指令压回飞控线性区后，
        真机等效延迟回落到 0.2~0.3 s（同一机体 MPC 通路的实测值）。

        1) 接近段速度剖面：水平速度上限随水平偏差收缩，抑制“到交接高度还很快”；
        2) 加速度限制：每步指令变化不超过 ``cmd_rate_max · dt``。
        """
        cap_xy = self.approach_speed_gain * d_xy + self.approach_speed_floor
        hspeed = float(np.hypot(v_cmd[0], v_cmd[1]))
        if 0.0 < cap_xy < hspeed:
            scale = cap_xy / hspeed
            v_cmd = np.array([v_cmd[0] * scale, v_cmd[1] * scale, v_cmd[2]])
        max_step = self.cmd_rate_max * self.dt
        self._v_shaped = self._v_shaped + np.clip(v_cmd - self._v_shaped, -max_step, max_step)
        return self._v_shaped

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
        # 指令整形（与部署端逐项一致）：接近段限速 + 加速度限制。
        # 部署端在“发布前”整形，故这里也放在延迟缓冲之前。
        shaped = self._shape_command(
            v_cmd_new, float(np.hypot(self.pos[0], self.pos[1])))
        # 指令低通滤波（默认关闭 cmd_filter_alpha=1.0）；顺序与部署端一致：整形→滤波
        if self.cmd_filter_alpha < 1.0:
            self._v_filt = (self.cmd_filter_alpha * shaped
                            + (1.0 - self.cmd_filter_alpha) * self._v_filt)
            shaped = self._v_filt.copy()
        # 控制延迟：指令进入缓冲，延迟 _action_delay 步后才作用于动力学
        self._act_buf.append(shaped)
        v_cmd = self._act_buf.pop(0)
        v_act = np.asarray(v_cmd, dtype=np.float64).copy()
        # 视觉不可用（上一帧无效/离轴超限）时，按部署端行为输出零速度悬停。
        # 不提前结束回合：否则“故意飞出视场以提前中止”会成为损回报的局部最优。
        if self.lost_time > 0.0:
            v_act = np.zeros(3, dtype=np.float64)
            self._v_filt = np.zeros(3, dtype=np.float64)
            self._v_shaped = np.zeros(3, dtype=np.float64)
        # 下降门控：用当前状态判断是否允许下降（与部署端一致）
        d_xy_now = float(np.hypot(self.pos[0], self.pos[1]))
        hspeed_now = float(np.hypot(self.vel[0], self.vel[1]))
        if v_act[2] > 0.0:
            v_act[2] *= self._descent_gate(d_xy_now, hspeed_now)
        # 动作平滑惩罚（在更新 prev_action 前计算差分）
        smooth_pen = self.action_smooth_w * float(np.sum((action - self.prev_action) ** 2))
        # 一阶惯性速度跟踪（水平/垂直各自的时间常数）+ 位置积分
        self.vel += (v_act - self.vel) * (self.dt / self.tau_vec)
        self.pos[0] += self.vel[0] * self.dt
        self.pos[1] += self.vel[1] * self.dt
        self.pos[2] -= self.vel[2] * self.dt  # v_down 为正时高度下降
        self.t += self.dt
        self.prev_action = action

        obs = self._get_obs()
        h = self.pos[2]
        d_xy = float(np.hypot(self.pos[0], self.pos[1]))
        hspeed = float(np.hypot(self.vel[0], self.vel[1]))
        v_down = float(self.vel[2])
        speed = float(np.linalg.norm(self.vel))

        # 过程奖励：只保留有界项，不设每步高度代价。
        # 无界的高度时间惩罚会使策略为了提前中止回合而故意飞出视场。
        # 位置惩罚限幅，避免漂远后奖励发散、破坏 VecNormalize 的尺度。
        reward = -self.w_pos * min(d_xy, self.pos_cap) * self.dt
        # 下降进程奖励：仅在接近对准时奖励下降（与门控一致），提供到达交接的正梯度
        if v_down > 0.0 and self.handoff_radius > 0.0:
            misalign = float(np.clip((d_xy - self.handoff_radius) / self.handoff_radius, 0.0, 1.0))
            reward += self.w_desc * v_down * (1.0 - misalign) * self.dt
            reward -= self.w_descent_misalign * v_down * misalign * self.dt
        # 全程水平速度代价（原为 0.1，偏弱，无法抑制接近段横向速度）
        reward -= self.w_vxy * hspeed * self.dt
        # 动作平滑惩罚：抑制 bang-bang 式振荡输出（不按 dt 缩放，逐步计）
        reward -= smooth_pen
        # 接近段速度惩罚：低于 low_alt_h 后随高度降低线性增大，压低交接速度余量
        if h < self.low_alt_h:
            reward -= self.low_alt_speed_w * speed * (1.0 - h / self.low_alt_h) * self.dt

        # 对齐 + 低速保持计时（交接判据，与部署 QLAND 稳定性一致）
        if d_xy <= self.handoff_radius and hspeed <= self.handoff_speed_xy:
            self._align_hold_t += self.dt
        else:
            self._align_hold_t = 0.0

        terminated = False
        truncated = False
        info = {"d_xy": d_xy, "h": h, "speed": speed, "horiz_speed": hspeed,
                "v_down": v_down, "align_hold_s": self._align_hold_t}

        if h <= self.handoff_altitude:  # 到达 QLAND 交接高度
            terminated = True
            success = self._align_hold_t >= self.handoff_hold_s
            if success:
                reward += self.success_bonus
                info["landing"] = "success"
            else:
                reward -= (self.fail_base + self.fail_xy * d_xy
                           + self.fail_v * hspeed + self.fail_v * abs(v_down))
                info["landing"] = "crash"
        elif self.t >= self.max_episode_time:  # 超时（含丢失目标）：截断，非真实终态
            truncated = True
            reward -= self.timeout_penalty
            info["landing"] = "lost" if self.lost_time >= self.lost_timeout else "timeout"

        return obs, reward, terminated, truncated, info


gym.register(id="Landing-v0", entry_point="rl_vls.envs.landing_env:LandingEnv")
