"""阶段 2 桥接节点：加载阶段 1 训练的 PPO 策略，闭环控制 ArduPilot。

数据流：
    /mavros/local_position/velocity_body (TwistStamped, 实测为 FLU 系) ─┐
    /mavros/local_position/pose (PoseStamped, 取 yaw)                  ├─> 观测组装 -> PPO
    /h_marker/position (Point, 相机光轴系 z前/x右/y下)                  ─┘
    PPO 动作 [v_right, v_forward, v_down]
        -> 按 yaw 旋转到 ENU -> /mavros/setpoint_velocity/cmd_vel_unstamped

实测结论（2026-09-28 SITL 阶跃响应，rl_vls/tools/step_response.csv）：
    - cmd_vel_unstamped 为本地 ENU 系语义（世界固定系，不随机头旋转），
      linear.x=东, linear.y=北, linear.z=上（负值下降）
    - 速度跟踪一阶时间常数 tau_xy ≈ 1.3s, tau_z ≈ 0.95s（训练已覆盖）

假设：
    - 相机安装方向：图像 x 右 = 机体右，图像 y 下 = 机体前（下视相机常见装法）
    - 飞行中航向基本固定（策略在固定 yaw 下训练；GUIDED 速度控制时航向自保持）
    TODO: 若实机航向会主动转动，观测 p_cam 与动作需按 yaw 变化旋转补偿

安全逻辑：
    - 仅在 GUIDED 模式且已解锁时发布策略指令；
    - 视觉消息超时 -> 发布零速度（悬停）；
    - 所有指令做限幅。
"""
from __future__ import annotations

import math
import pickle
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist, TwistStamped
from mavros_msgs.msg import State
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class LandingRLNode(Node):
    def __init__(self):
        super().__init__("landing_rl_node")
        self.declare_parameter("model_path", "")
        self.declare_parameter("vecnormalize_path", "")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("marker_timeout", 1.0)   # 视觉超时 (s) -> 悬停
        # 注：新鲜度按消息 header 时间戳判定（含检测延迟），故需大于检测管线延迟
        self.declare_parameter("v_xy_max", 1.0)
        self.declare_parameter("v_down_max", 0.8)
        self.declare_parameter("v_up_max", 0.8)
        # 话题 y 轴镜像补偿：2026-09-29 yaw 标定（yaw_calib.csv）发现该系统
        # /h_marker/position 的 y 与标准相机光轴系反号（视觉管线的符号问题，
        # 根治应修视觉节点；此处做补偿）
        self.declare_parameter("marker_flip_y", True)
        self._flip_y = -1.0 if self.get_parameter("marker_flip_y").value else 1.0
        # 控制器选择："rl"（策略网络）/ "p"（比例控制 baseline）/ "hybrid"
        # （高空 RL 接近 + 低空 P 降落，hybrid_alt 为切换高度）/ "fixed"（固定机体系速度，
        # 仅用于测试输出坐标映射：恒定机体系速度的 ENU 输出应以 1:1 斜率随 yaw 旋转）
        self.declare_parameter("controller", "rl")
        self.controller = self.get_parameter("controller").value
        self.declare_parameter("hybrid_alt", 3.0)     # hybrid 模式 RL->P 切换高度 (m)
        # P 控制器参数
        self.declare_parameter("kp_xy", 0.3)          # 水平比例增益 (1/s)
        self.declare_parameter("p_descent_vel", 0.3)  # 对准后的下降速度 (m/s)
        self.declare_parameter("p_align_radius", 0.5)  # 水平偏差小于该值才下降 (m)
        self.declare_parameter("p_min_alt", 0.3)      # 低于该高度停止下降 (m)
        # fixed 模式参数：恒定的机体系速度指令 [右, 前, 下]
        self.declare_parameter("fixed_v_right", 0.5)
        self.declare_parameter("fixed_v_forward", 0.0)
        self.declare_parameter("fixed_v_down", 0.0)
        # 默认仅 GUIDED/QGUIDED 且解锁时发布；测试时可关闭（关闭后非引导模式也发布）
        self.declare_parameter("require_guided", True)
        # 姿态反旋转：消除俯仰/横滚引起的 p_cam 摆动，恢复训练时的水平系观测
        # （aggressive 机动时 10m 高度可产生 ~1.5m 表观偏移，是振荡主因之一）
        self.declare_parameter("attitude_derotate", True)
        self._derotate_on = self.get_parameter("attitude_derotate").value
        # 视觉离群值门控：单帧 p_cam 变化超过阈值视为误检丢弃。
        # 阈值 = marker_max_step + marker_max_step_per_m * 深度，
        # 因为姿态反旋转与视觉延迟的错位产生的表观偏移随深度线性放大
        self.declare_parameter("marker_max_step", 0.4)
        self.declare_parameter("marker_max_step_per_m", 0.15)
        self._outlier_count = 0
        # 指令低通滤波：v_filt = alpha*v_new + (1-alpha)*v_filt_prev，
        # 抑制 bang-bang 策略输出的高频振荡；1.0=不滤波，0.2~0.5 合适
        self.declare_parameter("cmd_filter_alpha", 0.25)
        self._v_filt = np.zeros(3)
        self.quat = (1.0, 0.0, 0.0, 0.0)  # 完整姿态（反旋转用）

        model_path = self.get_parameter("model_path").value
        self.policy = None
        self.onnx_sess = None
        if model_path.endswith(".onnx"):
            # ONNX 推理（推荐）：部署环境只需 onnxruntime，不依赖 torch/sb3/numpy2
            import onnxruntime as ort
            self.onnx_sess = ort.InferenceSession(model_path)
            self.get_logger().info(f"已加载 ONNX 策略: {model_path}")
        elif model_path:
            from stable_baselines3 import PPO
            self.policy = PPO.load(model_path)
            self.get_logger().info(f"已加载策略: {model_path}")
        else:
            if self.controller == "p":
                self.get_logger().info("P 比例控制模式，无需模型")
            else:
                self.get_logger().warn("未提供 model_path，节点只转发安全零指令")

        # 观测归一化统计量（与训练时 VecNormalize 一致），不加载则直接用原始观测。
        # 优先用 .npz（export_vecnormalize.py 导出，跨 numpy 版本可用）；
        # .pkl 仅在部署环境 numpy 版本与训练一致时可用。
        self.obs_norm = None
        vn_path = self.get_parameter("vecnormalize_path").value
        if vn_path:
            if vn_path.endswith(".npz"):
                d = np.load(vn_path)
                self.obs_norm = (d["mean"], d["var"], float(d["epsilon"]), float(d["clip_obs"]))
            else:
                with open(vn_path, "rb") as f:
                    vn = pickle.load(f)
                self.obs_norm = (vn.obs_rms.mean, vn.obs_rms.var, vn.epsilon, vn.clip_obs)
            self.get_logger().info(f"已加载观测归一化: {vn_path}")

        self.p_cam: np.ndarray | None = None
        self.marker_stamp_ns: int | None = None   # 视觉帧时间戳（消息 header）
        self.vel_flu = np.zeros(3)      # 机体系 FLU: x前 y左 z上（阶跃数据实测，见 H2 假设验证）
        self.yaw = 0.0                  # ENU 航向（弧度，东为 0 逆时针为正）
        # 姿态历史缓冲：反旋转时按视觉帧时间戳取同时刻姿态，消除延迟错位
        self.pose_hist: deque = deque(maxlen=400)
        self.mav_state = State()
        self.prev_action = np.zeros(3, dtype=np.float32)
        self._last_dbg_ns = 0

        self.create_subscription(PointStamped, "/h_marker/position", self._marker_cb, 10)
        # MAVROS 遥测话题为 best_effort QoS，订阅方必须匹配
        self.create_subscription(TwistStamped, "/mavros/local_position/velocity_body",
                                 self._vel_cb, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "/mavros/local_position/pose",
                                 self._pose_cb, qos_profile_sensor_data)
        self.create_subscription(State, "/mavros/state", self._state_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, "/mavros/setpoint_velocity/cmd_vel_unstamped", 10)

        period = 1.0 / self.get_parameter("control_rate_hz").value
        self.create_timer(period, self._control_cb)

    # -------------------------------------------------------------- #
    # FLU 机体系 -> 相机光轴系（图像x右=机体右, 图像y下=机体前, z沿光轴向下）
    _C0 = np.array([[0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0]])

    def _attitude_at(self, stamp_ns: int) -> tuple:
        """取姿态历史中与指定时间戳最接近的姿态样本。"""
        if not self.pose_hist:
            return self.quat
        return min(self.pose_hist, key=lambda s: abs(s[0] - stamp_ns))[1]

    def _derotate(self, pc: np.ndarray, quat: tuple) -> np.ndarray:
        """姿态反旋转：pc_level = (C0·T·C0ᵀ)·pc，T 为去除 yaw 后的俯仰/横滚旋转。"""
        w, x, y, z = quat
        R_wb = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        yaw_q = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        c, s = math.cos(yaw_q), math.sin(yaw_q)
        R_level = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        T = R_level.T @ R_wb
        return (self._C0 @ T @ self._C0.T) @ pc

    def _marker_cb(self, msg: PointStamped):
        stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds
        # 容错：时间戳为 0 或明显异常（视觉端未正确赋值）时回退到接收时刻
        now_ns = self.get_clock().now().nanoseconds
        if stamp_ns <= 0 or abs(now_ns - stamp_ns) > 10e9:
            stamp_ns = now_ns
        # y 轴镜像补偿（话题 y 与标准相机光轴系反号），z 沿光轴不受影响
        new_p = np.array([msg.point.x, self._flip_y * msg.point.y, msg.point.z],
                         dtype=np.float64)
        # 姿态反旋转：用视觉帧时间戳对应的姿态，消除俯仰/横滚扰动且无延迟错位
        if self._derotate_on:
            new_p = self._derotate(new_p, self._attitude_at(stamp_ns))
        # 离群值门控：单帧变化超过物理极限（机体速度+姿态摆动+深度相关延迟错位）
        # 视为误检，丢弃该帧（不更新时间戳，超时后按丢失悬停）
        if self.p_cam is not None:
            allowed = self.get_parameter("marker_max_step").value \
                + self.get_parameter("marker_max_step_per_m").value * abs(float(new_p[2]))
            if np.linalg.norm(new_p - self.p_cam) > allowed:
                self._outlier_count += 1
                return
        self.p_cam = new_p.astype(np.float32)
        self.marker_stamp_ns = stamp_ns

    def _vel_cb(self, msg: TwistStamped):
        v = msg.twist.linear
        self.vel_flu = np.array([v.x, v.y, v.z])

    def _pose_cb(self, msg: PoseStamped):
        q = msg.pose.orientation
        self.quat = (q.w, q.x, q.y, q.z)
        self.yaw = yaw_from_quaternion(q)
        stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds
        if stamp_ns == 0:
            stamp_ns = self.get_clock().now().nanoseconds
        self.pose_hist.append((stamp_ns, self.quat))

    def _state_cb(self, msg: State):
        self.mav_state = msg

    # -------------------------------------------------------------- #
    def _build_obs(self) -> np.ndarray:
        """组装与 LandingEnv 一致的 10 维观测：p_cam(3)+vel(3)+prev_action(3)+stale(1)。

        训练环境的速度观测定义在 [右, 前, 下] 方向；
        velocity_body 实测为 FLU [x前, y左, z上]，故 [右,前,下] = [-y, x, -z]。
        """
        vel_env = np.array([-self.vel_flu[1], self.vel_flu[0], -self.vel_flu[2]])
        obs = np.concatenate([self.p_cam, vel_env, self.prev_action, [0.0]]).astype(np.float32)
        if self.obs_norm is not None:
            mean, var, eps, clip = self.obs_norm
            obs = (obs - mean) / np.sqrt(var + eps)
            obs = np.clip(obs, -clip, clip).astype(np.float32)
        return obs

    def _predict(self, obs: np.ndarray) -> np.ndarray:
        if self.onnx_sess is not None:
            return self.onnx_sess.run(None, {"obs": obs[None]})[0][0]
        action, _ = self.policy.predict(obs, deterministic=True)
        return action

    def _control_cb(self):
        cmd = Twist()

        require_guided = self.get_parameter("require_guided").value
        armed_guided = (self.mav_state.armed
                        and self.mav_state.mode in ("GUIDED", "QGUIDED")) or not require_guided
        marker_fresh = (
            self.p_cam is not None
            and self.marker_stamp_ns is not None
            and (self.get_clock().now().nanoseconds - self.marker_stamp_ns) * 1e-9
            < self.get_parameter("marker_timeout").value
        )
        if self.controller == "fixed":
            marker_fresh = True   # fixed 模式不需要视觉输入

        policy_loaded = self.policy is not None or self.onnx_sess is not None
        active = self.controller in ("p", "fixed") or policy_loaded
        if armed_guided and marker_fresh and active:
            # 控制器模式解析：hybrid = 高空 RL + 低空 P（hybrid_alt 切换）
            mode = self.controller
            if mode == "hybrid":
                hybrid_alt = self.get_parameter("hybrid_alt").value
                mode = "rl" if float(self.p_cam[2]) > hybrid_alt else "p"
            if mode == "fixed":
                v_right = self.get_parameter("fixed_v_right").value
                v_forward = self.get_parameter("fixed_v_forward").value
                v_down = self.get_parameter("fixed_v_down").value
            elif mode == "p":
                # P 比例控制 baseline：p_cam 已补偿到 [右, 前(下视), 下] 约定，
                # 水平速度正比于偏差；对准后才下降，低空停降
                kp = self.get_parameter("kp_xy").value
                v_right = kp * float(self.p_cam[0])
                v_forward = kp * float(self.p_cam[1])
                horiz_err = math.hypot(float(self.p_cam[0]), float(self.p_cam[1]))
                alt = float(self.p_cam[2])
                if horiz_err < self.get_parameter("p_align_radius").value \
                        and alt > self.get_parameter("p_min_alt").value:
                    v_down = self.get_parameter("p_descent_vel").value
                else:
                    v_down = 0.0
                if self.get_clock().now().nanoseconds - self._last_dbg_ns > 2e9:
                    self._last_dbg_ns = self.get_clock().now().nanoseconds
                    self.get_logger().info(
                        f"[P] p_cam=({self.p_cam[0]:+.2f},{self.p_cam[1]:+.2f},{self.p_cam[2]:+.2f}) "
                        f"v=({v_right:+.2f},{v_forward:+.2f},{v_down:+.2f}) "
                        f"horiz_err={horiz_err:.2f} outliers={self._outlier_count}"
                    )
            else:
                obs = self._build_obs()
                action = np.clip(self._predict(obs), -1.0, 1.0)
                self.prev_action = action.astype(np.float32)
                v_xy_max = self.get_parameter("v_xy_max").value
                v_down_max = self.get_parameter("v_down_max").value
                v_up_max = self.get_parameter("v_up_max").value
                v_right = float(action[0]) * v_xy_max
                v_forward = float(action[1]) * v_xy_max
                v_down = (float(action[2]) + 1.0) / 2.0 * (v_down_max + v_up_max) - v_up_max
                # 调试：每 2s 打印一次策略的输入输出，用于排查观测组装问题
                if self.get_clock().now().nanoseconds - self._last_dbg_ns > 2e9:
                    self._last_dbg_ns = self.get_clock().now().nanoseconds
                    # H 标的 ENU 方位角（转 yaw 时应恒定）与 cmd 的 ENU 方位角
                    cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
                    u_e = float(self.p_cam[1]) * cos_y + float(self.p_cam[0]) * sin_y
                    u_n = float(self.p_cam[1]) * sin_y - float(self.p_cam[0]) * cos_y
                    marker_enu = math.degrees(math.atan2(u_n, u_e))
                    self.get_logger().info(
                        f"[RL] p_cam=({self.p_cam[0]:+.2f},{self.p_cam[1]:+.2f},{self.p_cam[2]:+.2f}) "
                        f"action=({action[0]:+.2f},{action[1]:+.2f},{action[2]:+.2f}) "
                        f"yaw={math.degrees(self.yaw):+.1f} "
                        f"marker_enu={marker_enu:+.1f} "
                        f"obs_norm={'on' if self.obs_norm is not None else 'OFF!'} "
                        f"outliers={self._outlier_count}"
                    )

            # 指令低通滤波：抑制高频振荡分量（在限幅前对机体系速度滤波）
            alpha = self.get_parameter("cmd_filter_alpha").value
            if alpha < 1.0:
                self._v_filt = alpha * np.array([v_right, v_forward, v_down]) \
                    + (1.0 - alpha) * self._v_filt
                v_right, v_forward, v_down = float(self._v_filt[0]), float(self._v_filt[1]), float(self._v_filt[2])

            # 限幅
            v_xy_max = self.get_parameter("v_xy_max").value
            v_down_max = self.get_parameter("v_down_max").value
            v_up_max = self.get_parameter("v_up_max").value
            v_right = float(np.clip(v_right, -v_xy_max, v_xy_max))
            v_forward = float(np.clip(v_forward, -v_xy_max, v_xy_max))
            v_down = float(np.clip(v_down, -v_up_max, v_down_max))

            # 机体水平速度 -> ENU（实测 cmd_vel_unstamped 为世界固定 ENU 系）：
            # 机体前向 = (cosψ, sinψ)，机体右向 = (sinψ, -cosψ)
            cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
            cmd.linear.x = v_forward * cos_y + v_right * sin_y   # 东
            cmd.linear.y = v_forward * sin_y - v_right * cos_y   # 北
            cmd.linear.z = -v_down                               # 上为正
        else:
            # 非激活状态：滤波器归零，下次激活从零速起步，避免旧值残存
            self._v_filt = np.zeros(3)
            # 诊断：每 3s 打印一次未激活原因（此时没有 [RL] 日志，否则无法定位）
            if self.get_clock().now().nanoseconds - self._last_dbg_ns > 3e9:
                self._last_dbg_ns = self.get_clock().now().nanoseconds
                marker_age = (
                    (self.get_clock().now().nanoseconds - self.marker_stamp_ns) * 1e-9
                    if self.marker_stamp_ns is not None else -1.0
                )
                self.get_logger().warn(
                    f"[IDLE] 未激活: armed={self.mav_state.armed} mode='{self.mav_state.mode}' "
                    f"marker_age={marker_age:.2f}s outliers={self._outlier_count} "
                    f"policy={'loaded' if (self.policy or self.onnx_sess) else 'NONE'} "
                    f"require_guided={require_guided}"
                )

        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = LandingRLNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
