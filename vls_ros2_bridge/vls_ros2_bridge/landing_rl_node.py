"""阶段 2 桥接节点：加载阶段 1 训练的 PPO 策略，闭环控制 ArduPilot。

数据流：
    /mavros/local_position/velocity_body (TwistStamped, 实测为 FLU 系) ─┐
    /mavros/local_position/pose (PoseStamped, 取 yaw)                  ├─> 观测组装 -> PPO
    /h_marker/position (PointStamped, 相机光轴系 z前/x右/y下)           ─┘
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
    - 所有指令做限幅；
    - 所有控制器在高度 <2.0m 时交给 QLAND（默认），飞控确认落地后解除武装。
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist, TwistStamped
from mavros_msgs.msg import ExtendedState, State
from mavros_msgs.srv import CommandBool, SetMode
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

try:
    from .mpc_controller import LandingMPC
    from .landing_sequence import LandingSequence
    from .flight_logger import FlightLogger, VECTORS, quaternion_degrees
except ImportError:  # direct script deployment
    from mpc_controller import LandingMPC
    from landing_sequence import LandingSequence
    from flight_logger import FlightLogger, VECTORS, quaternion_degrees


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _fingerprint(path) -> dict | None:
    """模型/归一化文件指纹，写入飞行日志便于事后确认部署的是哪个模型。"""
    try:
        p = Path(path)
        st = p.stat()
        digest = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except OSError:
        return None
    return {"path": str(p), "size": st.st_size, "sha256_16": digest,
            "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}


class LandingRLNode(Node):
    def __init__(self):
        super().__init__("landing_rl_node")
        self.declare_parameter("model_path", "")
        self.declare_parameter("vecnormalize_path", "")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("flight_log_enabled", True)
        self.declare_parameter("flight_log_dir", "runs/flight_logs")
        self.declare_parameter("auto_land_enabled", True)
        self.declare_parameter("auto_land_altitude", 2.5)  # 自动降落触发高度 (m)，与契约 handoff_altitude 一致
        self.declare_parameter("auto_land_mode", "QLAND")
        self.declare_parameter("landing_retry_s", 1.0)
        self.declare_parameter("landing_request_timeout_s", 3.0)
        self.declare_parameter("landing_state_timeout_s", 2.0)
        self.declare_parameter("landed_state_timeout_s", 1.5)
        self.declare_parameter("disarm_ground_hold_s", 1.0)
        self.declare_parameter("marker_timeout", 1.0)   # 视觉超时 (s) -> 悬停
        # 注：新鲜度按消息 header 时间戳判定（含检测延迟），故需大于检测管线延迟
        self.declare_parameter("v_xy_max", 1.0)
        self.declare_parameter("v_down_max", 0.8)
        self.declare_parameter("v_up_max", 0.8)
        # 与训练契约一致的部署参数（见 rl_vls/contract.py）：
        # 视觉尺度补偿、指令低通滤波、下降门控，以及可选契约文件路径。
        # visual_scale：视觉位移/实际位移；1.0 表示视觉已是物理米制。
        self.declare_parameter("visual_scale", 1.0)
        self.declare_parameter("descent_gate_radius", 0.5)
        self.declare_parameter("descent_gate_speed", 0.4)
        self.declare_parameter("cmd_filter_alpha", 0.25)
        # 指令整形：把策略指令限在飞控可达范围内（与训练契约逐项一致）。
        # 真机辨识（2026-10-13）：MPC 指令 ≤0.4~0.8 m/s² 时幅值比 0.9~1.4、等效
        # 延迟仅 0.15~0.25 s；RL 指令要求 4~10 m/s² 时实测速度斜率被削平在
        # ~1~1.3 m/s²，幅值只剩 0.40~0.53、多出 ~90° 相位 → 闭环 0.5 Hz 振荡。
        self.declare_parameter("cmd_rate_max", 1.5)          # 指令最大变化率 (m/s²)
        self.declare_parameter("approach_speed_gain", 1.0)   # 接近段限速斜率 (1/s)
        self.declare_parameter("approach_speed_floor", 0.15)  # 接近段限速下限 (m/s)
        self.declare_parameter("contract_path", "")
        # 话题 y 轴镜像补偿：2026-09-29 yaw 标定（yaw_calib.csv）发现该系统
        # /h_marker/position 的 y 与标准相机光轴系反号（视觉管线的符号问题，
        # 根治应修视觉节点；此处做补偿）
        self.declare_parameter("marker_flip_y", True)
        self._flip_y = -1.0 if self.get_parameter("marker_flip_y").value else 1.0
        # 控制器选择："mpc"（模型预测控制）/ "rl"（策略网络）/ "p" / "hybrid"
        # （高空 RL 接近 + 低空 P 降落，hybrid_alt 为切换高度）/ "fixed"（固定机体系速度，
        # 仅用于测试输出坐标映射：恒定机体系速度的 ENU 输出应以 1:1 斜率随 yaw 旋转）
        self.declare_parameter("controller", "rl")
        self.controller = self.get_parameter("controller").value
        if self.controller not in ("rl", "p", "hybrid", "fixed", "mpc"):
            raise ValueError("Unknown controller: " + self.controller)
        # 加载部署契约（如提供），确保速度映射/滤波/整形/门控与训练一致
        self._apply_contract()
        # 指令整形缓存（契约可能已改写上面声明的参数，故在 _apply_contract 之后读）
        self.dt = 1.0 / float(self.get_parameter("control_rate_hz").value)
        self._shape_rate_max = float(self.get_parameter("cmd_rate_max").value)
        self._shape_gain = float(self.get_parameter("approach_speed_gain").value)
        self._shape_floor = float(self.get_parameter("approach_speed_floor").value)
        self._v_shaped = np.zeros(3)
        for name, default in (("mpc_horizon", 60), ("mpc_tau_xy", 0.6),
                              ("mpc_tau_z", 0.4), ("mpc_descent_speed", 0.3),
                              ("mpc_align_radius", 0.5), ("mpc_iterations", 80),
                              ("mpc_max_accel_xy", 0.4), ("mpc_max_accel_z", 0.4),
                              ("mpc_position_filter_tau", 0.25),
                              ("mpc_velocity_filter_tau", 0.1),
                              ("mpc_descent_filter_tau", 0.4),
                              ("mpc_command_smooth_w", 4.0), ("mpc_delay_xy", 0.3),
                              ("mpc_delay_z", 0.15), ("mpc_visual_scale", 1.0),
                              ("mpc_align_speed", 0.2), ("mpc_position_weight_xy", 2.0),
                              ("mpc_velocity_weight_xy", 4.0)):
            self.declare_parameter(name, default)
        self.mpc = None
        if self.controller == "mpc":
            self.mpc = LandingMPC(
                dt=1.0 / self.get_parameter("control_rate_hz").value,
                horizon=self.get_parameter("mpc_horizon").value,
                tau_xy=self.get_parameter("mpc_tau_xy").value,
                tau_z=self.get_parameter("mpc_tau_z").value,
                v_xy_max=self.get_parameter("v_xy_max").value,
                v_down_max=self.get_parameter("v_down_max").value,
                v_up_max=self.get_parameter("v_up_max").value,
                descent_speed=self.get_parameter("mpc_descent_speed").value,
                align_radius=self.get_parameter("mpc_align_radius").value,
                iterations=self.get_parameter("mpc_iterations").value,
                max_accel_xy=self.get_parameter("mpc_max_accel_xy").value,
                max_accel_z=self.get_parameter("mpc_max_accel_z").value,
                position_filter_tau=self.get_parameter("mpc_position_filter_tau").value,
                velocity_filter_tau=self.get_parameter("mpc_velocity_filter_tau").value,
                descent_filter_tau=self.get_parameter("mpc_descent_filter_tau").value,
                command_smooth_w=self.get_parameter("mpc_command_smooth_w").value,
                delay_xy=self.get_parameter("mpc_delay_xy").value,
                delay_z=self.get_parameter("mpc_delay_z").value,
                visual_scale=self.get_parameter("mpc_visual_scale").value,
                align_speed=self.get_parameter("mpc_align_speed").value,
                position_weight_xy=self.get_parameter("mpc_position_weight_xy").value,
                velocity_weight_xy=self.get_parameter("mpc_velocity_weight_xy").value)
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
        # 指令低通滤波状态；cmd_filter_alpha 参数已在上方按契约声明
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
            if self.controller in ("p", "mpc", "fixed"):
                self.get_logger().info(f"{self.controller} 控制模式，无需策略模型")
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

        # 记录模型/归一化文件指纹，并在 ONNX 比同目录 .zip 旧时告警。
        # 历史事故：重新训练后忘记重新导出 ONNX，SITL 实际跑的是旧策略，
        # 表现为“悬停不下降/横向来回抽”（门控因误差过大始终关闭）。
        self._model_fp = _fingerprint(model_path) if model_path else None
        self._vecnorm_fp = _fingerprint(vn_path) if vn_path else None
        for name, fp in (("model", self._model_fp), ("vecnormalize", self._vecnorm_fp)):
            if fp:
                self.get_logger().info(
                    f"{name} 指纹: sha256[:16]={fp['sha256_16']} size={fp['size']} mtime={fp['mtime']}")
        if model_path.endswith(".onnx"):
            sibling = Path(model_path).with_suffix(".zip")
            if sibling.exists() and Path(model_path).stat().st_mtime < sibling.stat().st_mtime:
                self.get_logger().warn(
                    f"{Path(model_path).name} 比 {sibling.name} 旧，可能忘记重新导出"
                    f"（rl_vls/tools/export_onnx.py）——请确认部署的是最新训练的策略！")

        self.marker_raw = None
        self.marker_raw_stamp_ns = None
        self.marker_status = "missing"
        self.vel_stamp_ns = self.vel_enu_stamp_ns = self.pose_stamp_ns = None
        self.vel_enu = np.full(3, np.nan)
        self.pos_enu = np.full(3, np.nan)
        self.flight_recorder = None
        self._flight_log_error_reported = False
        self.p_cam: np.ndarray | None = None
        self.marker_stamp_ns: int | None = None   # 视觉帧时间戳（消息 header）
        self.vel_flu = np.zeros(3)      # 机体系 FLU: x前 y左 z上（阶跃数据实测，见 H2 假设验证）
        self.yaw = 0.0                  # ENU 航向（弧度，东为 0 逆时针为正）
        # 姿态历史缓冲：反旋转时按视觉帧时间戳取同时刻姿态，消除延迟错位
        self.pose_hist: deque = deque(maxlen=400)
        self.mav_state = State()
        self.state_stamp_ns = None
        self.extended_state = ExtendedState()
        self.extended_state_stamp_ns = None
        self.land_client = self.create_client(SetMode, "/mavros/set_mode")
        self.disarm_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.landing = LandingSequence(
            self._request_land, self._request_disarm, self.get_logger(),
            altitude=self.get_parameter("auto_land_altitude").value,
            land_mode=self.get_parameter("auto_land_mode").value,
            retry_s=self.get_parameter("landing_retry_s").value,
            request_timeout_s=self.get_parameter("landing_request_timeout_s").value,
            disarm_hold_s=self.get_parameter("disarm_ground_hold_s").value)
        self.prev_action = np.zeros(3, dtype=np.float32)
        self._rl_gate = float("nan")
        self._last_dbg_ns = 0

        self.create_subscription(PointStamped, "/h_marker/position", self._marker_cb, 10)
        # MAVROS 遥测话题为 best_effort QoS，订阅方必须匹配
        self.create_subscription(TwistStamped, "/mavros/local_position/velocity_body",
                                 self._vel_cb, qos_profile_sensor_data)
        self.create_subscription(TwistStamped, "/mavros/local_position/velocity_local",
                                 self._vel_enu_cb, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "/mavros/local_position/pose",
                                 self._pose_cb, qos_profile_sensor_data)
        self.create_subscription(State, "/mavros/state", self._state_cb, qos_profile_sensor_data)
        self.create_subscription(ExtendedState, "/mavros/extended_state",
                                 self._extended_state_cb, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, "/mavros/setpoint_velocity/cmd_vel_unstamped", 10)

        if self.get_parameter("flight_log_enabled").value:
            try:
                params = {name: parameter.value
                          for name, parameter in self._parameters.items()}
                if self._model_fp:
                    params["model_fingerprint"] = self._model_fp
                if self._vecnorm_fp:
                    params["vecnormalize_fingerprint"] = self._vecnorm_fp
                self.flight_recorder = FlightLogger(
                    self.get_parameter("flight_log_dir").value, self.controller, params)
                self.get_logger().info(f"飞行日志: {self.flight_recorder.path}")
            except (OSError, ValueError) as exc:
                self.get_logger().error(f"无法创建飞行日志，继续控制: {exc}")
        period = 1.0 / self.get_parameter("control_rate_hz").value
        self.create_timer(period, self._control_cb)

    def _apply_contract(self):
        """加载部署契约（训练保存的 env_config.json）。

        若提供 contract_path，用其中 deploy 段的参数覆盖本节点参数，保证部署与
        训练逐项一致（速度映射、控制频率、指令滤波、视觉尺度、下降门控、交接高度）。
        无论是否加载成功，都打印生效的契约，便于与训练配置对比。
        """
        path = self.get_parameter("contract_path").value
        source = "defaults"
        if path:
            try:
                deploy = json.loads(Path(path).read_text(encoding="utf-8")).get("deploy", {})
                for name, value in deploy.items():
                    if self.has_parameter(name):
                        self.set_parameters([Parameter(name, value=value)])
                    else:
                        self.declare_parameter(name, value)
                source = path
            except (OSError, ValueError, KeyError) as exc:
                self.get_logger().error(f"读取 contract_path 失败，沿用默认参数: {exc}")
        keys = ("control_rate_hz", "v_xy_max", "v_down_max", "v_up_max",
                "cmd_filter_alpha", "cmd_rate_max", "approach_speed_gain",
                "approach_speed_floor", "visual_scale", "descent_gate_radius",
                "descent_gate_speed", "auto_land_altitude")
        effective = {key: self.get_parameter(key).value for key in keys}
        self.get_logger().info(f"执行器契约 (来源: {source}): {effective}")

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
        self.marker_raw = np.array([msg.point.x, msg.point.y, msg.point.z])
        self.marker_raw_stamp_ns = stamp_ns
        if not np.all(np.isfinite(self.marker_raw)):
            self.marker_status = "invalid"
            return
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
                self.marker_status = "outlier"
                return
        self.marker_status = "accepted"
        self.p_cam = new_p.astype(np.float32)
        self.marker_stamp_ns = stamp_ns

    def _vel_cb(self, msg: TwistStamped):
        v = msg.twist.linear
        self.vel_flu = np.array([v.x, v.y, v.z])
        self.vel_stamp_ns = self.get_clock().now().nanoseconds

    def _vel_enu_cb(self, msg: TwistStamped):
        v = msg.twist.linear
        self.vel_enu = np.array([v.x, v.y, v.z])
        self.vel_enu_stamp_ns = self.get_clock().now().nanoseconds

    def _pose_cb(self, msg: PoseStamped):
        position = msg.pose.position
        self.pos_enu = np.array([position.x, position.y, position.z])
        self.pose_stamp_ns = self.get_clock().now().nanoseconds
        q = msg.pose.orientation
        self.quat = (q.w, q.x, q.y, q.z)
        self.yaw = yaw_from_quaternion(q)
        stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds
        if stamp_ns == 0:
            stamp_ns = self.get_clock().now().nanoseconds
        self.pose_hist.append((stamp_ns, self.quat))

    def _state_cb(self, msg: State):
        self.mav_state = msg
        self.state_stamp_ns = self.get_clock().now().nanoseconds

    def _extended_state_cb(self, msg: ExtendedState):
        self.extended_state = msg
        self.extended_state_stamp_ns = self.get_clock().now().nanoseconds

    def _request_land(self, mode):
        if not self.land_client.service_is_ready():
            return None
        request = SetMode.Request()
        request.custom_mode = mode
        return self.land_client.call_async(request)

    def _request_disarm(self):
        if not self.disarm_client.service_is_ready():
            return None
        request = CommandBool.Request()
        request.value = False
        return self.disarm_client.call_async(request)

    def _landing_step(self, now_ns, marker_fresh):
        def fresh(stamp, timeout):
            return stamp is not None and 0 <= (now_ns - stamp)*1e-9 < timeout
        return self.landing.update(
            now_ns*1e-9, armed=self.mav_state.armed, mode=self.mav_state.mode,
            state_fresh=self.mav_state.connected and fresh(
                self.state_stamp_ns, self.get_parameter("landing_state_timeout_s").value),
            marker_altitude=self._handoff_altitude(),
            marker_fresh=marker_fresh,
            on_ground=self.extended_state.landed_state == ExtendedState.LANDED_STATE_ON_GROUND,
            ground_fresh=fresh(self.extended_state_stamp_ns,
                               self.get_parameter("landed_state_timeout_s").value),
            ground_stamp=self.extended_state_stamp_ns*1e-9
                if self.extended_state_stamp_ns is not None else None)

    # -------------------------------------------------------------- #
    def _rl_p_cam(self) -> np.ndarray:
        """RL 观测用的视觉目标位置（补偿到物理米制）。

        视觉位移约为实际位移的 visual_scale 倍，除以该值后与训练环境的物理米制
        观测一致；visual_scale=1.0 表示视觉已是物理米制。
        """
        if self.p_cam is None:
            return np.zeros(3, dtype=np.float32)
        scale = float(self.get_parameter("visual_scale").value)
        if scale > 0.0 and scale != 1.0:
            return (self.p_cam / scale).astype(np.float32)
        return np.asarray(self.p_cam, dtype=np.float32)

    def _rl_velocity(self) -> np.ndarray:
        """与训练环境一致的水平机体系 [右, 前, 下] 速度。

        优先用新鲜 ENU 实测速度旋转到水平机体系（与 MPC 一致），消除机体倾斜
        导致的位置/速度坐标不一致；无对应遥测时回退到机体系 FLU [-y, x, -z]。
        """
        now_ns = self.get_clock().now().nanoseconds
        if (self.vel_enu_stamp_ns is not None and self.pose_stamp_ns is not None
                and 0 <= (now_ns - self.vel_enu_stamp_ns) * 1e-9 < 0.2
                and 0 <= (now_ns - self.pose_stamp_ns) * 1e-9 < 0.2
                and np.all(np.isfinite(self.vel_enu))):
            c, sine = math.cos(self.yaw), math.sin(self.yaw)
            return np.array([self.vel_enu[0] * sine - self.vel_enu[1] * c,
                             self.vel_enu[0] * c + self.vel_enu[1] * sine,
                             -self.vel_enu[2]], dtype=np.float64)
        return np.array([-self.vel_flu[1], self.vel_flu[0], -self.vel_flu[2]],
                        dtype=np.float64)

    def _shape_command(self, v_cmd: np.ndarray, d_xy: float) -> np.ndarray:
        """指令整形：接近段限速 + 加速度（斜率）限制。

        与训练环境 ``LandingEnv._shape_command`` 逐项一致（参数取契约
        ``cmd_rate_max`` / ``approach_speed_gain`` / ``approach_speed_floor``）。

        动机（2026-10-13 真机低频辨识）：飞控对速度目标可达的加速度约 1~2.5 m/s²。
        策略输出的 ±1 m/s、~2 s 周期继电方波等效要求 4~10 m/s²，被削成慢三角波
        （幅值只剩 0.40~0.53、多出 ~90° 相位 → 拟合成 +0.5 s 假延迟），闭环因此
        变成 0.5 Hz 极限环。同一机体 MPC 通路指令 ≤0.4~0.8 m/s²，幅值比 0.9~1.4、
        等效延迟仅 0.15~0.25 s——区别就在指令是否超出飞控线性区。
        """
        cap_xy = self._shape_gain * d_xy + self._shape_floor
        hspeed = float(np.hypot(v_cmd[0], v_cmd[1]))
        if 0.0 < cap_xy < hspeed:
            scale = cap_xy / hspeed
            v_cmd = np.array([v_cmd[0] * scale, v_cmd[1] * scale, v_cmd[2]])
        max_step = self._shape_rate_max * self.dt
        self._v_shaped = self._v_shaped + np.clip(v_cmd - self._v_shaped, -max_step, max_step)
        return self._v_shaped

    def _rl_descent_gate(self, d_xy: float, horiz_speed: float) -> float:
        """下降门控：与训练环境 LandingEnv._descent_gate 逐项一致（平方软因子）。"""
        radius = float(self.get_parameter("descent_gate_radius").value)
        if radius <= 0.0:
            return 1.0
        ratio = min(d_xy / radius, 1.0)
        gate = 1.0 - ratio * ratio
        max_speed = float(self.get_parameter("descent_gate_speed").value)
        if max_speed > 0.0:
            ratio_v = min(horiz_speed / max_speed, 1.0)
            gate *= 1.0 - ratio_v * ratio_v
        return float(np.clip(gate, 0.0, 1.0))

    def _handoff_altitude(self):
        """交给 QLAND 的高度：RL/hybrid 用补偿后的物理高度，其余用原始视觉高度。"""
        if self.p_cam is None:
            return None
        if self.controller in ("rl", "hybrid"):
            return float(self._rl_p_cam()[2])
        return float(self.p_cam[2])

    def _build_obs(self) -> np.ndarray:
        """组装与 LandingEnv 一致的 10 维观测：p_cam(3)+vel(3)+prev_action(3)+stale(1)。

        视觉与速度都换算到训练环境约定的物理米制水平机体系 [右, 前, 下]：
        p_cam 用 _rl_p_cam()（尺度补偿），速度用 _rl_velocity()（ENU 旋转或 FLU 换轴）。
        """
        vel_env = self._rl_velocity()
        obs = np.concatenate([self._rl_p_cam(), vel_env, self.prev_action, [0.0]]).astype(np.float32)
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
        effective_controller = "idle"
        self._rl_gate = float("nan")

        now_ns = self.get_clock().now().nanoseconds
        require_guided = self.get_parameter("require_guided").value
        armed_guided = (self.mav_state.armed
                        and self.mav_state.mode in ("GUIDED", "QGUIDED")) or not require_guided
        marker_fresh = (
            self.p_cam is not None
            and self.marker_stamp_ns is not None
            and np.all(np.isfinite(self.p_cam))
            and 0 <= (now_ns - self.marker_stamp_ns) * 1e-9
            < self.get_parameter("marker_timeout").value
        )
        visual_fresh = marker_fresh
        # Run before every controller, including fixed test mode. The handoff
        # always requires real fresh vision and armed GUIDED/QGUIDED telemetry.
        if (self.get_parameter("auto_land_enabled").value or self.landing.phase != "idle") \
                and self._landing_step(now_ns, marker_fresh):
            self._v_filt = np.zeros(3)
            self.prev_action = np.zeros(3, dtype=np.float32)
            if self.mpc is not None:
                self.mpc.reset()
            # While awaiting LAND confirmation, hold zero velocity. Once LAND
            # owns control, stop publishing setpoints entirely; likewise on
            # manual takeover or completion.
            if self.landing.phase == "landing" and self.landing.confirmed_at is None \
                    and self.mav_state.mode in ("GUIDED", "QGUIDED"):
                self.cmd_pub.publish(cmd)
                self._record_flight_sample(now_ns, cmd, self.landing.phase, visual_fresh)
            else:
                self._record_flight_sample(now_ns, None, self.landing.phase, visual_fresh)
            return
        if self.controller == "fixed":
            marker_fresh = True   # fixed 模式不需要视觉输入

        policy_loaded = self.policy is not None or self.onnx_sess is not None
        active = self.controller in ("p", "fixed", "mpc") or policy_loaded
        if armed_guided and marker_fresh and active:
            # 控制器模式解析：hybrid = 高空 RL + 低空 P（hybrid_alt 切换）
            mode = self.controller
            if mode == "hybrid":
                hybrid_alt = self.get_parameter("hybrid_alt").value
                mode = "rl" if float(self._rl_p_cam()[2]) > hybrid_alt else "p"
            effective_controller = mode
            if mode == "fixed":
                v_right = self.get_parameter("fixed_v_right").value
                v_forward = self.get_parameter("fixed_v_forward").value
                v_down = self.get_parameter("fixed_v_down").value
            elif mode == "mpc":
                velocity = np.array([-self.vel_flu[1], self.vel_flu[0], -self.vel_flu[2]])
                # Match the level camera observation with level ENU velocity.
                if (self.vel_enu_stamp_ns is not None and self.pose_stamp_ns is not None
                        and 0 <= (now_ns-self.vel_enu_stamp_ns)*1e-9 < 0.2
                        and 0 <= (now_ns-self.pose_stamp_ns)*1e-9 < 0.2
                        and np.all(np.isfinite(self.vel_enu))):
                    c, sine = math.cos(self.yaw), math.sin(self.yaw)
                    velocity = np.array([self.vel_enu[0]*sine-self.vel_enu[1]*c,
                                         self.vel_enu[0]*c+self.vel_enu[1]*sine, -self.vel_enu[2]])
                v_right, v_forward, v_down = self.mpc.command(
                    self.p_cam, velocity, measurement_stamp=self.marker_stamp_ns)
                if now_ns - self._last_dbg_ns > 2e9:
                    self._last_dbg_ns = now_ns
                    estimate = self.mpc.filtered_position
                    self.get_logger().info(
                        f"[MPC] p_cam=({self.p_cam[0]:+.2f},{self.p_cam[1]:+.2f},{self.p_cam[2]:+.2f}) "
                        f"estimate_m={estimate} descent_gate={self.mpc.descent_gate:.2f} cmd=({v_right:+.2f},{v_forward:+.2f},{v_down:+.2f}) "
                        f"outliers={self._outlier_count}")
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

            # 指令整形（RL 通路）：接近段限速 + 加速度限制，顺序与训练环境一致
            # （整形 → 滤波 → 门控 → 发布）。动机见 _shape_command 注释。
            if mode == "rl":
                d_xy_cmd = float(np.hypot(*self._rl_p_cam()[:2]))
                v_right, v_forward, v_down = self._shape_command(
                    np.array([v_right, v_forward, v_down]), d_xy_cmd)

            # 指令低通滤波：默认 cmd_filter_alpha=1.0 关闭（改由上面速率限制整形）；
            # 若回退到 <1.0，则抑制高频振荡分量
            alpha = self.get_parameter("cmd_filter_alpha").value
            if alpha < 1.0 and self.controller != "mpc":
                self._v_filt = alpha * np.array([v_right, v_forward, v_down]) \
                    + (1.0 - alpha) * self._v_filt
                v_right, v_forward, v_down = float(self._v_filt[0]), float(self._v_filt[1]), float(self._v_filt[2])

            # 下降门控（RL）：未对准或横向速度过大时抑制下降，与训练环境一致；
            # 上升/悬停不受限。用补偿后的物理米制水平偏差与水平速度判定。
            if mode == "rl":
                if v_down > 0.0:
                    p_cam_phys = self._rl_p_cam()
                    d_xy = float(np.hypot(p_cam_phys[0], p_cam_phys[1]))
                    hspeed = float(np.hypot(*self._rl_velocity()[:2]))
                    gate = self._rl_descent_gate(d_xy, hspeed)
                    v_down *= gate
                    self._rl_gate = gate
                else:
                    self._rl_gate = 1.0

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
            # 非激活状态：滤波器/整形状态归零，下次激活从零速起步，避免旧值残存
            self._v_filt = np.zeros(3)
            self._v_shaped = np.zeros(3)
            if self.mpc is not None:
                self.mpc.reset()
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
        self._record_flight_sample(now_ns, cmd, effective_controller, visual_fresh)

    def _record_flight_sample(self, now_ns, cmd, phase, marker_fresh):
        recorder = self.flight_recorder
        if recorder is None:
            return
        if recorder.error is not None:
            if not self._flight_log_error_reported:
                self.get_logger().error(f"飞行日志写入失败，继续控制: {recorder.error}")
                self._flight_log_error_reported = True
            return
        def age(stamp):
            return (now_ns-stamp)*1e-9 if stamp is not None else float('nan')
        cmd_enu = np.array([cmd.linear.x, cmd.linear.y, cmd.linear.z]) if cmd else np.full(3, np.nan)
        c, sine = math.cos(self.yaw), math.sin(self.yaw)
        cmd_body = [cmd_enu[0]*sine-cmd_enu[1]*c,
                    cmd_enu[0]*c+cmd_enu[1]*sine, -cmd_enu[2]]
        row = dict(controller=self.controller, phase=phase, mode=self.mav_state.mode,
                   armed=int(self.mav_state.armed), connected=int(self.mav_state.connected),
                   landed_state=self.extended_state.landed_state,
                   cmd_published=int(cmd is not None), marker_fresh=int(marker_fresh),
                   marker_status=self.marker_status, outlier_count=self._outlier_count,
                   marker_stamp_ns=self.marker_stamp_ns, marker_raw_stamp_ns=self.marker_raw_stamp_ns,
                   marker_age_s=age(self.marker_stamp_ns), vel_age_s=age(self.vel_stamp_ns),
                   vel_enu_age_s=age(self.vel_enu_stamp_ns), pose_age_s=age(self.pose_stamp_ns),
                   state_age_s=age(self.state_stamp_ns), landed_age_s=age(self.extended_state_stamp_ns))
        row['mpc_descent_gate'] = self.mpc.descent_gate if self.mpc is not None else float('nan')
        row['rl_descent_gate'] = getattr(self, '_rl_gate', float('nan'))
        vectors = dict(mpc_position=self.mpc.filtered_position if self.mpc is not None else None,
                       mpc_velocity=self.mpc.filtered_velocity if self.mpc is not None else None,
                       marker_raw=self.marker_raw, marker=self.p_cam, cmd_enu=cmd_enu,
                       cmd_body=cmd_body, vel_flu=self.vel_flu if self.vel_stamp_ns is not None else None,
                       vel_enu=self.vel_enu, pos_enu=self.pos_enu,
                       attitude=quaternion_degrees(self.quat) if self.pose_stamp_ns is not None else None,
                       quat=self.quat if self.pose_stamp_ns is not None else None,
                       marker_rl=self._rl_p_cam() if self.p_cam is not None else None,
                       prev_action=getattr(self, 'prev_action', None))
        for prefix, axes in VECTORS.items():
            values = vectors[prefix]
            if values is not None:
                row.update({f"{prefix}_{axis}": float(value) for axis, value in zip(axes, values)})
        recorder.record(now_ns, row)

    def destroy_node(self):
        if self.flight_recorder is not None:
            self.flight_recorder.close()
        return super().destroy_node()


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
