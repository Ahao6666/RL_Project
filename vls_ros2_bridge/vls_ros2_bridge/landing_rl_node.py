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

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, Twist, TwistStamped
from mavros_msgs.msg import State
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


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
        self.declare_parameter("marker_timeout", 0.5)   # 视觉超时 (s) -> 悬停
        self.declare_parameter("v_xy_max", 2.0)
        self.declare_parameter("v_down_max", 1.5)
        self.declare_parameter("v_up_max", 0.8)
        # 话题 y 轴镜像补偿：2026-09-29 yaw 标定（yaw_calib.csv）发现该系统
        # /h_marker/position 的 y 与标准相机光轴系反号（视觉管线的符号问题，
        # 根治应修视觉节点；此处做补偿）
        self.declare_parameter("marker_flip_y", True)
        self._flip_y = -1.0 if self.get_parameter("marker_flip_y").value else 1.0
        # 控制器选择："rl"（策略网络）/ "p"（比例控制 baseline）/ "fixed"（固定机体系速度，
        # 仅用于测试输出坐标映射：恒定机体系速度的 ENU 输出应以 1:1 斜率随 yaw 旋转）
        self.declare_parameter("controller", "rl")
        self.controller = self.get_parameter("controller").value
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
        self.marker_stamp = None
        self.vel_flu = np.zeros(3)      # 机体系 FLU: x前 y左 z上（阶跃数据实测，见 H2 假设验证）
        self.yaw = 0.0                  # ENU 航向（弧度，东为 0 逆时针为正）
        self.mav_state = State()
        self.prev_action = np.zeros(3, dtype=np.float32)
        self._last_dbg_ns = 0

        self.create_subscription(Point, "/h_marker/position", self._marker_cb, 10)
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
    def _marker_cb(self, msg: Point):
        # y 轴镜像补偿（话题 y 与标准相机光轴系反号），z 沿光轴不受影响
        self.p_cam = np.array([msg.x, self._flip_y * msg.y, msg.z], dtype=np.float32)
        self.marker_stamp = self.get_clock().now()

    def _vel_cb(self, msg: TwistStamped):
        v = msg.twist.linear
        self.vel_flu = np.array([v.x, v.y, v.z])

    def _pose_cb(self, msg: PoseStamped):
        self.yaw = yaw_from_quaternion(msg.pose.orientation)

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
            and self.marker_stamp is not None
            and (self.get_clock().now() - self.marker_stamp).nanoseconds * 1e-9
            < self.get_parameter("marker_timeout").value
        )
        if self.controller == "fixed":
            marker_fresh = True   # fixed 模式不需要视觉输入

        active = self.controller in ("p", "fixed") or (self.policy is not None or self.onnx_sess is not None)
        if armed_guided and marker_fresh and active:
            if self.controller == "fixed":
                v_right = self.get_parameter("fixed_v_right").value
                v_forward = self.get_parameter("fixed_v_forward").value
                v_down = self.get_parameter("fixed_v_down").value
            elif self.controller == "p":
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
                    self.get_logger().info(
                        f"[RL] p_cam=({self.p_cam[0]:+.2f},{self.p_cam[1]:+.2f},{self.p_cam[2]:+.2f}) "
                        f"obs_vel=({obs[3]:+.2f},{obs[4]:+.2f},{obs[5]:+.2f}) "
                        f"action=({action[0]:+.2f},{action[1]:+.2f},{action[2]:+.2f}) "
                        f"obs_norm={'on' if self.obs_norm is not None else 'OFF!'}"
                    )

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
        # 否则 cmd 保持全零（悬停）

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
