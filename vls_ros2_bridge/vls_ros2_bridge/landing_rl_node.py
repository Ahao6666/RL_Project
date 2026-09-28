"""阶段 2 桥接节点：加载阶段 1 训练的 PPO 策略，闭环控制 ArduPilot。

数据流：
    /h_marker/position (Point, 相机光轴系 z前/x右/y下)        ─┐
    /mavros/local_position/velocity_body (TwistStamped, FRD)  ├─> 观测组装 -> PPO
    /mavros/local_position/pose (PoseStamped, 取 yaw)         ─┘
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

        model_path = self.get_parameter("model_path").value
        self.policy = None
        if model_path:
            from stable_baselines3 import PPO
            self.policy = PPO.load(model_path)
            self.get_logger().info(f"已加载策略: {model_path}")
        else:
            self.get_logger().warn("未提供 model_path，节点只转发安全零指令")

        # 观测归一化统计量（与训练时 VecNormalize 一致），不加载则直接用原始观测
        self.obs_rms = None
        vn_path = self.get_parameter("vecnormalize_path").value
        if vn_path:
            with open(vn_path, "rb") as f:
                vn = pickle.load(f)
            self.obs_rms = vn.obs_rms
            self.obs_clip = vn.clip_obs
            self.get_logger().info(f"已加载观测归一化: {vn_path}")

        self.p_cam: np.ndarray | None = None
        self.marker_stamp = None
        self.vel_frd = np.zeros(3)      # 机体系 FRD: x前 y右 z下
        self.yaw = 0.0                  # ENU 航向（弧度，东为 0 逆时针为正）
        self.mav_state = State()
        self.prev_action = np.zeros(3, dtype=np.float32)

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
        self.p_cam = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
        self.marker_stamp = self.get_clock().now()

    def _vel_cb(self, msg: TwistStamped):
        v = msg.twist.linear
        self.vel_frd = np.array([v.x, v.y, v.z])

    def _pose_cb(self, msg: PoseStamped):
        self.yaw = yaw_from_quaternion(msg.pose.orientation)

    def _state_cb(self, msg: State):
        self.mav_state = msg

    # -------------------------------------------------------------- #
    def _build_obs(self) -> np.ndarray:
        """组装与 LandingEnv 一致的 10 维观测：p_cam(3)+vel(3)+prev_action(3)+stale(1)。

        训练环境的速度观测定义在 [右, 前, 下] 方向，即机体系 FRD 的 [y, x, z]。
        """
        vel_env = np.array([self.vel_frd[1], self.vel_frd[0], self.vel_frd[2]])
        obs = np.concatenate([self.p_cam, vel_env, self.prev_action, [0.0]]).astype(np.float32)
        if self.obs_rms is not None:
            obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            obs = np.clip(obs, -self.obs_clip, self.obs_clip).astype(np.float32)
        return obs

    def _control_cb(self):
        cmd = Twist()

        armed_guided = self.mav_state.armed and self.mav_state.mode in ("GUIDED", "QGUIDED")
        marker_fresh = (
            self.p_cam is not None
            and self.marker_stamp is not None
            and (self.get_clock().now() - self.marker_stamp).nanoseconds * 1e-9
            < self.get_parameter("marker_timeout").value
        )

        if armed_guided and marker_fresh and self.policy is not None:
            action, _ = self.policy.predict(self._build_obs(), deterministic=True)
            action = np.clip(action, -1.0, 1.0)
            self.prev_action = action.astype(np.float32)
            v_xy_max = self.get_parameter("v_xy_max").value
            v_down_max = self.get_parameter("v_down_max").value
            v_up_max = self.get_parameter("v_up_max").value
            v_right = float(action[0]) * v_xy_max
            v_forward = float(action[1]) * v_xy_max
            v_down = (float(action[2]) + 1.0) / 2.0 * (v_down_max + v_up_max) - v_up_max

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
