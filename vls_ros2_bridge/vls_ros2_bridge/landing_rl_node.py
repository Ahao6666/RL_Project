"""阶段 2 桥接节点：加载阶段 1 训练的 PPO 策略，闭环控制 ArduPilot。

数据流：
    /h_marker/position (Point, 相机光轴系 z前/x右/y下)  ─┐
    /mavros/local_position/odom (Odometry, 自身速度)    ├─> 观测组装 -> PPO 策略
                                                         │
    PPO 动作 [v_right, v_forward, v_down]                │
        -> /mavros/setpoint_velocity/cmd_vel_unstamped (Twist)

坐标系说明（与训练环境 LandingEnv 保持一致）：
    策略输出 v_down 为正值表示下降；MAVROS 的 Twist 默认按机体 FLU 解释
    （x 前、y 左、z 上），因此 linear.z = -v_down。
    TODO(阶段2验证)：在 SITL 上实测确认 cmd_vel_unstamped 的坐标系约定，
    以及机体姿态对观测 p_cam 的影响（训练环境假设机体水平）。

安全逻辑：
    - 仅在 GUIDED 模式且已解锁时发布指令；
    - 视觉消息超时 -> 发布零速度（悬停）；
    - 所有指令做限幅。
"""
from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Twist
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from rclpy.node import Node


class LandingRLNode(Node):
    def __init__(self):
        super().__init__("landing_rl_node")
        self.declare_parameter("model_path", "")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("marker_timeout", 0.5)   # 视觉超时 (s) -> 悬停
        self.declare_parameter("v_xy_max", 2.0)
        self.declare_parameter("v_down_max", 1.5)
        self.declare_parameter("v_up_max", 0.8)

        model_path = self.get_parameter("model_path").value
        self.policy = None
        if model_path:
            from stable_baselines3 import PPO
            # TODO: 若训练时用了 VecNormalize，这里需要加载 vecnormalize.pkl
            # 并对接相同的环境封装，否则观测尺度不一致
            self.policy = PPO.load(model_path)
            self.get_logger().info(f"已加载策略: {model_path}")
        else:
            self.get_logger().warn("未提供 model_path，节点只转发安全零指令")

        self.p_cam: np.ndarray | None = None
        self.marker_stamp = None
        self.vel_body = np.zeros(3)
        self.mav_state = State()
        self.prev_action = np.zeros(3, dtype=np.float32)

        self.create_subscription(Point, "/h_marker/position", self._marker_cb, 10)
        self.create_subscription(Odometry, "/mavros/local_position/odom", self._odom_cb, 10)
        self.create_subscription(State, "/mavros/state", self._state_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, "/mavros/setpoint_velocity/cmd_vel_unstamped", 10)

        period = 1.0 / self.get_parameter("control_rate_hz").value
        self.create_timer(period, self._control_cb)

    # -------------------------------------------------------------- #
    def _marker_cb(self, msg: Point):
        self.p_cam = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
        self.marker_stamp = self.get_clock().now()

    def _odom_cb(self, msg: Odometry):
        v = msg.twist.twist.linear
        self.vel_body = np.array([v.x, v.y, v.z])  # TODO: 转换到训练环境的世界系约定

    def _state_cb(self, msg: State):
        self.mav_state = msg

    # -------------------------------------------------------------- #
    def _build_obs(self) -> np.ndarray:
        """组装与 LandingEnv 一致的 10 维观测：p_cam(3)+vel(3)+prev_action(3)+stale(1)。"""
        return np.concatenate([self.p_cam, self.vel_body, self.prev_action, [0.0]]).astype(np.float32)

    def _control_cb(self):
        cmd = Twist()

        armed_guided = self.mav_state.armed and self.mav_state.mode == "GUIDED"
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
            v_right = action[0] * v_xy_max
            v_forward = action[1] * v_xy_max
            v_down = (action[2] + 1.0) / 2.0 * (v_down_max + v_up_max) - v_up_max
            # 策略语义 [v_right, v_forward, v_down] -> MAVROS FLU [x前, y左, z上]
            cmd.linear.x = float(v_forward)
            cmd.linear.y = float(-v_right)
            cmd.linear.z = float(-v_down)
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
