"""速度阶跃响应测量：实测 ArduPilot 速度跟踪时间常数 tau 与 cmd_vel 坐标系约定。

前提：无人机已解锁并悬停在空中，且处于接受 cmd_vel_unstamped 的引导模式
（四旋翼 GUIDED / 垂起 QGUIDED）。本脚本只做速度控制，不切模式、不解锁。

流程（默认 20Hz 发布）：
    0~3s    零速度（悬停基线）
    3~9s    linear.x = +1.0 m/s 阶跃
    9~12s   零速度
    12~18s  linear.y = +1.0 m/s 阶跃
    18~21s  零速度
    21~25s  linear.z = -0.5 m/s 阶跃（若 ENU/FLU 语义则为下降）
    25~28s  零速度

记录 /mavros/local_position/velocity_body（机体系）与
/mavros/local_position/velocity_local（本地 ENU）及姿态，用于事后拟合：
  - tau：阶跃后速度到达 63.2% 稳态值的时间
  - 坐标系：对比指令方向与实际 local ENU 位移方向/航向的关系

用法：
    python3 step_response.py --out /tmp/step_response.csv
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from mavros_msgs.msg import State
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class StepResponseNode(Node):
    def __init__(self, out_path: str, rate_hz: float):
        super().__init__("step_response")
        self.out_path = out_path
        self.rate_hz = rate_hz
        self.state = State()
        self.vel_body = np.zeros(3)     # FRD 机体系
        self.vel_local = np.zeros(3)    # ENU 本地系
        self.yaw = 0.0
        self.ready = False
        self.log_rows = []

        self.create_subscription(State, "/mavros/state", self._state_cb, 10)
        # MAVROS 的遥测话题用 best_effort QoS 发布，订阅方必须匹配
        self.create_subscription(
            TwistStamped, "/mavros/local_position/velocity_body",
            self._vel_body_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            TwistStamped, "/mavros/local_position/velocity_local",
            self._vel_local_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            PoseStamped, "/mavros/local_position/pose",
            self._pose_cb, qos_profile_sensor_data
        )
        self.cmd_pub = self.create_publisher(
            Twist, "/mavros/setpoint_velocity/cmd_vel_unstamped", 10
        )

    def _state_cb(self, msg: State):
        self.state = msg

    def _vel_body_cb(self, msg: TwistStamped):
        self.vel_body = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.ready = True

    def _vel_local_cb(self, msg: TwistStamped):
        self.vel_local = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])

    def _pose_cb(self, msg: PoseStamped):
        self.yaw = yaw_from_quaternion(msg.pose.orientation)

    # (结束时刻 s, vx, vy, vz) 分段指令，vz 负值期望对应下降
    PHASES = [
        (3.0, 0.0, 0.0, 0.0),
        (9.0, 1.0, 0.0, 0.0),
        (12.0, 0.0, 0.0, 0.0),
        (18.0, 0.0, 1.0, 0.0),
        (21.0, 0.0, 0.0, 0.0),
        (25.0, 0.0, 0.0, -0.5),
        (28.0, 0.0, 0.0, 0.0),
    ]

    def run_sequence(self):
        period = 1.0 / self.rate_hz
        t0 = time.time()
        while rclpy.ok():
            t_loop = time.time()
            t = t_loop - t0
            phase = next((p for p in self.PHASES if t < p[0]), None)
            if phase is None:
                break
            cmd = Twist()
            cmd.linear.x, cmd.linear.y, cmd.linear.z = phase[1], phase[2], phase[3]
            self.cmd_pub.publish(cmd)
            self.log_rows.append(
                (t, phase[1], phase[2], phase[3],
                 *self.vel_body, *self.vel_local, self.yaw)
            )
            # 用 spin_once 代替 sleep：驱动订阅回调刷新 vel_body/vel_local/yaw
            rclpy.spin_once(self, timeout_sec=period)
            # spin_once 收到消息会提前返回，补足剩余周期保持恒定发布频率
            time.sleep(max(0.0, period - (time.time() - t_loop)))

    def save(self):
        np.savetxt(
            self.out_path, np.array(self.log_rows), delimiter=",",
            header="t,cmd_x,cmd_y,cmd_z,vb_x,vb_y,vb_z,vl_x_enu,vl_y_enu,vl_z_enu,yaw_rad",
            comments="",
        )
        self.get_logger().info(f"已保存 {len(self.log_rows)} 行到 {self.out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/step_response.csv")
    parser.add_argument("--rate", type=float, default=20.0)
    args = parser.parse_args()

    rclpy.init()
    node = StepResponseNode(args.out, args.rate)

    node.get_logger().info("等待飞控数据（需已解锁、悬停、处于引导模式）...")
    t0 = time.time()
    while not node.ready and time.time() - t0 < 10.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not node.ready:
        node.get_logger().error("未收到速度数据，退出")
        return

    node.get_logger().info(
        f"当前模式 {node.state.mode}，armed={node.state.armed}，3 秒后开始阶跃序列"
    )
    node.run_sequence()
    # 结束前显式发布零速度
    for _ in range(5):
        node.cmd_pub.publish(Twist())
        time.sleep(0.05)
    node.save()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
