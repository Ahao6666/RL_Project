# SITL 视觉着陆使用指南

控制节点在宿主机运行，连接 ArduPilot SITL + MAVROS + Gazebo。
当前复合翼使用 **GUIDED 控制、QLAND 降落**。以下命令均从项目根目录执行。

## 1. 准备环境

```bash
source /opt/ros/humble/setup.bash
# 若 mavros_msgs 安装在自己的工作空间，还需加载对应 install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

Python 环境需要 ROS2 消息包及 NumPy；ONNX 模式额外需要 `onnxruntime`。
如使用已有虚拟环境，执行 `source .venv/bin/activate`，并确认它可以导入 ROS2 包。

宿主机与 MAVROS 统一使用 Fast DDS，避免出现话题可接收、服务无响应的问题。

## 2. 检查通信

```bash
ros2 topic echo /h_marker/position --once
ros2 topic echo /mavros/state --once
ros2 topic echo /mavros/extended_state --once
ros2 service type /mavros/set_mode
ros2 service type /mavros/cmd/arming
```

视觉消息类型为 `geometry_msgs/msg/PointStamped`。确认话题实际收到消息，
速度指令能够送达 MAVROS，且模式切换和解除武装服务可用。
若视觉发布端为 BEST_EFFORT，节点视觉订阅需使用匹配的 QoS。

## 3. 打开 Foxglove

首次安装：

```bash
sudo apt install ros-$ROS_DISTRO-foxglove-bridge
```

在独立终端中加载 ROS2 环境，启动桥接并保持运行：

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765
```

打开 [Foxglove 仪表板](https://app.foxglove.dev/ahao/dashboard)。

## 4. 起飞并切换模式

手动起飞，在 H 靶标附近悬停，建议高度 **5～10 m**，确保目标持续可见。
随后切换 GUIDED：

```bash
ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode \
  "{base_mode: 0, custom_mode: 'GUIDED'}"
```

## 5. 选择控制策略

每次只运行一个控制节点；切换策略前先停止旧节点。

| 策略 | `controller` | 说明 |
| --- | --- | --- |
| PPO | `rl` | 强化学习策略，需要模型和归一化文件 |
| 混合控制 | `hybrid` | 高空 PPO，低于 3 m 切换 P 控制 |
| P 控制 | `p` | 比例控制，无需模型 |
| MPC | `mpc` | 模型预测控制，无需模型，内置滤波和指令平滑 |

### PPO / 混合控制

```bash
python3 vls_ros2_bridge/vls_ros2_bridge/landing_rl_node.py --ros-args \
  -p controller:=hybrid \
  -p model_path:=checkpoints/ppo_landing_final.onnx \
  -p vecnormalize_path:=checkpoints/obs_norm.npz \
  -p v_xy_max:=2.0 -p v_down_max:=1.5 -p v_up_max:=0.8
```

纯 PPO 将 `hybrid` 改为 `rl`。模型 `.onnx` 与外部权重 `.onnx.data` 应放在同一目录，
归一化文件须与模型匹配。

### P 控制

```bash
python3 vls_ros2_bridge/vls_ros2_bridge/landing_rl_node.py --ros-args \
  -p controller:=p
```

### MPC

```bash
python3 vls_ros2_bridge/vls_ros2_bridge/landing_rl_node.py --ros-args \
  -p controller:=mpc
```

节点默认速度限幅为水平 **1.0 m/s**、下降/上升 **0.8 m/s**。
上面的 PPO 命令使用训练时限幅；对比实验应为所有策略统一设置速度限幅。

## 6. 自动降落与解除武装

所有控制模式默认共用以下流程：

1. 视觉高度低于 **2.0 m** 时请求 **QLAND**，在约 1.5 m 的识别下限前交给飞控。
2. 等待模式确认期间输出零速度；确认后停止发布速度指令，由飞控完成降落。
3. 飞控持续确认 `ON_GROUND` **1 秒**后，请求解除武装。

进入流程后不再依赖 H 标识别。服务失败会重试；人工切换模式会取消自动流程。
缺少落地遥测时不会仅凭高度解除武装。

| 常用参数 | 默认值 | 说明 |
| --- | --- | --- |
| `auto_land_altitude` | `2.0` | 自动降落交接高度，m |
| `auto_land_mode` | `QLAND` | 当前复合翼降落模式；四旋翼改用 `LAND` |
| `auto_land_enabled` | `true` | 是否自动进入降落流程 |
| `marker_timeout` | `1.0` | 视觉超时阈值，s |
| `mpc_max_accel_xy` / `mpc_max_accel_z` | `0.8` / `0.4` | MPC 指令变化率上限，m/s² |

高度指姿态补偿后的相机到靶标平面的距离，按相机安装位置调整交接高度。
更多参数见 [README](../README.md)。

## 7. 观察与记录

随时准备切换 **QLOITER** 接管。观察目标是否收敛、速度指令是否振荡，以及
是否依次出现 QLAND 模式确认和解除武装日志。

每次测试记录：**初始高度/偏差、使用策略、最终落点偏差、振荡或丢失目标情况、接管原因**。
