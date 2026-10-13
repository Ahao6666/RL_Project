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

### PPO 

```bash
python3 vls_ros2_bridge/vls_ros2_bridge/landing_rl_node.py --ros-args \
  -p controller:=rl \
  -p model_path:=checkpoints/ppo_landing_final.onnx \
  -p vecnormalize_path:=checkpoints/obs_norm.npz \
  -p contract_path:=checkpoints/env_config.json
```

模型 `.onnx` 与外部权重 `.onnx.data` 应放在同一目录，归一化文件须与模型匹配。

**每次重新训练后必须重新导出**（否则节点会加载旧的 ONNX，实飞表现为“悬停不下降 /
横向来回抽”，因为水平误差始终过大导致下降门控恒为 0）：

```bash
python rl_vls/tools/export_onnx.py checkpoints/ppo_landing_final.zip checkpoints/ppo_landing_final.onnx
python rl_vls/tools/export_vecnormalize.py checkpoints/vecnormalize.pkl checkpoints/obs_norm.npz
```

节点启动时会打印模型与归一化文件的指纹（sha256 前 16 位 + mtime），并写入飞行日志
JSON 的 `parameters.model_fingerprint` / `vecnormalize_fingerprint`；若 ONNX 比同目录
`.zip` 旧会打印 WARN。事后排查时先看这两项即可确认跑的是哪个模型。

**必须带上 `contract_path`**（训练时由 `train_ppo` 写入的 `env_config.json`）：
节点会用它覆盖速度映射、控制频率、指令滤波、视觉尺度、下降门控与交接高度，
保证部署与训练逐项一致，并在启动日志里打印生效的“执行器契约”。不传则用节点默认值，
可能与训练不匹配。

RL 观测已把视觉尺度补偿到物理米制（`visual_scale`，视觉已修正后为 1.0），并对未对准/
横向速度过大的下降做门控；交接高度也按补偿后的物理高度判断。若视觉再次出现尺度偏差，
用飞行日志里 视觉位移/实际位移 的回归斜率设 `visual_scale` 并**重新训练**。

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

MPC 当前默认下降速度为 **0.3 m/s**，水平对准且速度低于 **0.2 m/s** 后逐渐允许下降。
预测模型包含延迟补偿；视觉尺度 `mpc_visual_scale` 默认 `1.0`（2026-10-13 起视觉节点
输出已是物理米制）。若更换视觉标定需重新校验，并用飞行日志里 视觉位移/实际位移
的回归斜率设回该值。

节点默认速度限幅为水平 **1.0 m/s**、下降/上升 **0.8 m/s**，与训练契约一致。
对比实验应为所有策略统一设置速度限幅与交接高度。RL 的 `v_xy_max`、`v_down_max`、
`v_up_max`、`cmd_filter_alpha`、`cmd_rate_max`、`approach_speed_gain`、
`approach_speed_floor`、`visual_scale`、`descent_gate_radius/speed` 建议
全部由 `contract_path` 提供，避免手工漂移。

### 指令整形（2026-10-13 起，RL 通路）

真机低频辨识（对指令/实测速度做 0.5 s 平滑后整段拟合一阶+延迟模型）得到：

| 通路 | 指令要求加速度 | 幅值传递比 | 等效延迟 |
| --- | --- | --- | --- |
| MPC（指令 ≤0.4~0.8 m/s²） | ≤0.8 m/s² | 0.9~1.4 | 0.15~0.25 s |
| RL（未整形） | 4~10 m/s² | 0.40~0.53 | 0.70~0.75 s |

飞控可达的加速度只有约 1~2.5 m/s²。未整形时策略的 ±1 m/s、~2 s 周期继电方波被削成
慢三角波（幅值减半 + 多出 ~90° 相位），闭环因此变成 0.5 Hz 极限环——表现就是
“在空中共振、悬停不下降”，且无论怎么重训都在策略 0.65 s 的可控悬崖之外。

因此 RL 通路在**发布前**对机体系速度指令做两级整形（与训练环境逐项一致）：

1. 接近段速度剖面：`|v_xy| ≤ approach_speed_gain · d_xy + approach_speed_floor`
   （默认 1.0 /s、0.15 m/s），让“到交接高度还很快”结构上不可能；
2. 加速度限制：每控制周期指令变化 ≤ `cmd_rate_max · dt`（默认 1.5 m/s²）。

`cmd_filter_alpha` 默认 **1.0（关闭）**：一阶低通在 0.5 Hz 上会额外引入 ~0.2 s
等效延迟，已由加速度限制取代。启动日志里的“执行器契约”一行会打印生效值，
飞行 JSON 也会记录全部参数与模型/归一化指纹。

### 混合控制（P+PPO）

```bash
python3 vls_ros2_bridge/vls_ros2_bridge/landing_rl_node.py --ros-args \
  -p controller:=hybrid \
  -p model_path:=checkpoints/ppo_landing_final.onnx \
  -p vecnormalize_path:=checkpoints/obs_norm.npz
```



## 6. 自动降落与解除武装

所有控制模式默认共用以下流程：

1. 视觉高度低于 **2.5 m** 时请求 **QLAND**，在约 1.5 m 的识别下限前交给飞控。
   （该值与训练契约 `handoff_altitude` 一致，由 `contract_path` 的 `deploy.auto_land_altitude` 提供；
   2026-10-13 由 2.0 m 上调到 2.5 m；改动后需重新训练，因为该值同时是训练回合的终点）
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
| `mpc_max_accel_xy` / `mpc_max_accel_z` | `0.4` / `0.4` | MPC 指令变化率上限，m/s² |

高度指姿态补偿后的相机到靶标平面的距离，按相机安装位置调整交接高度。
更多参数见 [README](../README.md)。

## 7. 观察与记录

随时准备切换 **QLOITER** 接管。观察目标是否收敛、速度指令是否振荡，以及
是否依次出现 QLAND 模式确认和解除武装日志。

每次测试记录：**初始高度/偏差、使用策略、最终落点偏差、振荡或丢失目标情况、接管原因**。

## 8. 飞行日志与分析

节点默认按控制频率记录日志，启动时显示 CSV 路径，文件保存在 `runs/flight_logs/`。
内容包括原始/补偿后的视觉目标位置、速度指令、实际速度、ENU 位置、姿态角、
飞控模式和降落阶段；同名 JSON 保存本次参数及日志丢行信息。

结束飞行后正常停止节点，再分析对应 CSV：

```bash
python3 -m rl_vls.tools.analyze_flight_log runs/flight_logs/实际日志名.csv
```

分析结果保存在同目录的 `日志名_analysis/`：`summary.json` 包含超调、稳定时间、
速度跟踪误差和指令振荡统计；`flight.png` 包含位置、速度、姿态和阶段曲线。
曲线需要 `matplotlib`；未安装时仍可生成统计报告。

默认稳定条件为：水平误差 ≤0.3 m、水平实际速度 ≤0.2 m/s，连续至少 2 秒。
策略控制与自主 QLAND 分开分析；最后视觉误差表示交接前偏差，并非触地偏差。
缺失或过期的遥测不会用于判定稳定。

可通过 `-p flight_log_dir:=其他目录` 修改保存位置，或
`-p flight_log_enabled:=false` 关闭记录。更多分析选项见 `--help`。


对比多次飞行：

```bash
python3 -m rl_vls.tools.compare_flight_logs
```

结果保存在 `runs/flight_logs/comparison/`，包括对比表、曲线以及视觉尺度/速度响应拟合。
不同初始条件下的结果不能直接当作严格的策略排名。
