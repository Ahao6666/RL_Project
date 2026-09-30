# SITL 闭环验证部署指南

将阶段 1 训练的 PPO 策略接入 docker 仿真（ArduPilot SITL + MAVROS + Gazebo）进行闭环降落验证。

- 策略模型：`checkpoints/ppo_landing_final.onnx`（由 `rl_vls/tools/export_onnx.py` 从
  SB3 `.zip` 导出，容器内只需 `onnxruntime`，不依赖 torch/sb3/numpy2；
  与 SB3 `predict(deterministic=True)` 输出误差 < 1e-6）
- 观测归一化：`checkpoints/obs_norm.npz`（由 `rl_vls/tools/export_vecnormalize.py` 从 `vecnormalize.pkl` 导出，跨 numpy 版本可用）
- 桥接节点：`vls_ros2_bridge/vls_ros2_bridge/landing_rl_node.py`
  （`model_path` 以 `.onnx` 结尾走 onnxruntime，以 `.zip` 结尾走 SB3）

前置实测结论（已反映在代码中）：

| 项目 | 实测值 |
| --- | --- |
| `cmd_vel_unstamped` 坐标系 | 本地 ENU（世界固定系，z 向上为正） |
| 水平速度响应 tau | ≈1.3 s |
| 垂直速度响应 tau | ≈0.95 s |

---

## 第 0 步：预检（仿真启动后先做）

```bash
# a. 确认 ardupilot 容器里能同时看到视觉与飞控两类话题
docker exec ardupilot_image-v1.1.5 bash -c \
  'source /home/insky/mavros_ws/install/setup.bash && ros2 topic list | grep -E "h_marker|mavros/state"'

# b. 检查 /h_marker/position 的 QoS
#    若 Reliability 为 BEST_EFFORT，需将桥接节点的订阅改为 qos_profile_sensor_data
docker exec ardupilot_image-v1.1.5 bash -c \
  'source /home/insky/mavros_ws/install/setup.bash && ros2 topic info /h_marker/position -v | head -20'
```

**故障排查**：若 a 中看不到 `/h_marker/position`，说明视觉容器与飞控容器之间 DDS 不通。
检查两个容器的网络模式（`docker inspect <容器> | grep NetworkMode`）与 `ROS_DOMAIN_ID` 是否一致。

## 第 1 步：准备容器环境

```bash
# 拷入模型（ONNX）、归一化统计量、桥接节点
docker cp checkpoints/ppo_landing_final.onnx ardupilot_image-v1.1.5:/tmp/
docker cp checkpoints/obs_norm.npz ardupilot_image-v1.1.5:/tmp/
docker cp vls_ros2_bridge/vls_ros2_bridge/landing_rl_node.py ardupilot_image-v1.1.5:/tmp/

# 容器内安装推理依赖（仅 onnxruntime，体积小、无 ABI 风险）
docker exec ardupilot_image-v1.1.5 pip install onnxruntime
```

> 若模型有更新：宿主机重新导出
> `.venv/bin/python rl_vls/tools/export_onnx.py checkpoints/ppo_landing_final.zip checkpoints/ppo_landing_final.onnx`
> 再 `docker cp` 进容器即可。

## 第 2 步：起飞就位

1. 手动控制无人机起飞，悬停在 H 靶标上方附近：
   - 高度 **5~10 m**，水平偏差 **< 5 m**（需在训练分布内）
2. 确认视觉话题持续输出：
   ```bash
   docker exec ardupilot_image-v1.1.5 bash -c \
     'source /home/insky/mavros_ws/install/setup.bash && ros2 topic hz /h_marker/position'
   ```
3. 切换到 `GUIDED` 模式（垂起固定翼用 `QGUIDED`）：
   ```bash
   docker exec ardupilot_image-v1.1.5 bash -c \
     'source /home/insky/mavros_ws/install/setup.bash && \
      ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode "{custom_mode: QGUIDED}"'
   ```

## 第 3 步：启动桥接节点

```bash
python3 landing_rl_node.py --ros-args \
	-p model_path:=ppo_landing_final.onnx \  
	-p vecnormalize_path:=obs_norm.npz \
	-p controller:=hybrid
```

节点参数（均有默认值，可按需覆盖）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `controller` | `"rl"` | `"rl"` 策略网络 / `"p"` P 比例控制 baseline（无需模型，用于链路测试） |
| `kp_xy` / `p_descent_vel` / `p_align_radius` / `p_min_alt` | `0.3` / `0.3` / `0.5` / `0.3` | P 控制：水平增益 (1/s)、下降速度、对准半径、停降高度 |
| `model_path` | `""` | 策略路径，`.onnx`（推荐）或 SB3 `.zip`，不填则只发零速度 |
| `vecnormalize_path` | `""` | 观测归一化 `.npz`（或训练同版本 numpy 下的 `.pkl`） |
| `marker_flip_y` | `true` | 话题 y 轴镜像补偿（本系统 y 轴镜像，见 yaw 标定记录） |
| `control_rate_hz` | `20.0` | 控制频率 |
| `marker_timeout` | `0.5` | 视觉超时 (s)，超时发零速度悬停 |
| `v_xy_max` / `v_down_max` / `v_up_max` | `2.0` / `1.5` / `0.8` | 速度限幅（须与训练一致） |

节点安全逻辑（无需人工干预）：

- 仅当 **已解锁 + GUIDED/QGUIDED 模式 + 视觉消息新鲜（< 0.5 s）** 三个条件同时满足时才发布策略指令
- 任一条件不满足即发布零速度（悬停）

## 安全措施

- 遥控器/地面站随时准备切回 `QLOITER` 接管
- 首次试飞在 **10 m 以上高度** 启动节点，观察是否朝靶标收敛，异常立即切走
- 策略触地速度训练均值 ~0.4 m/s，仿真可接受；上真机前需进一步压低

## 验证记录模板

试飞后记录以下信息便于后续迭代：

- [ ] 启动高度 / 水平偏差
- [ ] 是否收敛到靶标中心（着陆水平偏差）
- [ ] 下降节奏（匀速/阶梯/振荡）
- [ ] 触地速度
- [ ] 是否出现视觉丢失悬停
- [ ] 异常现象与接管时刻
