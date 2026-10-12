# RL_for_VLS — 基于强化学习的视觉着陆优化训练

用 PPO 训练无人机跟踪并降落到地面 H 靶标上。观测为 `/h_marker/position`
（`geometry_msgs/Point`，H 标中心在相机光轴系下的相对位置：z 前、x 右、y 下，单位米），
训练阶段用真值 + 误差模型模拟，实机验证时切换为真实视觉估计。
控制输出为 MAVROS 速度指令 `/mavros/setpoint_velocity/cmd_vel_unstamped`（ArduPilot GUIDED 模式）。

## 阶段路线

| 阶段 | 内容 | 退出标准 |
| --- | --- | --- |
| 0 | ROS2 + MAVROS + AP SITL 链路打通；P 控制器视觉伺服 baseline | 手动飞行中能闭环降落 |
| 1 | 纯 Python Gym 环境训练 PPO（本项目主体） | 仿真成功率 > 95% |
| 2 | `vls_ros2_bridge` 接入 SITL 闭环 | SITL 中策略稳定降落 |
| 3 | Domain randomization（风扰、延迟、姿态耦合、噪声随机化） | 扰动下成功率不显著下降 |
| 4 | 实机验证（观测换真实视觉，策略不变） | 实机安全着陆 |

## 目录结构

```
rl_vls/
├── envs/landing_env.py      # Gymnasium 环境：质点动力学 + FOV 约束 + 着陆奖励
├── vision/noise_model.py    # 视觉误差模型（距离/离轴相关噪声 + 丢帧）
└── training/
    ├── train_ppo.py         # PPO 训练入口（VecNormalize + Eval/Checkpoint 回调）
    └── evaluate.py          # 成功率 / 着陆偏差 / 触地速度统计
vls_ros2_bridge/             # 阶段 2 的 ROS2 ament_python 包
└── vls_ros2_bridge/landing_rl_node.py
```

## 快速开始（阶段 1）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m rl_vls.training.train_ppo --timesteps 500000
tensorboard --logdir runs/

python -m rl_vls.training.evaluate \
    --model checkpoints/ppo_landing_final.zip \
    --vecnormalize checkpoints/vecnormalize.pkl
```

## 阶段 2 使用（ROS2 Humble）

控制节点可以直接在宿主机运行，只需配置 ROS2/Python 依赖，并确保话题收发及
MAVROS 服务通信正常。

SITL 闭环验证的完整部署流程（预检、运行环境、起飞就位、节点启动）
见 **[docs/SITL_DEPLOYMENT.md](docs/SITL_DEPLOYMENT.md)**。

桥接节点也可作为普通 ament_python 包使用：把 `vls_ros2_bridge` 拷入 ros2 工作空间
`src/` 下 `colcon build` 后 `ros2 run vls_ros2_bridge landing_rl_node`。

## 环境关键参数（`LandingEnv`）

- 动作：`[v_right, v_forward, v_down]`，归一化 [-1,1]，限幅见构造参数
- 观测：`p_cam(3, 带噪) + vel(3) + prev_action(3) + stale_flag(1)`，共 10 维
- 动力学随机化：tau 每回合随机采样，水平 0.5~2.0 s、垂直 0.4~1.5 s
  （按 SITL 阶跃实测 tau_xy≈1.3s、tau_z≈0.95s 两侧留余量；`(t, t)` 可固定值做压力测试）
- 初始条件：高度 3~20 m、水平偏差 ±5 m，且保证目标初始在视场内
- 终止：触地（成功/坠毁按偏差与速度判定）、目标连续丢失 2 s、超时 40 s
- 奖励：`- (水平偏差 + 0.2*高度 + 0.1*水平速度) * dt` + 低空速度惩罚（<2 m 随高度线性增大）
  + 动作平滑惩罚（`- w·||a_t − a_{t−1}||²`，抗振荡，默认关闭，训练时由 `--action-smooth-w` 开启）
  + 终局奖惩
- 控制延迟随机化：`action_delay_range` 每回合采样（步），模拟视觉+通信链路延迟

## 飞行日志

ROS2 控制节点默认通过独立的 `flight_logger.py` 模块异步记录 CSV，保存在
`runs/flight_logs/`；同名 JSON 保存运行参数、坐标约定和丢行统计。
日志记录视觉原始/补偿位置、实际发布的速度指令、FLU/ENU 实际速度、ENU 位置、
姿态角，以及模式、武装状态、自动降落阶段和遥测新鲜度。
正常停止节点后运行：

```bash
python3 -m rl_vls.tools.analyze_flight_log runs/flight_logs/实际日志名.csv
```

输出 `summary.json` 和 `flight.png`（绘图需要 matplotlib），包括水平超调、
持续稳定时间、速度跟踪 RMSE、指令变化率/变号次数，以及遥测曲线。
超调按初始航向坐标系计算，目标为零；初始偏差 ≤0.05 m 时不计算超调百分比。
稳定默认要求水平误差 ≤0.3 m、水平速度 ≤0.2 m/s，且保持至少 2 秒直至该控制段结束。
失活、时间中断和自主降落会划分控制段，过期/缺失遥测打断稳定判定。
最后有效视觉偏差只代表策略控制阶段，无法代替视觉丢失后的最终落点偏差。

`flight_log_enabled`（默认 true）与 `flight_log_dir`（默认 `runs/flight_logs`）控制记录。
日志队列有界，写入失败不阻断控制；分析阈值与无图模式可通过 `--help` 查看。
