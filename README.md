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

把 `vls_ros2_bridge` 拷入你的 ros2 工作空间 `src/` 下：

```bash
colcon build --packages-select vls_ros2_bridge
ros2 run vls_ros2_bridge landing_rl_node --ros-args \
    -p model_path:=/abs/path/checkpoints/ppo_landing_final.zip
```

注意：桥接节点中的坐标系转换和 VecNormalize 对接处标有 TODO，
需先在 SITL 上验证 `cmd_vel_unstamped` 的坐标系约定后再接入策略。

## 环境关键参数（`LandingEnv`）

- 动作：`[v_right, v_forward, v_down]`，归一化 [-1,1]，限幅见构造参数
- 观测：`p_cam(3, 带噪) + vel(3) + prev_action(3) + stale_flag(1)`，共 10 维
- 终止：触地（成功/坠毁按偏差与速度判定）、目标连续丢失 2 s、超时 40 s
- 奖励：`- (水平偏差 + 0.2*高度 + 0.1*水平速度) * dt` + 终局奖惩
