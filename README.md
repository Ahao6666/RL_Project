# RL_for_VLS — 基于强化学习的视觉着陆优化训练

用 PPO 训练无人机跟踪并降落到地面 H 靶标上。观测为 `/h_marker/position`
（`geometry_msgs/Point`，H 标中心在相机光轴系下的相对位置：z 前、x 右、y 下，单位米），
训练阶段用真值 + 误差模型模拟，实机验证时切换为真实视觉估计。
控制输出为 MAVROS 速度指令 `/mavros/setpoint_velocity/cmd_vel_unstamped`（ArduPilot GUIDED 模式）。

## 阶段路线

| 阶段 | 内容 | 退出标准 |
| --- | --- | --- |
| 0 | ROS2 + MAVROS + AP SITL 链路打通；P 控制器视觉伺服 baseline | 手动飞行中能闭环降落 |
| 1 | 纯 Python Gym 环境训练 PPO（本项目主体） | 仿真交接成功率 > 95% |
| 2 | `vls_ros2_bridge` 接入 SITL 闭环 | SITL 中对准—减速—稳定交接 QLAND |
| 3 | Domain randomization（风扰、延迟、姿态耦合、噪声随机化） | 扰动下成功率不显著下降 |
| 4 | 实机验证（观测换真实视觉，策略不变） | 实机安全着陆 |

## 目录结构

```
rl_vls/
├── contract.py              # 训练/部署共享契约（速度映射/滤波/延迟/尺度/门控/交接判据）
├── envs/landing_env.py      # Gymnasium 环境：质点动力学 + 对齐-减速-稳定交接任务
├── vision/noise_model.py    # 视觉误差模型（距离/离轴相关噪声 + 丢帧）
└── training/
    ├── train_ppo.py         # PPO 训练入口（VecNormalize + Eval/Checkpoint 回调）
    └── evaluate.py          # 交接成功率/偏差/速度/稳定时间/指令振荡统计
vls_ros2_bridge/             # 阶段 2 的 ROS2 ament_python 包
└── vls_ros2_bridge/landing_rl_node.py
```

## 训练/部署契约

`rl_vls/contract.py` 是训练与部署共用的单一事实来源，统一了：速度映射与限幅、
控制频率、指令低通滤波、控制延迟、视觉尺度、下降门控，以及“对齐—减速—稳定交接”
的终点判据。`LandingEnv` 默认值、`train_ppo` 保存的 `checkpoints/env_config.json`、
ROS 部署节点读取的 `deploy` 段都以此为准。

历史问题是训练与部署各用一套参数：同一动作在两侧对应不同速度与中性点、策略最擅长的
低空减速段从未执行、真实延迟与视觉尺度未建模，导致 SITL 上振荡且拖偏。修改契约后
**必须重新训练**，旧模型不适用。

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 重新训练（默认 1e6 步；含延迟/滤波/尺度随机化与下降门控）
python -m rl_vls.training.train_ppo --timesteps 1000000
tensorboard --logdir runs/

# 评估：固定种子、多种子重复，关注交接成功率/偏差/速度/稳定时间
python -m rl_vls.training.evaluate \
    --model checkpoints/ppo_landing_final.zip \
    --vecnormalize checkpoints/vecnormalize.pkl \
    --env-config checkpoints/env_config.json --episodes 100

# 部署前把新模型导出为 ONNX 与 .npz（**每次重训后必做**，否则节点会加载旧 ONNX）
python -m rl_vls.tools.export_onnx checkpoints/ppo_landing_final.zip checkpoints/ppo_landing_final.onnx
python -m rl_vls.tools.export_vecnormalize checkpoints/vecnormalize.pkl checkpoints/obs_norm.npz
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
- 任务：把飞行器对齐、减速，并在交接高度（默认 2.5 m）满足水平偏差 ≤0.3 m 且
  水平速度 ≤0.2 m/s 并连续保持一段时间视为成功；终点不再是触地，与真实
  QLAND 交接流程一致
- 动力学随机化：tau 每回合随机采样（默认水平 0.6~1.3 s、垂直 0.7~1.2 s，取真机辨识值）
- 初始条件：高度 3~8 m、水平偏差 ±3 m，且保证目标初始在视场内
- 终止：到达交接高度（成功/未达标本按偏差与速度判定）；超时 48 s 按截断处理并
  计惩罚。视觉丢失不再提前终止，而是输出零速度悬停（与部署端一致），避免
  “飞出视场提前中止”的损回报局部最优
- 控制延迟随机化：`action_delay_range`（默认 **4~10 步**，20 Hz 下 0.20~0.50 s）。
  **口径（2026-10-13 真机低频辨识）**：同一机体上“指令→实测速度”的等效延迟，
  MPC 通路（指令 ≤0.4~0.8 m/s²）只有 0.15~0.25 s，而 RL 通路不加整形时高达
  0.70~0.75 s——因为飞控可达加速度只有约 1~2.5 m/s²，策略输出的 ±1 m/s、
  ~2 s 周期方波等效要求 4~10 m/s²，被削成慢三角波（幅值只剩 0.40~0.53、多出
  ~90° 相位，拟合成 +0.5 s 假延迟），闭环于是变成 0.5 Hz 极限环，越过策略
  0.65 s 的可控悬崖。**加入指令整形后飞控回到线性区**，故训练域收到 0.2~0.5 s。
- 指令整形（`cmd_rate_max` / `approach_speed_gain` / `approach_speed_floor`）：
  1) 加速度限制：机体系速度指令变化率 ≤ `cmd_rate_max`（默认 1.5 m/s²）；
  2) 接近段速度剖面：`|v_xy| ≤ gain·d_xy + floor`（默认 1.0 /s、0.15 m/s），
     使“到交接高度还很快”结构上不可能发生。训练与部署逐项一致，由
     `tests/test_command_shaping.py` 的一致性测试守护。
- 指令低通滤波：`cmd_filter_alpha` 默认 **1.0（关闭）**。一阶低通在 0.5 Hz 上会
  额外引入 ~0.2 s 等效延迟（真机辨识差值的 0.2 s 就是它），已改由上面的加速度
  限制整形替代；若回退到 <1.0，训练与部署仍保持同一顺序（整形→滤波→门控）
- 视觉尺度残差随机化：`visual_scale_range`（默认 0.9~1.1，部署已补偿主尺度）
- 下降门控：未对准（偏差 >`descent_gate_radius`）或横向速度 >`descent_gate_speed`
  时抑制下降，训练与部署一致；采用平方软因子，轻微偏差/摆动几乎不压制下降，
  避免高空因门控过紧而无法在限定时间内下降
- 奖励：`- 1.0*min(水平偏差,4m)*dt - 0.5*水平速度*dt`（**不设每步高度代价**，
  避免策略为中止回合而故意飞出视场；位置项限幅以稳住奖励尺度）
  + 对准时下降进程奖励 + 未对准时下降惩罚 + 接近段速度惩罚（<3 m 随高度线性增大）
  + 动作平滑惩罚（默认 0.1）+ 终局奖惩（成功 +50、未达标按偏差/速度惩罚、
  超时截断并 −10）

### 课程学习（默认开启）

直接从“水平偏差 ≤0.3 m + 保持 0.5 s + 未对准不许下降”的完整判据开始训练很难
bootstrap（实测 30 万步仍 0% 成功，策略只会停在目标附近）。训练默认启用课程学习
（`rl_vls/training/curriculum.py`），每 1 万步评估一次成功率，达标就沿着若干维度
从易到难插值推进：

| 维度 | 起点（EASY） | 终点（HARD，即正式判据） |
| --- | --- | --- |
| 初始水平偏差 | ±0.5 m | ±3.0 m |
| 初始高度 | 3~5 m | 3~8 m |
| 交接半径 | 1.0 m | 0.3 m |
| 交接水平速度 | 0.6 m/s | 0.2 m/s |
| 交接保持时间 | 0 s | 0.5 s |
| 下降门控半径 / 速度 | 1.5 m / 0.8 m/s | 0.5 m / 0.6 m/s |
| 回合上限 | 15 s | 48 s |

EASY 下起点本身就算“已对准”，门控基本不限制下降，策略只要学会“下降”即可拿到
成功奖励，从而快速 bootstrap；随后逐步加严。`--no-curriculum` 可关闭。

难度推进有**两条路**，缺一不可：

1. **成功率达标推进**（主）：评估难度**跟随当前训练难度**，当前难度下成功率
   ≥ `--advance-at`（默认 0.4）就 +0.25。注意判据必须在“当前训练难度”下测——
   早期版本在 HARD 上测成功率，策略解不了 HARD 时推进永不触发，训练域与评测域
   长期脱节（实测 1e6 步：训练难度 61% 成功，HARD 只有 1~9%）。
2. **时间下限**（兜底）：`_floor` 在 `--floor-frac`（默认 0.5）比例的预算内线性
   把难度下限升到 HARD，此后剩余预算全部在正式判据上打磨。旧版下限封顶 0.5
   （`0.5·t/T`），一旦成功率到不了阈值，1e6 步结束时**从没在 HARD 上训练过**。

另外每 `--hard-eval-every` 次评估会在 HARD 全判据上单独评一次，记到 `hard/*`
标量并据此保存 `best_model`（与 `best/vecnormalize.pkl` 成套）。评估间隔/回合数由
`--eval-freq`、`--eval-episodes` 控制（默认 20k 步 / 20 回合）——两者决定评估开销，
回合跑满时会明显拖低 `time/fps`（该值是含评估的累计平均，不是训练吞吐）。

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


## MPC 模型与当前 SITL 参数

MPC 使用带指令延迟队列的一阶速度模型，默认预测 60 步（20 Hz 下为 3 秒）。
按当前飞行日志调整的 ROS 默认值如下；独立 `LandingMPC` 类的视觉尺度默认仍为 1.0，
适用于真值米制的 Python 仿真。

| 参数 | ROS 默认值 | 含义 |
| --- | --- | --- |
| `mpc_tau_xy` / `mpc_tau_z` | `0.6` / `0.4` | 速度响应时间常数，s |
| `mpc_delay_xy` / `mpc_delay_z` | `0.3` / `0.15` | 模型中的等效指令延迟，s |
| `mpc_visual_scale` | `1.0` | 视觉位移/实际位移比例；输入除以该值转为模型米制（2026-10-13 起视觉已是物理米制） |
| `mpc_descent_speed` | `0.3` | 最大下降参考速度，m/s |
| `mpc_align_radius` / `mpc_align_speed` | `0.5` / `0.2` | 允许下降的水平误差/速度范围，m 与 m/s |
| `mpc_max_accel_xy` / `mpc_max_accel_z` | `0.4` / `0.4` | 速度指令变化率上限，m/s² |
| `mpc_position_weight_xy` / `mpc_velocity_weight_xy` | `2.0` / `4.0` | 水平位置/速度代价权重 |

MPC 优先使用新鲜 ENU 实际速度旋转到水平机体系，以匹配补偿后的视觉观测；
没有对应遥测时回退到 FLU 速度。下降同时考虑水平偏差和速度，避免高速掠过靶标时下降。
日志新增校准后的 `mpc_position_*`、`mpc_velocity_*` 和 `mpc_descent_gate`，便于诊断。
自动 QLAND 的 2.5 m 阈值仍使用原始尺度的补偿后视觉高度。

尺度和时间常数是当前场景的近似拟合，需要随视觉标定、飞控参数变化重新校验。
视觉模块本身的尺度标定应优先修正；输入已是物理米制时 `mpc_visual_scale` 为 1.0
（2026-10-13 视觉节点修好后即为 1.0；若再次出现尺度偏差，用飞行日志里
视觉位移/实际位移 的回归斜率设回该值）。
对比多次飞行及拟合诊断：

```bash
python3 -m rl_vls.tools.compare_flight_logs
```

输出 `runs/flight_logs/comparison/` 下的 Markdown/JSON 报告和曲线（绘图需 matplotlib）。
初始条件、下降速度、速度限幅和 QLAND 交接高度应保持一致后再比较控制器。
