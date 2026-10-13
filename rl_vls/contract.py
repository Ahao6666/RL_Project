"""训练与部署共用的执行器/任务契约（单一事实来源）。

历史上训练环境用一套默认值、ROS 部署节点用另一套，导致同一个动作在两侧
对应不同的速度、不同的中性点和不同的终点，是 PPO 策略在 SITL 上表现差的主因。

本模块把“训练/部署必须一致”的量集中定义：

* 执行器：控制频率、水平/垂直速度限幅与映射、指令整形（加速度限制 + 接近段限速）、
  控制延迟、视觉尺度；
* 任务：对齐—减速—稳定交接的判据（交接高度、水平偏差、水平速度、保持时间）；
* 奖励权重与 PPO 折扣：仅训练侧使用，但与契约一起落盘便于复现。

``LandingEnv`` 的默认值、``train_ppo`` 保存的 ``env_config.json``、ROS 部署节点
读取的部署参数都以本模块为准，避免再次出现“训练一套、部署另一套”。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Contract:
    # -- 执行器（训练/部署必须逐项一致）--
    control_rate_hz: float = 20.0
    v_xy_max: float = 1.0
    v_down_max: float = 0.8
    v_up_max: float = 0.8
    # 指令低通漏波：0 关闭（<1.0 时与部署端一致地对指令做一阶漏波）。
    # 2026-10-13 起默认关闭：改用下面的加速度限制，因为一阶漏波在 0.5 Hz 上
    # 额外引入 ~0.2 s 等效延迟（真机辨识差值的 0.2 s 就是它）。
    cmd_filter_alpha: float = 1.0
    # -- 指令整形：把策略指令限在飞控可达范围内（训练/部署必须逐项一致）--
    # 真机辨识（2026-10-13，低频整段拟合）：
    #   MPC 通路指令 ≤ 0.4~0.8 m/s²  → 幅值比 0.9~1.4，等效延迟仅 0.15~0.25 s；
    #   RL 通路指令要求 4~10 m/s² → 实测速度斜率被削平在 ~1~1.3 m/s²，
    #   幅值只剩 0.40~0.53，多出 ~90° 相位（拟合成 +0.5 s 假延迟），
    #   闭环变成 0.5 Hz 继电振荡（周期 2 s），越过策略 0.65 s 的可控悬崖。
    # 因此必须限制指令变化率，并让接近段速度随偏差收缩，使真机保持在线性区。
    cmd_rate_max: float = 1.5            # 机体系速度指令最大变化率 (m/s²)
    approach_speed_gain: float = 1.0     # 接近段限速斜率 (1/s)：|v_xy| ≤ gain·d_xy + floor
    approach_speed_floor: float = 0.15   # 接近段限速下限 (m/s)
    # 环路延迟（步）。整形后飞控回到线性区（实测 0.2~0.25 s），故域收到 0.2~0.5 s；
    # 仍保留一点裕度覆盖尺度/风扰等未建模因素。
    action_delay_range: tuple = (4, 10)     # 步；20 Hz 下 0.20~0.50 s
    tau_xy_range: tuple = (0.6, 1.3)        # 水平速度响应时间常数 (s)，实测 0.5~1.2
    tau_z_range: tuple = (0.7, 1.2)         # 垂直速度响应时间常数 (s)
    visual_scale_range: tuple = (0.9, 1.1)  # 视觉尺度残差（部署已补偿主尺度后）
    descent_gate_radius: float = 0.5        # 水平偏差超过该值即抑制下降 (m)
    descent_gate_speed: float = 0.6         # 水平速度超过该值即抑制下降 (m/s)

    # -- 任务：对齐—减速—稳定交接（终点为 QLAND 交接高度）--
    # 2026-10-13 起从 2.0 m 改到 2.5 m（给 QLAND 接管留更多余量）；
    # 该值同时是训练回合的终点与部署端 auto_land_altitude，二者必须一致，
    # 改动后必须重新训练（否则策略的“完成点”与部署交接点错位）。
    handoff_altitude: float = 2.5           # 交接高度 (m)
    handoff_radius: float = 0.3             # 交接水平偏差上限 (m)
    handoff_speed_xy: float = 0.2           # 交接水平速度上限 (m/s)
    handoff_hold_s: float = 0.5             # 交接前需连续保持达标的时间 (s)
    h_init_range: tuple = (3.0, 8.0)        # 匹配部署习惯（悬停 5~10 m）并缩短回合
    xy_init_range: float = 3.0
    max_episode_time: float = 48.0          # 失败回合上限；需容纳长延迟下的缓慢对准与下降

    # -- 奖励权重（训练侧）--
    # 注意：不设“每步高度代价”。无界的时间惩罚会让策略为了中止回合而
    # 故意飞出视场（“自杀式”局部最优），比悬停更划算。下降改为有界进程奖励。
    w_pos: float = 1.0
    pos_cap: float = 4.0                    # 位置惩罚限幅：防止漂远后奖励发散
    w_vxy: float = 0.5
    w_desc: float = 1.0
    w_descent_misalign: float = 1.0
    low_alt_h: float = 3.0
    low_alt_speed_w: float = 2.0
    action_smooth_w: float = 0.1
    success_bonus: float = 50.0
    fail_base: float = 5.0
    fail_v: float = 2.0
    fail_xy: float = 2.0
    timeout_penalty: float = 10.0           # 超时（未完成）惩罚，鼓励完成而非拖延

    # -- 部署端视觉尺度：视觉位移 / 实际位移，输入除以该值转为物理米制 --
    # 2026-10-13 起视觉节点输出已修正为物理米制（实测 raw/真实 ≈ 0.95~1.02），
    # 故为 1.0。若再次出现尺度偏差，用飞行日志里的 raw/真实 比值设成这个数。
    visual_scale: float = 1.0

    # -- PPO --
    gamma: float = 0.995

    @property
    def dt(self) -> float:
        return 1.0 / self.control_rate_hz

    def env_kwargs(self) -> dict:
        """``LandingEnv`` 构造参数（不含 noise_cfg / seed）。"""
        return {
            "dt": self.dt,
            "tau_xy_range": self.tau_xy_range,
            "tau_z_range": self.tau_z_range,
            "max_episode_time": self.max_episode_time,
            "v_xy_max": self.v_xy_max,
            "v_down_max": self.v_down_max,
            "v_up_max": self.v_up_max,
            "cmd_filter_alpha": self.cmd_filter_alpha,
            "cmd_rate_max": self.cmd_rate_max,
            "approach_speed_gain": self.approach_speed_gain,
            "approach_speed_floor": self.approach_speed_floor,
            "h_init_range": self.h_init_range,
            "xy_init_range": self.xy_init_range,
            "handoff_altitude": self.handoff_altitude,
            "handoff_radius": self.handoff_radius,
            "handoff_speed_xy": self.handoff_speed_xy,
            "handoff_hold_s": self.handoff_hold_s,
            "descent_gate_radius": self.descent_gate_radius,
            "descent_gate_speed": self.descent_gate_speed,
            "visual_scale_range": self.visual_scale_range,
            "w_pos": self.w_pos,
            "pos_cap": self.pos_cap,
            "w_vxy": self.w_vxy,
            "w_desc": self.w_desc,
            "w_descent_misalign": self.w_descent_misalign,
            "low_alt_h": self.low_alt_h,
            "low_alt_speed_w": self.low_alt_speed_w,
            "action_smooth_w": self.action_smooth_w,
            "success_bonus": self.success_bonus,
            "fail_base": self.fail_base,
            "fail_v": self.fail_v,
            "fail_xy": self.fail_xy,
            "timeout_penalty": self.timeout_penalty,
            "action_delay_range": self.action_delay_range,
        }

    def deploy_kwargs(self) -> dict:
        """ROS 部署节点需要与训练保持一致的关键参数（键名即 ROS 参数名）。"""
        return {
            "control_rate_hz": self.control_rate_hz,
            "v_xy_max": self.v_xy_max,
            "v_down_max": self.v_down_max,
            "v_up_max": self.v_up_max,
            "cmd_filter_alpha": self.cmd_filter_alpha,
            "cmd_rate_max": self.cmd_rate_max,
            "approach_speed_gain": self.approach_speed_gain,
            "approach_speed_floor": self.approach_speed_floor,
            "visual_scale": self.visual_scale,
            "descent_gate_radius": self.descent_gate_radius,
            "descent_gate_speed": self.descent_gate_speed,
            "auto_land_altitude": self.handoff_altitude,
        }

    def to_json_obj(self, ppo: dict | None = None) -> dict:
        return {
            "schema_version": 1,
            "env": self.env_kwargs(),
            "deploy": self.deploy_kwargs(),
            "curriculum": {"easy": curriculum_easy(), "hard": curriculum_hard(self)},
            "ppo": dict({"gamma": self.gamma}, **(ppo or {})),
        }

    def save(self, path, ppo: dict | None = None) -> None:
        Path(path).write_text(
            json.dumps(self.to_json_obj(ppo), indent=2, ensure_ascii=False),
            encoding="utf-8")


CONTRACT = Contract()

# --------------------------------------------------------------------------- #
# 课程学习端点：从“起点已对准、下降宽松、不要求保持”逐步过渡到完整任务。
# EASY 让随机/初期策略也能拿到成功奖励（bootstrap），HARD 即契约的正式判据。
# 由 rl_vls.training.curriculum.CurriculumCallback 按评估成功率插值。
CURRICULUM_KEYS = ("xy_init_range", "h_init_range", "handoff_radius",
                   "handoff_speed_xy", "handoff_hold_s",
                   "descent_gate_radius", "descent_gate_speed",
                   "max_episode_time")
CURRICULUM_EASY = {
    "xy_init_range": 0.5,
    "h_init_range": (3.0, 5.0),
    "handoff_radius": 1.0,
    "handoff_speed_xy": 0.6,
    "handoff_hold_s": 0.0,
    "descent_gate_radius": 1.5,
    "descent_gate_speed": 0.8,
    "max_episode_time": 15.0,   # 早期失败回合短，提升样本效率并降低评估开销
}


def curriculum_easy() -> dict:
    return dict(CURRICULUM_EASY)


def curriculum_hard(contract: Contract = CONTRACT) -> dict:
    """HARD 端点直接取契约值，避免两处参数漂移。"""
    return {key: getattr(contract, key) for key in CURRICULUM_KEYS}
