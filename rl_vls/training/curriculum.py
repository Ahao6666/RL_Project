"""课程学习回调：按评估成功率把任务从“易”逐步推进到“难”。

动机
----
正式任务的判据是“水平偏差 ≤0.3 m 且水平速度 ≤0.2 m/s 并保持 0.5 s 后交接”，
再加上“未对准不允许下降”的门控，从零开始的策略很难摸到第一次成功，
成功奖励（+50）长期拿不到，价值函数学不起来（实测 30 万步仍 0% 成功）。

做法
----
在若干个可调维度上从易到难线性插值（端点见 ``rl_vls.contract``）：

============== ================== ==================
维度            EASY（起点）        HARD（正式判据）
============== ================== ==================
初始水平偏差    ±0.5 m             ±3.0 m
初始高度        3~5 m              3~8 m
交接半径        1.0 m              0.3 m
交接水平速度    0.6 m/s            0.2 m/s
交接保持时间    0 s                0.5 s
下降门控半径    1.5 m              0.5 m
下降门控速度    0.8 m/s            0.6 m/s
回合上限        15 s               48 s
============== ================== ==================

EASY 下起点本身就算“对准”，门控几乎不限制下降，因此初期策略只要学会“下降”
就能成功，快速拿到成功奖励；随后每隔 ``eval_freq`` 步评估一次成功率，
达标（``advance_at``）就把进度提高 ``step_size``。

**推进判据的口径很重要**：成功率必须在“当前训练难度”下测。早期版本在 HARD
难度上测成功率，结果策略解不了 HARD 时推进永不触发（实测 1e6 步 progress 恒等于
下限、训练 61%@p=0.5 而 HARD 只有 1~9%，训练域与评测域脱节）。本版本评估难度
跟随训练难度，另每 ``hard_eval_every`` 次在 HARD 上单独评估一次（``hard/*``
标量），用于横向比较与保存最优模型。

**下限必须能升到 HARD**：除“按成功率提前推进”外还有一条时间下限，
``_floor`` 让它在 ``min_progress_frac × total_timesteps`` 步内线性升到 1.0，
其余预算全部用于在正式判据上打磨。旧版本下限封顶 0.5（``0.5·t/T``），一旦
成功率始终达不到 ``advance_at``，训练难度就停在 0.5、结束时从没在 HARD 上训练过——
这正是“HARD 1%/9%”的另一半原因。
"""
from __future__ import annotations

import copy
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from rl_vls.contract import curriculum_easy, curriculum_hard
from rl_vls.envs.landing_env import LandingEnv

KEYS = ("xy_init_range", "h_init_range", "handoff_radius", "handoff_speed_xy",
        "handoff_hold_s", "descent_gate_radius", "descent_gate_speed",
        "max_episode_time")


def _inner_envs(vec_env) -> list:
    """解包 VecNormalize / DummyVecEnv / Monitor，取到 LandingEnv 列表。"""
    envs = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
    out = []
    for env in getattr(envs, "envs", []):
        while not isinstance(env, LandingEnv) and hasattr(env, "env"):
            env = env.env
        if isinstance(env, LandingEnv):
            out.append(env)
    return out


def _interp(a, b, p):
    if isinstance(a, (tuple, list)):
        return tuple(x + (y - x) * p for x, y in zip(a, b))
    return a + (b - a) * p


class CurriculumCallback(BaseCallback):
    """课程学习 + 一站式评估回调。

    每 ``eval_freq`` 步评估一次，**评估难度跟随当前训练难度**，用它测到的成功率
    驱动课程推进（这是关键：早期版本在 HARD 难度上测成功率，策略解不了 HARD 时
    推进永不触发，训练域与评测域脱节）。

    推进有两条路：成功率达标（``advance_at``）就 +``step_size``；此外还有一条
    时间下限 ``_floor``（在 ``min_progress_frac`` 比例的预算内升到 HARD），
    保证无论学习快慢，后段预算都在正式判据上训练。

    另每 ``hard_eval_every`` 次额外在 HARD（契约默认）难度上评估一次，记录
    ``hard/*`` 标量并据此保存 ``best_model``（跨阶段可比），同时把本轮的
    VecNormalize 与最优模型成套保存，避免热启动时归一化错配。
    """

    def __init__(self, eval_env, total_timesteps: int, *, eval_freq: int = 20_000,
                 n_eval_episodes: int = 20, advance_at: float = 0.4,
                 step_size: float = 0.25, min_progress_frac: float = 0.5,
                 hard_eval_every: int = 5, best_model_save_path: str | None = None,
                 vecnormalize=None, enabled: bool = True, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.vecnormalize = vecnormalize
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.advance_at = advance_at
        self.step_size = step_size
        self.min_progress_frac = min_progress_frac
        self.hard_eval_every = max(1, hard_eval_every)
        self.best_model_save_path = best_model_save_path
        self.best_mean_reward = -np.inf
        self.enabled = enabled
        self.easy = curriculum_easy()
        self.hard = curriculum_hard()
        # 关闭课程时直接钉在 HARD（即契约默认），不做难度插值
        self.progress = 0.0 if enabled else 1.0
        self._next_eval = eval_freq
        self._eval_count = 0

    def params(self, progress: float) -> dict:
        # 端点直接返回契约值：插值会把 0.3 算成 0.30000000000000004，
        # 收尾阶段希望训练难度与部署判据逐位一致。
        if progress <= 0.0:
            return dict(self.easy)
        if progress >= 1.0:
            return dict(self.hard)
        return {key: _interp(self.easy[key], self.hard[key], progress) for key in KEYS}

    def _floor(self) -> float:
        """难度下限：在 ``min_progress_frac × total_timesteps`` 步内线性升到 HARD。

        上限必须是 1.0：否则策略可能在预算结束前都没在正式判据上训练过
        （旧版 ``0.5·t/T`` 封顶 0.5，1e6 步结束时训练难度仅 0.5，HARD 成功率 1%）。
        """
        if not self.total_timesteps:
            return 0.0
        span = max(1.0, self.total_timesteps * self.min_progress_frac)
        return min(1.0, self.num_timesteps / span)

    @staticmethod
    def _set(vec_env, params: dict) -> None:
        for env in _inner_envs(vec_env):
            for key, value in params.items():
                setattr(env, key, value)

    def _evaluate(self) -> tuple:
        """跑一遍评估，返回 (mean_reward, mean_len, success_rate)。"""
        if isinstance(self.eval_env, VecNormalize) and isinstance(self.training_env, VecNormalize):
            self.eval_env.obs_rms = copy.deepcopy(self.training_env.obs_rms)
        rewards, lengths, wins = [], [], 0
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done, info, ep_len, ep_reward = False, {}, 0, 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rew, dones, infos = self.eval_env.step(action)
                done = bool(dones[0])
                info = infos[0]
                ep_reward += float(rew[0])
                ep_len += 1
            rewards.append(ep_reward)
            lengths.append(ep_len)
            wins += info.get("landing") == "success"
        n = max(1, self.n_eval_episodes)
        return float(np.mean(rewards)), float(np.mean(lengths)), wins / n

    def _on_training_start(self) -> None:
        current = self.params(self.progress)
        self._set(self.training_env, current)
        self._set(self.eval_env, current)

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True
        self._next_eval = self.num_timesteps + self.eval_freq
        self._eval_count += 1

        # 1) 按“当前训练难度”评估 -> 驱动课程推进
        current = self.params(self.progress)
        self._set(self.training_env, current)
        self._set(self.eval_env, current)
        mean_reward, mean_len, rate = self._evaluate()

        if self.enabled:
            if rate >= self.advance_at:
                self.progress = min(1.0, self.progress + self.step_size)
            self.progress = max(self.progress, self._floor())
            current = self.params(self.progress)
            self._set(self.training_env, current)
            self._set(self.eval_env, current)

        log = self.model.logger.record
        log("eval/mean_reward", mean_reward)
        log("eval/mean_ep_length", mean_len)
        log("eval/success_rate", rate)
        log("curriculum/progress", self.progress)

        # 2) 每 hard_eval_every 次额外在 HARD 难度上评估（跨阶段可比 + 选最优模型）
        if self._eval_count % self.hard_eval_every == 0:
            self._set(self.eval_env, self.hard)
            hard_reward, hard_len, hard_rate = self._evaluate()
            log("hard/mean_reward", hard_reward)
            log("hard/mean_ep_length", hard_len)
            log("hard/success_rate", hard_rate)
            if self.best_model_save_path is not None and hard_reward > self.best_mean_reward:
                self.best_mean_reward = hard_reward
                os.makedirs(self.best_model_save_path, exist_ok=True)
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                if self.vecnormalize is not None:
                    self.vecnormalize.save(
                        os.path.join(self.best_model_save_path, "vecnormalize.pkl"))
            self._set(self.eval_env, current)  # 恢复当前难度
            if self.verbose:
                print(f"[curriculum] HARD eval: reward={hard_reward:.1f} "
                      f"len={hard_len:.0f} success={hard_rate:.2f}")

        if self.verbose:
            print(f"[curriculum] step={self.num_timesteps} success={rate:.2f} "
                  f"progress={self.progress:.2f} "
                  f"xy_init={current['xy_init_range']:.2f} "
                  f"r_handoff={current['handoff_radius']:.2f} "
                  f"hold={current['handoff_hold_s']:.2f} "
                  f"t_max={current['max_episode_time']:.0f}")
        return True
