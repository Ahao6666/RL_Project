"""指令整形（接近段限速 + 加速度限制）的单元测试与训练/部署一致性测试。

背景：2026-10-13 真机低频辨识发现，策略输出的 ±1 m/s、~2 s 周期继电方波等效要求
4~10 m/s²，而飞控可达加速度只有约 1~2.5 m/s²，指令被削成慢三角波（幅值只剩
0.40~0.53、多出 ~90° 相位 → 拟合成 +0.5 s 假延迟），闭环变成 0.5 Hz 极限环。
因此训练环境与部署节点都必须对指令做同样的整形；本文件守住"逐项一致"。
"""
import unittest

import numpy as np

from rl_vls.contract import CONTRACT
from rl_vls.envs.landing_env import LandingEnv

try:
    from vls_ros2_bridge.vls_ros2_bridge.landing_rl_node import LandingRLNode
except ImportError:      # 无 ROS 环境时跳过节点侧一致性测试
    LandingRLNode = None

RATE = CONTRACT.cmd_rate_max
DT = CONTRACT.dt


def make_env(**kw):
    return LandingEnv(**kw)


class TestShapingMath(unittest.TestCase):
    def setUp(self):
        self.env = make_env()

    def test_defaults_from_contract(self):
        self.assertEqual(self.env.cmd_rate_max, CONTRACT.cmd_rate_max)
        self.assertEqual(self.env.approach_speed_gain, CONTRACT.approach_speed_gain)
        self.assertEqual(self.env.approach_speed_floor, CONTRACT.approach_speed_floor)

    def test_rate_limit_is_respected_every_step(self):
        """每步指令变化不超过 cmd_rate_max·dt。"""
        env = self.env
        prev = np.zeros(3)
        for _ in range(40):
            out = env._shape_command(np.array([1.0, -1.0, 0.8]), 10.0)  # 远距离不限速
            self.assertTrue(np.all(np.abs(out - prev) <= RATE * DT + 1e-9),
                            f"单步变化超限: {out - prev}")
            prev = out.copy()
        # 足够多步后应达到指令值
        self.assertAlmostEqual(prev[0], 1.0, places=6)
        self.assertAlmostEqual(prev[2], 0.8, places=6)

    def test_rate_limit_ramp_time(self):
        """从静止到 1 m/s 至少要 1/(rate·dt) 步（末步恰好达到）。"""
        env = self.env
        steps = int(np.ceil(1.0 / (RATE * DT)))
        for i in range(steps - 1):
            out = env._shape_command(np.array([1.0, 0.0, 0.0]), 10.0)
            self.assertLess(out[0], 1.0 - 1e-9, f"第 {i} 步不应已到 1.0")
        out = env._shape_command(np.array([1.0, 0.0, 0.0]), 10.0)
        self.assertAlmostEqual(out[0], 1.0, places=6)

    def test_approach_profile_caps_horizontal_speed(self):
        """接近段水平速度上限 = gain·d_xy + floor。"""
        for d in (0.0, 0.2, 0.5, 1.0, 3.0):
            env = make_env()
            cap = CONTRACT.approach_speed_gain * d + CONTRACT.approach_speed_floor
            for _ in range(60):     # 跑到稳态，暴露限速
                out = env._shape_command(np.array([1.0, 1.0, 0.0]), d)
            hs = float(np.hypot(out[0], out[1]))
            self.assertLessEqual(hs, min(cap, CONTRACT.v_xy_max * 2 ** 0.5) + 1e-6,
                                 f"d_xy={d}: 水平速度 {hs} 超过上限 {cap}")

    def test_approach_profile_keeps_direction(self):
        """限速只缩放幅值，不改变水平方向。"""
        env = self.env
        for _ in range(40):
            out = env._shape_command(np.array([0.6, -0.8, 0.0]), 0.2)
        self.assertGreater(out[0], 0.0)
        self.assertLess(out[1], 0.0)
        self.assertAlmostEqual(out[0] / out[1], 0.6 / -0.8, places=5)

    def test_vertical_channel_not_scaled_by_profile(self):
        """接近段剖面只限水平通道，垂直指令不被缩放（下降速度由门控管理）。"""
        env = self.env
        for _ in range(40):
            out = env._shape_command(np.array([0.0, 0.0, 0.8]), 0.0)
        self.assertAlmostEqual(out[2], 0.8, places=6)

    def test_never_exceeds_actuator_limits(self):
        rng = np.random.default_rng(0)
        env = self.env
        for _ in range(300):
            cmd = rng.uniform(-1.0, 1.0, 3) * np.array(
                [CONTRACT.v_xy_max, CONTRACT.v_xy_max, CONTRACT.v_down_max])
            d = rng.uniform(0.0, 4.0)
            out = env._shape_command(cmd, d)
            self.assertLessEqual(abs(out[0]), CONTRACT.v_xy_max + 1e-9)
            self.assertLessEqual(abs(out[1]), CONTRACT.v_xy_max + 1e-9)
            self.assertLessEqual(abs(out[2]), CONTRACT.v_down_max + 1e-9)

    def test_demand_stays_inside_fc_capability(self):
        """整形后的指令变化率必须落在飞控可达范围内（本次故障的根因约束）。"""
        env = self.env
        prev = np.zeros(3)
        peak = 0.0
        for _ in range(200):
            out = env._shape_command(np.array([1.0, -1.0, 0.8]), 0.5)
            peak = max(peak, float(np.abs(out - prev).max()) / DT)
            prev = out.copy()
        self.assertLessEqual(peak, RATE + 1e-6)
        # 实测飞控可达约 1~2.5 m/s²，契约值必须落在其中
        self.assertLessEqual(RATE, 2.5)


class TestEnvIntegration(unittest.TestCase):
    def test_env_ramps_command_instead_of_jumping(self):
        """环境中指令也必须经整形后再进入延迟缓冲。"""
        env = make_env(action_delay_range=(0, 0))
        env.reset(seed=0)
        env.pos = np.array([3.0, 3.0, 6.0])
        v_prev = env.vel.copy()
        env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        step_v = np.abs(env.vel - v_prev).max()
        self.assertLessEqual(step_v, RATE * DT + 1e-6,
                             "指令未整形：速度一步跳变过大")

    def test_shaping_state_resets_between_episodes(self):
        env = make_env()
        env.reset(seed=1)
        for _ in range(20):
            env._shape_command(np.array([1.0, 1.0, 0.8]), 5.0)
        self.assertGreater(np.abs(env._v_shaped).max(), 0.5)
        env.reset(seed=2)
        self.assertEqual(np.abs(env._v_shaped).max(), 0.0)

    def test_filter_off_by_default(self):
        self.assertEqual(CONTRACT.cmd_filter_alpha, 1.0)


class TestNodeEnvParity(unittest.TestCase):
    """部署节点与训练环境的整形必须逐项一致（防再次出现训练/部署漂移）。"""

    def setUp(self):
        if LandingRLNode is None:
            self.skipTest("无 ROS 环境，跳过节点侧一致性测试")
        self.node = LandingRLNode.__new__(LandingRLNode)   # 不启动 ROS 节点
        self.node.dt = DT
        self.node._shape_rate_max = CONTRACT.cmd_rate_max
        self.node._shape_gain = CONTRACT.approach_speed_gain
        self.node._shape_floor = CONTRACT.approach_speed_floor
        self.node._v_shaped = np.zeros(3)
        self.env = make_env()

    def test_identical_trajectories(self):
        rng = np.random.default_rng(7)
        cmds = [np.array([1.0, -1.0, 0.8]), np.array([0.0, 0.0, -0.8]),
                np.array([-0.5, 0.9, 0.0])]
        ds = [3.0, 0.05, 1.0, 0.3, 8.0]
        for i in range(120):
            cmd = cmds[i % len(cmds)]
            d = ds[i % len(ds)]
            if i % 13 == 0:
                cmd = rng.uniform(-1.0, 1.0, 3)
            a = self.env._shape_command(cmd, d)
            b = self.node._shape_command(cmd, d)
            np.testing.assert_allclose(a, b, atol=1e-12,
                                       err_msg=f"第 {i} 步不一致: {a} vs {b}")

    def test_zero_speed_start_matches(self):
        self.node._v_shaped = np.zeros(3)
        for _ in range(30):
            a = self.env._shape_command(np.array([1.0, 0.0, -0.8]), 0.1)
            b = self.node._shape_command(np.array([1.0, 0.0, -0.8]), 0.1)
        np.testing.assert_allclose(a, b, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
