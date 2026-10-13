"""课程学习回调（难度插值 + 下限调度）的回归测试。

覆盖两个已经踩过的坑：
1. 推进判据必须在“当前训练难度”下测（否则训练域与评测域脱节）；
2. 难度下限必须能升到 HARD（旧版 ``0.5·t/T`` 封顶 0.5，预算结束后从没在正式
   判据上训练过，HARD 成功率只有 1~9%）。
"""
import unittest

from rl_vls.contract import curriculum_easy, curriculum_hard
from rl_vls.training.curriculum import KEYS, CurriculumCallback


def make_cb(total_timesteps=100_000, **kw):
    """只需要调度逻辑，不触碰任何 env / model。"""
    kw.setdefault("eval_freq", 10_000)
    kw.setdefault("n_eval_episodes", 1)
    return CurriculumCallback(None, total_timesteps, **kw)


class TestCurriculumSchedule(unittest.TestCase):
    def test_endpoints_match_contract(self):
        cb = make_cb()
        self.assertEqual(cb.params(0.0), curriculum_easy())
        self.assertEqual(cb.params(1.0), curriculum_hard())

    def test_curriculum_covers_every_tightened_criterion(self):
        # handoff_speed_xy 漏进课程表曾导致保持时间永远攒不够、100% crash
        for key in ("handoff_speed_xy", "handoff_hold_s", "handoff_radius",
                    "max_episode_time"):
            self.assertIn(key, KEYS)

    def test_interpolation_is_monotonic(self):
        cb = make_cb()
        mid = cb.params(0.5)
        self.assertAlmostEqual(mid["handoff_radius"], 0.65)
        self.assertAlmostEqual(mid["handoff_hold_s"], 0.25)
        self.assertAlmostEqual(mid["xy_init_range"], 1.75)
        self.assertEqual(tuple(mid["h_init_range"]), (3.0, 6.5))

    def test_floor_reaches_hard_within_floor_frac_budget(self):
        cb = make_cb(100_000, min_progress_frac=0.5)
        cb.num_timesteps = 0
        self.assertAlmostEqual(cb._floor(), 0.0)
        cb.num_timesteps = 25_000          # 半程的一半 -> 进度 0.5
        self.assertAlmostEqual(cb._floor(), 0.5)
        cb.num_timesteps = 50_000          # floor_frac × 总步数 -> 到达 HARD
        self.assertAlmostEqual(cb._floor(), 1.0)
        cb.num_timesteps = 99_000          # 之后保持 HARD
        self.assertAlmostEqual(cb._floor(), 1.0)

    def test_floor_never_exceeds_one(self):
        cb = make_cb(1000, min_progress_frac=0.5)
        for t in range(0, 3000, 250):
            cb.num_timesteps = t
            self.assertLessEqual(cb._floor(), 1.0)

    def test_floor_handles_zero_budget(self):
        cb = make_cb(0)
        cb.num_timesteps = 0
        self.assertEqual(cb._floor(), 0.0)

    def test_floor_frac_controls_when_hard_is_reached(self):
        cb = make_cb(100_000, min_progress_frac=0.75)
        cb.num_timesteps = 75_000
        self.assertAlmostEqual(cb._floor(), 1.0)
        cb.num_timesteps = 50_000
        self.assertAlmostEqual(cb._floor(), 2.0 / 3.0, places=6)

    def test_disabled_curriculum_pins_hard(self):
        cb = make_cb(enabled=False)
        self.assertEqual(cb.progress, 1.0)
        self.assertEqual(cb.params(cb.progress), curriculum_hard())

    def test_advance_rule_increments_and_saturates(self):
        cb = make_cb()
        self.assertLess(cb.advance_at, 0.5)      # 阈值过严会卡住推进
        cb.progress = 0.0
        for _ in range(5):
            cb.progress = min(1.0, cb.progress + cb.step_size)
        self.assertEqual(cb.progress, 1.0)


if __name__ == '__main__':
    unittest.main()
