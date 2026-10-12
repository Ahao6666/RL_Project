import unittest
import numpy as np
from rl_vls.envs.landing_env import LandingEnv
from vls_ros2_bridge.vls_ros2_bridge.mpc_controller import LandingMPC


class MPCTest(unittest.TestCase):
    def test_bounds_and_direction(self):
        mpc = LandingMPC()
        cmd = mpc.command([4., -3., 8.], [0., 0., 0.])
        self.assertGreater(cmd[0], 0.)
        self.assertLess(cmd[1], 0.)
        self.assertLessEqual(cmd[2], 0.)
        self.assertTrue(np.all(cmd >= mpc.lower))
        self.assertTrue(np.all(cmd <= mpc.upper))

    def test_slew_limits_cover_full_horizon_under_jitter(self):
        mpc = LandingMPC(iterations=5)
        rng = np.random.default_rng(3)
        for _ in range(100):
            previous = mpc.previous.copy()
            cmd = mpc.command([*rng.normal(0, 2, 2), rng.uniform(0.3, 10)],
                              rng.normal(0, 0.3, 3))
            increments = np.diff(np.column_stack([previous, mpc.sequence]), axis=1)
            self.assertTrue(np.all(np.abs(increments) <= mpc.max_delta[:, None]+1e-10))
            self.assertTrue(np.all(mpc.sequence >= mpc.lower[:, None]-1e-10))
            self.assertTrue(np.all(mpc.sequence <= mpc.upper[:, None]+1e-10))
            self.assertGreaterEqual(cmd[2], 0.)

    def test_held_camera_sample_is_not_fused_twice(self):
        mpc = LandingMPC()
        p = np.array([1., 2., 5.])
        velocity = np.array([0.2, 0., 0.1])
        mpc.command(p, velocity, measurement_stamp=1_000_000_000)
        mpc.command(p, velocity, measurement_stamp=1_000_000_000)
        np.testing.assert_allclose(mpc.filtered_position, p-velocity*mpc.dt)
        predicted = mpc.filtered_position-velocity*mpc.dt
        mpc.command(p, velocity, measurement_stamp=1_100_000_000)
        alpha = 0.1/(mpc.position_filter_tau+0.1)
        np.testing.assert_allclose(mpc.filtered_position, predicted+alpha*(p-predicted))

    def test_delayed_prediction_uses_issued_command_queue(self):
        mpc = LandingMPC(horizon=10, delay_xy=0.15, delay_z=0.1)
        commands = np.linspace(-0.3, 0.4, mpc.horizon)
        for axis, (F, G, _, _, _) in enumerate(mpc.models):
            delay = mpc.delays[axis]
            history = list(np.linspace(0.1, 0.2, delay))
            predicted = F @ np.r_[2., 0.1, history] + G @ commands
            p, v = 2., 0.1
            tau = 0.4 if axis == 2 else 0.6
            for k, command in enumerate(commands):
                history.append(command)
                applied = history.pop(0)
                v += (applied-v)*mpc.dt/tau
                p -= v*mpc.dt
                np.testing.assert_allclose(predicted[2*k:2*k+2], [p, v], atol=1e-10)

    def test_visual_calibration_and_speed_gate(self):
        mpc = LandingMPC(visual_scale=0.73, velocity_filter_tau=0.)
        mpc.command([0.073, 0., 5.11], [0., 0., 0.])
        np.testing.assert_allclose(mpc.filtered_position, [0.1, 0., 7.])
        self.assertGreater(mpc.descent_gate, 0.)
        mpc.command([0.073, 0., 5.11], [0.3, 0., 0.])
        self.assertEqual(mpc.descent_gate, 0.)
        mpc.reset()
        self.assertTrue(all(np.all(history == 0.) for history in mpc.command_history))

    def test_prediction_matches_environment(self):
        env = LandingEnv(tau_xy_range=(1.3, 1.3), tau_z_range=(0.95, 0.95),
                         noise_cfg=dict(dropout_prob=0.))
        env.reset(seed=4)
        env.pos = np.array([0.2, -0.1, 5.])
        env.vel = np.array([0.1, -0.2, 0.3])
        mpc = LandingMPC(tau_xy=1.3, tau_z=0.95, delay_xy=0., delay_z=0.)
        commands = np.tile([0.4, -0.3, 0.2], (mpc.horizon, 1))
        initial_p, initial_v = env._true_p_cam(), env.vel.copy()
        predictions = []
        for axis, (F, G, _, _, _) in enumerate(mpc.models):
            predictions.append(F @ [initial_p[axis], initial_v[axis]] + G @ commands[:, axis])
        for k, cmd in enumerate(commands):
            env.step([cmd[0]/2, cmd[1]/2, 2*(cmd[2]+0.8)/2.3-1])
            for axis in range(3):
                np.testing.assert_allclose(
                    predictions[axis][2*k:2*k+2],
                    [env._true_p_cam()[axis], env.vel[axis]], atol=1e-6)
        env.close()

    def test_invalid_and_reset(self):
        mpc = LandingMPC()
        np.testing.assert_array_equal(mpc.command([np.nan, 0, 2], [0, 0, 0]), np.zeros(3))
        a = mpc.command([0, 0, 4], [0, 0, 0])
        mpc.reset()
        np.testing.assert_allclose(a, mpc.command([0, 0, 4], [0, 0, 0]))

    def test_closed_loop_landing(self):
        env = LandingEnv(max_episode_time=120., tau_xy_range=(1.3, 1.3), tau_z_range=(0.95, 0.95),
                         noise_cfg=dict(sigma_dist=0., sigma_offaxis=0., sigma_floor=0., dropout_prob=0.))
        mpc = LandingMPC()
        obs, _ = env.reset(seed=7)
        for _ in range(2401):
            cmd = mpc.command(obs[:3], obs[3:6])
            action = [cmd[0]/2, cmd[1]/2, 2*(cmd[2]+0.8)/2.3-1]
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        self.assertEqual(info.get('landing'), 'success', info)
        env.close()


if __name__ == '__main__':
    unittest.main()
