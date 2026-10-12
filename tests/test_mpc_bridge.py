"""Exercise the ROS control callback without starting DDS or a flight stack."""
import math
from types import SimpleNamespace
import unittest

import numpy as np
from vls_ros2_bridge.vls_ros2_bridge.mpc_controller import LandingMPC

try:
    from vls_ros2_bridge.vls_ros2_bridge.landing_rl_node import LandingRLNode
except ImportError:
    LandingRLNode = None


@unittest.skipIf(LandingRLNode is None, 'ROS2 message dependencies unavailable')
class MPCBridgeTest(unittest.TestCase):
    def test_mpc_without_policy_and_idle_reset(self):
        params = dict(auto_land_enabled=False, require_guided=True, marker_timeout=1., cmd_filter_alpha=0.25,
                      v_xy_max=2., v_down_max=1.5, v_up_max=0.8)
        published = []
        node = SimpleNamespace(
            controller='mpc', landing=SimpleNamespace(phase='idle'), mpc=LandingMPC(), policy=None, onnx_sess=None,
            mav_state=SimpleNamespace(armed=True, mode='GUIDED'),
            p_cam=np.array([2., 0., 8.]), marker_stamp_ns=0,
            vel_flu=np.zeros(3), vel_enu_stamp_ns=None, pose_stamp_ns=None, yaw=math.pi/2,
            _last_dbg_ns=0, _v_filt=np.zeros(3),
            get_parameter=lambda name: SimpleNamespace(value=params[name]),
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0)),
            cmd_pub=SimpleNamespace(publish=published.append),
                    _record_flight_sample=lambda *args: None)
        LandingRLNode._control_cb(node)
        # At yaw=90 degrees, right is east. MPC bypasses the 0.25 filter.
        self.assertGreater(published[-1].linear.x, 0.)
        self.assertAlmostEqual(published[-1].linear.x, node.mpc.previous[0])
        self.assertAlmostEqual(published[-1].linear.y, 0.)
        # Fresh world velocity must be rotated to the same level frame as pc;
        # body velocity includes roll/pitch and is only a fallback.
        captured = []
        original_command = node.mpc.command
        node.mpc.command = lambda p, v, **kwargs: captured.append(v.copy()) or original_command(p, v, **kwargs)
        node.vel_enu_stamp_ns = node.pose_stamp_ns = 0
        node.vel_enu = np.array([0.1, 0.2, 0.3])
        LandingRLNode._control_cb(node)
        np.testing.assert_allclose(captured[-1], [0.1, 0.2, -0.3], atol=1e-12)
        node.mav_state.armed = False
        LandingRLNode._control_cb(node)
        self.assertEqual(published[-1].linear.x, 0.)
        self.assertEqual(published[-1].linear.z, 0.)
        np.testing.assert_array_equal(node.mpc.sequence, np.zeros_like(node.mpc.sequence))


if __name__ == '__main__':
    unittest.main()
