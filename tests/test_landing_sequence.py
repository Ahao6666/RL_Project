from types import MethodType, SimpleNamespace
import unittest

import numpy as np
from vls_ros2_bridge.vls_ros2_bridge.landing_sequence import LandingSequence

try:
    from vls_ros2_bridge.vls_ros2_bridge.landing_rl_node import LandingRLNode
    from mavros_msgs.msg import ExtendedState
except ImportError:
    LandingRLNode = None


class Future:
    def __init__(self):
        self.response = None
        self.cancelled = False

    def done(self):
        return self.response is not None

    def result(self):
        return self.response

    def cancel(self):
        self.cancelled = True


class SequenceTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.futures = []
        self.logger = SimpleNamespace(info=lambda msg: None, warn=lambda msg: None)
        self.sequence = LandingSequence(lambda mode: self.request('land', mode),
                                        lambda: self.request('disarm', False), self.logger,
                                        altitude=0.3, land_mode='AUTO')
        self.data = dict(armed=True, mode='GUIDED', state_fresh=True,
                         marker_altitude=0.29, marker_fresh=True,
                         on_ground=False, ground_fresh=True, ground_stamp=0.)

    def request(self, kind, value):
        self.calls.append((kind, value))
        future = Future()
        self.futures.append(future)
        return future

    def tick(self, now, **changes):
        self.data.update(changes)
        return self.sequence.update(now, **self.data)

    def test_default_qland_handoff_before_visual_loss(self):
        sequence = LandingSequence(lambda mode: self.request('land', mode),
                                   lambda: self.request('disarm', False), self.logger)
        data = dict(self.data, mode='GUIDED', marker_altitude=2.0)
        self.assertFalse(sequence.update(1., **data))
        self.assertEqual(self.calls, [])
        data['marker_altitude'] = 1.9
        self.assertTrue(sequence.update(1.1, **data))
        self.assertEqual(self.calls, [('land', 'QLAND')])
        data.update(mode='QLAND', marker_fresh=False)
        self.assertTrue(sequence.update(1.2, **data))
        self.assertEqual(sequence.confirmed_at, 1.2)

    def test_land_then_ground_confirmed_disarm(self):
        self.assertTrue(self.tick(1.))
        self.assertEqual(self.calls, [('land', 'LAND')])
        self.futures[-1].response = SimpleNamespace(mode_sent=True)
        self.tick(1.1)
        self.assertIsNone(self.sequence.confirmed_at)  # acknowledgement is not mode
        self.tick(1.2, mode='LAND', marker_fresh=False)
        self.tick(1.3, on_ground=True, ground_stamp=1.)  # pre-LAND report ignored
        self.assertIsNone(self.sequence.ground_since)
        self.tick(1.4, ground_stamp=1.4)
        self.tick(2.5, ground_stamp=2.5)
        self.assertEqual(self.calls[-1], ('disarm', False))
        self.futures[-1].response = SimpleNamespace(success=True)
        self.tick(2.6)
        self.assertEqual(self.sequence.phase, 'landing')  # wait for armed telemetry
        self.tick(2.7, armed=False)
        self.assertEqual(self.sequence.phase, 'complete')

    def test_threshold_and_valid_telemetry(self):
        for changes in [dict(marker_altitude=0.3), dict(marker_altitude=-0.01),
                        dict(marker_altitude=float('nan')), dict(marker_fresh=False),
                        dict(state_fresh=False), dict(armed=False), dict(mode='LOITER')]:
            data = dict(self.data)
            data.update(changes)
            self.assertFalse(self.sequence.update(1., **data), changes)
        self.assertEqual(self.calls, [])

    def test_qguided_and_operator_takeover(self):
        self.tick(1., mode='QGUIDED')
        self.assertEqual(self.calls[-1], ('land', 'QLAND'))
        self.tick(1.1, mode='QLOITER')
        self.assertTrue(self.futures[-1].cancelled)
        self.assertEqual(self.sequence.phase, 'aborted')
        self.tick(5., mode='QGUIDED')
        self.assertEqual(len(self.calls), 1)  # no fight with operator on this flight
        self.tick(6., armed=False)
        self.tick(7., armed=True, marker_altitude=5.)
        self.assertEqual(self.sequence.phase, 'idle')

    def test_rejection_timeout_and_retry(self):
        self.tick(1.)
        self.futures[-1].response = SimpleNamespace(mode_sent=False)
        self.tick(1.1)
        self.assertEqual(len(self.calls), 1)
        self.tick(2.1)
        pending = self.futures[-1]
        self.tick(2.2)
        self.assertEqual(len(self.calls), 2)
        self.tick(5.2)
        self.assertTrue(pending.cancelled)
        self.assertEqual(len(self.calls), 3)

    def test_ground_loss_staleness_and_disarm_retry(self):
        self.tick(1.)
        self.tick(1.1, mode='LAND')
        self.tick(1.2, on_ground=True, ground_stamp=1.2)
        self.tick(2.3, ground_fresh=False)
        self.assertEqual(len(self.calls), 1)
        self.tick(2.4, ground_fresh=True, ground_stamp=2.4)
        self.tick(3.5, ground_stamp=3.5)
        pending = self.futures[-1]
        self.tick(3.6, on_ground=False)
        self.assertTrue(pending.cancelled)
        self.tick(3.7, on_ground=True, ground_stamp=3.7)
        self.tick(4.8, ground_stamp=4.8)
        self.assertEqual(self.calls[-1], ('disarm', False))
        self.futures[-1].response = SimpleNamespace(success=False)
        self.tick(4.9)
        self.tick(6., ground_stamp=6.)
        self.assertEqual(sum(kind == 'disarm' for kind, _ in self.calls), 3)

    def test_unavailable_service_and_stale_flight_state(self):
        self.sequence.request_land = lambda mode: None
        self.assertTrue(self.tick(1.))
        self.assertIsNone(self.sequence.future)
        self.sequence.request_land = lambda mode: self.request('land', mode)
        self.tick(2.1, state_fresh=False)
        self.assertEqual(self.calls, [])
        self.tick(2.2, state_fresh=True)
        self.assertEqual(self.calls[-1], ('land', 'LAND'))

    @unittest.skipIf(LandingRLNode is None, 'ROS2 unavailable')
    def test_service_request_payloads(self):
        requests = []
        client = SimpleNamespace(service_is_ready=lambda: True,
                                 call_async=lambda request: requests.append(request) or Future())
        node = SimpleNamespace(land_client=client, disarm_client=client)
        LandingRLNode._request_land(node, 'LAND')
        LandingRLNode._request_disarm(node)
        self.assertEqual(requests[0].custom_mode, 'LAND')
        self.assertFalse(requests[1].value)
        client.service_is_ready = lambda: False
        self.assertIsNone(LandingRLNode._request_land(node, 'LAND'))
        self.assertIsNone(LandingRLNode._request_disarm(node))
        self.assertEqual(len(requests), 2)

    @unittest.skipIf(LandingRLNode is None, 'ROS2 unavailable')
    def test_every_controller_hands_off_before_predicting(self):
        for controller in ('rl', 'p', 'hybrid', 'mpc', 'fixed'):
            with self.subTest(controller=controller):
                self.setUp()
                params = dict(auto_land_enabled=True, require_guided=True,
                              marker_timeout=1., landing_state_timeout_s=2., landed_state_timeout_s=1.5)
                clock = [1_000_000_000]
                published = []
                node = SimpleNamespace(
                    controller=controller, landing=self.sequence, mpc=None,
                    policy=None, onnx_sess=None,
                    mav_state=SimpleNamespace(armed=True, mode='GUIDED', connected=True),
                    state_stamp_ns=clock[0], p_cam=np.array([0., 0., 0.29]),
                    marker_stamp_ns=clock[0], extended_state=ExtendedState(),
                    extended_state_stamp_ns=None, _v_filt=np.ones(3),
                    prev_action=np.ones(3),
                    get_parameter=lambda name: SimpleNamespace(value=params[name]),
                    get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=clock[0])),
                    cmd_pub=SimpleNamespace(publish=published.append),
                    _record_flight_sample=lambda *args: None)
                node._landing_step = MethodType(LandingRLNode._landing_step, node)
                LandingRLNode._control_cb(node)
                self.assertEqual(self.calls, [('land', 'LAND')])
                self.assertEqual(len(published), 1)
                self.assertEqual(published[-1].linear.z, 0.)
                node.mav_state.mode = 'LAND'
                clock[0] += 100_000_000
                LandingRLNode._control_cb(node)
                self.assertEqual(len(published), 1)  # autonomous LAND gets no setpoints


if __name__ == '__main__':
    unittest.main()
