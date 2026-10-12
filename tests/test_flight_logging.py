import csv
import json
import math
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from rl_vls.tools.analyze_flight_log import analyze, overshoot, settling
from vls_ros2_bridge.vls_ros2_bridge.flight_logger import FlightLogger, quaternion_degrees

try:
    from vls_ros2_bridge.vls_ros2_bridge.landing_rl_node import LandingRLNode
except ImportError:
    LandingRLNode = None


class FlightLoggingTest(unittest.TestCase):
    def test_async_csv_drain_and_metadata(self):
        with TemporaryDirectory() as directory:
            logger = FlightLogger(directory, 'mpc', {'auto_land_altitude': 2.}, queue_size=4)
            for i in range(100):
                logger.record(1_000_000_000+i*50_000_000, {'phase': 'mpc', 'cmd_published': 1})
            logger.close()
            logger.close()
            self.assertIsNone(logger.error)
            with logger.path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows)+logger.dropped_rows, 100)
            self.assertTrue(all(r['phase'] == 'mpc' for r in rows))
            metadata = json.loads(logger.metadata_path.read_text())
            self.assertTrue(metadata['writer_drained'])
            self.assertEqual(metadata['dropped_rows'], logger.dropped_rows)

    def test_disk_failure_does_not_escape_record_call(self):
        with TemporaryDirectory() as directory:
            logger = FlightLogger(directory, 'mpc')
            with patch.object(logger._writer, 'writerow', side_effect=OSError('disk full')):
                logger.record(1, {'phase': 'mpc'})
                logger.close()
            self.assertIn('disk full', logger.error)
            logger.record(2, {'phase': 'mpc'})  # disabled after error, no exception
            self.assertIn('disk full', json.loads(logger.metadata_path.read_text())['writer_error'])

    def test_overshoot_and_settling_definitions(self):
        self.assertAlmostEqual(overshoot([1., 0.3, -0.2, 0.])['percent'], 20.)
        self.assertIsNone(overshoot([0.01, -0.2])['percent'])
        result = settling(np.arange(8.), [False, True, True, True, False, True, True, True], 2., 1.1)
        self.assertEqual(result['first_stable_window_s'], 1.)
        self.assertEqual(result['settling_time_s'], 5.)

    @staticmethod
    def row(t, phase='mpc', fresh=1):
        return dict(t_s=str(t), controller='mpc', phase=phase, armed='1',
                    cmd_published='1' if phase == 'mpc' else '0', marker_fresh=str(fresh),
                    marker_age_s='0.01', marker_stamp_ns=str(int(t*1e9)),
                    marker_right='0.1', marker_forward='0.', pose_age_s='0.01',
                    attitude_yaw_deg='0.', attitude_roll_deg='0.', attitude_pitch_deg='0.',
                    vel_enu_age_s='0.01', vel_enu_east='0.', vel_enu_north='0.', vel_enu_up='-0.2',
                    cmd_enu_east='0.', cmd_enu_north='0.', cmd_enu_up='-0.2')

    def test_split_autonomous_land_and_stale_measurements(self):
        rows = [self.row(i*0.1) for i in range(31)]
        rows += [self.row(3.1+i*0.1, 'landing') for i in range(10)]
        rows += [self.row(4.1+i*0.1, fresh=0) for i in range(31)]
        report = analyze(rows)
        self.assertEqual(len(report['segments']), 2)
        self.assertEqual(report['segments'][0]['settling_time_s'], 0.)
        self.assertIsNone(report['segments'][1]['settling_time_s'])
        self.assertIsNone(report['segments'][1]['last_valid_horizontal_error_m'])
        self.assertEqual(report['segments'][0]['velocity']['up']['tracking_rmse_m_s'], 0.)
        json.dumps(report, allow_nan=False)

    def test_missing_velocity_does_not_imply_stability(self):
        rows = [self.row(i*0.1) for i in range(31)]
        for row in rows:
            row['vel_enu_age_s'] = 'nan'
        report = analyze(rows)
        self.assertIsNone(report['segments'][0]['settling_time_s'])
        self.assertIsNone(report['segments'][0]['velocity']['east']['tracking_rmse_m_s'])

    def test_quaternion_angles(self):
        np.testing.assert_allclose(quaternion_degrees([math.sqrt(0.5), 0, 0, math.sqrt(0.5)]), [0, 0, 90], atol=1e-8)
        self.assertTrue(all(math.isnan(v) for v in quaternion_degrees([0, 0, 0, 0])))

    @unittest.skipIf(LandingRLNode is None, 'ROS2 unavailable')
    def test_node_snapshot_marks_unpublished_commands_missing(self):
        samples = []
        node = SimpleNamespace(
            flight_recorder=SimpleNamespace(error=None, record=lambda stamp, row: samples.append(row)),
            controller='mpc', mpc=None, yaw=0., mav_state=SimpleNamespace(armed=True, connected=True, mode='QLAND'),
            extended_state=SimpleNamespace(landed_state=2), marker_status='accepted', _outlier_count=0,
            marker_stamp_ns=1_000_000_000, marker_raw_stamp_ns=1_000_000_000,
            vel_stamp_ns=1_000_000_000, vel_enu_stamp_ns=1_000_000_000,
            pose_stamp_ns=1_000_000_000, state_stamp_ns=1_000_000_000,
            extended_state_stamp_ns=1_000_000_000,
            marker_raw=[0, 0, 1.9], p_cam=[0, 0, 1.9], vel_flu=[1, 2, 3],
            vel_enu=[1, 2, 3], pos_enu=[4, 5, 6], quat=[1, 0, 0, 0])
        LandingRLNode._record_flight_sample(node, 1_100_000_000, None, 'landing', True)
        self.assertEqual(samples[-1]['cmd_published'], 0)
        self.assertTrue(math.isnan(samples[-1]['cmd_enu_east']))
        self.assertEqual(samples[-1]['pos_enu_north'], 5)
        self.assertAlmostEqual(samples[-1]['pose_age_s'], 0.1)
        cmd = SimpleNamespace(linear=SimpleNamespace(x=1., y=2., z=-0.5))
        LandingRLNode._record_flight_sample(node, 1_100_000_000, cmd, 'mpc', True)
        self.assertEqual(samples[-1]['cmd_published'], 1)
        self.assertEqual(samples[-1]['cmd_body_right'], -2.)
        self.assertEqual(samples[-1]['cmd_body_forward'], 1.)
        self.assertEqual(samples[-1]['cmd_body_down'], 0.5)


if __name__ == '__main__':
    unittest.main()
