import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from rl_vls.tools.compare_flight_logs import compare, fit_velocity


class FlightComparisonTest(unittest.TestCase):
    def test_identifies_known_delay_and_visual_scale(self):
        dt, tau, delay = 0.1, 0.5, 2
        times = np.arange(130)*dt
        commands = 0.4*np.sin(1.4*times)
        velocity = np.zeros(len(times))
        position = np.zeros(len(times))
        for i in range(len(times)-1):
            u = commands[max(0, i-delay)]
            velocity[i+1] = velocity[i]+(u-velocity[i])*(1-np.exp(-dt/tau))
            position[i+1] = position[i]+velocity[i+1]*dt
        rows = []
        for i, t in enumerate(times):
            rows.append(dict(t_s=t, phase='mpc', controller='mpc', armed=1, cmd_published=1,
                             marker_fresh=1, marker_age_s=0.01, marker_stamp_ns=int(t*1e9),
                             marker_right=0.1, marker_forward=0.73*(2-position[i]), marker_height=5.,
                             pose_age_s=0.01, vel_enu_age_s=0.01,
                             attitude_yaw_deg=0., attitude_roll_deg=0., attitude_pitch_deg=0.,
                             pos_enu_east=position[i], pos_enu_north=0., pos_enu_up=5.,
                             cmd_enu_east=commands[i], cmd_enu_north=0., cmd_enu_up=0.,
                             vel_enu_east=velocity[i], vel_enu_north=0., vel_enu_up=0.))
        fit = fit_velocity(rows, 'east')
        self.assertAlmostEqual(fit['effective_delay_s'], 0.2, places=6)
        self.assertAlmostEqual(fit['tau_s'], 0.5, places=6)
        with TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.csv'
            with path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            report = compare(path)
        scale = report['segments'][0]['estimated_visual_scales']['east']
        self.assertAlmostEqual(scale['visual_to_physical_ratio'], 0.73)
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
