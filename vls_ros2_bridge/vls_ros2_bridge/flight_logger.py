"""CSV flight recorder with a bounded background writer; no ROS dependency."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import threading
import time

VECTORS = {
    'marker_raw': ('x', 'y', 'z'), 'marker': ('right', 'forward', 'height'),
    'cmd_enu': ('east', 'north', 'up'), 'cmd_body': ('right', 'forward', 'down'),
    'vel_flu': ('forward', 'left', 'up'), 'vel_enu': ('east', 'north', 'up'),
    'pos_enu': ('east', 'north', 'up'), 'attitude': ('roll_deg', 'pitch_deg', 'yaw_deg'),
    'quat': ('w', 'x', 'y', 'z'),
}
FIELDS = ['t_s', 'ros_time_ns', 'wall_time_ns', 'controller', 'phase', 'mode',
          'armed', 'connected', 'landed_state', 'cmd_published', 'marker_fresh',
          'marker_status', 'outlier_count', 'marker_stamp_ns', 'marker_raw_stamp_ns',
          'marker_age_s', 'vel_age_s', 'vel_enu_age_s', 'pose_age_s',
          'state_age_s', 'landed_age_s', 'dropped_rows']
FIELDS += [f'{prefix}_{axis}' for prefix, axes in VECTORS.items() for axis in axes]


def quaternion_degrees(quat):
    w, x, y, z = quat
    norm = math.sqrt(w*w+x*x+y*y+z*z)
    if not math.isfinite(norm) or norm < 1e-9:
        return [math.nan]*3
    w, x, y, z = [q/norm for q in quat]
    return [math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))),
            math.degrees(math.asin(max(-1., min(1., 2*(w*y-z*x))))),
            math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))]


class FlightLogger:
    def __init__(self, directory, controller, metadata=None, queue_size=4096):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+controller
        self.path = directory / (name+'.csv')
        self.metadata_path = self.path.with_suffix('.json')
        self._metadata = dict(schema_version=1, controller=controller,
                              parameters=metadata or {}, dropped_rows=0,
                              frames='marker: level right/forward/down; pose and cmd: ENU; velocity_body: FLU',
                              sampling='latest received telemetry at each control tick; ages use reception time, marker age uses frame time')
        self.metadata_path.write_text(json.dumps(self._metadata, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
        self._file = self.path.open('x', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()
        self._file.flush()
        self._queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self.error = None
        self.dropped_rows = 0
        self._closed = False
        self._origin_ns = None
        self._thread = threading.Thread(target=self._run, name='flight-csv-writer', daemon=True)
        self._thread.start()

    def record(self, now_ns, row):
        if self._closed or self.error is not None:
            return
        if self._origin_ns is None:
            self._origin_ns = now_ns
        item = {key: math.nan for key in FIELDS}
        item.update(row)
        item.update(t_s=(now_ns-self._origin_ns)*1e-9, ros_time_ns=now_ns,
                    wall_time_ns=time.time_ns(), dropped_rows=self.dropped_rows)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self.dropped_rows += 1

    def _run(self):
        last_flush = time.monotonic()
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    row = self._queue.get(timeout=0.1)
                except queue.Empty:
                    row = None
                if row is not None:
                    self._writer.writerow(row)
                if time.monotonic()-last_flush >= 1.:
                    self._file.flush()
                    last_flush = time.monotonic()
        except Exception as exc:
            self.error = str(exc)
        finally:
            self._file.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=3.)
        self._metadata.update(dropped_rows=self.dropped_rows, writer_error=self.error,
                              writer_drained=not self._thread.is_alive())
        try:
            self.metadata_path.write_text(json.dumps(self._metadata, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
        except OSError:
            pass
