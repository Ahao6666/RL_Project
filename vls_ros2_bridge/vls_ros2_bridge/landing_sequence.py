"""Nonblocking LAND handoff and landed-confirmed disarming, shared by controllers.

Service acknowledgement never substitutes for flight-controller mode/armed
telemetry. An operator mode change cancels the sequence for this armed flight.
"""
from __future__ import annotations

import math


class LandingSequence:
    def __init__(self, request_land, request_disarm, logger, altitude=2.5,
                 land_mode='QLAND', retry_s=1.0, request_timeout_s=3.0,
                 disarm_hold_s=1.0):
        if not all(math.isfinite(x) and x > 0 for x in
                   (altitude, retry_s, request_timeout_s, disarm_hold_s)):
            raise ValueError('Landing thresholds and timeouts must be finite and positive')
        if land_mode not in ('AUTO', 'LAND', 'QLAND'):
            raise ValueError('auto_land_mode must be AUTO, LAND or QLAND')
        self.request_land = request_land
        self.request_disarm = request_disarm
        self.logger = logger
        self.altitude, self.land_mode = altitude, land_mode
        self.retry_s, self.request_timeout_s = retry_s, request_timeout_s
        self.disarm_hold_s = disarm_hold_s
        self.phase = 'idle'
        self.target_mode = None
        self.confirmed_at = None
        self.ground_since = None
        self.future = None
        self.request_kind = None
        self.request_at = None
        self.last_attempt = -math.inf
        self.was_armed = False

    def _cancel(self):
        if self.future is not None:
            self.future.cancel()
        self.future = None
        self.request_kind = None

    def _poll(self, now):
        if self.future is None:
            return
        if self.future.done():
            try:
                result = self.future.result()
                accepted = result.mode_sent if self.request_kind == 'land' else result.success
                if not accepted:
                    self.logger.warn(f'[LAND] {self.request_kind} 请求被拒绝，将重试')
            except Exception as exc:
                self.logger.warn(f'[LAND] {self.request_kind} 请求失败: {exc}')
            self.future = None
            self.request_kind = None
        elif now - self.request_at >= self.request_timeout_s:
            self.logger.warn(f'[LAND] {self.request_kind} 请求超时，将重试')
            self._cancel()

    def _request(self, kind, now):
        if self.future is not None or now - self.last_attempt < self.retry_s:
            return
        self.last_attempt = now
        try:
            self.future = self.request_land(self.target_mode) if kind == 'land' else self.request_disarm()
            if self.future is None:
                self.logger.warn(f'[LAND] {kind} 服务尚未就绪，将重试')
            else:
                self.request_kind, self.request_at = kind, now
        except Exception as exc:
            self.logger.warn(f'[LAND] {kind} 请求失败: {exc}')

    def update(self, now, *, armed, mode, state_fresh, marker_altitude,
               marker_fresh, on_ground, ground_fresh, ground_stamp):
        """Return True while the sequence owns control (suppress velocity output).

        All timestamps are seconds on the same ROS clock. ground_stamp is the
        reception time of ExtendedState, not visual altitude or local pose.
        """
        if not state_fresh:
            return self.phase != 'idle'
        if not armed:
            self._cancel()
            if self.phase not in ('idle', 'complete'):
                self.logger.info('[LAND] 飞控已解除武装，着陆流程完成')
                self.phase = 'complete'
            self.was_armed = False
            return self.phase != 'idle'
        if not self.was_armed:
            self.phase = 'idle'
            self.target_mode = self.confirmed_at = self.ground_since = None
            self.last_attempt = -math.inf
        self.was_armed = True
        if self.phase == 'idle':
            if not (mode in ('GUIDED', 'QGUIDED') and marker_fresh
                    and marker_altitude is not None and math.isfinite(marker_altitude)
                    and 0 <= marker_altitude < self.altitude):
                return False
            self.target_mode = ('QLAND' if mode == 'QGUIDED' else 'LAND') \
                if self.land_mode == 'AUTO' else self.land_mode
            self.phase = 'landing'
            self.logger.info(f'[LAND] 高度 {marker_altitude:.3f} m，切换 {self.target_mode}')
        if self.phase in ('aborted', 'complete'):
            return True
        if mode not in ('GUIDED', 'QGUIDED', self.target_mode) \
                or (self.confirmed_at is not None and mode != self.target_mode):
            self._cancel()
            self.phase = 'aborted'
            self.logger.warn(f'[LAND] 模式变为 {mode}，取消自动流程，交还人工控制')
            return True
        self._poll(now)
        if mode != self.target_mode:
            self._request('land', now)
            return True
        if self.confirmed_at is None:
            self.confirmed_at = now
            self._cancel()
            self.last_attempt = -math.inf
            self.logger.info(f'[LAND] 已确认 {self.target_mode}，等待飞控落地确认')
        # Require a new, fresh landed report after LAND was observed. A low
        # visual altitude alone must never authorize disarming in flight.
        grounded = on_ground and ground_fresh and ground_stamp is not None \
            and ground_stamp >= self.confirmed_at
        if not grounded:
            self.ground_since = None
            if self.request_kind == 'disarm':
                self._cancel()
            return True
        if self.ground_since is None:
            self.ground_since = now
        if now - self.ground_since >= self.disarm_hold_s:
            self._request('disarm', now)
        return True
