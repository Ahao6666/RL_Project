"""Analyze flight_logger CSV: tracking, overshoot, settling and telemetry plots."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

TRACKING_PHASES = {'rl', 'p', 'mpc', 'fixed'}


def number(row, name):
    try:
        return float(row.get(name, 'nan'))
    except (ValueError, TypeError):
        return float('nan')


def column(rows, name):
    return np.array([number(row, name) for row in rows])


def statistic(values, operation=np.mean):
    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    return float(operation(valid)) if len(valid) else None


def overshoot(error, deadband=0.05):
    """Peak opposite-side error relative to initial sign; undefined near zero."""
    valid = np.asarray(error)[np.isfinite(error)]
    if not len(valid) or abs(valid[0]) <= deadband:
        return {'peak_m': None, 'percent': None}
    peak = max(0., float(np.max(-np.sign(valid[0])*valid)))
    peak = peak if peak > deadband else 0.
    return {'peak_m': peak, 'percent': 100*peak/abs(valid[0])}


def settling(t, good, hold_s, max_gap_s):
    """Sustained entry and permanent settling; missing data breaks stability."""
    first_window = None
    start = None
    last_bad = -1
    for i in range(len(t)):
        gap = i > 0 and t[i]-t[i-1] > max_gap_s
        if not good[i] or gap:
            start = None
            last_bad = i
        if good[i]:
            if start is None:
                start = i
            if first_window is None and t[i]-t[start] >= hold_s:
                first_window = float(t[start]-t[0])
    permanent = None
    index = last_bad+1
    if index < len(t) and np.all(good[index:]) and t[-1]-t[index] >= hold_s:
        permanent = float(t[index]-t[0])
    return {'first_stable_window_s': first_window, 'settling_time_s': permanent}


def analyze_segment(rows, tolerance, speed_tolerance, hold_s, max_age, max_gap):
    t = column(rows, 't_s')
    p = np.column_stack([column(rows, 'marker_right'), column(rows, 'marker_forward')])
    marker_age = column(rows, 'marker_age_s')
    fresh = (column(rows, 'marker_fresh') == 1) & (marker_age >= 0) & (marker_age <= max_age)
    p[~fresh] = np.nan
    error = np.linalg.norm(p, axis=1)
    pose_age = column(rows, 'pose_age_s')
    yaw = np.radians(column(rows, 'attitude_yaw_deg'))
    pose_valid = (pose_age >= 0) & (pose_age <= max_age) & np.isfinite(yaw)
    # Rotate marker errors into the initial valid heading frame to avoid
    # interpreting yaw changes as horizontal target overshoot.
    yaw[~pose_valid] = np.nan
    east = p[:, 1]*np.cos(yaw)+p[:, 0]*np.sin(yaw)
    north = p[:, 1]*np.sin(yaw)-p[:, 0]*np.cos(yaw)
    valid = np.flatnonzero(np.isfinite(east) & np.isfinite(north))
    right = forward = np.full(len(rows), np.nan)
    if len(valid):
        heading = yaw[valid[0]]
        right = east*np.sin(heading)-north*np.cos(heading)
        forward = east*np.cos(heading)+north*np.sin(heading)
    velocity = np.column_stack([column(rows, 'vel_enu_'+axis) for axis in ('east', 'north', 'up')])
    age = column(rows, 'vel_enu_age_s')
    velocity[~((age >= 0) & (age <= max_age))] = np.nan
    speed_xy = np.linalg.norm(velocity[:, :2], axis=1)
    good = np.isfinite(error) & np.isfinite(speed_xy) & (error <= tolerance) & (speed_xy <= speed_tolerance)
    result = dict(start_s=float(t[0]), duration_s=float(t[-1]-t[0]), rows=len(rows),
                  controller=rows[0]['controller'], phases=sorted({r['phase'] for r in rows}),
                  horizontal_error_mean_m=statistic(error), horizontal_error_max_m=statistic(error, np.max),
                  last_valid_horizontal_error_m=float(error[np.isfinite(error)][-1]) if np.any(np.isfinite(error)) else None,
                  overshoot_right=overshoot(right), overshoot_forward=overshoot(forward),
                  visual_unusable_fraction=float(np.mean(~np.isfinite(error))),
                  unique_visual_frames=len({r.get('marker_stamp_ns') for r, ok in zip(rows, fresh) if ok}))
    result.update(settling(t, good, hold_s, max_gap))
    result['velocity'] = {}
    for axis_index, axis in enumerate(('east', 'north', 'up')):
        cmd = column(rows, 'cmd_enu_'+axis)
        delta_t = np.diff(t)
        valid_delta = (delta_t > 0) & (delta_t <= max_gap)
        rates = np.divide(np.diff(cmd), delta_t, out=np.full(len(delta_t), np.nan), where=valid_delta)
        signs = np.sign(cmd[np.isfinite(cmd) & (np.abs(cmd) > 0.05)])
        result['velocity'][axis] = dict(
            tracking_rmse_m_s=statistic((cmd-velocity[:, axis_index])**2, lambda a: np.sqrt(np.mean(a))),
            command_rate_max_m_s2=statistic(np.abs(rates), np.max),
            command_reversals=int(np.count_nonzero(np.diff(signs))) if len(signs) else 0)
    result['attitude_rms_deg'] = {axis: statistic(
        np.where(pose_valid, column(rows, 'attitude_'+axis+'_deg'), np.nan)**2,
        lambda a: np.sqrt(np.mean(a))) for axis in ('roll', 'pitch')}
    return result


def analyze(rows, tolerance=0.3, speed_tolerance=0.2, hold_s=2., max_age=0.5, max_gap=0.25):
    if not rows:
        raise ValueError('日志没有数据行')
    if not np.all(np.isfinite([tolerance, speed_tolerance, hold_s, max_age, max_gap])) or min(tolerance, speed_tolerance, hold_s, max_age, max_gap) <= 0:
        raise ValueError('分析阈值必须为正数')
    segments, current = [], []
    previous_time = None
    for row in rows:
        t = number(row, 't_s')
        tracking = row.get('phase') in TRACKING_PHASES and number(row, 'armed') == 1 \
            and number(row, 'cmd_published') == 1 and np.isfinite(t)
        split = previous_time is not None and (t <= previous_time or t-previous_time > max_gap)
        if not tracking or split:
            if current:
                segments.append(current)
                current = []
        if tracking:
            current.append(row)
        previous_time = t
    if current:
        segments.append(current)
    return dict(schema_version=1, rows=len(rows),
                thresholds=dict(position_m=tolerance, horizontal_speed_m_s=speed_tolerance,
                                hold_s=hold_s, telemetry_max_age_s=max_age, max_sample_gap_s=max_gap),
                phases={phase: sum(r.get('phase') == phase for r in rows) for phase in sorted({r.get('phase','') for r in rows})},
                definitions=dict(overshoot='opposite-side peak in initial heading frame, target zero, 0.05m deadband',
                                 settling='position and horizontal speed inside thresholds continuously through segment end; at least hold_s',
                                 last_error='last fresh visual error during strategy control, not touchdown accuracy',
                                 velocity_rmse='latest fresh ENU telemetry versus published ENU command; no latency compensation'),
                segments=[analyze_segment(s, tolerance, speed_tolerance, hold_s, max_age, max_gap) for s in segments])


def plot(rows, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = column(rows, 't_s')
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)
    marker_fresh = column(rows, 'marker_fresh') == 1
    for name in ('right', 'forward'):
        data = np.where(marker_fresh, column(rows, 'marker_'+name), np.nan)
        axes[0, 0].plot(t, data, label=name)
    axes[0, 0].axhline(0, color='gray', linewidth=0.5)
    axes[0, 0].set_title('Visual horizontal error (level body axes), m')
    axes[0, 1].plot(t, np.where(marker_fresh, column(rows, 'marker_height'), np.nan), label='visual depth')
    axes[0, 1].plot(t, column(rows, 'pos_enu_up'), label='local pose z (different origin)')
    axes[0, 1].set_title('Height / local altitude, m')
    for ax, axis in zip((axes[1, 0], axes[1, 1], axes[2, 0]), ('east', 'north', 'up')):
        cmd = np.where(column(rows, 'cmd_published') == 1, column(rows, 'cmd_enu_'+axis), np.nan)
        measured = np.where((column(rows, 'vel_enu_age_s') >= 0) & (column(rows, 'vel_enu_age_s') <= 0.5), column(rows, 'vel_enu_'+axis), np.nan)
        ax.plot(t, cmd, label='published command')
        ax.plot(t, measured, label='measured')
        ax.set_title(f'ENU {axis} velocity, m/s')
    for axis in ('roll', 'pitch', 'yaw'):
        axes[2, 1].plot(t, column(rows, 'attitude_'+axis+'_deg'), label=axis)
    axes[2, 1].set_title('Attitude, deg')
    axes[3, 0].plot(column(rows, 'pos_enu_east'), column(rows, 'pos_enu_north'), label='aircraft')
    axes[3, 0].set_title('Local ENU trajectory, m')
    axes[3, 0].set_xlabel('East, m')
    axes[3, 0].set_ylabel('North, m')
    axes[3, 0].axis('equal')
    phases = list(dict.fromkeys(r['phase'] for r in rows))
    axes[3, 1].step(t, [phases.index(r['phase']) for r in rows], where='post', label='phase')
    axes[3, 1].set_yticks(range(len(phases)), phases)
    axes[3, 1].set_title('Control / landing phase')
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(loc='best')
        if ax is not axes[3, 0]:
            ax.set_xlabel('Time, s')
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--position-tolerance', type=float, default=0.3)
    parser.add_argument('--speed-tolerance', type=float, default=0.2)
    parser.add_argument('--hold-s', type=float, default=2.)
    parser.add_argument('--max-age', type=float, default=0.5)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    with args.csv.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if rows and ('t_s' not in rows[0] or 'phase' not in rows[0]):
        parser.error('需要由 flight_logger 生成的 CSV')
    try:
        report = analyze(rows, args.position_tolerance, args.speed_tolerance, args.hold_s, args.max_age)
    except ValueError as exc:
        parser.error(str(exc))
    directory = args.output_dir or args.csv.parent / (args.csv.stem+'_analysis')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'summary.json'
    metadata_path = args.csv.with_suffix('.json')
    if metadata_path.exists():
        try:
            report['recording_metadata'] = json.loads(metadata_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            report['metadata_error'] = str(exc)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'分析报告: {path}')
    for index, segment in enumerate(report['segments'], 1):
        print(f"段 {index}: {segment['controller']} / {segment['phases']}, "
              f"稳定时间={segment['settling_time_s']} s, "
              f"最后视觉误差={segment['last_valid_horizontal_error_m']} m")
        print(f"  水平超调: right={segment['overshoot_right']['peak_m']} m, "
              f"forward={segment['overshoot_forward']['peak_m']} m")
    if not args.no_plots:
        try:
            plot(rows, directory/'flight.png')
            print(f'曲线: {directory / "flight.png"}')
        except ImportError:
            print('未安装 matplotlib，已输出统计报告；安装 matplotlib 后可生成曲线。')


if __name__ == '__main__':
    main()
