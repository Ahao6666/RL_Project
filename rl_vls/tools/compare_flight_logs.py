"""Compare real flight CSVs and estimate visual scale / effective velocity lag."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rl_vls.tools.analyze_flight_log import analyze, column


def fit_velocity(rows, axis):
    t = column(rows, 't_s')
    t = t-t[0]
    cmd, vel = column(rows, 'cmd_enu_'+axis), column(rows, 'vel_enu_'+axis)
    valid = np.isfinite(t+cmd+vel) & (column(rows, 'vel_enu_age_s') >= 0) & (column(rows, 'vel_enu_age_s') < 0.2)
    t, cmd, vel = t[valid], cmd[valid], vel[valid]
    if len(t) < 40 or t[-1] < 3 or np.ptp(cmd) < 0.05:
        return None
    grid = np.arange(0, t[-1], 0.1)
    u, v = np.interp(grid, t, cmd), np.interp(grid, t, vel)
    starts = np.arange(10, len(grid)-10, 2)
    if not len(starts):
        return None
    def rmse(delay, tau):
        predicted = v[starts].copy()
        for k in range(5):
            applied = np.interp(grid[starts+k]-delay, grid, u)
            predicted += (applied-predicted)*(1-np.exp(-0.1/tau))
        return float(np.sqrt(np.mean((predicted-v[starts+5])**2)))
    best = (float('inf'), 0., 0.)
    for delay in np.arange(0., 0.701, 0.05):
        for tau in np.arange(0.1, 2.501, 0.05):
            value = rmse(delay, tau)
            if value < best[0]:
                best = value, delay, tau
    return dict(prediction_horizon_s=0.5, rmse_m_s=best[0], effective_delay_s=float(best[1]),
                tau_s=float(best[2]), original_model_rmse_m_s=rmse(0., 0.95 if axis == 'up' else 1.3),
                interpretation='Local first-order approximation; excitation and closed-loop bias limit identification')


def compare(path):
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    report = analyze(rows)
    summaries = []
    # Match each analyzer segment independently, so restarts / manual takeover
    # are never identified as one continuous plant trajectory.
    for segment in report['segments']:
        active = [r for r in rows if r['phase'] in ('p', 'mpc', 'rl', 'fixed')
                  and float(r['t_s']) >= segment['start_s']-1e-7
                  and float(r['t_s']) <= segment['start_s']+segment['duration_s']+1e-7]
        yaw = np.radians(column(active, 'attitude_yaw_deg'))
        right, forward = column(active, 'marker_right'), column(active, 'marker_forward')
        relative = dict(east=forward*np.cos(yaw)+right*np.sin(yaw),
                        north=forward*np.sin(yaw)-right*np.cos(yaw), up=-column(active, 'marker_height'))
        scales = {}
        for axis, displacement in relative.items():
            position = column(active, 'pos_enu_'+axis)
            valid = np.isfinite(position+displacement) & (column(active, 'marker_fresh') == 1) \
                & (column(active, 'pose_age_s') < 0.2) & (column(active, 'marker_age_s') < 0.2)
            if np.count_nonzero(valid) > 10 and np.ptp(position[valid]) > 0.5:
                design = np.c_[position[valid], np.ones(np.count_nonzero(valid))]
                coef = np.linalg.lstsq(design, displacement[valid], rcond=None)[0]
                scales[axis] = dict(visual_to_physical_ratio=float(-coef[0]),
                                   residual_rms_m=float(np.sqrt(np.mean((displacement[valid]-design@coef)**2))),
                                   physical_span_m=float(np.ptp(position[valid])))
        first, last = active[0], active[-1]
        velocity_xy = [float(last['vel_enu_'+a]) for a in ('east', 'north')]
        end = rows[-1]
        drift = [float(end['pos_enu_'+a])-float(last['pos_enu_'+a]) for a in ('east', 'north')]
        summaries.append(dict(segment, initial_visual_error_m=float(np.hypot(float(first['marker_right']), float(first['marker_forward']))),
                              initial_visual_height_m=float(first['marker_height']),
                              handoff_horizontal_speed_m_s=float(np.linalg.norm(velocity_xy)),
                              recorded_post_control_drift_m=float(np.linalg.norm(drift)),
                              final_logged_armed=int(float(end['armed'])),
                              estimated_visual_scales=scales,
                              velocity_model_fits={axis: fit_velocity(active, axis) for axis in ('east', 'north', 'up')}))
    return dict(file=path.name, segments=summaries)


def plot_comparison(paths, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for index, path in enumerate(paths, 1):
        with path.open(encoding='utf-8', newline='') as f:
            rows = [r for r in csv.DictReader(f)
                    if r['phase'] in ('p', 'mpc', 'rl', 'fixed') and r['armed'] == '1']
        if not rows:
            continue
        t = column(rows, 't_s')
        t -= t[0]
        label = f"{index}: {rows[0]['controller'].upper()}"
        if rows[0]['controller'] == 'mpc' and path.with_suffix('.json').exists():
            parameters = json.loads(path.with_suffix('.json').read_text(encoding='utf-8')).get('parameters', {})
            label += ' revised' if 'mpc_delay_xy' in parameters else ' original'
        values = [
            np.hypot(column(rows, 'marker_right'), column(rows, 'marker_forward')),
            np.hypot(column(rows, 'vel_enu_east'), column(rows, 'vel_enu_north')),
            np.hypot(column(rows, 'cmd_enu_east'), column(rows, 'cmd_enu_north')),
            -column(rows, 'cmd_enu_up'), column(rows, 'marker_height'),
            np.hypot(column(rows, 'attitude_roll_deg'), column(rows, 'attitude_pitch_deg'))]
        for ax, value in zip(axes.flat, values):
            ax.plot(t, value, label=label)
    titles = ['Visual horizontal error, m', 'Measured horizontal speed, m/s',
              'Horizontal command magnitude, m/s', 'Downward command, m/s',
              'Visual depth, m', 'Roll/pitch magnitude, deg']
    axes[0, 0].axhline(0.3, linestyle='--', color='gray', linewidth=0.8)
    axes[0, 1].axhline(0.2, linestyle='--', color='gray', linewidth=0.8)
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel('Time since strategy activation, s')
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', nargs='*', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/flight_logs/comparison'))
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    paths = args.csv or sorted(Path('runs/flight_logs').glob('*.csv'))
    if not paths:
        parser.error('没有找到飞行 CSV')
    results = [compare(path) for path in paths]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(notes=['Initial conditions differ; these flights are not a controlled benchmark.',
                         'Scale fit assumes a stationary marker and matched camera/ENU axes.',
                         'Post-control drift is recorded movement, not independently measured touchdown error.',
                         'Estimated delays describe effective plant phase lag, not ROS message delivery latency.'], flights=results)
    (args.output_dir/'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    lines = ['# 飞行日志对比', '', '| 日志 | 初始偏差 / 高度 (m) | 控制时长 (s) | 稳定时间 (s) | 最后视觉误差 (m) | 交接水平速度 (m/s) | 后续记录位移 (m) |', '| --- | --- | --- | --- | --- | --- | --- |']
    for flight in results:
        for s in flight['segments']:
            stable = f"{s['settling_time_s']:.2f}" if s['settling_time_s'] is not None else '未持续稳定'
            error = f"{s['last_valid_horizontal_error_m']:.3f}" if s['last_valid_horizontal_error_m'] is not None else '无有效视觉'
            lines.append(f"| {flight['file']} | {s['initial_visual_error_m']:.2f} / {s['initial_visual_height_m']:.2f} | {s['duration_s']:.2f} | {stable} | {error} | {s['handoff_horizontal_speed_m_s']:.3f} | {s['recorded_post_control_drift_m']:.3f} |")
    lines += ['', '稳定：水平误差 ≤0.3m、水平速度 ≤0.2m/s，保持至少2秒且直到该控制段结束。',
              '', '最后视觉误差表示策略控制结束前的偏差；后续位移不是独立测量的落点误差。初始条件不同，需结合 JSON 中的武装状态判断日志是否完整。',
              '', '视觉尺度与速度响应拟合详见 comparison.json。拟合参数是近似值，需在新的 SITL 飞行中验证。']
    (args.output_dir/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))
    if not args.no_plots:
        try:
            plot_comparison(paths, args.output_dir/'comparison.png')
            print(f"对比曲线: {args.output_dir / 'comparison.png'}")
        except ImportError:
            print('未安装 matplotlib，已输出数值报告。')


if __name__ == '__main__':
    main()
