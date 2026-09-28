"""分析 step_response.py 采集的阶跃响应数据。

拟合内容：
  - 各轴速度阶跃的一阶惯性时间常数 tau（到达 63.2% 稳态值的时间）
  - cmd_vel_unstamped 坐标系约定（对比指令方向、ENU 实测速度方向与航向）

用法：
    python3 rl_vls/tools/analyze_step_response.py rl_vls/tools/step_response.csv
"""
from __future__ import annotations

import sys

import numpy as np


def fit_tau(t_seg: np.ndarray, v_seg: np.ndarray, v_ss: float, t_step: float):
    target = 0.632 * v_ss
    idx = np.where(np.abs(v_seg) >= abs(target))[0]
    return float(t_seg[idx[0]] - t_step) if len(idx) else None


def main(path: str):
    d = np.genfromtxt(path, delimiter=",", names=True)
    t = d["t"]
    print(f"共 {len(d)} 行, 时长 {t[-1]:.1f}s, 平均发布频率 {len(d) / t[-1]:.1f} Hz")
    print(f"航向 yaw: mean={np.degrees(np.mean(d['yaw_rad'])):.1f} deg, "
          f"范围 [{np.degrees(d['yaw_rad'].min()):.1f}, {np.degrees(d['yaw_rad'].max()):.1f}]")

    phases = [
        ("x 阶跃 (cmd_x=+1)", 3.0, 9.0, ["vb_x", "vl_x_enu", "vl_y_enu"]),
        ("y 阶跃 (cmd_y=+1)", 12.0, 18.0, ["vb_y", "vl_x_enu", "vl_y_enu"]),
        ("z 阶跃 (cmd_z=-0.5)", 21.0, 25.0, ["vb_z", "vl_z_enu"]),
    ]

    for name, t0, t1, axnames in phases:
        seg = (t >= t0) & (t < t1)
        settle = (t >= t1 - 1.5) & (t < t1)
        ss = {ax: float(np.mean(d[ax][settle])) for ax in axnames}
        main_ax = max(ss, key=lambda k: abs(ss[k]))
        tau = fit_tau(t[seg], d[main_ax][seg], ss[main_ax], t0)
        ss_str = "  ".join(f"{ax}={v:+.3f}" for ax, v in ss.items())
        print(f"\n== {name} ==")
        print(f"  稳态值: {ss_str}")
        if abs(ss[main_ax]) < 0.2:
            print("  响应幅度过小，指令可能未生效（检查模式是否为 GUIDED/QGUIDED）")
        elif tau is not None:
            print(f"  tau ≈ {tau:.2f} s (以 {main_ax} 拟合)")
        else:
            print("  tau 拟合失败")

    for label, t0, t1 in [("x 阶跃", 4.5, 9.0), ("y 阶跃", 13.5, 18.0)]:
        seg = (t >= t0) & (t < t1)
        vx_e = float(np.mean(d["vl_x_enu"][seg]))
        vy_e = float(np.mean(d["vl_y_enu"][seg]))
        yaw_mean = float(np.mean(d["yaw_rad"][seg]))
        if np.hypot(vx_e, vy_e) < 0.1:
            print(f"\n{label}: ENU 速度过小 ({vx_e:+.2f},{vy_e:+.2f})，无法判定坐标系")
            continue
        move_dir = np.arctan2(vy_e, vx_e)
        diff = np.degrees((move_dir - yaw_mean + np.pi) % (2 * np.pi) - np.pi)
        print(f"\n== 坐标系判定（{label}）==")
        print(f"  ENU 速度: ({vx_e:+.3f}, {vy_e:+.3f})，移动方向 {np.degrees(move_dir):+.1f} deg")
        print(f"  机头航向 {np.degrees(yaw_mean):+.1f} deg，移动方向-航向 = {diff:+.1f} deg")
        if abs(diff) < 30:
            print("  => cmd_vel 为【机体系】语义（移动方向≈机头航向）")
        else:
            print("  => 移动方向与航向不一致，cmd_vel 可能为【本地ENU系】语义")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rl_vls/tools/step_response.csv")
