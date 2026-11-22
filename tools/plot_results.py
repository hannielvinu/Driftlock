#!/usr/bin/env python3
"""
plot_results.py - figures for the README / deck.

  python tools/plot_results.py <degraded.csv> <gt.csv> <out_prefix> [title]

Writes <out_prefix>_traj.png (top view: ground truth vs ours vs naive, all
SE(3)-aligned) and <out_prefix>_forensics.png (what the forensics stage found).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import main  # noqa: E402
import evaluate  # noqa: E402

INK, MUTED, GT_C, OURS_C, NAIVE_C, STILL_C = "#16213e", "#6b7280", "#111827", "#0f9d8a", "#e4572e", "#9ca3af"


def aligned(est_path, gt_path):
    te, pe, _ = evaluate.load_est(est_path)
    tg, pg, _ = evaluate.load_gt(gt_path)
    _, first = np.unique(te, return_index=True)
    te, pe = te[first], pe[first]
    idx = np.clip(np.searchsorted(tg, te), 1, len(tg) - 1)
    ok = np.abs(tg[idx] - te) < 0.01
    R, t = evaluate.umeyama_se3(pe[ok], pg[idx[ok]])
    return te[ok], pe[ok] @ R.T + t, pg[idx[ok]]


def main_():
    imu, gt, pre = sys.argv[1:4]
    title = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(imu)
    ours, naive = pre + "_ours.txt", pre + "_naive.txt"
    st = main.run(imu, ours)
    main.run(imu, naive, p=dict(main.PARAMS, kp=0.0, ki=0.0, spike_thresh=1e9, hp_cut_hz=0.0, vel_hp=False))
    t1, p1, g1 = aligned(ours, gt)
    t2, p2, g2 = aligned(naive, gt)
    r_o = evaluate.rmse(p1 - g1)
    r_n = evaluate.rmse(p2 - g2)
    r_s = evaluate.rmse(g1 - g1.mean(0))

    plt.rcParams.update({"font.family": "Inter", "font.size": 11})
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), dpi=150, gridspec_kw={"width_ratios": [1.15, 1]})
    a = ax[0]
    a.plot(g1[:, 0], g1[:, 1], color=GT_C, lw=1.6, label="Ground truth (hidden from us)")
    a.plot(p1[:, 0], p1[:, 1], color=OURS_C, lw=1.6, label=f"Path Finder  ATE {r_o:.2f} m")
    blind = t1 >= t1[-1] - 5
    a.plot(p1[blind, 0], p1[blind, 1], color=OURS_C, lw=3.2, alpha=0.5)
    a.scatter(g1[blind][-1, 0], g1[blind][-1, 1], s=60, color=GT_C, marker="*", zorder=5, label="Target (end of blind 5 s)")
    a.scatter(p1[blind][-1, 0], p1[blind][-1, 1], s=60, color=OURS_C, marker="*", zorder=5)
    a.set_aspect("equal", "datalim")
    a.set_xlabel("x (m)")
    a.set_ylabel("y (m)")
    a.set_title("Top view after SE(3) alignment", color=INK, fontsize=12, loc="left")
    a.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    a.grid(alpha=0.25)

    b = ax[1]
    names = ["Naive double\nintegration", "Do-nothing\n(all at origin)", "Path Finder"]
    vals = [r_n, r_s, r_o]
    cols = [NAIVE_C, STILL_C, OURS_C]
    bars = b.barh(names, vals, color=cols)
    b.set_xscale("log")
    for bar, v in zip(bars, vals):
        b.text(v * 1.15, bar.get_y() + bar.get_height() / 2, f"{v:,.2f} m" if v < 100 else f"{v:,.0f} m",
               va="center", color=INK, fontsize=11)
    b.set_xlim(min(vals) * 0.5, max(vals) * 8)
    b.set_title("ATE RMSE (log scale, lower is better)", color=INK, fontsize=12, loc="left")
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(pre + "_traj.png")
    plt.close(fig)

    # forensics timeline
    ts, raw = main.read_imu(imu)
    t = (ts - ts[0]) * 1e-9
    dt = np.diff(t)
    nominal = np.median(dt)
    fig, ax = plt.subplots(2, 1, figsize=(12, 4.6), dpi=150, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    gi = slice(0, 3) if st.get("column_order") == "gyro,accel" else slice(3, 6)
    ax[0].plot(t, np.linalg.norm(raw[:, gi], axis=1), color=MUTED, lw=0.5)
    ax[0].set_ylabel("|gyro| (rad/s)")
    dead = np.all(raw == 0, axis=1)
    if dead.any():
        ax[0].axvspan(t[dead][0], t[-1], color=NAIVE_C, alpha=0.15)
        ax[0].text(t[dead][0], ax[0].get_ylim()[1] * 0.85, "  IMU dead:\n  blind 5 s", color=NAIVE_C, fontsize=9, va="top")
    ax[0].set_title("What the forensics stage sees in the degraded stream", color=INK, fontsize=12, loc="left")
    gaps = np.flatnonzero(dt > 2.5 * nominal)
    ax[1].vlines(t[gaps], 0, dt[gaps], color=OURS_C, lw=1.2, label=f"time gaps ({len(gaps)})")
    dup = np.flatnonzero(dt == 0)
    ax[1].scatter(t[dup], np.full(len(dup), nominal), s=8, color=NAIVE_C, label=f"duplicate rows ({len(dup)})")
    ax[1].set_ylabel("gap (s)")
    ax[1].set_xlabel("time (s)")
    ax[1].legend(frameon=False, fontsize=9, loc="upper left")
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(pre + "_forensics.png")
    plt.close(fig)
    print(f"ours {r_o:.3f}  still {r_s:.3f}  naive {r_n:.1f}")


if __name__ == "__main__":
    main_()
