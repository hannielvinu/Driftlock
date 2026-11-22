#!/usr/bin/env python3
"""
Self-contained tests for Driftlock (no external data needed).

    python tests/test_driftlock.py        # or: python -m pytest tests -q

A synthetic flight (a level circle at constant turn rate, 200 Hz, 184 s, the length of
EuRoC MH_01) is turned into an EuRoC-format IMU file with noise. It is then damaged the
way the organisers' degrade_imu.py does: dropped rows, two 2-second dropouts, duplicated
rows, frozen rows, spikes and a 5-second all-zero failure tail. The tests check the Track 3
contract (format, one pose per row, budgets) and that the estimate stays bounded.
"""
import os
import sys
import tempfile
import time
import tracemalloc
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import main  # noqa: E402

G = 9.80665
FS = 200.0
DUR = 184.0
R_CIRCLE, W_TURN = 1.5, 0.6          # m, rad/s
T0_NS = 1403636579758555392
HEADER = "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"


def truth(t):
    th = W_TURN * t
    p = np.stack([R_CIRCLE * np.cos(th), R_CIRCLE * np.sin(th), 1.0 + 0.2 * np.sin(0.3 * t)], 1)
    aw = np.stack([-R_CIRCLE * W_TURN ** 2 * np.cos(th), -R_CIRCLE * W_TURN ** 2 * np.sin(th),
                   -0.2 * 0.09 * np.sin(0.3 * t)], 1)
    yaw = th + np.pi / 2
    return p, aw, yaw


def make_flight(path, seed=0, swap=False):
    rng = np.random.default_rng(seed)
    n = int(DUR * FS)
    t = np.arange(n) / FS
    ts = T0_NS + np.round(t * 1e9).astype(np.int64)
    p, aw, yaw = truth(t)
    f = aw + np.array([0, 0, G])                       # specific force, world frame
    c, s = np.cos(yaw), np.sin(yaw)
    acc = np.stack([c * f[:, 0] + s * f[:, 1], -s * f[:, 0] + c * f[:, 1], f[:, 2]], 1)  # body = R^T f
    gyr = np.zeros((n, 3))
    gyr[:, 2] = W_TURN
    sev = 0.02 + 0.98 * (t / t[-1]) ** 1.7
    acc += rng.standard_normal((n, 3)) * 0.05 * sev[:, None] + np.cumsum(rng.standard_normal((n, 3)) * 0.0006 * sev[:, None], 0) * 0.07
    gyr += rng.standard_normal((n, 3)) * 0.01 * sev[:, None]
    # spikes on all six channels at once
    sp = rng.choice(np.arange(int(2 * FS), n - int(6 * FS)), 25, replace=False)
    acc[sp] += rng.choice([-1, 1], (25, 3)) * 3.0
    gyr[sp] += rng.choice([-1, 1], (25, 3)) * 0.8
    # frozen rows
    for i in rng.choice(np.arange(int(2 * FS), n - int(6 * FS)), 40, replace=False):
        acc[i + 1:i + 6] = acc[i]
        gyr[i + 1:i + 6] = gyr[i]
    # dead tail: last 5 s exactly zero
    dead = t >= t[-1] - 5.0
    acc[dead] = 0.0
    gyr[dead] = 0.0
    keep = np.ones(n, bool)
    keep[rng.random(n) < 0.03 * sev] = False                         # single drops
    for frac in (0.5, 0.8):                                          # two 2-second dropouts
        s0 = frac * DUR
        keep[(t >= s0) & (t < s0 + 2.0)] = False
    keep[dead] = True
    keep[0] = True
    idx = np.flatnonzero(keep)
    dup = idx[rng.random(len(idx)) < 0.002]                          # duplicated rows
    idx = np.sort(np.concatenate([idx, dup]))
    cols = np.concatenate([gyr, acc], 1) if not swap else np.concatenate([acc, gyr], 1)
    with open(path, "w") as fh:
        fh.write(HEADER + "\n")
        for i in idx:
            fh.write(f"{ts[i]}," + ",".join(f"{v:.6f}" for v in cols[i]) + "\n")
    return ts[idx], t[idx], dead[idx]


def load_out(path):
    rows = [ln.split() for ln in open(path) if not ln.startswith("#")]
    return [r[0] for r in rows], np.array([[float(x) for x in r[1:]] for r in rows])


def umeyama_ate(est, gt):
    me, mg = est.mean(0), gt.mean(0)
    U, _, Vt = np.linalg.svd((gt - mg).T @ (est - me))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    al = (est - me) @ R.T + mg
    return float(np.sqrt(np.mean(np.sum((al - gt) ** 2, 1))))


TMP = tempfile.mkdtemp()
IMU = os.path.join(TMP, "data.csv")
TS, T, DEAD = make_flight(IMU)
OUT = os.path.join(TMP, "estimated_trajectory.txt")
STATS = main.run(IMU, OUT)


def test_one_pose_per_input_row_with_exact_timestamps():
    stamps, poses = load_out(OUT)
    assert len(stamps) == len(TS) == poses.shape[0]
    expected = [f"{v // 1_000_000_000}.{v % 1_000_000_000:09d}" for v in TS.tolist()]
    assert stamps == expected                                       # ns -> s, no rounding


def test_eight_columns_finite_and_unit_quaternions():
    _, poses = load_out(OUT)
    assert poses.shape[1] == 7
    assert np.all(np.isfinite(poses))
    assert np.allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1.0, atol=1e-5)


def test_forensics_find_the_damage():
    assert STATS["blind_rows"] == int(DEAD.sum())                   # every dead row is found
    assert STATS["gaps"] >= 2 and STATS["gap_seconds"] >= 4.0
    assert STATS["duplicate_rows"] > 0
    assert STATS["stuck_rows"] > 0
    assert STATS["spike_rows"] >= 15
    assert STATS["column_order"] == "gyro,accel"


def test_bounded_and_better_than_doing_nothing():
    _, poses = load_out(OUT)
    p_true, _, _ = truth(T)
    _, first = np.unique(TS, return_index=True)
    est, gt = poses[first, :3], p_true[first]
    ate = umeyama_ate(est, gt)
    floor = float(np.sqrt(np.mean(np.sum((gt - gt.mean(0)) ** 2, 1))))
    assert ate < floor, (ate, floor)
    # textbook double integration of the same file
    nv = os.path.join(TMP, "naive.txt")
    main.run(IMU, nv, p=dict(main.PARAMS, kp=0.0, ki=0.0, spike_thresh=1e9, hp_cut_hz=0.0, vel_hp=False))
    _, pn = load_out(nv)
    assert umeyama_ate(pn[first, :3], gt) > 10 * ate


def test_column_order_detected_from_data():
    swapped = os.path.join(TMP, "swapped.csv")
    make_flight(swapped, swap=True)
    out2 = os.path.join(TMP, "out2.txt")
    st = main.run(swapped, out2)
    assert st["column_order"] == "accel,gyro"
    assert open(out2).read() == open(OUT).read()


def test_zip_and_folder_inputs():
    d = os.path.join(TMP, "ds", "imu0")
    os.makedirs(d, exist_ok=True)
    with open(IMU) as src, open(os.path.join(d, "data.csv"), "w") as dst:
        dst.write(src.read())
    zp = os.path.join(TMP, "ds.zip")
    with zipfile.ZipFile(zp, "w") as z:
        z.write(os.path.join(d, "data.csv"), "imu0/data.csv")
    for inp in (os.path.join(TMP, "ds"), zp):
        o = os.path.join(TMP, "o.txt")
        main.run(inp, o)
        assert open(o).read() == open(OUT).read()


def test_memory_budget_8mb():
    tracemalloc.start()
    main.run(IMU, os.path.join(TMP, "m.txt"))
    peak = tracemalloc.get_traced_memory()[1] / 2 ** 20
    tracemalloc.stop()
    assert peak < 8.0, peak


def test_runtime_budget_20ms_per_imu_second():
    best = min(main.run(IMU, os.path.join(TMP, "r.txt"))["ms_per_imu_second"] for _ in range(3))
    assert best < 20.0, best


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.perf_counter()
            try:
                fn()
                print(f"PASS  {name}  ({time.perf_counter() - t0:.2f}s)")
            except AssertionError as e:
                fails += 1
                print(f"FAIL  {name}  {e}")
    print(f"\n{'all tests passed' if not fails else f'{fails} failed'}  ·  rows={STATS['rows']}  "
          f"ms_per_imu_second={STATS['ms_per_imu_second']:.2f}")
    sys.exit(1 if fails else 0)
