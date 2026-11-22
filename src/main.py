#!/usr/bin/env python3
"""
Driftlock (Track 3: Path Finder) - 6-DoF trajectory from a degraded IMU stream (Track 3).

    python main.py <imu0/data.csv | dataset.zip | folder> [estimated_trajectory.txt]
                   [--meta metadata.json]

Pipeline (offline, single thread, numpy only)
  1. Stream-parse the CSV into compact arrays (no pandas, no copies).
  2. Forensics: find the dead-sensor tail (all-zero rows), duplicate
     timestamps, stuck/frozen rows, impulsive spikes and time gaps.
  3. Repair: flagged rows are rebuilt by time interpolation; gaps are bridged.
  4. Attitude: forward + backward Mahony filters (gyro integration, gravity
     correction weighted by how "clean" each accel sample looks), fused.
  5. Position: gravity-free world acceleration -> velocity -> position, with
     a zero-phase, edge-safe high-pass that removes the bias-driven drift
     while keeping real motion (offline smoothing: future samples are used).
  6. Blind window: after the IMU dies, dead-reckon position and attitude
     with a decaying constant-velocity / constant-rate model.
  7. Write exactly one TUM pose per input IMU row.
"""
import math
import os
import sys
import time
import zipfile
import io

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

PARAMS = {
    "init_seconds": 0.5,      # window used to level the initial attitude
    "kp": 0.5,                # Mahony proportional gain (rad/s per unit error)
    "ki": 0.005,              # Mahony integral gain (gyro bias)
    "acc_trust_sigma": 0.6,   # m/s^2; accel trusted when | |a| - g | is small
    "spike_thresh": 8.0,      # robust z threshold for impulsive spikes
    "hp_cut_hz": 0.12,       # high-pass corner for the drift removal (tuned)
    "hp_order": 2,            # steepness of the high-pass
    "vel_hp": True,           # also high-pass velocity before integration
    "snap_gap_s": 0.1,        # s, gaps longer than this re-level the tilt
    "blind_vel_tau": 1.0,     # s, velocity decay during the blind window
    "blind_rate_tau": 0.5,    # s, angular-rate decay during the blind window
    "blind_vel_avg": 0.25,    # s, velocity averaged at failure time
}


# --------------------------------------------------------------------------
# 1. parsing
# --------------------------------------------------------------------------
def _open_text(path):
    if os.path.isdir(path):
        for cand in (os.path.join(path, "data.csv"), os.path.join(path, "imu0", "data.csv"),
                     os.path.join(path, "mav0", "imu0", "data.csv")):
            if os.path.exists(cand):
                return open(cand, "r", encoding="utf-8-sig")
        raise FileNotFoundError(f"no imu0/data.csv under {path}")
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        names = [n for n in z.namelist() if n.endswith("imu0/data.csv")] or \
                [n for n in z.namelist() if n.endswith(".csv")]
        return io.TextIOWrapper(z.open(names[0]), encoding="utf-8-sig")
    return open(path, "r", encoding="utf-8-sig")


def read_imu(path):
    """Return int64 ns timestamps and an (n,6) float32 array in file column order."""
    with _open_text(path) as f:
        n = sum(1 for line in f if line.strip() and line.lstrip()[0] not in "#tT")
    ts = np.empty(n, np.int64)
    val = np.empty((n, 6), np.float32)
    i = 0
    with _open_text(path) as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#tT":
                continue
            parts = s.split(",")
            try:
                tok = parts[0].strip()
                ts[i] = int(tok) if tok.lstrip("-").isdigit() else int(round(float(tok) * 1e9))
                val[i] = [float(x) for x in parts[1:7]]
            except ValueError:
                continue
            i += 1
    return ts[:i], val[:i]


# --------------------------------------------------------------------------
# small quaternion helpers (w, x, y, z), pure python for the sequential loops
# --------------------------------------------------------------------------
def q_from_two(a, b):
    """Rotation that maps unit vector a onto unit vector b."""
    c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    w = 1.0 + c
    if w < 1e-9:
        return (0.0, 1.0, 0.0, 0.0)
    nrm = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / nrm, x / nrm, y / nrm, z / nrm)


def mahony(gyr, acc, dt, q0, g, p, order):
    """Complementary attitude filter. Returns (n,4) float64 quaternions (body->world)."""
    n = len(dt)
    out = np.empty((n, 4), np.float32)
    kp, ki, sig = p["kp"], p["ki"], p["acc_trust_sigma"]
    w, x, y, z = q0
    bx = by = bz = 0.0
    snap = False
    snap_dt = p.get("snap_gap_s", 0.1)
    sgn = 1.0 if order > 0 else -1.0
    CH = 2048                                    # convert to python floats in small chunks
    starts = range(0, n, CH) if order > 0 else range((n - 1) // CH * CH, -1, -CH)
    for c0 in starts:
      c1 = min(n, c0 + CH)
      G = gyr[c0:c1].tolist()
      A = acc[c0:c1].tolist()
      D = dt[max(0, c0 - 1):c1].tolist()
      off = 1 if c0 > 0 else 0
      Q = []
      rng = range(c1 - c0) if order > 0 else range(c1 - c0 - 1, -1, -1)
      for j in rng:
        k = c0 + j
        gx, gy, gz = G[j]
        ax, ay, az = A[j]
        an = math.sqrt(ax * ax + ay * ay + az * az)
        if snap and an > 1e-6:
            # after a long gap the tilt is re-levelled from gravity at once
            # (yaw is kept), so no tilt error leaks gravity into position
            vx = 2 * (x * z - w * y)
            vy = 2 * (w * x + y * z)
            vz = w * w - x * x - y * y + z * z
            cw, cx, cy, cz = q_from_two((vx, vy, vz), (ax / an, ay / an, az / an))
            w, x, y, z = (w * cw + x * cx + y * cy + z * cz,
                          -w * cx + x * cw - y * cz + z * cy,
                          -w * cy + x * cz + y * cw - z * cx,
                          -w * cz - x * cy + y * cx + z * cw)
            snap = False
        Q.append((j, w, x, y, z))
        ex = ey = ez = 0.0
        if an > 1e-6:
            trust = math.exp(-((an - g) / sig) ** 2)
            ax /= an; ay /= an; az /= an
            # world z expressed in body frame
            vx = 2 * (x * z - w * y)
            vy = 2 * (w * x + y * z)
            vz = w * w - x * x - y * y + z * z
            ex = (ay * vz - az * vy) * trust
            ey = (az * vx - ax * vz) * trust
            ez = (ax * vy - ay * vx) * trust
        h = D[j + off] if order > 0 else (D[j + off - 1] if k > 0 else D[0])
        snap = h > snap_dt
        bx += sgn * ki * ex * h; by += sgn * ki * ey * h; bz += sgn * ki * ez * h
        gx = sgn * (gx + bx) + kp * ex
        gy = sgn * (gy + by) + kp * ey
        gz = sgn * (gz + bz) + kp * ez
        hh = 0.5 * h
        w, x, y, z = (w + hh * (-x * gx - y * gy - z * gz),
                      x + hh * (w * gx + y * gz - z * gy),
                      y + hh * (w * gy - x * gz + z * gx),
                      z + hh * (w * gz + x * gy - y * gx))
        nrm = 1.0 / math.sqrt(w * w + x * x + y * y + z * z)
        w *= nrm; x *= nrm; y *= nrm; z *= nrm
      Q.sort()
      out[c0:c1] = [r[1:] for r in Q]
    return out


def rotate_inplace(q, v, blk=4096):
    """Rotate (n,3) body vectors in place by (n,4) quaternions (w,x,y,z) -> world."""
    for s0 in range(0, len(v), blk):
        qq = q[s0:s0 + blk].astype(np.float64)
        w, x, y, z = qq[:, 0], qq[:, 1], qq[:, 2], qq[:, 3]
        vv = v[s0:s0 + blk]
        vx, vy, vz = vv[:, 0].copy(), vv[:, 1].copy(), vv[:, 2].copy()
        tx = 2 * (y * vz - z * vy)
        ty = 2 * (z * vx - x * vz)
        tz = 2 * (x * vy - y * vx)
        vv[:, 0] = vx + w * tx + (y * tz - z * ty)
        vv[:, 1] = vy + w * ty + (z * tx - x * tz)
        vv[:, 2] = vz + w * tz + (x * ty - y * tx)


# --------------------------------------------------------------------------
# zero-phase high-pass (FFT, reflect-padded so the ends do not ring)
# --------------------------------------------------------------------------
def highpass(sig, fs, fc, order):
    """Zero-phase high-pass of each column of sig, in place, one axis at a time."""
    n = len(sig)
    if n < 8 or fc <= 0:
        sig -= sig.mean(0)
        return sig
    pad = min(n - 1, int(3 * fs / fc))
    for c in range(sig.shape[1]):
        x = sig[:, c]
        ext = np.concatenate([2 * x[0] - x[pad:0:-1], x, 2 * x[-1] - x[-2:-pad - 2:-1]])
        m = len(ext)
        ext -= np.linspace(ext[0], ext[-1], m)
        F = np.fft.rfft(ext)
        del ext
        f = np.fft.rfftfreq(m, 1.0 / fs)
        f[0] = 1e-12
        F *= 1.0 / np.sqrt(1.0 + (fc / f) ** (2 * order))
        F[0] = 0.0
        del f
        x[:] = np.fft.irfft(F, m)[pad:pad + n]
        del F
    return sig


# --------------------------------------------------------------------------
# main estimator
# --------------------------------------------------------------------------
def estimate(ts, holder, p=PARAMS):
    """holder = [raw]; raw is released as soon as it has been split (memory)."""
    stats = {}
    raw = holder.pop()
    n_rows = len(ts)

    # --- column order: the accel triplet is the one whose norm sits near g (or 1 g)
    def med_norm(sl):
        sub = raw[::7, sl].astype(np.float64)
        return float(np.median(np.sqrt((sub * sub).sum(1))))
    na, nb = med_norm(slice(0, 3)), med_norm(slice(3, 6))
    score_a = min(abs(na - 9.81), abs(na - 1.0) * 9.81)
    score_b = min(abs(nb - 9.81), abs(nb - 1.0) * 9.81)
    gi, ai = (slice(0, 3), slice(3, 6)) if score_b <= score_a else (slice(3, 6), slice(0, 3))
    acc_norm = nb if gi.start == 0 else na
    acc_mul = 9.80665 if acc_norm < 3.0 else 1.0
    stats["column_order"] = "gyro,accel" if gi.start == 0 else "accel,gyro"

    # --- dead tail: trailing rows whose six values are exactly zero
    nz = np.flatnonzero(np.any(raw != 0.0, axis=1))
    last_valid = int(nz[-1]) if len(nz) else n_rows - 1
    del nz
    stats["blind_rows"] = n_rows - 1 - last_valid

    # --- unique timestamps in the live part (duplicated rows share one pose)
    live_ts = ts[:last_valid + 1]
    dup = np.zeros(len(live_ts), bool)
    dup[1:] = live_ts[1:] == live_ts[:-1]
    stats["duplicate_rows"] = int(dup.sum())
    keep = ~dup
    t_ns = live_ts[keep]
    gyr = raw[:last_valid + 1][keep][:, gi].astype(np.float64)
    acc = raw[:last_valid + 1][keep][:, ai].astype(np.float64)
    del raw, keep
    if acc_mul != 1.0:
        acc *= acc_mul
    n = len(t_ns)
    dt = np.empty(n)
    dt[:-1] = np.diff(t_ns) * 1e-9
    dt[-1] = np.median(dt[:-1]) if n > 1 else 0.005
    nominal = float(np.median(dt))
    fs = 1.0 / nominal
    big = np.flatnonzero(dt > 2.5 * nominal)
    stats["gaps"] = int(len(big))
    stats["gap_seconds"] = float(dt[big].sum())

    # --- stuck rows: exactly identical to the previous row
    bad = np.zeros(n, bool)
    bad[1:] = np.all(gyr[1:] == gyr[:-1], axis=1) & np.all(acc[1:] == acc[:-1], axis=1)
    stats["stuck_rows"] = int(bad.sum())
    events = {"stuck": np.flatnonzero(bad).astype(np.int32)}   # small index lists, for reports

    # --- spikes: robust z-score of each channel vs a 7-sample running median;
    #     a row is a spike when at least 3 of its 6 channels jump together
    k, blk = 3, 4096
    scale = np.empty(6)
    for c in range(3):
        scale[c] = np.median(np.abs(np.diff(gyr[:, c])))
        scale[3 + c] = np.median(np.abs(np.diff(acc[:, c])))
    scale = 1.4826 * scale / math.sqrt(2) + 1e-9
    spike = np.zeros(n, bool)
    for s0 in range(0, n, blk):
        lo, hi = max(0, s0 - k), min(n, s0 + blk + k)
        votes = np.zeros(min(blk, n - s0), np.int8)
        for c in range(6):
            arr = gyr[lo:hi, c] if c < 3 else acc[lo:hi, c - 3]
            if hi - lo > 2 * k:
                med = np.median(np.lib.stride_tricks.sliding_window_view(arr, 2 * k + 1), axis=1)
                med = np.concatenate([np.full(k, med[0]), med, np.full(k, med[-1])])
            else:
                med = np.full(hi - lo, np.median(arr))
            dev = np.abs(arr - med)[s0 - lo:s0 - lo + len(votes)] / scale[c]
            votes += (dev > p["spike_thresh"]).astype(np.int8)
        spike[s0:s0 + len(votes)] = votes >= 3
    bad |= spike
    stats["spike_rows"] = int(spike.sum())
    events["spike"] = np.flatnonzero(spike).astype(np.int32)
    del spike

    # --- repair flagged rows by time interpolation
    if bad.any() and (~bad).sum() > 2:
        t = (t_ns - t_ns[0]) * 1e-9
        tb, tg = t[bad], t[~bad]
        for c in range(3):
            gyr[bad, c] = np.interp(tb, tg, gyr[~bad, c])
            acc[bad, c] = np.interp(tb, tg, acc[~bad, c])
        del t, tb, tg
    del bad

    # --- bridge time gaps: rotate through a gap at the mean of the rates on
    #     both sides (instead of holding one noisy sample for the whole gap)
    for k_ in big:
        if k_ + 1 < n:
            mid = 0.5 * (gyr[k_] + gyr[k_ + 1])
            gyr[k_] = mid
            gyr[k_ + 1] = mid

    # --- gravity magnitude and initial attitude
    g = float(np.median(np.sqrt((acc * acc).sum(1))))
    m0 = max(5, int(p["init_seconds"] * fs))
    a0 = acc[:m0].mean(0)
    a0 = a0 / np.linalg.norm(a0)
    q0 = q_from_two((a0[0], a0[1], a0[2]), (0.0, 0.0, 1.0))

    # --- attitude: forward filter, backward filter seeded by the forward end,
    #     blended so each part of the run leans on the better-informed pass
    q = mahony(gyr, acc, dt, q0, g, p, +1)
    qb = mahony(gyr, acc, dt, tuple(float(v) for v in q[-1]), g, p, -1)
    navg = max(2, int(p["blind_vel_avg"] * fs))
    w_end = gyr[-navg:].mean(0)
    del gyr
    for s0 in range(0, n, blk):
        a_, b_ = q[s0:s0 + blk].astype(np.float64), qb[s0:s0 + blk].astype(np.float64)
        sg = np.sign((a_ * b_).sum(1))[:, None]
        wgt = (1.0 - np.arange(s0, s0 + len(a_)) / max(1, n - 1))[:, None]
        m = a_ * wgt + b_ * sg * (1 - wgt)
        q[s0:s0 + blk] = m / np.sqrt((m * m).sum(1))[:, None]
    del qb

    # --- position: world accel without gravity -> velocity -> position,
    #     zero-phase high-pass removes bias-driven drift (offline smoothing)
    if p.get("_stop_after_attitude"):
        return {"t_ns": t_ns, "q": q, "acc": acc, "dt": dt, "g": g, "fs": fs, "dup": dup,
                "last_valid": last_valid, "stats": stats, "w_end": w_end}
    rotate_inplace(q, acc)
    acc[:, 2] -= g
    acc -= acc.mean(0)
    # Put world acceleration on a uniform time grid so the filters work in
    # real time even across gaps; missing slots get zero (mean) acceleration,
    # i.e. velocity is held through a dropout instead of being kicked by one
    # noisy sample stretched over the whole gap.
    gidx = np.rint((t_ns - t_ns[0]) * (1e-9 * fs)).astype(np.int64)
    np.maximum.accumulate(gidx, out=gidx)
    m = int(gidx[-1]) + 1
    grid = np.zeros((m, 3))
    grid[gidx] = acc
    del acc
    h = 1.0 / fs
    grid *= h
    np.cumsum(grid, axis=0, out=grid)        # velocity
    if p["vel_hp"]:
        highpass(grid, fs, p["hp_cut_hz"], p["hp_order"])
    grid *= h
    np.cumsum(grid, axis=0, out=grid)        # position
    highpass(grid, fs, p["hp_cut_hz"], p["hp_order"])
    pos = grid[gidx]
    del grid, gidx

    return {"t_ns": t_ns, "dup": dup, "q": q, "pos": pos, "w_end": w_end, "events": events,
            "fs": fs, "last_valid": last_valid, "stats": stats}


def write_output(path, ts, est, p=PARAMS):
    """One pose per input row. Duplicates reuse their pose; blind rows are dead-reckoned."""
    t_ns, dup, q, pos = est["t_ns"], est["dup"], est["q"], est["pos"]
    last_valid, fs = est["last_valid"], est["fs"]
    n_rows = len(ts)

    # state at failure time for dead reckoning
    navg = max(2, int(p["blind_vel_avg"] * fs))
    v_end = (pos[-1] - pos[-navg]) / ((t_ns[-1] - t_ns[-navg]) * 1e-9)
    w_end = est["w_end"]
    tv, tr = p["blind_vel_tau"], p["blind_rate_tau"]

    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        k = -1
        for i in range(n_rows):
            ns = int(ts[i])
            if i <= last_valid:
                if not dup[i]:
                    k += 1
                pw, px, py, pz = (float(v) for v in q[k])
                x, y, z = (float(v) for v in pos[k])
            else:
                tau = (ns - int(t_ns[-1])) * 1e-9
                fv = tv * (1 - math.exp(-tau / tv)) if tv > 0 else tau
                fr = tr * (1 - math.exp(-tau / tr)) if tr > 0 else tau
                x, y, z = pos[-1] + v_end * fv
                pw, px, py, pz = q[-1]
                ang = w_end * fr
                th = float(np.linalg.norm(ang))
                if th > 1e-12:
                    ax_, ay_, az_ = ang / th
                    s, c = math.sin(th / 2), math.cos(th / 2)
                    rw, rx, ry, rz = c, ax_ * s, ay_ * s, az_ * s
                    pw, px, py, pz = (pw * rw - px * rx - py * ry - pz * rz,
                                      pw * rx + px * rw + py * rz - pz * ry,
                                      pw * ry - px * rz + py * rw + pz * rx,
                                      pw * rz + px * ry - py * rx + pz * rw)
            x, y, z = round(x, 6) + 0.0, round(y, 6) + 0.0, round(z, 6) + 0.0   # no "-0.000000"
            sec, frac = divmod(ns, 1_000_000_000)
            f.write(f"{sec}.{frac:09d} {x:.6f} {y:.6f} {z:.6f} {px:.8f} {py:.8f} {pz:.8f} {pw:.8f}\n")


def run(inp, outp, meta=None, p=PARAMS):
    t0 = time.perf_counter()
    ts, raw = read_imu(inp)
    holder = [raw]
    del raw
    est = estimate(ts, holder, p)
    write_output(outp, ts, est, p)
    el = time.perf_counter() - t0
    st = est["stats"]
    dur = (int(ts[-1]) - int(ts[0])) * 1e-9
    st.update({"rows": int(len(ts)), "duration_s": dur, "runtime_s": el,
               "ms_per_imu_second": 1000 * el / max(dur, 1e-9)})
    return st


def main():
    args = [a for a in sys.argv[1:]]
    meta = None
    if "--meta" in args:
        i = args.index("--meta")
        meta = args[i + 1]
        del args[i:i + 2]
    profile = "--profile" in args
    if profile:
        args.remove("--profile")
    if not args:
        sys.exit(__doc__)
    inp = args[0]
    outp = args[1] if len(args) > 1 else "estimated_trajectory.txt"
    if profile:
        import tracemalloc
        tracemalloc.start()
    st = run(inp, outp, meta)
    if profile:
        st["peak_heap_MB"] = tracemalloc.get_traced_memory()[1] / 2 ** 20
        tracemalloc.stop()
    if meta:
        import json
        with open(meta, "w") as f:
            json.dump(st, f, indent=2)
    print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in st.items()))


if __name__ == "__main__":
    main()
