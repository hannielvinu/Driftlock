#!/usr/bin/env python3
"""
evaluate.py - score an estimated_trajectory.txt against ground truth, the way
the organisers describe it (evo, SE(3) Umeyama alignment, ATE RMSE + RPE),
plus the three Track 3 tiers we can reproduce locally:

  T1 waypoints : ATE RMSE on 1 Hz sampled ground-truth "checkpoints"
                 (the hidden waypoints are not published; 1 Hz is our proxy)
  T2 blind 5 s : ATE RMSE over the final 5 s (IMU failure window), using the
                 same global alignment
  T3 dropouts  : ATE RMSE over the 2 s that follow every timestamp gap

Usage
  python tools/evaluate.py <estimated_trajectory.txt> <gt.csv> <degraded_imu.csv>
      [--json out.json]

gt.csv is EuRoC/TUM-VI ground truth: ns, px, py, pz, qw, qx, qy, qz
"""
import argparse
import json
import sys

import numpy as np


def load_est(path):
    d = np.loadtxt(path, comments="#")
    return d[:, 0], d[:, 1:4], d[:, 4:8]          # t, p, q(xyzw)


def load_gt(path):
    d = np.loadtxt(path, delimiter=",", comments="#")
    q = d[:, [5, 6, 7, 4]]                           # -> xyzw
    return d[:, 0] * 1e-9, d[:, 1:4], q


def load_imu_t(path):
    d = np.loadtxt(path, delimiter=",", comments="#", usecols=0, dtype=np.int64)
    return d * 1e-9


def umeyama_se3(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cs, cd = src - mu_s, dst - mu_d
    U, _, Vt = np.linalg.svd(cd.T @ cs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    return R, mu_d - R @ mu_s


def qmat(q):
    x, y, z, w = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def rmse(e):
    return float(np.sqrt(np.mean(np.sum(e * e, axis=1)))) if len(e) else float("nan")


def evaluate(est_path, gt_path, imu_path):
    te, pe, qe = load_est(est_path)
    tg, pg, qg = load_gt(gt_path)
    ti = load_imu_t(imu_path)

    out = {"rows_imu": int(len(ti)), "rows_est": int(len(te)),
           "one_pose_per_frame": bool(len(ti) == len(te) and np.allclose(ti, te, atol=1e-6))}

    # associate each estimate with the nearest GT sample (max 10 ms), like evo
    _, first = np.unique(te, return_index=True)
    te, pe, qe = te[first], pe[first], qe[first]
    idx = np.clip(np.searchsorted(tg, te), 1, len(tg) - 1)
    idx -= (te - tg[idx - 1]) < (tg[idx] - te)
    ok = np.abs(tg[idx] - te) < 0.01
    te, pe, qe = te[ok], pe[ok], qe[ok]
    pg, qg = pg[idx[ok]], qg[idx[ok]]

    R, t = umeyama_se3(pe, pg)
    pa = pe @ R.T + t
    err = pa - pg
    out["ATE_rmse_m"] = rmse(err)
    out["GT_spread_m"] = rmse(pg - pg.mean(0))      # error of a "do-nothing" estimate

    # RPE over 1 s: translation (m) and rotation (deg)
    Rg, Re = qmat(qg), R @ qmat(qe)
    step = max(1, int(round(1.0 / np.median(np.diff(te)))))
    a, b = np.arange(0, len(te) - step), np.arange(step, len(te))
    dg = np.einsum("nji,nj->ni", Rg[a], pg[b] - pg[a])
    de = np.einsum("nji,nj->ni", Re[a], pa[b] - pa[a])
    out["RPE_trans_1s_m"] = rmse(de - dg)
    Rrel = np.einsum("nji,njk->nik", Rg[a], Rg[b])
    Erel = np.einsum("nji,njk->nik", Re[a], Re[b])
    D = np.einsum("nji,njk->nik", Rrel, Erel)
    ang = np.degrees(np.arccos(np.clip((np.trace(D, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    out["RPE_rot_1s_deg"] = float(np.sqrt(np.mean(ang ** 2)))

    # tiers
    t0, tend = te[0], te[-1]
    wp = np.zeros(len(te), bool)
    marks = np.arange(np.ceil(t0), tend, 1.0)
    wp[np.unique(np.clip(np.searchsorted(te, marks), 0, len(te) - 1))] = True
    out["T1_waypoint_ATE_m"] = rmse(err[wp])
    blind = te >= tend - 5.0
    out["T2_blind5s_ATE_m"] = rmse(err[blind])
    out["T2_blind5s_final_err_m"] = float(np.linalg.norm(err[-1]))
    dti = np.diff(ti)
    gaps = ti[1:][dti > 3 * np.median(dti)]
    post = np.zeros(len(te), bool)
    for g in gaps:
        post |= (te >= g) & (te < g + 2.0)
    post &= ~blind
    out["T3_postgap_ATE_m"] = rmse(err[post])
    out["n_gaps"] = int(len(gaps))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("est")
    ap.add_argument("gt")
    ap.add_argument("imu")
    ap.add_argument("--json")
    a = ap.parse_args()
    r = evaluate(a.est, a.gt, a.imu)
    print(json.dumps(r, indent=1))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
