#!/usr/bin/env python3
"""
benchmark.py - run main.py over every degraded test file and summarise.

  python tools/benchmark.py <test_dir> <data_dir> [--set key=value ...] [--baseline]

test_dir holds <seq>_<variant>.csv from make_test_set.py; data_dir holds
dataset-<seq>_512_16/mav0/mocap0/data.csv (TUM-VI) or <seq>/mav0/state_groundtruth_estimate0/data.csv.
--baseline adds two reference methods: "still" (all poses at the origin) and
"naive" (plain strapdown double integration, no repair, no smoothing).
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import main  # noqa: E402
import evaluate  # noqa: E402

KEYS = ["ATE_rmse_m", "T1_waypoint_ATE_m", "T2_blind5s_ATE_m", "T3_postgap_ATE_m",
        "RPE_trans_1s_m", "RPE_rot_1s_deg", "GT_spread_m"]


def gt_path(data_dir, seq):
    for c in (f"{data_dir}/dataset-{seq}_512_16/mav0/mocap0/data.csv",
              f"{data_dir}/{seq}/mav0/state_groundtruth_estimate0/data.csv"):
        if os.path.exists(c):
            return c
    raise FileNotFoundError(seq)


def write_still(imu, out):
    ts = np.loadtxt(imu, delimiter=",", comments="#", usecols=0, dtype=np.int64)
    with open(out, "w") as f:
        for t in ts:
            f.write(f"{t * 1e-9:.9f} 0 0 0 0 0 0 1\n")


def write_naive(imu, out):
    p = dict(main.PARAMS, kp=0.0, ki=0.0, spike_thresh=1e9, hp_cut_hz=0.0, vel_hp=False)
    main.run(imu, out, p=p)


def bench(test_dir, data_dir, params, method="ours"):
    rows = []
    for f in sorted(glob.glob(os.path.join(test_dir, "*_*.csv"))):
        base = os.path.basename(f)[:-4]
        if base.startswith("_"):
            continue
        seq, var = base.rsplit("_", 1)
        out = os.path.join(test_dir, f"_est_{method}_{base}.txt")
        st = {}
        if method == "ours":
            st = main.run(f, out, p=params)
        elif method == "still":
            write_still(f, out)
        else:
            write_naive(f, out)
        r = evaluate.evaluate(out, gt_path(data_dir, seq), f)
        r.update(seq=seq, variant=var, ms_per_s=st.get("ms_per_imu_second", float("nan")))
        rows.append(r)
    return rows


def summary(rows, label):
    print(f"\n== {label}")
    print(f"{'file':22s}" + "".join(f"{k[:14]:>15s}" for k in KEYS))
    for r in rows:
        print(f"{r['seq'] + '_' + r['variant']:22s}" + "".join(f"{r[k]:15.3f}" for k in KEYS))
    for var in sorted({r["variant"] for r in rows}):
        sub = [r for r in rows if r["variant"] == var]
        print(f"{'MEAN ' + var:22s}" + "".join(f"{np.nanmean([r[k] for r in sub]):15.3f}" for k in KEYS))
    print(f"{'MEAN all':22s}" + "".join(f"{np.nanmean([r[k] for r in rows]):15.3f}" for k in KEYS))


if __name__ == "__main__":
    args = sys.argv[1:]
    test_dir, data_dir = args[0], args[1]
    params = dict(main.PARAMS)
    base = "--baseline" in args
    js = None
    for i, a in enumerate(args):
        if a == "--set":
            k, v = args[i + 1].split("=")
            params[k] = type(params.get(k, 0.0))(eval(v))
        if a == "--json":
            js = args[i + 1]
    rows = bench(test_dir, data_dir, params)
    summary(rows, f"ours {json.dumps({k: params[k] for k in params if params[k] != main.PARAMS.get(k)})}")
    if base:
        for m in ("still", "naive"):
            summary(bench(test_dir, data_dir, params, m), m)
    if js:
        json.dump(rows, open(js, "w"), indent=1)
