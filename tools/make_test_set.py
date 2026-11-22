#!/usr/bin/env python3
"""
make_test_set.py - build degraded test inputs from a clean EuRoC-format IMU file
with the organisers' degrade_imu.py, in three variants:

  asis     : degrade_imu.py run directly on the EuRoC file. Note: EuRoC
             columns are gyro-then-accel, while degrade_imu.py assumes
             accel-then-gyro, so this variant puts the "accel" noise on the gyro.
  intended : columns re-ordered to accel-then-gyro before degrading and put
             back afterwards, so each sensor receives the noise meant for it.
  dropout2s: 'intended' plus two 2-second chunks of rows removed (FAQ: "2-second
             dropouts ... physically MISSING rows").

Usage
  python tools/make_test_set.py <clean imu0/data.csv> <out_dir> <name> <degrade_imu.py>
"""
import os
import subprocess
import sys

import numpy as np


def main():
    src, out, name, script = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    header = open(src).readline().rstrip("\n")
    d = np.loadtxt(src, delimiter=",", comments="#")
    ts = np.loadtxt(src, delimiter=",", comments="#", usecols=0, dtype=np.int64)

    def save(path, arr, tsv):
        with open(path, "w") as f:
            f.write(header + "\n")
            for t, r in zip(tsv, arr):
                f.write(f"{t}," + ",".join(f"{v:.6f}" for v in r) + "\n")

    # as-is
    asis = os.path.join(out, f"{name}_asis.csv")
    subprocess.run([sys.executable, "-I", script, src, asis], check=True, stdout=subprocess.DEVNULL)

    # intended
    tmp_in = os.path.join(out, f"_{name}_acc_first.csv")
    tmp_out = os.path.join(out, f"_{name}_acc_first_deg.csv")
    save(tmp_in, d[:, [4, 5, 6, 1, 2, 3]], ts)
    subprocess.run([sys.executable, "-I", script, tmp_in, tmp_out], check=True, stdout=subprocess.DEVNULL)
    e = np.loadtxt(tmp_out, delimiter=",", comments="#")
    ets = np.loadtxt(tmp_out, delimiter=",", comments="#", usecols=0, dtype=np.int64)
    save(os.path.join(out, f"{name}_intended.csv"), e[:, [4, 5, 6, 1, 2, 3]], ets)
    os.remove(tmp_in)
    os.remove(tmp_out)

    # two 2-second dropouts in the second half (before the 5 s failure window)
    te = e[:, 0] * 1e-9
    T0, T1 = te[0], te[-1]
    keep = np.ones(len(e), bool)
    for frac in (0.55, 0.80):
        s = T0 + frac * (T1 - T0)
        keep &= ~((te >= s) & (te < s + 2.0))
    save(os.path.join(out, f"{name}_dropout2s.csv"), e[keep][:, [4, 5, 6, 1, 2, 3]], ets[keep])


if __name__ == "__main__":
    main()
