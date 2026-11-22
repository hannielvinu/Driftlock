# src: Driftlock entry point (Track 3: Path Finder)

```bash
python main.py <imu0/data.csv | evaluation.zip | folder> [estimated_trajectory.txt] [--meta metadata.json] [--profile]
```

- Input: EuRoC IMU CSV (`#timestamp [ns], w_RS_S_x..z [rad s^-1], a_RS_S_x..z [m s^-2]`). Header optional; column order (gyro/accel) and accel units (m/s² or g) are detected from the data.
- Output: `estimated_trajectory.txt`, space-separated `timestamp tx ty tz qx qy qz qw`, timestamp in seconds with full nanosecond precision, exactly one pose per input row.
- Requirements: Python 3.9+ and NumPy. Single thread, offline, no other dependencies.
- `--profile` prints the peak heap measured with `tracemalloc`; `--meta` writes run statistics (forensics counts, runtime) as JSON.

All tunable constants are in the `PARAMS` dictionary at the top of `main.py`. They were tuned on TUM-VI room1/2/4/6 and EuRoC V1_01/MH_01/MH_05, and checked on held-out EuRoC MH_03/V2_01/V2_02 (see `../results/RESULTS.md`).
