# AI Disclosure

This project was built with help from an AI coding assistant (Anthropic Claude, used through Claude Code).

What the AI assistant did:
- Read the official Track 3 material (slides, FAQ, `degrade_imu.py`) and summarised the constraints.
- Drafted the estimator (`src/main.py`), the evaluation, benchmark and report tools (`tools/`), the README and the presentation.
- Ran the experiments: degrading public data with the organisers' script, parameter sweeps, memory and runtime profiling, and cross-checking our metrics against the open-source `evo` tool.
- Wrote the demo-video narration and the scripts that build the video.

What the team did:
- Chose the track and the approach, reviewed and tested the code, checked every result and claim, and prepared the final submission.

Generated media:
- The video voice-over is synthetic speech from Piper (open-source text-to-speech, voice `en_US-ryan-high`).
- The background music was composed in code (`video/build.py`), so no third-party recording is used.
- Every number and screen in the video comes from real runs of this code on public data.

What no AI is used for:
- No AI model runs inside the estimator. `src/main.py` is plain signal processing (complementary filtering, interpolation, FFT filtering) in NumPy. It has no learned weights and makes no network calls.

Data: only public datasets (TUM-VI, EuRoC MAV), degraded with the organisers' `degrade_imu.py`. No hidden evaluation data was used.
