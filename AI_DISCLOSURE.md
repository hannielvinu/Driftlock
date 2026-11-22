# AI Disclosure

This project was developed by Hanniel Vinu A (solo - Team HyperNova) with Claude Code as an AI pair-programmer.

**Team (direction and decisions):** chose Track 3 and the goal of a fully real, measurable solution with no mocks, and set the requirements and quality bar.

**Claude (implementation support, under the team's direction):** wrote and debugged the estimator and tools; ran the experiments (degrading public data with the organisers' script, parameter tuning, memory and runtime profiling); checked accuracy against the official `evo` tool.

**Inside the estimator:** no AI model is used. `src/main.py` is plain signal processing in NumPy, with no learned weights and no network calls.

**Demo video:** the voice-over is synthetic speech (Piper TTS).

**Data:** only public datasets (TUM-VI, EuRoC MAV). No hidden evaluation data was used.
