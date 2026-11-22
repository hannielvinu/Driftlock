# Demo video: how it was made

`docs/Driftlock_demo.mp4` (2:04, 1920×1080, H.264 + AAC) is built entirely from code in `video/`:

1. **Script:** one narration line per section (hook, stakes, problem, live run, forensics, pipeline, blind 5 s, proof, results, budget, close), in `video/narration.json`.
2. **Voice:** Piper TTS (`en_US-ryan-high`, length scale 0.9, slightly faster than default) turns each line into a WAV file.
3. **Visuals:** real recordings of the mission-report UI and of real terminal runs (`video/record.cjs`, `video/terminal.html`), plus deck slides, with a slow eased zoom and bold captions drawn by ffmpeg.
4. **Sync:** each section lasts exactly as long as its narration plus a short pause.
5. **Music:** a soft four-chord pad (Am–F–C–G) composed in Python inside `video/build.py`, ducked under the voice with ffmpeg's sidechain compressor.
6. **Export:** MP4, H.264 video, AAC audio.

Rebuild: synthesise `av/<id>.wav` with Piper for each narration line, record the clips, render slides to `slides/s-N.png`, then `python video/build.py <work_dir> docs/Driftlock_demo.mp4`.
