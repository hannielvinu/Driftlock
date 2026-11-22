#!/usr/bin/env python3
"""
build.py - assembles the Driftlock demo video.

  python video/build.py <work_dir> <out.mp4>

work_dir must contain:
  av/<id>.wav          Piper narration per section (see narration.json)
  clips/*.webm         real UI / terminal recordings (video/record.cjs)
  slides/s-N.png       deck slides rendered at 1920x1080, slides/outro.png

Each section lasts exactly as long as its narration plus a short pause. Visuals get a
slow zoom/pan and a bold caption. The music bed is composed here (four-chord pad), so
there is no third-party audio; ffmpeg ducks it under the voice. Output: H.264 + AAC MP4.
"""
import json
import math
import os
import subprocess
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W, OUT = sys.argv[1], sys.argv[2]
FPS = 30
PAD = 0.65
FB = os.path.expanduser("~/.fonts/Poppins-Bold.ttf")
FR = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"

# visual id -> (source, start offset s, zoom target, focus x, focus y) ; focus in 1920x1080 px
VIS = {
    "ui_room2_play": ("clips/room2_play.webm", 0.8, 1.35, 700, 430),
    "ui_room2_score": ("clips/room2_play.webm", 10.5, 1.9, 1640, 240),
    "slide_2": ("slides/s-2.png", 0, 1.10, 960, 540),
    "term1": ("clips/term1.webm", 0.0, 1.06, 960, 470),
    "ui_timeline": ("clips/room2_play.webm", 1.0, 1.55, 1150, 760),
    "slide_4": ("slides/s-4.png", 0, 1.10, 960, 400),
    "ui_blind": ("clips/room2_blind.webm", 0.8, 1.5, 760, 400),
    "term2": ("clips/term2.webm", 0.0, 1.06, 960, 450),
    "slide_8": ("slides/s-8.png", 0, 1.12, 960, 500),
    "ui_drone": ("clips/v201_play.webm", 0.8, 1.3, 760, 420),
    "outro": ("slides/outro.png", 0, 1.06, 960, 540),
}


def run(*a):
    subprocess.run(list(a), check=True)


def dur_wav(p):
    w = wave.open(p)
    return w.getnframes() / w.getframerate()


def dur_media(p):
    o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                       capture_output=True, text=True).stdout.strip()
    return float(o) if o and o != "N/A" else 0.0


def zoom_filter(L, z1, fx, fy):
    n = int(L * FPS) + 2
    # ease-in-out zoom from 1 to z1 over the section, drifting toward the focus point
    z = f"1+({z1}-1)*(0.5-0.5*cos(PI*min(on/{n},1)))"
    x = f"max(0,min(iw-iw/zoom,{fx * 2}-iw/zoom/2))"
    y = f"max(0,min(ih-ih/zoom,{fy * 2}-ih/zoom/2))"
    return f"scale=3840:2160,zoompan=z='{z}':x='{x}':y='{y}':d=1:s=1920x1080:fps={FPS}"


def section(i, item, L, txtdir):
    src, off, z1, fx, fy = VIS[item["visual"]]
    src = os.path.join(W, src)
    cap = os.path.join(txtdir, f"cap{i}.txt")
    open(cap, "w").write(item["caption"])
    draw = (f"drawtext=fontfile={FB}:textfile={cap}:fontcolor=white:fontsize=46:x=90:y=h-150:"
            f"box=1:boxcolor=0x070b16@0.78:boxborderw=26:alpha='min(1,max(0,(t-0.25)/0.4))'")
    tag = os.path.join(txtdir, "tag.txt")
    open(tag, "w").write("DRIFTLOCK  ·  TRACK 03")
    draw2 = f"drawtext=fontfile={FR}:textfile={tag}:fontcolor=0x22d3ee:fontsize=24:x=w-tw-60:y=46"
    fade = f"fade=t=in:st=0:d=0.25,fade=t=out:st={L - 0.25:.2f}:d=0.25"
    # slides already carry their own text: no caption on top of them
    overlays = [] if src.endswith(".png") else [draw]
    out = os.path.join(W, f"sec{i:02d}.mp4")
    if src.endswith(".png"):
        vf = ",".join([zoom_filter(L, z1, fx, fy)] + overlays + [fade, "format=yuv420p"])
        run("ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", str(FPS), "-i", src, "-t", f"{L:.3f}",
            "-vf", vf, "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-r", str(FPS), out)
    else:
        avail = dur_media(src) - off
        pre = [f"fps={FPS}"]
        if avail < L:
            pre.append(f"tpad=stop_mode=clone:stop_duration={L - avail + 0.2:.2f}")
        vf = ",".join(pre + [zoom_filter(L, z1, fx, fy)] + overlays + [fade, "format=yuv420p"])
        run("ffmpeg", "-y", "-loglevel", "error", "-ss", f"{off:.2f}", "-i", src, "-t", f"{L:.3f}", "-an",
            "-vf", vf, "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-r", str(FPS), out)
    return out


def music(total, path, sr=48000):
    """Soft four-chord pad (Am - F - C - G), original, written sample by sample."""
    t = np.arange(int(total * sr)) / sr
    chords = [[57, 60, 64], [53, 57, 60], [48, 55, 64], [55, 59, 62]]   # MIDI notes
    bar = 4.0
    out = np.zeros((len(t), 2))
    hz = lambda m: 440.0 * 2 ** ((m - 69) / 12)
    for c_i in range(int(total / bar) + 1):
        ch = chords[c_i % 4]
        s0, s1 = int(c_i * bar * sr), min(len(t), int((c_i + 1) * bar * sr + 1.2 * sr))
        if s0 >= len(t):
            break
        tt = t[s0:s1] - c_i * bar
        env = np.minimum(1, tt / 1.2) * np.exp(-np.maximum(0, tt - bar) / 0.5)
        sig = np.zeros(len(tt))
        for m in ch:
            for det in (-0.08, 0.08):
                sig += np.sin(2 * np.pi * hz(m + det) * tt) * 0.16
            sig += np.sin(2 * np.pi * hz(m + 12) * tt) * 0.04
        sig += np.sin(2 * np.pi * hz(ch[0] - 12) * tt) * 0.22                 # soft bass
        # gentle 8th-note shimmer on the top note
        pulse = 0.5 + 0.5 * np.cos(2 * np.pi * tt / 0.5)
        sig += np.sin(2 * np.pi * hz(ch[-1] + 12) * tt) * 0.035 * pulse
        pan = 0.5 + 0.15 * np.sin(2 * np.pi * tt / 7)
        out[s0:s1, 0] += sig * env * (1 - pan) * 2
        out[s0:s1, 1] += sig * env * pan * 2
    # one-pole low-pass for warmth, fade in/out
    a = math.exp(-2 * math.pi * 1800 / sr)
    for ch_ in range(2):
        y = out[:, ch_]
        for i in range(1, len(y)):
            y[i] = (1 - a) * y[i] + a * y[i - 1]
    fade = np.minimum(1, np.minimum(t / 2.0, (total - t) / 3.0))[:, None]
    out *= fade
    out /= np.max(np.abs(out)) + 1e-9
    out *= 0.6
    pcm = (out * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main():
    items = json.load(open(os.path.join(HERE, "narration.json")))
    txtdir = os.path.join(W, "txt")
    os.makedirs(txtdir, exist_ok=True)
    secs, voice_parts, total = [], [], 0.0
    for i, it in enumerate(items):
        v = os.path.join(W, "av", it["id"] + ".wav")
        L = dur_wav(v) + PAD + (0.8 if it["id"] == "close" else 0.0) + (0.3 if i == 0 else 0.0)
        secs.append(section(i, it, L, txtdir))
        vp = os.path.join(W, f"vp{i:02d}.wav")
        lead = 300 if i == 0 else 150
        run("ffmpeg", "-y", "-loglevel", "error", "-i", v, "-af",
            f"adelay={lead}|{lead},apad,atrim=0:{L:.3f}", "-ar", "48000", "-ac", "2", vp)
        voice_parts.append(vp)
        total += L
        print(f"{it['id']:10s} {L:6.2f}s")
    lst = os.path.join(W, "vlist.txt")
    open(lst, "w").write("".join(f"file '{s}'\n" for s in secs))
    run("ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
        os.path.join(W, "video_noaudio.mp4"))
    alst = os.path.join(W, "alist.txt")
    open(alst, "w").write("".join(f"file '{p}'\n" for p in voice_parts))
    run("ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", alst, "-c", "copy",
        os.path.join(W, "voice.wav"))
    music(total, os.path.join(W, "music.wav"))
    fc = ("[1:a]volume=1.55,asplit=2[v1][v2];[2:a]volume=0.32[m];"
          "[m][v1]sidechaincompress=threshold=0.02:ratio=9:attack=15:release=450:makeup=1[md];"
          "[md][v2]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.89[a]")
    run("ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(W, "video_noaudio.mp4"),
        "-i", os.path.join(W, "voice.wav"), "-i", os.path.join(W, "music.wav"),
        "-filter_complex", fc, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-shortest", OUT)
    print("total", round(total, 1), "s ->", OUT)


if __name__ == "__main__":
    main()
