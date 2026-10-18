#!/usr/bin/env python3
"""Narration and music for the JitJet films (cover/ and example/*/animate_*.py).

Narration: one clip per line of a script, either synthesised (edge-tts, Microsoft's
online neural voices; or the macOS `say` command) or recorded (--narration-dir).
Synthesised clips are cached by a hash of engine, voice, rate and text, so a
re-run with the same text costs nothing.

Score: a numpy synthesiser driven by a ScoreSpec of cue times: a drone, a pad
whose chord changes at given times, a plucked pulse in sections of growing
weight, bells at given times, a low thud at given times, and a plain hall.
mix_audio() lays the clips on the clock, ducks the score under the voice and
writes the stereo WAV; write_srt() writes the subtitles.

The scripts add the options with add_narration_args() and resolve them with
narration_defaults(); see cover/animate_short_cover_timeline.py for the use.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SR = 44100
DEFAULT_VOICE = {"edge": "en-GB-RyanNeural", "say": "Daniel"}
DEFAULT_RATE = {"edge": -6, "say": 156}

# --- the script text -------------------------------------------------------------
def display_text(text: str) -> str:
    """The pauses "..." of a line as an ellipsis for subtitles and printing."""
    return re.sub(r"\s*\.\.\.\s*", "… ", text).strip()


def spoken_for_say(text: str) -> str:
    return re.sub(r"\s*\.\.\.\s*", " [[slnc 450]] ", text)


def print_script(lines: Dict[str, str]) -> None:
    for key, text in lines.items():
        print(f"[{key}]\n{display_text(text)}\n")


# --- the clips -------------------------------------------------------------------
def read_audio(path: Path, ffmpeg: str) -> np.ndarray:
    """Any audio file -> float32 mono at SR, via ffmpeg."""
    out = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


def trim_silence(x: np.ndarray, thr: float = 0.004, keep: float = 0.15) -> np.ndarray:
    """Cut the leading and trailing silence, keeping `keep` seconds of it."""
    idx = np.flatnonzero(np.abs(x) > thr)
    if idx.size == 0:
        return x
    k = int(keep * SR)
    return x[max(idx[0] - k, 0): min(idx[-1] + k, x.size)]


def synth_say(text: str, voice: str, rate: int, path: Path) -> None:
    if shutil.which("say") is None:
        sys.exit("`say` (macOS) not found: use --engine edge, --narration-dir, or --no-narration")
    subprocess.run(["say", "-v", voice, "-r", str(rate), f"--data-format=LEI16@{SR}", "-o", str(path),
                    spoken_for_say(text)], check=True)


def synth_edge(text: str, voice: str, rate: int, path: Path) -> None:
    """One line through edge-tts (Microsoft's neural voices, needs the network)."""
    try:
        import edge_tts
    except ImportError:
        sys.exit("edge_tts not found: `pip install --user edge-tts` (with the python running this script), "
                 "or --engine say, --narration-dir, or --no-narration")
    text = re.sub(r"\s*\.\.\.\s*", "... ", text)
    try:
        asyncio.run(edge_tts.Communicate(text, voice, rate=f"{rate:+d}%").save(str(path)))
    except Exception as e:  # noqa: BLE001
        sys.exit(f"edge-tts failed for voice {voice!r}: {e}\n(needs the network; `edge-tts --list-voices` lists the voices)")


def add_narration_args(p) -> None:
    """The narration options shared by the films."""
    p.add_argument("--engine", choices=("edge", "say"), default="edge",
                   help="edge: Microsoft neural voices via edge-tts (network); say: the macOS synthesiser")
    p.add_argument("--voice", default=None,
                   help="edge-tts voice (default en-GB-RyanNeural; `edge-tts --list-voices`) or `say` voice (default Daniel)")
    p.add_argument("--rate", type=int, default=None,
                   help="speaking rate: percent change for edge (default -6), words per minute for say (default 156)")
    p.add_argument("--narration-dir", type=Path, default=None,
                   help="use recorded clips <key>.wav|mp3 (keys as in --print-script) instead of synthesis")
    p.add_argument("--no-narration", action="store_true", help="music only, the nominal timing")
    p.add_argument("--print-script", action="store_true", help="print the narration and exit")
    p.add_argument("--music-db", type=float, default=-6.0, help="music level (it ducks under the voice); --no-music for silence")
    p.add_argument("--no-music", action="store_true")


def narration_defaults(args) -> None:
    if args.voice is None:
        args.voice = DEFAULT_VOICE[args.engine]
    if args.rate is None:
        args.rate = DEFAULT_RATE[args.engine]


def load_narration(lines: Dict[str, str], args, cache_dir: Path, ffmpeg: str) -> Dict[str, np.ndarray]:
    """One clip per line, from --narration-dir or synthesised (cached in cache_dir); {} with --no-narration."""
    clips: Dict[str, np.ndarray] = {}
    if args.no_narration:
        return clips
    cache_dir.mkdir(parents=True, exist_ok=True)
    for key, text in lines.items():
        if args.narration_dir:
            hits = sorted(p for p in Path(args.narration_dir).iterdir() if p.stem == key)
            if not hits:
                sys.exit(f"--narration-dir: no clip named {key}.* in {args.narration_dir}")
            path = hits[0]
        else:
            tag = hashlib.sha1(f"{args.engine}|{args.voice}|{args.rate}|{text}".encode()).hexdigest()[:10]
            path = cache_dir / f"{key}_{tag}.{'mp3' if args.engine == 'edge' else 'wav'}"
            if not path.exists():
                print(f"  narration: {key} ({args.engine}, {args.voice})", flush=True)
                (synth_edge if args.engine == "edge" else synth_say)(text, args.voice, args.rate, path)
        clip = trim_silence(read_audio(path, ffmpeg))
        peak = float(np.max(np.abs(clip))) if clip.size else 1.0
        clips[key] = clip * (0.85 / peak if peak > 0 else 1.0)
    return clips


def clip_lengths(clips: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {k: c.size / SR for k, c in clips.items()}


# --- the instruments -------------------------------------------------------------
def note(f: float, n: int, sr: int = SR, detune: float = 0.0) -> np.ndarray:
    """A soft pad note: a few harmonics with a slow vibrato, length n samples."""
    t = np.arange(n, dtype=np.float32) / sr
    vib = 1.0 + 0.0025 * np.sin(2 * np.pi * 0.17 * t + f)
    out = np.zeros(n, dtype=np.float32)
    for h, amp in ((1, 1.0), (2, 0.35), (3, 0.16), (4, 0.06)):
        out += amp * np.sin(2 * np.pi * (f + detune) * h * vib * t).astype(np.float32)
    return out


def env_adsr(n: int, attack: float, decay: float, sr: int = SR) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    e = np.exp(-t / decay).astype(np.float32)
    k = max(int(attack * sr), 1)
    e[:k] *= np.linspace(0, 1, k, dtype=np.float32)
    return e


def pluck(f: float, sr: int = SR) -> np.ndarray:
    """A soft plucked note, 1.4 s."""
    n = int(1.4 * sr)
    t = np.arange(n, dtype=np.float32) / sr
    out = np.sin(2 * np.pi * f * t) * env_adsr(n, 0.004, 0.32)
    out += 0.35 * np.sin(2 * np.pi * 2 * f * t) * env_adsr(n, 0.002, 0.14)
    out += 0.12 * np.sin(2 * np.pi * 3 * f * t) * env_adsr(n, 0.002, 0.08)
    return out.astype(np.float32)


def bell(f: float, seconds: float = 3.0, sr: int = SR) -> np.ndarray:
    """A bell: inharmonic partials with their own decays."""
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    out = np.zeros(n, dtype=np.float32)
    for ratio, amp, dec in ((1.0, 1.0, 1.4), (2.0, 0.45, 0.8), (2.76, 0.3, 0.5), (4.07, 0.12, 0.3), (0.5, 0.25, 1.8)):
        out += amp * np.sin(2 * np.pi * f * ratio * t) * env_adsr(n, 0.003, dec)
    return out


def reverb(x: np.ndarray, seconds: float, wet: float, sr: int = SR) -> np.ndarray:
    """A plain hall: convolution with decaying noise (FFT), mixed at `wet`."""
    m = int(seconds * sr)
    rng = np.random.default_rng(7)
    ir = rng.standard_normal(m).astype(np.float32) * np.exp(-np.arange(m, dtype=np.float32) / (seconds * sr / 4.5))
    ir /= np.sqrt(np.sum(ir ** 2)) * 4
    n = x.size + m - 1
    nfft = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[: x.size].astype(np.float32)
    return x + wet * y


def add(buf: np.ndarray, t0: float, x: np.ndarray, gain: float = 1.0, sr: int = SR) -> None:
    i0 = int(t0 * sr)
    if i0 < 0 or i0 >= buf.shape[0]:
        return
    m = min(x.shape[0], buf.shape[0] - i0)
    buf[i0:i0 + m] += gain * x[:m]


# --- the score -------------------------------------------------------------------
# the key of D minor
D2, A2, Bb2, C3, D3, E3, F3, G3, A3, Bb3, C4, Cs4, D4, E4, F4 = (
    73.42, 110.0, 116.54, 130.81, 146.83, 164.81, 174.61, 196.0, 220.0, 233.08, 261.63, 277.18, 293.66, 329.63, 349.23)
Dm, Bb, F, C, Gm, A, Dm_hi = (D3, F3, A3), (Bb2, D3, F3), (F3, A3, C4), (C3, E3, G3), (G3, Bb3, D4), (A2, Cs4, E4), (D3, A3, D4, F4)
BEAT = 60.0 / 76.0
PATTERNS = ([0, 1, 2, 3, 2, 1], [0, 2, 1, 3, 1, 2])


@dataclass
class Pulse:
    """Eighth notes on `tones` from t0 to t1, louder before t_accent (a growth, a reveal)."""
    t0: float
    t_accent: float
    t1: float
    gain: float
    tones: Sequence[float]
    pattern: Sequence[int]


@dataclass
class Bell:
    t: float
    f: float
    gain: float = 0.42
    seconds: float = 3.0
    spread: float = 0.012          # the right channel rings a hair later


@dataclass
class ScoreSpec:
    duration: float
    fade_out: float
    chords: List[Tuple[float, Sequence[float]]]      # (when it starts, the chord); the first starts at 0
    pulses: List[Pulse] = field(default_factory=list)
    bells: List[Bell] = field(default_factory=list)
    thuds: List[float] = field(default_factory=list)  # a low pulse (a collision)
    outro: Optional[Tuple[float, Sequence[float]]] = None   # (from when, tones): slow quarter notes dying over 6 s


def pulse_for(i: int, t0: float, t_accent: float, t1: float, chord: Sequence[float]) -> Pulse:
    """The pulse of the i-th stop of a film: one more voice with every stop."""
    return Pulse(t0, t_accent, t1, 0.10 + 0.045 * i, [f * 2 for f in chord] + [chord[0] * 4], PATTERNS[i % 2])


def make_score(spec: ScoreSpec, sr: int = SR) -> np.ndarray:
    """Stereo, peak 0.5."""
    n = int(spec.duration * sr) + sr
    L, R = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    t = np.arange(n, dtype=np.float32) / sr

    # the pad: one chord after the other, crossfading
    sched = spec.chords
    xfade = 2.5
    for j, (t0, chord) in enumerate(sched):
        t1 = sched[j + 1][0] + xfade if j + 1 < len(sched) else spec.duration + 1.0
        n0, n1 = int(t0 * sr), min(int(t1 * sr), n)
        m = n1 - n0
        env = np.ones(m, dtype=np.float32)
        k_in, k_out = min(int((4.0 if j == 0 else xfade) * sr), m), min(int(xfade * sr), m)
        env[:k_in] *= np.linspace(0, 1, k_in, dtype=np.float32) ** 1.5
        if j + 1 < len(sched):
            env[-k_out:] *= np.linspace(1, 0, k_out, dtype=np.float32) ** 1.5
        for f in chord:
            L[n0:n1] += 0.33 * env * note(f, m, sr, detune=+0.25 * f / 220)
            R[n0:n1] += 0.33 * env * note(f, m, sr, detune=-0.25 * f / 220)

    # the drone, breathing slowly
    drone = (0.8 + 0.2 * np.sin(2 * np.pi * 0.045 * t)) * (np.sin(2 * np.pi * D2 * t) + 0.3 * np.sin(2 * np.pi * 2 * D2 * t))
    L += 0.45 * drone.astype(np.float32)
    R += 0.45 * drone.astype(np.float32)

    # the thuds
    thud = np.sin(2 * np.pi * 55.0 * t[: int(1.5 * sr)]) * env_adsr(int(1.5 * sr), 0.01, 0.45)
    for t0 in spec.thuds:
        add(L, t0, thud, 0.5)
        add(R, t0, thud, 0.5)

    # the pulse
    plucks: Dict[float, np.ndarray] = {}
    for pu in spec.pulses:
        k, tk = 0, pu.t0
        while tk < pu.t1 - 0.3:
            f = pu.tones[pu.pattern[k % len(pu.pattern)]]
            if f not in plucks:
                plucks[f] = pluck(f, sr)
            g = pu.gain * (1.25 if tk < pu.t_accent else 1.0) * (0.75 if k % 2 else 1.0)
            pan = 0.5 + 0.3 * math.sin(k * 1.3)
            add(L, tk, plucks[f], g * (1 - pan) * 1.4)
            add(R, tk, plucks[f], g * pan * 1.4)
            tk += BEAT / 2
            k += 1
    if spec.outro is not None:
        t_out, tones = spec.outro
        tk, k = t_out, 0
        while tk < t_out + 6.0:
            f = tones[k % len(tones)]
            if f not in plucks:
                plucks[f] = pluck(f, sr)
            g = 0.3 * max(0.0, 1.0 - (tk - t_out) / 6.0)
            add(L, tk, plucks[f], g)
            add(R, tk, plucks[f], g)
            tk += BEAT
            k += 1

    # the bells
    for b in spec.bells:
        x = bell(b.f, b.seconds, sr)
        add(L, b.t, x, b.gain)
        add(R, b.t + b.spread, x, b.gain)

    # the hall, and the fades
    L, R = reverb(L, 2.2, 0.35, sr), reverb(R, 2.2, 0.35, sr)
    glob = np.ones(n, dtype=np.float32)
    k = int(3 * sr)
    glob[:k] = np.linspace(0, 1, k) ** 2
    kf = int((spec.fade_out + 2.5) * sr)
    end = int(spec.duration * sr)
    glob[end - kf:end] *= np.linspace(1, 0, kf) ** 1.5
    glob[end:] = 0
    st = np.stack([L, R], axis=1) * glob[:, None]
    st *= 0.5 / max(float(np.max(np.abs(st))), 1e-6)
    return st[:int(spec.duration * sr)]


# --- the mix and the subtitles ------------------------------------------------------
def mix_audio(duration: float, voice_at: Dict[str, float], clips: Dict[str, np.ndarray],
              score: Optional[ScoreSpec], music_db: float, out: Path) -> None:
    """The narration on the clock (voice_at: key -> start time) with the score ducked under it, as a WAV."""
    n = int(duration * SR)
    voice = np.zeros(n, dtype=np.float32)
    for key, t0 in voice_at.items():
        clip = clips.get(key)
        if clip is not None:
            add(voice, t0, clip)
    if voice.any():
        voice = reverb(voice, 0.6, 0.10)     # a touch of room, the voice stays close
    stereo = np.stack([voice, voice], axis=1)
    if score is not None:
        music = make_score(score)
        # duck the music under the voice: an envelope at 100 Hz, smoothed over 0.3 s
        dec = 441
        env = np.abs(voice[: (n // dec) * dec]).reshape(-1, dec).max(axis=1)
        k = 30
        env = np.convolve(env, np.ones(k) / k, mode="same")
        env = np.maximum(env, np.convolve(env, np.ones(2 * k) / (2 * k), mode="same"))
        gain = 1.0 - 0.55 * np.clip(env / 0.08, 0, 1)
        gain_full = np.interp(np.arange(n) / dec, np.arange(env.size), gain).astype(np.float32)
        stereo += music[:n] * (10 ** (music_db / 20)) * gain_full[:, None]
    stereo = np.clip(stereo, -1, 1)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((stereo * 32767).astype("<i2").tobytes())


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(voice_at: Dict[str, float], clips: Dict[str, np.ndarray], lines: Dict[str, str], path: Path) -> None:
    """One subtitle per sentence of each line, with the time shared out by length."""
    k = 1
    with open(path, "w") as f:
        for key, t0 in voice_at.items():
            clip = clips.get(key)
            if clip is None:
                continue
            total_len = clip.size / SR
            parts = [p.strip() for p in re.split(r"(?<=[.?!:…])\s+", display_text(lines[key])) if p.strip()]
            merged: List[str] = []
            for p in parts:
                if merged and len(p) < 25:
                    merged[-1] += " " + p
                else:
                    merged.append(p)
            total = sum(len(p) for p in merged)
            t = t0
            for p in merged:
                d = total_len * len(p) / total
                f.write(f"{k}\n{srt_time(t)} --> {srt_time(t + d - 0.05)}\n{p}\n\n")
                t += d
                k += 1


def mux(ffmpeg: str, video: Path, audio: Path, out: Path) -> None:
    """Join a silent video and the WAV soundtrack into out (video copied, AAC audio)."""
    subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(video), "-i", str(audio), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)],
                   check=True)
