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

Film clock: the chapter films (example/*/animate_*.py) are scenes made of frames
at NOMINAL_FPS; each scene lists its phases (key, first frame, frame after the
last), one line of narration per phase. Film puts the phases on the clock so that
each lasts at least its line (the motion slowed down up to 2.5x, then held), and
film_score() writes the score from that clock: a chord per scene, a bell as each
scene opens, the pulse gaining a voice with every scene, the outro resolving.
render_video() draws the frames of the clock (in several processes with jobs > 1,
the segments joined losslessly) and muxes the soundtrack under them.

The scripts add the options with add_narration_args() and resolve them with
narration_defaults(); see cover/animate_short_cover_timeline.py for the use of the
narration alone and example/ME/Standalone/animate_standalone_qqbar_bbbar_xsec.py
for a film.
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


# Written -> spoken: words the voices read as initials or mangle. Subtitles and --print-script keep the written form.
SPOKEN = {"JitJet": "Jit Jet"}       # "JitJet" comes out as "J-I-T Jet"


def spoken_text(text: str) -> str:
    for written, spoken in SPOKEN.items():
        text = text.replace(written, spoken)
    return text


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
            spoken = spoken_text(text)
            tag = hashlib.sha1(f"{args.engine}|{args.voice}|{args.rate}|{spoken}".encode()).hexdigest()[:10]
            path = cache_dir / f"{key}_{tag}.{'mp3' if args.engine == 'edge' else 'wav'}"
            if not path.exists():
                print(f"  narration: {key} ({args.engine}, {args.voice})", flush=True)
                (synth_edge if args.engine == "edge" else synth_say)(spoken, args.voice, args.rate, path)
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


# --- the film clock ----------------------------------------------------------------
NOMINAL_FPS = 24                  # the scenes are drawn for this: one frame index per 1/24 s


@dataclass
class Phase:
    scene: object
    key: str                      # <scene name>_<phase key>: the narration key
    k0: int
    k1: int                       # frame after the last of the phase
    t0: float
    t1: float
    motion: float                 # the frames k0 .. k1-1 play over this long, then the last one holds


class Film:
    """The phases of the scenes on the clock: each lasts at least its narration (motion slowed up to
    slow_max, then held). Scenes have .number and .phases() -> [(key, k0, k1)]; names maps number -> name."""

    def __init__(self, scenes: Sequence, lengths: Dict[str, float], names: Dict[int, str], lead: float = 0.4,
                 tail: float = 0.7, slow_max: float = 2.5, outro_lead: float = 0.6, outro_tail: float = 2.0,
                 fade_out: float = 1.5):
        self.phases: List[Phase] = []
        self.voice_at: Dict[str, float] = {}
        self.scene_start: Dict[int, float] = {}
        t = 0.0
        for sc in scenes:
            self.scene_start[sc.number] = t
            for key, k0, k1 in sc.phases():
                full = f"{names.get(sc.number, 'scene')}_{key}"
                nominal = (k1 - k0) / NOMINAL_FPS
                dur = max(nominal, lead + lengths.get(full, 0.0) + tail)
                self.phases.append(Phase(sc, full, k0, k1, t, t + dur, min(dur, slow_max * nominal)))
                if full in lengths:
                    self.voice_at[full] = t + lead
                t += dur
        self.t_outro = t
        if "outro" in lengths:
            self.voice_at["outro"] = t + outro_lead
            t += outro_lead + lengths["outro"] + outro_tail
        self.duration = t
        self.fade_out = fade_out

    def frame_at(self, t: float) -> Tuple[object, float]:
        """(scene, frame index, a float) at film time t."""
        for ph in self.phases:
            if t < ph.t1 or ph is self.phases[-1]:
                u = min(max((t - ph.t0) / ph.motion, 0.0), 1.0) if ph.motion > 0 else 1.0
                return ph.scene, ph.k0 + (ph.k1 - 1 - ph.k0) * u
        raise RuntimeError("no phases")

    def frames(self, fps: float) -> List[Tuple[object, float]]:
        n = int(math.ceil(self.duration * fps))
        return [self.frame_at(i / fps) for i in range(n)]

    def scene_phases(self, number: int) -> List[Phase]:
        return [ph for ph in self.phases if ph.scene.number == number]

    def describe(self) -> str:
        return (f"{self.duration:.1f} s, {len(self.phases)} phases" +
                ("" if self.voice_at else " (silent)"))


def film_score(film: Film, progression: Sequence[Sequence[float]], thuds: Sequence[float] = (),
               max_voices: int = 6) -> ScoreSpec:
    """The score of a film from its clock: one chord per scene cycling through `progression` (resolving to
    Dm_hi at the outro), a bell as each scene after the first opens, the pulse gaining a voice with every
    scene (up to max_voices), a thud at each time of `thuds`, the final bell under the outro."""
    starts = sorted(film.scene_start.items())
    spec = ScoreSpec(film.duration, film.fade_out,
                     chords=[(t, progression[i % len(progression)]) for i, (_, t) in enumerate(starts)]
                            + [(film.t_outro, Dm_hi)],
                     thuds=list(thuds), outro=(film.t_outro, [f * 2 for f in Dm_hi]))
    if spec.chords[0][0] > 0:
        spec.chords.insert(0, (0.0, Dm))
    for i, (number, t) in enumerate(starts):
        chord = progression[i % len(progression)]
        phases = film.scene_phases(number)
        spec.pulses.append(pulse_for(min(i, max_voices), t, phases[0].t1, phases[-1].t1, chord))
        if i > 0:
            spec.bells.append(Bell(t, chord[0] * 4))
    spec.bells.append(Bell(film.t_outro + 0.3, D4, gain=0.55, seconds=6.0, spread=0.015))
    return spec


def voice_at_phase(film: Film, key: str) -> Optional[float]:
    """When the phase `key` starts (for a cue of the score), None if the film has no such phase."""
    return next((ph.t0 for ph in film.phases if ph.key == key), None)


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


# --- rendering a film ----------------------------------------------------------------
VIDEO_ARGS = ["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", "-movflags", "+faststart"]


def _writer(out: Path, fps: float, title: str):
    from matplotlib import animation
    if out.suffix.lower() == ".gif":
        return animation.PillowWriter(fps=fps)
    if not animation.FFMpegWriter.isAvailable():
        sys.exit("ffmpeg not found on PATH; write a .gif instead or install ffmpeg")
    return animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=VIDEO_ARGS, metadata={"title": title})


def render_video(film: Film, out: Path, fps: float, dpi: int, audio: Optional[Path], draw, figsize, title: str,
                 jobs: int = 1, every: int = 50) -> None:
    """Render film.frames(fps) with draw(fig, scene, k, t_film) to `out` (H.264, crf 18, or a silent .gif) and
    mux `audio` (a WAV) under it. With jobs > 1 the frames are split into that many segments rendered by forked
    worker processes and joined losslessly with ffmpeg's concat demuxer."""
    frames = film.frames(fps)
    ext = out.suffix.lower()
    if ext not in (".gif", ".mp4", ".m4v", ".mov", ".webm"):
        sys.exit(f"unknown output type {ext}; use .mp4 or .gif")
    if ext == ".gif":
        audio, jobs = None, 1
    ffmpeg = shutil.which("ffmpeg")
    silent = out.with_name(out.stem + "_silent" + out.suffix) if audio else out
    n = len(frames)
    jobs = max(1, min(jobs, n // 48 or 1))
    print(f"writing {n} frames at {fps:g} fps ({n / fps:.1f} s) to {out}" + (f" with {jobs} processes" if jobs > 1 else ""),
          flush=True)
    if jobs == 1:
        _run_segment(draw, frames, silent, fps, dpi, figsize, title, "", 0, n, every)
    else:
        import multiprocessing as mp
        import tempfile as tf
        ctx = mp.get_context("fork")
        with tf.TemporaryDirectory(prefix="jitjet_segments_") as tmp:
            bounds = [round(j * n / jobs) for j in range(jobs + 1)]
            segs = [Path(tmp) / f"seg_{j:02d}{ext}" for j in range(jobs)]
            procs = [ctx.Process(target=_run_segment, args=(draw, frames[bounds[j]:bounds[j + 1]], segs[j], fps, dpi, figsize,
                                                             title, f"[{j + 1}/{jobs}] ", bounds[j], n, every))
                     for j in range(jobs)]
            for pr in procs:
                pr.start()
            for pr in procs:
                pr.join()
            if any(pr.exitcode for pr in procs):
                sys.exit("a rendering process failed")
            lst = Path(tmp) / "segments.txt"
            lst.write_text("".join(f"file '{sg}'\n" for sg in segs))
            subprocess.run([ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                            "-movflags", "+faststart", str(silent)], check=True)
    if audio:
        mux(ffmpeg, silent, audio, out)
        silent.unlink()


def _run_segment(draw, frames, out: Path, fps: float, dpi: int, figsize, title: str, label: str,
                 first: int, total: int, every: int) -> None:
    """The frames of one segment, in this process."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize)
    writer = _writer(out, fps, title)
    with writer.saving(fig, str(out), dpi):
        for i, (sc, k) in enumerate(frames):
            draw(fig, sc, k, (first + i) / fps)
            writer.grab_frame(facecolor="white")
            if i % every == 0 or i == len(frames) - 1:
                print(f"  {label}frame {first + i + 1}/{total}  (scene {getattr(sc, 'number', '?')}, k={k:.1f})", flush=True)
    plt.close(fig)


def mux(ffmpeg: str, video: Path, audio: Path, out: Path) -> None:
    """Join a silent video and the WAV soundtrack into out (video copied, AAC audio)."""
    subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(video), "-i", str(audio), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)],
                   check=True)
