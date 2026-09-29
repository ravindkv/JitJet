#!/usr/bin/env python3
"""The one-minute trailer: the cone of cover_timeline.tex grows part by part.

No narration. The film starts at the hard scattering, a glow on an empty page,
and the cone grows along its axis, one part of the book at a time: the b quark
and her shower, the hadrons, the pileup crowd and the detector wall, the two
trigger gates, the Particle-Flow census, PUPPI and the R = 0.4 cone, the JEC
scale, the b tag, and finally the R = 0.8 top jet that caps the cone. At each
part the growth pauses, the cover's own label for that part appears, and a
caption names the part. The camera pulls back as the cone grows, and the film
ends on the whole cone with the title of the book. A synthesised pad plays
underneath.

How the frames are made
  cover_timeline.tex takes two macros, \\smax (draw the cone up to this point on
  the axis) and \\partmax (show the labels of the first N parts). The script
  compiles the cover once per growth step (pdflatex, in parallel), renders each
  PDF with ghostscript, and dissolves between consecutive steps while the
  camera moves, in the manner of the animate_*.py scripts of example/. The
  compiled frames can be kept (--work-dir) so that a re-run only re-renders.

Examples (needs numpy, Pillow, pdflatex, gs and ffmpeg; on this Mac use /usr/bin/python3):
  python3 animate_short_cover_timeline.py                 # -> animate_short_cover_timeline.mp4
  python3 animate_short_cover_timeline.py --fast          # 960x540, fewer steps, quick
  python3 animate_short_cover_timeline.py --stills stills.pdf --no-video
  python3 animate_short_cover_timeline.py --steps 24 --grow 3.0     # finer growth, slower film

The compiled frames go to build/short_cover_frames/ (about 4 min the first
time, six pdflatex jobs in parallel) and are re-used by later runs; after a
change to cover_timeline.tex delete that directory (or `make clean`).
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import animate_cover_timeline as long_film  # noqa: E402  (same directory: cover, fonts, ruler, music)
from animate_cover_timeline import (  # noqa: E402
    GOLD, NAVY, PALE, SR, WHITE, Cover, Fonts, Ruler, _SUP, alpha_paste, clamp01, draw_rich, make_music, smooth,
)

COVER_TEX = HERE / "cover_timeline.tex"


# --- the parts: where the cone stops, what the caption says ------------------
@dataclass
class Part:
    number: int
    smax: float                       # the cone is complete up to here when the part is reached
    kicker: str
    title: str
    sub: str
    clock: str
    shot: Tuple[float, float, float]  # camera centre (cm) and width (cm) while the part is shown


PARTS = [
    Part(1, 6.6, "PART I", "The family is born", "matrix element  ·  parton shower  ·  hadrons",
         "10^{-27} s  to  10^{-23} s", (5.6, 8.9, 10.5)),
    Part(2, 11.3, "PART II", "On the road", "pileup  ·  detector  ·  L1T  ·  HLT",
         "25 ns  ·  4 μs  ·  0.3 s", (7.4, 10.9, 14.5)),
    Part(3, 12.9, "PART III", "Taking the census", "RecHits  ·  Particle Flow",
         "hours", (8.4, 11.9, 17.5)),
    Part(4, 14.55, "PART IV", "Who belongs to the family?", "PUPPI  ·  anti-kT clustering, R = 0.4",
         "hours", (9.2, 12.5, 20.5)),
    Part(5, 16.05, "PART V", "Settling the accounts", "jet energy corrections  ·  resolution",
         "hours to months", (9.9, 13.2, 24.0)),
    Part(6, 17.6, "PART VI", "Arrival", "b tagging",
         "months", (10.5, 13.9, 28.0)),
    Part(7, 19.0, "PART VII", "The top family", "three prongs in one R = 0.8 jet  ·  top tagging",
         "from bottom to top", (10.8, 14.8, 31.0)),
]
S_START = -1.3                      # before the apex: only the glow of the collision is visible
TITLE_SHOT = (10.5, 26.9, 13.5)
WHOLE_SHOT = (10.8, 14.8, 31.0)


# --- the compiled frames ----------------------------------------------------
@dataclass
class State:
    smax: float
    partmax: int

    def name(self) -> str:
        return f"s{self.smax:06.2f}_p{self.partmax}"


def build_states(steps: int) -> Tuple[List[State], List[int]]:
    """All truncations to compile, and the index of the state that ends each part."""
    states = [State(S_START, 0)]
    ends = []
    s0 = S_START
    for part in PARTS:
        for j in range(1, steps + 1):
            u = smooth(j / steps)          # the growth eases in and out within a part
            s = s0 + (part.smax - s0) * u
            states.append(State(round(s, 2), part.number if j == steps else part.number - 1))
        ends.append(len(states) - 1)
        s0 = part.smax
    return states, ends


def compile_state(state: State, workdir: Path, dpi: float) -> Path:
    """pdflatex + gs for one truncation; skipped when the PNG already exists."""
    png = workdir / f"{state.name()}_{dpi:g}.png"
    if png.exists():
        return png
    job = state.name()
    tex = f"\\def\\smax{{{state.smax}}}\\def\\partmax{{{state.partmax}}}\\input{{{COVER_TEX.name}}}"
    r = subprocess.run(["pdflatex", "-interaction=batchmode", "-halt-on-error", f"-output-directory={workdir}",
                        f"-jobname={job}", tex], cwd=HERE, capture_output=True)
    pdf = workdir / f"{job}.pdf"
    if r.returncode != 0 or not pdf.exists():
        sys.exit(f"pdflatex failed for {state}: see {workdir / (job + '.log')}")
    subprocess.run(["gs", "-q", "-sDEVICE=png16m", f"-r{dpi:g}", "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
                    "-o", str(png), str(pdf)], check=True)
    for ext in (".pdf", ".aux", ".log"):
        (workdir / (job + ext)).unlink(missing_ok=True)
    return png


def compile_all(states: List[State], workdir: Path, dpi: float, jobs: int) -> List[Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in states if not (workdir / f"{s.name()}_{dpi:g}.png").exists()]
    print(f"compiling {len(todo)} of {len(states)} frames of the cover ({jobs} at a time) ...")
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        pngs = list(ex.map(lambda s: compile_state(s, workdir, dpi), states))
    return pngs


class Frames:
    """The compiled frames as Cover objects, loaded on demand (a few at a time)."""

    def __init__(self, pngs: List[Path], dpi: float, keep: int = 4):
        self.pngs, self.dpi, self.keep = pngs, dpi, keep
        self.cache: Dict[int, Cover] = {}
        self.order: List[int] = []

    def get(self, i: int) -> Cover:
        i = min(max(i, 0), len(self.pngs) - 1)
        if i not in self.cache:
            self.cache[i] = Cover(self.pngs[i], self.dpi)
            self.order.append(i)
            while len(self.order) > self.keep:
                del self.cache[self.order.pop(0)]
        return self.cache[i]


# --- the timeline -----------------------------------------------------------
@dataclass
class Cue:
    part: Part
    t_grow: float      # the cone starts growing towards this part
    t_hold: float      # the growth is complete; the label and the caption appear
    t_end: float       # the next growth starts (or the finale)
    i0: int            # frame index at t_grow
    i1: int            # frame index at t_hold


class Timeline:
    def __init__(self, ends: List[int], t_title: float, grow: float, hold: float, finale: float, fade_out: float):
        self.t_title = t_title                # the title, then the move down to the apex
        self.t_first = t_title + 3.0          # the camera reaches the apex; a beat before growth starts
        self.cues: List[Cue] = []
        t = self.t_first + 1.0
        i0 = 0
        for part, i1 in zip(PARTS, ends):
            cue = Cue(part, t, t + grow, t + grow + hold, i0, i1)
            self.cues.append(cue)
            t, i0 = cue.t_end, i1
        self.t_finale = t                     # pull back a touch and show the title card
        self.duration = t + finale
        self.fade_in = 1.2
        self.fade_out = fade_out
        self.last_index = ends[-1]

    def frame_index(self, t: float) -> Tuple[int, float]:
        """Which compiled frame to show at time t: index i and the blend towards i + 1."""
        if t < self.cues[0].t_grow:
            return 0, 0.0
        for cue in self.cues:
            if cue.t_grow <= t < cue.t_hold:
                x = (t - cue.t_grow) / (cue.t_hold - cue.t_grow) * (cue.i1 - cue.i0)
                i = int(math.floor(x))
                return cue.i0 + i, x - i
            if cue.t_hold <= t < cue.t_end:
                return cue.i1, 0.0
        return self.last_index, 0.0

    def camera(self, t: float) -> Tuple[float, float, float]:
        def lerp(a, b, u):
            return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] * (b[2] / a[2]) ** u)
        first = PARTS[0].shot
        if t < self.t_title:
            return lerp(TITLE_SHOT, (TITLE_SHOT[0], TITLE_SHOT[1], TITLE_SHOT[2] * 0.96), t / self.t_title)
        if t < self.t_first:
            return lerp(TITLE_SHOT, first, smooth((t - self.t_title) / (self.t_first - self.t_title)))
        prev = first
        for cue in self.cues:
            if t < cue.t_grow:
                return prev
            if t < cue.t_hold:
                return lerp(prev, cue.part.shot, smooth((t - cue.t_grow) / (cue.t_hold - cue.t_grow)))
            if t < cue.t_end:
                return cue.part.shot
            prev = cue.part.shot
        # finale: a slow pull-back on the whole cone
        u = clamp01((t - self.t_finale) / (self.duration - self.t_finale))
        return lerp(WHOLE_SHOT, (WHOLE_SHOT[0], WHOLE_SHOT[1], WHOLE_SHOT[2] * 1.08), u)


# --- overlays ----------------------------------------------------------------
class Captions:
    """One caption per part (kicker, title, one line of keywords, the clock) and the end card."""

    def __init__(self, size: Tuple[int, int], fonts: Fonts):
        self.W, self.H = size
        self.f, self.s = fonts, fonts.scale
        self.margin = int(40 * self.s)
        self.cards = {p.number: self._card(p) for p in PARTS}
        self.end_card = self._end_card()

    def _card(self, p: Part) -> Image.Image:
        s = self.s
        f_kick, f_title, f_sub = self.f.get("medium", 22), self.f.get("bold", 46), self.f.get("regular", 24)
        f_clock, f_small = self.f.get("regular", 18), self.f.get("regular", 12)
        pad = int(24 * s)
        w = int(max(f_title.getlength(p.title), f_sub.getlength(p.sub)) + 2 * pad + 20 * s)
        w = max(w, int(0.30 * self.W))
        h = pad + int(30 * s) + int(56 * s) + int(34 * s) + pad // 2
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=int(8 * s), fill=(*NAVY, 205))
        d.rectangle((0, 0, int(6 * s), h - 1), fill=GOLD)
        x0 = pad + int(8 * s)
        d.text((x0, pad), p.kicker, font=f_kick, fill=GOLD)
        d.text((x0, pad + int(30 * s)), p.title, font=f_title, fill=WHITE)
        d.text((x0, pad + int(30 * s) + int(56 * s)), p.sub, font=f_sub, fill=PALE)
        # the clock, to the right of the kicker
        cw = sum(f_clock.getlength(t) for t in _SUP.split(p.clock))
        draw_rich(d, (w - pad - cw, pad + int(2 * s)), p.clock, f_clock, f_small, GOLD)
        return im

    def _end_card(self) -> Image.Image:
        s = self.s
        im = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, self.W, self.H), fill=(*NAVY, 150))
        f_big, f_mid, f_it, f_small = self.f.get("bold", 150), self.f.get("regular", 54), self.f.get("italic", 34), self.f.get("regular", 22)
        cx, cy = self.W // 2, int(self.H * 0.44)
        wj, wjet = f_big.getlength("Jit"), f_big.getlength("Jet")
        x = cx - (wj + wjet) / 2
        d.text((x, cy), "Jit", font=f_big, fill=WHITE, anchor="ls")
        d.text((x + wj, cy), "Jet", font=f_big, fill=GOLD, anchor="ls")
        d.line((cx - int(250 * s), cy + int(28 * s), cx + int(250 * s), cy + int(28 * s)), fill=GOLD, width=max(int(2 * s), 1))
        d.text((cx, cy + int(60 * s)), "Journey Is The Jet", font=f_mid, fill=WHITE, anchor="ma")
        d.text((cx, cy + int(130 * s)), "(bottom to top)", font=f_it, fill=GOLD, anchor="ma")
        d.text((cx, self.H - int(120 * s)), "Ravindra Verma   ·   University of Helsinki   ·   CMS Collaboration",
               font=f_small, fill=PALE, anchor="ma")
        return im


class Renderer:
    def __init__(self, frames: Frames, tl: Timeline, size: Tuple[int, int]):
        self.frames, self.tl, self.size = frames, tl, size
        self.fonts = Fonts(size[1] / 1080.0)
        self.captions = Captions(size, self.fonts)
        self.ruler = Ruler(size, self.fonts)

    def frame(self, t: float) -> Image.Image:
        tl = self.tl
        cx, cy, w = tl.camera(t)
        i, frac = tl.frame_index(t)
        base = self.frames.get(i).view(cx, cy, w, self.size)
        if frac > 0.02:
            base = Image.blend(base, self.frames.get(i + 1).view(cx, cy, w, self.size), frac)
        frame = base.convert("RGBA")
        # caption of the part being shown: in from the end of the growth, out when the next growth starts
        for cue in tl.cues:
            a = min(clamp01((t - cue.t_hold + 0.3) / 0.6), 1.0 - clamp01((t - cue.t_end) / 0.5))
            if a > 0:
                card = self.captions.cards[cue.part.number]
                alpha_paste(frame, card, (self.captions.margin, self.size[1] - self.ruler.strip_h - card.height), a)
        # the progress ruler, from the first growth to the end card
        cur, done = 0, 0.0
        for cue in tl.cues:
            if t >= cue.t_hold:
                cur, done = cue.part.number, (cue.part.number - 1) / 7.0
            elif t >= cue.t_grow:
                done = (cue.part.number - 2 + (t - cue.t_grow) / (cue.t_hold - cue.t_grow)) / 7.0
                cur = cue.part.number - 1
                break
        # the end card; the ruler leaves with it
        a_end = clamp01((t - tl.t_finale - 1.0) / 1.5)
        a_ruler = clamp01((t - tl.cues[0].t_grow + 1.0) / 1.0) * (1.0 - a_end)
        if a_ruler > 0:
            self.ruler.draw(frame, max(done, 0.0), cur, a_ruler)
        if a_end > 0:
            alpha_paste(frame, self.captions.end_card, (0, 0), a_end)
        g = min(clamp01(t / tl.fade_in), clamp01((tl.duration - t) / tl.fade_out))
        if g < 1.0:
            arr = (np.asarray(frame.convert("RGB")).astype(np.float32) * g).astype(np.uint8)
            return Image.fromarray(arr)
        return frame.convert("RGB")


# --- output -------------------------------------------------------------------
def write_music(duration: float, level_db: float, out: Path) -> None:
    st = make_music(duration) * (10 ** (level_db / 20))
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(st, -1, 1) * 32767).astype("<i2").tobytes())


def write_stills(rend: Renderer, tl: Timeline, out: Path) -> None:
    times = [(0.5 * tl.t_title, "title")] + [(c.t_hold + 1.5, f"part{c.part.number}") for c in tl.cues] + \
            [(tl.duration - tl.fade_out - 0.5, "end")]
    ims = [rend.frame(t) for t, _ in times]
    if out.suffix.lower() == ".pdf":
        ims[0].save(out, save_all=True, append_images=ims[1:], resolution=96)
    else:
        for (_, key), im in zip(times, ims):
            im.save(out.with_name(f"{out.stem}_{key}{out.suffix}"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", type=Path, default=HERE / "animate_short_cover_timeline.mp4")
    p.add_argument("--size", default="1920x1080")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dpi", type=float, default=300, help="resolution of the compiled frames")
    p.add_argument("--steps", type=int, default=16, help="compiled growth steps per part")
    p.add_argument("--jobs", type=int, default=6, help="parallel pdflatex jobs")
    p.add_argument("--grow", type=float, default=2.6, help="seconds of growth per part")
    p.add_argument("--hold", type=float, default=3.4, help="seconds of pause per part")
    p.add_argument("--title", type=float, default=3.0, help="seconds on the title before the descent")
    p.add_argument("--finale", type=float, default=9.0, help="seconds on the whole cone and the end card")
    p.add_argument("--music-db", type=float, default=-9.0, help="music level; --no-music for silence")
    p.add_argument("--no-music", action="store_true")
    p.add_argument("--work-dir", type=Path, default=HERE.parent / "build" / "short_cover_frames",
                   help="the compiled frames are kept here and re-used on the next run (make clean removes them)")
    p.add_argument("--stills", type=Path, default=None, help="one key frame per part (.pdf stack or .png files)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--fast", action="store_true", help="preview: 960x540, 12 fps, 150 dpi, 6 steps")
    args = p.parse_args()
    if args.fast:
        args.size, args.fps, args.dpi, args.steps = "960x540", 12, 150.0, 6
    W, H = (int(v) for v in args.size.lower().split("x"))
    for tool in ("pdflatex", "gs", "ffmpeg"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found on PATH")

    states, ends = build_states(args.steps)
    tl = Timeline(ends, args.title, args.grow, args.hold, args.finale, fade_out=2.0)
    print(f"{len(states)} cone frames; film {tl.duration:.1f} s")

    tmp = tempfile.TemporaryDirectory(prefix="jitjet_short_")
    pngs = compile_all(states, args.work_dir, args.dpi, args.jobs)
    rend = Renderer(Frames(pngs, args.dpi), tl, (W, H))
    if args.stills:
        write_stills(rend, tl, args.stills)
        print(f"wrote {args.stills}")
    if args.no_video:
        return
    audio = Path(tmp.name) / "music.wav"
    write_music(tl.duration, -60.0 if args.no_music else args.music_db, audio)
    print(f"film: {W}x{H} at {args.fps} fps ...")
    long_film.write_video(rend, SimpleNamespace(duration=tl.duration), audio, args.output, args.fps,
                          shutil.which("ffmpeg"), "ultrafast" if args.fast else "medium")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
