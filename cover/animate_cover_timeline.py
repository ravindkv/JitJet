#!/usr/bin/env python3
"""The book trailer: a narrated flight along the cone of cover_timeline.tex.

The film is the cover itself, seen through a moving camera. It opens on the
title, descends to the apex where the b quark is born, and then travels up the
cone, stopping at each of the seven milestones drawn on the cover (the seven
parts of the book) while a narrator introduces the part and its chapters. At
the top it pulls back to show the whole cone, from the hard scattering to the
analysis, closes with the eighth part, and pans down to the credits.

The narration is written in the manner of a natural-history documentary: a
calm, unhurried voice, the present tense, one traveller followed from birth to
arrival. It is *not* the voice of Sir David Attenborough and the script makes
no attempt to imitate him: cloning a real person's voice is not something this
script does. By default the words are spoken by the British voice of macOS
("Daniel", via the `say` command). To use another voice, record the script
(--print-script, or the .srt written next to the film) and pass the clips with
--narration-dir; one file per scene, named after the scene key (title.wav,
part1.wav, ..., part8.wav; any format ffmpeg can read).

The music is synthesised here with numpy: a slow four-chord pad over a drone,
ducked under the voice. There is nothing to license.

Scenes (the keys accepted by --scene)
  title   the title of the book, on the cover
  part1   I    The family is born        ME, parton shower, hadrons     (apex)
  part2   II   On the road               pileup, detector, L1T, HLT
  part3   III  Taking the census         RecHits, Particle Flow
  part4   IV   Who belongs to the family PUPPI, anti-kT clustering
  part5   V    Settling the accounts     JEC, JER
  part6   VI   Arrival                   b tagging
  part7   VII  The top family            top tagging                    (the AK8 cap)
  part8   VIII The complete pilgrimage   the whole cone, thirty orders of magnitude
  credits the credits of the cover

Examples (needs numpy, Pillow and ffmpeg; on this Mac use /usr/bin/python3):
  python3 animate_cover_timeline.py                       # -> animate_cover_timeline.mp4 (+ .srt)
  python3 animate_cover_timeline.py --fast                # 960x540 preview, quick to render
  python3 animate_cover_timeline.py --stills stills.pdf --no-video   # one key frame per scene
  python3 animate_cover_timeline.py --scene part1,part2 -o test.mp4
  python3 animate_cover_timeline.py --print-script        # the narration, for recording it yourself
  python3 animate_cover_timeline.py --narration-dir myvoice/   # your own recording of the script

The cover is rendered from cover_timeline.pdf with ghostscript (gs); the PDF is
compiled from cover_timeline.tex with pdflatex if it is missing.
"""
from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
COVER_TEX = HERE / "cover_timeline.tex"
COVER_PDF = HERE / "cover_timeline.pdf"

# --- the cover's geometry (copied from cover_timeline.tex) -------------------
PAGE_W, PAGE_H = 21.0, 29.7        # cm
IP = (2.8, 6.6)                    # the apex of the cone on the page, cm
ANGLE = 40.0                       # the cone axis is rotated by this angle, degrees
_CA, _SA = math.cos(math.radians(ANGLE)), math.sin(math.radians(ANGLE))


def axis_point(s: float, y: float = 0.0) -> Tuple[float, float]:
    """Page coordinates (cm) of the point at distance s along the cone axis, offset y across it."""
    return (IP[0] + _CA * s - _SA * y, IP[1] + _SA * s + _CA * y)


# --- palette (as in cover_timeline.tex) ---------------------------------------
NAVY = (11, 29, 58)
NAVYTOP = (23, 54, 95)
GOLD = (244, 185, 66)
BRED = (255, 90, 90)
WHITE = (255, 255, 255)
PALE = (206, 214, 228)            # white!75!navy, roughly
SR = 44100                        # audio sample rate


def smooth(t: float) -> float:
    """Ease-in/out in [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def clamp01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


# --- the scenes ---------------------------------------------------------------
@dataclass
class Segment:
    key: str
    kicker: str                       # small gold line above the title, e.g. "PART I  ·  THE FAMILY IS BORN"
    title: str                        # the title of the part in the book
    chapters: List[str]               # chapter lines, revealed one by one
    clock: str                        # the stage on the cover's ruler ("10^{-27} s  to  10^{-23} s")
    shot: Tuple[float, float, float]  # camera centre (cm, cm) and width (cm) of the view
    card: str                         # where the caption card sits: bl, br, tl, tr, or "" for none
    spoken: str                       # the narration ("..." is a pause)
    hold_min: float = 4.0             # shortest stay, s
    move: float = 3.0                 # travel time from the previous scene, s
    milestone: int = 0                # 1..8 lights the node on the progress ruler


def build_segments() -> List[Segment]:
    p1 = axis_point(2.6, 0.9)     # apex, shower and hadron slices, and the label of part I
    p2 = axis_point(9.2, 1.3)     # pileup, the detector wall and the two gates
    p3 = axis_point(12.4, 1.5)
    p4 = axis_point(14.1, 1.6)
    p5 = axis_point(15.6, 1.7)
    p6 = axis_point(17.0, 1.9)
    p7 = axis_point(18.9, -0.4)   # the AK8 cap and the label of part VII
    return [
        Segment("title", "", "", [], "",
                (10.5, 26.9, 13.5), "",
                "There are journeys on this planet that take a lifetime. ... And there are journeys "
                "that are over in a fraction of a second, ... and yet contain a whole life. ... "
                "This is the story of one such traveller. ... A jet. ... "
                "And its story is told, as every good story is told, from the beginning to the end. "
                "... From bottom, ... to top.",
                hold_min=6.0, move=0.0),
        Segment("part1", "PART I  ·  THE FAMILY IS BORN", "Before the Journey: The Family Is Born",
                ["1  The Ancestral Home: Protons, PDFs and the Hard Scattering",
                 "2  The Mother Gives Birth: Parton Showering",
                 "3  Whose Child Is It? Matching and Merging",
                 "4  Dressing the Children: Hadronisation and Heavy-Flavour Decays",
                 "5  Cousins Nobody Invited: Underlying Event and Colour Reconnection",
                 "6  The Family Photograph: Generator-Level Jets"],
                "10^{-27} s  to  10^{-23} s",
                (p1[0] + 1.1, p1[1] - 0.15, 8.6), "br",
                "Here, at a single point, two protons meet. ... At ten to the minus twenty-seven "
                "seconds, a bottom quark is born, together with her twin, who leaves at once in the "
                "opposite direction. ... She cannot travel alone: within moments she is radiating "
                "gluons, a shower of children, each a little softer than the last. ... By ten to the "
                "minus twenty-three seconds the family has dressed itself in hadrons: pions, kaons, a "
                "photon or two, and one heavy B hadron that carries the mother's name. ... "
                "Part One: The Family Is Born. Six chapters, from the ancestral home of the proton, "
                "to the family photograph.",
                milestone=1, move=4.0),
        Segment("part2", "PART II  ·  ON THE ROAD", "On the Road: Crowds, Borders and Checkpoints",
                ["7  The Crowd on the Road: Pileup",
                 "8  Crossing the Border: Detector Simulation and Digitisation",
                 "9  The First Checkpoint: Level-1 Trigger",
                 "10  The Second Checkpoint: High-Level Trigger"],
                "same crossing  ·  25 ns  ·  4 μs  ·  0.3 s",
                (p2[0] - 0.3, p2[1] - 0.1, 9.5), "br",
                "The road is not empty. ... In the same crossing of the beams, some sixty other "
                "collisions take place, and their debris joins the travellers: the crowd on the road. "
                "... Twenty-five nanoseconds after the birth, the family reaches the border: the "
                "tracker, the calorimeters, the muon chambers, each taking its toll. ... One member "
                "does not stop at the wall at all. The neutrino ... simply never arrives. ... Then, two "
                "checkpoints: the Level-One trigger, which must decide in four microseconds, ... and "
                "the High-Level Trigger, given a third of a second. Fail either, and the journey ends "
                "here, unrecorded. ... Part Two: On the Road. Crowds, borders, and checkpoints.",
                milestone=2),
        Segment("part3", "PART III  ·  TAKING THE CENSUS", "Taking the Census: Reconstruction",
                ["11  Footprints and Fingerprints: Local Reconstruction",
                 "12  The Census Takers: Particle Flow"],
                "hours later",
                (p3[0] - 0.2, p3[1] - 0.2, 8.0), "br",
                "Hours later, in the quiet of the computing farms, the census begins. ... Every "
                "footprint on the wall, every hit and every cell, is gathered up, and Particle Flow "
                "names them one by one: charged hadron, ... photon, ... neutral hadron, ... muon. ... "
                "Part Three: Taking the Census. Local reconstruction, and the census takers themselves.",
                milestone=3),
        Segment("part4", "PART IV  ·  WHO BELONGS TO THE FAMILY?", "Who Belongs to the Family? Pileup Mitigation and Clustering",
                ["13  The Police: Charged-Hadron Subtraction and PUPPI",
                 "14  The Inn Where Everyone Gathers: Jet Clustering",
                 "15  Impostors at the Inn: Jet ID, Noise and Veto Maps"],
                "hours later",
                (p4[0] - 0.2, p4[1] - 0.2, 8.0), "br",
                "But who, among all of these, truly belongs to the family? ... The police arrive, in "
                "the form of charged-hadron subtraction and PUPPI, and every candidate is given a "
                "weight: how likely it is to be a genuine child, and not a stranger from the crowd. "
                "... Then the anti-k-T algorithm gathers what is left into a cone of radius nought "
                "point four. The inn, where everyone gathers. ... "
                "Part Four: Who Belongs to the Family?",
                milestone=4),
        Segment("part5", "PART V  ·  SETTLING THE ACCOUNTS", "Settling the Accounts: Jet Energy Corrections",
                ["16  Paying for the Uninvited: L1 Pileup Offset Correction",
                 "17  The Fair Price: MC-Truth Corrections (L2Relative, L3Absolute)",
                 "18  The Local Tax: Residual Corrections on Data",
                 "19  The Wobbly Ruler: Jet Energy Resolution",
                 "20  The Receipt: JES and JER Uncertainties"],
                "hours to months",
                (p5[0] - 0.3, p5[1] - 0.3, 8.4), "br",
                "Now the accounts must be settled. ... The detector has under-counted, the crowd has "
                "over-paid, and so the jet energy corrections are applied, level by level: the pile-up "
                "offset, ... the fair price of the Monte Carlo truth, ... the local tax of the "
                "residuals on data. ... And one must admit that the ruler itself is a little wobbly: "
                "the jet energy resolution. ... "
                "Part Five: Settling the Accounts. Five chapters, ending with the receipt.",
                milestone=5),
        Segment("part6", "PART VI  ·  ARRIVAL", "Arrival: Recognising the Family",
                ["21  The Registry Office: b-Tagging",
                 "22  The Families That Never Arrived: MET and Unclustered Energy"],
                "months",
                (p6[0] - 0.4, p6[1] - 0.5, 8.8), "tl",
                "And so, at last: ... arrival. ... The registry office examines the family. A "
                "secondary vertex, ... a displaced track, ... a soft muon; ... and it stamps the "
                "passport. This is a b jet. ... "
                "Part Six: Arrival. ... With a chapter for the families that never arrived: the "
                "missing transverse energy.",
                milestone=6),
        Segment("part7", "PART VII  ·  THE TOP FAMILY", "The Top Family",
                ["23  Three Families as One: The Boosted Top Jet",
                 "24  The Registry Office II: Top-Tagging"],
                "R = 0.8",
                (p7[0] - 0.3, p7[1] - 0.2, 15.0), "bl",
                "But the road does not end there. ... On a larger scale, in a cone of radius nought "
                "point eight, three families travel as one: a b quark, and the two children of a W "
                "boson: ... the whole family of a top quark, boosted so hard that it arrives together. "
                "... Three prongs, ... one jet. ... "
                "Part Seven: The Top Family. ... From bottom, ... to top.",
                milestone=7, move=3.5),
        Segment("part8", "PART VIII  ·  THE COMPLETE PILGRIMAGE", "The Complete Pilgrimage",
                ["25  The Energy Ledger: The Whole Journey of Our Two Jets",
                 "26  Practical Life in JME",
                 "27  The Next Pilgrimage: Run 3, Phase-2 and Beyond"],
                "10^{-27} s  to  months",
                (10.8, 14.4, 31.0), "bl",
                "Thirty orders of magnitude in time: from ten to the minus twenty-seven seconds, to "
                "the months of an analysis. ... And every slice of the cone is the same small patch of "
                "eta and phi, around the same axis. ... The cone is the jet. ... The journey is the "
                "jet. ... Part Eight closes the pilgrimage: with the energy ledger, with practical life "
                "in the J M E group, and with the next pilgrimage: Run Three, Phase Two, and beyond. ... "
                "Welcome aboard.",
                milestone=8, move=4.5),
        Segment("credits", "", "", [], "",
                (10.5, 3.2, 16.0), "", "", hold_min=7.0, move=4.0),
    ]


# --- narration audio ----------------------------------------------------------
def spoken_for_say(text: str) -> str:
    """The pauses of the script as embedded silences of the macOS synthesiser."""
    return re.sub(r"\s*\.\.\.\s*", " [[slnc 450]] ", text)


def display_text(text: str) -> str:
    return re.sub(r"\s*\.\.\.\s*", " ... ", text).strip()


def read_audio(path: Path, ffmpeg: str) -> np.ndarray:
    """Any audio file -> float32 mono at SR, via ffmpeg."""
    out = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


def synth_say(text: str, voice: str, rate: int, path: Path) -> None:
    subprocess.run(["say", "-v", voice, "-r", str(rate), f"--data-format=LEI16@{SR}", "-o", str(path),
                    spoken_for_say(text)], check=True)


def trim_silence(x: np.ndarray, thr: float = 0.004, keep: float = 0.15) -> np.ndarray:
    """Cut the leading and trailing silence, keeping `keep` seconds of it."""
    idx = np.flatnonzero(np.abs(x) > thr)
    if idx.size == 0:
        return x
    k = int(keep * SR)
    return x[max(idx[0] - k, 0): min(idx[-1] + k, x.size)]


def load_narration(segments: Sequence[Segment], args, workdir: Path, ffmpeg: str) -> Dict[str, np.ndarray]:
    """One clip per scene, from --narration-dir or from `say`; empty dict with --no-narration."""
    clips: Dict[str, np.ndarray] = {}
    if args.no_narration:
        return clips
    for seg in segments:
        if not seg.spoken:
            continue
        if args.narration_dir:
            hits = sorted(p for p in Path(args.narration_dir).iterdir() if p.stem == seg.key)
            if not hits:
                sys.exit(f"--narration-dir: no clip named {seg.key}.* in {args.narration_dir}")
            clip = read_audio(hits[0], ffmpeg)
        else:
            if shutil.which("say") is None:
                sys.exit("`say` (macOS) not found: pass --narration-dir with recorded clips, or --no-narration")
            path = workdir / f"{seg.key}.wav"
            synth_say(seg.spoken, args.voice, args.rate, path)
            clip = read_audio(path, ffmpeg)
        clip = trim_silence(clip)
        peak = float(np.max(np.abs(clip))) if clip.size else 1.0
        clips[seg.key] = clip * (0.85 / peak if peak > 0 else 1.0)
    return clips


# --- the timeline -------------------------------------------------------------
@dataclass
class Cue:
    seg: Segment
    t_arrive: float
    t_leave: float
    t_voice: float
    voice_len: float


@dataclass
class Timeline:
    cues: List[Cue]
    duration: float
    fade_in: float = 1.5
    fade_out: float = 2.5

    def cue_at(self, t: float) -> Tuple[Cue, Optional[Cue]]:
        """The cue the camera sits on (or is leaving) at time t, and the next one."""
        cur = self.cues[0]
        for i, c in enumerate(self.cues):
            if t >= c.t_arrive - c.seg.move:
                cur = c
        nxt = None
        for c in self.cues:
            if c.t_arrive > cur.t_arrive:
                nxt = c
                break
        return cur, nxt


def build_timeline(segments: Sequence[Segment], clips: Dict[str, np.ndarray], no_narration: bool) -> Timeline:
    cues: List[Cue] = []
    t = 0.0
    lead = 1.0          # the voice starts this long before the camera arrives
    tail = 1.2          # and the camera stays this long after the voice stops
    for i, seg in enumerate(segments):
        arrive = t if i == 0 else t + seg.move
        if seg.key in clips:
            vlen = clips[seg.key].size / SR
        elif seg.spoken and no_narration:
            vlen = len(seg.spoken.split()) / 2.6 + 1.5 * seg.spoken.count("...") * 0.4
        else:
            vlen = 0.0
        t_voice = (arrive + 1.5) if i == 0 else (arrive - lead)
        leave = max(arrive + seg.hold_min, t_voice + vlen + tail) if vlen > 0 else arrive + seg.hold_min
        cues.append(Cue(seg, arrive, leave, t_voice, vlen))
        t = leave
    return Timeline(cues, duration=t + 0.5)


# --- the camera ---------------------------------------------------------------
class Camera:
    """Keyframes (t, cx, cy, w) with eased moves between scenes and a slow drift within one."""

    def __init__(self, tl: Timeline):
        self.keys: List[Tuple[float, float, float, float, bool]] = []   # t, cx, cy, w, ease-to-next
        for cue in tl.cues:
            cx, cy, w = cue.seg.shot
            if cue.seg.key in ("title", "credits", "part8"):
                drift = (cx, cy, w * 0.95)
            else:
                drift = (cx + 0.012 * w * _CA, cy + 0.012 * w * _SA, w * 0.93)
            self.keys.append((cue.t_arrive, cx, cy, w, False))
            self.keys.append((cue.t_leave, *drift, True))

    def at(self, t: float) -> Tuple[float, float, float]:
        keys = self.keys
        if t <= keys[0][0]:
            return keys[0][1:4]
        if t >= keys[-1][0]:
            return keys[-1][1:4]
        for (t0, x0, y0, w0, ease), (t1, x1, y1, w1, _) in zip(keys, keys[1:]):
            if t0 <= t < t1:
                u = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
                u = smooth(u) if ease else u
                return (x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, w0 * (w1 / w0) ** u)
        return keys[-1][1:4]


# --- the cover as an image ----------------------------------------------------
def ensure_cover_pdf() -> Path:
    if COVER_PDF.exists():
        return COVER_PDF
    if shutil.which("pdflatex") is None:
        sys.exit(f"{COVER_PDF} missing and pdflatex not found")
    print(f"compiling {COVER_TEX.name} ...")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", COVER_TEX.name],
                   cwd=HERE, check=True, stdout=subprocess.DEVNULL)
    return COVER_PDF


class Cover:
    """The rendered cover, padded with its own background, at several resolutions."""

    PAD = (7.0, 7.0)   # cm added on each side (x, y): the widest shot leaves the page

    def __init__(self, png: Path, dpi: float):
        self.dpi = dpi
        arr = np.asarray(Image.open(png).convert("RGB"))
        px, py = int(round(self.PAD[0] * dpi / 2.54)), int(round(self.PAD[1] * dpi / 2.54))
        arr = np.pad(arr, ((py, py), (px, px), (0, 0)), mode="edge")
        self.levels = [Image.fromarray(arr)]
        for _ in range(3):
            self.levels.append(self.levels[-1].reduce(2))

    def px(self, x_cm: float, y_cm: float) -> Tuple[float, float]:
        k = self.dpi / 2.54
        return ((x_cm + self.PAD[0]) * k, (PAGE_H - y_cm + self.PAD[1]) * k)

    def view(self, cx: float, cy: float, w: float, size: Tuple[int, int]) -> Image.Image:
        W, H = size
        h = w * H / W
        x0, y0 = self.px(cx - w / 2, cy + h / 2)
        x1, y1 = self.px(cx + w / 2, cy - h / 2)
        ratio = (x1 - x0) / W                      # source pixels per output pixel at level 0
        k = 0
        while ratio / 2 ** k > 1.75 and k < len(self.levels) - 1:
            k += 1
        f = float(2 ** k)
        # resize() takes a float box, so the camera moves with sub-pixel smoothness
        return self.levels[k].resize((W, H), Image.BICUBIC, box=(x0 / f, y0 / f, x1 / f, y1 / f))


# --- text and cards -----------------------------------------------------------
class Fonts:
    """Helvetica Neue if the Mac has it (the cover uses its TeX clone), else DejaVu Sans."""

    def __init__(self, scale: float):
        self.scale = scale
        self.faces = {"regular": None, "bold": None, "medium": None, "italic": None}
        ttc = Path("/System/Library/Fonts/HelveticaNeue.ttc")
        if ttc.exists():
            for i in range(20):
                try:
                    style = ImageFont.truetype(str(ttc), 20, index=i).getname()[1]
                except OSError:
                    break
                key = {"Regular": "regular", "Bold": "bold", "Medium": "medium", "Italic": "italic"}.get(style)
                if key and self.faces[key] is None:
                    self.faces[key] = (str(ttc), i)
        try:
            import matplotlib.font_manager as fm
            dejavu = Path(fm.findfont("DejaVu Sans"))
        except Exception:
            dejavu = None
        for key in self.faces:
            if self.faces[key] is None:
                if dejavu is None:
                    sys.exit("no usable font: Helvetica Neue or matplotlib's DejaVu Sans is needed")
                name = {"bold": "DejaVuSans-Bold.ttf", "medium": "DejaVuSans-Bold.ttf",
                        "italic": "DejaVuSans-Oblique.ttf"}.get(key, "DejaVuSans.ttf")
                self.faces[key] = (str(dejavu.with_name(name)), 0)
        self._cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

    def get(self, face: str, size: float) -> ImageFont.FreeTypeFont:
        px = max(int(round(size * self.scale)), 6)
        key = (face, px)
        if key not in self._cache:
            path, idx = self.faces[face]
            self._cache[key] = ImageFont.truetype(path, px, index=idx)
        return self._cache[key]


_SUP = re.compile(r"\^\{([^}]*)\}")


def draw_rich(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, font, small, fill) -> float:
    """Draw text with ^{...} superscripts; returns the end x."""
    x, y = xy
    pos = 0
    for m in _SUP.finditer(text):
        chunk = text[pos:m.start()]
        draw.text((x, y), chunk, font=font, fill=fill)
        x += font.getlength(chunk)
        draw.text((x, y - 0.35 * font.size), m.group(1), font=small, fill=fill)
        x += small.getlength(m.group(1))
        pos = m.end()
    chunk = text[pos:]
    draw.text((x, y), chunk, font=font, fill=fill)
    return x + font.getlength(chunk)


def wrap(text: str, font, width: float) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        if font.getlength(trial) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


@dataclass
class Card:
    """The caption of a scene: a header image and one image per chapter line, all RGBA."""
    x: int
    y: int
    header: Image.Image
    lines: List[Image.Image]
    line_y: List[int]
    total_h: int


class CardArtist:
    def __init__(self, size: Tuple[int, int], fonts: Fonts):
        self.W, self.H = size
        self.f = fonts
        self.s = fonts.scale
        self.cw = int(0.44 * self.W)          # card width
        self.margin = int(40 * self.s)
        self.ruler_h = int(70 * self.s)

    def build(self, seg: Segment) -> Optional[Card]:
        if not seg.card:
            return None
        s = self.s
        f_kick, f_small = self.f.get("medium", 20), self.f.get("regular", 14)
        f_title = self.f.get("bold", 34)
        f_ch, f_num = self.f.get("regular", 22), self.f.get("medium", 22)
        pad = int(22 * s)
        inner = self.cw - 2 * pad - int(10 * s)

        # header: kicker line (left) and the clock (right), then the title, wrapped
        title_lines = wrap(seg.title, f_title, inner)
        hh = pad + int(30 * s) + len(title_lines) * int(42 * s) + int(6 * s)
        header = Image.new("RGBA", (self.cw, hh), (0, 0, 0, 0))
        d = ImageDraw.Draw(header)
        d.rounded_rectangle((0, 0, self.cw - 1, hh - 1), radius=int(8 * s), fill=(*NAVY, 200))
        d.rectangle((0, 0, int(6 * s), hh - 1), fill=GOLD)
        x0 = pad + int(10 * s)
        d.text((x0, pad), seg.kicker, font=f_kick, fill=GOLD)
        if seg.clock:
            cw = sum(f_small.getlength(t) for t in _SUP.split(seg.clock))
            draw_rich(d, (self.cw - pad - cw, pad + int(4 * s)), seg.clock, f_small, self.f.get("regular", 10), PALE)
        y = pad + int(30 * s)
        for ln in title_lines:
            d.text((x0, y), ln, font=f_title, fill=WHITE)
            y += int(42 * s)

        # chapter lines, each its own image so that they can appear one after the other
        lines, line_y = [], []
        lh = int(30 * s)
        num_w = int(44 * s)
        ytot = hh
        for ch in seg.chapters:
            num, _, rest = ch.partition("  ")
            wrapped = wrap(rest, f_ch, inner - num_w)
            h = lh * len(wrapped) + int(4 * s)
            im = Image.new("RGBA", (self.cw, h), (0, 0, 0, 0))
            dd = ImageDraw.Draw(im)
            dd.rectangle((0, 0, self.cw - 1, h - 1), fill=(*NAVY, 200))
            dd.rectangle((0, 0, int(6 * s), h - 1), fill=GOLD)
            dd.text((x0 + num_w - int(10 * s), int(2 * s)), num, font=f_num, fill=GOLD, anchor="ra")
            for j, ln in enumerate(wrapped):
                dd.text((x0 + num_w, int(2 * s) + j * lh), ln, font=f_ch, fill=WHITE)
            lines.append(im)
            line_y.append(ytot)
            ytot += h
        ytot += int(10 * s)
        # a closing strip with rounded bottom corners
        foot = Image.new("RGBA", (self.cw, int(10 * s)), (0, 0, 0, 0))
        ImageDraw.Draw(foot).rounded_rectangle((0, -int(10 * s), self.cw - 1, int(10 * s) - 1), radius=int(8 * s), fill=(*NAVY, 200))
        ImageDraw.Draw(foot).rectangle((0, 0, int(6 * s), int(4 * s)), fill=GOLD)
        lines.append(foot)
        line_y.append(ytot - int(10 * s))

        if seg.card[0] == "b":
            y0 = self.H - self.ruler_h - self.margin // 2 - ytot
        else:
            y0 = self.margin
        x = self.margin if seg.card[1] == "l" else self.W - self.margin - self.cw
        return Card(x, y0, header, lines, line_y, ytot)


def alpha_paste(frame: Image.Image, im: Image.Image, xy: Tuple[int, int], alpha: float) -> None:
    if alpha <= 0.0:
        return
    if alpha < 1.0:
        a = im.getchannel("A").point(lambda v: int(v * alpha))
        im = im.copy()
        im.putalpha(a)
    frame.alpha_composite(im, dest=xy)


class Ruler:
    """The progress ruler at the bottom: the eight parts, the current one lit."""
    ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]

    def __init__(self, size: Tuple[int, int], fonts: Fonts):
        self.W, self.H = size
        self.s = fonts.scale
        self.font = fonts.get("medium", 15)
        self.font_b = fonts.get("bold", 15)
        self.x0, self.x1 = int(0.10 * self.W), int(0.90 * self.W)
        self.strip_h = int(70 * self.s)
        self.y = self.strip_h - int(34 * self.s)

    def draw(self, frame: Image.Image, done: float, current: int, alpha: float) -> None:
        """done: fraction of the ruler travelled (0..1); current: lit node (1..8, 0 for none)."""
        if alpha <= 0:
            return
        s = self.s
        layer = Image.new("RGBA", (self.W, self.strip_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        a = int(255 * alpha)
        gold, dim = (*GOLD, a), (*GOLD, int(0.35 * a))
        d.line((self.x0, self.y, self.x1, self.y), fill=dim, width=max(int(2 * s), 1))
        xd = self.x0 + (self.x1 - self.x0) * clamp01(done)
        d.line((self.x0, self.y, xd, self.y), fill=gold, width=max(int(2 * s), 1))
        n = len(self.ROMAN)
        for i, rom in enumerate(self.ROMAN):
            x = self.x0 + (self.x1 - self.x0) * i / (n - 1)
            r = int((9 if i + 1 == current else 6) * s)
            lit = i + 1 <= current
            d.ellipse((x - r, self.y - r, x + r, self.y + r), fill=(*NAVY, a) if not (i + 1 == current) else gold,
                      outline=gold if lit else dim, width=max(int(2 * s), 1))
            d.text((x, self.y - int(14 * s)), rom, font=self.font_b if i + 1 == current else self.font,
                   fill=gold if lit else dim, anchor="ms")
        frame.alpha_composite(layer, dest=(0, self.H - self.strip_h))


# --- rendering ----------------------------------------------------------------
class Renderer:
    def __init__(self, cover: Cover, tl: Timeline, size: Tuple[int, int]):
        self.cover, self.tl, self.size = cover, tl, size
        self.fonts = Fonts(size[1] / 1080.0)
        self.camera = Camera(tl)
        artist = CardArtist(size, self.fonts)
        self.cards = {c.seg.key: artist.build(c.seg) for c in tl.cues}
        self.ruler = Ruler(size, self.fonts)
        self.n_parts = 8
        marked = [c.t_arrive for c in tl.cues if c.seg.milestone > 0]
        self.t_ruler = min(marked) if marked else None    # the ruler appears just before the first part

    def frame(self, t: float) -> Image.Image:
        cx, cy, w = self.camera.at(t)
        frame = self.cover.view(cx, cy, w, self.size).convert("RGBA")
        cur, nxt = self.tl.cue_at(t)
        # caption card of the current scene (fades in on arrival, lines one by one, fades out when leaving)
        card = self.cards.get(cur.seg.key)
        if card is not None:
            a_in = clamp01((t - (cur.t_arrive - 0.4)) / 0.7)
            a_out = 1.0 - clamp01((t - cur.t_leave) / 0.6)
            a = min(a_in, a_out)
            alpha_paste(frame, card.header, (card.x, card.y), a)
            for i, (im, ly) in enumerate(zip(card.lines, card.line_y)):
                ai = clamp01((t - (cur.t_arrive + 0.7 + 0.55 * i)) / 0.5) if i < len(card.lines) - 1 else \
                    clamp01((t - (cur.t_arrive + 0.7 + 0.55 * (len(card.lines) - 2))) / 0.5)
                alpha_paste(frame, im, (card.x, card.y + ly), min(a, ai))
        # progress ruler: appears just before the first part, travels during the moves, fades before the credits
        m_cur = cur.seg.milestone
        moving = nxt is not None and t > cur.t_leave
        if self.t_ruler is not None and (m_cur > 0 or (moving and nxt.seg.milestone > 0)):
            if moving and nxt.seg.milestone > 0 and m_cur > 0:
                u = smooth((t - cur.t_leave) / (nxt.t_arrive - cur.t_leave))
                done = ((m_cur - 1) + u * (nxt.seg.milestone - m_cur)) / (self.n_parts - 1)
            else:
                done = max(m_cur - 1, 0) / (self.n_parts - 1)
            a = clamp01((t - self.t_ruler + 2.0) / 1.0)
            if moving and nxt.seg.milestone == 0:
                a = min(a, 1.0 - clamp01((t - cur.t_leave) / 1.0))
            elif m_cur == 0 and not moving:
                a = 0.0
            self.ruler.draw(frame, done, m_cur, a)
        # fades
        g = min(clamp01(t / self.tl.fade_in), clamp01((self.tl.duration - t) / self.tl.fade_out))
        if g < 1.0:
            arr = (np.asarray(frame.convert("RGB")).astype(np.float32) * g).astype(np.uint8)
            return Image.fromarray(arr)
        return frame.convert("RGB")


# --- music and the mix --------------------------------------------------------
def note(f: float, n: int, sr: int = SR, detune: float = 0.0) -> np.ndarray:
    """A soft pad note: a few harmonics with a slow vibrato, length n samples."""
    t = np.arange(n, dtype=np.float32) / sr
    vib = 1.0 + 0.0025 * np.sin(2 * np.pi * 0.17 * t + f)
    out = np.zeros(n, dtype=np.float32)
    for h, amp in ((1, 1.0), (2, 0.35), (3, 0.16), (4, 0.06)):
        out += amp * np.sin(2 * np.pi * (f + detune) * h * vib * t).astype(np.float32)
    return out


def make_music(duration: float, sr: int = SR) -> np.ndarray:
    """A slow four-chord pad over a drone, stereo, peak 0.5, fading in and out."""
    D2, C3, D3, E3, F3, G3, A3, Bb2, C4 = 73.42, 130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 116.54, 261.63
    chords = [(D3, F3, A3), (Bb2, D3, F3), (F3, A3, C4), (C3, E3, G3)]
    n = int(duration * sr) + sr
    L, R = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    chord_len, xfade = 11.0, 4.0
    t0 = 0.0
    i = 0
    while t0 < duration:
        c = chords[i % len(chords)]
        n0, n1 = int(t0 * sr), min(int((t0 + chord_len + xfade) * sr), n)
        m = n1 - n0
        env = np.ones(m, dtype=np.float32)
        k = min(int(xfade * sr), m)
        env[:k] = np.linspace(0, 1, k) ** 1.5
        env[-k:] *= np.linspace(1, 0, k) ** 1.5
        for f in c:
            L[n0:n1] += env * note(f, m, sr, detune=+0.25 * f / 220)
            R[n0:n1] += env * note(f, m, sr, detune=-0.25 * f / 220)
        t0 += chord_len
        i += 1
    t = np.arange(n, dtype=np.float32) / sr
    drone = (0.8 + 0.2 * np.sin(2 * np.pi * 0.045 * t)) * (np.sin(2 * np.pi * D2 * t) + 0.3 * np.sin(2 * np.pi * 2 * D2 * t))
    L += 0.9 * drone.astype(np.float32)
    R += 0.9 * drone.astype(np.float32)
    glob = np.ones(n, dtype=np.float32)
    k = int(5 * sr)
    glob[:k] = np.linspace(0, 1, k) ** 2
    kf = int(7 * sr)
    end = int(duration * sr)
    glob[end - kf:end] *= np.linspace(1, 0, kf) ** 1.5
    glob[end:] = 0
    st = np.stack([L, R], axis=1) * glob[:, None]
    st *= 0.5 / max(float(np.max(np.abs(st))), 1e-6)
    return st[:int(duration * sr)]


def mix_audio(tl: Timeline, clips: Dict[str, np.ndarray], music_db: Optional[float], out: Path) -> None:
    n = int(tl.duration * SR)
    voice = np.zeros(n, dtype=np.float32)
    for cue in tl.cues:
        clip = clips.get(cue.seg.key)
        if clip is None:
            continue
        i0 = int(cue.t_voice * SR)
        m = min(clip.size, n - i0)
        if m > 0:
            voice[i0:i0 + m] += clip[:m]
    stereo = np.stack([voice, voice], axis=1)
    if music_db is not None:
        music = make_music(tl.duration)
        # duck the music under the voice: an envelope at 100 Hz, smoothed over 0.3 s
        dec = 441
        env = np.abs(voice[: (n // dec) * dec]).reshape(-1, dec).max(axis=1)
        k = 30
        env = np.convolve(env, np.ones(k) / k, mode="same")
        env = np.maximum(env, np.convolve(env, np.ones(2 * k) / (2 * k), mode="same"))
        gain = 1.0 - 0.6 * np.clip(env / 0.08, 0, 1)
        gain_full = np.interp(np.arange(n) / dec, np.arange(env.size), gain).astype(np.float32)
        stereo += music[:n] * (10 ** (music_db / 20)) * gain_full[:, None]
    stereo = np.clip(stereo, -1, 1)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((stereo * 32767).astype("<i2").tobytes())


# --- subtitles, script --------------------------------------------------------
def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(tl: Timeline, path: Path) -> None:
    k = 1
    with open(path, "w") as f:
        for cue in tl.cues:
            if not cue.seg.spoken or cue.voice_len <= 0:
                continue
            text = re.sub(r"\s*\.\.\.\s*", " ", cue.seg.spoken)
            parts = [p.strip() for p in re.split(r"(?<=[.?!])\s+", text) if p.strip()]
            # merge fragments that are only a pause or very short
            merged: List[str] = []
            for p in parts:
                if merged and len(p) < 25:
                    merged[-1] += " " + p
                else:
                    merged.append(p)
            total = sum(len(p) for p in merged)
            t = cue.t_voice
            for p in merged:
                d = cue.voice_len * len(p) / total
                f.write(f"{k}\n{srt_time(t)} --> {srt_time(t + d - 0.05)}\n{p}\n\n")
                t += d
                k += 1


def print_script(segments: Sequence[Segment]) -> None:
    for seg in segments:
        if seg.spoken:
            print(f"[{seg.key}]\n{display_text(seg.spoken)}\n")


# --- output -------------------------------------------------------------------
def write_video(rend: Renderer, tl: Timeline, audio: Path, out: Path, fps: int, ffmpeg: str, preset: str) -> None:
    W, H = rend.size
    n = int(math.ceil(tl.duration * fps))
    cmd = [ffmpeg, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-i", str(audio),
           "-c:v", "libx264", "-preset", preset, "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for i in range(n):
        proc.stdin.write(rend.frame(i / fps).tobytes())
        if i % (10 * fps) == 0:
            print(f"  frame {i}/{n}  t = {i / fps:6.1f} s", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")


def write_frames(rend: Renderer, tl: Timeline, outdir: Path, fps: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    n = int(math.ceil(tl.duration * fps))
    for i in range(n):
        rend.frame(i / fps).save(outdir / f"frame_{i:05d}.png")


def write_stills(rend: Renderer, tl: Timeline, out: Path) -> None:
    """One key frame per scene, when the card is complete: a PDF stack or numbered PNGs."""
    ims = []
    for cue in tl.cues:
        t = cue.t_arrive + min(0.7 * (cue.t_leave - cue.t_arrive), 0.7 + 0.55 * (len(cue.seg.chapters) + 1))
        ims.append(rend.frame(t))
    if out.suffix.lower() == ".pdf":
        ims[0].save(out, save_all=True, append_images=ims[1:], resolution=96)
    else:
        for cue, im in zip(tl.cues, ims):
            im.save(out.with_name(f"{out.stem}_{cue.seg.key}{out.suffix}"))


# --- main ---------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", type=Path, default=HERE / "animate_cover_timeline.mp4")
    p.add_argument("--scene", default="all", help="all, or a comma list of scene keys (title,part1,...,part8,credits)")
    p.add_argument("--size", default="1920x1080", help="frame size WxH (16:9)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dpi", type=float, default=400, help="resolution at which the cover PDF is rendered")
    p.add_argument("--fast", action="store_true", help="preview: 960x540, 12 fps, 150 dpi, fast x264 preset")
    p.add_argument("--voice", default="Daniel", help="macOS `say` voice (Daniel is British English)")
    p.add_argument("--rate", type=int, default=156, help="speaking rate in words per minute")
    p.add_argument("--narration-dir", type=Path, default=None, help="use recorded clips <key>.wav|mp3|... instead of `say`")
    p.add_argument("--no-narration", action="store_true", help="captions and music only (scenes keep a nominal length)")
    p.add_argument("--no-music", action="store_true")
    p.add_argument("--music-db", type=float, default=-14.0, help="music level relative to the voice")
    p.add_argument("--stills", type=Path, default=None, help="write one key frame per scene (.pdf stack or .png files)")
    p.add_argument("--no-video", action="store_true", help="skip the film (with --stills or --frames-dir)")
    p.add_argument("--frames-dir", type=Path, default=None, help="write the frames as PNG files instead of a film")
    p.add_argument("--keep-audio", action="store_true", help="also write the mixed soundtrack as a .wav next to the film")
    p.add_argument("--print-script", action="store_true", help="print the narration and exit")
    args = p.parse_args()

    segments = build_segments()
    if args.print_script:
        print_script(segments)
        return
    if args.scene != "all":
        keys = [k.strip() for k in args.scene.split(",")]
        unknown = [k for k in keys if k not in {s.key for s in segments}]
        if unknown:
            sys.exit(f"unknown scene(s): {', '.join(unknown)}")
        segments = [s for s in segments if s.key in keys]
    if args.fast:
        args.size, args.fps, args.dpi = "960x540", 12, 150.0
    W, H = (int(v) for v in args.size.lower().split("x"))
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        sys.exit("ffmpeg not found on PATH")
    gs = shutil.which("gs")
    if gs is None:
        sys.exit("ghostscript (gs) not found on PATH")

    with tempfile.TemporaryDirectory(prefix="jitjet_trailer_") as tmp:
        work = Path(tmp)
        pdf = ensure_cover_pdf()
        png = work / "cover.png"
        print(f"rendering {pdf.name} at {args.dpi:.0f} dpi ...")
        subprocess.run([gs, "-q", "-sDEVICE=png16m", f"-r{args.dpi:g}", "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
                        "-dFirstPage=1", "-dLastPage=1", "-o", str(png), str(pdf)], check=True)
        cover = Cover(png, args.dpi)

        print("narration ...")
        clips = load_narration(segments, args, work, ffmpeg)
        tl = build_timeline(segments, clips, args.no_narration)
        for cue in tl.cues:
            print(f"  {cue.seg.key:8s} arrive {cue.t_arrive:6.1f} s  leave {cue.t_leave:6.1f} s  voice {cue.voice_len:5.1f} s")
        print(f"  total {tl.duration:.1f} s")

        rend = Renderer(cover, tl, (W, H))
        if args.stills:
            write_stills(rend, tl, args.stills)
            print(f"wrote {args.stills}")
        if args.frames_dir:
            write_frames(rend, tl, args.frames_dir, args.fps)
            print(f"wrote frames to {args.frames_dir}")
        if args.no_video:
            return

        audio = work / "mix.wav"
        print("soundtrack ...")
        mix_audio(tl, clips, None if args.no_music else args.music_db, audio)
        if args.keep_audio:
            shutil.copy(audio, args.output.with_suffix(".wav"))
        write_srt(tl, args.output.with_suffix(".srt"))
        print(f"film: {W}x{H} at {args.fps} fps, {tl.duration:.0f} s ...")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_video(rend, tl, audio, args.output, args.fps, ffmpeg, "ultrafast" if args.fast else "medium")
        print(f"wrote {args.output} and {args.output.with_suffix('.srt').name}")


if __name__ == "__main__":
    main()
