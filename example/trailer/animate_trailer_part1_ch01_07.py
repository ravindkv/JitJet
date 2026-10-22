#!/usr/bin/env python3
"""The Part I trailer (chapters 1-7): six pages of the CMSSW stack in a row, narrated.

One simulated event, followed through the first seven chapters of the book. The
film shows the first six pages of
example/GenJet/CMSSW/genJetAnalyzer_ME_PS_MPI_Had_Decay_GenJet_out.pdf one after
the other: the eta-phi plane after the hard process, after the parton shower,
after MPI and the beam remnants, after hadronisation, after the hadron decays,
and then the anti-kT jets (R = 0.4). The two further pages of the PDF (the
same jets without neutrinos, R = 0.8) are left out, for length. Each page fills
the height of the frame, whole, and a
counter in the sidebar counts the particles the stage produced:
2, 57, 169, 320, 634, and then 239 of the 634 inside the jets. A card names the
chapter (and so the film) that explains the stage, and one question per stop
asks what that chapter answers; the narration does not spell the chapter
numbers, the card does. A ruler at the bottom lights the chapters as they pass.

Narration and music
  One line per stop (NARRATION below), spoken by the same neural voice as the
  chapter films through example/soundtrack.py (edge-tts, needs the network;
  --engine say for the macOS voice; --narration-dir for recordings; keys:
  title, s1 ... s6, end; see --print-script). Each stop lasts as long as its
  line, so the film is about 100 s narrated and about 55 s with
  --no-narration. The score follows the cues: the pad changes chord with every
  page, the pulse gains a voice with every stop (as the event gains particles),
  a bell rings as each page settles, a thud marks the collision. It ducks under
  the voice. An .srt with the narration is written next to the .mp4.

Frames
  The pages are rendered once with ghostscript at --dpi into build/trailer_frames
  (re-used by later runs; delete the directory after a change to the PDF), and
  the film is drawn with Pillow and piped to ffmpeg (H.264 at --crf 26, about
  12 MB; 1920x1080, 30 fps). The page fills the height of the frame, whole, and
  the cards live in a sidebar on the right.

Examples
  /usr/bin/python3 example/trailer/animate_trailer_part1_ch01_07.py
  /usr/bin/python3 example/trailer/animate_trailer_part1_ch01_07.py --fast      # 960x540 preview in a minute
  /usr/bin/python3 example/trailer/animate_trailer_part1_ch01_07.py --print-script
  /usr/bin/python3 example/trailer/animate_trailer_part1_ch01_07.py --audio-only build/trailer.wav
  /usr/bin/python3 example/trailer/animate_trailer_part1_ch01_07.py --stills example/trailer/stills.png --no-video
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "cover"))
from animate_cover_timeline import GOLD, NAVY, NAVYTOP, PALE, WHITE, Fonts, Ruler, alpha_paste, clamp01, smooth, wrap  # noqa: E402
sys.path.insert(0, str(ROOT / "example"))
import soundtrack as snd  # noqa: E402  (example/soundtrack.py: the voice, the score, the mix)

PDF = ROOT / "example" / "GenJet" / "CMSSW" / "genJetAnalyzer_ME_PS_MPI_Had_Decay_GenJet_out.pdf"


# --- the stops: one page of the PDF each --------------------------------------------
@dataclass
class Stop:
    key: str
    page: int                       # page of the PDF (1-based)
    count: int                      # the number in the counter card
    unit: str                       # what the number counts
    kicker: str                     # "CHAPTER 2 · 3"
    title: str                      # the chapter title (the film has the same)
    sub: str                        # the physics of the chapter, in a few words
    question: str                   # what the chapter answers
    chapters: Tuple[int, ...]       # lit on the ruler
    shot: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    # camera (cx, cy, w) at the start and the end of the stop: the centre of the view in fractions
    # of the page, and the width of the view as a fraction of the page width


# The camera: the whole page, still (w = 1: the page fills the height of the frame, every label of it
# visible). Consecutive pages are then in register when they dissolve, and the particles appear in place.
# A slow push-in (say (0.5, 0.5, 1.0) to (0.5, 0.503, 0.97), reversed on the next stop) looks nice but costs
# four times the bitrate: every frame of the fine text changes.
STILL = ((0.5, 0.5, 1.0), (0.5, 0.5, 1.0))

STOPS = [
    Stop("s1", 1, 2, "particles: a b quark and her antiquark",
         "CHAPTER 1", "The Ancestral Home", "protons  ·  parton densities  ·  the hard scattering",
         "Where do two quarks of 113 GeV come from?", (1,), STILL),
    Stop("s2", 2, 57, "particles after the parton shower",
         "CHAPTERS 2 · 3", "The Mother Gives Birth", "parton shower  ·  matching and merging",
         "Why does a colour charge radiate, when does it stop, and whose child is each gluon?", (2, 3), STILL),
    Stop("s3", 3, 169, "particles with MPI and the beam remnants",
         "CHAPTER 4", "Cousins Nobody Invited", "multiparton interactions  ·  beam remnants  ·  colour reconnection",
         "Who invited the other 112?", (4,), STILL),
    Stop("s4", 4, 320, "hadrons: not one quark or gluon is left",
         "CHAPTER 5", "The Children Get Married", "Lund strings  ·  hadronisation  ·  b fragmentation",
         "How does a string break into hadrons?", (5,), STILL),
    Stop("s5", 5, 634, "stable particles after the hadron decays",
         "CHAPTER 6", "The Children Produce Grandchildren", "B-hadron decays  ·  displaced vertices  ·  the neutrino",
         "When does a B meson decay, and what leaves for good?", (6,), STILL),
    Stop("s6", 6, 239, "of the 634 are inside the 27 anti-kT jets (R = 0.4)",
         "CHAPTER 7", "The Family Photograph", "anti-kT R = 0.4  ·  the b jet: 118 GeV from a 92 GeV quark",
         "Why does the b jet carry more than the b quark?", (7,), STILL),
]
# Pages 7 and 8 of the PDF (ak4GenJetsNoNu: 80 GeV in the same b jet without its neutrino; ak8GenJets: 488 of
# the 634 inside the 28 wide jets) are not shown: they cost twenty seconds and the chapter 7 film has them.
CHAIN = [2, 57, 169, 320, 634]          # the breadcrumb under the counter
N_CHAPTERS = 7

# The narration, one line per stop ("..." is a pause). The clip of a line sets how long its stop lasts.
# The chapter numbers are not spoken: the card in the sidebar names the chapter.
NARRATION = {
    "title": "One collision, one simulated event, seven chapters... How many particles does it take to make a jet?",
    "s1": "It begins with two: a bottom quark and her antiquark, back to back, one hundred and thirteen GeV each... "
          "Where do they come from?",
    "s2": "Then fifty-seven. The quarks radiate and the gluons split... "
          "Why does a colour charge shower, and whose child is each gluon?",
    "s3": "One hundred and sixty-nine. The rest of the protons joins in: extra scatterings, beam remnants... "
          "Who invited the other hundred and twelve?",
    "s4": "Three hundred and twenty. Not one quark or gluon is left: every parton has married into a hadron... "
          "How does a string break?",
    "s5": "Six hundred and thirty-four. The hadrons decay; the B meson leaves nine grandchildren, and a neutrino "
          "walks away for good... When, and how far?",
    "s6": "Then the family photograph: anti-k-T draws twenty-seven circles, and only two hundred and thirty-nine "
          "of the six hundred and thirty-four are inside. The b jet carries more than the b quark... Why more?",
    "end": "From two to six hundred and thirty-four, and into twenty-seven jets. Every stage has its film... "
           "JitJet: Journey Is The Jet. Watch part one.",
}


# --- the pages ---------------------------------------------------------------------
def render_pages(pdf: Path, workdir: Path, dpi: float, n: int) -> List[Path]:
    """The pages of the PDF as PNGs (ghostscript), skipped when they exist already."""
    workdir.mkdir(parents=True, exist_ok=True)
    pngs = [workdir / f"page_{i:02d}_{dpi:g}.png" for i in range(1, n + 1)]
    if all(p.exists() for p in pngs):
        return pngs
    print(f"rendering {n} pages of {pdf.name} at {dpi:g} dpi ...", flush=True)
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", f"-r{dpi:g}", "-dTextAlphaBits=4",
                    "-dGraphicsAlphaBits=4", f"-sOutputFile={workdir / f'page_%02d_{dpi:g}.png'}", str(pdf)],
                   check=True)
    missing = [p for p in pngs if not p.exists()]
    if missing:
        sys.exit(f"ghostscript did not write {missing[0]}: has the PDF {n} pages?")
    return pngs


class Page:
    """One rendered page; view() cuts a 16:9 window out of it."""

    def __init__(self, png: Path):
        self.im = Image.open(png).convert("RGB")
        self.W0, self.H0 = self.im.size
        self.half = self.im.reduce(2)

    def view(self, cx: float, cy: float, w: float, size: Tuple[int, int]) -> Image.Image:
        W, H = size
        bw = min(w * self.W0, self.W0)
        bh = min(bw * H / W, self.H0)          # (rounding of the region's size must not push the box past the page)
        x0 = min(max(cx * self.W0 - bw / 2, 0.0), self.W0 - bw)
        y0 = min(max(cy * self.H0 - bh / 2, 0.0), max(self.H0 - bh, 0.0))
        if bw / W > 2.6:                       # far out: resize from the half-size copy
            return self.half.resize((W, H), Image.BICUBIC, box=(x0 / 2, y0 / 2, (x0 + bw) / 2, (y0 + bh) / 2))
        return self.im.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + bw, y0 + bh))


# --- the clock -----------------------------------------------------------------------
@dataclass
class Cue:
    stop: Stop
    t_in: float        # the page starts dissolving in
    t_hold: float      # the dissolve is complete; the counter counts, the card appears
    t_end: float       # the next page starts dissolving in (or the finale)


class Timeline:
    """The film clock. `lengths` are the narration clips in seconds (empty: the nominal timing)."""

    def __init__(self, stops: Sequence[Stop], lengths: Dict[str, float], t_title: float, hold: float, xfade: float,
                 finale: float, fade_out: float):
        d = lambda key: lengths.get(key, 0.0)  # noqa: E731
        self.voice: Dict[str, float] = {"title": 0.8}
        self.xfade = xfade
        self.t_first = max(t_title, 0.8 + d("title") + 0.4)     # the first page comes in after the question
        self.cues: List[Cue] = []
        t = self.t_first
        for st in stops:
            t_hold = t + xfade
            self.voice[st.key] = t_hold + 0.3
            length = max(hold, 0.3 + d(st.key) + 1.0)            # the stop lasts until the line has been said
            self.cues.append(Cue(st, t, t_hold, t_hold + length))
            t = t_hold + length
        self.t_finale = t
        self.voice["end"] = t + 1.6
        self.duration = t + max(finale, 1.6 + d("end") + 3.5)
        self.fade_in = 1.0
        self.fade_out = fade_out

    def cue_at(self, t: float) -> Optional[Cue]:
        """The cue whose page is (coming) on screen at t; None on the title card."""
        cur = None
        for cue in self.cues:
            if t >= cue.t_in:
                cur = cue
        return cur


# --- the overlays --------------------------------------------------------------------
def chain_width(items: Sequence[str], font, gap: float) -> float:
    h = font.size
    return sum(font.getlength(it) for it in items) + (len(items) - 1) * (2 * gap + 1.1 * h)


def draw_chain(d: ImageDraw.ImageDraw, xy: Tuple[float, float], items: Sequence[str], font, fill, gap: float) -> float:
    """`items` in a row with an arrow drawn between them (the fonts have no arrow glyph); returns the end x."""
    x, y = xy
    h = font.size
    for i, it in enumerate(items):
        d.text((x, y), it, font=font, fill=fill)
        x += font.getlength(it)
        if i < len(items) - 1:
            ym = y + 0.62 * h
            x0, x1 = x + gap, x + gap + 1.1 * h
            d.line((x0, ym, x1, ym), fill=fill, width=max(int(h / 12), 1))
            d.polygon([(x1, ym), (x1 - 0.38 * h, ym - 0.3 * h), (x1 - 0.38 * h, ym + 0.3 * h)], fill=fill)
            x = x1 + gap
    return x


class ChapterRuler(Ruler):
    ROMAN = [str(i) for i in range(1, N_CHAPTERS + 1)]


class Cards:
    """The cards of the sidebar (the counter at the top, the question and the chapter card at the bottom),
    the title card and the end card. The sidebar is the strip right of the page, `width` wide."""

    def __init__(self, size: Tuple[int, int], width: int, fonts: Fonts):
        self.W, self.H = size
        self.f, self.s = fonts, fonts.scale
        self.margin = int(18 * self.s)
        self.width = width - 2 * self.margin            # of a card
        self.chapter_cards = {st.key: self._chapter_card(st) for st in STOPS}
        self.questions = {st.key: self._question(st) for st in STOPS}
        self.title_card = self._title_card()
        self.end_card = self._end_card()

    def _panel(self, h: int) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        im = Image.new("RGBA", (self.width, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, self.width - 1, h - 1), radius=int(8 * self.s), fill=(*NAVYTOP, 255))
        d.rectangle((0, 0, int(5 * self.s), h - 1), fill=GOLD)
        return im, d

    def counter(self, value: float, unit: str, chain: Sequence[int]) -> Image.Image:
        """The counter card: the number (counting), what it counts, the chain of counts so far."""
        s = self.s
        f_big, f_unit, f_chain = self.f.get("bold", 96), self.f.get("regular", 20), self.f.get("medium", 17)
        pad, x0 = int(18 * s), int(18 * s) + int(8 * s)
        lines = wrap(unit, f_unit, self.width - x0 - pad)
        lh = int(26 * s)
        h = pad + int(100 * s) + lh * len(lines) + (int(28 * s) if chain else int(6 * s)) + pad // 2
        im, d = self._panel(h)
        d.text((x0 - int(4 * s), pad - int(8 * s)), f"{int(round(value))}", font=f_big, fill=GOLD)
        for i, ln in enumerate(lines):
            d.text((x0, pad + int(100 * s) + i * lh), ln, font=f_unit, fill=WHITE)
        if chain:
            draw_chain(d, (x0, pad + int(100 * s) + lh * len(lines) + int(6 * s)), [str(c) for c in chain],
                       f_chain, PALE, 6 * s)
        return im

    def _chapter_card(self, st: Stop) -> Image.Image:
        s = self.s
        f_kick, f_title, f_sub, f_small = (self.f.get("medium", 19), self.f.get("bold", 30), self.f.get("regular", 18),
                                           self.f.get("italic", 15))
        pad, x0 = int(18 * s), int(18 * s) + int(8 * s)
        width = self.width - x0 - pad
        title, sub = wrap(st.title, f_title, width), wrap(st.sub, f_sub, width)
        lt, ls = int(36 * s), int(24 * s)
        y_title = pad + int(26 * s)
        y_sub = y_title + lt * len(title) + int(4 * s)
        y_small = y_sub + ls * len(sub) + int(8 * s)
        im, d = self._panel(y_small + int(22 * s) + pad // 2)
        d.text((x0, pad), st.kicker, font=f_kick, fill=GOLD)
        for i, ln in enumerate(title):
            d.text((x0, y_title + i * lt), ln, font=f_title, fill=WHITE)
        for i, ln in enumerate(sub):
            d.text((x0, y_sub + i * ls), ln, font=f_sub, fill=PALE)
        d.text((x0, y_small), "watch the film  ·  read the chapter", font=f_small, fill=GOLD)
        return im

    def _question(self, st: Stop) -> Image.Image:
        """The question of the stop, in gold italics."""
        s = self.s
        f_q = self.f.get("italic", 23)
        pad, x0 = int(18 * s), int(18 * s) + int(8 * s)
        lines = wrap(st.question, f_q, self.width - x0 - pad)
        lh = int(30 * s)
        h = pad // 2 + lh * len(lines) + pad // 2
        im = Image.new("RGBA", (self.width, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, self.width - 1, h - 1), radius=int(8 * s), fill=(*NAVYTOP, 160))
        for i, ln in enumerate(lines):
            d.text((x0, pad // 2 + i * lh), ln, font=f_q, fill=GOLD)
        return im

    def _title_card(self) -> Image.Image:
        s = self.s
        im = Image.new("RGB", (self.W, self.H), NAVY)
        d = ImageDraw.Draw(im)
        f_big, f_mid, f_it, f_small = (self.f.get("bold", 170), self.f.get("regular", 50), self.f.get("italic", 34),
                                       self.f.get("regular", 24))
        cx, cy = self.W // 2, int(self.H * 0.42)
        wj, wjet = f_big.getlength("Jit"), f_big.getlength("Jet")
        x = cx - (wj + wjet) / 2
        d.text((x, cy), "Jit", font=f_big, fill=WHITE, anchor="ls")
        d.text((x + wj, cy), "Jet", font=f_big, fill=GOLD, anchor="ls")
        d.line((cx - int(260 * s), cy + int(30 * s), cx + int(260 * s), cy + int(30 * s)), fill=GOLD,
               width=max(int(2 * s), 1))
        d.text((cx, cy + int(62 * s)), "Part I  ·  The Family Is Born", font=f_mid, fill=WHITE, anchor="ma")
        d.text((cx, cy + int(136 * s)), "one simulated event  ·  seven chapters  ·  seven films", font=f_it,
               fill=GOLD, anchor="ma")
        d.text((cx, self.H - int(110 * s)), "Ravindra Verma   ·   University of Helsinki   ·   CMS Collaboration",
               font=f_small, fill=PALE, anchor="ma")
        return im

    def _end_card(self) -> Image.Image:
        s = self.s
        im = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, self.W, self.H), fill=(*NAVY, 225))
        f_big, f_mid, f_it = self.f.get("bold", 140), self.f.get("regular", 46), self.f.get("italic", 32)
        f_chain = self.f.get("medium", 30)
        cx, cy = self.W // 2, int(self.H * 0.40)
        wj, wjet = f_big.getlength("Jit"), f_big.getlength("Jet")
        x = cx - (wj + wjet) / 2
        d.text((x, cy), "Jit", font=f_big, fill=WHITE, anchor="ls")
        d.text((x + wj, cy), "Jet", font=f_big, fill=GOLD, anchor="ls")
        d.line((cx - int(250 * s), cy + int(26 * s), cx + int(250 * s), cy + int(26 * s)), fill=GOLD,
               width=max(int(2 * s), 1))
        d.text((cx, cy + int(56 * s)), "Journey Is The Jet", font=f_mid, fill=WHITE, anchor="ma")
        items = [str(c) for c in CHAIN] + ["27 jets"]
        draw_chain(d, (cx - chain_width(items, f_chain, 10 * s) / 2, cy + int(126 * s)), items, f_chain, GOLD, 10 * s)
        d.text((cx, cy + int(182 * s)), "Part I, chapters 1 to 7: one film and one chapter per stage", font=f_it,
               fill=PALE, anchor="ma")
        return im


# --- rendering --------------------------------------------------------------------------
class Renderer:
    """The page fills the height on the left, whole (no label of it is cut); the cards live in the sidebar
    on the right; the ruler runs along the bottom."""

    def __init__(self, pages: Dict[int, Page], tl: Timeline, size: Tuple[int, int]):
        self.pages, self.tl, self.size = pages, tl, size
        W, H = size
        first = next(iter(pages.values()))
        self.page_w = int(round(H * first.W0 / first.H0))         # the page region, the sidebar is the rest
        self.fonts = Fonts(H / 1080.0)
        self.cards = Cards(size, W - self.page_w, self.fonts)
        self.ruler = ChapterRuler(size, self.fonts)
        self.ruler.x0, self.ruler.x1 = int(0.08 * self.page_w), int(0.92 * self.page_w)   # along the page only
        self.canvas = Image.new("RGB", size, NAVY)
        ImageDraw.Draw(self.canvas).rectangle((self.page_w, 0, self.page_w + int(3 * self.fonts.scale), H), fill=GOLD)

    def camera(self, cue: Cue, t: float) -> Tuple[float, float, float]:
        a, b = cue.stop.shot
        u = clamp01((t - cue.t_in) / max(cue.t_end - cue.t_in, 1e-6))
        return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] * (b[2] / a[2]) ** u)

    def page_view(self, cue: Cue, t: float) -> Image.Image:
        """The whole frame with the page in its region (the sidebar empty)."""
        cx, cy, w = self.camera(cue, t)
        frame = self.canvas.copy()
        frame.paste(self.pages[cue.stop.page].view(cx, cy, w, (self.page_w, self.size[1])), (0, 0))
        return frame

    def frame(self, t: float) -> Image.Image:
        tl, cards = self.tl, self.cards
        cue = tl.cue_at(t)
        if cue is None:
            base = cards.title_card.copy()
        else:
            base = self.page_view(cue, t)
            u = clamp01((t - cue.t_in) / tl.xfade)
            if u < 1.0:                                     # dissolving in from what was there before
                i = tl.cues.index(cue)
                prev = cards.title_card if i == 0 else self.page_view(tl.cues[i - 1], t)
                base = Image.blend(prev, base, smooth(u))
        frame = base.convert("RGBA")
        a_end = clamp01((t - tl.t_finale - 0.3) / 1.5)
        x_side = self.page_w + int(3 * self.fonts.scale) + cards.margin
        if cue is not None:
            i = tl.cues.index(cue)
            st = cue.stop
            # the counter: counts from the previous value from the middle of the dissolve, 1.3 s
            prev_val = tl.cues[i - 1].stop.count if i > 0 else 0
            t_count = cue.t_in + 0.5 * tl.xfade
            v = prev_val + (st.count - prev_val) * smooth(clamp01((t - t_count) / 1.3))
            if t < t_count:                                 # still the previous stop's card
                v, unit, chain = prev_val, (tl.cues[i - 1].stop.unit if i > 0 else st.unit), CHAIN[:max(i - 1, 0)]
            else:
                unit, chain = st.unit, CHAIN[:i]            # the counts so far (all five under the jets)
            a_counter = clamp01((t - tl.t_first - 0.5) / 0.8) * (1.0 - a_end)
            if a_counter > 0:
                alpha_paste(frame, cards.counter(v, unit, chain), (x_side, cards.margin), a_counter)
            # the chapter card at the bottom of the sidebar, its question above it
            for c in tl.cues:
                a = min(clamp01((t - c.t_hold - 0.3) / 0.5), 1.0 - clamp01((t - c.t_end) / 0.5)) * (1.0 - a_end)
                if a > 0:
                    cc = cards.chapter_cards[c.stop.key]
                    y = self.size[1] - self.ruler.strip_h - cc.height
                    alpha_paste(frame, cc, (x_side, y), a)
                    q = cards.questions[c.stop.key]
                    aq = min(clamp01((t - c.t_hold - 2.2) / 0.6), 1.0 - clamp01((t - c.t_end) / 0.5)) * (1.0 - a_end)
                    if aq > 0:
                        alpha_paste(frame, q, (x_side, y - q.height - int(10 * self.fonts.scale)), aq)
            # the ruler: the chapters lit as their stops pass
            lit = max(st.chapters)
            done = (lit - 1) / (N_CHAPTERS - 1) + (1.0 / (N_CHAPTERS - 1)) * (
                0.0 if lit >= N_CHAPTERS else clamp01((t - cue.t_hold) / (cue.t_end - cue.t_hold)) * 0.6)
            a_ruler = clamp01((t - tl.t_first - 1.0) / 1.0) * (1.0 - a_end)
            if a_ruler > 0:
                self.ruler.draw(frame, min(done, 1.0), lit, a_ruler)
        if a_end > 0:
            alpha_paste(frame, cards.end_card, (0, 0), a_end)
        g = min(clamp01(t / tl.fade_in), clamp01((tl.duration - t) / tl.fade_out))
        if g < 1.0:
            arr = (np.asarray(frame.convert("RGB")).astype(np.float32) * g).astype(np.uint8)
            return Image.fromarray(arr)
        return frame.convert("RGB")


# --- the score ------------------------------------------------------------------------
def score_spec(tl: Timeline) -> snd.ScoreSpec:
    """A chord per page, the pulse gaining a voice with every stop, a bell as each page settles, a thud
    at the collision, the end card resolving to D minor."""
    progression = [snd.Dm, snd.Bb, snd.F, snd.C, snd.Gm, snd.Bb, snd.F, snd.A]
    spec = snd.ScoreSpec(tl.duration, tl.fade_out,
                         chords=[(0.0, snd.Dm)] + [(cue.t_in, ch) for cue, ch in zip(tl.cues, progression)]
                                + [(tl.t_finale, snd.Dm_hi)],
                         thuds=[tl.t_first + 0.5 * tl.xfade], outro=(tl.t_finale, [f * 2 for f in snd.Dm_hi]))
    for i, cue in enumerate(tl.cues):
        spec.pulses.append(snd.pulse_for(min(i, 6), cue.t_in, cue.t_hold + 1.5, cue.t_end, progression[i]))
        spec.bells.append(snd.Bell(cue.t_hold, progression[i][0] * 4))
    spec.bells.append(snd.Bell(tl.t_finale + 1.0, snd.D4, gain=0.55, seconds=6.0, spread=0.015))
    return spec


# --- output -------------------------------------------------------------------------------
def write_video(rend: Renderer, duration: float, audio: Optional[Path], out: Path, fps: int, ffmpeg: str, preset: str,
                crf: int, clip: Optional[Tuple[float, float]] = None) -> None:
    """The frames piped to ffmpeg (H.264 at the given crf, tuned for still images: the pages barely move) with the
    WAV under them; `clip` renders only that stretch of the clock (for checking)."""
    W, H = rend.size
    t0, t1 = clip if clip else (0.0, duration)
    n = int(math.ceil((t1 - t0) * fps))
    cmd = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    if audio:
        cmd += ["-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", str(audio)]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-tune", "stillimage", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "160k", "-shortest"]
    cmd += ["-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for i in range(n):
        proc.stdin.write(rend.frame(t0 + i / fps).tobytes())
        if i % (10 * fps) == 0:
            print(f"  frame {i}/{n}  t = {t0 + i / fps:6.1f} s", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")


def write_stills(rend: Renderer, tl: Timeline, out: Path) -> None:
    """One frame per stop (title, s1 ... s6, end): a PDF stack or numbered PNGs."""
    times = [(0.6 * tl.t_first, "title")] + [(c.t_hold + 3.2, c.stop.key) for c in tl.cues] + \
            [(tl.duration - tl.fade_out - 0.5, "end")]
    ims = [rend.frame(t) for t, _ in times]
    if out.suffix.lower() == ".pdf":
        ims[0].save(out, save_all=True, append_images=ims[1:], resolution=96)
    else:
        for (_, key), im in zip(times, ims):
            im.save(out.with_name(f"{out.stem}_{key}{out.suffix}"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", type=Path, default=HERE / "animate_trailer_part1_ch01_07.mp4")
    p.add_argument("--pdf", type=Path, default=PDF, help="the CMSSW stack; its first six pages are shown")
    p.add_argument("--size", default="1920x1080")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dpi", type=float, default=300, help="resolution of the rendered pages")
    p.add_argument("--hold", type=float, default=6.0, help="minimum seconds per page (longer if the line is)")
    p.add_argument("--xfade", type=float, default=1.0, help="seconds of dissolve between pages")
    p.add_argument("--title", type=float, default=5.0, help="minimum seconds on the title card")
    p.add_argument("--finale", type=float, default=9.0, help="minimum seconds on the end card")
    snd.add_narration_args(p)             # --engine --voice --rate --narration-dir --no-narration --print-script --music-db --no-music
    p.add_argument("--work-dir", type=Path, default=ROOT / "build" / "trailer_frames",
                   help="the rendered pages are kept here and re-used (make clean removes them)")
    p.add_argument("--stills", type=Path, default=None, help="one key frame per stop (.pdf stack or .png files)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--audio-only", type=Path, default=None, metavar="WAV",
                   help="write just the soundtrack (narration + score) and exit, without rendering the pages")
    p.add_argument("--crf", type=int, default=26, help="H.264 quality (lower is better and bigger; 26 gives ~12 MB)")
    p.add_argument("--clip", type=float, nargs=2, default=None, metavar=("T0", "T1"),
                   help="render only this stretch of the clock, for checking a stop")
    p.add_argument("--fast", action="store_true", help="preview: 960x540, 12 fps, 150 dpi")
    args = p.parse_args()
    if args.print_script:
        snd.print_script(NARRATION)
        return
    if args.rate is None and args.engine == "edge":
        args.rate = -3                    # a touch brisker than the chapter films (-6): a trailer
    snd.narration_defaults(args)
    if args.fast:
        args.size, args.fps, args.dpi = "960x540", 12, 150.0
    W, H = (int(v) for v in args.size.lower().split("x"))
    ffmpeg = shutil.which("ffmpeg")
    for tool in ("gs", "ffmpeg"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found on PATH")

    print("narration ..." if not args.no_narration else "no narration")
    clips = snd.load_narration(NARRATION, args, ROOT / "build" / "narration" / "trailer", ffmpeg)
    tl = Timeline(STOPS, snd.clip_lengths(clips), args.title, args.hold, args.xfade, args.finale, fade_out=2.0)
    print(f"film {tl.duration:.1f} s: title to {tl.t_first:.1f} s, "
          + ", ".join(f"{c.stop.key} {c.t_in:.1f}-{c.t_end:.1f}" for c in tl.cues) + f", end card from {tl.t_finale:.1f} s")
    score = None if args.no_music else score_spec(tl)
    if args.audio_only:
        snd.mix_audio(tl.duration, tl.voice, clips, score, args.music_db, args.audio_only)
        print(f"wrote {args.audio_only}")
        return

    pngs = render_pages(args.pdf, args.work_dir, args.dpi, max(st.page for st in STOPS))
    pages = {st.page: Page(pngs[st.page - 1]) for st in STOPS}
    rend = Renderer(pages, tl, (W, H))
    if args.stills:
        write_stills(rend, tl, args.stills)
        print(f"wrote {args.stills}")
    if args.no_video:
        return
    tmp = tempfile.TemporaryDirectory(prefix="jitjet_trailer_")
    audio = Path(tmp.name) / "soundtrack.wav"
    snd.mix_audio(tl.duration, tl.voice, clips, score, args.music_db, audio)
    if clips:
        snd.write_srt(tl.voice, clips, NARRATION, args.output.with_suffix(".srt"))
    print(f"film: {W}x{H} at {args.fps} fps ...")
    write_video(rend, tl.duration, audio, args.output, args.fps, ffmpeg, "ultrafast" if args.fast else "slow",
                args.crf, tuple(args.clip) if args.clip else None)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
