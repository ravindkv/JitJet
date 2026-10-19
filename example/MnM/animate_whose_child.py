#!/usr/bin/env python3
"""The film of chapter 3, "Whose Child Is It? Matching and Merging": a narrated minute.

Chapter 3 is a stop on the road at which our event does not get off. The clip shows the
road of the book with its stops (chapter 1, the ancestral home; chapter 2, the mother
gives birth; chapter 3, whose child is it; chapter 4, cousins nobody invited), the cart
carrying our event, u ubar -> b bbar, rolling up to stop 3, the sign that says who this
stop is for (events with an extra-parton subprocess, b bbar + 1, 2 partons, or a
next-to-leading-order matrix element: MLM, CKKW-L, MC@NLO, POWHEG, FxFx), the passengers
who do get off here (ttbar+jets, W/Z+jets, multijet samples), and the card that says why
we do not: one leading-order 2->2 process, no second parent for any parton, nothing to
match. The ledger row repeats the previous one, 100.45 GeV with Delta = 0, and the cart
rolls on to stop 4. The numbers are the ones typed in the chapter's ledger.

Three beats on one continuous camera pan, drawn against a 12.5-second script clock:

  approach   0.0 - 4.7 s   the cart arrives at stop 3 and the sign unfolds (0-3 s); the
                           passengers of other samples gather on the platform (3.5-4.7 s)
  verdict    4.7 - 8.5 s   the card for our event (4.7 s), the unchanged ledger row
                           (6.1 s) and the stamp "not for us" (7.3 s)
  move on    8.5 - 12.5 s  the cart accelerates past the stop and the camera pans to
                           stop 4

Narration and music
  Each beat is a few phases (the cart arrives, the sign unfolds, the passengers get off,
  ...) and each phase has one line of narration (narration() below), spoken by a neural
  voice through edge-tts (Microsoft's online voices; `pip install --user edge-tts`, needs
  the network) or, with --engine say, by a macOS voice; --narration-dir takes recorded
  clips named <key>.wav|mp3 (--print-script lists the keys and the text). A phase lasts at
  least as long as its line: its motion is slowed down (up to 2.5x) and then held, so the
  narrated film is about two minutes and the silent one (--no-narration) the 12.5 s of
  the script clock; the phases end where the picture rests (the cart stopped, the sign up,
  the card shown, the end frame), so a hold never freezes a moving cart. The score comes from example/soundtrack.py: a drone, a pad with one
  chord per beat, a pulse, a bell as each beat opens, a thud when the stamp lands. The
  clips are cached in build/narration/MnM/. An .srt is written next to the .mp4.

Examples (matplotlib is needed; on this Mac use /usr/bin/python3):
  python3 animate_whose_child.py                   # -> animate_whose_child.mp4 + .srt next to this file
  python3 animate_whose_child.py --fast            # quick low-resolution preview
  python3 animate_whose_child.py --stills animate_whose_child_stills.pdf --no-video
  python3 animate_whose_child.py --print-script    # the narration, for recording it yourself
  python3 animate_whose_child.py --audio-only x.wav   # the soundtrack alone
  python3 animate_whose_child.py --frames-dir frames --step 12 --dpi 40

An .mp4 needs ffmpeg on the PATH; a .gif needs only Pillow and is silent.
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import soundtrack as snd  # noqa: E402  (example/soundtrack.py: the voice, the score, the film clock)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

FIGSIZE = (12.8, 7.2)
# the script clock (seconds): the beats below are drawn against it
T_ARRIVE, T_PASS, T_CARD, T_LEDGER, T_STAMP, T_GO, DURATION = 3.0, 3.5, 4.7, 6.1, 7.3, 8.5, 12.5
FPS = snd.NOMINAL_FPS             # one frame index per 1/24 s of the script clock

# colours shared with the other films of the book
COL_B, COL_BBAR = "#ff0000", "#0000ff"
COL_ISR, COL_FSR = "#ff7f00", "#009900"
COL_ME, COL_PS = "#2a5db0", "#c8780a"
COL_ROAD, COL_STOP, COL_NEXT = "#444444", "#1f1f1f", "#2f8f5b"
COL_STAMP = "#c0392b"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d6d3ca"

# the stops of the road (road units; the camera window is 24 units wide)
STOPS = [
    (0.0, "1", "The Ancestral Home", "hard scattering"),
    (10.0, "2", "The Mother Gives Birth", "parton shower"),
    (20.0, "3", "Whose Child Is It?", "matching and merging"),
    (30.0, "4", "Cousins Nobody Invited", "MPI, colour reconnection"),
    (40.0, "5", "The Children Get Married", "hadronisation"),
]
X3, X4 = STOPS[2][0], STOPS[3][0]

# the ledger row of chapter 3 (typed in chapter/chapter_03_whose_child.tex)
LEDGER_PT, LEDGER_M, LEDGER_N = 100.45, 13.52, 3

# who does get off at this stop
PASSENGERS = [
    (r"$t\bar t$+jets", "FxFx"),
    (r"$W/Z$+jets", "MLM"),
    (r"$t\bar t$, single top", "POWHEG"),
]


def smooth(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ramp(t: float, t0: float, t1: float) -> float:
    return smooth((t - t0) / max(t1 - t0, 1e-9))


def cart_x(t: float) -> float:
    """Road position of our event's cart against time (seconds)."""
    # roll in from stop 2's side, brake in front of the sign, pause, then accelerate past
    x_arrive = X3 - 3.4
    if t < T_ARRIVE:
        return 13.6 + (x_arrive - 13.6) * (1 - (1 - min(t / T_ARRIVE, 1.0)) ** 2.2)
    if t < T_GO:
        return x_arrive
    return x_arrive + (X4 - 3.8 - x_arrive) * smooth((t - T_GO) / 3.0)


def camera_x(t: float) -> float:
    """Centre of the camera window."""
    if t < T_GO:
        return 10.5 + (X3 - 1.0 - 10.5) * ramp(t, 0.0, 2.6)
    return (X3 - 1.0) + (X4 - 2.0 - (X3 - 1.0)) * ramp(t, T_GO + 0.4, T_GO + 3.0)


# --- drawing --------------------------------------------------------------------------------------------
def gluon(ax, x0, y0, x1, y1, color, n=5, amp=0.12, lw=1.4, alpha=1.0):
    s = np.linspace(0, 1, 60)
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    wob = amp * np.sin(2 * math.pi * n * s)
    ax.plot(x0 + dx * s + nx * wob, y0 + dy * s + ny * wob, color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=6)


def draw_road(ax, t: float):
    ax.axhline(0, color=COL_ROAD, lw=5, zorder=1)
    for x in np.arange(-20, 70, 1.6):
        ax.plot([x, x + 0.8], [0, 0], color="white", lw=1.6, zorder=2)
    for xs, num, title, sub in STOPS:
        here = abs(xs - X3) < 1e-6
        nxt = abs(xs - X4) < 1e-6
        col = COL_NEXT if nxt and t > T_GO + 0.4 else COL_STOP
        ax.add_patch(Circle((xs, 0), 0.55, facecolor="white", edgecolor=col, lw=3, zorder=3))
        ax.text(xs, 0, num, ha="center", va="center", fontsize=15, weight="bold", color=col, zorder=4)
        ax.plot([xs, xs], [0.55, 1.6], color=col, lw=2.5, zorder=3)
        ax.text(xs, 1.75, title, ha="center", va="bottom", fontsize=12 if here else 10.5, weight="bold" if here else "normal", color=col, zorder=4)
        ax.text(xs, -0.95, sub, ha="center", va="top", fontsize=9, color=TEXT2, style="italic", zorder=4)


def draw_cart(ax, x: float, t: float, speed: float):
    """Our event as a little cart: the b and bbar with their gluons as passengers."""
    y = 0.95
    body = FancyBboxPatch((x - 1.5, y - 0.45), 3.0, 1.5, boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor="#fff7e6", edgecolor=TEXT, lw=1.8, zorder=7)
    ax.add_patch(body)
    for wx in (x - 0.95, x + 0.95):
        ax.add_patch(Circle((wx, y - 0.62), 0.26, facecolor="#333333", edgecolor="black", lw=1.2, zorder=8))
        # spokes turn with the speed
        ang = -speed * t * 2.2
        for k in range(3):
            a = ang + k * math.pi / 3
            ax.plot([wx - 0.2 * math.cos(a), wx + 0.2 * math.cos(a)], [y - 0.62 - 0.2 * math.sin(a), y - 0.62 + 0.2 * math.sin(a)],
                    color="#bbbbbb", lw=1.2, zorder=9)
    # passengers: b (red), bbar (blue), ISR gluon, FSR gluons
    ax.plot([x - 0.9, x - 0.25], [y + 0.15, y + 0.75], color=COL_B, lw=2.6, zorder=8, solid_capstyle="round")
    ax.text(x - 0.95, y + 0.78, "$b$", color=COL_B, fontsize=11, weight="bold", ha="center", va="bottom", zorder=9)
    ax.plot([x + 0.9, x + 0.25], [y + 0.15, y + 0.75], color=COL_BBAR, lw=2.6, zorder=8, solid_capstyle="round")
    ax.text(x + 0.95, y + 0.78, r"$\bar b$", color=COL_BBAR, fontsize=11, weight="bold", ha="center", va="bottom", zorder=9)
    gluon(ax, x - 0.6, y + 0.45, x - 1.25, y + 0.95, COL_FSR, n=4, amp=0.07)
    gluon(ax, x + 0.6, y + 0.45, x + 1.25, y + 0.95, COL_FSR, n=4, amp=0.07)
    gluon(ax, x, y + 0.1, x, y + 0.95, COL_ISR, n=5, amp=0.07)
    ax.text(x, y - 0.2, r"our event: $u\bar u\to b\bar b$", ha="center", va="center", fontsize=9.5, color=TEXT, zorder=9)
    if speed > 0.5:  # motion lines
        for k, dy in enumerate((0.3, 0.75, 1.2)):
            ax.plot([x - 1.9 - 0.3 * k, x - 1.6 - 0.1 * k - 0.6 * min(speed, 3) / 3], [y + dy, y + dy], color=TEXT2, lw=1.2, alpha=0.6, zorder=6)


def draw_sign(ax, t: float):
    """The sign at stop 3: who this stop is for. Unfolds between 1.2 and 2.2 s."""
    a = ramp(t, 1.2, 2.2)
    if a <= 0:
        return
    xs = X3
    # the station canopy with the title
    w, h = 9.8 * a, 3.6
    x0, y0 = xs - 1.4, 2.9
    xs = x0 + w / 2
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.05,rounding_size=0.25",
                                facecolor="#eef3fb", edgecolor=COL_ME, lw=2, zorder=5, alpha=a))
    if a < 0.98:
        return
    ax.text(xs, y0 + h - 0.35, "This stop is for events with", ha="center", va="top", fontsize=11.5, color=TEXT2, zorder=6)
    ax.text(xs, y0 + h - 0.95, "a second parent for the extra parton", ha="center", va="top", fontsize=13, weight="bold", color=COL_ME, zorder=6)
    lines = [
        r"$b\bar b+1,\,2$-parton subprocesses:  MLM, CKKW-L merging",
        r"a next-to-leading-order matrix element:  MC@NLO, POWHEG matching",
        r"both at once:  FxFx",
    ]
    for i, line in enumerate(lines):
        yy = y0 + h - 1.65 - 0.62 * i
        ax.text(x0 + 0.35, yy, "• " + line, ha="left", va="center", fontsize=10.5, color=TEXT, zorder=6)


def draw_passengers(ax, t: float):
    """The samples that do get off here walk onto the platform from T_PASS on and into the
    station as our cart starts to roll past."""
    gone = ramp(t, T_GO, T_GO + 0.8)
    for i, (name, recipe) in enumerate(PASSENGERS):
        a = ramp(t, T_PASS + 0.25 * i, T_PASS + 0.7 + 0.25 * i) * (1 - gone)
        if a <= 0:
            continue
        x = X3 + 2.6 + 1.6 * i
        y = 0.25 + 1.0 * a
        ax.add_patch(Circle((x, y + 0.75), 0.22, facecolor=COL_ME, edgecolor="none", zorder=7, alpha=a))
        ax.add_patch(Rectangle((x - 0.3, y), 0.6, 0.6, facecolor=COL_ME, edgecolor="none", zorder=7, alpha=a))
        ax.text(x, y - 0.12, name, ha="center", va="top", fontsize=8.5, color=COL_ME, zorder=8, alpha=a)
        ax.text(x, y - 0.5, recipe, ha="center", va="top", fontsize=7.5, color=TEXT2, style="italic", zorder=8, alpha=a)
    if t > T_PASS + 1.0 and gone < 1:
        ax.text(X3 + 4.2, 2.35, "they get off here", ha="center", va="bottom", fontsize=9.5, color=COL_ME, style="italic", zorder=8, alpha=1 - gone)


def draw_verdict(ax, cx: float, t: float):
    """The card for our event: one 2->2 LO process, nothing to match; the ledger row repeats."""
    a = ramp(t, T_CARD, T_CARD + 0.6)
    if a <= 0:
        return
    fade = 1.0 - ramp(t, T_GO + 0.6, T_GO + 1.4)
    if fade <= 0:
        return
    al = a * fade
    x0, y0, w, h = cx - 11.6, 4.5, 10.3, 5.6
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.05,rounding_size=0.25",
                                facecolor="#fff7e6", edgecolor=COL_PS, lw=2, zorder=10, alpha=al))
    ax.text(x0 + 0.35, y0 + h - 0.35, "Our event", ha="left", va="top", fontsize=13, weight="bold", color=COL_PS, zorder=11, alpha=al)
    rows = [
        r"one leading-order $2\to2$ process: $u\bar u\to b\bar b$",
        r"no $u\bar u\to b\bar b g$, no $ug\to b\bar b u$ in the sample",
        "every extra parton has one parent, the shower",
        "nothing to match, nothing to merge",
    ]
    for i, r in enumerate(rows):
        ax.text(x0 + 0.35, y0 + h - 1.0 - 0.6 * i, "• " + r, ha="left", va="top", fontsize=10.5, color=TEXT, zorder=11, alpha=al)
    # the ledger row, unchanged
    b = ramp(t, T_LEDGER, T_LEDGER + 0.6) * fade
    if b > 0:
        ly = y0 + 0.55
        ax.plot([x0 + 0.35, x0 + w - 0.35], [ly + 1.05, ly + 1.05], color=GRID, lw=1, zorder=11, alpha=b)
        ax.text(x0 + 0.35, ly + 0.8, "ledger", ha="left", va="center", fontsize=9, color=TEXT2, zorder=11, alpha=b)
        ax.text(x0 + 0.35, ly + 0.35, r"$b$ family after the shower", ha="left", va="center", fontsize=10, color=TEXT, zorder=11, alpha=b)
        ax.text(x0 + w - 0.35, ly + 0.35, rf"{LEDGER_PT:.2f} GeV,  $m={LEDGER_M:.2f}$ GeV,  {LEDGER_N} partons", ha="right", va="center", fontsize=10, color=TEXT, zorder=11, alpha=b)
        c = ramp(t, T_LEDGER + 0.6, T_LEDGER + 1.2) * fade
        if c > 0:
            ax.text(x0 + 0.35, ly - 0.1, "after matching and merging (none applied)", ha="left", va="center", fontsize=10, color=TEXT, zorder=11, alpha=c)
            ax.text(x0 + w - 0.35, ly - 0.1, rf"{LEDGER_PT:.2f} GeV,  $\Delta p_T = 0$", ha="right", va="center", fontsize=10, weight="bold", color=COL_NEXT, zorder=11, alpha=c)
    # the stamp
    s = ramp(t, T_STAMP, T_STAMP + 0.3)
    if s > 0:
        scale = 1.6 - 0.6 * s
        ax.text(x0 + w - 2.5, y0 + h - 2.0, "NOT FOR US", ha="center", va="center", rotation=-14, fontsize=19 * scale, weight="bold",
                color=COL_STAMP, alpha=s * fade * 0.9, zorder=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="none", edgecolor=COL_STAMP, lw=2.5 * scale, alpha=s * fade * 0.9))


def draw_next(ax, cx: float, t: float):
    a = ramp(t, T_GO + 1.0, T_GO + 1.8)
    if a <= 0:
        return
    ax.text(cx + 6.0, 8.6, "no stop here; on to the next one", ha="center", va="center", fontsize=14, weight="bold", color=COL_NEXT, alpha=a, zorder=11)
    ax.annotate("", xy=(cx + 10.2, 7.7), xytext=(cx + 1.8, 7.7), arrowprops=dict(arrowstyle="-|>", lw=2.5, color=COL_NEXT, alpha=a), zorder=11)


def caption(t: float) -> str:
    if t < T_PASS:
        return "Chapter 3 is a stop on the road: the sign says who it is for."
    if t < T_GO:
        return "Our event has one parent for every parton. There is nothing to match here, and the ledger row repeats."
    return "So we roll on: chapter 4, the rest of the two protons."


def draw_frame(fig, t: float, t_film: Optional[float] = None):
    """The picture at script time t; the stamp in the corner shows t_film (the narrated clock) when given."""
    fig.clf()
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    cx = camera_x(t)
    ax.set_xlim(cx - 12, cx + 12)
    ax.set_ylim(-2.0, 11.5)
    draw_road(ax, t)
    draw_sign(ax, t)
    draw_passengers(ax, t)
    x = cart_x(t)
    speed = abs(cart_x(t + 1 / 24) - x) * 24
    draw_cart(ax, x, t, speed)
    draw_verdict(ax, cx, t)
    draw_next(ax, cx, t)
    fig.text(0.02, 0.965, "Stop 3: Whose Child Is It? Matching and merging", fontsize=15, weight="bold", color=TEXT, va="center")
    fig.text(0.98, 0.965, "chapter_03_whose_child.tex", fontsize=9.5, color=TEXT2, ha="right", va="center")
    fig.text(0.02, 0.03, caption(t), fontsize=10.5, color=TEXT, va="bottom")
    fig.text(0.98, 0.03, f"{(t if t_film is None else t_film):4.1f} s", fontsize=9, color=TEXT2, ha="right", va="bottom")


KEYFRAMES = [2.9, 8.0, 12.0]  # one per beat


# --- the beats as scenes, their phases and the narration -------------------------------------------------
class Beat:
    """A beat of the script clock, seconds [t0, t1), as a scene of nframes frames for the film clock."""

    def __init__(self, number: int, name: str, t0: float, t1: float, phase_times: List[Tuple[str, float]]):
        self.number, self.name, self.t0 = number, name, t0
        self.k0 = int(round(t0 * FPS))
        self.nframes = int(round(t1 * FPS)) - self.k0
        self._phases = phase_times                     # (key, script time at which the phase starts)

    def phases(self) -> List[Tuple[str, int, int]]:
        ks = [int(round(t * FPS)) - self.k0 for _, t in self._phases] + [self.nframes]
        return [(key, ks[i], ks[i + 1]) for i, (key, _) in enumerate(self._phases)]

    def draw(self, fig, k: float, t_film: Optional[float] = None):
        draw_frame(fig, (self.k0 + k) / FPS, t_film)

    def keyframe(self) -> float:
        return next(t for t in KEYFRAMES if self.t0 <= t < self.t0 + self.nframes / FPS) * FPS - self.k0


BEATS = [
    Beat(1, "approach", 0.0, T_CARD, [("arrive", 0.0), ("sign", T_ARRIVE), ("passengers", T_PASS)]),
    Beat(2, "verdict", T_CARD, T_GO, [("card", T_CARD), ("ledger", T_LEDGER), ("stamp", T_STAMP)]),
    Beat(3, "moveon", T_GO, DURATION, [("roll", T_GO), ("next", T_GO + 3.0)]),
]
NAMES = {b.number: b.name for b in BEATS}


def narration() -> Dict[str, str]:
    """One line per phase, keyed <beat>_<phase>, plus the closing line; the numbers are the ledger's."""
    return {
        "approach_arrive": ("Stop three. Our event rolls up to the platform: an up quark and an anti-up quark that became "
                            "a bottom pair, with the gluons of the shower on board."),
        "approach_sign": ("The sign says who this stop is for. Events with a second parent for an extra parton... "
                          "Merging, M L M or C K K W L, when the sample also has subprocesses with one or two more partons. "
                          "Matching, M C at N L O or POWHEG, when the matrix element is next-to-leading order. "
                          "F x F x, for both at once."),
        "approach_passengers": ("Top pairs with jets. W and Z with jets. Single top... They get off here. Their samples "
                                "have two ways of making the same parton, and someone has to decide whose child it is."),
        "verdict_card": ("Our event stays on the cart. One leading-order two-to-two process, and no sample beside it with "
                         "an extra parton from the matrix element. Every parton beyond the b and the anti-b has exactly "
                         "one parent, the shower... Nothing to match. Nothing to merge."),
        "verdict_ledger": (f"The ledger row repeats the previous one: the b family after the shower, {LEDGER_PT:.2f} GeV, "
                           f"{LEDGER_N} partons... after matching and merging, the same, and a change of zero."),
        "verdict_stamp": "Not for us.",
        "moveon_roll": "So the cart rolls on, past the platform and past the passengers walking into the station.",
        "moveon_next": ("Stop four. The cousins nobody invited... the rest of the two protons, which have been waiting all "
                        "this time to join the party."),
        "outro": "No stop here. Next stop: the underlying event.",
    }


def score_spec(film: snd.Film) -> snd.ScoreSpec:
    """D minor, G minor and the dominant A for the three beats, resolving under the outro; a thud as the stamp lands."""
    t_stamp = snd.voice_at_phase(film, "verdict_stamp")
    return snd.film_score(film, [snd.Dm, snd.Gm, snd.A], thuds=[t_stamp + 0.3] if t_stamp is not None else [])


# --- output ---------------------------------------------------------------------------------------------
def write_video(film: snd.Film, out: Path, fps: float, dpi: int, audio: Optional[Path], jobs: int):
    snd.render_video(film, out, fps, dpi, audio, lambda fig, beat, k, t: beat.draw(fig, k, t), FIGSIZE,
                     "Whose child is it? Matching and merging", jobs=jobs)
    print(f"wrote {out} ({film.duration:.1f} s)")


def write_frames(film: snd.Film, outdir: Path, dpi: int, fps: float):
    outdir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=FIGSIZE)
    for i, (beat, k) in enumerate(film.frames(fps)):
        beat.draw(fig, k, i / fps)
        fig.savefig(outdir / f"frame_{i:04d}_t{i / fps:05.1f}.png", dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"wrote PNG frames to {outdir}")


def write_stills(out: Path, dpi: int):
    from matplotlib.backends.backend_pdf import PdfPages
    fig = plt.figure(figsize=FIGSIZE)
    if out.suffix.lower() == ".pdf":
        with PdfPages(str(out)) as pdf:
            for t in KEYFRAMES:
                draw_frame(fig, t)
                pdf.savefig(fig, facecolor="white")
    else:
        for i, t in enumerate(KEYFRAMES, 1):
            draw_frame(fig, t)
            fig.savefig(out.with_name(f"{out.stem}_beat{i}{out.suffix}"), dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"wrote stills to {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", type=Path, default=HERE / "animate_whose_child.mp4", help=".mp4 (needs ffmpeg) or .gif (silent)")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--dpi", type=int, default=150, help="150 gives 1920x1080, 100 gives 1280x720")
    p.add_argument("--step", type=int, default=1, help="render every step-th frame")
    p.add_argument("--fast", action="store_true", help="preview: --step 4 --dpi 60")
    p.add_argument("--frames-dir", type=Path, default=None)
    p.add_argument("--stills", type=Path, default=None, help="one key frame per beat (.pdf or .png)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--jobs", type=int, default=4, help="render the frames in this many processes (joined losslessly)")
    p.add_argument("--audio-only", type=Path, default=None, metavar="WAV",
                   help="write just the soundtrack (narration + score) and exit, without rendering")
    p.add_argument("--cache-dir", type=Path, default=HERE.parent.parent / "build" / "narration" / "MnM",
                   help="the synthesised narration clips are kept here (make clean removes them)")
    snd.add_narration_args(p)             # --engine --voice --rate --narration-dir --no-narration --print-script --music-db --no-music
    args = p.parse_args()
    if args.fast:
        args.step, args.dpi = max(args.step, 4), min(args.dpi, 60)
    snd.narration_defaults(args)
    fps = args.fps / args.step
    print(f"{DURATION:g} s script clock at {FPS} fps: cart from x = {cart_x(0):.1f} past stop 3 at x = {X3:.0f} to x = {cart_x(DURATION):.1f} at stop 4 (x = {X4:.0f})")

    lines = narration()
    if args.print_script:
        snd.print_script(lines)
        return
    ffmpeg = shutil.which("ffmpeg")
    if args.output.suffix.lower() != ".gif" and ffmpeg is None and not args.no_narration:
        sys.exit("ffmpeg not found on PATH (needed for the narration clips)")
    clips = snd.load_narration(lines, args, args.cache_dir, ffmpeg) if ffmpeg else {}
    film = snd.Film(BEATS, snd.clip_lengths(clips), NAMES)
    print(f"film: {film.describe()}")
    score = None if args.no_music else score_spec(film)
    if args.audio_only:
        snd.mix_audio(film.duration, film.voice_at, clips, score, args.music_db, args.audio_only)
        print(f"wrote {args.audio_only}")
        return

    if args.stills is not None:
        write_stills(args.stills, args.dpi)
    if args.frames_dir is not None:
        write_frames(film, args.frames_dir, args.dpi, fps)
    elif not args.no_video:
        audio = None
        tmp = tempfile.TemporaryDirectory(prefix="jitjet_mnm_")
        if ffmpeg and args.output.suffix.lower() != ".gif" and (clips or score):
            audio = Path(tmp.name) / "soundtrack.wav"
            snd.mix_audio(film.duration, film.voice_at, clips, score, args.music_db, audio)
            if clips:
                snd.write_srt(film.voice_at, clips, lines, args.output.with_suffix(".srt"))
        write_video(film, args.output, fps, args.dpi, audio, args.jobs)


if __name__ == "__main__":
    sys.exit(main())
