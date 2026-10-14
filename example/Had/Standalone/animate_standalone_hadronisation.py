#!/usr/bin/env python3
"""Animated companion to standalone_hadronisation.py.

Nine scenes, about 90 s in total, show how the partons of the chapter-4 record turn into
the primary hadrons of chapter 5. Every number on screen is computed with the classes of
that script: the seed-1 hadronisation of ../../MPI/event_MPI.lhe (the record
../event_Had.lhe) is replayed. The first four are sketches that introduce the picture
(the TikZ figures of the chapter, animated); the last five follow the record.

  1  guests     The guest list: which combinations of quarks and antiquarks QCD lets out
                of the house (mesons, baryons, the rare tetraquarks, pentaquarks and
                glueballs) and which carry a net colour charge and are never free; the
                string makes mesons and baryons only.
  2  wedding    The mother's rope, b g g g g ubar, with its four knots; the rope snapping
                in the order the script drew, a pair from the vacuum at every break; and
                the procession of its sixteen hadrons with their quark content, the b in
                the first and the ubar in the last.
  3  snapshots  A generic rope in its rest frame, snapshot by snapshot: the ends fly
                apart, pairs appear at the breaks, every piece becomes a yo-yo; on the
                right the same history on the world sheet, with light-cone coordinates
                x+- = t +- x and x+ x- = tau^2 read off the newest break.
  4  ladder     The mother's rope break by break from the -v log: each row the string that
                is left, the end that fires, the pair and the hadron, down to the final
                two; then the chain of sixteen with the fifteen vacuum pairs between
                neighbours.
  5  yoyo       A two-parton string of the event, drawn in the string rest frame: the
                quark and the antiquark fly apart at the speed of light with the string
                between them, the string breaks at the vertices of the record one after
                the other in time, and the pieces between neighbouring vertices become
                the hadrons; on the right the same vertices appear in the (x+, x-)
                plane on which the area law lines up.
  6  bstring    The b quark's string in the (eta, phi) plane: the partons with the colour
                line through them, then the hadrons popping out one by one in the order
                the script produced them, from both ends, with the flavour chain and the
                remaining mass of the string running down to W_min on the right.
  7  kinks      The regions of the same string's world sheet: the two ends walk through
                the (i, k) grid of regions, swallowing gluon halves on the way; the
                hadrons are placed where they formed.
  8  event      The whole event: the 211 partons of stage 3 fade string by string, in
                order of mass, and the 365 primary hadrons take their place; the counter
                follows the strings, the ministrings and the hadrons.
  9  ensemble   The same partons hadronised again and again: the x_E = E_B/E_b of the b
                quark's hadron fills a histogram, its running mean settles on the
                Lund-Bowler value next to PYTHIA 8 on the same partons, and the B species
                fractions grow.

Examples (matplotlib is needed; on this Mac use /usr/bin/python3):
  python3 animate_standalone_hadronisation.py                  # -> .mp4 next to this file
  python3 animate_standalone_hadronisation.py --fast           # quick low-resolution preview
  python3 animate_standalone_hadronisation.py --scene bstring -o bstring.gif
  python3 animate_standalone_hadronisation.py --stills stills.pdf --no-video

An .mp4 needs ffmpeg on the PATH; a .gif needs only Pillow. --frames-dir writes the
individual frames as PNG files instead. --seed is the one of the standalone script
(default: seed 1, the committed record).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import standalone_hadronisation as core  # noqa: E402
import plot_journey_Had as journey  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402

COL_B, COL_BBAR, COL_HAD = "#ff0000", "#0000ff", "#9b4dff"
COL_STR, COL_GLU, COL_A, COL_PY, COL_EVENT = "#2a78d6", "#2f8f5b", "#8e5bb5", "#7f7f7f", "#c0392b"
COL_ISR, COL_FSR, COL_MPI, COL_REM = "#ff7f00", "#009900", "#009999", "#666666"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d6d3ca"
FIGSIZE = (12.8, 7.2)
STATS_FILE = HERE / "standalone_had_stats.stats.json"
PY_FILE = HERE / "validate_hadronisation_with_pythia_record.json"
SCENE_NAMES = {"1": "guests", "2": "wedding", "3": "snapshots", "4": "ladder", "5": "yoyo", "6": "bstring", "7": "kinks", "8": "event", "9": "ensemble"}
HBARC = 0.19732698


def smooth(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ramp(k: int, k0: int, k1: int) -> float:
    return smooth((k - k0) / max(k1 - k0, 1))


class Physics:
    def __init__(self, seed: int, input_path: Path):
        self.args = SimpleNamespace(input=input_path, alund=core.A_LUND, blund=core.B_LUND, rfactb=core.R_FACT_B,
                                    rfactc=core.R_FACT_C, peterson=False, sigma=core.SIGMA_PT, prob_s=core.PROB_S_TO_UD,
                                    prob_qq=core.PROB_QQ_TO_Q, radius=0.4)
        self.ev, self.had = core.run(self.args, seed)
        self.summary = core.summarise(self.ev, self.had)
        self.b = self.summary["b"]["b"]
        self.bstring = next(x for x in self.had.strings if self.b["quark_row"] in x["rows"])
        two = [x for x in self.had.strings if x["kind"] == "string" and len(x["partons"]) == 2 and x.get("system") is not None]
        self.yoyo = min(two, key=lambda x: abs(x["w"] - 30.0))
        self.seed = seed
        self.stats = json.loads(STATS_FILE.read_text()) if STATS_FILE.exists() else None
        self.py = json.loads(PY_FILE.read_text()) if PY_FILE.exists() else None

    def more_xe(self, seeds):
        out = []
        for s in seeds:
            ev, had = core.run(self.args, s)
            sm = core.summarise(ev, had)
            out.append((sm["b"]["b"]["x_e"], sm["b"]["b"]["hadron"]))
        return out


# --- drawing helpers ------------------------------------------------------------------------------------
def chrome(fig, scene_no: int, title: str, caption: str):
    fig.clf()
    fig.patch.set_facecolor("white")
    fig.text(0.02, 0.965, f"Scene {scene_no}: {title}", fontsize=15, weight="bold", color=TEXT, va="center")
    fig.text(0.98, 0.965, "standalone_hadronisation.py", fontsize=9.5, color=TEXT2, ha="right", va="center")
    fig.text(0.02, 0.03, caption, fontsize=10.5, color=TEXT, va="bottom", wrap=True)


def style_2d(ax, fontsize=8):
    ax.tick_params(labelsize=fontsize, colors=TEXT2)
    for sp in ax.spines.values():
        sp.set_color(TEXT2)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def eta_phi_frame(ax, xlim=None):
    for label, lo, hi, shade in journey.ps.REGIONS:
        for sgn in (-1, 1):
            ax.axvspan(sgn * lo, sgn * hi, color=shade, lw=0, zorder=0)
    ax.set_xlim(*(xlim or (-journey.ETA_MAX, journey.ETA_MAX)))
    ax.set_ylim(-math.pi, math.pi)
    ax.set_xlabel(r"$\eta$", fontsize=10)
    ax.set_ylabel(r"$\phi$", fontsize=10)
    style_2d(ax, 8)


def msize(pt: float, scale=1.0) -> float:
    return scale * max(3.0, min(16.0, 3.0 + 3.0 * math.log10(1 + pt)))


class Scene:
    number, title, nframes = 0, "", 1

    def __init__(self, phys: Physics, fps: int):
        self.phys, self.fps = phys, fps

    def draw(self, fig, k: int):
        raise NotImplementedError

    def keyframe(self) -> int:
        return self.nframes - 1


# --- quark content helpers for the introductory scenes --------------------------------------------------
FLAV_ORDER = {2: 0, 1: 1, 3: 2, 4: 3, 5: 4}
Q_COL = {"r": "#d62728", "g": "#2ca02c", "b": "#1f77b4", "ar": "#5fd3e8", "ag": "#e377c2", "ab": "#f2d74e"}
COL_VAC = COL_GLU


def flavs(pid: int):
    """the quark flavours (signed) inside a quark, antiquark or diquark id"""
    if core.is_diquark(pid):
        q1, q2, _ = core.diquark_flavours(pid)
        return [q1, q2] if pid > 0 else [-q1, -q2]
    return [pid]


def flav_tex(f: int) -> str:
    n = core.QUARK_NAME[abs(f)]
    return n if f > 0 else r"\bar{" + n + "}"


def fl_tex(pid: int) -> str:
    """a quark, antiquark or diquark id as TeX (without the dollars)"""
    return "".join(flav_tex(f) for f in flavs(pid))


def content_tex(info) -> str:
    """the constituents of a hadron from the end flavour and the new one: quarks first, u d s order"""
    fl = flavs(info["old"]) + flavs(info["new"])
    q = sorted([f for f in fl if f > 0], key=lambda f: FLAV_ORDER.get(f, 9))
    a = sorted([f for f in fl if f < 0], key=lambda f: FLAV_ORDER.get(-f, 9))
    return "$" + "".join(flav_tex(f) for f in q + a) + "$"


def pair_tex(new: int) -> str:
    """the vacuum pair of a break, from the flavour that joined the hadron"""
    q = [f for f in flavs(new) + flavs(-new) if f > 0]
    a = [f for f in flavs(new) + flavs(-new) if f < 0]
    if len(q) == 1:
        return "$" + flav_tex(q[0]) + flav_tex(a[0]) + "$"
    return "$(" + "".join(flav_tex(f) for f in q) + r")(" + "".join(flav_tex(f) for f in a) + ")$"


def constituents(info):
    """(left, right) constituent of a hadron along the string: quark end hadrons keep the old flavour on the left"""
    return (info["old"], info["new"]) if info["end"] == "+" else (info["new"], info["old"])


def ball(ax, x, y, col, ms=9, edge=None, alpha=1.0, z=6):
    ax.plot([x], [y], "o", color=col, ms=ms, mec=edge or col, mew=1.0 if edge else 0.0, alpha=alpha, zorder=z, clip_on=False)


def box(ax, x, y, w, h, fc, ec, alpha=1.0, ls="-", z=3):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.0, ls=ls, alpha=alpha, zorder=z, clip_on=False))


def coil(ax, x0, x1, y, col, alpha=1.0, amp=0.08, n=7):
    s = np.linspace(0, 1, 120)
    ax.plot(x0 + s * (x1 - x0), y + amp * np.sin(2 * math.pi * n * s), color=col, lw=1.6, alpha=alpha, zorder=5, clip_on=False)


def blank_axes(fig, rect, xlim, ylim):
    ax = fig.add_axes(rect)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return ax


# --- scene 1: the guest list ------------------------------------------------------------------------------
class GuestsScene(Scene):
    number, title = 1, "the guest list: which combinations of quarks QCD lets out"
    ALLOWED = [
        ("meson  $q\\bar q$", r"$3\otimes\bar 3 = 1\oplus 8$", r"$\pi^+\,(u\bar d)$, $K^-\,(s\bar u)$, $\bar B^{*0}\,(b\bar d)$", ["r", "ar"], None),
        ("baryon  $qqq$", r"$3\otimes3\otimes3 = 1\oplus8\oplus8\oplus10$", r"$p\,(uud)$, $n\,(udd)$, $\Sigma^0\,(uds)$", ["r", "g", "b"], None),
        (r"antibaryon  $\bar q\bar q\bar q$", "the conjugate", r"$\bar n\,(\bar u\bar d\bar d)$, $\bar\Sigma^0\,(\bar u\bar d\bar s)$", ["ar", "ag", "ab"], None),
        (r"tetraquark  $qq\bar q\bar q$", "rare (LHCb); not from a string", r"$T_{cc}^+\,(cc\bar u\bar d)$", ["r", "g", "ar", "ag"], None),
        (r"pentaquark  $qqqq\bar q$", "rare (LHCb); not from a string", r"$P_c^+\,(uudc\bar c)$", ["r", "g", "b", "r", "ar"], None),
        ("glueball  $gg$, $ggg$", r"$8\otimes8 \ni 1$", "predicted, not yet certain", [], "gg"),
    ]
    FORBIDDEN = [
        ("a lone quark  $q$", "a triplet 3", "never seen", ["r"], None),
        ("a diquark  $qq$", r"$3\otimes3=\bar3\oplus6$: no singlet", r"but a fine string end (a $\bar 3$)", ["r", "g"], None),
        (r"two quarks, one antiquark  $qq\bar q$", r"$n_q - n_{\bar q} = 1$", "no singlet", ["r", "g", "ar"], None),
        ("a lone gluon  $g$", "an octet 8", "never seen", [], "g"),
    ]

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.per = max(int(0.55 * fps), 3)
        self.nframes = self.per * 10 + int(2.5 * fps)

    def keyframe(self):
        return self.nframes - 1

    def item(self, ax, x, y, spec, k0, k, bad):
        a = ramp(k, k0, k0 + self.per)
        if a <= 0:
            return
        name, line1, line2, balls, glue = spec
        w = 0.55 * max(len(balls), 2) + 0.5
        box(ax, x, y, w, 0.95, "#fdeaea" if bad else "#f4f4f2", "#c0392b" if bad else "#9a9a94", alpha=a, ls="--" if bad else "-")
        if glue:
            for dy in ([0.12, -0.12] if glue == "gg" else [0.0]):
                coil(ax, x - w / 2 + 0.25, x + w / 2 - 0.25, y + dy, COL_GLU, alpha=a)
        for j, c in enumerate(balls):
            ball(ax, x + 0.55 * (j - (len(balls) - 1) / 2), y, Q_COL[c], ms=13, edge="#444444" if c.startswith("a") else None, alpha=a)
        ax.text(x, y - 0.68, name, fontsize=9.5, weight="bold", ha="center", va="top", color=TEXT, alpha=a)
        ax.text(x, y - 1.02, line1, fontsize=8.5, ha="center", va="top", color=TEXT2, alpha=a)
        ax.text(x, y - 1.34, line2, fontsize=8.5, ha="center", va="top", color=TEXT2, alpha=a)

    def draw(self, fig, k):
        chrome(fig, self.number, self.title,
               "A free particle must be a colour singlet: the colours of its quarks (red, green, blue) and the anticolours of its antiquarks "
               "must cancel, otherwise the field it drags behind it would stretch to infinity. The rule is that the number of quarks minus the "
               "number of antiquarks is a multiple of three. The Lund string makes only the first two kinds, mesons and baryons: a string end is "
               "a quark, an antiquark or a diquark.")
        ax = blank_axes(fig, [0.03, 0.12, 0.94, 0.80], (0, 16), (0, 7.4))
        ax.text(0.2, 7.2, "allowed: colour singlets, the colours cancel", fontsize=11, weight="bold", color=TEXT, va="top")
        xs = [1.3, 4.0, 6.7, 9.5, 12.3, 14.7]
        for j, (x, spec) in enumerate(zip(xs, self.ALLOWED)):
            self.item(ax, x, 5.8, spec, j * self.per, k, False)
        a_head = ramp(k, 6 * self.per - self.per // 2, 6 * self.per)
        ax.text(0.2, 3.55, "forbidden: a net colour charge, the field would stretch to infinity", fontsize=11, weight="bold", color="#c0392b", va="top", alpha=a_head)
        for j, (x, spec) in enumerate(zip([1.3, 4.0, 7.0, 9.8], self.FORBIDDEN)):
            self.item(ax, x, 2.2, spec, (6 + j) * self.per, k, True)
        a_rule = ramp(k, 10 * self.per, 10 * self.per + self.fps)
        ax.text(11.6, 2.95, "the rule", fontsize=10, weight="bold", color=TEXT, va="top", alpha=a_rule)
        ax.text(11.6, 2.6, "a free hadron is invariant under SU(3) of colour:\n"
                "$n_q - n_{\\bar q}$ a multiple of three, any number of gluons.\n"
                "Every string break creates a pair, $q'\\bar q'$ or a diquark pair,\n"
                "so flavour, charge and baryon number are conserved,\n"
                "and the string's hadrons are mesons and baryons only.",
                fontsize=8.5, color=TEXT2, va="top", alpha=a_rule, linespacing=1.35)


# --- scene 2: the wedding -------------------------------------------------------------------------------
class WeddingScene(Scene):
    number, title = 2, "the wedding: the mother's rope snaps into a chain of mesons and baryons"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.x = phys.bstring
        self.sysm = self.x["system"]
        self.chain = self.sysm.ordered()                      # string order, from the b end
        self.prod = list(self.sysm.hadrons)                   # production order
        n = len(self.chain)
        # which break (index j = between chain[j] and chain[j+1]) each production step creates
        self.break_order, seen = [], set()
        n_pos = len(self.sysm.pos.hadrons)
        for hid, p, info in self.prod:
            if info.get("final"):
                j = n_pos - 1
            elif info["end"] == "+":
                j = info["rank"] - 1
            else:
                j = n - 1 - info["rank"]
            if j not in seen and 0 <= j < n - 1:
                seen.add(j)
                self.break_order.append((j, info))
        self.per = max(int(0.4 * fps), 3)
        self.t_rope = int(1.6 * fps)
        self.t_snap = self.t_rope + int(0.6 * fps)
        self.nframes = self.t_snap + self.per * len(self.break_order) + int(2.5 * fps)

    def keyframe(self):
        return self.nframes - 1

    def draw(self, fig, k):
        ph = self.phys
        x = self.x
        n = len(self.chain)
        chrome(fig, self.number, self.title,
               f"Every parton of the record holds one end of a colour line and nobody may leave alone. The mother's rope, string 1, runs from the b "
               f"through {len(x['partons']) - 2} gluons to a $\\bar u$ from the rest of the event (W = {x['w']:.0f} GeV). As the partons fly apart the rope "
               f"snaps, in the order the script drew, and at every break a pair from the vacuum holds the two new ends; what walks out is a chain of "
               f"{n} hadrons, couples and triples, with the b in the first and the $\\bar u$ in the last.")
        ax = blank_axes(fig, [0.03, 0.12, 0.94, 0.80], (0, 16), (0, 7.6))
        # (a) the rope with its kinks
        ax.text(0.1, 7.5, "(a) the ropes of the record: the mother's, with four knots", fontsize=10.5, weight="bold", color=TEXT, va="top")
        partons = x["partons"]
        m = len(partons)
        xs = np.linspace(1.0, 12.0, m)
        ys = [6.35 + (0.35 if (j % 2 == 1 and 0 < j < m - 1) else (-0.25 if 0 < j < m - 1 else 0.0)) for j in range(m)]
        a_rope = ramp(k, 0, self.t_rope)
        seg = a_rope * (m - 1)
        for j in range(m - 1):
            f = min(max(seg - j, 0.0), 1.0)
            if f > 0:
                ax.plot([xs[j], xs[j] + f * (xs[j + 1] - xs[j])], [ys[j], ys[j] + f * (ys[j + 1] - ys[j])], color="#d98c1f", lw=4, solid_capstyle="round", zorder=2)
        for j, (pid, p, mass, rows) in enumerate(partons):
            if seg >= j - 0.05 or j == 0:
                col = COL_B if pid == 5 else COL_BBAR if pid < 0 else COL_GLU
                ball(ax, xs[j], ys[j], col, ms=11 if pid != 21 else 9)
                lab = "g" if pid == 21 else "$" + fl_tex(pid) + "$"
                ax.text(xs[j], ys[j] + (0.32 if ys[j] > 6.35 else -0.32), f"{lab}  {core.pt(p):.1f} GeV", fontsize=8, ha="center",
                        va="bottom" if ys[j] > 6.35 else "top", color=col)
        ax.text(13.0, 6.4, f"string 1 of {len(ph.had.strings)} colour singlets;\nthe mother's b carries {core.pt(partons[0][1]):.1f} GeV of $p_T$",
                fontsize=8.5, color=TEXT2, va="center")
        # (b) the rope snaps
        a_b = ramp(k, self.t_rope, self.t_snap)
        if a_b > 0:
            ax.text(0.1, 5.15, "(b) the rope snaps: a pair from the vacuum at every break, in the order the script drew them", fontsize=10.5,
                    weight="bold", color=TEXT, va="top", alpha=a_b)
            x0, x1, yb = 1.0, 15.2, 4.3
            slot = (x1 - x0) / n
            n_br = 0 if k < self.t_snap else min(len(self.break_order), (k - self.t_snap) // self.per + 1)
            made = self.break_order[:n_br]
            cuts = sorted(j for j, _ in made)
            edges = [x0] + [x0 + slot * (j + 1) for j in cuts] + [x1]
            for e0, e1 in zip(edges, edges[1:]):
                ax.plot([e0 + 0.16, e1 - 0.16], [yb, yb], color="#d98c1f", lw=4, alpha=a_b, solid_capstyle="round", zorder=2)
            ball(ax, x0, yb, COL_B, ms=11, alpha=a_b)
            ball(ax, x1, yb, COL_BBAR, ms=11, alpha=a_b)
            ax.text(x0, yb - 0.3, "b", fontsize=9, ha="center", va="top", color=COL_B, alpha=a_b)
            ax.text(x1, yb - 0.3, r"$\bar u$", fontsize=9, ha="center", va="top", color=COL_BBAR, alpha=a_b)
            for i, (j, info) in enumerate(made):
                a = 1.0 if i < n_br - 1 else ramp(k - self.t_snap - (n_br - 1) * self.per, 0, self.per)
                xc = x0 + slot * (j + 1)
                new = info["new"]
                left, right = (new, -new) if info["end"] == "+" else (-new, new)
                ball(ax, xc - 0.17, yb, COL_VAC, ms=8, edge="#222222" if left < 0 else None, alpha=a)
                ball(ax, xc + 0.17, yb, COL_VAC, ms=8, edge="#222222" if right < 0 else None, alpha=a)
                ax.text(xc, yb + 0.22, "$" + fl_tex(left) + r"\;" + fl_tex(right) + "$", fontsize=7.5, ha="center", va="bottom", color=COL_GLU, alpha=a)
                ax.annotate("", (xc - 0.5, yb - 0.3), (xc - 0.2, yb - 0.3), arrowprops=dict(arrowstyle="->", color=COL_GLU, lw=0.8, alpha=a))
                ax.annotate("", (xc + 0.5, yb - 0.3), (xc + 0.2, yb - 0.3), arrowprops=dict(arrowstyle="->", color=COL_GLU, lw=0.8, alpha=a))
            ax.text(x0, 3.75, f"{n_br} of {n - 1} breaks: each costs the rope's own energy, 1 GeV per fermi, and pays for one pair", fontsize=8.5,
                    color=TEXT2, va="top", alpha=a_b)
        # (c) the procession
        if k >= self.t_snap:
            ax.text(0.1, 3.0, f"(c) the procession: the {n} hadrons in their order along the rope, with their quark content", fontsize=10.5,
                    weight="bold", color=TEXT, va="top")
            n_br = min(len(self.break_order), (k - self.t_snap) // self.per + 1)
            have = set(j for j, _ in self.break_order[:n_br])
            slot = 14.2 / n
            for i, (hid, p, info) in enumerate(self.chain):
                ready = (i == 0 or (i - 1) in have) and (i == n - 1 or i in have)
                if not ready:
                    continue
                xc = 1.0 + slot * (i + 0.5)
                left, right = constituents(info)
                fl = flavs(left) + flavs(right)
                bary = len(fl) == 3
                col = COL_B if info["end"] == "+" else COL_BBAR
                box(ax, xc, 1.95, slot * 0.9, 0.95, "white", col)
                for jq, f in enumerate(fl):
                    orig = (f == 5 and i == 0) or (f == -2 and i == n - 1)
                    c = COL_B if (orig and f > 0) else COL_BBAR if orig else COL_VAC
                    ball(ax, xc + 0.22 * (jq - (len(fl) - 1) / 2), 2.1, c, ms=7, edge="#222222" if (f < 0 and not orig) else None)
                ax.text(xc, 1.75, content_tex(info), fontsize=7.5, ha="center", va="center", color=TEXT2)
                ax.text(xc, 2.5, journey.hadron_tex(core.pname(hid)), fontsize=8.5, ha="center", va="bottom", color=col)
            if n_br >= len(self.break_order):
                a = ramp(k - self.t_snap - len(self.break_order) * self.per, 0, self.fps)
                nb = sum(1 for _, _, info in self.chain if len(flavs(info["old"]) + flavs(info["new"])) == 3)
                ax.text(1.0, 1.25, f"{n - nb} mesons and {nb} baryons (a diquark pair makes a baryon and an antibaryon side by side); red boxes from the b end, "
                        f"blue from the $\\bar u$ end; the {ph.b['hadron']} ($b\\bar d$) keeps {100 * ph.b['x_e']:.0f}% of the mother's energy",
                        fontsize=8.5, color=TEXT, va="top", alpha=a)


# --- scene 3: the rope snapshot by snapshot, and light-cone coordinates ------------------------------------
class SnapshotScene(Scene):
    number, title = 3, "the rope in time, and why light-cone coordinates: a yo-yo at every break"
    T_END = 6.5            # fm of string time shown
    T_TURN = 3.0           # W / 2 kappa: when the original ends have given all their momentum to the rope
    VERTICES = [(1.8, -1.0), (1.5, 0.5), (2.2, 1.7)]      # (t, x) in fm, string order from the q end

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.t_run = int(6.5 * fps)
        self.nframes = self.t_run + int(1.5 * fps)
        self.dt = self.T_END / self.t_run
        self.simulate()

    def keyframe(self):
        return int(self.t_run * 0.62)

    def simulate(self):
        """ends move at the speed of light; a piece reverses its two ends when their separation reaches its size"""
        ends = [dict(x=0.0, v=-1.0, kind="q", born=0.0, id=0), dict(x=0.0, v=+1.0, kind="qbar", born=0.0, id=1)]
        next_id = 2
        pieces = [dict(L=None)]                       # piece i joins ends[2i] (left) and ends[2i+1] (right)
        pending = sorted(self.VERTICES)
        self.hist = []                                # per frame: list of (x, kind) and the pieces
        self.vertex_frames = []
        t = 0.0
        for k in range(self.t_run + 1):
            while pending and pending[0][0] <= t:
                tb, xb = pending.pop(0)
                # the piece that contains xb
                for i, pc in enumerate(pieces):
                    a, b = ends[2 * i]["x"], ends[2 * i + 1]["x"]
                    if min(a, b) <= xb <= max(a, b):
                        ends.insert(2 * i + 1, dict(x=xb, v=+1.0, kind="q", born=t, id=next_id + 1))
                        ends.insert(2 * i + 1, dict(x=xb, v=-1.0, kind="qbar", born=t, id=next_id))
                        next_id += 2
                        pieces.insert(i + 1, dict(L=None))
                        pieces[i]["L"] = None
                        break
                else:
                    raise RuntimeError(f"vertex at x = {xb} lies outside the string at t = {tb}")
                self.vertex_frames.append((k, t, xb))
            if t >= self.T_TURN:
                for e in (ends[0], ends[-1]):
                    if e.get("turned") is None:
                        e["v"], e["turned"] = -e["v"], True
            for i, pc in enumerate(pieces):
                le, ri = ends[2 * i], ends[2 * i + 1]
                sep = ri["x"] - le["x"]
                approaching = le["v"] > 0 and ri["v"] < 0
                if pc["L"] is None and approaching:
                    pc["L"] = max(sep, 0.05)
                if pc["L"] is not None:
                    if approaching and sep <= -pc["L"]:
                        le["v"], ri["v"] = -1.0, +1.0
                    elif (not approaching) and le["v"] < 0 and ri["v"] > 0 and sep >= pc["L"]:
                        le["v"], ri["v"] = +1.0, -1.0
            self.hist.append(([(e["x"], e["kind"], e["born"], e["id"]) for e in ends], [pc["L"] for pc in pieces], t))
            for e in ends:
                e["x"] += e["v"] * self.dt
            t += self.dt

    def draw(self, fig, k):
        kk = min(k, self.t_run)
        ends, Ls, t = self.hist[kk]
        chrome(fig, self.number, self.title,
               "Left: a quark and an antiquark created at one point fly apart at the speed of light, the rope between them stores 1 GeV per fermi, and "
               "pairs appear at rest in the field at the break-ups; the pieces between neighbouring breaks are short strings whose ends run towards "
               "each other, cross and run apart again for ever, the yo-yos that are the hadrons. Right: the same history on the world sheet. The light-like "
               "directions are the natural axes, $x^\\pm = t \\pm x$: a massless end keeps one of them fixed and a break at proper time $\\tau$ has $x^+ x^- = \\tau^2$.")
        # left: the snapshot
        ax = blank_axes(fig, [0.04, 0.14, 0.50, 0.76], (-6.8, 6.8), (-0.9, 7.2))
        ax.text(-6.6, 7.0, f"the rope at $t$ = {t:.1f} fm", fontsize=11, weight="bold", color=TEXT, va="top")
        y0 = 4.6
        n_pc = len(Ls)
        for i in range(n_pc):
            a, b = ends[2 * i][0], ends[2 * i + 1][0]
            ax.plot([a, b], [y0, y0], color="#d98c1f", lw=6, solid_capstyle="round", zorder=2)
        for i, (xx, kind, born, _) in enumerate(ends):
            orig = born == 0.0
            col = (COL_B if kind == "q" else COL_BBAR) if orig else COL_VAC
            ball(ax, xx, y0, col, ms=12 if orig else 9, edge="#222222" if (kind == "qbar" and not orig) else None, z=7)
            lab = ("$q$" if kind == "q" else r"$\bar q$") if orig else ("$q'$" if kind == "q" else r"$\bar q'$")
            ax.text(xx, y0 + 0.32 + 0.3 * (i % 2), lab, fontsize=9, ha="center", va="bottom", color=col)
        # the earlier snapshots, faded, as a ladder below
        shown = [j for j in range(0, kk + 1, max(1, self.t_run // 9))][-5:]
        for r, j in enumerate(shown):
            e_j, L_j, t_j = self.hist[j]
            yy = y0 - 0.65 * (len(shown) - r)
            al = 0.25 + 0.5 * (r + 1) / len(shown)
            for i in range(len(L_j)):
                ax.plot([e_j[2 * i][0], e_j[2 * i + 1][0]], [yy, yy], color="#d98c1f", lw=3, alpha=al, solid_capstyle="round")
            for xx, kind, born, _ in e_j:
                col = (COL_B if kind == "q" else COL_BBAR) if born == 0.0 else COL_VAC
                ball(ax, xx, yy, col, ms=6, alpha=al, z=5)
            ax.text(-6.6, yy, f"$t$ = {t_j:.1f}", fontsize=7.5, color=TEXT2, va="center")
        msg = (f"{n_pc} piece{'s' if n_pc > 1 else ''}: " + ("the rope fills with energy, $\\kappa r$" if n_pc == 1 else
               f"{n_pc - 1} break{'s' if n_pc > 2 else ''}; " + ("every piece is a hadron, a yo-yo" if not any(L is None for L in Ls) else "the new pairs are pulled apart")))
        ax.text(-6.6, 6.2, msg, fontsize=9, color=TEXT2, va="top")
        ax.text(-6.6, -0.3, "earlier snapshots, bottom to top", fontsize=7.5, color=TEXT2, va="top")
        # right: the world sheet in light-cone coordinates
        ax2 = fig.add_axes([0.60, 0.19, 0.37, 0.70])
        ax2.set_xlim(-6.8, 6.8)
        ax2.set_ylim(0, 7.2)
        ax2.set_xlabel("$x$  [fm]", fontsize=10)
        ax2.set_ylabel("$t$  [fm]", fontsize=10)
        style_2d(ax2, 8)
        tt = np.linspace(0, t, 50)
        ax2.fill_between(np.concatenate([-tt[::-1], tt]), np.concatenate([tt[::-1], tt]), t, color="#fbe9d2", lw=0, zorder=0)
        ax2.plot([0, 6.8], [0, 6.8], color=COL_B, lw=1.5, ls="--")
        ax2.plot([0, -6.8], [0, 6.8], color=COL_BBAR, lw=1.5, ls="--")
        ax2.text(6.6, 7.05, "$x^- = 0$: the $q$ end\n(the $x^+$ axis)", fontsize=8, color=COL_B, ha="right", va="top")
        ax2.text(-6.6, 7.05, "$x^+ = 0$: the $\\bar q$ end\n(the $x^-$ axis)", fontsize=8, color=COL_BBAR, ha="left", va="top")
        xx = np.linspace(-6.5, 6.5, 200)
        for tau in (1.0, 1.45, 2.0):
            ax2.plot(xx, np.sqrt(xx ** 2 + tau ** 2), color=GRID, lw=0.8)
        ax2.text(5.6, math.sqrt(5.6 ** 2 + 1.45 ** 2) + 0.15, r"$x^+ x^- = \tau^2$", fontsize=8, color=TEXT2, ha="right")
        # world lines of every end so far
        for xx, kind, born, eid in ends:
            pts = [(e[0], self.hist[j][2]) for j in range(0, kk + 1) for e in self.hist[j][0] if e[3] == eid]
            col = (COL_B if kind == "q" else COL_BBAR) if born == 0.0 else COL_VAC
            ax2.plot([q[0] for q in pts], [q[1] for q in pts], color=col, lw=1.4, zorder=4)
        ax2.axhline(t, color="#d98c1f", lw=2.5, zorder=3)
        for i in range(len(Ls)):
            ax2.plot([ends[2 * i][0], ends[2 * i + 1][0]], [t, t], color="#d98c1f", lw=5, solid_capstyle="round", zorder=3)
        for kv, tv, xv in self.vertex_frames:
            if kv <= kk:
                ax2.plot([xv], [tv], "o", color="black", ms=6, mec="white", zorder=6)
        # the newest vertex, projected on the light-cone axes
        done = [(kv, tv, xv) for kv, tv, xv in self.vertex_frames if kv <= kk]
        if done:
            kv, tv, xv = done[-1]
            xp, xm = tv + xv, tv - xv
            ax2.plot([xv, xp / 2], [tv, xp / 2], color=COL_B, lw=1, ls=":")
            ax2.plot([xv, -xm / 2], [tv, xm / 2], color=COL_BBAR, lw=1, ls=":")
            ax2.annotate(f"$x^+$ = {xp:.1f}", (xp / 2, xp / 2), xytext=(6, -2), textcoords="offset points", fontsize=8, color=COL_B)
            ax2.annotate(f"$x^-$ = {xm:.1f}", (-xm / 2, xm / 2), xytext=(-6, -2), textcoords="offset points", fontsize=8, color=COL_BBAR, ha="right")
            ax2.text(0.5, 0.97, f"newest break: $x$ = {xv:.1f}, $t$ = {tv:.1f} fm\n$x^\\pm = t \\pm x$ = {xp:.1f}, {xm:.1f}; "
                     f"$\\tau = \\sqrt{{x^+ x^-}}$ = {math.sqrt(max(xp * xm, 0)):.2f} fm", transform=ax2.transAxes, fontsize=8.5, color=TEXT, va="top", ha="center")
        ax2.set_title("the world sheet: every end moves on a light-like line", fontsize=9, color=TEXT2)


# --- scene 4: the ladder: the mother's rope break by break ---------------------------------------------------
class LadderScene(Scene):
    number, title = 4, "break by break: the mother's rope from both ends"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.x = phys.bstring
        self.sysm = self.x["system"]
        self.prod = list(self.sysm.hadrons)
        self.chain = self.sysm.ordered()
        # rows: (description, plus_end, minus_end, left hadrons, right hadrons, W after, new hadron info)
        self.rows = [("the string the shower\nand the cousins left", self.sysm.ids[0], self.sysm.ids[-1], [], [], self.x["w"], None)]
        plus, minus, left, right = self.sysm.ids[0], self.sysm.ids[-1], [], []
        i = 0
        while i < len(self.prod):
            hid, p, info = self.prod[i]
            if info.get("final"):
                h2 = self.prod[i + 1]
                fl = [(hid, info), (h2[0], h2[2])]
                for h, inf in fl:
                    (left if inf["end"] == "+" else right).append((h, inf))
                self.rows.append((f"W = {info['w_rem']:.1f} < W_min: one last\n{pair_tex(info['new'])} pair, the final two", None, None, list(left), list(right), None, (hid, info, h2[0], h2[2])))
                break
            w_after = self.prod[i + 1][2]["w_rem"] if i + 1 < len(self.prod) else None
            endname = "b end" if info["end"] == "+" else r"$\bar u$ end"
            if core.is_diquark(info["new"]):
                desc = f"{endname}: a diquark pair {pair_tex(info['new'])}"
            elif core.is_diquark(info["old"]):
                desc = f"{endname}: {pair_tex(info['new'])}; the quark\njoins the diquark, a baryon"
            else:
                desc = f"{endname}: a {pair_tex(info['new'])} pair"
            if info["end"] == "+":
                left.append((hid, info))
                plus = -info["new"]
            else:
                right.append((hid, info))
                minus = -info["new"]
            self.rows.append((desc, plus, minus, list(left), list(right), w_after, (hid, info)))
            i += 1
        self.per = max(int(0.5 * fps), 3)
        self.t_rows = self.per * len(self.rows)
        self.nframes = self.t_rows + int(3.5 * fps)

    def keyframe(self):
        return self.nframes - 1

    def draw(self, fig, k):
        ph = self.phys
        n = len(self.chain)
        chrome(fig, self.number, self.title,
               f"String 1 break by break, as the -v log recorded it (the gluon kinks are left out of the drawing). A break at the $\\bar u$ end creates a "
               f"$u\\bar u$ pair: the new u joins the old $\\bar u$ into a $\\pi^0$ and the new $\\bar u$ is the end of what is left; a break at the b end "
               f"creates a $d\\bar d$ pair and the $\\bar d$ joins the mother. The two ends take turns in random order until the remaining mass is below "
               f"W_min and the final two share what is left: {n} hadrons, {n - 1} vacuum pairs, the b in the first, the $\\bar u$ in the last.")
        ax = blank_axes(fig, [0.03, 0.11, 0.94, 0.82], (0, 16.5), (0, 18.6))
        n_rows = min(len(self.rows), k // self.per + 1)
        xc, bw = 8.6, 0.56
        for r in range(n_rows):
            desc, plus, minus, left, right, w_after, new = self.rows[r]
            a = 1.0 if r < n_rows - 1 else ramp(k - r * self.per, 0, self.per)
            y = 18.0 - 0.93 * r
            ax.text(2.35, y, desc, fontsize=7, color=TEXT2, ha="right", va="center", alpha=a, linespacing=1.1)
            if plus is not None:
                W = w_after if w_after is not None else 0.0
                half = (1.2 + 0.07 * (self.rows[r][5] or 1.2)) / 2
                ax.plot([xc - half, xc + half], [y, y], color="#d98c1f", lw=4, solid_capstyle="round", alpha=a, zorder=2)
                orig_p, orig_m = plus == self.sysm.ids[0], minus == self.sysm.ids[-1]
                ball(ax, xc - half, y, COL_B if orig_p else COL_VAC, ms=8, alpha=a, edge=None if (orig_p or plus > 0) else "#222222")
                ball(ax, xc + half, y, COL_BBAR if orig_m else COL_VAC, ms=8, alpha=a, edge=None if (orig_m or minus > 0) else "#222222")
                ax.text(xc - half, y + 0.22, "$" + fl_tex(plus) + "$", fontsize=7.5, ha="center", va="bottom", color=COL_B if orig_p else COL_GLU, alpha=a)
                ax.text(xc + half, y + 0.22, "$" + fl_tex(minus) + "$", fontsize=7.5, ha="center", va="bottom", color=COL_BBAR if orig_m else COL_GLU, alpha=a)
                ax.text(15.3, y, f"W = {self.rows[r][5]:.1f} GeV" if r == 0 else f"{self.rows[r][5]:.1f}", fontsize=8, color=TEXT2, va="center", alpha=a)
            else:
                half = 0.5
                ax.plot([xc - half, xc + half], [y, y], color="#d98c1f", lw=4, solid_capstyle="round", alpha=a, zorder=2)
                hid1, info1, hid2, info2 = new
                ball(ax, xc - 0.15, y, COL_VAC, ms=7, alpha=a, edge="#222222" if info1["new"] < 0 else None)
                ball(ax, xc + 0.15, y, COL_VAC, ms=7, alpha=a, edge="#222222" if info1["new"] > 0 else None)
                ax.text(xc, y + 0.22, pair_tex(info1["new"]), fontsize=7.5, ha="center", va="bottom", color=COL_GLU, alpha=a)
                ax.text(15.3, y, f"{info1['w_rem']:.1f} < {info1['w_min']:.1f}", fontsize=8, color=TEXT2, va="center", alpha=a)
            for j, (hid, info) in enumerate(reversed(left)):
                xb = xc - half - 0.45 - bw / 2 - j * (bw + 0.05)
                is_new = new is not None and (hid, info) in [(new[0], new[1])] + ([(new[2], new[3])] if len(new) == 4 else [])
                box(ax, xb, y, bw, 0.62, "#fde8e8" if is_new else "white", COL_B, alpha=a)
                ax.text(xb, y, journey.hadron_tex(core.pname(hid)), fontsize=7.5, ha="center", va="center", color=COL_B, alpha=a)
            for j, (hid, info) in enumerate(reversed(right)):
                xb = xc + half + 0.45 + bw / 2 + j * (bw + 0.05)
                is_new = new is not None and (hid, info) in [(new[0], new[1])] + ([(new[2], new[3])] if len(new) == 4 else [])
                box(ax, xb, y, bw, 0.62, "#e8ecfd" if is_new else "white", COL_BBAR, alpha=a)
                ax.text(xb, y, journey.hadron_tex(core.pname(hid)), fontsize=7.5, ha="center", va="center", color=COL_BBAR, alpha=a)
        # the chain with the pairs
        if k >= self.t_rows:
            a = ramp(k - self.t_rows, 0, self.fps)
            yc = 18.0 - 0.93 * len(self.rows) - 0.9
            ax.text(0.2, yc + 0.75, f"the chain: {n} hadrons along the rope, the {n - 1} vacuum pairs between neighbours (quark to the right, antiquark to the left)",
                    fontsize=9, weight="bold", color=TEXT, va="bottom", alpha=a)
            slot = 16.0 / n
            for i, (hid, p, info) in enumerate(self.chain):
                x0 = 0.25 + slot * (i + 0.5)
                col = COL_B if info["end"] == "+" else COL_BBAR
                fc = "#eeeeee" if info.get("final") else "white"
                box(ax, x0, yc, slot * 0.86, 0.95, fc, col, alpha=a)
                ax.text(x0, yc + 0.18, journey.hadron_tex(core.pname(hid)), fontsize=8, ha="center", va="center", color=col, alpha=a)
                ax.text(x0, yc - 0.22, content_tex(info), fontsize=7, ha="center", va="center", color=TEXT2, alpha=a)
                if i < n - 1:
                    nxt = self.chain[i + 1][2]
                    new = nxt["new"] if nxt["end"] == "-" else -constituents(info)[1]
                    ax.text(x0 + slot / 2, yc - 0.72, pair_tex(new if new > 0 or core.is_diquark(new) else -new), fontsize=6.5, ha="center", va="center",
                            color=COL_GLU, alpha=a, rotation=0)
            ax.text(0.25 + slot * 0.5, yc - 1.15, "holds the mother's b", fontsize=7.5, ha="center", va="top", color=COL_B, alpha=a)
            ax.text(0.25 + slot * (n - 0.5), yc - 1.15, r"holds the original $\bar u$", fontsize=7.5, ha="center", va="top", color=COL_BBAR, alpha=a)
            ax.text(0.25 + slot * n / 2, yc - 1.15, "every other quark came out of the vacuum at a break; flavour, charge and baryon number are conserved at every step",
                    fontsize=7.5, ha="center", va="top", color=TEXT2, alpha=a)


# --- scene 5: the yo-yo -----------------------------------------------------------------------------------
class YoyoScene(Scene):
    number, title = 5, "a string breaks: the two-parton string in its rest frame"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.nframes = 7 * fps
        x = phys.yoyo
        self.sysm = x["system"]
        self.W = x["w"]
        verts = []
        for end in (self.sysm.pos, self.sysm.neg):
            for hid, p, info in end.hadrons:
                if "coords" in info:
                    tp, tm = info["coords"]
                    verts.append(dict(xp=tp * self.W, xm=tm * self.W, t=(tp + tm) * self.W / 2, z=(tp - tm) * self.W / 2,
                                      end=info["end"], name=core.pname(hid), rank=info["rank"]))
        self.verts = sorted(verts, key=lambda v: v["t"])
        self.names = [core.pname(h) for h, _, _ in x["hadrons"]]
        self.tmax = self.W / 2 * 1.05

    def keyframe(self) -> int:
        return int(self.nframes * 0.8)

    def draw(self, fig, k):
        ph = self.phys
        W = self.W
        q, qbar = core.pname(self.sysm.ids[0]), core.pname(self.sysm.ids[-1])
        t_now = ramp(k, 0, int(self.nframes * 0.8)) * self.tmax
        chrome(fig, self.number, self.title,
               f"A {W:.0f} GeV {q} {qbar} string of our event in its own rest frame (seed {ph.seed}). The quark and the antiquark fly apart at the "
               f"speed of light, the string between them stores 1 GeV per fermi, and it breaks where the record's vertices are: every piece "
               f"between two neighbouring vertices is one of the {len(self.names)} hadrons. On the right the same vertices in light-cone "
               r"coordinates: the area law puts them near $\Gamma = x^+ x^- \sim$ 1-3 GeV$^2$.")
        ax = fig.add_axes([0.06, 0.14, 0.50, 0.74])
        ax.set_xlim(-W / 2 * 1.1, W / 2 * 1.1)
        ax.set_ylim(0, self.tmax * 1.12)
        ax.set_xlabel(r"$\kappa z$  [GeV]   (1 GeV = 1 fm)", fontsize=10)
        ax.set_ylabel(r"$\kappa t$  [GeV]", fontsize=10)
        style_2d(ax, 8)
        ax.plot([0, t_now], [0, t_now], color=COL_STR, lw=2)
        ax.plot([0, -t_now], [0, t_now], color=COL_STR, lw=2)
        ax.text(t_now + 0.3, t_now, q, color=COL_STR, fontsize=10, va="center")
        ax.text(-t_now - 0.3, t_now, qbar, color=COL_STR, fontsize=10, va="center", ha="right")
        # the string at time t_now: a horizontal line between the two quarks, cut at the vertices already formed
        done = [v for v in self.verts if v["t"] <= t_now]
        cuts = sorted([v["z"] for v in done])
        segs = [-t_now] + cuts + [t_now]
        for a, b in zip(segs, segs[1:]):
            ax.plot([a, b], [t_now, t_now], color=COL_HAD if (a in cuts or b in cuts) else COL_STR, lw=4, solid_capstyle="butt", alpha=0.9)
        for v in done:
            col = COL_B if v["end"] == "+" else COL_BBAR
            ax.plot([v["z"]], [v["t"]], "o", color=col, ms=6, mec="white", zorder=5)
            ax.plot([v["z"], v["z"]], [v["t"], t_now], color=col, lw=0.8, ls=":")
            ax.annotate(journey.hadron_tex(v["name"]), (v["z"], v["t"]), xytext=(4, -10), textcoords="offset points", fontsize=8, color=col)
        for g in (1, 3, 10):
            zz = np.linspace(-W / 2, W / 2, 200)
            ax.plot(zz, np.sqrt(g + zz ** 2), color=GRID, lw=0.7)
        ax.text(0.02, 0.96, f"$t$ = {t_now / core.KAPPA:.1f} fm:  {len(done)} of {len(self.verts)} break-ups so far", transform=ax.transAxes,
                fontsize=10, color=TEXT, va="top")
        # right: the (x+, x-) plane
        ax2 = fig.add_axes([0.64, 0.14, 0.33, 0.74])
        lo = 1e-3 * W
        for g in (0.3, 1, 3, 10):
            xp = np.logspace(math.log10(lo), math.log10(W), 100)
            ax2.plot(xp, g / xp, color=GRID, lw=0.8)
        for v in done:
            col = COL_B if v["end"] == "+" else COL_BBAR
            ax2.loglog([max(v["xp"], lo)], [max(v["xm"], lo)], "o", color=col, ms=6, mec="white")
        ax2.set_xlim(lo, W * 1.1)
        ax2.set_ylim(lo, W * 1.1)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel(r"$\kappa x^+$  [GeV]", fontsize=10)
        ax2.set_ylabel(r"$\kappa x^-$  [GeV]", fontsize=10)
        ax2.set_title(r"$\Gamma = \kappa^2\tau^2 = x^+ x^-$: lines at 0.3, 1, 3, 10 GeV$^2$", fontsize=9, color=TEXT2)
        style_2d(ax2, 8)
        ax2.plot([], [], "o", color=COL_B, label=f"vertices of the {q} end")
        ax2.plot([], [], "o", color=COL_BBAR, label=f"vertices of the {qbar} end")
        ax2.legend(loc="upper right", fontsize=8, frameon=False)


# --- scene 6: the b string --------------------------------------------------------------------------------
class BStringScene(Scene):
    number, title = 6, "the b quark's string: hadrons from both ends"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.x = phys.bstring
        self.sysm = self.x["system"]
        self.order = list(self.sysm.hadrons)            # production order
        self.per = max(int(0.45 * fps), 4)
        self.nframes = self.per * (len(self.order) + 1) + 2 * fps

    def draw(self, fig, k):
        ph = self.phys
        x, sysm = self.x, self.sysm
        n_show = min(len(self.order), k // self.per)
        shown = self.order[:n_show]
        chrome(fig, self.number, self.title,
               f"The string that holds the b quark: {len(x['partons'])} partons (the b, {len(x['partons']) - 2} gluons and the other end), "
               f"W = {x['w']:.0f} GeV. Hadrons are produced from the two ends in random order; each takes a fraction z of the remaining "
               f"light-cone momentum of its end (Lund-Bowler for the b: z = {ph.b['z_drawn']:.3f} gives the {ph.b['hadron']} with "
               f"x_E = {ph.b['x_e']:.3f}) until the remaining mass drops below W_min and the last two hadrons share what is left.")
        ax = fig.add_axes([0.05, 0.13, 0.52, 0.76])
        etas = [core.eta(p) for _, p, _, _ in x["partons"]]
        phis = [core.phi(p) for _, p, _, _ in x["partons"]]
        xlim = (min(etas) - 0.8, max(etas) + 0.8)
        eta_phi_frame(ax, xlim)
        for (e0, f0), (e1, f1) in zip(zip(etas, phis), zip(etas[1:], phis[1:])):
            d = f1 - f0
            f1u = f1 - 2 * math.pi if d > math.pi else f1 + 2 * math.pi if d < -math.pi else f1
            ax.plot([e0, e1], [f0, f1u], color=COL_STR, lw=1.4, alpha=0.8)
        for (pid, p, m, rows), e, f in zip(x["partons"], etas, phis):
            ax.plot([e], [f], "s" if pid == 21 else "o", color=COL_B if abs(pid) == 5 else COL_GLU, ms=msize(core.pt(p)), mec="white", alpha=0.9)
            ax.annotate(f"{core.pname(pid)} {core.pt(p):.1f}", (e, f), xytext=(5, 5), textcoords="offset points", fontsize=8, color=TEXT2)
        for j, (hid, p, info) in enumerate(shown):
            col = COL_B if info["end"] == "+" else COL_BBAR
            a = 1.0 if j < n_show - 1 else ramp(k - n_show * self.per, 0, self.per)
            ax.plot([core.eta(p)], [core.phi(p)], "o", color=col, ms=msize(core.pt(p)) * (0.6 + 0.4 * a), mfc="none", mew=1.5, alpha=a)
            ax.annotate(journey.hadron_tex(core.pname(hid)), (core.eta(p), core.phi(p)), xytext=(4, -10), textcoords="offset points", fontsize=8, color=col, alpha=a)
        ax.set_title(f"the b quark's string in the (eta, phi) plane: {n_show} of {len(self.order)} hadrons", fontsize=10, color=TEXT2)
        # right: the chain and the remaining mass
        ax2 = fig.add_axes([0.63, 0.13, 0.34, 0.76])
        ax2.axis("off")
        y = 0.98
        ax2.text(0, y, "hadron   end   z        remaining W -> W_min", fontsize=9, color=TEXT2, va="top", family="monospace")
        for j, (hid, p, info) in enumerate(shown):
            y -= 0.045
            col = COL_B if info["end"] == "+" else COL_BBAR
            ztxt = f"{info['z']:.3f}" if info.get("z") is not None else "final"
            wr = info.get("w_rem", 0.0)
            wm = info.get("w_min", 0.0)
            ax2.text(0, y, f"{core.pname(hid):9s} {info['end']:3s}  {ztxt:6s}  {wr:6.1f} > {wm:5.1f}", fontsize=9, color=col, va="top", family="monospace")
        if n_show == len(self.order):
            y -= 0.06
            ax2.text(0, y, f"{len(self.order)} hadrons; momentum conserved to 1e-12 GeV;\nthe {ph.b['hadron']} carries {100 * ph.b['x_e']:.0f}% of the b quark's energy",
                     fontsize=9.5, color=TEXT, va="top")


# --- scene 7: the regions ---------------------------------------------------------------------------------
class KinkScene(Scene):
    number, title = 7, "gluons as kinks: the two ends walk through the regions of the world sheet"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.x = phys.bstring
        self.sysm = self.x["system"]
        self.order = list(self.sysm.hadrons)
        self.per = max(int(0.45 * fps), 4)
        self.nframes = self.per * (len(self.order) + 1) + 2 * fps

    def draw(self, fig, k):
        sysm = self.sysm
        n = sysm.n
        n_show = min(len(self.order), k // self.per)
        chrome(fig, self.number, self.title,
               f"The world sheet of the b quark's string is made of regions (i, k), one for every pair of a piece from the b side "
               f"(the b, then half of each gluon) and a piece from the other side. Each hadron from the b end takes a fraction of its own "
               f"piece and as much of the other side as its mass needs; when a region is used up the end steps to the next one, swallowing "
               f"the rest of the gluon half, which is how a hadron forms across a kink. The map darkens with the region's mass.")
        ax = fig.add_axes([0.07, 0.13, 0.50, 0.76])
        w2 = np.full((n, n + 1), np.nan)
        for i in range(n):
            for kk in range(i + 1, n + 1):
                w2[i, kk] = sysm.region(i, kk).w2
        ax.imshow(np.log10(np.where(w2 > 0, w2, np.nan)).T, origin="lower", cmap="YlGnBu", aspect="auto", extent=(-0.5, n - 0.5, 0.5, n + 0.5), alpha=0.9)
        ax.set_xticks(range(n))
        ax.set_yticks(range(1, n + 1))
        ax.set_xlabel("$i$: piece from the $b$ side", fontsize=10)
        ax.set_ylabel("$k$: piece from the other side", fontsize=10)
        style_2d(ax, 8)
        ax.grid(False)
        paths = {"+": [(0, 1)], "-": [(n - 1, n)]}
        cells = {}
        for hid, p, info in self.order[:n_show]:
            if "region" in info:
                paths[info["end"]].append(info["region"])
                cells.setdefault(info["region"], []).append(core.pname(hid))
        for (i, kk), names in cells.items():
            txt = journey.hadron_tex(names[-1]) + (f" +{len(names) - 1}" if len(names) > 1 else "")
            ax.text(i, kk - 0.3, txt, fontsize=8, ha="center", va="center", color=TEXT)
        for e, col in (("+", COL_B), ("-", COL_BBAR)):
            ii, kk = zip(*paths[e])
            ax.plot(ii, kk, "o-", color=col, ms=7, lw=1.5, mec="white", label=f"path of the {'b' if e == '+' else 'other'} end")
        ax.legend(loc="lower right", fontsize=9, frameon=False)
        ax.set_title(f"{n_show} of {len(self.order)} hadrons placed", fontsize=10, color=TEXT2)
        # right: the pieces used up
        ax2 = fig.add_axes([0.65, 0.13, 0.32, 0.76])
        rem_p = [1.0] * n
        rem_m = [None] + [1.0] * n
        for hid, p, info in self.order[:n_show]:
            if "region" not in info:
                continue
            if info["end"] == "+":
                for j in info.get("steps_dir", []):
                    rem_p[j] = 0.0
                for j in info.get("steps", []):
                    rem_m[j] = 0.0
                rem_p[info["region"][0]] = max(rem_p[info["region"][0]] - info["x_dir"], 0)
                rem_m[info["region"][1]] = max(rem_m[info["region"][1]] - info["x_inv"], 0)
            else:
                for j in info.get("steps_dir", []):
                    rem_m[j] = 0.0
                for j in info.get("steps", []):
                    rem_p[j] = 0.0
                rem_m[info["region"][1]] = max(rem_m[info["region"][1]] - info["x_dir"], 0)
                rem_p[info["region"][0]] = max(rem_p[info["region"][0]] - info["x_inv"], 0)
        ax2.barh(np.arange(n) + 0.2, rem_p, height=0.35, color=COL_B, alpha=0.8, label="piece from the $b$ side still unused")
        ax2.barh(np.arange(1, n + 1) - 0.2, rem_m[1:], height=0.35, color=COL_BBAR, alpha=0.8, label="piece from the other side still unused")
        ax2.set_xlim(0, 1.05)
        ax2.set_ylim(-0.6, n + 0.6)
        ax2.set_yticks(range(n + 1))
        ax2.set_yticklabels([core.pname(sysm.ids[j]) + f" ({j})" for j in range(n + 1)], fontsize=8)
        ax2.set_xlabel("fraction of the light-cone piece left", fontsize=9)
        style_2d(ax2, 8)
        ax2.legend(loc="upper right", fontsize=8, frameon=False)


# --- scene 8: the whole event -----------------------------------------------------------------------------
class EventScene(Scene):
    number, title = 8, "the whole event: 33 strings become 365 hadrons"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.strings = sorted(phys.had.strings, key=lambda s: -s["w"])
        self.per = max(int(0.14 * fps), 2)
        self.nframes = self.per * (len(self.strings) + 2) + 3 * fps
        self.n_partons = sum(len(rows) for s in self.strings for _, _, _, rows in s["partons"])
        ev = phys.ev
        self.parton_rows = {}
        for s in self.strings:
            for pid, p, m, rows in s["partons"]:
                self.parton_rows[rows[0]] = (pid, p)

    def keyframe(self) -> int:
        return self.per * (len(self.strings) // 2)

    def draw(self, fig, k):
        ph = self.phys
        n_done = min(len(self.strings), k // self.per)
        prog = ramp(k - n_done * self.per, 0, self.per)
        chrome(fig, self.number, self.title,
               f"The {self.n_partons} partons of stage 3 are grouped into {len(self.strings)} colour singlets "
               f"by their colour tags ({sum(1 for s in self.strings if s['kind'] == 'mini')} of them ministrings). String by string, heaviest "
               f"first, the partons go and the hadrons come: {ph.summary['n_hadrons']} primary hadrons, {ph.summary['n_charged']} of them charged, "
               f"the momentum sum unchanged. The B hadrons of the two b quarks are the red and blue dots.")
        ax = fig.add_axes([0.05, 0.13, 0.70, 0.76])
        eta_phi_frame(ax)
        n_had = 0
        for j, s in enumerate(self.strings):
            if j < n_done:
                a_p, a_h = 0.0, 1.0
            elif j == n_done:
                a_p, a_h = 1.0 - prog, prog
            else:
                a_p, a_h = 1.0, 0.0
            if a_p > 0:
                for pid, p, m, rows in s["partons"]:
                    e, f = core.eta(p), core.phi(p)
                    if abs(e) > journey.ETA_MAX:
                        e = math.copysign(journey.ETA_MAX - 0.05, e)
                    col = COL_B if rows[0] == ph.b["quark_row"] else COL_BBAR if rows[0] == ph.summary["b"]["bbar"]["quark_row"] else (COL_GLU if pid == 21 else COL_STR)
                    ax.plot([e], [f], "s" if pid == 21 else "D", color=col, ms=msize(core.pt(p), 0.8), mfc="none", mew=1.0, alpha=a_p)
            if a_h > 0:
                for hid, p, info in (s["hadrons"] or []):
                    e, f = core.eta(p), core.phi(p)
                    if abs(e) > journey.ETA_MAX:
                        e = math.copysign(journey.ETA_MAX - 0.05, e)
                    row = info.get("row")
                    col = COL_B if row == ph.b["hadron_row"] else COL_BBAR if row == ph.summary["b"]["bbar"]["hadron_row"] else COL_HAD
                    ax.plot([e], [f], "o", color=col, ms=msize(core.pt(p), 0.9), mec="none", alpha=0.85 * a_h)
                    n_had += 1 if j < n_done else 0
        ax.set_title(f"{n_done} of {len(self.strings)} strings fragmented, {n_had} hadrons so far", fontsize=10, color=TEXT2)
        ax2 = fig.add_axes([0.79, 0.13, 0.19, 0.76])
        ax2.axis("off")
        ax2.plot([], [], "D", color=COL_STR, mfc="none", label="quark / diquark")
        ax2.plot([], [], "s", color=COL_GLU, mfc="none", label="gluon")
        ax2.plot([], [], "o", color=COL_HAD, label="primary hadron")
        ax2.plot([], [], "o", color=COL_B, label="the b quark / its B hadron")
        ax2.plot([], [], "o", color=COL_BBAR, label="the bbar quark / its B hadron")
        ax2.legend(loc="upper left", fontsize=8.5, frameon=False)
        y = 0.6
        for s in self.strings[max(0, n_done - 8):n_done]:
            ax2.text(0, y, f"W = {s['w']:7.1f} GeV, {len(s['partons']):2d} partons -> {len(s['hadrons'] or []):2d} hadrons{' (mini)' if s['kind'] == 'mini' else ''}",
                     fontsize=8, color=TEXT2, family="monospace")
            y -= 0.04


# --- scene 9: the ensemble --------------------------------------------------------------------------------
class EnsembleScene(Scene):
    number, title = 9, "the same partons hadronised again and again: x_E of the b quark's hadron"

    def __init__(self, phys, fps, nevents):
        super().__init__(phys, fps)
        if phys.stats:
            d = phys.stats["b"]["b"]
            xe = d["x_e"]
            names = []
            for h, cnt in phys.stats["b_species"]["b"].items():
                names += [h] * cnt
            names = names[:len(xe)] + [""] * max(0, len(xe) - len(names))
            rng = np.random.default_rng(7)
            rng.shuffle(names)
            names[0] = phys.b["hadron"]                   # the stats file does not pair species and events; seed 1 is known
            self.xe = list(zip(xe, names))
        else:
            self.xe = phys.more_xe(range(phys.seed, phys.seed + nevents))
        self.xe = self.xe[:max(nevents, 20)]
        self.py_xe = np.array(phys.py["b"]["b"]["x_e"]) if phys.py else None
        self.nframes = 6 * fps

    def draw(self, fig, k):
        ph = self.phys
        n = max(1, int(ramp(k, 0, int(self.nframes * 0.85)) * len(self.xe)))
        xs = np.array([x for x, _ in self.xe[:n]])
        chrome(fig, self.number, self.title,
               f"The partons of stage 3 are kept and only the hadronisation is redrawn, seeds 1, 2, 3, ...: the energy fraction the B hadron keeps, "
               f"x_E = E_B/E_b, scatters around its Lund-Bowler mean; PYTHIA 8 on the very same partons "
               f"({ph.py['nevents'] if ph.py else 0} times) gives the dashed line. Our event (seed 1) is the first entry. Right: which B hadron it was.")
        ax = fig.add_axes([0.06, 0.13, 0.55, 0.76])
        ax.hist(xs, bins=np.linspace(0.4, 1.3, 28), color=COL_A, alpha=0.7, label=f"this script, {n} hadronisations")
        if self.py_xe is not None:
            ax.hist(self.py_xe, bins=np.linspace(0.4, 1.3, 28), histtype="step", color=COL_PY, lw=1.5, ls="--",
                    weights=np.full(len(self.py_xe), n / len(self.py_xe)), label=f"PYTHIA 8, same partons (scaled): mean {self.py_xe.mean():.3f}")
        ax.axvline(xs.mean(), color=COL_A, lw=2)
        ax.axvline(self.xe[0][0], color=COL_EVENT, lw=1.5, ls=":")
        ax.text(self.xe[0][0], ax.get_ylim()[1] * 0.9, f" our event {self.xe[0][0]:.3f}", color=COL_EVENT, fontsize=9)
        ax.set_xlabel("$x_E = E_B / E_b$", fontsize=10)
        ax.set_ylabel("hadronisations", fontsize=10)
        ax.set_title(f"running mean {xs.mean():.3f}  (std {xs.std():.3f})", fontsize=10, color=TEXT2)
        style_2d(ax, 8)
        ax.legend(loc="upper left", fontsize=9, frameon=False)
        ax2 = fig.add_axes([0.68, 0.13, 0.30, 0.76])
        counts = {}
        for _, nm in self.xe[:n]:
            counts[nm] = counts.get(nm, 0) + 1
        items = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
        if items:
            ax2.barh(range(len(items)), [100 * c / n for _, c in items], color=COL_B, alpha=0.8)
            ax2.set_yticks(range(len(items)))
            ax2.set_yticklabels([journey.hadron_tex(nm) for nm, _ in items], fontsize=10)
            ax2.invert_yaxis()
        ax2.set_xlabel("% of the b quark's hadrons", fontsize=10)
        ax2.set_title("vector : pseudoscalar = 2.2 : 1, strange 0.217, baryons 0.081", fontsize=9, color=TEXT2)
        style_2d(ax2, 8)


def build_scenes(names: List[str], phys: Physics, args) -> List[Scene]:
    out = []
    for n in names:
        if n == "guests":
            out.append(GuestsScene(phys, args.fps))
        elif n == "wedding":
            out.append(WeddingScene(phys, args.fps))
        elif n == "snapshots":
            out.append(SnapshotScene(phys, args.fps))
        elif n == "ladder":
            out.append(LadderScene(phys, args.fps))
        elif n == "yoyo":
            out.append(YoyoScene(phys, args.fps))
        elif n == "bstring":
            out.append(BStringScene(phys, args.fps))
        elif n == "kinks":
            out.append(KinkScene(phys, args.fps))
        elif n == "event":
            out.append(EventScene(phys, args.fps))
        elif n == "ensemble":
            out.append(EnsembleScene(phys, args.fps, args.events))
    return out


def frame_list(scenes: List[Scene], step: int):
    frames = []
    for sc in scenes:
        for k in range(0, sc.nframes, step):
            frames.append((sc, k))
    return frames


def write_video(scenes, out: Path, fps: int, dpi: int, step: int):
    fig = plt.figure(figsize=FIGSIZE)
    frames = frame_list(scenes, step)
    if out.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=max(1, fps // step))
    else:
        writer = animation.FFMpegWriter(fps=max(1, fps // step), bitrate=3500)
    with writer.saving(fig, str(out), dpi):
        for i, (sc, k) in enumerate(frames):
            sc.draw(fig, k)
            writer.grab_frame(facecolor="white")
            if i % 50 == 0:
                print(f"  frame {i}/{len(frames)}")
    plt.close(fig)
    print(f"wrote {out} ({len(frames)} frames)")


def write_frames(scenes, outdir: Path, dpi: int, step: int):
    outdir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=FIGSIZE)
    for i, (sc, k) in enumerate(frame_list(scenes, step)):
        sc.draw(fig, k)
        fig.savefig(outdir / f"frame_{i:04d}_s{sc.number}_k{k:03d}.png", dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"wrote PNG frames to {outdir}")


def write_stills(scenes, out: Path, dpi: int):
    from matplotlib.backends.backend_pdf import PdfPages
    fig = plt.figure(figsize=FIGSIZE)
    if out.suffix.lower() == ".pdf":
        with PdfPages(str(out)) as pdf:
            for sc in scenes:
                sc.draw(fig, sc.keyframe())
                pdf.savefig(fig, facecolor="white")
    else:
        for sc in scenes:
            sc.draw(fig, sc.keyframe())
            fig.savefig(out.with_name(f"{out.stem}_scene{sc.number}{out.suffix}"), dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"wrote stills to {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", type=Path, default=HERE / "animate_standalone_hadronisation.mp4")
    p.add_argument("--scene", default="all", help="all, or a comma list of guests,wedding,snapshots,ladder,yoyo,bstring,kinks,event,ensemble (or 1..9)")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--dpi", type=int, default=100, help="100 gives 1280x720")
    p.add_argument("--step", type=int, default=1, help="render every step-th frame")
    p.add_argument("--fast", action="store_true", help="preview: --step 4 --dpi 60, fewer events in scene 5")
    p.add_argument("--frames-dir", type=Path, default=None)
    p.add_argument("--stills", type=Path, default=None, help="one key frame per scene (.pdf or .png)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("-i", "--input", type=Path, default=HERE.parent.parent / "MPI" / "event_MPI.lhe")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--events", type=int, default=400, help="hadronisations shown in scene 5 (from the stats file when present)")
    args = p.parse_args()
    if args.fast:
        args.step, args.dpi, args.events = max(args.step, 4), min(args.dpi, 60), min(args.events, 40)
    names = list(SCENE_NAMES.values()) if args.scene == "all" else [SCENE_NAMES.get(s.strip(), s.strip()) for s in args.scene.split(",")]
    bad = [n for n in names if n not in SCENE_NAMES.values()]
    if bad:
        sys.exit(f"unknown scene(s) {bad}; choose from {list(SCENE_NAMES.values())}")
    phys = Physics(args.seed, args.input)
    s = phys.summary
    print(f"event: seed {args.seed}: {len(phys.had.strings)} systems -> {s['n_hadrons']} hadrons; the b quark's {phys.b['hadron']} has x_E = {phys.b['x_e']:.3f}")
    scenes = build_scenes(names, phys, args)
    print("frames: " + ", ".join(f"{sc.number} {SCENE_NAMES[str(sc.number)]} {sc.nframes}" for sc in scenes)
          + f"  ({sum(sc.nframes for sc in scenes) / args.fps:.1f} s at {args.fps} fps)")
    if args.stills is not None:
        write_stills(scenes, args.stills, args.dpi)
    if args.frames_dir is not None:
        write_frames(scenes, args.frames_dir, args.dpi, args.step)
    elif not args.no_video:
        write_video(scenes, args.output, args.fps, args.dpi, args.step)


if __name__ == "__main__":
    main()
