#!/usr/bin/env python3
"""Animated companion to standalone_hadronisation.py.

Five scenes, about 50 s in total, show how the partons of the chapter-4 record turn into
the primary hadrons of chapter 5. Every number on screen is computed with the classes of
that script: the seed-1 hadronisation of ../../MPI/event_MPI.lhe (the record
../event_Had.lhe) is replayed.

  1  yoyo       A two-parton string of the event, drawn in the string rest frame: the
                quark and the antiquark fly apart at the speed of light with the string
                between them, the string breaks at the vertices of the record one after
                the other in time, and the pieces between neighbouring vertices become
                the hadrons; on the right the same vertices appear in the (x+, x-)
                plane on which the area law lines up.
  2  bstring    The b quark's string in the (eta, phi) plane: the partons with the colour
                line through them, then the hadrons popping out one by one in the order
                the script produced them, from both ends, with the flavour chain and the
                remaining mass of the string running down to W_min on the right.
  3  kinks      The regions of the same string's world sheet: the two ends walk through
                the (i, k) grid of regions, swallowing gluon halves on the way; the
                hadrons are placed where they formed.
  4  event      The whole event: the 211 partons of stage 3 fade string by string, in
                order of mass, and the 365 primary hadrons take their place; the counter
                follows the strings, the ministrings and the hadrons.
  5  ensemble   The same partons hadronised again and again: the x_E = E_B/E_b of the b
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
SCENE_NAMES = {"1": "yoyo", "2": "bstring", "3": "kinks", "4": "event", "5": "ensemble"}
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


# --- scene 1: the yo-yo -----------------------------------------------------------------------------------
class YoyoScene(Scene):
    number, title = 1, "a string breaks: the two-parton string in its rest frame"

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


# --- scene 2: the b string --------------------------------------------------------------------------------
class BStringScene(Scene):
    number, title = 2, "the b quark's string: hadrons from both ends"

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


# --- scene 3: the regions ---------------------------------------------------------------------------------
class KinkScene(Scene):
    number, title = 3, "gluons as kinks: the two ends walk through the regions of the world sheet"

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


# --- scene 4: the whole event -----------------------------------------------------------------------------
class EventScene(Scene):
    number, title = 4, "the whole event: 33 strings become 365 hadrons"

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


# --- scene 5: the ensemble --------------------------------------------------------------------------------
class EnsembleScene(Scene):
    number, title = 5, "the same partons hadronised again and again: x_E of the b quark's hadron"

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
        if n == "yoyo":
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
    p.add_argument("--scene", default="all", help="all, or a comma list of yoyo,bstring,kinks,event,ensemble (or 1..5)")
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
