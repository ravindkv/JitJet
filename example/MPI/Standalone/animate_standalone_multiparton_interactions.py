#!/usr/bin/env python3
"""Animated companion to standalone_multiparton_interactions.py.

Five scenes, about 50 s in total, show what happens to the showered b bbar
event of chapter 2 when the rest of the two protons joins in. Every number on
screen is computed with the classes of that script: the seed-1 event at the
impact parameter of the CMSSW event (b = 0.3136 <b>, the record
../event_MPI.lhe) is replayed, once without and once with colour reconnection.

  1  overlap    Two protons, drawn as the CP5 double-Gaussian matter profile,
                approach each other in the transverse plane at impact parameter
                b. The overlap O(b) and the enhancement f(b) = n(b)/<n> follow
                the sliding b; the collision stops at the b of our event and
                the Poisson mean of interactions appears.
  2  chain      The pT scale runs down from the hard scale (113 GeV) to pTmin;
                every time it passes the pT of a secondary interaction the
                interaction happens: the ladder on the left fills rung by rung,
                the (eta, phi) picture on the right fills with the new systems
                (cyan) among the partons of the hard system, and the
                no-interaction probability curve shows how unlikely the quiet
                stretches were.
  3  remnants   The momentum bars of the two protons are eaten interaction by
                interaction; what is left goes to the remnant partons, whose
                flavours appear as the valence content is used up. On the
                right the primordial kT kicks of all initiators and remnants
                sum to zero.
  4  strings    The colour strings in the (y, phi) plane before and after the
                MPI-based colour reconnection: the soft systems' gluons move
                onto the strings of the harder systems and the total string
                length lambda drops.
  5  ensemble   The same hard event dressed again and again: the running
                means of the number of interactions, the transverse
                underlying-event density and the extra pT inside R = 0.4
                around the b quark settle, against PYTHIA 8 (when the
                validation file is present).

Examples (matplotlib is needed; on this Mac use /usr/bin/python3):
  python3 animate_standalone_multiparton_interactions.py                  # -> .mp4 next to this file
  python3 animate_standalone_multiparton_interactions.py --fast           # quick low-resolution preview
  python3 animate_standalone_multiparton_interactions.py --scene chain -o chain.gif
  python3 animate_standalone_multiparton_interactions.py --stills stills.pdf --no-video

An .mp4 needs ffmpeg on the PATH; a .gif needs only Pillow. --frames-dir writes the
individual frames as PNG files instead. --seed and --impact are those of the
standalone script (default: seed 1 at b = 0.3136 <b>, the committed record).
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
import standalone_multiparton_interactions as core  # noqa: E402
import plot_journey_MPI as journey  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

FAMILY = journey.FAMILY
COL_B, COL_BBAR, COL_ISR, COL_FSR = FAMILY["b"]["color"], FAMILY["bbar"]["color"], FAMILY["ISR"]["color"], FAMILY["FSR"]["color"]
COL_MPI, COL_REM = FAMILY["MPI"]["color"], FAMILY["remnant"]["color"]
COL_A, COL_BEAM, COL_EVENT, COL_PY = "#8e5bb5", "#2a78d6", "#c0392b", "#7f7f7f"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d6d3ca"
ETA_MAX = journey.ETA_MAX
FIGSIZE = (12.8, 7.2)
CMSSW_B, CMSSW_F, CMSSW_NMPI = 0.3136, 3.709, 32
PY_FILE = HERE / "validate_mpi_with_pythia_fixedb.json"
STATS_FILE = HERE / "standalone_mpi_stats_fixedb.stats.json"
SCENE_NAMES = {"1": "overlap", "2": "chain", "3": "remnants", "4": "strings", "5": "ensemble"}


def smooth(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def ramp(k: int, k0: int, k1: int) -> float:
    if k1 <= k0:
        return 1.0
    return min(max((k - k0) / float(k1 - k0), 0.0), 1.0)


def rapidity(p) -> float:
    return 0.5 * math.log((p[0] + p[3]) / max(p[0] - p[3], 1e-9))


# --- physics: the event, with and without colour reconnection -----------------------------------------
class Physics:
    def __init__(self, seed: int, impact: float, input_path: Path):
        self.args = SimpleNamespace(input=input_path, ptmax=None, ptmin_isr=0.2, ptmin_fsr=0.5, pt0=2.0, no_isr=False,
                                    no_fsr=False, no_shower=False, no_cr=False, no_kt=False, check=False,
                                    cr_range=core.CR_RANGE, sigma_nd=core.sigma_nd(), radius=0.4, impact=impact)
        self.alpha = core.AlphaStrong(0.118, 2)
        self.pdf = core.TablePDF(core.TABLES)
        self.pt0 = core.pt0_at(core.ECM)
        self.si = core.SigmaInt(self.pt0, self.pdf, self.alpha)
        self.ov = core.Overlap(self.si.total / self.args.sigma_nd)
        self.ev, self.gen, self.hard = core.generate(self.args, seed, self.pdf, self.alpha, self.si, self.ov)
        self.summary = core.summarise(self.ev, self.gen, self.hard, 0.4)
        args_nocr = SimpleNamespace(**vars(self.args))
        args_nocr.no_cr = True
        self.ev0, self.gen0, _ = core.generate(args_nocr, seed, self.pdf, self.alpha, self.si, self.ov)
        self.seed, self.impact = seed, impact

    def generate_more(self, seeds, impact=None):
        """Further events (for the ensemble scene); returns summaries."""
        out = []
        for s in seeds:
            a = SimpleNamespace(**vars(self.args))
            a.impact = impact if impact is not None else self.args.impact
            try:
                ev, gen, hard = core.generate(a, s, self.pdf, self.alpha, self.si, self.ov)
                out.append(core.summarise(ev, gen, hard, 0.4))
            except RuntimeError:
                continue
        return out


# --- drawing helpers ------------------------------------------------------------------------------------
def chrome(fig, scene_no: int, title: str, caption: str):
    fig.clf()
    fig.patch.set_facecolor("white")
    fig.text(0.02, 0.965, f"Scene {scene_no}: {title}", fontsize=15, weight="bold", color=TEXT, va="center")
    fig.text(0.98, 0.965, "standalone_multiparton_interactions.py", fontsize=9.5, color=TEXT2, ha="right", va="center")
    fig.text(0.02, 0.03, caption, fontsize=10.5, color=TEXT, va="bottom", wrap=True)


def style_2d(ax, fontsize=8):
    ax.tick_params(labelsize=fontsize, colors=TEXT2)
    for sp in ax.spines.values():
        sp.set_color(TEXT2)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def eta_phi_frame(ax):
    for label, lo, hi, shade in journey.ps.REGIONS:
        for sgn in (-1, 1):
            ax.axvspan(sgn * lo, sgn * hi, color=shade, lw=0, zorder=0)
    ax.set_xlim(-ETA_MAX, ETA_MAX)
    ax.set_ylim(-math.pi, math.pi)
    ax.set_xlabel(r"$\eta$", fontsize=10)
    ax.set_ylabel(r"$\phi$", fontsize=10)
    style_2d(ax, 8)


def draw_partons(ax, ev, rows, cat_of, size_scale=1.0, alpha=1.0, labels=True):
    for i in rows:
        p = ev.p[i].p
        ptv, e, f = core.pt(p), core.eta(p), core.phi(p)
        if abs(e) > ETA_MAX:
            e = math.copysign(ETA_MAX - 0.05, e)
        cat = cat_of(i)
        st = FAMILY[cat]
        hollow = st["marker"] in ("^", "v", "s", "d")
        ax.scatter(e, f, s=journey.area(ptv) * size_scale, marker=st["marker"], facecolor="none" if hollow else st["color"],
                   edgecolor=st["color"], linewidth=1.0, alpha=alpha, zorder=4 if not hollow else 3)
        if labels and ptv > 20:
            ax.annotate(f"{ptv:.0f}", (e, f), xytext=(3, 3), textcoords="offset points", fontsize=7, color=st["color"])


class Scene:
    number, title, nframes = 0, "", 1

    def __init__(self, phys: Physics, fps: int):
        self.phys, self.fps = phys, fps

    def draw(self, fig, k: int):
        raise NotImplementedError

    def keyframe(self) -> int:
        return self.nframes - 1


# --- scene 1: overlap ------------------------------------------------------------------------------------
class OverlapScene(Scene):
    number, title = 1, "two protons meet at impact parameter b"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.nframes = int(10 * fps)
        self.b_final = phys.gen.b / phys.ov.b_avg

    def draw(self, fig, k):
        ph, ov = self.phys, self.phys.ov
        t_app = ramp(k, 0, int(0.65 * self.nframes))
        b_now = 2.6 - (2.6 - self.b_final) * smooth(t_app)         # in units of <b>
        settled = k >= int(0.65 * self.nframes)
        chrome(fig, 1, self.title,
               "The matter of each proton is a double Gaussian (CP5: 63 % of it in a core of 0.76 of the radius). The chance of a parton-parton\n"
               "interaction is the overlap O(b) of the two profiles; the mean number of interactions is n(b) = f(b) <n> with <n> = sigma_int/sigma_ND = "
               f"{ph.si.total / ph.args.sigma_nd:.2f}.\nAn event with a hard process is more central than average: b is drawn with weight O(b). "
               f"Our event sits at b = {self.b_final:.4f} <b>, f = {ph.gen.fb:.2f} (the CMSSW event: {CMSSW_B}, {CMSSW_F}).")
        ax = fig.add_axes([0.05, 0.14, 0.42, 0.76])
        ax.set_aspect("equal")
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-2.6, 2.6)
        ax.set_xlabel("x  [units of the average impact parameter <b>]", fontsize=9.5)
        ax.set_ylabel("y", fontsize=9.5)
        style_2d(ax)
        bu = ov.b_avg                                   # internal units per <b>
        for cx, col in ((-0.5 * b_now, COL_A), (0.5 * b_now, COL_BEAM)):
            for rr, al in ((2.2, 0.08), (1.6, 0.10), (1.1, 0.13), (0.7, 0.18), (0.4, 0.25)):
                ax.add_patch(Circle((cx, 0), rr / bu * 0.9, color=col, alpha=al, lw=0))
        ax.annotate("", xy=(0.5 * b_now, -2.1), xytext=(-0.5 * b_now, -2.1), arrowprops=dict(arrowstyle="<->", color=TEXT, lw=1.2))
        ax.text(0, -2.3, f"b = {b_now:.3f} <b>", ha="center", fontsize=10.5, color=TEXT)
        ax.text(-0.5 * b_now, 2.3, "proton A", ha="center", fontsize=10, color=COL_A)
        ax.text(0.5 * b_now, 2.3, "proton B", ha="center", fontsize=10, color=COL_BEAM)
        ax2 = fig.add_axes([0.56, 0.52, 0.41, 0.38])
        bs = ov.b / bu
        ax2.plot(bs, ov.O(ov.b) / ov.O(0.0), color=COL_MPI, lw=2, label="O(b)/O(0)")
        ax2.plot(bs, ov.p_int, color=COL_A, lw=1.6, label=r"$P_{\rm int}(b)$")
        ax2.axvline(b_now, color=COL_EVENT, lw=1.2, ls="--")
        ax2.set_xlim(0, 2.6)
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel("b / <b>", fontsize=9.5)
        ax2.legend(fontsize=9, loc="upper right")
        ax2.set_title("overlap and interaction probability", fontsize=10)
        style_2d(ax2)
        ax3 = fig.add_axes([0.56, 0.14, 0.41, 0.3])
        ax3.plot(bs, ov.O(ov.b) * ov.p_int_total, color=COL_EVENT, lw=2)
        f_now = ov.f(b_now * bu)
        ax3.plot([b_now], [f_now], "o", color=COL_EVENT, ms=9, mec="white")
        ax3.axhline(1.0, color=TEXT2, lw=0.8, ls=":")
        ax3.set_xlim(0, 2.6)
        ax3.set_ylim(0, 5)
        ax3.set_xlabel("b / <b>", fontsize=9.5)
        ax3.set_ylabel("f(b)", fontsize=9.5)
        ax3.set_title(f"enhancement f(b) = {f_now:.2f}   ->   mean number of interactions n(b) = {f_now * ph.si.total / ph.args.sigma_nd:.1f}",
                      fontsize=10, color=COL_EVENT if settled else TEXT)
        style_2d(ax3)
        if settled:
            ax.text(0, 1.6, f"{ph.summary['n_mpi'] - 1} secondary interactions happened in our event", ha="center", fontsize=11, color=COL_MPI,
                    weight="bold")

    def keyframe(self):
        return self.nframes - 1


# --- scene 2: the chain ----------------------------------------------------------------------------------
class ChainScene(Scene):
    number, title = 2, "the chain of interactions runs down in pT"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.nframes = int(16 * fps)
        gen = phys.gen
        self.pts = [s["pt"] for s in gen.systems[1:]]
        self.pt_max, self.pt_min = gen.pt_max, core.PTMIN_MPI
        self.log0, self.log1 = math.log(self.pt_max), math.log(self.pt_min)
        self.orig = core.origins(phys.ev, gen)

    def scale_at(self, k):
        t = smooth(ramp(k, int(0.5 * self.fps), self.nframes - int(1.5 * self.fps)))
        return math.exp(self.log0 + (self.log1 - self.log0) * t)

    def draw(self, fig, k):
        ph, gen, ev, si = self.phys, self.phys.gen, self.phys.ev, self.phys.si
        scale = self.scale_at(k)
        n_done = sum(1 for p in self.pts if p >= scale)
        chrome(fig, 2, self.title,
               f"Below the hard scale pTmax = {self.pt_max:.1f} GeV the secondary scatterings are generated like shower emissions: a Sudakov veto algorithm with "
               f"dP/dpT^2 = f(b) (1/sigma_ND) dsigma/dpT^2 exp(-f(b) sigma(>pT)/sigma_ND).\nEach accepted scale gives two rapidities, two momentum fractions, two flavours and a process, "
               f"drawn with the depleted PDFs of the two protons; the new system is then showered from its own pT. Scale now: {scale:.2f} GeV, "
               f"{n_done} of {len(self.pts)} interactions happened.")
        axl = fig.add_axes([0.05, 0.14, 0.22, 0.76])
        axl.set_yscale("log")
        axl.set_ylim(0.15, 200)
        axl.set_xlim(0, 1)
        axl.set_xticks([])
        axl.set_ylabel(r"$p_T$  [GeV]", fontsize=10)
        axl.axhline(self.pt_max, color=COL_B, lw=1.5)
        axl.text(0.02, self.pt_max * 1.15, "hard process 113.3", fontsize=8.5, color=COL_B)
        axl.axhline(self.pt_min, color=TEXT2, lw=1.0, ls=":")
        axl.text(0.02, self.pt_min * 0.72, r"$p_{T\min}$ = 0.2", fontsize=8.5, color=TEXT2)
        axl.axhline(ph.pt0, color=TEXT2, lw=0.8, ls="--")
        axl.text(0.6, ph.pt0 * 1.1, r"$p_{T0}$", fontsize=8.5, color=TEXT2)
        for j, p in enumerate(self.pts):
            if p >= scale:
                axl.axhline(p, color=COL_MPI, lw=1.4)
                if j < 12:
                    axl.text(0.98, p, f"{p:.2f}", fontsize=7.5, color=COL_MPI, ha="right", va="bottom")
        axl.axhspan(self.pt_min, scale, color="#f2e0dc", zorder=0)
        axl.axhline(scale, color=COL_EVENT, lw=2)
        axl.text(0.5, scale * 1.1, "scale", color=COL_EVENT, fontsize=8.5, ha="center")
        axl.set_title("the pT ladder", fontsize=10)
        style_2d(axl)
        axr = fig.add_axes([0.33, 0.14, 0.46, 0.76])
        eta_phi_frame(axr)
        hard_rows = gen.systems[0]["finals"]
        draw_partons(axr, ev, hard_rows, lambda i: ("b" if i == self.star_row(5) else "bbar" if i == self.star_row(-5)
                                                   else self.hard_cat(i)), labels=True)
        for s in gen.systems[1:]:
            if s["pt"] >= scale:
                fresh = s["pt"] >= scale and s["pt"] < scale * 1.35
                draw_partons(axr, ev, s["finals"], lambda i: "MPI", size_scale=1.9 if fresh else 1.0, labels=False)
        axr.set_title(f"the event in the (eta, phi) plane: {len(hard_rows)} partons of the hard system, "
                      f"{sum(len(s['finals']) for s in gen.systems[1:] if s['pt'] >= scale)} from secondary systems", fontsize=10)
        axs = fig.add_axes([0.84, 0.14, 0.14, 0.76])
        pts = si.pt[si.pt < self.pt_max]
        surv = [math.exp(-gen.fb * (si.above(p) - si.above(self.pt_max)) / ph.args.sigma_nd) for p in pts]
        axs.plot(surv, pts, color=COL_EVENT, lw=1.6)
        axs.set_yscale("log")
        axs.set_ylim(0.15, 200)
        axs.set_xlim(0, 1.02)
        axs.axhline(scale, color=COL_EVENT, lw=2)
        axs.set_xlabel("no interaction\nbetween pTmax and pT", fontsize=8.5)
        axs.set_yticklabels([])
        style_2d(axs, 7.5)

    def star_row(self, pid):
        return next(p["row"] for p in self.phys.summary["partons"] if p["name"] == core.pname(pid))

    def hard_cat(self, i):
        rows = journey.parse_listing(core.ps.listing(self.phys.ev, "x").splitlines()) if not hasattr(self, "_rows") else self._rows
        self._rows = rows
        o = journey.origin_of(rows, i)
        return "ISR" if o == "ISR" else "FSR"

    def keyframe(self):
        return self.nframes - int(1.5 * self.fps)


# --- scene 3: remnants -----------------------------------------------------------------------------------
class RemnantScene(Scene):
    number, title = 3, "what is left of the protons: the beam remnants"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.nframes = int(10 * fps)

    def draw(self, fig, k):
        ph, gen, ev = self.phys, self.phys.gen, self.phys.ev
        nsys = len(gen.systems)
        t = smooth(ramp(k, int(0.3 * self.fps), int(0.7 * self.nframes)))
        n_show = int(round(t * nsys))
        done = k >= int(0.7 * self.nframes)
        beams = gen.beams
        chrome(fig, 3, self.title,
               "Every interaction takes a momentum fraction x out of a proton; the PDFs are rescaled to what is left, a kicked-out valence quark is subtracted from the valence\n"
               f"content and a kicked-out sea quark leaves a companion. Proton A gave away {beams[0].n_init} partons (X = {beams[0].x_left():.3f} left), proton B {beams[1].n_init} "
               f"(X = {beams[1].x_left():.3f}).\nThe rest becomes the remnant partons (A: {', '.join(core.name(r.id) for r in beams[0].resolved[beams[0].n_init:])}; "
               f"B: {', '.join(core.name(r.id) for r in beams[1].resolved[beams[1].n_init:])}); every parton gets a primordial kT and the kicks cancel within each proton.")
        ax = fig.add_axes([0.05, 0.14, 0.55, 0.76])
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["proton B\n(p-)", "proton A\n(p+)"], fontsize=10)
        ax.set_xlabel("light-cone momentum fraction of the proton", fontsize=10)
        style_2d(ax)
        for side, y in ((0, 1.0), (1, 0.0)):
            beam = beams[side]
            left = 0.0
            for i, r in enumerate(beam.resolved[:n_show]):
                x = ev.x(gen.systems[i]["in_a_before_kt" if side == 0 else "in_b_before_kt"])
                col = COL_B if i == 0 else COL_MPI
                ax.barh(y, x, left=left, height=0.5, color=col, edgecolor="white", lw=0.5)
                if i == 0:
                    ax.text(left + x / 2, y + 0.33, f"hard: {core.pname(r.id)} x = {x:.3f}" + (" (valence)" if r.is_valence else ""),
                            ha="center" if x > 0.1 else "left", fontsize=8.5, color=COL_B)
                left += x
            if done:
                for r in beam.resolved[beam.n_init:]:
                    share = (r.p[0] + r.p[3]) / gen.ecm if side == 0 else (r.p[0] - r.p[3]) / gen.ecm
                    ax.barh(y, share, left=left, height=0.5, color=COL_REM, edgecolor="white", lw=0.5)
                    if share > 0.03:
                        ax.text(left + share / 2, y, core.name(r.id), ha="center", va="center", fontsize=8.5, color="white")
                    left += share
                ax.text(1.0, y - 0.38, f"remnants take X = {beam.x_left():.3f}", ha="right", fontsize=9, color=COL_REM)
            else:
                ax.text(1.0, y - 0.38, f"used: {left:.3f}", ha="right", fontsize=9, color=TEXT2)
            if done:
                ax.text(0.0, y - 0.38, f"valence left: u x{beam.n_val_left(2)}, d x{beam.n_val_left(1)}", fontsize=9, color=TEXT2)
        ax.set_title(f"interaction {min(n_show, nsys)} of {nsys}: the protons are being used up", fontsize=10.5)
        ax2 = fig.add_axes([0.66, 0.14, 0.31, 0.76])
        ax2.set_aspect("equal")
        ax2.set_xlim(-4, 4)
        ax2.set_ylim(-4, 4)
        ax2.set_xlabel(r"$k_x$  [GeV]", fontsize=9.5)
        ax2.set_ylabel(r"$k_y$  [GeV]", fontsize=9.5)
        ax2.set_title("primordial kT of initiators (filled) and remnants (hollow)", fontsize=9.5)
        style_2d(ax2)
        if done:
            tk = smooth(ramp(k, int(0.7 * self.nframes), int(0.9 * self.nframes)))
            for side, col in ((0, COL_A), (1, COL_BEAM)):
                for i, r in enumerate(beams[side].resolved):
                    kx, ky = r.p[1] * tk, r.p[2] * tk
                    ax2.arrow(0, 0, kx, ky, color=col, alpha=0.7, lw=0.8, head_width=0.08, length_includes_head=True,
                              fill=i < beams[side].n_init)
                sx = sum(r.p[1] for r in beams[side].resolved)
                sy = sum(r.p[2] for r in beams[side].resolved)
                ax2.text(-3.8, 3.5 - 0.5 * side, f"proton {'A' if side == 0 else 'B'}: sum of kicks = ({sx:+.3f}, {sy:+.3f}) GeV", fontsize=8.5, color=col)

    def keyframe(self):
        return self.nframes - 1


# --- scene 4: strings ------------------------------------------------------------------------------------
class StringScene(Scene):
    number, title = 4, "colour reconnection shortens the strings"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.nframes = int(9 * fps)
        self.chains0 = phys.gen0.strings()
        self.chains1 = phys.gen.strings()
        self.lam0 = phys.gen0.lambda_measure()
        self.lam1 = phys.gen.lambda_measure()
        self.orig = core.origins(phys.ev, phys.gen)

    def draw_strings(self, ax, ev, chains, alpha):
        for ch in chains:
            pts = [(rapidity(ev.p[i].p), core.phi(ev.p[i].p)) for i in ch]
            for (y0, f0), (y1, f1) in zip(pts, pts[1:]):
                d = f1 - f0
                if d > math.pi:
                    f1 -= 2 * math.pi
                elif d < -math.pi:
                    f1 += 2 * math.pi
                ax.plot([y0, y1], [f0, f1], color=COL_MPI, lw=0.9, alpha=alpha)
            if ev.p[ch[0]].id == 21 and len(ch) > 1:
                (y0, f0), (y1, f1) = pts[-1], pts[0]
                ax.plot([y0, y1], [f0, f1], color=COL_MPI, lw=0.9, alpha=alpha)
        for ch in chains:
            for i in ch:
                p = ev.p[i].p
                o = self.orig.get(i)
                col = {"hard": COL_FSR, "MPI": COL_MPI, "remnant": COL_REM}.get(o, COL_ISR)
                if ev.p[i].id in (5, -5) and o == "hard":
                    col = COL_B if ev.p[i].id == 5 else COL_BBAR
                ax.plot([rapidity(p)], [core.phi(p)], "o", color=col, ms=max(2.5, min(10, 2 + 2 * math.log10(1 + core.pt(p)))), mec="white",
                        mew=0.4, alpha=alpha)

    def draw(self, fig, k):
        ph, ev = self.phys, self.phys.ev
        t = smooth(ramp(k, int(0.3 * self.nframes), int(0.75 * self.nframes)))
        n_dec = len(getattr(ph.gen, "cr_decisions", []))
        n_merged = ph.summary["n_reconnected"]
        chrome(fig, 4, self.title,
               f"Before hadronisation PYTHIA may rearrange the colour lines. In the MPI-based model a system of scale pT is merged into a harder one with probability "
               f"P = (R pT0)^2/((R pT0)^2 + pT^2), R = {core.CR_RANGE}: soft systems almost always.\nIts gluons are inserted into the dipoles of the harder system where "
               f"they disturb it least, so the total string length lambda = sum ln(1 + m_ij^2/m0^2) drops: {self.lam0:.0f} -> {self.lam1:.0f} here, "
               f"{n_merged} of {n_dec} systems merged, {len(self.chains0)} -> {len(self.chains1)} strings.")
        ax = fig.add_axes([0.05, 0.14, 0.62, 0.76])
        ax.set_xlim(-10, 10)
        ax.set_ylim(-math.pi - 0.2, math.pi + 0.2)
        ax.set_xlabel("rapidity y", fontsize=10)
        ax.set_ylabel(r"$\phi$", fontsize=10)
        style_2d(ax)
        self.draw_strings(ax, ev, self.chains0, 1.0 - t)
        self.draw_strings(ax, ev, self.chains1, t)
        ax.set_title("colour strings (lines) between the final-state partons: before (fading out) and after reconnection (fading in)", fontsize=10)
        ax2 = fig.add_axes([0.74, 0.14, 0.23, 0.76])
        ax2.bar([0, 1], [self.lam0, self.lam1 * t + self.lam0 * (1 - t)], color=[TEXT2, COL_MPI], width=0.6)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["before", "after"], fontsize=10)
        ax2.set_ylabel(r"string length $\lambda$", fontsize=10)
        ax2.set_ylim(0, self.lam0 * 1.15)
        ax2.text(1, self.lam1 * t + self.lam0 * (1 - t) + 10, f"{self.lam1 * t + self.lam0 * (1 - t):.0f}", ha="center", fontsize=10, color=COL_MPI)
        ax2.text(0, self.lam0 + 10, f"{self.lam0:.0f}", ha="center", fontsize=10, color=TEXT2)
        style_2d(ax2)

    def keyframe(self):
        return self.nframes - 1


# --- scene 5: ensemble -----------------------------------------------------------------------------------
class EnsembleScene(Scene):
    number, title = 5, "the same event dressed again and again"

    def __init__(self, phys, fps, n_events: int):
        super().__init__(phys, fps)
        stats = json.loads(STATS_FILE.read_text()) if STATS_FILE.exists() else None
        if stats and len(stats["counts"]["n_mpi"]) >= n_events:
            c = stats["counts"]
            fam = stats["families"]["b"]["0.4"] if "0.4" in stats["families"]["b"] else stats["families"]["b"][0.4]
            self.n_mpi = np.array(c["n_mpi"][:n_events], dtype=float)
            self.ue = np.array(c["ue_transverse"][:n_events], dtype=float)
            self.cone = np.array(fam["ue_scalar"][:n_events], dtype=float)
            self.source = "standalone_mpi_stats_fixedb.stats.json"
        else:
            summaries = phys.generate_more(range(phys.seed, phys.seed + n_events))
            self.n_mpi = np.array([s["n_mpi"] for s in summaries], dtype=float)
            self.ue = np.array([s["ue"]["density_transverse"] for s in summaries], dtype=float)
            self.cone = np.array([next(p for p in s["partons"] if p["name"] == "b")["cones"][0.4]["pt_ue_scalar"] for s in summaries], dtype=float)
            self.source = f"{len(summaries)} events generated now"
        self.n = len(self.n_mpi)
        self.py = json.loads(PY_FILE.read_text()) if PY_FILE.exists() else None
        self.nframes = int(8 * fps)

    def draw(self, fig, k):
        ph = self.phys
        n_show = max(1, int(round(smooth(ramp(k, 0, self.nframes - int(1.5 * self.fps))) * self.n)))
        chrome(fig, 5, self.title,
               f"Seeds {ph.seed}.. at b = {ph.impact} <b>: the number of interactions, the transverse underlying-event density (partons not from the hard system, |eta| < 2.5, "
               "60-120 degrees from the leading b)\nand the extra scalar pT inside R = 0.4 around the b quark fluctuate strongly from event to event; their means settle "
               f"towards the values PYTHIA 8 gives for the same hard event ({self.source}).")
        series = [("number of interactions", self.n_mpi, "n_mpi", CMSSW_NMPI, "CMSSW event"),
                  ("transverse UE density  [GeV per unit eta-phi]", self.ue, "ue_transverse", None, None),
                  ("UE scalar pT inside R = 0.4 around the b  [GeV]", self.cone, None, None, None)]
        for j, (lab, v, pykey, ref, reflab) in enumerate(series):
            ax = fig.add_axes([0.06 + 0.32 * j, 0.42, 0.27, 0.48])
            vals = v[:n_show]
            bins = np.linspace(0, max(1.0, float(np.percentile(v, 99)) * 1.1), 21)
            ax.hist(vals, bins=bins, color=COL_MPI, alpha=0.7, label=f"this script ({n_show} events)")
            pyv = None
            if self.py:
                if pykey:
                    pyv = self.py["counts"].get(pykey)
                else:
                    fam = self.py["families"].get("b", {})
                    pyv = fam.get("0.4", fam.get(0.4, {})).get("ue_scalar")
            if pyv:
                w = np.ones(len(pyv)) * n_show / len(pyv)
                ax.hist(pyv, bins=bins, histtype="step", color=COL_PY, lw=1.6, ls="--", weights=w, label=f"PYTHIA 8 ({len(pyv)} events, scaled)")
            if ref is not None:
                ax.axvline(ref, color=TEXT, lw=1.2, ls=":")
                ax.text(ref, 0.95, reflab, transform=ax.get_xaxis_transform(), fontsize=8.5, ha="left", va="top")
            ax.set_xlabel(lab, fontsize=9)
            ax.legend(fontsize=8, loc="upper right")
            style_2d(ax)
            ax2 = fig.add_axes([0.06 + 0.32 * j, 0.14, 0.27, 0.2])
            run = np.cumsum(v) / np.arange(1, self.n + 1)
            ax2.plot(np.arange(1, n_show + 1), run[:n_show], color=COL_MPI, lw=1.8)
            if pyv:
                ax2.axhline(float(np.mean(pyv)), color=COL_PY, lw=1.2, ls="--")
                ax2.text(self.n, float(np.mean(pyv)), f" PYTHIA {np.mean(pyv):.2f}", fontsize=8, color=COL_PY, va="bottom", ha="right")
            ax2.set_xlim(0, self.n)
            ax2.set_ylim(0, max(run.max(), float(np.mean(pyv)) if pyv else 0) * 1.3)
            ax2.set_xlabel("events", fontsize=8.5)
            ax2.set_ylabel("running mean", fontsize=8.5)
            ax2.text(0.02, 0.9, f"{run[n_show - 1]:.2f}", transform=ax2.transAxes, fontsize=9, color=COL_MPI, va="top")
            style_2d(ax2, 7.5)

    def keyframe(self):
        return self.nframes - 1


# --- driver -------------------------------------------------------------------------------------------------
def build_scenes(names: List[str], phys: Physics, args) -> List[Scene]:
    out = []
    for n in names:
        if n == "overlap":
            out.append(OverlapScene(phys, args.fps))
        elif n == "chain":
            out.append(ChainScene(phys, args.fps))
        elif n == "remnants":
            out.append(RemnantScene(phys, args.fps))
        elif n == "strings":
            out.append(StringScene(phys, args.fps))
        elif n == "ensemble":
            out.append(EnsembleScene(phys, args.fps, args.events))
    return out


def frame_list(scenes: List[Scene], step: int):
    frames = []
    for sc in scenes:
        for k in range(0, sc.nframes, step):
            frames.append((sc, k))
    return frames


def render_frame(fig, sc: Scene, k: int):
    sc.draw(fig, k)


def write_video(scenes, out: Path, fps: int, dpi: int, step: int):
    fig = plt.figure(figsize=FIGSIZE)
    frames = frame_list(scenes, step)
    if out.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=max(1, fps // step))
    else:
        writer = animation.FFMpegWriter(fps=max(1, fps // step), bitrate=3500)
    with writer.saving(fig, str(out), dpi):
        for i, (sc, k) in enumerate(frames):
            render_frame(fig, sc, k)
            writer.grab_frame(facecolor="white")
            if i % 50 == 0:
                print(f"  frame {i}/{len(frames)}")
    plt.close(fig)
    print(f"wrote {out} ({len(frames)} frames)")


def write_frames(scenes, outdir: Path, dpi: int, step: int):
    outdir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=FIGSIZE)
    frames = frame_list(scenes, step)
    for i, (sc, k) in enumerate(frames):
        render_frame(fig, sc, k)
        fig.savefig(outdir / f"frame_{i:04d}_s{sc.number}_k{k:03d}.png", dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"wrote {len(frames)} PNG frames to {outdir}")


def write_stills(scenes, out: Path, dpi: int):
    from matplotlib.backends.backend_pdf import PdfPages
    fig = plt.figure(figsize=FIGSIZE)
    if out.suffix.lower() == ".pdf":
        with PdfPages(str(out)) as pdf:
            for sc in scenes:
                render_frame(fig, sc, sc.keyframe())
                pdf.savefig(fig, facecolor="white")
    else:
        for sc in scenes:
            render_frame(fig, sc, sc.keyframe())
            fig.savefig(out.with_name(f"{out.stem}_scene{sc.number}{out.suffix}"), dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"wrote stills to {out}")


def show_interactive(scenes, fps: int, step: int):
    matplotlib.use("MacOSX" if sys.platform == "darwin" else "TkAgg", force=True)
    import matplotlib.pyplot as plt_i
    fig = plt_i.figure(figsize=FIGSIZE)
    frames = frame_list(scenes, step)

    def update(i):
        sc, k = frames[i]
        render_frame(fig, sc, k)
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, repeat=True)  # noqa: F841
    plt_i.show()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", type=Path, default=HERE / "animate_standalone_multiparton_interactions.mp4")
    p.add_argument("--scene", default="all", help="all, or a comma list of overlap,chain,remnants,strings,ensemble (or 1..5)")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--dpi", type=int, default=100, help="100 gives 1280x720")
    p.add_argument("--step", type=int, default=1, help="render every step-th frame")
    p.add_argument("--fast", action="store_true", help="preview: --step 4 --dpi 60, fewer events in scene 5")
    p.add_argument("--frames-dir", type=Path, default=None)
    p.add_argument("--stills", type=Path, default=None, help="one key frame per scene (.pdf or .png)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("-i", "--input", type=Path, default=HERE.parent.parent / "PS" / "event_PS.lhe")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--impact", type=float, default=CMSSW_B)
    p.add_argument("--events", type=int, default=120, help="events shown in scene 5 (from the stats file when present)")
    args = p.parse_args()
    if args.fast:
        args.step, args.dpi, args.events = max(args.step, 4), min(args.dpi, 60), min(args.events, 24)
    names = list(SCENE_NAMES.values()) if args.scene == "all" else [SCENE_NAMES.get(s.strip(), s.strip()) for s in args.scene.split(",")]
    bad = [n for n in names if n not in SCENE_NAMES.values()]
    if bad:
        sys.exit(f"unknown scene(s) {bad}; choose from {list(SCENE_NAMES.values())}")
    phys = Physics(args.seed, args.impact, args.input)
    s = phys.summary
    print(f"event: seed {args.seed}, b = {args.impact} <b>, f = {phys.gen.fb:.3f}: {s['n_mpi'] - 1} secondary interactions, "
          f"{s['n_final']} final partons, lambda {s['lambda_nocr']:.0f} -> {s['lambda_cr']:.0f}")
    scenes = build_scenes(names, phys, args)
    print("frames: " + ", ".join(f"{sc.number} {SCENE_NAMES[str(sc.number)]} {sc.nframes}" for sc in scenes)
          + f"  ({sum(sc.nframes for sc in scenes) / args.fps:.1f} s at {args.fps} fps)")
    if args.show:
        show_interactive(scenes, args.fps, args.step)
        return
    if args.stills is not None:
        write_stills(scenes, args.stills, args.dpi)
    if args.frames_dir is not None:
        write_frames(scenes, args.frames_dir, args.dpi, args.step)
    elif not args.no_video:
        write_video(scenes, args.output, args.fps, args.dpi, args.step)


if __name__ == "__main__":
    main()
