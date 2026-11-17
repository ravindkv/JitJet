#!/usr/bin/env python3
"""Animated companion to standalone_G4.py: the gate at the border of the detector.

Six scenes, narrated, show the family of chapter 7 and the crowd of chapter 8 arriving at the gate of Geant4 and
the decision that is made there, member by member, before Geant4 takes its first step. Every number on screen and in
the narration is computed by standalone_G4.py on the record of chapter 8 (../../PU/event_PU.lhe) and its library.
The film picks up where the chapter-8 film ended (the crowd on the road, the border ahead) and ends with the CMS
detector of 2025 waiting for the primaries: the crossing itself is the next film.

  1  road       The three crowds arrive: the particles per unit of eta of the family, of the 63 in-time and of the
                897 out-of-time collisions build up, with the edges of the detector and of the gate.
  2  line       The eta cut is a line: the cylinder r < 2.9 cm to |z| = 354.8 cm (stretched), the vertices of the
                window along the beam, and the lines of the border particles, kept or dropped by where they are at
                354.8 cm, not by their angle.
  3  soft       The momentum spectrum with the 40 MeV cut, and the neutrinos of the family, killed before their
                first step: the missing transverse momentum of the event decided at the gate.
  4  decays     The decays the generator made, across the beam: the fourteen weak decays of the family appear,
                every one inside the 2.9 cm cylinder, so every daughter starts as a primary.
  5  forward    The forward crowd in (|eta|, E): the neutrons and photons for the ZDC, the protons for the Roman
                pots, the rest down the beam pipe.
  6  ready      The primaries gathered in front of the CMS detector of 2025, its quadrant cut away: ready to cross.

Narration and music: as in the other chapter films (example/soundtrack.py): one line per phase (--print-script),
edge-tts or the macOS `say` voice, clips cached in build/narration/G4/, an .srt next to the .mp4.

Examples (matplotlib is needed; on this Mac use /usr/bin/python3):
  python3 animate_standalone_G4.py                       # -> .mp4 + .srt next to this file (1920x1080)
  python3 animate_standalone_G4.py --fast                # quick low-resolution preview
  python3 animate_standalone_G4.py --stills animate_standalone_G4_stills.pdf --no-video --no-narration --no-music
  python3 animate_standalone_G4.py --print-script
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
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(HERE))
import standalone_G4 as core  # noqa: E402
import draw_cms3d as cms  # noqa: E402

sys.path.insert(0, str(HERE.parent.parent))
import soundtrack as snd  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL_SIG, COL_IN, COL_OOT = "#d62728", "#8a7f5a", "#d4a017"
COL_KEPT, COL_DROP, COL_VF, COL_PPS, COL_NU = "#2a78d6", "#9a9a9a", "#2f8f5b", "#8e5bb5", "#eb6834"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d6d3ca"
FIGSIZE = (12.8, 7.2)
SCENE_NAMES = {"1": "road", "2": "line", "3": "soft", "4": "decays", "5": "forward", "6": "ready"}


def smooth(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ramp(k: float, k0: int, k1: int) -> float:
    return smooth((k - k0) / max(k1 - k0, 1))


class Physics:
    def __init__(self, record: Path, library: Path):
        args = argparse.Namespace(input=record, library=library, eta_cut=core.ETA_CUT, eta_hector=core.ETA_HECTOR, r_dec=core.R_DEC, p_min=core.P_MIN)
        self.st = core.Stage(args)
        self.R = self.st.compute()
        self.c, self.g, self.G = self.st.crowd, self.st.g, self.st.gate
        self.L = self.R["L"]
        self.view = cms.View()
        self.faces = cms.scene(self.view)


def chrome(fig, scene_no: int, title: str, caption: str):
    fig.clf()
    fig.patch.set_facecolor("white")
    fig.text(0.02, 0.965, f"Scene {scene_no}: {title}", fontsize=15, weight="bold", color=TEXT, va="center")
    fig.text(0.98, 0.965, "standalone_G4.py", fontsize=9.5, color=TEXT2, ha="right", va="center")
    fig.text(0.02, 0.03, caption, fontsize=10.5, color=TEXT, va="bottom", wrap=True)


def style_2d(ax, fontsize=9):
    ax.tick_params(labelsize=fontsize, colors=TEXT2)
    for sp in ax.spines.values():
        sp.set_color(TEXT2)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def counter(fig, x, y, label, value, color):
    fig.text(x, y, label, fontsize=10, color=TEXT2, va="center")
    fig.text(x, y - 0.04, value, fontsize=17, color=color, va="center", weight="bold")


class Scene:
    number, title, nframes = 0, "", 1

    def __init__(self, phys: Physics, fps: int):
        self.phys, self.fps = phys, fps

    def keyframe(self) -> int:
        return self.nframes - 1

    def phases(self) -> List[Tuple[str, int, int]]:
        return [("all", 0, self.nframes)]


# --- scene 1: the three crowds arrive ---------------------------------------------------------------------
class RoadScene(Scene):
    number, title = 1, "the road: three crowds arrive at the border"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.k1, self.k2 = 4 * fps, 9 * fps
        self.nframes = 14 * fps

    def phases(self):
        return [("signal", 0, self.k1), ("intime", self.k1, self.k2), ("oot", self.k2, self.nframes)]

    def draw(self, fig, k):
        ph = self.phys
        c = ph.c
        chrome(fig, 1, self.title, "Each crowd at its own vertex along the beam; the gate stands at |eta| = 5.5, where the particles thin out and the energy piles up.")
        ax = fig.add_axes([0.07, 0.14, 0.62, 0.74])
        style_2d(ax)
        bins = np.linspace(-13, 13, 53)
        w = bins[1] - bins[0]
        parts = [(0, COL_SIG, "the family", 1.0, ramp(k, 0, self.k1)), (1, COL_IN, f"in time, per collision ({ph.st.n_intime})", 1.0 / ph.st.n_intime, ramp(k, self.k1, self.k2)),
                 (2, COL_OOT, f"out of time, per collision ({ph.st.n_oot})", 1.0 / max(ph.st.n_oot, 1), ramp(k, self.k2, self.nframes - fps_hold(self.fps)))]
        for blk, col, lab, scale, f in parts:
            if f <= 0:
                continue
            m = (c.block == blk) & ~c.nu
            h, _ = np.histogram(c.eta[m], bins=bins)
            ax.stairs(h * scale / w * f, bins, color=col, lw=2.0, label=lab)
        for x, lab, ls in core_edges():
            for s in (-1, 1):
                ax.axvline(s * x, color=TEXT if ls == "-" else TEXT2, lw=1.4 if ls == "-" else 0.9, ls=ls)
            if lab:
                ax.text(x + 0.1, 220, lab, rotation=90, va="top", fontsize=9, color=TEXT if ls == "-" else TEXT2)
        ax.set_yscale("log")
        ax.set_ylim(0.05, 300)
        ax.set_xlim(-13, 13)
        ax.set_xlabel(r"$\eta$", fontsize=12)
        ax.set_ylabel(r"stable particles per unit of $\eta$", fontsize=11)
        ax.legend(loc="lower center", fontsize=10, frameon=True)
        L = ph.L
        counter(fig, 0.73, 0.82, "the family", f"{L['signal']['stable']['n']} stable, 13.6 TeV", COL_SIG)
        if k >= self.k1:
            counter(fig, 0.73, 0.68, f"in time: {ph.st.n_intime} collisions", f"{L['intime']['stable']['n']:,} stable", COL_IN)
        if k >= self.k2:
            counter(fig, 0.73, 0.54, f"out of time: {ph.st.n_oot} collisions", f"{L['oot']['stable']['n']:,} stable", COL_OOT)
            fig.text(0.73, 0.40, "25 ns x bx early or late:\nthe same gate, the same rules", fontsize=10.5, color=TEXT2, va="top")


def fps_hold(fps):
    return fps


def core_edges():
    return [(2.5, "tracker", ":"), (3.0, "endcaps", ":"), (core.ETA_HF_FRONT, "HF hole", "--"), (core.ETA_CUT, "the gate 5.5", "-"), (core.ETA_HECTOR, "8.2", "-")]


# --- scene 2: the line ------------------------------------------------------------------------------------
class LineScene(Scene):
    number, title = 2, "the eta cut is a line, not an angle"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        c, g, G = phys.c, phys.g, phys.G
        a = np.abs(c.eta)
        m = (c.block > 0) & (a > G.eta_cut - 0.02) & (a < G.eta_cut + 0.02) & ~g["very_forward"] & (c.p4[:, 3] > 0)
        self.idx = np.where(m)[0]
        self.k1, self.k2 = 3 * fps, 10 * fps
        self.nframes = 14 * fps

    def phases(self):
        return [("cylinder", 0, self.k1), ("lines", self.k1, self.k2), ("verdict", self.k2, self.nframes)]

    def draw(self, fig, k):
        ph = self.phys
        c, g, G = ph.c, ph.g, ph.G
        chrome(fig, 2, self.title, f"Followed from its own vertex to |z| = {G.z_lmax:.1f} cm: still inside r < {G.r} cm, dropped; outside, handed over. "
                                   f"In our window {len(ph.R['line_vs_angle']['kept_beyond'])} members are kept beyond |eta| = 5.5 and {len(ph.R['line_vs_angle']['dropped_inside'])} dropped inside it.")
        ax = fig.add_axes([0.07, 0.15, 0.60, 0.73])
        style_2d(ax)
        ax.axhspan(0, G.r, color="#fbe3e3", zorder=0)
        ax.axhspan(0, core.R_BEAMPIPE, color="#e6e6e6", zorder=0)
        ax.plot([-20, G.z_lmax], [G.r, G.r], color="#cc0000", lw=1.8)
        ax.plot([G.z_lmax, G.z_lmax], [0, G.r], color="#cc0000", lw=1.8)
        ax.text(5, G.r + 0.05, f"the cylinder of the gate, r < {G.r} cm", color="#cc0000", fontsize=10)
        ax.text(G.z_lmax - 2, 0.15, f"$Z_{{\\max}}$ = R sinh 5.5 = {G.z_lmax:.1f} cm", ha="right", fontsize=10, color="#cc0000")
        zs = np.array([col["z"] for col in ph.st.collisions])
        ax.scatter(zs, np.zeros_like(zs) + 0.02, s=6, color=TEXT2, zorder=3)
        f = ramp(k, self.k1, self.k2)
        n = int(round(f * len(self.idx)))
        verdict = k >= self.k2
        for j in self.idx[:n]:
            z0 = c.vtx[j, 2]
            z1 = 380.0
            r1 = (z1 - z0) / math.sinh(abs(c.eta[j]))
            col = (COL_KEPT if g["primary"][j] else COL_DROP) if verdict else "#555555"
            ax.plot([z0, z1], [0, r1], color=col, lw=0.6, alpha=0.7)
            if verdict:
                ax.plot(G.z_lmax, g["r_line"][j], "o", ms=3.5, color=col)
        ax.set_xlim(-20, 390)
        ax.set_ylim(0, 3.4)
        ax.set_xlabel("z [cm]", fontsize=11)
        ax.set_ylabel("r [cm] (stretched)", fontsize=11)
        a2 = fig.add_axes([0.72, 0.40, 0.25, 0.42])
        style_2d(a2, 8)
        zv = np.linspace(-15, 15, 100)
        a2.plot(zv, np.arcsinh((G.z_lmax - zv) / G.r), color=COL_KEPT, lw=1.6, label=r"towards $+z$")
        a2.plot(zv, np.arcsinh((G.z_lmax + zv) / G.r), color=COL_NU, lw=1.6, label=r"towards $-z$")
        a2.axhline(G.eta_cut, color=TEXT2, lw=0.8, ls=":")
        a2.set_xlabel(r"vertex $z_v$ [cm]", fontsize=9)
        a2.set_ylabel(r"effective cut $\eta_c$", fontsize=9)
        a2.legend(fontsize=8, loc="upper center")
        fig.text(0.72, 0.30, r"$\eta_c = \sinh^{-1}[(Z_{\max} \mp z_v)/R] \simeq 5.5 \mp z_v / Z_{\max}$", fontsize=11, color=TEXT)
        if verdict:
            nk = int(g["primary"][self.idx].sum())
            fig.text(0.72, 0.22, f"{n} border lines towards +z:\n{nk} handed over (blue), {n - nk} dropped (grey)", fontsize=10.5, color=TEXT2, va="top")


# --- scene 3: soft and invisible --------------------------------------------------------------------------
class SoftScene(Scene):
    number, title = 3, "the soft and the invisible"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.k1 = 5 * fps
        self.nframes = 11 * fps

    def phases(self):
        return [("soft", 0, self.k1), ("neutrinos", self.k1, self.nframes)]

    def draw(self, fig, k):
        ph = self.phys
        c, g = ph.c, ph.g
        L = ph.L
        chrome(fig, 3, self.title, f"Below 40 MeV: {L['signal']['drops']['p']['n']} of the family, {L['intime']['drops']['p']['n']} of the in-time crowd. "
                                   f"The {L['signal']['drops']['nu']['n']} neutrinos of the family pass every cut and are killed before their first step.")
        ax = fig.add_axes([0.07, 0.15, 0.40, 0.73])
        style_2d(ax)
        m0 = c.block < 2
        bins = np.logspace(-3.3, 3.5, 70)
        f = ramp(k, 0, self.k1 - self.fps)
        for kind, col in (("photon", "#2f8f5b"), ("charged hadron", TEXT), ("neutral hadron", "#6f6f6f")):
            m = m0 & (c.kind == kind)
            h, _ = np.histogram(c.p[m], bins=bins)
            ax.stairs(h * f, bins, color=col, lw=1.6, label=f"{kind}s")
        ax.axvline(core.P_MIN, color="#cc0000", lw=1.6)
        ax.text(core.P_MIN * 1.15, 2500, "40 MeV", color="#cc0000", fontsize=11)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.5, 5e3)
        ax.set_xlabel("|p| [GeV]", fontsize=11)
        ax.set_ylabel("stable particles (family + in-time crowd)", fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        a2 = fig.add_axes([0.55, 0.15, 0.42, 0.73])
        style_2d(a2)
        a2.set_xlim(-6, 6)
        a2.set_ylim(-math.pi, math.pi)
        a2.set_xlabel(r"$\eta$", fontsize=11)
        a2.set_ylabel(r"$\phi$", fontsize=11)
        sig = np.where(c.block == 0)[0]
        prim = sig[g["primary"][sig]]
        a2.scatter(np.clip(c.eta[prim], -5.95, 5.95), c.phi[prim], s=np.clip(4 + 12 * np.log10(1 + c.pt[prim]), 3, 60), color="#7a7a7a", lw=0)
        if k >= self.k1:
            nu = sig[c.nu[sig]]
            u = ramp(k, self.k1, self.nframes - self.fps)
            a2.scatter(np.clip(c.eta[nu], -5.95, 5.95), c.phi[nu], s=180, facecolors="none", edgecolors=COL_NU, lw=2.0 * max(u, 0.2), label="neutrinos (killed)")
            for j in nu:
                a2.annotate(f"{core.dec.name(int(c.ids[j]))} {c.pt[j]:.1f}", (c.eta[j], c.phi[j]), xytext=(6, 6), textcoords="offset points", fontsize=9, color=COL_NU)
            vx, vy = c.p4[nu, 1].sum(), c.p4[nu, 2].sum()
            a2.text(-5.8, 2.8, f"missing pT decided here: {math.hypot(vx, vy):.1f} GeV", fontsize=11, color=COL_NU, weight="bold")
            a2.legend(fontsize=9, loc="lower left")


# --- scene 4: the decays ----------------------------------------------------------------------------------
class DecayScene(Scene):
    number, title = 4, "the decays the generator made"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.weak = sorted(phys.st.decays.weak(), key=lambda d: d["r_dec"])
        self.k1 = 9 * fps
        self.nframes = 13 * fps

    def phases(self):
        return [("decays", 0, self.k1), ("inside", self.k1, self.nframes)]

    def draw(self, fig, k):
        ph = self.phys
        G = ph.G
        pv = ph.st.pv
        far = ph.R["decays"]["far"]
        chrome(fig, 4, self.title, f"PYTHIA decays what lives less than 10 mm; {len(self.weak)} weak decays in the family, the farthest a D0 at r = {far['r_dec']:.2f} cm: "
                                   f"all inside the {G.r} cm cylinder, so every daughter starts as a primary where its parent decayed.")
        ax = fig.add_axes([0.05, 0.13, 0.45, 0.76])
        style_2d(ax)
        th = np.linspace(0, 2 * math.pi, 300)
        ax.fill(G.r * np.cos(th), G.r * np.sin(th), color="#fbe3e3", zorder=0)
        ax.plot(G.r * np.cos(th), G.r * np.sin(th), color="#cc0000", lw=1.8)
        ax.fill(core.R_BEAMPIPE * np.cos(th), core.R_BEAMPIPE * np.sin(th), color="#ececec", zorder=0)
        ax.plot(pv[0], pv[1], "+", color=TEXT, ms=12, mew=2)
        n = int(round(ramp(k, 0, self.k1) * len(self.weak)))
        for d in self.weak[:n]:
            x, y = d["dec"][0], d["dec"][1]
            ax.plot([pv[0], x], [pv[1], y], color=TEXT2, lw=0.8)
            ax.plot(x, y, "o", ms=7, color=COL_KEPT)
        ax.set_aspect("equal")
        ax.set_xlim(-3.3, 3.3)
        ax.set_ylim(-3.3, 3.3)
        ax.set_xlabel("x [cm]", fontsize=11)
        ax.set_ylabel("y [cm]", fontsize=11)
        ax.text(0, 3.0, f"r = {G.r} cm", ha="center", color="#cc0000", fontsize=11)
        a2 = fig.add_axes([0.57, 0.13, 0.40, 0.76])
        a2.axis("off")
        a2.text(0.0, 0.98, f"{'hadron':<12s}{'pT':>7s}{'eta':>7s}{'r_dec [cm]':>12s}", fontsize=11, family="monospace", weight="bold", va="top")
        for i, d in enumerate(self.weak[:n]):
            a2.text(0.0, 0.92 - 0.055 * i, f"{d['name']:<12s}{d['pt']:7.1f}{d['eta']:7.2f}{d['r_dec']:12.3f}", fontsize=11, family="monospace", va="top", color=COL_KEPT if d["inside"] else "#cc0000")
        if k >= self.k1:
            a2.text(0.0, 0.08, "P(r > 2.9 cm) = exp(-R m / (pT c tau))\nall inside: no predefined decays", fontsize=12, color=TEXT, weight="bold", va="bottom")


# --- scene 5: the forward ---------------------------------------------------------------------------------
class ForwardScene(Scene):
    number, title = 5, "the forward: the ZDC, the Roman pots, and the beam pipe"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.k1, self.k2 = 4 * fps, 8 * fps
        self.nframes = 12 * fps

    def phases(self):
        return [("crowd", 0, self.k1), ("zdc", self.k1, self.k2), ("pps", self.k2, self.nframes)]

    def draw(self, fig, k):
        ph = self.phys
        c, g = ph.c, ph.g
        L = ph.L
        chrome(fig, 5, self.title, f"In time: {L['intime']['very_forward']['n']} neutrons and photons handed over towards the ZDC, {L['intime']['pps']['n']} protons carried to the Roman pots, "
                                   f"{L['intime']['drops']['eta']['n']:,} members down the beam pipe with {L['intime']['drops']['eta']['e'] / 1000:.0f} TeV.")
        ax = fig.add_axes([0.07, 0.14, 0.62, 0.74])
        style_2d(ax)
        a = np.abs(c.eta)
        fw = (c.block > 0) & (a > 4.5)
        ax.scatter(a[fw], c.p4[fw, 0], s=2, color="#c8c8c8", lw=0, rasterized=True)
        hp = fw & g["primary"] & ~g["very_forward"]
        ax.scatter(a[hp], c.p4[hp, 0], s=3, color=COL_KEPT, lw=0, rasterized=True, label="handed over at the vertex")
        if k >= self.k1:
            m = fw & g["primary"] & g["very_forward"]
            ax.scatter(a[m], c.p4[m, 0], s=9, color=COL_VF, lw=0, label=r"n, $\bar n$, $\gamma$: handed over towards the ZDC")
        if k >= self.k2:
            m = fw & g["pps"]
            ax.scatter(a[m], c.p4[m, 0], s=22, color=COL_PPS, lw=0, label="protons: the Roman pots at 203 m")
        for x in (core.ETA_CUT, core.ETA_HECTOR):
            ax.axvline(x, color=TEXT, lw=1.3)
        ax.text(core.ETA_CUT + 0.08, 6000, "5.5", fontsize=11)
        ax.text(core.ETA_HECTOR + 0.08, 6000, "8.2", fontsize=11)
        ax.set_yscale("log")
        ax.set_xlim(4.5, 13.5)
        ax.set_ylim(0.05, 1e4)
        ax.set_xlabel(r"$|\eta|$", fontsize=12)
        ax.set_ylabel("E [GeV]", fontsize=11)
        ax.legend(loc="lower right", fontsize=10, markerscale=2)
        counter(fig, 0.73, 0.82, "dropped beyond 5.5 (in time)", f"{L['intime']['drops']['eta']['n']:,}", "#777777")
        if k >= self.k1:
            counter(fig, 0.73, 0.66, "handed over towards the ZDC", f"{L['intime']['very_forward']['n']} ({L['intime']['very_forward']['e'] / 1000:.0f} TeV)", COL_VF)
        if k >= self.k2:
            counter(fig, 0.73, 0.50, "to the Roman pots", f"{L['intime']['pps']['n']} ({L['intime']['pps']['e'] / 1000:.0f} TeV)", COL_PPS)


# --- scene 6: ready to cross ------------------------------------------------------------------------------
class ReadyScene(Scene):
    number, title = 6, "ready to cross: the primaries in front of CMS"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.k1 = 5 * fps
        self.nframes = 10 * fps

    def phases(self):
        return [("gather", 0, self.k1), ("waiting", self.k1, self.nframes)]

    def draw(self, fig, k):
        ph = self.phys
        L = ph.L
        chrome(fig, 6, self.title, f"The family hands {L['signal']['primary']['n']} of its {L['signal']['stable']['n']} members to Geant4, "
                                   f"{100 * L['signal']['fractions']['e']:.0f} % of its energy and {100 * L['signal']['fractions']['pt']:.0f} % of its pT; the crowd a quarter of its energy. The crossing is the next film.")
        ax = fig.add_axes([0.30, 0.10, 0.69, 0.80])
        ax.axis("off")
        bl, br = ph.view.project([0, 0, -15]), ph.view.project([0, 0, 15])
        ax.plot([bl[0], br[0]], [bl[1], br[1]], color="#555555", lw=1.0, zorder=0)
        u = ramp(k, 0, self.k1)
        cms.draw_matplotlib(ax, ph.faces, ph.view, alpha=0.25 + 0.75 * u)
        ax.set_aspect("equal")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-9, 9.5)
        f = ramp(k, 0, self.k1)
        for i, (b, col, lab) in enumerate((("signal", COL_SIG, "the family"), ("intime", COL_IN, f"in time ({ph.st.n_intime})"), ("oot", COL_OOT, f"out of time ({ph.st.n_oot})"))):
            n = int(round(f * L[b]["primary"]["n"]))
            counter(fig, 0.03, 0.80 - 0.16 * i, lab + ": primaries", f"{n:,}", col)
        if k >= self.k1:
            fig.text(0.03, 0.28, "the tracker, ECAL, HCAL,\nthe solenoid, the iron and\nthe muon chambers, HF:\nthe CMS detector of 2025", fontsize=10.5, color=TEXT2, va="top")


def narration(ph: Physics, scenes) -> Dict[str, str]:
    L = ph.L
    s, i, o = L["signal"], L["intime"], L["oot"]
    far = ph.R["decays"]["far"]
    nu = [j for j in np.where((ph.c.block == 0) & ph.c.nu)[0]]
    vx, vy = ph.c.p4[nu, 1].sum(), ph.c.p4[nu, 2].sum()
    lines = {
        "road_signal": (f"The crowd has arrived, and the road ends at a gate. ... Before the detector can see a single particle, every member of the event is handed to Geant4, "
                        f"the toolkit that carries particles through matter, and the hand-over is a selection. Our family comes first: {s['stable']['n']} stable particles, thirteen point six T e V."),
        "road_intime": (f"Then the in-time crowd of chapter eight: {ph.st.n_intime} collisions, {i['stable']['n']:,} particles. Look at where they go: the plateau in the middle, and the beam remnants far forward."),
        "road_oot": (f"And the out-of-time crowd, {ph.st.n_oot} collisions of the neighbouring crossings, {o['stable']['n']:,} particles, simulated with the same rules. "
                     "The gate stands at an eta of five and a half, where the particles are already thinning out, and the energy is not."),
        "line_cylinder": (f"The rule of the gate is a cylinder. ... Two point nine centimetres around the beam, out to {ph.G.z_lmax:.0f} centimetres along it, which is two point nine times the sinh of five and a half."),
        "line_lines": ("Every particle is followed in a straight line from its own vertex. If the line is still inside the cylinder at the end, the particle is dropped. "
                       "From the centre of the detector, that is an eta cut at five and a half. From a vertex a few centimetres away, it is not."),
        "line_verdict": (f"So the verdict depends on where the collision happened. In our window, {len(ph.R['line_vs_angle']['kept_beyond'])} members beyond five and a half are kept, "
                         f"and {len(ph.R['line_vs_angle']['dropped_inside'])} inside it are dropped. The cut is a line, not an angle."),
        "soft_soft": (f"Below forty M e V of momentum, nothing enters: {s['drops']['p']['n']} soft members of the family, most of them photons from neutral pion decays, "
                      f"and {i['drops']['p']['n']} of the crowd. A tenth of a GeV, all together."),
        "soft_neutrinos": (f"The neutrinos are different. They pass every cut, and the stacking action kills them before their first step. Our family loses {s['drops']['nu']['n']}, "
                           f"{s['drops']['nu']['e']:.0f} GeV of energy. Whatever transverse momentum they carry, {math.hypot(vx, vy):.0f} GeV here, is missing from every detector, from now on."),
        "decays_decays": (f"Some members never reach the gate themselves. Pythia decays every hadron that lives less than ten millimetres. Our family had {len(ph.st.decays.weak())} weak decays, "
                          "B and D hadrons, a b baryon, a tau, and each happened a few millimetres from the collision."),
        "decays_inside": (f"The farthest, a D zero, decayed one point five centimetres from the beam: inside the cylinder. So every daughter starts as a primary, where its parent died. "
                          "Only a decay outside the cylinder would hand the parent itself to Geant4."),
        "forward_crowd": (f"Far forward the crowd is all energy. {i['drops']['eta']['n']:,} members of the in-time crowd go down the beam pipe, {i['drops']['eta']['e'] / 1000:.0f} T e V, and no detector will ever see them."),
        "forward_zdc": (f"But beyond an eta of eight point two, neutrons and photons are handed over, {i['very_forward']['n']} of them here, for the zero degree calorimeter one hundred and forty metres away. "
                        "By default the simulation stops them in the dead region of the beam line, a metre before they arrive; a study of the forward region lets them through."),
        "forward_pps": (f"And the protons that survived the collision, with more than three T e V along the beam, are carried by the optics of the L H C to the Roman pots, two hundred and three metres away, "
                        f"and handed to Geant4 there. {i['pps']['n']} of them."),
        "ready_gather": (f"What is left gathers in front of the detector. {s['primary']['n']} members of the family: ninety percent of them, {100 * s['fractions']['e']:.0f} percent of the energy, "
                         f"{100 * s['fractions']['pt']:.0f} percent of the transverse momentum. And {i['primary']['n']:,} and {o['primary']['n']:,} members of the two crowds, a quarter of their energy."),
        "ready_waiting": ("The detector of 2025 is waiting: the tracker, the calorimeters, the solenoid, the iron and the muon chambers."),
        "outro": ("Nothing has moved yet. The hand-over is a selection, not a transformation. ... The crossing, with the field, the material and the showers, is the next film."),
    }
    return lines


def score_spec(film: snd.Film) -> snd.ScoreSpec:
    return snd.film_score(film, [snd.Dm, snd.Bb, snd.F, snd.C, snd.Gm, snd.A])


def build_scenes(names, phys, args):
    cls = dict(road=RoadScene, line=LineScene, soft=SoftScene, decays=DecayScene, forward=ForwardScene, ready=ReadyScene)
    return [cls[n](phys, args.fps) for n in names]


def write_stills(scenes, out: Path, dpi: int):
    from matplotlib.backends.backend_pdf import PdfPages
    fig = plt.figure(figsize=FIGSIZE)
    with PdfPages(str(out)) as pdf:
        for sc in scenes:
            sc.draw(fig, sc.keyframe())
            pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    print(f"wrote stills to {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", type=Path, default=HERE / "animate_standalone_G4.mp4")
    p.add_argument("--scene", default="all")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--stills", type=Path, default=None)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--cache-dir", type=Path, default=REPO / "build" / "narration" / "G4")
    snd.add_narration_args(p)
    p.add_argument("-i", "--input", type=Path, default=HERE.parent.parent / "PU" / "event_PU.lhe")
    p.add_argument("--library", type=Path, default=HERE.parent.parent / "PU" / "pileup_library.lhe.gz")
    args = p.parse_args()
    if args.fast:
        args.step, args.dpi = max(args.step, 4), min(args.dpi, 60)
    snd.narration_defaults(args)
    fps = args.fps / args.step
    args.fps = snd.NOMINAL_FPS
    names = list(SCENE_NAMES.values()) if args.scene == "all" else [SCENE_NAMES.get(s.strip(), s.strip()) for s in args.scene.split(",")]
    phys = Physics(args.input, args.library)
    scenes = build_scenes(names, phys, args)
    lines = narration(phys, scenes)
    if args.print_script:
        snd.print_script(lines)
        return
    if args.stills is not None:
        write_stills(scenes, args.stills, args.dpi)
    if args.no_video:
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None and not args.no_narration:
        sys.exit("ffmpeg not found on PATH (needed for the narration clips)")
    clips = snd.load_narration(lines, args, args.cache_dir, ffmpeg) if ffmpeg else {}
    film = snd.Film(scenes, snd.clip_lengths(clips), {int(n): name for n, name in SCENE_NAMES.items()}, slow_max=4.0)
    print(f"film: {film.describe()}")
    score = None if args.no_music else score_spec(film)
    audio = None
    tmp = tempfile.TemporaryDirectory(prefix="jitjet_g4_")
    if ffmpeg and (clips or score):
        audio = Path(tmp.name) / "soundtrack.wav"
        snd.mix_audio(film.duration, film.voice_at, clips, score, args.music_db, audio)
        if clips:
            snd.write_srt(film.voice_at, clips, lines, args.output.with_suffix(".srt"))
    snd.render_video(film, args.output, fps, args.dpi, audio, lambda fig, sc, k, t: sc.draw(fig, k), FIGSIZE, "b bbar event, the hand-over to Geant4", jobs=args.jobs)
    print(f"wrote {args.output} ({film.duration:.1f} s)")


if __name__ == "__main__":
    main()
