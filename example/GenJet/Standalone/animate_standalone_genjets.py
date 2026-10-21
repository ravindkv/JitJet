#!/usr/bin/env python3
"""Animated companion to standalone_genjets.py: the family photograph.

Six scenes, narrated, show how the 749 particles of chapter 6 become generator-level jets and how a jet
gets its flavour. Every number on screen and in the narration is computed with the classes of that
script on the seed-1 record (../../Decay/event_Decay.lhe, the record ../event_GenJet.lhe). The film picks
up where the chapter-6 film ended (the grandchildren are born) and ends at the next stop, the crowd on
the road of chapter 8.

  1  photograph  The final state appears in the (eta, phi) plane, hardest particles first, then the 42
                 circles of the anti-kT jets settle over it: the photograph, and the frame of every
                 later comparison.
  2  gather      Zoom on the b quark's jet: the 36 merges of anti-kT in their order, the hard core first,
                 the soft periphery last, the running pT, the ladder of distances, the circle of R = 0.4
                 drawn when d_iB wins.
  3  choices     The same family, other photographs: the radius grows from 0.1 to 0.8 and the pT with it;
                 a decay of the record in which the B went semileptonic shows what the neutrino choice
                 costs; the threshold decides how many jets the picture holds.
  4  ghosts      The B hadron shrinks to a ghost of pT x 1e-18 and still lands in the jet: hadronFlavour
                 5; the b quark's ghost gives partonFlavour 5; at the other end of the event the
                 gluon-splitting pair shares one jet and the oscillated B gives the bbar jet a b.
  5  whisper     A soft B0 Bbar0 pair appears between two light jets: the ghost and the net labels flip,
                 the IFN neutralisation distance cancels the pair, and the scan in the pair's pT shows
                 which definitions are infrared safe.
  6  ensemble    The same primary hadrons decayed and clustered again and again: the response of the
                 AK4, AK4-no-neutrino and AK8 jets to the b quark.

Narration and music: as in the other chapter films (example/soundtrack.py): one line per phase
(--print-script), edge-tts or the macOS `say` voice, clips cached in build/narration/GenJet/, an .srt
next to the .mp4; the score has a chord per scene and a thud when the soft pair flips the labels.

Examples (matplotlib is needed; on this Mac use /usr/bin/python3):
  python3 animate_standalone_genjets.py                  # -> .mp4 + .srt next to this file (1920x1080)
  python3 animate_standalone_genjets.py --fast           # quick low-resolution preview
  python3 animate_standalone_genjets.py --scene gather,ghosts -o g.gif   # silent
  python3 animate_standalone_genjets.py --stills stills.pdf --no-video
  python3 animate_standalone_genjets.py --print-script
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "Had"))
import standalone_genjets as core  # noqa: E402
import plot_journey_Had as jh  # noqa: E402

sys.path.insert(0, str(HERE.parent.parent))
import soundtrack as snd  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

COL_B, COL_BBAR, COL_HAD, COL_DEC, COL_UNCL = "#ff0000", "#0000ff", "#9b4dff", "#8a8a8a", "#c0c0c0"
COL_STR, COL_MINI, COL_A, COL_EVENT, COL_PY = "#2a78d6", "#eb6834", "#8e5bb5", "#c0392b", "#7f7f7f"
COL_NU, COL_LEP, COL_CH, COL_GAM, COL_GLU = "#1b9e77", "#d95f02", "#7570b3", "#e6ab02", "#2f8f5b"
PALETTE = ["#00999a", "#9b59d6", "#3b9ae1", "#ff1493", "#2e9e2e", "#a3a300"]
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d6d3ca"
FIGSIZE = (12.8, 7.2)
STATS_FILE = HERE / "standalone_genjet_stats.stats.json"
SCENE_NAMES = {"1": "photograph", "2": "gather", "3": "choices", "4": "ghosts", "5": "whisper", "6": "ensemble"}


def smooth(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ramp(k: float, k0: int, k1: int) -> float:
    return smooth((k - k0) / max(1, k1 - k0))


def tex(name: str) -> str:
    return jh.hadron_tex(name)


class Physics:
    def __init__(self, input_path: Path):
        self.args = SimpleNamespace(input=input_path, algorithm="antikt", ptmin=core.JET_PT_MIN, radius=None, no_flavour_kt=True, soft_pair=None, soft_sep=0.5,
                                    repeat=0, seed=1)
        self.ev, self.st, _ = core.run(self.args, 0)
        self.summary = core.stage_summary(self.st)
        self.b, self.bbar = self.summary["b"]["b"], self.summary["b"]["bbar"]
        self.stats = json.loads(STATS_FILE.read_text()) if STATS_FILE.exists() else None
        # the soft pair of the infrared test at three strengths
        self.pairs = {}
        for p in (0.5, 2.0, 8.0):
            extra, between = core.soft_pair(self.ev, self.st.coll["ak4"].info, p, 0.5)
            st2 = core.Stage(self.ev, self.args, verbose=0, soft_pair=extra)
            self.pairs[p] = (extra, between, st2)
        self.between = self.pairs[0.5][1]
        # a decay of the record in which the b quark's B went semileptonic: the neutrino choice
        dargs = SimpleNamespace(input=HERE.parent.parent / "Had" / "event_Had.lhe", tau0max=core.dec.TAU0_MAX, limit_tau0=True, no_mix=False, stable_b=False, radius=0.4)
        self.semi = None
        for s in range(2, 60):
            e2, d2 = core.dec.run(dargs, s)
            st2 = core.Stage(e2, self.args, verbose=0)
            s2 = core.stage_summary(st2)
            j = s2["b"]["b"]["jets"]["ak4"]
            if j and j["pt_nu"] > 5.0:
                self.semi = (s, s2)
                break


# --- drawing helpers ------------------------------------------------------------------------------------
def chrome(fig, scene_no: int, title: str, caption: str):
    fig.clf()
    fig.patch.set_facecolor("white")
    fig.text(0.02, 0.965, f"Scene {scene_no}: {title}", fontsize=15, weight="bold", color=TEXT, va="center")
    fig.text(0.98, 0.965, "standalone_genjets.py", fontsize=9.5, color=TEXT2, ha="right", va="center")
    fig.text(0.02, 0.03, caption, fontsize=10.5, color=TEXT, va="bottom", wrap=True)


def style_2d(ax, fontsize=8):
    ax.tick_params(labelsize=fontsize, colors=TEXT2)
    for sp in ax.spines.values():
        sp.set_color(TEXT2)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def msize(pt: float, scale=1.0) -> float:
    return scale * max(6.0, min(260.0, 6.0 + 40.0 * math.log10(1 + pt)))


def jet_colour(rank, roles):
    if roles.get(rank) == "b":
        return COL_B
    if roles.get(rank) == "bbar":
        return COL_BBAR
    others = [r for r in range(rank + 1) if r not in roles]
    k = others.index(rank) if rank in others else 99
    return PALETTE[k] if k < len(PALETTE) else COL_DEC


def roles_of(phys) -> dict:
    return {phys.b["jets"]["ak4"]["rank"]: "b", phys.bbar["jets"]["ak4"]["rank"]: "bbar"}


def kin(p):
    return core.eta(p), core.phi(p), core.pt(p)


class Scene:
    number, title, nframes = 0, "", 1

    def __init__(self, phys: Physics, fps: int):
        self.phys, self.fps = phys, fps

    def draw(self, fig, k: float):
        raise NotImplementedError

    def keyframe(self) -> int:
        return self.nframes - 1

    def phases(self) -> List[Tuple[str, int, int]]:
        return [("all", 0, self.nframes)]


# --- scene 1: the photograph ------------------------------------------------------------------------------
class PhotographScene(Scene):
    number, title = 1, "the photograph: 749 particles, 42 jets"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.nframes = 10 * fps
        ev = phys.ev
        c = phys.st.coll["ak4"]
        self.parts = sorted([(i,) + kin(ev.p[i].p) for i in ev.final_indices()], key=lambda t: -t[3])
        self.jet_of = c.jet_of
        self.jets = c.info
        self.roles = roles_of(phys)
        self.k_crowd, self.k_jets = 4 * fps, 8 * fps

    def phases(self):
        return [("crowd", 0, self.k_crowd), ("jets", self.k_crowd, self.nframes)]

    def keyframe(self):
        return self.nframes - 1

    def draw(self, fig, k):
        s = self.phys.summary
        c4 = s["collections"]["ak4"]
        chrome(fig, self.number, self.title,
               f"Stage 5 left {s['n_final']} particles ({s['n_neutrinos']} neutrinos). Anti-$k_T$ with R = 0.4 gathers {c4['n_clustered']} of them into {c4['n_jets']} jets above 3 GeV "
               f"({c4['n_jets_10']} above 10 GeV, {c4['n_jets_30']} above 30); the rest stays loose. The $b$ quark's family is jet [{self.phys.b['jets']['ak4']['rank']}], "
               f"{self.phys.b['jets']['ak4']['pt']:.1f} GeV, the $\\bar b$'s jet [{self.phys.bbar['jets']['ak4']['rank']}], {self.phys.bbar['jets']['ak4']['pt']:.1f} GeV.")
        ax = fig.add_axes([0.06, 0.14, 0.9, 0.75])
        style_2d(ax, 9)
        n = int(len(self.parts) * ramp(k, 0, self.k_crowd))
        f_jets = ramp(k, self.k_crowd, self.k_jets)
        for (i, e, f, pt) in self.parts[:n]:
            rank = self.jet_of.get(i, -1)
            col = jet_colour(rank, self.roles) if (rank >= 0 and f_jets > 0) else COL_DEC
            if f_jets > 0 and rank < 0:
                ax.scatter(min(max(e, -6), 6), f, s=msize(pt) * 0.5, facecolors="none", edgecolors=COL_UNCL, lw=0.8, zorder=2)
            else:
                ax.scatter(min(max(e, -6), 6), f, s=msize(pt) * 0.5, color=col, alpha=0.85, lw=0, zorder=3)
        if f_jets > 0:
            nj = int(len(self.jets) * f_jets)
            for j in self.jets[:nj]:
                col = jet_colour(j["rank"], self.roles)
                lead = col != COL_DEC
                ax.add_patch(Circle((j["eta"], j["phi"]), 0.4, fill=False, edgecolor=col, lw=1.8 if lead else 0.8, ls="-" if lead else "--", zorder=4))
                ax.scatter(j["eta"], j["phi"], s=msize(j["pt"]) * 0.8, marker="*", color=col, zorder=5, lw=0)
                if j["pt"] > 5:
                    ax.annotate(f"[{j['rank']}] {j['pt']:.1f}", (j["eta"], j["phi"]), xytext=(4, 4), textcoords="offset points", fontsize=8 if lead else 7, color=col, weight="bold" if lead else "normal")
        for q, rec in (("b", self.phys.b), ("bbar", self.phys.bbar)):
            ax.scatter(rec["eta_quark"], rec["phi_quark"], s=160, marker="*", facecolors="none", edgecolors="black", lw=1.2, zorder=6)
            ax.annotate(("$b$" if q == "b" else r"$\bar b$") + f" quark {rec['pt_quark']:.1f}", (rec["eta_quark"], rec["phi_quark"]), xytext=(6, -12), textcoords="offset points", fontsize=9)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-math.pi, math.pi)
        ax.set_xlabel(r"$\eta$", fontsize=11)
        ax.set_ylabel(r"$\phi$", fontsize=11)
        ax.set_title(f"{n} of {len(self.parts)} particles drawn (hardest first)" + (f"; {min(nj, len(self.jets))} of {len(self.jets)} anti-$k_T$ jets" if f_jets > 0 else ""), fontsize=11, color=TEXT)


# --- scene 2: gathering the b jet -------------------------------------------------------------------------
class GatherScene(Scene):
    number, title = 2, "anti-$k_T$ gathers the $b$ quark's family, hardest first"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.nframes = 14 * fps
        c = phys.st.coll["ak4"]
        self.c = c
        self.cl = c.cl
        self.jet = c.info[phys.b["jets"]["ak4"]["rank"]]
        self.steps = self.cl.merges_into(self.jet["index"])
        self.n = len(self.steps)
        self.e0, self.f0 = self.jet["eta"], self.jet["phi"]
        ev = phys.ev
        self.near = [(i,) + kin(ev.p[i].p) for i in ev.final_indices() if abs(core.eta(ev.p[i].p) - self.e0) < 1.1 and abs(core.dphi(core.phi(ev.p[i].p), self.f0)) < 1.1]
        self.k_core, self.k_all = 5 * fps, 11 * fps
        self.n_core = 12

    def phases(self):
        return [("core", 0, self.k_core), ("periphery", self.k_core, self.k_all), ("declared", self.k_all, self.nframes)]

    def keyframe(self):
        return self.k_all + self.fps

    def merged_at(self, k):
        if k < self.k_core:
            return int(self.n_core * ramp(k, 0, self.k_core))
        return self.n_core + int((self.n - self.n_core) * ramp(k, self.k_core, self.k_all))

    def draw(self, fig, k):
        m = self.merged_at(k)
        declared = k >= self.k_all
        j = self.jet
        chrome(fig, self.number, self.title,
               f"$d_{{ij}}$ = min($1/p_T^2$) $\\Delta R^2/R^2$ is smallest for the hardest pair that is close: the core of the jet (the $\\bar B^0$'s K$^-$, $\\pi^-$, photons) merges "
               f"first, the soft periphery joins last, each merge at a larger $\\Delta R$. After {self.n} merges nothing within R = 0.4 is left and $d_{{iB}}$ = $1/p_T^2$ wins: "
               f"the jet is declared with {j['pt']:.2f} GeV and {j['n']} particles, its axis {self.phys.b['jets']['ak4']['dr']:.3f} from the quark.")
        ax = fig.add_axes([0.05, 0.14, 0.48, 0.76])
        style_2d(ax, 9)
        cl = self.cl
        done = self.steps[:m]
        merged = set()
        for (_, i, jj, d, kk) in done:
            for x in cl.constituents(kk):
                merged.add(x)
        rows_merged = {self.c.tags[x][1] for x in merged if x < cl.n0 and self.c.tags[x][0] == "particle"}
        for (i, e, f, pt) in self.near:
            inside = i in rows_merged
            ax.scatter(e, f, s=msize(pt), color=COL_B if inside else COL_DEC, alpha=0.95 if inside else 0.5, lw=0, zorder=3 if inside else 2)
        if done:
            kk = done[-1][4]
            ax.scatter(cl.y[kk] * 0 + core.eta(cl.p4[kk]), cl.phi[kk], s=220, marker="*", color=COL_EVENT, zorder=6, lw=0)
            last = done[-1]
            soft = last[2] if cl.pt2[last[2]] < cl.pt2[last[1]] else last[1]
            ax.add_patch(Circle((core.eta(cl.p4[soft]), cl.phi[soft]), 0.03, fill=False, edgecolor=COL_GAM, lw=2, zorder=7))
        if declared:
            ax.add_patch(Circle((self.e0, self.f0), 0.4, fill=False, edgecolor=COL_B, lw=2.2, zorder=5))
        else:
            ax.add_patch(Circle((self.e0, self.f0), 0.4, fill=False, edgecolor=COL_B, lw=0.8, ls=":", zorder=5))
        ax.scatter(self.phys.b["eta_quark"], self.phys.b["phi_quark"], s=180, marker="*", facecolors="none", edgecolors="black", lw=1.2, zorder=6)
        ax.set_xlim(self.e0 - 1.0, self.e0 + 1.0)
        ax.set_ylim(self.f0 - 1.0, self.f0 + 1.0)
        ax.set_xlabel(r"$\eta$", fontsize=11)
        ax.set_ylabel(r"$\phi$", fontsize=11)
        pt_now = math.sqrt(cl.pt2[done[-1][4]]) if done else 0.0
        ax.set_title(f"merge {m} of {self.n}: pseudojet $p_T$ = {pt_now:.2f} GeV" + ("  -  declared a jet" if declared else ""), fontsize=11, color=TEXT)
        # right: the ladder
        ax2 = fig.add_axes([0.6, 0.14, 0.37, 0.76])
        style_2d(ax2, 9)
        ns = np.arange(1, self.n + 1)
        pts = [math.sqrt(cl.pt2[s[4]]) for s in self.steps]
        drs = [math.sqrt(core.dr2(cl.y[s[1]], cl.phi[s[1]], cl.y[s[2]], cl.phi[s[2]])) for s in self.steps]
        ax2.plot(ns[:m], pts[:m], "o-", color=COL_B, ms=4, mec="white", label="$p_T$ of the growing pseudojet")
        ax2.set_xlim(0, self.n + 1)
        ax2.set_ylim(0, 110)
        ax2.set_xlabel("merge number", fontsize=10)
        ax2.set_ylabel("$p_T$  [GeV]", fontsize=10, color=COL_B)
        ax3 = ax2.twinx()
        ax3.plot(ns[:m], drs[:m], "s", color=COL_STR, ms=4, label=r"$\Delta R$ of the merge")
        ax3.axhline(0.4, color=COL_STR, lw=0.8, ls=":")
        ax3.set_ylim(0, 0.5)
        ax3.set_ylabel(r"$\Delta R$", fontsize=10, color=COL_STR)
        ax3.grid(False)
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax3.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)
        if declared:
            ax2.text(0.03, 0.95, f"$d_{{iB}}$ = 1/$p_T^2$ = {1 / j['pt'] ** 2:.2e} wins:\njet [{j['rank']}], {j['pt']:.2f} GeV, {j['n']} particles", transform=ax2.transAxes, va="top", fontsize=10, color=COL_B)


# --- scene 3: the choices ---------------------------------------------------------------------------------
class ChoicesScene(Scene):
    number, title = 3, "the choices: the radius, the neutrinos, the threshold"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.nframes = 14 * fps
        ev = phys.ev
        j = phys.st.coll["ak4"].info[phys.b["jets"]["ak4"]["rank"]]
        self.e0, self.f0 = j["eta"], j["phi"]
        fin = ev.final_indices()
        self.parts = [(i,) + kin(ev.p[i].p) for i in fin if abs(core.eta(ev.p[i].p) - self.e0) < 1.3 and abs(core.dphi(core.phi(ev.p[i].p), self.f0)) < 1.3]
        self.radii = np.linspace(0.05, 0.9, 60)
        self.cone = []
        for R in self.radii:
            rows = [i for (i, e, f, pt) in self.parts if math.hypot(e - self.e0, core.dphi(f, self.f0)) < R]
            self.cone.append(core.pt(sum((ev.p[i].p for i in rows), np.zeros(4))))
        self.k_radius, self.k_nu = 5 * fps, 10 * fps

    def phases(self):
        return [("radius", 0, self.k_radius), ("neutrino", self.k_radius, self.k_nu), ("threshold", self.k_nu, self.nframes)]

    def keyframe(self):
        return self.k_nu + self.fps

    def draw(self, fig, k):
        b = self.phys.b
        j4, j4n, j8 = b["jets"]["ak4"], b["jets"]["ak4nonu"], b["jets"]["ak8"]
        semi = self.phys.semi
        R = 0.05 + 0.85 * ramp(k, 0, self.k_radius)
        f_nu = ramp(k, self.k_radius, self.k_nu)
        f_thr = ramp(k, self.k_nu, self.nframes - self.fps)
        cap = (f"Around the same axis the pT inside R grows with R: {j4['pt']:.1f} GeV at 0.4 (the AK4 jet), {j8['pt']:.1f} at 0.8 (the AK8 jet). Our $\\bar B^0$ decayed without a neutrino, "
               f"so the two AK4 collections agree here")
        if semi:
            s2 = semi[1]["b"]["b"]["jets"]
            cap += f"; in decay seed {semi[0]} of the same hadrons the B went semileptonic and the jet without neutrinos has {s2['ak4nonu']['pt']:.1f} GeV instead of {s2['ak4']['pt']:.1f}"
        cap += ". CMSSW's event: 117.8 GeV with its 37.5 GeV neutrino, 79.8 without."
        chrome(fig, self.number, self.title, cap)
        ax = fig.add_axes([0.05, 0.14, 0.42, 0.76])
        style_2d(ax, 9)
        for (i, e, f, pt) in self.parts:
            inside = math.hypot(e - self.e0, core.dphi(f, self.f0)) < R
            ax.scatter(e, f, s=msize(pt), color=COL_B if inside else COL_DEC, alpha=0.95 if inside else 0.45, lw=0, zorder=3)
        ax.add_patch(Circle((self.e0, self.f0), R, fill=False, edgecolor=COL_B, lw=2.2, zorder=5))
        for RR, col in ((0.4, COL_B), (0.8, COL_BBAR)):
            ax.add_patch(Circle((self.e0, self.f0), RR, fill=False, edgecolor=col, lw=0.8, ls=":", zorder=4))
        pt_R = float(np.interp(R, self.radii, self.cone))
        ax.set_xlim(self.e0 - 1.2, self.e0 + 1.2)
        ax.set_ylim(self.f0 - 1.2, self.f0 + 1.2)
        ax.set_xlabel(r"$\eta$", fontsize=11)
        ax.set_ylabel(r"$\phi$", fontsize=11)
        ax.set_title(f"R = {R:.2f}: {pt_R:.1f} GeV inside the circle", fontsize=11, color=TEXT)
        ax2 = fig.add_axes([0.55, 0.56, 0.42, 0.34])
        style_2d(ax2, 9)
        ax2.plot(self.radii[self.radii <= R + 1e-9], np.array(self.cone)[self.radii <= R + 1e-9], color=COL_B, lw=2)
        ax2.plot([0.4], [j4["pt"]], "s", color=COL_B, ms=7, label=f"AK4 jet {j4['pt']:.1f}")
        ax2.plot([0.8], [j8["pt"]], "s", color=COL_BBAR, ms=7, label=f"AK8 jet {j8['pt']:.1f}")
        ax2.axhline(b["pt_quark"], color=TEXT2, lw=0.8, ls=":")
        ax2.text(0.06, b["pt_quark"] + 1.5, f"$b$ quark {b['pt_quark']:.1f}", fontsize=8.5, color=TEXT2)
        ax2.set_xlim(0, 0.9)
        ax2.set_ylim(0, 125)
        ax2.set_xlabel("R", fontsize=10)
        ax2.set_ylabel("$p_T$ inside R  [GeV]", fontsize=10)
        ax2.legend(loc="lower right", fontsize=8.5)
        ax3 = fig.add_axes([0.55, 0.14, 0.42, 0.3])
        style_2d(ax3, 9)
        bars = [("our event\nAK4", j4["pt"], COL_B), ("our event\nAK4 no $\\nu$", j4n["pt"], COL_NU), ("our event\nAK8", j8["pt"], COL_BBAR)]
        if semi:
            s2 = semi[1]["b"]["b"]["jets"]
            bars += [(f"seed {semi[0]}\nAK4", s2["ak4"]["pt"], COL_B), (f"seed {semi[0]}\nAK4 no $\\nu$", s2["ak4nonu"]["pt"], COL_NU)]
        bars += [("CMSSW\nAK4", 117.79, COL_PY), ("CMSSW\nAK4 no $\\nu$", 79.82, COL_PY)]
        nb = int(len(bars) * f_nu) if f_nu > 0 else 0
        for x, (lab, v, col) in enumerate(bars[:nb]):
            ax3.bar(x, v, 0.7, color=col, alpha=0.8)
            ax3.text(x, v + 2, f"{v:.1f}", ha="center", fontsize=8.5)
        ax3.set_xticks(range(len(bars)))
        ax3.set_xticklabels([b_[0] for b_ in bars], fontsize=7.5)
        ax3.set_xlim(-0.6, len(bars) - 0.4)
        ax3.set_ylim(0, 150)
        ax3.set_ylabel("jet $p_T$  [GeV]", fontsize=10)
        if f_thr > 0:
            c4 = self.phys.summary["collections"]["ak4"]
            thr = [3, 10, 30]
            vals = [c4["n_jets"], c4["n_jets_10"], c4["n_jets_30"]]
            txt = "the threshold: jets above " + ", ".join(f"{t} GeV: {v}" for t, v in zip(thr, vals))
            fig.text(0.55, 0.47, txt, fontsize=10.5, color=TEXT, alpha=min(1.0, 3 * f_thr), weight="bold")


# --- scene 4: the ghosts ----------------------------------------------------------------------------------
class GhostScene(Scene):
    number, title = 4, "the ghosts: who decides the jet is a $b$ jet"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.nframes = 14 * fps
        ev = phys.ev
        self.c = phys.st.coll["ak4"]
        self.jb = self.c.info[phys.b["jets"]["ak4"]["rank"]]
        self.jbb = self.c.info[phys.bbar["jets"]["ak4"]["rank"]]
        g = phys.st.ghosts
        self.ghosts = [(kind, r) + kin(ev.p[r].p) + (core.name(ev.p[r].id),) for kind in ("bhad", "chad", "physics") for r in g[kind]]
        self.ghosts += [("parton", r) + kin(ev.p[r].p) + (core.name(ev.p[r].id),) for r in g["parton"] if core.pt(ev.p[r].p) > 3]
        self.k_shrink, self.k_labels = 5 * fps, 9 * fps
        self.roles = roles_of(phys)

    def phases(self):
        return [("shrink", 0, self.k_shrink), ("labels", self.k_shrink, self.k_labels), ("pair", self.k_labels, self.nframes)]

    def keyframe(self):
        return self.nframes - 1

    def panel(self, fig, rect, centre, f_shrink, f_labels, which):
        ax = fig.add_axes(rect)
        style_2d(ax, 9)
        e0, f0 = centre
        ev = self.phys.ev
        for j in self.c.info:
            if abs(j["eta"] - e0) < 1.4 and abs(core.dphi(j["phi"], f0)) < 1.4 and j["pt"] > 3:
                col = jet_colour(j["rank"], self.roles)
                ax.add_patch(Circle((j["eta"], j["phi"]), 0.4, fill=False, edgecolor=col, lw=1.6 if col != COL_DEC else 0.8, ls="-" if col != COL_DEC else "--", zorder=4))
                for r in j["rows"]:
                    e, f, pt = kin(ev.p[r].p)
                    ax.scatter(e, f, s=msize(pt) * 0.6, color=col, alpha=0.35, lw=0, zorder=2)
                if f_labels > 0 and j["pt"] > 5:
                    lab = f"[{j['rank']}] {j['pt']:.0f} GeV\nhadronFlavour {j['hadron_flavour']}, partonFlavour {j['parton_flavour']:+d}"
                    if which == "bbar" and j["net_flavour"]:
                        lab += f"\nnet {core.flavour_text(j['net_flavour'])}"
                    ax.annotate(lab, (j["eta"], j["phi"]), xytext=(0, 34 if which == "b" else -46), textcoords="offset points", ha="center", fontsize=8, color=col,
                                weight="bold", alpha=min(1.0, 3 * f_labels), zorder=8)
        for kind, r, e, f, pt, nm in self.ghosts:
            if abs(e - e0) > 1.4 or abs(core.dphi(f, f0)) > 1.4:
                continue
            scale = (1 - f_shrink) if kind in ("bhad", "chad") else 1.0
            col = {"bhad": COL_B if "bar" in nm or nm.startswith("Lambda_b") else COL_BBAR, "chad": COL_GAM, "physics": "black", "parton": COL_GLU if nm == "g" else COL_STR}[kind]
            if kind == "bhad":
                col = COL_B if core.flavour_of(ev.p[r].id).get(5, 0) > 0 else COL_BBAR
            marker = {"bhad": "*", "chad": "D", "physics": "P", "parton": "x"}[kind]
            size = msize(pt) * 2.2 * scale + 40 if kind in ("bhad", "chad") else 60
            ax.scatter(e, f, s=size, marker=marker, facecolors="none" if kind != "parton" else col, edgecolors=col, lw=1.4, zorder=7)
            if kind in ("bhad", "physics") or (kind == "chad" and pt > 5):
                ax.annotate(tex(nm) + (f" {pt:.0f} GeV" if f_shrink < 0.5 else " ghost" if kind != "physics" else f" ({pt:.0f}, status 23)"), (e, f),
                            xytext=(6, 6), textcoords="offset points", fontsize=8.5, color=col, zorder=8)
        ax.set_xlim(e0 - 1.4, e0 + 1.4)
        ax.set_ylim(f0 - 1.4, f0 + 1.4)
        ax.set_xlabel(r"$\eta$", fontsize=10)
        ax.set_ylabel(r"$\phi$", fontsize=10)
        return ax

    def draw(self, fig, k):
        f_shrink = ramp(k, 0, self.k_shrink)
        f_labels = ramp(k, self.k_shrink, self.k_shrink + self.fps)
        f_pair = ramp(k, self.k_labels, self.k_labels + self.fps)
        b, bb = self.phys.b, self.phys.bbar
        jb, jbb = b["jets"]["ak4"], bb["jets"]["ak4"]
        chrome(fig, self.number, self.title,
               f"The last b and c hadrons of every decay chain and the partons handed to the strings are added with their momentum times $10^{{-18}}$: ghosts that follow the clustering "
               f"without changing a jet. The $\\bar B^0$'s ghost lands in jet [{jb['rank']}]: hadronFlavour 5; the hardest b-parton ghost gives partonFlavour {jb['parton_flavour']:+d}. "
               f"The status-23 $b$ quark sits 0.56 away: outside the AK4 jet. At $\\eta$ = -3.4 the gluon-splitting pair ($\\bar\\Lambda_b^0$ + $\\bar B^0$) shares jet [2]: hadronFlavour 5, "
               f"net flavour 0; the $\\bar b$ quark's $B^0$ oscillated, so its jet [{jbb['rank']}] has net flavour +1 b.")
        ax1 = self.panel(fig, [0.05, 0.14, 0.42, 0.76], (self.jb["eta"], self.jb["phi"]), f_shrink, f_labels, "b")
        ax1.set_title(f"the $b$ quark's jet [{jb['rank']}]: ghosts at " + (f"full size" if f_shrink < 0.5 else "$10^{-18}$ of their momentum"), fontsize=11, color=TEXT)
        if f_pair > 0:
            ax2 = self.panel(fig, [0.55, 0.14, 0.42, 0.76], (self.jbb["eta"], self.jbb["phi"]), 1.0, f_pair, "bbar")
            ax2.set_title(f"the $\\bar b$ quark's jet [{jbb['rank']}] and the gluon-splitting pair", fontsize=11, color=TEXT)
        else:
            ax2 = fig.add_axes([0.55, 0.14, 0.42, 0.76])
            ax2.axis("off")
            rules = ("hadronFlavour  = 5 if a b-hadron ghost is in the jet,\n                           4 if a c-hadron ghost, else 0\n\n"
                     "partonFlavour  = id of the hardest b parton (status 71-79)\n                           in the jet, else c, else the hardest parton\n\n"
                     f"our $b$ jet: hadronFlavour {jb['hadron_flavour']}, partonFlavour {jb['parton_flavour']:+d}\n"
                     f"   b hadrons: {', '.join(tex(h) for h in jb['b_hadrons'])}\n   c hadrons: {', '.join(tex(h) for h in jb['c_hadrons'])}\n"
                     f"   partons: {', '.join(jb['partons'])}")
            ax2.text(0.02, 0.95, rules, transform=ax2.transAxes, va="top", fontsize=11, family="monospace", alpha=min(1.0, 3 * f_labels) if f_labels > 0 else 0.0)


# --- scene 5: the whisper (infrared safety) ---------------------------------------------------------------
class WhisperScene(Scene):
    number, title = 5, "the whisper: a soft $b\\bar b$ pair and infrared safety"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.nframes = 16 * fps
        self.extra, self.between, self.st2 = phys.pairs[0.5]
        c = phys.st.coll["ak4"]
        self.ja, self.jb_ = c.info[self.between[0]], c.info[self.between[1]]
        self.e0 = 0.5 * (self.ja["eta"] + self.jb_["eta"])
        self.f0 = self.ja["phi"] + 0.5 * ((self.jb_["phi"] - self.ja["phi"] + math.pi) % (2 * math.pi) - math.pi)
        # the soft hadrons are placed and clustered by rapidity; a 0.5 GeV B meson sits 0.7 away in eta, so draw them at y
        self.pos = [(core.rapidity(e[0]), core.phi(e[0]), core.pt(e[0])) for e in self.extra]
        ev = phys.ev
        self.parts = [(i,) + kin(ev.p[i].p) for i in ev.final_indices() if abs(core.eta(ev.p[i].p) - self.e0) < 1.4 and abs(core.dphi(core.phi(ev.p[i].p), self.f0)) < 1.4]
        self.jet_of = c.jet_of
        self.roles = roles_of(phys)
        # the labels of the two jets at the three strengths
        self.scan = {}
        for p, (extra, between, st2) in phys.pairs.items():
            ghost = sum(1 for j in st2.coll["ak4"].info if j["extras"] and j["hadron_flavour"] == 5)
            net = sum(1 for j in st2.stable_jets if any(l.startswith("soft") for l in j["flavoured"]) and j["net_flavour"].get(5))
            ifn = sum(1 for j in st2.stable_jets if any(l.startswith("soft") for l in j["flavoured"]) and j["ifn_flavour"].get(5))
            self.scan[p] = (ghost, net, ifn)
        self.ifn_event = next((e for e in self.st2.ifn.events if "soft" in self.st2.labels[e[1]] or "soft" in self.st2.labels[e[2]]), None) if self.st2.ifn.events else None
        self.k_pair, self.k_flip, self.k_neut = 3 * fps, 7 * fps, 11 * fps

    def phases(self):
        return [("pair", 0, self.k_pair), ("flip", self.k_pair, self.k_flip), ("neutralise", self.k_flip, self.k_neut), ("scan", self.k_neut, self.nframes)]

    def keyframe(self):
        return self.k_neut - 1

    def draw(self, fig, k):
        f_pair = ramp(k, 0, self.k_pair)
        f_flip = ramp(k, self.k_pair, self.k_pair + self.fps)
        f_neut = ramp(k, self.k_flip, self.k_flip + self.fps)
        f_scan = ramp(k, self.k_neut, self.nframes - self.fps)
        g, n_, i_ = self.scan[0.5]
        ev_ = self.ifn_event
        chrome(fig, self.number, self.title,
               f"A gluon splits into a $b\\bar b$ pair so soft (0.5 GeV each) that nothing in the event changes, and the two B mesons fall into jets [{self.between[0]}] and [{self.between[1]}]. "
               f"hadronFlavour now says 5 for {g} jets, the net flavour says $b$ and $\\bar b$: two labels changed by a whisper, which no infrared-safe quantity may do. "
               f"IFN checks, before the soft $B^0$ joins its jet, whether an opposite flavour is closer in $u_{{ik}}$ = max($p_T$)$^2\\,\\Omega^2_{{ik}}$ than the jet it is joining"
               + (f": $u$ = {ev_[3]:.3g} against {ev_[4]:.3g}, the pair is neutralised and both jets stay flavourless." if ev_ else "; here it neutralises the pair."))
        ax = fig.add_axes([0.05, 0.14, 0.5, 0.76])
        style_2d(ax, 9)
        for (i, e, f, pt) in self.parts:
            rank = self.jet_of.get(i, -1)
            col = jet_colour(rank, self.roles) if rank >= 0 else COL_UNCL
            ax.scatter(e, f, s=msize(pt) * 0.7, color=col, alpha=0.5, lw=0, zorder=2)
        for j in (self.ja, self.jb_):
            col = jet_colour(j["rank"], self.roles)
            ax.add_patch(Circle((j["eta"], j["phi"]), 0.4, fill=False, edgecolor=col, lw=1.8, zorder=4))
            lab = f"[{j['rank']}] {j['pt']:.1f} GeV"
            if f_flip > 0:
                which = 0 if j is self.ja else 1
                fl = "+1 b" if self.extra[which][1].get(5, 0) > 0 else "-1 b"
                neut = f_neut > 0 and ev_ is not None
                lab += f"\nghost: hadronFlavour 5\nnet flavour {fl}" + ("\nIFN: none" if neut else "")
            left = j is self.ja
            ax.annotate(lab, (j["eta"], j["phi"]), xytext=(-30 if left else 30, 46), textcoords="offset points", ha="right" if left else "left", fontsize=9, color=col,
                        weight="bold", zorder=8)
        if f_pair > 0:
            for (e, f, pt), ex in zip(self.pos, self.extra):
                col = COL_B if ex[1].get(5, 0) > 0 else COL_BBAR
                ax.scatter(e, f, s=40 + 260 * f_pair, marker="*", color=col, zorder=7, lw=0)
                ax.annotate(tex(ex[2].split()[-1]) + f" {pt:.1f} GeV (at its rapidity)", (e, f), xytext=(6, -14 if ex is self.extra[0] else 8), textcoords="offset points",
                            fontsize=9, color=col, zorder=8)
            if f_neut > 0 and ev_ is not None:
                (e1, f1, _), (e2, f2, _) = self.pos
                ax.plot([e1, e2], [f1, f2], color=COL_GLU, lw=3 * f_neut, zorder=6)
                ax.text(0.5 * (e1 + e2), 0.5 * (f1 + f2) + 0.08, f"IFN: $u_{{ik}}$ = {ev_[3]:.2g} < $u_{{ij}}$ = {ev_[4]:.2g}\nneutralised", ha="center", fontsize=9, color=COL_GLU, zorder=8,
                        alpha=min(1.0, 2 * f_neut))
        ax.set_xlim(self.e0 - 1.2, self.e0 + 1.2)
        ax.set_ylim(self.f0 - 1.2, self.f0 + 1.2)
        ax.set_xlabel(r"$\eta$", fontsize=11)
        ax.set_ylabel(r"$\phi$", fontsize=11)
        ax.set_title("two light jets and a soft $g \\to b\\bar b$ between them", fontsize=11, color=TEXT)
        ax2 = fig.add_axes([0.63, 0.2, 0.34, 0.64])
        style_2d(ax2, 9)
        ps = sorted(self.scan)
        labels = [("ghost (hadronFlavour)", COL_B, 0), ("net flavour", COL_HAD, 1), ("IFN", COL_GLU, 2)]
        npts = int(len(ps) * f_scan + 0.999) if f_scan > 0 else 0
        for lab, col, idx in labels:
            ax2.plot(ps[:npts], [self.scan[p][idx] + 0.04 * (idx - 1) for p in ps[:npts]], "o-", color=col, ms=7, mec="white", lw=1.6, label=lab)
        ax2.set_xscale("log")
        ax2.set_xlim(0.3, 12)
        ax2.set_ylim(-0.3, 2.3)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(["no jet\nlabelled b", "one jet", "both jets"], fontsize=9)
        ax2.set_xlabel("$p_T$ of each hadron of the pair  [GeV]", fontsize=10)
        ax2.set_title("the labels as the pair gets harder", fontsize=11, color=TEXT)
        ax2.legend(loc="center right", fontsize=9)
        if f_scan <= 0:
            ax2.text(0.5, 0.5, "scan follows", ha="center", transform=ax2.transAxes, fontsize=10, color=TEXT2)


# --- scene 6: the ensemble --------------------------------------------------------------------------------
class EnsembleScene(Scene):
    number, title = 6, "the same family photographed again and again"

    def __init__(self, phys, fps, events=400):
        super().__init__(phys, fps)
        self.nframes = 9 * fps
        st = phys.stats
        self.resp = {}
        if st:
            for key in ("ak4", "ak4nonu", "ak8"):
                v = np.array(st["b"]["b"]["jets"][key]["response"], dtype=float)[:events]
                self.resp[key] = v[~np.isnan(v)]
        else:
            for key in ("ak4", "ak4nonu", "ak8"):
                self.resp[key] = np.array([phys.b["jets"][key]["response"]])
        self.k_run = self.nframes - 2 * fps

    def phases(self):
        return [("fill", 0, self.k_run), ("result", self.k_run, self.nframes)]

    def draw(self, fig, k):
        f = ramp(k, 0, self.k_run)
        b = self.phys.b
        N = len(self.resp["ak4"])
        n = max(1, int(N * f))
        chrome(fig, self.number, self.title,
               f"The primary hadrons of chapter 5 decayed {N} times with different seeds and clustered each time. The AK4 jet keeps {np.mean(self.resp['ak4']):.3f} of the "
               f"quark's $p_T$ on average, the jet without neutrinos {np.mean(self.resp['ak4nonu']):.3f} with a tail from the semileptonic decays, the AK8 jet "
               f"{np.mean(self.resp['ak8']):.3f}: the photograph depends on the choices, the family does not.")
        ax = fig.add_axes([0.07, 0.15, 0.9, 0.73])
        style_2d(ax, 9)
        bins = np.linspace(0.6, 1.8, 49)
        for key, col, ls, lab in (("ak4", COL_B, "-", "ak4GenJets"), ("ak4nonu", COL_NU, "--", "ak4GenJetsNoNu"), ("ak8", COL_BBAR, "-", "ak8GenJets")):
            v = self.resp[key][:n]
            ax.hist(v, bins=bins, histtype="step", color=col, lw=2, ls=ls, label=f"{lab}: mean {v.mean():.3f}, std {v.std():.3f}")
        for key, col, y in (("ak4", COL_B, 0.93), ("ak8", COL_BBAR, 0.85)):
            ax.axvline(b["jets"][key]["response"], color=col, lw=1.2, ls=":")
            ax.text(b["jets"][key]["response"] + 0.01, y, f"seed 1 {key}: {b['jets'][key]['response']:.3f}", color=col, fontsize=9, transform=ax.get_xaxis_transform())
        ax.set_yscale("log")
        ax.set_ylim(0.7, 1.5 * max(N, 2))
        ax.set_xlabel("$p_T$(jet) / $p_T$($b$ quark)", fontsize=11)
        ax.set_ylabel("decays", fontsize=11)
        ax.set_title(f"{n} of {N} decays clustered", fontsize=11, color=TEXT)
        ax.legend(loc="upper left", fontsize=10)


# --- narration --------------------------------------------------------------------------------------------
def narration(phys: Physics, scenes: List[Scene]) -> Dict[str, str]:
    s, b, bb = phys.summary, phys.b, phys.bbar
    c4 = s["collections"]["ak4"]
    j4, j4n, j8 = b["jets"]["ak4"], b["jets"]["ak4nonu"], b["jets"]["ak8"]
    semi = phys.semi
    between = phys.between
    ev_ = next((e for e in phys.pairs[0.5][2].ifn.events), None)
    lines = {
        "photograph_crowd": (f"The grandchildren are born: {s['n_final']} particles, five of them neutrinos, stand where the decays left them. Before the family sets out for the "
                             "detector we take a photograph, because everyone who later claims to be a member will be compared against it."),
        "photograph_jets": (f"The photographer is anti k T with a radius of nought point four: it gathers {c4['n_clustered']} of the particles into {c4['n_jets']} jets above "
                            f"three G e V, {c4['n_jets_30']} of them above thirty. The b quark's family is the red jet with {j4['pt']:.1f} G e V, the anti-b's the blue one "
                            f"with {bb['jets']['ak4']['pt']:.1f}."),
        "gather_core": ("How the picture is taken. The distance between two particles is the smaller of their inverse p T squared, times their separation squared over R squared. "
                        "The hardest pair that is close has the smallest distance, so the core of the jet forms first: the B meson's kaon and pion, then its photons."),
        "gather_periphery": (f"The periphery joins last, each merge at a larger separation, and the ghosts of the B meson and of the b quark slip in along the way. "
                             f"After {len(scenes[1].steps) if len(scenes) > 1 and isinstance(scenes[1], GatherScene) else 36} merges nothing within the radius is left."),
        "gather_declared": (f"Then the distance to the beam, one over p T squared, is the smallest left, and the jet is declared: {j4['pt']:.2f} G e V in {j4['n']} particles, "
                            f"its axis {j4['dr']:.3f} from the quark. The cone of the same radius around the quark held {j4['pt_cone']:.2f}: two particles differ at the border."),
        "choices_radius": (f"The choices. Around the same axis the momentum inside the circle grows with the radius: {j4['pt']:.1f} G e V at nought point four, the A K 4 jet, "
                           f"and {j8['pt']:.1f} at nought point eight, the A K 8 jet, which has taken in a gluon's family as well."),
        "choices_neutrino": ((f"Our B meson decayed without a neutrino, so the jet without neutrinos is the same jet. In decay seed {semi[0]} of the same hadrons the B went "
                              f"semileptonic, and the jet without neutrinos has {semi[1]['b']['b']['jets']['ak4nonu']['pt']:.1f} G e V instead of "
                              f"{semi[1]['b']['b']['jets']['ak4']['pt']:.1f}. ") if semi else "Our B meson decayed without a neutrino, so the jet without neutrinos is the same jet. ")
                             + "C M S S W's event lost thirty-eight G e V this way: a hundred and eighteen with the neutrino, eighty without.",
        "choices_threshold": (f"And the threshold: {c4['n_jets']} jets above three G e V, {c4['n_jets_10']} above ten, {c4['n_jets_30']} above thirty. "
                              "The family is the same; the photograph depends on the radius, the neutrinos and the threshold, and every later chapter must say which one it uses."),
        "ghosts_shrink": ("Who decides that this is a b jet? The B meson has decayed; it is no longer in the picture. So we add it back as a ghost: its direction, its momentum "
                          "times ten to the minus eighteen. A ghost changes no jet, but anti k T still assigns it to one."),
        "ghosts_labels": (f"The B meson's ghost lands in the red jet: hadron flavour five. The b quark of the strings, also a ghost, lands there too: parton flavour {j4['parton_flavour']:+d}. "
                          f"The b quark of the hard process does not: the shower moved its family by more than half a unit, and the physics definition finds nothing."),
        "ghosts_pair": ("At the other end of the event, near eta minus three and a half, a gluon of the shower had split into a b anti-b pair. Its two hadrons share one jet: "
                        "hadron flavour five, net flavour zero. And the anti-b quark's B zero oscillated before it decayed, so its jet carries a b by the hadron's sign."),
        "whisper_pair": (f"The whisper. Imagine a gluon splitting into a b anti-b pair so soft, half a G e V each, that nothing in the event changes, and its two mesons "
                         f"fall into jets {between[0]} and {between[1]}."),
        "whisper_flip": ("Hadron flavour now says five for both jets; the net flavour says b and anti-b. Two labels changed by a whisper. A quantity that changes under an "
                         "arbitrarily soft emission is not infrared safe, and a calculation of it does not converge."),
        "whisper_neutralise": (f"Interleaved flavour neutralisation asks, before the soft B joins its jet, whether an opposite flavour is closer to it in the distance u than the jet it is joining. "
                               + (f"Here u is {ev_[3]:.2g} against {ev_[4]:.2g}: the pair cancels, and both jets stay flavourless." if ev_ else "Here it is: the pair cancels.")),
        "whisper_scan": ("As the pair gets harder the picture changes: above a few G e V the pair is a real b anti-b and every definition labels the jets. The safe definitions "
                         "agree with the naive ones where it matters and disagree where a soft pair would have fooled them."),
        "ensemble_fill": (f"The same family photographed again and again: the primary hadrons of chapter five decayed {len(scenes[-1].resp['ak4']) if scenes and isinstance(scenes[-1], EnsembleScene) else 400} "
                          "times with different seeds, and clustered each time."),
        "ensemble_result": (f"The A K 4 jet keeps {np.mean(scenes[-1].resp['ak4']) if scenes and isinstance(scenes[-1], EnsembleScene) else j4['response']:.2f} of the quark's momentum on average, "
                            "the jet without neutrinos a little less with a tail from the semileptonic decays, the A K 8 jet more. This is the reference: every jet "
                            "the detector reconstructs in the chapters to come will be measured against it."),
        "outro": ("The photograph is taken. Now the family leaves the vertex, and on the road it will meet a crowd: the other collisions of the same bunch crossing. "
                  "That is the next stop."),
    }
    return lines


def score_spec(film: snd.Film) -> snd.ScoreSpec:
    t_flip = snd.voice_at_phase(film, "whisper_flip")
    return snd.film_score(film, [snd.Dm, snd.Bb, snd.F, snd.C, snd.Gm, snd.A], thuds=[t_flip + 0.2] if t_flip is not None else [])


# --- driver -----------------------------------------------------------------------------------------------
def build_scenes(names: List[str], phys: Physics, args) -> List[Scene]:
    out = []
    for n in names:
        if n == "photograph":
            out.append(PhotographScene(phys, args.fps))
        elif n == "gather":
            out.append(GatherScene(phys, args.fps))
        elif n == "choices":
            out.append(ChoicesScene(phys, args.fps))
        elif n == "ghosts":
            out.append(GhostScene(phys, args.fps))
        elif n == "whisper":
            out.append(WhisperScene(phys, args.fps))
        elif n == "ensemble":
            out.append(EnsembleScene(phys, args.fps, args.events))
    return out


def write_video(film: snd.Film, out: Path, fps: float, dpi: int, audio: Optional[Path], jobs: int):
    snd.render_video(film, out, fps, dpi, audio, lambda fig, sc, k, t: sc.draw(fig, k), FIGSIZE, "b bbar event, standalone GenJets", jobs=jobs)
    print(f"wrote {out} ({film.duration:.1f} s)")


def write_frames(film: snd.Film, outdir: Path, dpi: int, fps: float):
    outdir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=FIGSIZE)
    for i, (sc, k) in enumerate(film.frames(fps)):
        sc.draw(fig, k)
        fig.savefig(outdir / f"frame_{i:04d}_s{sc.number}_k{int(round(k)):03d}.png", dpi=dpi, facecolor="white")
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
    p.add_argument("-o", "--output", type=Path, default=HERE / "animate_standalone_genjets.mp4", help=".mp4 (needs ffmpeg) or .gif (silent)")
    p.add_argument("--scene", default="all", help="all, or a comma list of photograph,gather,choices,ghosts,whisper,ensemble (or 1..6)")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--dpi", type=int, default=150, help="150 gives 1920x1080, 100 gives 1280x720")
    p.add_argument("--step", type=int, default=1, help="render every step-th frame")
    p.add_argument("--fast", action="store_true", help="preview: --step 4 --dpi 60")
    p.add_argument("--frames-dir", type=Path, default=None)
    p.add_argument("--stills", type=Path, default=None, help="one key frame per scene (.pdf or .png)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--jobs", type=int, default=4, help="render the frames in this many processes (joined losslessly)")
    p.add_argument("--audio-only", type=Path, default=None, metavar="WAV", help="write just the soundtrack and exit")
    p.add_argument("--cache-dir", type=Path, default=HERE.parent.parent.parent / "build" / "narration" / "GenJet")
    snd.add_narration_args(p)
    p.add_argument("-i", "--input", type=Path, default=HERE.parent.parent / "Decay" / "event_Decay.lhe")
    p.add_argument("--events", type=int, default=400, help="decays shown in scene 6 (from the stats file when present)")
    args = p.parse_args()
    if args.fast:
        args.step, args.dpi = max(args.step, 4), min(args.dpi, 60)
    snd.narration_defaults(args)
    fps = args.fps / args.step
    args.fps = snd.NOMINAL_FPS
    names = list(SCENE_NAMES.values()) if args.scene == "all" else [SCENE_NAMES.get(s.strip(), s.strip()) for s in args.scene.split(",")]
    bad = [n for n in names if n not in SCENE_NAMES.values()]
    if bad:
        sys.exit(f"unknown scene(s) {bad}; choose from {list(SCENE_NAMES.values())}")
    phys = Physics(args.input)
    s = phys.summary
    print(f"event: {s['n_final']} final particles, {s['collections']['ak4']['n_jets']} ak4 jets; the b quark's jet {phys.b['jets']['ak4']['pt']:.2f} GeV"
          + (f"; semileptonic example: decay seed {phys.semi[0]}" if phys.semi else "") + f"; soft pair between jets {phys.between}")
    scenes = build_scenes(names, phys, args)
    print("frames: " + ", ".join(f"{sc.number} {SCENE_NAMES[str(sc.number)]} {sc.nframes}" for sc in scenes)
          + f"  ({sum(sc.nframes for sc in scenes) / snd.NOMINAL_FPS:.1f} s silent)")
    lines = narration(phys, scenes)
    if args.print_script:
        snd.print_script(lines)
        return
    ffmpeg = shutil.which("ffmpeg")
    if args.output.suffix.lower() != ".gif" and ffmpeg is None and not args.no_narration:
        sys.exit("ffmpeg not found on PATH (needed for the narration clips)")
    clips = snd.load_narration(lines, args, args.cache_dir, ffmpeg) if ffmpeg else {}
    film = snd.Film(scenes, snd.clip_lengths(clips), {int(n): name for n, name in SCENE_NAMES.items()}, slow_max=4.0)
    print(f"film: {film.describe()}")
    score = None if args.no_music else score_spec(film)
    if args.audio_only:
        snd.mix_audio(film.duration, film.voice_at, clips, score, args.music_db, args.audio_only)
        print(f"wrote {args.audio_only}")
        return
    if args.stills is not None:
        write_stills(scenes, args.stills, args.dpi)
    if args.frames_dir is not None:
        write_frames(film, args.frames_dir, args.dpi, fps)
    elif not args.no_video:
        audio = None
        tmp = tempfile.TemporaryDirectory(prefix="jitjet_genjet_")
        if ffmpeg and args.output.suffix.lower() != ".gif" and (clips or score):
            audio = Path(tmp.name) / "soundtrack.wav"
            snd.mix_audio(film.duration, film.voice_at, clips, score, args.music_db, audio)
            if clips:
                snd.write_srt(film.voice_at, clips, lines, args.output.with_suffix(".srt"))
        write_video(film, args.output, fps, args.dpi, audio, args.jobs)


if __name__ == "__main__":
    main()
