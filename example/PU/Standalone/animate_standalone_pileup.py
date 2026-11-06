#!/usr/bin/env python3
"""Animated companion to standalone_pileup.py: the crowd on the road.

Six scenes, narrated, show the b bbar event of chapter 7 meeting the sixty other collisions of its
bunch crossing in Run 3 (2024). Every number on screen and in the narration is read from the record
(../event_PU.lhe), the library (../pileup_library.lhe.gz), the summary (standalone_pileup.json) and the
ensemble (standalone_pileup_stats.json) that the standalone script wrote. The film picks up where the
chapter-7 film ended (the photograph is taken) and ends at the next stop, the border of the detector.

  1  crowd      The photograph of chapter 7 in the (eta, phi) plane; then the in-time collisions of our
                crossing stack onto it one by one, the counter and rho running, until the crowd is complete.
  2  road       The beam spot in (z, t): the two bunches overlap, the vertices of our crossing appear
                with the signal among them, then the fifteen other crossings of the production window
                (bx = -12 ... +3), 25 ns apart.
  3  member     The four kinds of inelastic collision, one example of each from the library (ND, SD, DD,
                CD) with the table of their cross sections and multiplicities filling row by row, then the
                multiplicity of the two thousand collisions building up.
  4  median     The grid cells fill with pT, the histogram of the cell densities and its median: rho.
  5  ourjet     Zoom on our jet: the family, the crowd pouring into the circle, the pT rising; then rho A
                taken out, and what is left against the signal.
  6  ensemble   The crowd drawn four hundred times on the same photograph: the pileup inside the circle,
                the estimate rho A, and the residual.

Narration and music: as in the other chapter films (example/soundtrack.py): one line per phase
(--print-script), edge-tts or the macOS `say` voice, clips cached in build/narration/PU/, an .srt next
to the .mp4; the score has a chord per scene and a thud when the crowd is complete.

Examples (matplotlib is needed; on this Mac use /usr/bin/python3):
  python3 animate_standalone_pileup.py                  # -> .mp4 + .srt next to this file (1920x1080)
  python3 animate_standalone_pileup.py --fast           # quick low-resolution preview
  python3 animate_standalone_pileup.py --scene crowd,ourjet -o g.gif   # silent
  python3 animate_standalone_pileup.py --stills stills.pdf --no-video
  python3 animate_standalone_pileup.py --print-script
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import standalone_pileup as core  # noqa: E402

gj = core.gj
sys.path.insert(0, str(HERE.parent.parent))
import soundtrack as snd  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

COL_B, COL_BBAR, COL_SIG, COL_UNCL = "#ff0000", "#0000ff", "#9b4dff", "#c0c0c0"
COL_PU, COL_MODEL, COL_EVENT, COL_CMSSW = "#8a7f5a", "#1b9e77", "#c0392b", "#1f1f1f"
COL_EARLY, COL_IN, COL_LATE = "#b8860b", "#5aa9e6", "#fa8072"
COL_ND, COL_SD, COL_DD, COL_CD = "#2a78d6", "#2f8f5b", "#eb6834", "#8e5bb5"
PALETTE = ["#00999a", "#9b59d6", "#3b9ae1", "#ff1493", "#2e9e2e", "#a3a300"]
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d6d3ca"
FIGSIZE = (12.8, 7.2)
SCENE_NAMES = {"1": "crowd", "2": "road", "3": "member", "4": "median", "5": "ourjet", "6": "ensemble"}


def smooth(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ramp(k: float, k0: int, k1: int) -> float:
    return smooth((k - k0) / max(1, k1 - k0))


def kin(p):
    return gj.eta(p), gj.phi(p), gj.pt(p)


def dr2v(eta1, phi1, eta2, phi2):
    """dR^2 for arrays."""
    d = np.abs(np.asarray(phi1) - phi2) % (2 * math.pi)
    d = np.where(d > math.pi, 2 * math.pi - d, d)
    return (np.asarray(eta1) - eta2) ** 2 + d * d


class Physics:
    """Everything the scenes draw, read from the files the standalone script wrote."""

    def __init__(self, record: Path, library: Path, summary: Path, stats: Path):
        self.summary, self.collisions, self.particles, self.jets, self.signal_of = core.read_pileup_tables(record)
        self.s = json.loads(summary.read_text())
        self.stats = json.loads(stats.read_text())["results"] if stats.exists() else []
        self.lib = core.Library(library)
        ev, _, _ = gj.dec.read_record(record)
        self.ev = ev
        self.genjets, self.genjets_of = gj.read_jet_tables(record)
        self.ref = self.genjets["ak4"][0]
        self.mixed_b = next((j for j in self.jets if j["genjet"] == 0), None)
        # the signal particles that entered the mixed clustering
        self.sig_rows = [i for i in ev.final_indices() if abs(ev.p[i].id) not in gj.IGNORE_IDS and not gj.is_neutrino(ev.p[i].id) and abs(gj.eta(ev.p[i].p)) < core.ETA_MAX]
        self.sig_kin = np.array([kin(ev.p[i].p) for i in self.sig_rows])
        self.sig_y = np.array([gj.rapidity(ev.p[i].p) for i in self.sig_rows])
        # the in-time pileup particles by collision, in the order of the record
        self.in_time = [c for c in self.collisions if c["bx"] == 0 and c["k"] > 0]
        self.pu = {}
        for p in self.particles:
            if gj.is_neutrino(p["id"]):
                continue
            ptv = math.hypot(p["px"], p["py"])
            if ptv <= 0:
                continue
            e = math.asinh(p["pz"] / ptv)
            if abs(e) >= core.ETA_MAX:
                continue
            y = 0.5 * math.log((p["e"] + p["pz"]) / (p["e"] - p["pz"])) if p["e"] > abs(p["pz"]) else e
            self.pu.setdefault(p["k"], []).append((e, math.atan2(p["py"], p["px"]), ptv, p["jet"], y))
        self.pu = {k: np.array(v) for k, v in self.pu.items()}
        self.order = [c["k"] for c in self.in_time]
        # the running rho: the grid cells per collision, cumulated
        ny, nphi = int(2 * core.GRID_RAP_MAX / core.GRID_SPACING), int(2 * math.pi / core.GRID_SPACING)
        self.dy, self.dphi = 2 * core.GRID_RAP_MAX / ny, 2 * math.pi / nphi

        def cells_of(arr, yv):
            """FastJet's grid, as grid_rho: rapidity, phi from 0."""
            c = np.zeros((ny, nphi))
            if len(arr):
                iy = np.floor((yv + core.GRID_RAP_MAX) / self.dy).astype(int)
                ip = np.floor((arr[:, 1] % (2 * math.pi)) / self.dphi).astype(int) % nphi
                ok = (iy >= 0) & (iy < ny)
                np.add.at(c, (iy[ok], ip[ok]), arr[ok, 2])
            return c / (self.dy * self.dphi)
        self.cells_sig = cells_of(self.sig_kin, self.sig_y)
        self.cells_cum = [self.cells_sig.copy()]
        for k in self.order:
            arr = self.pu.get(k, np.zeros((0, 5)))
            self.cells_cum.append(self.cells_cum[-1] + cells_of(arr, arr[:, 4] if len(arr) else np.zeros(0)))
        self.rho_run = [float(np.median(c)) for c in self.cells_cum]
        self.n_run = [0] + [sum(len(self.pu.get(k, [])) for k in self.order[:n + 1]) for n in range(len(self.order))]
        # the pileup inside the circle of our jet, cumulated
        self.circle_run = [0.0]
        for k in self.order:
            arr = self.pu.get(k, np.zeros((0, 4)))
            inside = dr2v(arr[:, 0], arr[:, 1], self.ref["eta"], self.ref["phi"]) < 0.16 if len(arr) else np.zeros(0, dtype=bool)
            self.circle_run.append(self.circle_run[-1] + float(arr[inside, 2].sum()) if len(arr) else self.circle_run[-1])
        self.region = core.region_of(self.s.get("region_model", "run3"))
        # one library event of each kind for scene 3, and the means per kind (the catalogue of minbias_catalogue.py)
        self.ex_nd = max(self.lib.events, key=lambda e: e["nfinal"] if e["kind"] == "ND" and e["nfinal"] < 400 else -1)
        self.ex_sd = max(self.lib.events, key=lambda e: e["nfinal"] if e["kind"] == "SD" and 20 < e["nfinal"] < 80 else -1)
        self.ex_dd = max(self.lib.events, key=lambda e: e["nfinal"] if e["kind"] == "DD" and 30 < e["nfinal"] < 120 else -1)
        self.ex_cd = max(self.lib.events, key=lambda e: e["nfinal"] if e["kind"] == "CD" and 15 < e["nfinal"] < 60 else -1)
        self.examples = dict(ND=self.ex_nd, SD=self.ex_sd, DD=self.ex_dd, CD=self.ex_cd)
        self.sigma = dict(ND=55.42, SD=12.86, DD=8.85, CD=1.29)
        self.per_kind = {}
        for kd in ("ND", "SD", "DD", "CD"):
            sel = [e for e in self.lib.events if e["kind"] == kd]
            self.per_kind[kd] = dict(events=len(sel), share=100.0 * len(sel) / len(self.lib), nmpi=np.mean([max(1, e.get("nmpi", 0)) for e in sel]),
                                     stable=np.mean([e["nfinal"] for e in sel]), charged=np.mean([e["nch"] for e in sel]),
                                     central=np.mean([int(((e["pt"] > 0.5) & (np.abs(e["eta"]) < 2.5)).sum()) for e in sel]),
                                     lead=np.mean([e["pt_max"] for e in sel]), sumpt=np.mean([e["sumpt"] for e in sel]))


# --- drawing helpers ------------------------------------------------------------------------------------
def chrome(fig, scene_no: int, title: str, caption: str):
    fig.clf()
    fig.patch.set_facecolor("white")
    fig.text(0.02, 0.965, f"Scene {scene_no}: {title}", fontsize=15, weight="bold", color=TEXT, va="center")
    fig.text(0.98, 0.965, "standalone_pileup.py", fontsize=9.5, color=TEXT2, ha="right", va="center")
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
    return PALETTE[k] if k < len(PALETTE) else "#8a8a8a"


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


def eta_phi_axes(fig, rect, fontsize=9):
    ax = fig.add_axes(rect)
    style_2d(ax, fontsize)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-math.pi, math.pi)
    ax.set_xlabel(r"$\eta$", fontsize=11)
    ax.set_ylabel(r"$\phi$", fontsize=11)
    return ax


# --- scene 1: the crowd arrives --------------------------------------------------------------------------
class CrowdScene(Scene):
    number, title = 1, "the crowd arrives: sixty collisions in one bunch crossing"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.title = f"the crowd arrives: {len(phys.order)} collisions in one bunch crossing"
        self.k_hold = 3 * fps
        self.k_stack = self.k_hold + 12 * fps
        self.nframes = self.k_stack + 3 * fps
        self.roles_gen = {0: "b", 1: "bbar"}
        self.roles_mix = {j["rank"]: ("b" if j["genjet"] == 0 else "bbar") for j in phys.jets if j["genjet"] in (0, 1)}

    def phases(self):
        return [("photo", 0, self.k_hold), ("stack", self.k_hold, self.k_stack), ("full", self.k_stack, self.nframes)]

    @staticmethod
    def sizes(ptv):
        return np.clip(6.0 + 40.0 * np.log10(1 + ptv), 6.0, 260.0) ** 2 * 0.55 ** 2 / 6

    def draw(self, fig, k):
        ph = self.phys
        f = ramp(k, self.k_hold, self.k_stack)
        n = int(round(f * len(ph.order)))
        done = k >= self.k_stack
        s = ph.s
        cap = (f"The photograph of chapter 7: {len(ph.sig_rows)} signal particles within |eta| < 5, the b jet [0] at {ph.ref['pt']:.1f} GeV. " if n == 0 else
               f"{n} of the {len(ph.order)} in-time collisions stacked: {ph.n_run[n]} pileup particles, grid median rho = {ph.rho_run[n]:.1f} GeV per unit area, "
               f"{ph.circle_run[n]:.1f} GeV of crowd inside the circle of our jet.")
        if done:
            cap = (f"The crowd is complete: {len(ph.order)} collisions, {ph.n_run[-1]} particles, rho = {s['rho_grid']:.1f} GeV per unit area; anti-kT on all of it gives {s['n_jets']} jets, "
                   f"our jet [0] now {ph.mixed_b['pt']:.1f} GeV ({ph.mixed_b['pt_signal']:.1f} signal + {ph.mixed_b['pt_pileup']:.1f} crowd).")
        chrome(fig, self.number, self.title, cap)
        ax = eta_phi_axes(fig, [0.05, 0.15, 0.62, 0.77])
        # the crowd: the collisions already stacked in one scatter, the newest one brighter
        if n:
            stacked = [ph.pu[kk] for kk in ph.order[:n - 1] if kk in ph.pu and len(ph.pu[kk])]
            if stacked:
                arr = np.vstack(stacked)
                cols = [jet_colour(int(j), self.roles_mix) if done and j >= 0 else COL_PU for j in arr[:, 3]] if done else COL_PU
                ax.scatter(arr[:, 0], arr[:, 1], s=self.sizes(arr[:, 2]), c=cols, alpha=0.35, lw=0, zorder=2, rasterized=True)
            last = ph.pu.get(ph.order[n - 1])
            if last is not None and len(last):
                ax.scatter(last[:, 0], last[:, 1], s=self.sizes(last[:, 2]), c=COL_PU, alpha=0.35 if done else 0.9, lw=0, zorder=2, rasterized=True)
        # the signal
        roles = self.roles_mix if done else self.roles_gen
        jet_of = ph.signal_of if done else ph.genjets_of["ak4"]
        cols = [jet_colour(jet_of.get(r, -1), roles) if jet_of.get(r, -1) >= 0 else COL_UNCL for r in ph.sig_rows]
        ax.scatter(ph.sig_kin[:, 0], ph.sig_kin[:, 1], s=[msize(p, 1.1) ** 2 / 4 for p in ph.sig_kin[:, 2]], c=cols, alpha=0.95, lw=0, zorder=4)
        jets = ph.jets if done else [dict(g, pt_sub=g["pt"]) for g in ph.genjets["ak4"]]
        for j in jets:
            if j["pt"] < 10:
                continue
            col = jet_colour(j["rank"], roles)
            lead = j["rank"] in roles
            ax.add_patch(Circle((j["eta"], j["phi"]), 0.4, fill=False, ec=col, lw=1.8 if lead else 0.8, ls="-" if lead else "--", zorder=5))
            ax.annotate(f"[{j['rank']}] {j['pt']:.1f}" + (f" $\\to$ {j['pt_sub']:.1f}" if done else ""), (j["eta"], j["phi"]), xytext=(4, 3), textcoords="offset points",
                        fontsize=8 if lead else 7, color=col, weight="bold" if lead else "normal", zorder=6)
        ax.set_title("the signal in colour, the crowd in khaki" if not done else "anti-kT on signal + crowd: the mixed jets, label = pT -> pT - rho A", fontsize=10.5, color=TEXT)
        # the counters
        ax2 = fig.add_axes([0.72, 0.15, 0.26, 0.77])
        ax2.axis("off")
        rows = [("in-time collisions", f"{n} of {len(ph.order)}"), ("pileup particles", f"{ph.n_run[n]}"), ("signal particles", f"{len(ph.sig_rows)}"),
                ("rho (grid median)", f"{ph.rho_run[n]:.1f} GeV / area"), ("crowd in our circle", f"{ph.circle_run[n]:.1f} GeV")]
        for i, (lab, v) in enumerate(rows):
            ax2.text(0.0, 0.95 - 0.08 * i, f"{lab}:", fontsize=11, color=TEXT2, transform=ax2.transAxes)
            ax2.text(0.0, 0.91 - 0.08 * i, v, fontsize=13, color=TEXT, weight="bold", transform=ax2.transAxes)
        # the running rho
        ax3 = fig.add_axes([0.74, 0.17, 0.23, 0.33])
        style_2d(ax3, 8)
        ax3.plot(range(len(ph.order) + 1), ph.rho_run, "-", color=GRID, lw=1.5)
        ax3.plot(range(n + 1), ph.rho_run[:n + 1], "-", color=COL_MODEL, lw=2.2)
        ax3.plot([n], [ph.rho_run[n]], "o", color=COL_EVENT, ms=6)
        ax3.set_xlabel("collisions stacked", fontsize=9)
        ax3.set_ylabel(r"$\rho$  [GeV per unit area]", fontsize=9)
        ax3.set_title(f"{ph.rho_run[-1] / max(1, len(ph.order)):.2f} GeV per collision", fontsize=9, color=TEXT2)


# --- scene 2: the road ------------------------------------------------------------------------------------
class RoadScene(Scene):
    number, title = 2, "the road: where and when the collisions happen"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.k_bunch = 4 * fps
        self.k_in = self.k_bunch + 5 * fps
        self.nframes = self.k_in + 7 * fps
        z = np.linspace(-20, 20, 201)
        t = np.linspace(-0.8, 0.8, 201)
        self.Z, self.T = np.meshgrid(z, t, indexing="ij")
        self.D = phys.region.density(self.Z, self.T)
        self.D /= self.D.max()
        self.m = phys.region.moments()

    def phases(self):
        return [("bunches", 0, self.k_bunch), ("intime", self.k_bunch, self.k_in), ("outoftime", self.k_in, self.nframes)]

    def draw(self, fig, k):
        ph = self.phys
        r = ph.region
        f1 = ramp(k, 0, self.k_bunch)
        f2 = ramp(k, self.k_bunch, self.k_in)
        f3 = ramp(k, self.k_in, self.nframes - 2 * self.fps)
        sig = [c for c in ph.collisions if c["k"] == 0][0]
        it = ph.in_time
        allv = [c for c in ph.collisions if c["k"] > 0]
        bxs_all = sorted(set(c["bx"] for c in allv))
        cap = ((f"The 2024 beam spot (BetafuncEvtVtxGenerator, the payload of the 2024 global tag): the collisions spread along the beam over sigma_z = {self.m[0]:.1f} cm around z0 = {r.z0:.1f} cm "
                f"and over sigma_t = {self.m[1] * 1e3:.0f} ps in time, {r.sigma_x0 * 1e4:.0f} um across. " if r.name == "run3" else
                f"Two bunches of 9 cm cross at 510 urad, crab-rotated by 380 urad and focused to beta* = 20 cm: the collisions spread over sigma_z = {self.m[0]:.1f} cm and sigma_t = {self.m[1] * 1e3:.0f} ps. ")
               if k < self.k_bunch else
               f"The {int(f2 * len(it))} of {len(it)} in-time collisions of our crossing, and the signal (star) at z = {sig['z']:.2f} cm, t = {sig['t'] * 1e3:.0f} ps; "
               f"the nearest neighbour is {ph.s['nearest']['dz_mm']:.1f} mm away in z and {ph.s['nearest']['dt_ps']:.0f} ps in time." if k < self.k_in else
               f"The {len(bxs_all) - 1} other crossings of the window bx = {bxs_all[0]:+d} ... {bxs_all[-1]:+d}, 25 ns apart: {len(allv)} collisions in all, early in dark yellow, late in salmon; "
               f"in space they are the same crowd, only the clock tells them apart.")
        chrome(fig, self.number, self.title, cap)
        ax = fig.add_axes([0.06, 0.15, 0.40, 0.75])
        style_2d(ax, 9)
        # the bunches: two Gaussians moving in from +-30 cm (f1), then the overlap density
        if f1 < 1:
            zz = np.linspace(-40, 40, 400)
            zc = 30 * (1 - f1)
            for sign, col in ((+1, COL_ND), (-1, COL_DD)):
                g = np.exp(-((zz + sign * zc) ** 2) / (2 * r.sigma_s ** 2))
                ax.fill_between(zz, -0.8 + 0.0 * zz, -0.8 + 0.8 * g, color=col, alpha=0.25, lw=0)
            ax.text(0.5, 0.92, f"two bunches approach at c; the overlap density of the model appears as they meet", transform=ax.transAxes, ha="center", fontsize=9, color=TEXT2)
        ax.contourf(self.Z, self.T * 1e3, self.D * min(1.0, f1 * 1.5), levels=np.linspace(0.02, 1, 12), cmap="Greens", alpha=0.9 * f1)
        ax.set_xlim(-20, 20)
        ax.set_ylim(-800, 800)
        ax.set_xlabel("z  [cm]", fontsize=11)
        ax.set_ylabel("t  [ps]", fontsize=11)
        n = int(f2 * len(it))
        if n:
            ax.plot([c["z"] for c in it[:n]], [c["t"] * 1e3 for c in it[:n]], "o", color=COL_IN, ms=4, mec="none")
        if f2 > 0.3:
            ax.plot([sig["z"]], [sig["t"] * 1e3], "*", color=COL_B, ms=16, mec="white", zorder=5)
        ax.set_title(f"the beam spot: sigma_z = {self.m[0]:.2f} cm, sigma_t = {self.m[1] * 1e3:.0f} ps", fontsize=10.5, color=TEXT)
        # the clock: the crossings of the window
        ax2 = fig.add_axes([0.54, 0.15, 0.44, 0.75])
        style_2d(ax2, 9)
        bxs = sorted(set(c["bx"] for c in allv))
        far = max(abs(bxs[0]), abs(bxs[-1])) if bxs else 1
        shown = [0] if k < self.k_in else [bx for bx in bxs if abs(bx) <= 1 + (far - 1) * f3 * 1.0001]
        for c in allv:
            if c["bx"] not in shown or (c["bx"] == 0 and c["k"] > it[min(n, len(it)) - 1]["k"] if n else c["bx"] == 0):
                continue
            col = COL_EARLY if c["bx"] < 0 else COL_IN if c["bx"] == 0 else COL_LATE
            ax2.plot(c["t"], c["z"], "o", color=col, ms=3, mec="none", alpha=0.8)
        if f2 > 0.3:
            ax2.plot([sig["t"]], [sig["z"]], "*", color=COL_B, ms=16, mec="white", zorder=5)
        for bx in shown:
            nn = sum(1 for c in allv if c["bx"] == bx)
            ax2.text(bx * core.BUNCH_SPACING, 23, f"{bx:+d}\n{nn}", ha="center", va="top", fontsize=7.5 if far > 4 else 9, color=COL_EARLY if bx < 0 else COL_IN if bx == 0 else COL_LATE)
        ax2.set_xlim((bxs[0] - 0.7) * core.BUNCH_SPACING if bxs else -90, (bxs[-1] + 0.7) * core.BUNCH_SPACING if bxs else 90)
        ax2.set_ylim(-25, 25)
        ax2.set_xlabel("t  [ns]: 25 ns between crossings (labels: bx and its collisions)", fontsize=11)
        ax2.set_ylabel("z  [cm]", fontsize=11)
        ax2.set_title("in time (blue), early (dark yellow), late (salmon)", fontsize=10.5, color=TEXT)


# --- scene 3: one member of the crowd -----------------------------------------------------------------------
class MemberScene(Scene):
    number, title = 3, "one member of the crowd: the four kinds of minimum-bias collision"
    KINDS = ["ND", "SD", "DD", "CD"]
    LONG = dict(ND="non-diffractive", SD="single diffractive", DD="double diffractive", CD="central diffractive")
    COLS = dict(ND=COL_ND, SD=COL_SD, DD=COL_DD, CD=COL_CD)

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.k_kind = [0, 4 * fps, 8 * fps, 11 * fps, 14 * fps]      # nd, sd, dd, cd end frames
        self.k_table = self.k_kind[-1] + 5 * fps
        self.nframes = self.k_table + 7 * fps
        self.nf = np.array([e["nfinal"] for e in phys.lib.events])
        self.kind = np.array([e["kind"] for e in phys.lib.events])

    def phases(self):
        return [("nd", self.k_kind[0], self.k_kind[1]), ("sd", self.k_kind[1], self.k_kind[2]), ("dd", self.k_kind[2], self.k_kind[3]), ("cd", self.k_kind[3], self.k_kind[4]),
                ("table", self.k_kind[4], self.k_table), ("library", self.k_table, self.nframes)]

    def stage(self, k):
        """0..3 the four kinds, 4 the table, 5 the library."""
        for i in range(4):
            if k < self.k_kind[i + 1]:
                return i
        return 4 if k < self.k_table else 5

    def table(self, ax, n_rows, highlight=None, full=False):
        ph = self.phys
        ax.axis("off")
        cols = ["kind", "sigma [mb]", "share", "MPI", "stable", "charged", "central", "lead pT", "sum pT"]
        x = [0.0, 0.22, 0.33, 0.42, 0.53, 0.65, 0.77, 0.88, 1.0]
        y = 0.95
        for xx, c in zip(x, cols):
            ax.text(xx, y, c, fontsize=9, color=TEXT2, transform=ax.transAxes, ha="left" if xx == 0 else "right", weight="bold")
        ax.plot([0, 1], [y - 0.025, y - 0.025], "-", color=TEXT2, lw=0.8, transform=ax.transAxes)
        for i, kd in enumerate(self.KINDS[:n_rows]):
            p = ph.per_kind[kd]
            y -= 0.075
            col = self.COLS[kd]
            vals = [kd, f"{ph.sigma[kd]:.2f}", f"{p['share']:.1f}%", f"{p['nmpi']:.1f}", f"{p['stable']:.0f}", f"{p['charged']:.0f}", f"{p['central']:.0f}", f"{p['lead']:.1f}", f"{p['sumpt']:.0f}"]
            for xx, v in zip(x, vals):
                ax.text(xx, y, v, fontsize=10.5 if kd == highlight else 9.5, color=col, weight="bold" if kd == highlight else "normal", transform=ax.transAxes,
                        ha="left" if xx == 0 else "right", family="monospace" if xx > 0 else None)
            ax.text(0.0, y - 0.03, self.LONG[kd], fontsize=7.5, color=col, transform=ax.transAxes)
        if full:
            y -= 0.075
            ax.plot([0, 1], [y + 0.055, y + 0.055], "-", color=TEXT2, lw=0.8, transform=ax.transAxes)
            vals = ["all", f"{sum(ph.sigma.values()):.2f}", "100%", f"{np.mean([max(1, e.get('nmpi', 0)) for e in ph.lib.events]):.1f}", f"{self.nf.mean():.0f}",
                    f"{np.mean([e['nch'] for e in ph.lib.events]):.0f}", f"{np.mean([int(((e['pt'] > 0.5) & (np.abs(e['eta']) < 2.5)).sum()) for e in ph.lib.events]):.0f}",
                    f"{np.mean([e['pt_max'] for e in ph.lib.events]):.1f}", f"{np.mean([e['sumpt'] for e in ph.lib.events]):.0f}"]
            for xx, v in zip(x, vals):
                ax.text(xx, y, v, fontsize=9.5, color=TEXT, weight="bold", transform=ax.transAxes, ha="left" if xx == 0 else "right", family="monospace" if xx > 0 else None)
            ax.text(0.0, y - 0.03, "minimum bias: all four in their natural proportions", fontsize=7.5, color=TEXT, transform=ax.transAxes)
            ax.text(0.0, y - 0.11, "means over the 2000 collisions of the library; MPI = parton-parton\ninteractions (the hardest one plus the multiparton ones);\n"
                    "central: |eta| < 2.5, pT > 0.5 GeV; sigma: Pythia at 13.6 TeV, the total and\nthe elastic cross section fixed by the tune (100.3 and 21.9 mb),\n"
                    "the diffractive pieces from Pythia's parametrisation, ND what remains.",
                    fontsize=8.5, color=TEXT2, transform=ax.transAxes, va="top")

    def draw(self, fig, k):
        ph = self.phys
        st = self.stage(k)
        f = ramp(k, self.k_table, self.nframes - 2 * self.fps)
        if st < 4:
            kd = self.KINDS[st]
            e = ph.examples[kd]
            p = ph.per_kind[kd]
            what = {"ND": "both protons break up, several parton pairs interact: a plateau of soft particles over ten units of rapidity. Seven of ten members of the crowd",
                    "SD": "one proton survives intact, the other dissociates: a gap in rapidity on the surviving side. One in six of the crowd",
                    "DD": "both protons dissociate into low-mass systems: a gap in the middle, little in the tracker. One in nine",
                    "CD": "both protons survive and a small system is made in the centre: a few particles. One in fifty"}[kd]
            cap = (f"{self.LONG[kd].capitalize()} ({kd}), library event {e['no']}: {what}; this one has {e['nfinal']} particles, {max(1, e.get('nmpi', 0))} parton-parton interactions, "
                   f"{e['n_central']} charged in the tracker's reach, the hardest at {e['pt_max']:.1f} GeV. On average a {kd} collision brings {p['stable']:.0f} particles.")
        elif st == 4:
            cap = (f"The table of the crowd: sigma_inel = {sum(ph.sigma.values()):.2f} mb at 13.6 TeV, the four kinds in the proportions of their cross sections, "
                   f"and what each brings on average: from {ph.per_kind['CD']['stable']:.0f} to {ph.per_kind['ND']['stable']:.0f} particles, {ph.per_kind['CD']['central']:.0f} to "
                   f"{ph.per_kind['ND']['central']:.0f} in the tracker's reach, and never more than a few GeV in one particle.")
        else:
            cap = (f"The whole library: {len(ph.lib)} collisions with {self.nf.mean():.0f} particles on average and a median of {np.median(self.nf):.0f}, from 3 to {self.nf.max()}; "
                   f"shares ND {100 * (self.kind == 'ND').mean():.0f}%, SD {100 * (self.kind == 'SD').mean():.0f}%, DD {100 * (self.kind == 'DD').mean():.0f}%, CD {100 * (self.kind == 'CD').mean():.0f}%, "
                   f"as PYTHIA's cross sections say.")
        chrome(fig, self.number, self.title, cap)
        ax = eta_phi_axes(fig, [0.05, 0.15, 0.42, 0.75])
        ax.set_xlim(-10, 10)
        kd = self.KINDS[min(st, 3)]
        e = ph.examples[kd]
        col = self.COLS[kd]
        ax.scatter(np.clip(e["eta"], -9.9, 9.9), np.arctan2(e["p4"][:, 2], e["p4"][:, 1]), s=[msize(p, 1.0) ** 2 / 4 for p in e["pt"]], color=col, alpha=0.85, lw=0)
        prot = np.abs(e["ids"]) == 2212
        if prot.any() and kd != "ND":
            far = prot & (e["p4"][:, 0] > 0.45 * 6800.0)
            for ee in np.clip(e["eta"][far], -9.9, 9.9):
                ax.annotate("p", (ee, 0.0), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=10, color=col, weight="bold")
        for x in (-5, 5):
            ax.axvline(x, color=GRID, lw=2)
        for x in (-2.5, 2.5):
            ax.axvline(x, color=GRID, lw=1.2, ls="--")
        ax.set_title(f"{self.LONG[kd]} ({kd}), event {e['no']}: {e['nfinal']} particles (tracker |eta| < 2.5 dashed, HF to 5)", fontsize=10.5, color=TEXT)
        ax2 = fig.add_axes([0.52, 0.15, 0.46, 0.75])
        if st <= 4:
            self.table(ax2, st + 1 if st < 4 else 4, highlight=kd if st < 4 else None, full=(st == 4))
            return
        style_2d(ax2, 9)
        n_show = int(f * len(ph.lib))
        edges = np.concatenate([[0], np.logspace(0, 3, 31)])
        if n_show:
            ax2.hist([self.nf[:n_show][self.kind[:n_show] == kk] for kk in self.KINDS], bins=edges, stacked=True, color=[self.COLS[kk] for kk in self.KINDS], label=self.KINDS)
            ax2.legend(loc="upper left", fontsize=9)
        ax2.axvline(e["nfinal"], color=col, lw=2)
        ax2.text(e["nfinal"] * 1.1, 0.9, f"this one: {e['nfinal']}", transform=ax2.get_xaxis_transform(), color=col, fontsize=9)
        ax2.axvline(749, color=COL_B, lw=1.2, ls=":")
        ax2.text(749 * 0.9, 0.75, "our signal\nevent: 749", transform=ax2.get_xaxis_transform(), color=COL_B, fontsize=8.5, ha="right")
        ax2.set_xscale("log")
        ax2.set_xlim(1, 1200)
        ax2.set_ylim(0, 330)
        ax2.set_xlabel("final particles per collision", fontsize=11)
        ax2.set_ylabel("collisions", fontsize=11)
        ax2.set_title(f"the library: {n_show} of {len(ph.lib)} collisions", fontsize=10.5, color=TEXT)


# --- scene 4: the median ------------------------------------------------------------------------------------
class MedianScene(Scene):
    number, title = 4, "the median energy density: what one unit of area carries"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.k_grid = 7 * fps
        self.nframes = self.k_grid + 5 * fps

    def phases(self):
        return [("grid", 0, self.k_grid), ("median", self.k_grid, self.nframes)]

    def draw(self, fig, k):
        ph = self.phys
        f = ramp(k, 0, self.k_grid)
        n = int(round(f * len(ph.order)))
        cells = ph.cells_cum[n]
        vals = cells.ravel()
        med, mean = float(np.median(vals)), float(vals.mean())
        if k < self.k_grid:
            cap = (f"The (eta, phi) plane is cut into {cells.size} cells of {ph.dy:.2f} x {ph.dphi:.2f} and the pT of every particle goes to its cell: {n} collisions stacked, "
                   f"the cells hold from {vals.min():.0f} to {vals.max():.0f} GeV per unit area.")
        else:
            cap = (f"The median of the cells is rho = {med:.1f} GeV per unit area; the mean, {mean:.1f}, is pulled up by the jets in the tail, which is why the median is used. "
                   f"The jet median of Cacciari and Salam on kT jets gives {ph.s['rho_kt']:.1f}." if ph.s.get("rho_kt") else f"The median of the cells is rho = {med:.1f} GeV per unit area.")
        chrome(fig, self.number, self.title, cap)
        ax = fig.add_axes([0.05, 0.15, 0.50, 0.75])
        style_2d(ax, 9)
        im = ax.imshow(cells.T, origin="lower", extent=(-core.GRID_RAP_MAX, core.GRID_RAP_MAX, 0, 2 * math.pi), aspect="auto", cmap="YlOrBr", vmin=0,
                       vmax=max(3.0 * max(float(np.median(ph.cells_cum[-1])), 1.0), vals.max() * 0.5))
        ax.set_xlabel(r"$y$", fontsize=11)
        ax.set_ylabel(r"$\phi$", fontsize=11)
        ax.set_title(f"pT per unit area in the {cells.shape[0]} x {cells.shape[1]} cells, {n} collisions", fontsize=10.5, color=TEXT)
        for j in ph.jets[:6]:
            ax.add_patch(Circle((j["y"], j["phi"] % (2 * math.pi)), 0.4, fill=False, ec=COL_EVENT if j["genjet"] == 0 else TEXT2, lw=1.2))
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("GeV per unit area", fontsize=9)
        ax2 = fig.add_axes([0.64, 0.15, 0.33, 0.75])
        style_2d(ax2, 9)
        top = max(vals.max(), 50.0)
        ax2.hist(vals, bins=np.linspace(0, top * 1.02, 30), color=COL_PU, alpha=0.7)
        ax2.axvline(med, color=COL_EVENT, lw=2.5)
        ax2.text(med, 0.95, f"  median = rho = {med:.1f}", transform=ax2.get_xaxis_transform(), color=COL_EVENT, fontsize=10, weight="bold")
        if k >= self.k_grid:
            ax2.axvline(mean, color=COL_CMSSW, lw=1.2, ls="--")
            ax2.text(mean, 0.85, f"  mean {mean:.1f}", transform=ax2.get_xaxis_transform(), color=COL_CMSSW, fontsize=9)
            ax2.text(0.98, 0.6, "the jets sit here,\nin the tail", transform=ax2.transAxes, ha="right", fontsize=9, color=TEXT2)
        ax2.set_xlabel("pT per unit area in a cell  [GeV]", fontsize=11)
        ax2.set_ylabel("cells", fontsize=11)
        ax2.set_title("the cells, and their median", fontsize=10.5, color=TEXT)


# --- scene 5: our jet ----------------------------------------------------------------------------------------
class OurJetScene(Scene):
    number, title = 5, "our jet: the crowd inside the circle, and the subtraction"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.k_family = 3 * fps
        self.k_crowd = self.k_family + 9 * fps
        self.nframes = self.k_crowd + 6 * fps
        ref = phys.ref
        self.ref_rows = [r for r, j in phys.genjets_of["ak4"].items() if j == 0]
        self.fam = np.array([kin(phys.ev.p[r].p) for r in self.ref_rows])
        # the pileup constituents of the mixed jet, in the order of the collisions
        j = phys.mixed_b
        self.pu_in = []
        for kk in phys.order:
            arr = phys.pu.get(kk)
            if arr is None:
                continue
            for e, f, p, jj, _y in arr:
                if j is not None and int(jj) == j["rank"]:
                    self.pu_in.append((e, f, p, kk))
        self.pu_in = np.array(self.pu_in) if self.pu_in else np.zeros((0, 4))
        self.cum = np.concatenate([[0.0], np.cumsum(self.pu_in[:, 2])]) if len(self.pu_in) else np.zeros(1)

    def phases(self):
        return [("family", 0, self.k_family), ("crowd", self.k_family, self.k_crowd), ("subtract", self.k_crowd, self.nframes)]

    def draw(self, fig, k):
        ph = self.phys
        ref, j = ph.ref, ph.mixed_b
        f = ramp(k, self.k_family, self.k_crowd)
        n = int(round(f * len(self.pu_in)))
        sub = k >= self.k_crowd
        if k < self.k_family:
            cap = f"The family alone: the photograph of chapter 7, jet [0] with {ref['pt']:.2f} GeV in {ref['n']} particles, R = 0.4."
        elif not sub:
            cap = (f"The crowd pours in: {n} pileup particles inside the jet, {self.cum[n]:.1f} GeV, from {len(set(int(x) for x in self.pu_in[:n, 3]))} different collisions; "
                   f"the mixed jet will hold {j['pt']:.2f} GeV in {j['n']} particles.")
        else:
            cap = (f"The subtraction: A = {j['area']:.3f} (ghosts), rho A = {j['rho_area']:.2f} GeV, pT - rho A = {j['pt_sub']:.2f} against the signal's {j['pt_signal']:.2f} inside: "
                   f"residual {j['pt_sub'] - j['pt_signal']:+.2f} GeV. The mean of the crowd is gone; its fluctuation is not.")
        chrome(fig, self.number, self.title, cap)
        ax = fig.add_axes([0.05, 0.15, 0.50, 0.75])
        style_2d(ax, 9)
        ax.set_xlim(ref["eta"] - 0.75, ref["eta"] + 0.75)
        ax.set_ylim(ref["phi"] - 0.75, ref["phi"] + 0.75)
        ax.set_aspect("equal")
        ax.add_patch(Circle((ref["eta"], ref["phi"]), 0.4, fill=False, ec=COL_B, lw=2))
        if j is not None and f > 0:
            ax.add_patch(Circle((j["eta"], j["phi"]), 0.4, fill=False, ec=COL_EVENT, lw=1.2, ls="--"))
        # the rest of the signal nearby in grey
        for i, r in enumerate(ph.sig_rows):
            e, fphi, p = ph.sig_kin[i]
            if r not in self.ref_rows and abs(e - ref["eta"]) < 0.8 and abs(fphi - ref["phi"]) < 0.8:
                ax.scatter(e, fphi, s=msize(p, 1.0) ** 2 / 4, color=COL_UNCL, lw=0)
        ax.scatter(self.fam[:, 0], self.fam[:, 1], s=[msize(p, 1.3) ** 2 / 4 for p in self.fam[:, 2]], color=COL_B, alpha=0.95, lw=0, zorder=4)
        if n:
            ax.scatter(self.pu_in[:n, 0], self.pu_in[:n, 1], s=[msize(p, 1.0) ** 2 / 4 for p in self.pu_in[:n, 2]], color=COL_PU, alpha=0.8, lw=0, zorder=3)
        ax.set_xlabel(r"$\eta$", fontsize=11)
        ax.set_ylabel(r"$\phi$", fontsize=11)
        ax.set_title(f"the circle of R = 0.4 around our jet: {ref['n']} family particles" + (f" + {n} of the crowd" if n else ""), fontsize=10.5, color=TEXT)
        # the ledger bars
        ax2 = fig.add_axes([0.63, 0.15, 0.34, 0.75])
        style_2d(ax2, 9)
        bars = [("photograph", ref["pt"], COL_B), ("+ crowd", ref["pt"] + self.cum[n] if not sub else j["pt"], COL_PU)]
        if sub:
            bars += [("- rho A", j["pt_sub"], COL_MODEL), ("signal inside", j["pt_signal"], COL_B)]
        xs = np.arange(len(bars))
        ax2.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], alpha=0.8)
        for x, b in zip(xs, bars):
            ax2.text(x, b[1] + 2, f"{b[1]:.1f}", ha="center", fontsize=10, color=TEXT)
        ax2.set_xticks(xs)
        ax2.set_xticklabels([b[0] for b in bars], fontsize=9)
        ax2.set_ylim(0, max(ref["pt"] + (self.cum[-1] if len(self.cum) else 0), j["pt"] if j else 0) * 1.2)
        ax2.set_ylabel("pT  [GeV]", fontsize=11)
        ax2.set_title("the ledger of our jet at this stop", fontsize=10.5, color=TEXT)
        if sub:
            ax2.text(0.5, 0.92, f"rho = {ph.s['rho_grid']:.1f} x A = {j['area']:.3f} = {j['rho_area']:.1f} GeV", transform=ax2.transAxes, ha="center", fontsize=9.5, color=COL_MODEL)


# --- scene 6: the ensemble ------------------------------------------------------------------------------------
class EnsembleScene(Scene):
    number, title = 6, "the crowd drawn again and again"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.k_fill = 8 * fps
        self.nframes = self.k_fill + 5 * fps
        st = phys.stats
        self.pu = np.array([r["circle"]["b"]["pt_pileup"] for r in st]) if st else np.zeros(0)
        self.ra = np.array([r["circle"]["b"]["rho_area"] for r in st]) if st else np.zeros(0)
        self.n = np.array([r["n_in_time"] for r in st]) if st else np.zeros(0)

    def phases(self):
        return [("fill", 0, self.k_fill), ("result", self.k_fill, self.nframes)]

    def draw(self, fig, k):
        ph = self.phys
        f = ramp(k, 0, self.k_fill)
        n = int(f * len(self.pu))
        pu, ra = self.pu[:n], self.ra[:n]
        res = pu - ra
        if k < self.k_fill:
            cap = f"{n} draws of the crowd on the same photograph: the pileup inside the circle of our jet and the estimate rho pi R^2."
        else:
            cap = (f"Over {len(self.pu)} draws the crowd puts {self.pu.mean():.1f} +- {self.pu.std():.1f} GeV inside the circle; rho A estimates {self.ra.mean():.1f} +- {self.ra.std():.1f}; "
                   f"the residual is {res.mean():+.1f} +- {res.std():.1f} GeV. The subtraction removes the average; the {res.std():.0f} GeV of spread is the price of the crowd, "
                   f"and the next chapters will try to lower it with tracks and time.")
        chrome(fig, self.number, self.title, cap)
        ax = fig.add_axes([0.06, 0.15, 0.40, 0.75])
        style_2d(ax, 9)
        if n:
            ax.plot(ra, pu, "o", color=COL_PU, ms=4, mec="none", alpha=0.7)
        lo, hi = (min(self.pu.min(), self.ra.min()) * 0.9, max(self.pu.max(), self.ra.max()) * 1.05) if len(self.pu) else (0, 1)
        ax.plot([lo, hi], [lo, hi], "-", color=TEXT2, lw=1)
        if ph.mixed_b:
            ax.plot([ph.mixed_b["rho_area"]], [ph.mixed_b["pt_pileup"]], "*", color=COL_B, ms=16, mec="white", zorder=5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("rho A, the estimate  [GeV]", fontsize=11)
        ax.set_ylabel("pileup inside the circle, the truth  [GeV]", fontsize=11)
        ax.set_title(f"{n} draws; the star is our crossing", fontsize=10.5, color=TEXT)
        ax2 = fig.add_axes([0.55, 0.15, 0.42, 0.75])
        style_2d(ax2, 9)
        top = float(self.pu.max()) * 1.1 if len(self.pu) else 100.0
        low = -0.45 * top
        if n > 1:
            ax2.hist(pu, bins=np.linspace(0, top, 36), color=COL_PU, alpha=0.6, label=f"crowd inside: {pu.mean():.1f} +- {pu.std():.1f}")
            ax2.hist(res, bins=np.linspace(low, top - low + low, 36), color=COL_MODEL, alpha=0.6, label=f"after rho A: {res.mean():+.1f} +- {res.std():.1f}")
            ax2.legend(loc="upper right", fontsize=10)
        ax2.axvline(0, color=TEXT2, lw=1)
        ax2.set_xlim(low, top)
        ax2.set_xlabel("GeV", fontsize=11)
        ax2.set_ylabel("draws", fontsize=11)
        ax2.set_title("before and after the subtraction", fontsize=10.5, color=TEXT)


# --- the narration --------------------------------------------------------------------------------------------
def narration(phys: Physics, scenes: List[Scene]) -> Dict[str, str]:
    s = phys.s
    j = phys.mixed_b
    lib = phys.lib
    nf = np.array([e["nfinal"] for e in lib.events])
    st = phys.stats
    pu = np.array([r["circle"]["b"]["pt_pileup"] for r in st]) if st else np.zeros(1)
    ra = np.array([r["circle"]["b"]["rho_area"] for r in st]) if st else np.zeros(1)
    m = phys.region.moments()
    n_in = s["n_in_time"]
    lines = {
        "crowd_photo": (f"The photograph is taken, and now the family sets out. ... At the L H C of 2024, every bunch crossing held about {s['mu']:.0f} proton proton collisions, "
                        "and ours is only one of them. This is the photograph of chapter seven, in the eta phi plane: the signal alone."),
        "crowd_stack": (f"The other collisions of the same crossing arrive, one by one. ... They are soft, they come from everywhere, and there are {n_in} of them in ours, "
                        f"a Poisson draw with mean {s['mu']:.0f}. Watch the counter: {phys.n_run[-1]} particles, and an energy density, rho, that grows by about {s['rho_grid'] / n_in:.1f} GeV per unit area per collision."),
        "crowd_full": (f"The crowd is complete. Anti k T, run on everything, finds {s['n_jets']} jets, and our jet now holds {j['pt']:.1f} GeV instead of {phys.ref['pt']:.1f}: "
                       f"{j['pt_pileup']:.0f} GeV of strangers have walked into the circle."),
        "road_bunches": (f"Where do they come from? ... Two bunches of protons pass through each other, and where they overlap is the beam spot: in 2024, "
                         f"{m[0]:.1f} centimetres long and {m[1] * 1e3:.0f} picoseconds wide, as the simulation draws it from the measured beam parameters."),
        "road_intime": (f"The in-time collisions of our crossing are spread along it, each with its own position and time. The signal is the star. "
                        f"Its nearest neighbour is {s['nearest']['dz_mm']:.2f} millimetres away in z and {s['nearest']['dt_ps']:.0f} picoseconds in time: "
                        f"a vertex finder will have to tell them apart, and Run three has no timing layer to help it."),
        "road_outoftime": (f"And the road does not stop at one crossing. Twenty five nanoseconds apart, the neighbouring bunches collide too, "
                           f"{-s['bunches'][0]} crossings before ours and {s['bunches'][1]} after in the production window; "
                           "their collisions sit in the same place, only earlier or later, and the slow detectors will see them. In time, early and late: the colours of the pages to come."),
        "member_nd": (f"One member of the crowd. Pythia makes four kinds of inelastic collision. The non-diffractive one: both protons break up, several parton pairs interact, "
                      f"and the particles spread over the whole rapidity range. This one has {phys.ex_nd['nfinal']} particles, {max(1, phys.ex_nd.get('nmpi', 0))} parton pairs, the hardest particle at "
                      f"{phys.ex_nd['pt_max']:.1f} GeV. Seven of ten collisions look like this, and they carry most of the crowd."),
        "member_sd": (f"The single-diffractive one: a proton survives, the other dissociates, and a gap in rapidity opens on the surviving side. {phys.ex_sd['nfinal']} particles here, "
                      f"{phys.per_kind['SD']['stable']:.0f} on average, and only {phys.per_kind['SD']['central']:.0f} in the tracker's reach. One collision in six."),
        "member_dd": (f"The double-diffractive one: both protons dissociate into low-mass systems, with a gap in the middle. {phys.per_kind['DD']['stable']:.0f} particles on average, "
                      f"almost none in the tracker. One in nine."),
        "member_cd": (f"And the central-diffractive one: both protons survive, and a small system is made in the centre. {phys.per_kind['CD']['stable']:.0f} particles on average. "
                      f"One in fifty, the rarest and the emptiest."),
        "member_table": (f"Put together, this is the table of the crowd. The inelastic cross section, {sum(phys.sigma.values()):.1f} millibarn, is fixed by the tune; the four kinds come in the "
                         f"proportions of their cross sections; and on average a collision brings {nf.mean():.0f} particles, "
                         f"{np.mean([int(((e['pt'] > 0.5) & (np.abs(e['eta']) < 2.5)).sum()) for e in phys.lib.events]):.0f} of them in the tracker's reach, and nothing above a few GeV."),
        "member_library": (f"The library: two thousand collisions generated once with Pythia at thirteen point six T e V, {nf.mean():.0f} particles on average, a median of {np.median(nf):.0f}, "
                           f"from three to {nf.max()}. The crowd is soft, and very uneven. The mixing reads them at random, as the production does."),
        "median_grid": ("How much does the crowd put in one unit of area? Cut the plane into cells, add the transverse momentum of every particle to its cell, "
                        "and watch the cells fill as the collisions stack."),
        "median_median": (f"The median of the cells is rho: {s['rho_grid']:.0f} GeV per unit area. The mean would be higher, because the jets sit in the tail; "
                          "the median does not see them. That is the one number the subtraction needs."),
        "ourjet_family": (f"Back to our jet. The family alone: {phys.ref['n']} particles, {phys.ref['pt']:.1f} GeV, in a circle of radius zero point four."),
        "ourjet_crowd": (f"The crowd pours in: {j['n_pileup']} particles from {j['n_collisions']} different collisions, {j['pt_pileup']:.0f} GeV, and the jet reads {j['pt']:.0f}."),
        "ourjet_subtract": (f"The subtraction. The area of the jet, measured with ghosts, is {j['area']:.2f}; rho times the area is {j['rho_area']:.0f} GeV; "
                            f"take it out, and {j['pt_sub']:.1f} is left, against the {j['pt_signal']:.1f} the family really has inside. "
                            f"The average of the crowd is gone. What stays is the fluctuation."),
        "ensemble_fill": (f"Draw the crowd again and again on the same photograph, four hundred times. Each time, the pileup inside the circle, and the estimate rho A."),
        "ensemble_result": (f"The crowd puts {pu.mean():.0f} GeV inside on average, with a spread of {pu.std():.0f}. The estimate removes the average, and {np.std(pu - ra):.0f} GeV of spread remain. "
                            "That spread is the price of the crowd on a hundred GeV jet, and the chapters ahead, with tracks, with vertices and with time, are about paying less of it."),
        "outro": ("The family is on the road, in the middle of the crowd. ... Ahead lies the border: the detector, where every particle will be seen, "
                  "smeared and counted, and the crowd with it. The next film crosses it."),
    }
    return lines


def score_spec(film: snd.Film) -> snd.ScoreSpec:
    t_full = snd.voice_at_phase(film, "crowd_full")
    return snd.film_score(film, [snd.Dm, snd.Gm, snd.Bb, snd.F, snd.C, snd.A], thuds=[t_full + 0.2] if t_full is not None else [])


# --- driver -----------------------------------------------------------------------------------------------
def build_scenes(names: List[str], phys: Physics, args) -> List[Scene]:
    cls = dict(crowd=CrowdScene, road=RoadScene, member=MemberScene, median=MedianScene, ourjet=OurJetScene, ensemble=EnsembleScene)
    return [cls[n](phys, args.fps) for n in names]


def write_video(film: snd.Film, out: Path, fps: float, dpi: int, audio: Optional[Path], jobs: int):
    snd.render_video(film, out, fps, dpi, audio, lambda fig, sc, k, t: sc.draw(fig, k), FIGSIZE, "b bbar event, standalone pileup", jobs=jobs)
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
    p.add_argument("-o", "--output", type=Path, default=HERE / "animate_standalone_pileup.mp4", help=".mp4 (needs ffmpeg) or .gif (silent)")
    p.add_argument("--scene", default="all", help="all, or a comma list of crowd,road,member,median,ourjet,ensemble (or 1..6)")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--dpi", type=int, default=150, help="150 gives 1920x1080, 100 gives 1280x720")
    p.add_argument("--step", type=int, default=1, help="render every step-th frame")
    p.add_argument("--fast", action="store_true", help="preview: --step 4 --dpi 60")
    p.add_argument("--frames-dir", type=Path, default=None)
    p.add_argument("--stills", type=Path, default=None, help="one key frame per scene (.pdf or .png)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--jobs", type=int, default=4, help="render the frames in this many processes (joined losslessly)")
    p.add_argument("--audio-only", type=Path, default=None, metavar="WAV", help="write just the soundtrack and exit")
    p.add_argument("--cache-dir", type=Path, default=HERE.parent.parent.parent / "build" / "narration" / "PU")
    snd.add_narration_args(p)
    p.add_argument("-i", "--input", type=Path, default=HERE.parent / "event_PU.lhe")
    p.add_argument("--library", type=Path, default=HERE.parent / "pileup_library.lhe.gz")
    p.add_argument("--summary", type=Path, default=HERE / "standalone_pileup.json")
    p.add_argument("--stats", type=Path, default=HERE / "standalone_pileup_stats.json")
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
    phys = Physics(args.input, args.library, args.summary, args.stats)
    print(f"event: mu = {phys.s['mu']:g}, {phys.s['n_in_time']} in-time collisions, rho = {phys.s['rho_grid']:.1f}; our jet {phys.ref['pt']:.2f} -> {phys.mixed_b['pt']:.2f} -> {phys.mixed_b['pt_sub']:.2f}")
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
        tmp = tempfile.TemporaryDirectory(prefix="jitjet_pu_")
        if ffmpeg and args.output.suffix.lower() != ".gif" and (clips or score):
            audio = Path(tmp.name) / "soundtrack.wav"
            snd.mix_audio(film.duration, film.voice_at, clips, score, args.music_db, audio)
            if clips:
                snd.write_srt(film.voice_at, clips, lines, args.output.with_suffix(".srt"))
        write_video(film, args.output, fps, args.dpi, audio, args.jobs)


if __name__ == "__main__":
    main()
