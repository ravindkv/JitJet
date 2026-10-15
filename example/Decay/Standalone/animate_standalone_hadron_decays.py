#!/usr/bin/env python3
"""Animated companion to standalone_hadron_decays.py.

Seven scenes, about 70 s in total, show how the primary hadrons of chapter 5 turn into what the
detector will see. Every number on screen is computed with the classes of that script: the
seed-1 decays of ../../Had/event_Had.lhe (the record ../event_Decay.lhe) are replayed.

  1  clock      A hundred B mesons born at once decay one by one: the survivors follow
                e^(-t/tau0), the clock has no memory, and the lifetime drawn for our event's
                B hadron is marked on the curve.
  2  walk       The b quark's B hadron walks away from the primary vertex in the transverse
                plane at gamma beta = 31 (one picosecond of proper time is a centimetre),
                decays, its charm hadron walks on and decays too, and the tracks of the
                charged grandchildren fan out from the displaced vertices.
  3  table      The decay table of the B hadron as a cumulative bar of branching ratios,
                coloured by class; the random number slides down the bar to the channel of
                our event; the partons of the channel turn into hadrons.
  4  cascade    The decay tree of the b quark's hadron grows generation by generation in the
                (eta, phi) plane around the B: B* -> B gamma, the B's products, their
                products, down to the stable particles, with the mothers fading.
  5  emigrant   A semileptonic decay from the ensemble: the lepton energy spectrum of the V-A
                matrix element fills up against phase space, then the neutrino leaves the
                cone and the visible pT drops.
  6  event      The whole event: the 365 primary hadrons decay in order of their laboratory
                decay time, the 7xx final particles take their place, with the counters.
  7  ensemble   The same primary hadrons decayed again and again: the flight distance of the
                B and the visible share of the jet's pT fill their histograms.

Examples (matplotlib is needed; on this Mac use /usr/bin/python3):
  python3 animate_standalone_hadron_decays.py                  # -> .mp4 next to this file
  python3 animate_standalone_hadron_decays.py --fast           # quick low-resolution preview
  python3 animate_standalone_hadron_decays.py --scene walk,cascade -o walk.gif
  python3 animate_standalone_hadron_decays.py --stills stills.pdf --no-video

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
sys.path.insert(0, str(HERE.parent.parent / "Had"))
import standalone_hadron_decays as core  # noqa: E402
import plot_journey_Had as jh  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402

COL_B, COL_BBAR, COL_HAD, COL_DEC = "#ff0000", "#0000ff", "#9b4dff", "#999999"
COL_STR, COL_MINI, COL_A, COL_EVENT, COL_PY = "#2a78d6", "#eb6834", "#8e5bb5", "#c0392b", "#7f7f7f"
COL_NU, COL_LEP, COL_CH, COL_GAM = "#1b9e77", "#d95f02", "#7570b3", "#e6ab02"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d6d3ca"
FIGSIZE = (12.8, 7.2)
STATS_FILE = HERE / "standalone_decay_stats.stats.json"
SCENE_NAMES = {"1": "clock", "2": "walk", "3": "table", "4": "cascade", "5": "emigrant", "6": "event", "7": "ensemble"}
C_MM_PER_PS = core.C_MM_PER_PS


def smooth(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ramp(k: int, k0: int, k1: int) -> float:
    return smooth((k - k0) / max(1, k1 - k0))


def tex(name: str) -> str:
    return jh.hadron_tex(name)


class Physics:
    def __init__(self, seed: int, input_path: Path):
        self.args = SimpleNamespace(input=input_path, tau0max=core.TAU0_MAX, limit_tau0=True, no_mix=False, stable_b=False, radius=0.4)
        self.ev, self.dec = core.run(self.args, seed)
        self.summary = core.summarise(self.ev, self.dec)
        self.b = self.summary["b"]["b"]
        self.seed = seed
        self.stats = json.loads(STATS_FILE.read_text()) if STATS_FILE.exists() else None
        self.semi = None          # (ev, dec, summary) of a decay of the record in which the b quark's B went semileptonic
        for s in range(seed + 1, seed + 40):
            ev, dec = core.run(self.args, s)
            sm = core.summarise(ev, dec)
            r = sm["b"]["b"]
            if r["cls"].startswith("semileptonic (mu)") or r["cls"].startswith("semileptonic (e)"):
                self.semi = (ev, dec, sm, s)
                break


# --- drawing helpers ------------------------------------------------------------------------------------
def chrome(fig, scene_no: int, title: str, caption: str):
    fig.clf()
    fig.patch.set_facecolor("white")
    fig.text(0.02, 0.965, f"Scene {scene_no}: {title}", fontsize=15, weight="bold", color=TEXT, va="center")
    fig.text(0.98, 0.965, "standalone_hadron_decays.py", fontsize=9.5, color=TEXT2, ha="right", va="center")
    fig.text(0.02, 0.03, caption, fontsize=10.5, color=TEXT, va="bottom", wrap=True)


def style_2d(ax, fontsize=8):
    ax.tick_params(labelsize=fontsize, colors=TEXT2)
    for sp in ax.spines.values():
        sp.set_color(TEXT2)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


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


# --- scene 1: the clock ---------------------------------------------------------------------------------
class ClockScene(Scene):
    number, title = 1, "the clock without memory"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.nframes = 9 * fps
        rng = np.random.default_rng(7)
        self.tau0 = core.tau0(511) / C_MM_PER_PS
        self.n = 100
        self.t_dec = -self.tau0 * np.log(rng.random(self.n))
        self.pos = rng.random((self.n, 2))
        self.t_max = 6.0

    def draw(self, fig, k):
        t = self.t_max * ramp(k, 0, self.nframes - 2 * self.fps)
        r = self.phys.b
        chrome(fig, self.number, self.title,
               f"A hundred $B^0$ mesons born at t = 0: each decays at a random moment drawn from $e^{{-t/\\tau_0}}$, $\\tau_0$ = {self.tau0:.2f} ps. "
               f"The survivors never age: the chance to decay in the next picosecond is the same at t = 0 and at t = 5 ps.\n"
               f"Our event's {tex(r['hadron'])} drew t = {r['tau_ps']:.2f} ps ({r['tau_over_tau0']:.2f} lifetimes).")
        ax = fig.add_axes([0.05, 0.17, 0.42, 0.71])
        alive = self.t_dec > t
        ax.scatter(self.pos[alive, 0], self.pos[alive, 1], s=140, color=COL_B, edgecolors="white", zorder=4)
        just = (~alive) & (self.t_dec > t - 0.25)
        ax.scatter(self.pos[just, 0], self.pos[just, 1], s=260, facecolors="none", edgecolors=COL_GAM, lw=2, zorder=3)
        ax.scatter(self.pos[~alive, 0], self.pos[~alive, 1], s=30, color=COL_DEC, zorder=2)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t = {t:.2f} ps: {alive.sum()} of {self.n} still alive", fontsize=11, color=TEXT)
        ax2 = fig.add_axes([0.55, 0.17, 0.42, 0.71])
        style_2d(ax2, 9)
        tt = np.linspace(0, self.t_max, 200)
        ax2.plot(tt, self.n * np.exp(-tt / self.tau0), color=COL_STR, lw=2, label=r"$N(t) = N_0\, e^{-t/\tau_0}$")
        tt2 = np.linspace(0, t, 100)
        ax2.plot(tt2, [np.sum(self.t_dec > x) for x in tt2], color=COL_B, lw=2.2, label="survivors of the hundred")
        ax2.axvline(r["tau_ps"], color=COL_EVENT, ls="--", lw=1.2)
        ax2.text(r["tau_ps"] + 0.08, 92, f"our {tex(r['hadron'])}: {r['tau_ps']:.2f} ps", color=COL_EVENT, fontsize=9.5)
        ax2.axvline(self.tau0, color=TEXT2, ls=":", lw=1)
        ax2.text(self.tau0 + 0.08, 100 * math.exp(-1) + 3, r"$\tau_0$: 37% left", color=TEXT2, fontsize=9)
        ax2.set_xlim(0, self.t_max)
        ax2.set_ylim(0, 105)
        ax2.set_xlabel("proper time t  [ps]", fontsize=10)
        ax2.set_ylabel("B mesons alive", fontsize=10)
        ax2.legend(loc="upper right", fontsize=9.5)


# --- scene 2: the walk ----------------------------------------------------------------------------------
class WalkScene(Scene):
    number, title = 2, "the walk: a centimetre per picosecond"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.nframes = 12 * fps
        ev, dec, r = phys.ev, phys.dec, phys.b
        self.r = r
        rw = r["hadron_row"]
        desc = core.descendants(ev, rw)
        # the weakly decaying particles of the chain with their flight (start, end, t_start, t_end in lab mm/c)
        self.flights = []
        for d in dec.decays:
            if (d["row"] == rw or d["row"] in desc) and d["tau"] > 1e-4:
                v0 = dec.vertex.get(d["row"], np.zeros(4))
                self.flights.append(dict(name=core.name(d["id"]), v0=v0, v1=d["v_dec"], t0=v0[0], t1=d["v_dec"][0], heavy=core.heavy_content(d["id"])))
        self.tracks = []
        for d in desc:
            q = ev.p[d]
            if q.status > 0 and core.is_charged(q.id):
                v = dec.vertex.get(d, np.zeros(4))
                self.tracks.append((v, q.p[1:] / np.linalg.norm(q.p[1:]), v[0], core.name(q.id)))
        self.t_end = max(f["t1"] for f in self.flights) * 1.4 + 5.0

    def draw(self, fig, k):
        r = self.r
        t = self.t_end * ramp(k, 0, self.nframes - 2 * self.fps)
        chrome(fig, self.number, self.title,
               f"The {tex(r['hadron'])} ($p_T$ = {r['pt_hadron']:.1f} GeV, $\\gamma\\beta$ = {r['gamma_beta']:.0f}) leaves the primary vertex: its proper time "
               f"t = {r['tau_ps']:.2f} ps is stretched to {r['tau_ps'] * r['gamma']:.0f} ps in the laboratory, a walk of L = {r['L']:.1f} mm "
               f"($L_{{xy}}$ = {r['Lxy']:.1f} mm). It decays into {r['channel']}; the {r['charm']} walks on another {r['L_charm']:.1f} mm.\n"
               f"The tracks of the {r['n_tracks']} charged grandchildren start from the displaced vertices: {r['n_tracks_d0_100um']} of them miss the beam line by more than 100 um.")
        lim = max(12.0, 1.3 * max(abs(f["v1"][j]) for f in self.flights for j in (1, 2)))
        ax = fig.add_axes([0.05, 0.17, 0.5, 0.73])
        style_2d(ax, 9)
        ax.plot([0], [0], "*", color=COL_EVENT, ms=16, zorder=6)
        ax.text(0.4, -0.9, "primary vertex", color=COL_EVENT, fontsize=9)
        for f in self.flights:
            if t <= f["t0"]:
                continue
            frac = min(1.0, (t - f["t0"]) / max(f["t1"] - f["t0"], 1e-9))
            v = f["v0"] + frac * (f["v1"] - f["v0"])
            col = COL_B if f["heavy"] == 5 else COL_CH
            ax.plot([f["v0"][1], v[1]], [f["v0"][2], v[2]], color=col, lw=2, ls="--", zorder=4)
            ax.plot([v[1]], [v[2]], "o", color=col, ms=10, mec="white", zorder=5)
            ax.annotate(tex(f["name"]), (v[1], v[2]), xytext=(6, 6), textcoords="offset points", fontsize=10, color=col)
        for v, n, t0, nm in self.tracks:
            if t <= t0:
                continue
            s = min(lim, (t - t0) * 1.0)
            ax.plot([v[1], v[1] + s * n[0]], [v[2], v[2] + s * n[1]], color=COL_DEC, lw=1.0, zorder=3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x  [mm]", fontsize=10)
        ax.set_ylabel("y  [mm]", fontsize=10)
        ax.set_title(f"laboratory time t = {t / C_MM_PER_PS:.0f} ps  (= {t:.1f} mm of light travel)", fontsize=11, color=TEXT)
        ax2 = fig.add_axes([0.62, 0.17, 0.35, 0.73])
        style_2d(ax2, 9)
        zlim = max(30.0, 1.3 * max(abs(f["v1"][3]) for f in self.flights))
        for f in self.flights:
            if t <= f["t0"]:
                continue
            frac = min(1.0, (t - f["t0"]) / max(f["t1"] - f["t0"], 1e-9))
            v = f["v0"] + frac * (f["v1"] - f["v0"])
            col = COL_B if f["heavy"] == 5 else COL_CH
            ax2.plot([f["v0"][3], v[3]], [math.hypot(*f["v0"][1:3]), math.hypot(*v[1:3])], color=col, lw=2, ls="--")
            ax2.plot([v[3]], [math.hypot(*v[1:3])], "o", color=col, ms=9, mec="white")
        ax2.plot([0], [0], "*", color=COL_EVENT, ms=14)
        ax2.set_xlim(-zlim if r["vertex"][2] < 0 else -2, 2 if r["vertex"][2] < 0 else zlim)
        ax2.set_ylim(-0.5, lim)
        ax2.set_xlabel("z  [mm]", fontsize=10)
        ax2.set_ylabel("r  [mm]", fontsize=10)
        ax2.set_title("along the beam", fontsize=11, color=TEXT)


# --- scene 3: the table ---------------------------------------------------------------------------------
CLASS_COL = {"partonic": COL_STR, "explicit": COL_A, "semileptonic (e)": COL_LEP, "semileptonic (mu)": COL_NU, "semileptonic (tau)": COL_PY}


def channel_kind(prods, me):
    c = core.channel_class(prods, me)
    if c.startswith("semileptonic"):
        return c
    return "partonic" if any(core.is_parton(p) for p in prods) else "explicit"


class TableScene(Scene):
    number, title = 3, "the decay table and the roll of the dice"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.nframes = 10 * fps
        r = phys.b
        self.r = r
        d = next(x for x in phys.dec.decays if x["row"] == r["hadron_row"])
        self.d = d
        self.chans, self.total = core.open_channels(d["id_dec"])
        self.cum = np.cumsum([c[0] for c in self.chans]) / self.total
        self.k_chan = d["channel"]
        self.r_pick = (self.cum[self.k_chan - 1] if self.k_chan > 0 else 0.0) + 0.5 * self.chans[self.k_chan][0] / self.total

    def draw(self, fig, k):
        r, d = self.r, self.d
        f1 = ramp(k, 0, 5 * self.fps)
        f2 = ramp(k, 5 * self.fps, 8 * self.fps)
        chrome(fig, self.number, self.title,
               f"The {tex(core.name(d['id_dec']))} has {len(self.chans)} open channels whose branching ratios add up to one. A random number r picks the channel "
               f"in proportion to its branching ratio: ours lands on {tex(core.name(d['id_dec']))} $\\to$ {' '.join(tex(core.name(p)) for p in d['prods'])} "
               f"({100 * d['br'] / self.total:.1f}%, meMode {d['me']}).\n"
               + ("The channel is written at quark level: the partons are paired into colour singlets and the string flavour machinery of chapter 5 turns them into hadrons: "
                  f"{' '.join(tex(core.name(i)) for i in d['ids'])}." if any(core.is_parton(p) for p in d["prods"]) else "The channel lists hadrons: they are placed in phase space."))
        ax = fig.add_axes([0.06, 0.17, 0.18, 0.73])
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_xticks([])
        ax.set_ylabel("cumulative branching ratio", fontsize=10)
        style_2d(ax, 9)
        ax.grid(False)
        lo = 0.0
        for (br, me, prods), hi in zip(self.chans, self.cum):
            ax.axhspan(lo, hi, xmin=0.1, xmax=0.9, color=CLASS_COL[channel_kind(prods, me)], lw=0)
            lo = hi
        y = self.r_pick * f1
        ax.annotate("", xy=(0.95, y), xytext=(1.6, y), arrowprops=dict(arrowstyle="->", color=COL_EVENT, lw=2))
        ax.text(1.65, y, f"r = {y:.3f}", color=COL_EVENT, fontsize=10, va="center")
        for kind, col in CLASS_COL.items():
            share = sum(c[0] for c in self.chans if channel_kind(c[2], c[1]) == kind) / self.total
            ax.plot([], [], color=col, lw=8, label=f"{kind} {100 * share:.0f}%")
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=9)
        ax2 = fig.add_axes([0.52, 0.17, 0.45, 0.73])
        ax2.axis("off")
        lines = [f"the channel drawn ({100 * d['br'] / self.total:.2f}%):", f"   {tex(core.name(d['id_dec']))} $\\to$ " + " ".join(tex(core.name(p)) for p in d["prods"])]
        if any(core.is_parton(p) for p in d["prods"]):
            lines += ["", "partons $\\to$ hadrons (chapter 5's flavour machinery):", "   " + " ".join(tex(core.name(i)) for i in d["ids"])]
        lines += ["", f"{len(d['ids'])} direct products, {r['n_final']} final particles in the end", f"class: {r['cls']}"]
        n_show = int(len(lines) * f2) if f1 >= 1 else 0
        for i, ln in enumerate(lines[:n_show]):
            ax2.text(0.0, 0.9 - 0.08 * i, ln, fontsize=12 if i in (1, 4) else 10.5, color=TEXT if i not in (1, 4) else COL_B, transform=ax2.transAxes, va="top")


# --- scene 4: the cascade -------------------------------------------------------------------------------
class CascadeScene(Scene):
    number, title = 4, "the cascade: grandchildren, great-grandchildren"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.nframes = 11 * fps
        ev, dec, r = phys.ev, phys.dec, phys.b
        self.r = r
        self.gen = {r["string_row"]: 0}
        order = [r["string_row"]]
        i = 0
        while i < len(order):
            row = order[i]
            q = ev.p[row]
            if q.d1 > 0:
                for d_ in range(q.d1, max(q.d1, q.d2) + 1):
                    self.gen[d_] = self.gen[row] + 1
                    order.append(d_)
            i += 1
        self.rows = order
        self.n_gen = max(self.gen.values()) + 1
        k0 = core.eta(ev.p[r["string_row"]].p), core.phi(ev.p[r["string_row"]].p)
        self.center = k0
        self.span = 0.25

    def draw(self, fig, k):
        ev, r = self.phys.ev, self.r
        g = self.n_gen * ramp(k, 0, self.nframes - 2 * self.fps)
        chain_txt = " $\\to$ ".join(tex(n) for n in r["chain"])
        chrome(fig, self.number, self.title,
               f"The {tex(r['string_hadron'])} of the string decays generation by generation: {chain_txt} ... "
               f"until only particles the detector can see are left: {r['n_final']} of them ({r['n_charged_final']} charged, {r['n_photons_final']} photons, "
               f"{r['n_neutrinos']} neutrinos), all within a few tenths in $\\eta$ and $\\phi$ of the original $b$ quark.")
        ax = fig.add_axes([0.06, 0.17, 0.62, 0.73])
        style_2d(ax, 9)
        cx, cy = self.center
        for row in self.rows:
            gen = self.gen[row]
            if gen > g:
                continue
            q = ev.p[row]
            e_, f_ = core.eta(q.p), core.phi(q.p)
            f_ = f_ - 2 * math.pi * round((f_ - cy) / (2 * math.pi))
            alive = q.status > 0 or gen + 1 > g
            col = COL_B if core.heavy_content(q.id) == 5 else COL_CH if core.heavy_content(q.id) == 4 else (COL_NU if core.is_neutrino(q.id) else COL_GAM if q.id == 22 else COL_DEC if q.status > 0 else COL_HAD)
            ax.scatter(e_, f_, s=msize(core.pt(q.p), 3.0) ** 2 / 4, color=col, alpha=1.0 if alive else 0.25, edgecolors="white", zorder=5 if alive else 3)
            if q.m1 > 0 and row != r["string_row"]:
                m = ev.p[q.m1]
                me_, mf_ = core.eta(m.p), core.phi(m.p)
                mf_ = mf_ - 2 * math.pi * round((mf_ - cy) / (2 * math.pi))
                ax.plot([me_, e_], [mf_, f_], color=col, lw=0.8, alpha=0.5 if alive else 0.2, zorder=2)
            if alive and (core.pt(q.p) > 3 or core.heavy_content(q.id) or core.is_neutrino(q.id)):
                ax.annotate(tex(core.name(q.id)), (e_, f_), xytext=(4, 4), textcoords="offset points", fontsize=8.5, color=col)
        ax.set_xlim(cx - self.span, cx + self.span)
        ax.set_ylim(cy - self.span, cy + self.span)
        ax.set_xlabel(r"$\eta$", fontsize=11)
        ax.set_ylabel(r"$\phi$", fontsize=11)
        ax.set_title(f"generation {min(int(g), self.n_gen - 1)} of {self.n_gen - 1}", fontsize=11, color=TEXT)
        ax2 = fig.add_axes([0.72, 0.17, 0.26, 0.73])
        ax2.axis("off")
        shown = [row for row in self.rows if self.gen[row] <= g]
        alive = [row for row in shown if ev.p[row].status > 0 or self.gen[row] + 1 > g]
        ax2.text(0, 0.95, f"particles alive: {len(alive)}", fontsize=12, color=TEXT, transform=ax2.transAxes, weight="bold")
        counts = {}
        for row in alive:
            n = core.name(ev.p[row].id)
            counts[n] = counts.get(n, 0) + 1
        for i, (n, c) in enumerate(sorted(counts.items(), key=lambda kv: -kv[1])[:16]):
            ax2.text(0.05, 0.88 - 0.05 * i, f"{tex(n)}  x {c}", fontsize=10, color=TEXT, transform=ax2.transAxes)


# --- scene 5: the emigrant ------------------------------------------------------------------------------
class EmigrantScene(Scene):
    number, title = 5, "the emigrant: a neutrino leaves the family"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.nframes = 10 * fps
        M = core.m0(511)
        masses, ids = [0.0, 0.10566, core.m0(413)], [14, -13, -413]
        P = np.array([M, 0, 0, 0])
        self.e_va, self.e_ps = [], []
        dec = phys.dec
        for _ in range(4000):
            p, _ = dec.mgenerator(M, masses, 22, P, ids)
            if p is not None:
                self.e_va.append(p[1][0])
            p, _ = dec.mgenerator(M, masses, 0, P, ids)
            if p is not None:
                self.e_ps.append(p[1][0])
        self.e_va, self.e_ps = np.array(self.e_va), np.array(self.e_ps)
        self.semi = phys.semi

    def draw(self, fig, k):
        f1 = ramp(k, 0, 5 * self.fps)
        f2 = ramp(k, 5 * self.fps, 8 * self.fps)
        cap = (f"In a semileptonic decay $B\\to D^{{(*)}} \\ell\\nu$ the V-A matrix element $|M|^2 \\propto (p_B\\cdot p_\\nu)(p_D\\cdot p_\\ell)$ pushes the lepton and the "
               f"neutrino to high energies. The neutrino then leaves the jet for good: the detector will measure less $p_T$ than the quark had.")
        if self.semi:
            ev2, dec2, sm2, s2 = self.semi
            r2 = sm2["b"]["b"]
            c = r2["cones"][0.4]
            cap += (f"\nDecaying the same hadrons with seed {s2}: {tex(r2['hadron'])} $\\to$ {' '.join(tex(x) for x in r2['channel'].split())}, the neutrino takes "
                    f"{r2['e_nu']:.1f} GeV ({100 * r2['e_nu_frac']:.0f}% of the B); the cone R = 0.4 drops from {c['primaries']['pt']:.1f} to {c['visible']['pt']:.1f} GeV visible.")
        chrome(fig, self.number, self.title, cap)
        ax = fig.add_axes([0.06, 0.17, 0.42, 0.71])
        style_2d(ax, 9)
        n = max(1, int(len(self.e_va) * f1))
        bins = np.linspace(0, 2.6, 27)
        ax.hist(self.e_ps[:n], bins=bins, density=True, color=COL_PY, alpha=0.5, label="phase space")
        ax.hist(self.e_va[:n], bins=bins, density=True, histtype="step", color=COL_LEP, lw=2.2, label="V-A matrix element")
        ax.set_xlabel(r"$E_\mu$ in the B rest frame  [GeV]  ($B^0\to D^{*-}\mu^+\nu_\mu$)", fontsize=10)
        ax.set_ylabel("probability density", fontsize=10)
        ax.set_title(f"{n} decays drawn: <E> = {self.e_va[:n].mean():.2f} (V-A) against {self.e_ps[:n].mean():.2f} GeV (phase space)", fontsize=10, color=TEXT)
        ax.legend(loc="upper left", fontsize=9.5)
        ax2 = fig.add_axes([0.56, 0.17, 0.41, 0.71])
        style_2d(ax2, 9)
        if self.semi and f1 >= 1:
            ev2, dec2, sm2, s2 = self.semi
            r2 = sm2["b"]["b"]
            c = r2["cones"][0.4]
            labels = ["primary hadrons", "all final", "visible (no $\\nu$)", "charged"]
            vals = [c["primaries"]["pt"], c["final"]["pt"], c["visible"]["pt"], c["charged"]["pt"]]
            cols = [COL_HAD, COL_DEC, COL_STR, COL_CH]
            nshow = int(len(vals) * f2 + 0.999)
            ax2.bar(range(nshow), vals[:nshow], color=cols[:nshow])
            for i in range(nshow):
                ax2.text(i, vals[i] + 1, f"{vals[i]:.1f}", ha="center", fontsize=10, color=TEXT)
            ax2.set_xticks(range(4))
            ax2.set_xticklabels(labels, fontsize=9)
            ax2.set_ylim(0, max(vals) * 1.15)
            ax2.set_ylabel("$p_T$ in the cone R = 0.4 around the $b$ quark  [GeV]", fontsize=10)
            ax2.set_title(f"seed {s2}: the neutrino carries {r2['pt_nu']:.1f} GeV of $p_T$ away", fontsize=10, color=TEXT)
        else:
            ax2.axis("off")


# --- scene 6: the event ---------------------------------------------------------------------------------
class EventScene(Scene):
    number, title = 6, "the whole event: 365 hadrons become what the detector sees"

    def __init__(self, phys, fps):
        super().__init__(phys, fps)
        self.nframes = 11 * fps
        ev, dec = phys.ev, phys.dec
        self.prim = dec.primaries
        # every decay happens at its laboratory time; the frames follow the time on a log-like scale
        self.t_dec = {d["row"]: d["v_dec"][0] for d in dec.decays}
        ts = sorted(t for t in self.t_dec.values() if t > 0)
        self.t_max = ts[-1] * 1.05 if ts else 1.0
        self.order = sorted(self.t_dec, key=lambda r_: self.t_dec[r_])

    def draw(self, fig, k):
        ev, dec = self.phys.ev, self.phys.dec
        f = ramp(k, 0, self.nframes - 2 * self.fps)
        n_done = int(len(self.order) * f)
        done = set(self.order[:n_done])
        t_now = self.t_dec[self.order[n_done - 1]] if n_done else 0.0
        s = self.phys.summary
        chrome(fig, self.number, self.title,
               f"The decays in the order they happen in the laboratory: first the strong and electromagnetic ones at the primary vertex ($\\rho$, $K^*$, $\\pi^0$, ...), "
               f"then the charm and bottom hadrons millimetres and centimetres away. {s['n_primaries']} primary hadrons, {s['n_decays']} decays, "
               f"{s['n_final']} final particles: {s['n_charged']} charged, {s['n_photons']} photons, {s['n_neutrinos']} neutrinos; {s['n_primary_stable']} primaries survive unchanged.")
        ax = fig.add_axes([0.05, 0.17, 0.66, 0.74])
        style_2d(ax, 9)
        alive = []
        for i, q in enumerate(ev.p):
            if q.status == 0:
                continue
            is_prim = i in self.prim
            born = is_prim or (q.m1 in done)
            if not born:
                continue
            decayed = i in done
            if decayed:
                continue
            alive.append(i)
        for i in alive:
            q = ev.p[i]
            e_, f_ = core.eta(q.p), core.phi(q.p)
            if abs(e_) > 6:
                continue
            col = COL_HAD if i in self.prim else (COL_NU if core.is_neutrino(q.id) else COL_DEC)
            ax.scatter(e_, f_, s=msize(core.pt(q.p), 1.2) ** 2 / 4, color=col, alpha=0.85, lw=0)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-math.pi, math.pi)
        ax.set_xlabel(r"$\eta$", fontsize=11)
        ax.set_ylabel(r"$\phi$", fontsize=11)
        ax.set_title(f"laboratory time {t_now / C_MM_PER_PS:.1f} ps: {n_done} decays done, {len(alive)} particles", fontsize=11, color=TEXT)
        ax2 = fig.add_axes([0.75, 0.17, 0.23, 0.74])
        ax2.axis("off")
        n_prim_alive = sum(1 for i in alive if i in self.prim)
        n_ch = sum(1 for i in alive if core.is_charged(ev.p[i].id))
        n_gam = sum(1 for i in alive if ev.p[i].id == 22)
        n_nu = sum(1 for i in alive if core.is_neutrino(ev.p[i].id))
        rows = [("decays done", n_done), ("particles", len(alive)), ("primary hadrons left", n_prim_alive), ("charged", n_ch), ("photons", n_gam), ("neutrinos", n_nu)]
        for j, (lab, v) in enumerate(rows):
            ax2.text(0.0, 0.95 - 0.09 * j, f"{lab}: {v}", fontsize=12, color=TEXT, transform=ax2.transAxes)
        ax2.scatter([0.05], [0.33], s=60, color=COL_HAD, transform=ax2.transAxes)
        ax2.text(0.12, 0.33, "primary hadron", fontsize=10, transform=ax2.transAxes, va="center")
        ax2.scatter([0.05], [0.27], s=60, color=COL_DEC, transform=ax2.transAxes)
        ax2.text(0.12, 0.27, "decay product", fontsize=10, transform=ax2.transAxes, va="center")
        ax2.scatter([0.05], [0.21], s=60, color=COL_NU, transform=ax2.transAxes)
        ax2.text(0.12, 0.21, "neutrino", fontsize=10, transform=ax2.transAxes, va="center")


# --- scene 7: the ensemble ------------------------------------------------------------------------------
class EnsembleScene(Scene):
    number, title = 7, "the same hadrons decayed again and again"

    def __init__(self, phys, fps, events=400):
        super().__init__(phys, fps)
        self.nframes = 9 * fps
        st = phys.stats
        if st:
            d = st["b"]["b"]
            self.L = np.array(d["L"])[:events]
            self.ratio = (np.array(d["cones"]["0.4"]["visible"]["pt"]) / np.array(d["cones"]["0.4"]["primaries"]["pt"]))[:events]
        else:
            self.L, self.ratio = np.array([phys.b["L"]]), np.array([phys.b["cones"][0.4]["visible"]["pt"] / phys.b["cones"][0.4]["primaries"]["pt"]])

    def draw(self, fig, k):
        f = ramp(k, 0, self.nframes - 2 * self.fps)
        n = max(1, int(len(self.L) * f))
        r = self.phys.b
        chrome(fig, self.number, self.title,
               f"The {self.phys.summary['n_primaries']} primary hadrons of the record decayed {len(self.L)} times with different seeds. The flight distance of the $b$ quark's B "
               f"follows the exponential of scene 1 stretched by $\\gamma\\beta$ = {r['gamma_beta']:.0f}; the visible $p_T$ of the cone R = 0.4 stays at the primary-hadron value "
               f"in four decays out of five and drops by 10-50% when a neutrino leaves.")
        ax = fig.add_axes([0.06, 0.17, 0.42, 0.71])
        style_2d(ax, 9)
        ax.hist(self.L[:n], bins=np.linspace(0, 80, 33), color=COL_B, alpha=0.5)
        lam = r["gamma_beta"] * core.tau0(511)
        x = np.linspace(0.01, 80, 200)
        ax.plot(x, n * 2.5 * np.exp(-x / lam) / lam, color=COL_STR, lw=1.5, ls="--", label=rf"$\gamma\beta\,c\tau_0$ = {lam:.1f} mm")
        ax.axvline(r["L"], color=COL_EVENT, ls="--", lw=1.2)
        ax.text(r["L"] + 1, 0.9, f"seed 1: {r['L']:.1f} mm", color=COL_EVENT, fontsize=9.5, transform=ax.get_xaxis_transform())
        ax.set_xlabel("flight distance L of the B hadron  [mm]", fontsize=10)
        ax.set_ylabel("decays", fontsize=10)
        ax.set_title(f"{n} decays: <L> = {self.L[:n].mean():.1f} mm", fontsize=11, color=TEXT)
        ax.legend(loc="upper right", fontsize=9.5)
        ax2 = fig.add_axes([0.56, 0.17, 0.41, 0.71])
        style_2d(ax2, 9)
        ax2.hist(self.ratio[:n], bins=np.linspace(0.3, 1.1, 33), color=COL_STR, alpha=0.5)
        ax2.set_yscale("log")
        ax2.set_xlabel(r"visible $p_T$ / primary-hadron $p_T$ in the cone R = 0.4", fontsize=10)
        ax2.set_ylabel("decays", fontsize=10)
        ax2.set_title(f"<ratio> = {self.ratio[:n].mean():.3f}, below 0.95 in {100 * np.mean(self.ratio[:n] < 0.95):.0f}%", fontsize=11, color=TEXT)


# --- driver -----------------------------------------------------------------------------------------------
def build_scenes(names: List[str], phys: Physics, args) -> List[Scene]:
    out = []
    for n in names:
        if n == "clock":
            out.append(ClockScene(phys, args.fps))
        elif n == "walk":
            out.append(WalkScene(phys, args.fps))
        elif n == "table":
            out.append(TableScene(phys, args.fps))
        elif n == "cascade":
            out.append(CascadeScene(phys, args.fps))
        elif n == "emigrant":
            out.append(EmigrantScene(phys, args.fps))
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
    p.add_argument("-o", "--output", type=Path, default=HERE / "animate_standalone_hadron_decays.mp4")
    p.add_argument("--scene", default="all", help="all, or a comma list of clock,walk,table,cascade,emigrant,event,ensemble (or 1..7)")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--dpi", type=int, default=100, help="100 gives 1280x720")
    p.add_argument("--step", type=int, default=1, help="render every step-th frame")
    p.add_argument("--fast", action="store_true", help="preview: --step 4 --dpi 60")
    p.add_argument("--frames-dir", type=Path, default=None)
    p.add_argument("--stills", type=Path, default=None, help="one key frame per scene (.pdf or .png)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("-i", "--input", type=Path, default=HERE.parent.parent / "Had" / "event_Had.lhe")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--events", type=int, default=400, help="decays shown in scene 7 (from the stats file when present)")
    args = p.parse_args()
    if args.fast:
        args.step, args.dpi = max(args.step, 4), min(args.dpi, 60)
    names = list(SCENE_NAMES.values()) if args.scene == "all" else [SCENE_NAMES.get(s.strip(), s.strip()) for s in args.scene.split(",")]
    bad = [n for n in names if n not in SCENE_NAMES.values()]
    if bad:
        sys.exit(f"unknown scene(s) {bad}; choose from {list(SCENE_NAMES.values())}")
    phys = Physics(args.seed, args.input)
    s = phys.summary
    print(f"event: seed {args.seed}: {s['n_primaries']} primary hadrons -> {s['n_final']} final particles; the b quark's {phys.b['hadron']} flew {phys.b['L']:.1f} mm"
          + (f"; semileptonic example: seed {phys.semi[3]}" if phys.semi else ""))
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
