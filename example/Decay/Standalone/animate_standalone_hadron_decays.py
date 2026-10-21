#!/usr/bin/env python3
"""Animated companion to standalone_hadron_decays.py.

Seven scenes, narrated, show how the primary hadrons of chapter 5 turn into what the detector
will see. Every number on screen and in the narration is computed with the classes of that
script: the seed-1 decays of ../../Had/event_Had.lhe (the record ../event_Decay.lhe) are
replayed. The film picks up where the chapter-5 film ended ("the children are married") and
ends at the door of the inn, where the anti-kT algorithm of chapter 15 gathers the grandchildren.

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

Narration and music
  Each scene is a few phases (the hundred mesons born, their decays, the clock without memory,
  our draw; the walk, the decay, the tracks, the view along the beam; ...) and each phase has
  one line of narration (narration() below), spoken by a neural voice through edge-tts
  (Microsoft's online voices; `pip install --user edge-tts`, needs the network) or, with
  --engine say, by a macOS voice; --narration-dir takes recorded clips named <key>.wav|mp3
  (--print-script lists the keys and the text). A phase lasts at least as long as its line:
  its motion is slowed down (up to 4x) and then held, so the narrated film is about 7 min and
  the silent one (--no-narration) about 70 s. The score comes from example/soundtrack.py: a
  drone, a pad with one chord per scene, a pulse that gains a voice with every scene, a bell as
  each scene opens, a thud when the B hadron of the walk decays. The clips are cached in
  build/narration/Decay/. An .srt is written next to the .mp4.

Examples (matplotlib is needed; on this Mac use /usr/bin/python3):
  python3 animate_standalone_hadron_decays.py                  # -> .mp4 + .srt next to this file (1920x1080)
  python3 animate_standalone_hadron_decays.py --fast           # quick low-resolution preview
  python3 animate_standalone_hadron_decays.py --scene walk,cascade -o walk.gif   # silent
  python3 animate_standalone_hadron_decays.py --stills stills.pdf --no-video
  python3 animate_standalone_hadron_decays.py --print-script   # the narration, for recording it yourself
  python3 animate_standalone_hadron_decays.py --audio-only x.wav   # the soundtrack alone

An .mp4 needs ffmpeg on the PATH; a .gif needs only Pillow and is silent. --frames-dir
writes the individual frames as PNG files instead. --seed is the one of the standalone
script (default: seed 1, the committed record). --jobs renders the frames in parallel.
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
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "Had"))
import standalone_hadron_decays as core  # noqa: E402
import plot_journey_Had as jh  # noqa: E402

sys.path.insert(0, str(HERE.parent.parent))
import soundtrack as snd  # noqa: E402  (example/soundtrack.py: the voice, the score, the film clock)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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


def ramp(k: float, k0: int, k1: int) -> float:
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

    def draw(self, fig, k: float):
        raise NotImplementedError

    def keyframe(self) -> int:
        return self.nframes - 1

    def phases(self) -> List[Tuple[str, int, int]]:
        """(key, first frame, frame after the last) of each phase; one line of narration per phase."""
        return [("all", 0, self.nframes)]


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
        self.k_run = self.nframes - 2 * self.fps

    def phases(self):
        return [("born", 0, self.fps), ("decay", self.fps, 5 * self.fps), ("memory", 5 * self.fps, self.k_run), ("ours", self.k_run, self.nframes)]

    def draw(self, fig, k):
        t = self.t_max * ramp(k, 0, self.k_run)
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
        self.k_run = self.nframes - 2 * self.fps
        # the frame at which the B (first flight) and the last flight decay, by inverting the smooth ramp
        self.k_bdecay = self.frame_at_time(self.flights[0]["t1"])
        self.k_alldecay = self.frame_at_time(max(f["t1"] for f in self.flights))

    def frame_at_time(self, t: float) -> int:
        for kk in range(self.k_run):
            if self.t_end * ramp(kk, 0, self.k_run) >= t:
                return kk
        return self.k_run

    def phases(self):
        return [("leave", 0, self.k_bdecay), ("decay", self.k_bdecay, self.k_alldecay + self.fps), ("tracks", self.k_alldecay + self.fps, self.k_run),
                ("beam", self.k_run, self.nframes)]

    def draw(self, fig, k):
        r = self.r
        t = self.t_end * ramp(k, 0, self.k_run)
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

    def phases(self):
        return [("bar", 0, self.fps), ("roll", self.fps, 5 * self.fps), ("channel", 5 * self.fps, 8 * self.fps), ("hadrons", 8 * self.fps, self.nframes)]

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
        self.k_run = self.nframes - 2 * self.fps
        self.k_gen1 = next(kk for kk in range(self.k_run + 1) if self.n_gen * ramp(kk, 0, self.k_run) >= 1.0)   # the B* has decayed

    def phases(self):
        return [("star", 0, self.k_gen1), ("generations", self.k_gen1, self.k_run), ("stable", self.k_run, self.nframes)]

    def draw(self, fig, k):
        ev, r = self.phys.ev, self.r
        g = self.n_gen * ramp(k, 0, self.k_run)
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

    def phases(self):
        return [("spectrum", 0, 5 * self.fps), ("neutrino", 5 * self.fps, 8 * self.fps), ("visible", 8 * self.fps, self.nframes)]

    def draw(self, fig, k):
        f1 = ramp(k, 0, 5 * self.fps)
        f2 = ramp(k, 5 * self.fps, 8 * self.fps)
        cap = (f"In a semileptonic decay $B\\to D^{{(*)}} \\ell\\nu$ the V-A matrix element $|M|^2 \\propto (p_B\\cdot p_\\nu)(p_D\\cdot p_\\ell)$ pushes the lepton towards "
               f"the end point of its spectrum. The neutrino then leaves the jet for good: the detector will measure less $p_T$ than the quark had.")
        if self.semi:
            ev2, dec2, sm2, s2 = self.semi
            r2 = sm2["b"]["b"]
            c = r2["cones"][0.4]
            cap += (f"\nDecaying the same hadrons with seed {s2}: {tex(r2['hadron'])} $\\to$ {' '.join(tex(x) for x in r2['channel'].split())}, the neutrino takes "
                    f"{r2['e_nu']:.1f} GeV ({100 * r2['e_nu_frac']:.0f}% of the B); the cone R = 0.4 drops from {c['primaries']['pt']:.1f} to {c['visible']['pt']:.1f} GeV visible.")
        chrome(fig, self.number, self.title, cap)
        ax = fig.add_axes([0.06, 0.22, 0.42, 0.66])
        style_2d(ax, 9)
        n = max(1, int(len(self.e_va) * f1))
        bins = np.linspace(0, 2.6, 27)
        ax.hist(self.e_ps[:n], bins=bins, density=True, color=COL_PY, alpha=0.5, label="phase space")
        ax.hist(self.e_va[:n], bins=bins, density=True, histtype="step", color=COL_LEP, lw=2.2, label="V-A matrix element")
        ax.set_xlabel(r"$E_\mu$ in the B rest frame  [GeV]  ($B^0\to D^{*-}\mu^+\nu_\mu$)", fontsize=10)
        ax.set_ylabel("probability density", fontsize=10)
        ax.set_title(f"{n} decays drawn: <E> = {self.e_va[:n].mean():.2f} (V-A) against {self.e_ps[:n].mean():.2f} GeV (phase space)", fontsize=10, color=TEXT)
        ax.legend(loc="upper left", fontsize=9.5)
        ax2 = fig.add_axes([0.56, 0.22, 0.41, 0.66])
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
        self.k_run = self.nframes - 2 * self.fps
        n_prompt = sum(1 for t in self.t_dec.values() if t < 0.01)          # within 10 um of the collision point
        self.frac_prompt = n_prompt / max(1, len(self.order))
        self.k_prompt = next(kk for kk in range(self.k_run + 1) if int(len(self.order) * ramp(kk, 0, self.k_run)) >= n_prompt)

    def phases(self):
        return [("start", 0, self.k_prompt), ("heavy", self.k_prompt, self.k_run), ("final", self.k_run, self.nframes)]

    def draw(self, fig, k):
        ev, dec = self.phys.ev, self.phys.dec
        f = ramp(k, 0, self.k_run)
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
        self.k_run = self.nframes - 2 * self.fps

    def phases(self):
        return [("fill", 0, self.k_run), ("result", self.k_run, self.nframes)]

    def draw(self, fig, k):
        f = ramp(k, 0, self.k_run)
        n = max(1, int(len(self.L) * f))
        r = self.phys.b
        chrome(fig, self.number, self.title,
               f"The {self.phys.summary['n_primaries']} primary hadrons of the record decayed {len(self.L)} times with different seeds. The flight distance of the $b$ quark's B "
               f"follows the exponential of scene 1 stretched by $\\gamma\\beta$ = {r['gamma_beta']:.0f}; the visible $p_T$ of the cone R = 0.4 stays within 5% of the "
               f"primary-hadron value in {100 * np.mean(self.ratio >= 0.95):.0f}% of the decays and drops by 5-50% when a neutrino leaves the cone.")
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


# --- narration --------------------------------------------------------------------------------------------
HADRON_WORD = {"pi": "pi", "K": "K", "B": "B", "D": "D", "rho": "rho", "eta": "eta", "omega": "omega", "phi": "phi", "J/psi": "J psi",
               "Sigma": "Sigma", "Lambda": "Lambda", "Xi": "Xi", "Omega": "Omega", "Delta": "Delta", "n": "neutron", "p": "proton",
               "nu": "neutrino", "mu": "mu", "e": "e", "tau": "tau", "gamma": "photon"}


def say_hadron(name: str) -> str:
    """A particle name of the standalone script (B*bar0, D*_s-, nu_mubar, K_S0, ...) in words."""
    anti = "bar" in name
    n = name.replace("bar", "")
    star = "*" in n
    n = n.replace("*", "")
    charge = ""
    for c, w in (("+", "plus"), ("-", "minus"), ("0", "zero"), ("'", "prime")):
        if n.endswith(c):
            charge, n = w, n[:-1]
    sub = ""
    if "_" in n:
        n, sub = n.split("_", 1)
    base = HADRON_WORD.get(n, n)
    if base in ("neutron", "proton"):
        charge = ""
    if base == "neutrino":                 # nu_e, nu_mubar, ...
        flavour = {"e": "electron", "mu": "muon", "tau": "tau"}.get(sub, sub)
        return ("an anti-" if anti else "a ") + f"{flavour} neutrino"
    if base in ("e", "mu", "tau") and not sub:
        return {"e": "an electron" if charge == "minus" else "a positron", "mu": f"a mu {charge}", "tau": f"a tau {charge}"}[base]
    words = ("anti-" if anti else "") + base + (f"-{sub}" if sub else "") + (" star" if star else "") + (f" {charge}" if charge else "")
    return words


def cap(words: str) -> str:
    return words[0].upper() + words[1:]


def with_article(words: str) -> str:
    return words if words.startswith(("a ", "an ")) else ("an " if words[0].lower() in "aeiou" else "a ") + words


def say_list(names: List[str], article: bool = True) -> str:
    """'a D-s star minus, a D star zero and a rho plus'"""
    said = []
    for nm in names:
        w = say_hadron(nm)
        if article:
            w = with_article(w)
        said.append(w)
    return said[0] if len(said) == 1 else ", ".join(said[:-1]) + " and " + said[-1]


def narration(phys: Physics, scenes: List[Scene]) -> Dict[str, str]:
    """One line per phase, keyed <scene>_<phase>, plus the closing line; the numbers are the event's."""
    r, s = phys.b, phys.summary
    tau0_ps = core.tau0(511) / C_MM_PER_PS
    mm_per_ps = r["gamma_beta"] * C_MM_PER_PS
    bhad = say_hadron(r["hadron"])
    survive = 100 * math.exp(-r["tau_over_tau0"])
    lines = {
        "clock_born": (f"The children are married: {s['n_primaries']} primary hadrons walked out of the strings of chapter five, and most of "
                       "them will not live to reach the detector. The clock. A hundred B zero mesons, born at the same instant. Each carries a "
                       "clock, and the clock is a die: at every picosecond the same chance to decay, whatever happened before."),
        "clock_decay": (f"They go one by one, at random moments drawn from the exponential, with the mean life tau zero of {tau0_ps:.2f} "
                        "picoseconds. The survivors trace the curve. After one lifetime, thirty-seven percent are left."),
        "clock_memory": ("The clock has no memory. The survivors have not aged: the chance to decay in the next picosecond is the same at five "
                         "picoseconds as at zero. That is why a decay can follow no other law than the exponential."),
        "clock_ours": (f"Our event's {bhad} drew {r['tau_ps']:.2f} picoseconds, {r['tau_over_tau0']:.1f} lifetimes... a long life, but not a rare "
                       f"one: {survive:.0f} percent of B mesons live longer."),
        "walk_leave": (f"The walk. In the laboratory the clock runs slow: at gamma beta of {r['gamma_beta']:.0f}, one picosecond of proper time "
                       f"is {mm_per_ps:.0f} millimetres, almost a centimetre. The {bhad} is neutral and leaves no track; it simply walks away from "
                       "the primary vertex."),
        "walk_decay": (f"After {r['L']:.0f} millimetres it decays, {r['Lxy']:.1f} of them across the beam, into {say_list(r['channel'].split())}. "
                       f"The charm hadron, a {say_hadron(r['charm'])}, walks on another {r['L_charm']:.0f} millimetres before it decays too."),
        "walk_tracks": (f"The tracks of the {r['n_tracks']} charged grandchildren start from the displaced vertices, not from the collision point. "
                        f"Extrapolated back to the beam, all {r['n_tracks_d0_100um']} miss it by more than a hundred microns, the largest by "
                        f"{r['max_d0']:.1f} millimetres: the birthmark of a B hadron."),
        "walk_beam": ("Along the beam the walk is longer than across it, because the B flies forward. And the miss distance does not grow with "
                      "the energy: a faster B walks further, but its children fly in a narrower cone, and the product stays at a few hundred "
                      "microns. That is what a b tag looks for."),
        "outro": ("The grandchildren are born. The family has scattered into tracks, photons and a few neutrinos that left for good; what "
                  "the detector receives is these, not the quark. Next stop: the inn, where everyone gathers... the anti-k-T algorithm "
                  "will seat the grandchildren at their tables, and call each table a jet."),
    }
    for sc in scenes:
        if isinstance(sc, TableScene):
            d = sc.d
            n_ch = len(sc.chans)
            prods = " ".join(core.name(p) for p in d["prods"])
            lines["table_bar"] = (f"The decay table. {cap(with_article(bhad))} has {n_ch} open channels, drawn here as one bar: each slice is a branching ratio, and "
                                  "the slices add up to one. Blue, the channels written at quark level; purple, the ones listed as hadrons; "
                                  "orange, green and grey, the semileptonic ones, with an electron, a muon or a tau.")
            lines["table_roll"] = (f"One random number decides. It slides down the bar and stops at the first slice whose cumulative branching "
                                   f"ratio exceeds it: here, channel {sc.k_chan + 1} of {n_ch}, a slice of {100 * d['br'] / sc.total:.1f} percent.")
            lines["table_channel"] = (f"The channel is written at quark level, {say_quarks(d['prods'])}: the b quark became a charm quark and a W, "
                                      "the W became a strange quark and an anti-charm, and the anti-down of the meson watched, the spectator.")
            lines["table_hadrons"] = (f"The four quarks are paired into two colour singlets and the flavour machinery of chapter five turns them into "
                                      f"hadrons: {say_list([core.name(i) for i in d['ids']])}. {len(d['ids'])} direct products; {r['n_final']} final "
                                      "particles in the end.")
        elif isinstance(sc, CascadeScene):
            neutrinos = "no neutrino this time" if r["n_neutrinos"] == 0 else f"{r['n_neutrinos']} neutrinos"
            lines["cascade_star"] = (f"The cascade, in the eta-phi plane around the b quark. First the {say_hadron(r['string_hadron'])} of the string "
                                     f"radiates a photon and becomes the {bhad}: an electromagnetic decay, within attoseconds, at the primary vertex.")
            lines["cascade_generations"] = (f"Then the generations: the {bhad}'s {r['n_direct']} products, their products, and theirs, each a step closer to "
                                            "stability. The mothers fade as they decay; bottom hadrons in red, charm in violet, photons in yellow, "
                                            "and grey for the particles the detector will see.")
            lines["cascade_stable"] = (f"After {sc.n_gen - 1} generations nothing unstable is left: {r['n_final']} particles, {r['n_charged_final']} charged and "
                                       f"{r['n_photons_final']} photons, {neutrinos}, all within a few "
                                       "tenths of a unit of the original b quark. The cone has filled with grandchildren.")
        elif isinstance(sc, EmigrantScene):
            lines["emigrant_spectrum"] = ("One decay in five is different in kind: the W inside the hadron makes a lepton and a neutrino instead of "
                                          "quarks. The V minus A matrix element pushes the lepton towards the end point of its spectrum: four thousand "
                                          f"B zero to D star mu nu decays at rest, mean {sc.e_va.mean():.2f} GeV, against {sc.e_ps.mean():.2f} for flat phase space.")
            if sc.semi:
                ev2, dec2, sm2, s2 = sc.semi
                r2 = sm2["b"]["b"]
                c = r2["cones"][0.4]
                lines["emigrant_neutrino"] = (f"Our seed-one B went hadronic, so the same hadrons are decayed again with seed {s2}: now the {say_hadron(r2['hadron'])} "
                                              f"decays to {say_list(r2['channel'].split())}. The neutrino takes {r2['e_nu']:.0f} GeV, {100 * r2['e_nu_frac']:.0f} percent "
                                              "of the B's energy, and leaves the family for good.")
                lines["emigrant_visible"] = (f"The cone of radius point four around the b quark drops from {c['primaries']['pt']:.1f} GeV of primary hadrons to "
                                             f"{c['visible']['pt']:.1f} GeV visible, {c['charged']['pt']:.0f} of them charged. The detector will measure less than the "
                                             "quark had; the emigrant, and the soft muon it left behind, are the two handles the corrections will reach for later.")
            else:
                lines["emigrant_neutrino"] = "No semileptonic decay of the B was found among the first forty seeds."
                lines["emigrant_visible"] = "The neutrino would take its share of the cone away."
        elif isinstance(sc, EventScene):
            lines["event_start"] = (f"The whole event, in laboratory time. {s['n_primaries']} primary hadrons; the strong and electromagnetic decays go first, at the "
                                    f"primary vertex itself: rho, K star, pi zero, eta... {100 * sc.frac_prompt:.0f} percent of all decays happen within ten "
                                    "microns of the collision point.")
            lines["event_heavy"] = ("Then the weak decays, which have walked: the charm hadrons after millimetres, the B hadrons after centimetres, the "
                                    "K short and the Lambda after centimetres to metres. "
                                    + (f"{'One B meson' if s['n_oscillated'] == 1 else str(s['n_oscillated']) + ' B mesons'} of the event changed name on the way, "
                                       "oscillating into the antiparticle before decaying." if s["n_oscillated"] else "No B meson oscillated this time."))
            lines["event_final"] = (f"{s['n_final']} final particles: {s['n_charged']} charged, {s['n_photons']} photons, {s['n_neutrinos']} neutrinos. "
                                    f"{s['n_primary_stable']} of the primary hadrons survive unchanged, pions, kaons, protons. This is what the detector receives.")
        elif isinstance(sc, EnsembleScene):
            lines["ensemble_fill"] = (f"The same {s['n_primaries']} primary hadrons decayed again and again, {len(sc.L)} times, only the decays redrawn. The flight "
                                      f"distance of the b quark's hadron fills the exponential of scene one, stretched by gamma beta: mean {sc.L.mean():.1f} millimetres.")
            lines["ensemble_result"] = (f"And the visible share of the cone. In {100 * np.mean(sc.ratio >= 0.95):.0f} percent of the decays it stays within five percent of "
                                        f"what the primary hadrons carried; in the other {100 * np.mean(sc.ratio < 0.95):.0f} it drops by up to half, when a neutrino has left. "
                                        f"Seed one, our event, sits at {sc.ratio[0]:.2f}.")
    keys = [f"{SCENE_NAMES[str(sc.number)]}_{key}" for sc in scenes for key, _, _ in sc.phases()] + ["outro"]
    return {k: lines[k] for k in keys if k in lines and lines[k]}


def say_quarks(ids: List[int]) -> str:
    """'anti-charm, strange, charm, anti-down'"""
    word = {1: "down", 2: "up", 3: "strange", 4: "charm", 5: "bottom"}
    return ", ".join(("anti-" if i < 0 else "") + word.get(abs(i), str(i)) for i in ids)


def score_spec(film: snd.Film) -> snd.ScoreSpec:
    """The progression of the cover (D minor, B flat, F, C, G minor, B flat, A) over the seven scenes, resolving
    under the outro; a thud when the B hadron of the walk decays."""
    t_dec = snd.voice_at_phase(film, "walk_decay")
    return snd.film_score(film, [snd.Dm, snd.Bb, snd.F, snd.C, snd.Gm, snd.Bb, snd.A],
                          thuds=[t_dec + 0.2] if t_dec is not None else [])


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


def write_video(film: snd.Film, out: Path, fps: float, dpi: int, audio: Optional[Path], jobs: int):
    snd.render_video(film, out, fps, dpi, audio, lambda fig, sc, k, t: sc.draw(fig, k), FIGSIZE, "b bbar event, standalone hadron decays", jobs=jobs)
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
    p.add_argument("-o", "--output", type=Path, default=HERE / "animate_standalone_hadron_decays.mp4",
                   help=".mp4 (needs ffmpeg) or .gif (silent)")
    p.add_argument("--scene", default="all", help="all, or a comma list of clock,walk,table,cascade,emigrant,event,ensemble (or 1..7)")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--dpi", type=int, default=150, help="150 gives 1920x1080, 100 gives 1280x720")
    p.add_argument("--step", type=int, default=1, help="render every step-th frame")
    p.add_argument("--fast", action="store_true", help="preview: --step 4 --dpi 60, fewer decays in scene 7")
    p.add_argument("--frames-dir", type=Path, default=None)
    p.add_argument("--stills", type=Path, default=None, help="one key frame per scene (.pdf or .png)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--jobs", type=int, default=4, help="render the frames in this many processes (joined losslessly)")
    p.add_argument("--audio-only", type=Path, default=None, metavar="WAV",
                   help="write just the soundtrack (narration + score) and exit, without rendering")
    p.add_argument("--cache-dir", type=Path, default=HERE.parent.parent.parent / "build" / "narration" / "Decay",
                   help="the synthesised narration clips are kept here (make clean removes them)")
    snd.add_narration_args(p)             # --engine --voice --rate --narration-dir --no-narration --print-script --music-db --no-music
    p.add_argument("-i", "--input", type=Path, default=HERE.parent.parent / "Had" / "event_Had.lhe")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--events", type=int, default=400, help="decays shown in scene 7 (from the stats file when present)")
    args = p.parse_args()
    if args.fast:
        args.step, args.dpi, args.events = max(args.step, 4), min(args.dpi, 60), min(args.events, 40)
    snd.narration_defaults(args)
    fps = args.fps / args.step
    args.fps = snd.NOMINAL_FPS            # the scenes are drawn at the nominal rate; the film clock renders at fps
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
        tmp = tempfile.TemporaryDirectory(prefix="jitjet_decay_")
        if ffmpeg and args.output.suffix.lower() != ".gif" and (clips or score):
            audio = Path(tmp.name) / "soundtrack.wav"
            snd.mix_audio(film.duration, film.voice_at, clips, score, args.music_db, audio)
            if clips:
                snd.write_srt(film.voice_at, clips, lines, args.output.with_suffix(".srt"))
        write_video(film, args.output, fps, args.dpi, audio, args.jobs)


if __name__ == "__main__":
    main()
