#!/usr/bin/env python3
"""Animated companion to standalone_parton_showering.py.

Four scenes, about 40 s in total, show what the standalone shower does to the
b bbar event of chapter 1. Every number on screen is computed with the classes
of that script (AlphaStrong, TablePDF, Shower, family): scene 1 replays the
seed-1 shower that wrote ../event_PS.lhe branching by branching, the other
scenes call the same trial-emission and shower functions again.

  1  shower    The evolution in pT. A scale runs down from pTmax = 113.3 GeV
               (the factorisation scale of the hard process) to the cut-off,
               and every time it passes the pT of a branching the branching
               happens: the ladder on the left fills rung by rung, the
               (eta, phi) picture on the right fills with partons. Marker area
               is pT; colour is the family (FSR of b, FSR of bbar, ISR); thin
               lines are the colour dipoles that do the radiating. At the end
               the family of each b quark (partons within dR < 0.4) is circled.
  2  sudakov   The Sudakov veto algorithm, seen as an ensemble: the four ends
               that can radiate first (the b, the bbar, the two incoming
               partons) each propose their next emission from pTmax many
               times. The survival curves are the no-emission probabilities
               Delta(pTmax, pT) of each end; the hardest proposal wins the
               first branching, and the histogram shows how often each end
               wins and at which pT. The seed-1 event's first branching is
               marked.
  3  recoil    Momentum bookkeeping while the same shower runs again: in the
               transverse plane the b bbar system acquires a pT that the ISR
               partons balance, so the sum over all partons stays at zero.
               On the right the two incoming partons walk up in x on top of
               the PDFs at the current scale: backward evolution.
  4  ensemble  The same hard event showered again and again (seeds 1, 2, 3,
               ...): twelve outcomes side by side, and the running means of
               the branching counts and of the family pT of each b quark
               settling towards the values that PYTHIA 8 gives for the same
               event (validate_shower_with_pythia.py, README.md).

Examples (matplotlib is needed; on this Mac use /usr/bin/python3):
  python3 animate_standalone_parton_showering.py                  # -> .mp4 next to this file
  python3 animate_standalone_parton_showering.py --fast           # quick low-resolution preview
  python3 animate_standalone_parton_showering.py --scene shower -o shower.gif
  python3 animate_standalone_parton_showering.py --stills stills.pdf --no-video
  python3 animate_standalone_parton_showering.py --show           # interactive window

An .mp4 needs ffmpeg on the PATH; a .gif needs only Pillow. --frames-dir writes
the individual frames as PNG files instead. The physics options (--seed,
--ptmin-fsr, --ptmin-isr, --pt0, --ptmax, --no-isr, --no-dampen, --radius) are
those of the standalone script.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import standalone_parton_showering as core  # noqa: E402  (same directory)
import plot_journey_PS as journey  # noqa: E402  (../plot_journey_PS.py: colours, regions, marker area)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch  # noqa: E402

# --- appearance --------------------------------------------------------------
FAMILY = journey.FAMILY                      # b: red, bbar: blue, ISR: orange (as in plot_journey_PS.py)
STAR = journey.STAR                          # marker of the final-state copy of the hard b / bbar
COL_B, COL_BBAR, COL_ISR = FAMILY["b"]["color"], FAMILY["bbar"]["color"], FAMILY["ISR"]["color"]
COL_A, COL_BEAM = "#8e5bb5", "#2f8f5b"       # incoming parton from +z (side A) / from -z (side B)
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d6d3ca"
ETA_MAX = journey.ETA_MAX
FIGSIZE = (12.8, 7.2)                        # 16:9; dpi 100 -> 1280x720
# PYTHIA 8.315 averages for this hard event, 2000 showers (README.md, validate_shower_with_pythia.py)
PYTHIA_REF = dict(n_isr=5.05, n_fsr=19.5, n_final=26.5, fam_pt={"b": 108.4, "bbar": 104.8})
STANDALONE_REF = dict(n_isr=4.88, n_fsr=20.9, n_final=27.7, fam_pt={"b": 108.4, "bbar": 103.4})


def smooth(t: float) -> float:
    """Ease-in/out in [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def ramp(k: int, k0: int, k1: int) -> float:
    """Fraction of the way from frame k0 to frame k1 (clamped)."""
    if k1 <= k0:
        return 1.0
    return min(max((k - k0) / float(k1 - k0), 0.0), 1.0)


def tex_name(pid: int) -> str:
    return journey.parton_tex(pid)


# --- physics setup -------------------------------------------------------------
@dataclass
class Setup:
    input: Path
    seed: int = 1
    alphas: float = 0.118
    order: int = 2
    ptmin_fsr: float = 0.5
    ptmin_isr: float = 0.2
    pt0: float = 2.0
    ptmax: Optional[float] = None
    radius: float = 0.4
    no_isr: bool = False
    no_fsr: bool = False
    no_dampen: bool = False

    def as_args(self) -> SimpleNamespace:
        """The argparse namespace that core.shower_event expects."""
        return SimpleNamespace(input=self.input, ptmax=self.ptmax, ptmin_fsr=self.ptmin_fsr,
                               ptmin_isr=self.ptmin_isr, pt0=self.pt0, no_isr=self.no_isr, no_fsr=self.no_fsr,
                               no_dampen=self.no_dampen, check=False)


@dataclass
class Snap:
    """The event record after `step` branchings."""
    step: int
    pT: float                                  # scale of this branching (pTmax for step 0)
    rows: List[core.Particle]
    inA: int
    inB: int
    track: Dict[int, int]                      # hard parton index -> its current row
    record: Optional[dict] = None              # the history entry of this branching
    _fam: Dict[int, str] = field(default_factory=dict)

    @staticmethod
    def take(ev: core.Event, sh: core.Shower, step: int, pT: float, record: Optional[dict]) -> "Snap":
        rows = [core.Particle(q.id, q.status, q.m1, q.m2, q.d1, q.d2, q.col, q.acol, q.p.copy(), q.m) for q in ev.p]
        return Snap(step, pT, rows, ev.inA, ev.inB, dict(sh.track), record)

    @property
    def final(self) -> List[int]:
        return [i for i, q in enumerate(self.rows) if q.final]

    @property
    def stars(self) -> Dict[int, int]:
        """pid of the hard b / bbar -> its current row."""
        return {self.rows[orig].id: cur for orig, cur in self.track.items()}

    def family(self, i: int) -> str:
        if i not in self._fam:
            self._fam[i] = family_of(self.rows, i)
        return self._fam[i]

    def new_rows(self) -> List[int]:
        return list(self.record.get("new_rows", [])) if self.record else []

    def dipoles(self):
        """(i, j) pairs joined by a colour line; j = -1/-2 for the incoming parton on side A/B."""
        ends: Dict[int, list] = {}
        for i in self.final + [self.inA, self.inB]:
            q = self.rows[i]
            for tag in (q.col, q.acol):
                if tag:
                    ends.setdefault(tag, []).append(i)
        pairs = []
        for tag, ij in ends.items():
            if len(ij) == 2:
                pairs.append(tuple(ij))
        return pairs


def family_of(rows, i: int) -> str:
    """Walk mother1 back to the hard b (-23, id 5), the hard bbar or an ISR emission (43)."""
    seen = set()
    while 0 <= i < len(rows) and i not in seen:
        seen.add(i)
        q = rows[i]
        if abs(q.status) == 23:
            return "b" if q.id == 5 else "bbar" if q.id == -5 else "ISR"
        if abs(q.status) in (41, 43, 53, 21, 12, 11):
            return "ISR"
        i = q.m1
    return "ISR"


class Physics:
    """The shower of the standalone script, replayed with a snapshot after every branching."""

    def __init__(self, st: Setup):
        self.st = st
        self.alpha = core.AlphaStrong(st.alphas, st.order)
        self.pdf = None if st.no_isr else core.TablePDF(core.TABLES)
        self.snaps: List[Snap] = []
        self.ev, self.sh, self.hard, self.pTmax = self.replay(st.seed, self.snaps)
        self.history = self.sh.history
        self.pTmin_fsr = math.sqrt(self.sh.pT2min_fsr)
        self.pTmin_isr = math.sqrt(self.sh.pT2min_isr)

    def new_shower(self, seed: int):
        st = self.st
        ev = core.read_event(st.input)
        hard = [i for i, q in enumerate(ev.p) if q.status == 23 and q.coloured]
        if st.ptmax is None:
            pT2max = core.pt(ev.p[hard[0]].p) ** 2 + sum(ev.p[i].m ** 2 for i in hard) / len(hard)
        else:
            pT2max = st.ptmax ** 2
        sh = core.Shower(ev, self.alpha, self.pdf, np.random.default_rng(seed), ptmin_fsr=st.ptmin_fsr,
                         ptmin_isr=st.ptmin_isr, pt0_isr=st.pt0, do_isr=not st.no_isr, do_fsr=not st.no_fsr,
                         dampen_beam_recoil=not st.no_dampen)
        sh.track = {i: i for i in hard}
        return ev, sh, hard, pT2max

    def replay(self, seed: int, snaps: List[Snap]):
        """Run Shower.run() of the standalone script, taking a snapshot after every branching."""
        ev, sh, hard, pT2max = self.new_shower(seed)
        snaps.append(Snap.take(ev, sh, 0, math.sqrt(pT2max), None))
        branch_fsr, branch_isr = sh.branch_fsr, sh.branch_isr

        def wrap(branch):
            def wrapped(t):
                branch(t)
                snaps.append(Snap.take(ev, sh, len(sh.history), math.sqrt(t["pT2"]), sh.history[-1]))
            return wrapped

        sh.branch_fsr, sh.branch_isr = wrap(branch_fsr), wrap(branch_isr)
        sh.run(pT2max)
        return ev, sh, hard, math.sqrt(pT2max)

    def first_emission_trials(self, n: int, seed: int) -> Dict[str, np.ndarray]:
        """pT of the first emission proposed from pTmax by each end, n times (NaN = none above the cut-off)."""
        ev, sh, hard, pT2max = self.new_shower(seed)
        ends = sh.dipole_ends()
        out: Dict[str, np.ndarray] = {}
        for (i, k, side) in ends:
            key = "b" if ev.p[i].id == 5 else "bbar" if ev.p[i].id == -5 else core.pname(ev.p[i].id)
            vals = np.full(n, np.nan)
            for j in range(n):
                t = sh.pT2next_fsr((i, k, side), pT2max)
                if t:
                    vals[j] = math.sqrt(t["pT2"])
            out[key] = vals
        if sh.do_isr:
            for s, key in ((0, "A"), (1, "B")):
                vals = np.full(n, np.nan)
                for j in range(n):
                    t = sh.pT2next_isr(s, pT2max)
                    if t:
                        vals[j] = math.sqrt(t["pT2"])
                out[key] = vals
        return out

    def ensemble(self, n: int, seed0: int):
        """Summaries (core.summarise) and final states of n showers with seeds seed0, seed0+1, ..."""
        out = []
        args = self.st.as_args()
        for s in range(n):
            ev, sh, hard, _ = core.shower_event(args, seed0 + s, self.pdf, self.alpha)
            summ = core.summarise(ev, sh, hard, self.st.radius)
            snap = Snap.take(ev, sh, len(sh.history), 0.0, None)
            out.append((seed0 + s, summ, snap))
        return out


def check_against_record(phys: Physics, path: Path) -> str:
    """Compare the replayed final state with the committed ../event_PS.lhe."""
    if not path.exists():
        return f"{path.name} not found, nothing to compare with"
    try:
        rows = journey.parse_listing(path.read_text().splitlines())
    except ValueError as exc:
        return f"{path.name}: {exc}"
    last = phys.snaps[-1]
    n_file = sum(1 for r in rows.values() if r["status"] > 0)
    if len(rows) != len(last.rows) or n_file != len(last.final):
        return (f"replay differs from {path.name}: {len(last.rows)} rows / {len(last.final)} final partons here, "
                f"{len(rows)} / {n_file} in the file (different seed or options?)")
    worst = 0.0
    for i in last.final:
        r = rows.get(i)
        if r is None:
            continue
        q = last.rows[i]
        worst = max(worst, abs(q.p[1] - r["px"]), abs(q.p[2] - r["py"]), abs(q.p[3] - r["pz"]))
    return f"replay reproduces {path.name}: {len(rows)} rows, largest momentum difference {worst:.4f} GeV"


# --- drawing helpers --------------------------------------------------------------
def chrome(fig, scene_no: int, title: str, caption: str):
    fig.text(0.015, 0.955, title, fontsize=15, color=TEXT, weight="bold", ha="left", va="center")
    fig.text(0.985, 0.955, f"scene {scene_no}/4", fontsize=10, color=TEXT2, ha="right", va="center")
    fig.text(0.5, 0.045, caption, fontsize=10.5, color=TEXT, ha="center", va="center", wrap=True)


def style_2d(ax, fontsize=8):
    ax.tick_params(colors=TEXT2, labelsize=fontsize)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT2)
    ax.grid(color=GRID, lw=0.6)


def wrap_phi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def draw_eta_phi(ax, snap: Snap, scale: float = 1.0, dipoles: bool = True, highlight=(), labels: bool = True,
                 regions: bool = True, fontsize: float = 8.0):
    """The (eta, phi) picture of plot_journey_PS.py for one snapshot."""
    if regions:
        for label, lo, hi, shade in journey.REGIONS:
            for sgn in (-1, 1):
                ax.axvspan(sgn * lo, sgn * hi, color=shade, lw=0, zorder=0)
            if labels:
                for x in ({0.0} if lo == 0 else {0.5 * (lo + hi), -0.5 * (lo + hi)}):
                    ax.text(x, math.pi + 0.1, label, ha="center", va="bottom", fontsize=fontsize - 1.5, color=TEXT2)
        for edge in (2.5, 3.0):
            for sgn in (-1, 1):
                ax.axvline(sgn * edge, color="#ffffff", lw=1.0, zorder=1)
    ax.axhline(0.0, color=TEXT2, lw=0.5, ls=":", zorder=1)
    rows = snap.rows
    pos = {}
    for i in snap.final:
        p = rows[i].p
        pos[i] = (max(-ETA_MAX + 0.1, min(ETA_MAX - 0.1, core.eta(p))), core.phi(p))
    if dipoles:
        for i, j in snap.dipoles():
            if i in pos and j in pos:
                (e1, f1), (e2, f2) = pos[i], pos[j]
                d = wrap_phi(f2 - f1)
                ax.plot([e1, e2], [f1, f1 + d], color=TEXT2, lw=0.6, alpha=0.35, zorder=2)
                if abs(f1 + d) > math.pi:               # the short way round crosses the phi boundary
                    ax.plot([e1, e2], [f1 - math.copysign(2 * math.pi, d), f1 + d - math.copysign(2 * math.pi, d)],
                            color=TEXT2, lw=0.6, alpha=0.35, zorder=2)
            else:                                        # one end is an incoming parton: run to the eta edge
                fin = i if i in pos else j
                inc = j if fin == i else i
                if fin not in pos:
                    continue
                e1, f1 = pos[fin]
                side = 1 if inc == snap.inA else -1
                ax.plot([e1, side * ETA_MAX], [f1, f1], color=TEXT2, lw=0.6, alpha=0.35, ls="--", zorder=2)
    stars = snap.stars
    order = sorted(snap.final, key=lambda i: -core.pt(rows[i].p))
    for i in order:
        q = rows[i]
        e, f = pos[i]
        ptv = core.pt(q.p)
        col = FAMILY[snap.family(i)]["color"]
        size = journey.area(ptv) * scale
        if i in highlight:
            ax.scatter(e, f, s=size * 3.0 + 80 * scale, marker="o", facecolor="none", edgecolor=col, linewidth=1.5,
                       zorder=5)
        if i in stars.values():
            pid = next(p for p, r in stars.items() if r == i)
            ax.scatter(e, f, s=size, marker=STAR[pid]["marker"], color=col, edgecolor=TEXT, linewidth=1.0, zorder=4)
            if labels:
                dx = math.sqrt(size / math.pi) + 3.0
                ax.annotate(f"{STAR[pid]['name']}  {ptv:.1f} GeV", (e, f), xytext=(dx if e < 0 else -dx, 0),
                            textcoords="offset points", ha="left" if e < 0 else "right", va="center",
                            fontsize=fontsize, color=TEXT, zorder=6)
        else:
            ax.scatter(e, f, s=size, marker="o" if q.id == 21 else "D", color=col, alpha=0.8, edgecolor="#ffffff",
                       linewidth=0.6, zorder=3)
    ax.set_xlim(-ETA_MAX, ETA_MAX)
    ax.set_ylim(-math.pi, math.pi)
    if labels:
        ax.set_yticks([-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi])
        ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        ax.set_xlabel(r"pseudorapidity $\eta$", color=TEXT, fontsize=fontsize + 1)
        ax.set_ylabel(r"azimuth $\phi$", color=TEXT, fontsize=fontsize + 1)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.tick_params(colors=TEXT2, labelsize=fontsize)
    ax.grid(color="#ffffff", lw=0.6, zorder=1)
    ax.set_axisbelow(True)


def describe(rec: dict) -> str:
    """One line for a history entry, as in print_summary of the standalone script."""
    if rec["kind"] == "FSR":
        against = ("beam " if rec["recoiler_incoming"] else "") + rec["recoiler"]
        return (f"FSR {rec['branching']}: {rec['radiator']} radiates against {against}\n"
                f"pT = {rec['pT']:.2f} GeV, z = {rec['z']:.3f}, virtuality {rec['m_virtual']:.1f} GeV")
    return (f"ISR {rec['branching']} on side {rec['side']}: {rec['mother']} -> {rec['daughter']} + {rec['emitted']}\n"
            f"pT = {rec['pT']:.2f} GeV, z = {rec['z']:.3f}, x -> {rec['x_new']:.3g}")


# --- scenes -----------------------------------------------------------------------
class Scene:
    """A scene is a number of frames and a draw(fig, k) method for frame k."""
    number = 0
    title = ""
    nframes = 1

    def draw(self, fig, k: int) -> None:
        raise NotImplementedError

    def keyframe(self) -> int:
        return self.nframes // 2


class StepClock:
    """Maps frames to (snapshot index, current scale) for a per-branching sweep."""

    def __init__(self, phys: Physics, k_intro: int, per_step: int, k_tail: int, k_hold: int):
        self.phys = phys
        self.k_intro, self.per_step, self.k_tail, self.k_hold = k_intro, per_step, k_tail, k_hold
        self.nsteps = len(phys.snaps) - 1
        self.k_sweep_end = k_intro + per_step * self.nsteps
        self.nframes = self.k_sweep_end + k_tail + k_hold

    def at(self, k: int):
        """(snapshot shown, scale pT_now, frames since the last branching)."""
        snaps, phys = self.phys.snaps, self.phys
        pTmin = min(phys.pTmin_fsr, phys.pTmin_isr if phys.sh.do_isr else phys.pTmin_fsr)
        if k < self.k_intro:
            return 0, phys.pTmax, k
        if k >= self.k_sweep_end:
            t = smooth(ramp(k, self.k_sweep_end, self.k_sweep_end + self.k_tail))
            return self.nsteps, math.exp(math.log(snaps[-1].pT) * (1 - t) + math.log(pTmin) * t), k - self.k_sweep_end
        s = (k - self.k_intro) // self.per_step + 1          # branching about to happen / just happened
        j = (k - self.k_intro) % self.per_step
        fire = max(1, self.per_step - 3)                      # frame of the block at which the branching appears
        t = smooth(min(1.0, j / float(fire)))
        pT = math.exp(math.log(snaps[s - 1].pT) * (1 - t) + math.log(snaps[s].pT) * t)
        if j >= fire:
            return s, snaps[s].pT, j - fire
        return s - 1, pT, self.per_step


class ShowerScene(Scene):
    number, title = 1, "The shower unwinds in $p_T$: branching by branching from 113 GeV down to the cut-off"

    def __init__(self, phys: Physics):
        self.phys = phys
        self.clock = StepClock(phys, k_intro=42, per_step=6, k_tail=16, k_hold=50)
        self.nframes = self.clock.nframes
        self.has_split = any(s.record and s.record["kind"] == "FSR" and s.record["branching"] == "g->qqbar"
                             for s in phys.snaps)

    def keyframe(self):
        return self.clock.k_intro + self.clock.per_step * (self.clock.nsteps * 2 // 3)

    def draw(self, fig, k):
        phys, snaps = self.phys, self.phys.snaps
        s, pT_now, since = self.clock.at(k)
        snap = snaps[s]
        done = snaps[1:s + 1]

        # ladder of branchings: pT against branching number
        ax = fig.add_axes([0.06, 0.13, 0.30, 0.76])
        style_2d(ax)
        ax.set_yscale("log")
        ax.set_xlim(0, self.clock.nsteps + 1)
        ax.set_ylim(0.3, 700)
        ax.set_xlabel("branching number (in order of decreasing $p_T$)", color=TEXT, fontsize=9)
        ax.set_ylabel(r"$p_T$ of the branching  [GeV]", color=TEXT, fontsize=9)
        ax.axhline(phys.pTmax, color=TEXT2, lw=0.8, ls="--")
        ax.text(0.3, phys.pTmax * 0.78, rf"$p_{{T\max}}$ = {phys.pTmax:.1f} GeV (hard process)", fontsize=8, color=TEXT2)
        ax.axhspan(0.3, phys.pTmin_fsr, color="#efede6", zorder=0)
        ax.text(0.3, phys.pTmin_fsr * 0.72, rf"FSR cut-off {phys.pTmin_fsr:.2f} GeV", fontsize=7.5, color=TEXT2)
        if phys.sh.do_isr:
            ax.axhline(phys.pTmin_isr, color=TEXT2, lw=0.6, ls=":")
            ax.text(self.clock.nsteps + 0.8, phys.pTmin_isr * 1.1, rf"ISR cut-off {phys.pTmin_isr:.2f}", fontsize=7,
                    color=TEXT2, ha="right")
        for m, lab in ((core.MB, "$m_b$"), (core.MC, "$m_c$")):
            ax.axhline(m, color=GRID, lw=0.8)
            ax.text(self.clock.nsteps + 0.8, m * 1.08, lab + rf" = {m:g} GeV: $n_f$ changes", fontsize=7, color=TEXT2,
                    ha="right")
        ax.axhline(pT_now, color=TEXT, lw=1.2)
        ax.text(self.clock.nsteps + 0.8, pT_now * 1.12, rf"$p_T^{{\rm now}}$ = {pT_now:.2f} GeV", fontsize=8.5,
                color=TEXT, ha="right", weight="bold")
        for sn in done:
            rec = sn.record
            new = sn.new_rows()
            fam = sn.family(new[0]) if new else "ISR"
            col = COL_ISR if rec["kind"] == "ISR" else FAMILY[fam]["color"]
            marker = "s" if rec["kind"] == "ISR" else "o"
            ax.plot([sn.step], [rec["pT"]], marker=marker, ms=6 if sn is snap else 4.5, color=col, mec=TEXT if sn is snap else "white",
                    mew=0.8, ls="")
        if len(done) > 1:
            ax.plot([sn.step for sn in done], [sn.record["pT"] for sn in done], color=TEXT2, lw=0.6, alpha=0.5, zorder=1)
        ax.legend(handles=[Line2D([], [], marker="o", color=COL_B, ls="", ms=5, label="FSR of the $b$ (family)"),
                           Line2D([], [], marker="o", color=COL_BBAR, ls="", ms=5, label=r"FSR of the $\bar b$ (family)"),
                           Line2D([], [], marker="s", color=COL_ISR, ls="", ms=5, label="ISR, either beam (square rung)"),
                           Line2D([], [], marker="o", color="#b9b7b0", ls="", ms=5, label="gluon (right: circle)"),
                           Line2D([], [], marker="D", color="#b9b7b0", ls="", ms=4.5, label="quark (right: diamond)"),
                           Line2D([], [], color=TEXT2, lw=0.8, alpha=0.6, label="colour dipole (right)")],
                  loc="upper right", fontsize=7.5, frameon=False, labelspacing=0.35)
        if snap.record:
            ax.text(0.02, 0.02, f"branching {snap.step} of {self.clock.nsteps}\n" + describe(snap.record),
                    transform=ax.transAxes, fontsize=8.3, color=TEXT, va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#fbfaf7", ec=GRID))
        else:
            ax.text(0.02, 0.02, "hard process: no branching yet\ntwo colour dipoles, each from a $b$ quark\nto the incoming parton it is connected to",
                    transform=ax.transAxes, fontsize=8.3, color=TEXT, va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#fbfaf7", ec=GRID))

        # eta-phi picture
        ax2 = fig.add_axes([0.43, 0.13, 0.545, 0.76])
        highlight = set(snap.new_rows()) if since < 6 else set()
        draw_eta_phi(ax2, snap, highlight=highlight)
        fin = snap.final
        n_isr = sum(1 for sn in done if sn.record["kind"] == "ISR")
        info = [f"{len(fin)} final-state partons ({sum(1 for i in fin if snap.rows[i].id == 21)} gluons)",
                f"{n_isr} ISR + {len(done) - n_isr} FSR branchings so far"]
        if phys.sh.do_isr:
            xa, xb = abs(snap.rows[snap.inA].p[3]) / phys.ev.ebeam, abs(snap.rows[snap.inB].p[3]) / phys.ev.ebeam
            info.append(f"incoming: {core.pname(snap.rows[snap.inA].id)} x = {xa:.3g} (+z), "
                        f"{core.pname(snap.rows[snap.inB].id)} x = {xb:.3g} (-z)")
        ax2.text(0.99, 0.985, "\n".join(info), transform=ax2.transAxes, fontsize=8, color=TEXT, va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
        # at the end: the family of each b quark
        if k >= self.clock.k_sweep_end + self.clock.k_tail // 2:
            g = smooth(ramp(k, self.clock.k_sweep_end + self.clock.k_tail // 2, self.clock.k_sweep_end + self.clock.k_tail + 10))
            for pid, cur in snap.stars.items():
                p = snap.rows[cur].p
                fam = core.family(core.Event(snap.rows, phys.ev.ebeam, snap.inA, snap.inB), cur, phys.st.radius)
                e, f = core.eta(p), core.phi(p)
                col = FAMILY["b" if pid == 5 else "bbar"]["color"]
                ax2.add_patch(Circle((e, f), phys.st.radius * g, fill=False, ec=col, lw=1.4, ls="--", zorder=6))
                ax2.annotate(f"family within $\\Delta R$ < {phys.st.radius}: {fam['n']} partons,\n"
                             f"$p_T$ = {fam['pt']:.1f} GeV, $m$ = {fam['mass']:.1f} GeV",
                             (e, f + phys.st.radius), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                             fontsize=8, color=col, alpha=g, zorder=6,
                             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8 * g))

        n = self.clock.nsteps
        if k < self.clock.k_intro:
            cap = (r"Chapter 1 left a $b$ and a $\bar b$ at $p_T$ = 113 GeV, each colour-connected to an incoming parton: "
                   r"two dipoles, free to radiate from $p_{T\max}$ = 113.3 GeV downwards.")
        elif s <= n // 4:
            cap = (r"Every dipole end and both beams propose their next emission; the hardest wins and all restart from "
                   r"its $p_T$. Marker area is $p_T$; a ring marks the partons just created.")
        elif s <= n // 2:
            cap = (r"An ISR branching (square rung) replaces an incoming parton by one of larger $x$ and adds a parton "
                   r"to the final state; the whole event recoils, so every marker shifts.")
        elif s < n:
            cap = (r"Lower in $p_T$ the emissions get softer and more frequent: $\alpha_s$ grows and $1/p_T^2$ takes over. "
                   + (r"A gluon may also split into $q\bar q$, here even into $b\bar b$." if self.has_split
                      else r"Gluons may also split into $q\bar q$."))
        else:
            cap = (rf"At the cut-off the shower stops with {len(fin)} partons. Within $\Delta R$ < {phys.st.radius} of each $b$ "
                   r"quark most of its radiation is recovered: the family a jet algorithm will collect.")
        chrome(fig, self.number, self.title, cap)


class SudakovScene(Scene):
    number, title = 2, "Who radiates first? The Sudakov veto algorithm as an ensemble of proposals"

    KEYS = {"b": (COL_B, "$b$ dipole end"), "bbar": (COL_BBAR, r"$\bar b$ dipole end"),
            "A": (COL_A, "ISR, incoming from $+z$"), "B": (COL_BEAM, "ISR, incoming from $-z$")}

    def __init__(self, phys: Physics, trials: int, seed: int):
        self.phys = phys
        self.trials = phys.first_emission_trials(trials, seed)
        self.keys = [k for k in self.KEYS if k in self.trials]
        self.n = trials
        arr = np.vstack([self.trials[k] for k in self.keys])            # (ends, n), NaN = no emission
        filled = np.where(np.isnan(arr), -1.0, arr)
        self.winner = filled.argmax(axis=0)
        self.win_pt = filled.max(axis=0)
        self.pmin = min(phys.pTmin_fsr, phys.pTmin_isr) * 0.8
        self.bins = np.geomspace(self.pmin, phys.pTmax, 28)
        self.grid = np.geomspace(self.pmin, phys.pTmax, 200)
        self.first = phys.history[0] if phys.history else None
        self.nframes = 190
        self.n_of_frame = [int(round(4 * (self.n / 4.0) ** ramp(k, 0, self.nframes - 45))) for k in range(self.nframes)]

    def keyframe(self):
        return self.nframes - 30

    def survival(self, vals: np.ndarray, n: int) -> np.ndarray:
        v = vals[:n]
        v = np.where(np.isnan(v), -1.0, v)
        return np.array([(v < g).mean() for g in self.grid])

    def draw(self, fig, k):
        phys = self.phys
        n = self.n_of_frame[k]
        ax = fig.add_axes([0.06, 0.15, 0.42, 0.74])
        style_2d(ax)
        ax.set_xscale("log")
        ax.set_xlim(phys.pTmax * 1.05, self.pmin)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(r"$p_T$  [GeV]   (the evolution runs from left to right, downwards in $p_T$)", color=TEXT, fontsize=9)
        ax.set_ylabel(r"probability of no emission above $p_T$:  $\Delta(p_{T\max}, p_T)$", color=TEXT, fontsize=9)
        prod = np.ones_like(self.grid)
        for key in self.keys:
            col, lab = self.KEYS[key]
            surv = self.survival(self.trials[key], n)
            prod *= surv
            ax.plot(self.grid, surv, color=col, lw=1.8, label=lab)
        ax.plot(self.grid, prod, color=TEXT, lw=1.4, ls="--", label="all four together (product)")
        if self.first:
            ax.axvline(self.first["pT"], color=TEXT, lw=0.8, ls=":")
            ax.text(self.first["pT"] * 0.95, 0.5, f"seed-1 event:\nfirst branching at\n{self.first['pT']:.1f} GeV",
                    fontsize=8, color=TEXT, ha="left", va="center")
        ax.legend(loc="lower left", fontsize=8, frameon=False)
        ax.set_title(f"$N$ = {n} proposals per end, from $p_{{T\\max}}$ = {phys.pTmax:.1f} GeV", fontsize=10,
                     color=TEXT, loc="left")

        ax2 = fig.add_axes([0.56, 0.15, 0.42, 0.74])
        style_2d(ax2)
        ax2.set_xscale("log")
        ax2.set_xlim(self.pmin, phys.pTmax * 1.05)
        ax2.set_xlabel(r"$p_T$ of the first branching (hardest of the four proposals)  [GeV]", color=TEXT, fontsize=9)
        ax2.set_ylabel("trials", color=TEXT, fontsize=9)
        bottom = np.zeros(len(self.bins) - 1)
        shares = []
        for idx, key in enumerate(self.keys):
            col, lab = self.KEYS[key]
            sel = (self.winner[:n] == idx) & (self.win_pt[:n] > 0)
            h, _ = np.histogram(self.win_pt[:n][sel], bins=self.bins)
            ax2.bar(self.bins[:-1], h, width=np.diff(self.bins), bottom=bottom, align="edge", color=col, alpha=0.85,
                    lw=0)
            bottom += h
            shares.append((lab, sel.sum() / max(n, 1)))
        none = float(((self.win_pt[:n] <= 0)).mean())
        ax2.set_ylim(0, max(bottom.max() * 1.9, 4))
        if self.first:
            ax2.axvline(self.first["pT"], color=TEXT, lw=0.8, ls=":")
        txt = "\n".join([f"who wins the first branching (N = {n}):"] +
                        [f"  {lab}: {100 * f:.0f} %" for lab, f in shares] +
                        [f"  nobody (no emission at all): {100 * none:.1f} %", "",
                         f"seed-1 event: {describe(self.first).splitlines()[0] if self.first else '-'}"])
        ax2.text(0.02, 0.98, txt, transform=ax2.transAxes, fontsize=8.3, color=TEXT, va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#fbfaf7", ec=GRID))

        if k < 50:
            cap = (r"Each end samples its next emission from $p_{T\max}$ by the veto algorithm: an overestimate of "
                   r"$\alpha_s P(z)/p_T^2$ proposes, the exact kernel (PDF ratio, two-loop $\alpha_s$, dead cone) accepts or rejects.")
        elif k < 120:
            cap = (r"The fraction of proposals still without an emission at $p_T$ is that end's Sudakov form factor "
                   r"$\Delta(p_{T\max}, p_T) = \exp[-\int_{p_T}^{p_{T\max}} \mathrm{d}p_T'^2 \int \mathrm{d}z\, \alpha_s P/(2\pi p_T'^2)]$.")
        else:
            best = max(shares, key=lambda t: t[1])
            why = (r"the PDF ratio $x'f(x')/xf(x)$ is large at small $x$" if best[0].startswith("ISR")
                   else r"its dipole has the largest mass")
            cap = rf"The hardest proposal wins: here {best[0]} does in {100 * best[1]:.0f} % of the trials, because {why}."
        chrome(fig, self.number, self.title, cap)


class RecoilScene(Scene):
    number, title = 3, "Momentum bookkeeping: the $b\\bar b$ system recoils against the ISR, the beams walk up in $x$"

    def __init__(self, phys: Physics):
        self.phys = phys
        self.clock = StepClock(phys, k_intro=24, per_step=4, k_tail=10, k_hold=50)
        self.nframes = self.clock.nframes
        lim = 1.0
        for sn in phys.snaps:
            for i in sn.final:
                lim = max(lim, abs(sn.rows[i].p[1]), abs(sn.rows[i].p[2]))
        self.lim = 1.15 * lim
        self.xgrid = np.geomspace(1.0e-3, 0.99, 120)

    def keyframe(self):
        return self.clock.k_intro + self.clock.per_step * (self.clock.nsteps // 2)

    def sums(self, snap: Snap):
        rows = snap.rows
        stars = snap.stars
        bb = sum((rows[i].p for i in stars.values()), np.zeros(4))
        isr = sum((rows[i].p for i in snap.final if snap.family(i) == "ISR"), np.zeros(4))
        tot = sum((rows[i].p for i in snap.final), np.zeros(4))
        return bb, isr, tot

    def draw(self, fig, k):
        phys, snaps = self.phys, self.phys.snaps
        s, pT_now, since = self.clock.at(k)
        snap = snaps[s]
        rows = snap.rows
        bb, isr, tot = self.sums(snap)

        ax = fig.add_axes([0.05, 0.15, 0.42, 0.75])
        style_2d(ax)
        ax.set_xlim(-self.lim, self.lim)
        ax.set_ylim(-self.lim, self.lim)
        ax.set_aspect("equal")
        ax.set_xlabel("$p_x$  [GeV]", color=TEXT, fontsize=9)
        ax.set_ylabel("$p_y$  [GeV]", color=TEXT, fontsize=9)
        ax.axhline(0, color=TEXT2, lw=0.4)
        ax.axvline(0, color=TEXT2, lw=0.4)
        stars = snap.stars
        new = set(snap.new_rows()) if since < 4 else set()
        for i in snap.final:
            p = rows[i].p
            col = FAMILY[snap.family(i)]["color"]
            star = i in stars.values()
            ax.annotate("", (p[1], p[2]), (0, 0), zorder=3 if star else 2,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0 if star else (1.4 if i in new else 0.7),
                                        alpha=1.0 if (star or i in new) else 0.55, shrinkA=0, shrinkB=0,
                                        mutation_scale=9 if star else 7))
        for v, col, lab, ls in ((bb, TEXT, r"$b\bar b$ system", "--"), (isr, COL_ISR, "ISR sum", "--")):
            if math.hypot(v[1], v[2]) < 0.05:
                continue
            ax.annotate("", (v[1], v[2]), (0, 0), zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=10))
            nrm = math.hypot(v[1], v[2])
            ax.annotate(lab, (v[1], v[2]), xytext=(10 * v[1] / nrm, 10 * v[2] / nrm), textcoords="offset points",
                        fontsize=8.5, color=col, ha="right" if v[1] < 0 else "left", va="bottom" if v[2] > 0 else "top")
        ax.plot([tot[1]], [tot[2]], marker="x", ms=8, color=TEXT, mew=1.5)
        info = "\n".join([
            f"after {s} of {self.clock.nsteps} branchings, $p_T^{{\\rm now}}$ = {pT_now:.2f} GeV",
            rf"$b$: $p_T$ = {core.pt(rows[stars[5]].p):.1f} GeV,  $\bar b$: $p_T$ = {core.pt(rows[stars[-5]].p):.1f} GeV"
            if 5 in stars and -5 in stars else "",
            rf"$b\bar b$ system: $|\sum \vec p_T|$ = {math.hypot(bb[1], bb[2]):.1f} GeV, $m$ = {math.sqrt(max(core.mass2(bb), 0)):.0f} GeV",
            rf"ISR partons: $|\sum \vec p_T|$ = {math.hypot(isr[1], isr[2]):.1f} GeV",
            rf"all {len(snap.final)} partons: $|\sum \vec p_T|$ = {math.hypot(tot[1], tot[2]):.2f} GeV",
        ])
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8.5, color=TEXT, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="#fbfaf7", ec=GRID))

        # PDFs at the current scale and the two incoming partons
        ax2 = fig.add_axes([0.56, 0.15, 0.42, 0.75])
        style_2d(ax2)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlim(1.0e-3, 1.0)
        ax2.set_ylim(0.02, 30)
        ax2.set_xlabel("momentum fraction $x$", color=TEXT, fontsize=9)
        ax2.set_ylabel(r"$x\,f(x, Q^2)$ at $Q$ = $p_T^{\rm now}$", color=TEXT, fontsize=9)
        if phys.pdf is not None:
            q2 = max(pT_now ** 2, phys.pdf.q2min)
            ax2.set_title(f"NNPDF3.1 at $Q$ = {math.sqrt(q2):.2f} GeV" +
                          (" (frozen below 1.65 GeV)" if pT_now ** 2 < phys.pdf.q2min else ""), fontsize=10, color=TEXT, loc="left")
            for pid, col, lab, ls in ((21, COL_ISR, "gluon", "-"), (2, COL_BEAM, "$u$", "-"), (-2, COL_A, r"$\bar u$", "-"),
                                      (1, "#9a6b2f", "$d$", ":")):
                y = [phys.pdf.xf(pid, float(x), q2) for x in self.xgrid]
                ax2.plot(self.xgrid, y, color=col, lw=1.4, ls=ls, label=lab, alpha=0.9)
            for side, name, col in ((snap.inA, "from $+z$", COL_A), (snap.inB, "from $-z$", COL_BEAM)):
                q = rows[side]
                x = abs(q.p[3]) / phys.ev.ebeam
                y = phys.pdf.xf(q.id, x, q2)
                trail = []
                for sn in snaps[:s + 1]:
                    r = sn.rows[sn.inA if col == COL_A else sn.inB]
                    trail.append((abs(r.p[3]) / phys.ev.ebeam, phys.pdf.xf(r.id, abs(r.p[3]) / phys.ev.ebeam, q2)))
                ax2.plot([t[0] for t in trail], [t[1] for t in trail], color=col, lw=1.0, ls="--", alpha=0.6)
                ax2.plot([x], [y], marker="o", ms=9, color=col, mec=TEXT, mew=1.0)
                ax2.annotate(f"{tex_name(q.id)} {name}\n$x$ = {x:.3g}", (x, y), xytext=(8, -14), textcoords="offset points",
                             fontsize=8.5, color=col, ha="left", va="top",
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
            ax2.legend(loc="lower left", fontsize=8, frameon=False, title="flavour", title_fontsize=8)
        else:
            ax2.text(0.5, 0.5, "ISR switched off (--no-isr)", transform=ax2.transAxes, ha="center", color=TEXT2)

        if k < self.clock.k_intro:
            cap = (r"Before the shower the $b$ and $\bar b$ are back to back with 113 GeV each and the incoming partons "
                   r"have no $p_T$: the total transverse momentum is zero and must stay zero.")
        elif s < self.clock.nsteps // 2:
            cap = (r"FSR shares the radiator's momentum with the new parton and the recoiler; ISR kicks the whole final "
                   r"state. The $b\bar b$ system drifts off zero, the ISR sum balances it.")
        elif s < self.clock.nsteps:
            cap = (r"Backward evolution: each ISR branching replaces the incoming parton by its mother at $x/z$, a step "
                   r"up in $x$, while the PDFs themselves change with the falling scale.")
        else:
            cap = (rf"Final state: the $b\bar b$ pair carries {math.hypot(bb[1], bb[2]):.1f} GeV of net $p_T$, the ISR partons "
                   rf"{math.hypot(isr[1], isr[2]):.1f} GeV the other way, the rest is FSR; all {len(snap.final)} partons sum to zero.")
        chrome(fig, self.number, self.title, cap)


class EnsembleScene(Scene):
    number, title = 4, "The same hard event showered again and again: no two alike, the averages agree with PYTHIA"

    def __init__(self, phys: Physics, n_showers: int, seed0: int):
        self.phys = phys
        self.runs = phys.ensemble(n_showers, seed0)
        self.n = len(self.runs)
        self.n_thumbs = min(12, self.n)
        self.k_thumbs = 9 * self.n_thumbs
        self.k_conv = self.k_thumbs + 110
        self.nframes = self.k_conv + 45
        self.counts = np.arange(1, self.n + 1)
        self.series = {}
        for key in ("n_isr", "n_fsr", "n_final"):
            v = np.array([r[1][key] for r in self.runs], dtype=float)
            self.series[key] = np.cumsum(v) / self.counts
        for name in ("b", "bbar"):
            v = np.array([next(q["family"]["pt"] for q in r[1]["partons"] if q["name"] == name) for r in self.runs])
            self.series["fam_" + name] = np.cumsum(v) / self.counts
            self.series["raw_" + name] = v

    def keyframe(self):
        return self.k_conv - 15

    def draw(self, fig, k):
        shown = min(self.n_thumbs, k // 9 + 1)
        n_now = 1 if k < self.k_thumbs else min(self.n, 1 + int(round((self.n - 1) * ramp(k, self.k_thumbs, self.k_conv))))
        n_now = max(n_now, shown)
        cols, rws = 4, 3
        w, h = 0.135, 0.245
        for j in range(shown):
            seed, summ, snap = self.runs[j]
            c, r = j % cols, j // cols
            ax = fig.add_axes([0.03 + c * (w + 0.008), 0.62 - r * (h + 0.015), w, h])
            draw_eta_phi(ax, snap, scale=0.35, dipoles=False, labels=False, regions=True, fontsize=6)
            ax.set_title(f"seed {seed}: {summ['n_isr']} ISR, {summ['n_fsr']} FSR, {summ['n_final']} partons",
                         fontsize=6.3, color=TEXT, loc="left", pad=2)
            for spine in ax.spines.values():
                spine.set_color(COL_B if seed == self.phys.st.seed else GRID)
        fig.text(0.03, 0.9, rf"$(\eta, \phi)$ of the final state, seeds 1 to {self.n_thumbs} (the book's event is seed {self.phys.st.seed})",
                 fontsize=9, color=TEXT2)

        x = self.counts[:n_now]
        ax_top = fig.add_axes([0.66, 0.55, 0.32, 0.33])
        ax_bot = fig.add_axes([0.66, 0.12, 0.32, 0.33])
        for a in (ax_top, ax_bot):
            style_2d(a)
            a.set_xlim(1, self.n * 1.05)
        for key, col, lab in (("n_isr", COL_ISR, "ISR branchings"), ("n_fsr", TEXT, "FSR branchings")):
            ax_top.plot(x, self.series[key][:n_now], color=col, lw=1.8, label=f"{lab}: {self.series[key][n_now - 1]:.2f}")
            ax_top.axhline(PYTHIA_REF[key], color=col, lw=0.8, ls="--")
            ax_top.text(self.n * 1.04, PYTHIA_REF[key], f"PYTHIA {PYTHIA_REF[key]}", fontsize=7, color=col, va="bottom", ha="right")
        ax_top.set_ylim(0, 38)
        ax_top.set_ylabel("running mean of the count", color=TEXT, fontsize=9)
        ax_top.set_title(f"$N$ = {n_now} showers of the same hard event", fontsize=10, color=TEXT, loc="left")
        ax_top.legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.85, facecolor="white", edgecolor="none")
        for name, col in (("b", COL_B), ("bbar", COL_BBAR)):
            lab = "$b$" if name == "b" else r"$\bar b$"
            ax_bot.plot(x, self.series["fam_" + name][:n_now], color=col, lw=1.8,
                        label=f"{lab} family: {self.series['fam_' + name][n_now - 1]:.1f} GeV")
            ax_bot.scatter(x, self.series["raw_" + name][:n_now], s=6, color=col, alpha=0.35, lw=0)
            ax_bot.axhline(PYTHIA_REF["fam_pt"][name], color=col, lw=0.8, ls="--")
            ax_bot.text(self.n * 1.04, PYTHIA_REF["fam_pt"][name], f"PYTHIA {PYTHIA_REF['fam_pt'][name]}", fontsize=7,
                        color=col, va="bottom", ha="right")
        ax_bot.axhline(113.2, color=TEXT2, lw=0.8, ls=":")
        ax_bot.text(1.2, 114.5, "before the shower: 113.2 GeV", fontsize=7, color=TEXT2)
        ax_bot.set_ylim(40, 130)
        ax_bot.set_xlabel("number of showers $N$", color=TEXT, fontsize=9)
        ax_bot.set_ylabel(rf"family $p_T$ ($\Delta R$ < {self.phys.st.radius})  [GeV]", color=TEXT, fontsize=9)
        ax_bot.legend(fontsize=8, loc="lower right", frameon=False)

        if k < self.k_thumbs:
            cap = (r"Same hard process, different random numbers: how many branchings, where the gluons go and how much "
                   r"$p_T$ the $b$ quarks keep all vary. Only distributions are predicted.")
        elif k < self.k_conv - 20:
            cap = (r"Averaged over many showers the counts settle near 5 ISR and 20 FSR branchings, and each $b$ family "
                   r"keeps about 105 of its 113 GeV within $\Delta R$ < 0.4; the rest leaks out of the cone.")
        else:
            cap = (rf"After {self.n} showers: {self.series['n_isr'][-1]:.1f} ISR, {self.series['n_fsr'][-1]:.1f} FSR branchings, family "
                   rf"$p_T$ {self.series['fam_b'][-1]:.0f} / {self.series['fam_bbar'][-1]:.0f} GeV; PYTHIA 8 on the same event: "
                   rf"{PYTHIA_REF['n_isr']} / {PYTHIA_REF['n_fsr']} and {PYTHIA_REF['fam_pt']['b']:.0f} / {PYTHIA_REF['fam_pt']['bbar']:.0f} GeV.")
        chrome(fig, self.number, self.title, cap)


# --- rendering ----------------------------------------------------------------------
SCENE_NAMES = {"1": "shower", "2": "sudakov", "3": "recoil", "4": "ensemble"}


def build_scenes(names: List[str], phys: Physics, args) -> List[Scene]:
    out: List[Scene] = []
    for name in names:
        if name == "shower":
            out.append(ShowerScene(phys))
        elif name == "sudakov":
            out.append(SudakovScene(phys, args.trials, args.seed + 1000))
        elif name == "recoil":
            out.append(RecoilScene(phys))
        elif name == "ensemble":
            out.append(EnsembleScene(phys, args.showers, args.seed))
    return out


def frame_list(scenes: List[Scene], step: int):
    frames = []
    for sc in scenes:
        frames += [(sc, k) for k in range(0, sc.nframes, step)]
        if (sc.nframes - 1) % step:
            frames.append((sc, sc.nframes - 1))
    return frames


def render_frame(fig, sc: Scene, k: int):
    fig.clf()
    fig.patch.set_facecolor("white")
    sc.draw(fig, k)


def write_video(scenes, out: Path, fps: int, dpi: int, step: int):
    frames = frame_list(scenes, step)
    ext = out.suffix.lower()
    if ext == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif ext in (".mp4", ".m4v", ".mov", ".webm"):
        if not animation.FFMpegWriter.isAvailable():
            sys.exit("ffmpeg not found on PATH; write a .gif instead or install ffmpeg")
        writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=3500,
                                        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                                        metadata={"title": "b bbar event, standalone parton shower"})
    else:
        sys.exit(f"unknown output type {ext}; use .mp4 or .gif")
    fig = plt.figure(figsize=FIGSIZE)
    print(f"writing {len(frames)} frames at {fps} fps ({len(frames) / fps:.1f} s) to {out}", flush=True)
    with writer.saving(fig, str(out), dpi):
        for i, (sc, k) in enumerate(frames):
            render_frame(fig, sc, k)
            writer.grab_frame(facecolor="white")
            if i % 25 == 0 or i == len(frames) - 1:
                print(f"  frame {i + 1}/{len(frames)}  (scene {sc.number}, k={k})", flush=True)
    plt.close(fig)


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
    """One page per scene at its representative frame (for the book)."""
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
    p.add_argument("-o", "--output", type=Path, default=HERE / "animate_standalone_parton_showering.mp4",
                   help=".mp4 (needs ffmpeg) or .gif")
    p.add_argument("--scene", default="all", help="all, or a comma list of shower,sudakov,recoil,ensemble (or 1,2,3,4)")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--dpi", type=int, default=100, help="100 gives 1280x720")
    p.add_argument("--step", type=int, default=1, help="render every step-th frame (faster, choppier)")
    p.add_argument("--fast", action="store_true", help="preview: --step 4 --dpi 60, fewer trials and showers")
    p.add_argument("--frames-dir", type=Path, default=None, help="write PNG frames here instead of a video")
    p.add_argument("--stills", type=Path, default=None, help="also write one key frame per scene (.pdf or .png)")
    p.add_argument("--no-video", action="store_true", help="skip the video (with --stills or --frames-dir)")
    p.add_argument("--show", action="store_true", help="play in an interactive window instead of writing a file")
    p.add_argument("-i", "--input", type=Path, default=HERE.parent.parent / "ME" / "event_ME.lhe",
                   help="hard-process record to shower")
    p.add_argument("--record", type=Path, default=HERE.parent / "event_PS.lhe",
                   help="showered record to compare the replay with")
    p.add_argument("--trials", type=int, default=1500, help="proposals per end in scene 2")
    p.add_argument("--showers", type=int, default=64, help="showers in scene 4 (about 0.3 s each)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--alphas", type=float, default=0.118)
    p.add_argument("--order", type=int, default=2, choices=(1, 2))
    p.add_argument("--ptmin-fsr", type=float, default=0.5)
    p.add_argument("--ptmin-isr", type=float, default=0.2)
    p.add_argument("--pt0", type=float, default=2.0)
    p.add_argument("--ptmax", type=float, default=None)
    p.add_argument("--radius", type=float, default=0.4)
    p.add_argument("--no-isr", action="store_true")
    p.add_argument("--no-dampen", action="store_true")
    args = p.parse_args()
    if args.fast:
        args.step, args.dpi = max(args.step, 4), min(args.dpi, 60)
        args.trials, args.showers = min(args.trials, 300), min(args.showers, 24)

    names = list(SCENE_NAMES.values()) if args.scene == "all" else \
        [SCENE_NAMES.get(s.strip(), s.strip()) for s in args.scene.split(",")]
    bad = [n for n in names if n not in SCENE_NAMES.values()]
    if bad:
        sys.exit(f"unknown scene(s) {bad}; choose from {list(SCENE_NAMES.values())}")

    st = Setup(args.input, args.seed, args.alphas, args.order, args.ptmin_fsr, args.ptmin_isr, args.pt0, args.ptmax,
               args.radius, args.no_isr, False, args.no_dampen)
    phys = Physics(st)
    last = phys.snaps[-1]
    n_isr = sum(1 for h in phys.history if h["kind"] == "ISR")
    print(f"shower of {args.input.name}, seed {args.seed}: pTmax = {phys.pTmax:.2f} GeV, cut-offs {phys.pTmin_fsr:.3f} (FSR) / "
          f"{phys.pTmin_isr:.3f} (ISR) GeV; {n_isr} ISR + {len(phys.history) - n_isr} FSR branchings, "
          f"{len(last.final)} final-state partons")
    print(check_against_record(phys, args.record))
    scenes = build_scenes(names, phys, args)
    for sc in scenes:
        if isinstance(sc, SudakovScene):
            shares = {key: float(np.mean(sc.winner == i)) for i, key in enumerate(sc.keys)}
            print(f"sudakov: {sc.n} proposals per end; first branching won by " +
                  ", ".join(f"{k}: {100 * v:.0f}%" for k, v in shares.items()))
        if isinstance(sc, EnsembleScene):
            print(f"ensemble: {sc.n} showers, mean n_ISR = {sc.series['n_isr'][-1]:.2f}, n_FSR = {sc.series['n_fsr'][-1]:.2f}, "
                  f"family pT b = {sc.series['fam_b'][-1]:.1f}, bbar = {sc.series['fam_bbar'][-1]:.1f} GeV")
    print("frames: " + ", ".join(f"{sc.number} {SCENE_NAMES[str(sc.number)]} {sc.nframes}" for sc in scenes) +
          f"  ({sum(sc.nframes for sc in scenes) / args.fps:.1f} s at {args.fps} fps)")

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
