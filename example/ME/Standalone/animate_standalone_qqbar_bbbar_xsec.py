#!/usr/bin/env python3
"""Animated 3D companion to standalone_qqbar_bbbar_xsec.py.

Four scenes, about 30 s in total, show what the standalone integrator does.
Every number on screen is computed with the functions of that script (the
tables, the PDF and alpha_s interpolators and born_weights), so the pictures
and the printed cross section come from the same code.

  1  collision  The event of ../event_ME.lhe. In the partonic centre-of-mass
                frame the u and ubar collide head on and the b and bbar leave
                back to back at the angle theta* to the beam. The translucent
                tube is the generator cut pTHat >= 100 GeV: the quarks must leave
                through its wall. The system is then boosted along the beam by
                the pair rapidity Y = -2.59 into the lab frame: the tips slide
                along the tube (pT is boost invariant) and the pair ends up far
                forward, exactly as listed in event_ME.lhe.
  2  cube       Sobol points in the unit cube (u1, u2, u3) that born_weights()
                samples, coloured by log10 of the weight sum_q w_q. The cube is
                then morphed into the physical variables (log10 x1, log10 x2,
                |cos theta|): the change of variables of step 1 of the chapter,
                whose Jacobian is part of the weight.
  3  surface    The integrand with cos theta integrated out,
                d sigma / (d log10 x1 d log10 x2), as a surface over the triangle
                x1 x2 >= tau_min. Its volume is the cross section, 0.345 nb. The
                event of scene 1 is marked on it.
  4  converge   One Sobol pass: the cube fills up point by point while the
                running average of the weights settles to 0.345 nb, per incoming
                flavour and in total.

Examples (matplotlib is needed; on this Mac use /usr/bin/python3):
  python3 animate_standalone_qqbar_bbbar_xsec.py                 # -> .mp4 next to this file
  python3 animate_standalone_qqbar_bbbar_xsec.py --fast          # quick low-resolution preview
  python3 animate_standalone_qqbar_bbbar_xsec.py --scene surface -o surface.gif
  python3 animate_standalone_qqbar_bbbar_xsec.py --stills stills.pdf --no-video
  python3 animate_standalone_qqbar_bbbar_xsec.py --show          # interactive window

An .mp4 needs ffmpeg on the PATH; a .gif needs only Pillow. --frames-dir writes
the individual frames as PNG files instead. The physics options (--ecm, --mb,
--ptmin, --ptmax, --power, --seed) are those of the standalone script.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import standalone_qqbar_bbbar_xsec as core  # noqa: E402  (same directory)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation, colors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# --- appearance --------------------------------------------------------------
# Same colours as plot_journey_ME.py for the b (blue) and bbar (orange).
COL_B, COL_BBAR = "#2a78d6", "#eb6834"
COL_U, COL_UBAR = "#2f8f5b", "#8e5bb5"          # incoming quark / antiquark
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d6d3ca"
FLAVOUR_COLOURS = {"d": "#9a6b2f", "u": "#2a78d6", "s": "#2f8f5b", "c": "#eb6834", "b": "#8e5bb5"}
CMAP = "viridis"
PYTHIA_NB, PYTHIA_ERR_NB = 0.3451, 0.0061           # CMSSW step1_GEN_cfg.py GenXSecAnalyzer, pTHat>=100 GeV
REPEATS8_NB = 0.345324                              # qqbar_bbbar_xsec.py --output result.json (repeats=8)
SOBOL1_NB = 0.345320                                # standalone_qqbar_bbbar_xsec.py, single 2^16-point Sobol pass
FIGSIZE = (12.8, 7.2)                               # 16:9; dpi 100 -> 1280x720


def smooth(t: float) -> float:
    """Ease-in/out in [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def ramp(k: int, k0: int, k1: int) -> float:
    """Fraction of the way from frame k0 to frame k1 (clamped)."""
    if k1 <= k0:
        return 1.0
    return min(max((k - k0) / float(k1 - k0), 0.0), 1.0)


# --- physics setup -------------------------------------------------------------
@dataclass
class Setup:
    ecm: float = 13600.0
    mb: float = 4.8
    ptmin: float = 100.0
    ptmax: Optional[float] = None
    power: int = 16
    seed: int = 24680

    @property
    def s(self) -> float:
        return self.ecm ** 2

    @property
    def taumin(self) -> float:
        return 4.0 * (self.mb ** 2 + self.ptmin ** 2) / self.s


def map_cube(u: np.ndarray, st: Setup) -> Dict[str, np.ndarray]:
    """Unit-cube points (N, 3) -> physical variables. The same map as core.born_weights."""
    logwidth = math.log(1.0 / st.taumin)
    logtau = math.log(st.taumin) + logwidth * u[:, 0]
    tau = np.exp(logtau)
    shat = tau * st.s
    ymax = -0.5 * logtau
    Y = (2.0 * u[:, 1] - 1.0) * ymax
    x1 = np.exp(0.5 * logtau + Y)
    x2 = np.exp(0.5 * logtau - Y)
    p2 = np.clip(shat / 4.0 - st.mb ** 2, 0.0, None)
    p2s = np.maximum(p2, st.ptmin ** 2 + 1.0e-9)
    cmax = np.sqrt(np.clip(1.0 - st.ptmin ** 2 / p2s, 0.0, None))
    if st.ptmax is None:
        cmin = np.zeros_like(cmax)
    else:
        cmin = np.where(st.ptmax ** 2 < p2s, np.sqrt(np.clip(1.0 - st.ptmax ** 2 / p2s, 0.0, None)), 0.0)
    c = cmin + (cmax - cmin) * u[:, 2]
    pt = np.sqrt(p2 * (1.0 - c) * (1.0 + c))
    return dict(tau=tau, shat=shat, Y=Y, ymax=ymax, x1=x1, x2=x2, c=c, cmax=cmax, pt=pt, pstar=np.sqrt(p2))


class Physics:
    """Tables, interpolators and weights from the standalone script."""

    def __init__(self, st: Setup):
        if core.TABLES is None:
            sys.exit("Tables missing in standalone_qqbar_bbbar_xsec.py: run dump_standalone_tables.py first.")
        self.st = st
        self.pdf = core.TablePDF(core.TABLES)
        self.alpha = core.TableAlphaS(core.TABLES)

    def weights(self, u: np.ndarray) -> np.ndarray:
        """Per-flavour weights (5, N) in pb; their average over the cube is sigma."""
        st = self.st
        return core.born_weights(u, st.ecm, st.mb, st.ptmin, st.ptmax, self.pdf, self.alpha)

    def sobol(self, power: int, seed_offset: int = 0) -> np.ndarray:
        return qmc.Sobol(d=3, scramble=True, seed=self.st.seed + seed_offset).random_base2(power)


# --- the event -------------------------------------------------------------------
@dataclass
class Event:
    """Hard-process kinematics of one q qbar -> b bbar event (GeV)."""
    ebeam: float
    pa: np.ndarray            # incoming parton from +z (px, py, pz, E)
    pb: np.ndarray            # incoming parton from -z
    pq: np.ndarray            # outgoing b
    pqbar: np.ndarray         # outgoing bbar
    name_a: str = "ubar"
    name_b: str = "u"

    @staticmethod
    def rapidity(p: np.ndarray) -> float:
        return 0.5 * math.log((p[3] + p[2]) / (p[3] - p[2]))

    def __post_init__(self):
        tot = self.pa + self.pb
        self.shat = float(tot[3] ** 2 - tot[0] ** 2 - tot[1] ** 2 - tot[2] ** 2)
        self.Y = self.rapidity(tot)
        self.x1 = float(self.pa[3] / self.ebeam)
        self.x2 = float(self.pb[3] / self.ebeam)
        self.mb = float(math.sqrt(max(self.pq[3] ** 2 - np.dot(self.pq[:3], self.pq[:3]), 0.0)))
        self.pt = float(math.hypot(self.pq[0], self.pq[1]))
        self.phi = float(math.atan2(self.pq[1], self.pq[0]))
        self.mt = math.sqrt(self.pt ** 2 + self.mb ** 2)
        self.pstar = math.sqrt(max(self.shat / 4.0 - self.mb ** 2, 0.0))
        self.y_q, self.y_qbar = self.rapidity(self.pq), self.rapidity(self.pqbar)
        self.ystar_q, self.ystar_qbar = self.y_q - self.Y, self.y_qbar - self.Y   # CM-frame rapidities
        self.costheta = self.mt * math.sinh(self.ystar_q) / self.pstar             # cos theta* of the b
        self.tau = self.shat / (2.0 * self.ebeam) ** 2

    def outgoing(self, yboost: float):
        """Momenta of b and bbar after a boost by rapidity yboost from the CM frame."""
        out = []
        for ystar, sign in ((self.ystar_q, 1.0), (self.ystar_qbar, -1.0)):
            px, py = sign * self.pt * math.cos(self.phi), sign * self.pt * math.sin(self.phi)
            out.append(np.array([px, py, self.mt * math.sinh(ystar + yboost)]))
        return out

    def incoming(self, yboost: float):
        """pz of the two massless incoming partons after the boost (CM frame: +-sqrt(shat)/2)."""
        e = math.sqrt(self.shat) / 2.0
        return e * math.exp(yboost), -e * math.exp(-yboost)


def read_event(path: Path) -> Optional[Event]:
    """Parse a PYTHIA hard-process listing or an LHE <event> block (like plot_journey_ME.py)."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    incoming, outgoing, beams, in_lhe = [], [], [], False
    for line in lines:
        t = line.split()
        if not t:
            continue
        if t[0] == "<event>":
            in_lhe = True
            continue
        if t[0] == "</event>":
            in_lhe = False
            continue
        try:
            if in_lhe and len(t) >= 11:
                pid, status = int(t[0]), int(t[1])
                p = np.array([float(v) for v in t[6:10]])
            elif len(t) >= 13 and t[0].isdigit():
                pid, status = int(t[1]), int(t[3])
                p = np.array([float(v) for v in t[-5:-1]])
            else:
                continue
        except ValueError:
            continue
        if status in (-12,) or (in_lhe and status == -9):
            beams.append(p)
        elif status in (-21, -1):
            incoming.append((pid, p))
        elif status in (23, 1):
            outgoing.append((pid, p))
    if len(incoming) != 2 or len(outgoing) != 2:
        return None
    ebeam = beams[0][3] if beams else max(abs(p[2]) for _, p in incoming)
    (pid_a, pa), (pid_b, pb) = sorted(incoming, key=lambda ip: -ip[1][2])
    pq = next(p for pid, p in outgoing if pid > 0)
    pqbar = next(p for pid, p in outgoing if pid < 0)
    names = {1: "d", 2: "u", 3: "s", 4: "c", 5: "b"}
    name = lambda pid: names.get(abs(pid), str(pid)) + ("bar" if pid < 0 else "")  # noqa: E731
    return Event(ebeam, pa, pb, pq, pqbar, name(pid_a), name(pid_b))


def default_event() -> Event:
    """The event of example/ME/event_ME.lhe, in case the file is not found."""
    return Event(6800.0,
                 np.array([0.0, 0.0, 14.785, 14.785]), np.array([0.0, 0.0, -2652.934, 2652.934]),
                 np.array([-62.071, -94.690, -225.148, 252.059]), np.array([62.071, 94.690, -2413.001, 2415.661]))


# --- drawing helpers --------------------------------------------------------------
def style_3d(ax, xlabel, ylabel, zlabel):
    ax.set_xlabel(xlabel, color=TEXT, labelpad=8)
    ax.set_ylabel(ylabel, color=TEXT, labelpad=8)
    ax.set_zlabel(zlabel, color=TEXT, labelpad=8)
    ax.tick_params(colors=TEXT2, labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#fbfaf7")
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"]["color"] = GRID
    ax.set_facecolor("white")


def chrome(fig, scene_no: int, title: str, caption: str):
    fig.text(0.015, 0.955, title, fontsize=15, color=TEXT, weight="bold", ha="left", va="center")
    fig.text(0.985, 0.955, f"scene {scene_no}/4", fontsize=10, color=TEXT2, ha="right", va="center")
    fig.text(0.5, 0.035, caption, fontsize=11, color=TEXT, ha="center", va="center", wrap=True)


def draw_tube(ax, radius: float, zlo: float, zhi: float, color="#9a9890", alpha=0.13):
    """Translucent cylinder of given radius around the z axis: the pTHat cut."""
    phi = np.linspace(0, 2 * np.pi, 48)
    z = np.array([zlo, zhi])
    P, Z = np.meshgrid(phi, z)
    ax.plot_surface(radius * np.cos(P), radius * np.sin(P), Z, color=color, alpha=alpha,
                    linewidth=0, edgecolor="none", antialiased=False, shade=False)
    for zz in (zlo, zhi):
        ax.plot(radius * np.cos(phi), radius * np.sin(phi), zz, color=color, lw=0.8, alpha=0.7)
    ax.plot(radius * np.cos(phi), radius * np.sin(phi), 0.0, color=color, lw=0.8, ls="--", alpha=0.9)


def arrow3d(ax, tail, head, color, lw=2.2, ms=9, alpha=1.0, label=None, dashed_projection=False):
    """A momentum vector: a line from tail to head with a ball at the head."""
    tail, head = np.asarray(tail, float), np.asarray(head, float)
    ax.plot([tail[0], head[0]], [tail[1], head[1]], [tail[2], head[2]], color=color, lw=lw, alpha=alpha,
            solid_capstyle="round")
    ax.plot([head[0]], [head[1]], [head[2]], marker="o", ms=ms, color=color, alpha=alpha, mec="white", mew=0.8)
    if dashed_projection:                         # tip -> transverse plane -> axis, to show pT
        ax.plot([head[0], head[0]], [head[1], head[1]], [head[2], 0.0], color=color, lw=0.8, ls=":", alpha=0.8)
        ax.plot([0.0, head[0]], [0.0, head[1]], [0.0, 0.0], color=color, lw=0.8, ls=":", alpha=0.8)
    if label:
        ax.text(head[0], head[1], head[2], "  " + label, color=color, fontsize=10, weight="bold")


# --- scenes -----------------------------------------------------------------------
class Scene:
    """A scene is a number of frames and a draw(fig, k) method for frame k."""
    number = 0
    title = ""
    nframes = 1

    def draw(self, fig, k: int) -> None:
        raise NotImplementedError

    def keyframe(self) -> int:
        """Representative frame for the stills."""
        return self.nframes // 2


class CollisionScene(Scene):
    number, title = 1, "The hard process in three dimensions: partonic CM frame, then the lab frame"

    def __init__(self, ev: Event, st: Setup):
        self.ev, self.st = ev, st
        # phases (frame indices): partons fly in, quarks fly out, boost, hold
        self.k_in, self.k_out, self.k_boost, self.k_hold = 45, 95, 175, 210
        self.nframes = self.k_hold

    def keyframe(self):
        return self.k_boost - 25

    def draw(self, fig, k):
        ev, st = self.ev, self.st
        t_in, t_out, t_boost = ramp(k, 0, self.k_in), ramp(k, self.k_in, self.k_out), ramp(k, self.k_out, self.k_boost)
        yboost = ev.Y * smooth(t_boost)
        outgoing = ev.outgoing(yboost)
        pz_a, pz_b = ev.incoming(yboost)
        R = 1.35 * max(ev.pt, st.ptmin, 1.0)
        zlim_cm = 1.4 * ev.pstar
        zlim = max(zlim_cm, 1.12 * max(abs(v[2]) for v in outgoing), 1.05 * max(abs(pz_a), abs(pz_b)) * t_boost)

        ax = fig.add_axes([0.02, 0.11, 0.66, 0.80], projection="3d")
        ax.computed_zorder = False
        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-zlim, zlim)
        ax.set_box_aspect((1.0, 1.0, 2.1))
        ax.view_init(elev=18, azim=-58 + 0.12 * k)
        style_3d(ax, "$p_x$ [GeV]", "$p_y$ [GeV]", "$p_z$ [GeV]  (beam axis)")

        # beam axis and the pT >= ptmin tube; the incoming partons are labelled at the axis ends
        tex_a = "$\\bar{" + ev.name_a[0] + "}$" if "bar" in ev.name_a else f"${ev.name_a}$"
        tex_b = "$\\bar{" + ev.name_b[0] + "}$" if "bar" in ev.name_b else f"${ev.name_b}$"
        ax.plot([0, 0], [0, 0], [-zlim, zlim], color=TEXT2, lw=0.9, ls="--", alpha=0.7)
        ax.text(0, 0, zlim, f"  $p$ ($+z$): {tex_a}, $x_1={ev.x1:.1e}$, $p_z={pz_a:+.1f}$ GeV",
                color=COL_UBAR, fontsize=9)
        ax.text(0, 0, -zlim, f"  $p$ ($-z$): {tex_b}, $x_2={ev.x2:.3f}$, $p_z={pz_b:+.1f}$ GeV",
                color=COL_U, fontsize=9, va="top")
        draw_tube(ax, st.ptmin, -zlim, zlim)
        ax.text(-st.ptmin * 0.75, -st.ptmin * 0.75, 0.0, f"$\\hat p_T \\geq {st.ptmin:g}$ GeV  ",
                color="#6b6a66", fontsize=9, ha="right")

        # incoming partons (massless, along +-z)
        if k < self.k_in:
            za, zb = zlim * (1.0 - 0.97 * smooth(t_in)), -zlim * (1.0 - 0.97 * smooth(t_in))
            arrow3d(ax, (0, 0, zlim), (0, 0, za), COL_UBAR, lw=2.0, ms=7, label=tex_a)
            arrow3d(ax, (0, 0, -zlim), (0, 0, zb), COL_U, lw=2.0, ms=7, label=tex_b)
        else:
            arrow3d(ax, (0, 0, max(pz_a, 0.0) + 1e-9), (0, 0, 0), COL_UBAR, lw=1.6, ms=0, alpha=0.55)
            arrow3d(ax, (0, 0, min(pz_b, 0.0) - 1e-9), (0, 0, 0), COL_U, lw=1.6, ms=0, alpha=0.55)

        # outgoing b and bbar
        if k >= self.k_in:
            g = smooth(t_out)
            for vec, col, lab in ((outgoing[0], COL_B, "b"), (outgoing[1], COL_BBAR, r"\bar b")):
                head = g * vec
                pz_txt = f"  $p_z={vec[2]:+.0f}$" if t_boost > 0 else ""
                arrow3d(ax, (0, 0, 0), head, col, label=f"${lab}$: $p_T={ev.pt:.1f}$" + pz_txt,
                        dashed_projection=(g > 0.99))
            if t_boost == 0 and g > 0.99:            # angle theta* to the beam in the CM frame
                th = math.acos(ev.costheta)
                arc = np.linspace(0, th, 20)
                r = 0.55 * ev.pstar
                ax.plot(r * np.sin(arc) * math.cos(ev.phi), r * np.sin(arc) * math.sin(ev.phi), r * np.cos(arc),
                        color=COL_B, lw=1.0)
                ax.text(r * math.sin(th / 2) * math.cos(ev.phi) * 1.15, r * math.sin(th / 2) * math.sin(ev.phi) * 1.15,
                        r * math.cos(th / 2) * 1.15, r"$\theta^*$", color=COL_B, fontsize=11)

        # numbers panel
        boost_line = (f"boost so far  = {yboost:+.2f}" if 0 < t_boost < 1 else
                      ("frame: partonic CM (Y = 0)" if t_boost == 0 else f"frame: lab (boosted by Y = {ev.Y:+.2f})"))
        y_q, y_qbar = ev.ystar_q + yboost, ev.ystar_qbar + yboost
        panel = "\n".join([
            "event_ME.lhe  (process 124)",
            "",
            f"{ev.name_a:<5s} from +z   x1 = {ev.x1:.3e}",
            f"{ev.name_b:<5s} from -z   x2 = {ev.x2:.4f}",
            f"sqrt(shat) = sqrt(x1 x2 s) = {math.sqrt(ev.shat):5.1f} GeV",
            f"tau = shat/s          = {ev.tau:.2e}",
            f"p*  = sqrt(shat/4-mb^2) = {ev.pstar:5.1f} GeV",
            f"cos theta*            = {ev.costheta:+.3f}",
            f"pT  = p* sin theta*   = {ev.pt:6.2f} GeV",
            f"mT  = sqrt(pT^2+mb^2) = {ev.mt:6.2f} GeV",
            f"pair rapidity Y       = {ev.Y:+.2f}",
            "",
            boost_line,
            f"y(b)    = {y_q:+.2f}    pz = {outgoing[0][2]:+8.1f}",
            f"y(bbar) = {y_qbar:+.2f}    pz = {outgoing[1][2]:+8.1f}",
            f"pz = mT sinh(y),  pT unchanged",
        ])
        fig.text(0.70, 0.88, panel, family="monospace", fontsize=9.5, color=TEXT, va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.6", fc="#fbfaf7", ec=GRID))

        if k < self.k_in:
            cap = (f"Partonic centre-of-mass frame: {tex_a} ($x_1 = {ev.x1:.1e}$) and {tex_b} ($x_2 = {ev.x2:.2f}$) "
                   f"collide head on with $\\sqrt{{\\hat s}} = \\sqrt{{x_1 x_2 s}} = {math.sqrt(ev.shat):.0f}$ GeV.")
        elif t_boost == 0:
            cap = (f"$b$ and $\\bar b$ leave back to back with $p^* = {ev.pstar:.1f}$ GeV at $\\cos\\theta^* = {ev.costheta:.2f}$. "
                   f"Their $p_T = p^*\\sin\\theta^* = {ev.pt:.1f}$ GeV must clear the {st.ptmin:g} GeV tube of the generator cut.")
        elif t_boost < 1:
            cap = (f"Boost along the beam by the pair rapidity $Y = {ev.Y:+.2f}$: the tips slide along the tube "
                   "($p_T$ is boost invariant, $p_z = m_T\\sinh y$ is not) and the pair moves far forward.")
        else:
            cap = (f"Lab frame: $p_z(\\bar b) = {outgoing[1][2]:+.0f}$ GeV, $p_z(b) = {outgoing[0][2]:+.0f}$ GeV, "
                   f"both with $p_T = {ev.pt:.1f}$ GeV. These are the momenta listed in event_ME.lhe.")
        chrome(fig, self.number, self.title, cap)


class CubeScene(Scene):
    number, title = 2, "What is integrated: Sobol points in the unit cube, then in physical variables"

    def __init__(self, phys: Physics, power: int = 11):
        self.phys, self.st = phys, phys.st
        self.u = phys.sobol(power, seed_offset=0)
        self.w = phys.weights(self.u).sum(axis=0)
        self.logw = np.log10(np.clip(self.w, 1e-300, None))
        good = self.w > 0
        self.vmax = self.logw[good].max()
        self.vmin = max(np.percentile(self.logw[good], 2), self.vmax - 6.0)     # at most six decades
        v = map_cube(self.u, self.st)
        l10 = math.log10(self.st.taumin)
        # physical coordinates normalised to [0, 1] for the morph
        self.phys_xyz = np.column_stack([1.0 - np.log10(v["x1"]) / l10, 1.0 - np.log10(v["x2"]) / l10, v["c"]])
        self.k_reveal, self.k_rotate, self.k_morph, self.k_hold = 45, 135, 205, 240
        self.nframes = self.k_hold

    def keyframe(self):
        return self.k_morph + 10

    def draw(self, fig, k):
        st = self.st
        n = len(self.u)
        n_show = n if k >= self.k_reveal else int(2 ** (4 + (math.log2(n) - 4) * ramp(k, 0, self.k_reveal)))
        m = smooth(ramp(k, self.k_rotate, self.k_morph))
        xyz = (1.0 - m) * self.u[:n_show] + m * self.phys_xyz[:n_show]

        ax = fig.add_axes([0.02, 0.05, 0.72, 0.88], projection="3d")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_box_aspect((1, 1, 0.9))
        ax.view_init(elev=22, azim=-55 + 360.0 * ramp(k, self.k_reveal, self.k_morph) + 0.1 * k)
        if m < 0.5:
            style_3d(ax, "$u_1$  ($\\ln\\tau$)", "$u_2$  ($Y$)", "$u_3$  ($|\\cos\\theta|$)")
        else:
            style_3d(ax, "$\\log_{10} x_1$", "$\\log_{10} x_2$", "$|\\cos\\theta|$")
        l10 = math.log10(st.taumin)
        if m > 0.5:
            fmt = FuncFormatter(lambda v, pos: f"{round(l10 * (1.0 - v), 1) + 0.0:.1f}")
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
        if 0.05 < m < 0.95:
            ax.set_xticklabels([]); ax.set_yticklabels([])

        sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=self.logw[:n_show], cmap=CMAP, vmin=self.vmin, vmax=self.vmax,
                        s=9, alpha=0.85, linewidths=0, depthshade=False)
        if m > 0.5:                                    # threshold line x1 x2 = tau_min on the floor
            tline = np.linspace(0, 1, 50)
            ax.plot(tline, 1.0 - tline, 0.0, color=COL_BBAR, lw=1.6)
            ax.text(0.5, 0.5, 0.0, f"  $x_1 x_2 = \\tau_{{\\min}} = {st.taumin:.1e}$", color=COL_BBAR, fontsize=9)
            ax.plot([0.5, 1.0], [0.5, 1.0], [0.0, 0.0], color=TEXT2, lw=1.0, ls="--")
            ax.text(1.0, 1.0, 0.0, " $Y=0$ ($x_1=x_2$)", color=TEXT2, fontsize=9)
        cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.08, location="right")
        cb.set_label(r"$\log_{10}\ \sum_q w_q$  [pb]", color=TEXT)
        cb.ax.tick_params(colors=TEXT2, labelsize=8)

        panel = "\n".join([
            f"Sobol points: {n_show:5d} of {n}",
            f"seed {st.seed}, scrambled",
            "",
            "u1 -> ln tau in [ln tau_min, 0]",
            "  tau_min = 4(mb^2+ptmin^2)/s",
            f"          = {st.taumin:.2e}",
            "u2 -> Y in [-Ymax, Ymax]",
            "  Ymax = -ln(tau)/2",
            "u3 -> |cos theta| in [0, c_max]",
            "  c_max = sqrt(1 - ptmin^2/p*^2)",
            "",
            "x1 = sqrt(tau) e^{+Y}",
            "x2 = sqrt(tau) e^{-Y}",
            "",
            "w_q = ln(1/tau_min) 2Ymax 2c_max",
            "    x dsigma/dcos",
            "    x [x1f_q x2f_qbar + (q<->qbar)]",
            "",
            f"<sum_q w_q> here: {self.w.mean() / 1000:.3f} nb",
            f"2^{st.power} points:  {SOBOL1_NB:.5f} nb",
        ])
        fig.text(0.745, 0.87, panel, family="monospace", fontsize=8.8, color=TEXT, va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.6", fc="#fbfaf7", ec=GRID))

        if k < self.k_reveal:
            cap = ("born_weights() samples the unit cube $(u_1,u_2,u_3)$ with a Sobol sequence, which fills the cube "
                   "evenly at every power of two. The colour is $\\log_{10}$ of the weight $\\sum_q w_q$.")
        elif k < self.k_rotate:
            cap = ("The weight spans orders of magnitude: largest near $u_1 = 0$ (threshold, small $\\tau$), "
                   "smallest at the $u_2$ edges ($|Y| \\to Y_{\\max}$, where one $x \\to 1$ and the PDF vanishes).")
        elif m < 1:
            cap = ("The same points in physical variables: $(\\ln\\tau, Y)$ become $(\\log_{10}x_1, \\log_{10}x_2)$ and the "
                   "cube becomes the wedge $x_1 x_2 \\geq \\tau_{\\min}$, $|\\cos\\theta| \\leq c_{\\max}(\\tau)$.")
        else:
            cap = ("Only the region above the threshold line $x_1 x_2 = \\tau_{\\min}$ is kinematically allowed; near it "
                   "$c_{\\max} \\to 0$. The stretch of this map is the Jacobian carried in each weight.")
        chrome(fig, self.number, self.title, cap)


class SurfaceScene(Scene):
    number, title = 3, r"The integrand over the $(x_1, x_2)$ plane: its volume is the cross section"

    def __init__(self, phys: Physics, ev: Event, n1: int = 120, n2: int = 120, nc: int = 32):
        self.phys, self.st, self.ev = phys, phys.st, ev
        self.nc = nc
        st = self.st
        # cell midpoints in (u1, u2); u1 stops short of 1 where the triangle closes to a point
        u1 = (np.arange(n1) + 0.5) / n1 * 0.985
        u2 = (np.arange(n2) + 0.5) / n2
        uc = (np.arange(nc) + 0.5) / nc
        U1, U2, UC = np.meshgrid(u1, u2, uc, indexing="ij")
        pts = np.column_stack([U1.ravel(), U2.ravel(), UC.ravel()])
        w = phys.weights(pts).sum(axis=0).reshape(n1, n2, nc)
        h_cube = w.mean(axis=2)                                    # pb per unit (u1,u2) area
        v = map_cube(np.column_stack([U1[:, :, 0].ravel(), U2[:, :, 0].ravel(), np.zeros(n1 * n2)]), st)
        jac = math.log(1.0 / st.taumin) * 2.0 * v["ymax"]           # d(ln tau) dY / (du1 du2)
        self.X = np.log10(v["x1"]).reshape(n1, n2)
        self.Yc = np.log10(v["x2"]).reshape(n1, n2)
        self.Z = (h_cube / jac.reshape(n1, n2)) * math.log(10) ** 2 / 1000.0   # nb per (log10 x1)(log10 x2)
        self.sigma_nb = float(h_cube.mean() * 0.985 / 1000.0)      # Riemann sum over the cube in nb
        self.k_grow, self.k_rotate, self.k_hold = 50, 175, 200
        self.nframes = self.k_hold

    def keyframe(self):
        return self.k_grow + 5

    def draw(self, fig, k):
        st, ev = self.st, self.ev
        g = smooth(ramp(k, 0, self.k_grow))
        zmax = self.Z.max() * 1.05
        l10 = math.log10(st.taumin)

        ax = fig.add_axes([0.02, 0.07, 0.72, 0.86], projection="3d")
        ax.set_xlim(l10, 0); ax.set_ylim(l10, 0); ax.set_zlim(0, zmax)
        ax.set_box_aspect((1, 1, 0.75))
        # look from the small-x corner at the threshold edge; swing +-55 degrees rather than circle behind the dome
        swing = 55.0 * math.sin(2.0 * math.pi * ramp(k, self.k_grow, self.k_rotate))
        ax.view_init(elev=27 + 6.0 * math.sin(math.pi * ramp(k, self.k_grow, self.k_rotate)), azim=-135 + swing)
        style_3d(ax, "$\\log_{10} x_1$", "$\\log_{10} x_2$",
                 "$\\mathrm{d}\\sigma / (\\mathrm{d}\\log_{10}x_1\\, \\mathrm{d}\\log_{10}x_2)$  [nb]")
        ax.zaxis.labelpad = 12
        # manual draw order: floor, edges, surface, then the annotation on top of everything
        ax.computed_zorder = False
        ax.contourf(self.X, self.Yc, self.Z, levels=14, cmap=CMAP, alpha=0.35, zdir="z", offset=0)
        t = np.linspace(l10, 0, 50)                       # threshold edge and the Y = 0 diagonal on the floor
        ax.plot(t, l10 - t, 0, color=COL_BBAR, lw=1.6)
        ax.text(l10 * 0.5, l10 * 0.5, 0, f"  $x_1 x_2 = \\tau_{{\\min}}$ ($\\hat p_T = {st.ptmin:g}$ GeV, $\\cos\\theta=0$)",
                color=COL_BBAR, fontsize=9)
        ax.plot([l10 / 2, 0], [l10 / 2, 0], [0, 0], color=TEXT2, lw=1.0, ls="--")
        ax.text(0, 0, 0, " $Y=0$", color=TEXT2, fontsize=9)
        norm = colors.Normalize(0, self.Z.max())
        ax.plot_surface(self.X, self.Yc, g * self.Z, facecolors=matplotlib.colormaps[CMAP](norm(self.Z)),
                        rstride=1, cstride=1, linewidth=0, edgecolor="none", antialiased=False, shade=False)
        # our event: vertical stem up to the surface
        if g > 0.5:
            i = np.abs(self.X[:, 0] + self.Yc[:, 0] - math.log10(ev.tau)).argmin()
            j = np.abs(self.X[i] - math.log10(ev.x1)).argmin()
            zev = self.Z[i, j] * g
            ax.plot([math.log10(ev.x1)] * 2, [math.log10(ev.x2)] * 2, [0, zev], color=COL_B, lw=1.8)
            ax.plot([math.log10(ev.x1)], [math.log10(ev.x2)], [zev], marker="o", ms=8, color=COL_B, mec="white")
            ax.text(math.log10(ev.x1), math.log10(ev.x2), zev, f"  our event\n  $x_1={ev.x1:.1e}$, $x_2={ev.x2:.2f}$, $Y={ev.Y:+.1f}$",
                    color=COL_B, fontsize=9, weight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

        ipk, jpk = np.unravel_index(self.Z.argmax(), self.Z.shape)
        panel = "\n".join([
            "surface height:",
            "  integrand with cos theta integrated",
            "  = dsigma/(dlog10 x1 dlog10 x2) [nb]",
            "",
            f"grid: {self.Z.shape[0]} x {self.Z.shape[1]} cells in (u1,u2)",
            f"      {self.nc} midpoints in u3",
            "",
            f"peak {self.Z.max():.4f} nb at",
            f"  x1 = {10 ** self.X[ipk, jpk]:.1e}, x2 = {10 ** self.Yc[ipk, jpk]:.1e}",
            f"  (Y = 0, just above threshold)",
            f"volume on this grid: {self.sigma_nb:.4f} nb",
            f"  -> {SOBOL1_NB:.4f} nb as the grid is refined",
            f"Sobol 2^16 (script):  {SOBOL1_NB:.5f} nb",
            "",
            "front edge: x1 x2 = tau_min, c_max -> 0",
            "flanks: |Y| -> Ymax, one x -> 1, PDF -> 0",
            "back:  tau -> 1, PDFs and 1/shat small",
        ])
        fig.text(0.745, 0.87, panel, family="monospace", fontsize=8.8, color=TEXT, va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.6", fc="#fbfaf7", ec=GRID))

        if g < 1:
            cap = ("Integrate out $\\cos\\theta$ at every $(x_1, x_2)$: the height is "
                   "$\\mathrm{d}\\sigma/(\\mathrm{d}\\log_{10}x_1\\,\\mathrm{d}\\log_{10}x_2)$, whose volume over the triangle "
                   "is the cross section.")
        elif k < self.k_grow + 60:
            cap = ("The dome sits just above the threshold $x_1 x_2 = \\tau_{\\min}$ at $Y = 0$: the PDFs grow towards "
                   "small $x$ and $\\hat\\sigma \\propto 1/\\hat s$, so most of the rate is just above the $p_T$ cut.")
        elif k < self.k_rotate:
            cap = ("The density falls along the threshold edge as $|Y|$ grows (one $x$ becomes large). Our event "
                   "($x_1 = 2.2\\times10^{-3}$, $x_2 = 0.39$, $Y = -2.6$) sits far out on that flank.")
        else:
            cap = (f"Volume under the surface on this grid: {self.sigma_nb:.4f} nb, tending to {SOBOL1_NB:.4f} nb as the grid is "
                   f"refined. The Sobol pass of the standalone script gives {SOBOL1_NB:.5f} nb, PYTHIA {PYTHIA_NB} $\\pm$ {PYTHIA_ERR_NB} nb.")
        chrome(fig, self.number, self.title, cap)


class ConvergeScene(Scene):
    number, title = 4, "Averaging over the cube: one Sobol pass converging to the cross section"

    def __init__(self, phys: Physics, show_cap: int = 16384):
        self.phys, self.st = phys, phys.st
        self.u = phys.sobol(self.st.power, seed_offset=0)
        self.w = phys.weights(self.u)                                  # (5, N)
        n = self.w.shape[1]
        self.counts = np.arange(1, n + 1)
        self.running = np.cumsum(self.w, axis=1) / self.counts / 1000.0   # nb, per flavour
        self.total = self.running.sum(axis=0)
        self.logw = np.log10(np.clip(self.w.sum(axis=0), 1e-300, None))
        good = self.w.sum(axis=0) > 0
        self.vmax = self.logw[good].max()
        self.vmin = max(np.percentile(self.logw[good], 2), self.vmax - 6.0)
        self.show_cap = min(show_cap, n)
        self.nframes = 170
        self.n_of_frame = [int(round(2 ** (4 + (math.log2(n) - 4) * ramp(k, 0, self.nframes - 15)))) for k in range(self.nframes)]

    def keyframe(self):
        return self.nframes - 30

    def draw(self, fig, k):
        st = self.st
        n_now = self.n_of_frame[k]
        n_show = min(n_now, self.show_cap)
        ax = fig.add_axes([0.0, 0.06, 0.56, 0.86], projection="3d")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_box_aspect((1, 1, 0.9))
        ax.view_init(elev=22, azim=-55 + 0.12 * k)
        style_3d(ax, "$u_1$  ($\\ln\\tau$)", "$u_2$  ($Y$)", "$u_3$  ($|\\cos\\theta|$)")
        ax.scatter(self.u[:n_show, 0], self.u[:n_show, 1], self.u[:n_show, 2], c=self.logw[:n_show], cmap=CMAP,
                   vmin=self.vmin, vmax=self.vmax, s=max(1.5, 60.0 / math.sqrt(n_show)), alpha=0.8, linewidths=0,
                   depthshade=False)
        shown = f"{n_show} of {n_now} shown" if n_show < n_now else f"{n_now} points"
        ax.set_title(f"first $N = {n_now}$ Sobol points  ({shown})", fontsize=10, color=TEXT2, pad=0)

        # running totals
        ax_top = fig.add_axes([0.63, 0.55, 0.34, 0.33])
        ax_bot = fig.add_axes([0.63, 0.12, 0.34, 0.33])
        x = self.counts[:n_now]
        for a in (ax_top, ax_bot):
            a.set_xscale("log")
            a.set_xlim(8, self.counts[-1] * 1.15)
            a.grid(color=GRID, lw=0.6)
            a.tick_params(colors=TEXT2, labelsize=8)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
        ax_top.axhspan(PYTHIA_NB - PYTHIA_ERR_NB, PYTHIA_NB + PYTHIA_ERR_NB, color="#e6e4de", zorder=0)
        ax_top.axhline(PYTHIA_NB, color=TEXT2, lw=0.8)
        ax_top.axhline(REPEATS8_NB, color=TEXT, lw=0.8, ls="--")
        ax_top.plot(x, self.total[:n_now], color=TEXT, lw=1.8)
        ax_top.plot(x[-1:], self.total[n_now - 1:n_now], marker="o", ms=6, color=TEXT)
        ax_top.set_ylim(0.31, 0.38)
        ax_top.set_ylabel(r"running $\langle \sum_q w_q \rangle$  [nb]", color=TEXT, fontsize=9)
        ax_top.set_title(f"$N = {n_now}$:  estimate $= {self.total[n_now - 1]:.4f}$ nb", fontsize=10, color=TEXT, loc="left")
        ax_top.legend(handles=[Line2D([], [], color=TEXT, lw=1.8, label="this pass"),
                               Line2D([], [], color=TEXT, lw=0.8, ls="--", label=f"8 passes: {REPEATS8_NB} nb"),
                               Line2D([], [], color="#c9c6bc", lw=6, label=f"PYTHIA {PYTHIA_NB} $\\pm$ {PYTHIA_ERR_NB} nb")],
                      fontsize=8, loc="lower right", frameon=False)
        for i, name in enumerate(core.FLAVOUR_NAMES):
            ax_bot.plot(x, self.running[i, :n_now], color=FLAVOUR_COLOURS[name], lw=1.4,
                        label=f"${name}\\bar {name}$: {self.running[i, n_now - 1]:.3f}")
        ax_bot.set_ylim(0, 0.22)
        ax_bot.set_xlabel("number of Sobol points $N$", color=TEXT, fontsize=9)
        ax_bot.set_ylabel("per incoming flavour  [nb]", color=TEXT, fontsize=9)
        ax_bot.legend(fontsize=7.5, loc="upper right", frameon=False, ncol=3, columnspacing=1.0, handlelength=1.5)

        if k < 40:
            cap = ("Every point contributes $\\sum_q w_q$; the cross section is the average over the cube. "
                   "The first few points scatter wildly because the weights span orders of magnitude.")
        elif k < 110:
            cap = ("A Sobol sequence fills the cube evenly at each power of two, so the average settles faster than "
                   "the $1/\\sqrt{N}$ of random points; $u\\bar u$ and $d\\bar d$ dominate.")
        else:
            cap = (f"After $2^{{{st.power}}}$ points this pass gives {self.total[-1]:.4f} nb; eight scrambled passes give "
                   f"{REPEATS8_NB} $\\pm$ 0.00005 nb, PYTHIA's own Monte Carlo {PYTHIA_NB} $\\pm$ {PYTHIA_ERR_NB} nb.")
        chrome(fig, self.number, self.title, cap)


# --- rendering ----------------------------------------------------------------------
SCENE_NAMES = {"1": "collision", "2": "cube", "3": "surface", "4": "converge"}


def build_scenes(names: List[str], phys: Physics, ev: Event) -> List[Scene]:
    out: List[Scene] = []
    for name in names:
        if name == "collision":
            out.append(CollisionScene(ev, phys.st))
        elif name == "cube":
            out.append(CubeScene(phys))
        elif name == "surface":
            out.append(SurfaceScene(phys, ev))
        elif name == "converge":
            out.append(ConvergeScene(phys))
    return out


def frame_list(scenes: List[Scene], step: int):
    """(scene, k) pairs for the whole film, sampling every step-th frame."""
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
                                        metadata={"title": "q qbar -> b bbar, standalone LO integration"})
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
    p.add_argument("-o", "--output", type=Path, default=HERE / "animate_standalone_qqbar_bbbar_xsec.mp4",
                   help=".mp4 (needs ffmpeg) or .gif")
    p.add_argument("--scene", default="all",
                   help="all, or a comma list of collision,cube,surface,converge (or 1,2,3,4)")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--dpi", type=int, default=100, help="100 gives 1280x720")
    p.add_argument("--step", type=int, default=1, help="render every step-th frame (faster, choppier)")
    p.add_argument("--fast", action="store_true", help="preview: --step 4 --dpi 60")
    p.add_argument("--frames-dir", type=Path, default=None, help="write PNG frames here instead of a video")
    p.add_argument("--stills", type=Path, default=None, help="also write one key frame per scene (.pdf or .png)")
    p.add_argument("--no-video", action="store_true", help="skip the video (with --stills or --frames-dir)")
    p.add_argument("--show", action="store_true", help="play in an interactive window instead of writing a file")
    p.add_argument("--event", type=Path, default=HERE.parent / "event_ME.lhe", help="hard-process record for scene 1")
    p.add_argument("--ecm", type=float, default=13600.0)
    p.add_argument("--mb", type=float, default=4.8)
    p.add_argument("--ptmin", type=float, default=100.0)
    p.add_argument("--ptmax", type=float, default=None)
    p.add_argument("--power", type=int, default=16, help="2**power Sobol points in scene 4")
    p.add_argument("--seed", type=int, default=24680)
    args = p.parse_args()
    if args.fast:
        args.step, args.dpi = max(args.step, 4), min(args.dpi, 60)

    names = list(SCENE_NAMES.values()) if args.scene == "all" else \
        [SCENE_NAMES.get(s.strip(), s.strip()) for s in args.scene.split(",")]
    bad = [n for n in names if n not in SCENE_NAMES.values()]
    if bad:
        sys.exit(f"unknown scene(s) {bad}; choose from {list(SCENE_NAMES.values())}")

    st = Setup(args.ecm, args.mb, args.ptmin, args.ptmax, args.power, args.seed)
    phys = Physics(st)
    ev = read_event(args.event) or default_event()
    print(f"tables: {core.TABLES['pdf_set']}/{core.TABLES['member']}; eCM={st.ecm:g}, mb={st.mb:g}, "
          f"pTHat >= {st.ptmin:g} GeV, tau_min = {st.taumin:.3e}")
    print(f"event: {ev.name_a} (x1={ev.x1:.3e}) + {ev.name_b} (x2={ev.x2:.4f}) -> b bbar, sqrt(shat)={math.sqrt(ev.shat):.1f} GeV, "
          f"pT={ev.pt:.2f} GeV, cos theta*={ev.costheta:+.3f}, Y={ev.Y:+.2f}")
    scenes = build_scenes(names, phys, ev)
    for sc in scenes:
        if isinstance(sc, SurfaceScene):
            print(f"surface: Riemann sum {sc.sigma_nb:.4f} nb, peak {sc.Z.max():.3f} nb per unit log10^2 area")
        if isinstance(sc, ConvergeScene):
            print(f"converge: 2^{st.power} points give {sc.total[-1]:.5f} nb")

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
