#!/usr/bin/env python3
"""The plots behind the theory sections of chapter 2, drawn from the standalone shower.

One page per theory (sub)section of the chapter. Every curve is computed with
the classes of standalone_parton_showering.py (AlphaStrong, TablePDF, Shower),
and every marker is a number of the seed-1 shower of our event, read from the
history the script writes (--history) and from its -v log
(standalone_parton_showering.log), so that each figure can be checked line by
line against that log.

  page  1  propagator      the factor 1/(1 - beta cos theta) for a massless and for the b quark,
                           and the number of branchings per octave of pT in our event
  page  2  kernels         the four splitting functions with the z values drawn in our event,
                           and the distribution of z per kind of branching
  page  3  sudakov         the measured no-emission probability of the four radiators against the
                           leading-log formula, and the scale of the first branching
  page  4  fsr             evolution variable against the laboratory pT of the emitted parton for the
                           33 final-state branchings, and the beam-recoil damping factor
  page  5  isr             the backward-evolution integrand P(z) x PDF ratio for both beams, and the
                           infrared regularisation of the initial-state shower
  page  6  interleaving    the ladder of 38 rungs with the thresholds and cut-offs, and the number of
                           competitors in every round (parsed from the -v log)
  page  7  coupling        the second-order coupling with its thresholds against the first-order
                           overestimate, and their ratio (the veto weight)
  page  8  competition     every proposal of every round of our event and the winner (parsed from the -v log)
  page  9  dead cone       the mass correction of the b -> b g kernel and the angular density of soft
                           gluons around a massive quark, with the mother's three emissions
  page 10  runaways        the mother's pT copy by copy, her descendants in (Delta R, pT), and the
                           family pT as a function of the cone radius

    python3 plot_theory_PS.py                    # -> theory_parton_showering.pdf (all pages)
    python3 plot_theory_PS.py --page 3 -o s.png  # one page as PNG
    python3 plot_theory_PS.py --trials 400       # faster survival curves

Needs numpy, scipy and matplotlib (on this Mac: /usr/bin/python3). About a minute
with the default 1500 proposals per radiator.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import standalone_parton_showering as core  # noqa: E402
import plot_journey_PS as journey  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

# --- appearance: the colours of plot_journey_PS.py and the animation ---------------------------------
COL_B, COL_BBAR = "#ff0000", "#0000ff"
COL_ISR, COL_FSR = "#eb6834", "#2f8f5b"
COL_A, COL_BEAM = "#8e5bb5", "#2a78d6"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d6d3ca"
COL_EVENT = "#c0392b"
KIND_COLOURS = {"q->qg": "#2a78d6", "g->gg": "#2f8f5b", "g->qqbar": "#8e5bb5", "ISR": "#eb6834"}
KIND_TEX = {"q->qg": r"$q\to qg$", "g->gg": r"$g\to gg$", "g->qqbar": r"$g\to q\bar q$", "ISR": "initial state"}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "legend.fontsize": 10,
    "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 100,
})

CF, CA, TR = 4.0 / 3.0, 3.0, 0.5
MB, MC = 4.8, 1.5
EVENT_ME = HERE.parent.parent / "ME" / "event_ME.lhe"
EVENT_PS = HERE.parent / "event_PS.lhe"
LOG = HERE / "standalone_parton_showering.log"


# --- the shower of our event, replayed ----------------------------------------------------------------
class Setup:
    """The seed-1 shower of our event, its history, its record and its -v log."""

    def __init__(self, seed=1):
        self.alpha = core.AlphaStrong(0.118, 2)
        self.pdf = core.TablePDF(core.TABLES)
        self.ev0 = core.read_event(EVENT_ME)
        hard = [i for i, q in enumerate(self.ev0.p) if q.status == 23 and q.coloured]
        m2avg = sum(self.ev0.p[i].m ** 2 for i in hard) / len(hard)
        self.pT2max = core.pt(self.ev0.p[hard[0]].p) ** 2 + m2avg
        self.pTmax = math.sqrt(self.pT2max)
        # the shower itself, for its history and final record
        ev = core.read_event(EVENT_ME)
        self.sh = core.Shower(ev, self.alpha, self.pdf, np.random.default_rng(seed))
        self.sh.track = {i: i for i in hard}
        self.sh.run(self.pT2max)
        self.ev = ev
        self.hard = hard
        self.history = self.sh.history
        self.pTmin_fsr = math.sqrt(self.sh.pT2min_fsr)
        self.pTmin_isr = math.sqrt(self.sh.pT2min_isr)
        # the committed record, parsed by the journey plot's reader
        self.rows = journey.parse_listing(EVENT_PS.read_text().splitlines())
        self.final, self.beams, self.inc0, self.inc = journey.classify(self.rows)
        self.log = LOG.read_text().splitlines() if LOG.exists() else []

    def fresh_shower(self, seed):
        ev = core.read_event(EVENT_ME)
        sh = core.Shower(ev, self.alpha, self.pdf, np.random.default_rng(seed))
        return ev, sh

    def first_emission_trials(self, n, seed=12345):
        """n proposals from pTmax for each of the four radiators of the hard event (NaN = no emission)."""
        ev, sh = self.fresh_shower(seed)
        out = {}
        ends = sh.dipole_ends()
        for end in ends:
            key = "b" if ev.p[end[0]].id == 5 else "bbar"
            vals = []
            for _ in range(n):
                t = sh.pT2next_fsr(end, self.pT2max)
                vals.append(math.sqrt(t["pT2"]) if t else float("nan"))
            out[key] = np.array(vals)
        for side, key in ((0, "A"), (1, "B")):
            vals = []
            for _ in range(n):
                t = sh.pT2next_isr(side, self.pT2max)
                vals.append(math.sqrt(t["pT2"]) if t else float("nan"))
            out[key] = np.array(vals)
        return out


def kind_of(h):
    if h["kind"] == "ISR":
        return "ISR"
    return h["branching"]


def row_kin(r):
    pt = math.hypot(r["px"], r["py"])
    eta = math.asinh(r["pz"] / pt) if pt > 0 else 0.0
    return pt, eta, math.atan2(r["py"], r["px"])


def mark(ax, x, label, color=COL_EVENT, y=0.93, ha="left", dx=4, ls="--"):
    ax.axvline(x, color=color, lw=1.2, ls=ls)
    ax.annotate(label, xy=(x, y), xycoords=("data", "axes fraction"), xytext=(dx if ha == "left" else -dx, 0),
                textcoords="offset points", ha=ha, va="top", fontsize=10, color=color)


# --- the -v log, parsed ------------------------------------------------------------------------------------
RE_STEP = re.compile(r"^==== step (\d+): evolve downwards from pT = ([\d.]+) GeV")
RE_FSR = re.compile(r"^  FSR .*->\s+proposes (\S+)\s+pT =\s+([\d.]+)\s+z = ([\d.]+)")
RE_ISR = re.compile(r"^  ISR side (A|B).*->\s+proposes (\S+)\s+pT =\s+([\d.]+)")
RE_SILENT = re.compile(r"^  no emission above the cut-off from (\d+) competitor")
RE_WIN = re.compile(r"^  winner: (FSR|ISR) .* at pT = ([\d.]+) GeV")


def parse_rounds(lines):
    """For every round of the -v log: the scale, every proposal (kind, pT), the silent count and the winner."""
    rounds, cur = [], None
    for line in lines:
        m = RE_STEP.match(line)
        if m:
            cur = dict(step=int(m.group(1)), start=float(m.group(2)), fsr=[], isr=[], silent=0, winner=None)
            rounds.append(cur)
            continue
        if cur is None:
            continue
        m = RE_FSR.match(line)
        if m:
            cur["fsr"].append((m.group(1), float(m.group(2)), float(m.group(3))))
            continue
        m = RE_ISR.match(line)
        if m:
            cur["isr"].append((m.group(1), float(m.group(3))))
            continue
        m = RE_SILENT.match(line)
        if m:
            cur["silent"] = int(m.group(1))
            continue
        m = RE_WIN.match(line)
        if m:
            cur["winner"] = (m.group(1), float(m.group(2)))
    return rounds


# --- page 1: the propagator and the two logarithms ---------------------------------------------------------
def page_propagator(st):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    fig.suptitle(r"Why a quark cannot travel alone: the propagator $1/(1-\beta\cos\theta)$ and the branchings per octave of our event", y=0.995)
    th = np.geomspace(1e-3, math.pi, 600)
    Eb = 252.059
    beta_b = math.sqrt(1 - MB**2 / Eb**2)
    a1.plot(th, 1 / (1 - np.cos(th)), color=TEXT2, lw=1.6, ls="--", label=r"massless quark: $1/(1-\cos\theta)\to2/\theta^2$")
    a1.plot(th, 1 / (1 - beta_b * np.cos(th)), color=COL_B, lw=1.8, label=r"our mother: $E_b=252\,$GeV, $\beta_b=%.5f$" % beta_b)
    th0 = MB / Eb
    a1.axvline(th0, color=COL_B, lw=1, ls=":")
    a1.annotate(r"$\theta_0=m_b/E_b=%.3f$" % th0, xy=(th0, 2e4), xytext=(6, 0), textcoords="offset points", fontsize=10, color=COL_B)
    a1.axhline(1 / (1 - beta_b), color=COL_B, lw=0.8, ls=":")
    a1.text(1.2e-3, 1 / (1 - beta_b) * 1.3, r"plateau $1/(1-\beta_b)=%.0f$: the dead cone" % (1 / (1 - beta_b)), fontsize=10, color=COL_B)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel(r"angle $\theta$ between the gluon and the quark  [rad]")
    a1.set_ylabel(r"$1/(1-\beta\cos\theta)$")
    a1.set_title(r"(a) the collinear singularity, and where the mass stops it")
    a1.legend(loc="upper right")
    # (b) branchings per octave
    pts = np.array([h["pT"] for h in st.history])
    kinds = [kind_of(h) for h in st.history]
    octs = np.log2(st.pTmax / pts)
    edges = np.arange(0, 8.5, 1.0)
    bottom = np.zeros(len(edges) - 1)
    for k in ("ISR", "q->qg", "g->gg", "g->qqbar"):
        sel = np.array([kk == k for kk in kinds])
        h, _ = np.histogram(octs[sel], bins=edges)
        a2.bar(edges[:-1], h, width=1.0, bottom=bottom, align="edge", color=KIND_COLOURS[k], alpha=0.85, lw=0, label=KIND_TEX[k])
        bottom += h
    a2.set_xlabel(r"octaves below $p_{\mathrm{T}}^{\max}$:  $\log_2(p_{\mathrm{T}}^{\max}/p_{\mathrm{T,evol}})$")
    a2.set_ylabel("branchings of our event per octave")
    a2.set_title(r"(b) 38 branchings: the rate per octave grows as the scale falls")
    sec = a2.twinx()
    q = np.geomspace(st.pTmin_fsr, st.pTmax, 200)
    sec.plot(np.log2(st.pTmax / q), [st.alpha.alphaS(v * v) for v in q], color=TEXT, lw=1.4, ls="--")
    sec.set_ylabel(r"$\alpha_{\mathrm{s}}(p_{\mathrm{T}}^2)$ (dashed)")
    sec.set_ylim(0, 1.7)
    sec.grid(False)
    for x, lab in ((math.log2(st.pTmax / MB), r"$m_b$"), (math.log2(st.pTmax / MC), r"$m_c$"), (math.log2(st.pTmax / st.pTmin_fsr), r"$p_{\mathrm{T}}^{\min}$")):
        a2.axvline(x, color=TEXT2, lw=0.8, ls=":")
        a2.text(x, 0.97 * a2.get_ylim()[1] if False else 15.5, lab, ha="center", fontsize=10, color=TEXT2)
    a2.set_ylim(0, 17)
    a2.text(0.03, 0.9, "first 10 branchings: 3 octaves\nlast 18 branchings: < 1 octave", transform=a2.transAxes, fontsize=10, color=TEXT2, va="top")
    a2.legend(loc="center left")
    fig.tight_layout()
    return fig


# --- page 2: the splitting functions -------------------------------------------------------------------------
def page_kernels(st):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    fig.suptitle(r"The four splitting functions, and the $z$ values the shower of our event actually drew", y=0.995)
    z = np.linspace(0.005, 0.995, 400)
    Pqq = CF * (1 + z**2) / (1 - z)
    Pgg = 2 * CA * (1 - z * (1 - z)) ** 2 / (z * (1 - z))
    Pqg = TR * (z**2 + (1 - z) ** 2)
    Pgq = CF * (1 + (1 - z) ** 2) / z
    a1.plot(z, Pqq, color=KIND_COLOURS["q->qg"], lw=1.8, label=r"$P_{qq}=C_F(1+z^2)/(1-z)$")
    a1.plot(z, Pgg, color=KIND_COLOURS["g->gg"], lw=1.8, label=r"$P_{gg}=2C_A(1-z(1-z))^2/(z(1-z))$")
    a1.plot(z, Pqg, color=KIND_COLOURS["g->qqbar"], lw=1.8, label=r"$P_{qg}=T_R(z^2+(1-z)^2)$")
    a1.plot(z, Pgq, color=KIND_COLOURS["ISR"], lw=1.4, ls="--", label=r"$P_{gq}=C_F(1+(1-z)^2)/z$")
    a1.plot(z, 2 * CF / (1 - z), color=TEXT2, lw=1, ls=":", label=r"overestimate $2C_F/(1-z)$")
    for h in st.history:
        k = kind_of(h)
        if k == "ISR":
            continue
        P = {"q->qg": CF * (1 + h["z"] ** 2) / (1 - h["z"]),
             "g->gg": 2 * CA * (1 - h["z"] * (1 - h["z"])) ** 2 / (h["z"] * (1 - h["z"])),
             "g->qqbar": TR * (h["z"] ** 2 + (1 - h["z"]) ** 2)}[k]
        a1.plot(h["z"], P, marker="o", ms=5, color=KIND_COLOURS[k], mec="white", zorder=5)
    for step in (5, 13):
        h = st.history[step - 1]
        a1.annotate("the mother, branching %d:\n$z=%.3f$, $P_{qq}=%.2f$" % (step, h["z"], CF * (1 + h["z"] ** 2) / (1 - h["z"])),
                    xy=(h["z"], CF * (1 + h["z"] ** 2) / (1 - h["z"])), xytext=(-70 if step == 5 else -90, 25), textcoords="offset points",
                    fontsize=9.5, color=COL_B, arrowprops=dict(arrowstyle="-", color=COL_B, lw=0.8))
    a1.set_yscale("log")
    a1.set_ylim(0.1, 400)
    a1.set_xlabel(r"$z$, the momentum fraction kept by the first daughter")
    a1.set_ylabel(r"$P(z)$  (a rate, not a probability)")
    a1.set_title(r"(a) the kernels; dots: the 33 final-state branchings of our event")
    a1.legend(loc="upper center", ncol=2, fontsize=9)
    # (b) z distribution per kind
    bins = np.linspace(0, 1, 11)
    bottom = np.zeros(10)
    for k in ("q->qg", "g->gg", "g->qqbar", "ISR"):
        zs = [h["z"] for h in st.history if kind_of(h) == k]
        h, _ = np.histogram(zs, bins=bins)
        a2.bar(bins[:-1], h, width=0.1, bottom=bottom, align="edge", color=KIND_COLOURS[k], alpha=0.85, lw=0, label="%s (%d)" % (KIND_TEX[k], len(zs)))
        bottom += h
    a2.set_xlabel(r"$z$")
    a2.set_ylabel("branchings of our event")
    a2.set_title(r"(b) soft daughters dominate the gluon splittings")
    a2.legend(loc="upper center")
    fig.tight_layout()
    return fig


# --- page 3: the Sudakov form factor -------------------------------------------------------------------------
def page_sudakov(st, trials):
    tr = st.first_emission_trials(trials)
    grid = np.geomspace(min(st.pTmin_fsr, st.pTmin_isr) * 0.8, st.pTmax, 200)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    fig.suptitle(r"The Sudakov form factor: measured from %d proposals per radiator, against the leading-log formula" % trials, y=0.995)
    cols = {"b": (COL_B, r"$b$ dipole end"), "bbar": (COL_BBAR, r"$\bar b$ dipole end"), "A": (COL_A, r"$+z$ beam, $\bar u$ at $x=2.2\times10^{-3}$"), "B": (COL_BEAM, r"$-z$ beam, $u$ at $x=0.39$")}
    prod = np.ones_like(grid)
    for key in ("b", "bbar", "A", "B"):
        v = np.where(np.isnan(tr[key]), -1.0, tr[key])
        surv = np.array([(v < g).mean() for g in grid])
        prod *= surv
        a1.plot(grid, surv, color=cols[key][0], lw=1.8, label=cols[key][1])
    a1.plot(grid, prod, color=TEXT, lw=1.3, ls="--", label="product: nobody has radiated yet")
    # leading-log estimates for the mother: Q = 378 GeV dipole mass
    Q = 377.9
    fixed = np.exp(-(0.118 * CF / math.pi) * (np.log(Q / grid) ** 2 - math.log(Q / st.pTmax) ** 2))
    a1.plot(grid, fixed, color=COL_B, lw=1.2, ls=":", label=r"LL, fixed $\alpha_{\mathrm{s}}=0.118$ (eq. double log)")
    run = []
    for g in grid:
        if g < st.pTmin_fsr:
            run.append(float("nan"))
            continue
        qs = np.geomspace(g, st.pTmax, 400)
        f = np.array([st.alpha.alphaS(q * q) for q in qs]) * CF / math.pi * 2 * np.log(Q / qs) / qs
        run.append(math.exp(-np.trapz(f, qs)))
    run = np.array(run)
    a1.plot(grid, run, color=COL_B, lw=1.2, ls="-.", label=r"LL, running $\alpha_{\mathrm{s}}$ (down to the cut-off)")
    a1.set_xscale("log")
    a1.set_xlim(st.pTmax * 1.05, grid[0])
    a1.set_ylim(0, 1.02)
    a1.set_xlabel(r"$p_{\mathrm{T}}$  [GeV]   (the evolution runs left to right)")
    a1.set_ylabel(r"$\Delta(p_{\mathrm{T}}^{\max},p_{\mathrm{T}})$: no emission above $p_{\mathrm{T}}$")
    i10 = np.searchsorted(grid, 10.0)
    vb = np.where(np.isnan(tr["b"]), -1.0, tr["b"])
    a1.set_title(r"(a) at $10\,$GeV: measured %.2f, LL fixed %.2f, LL running %.2f" % ((vb < 10).mean(), fixed[i10], run[i10]))
    a1.legend(loc="lower left", fontsize=8.5)
    # (b) who radiates first and at which scale
    arr = np.vstack([np.where(np.isnan(tr[k]), -1.0, tr[k]) for k in ("b", "bbar", "A", "B")])
    winner, winpt = arr.argmax(axis=0), arr.max(axis=0)
    bins = np.geomspace(grid[0], st.pTmax, 26)
    bottom = np.zeros(len(bins) - 1)
    for idx, key in enumerate(("b", "bbar", "A", "B")):
        sel = (winner == idx) & (winpt > 0)
        h, _ = np.histogram(winpt[sel], bins=bins)
        a2.bar(bins[:-1], h, width=np.diff(bins), bottom=bottom, align="edge", color=cols[key][0], alpha=0.85, lw=0,
               label="%s: %.0f%%" % (cols[key][1].split(",")[0], 100 * sel.mean()))
        bottom += h
    a2.set_xscale("log")
    a2.set_xlabel(r"$p_{\mathrm{T}}$ of the first branching (hardest of the four proposals)  [GeV]")
    a2.set_ylabel("trials")
    a2.set_title(r"(b) who radiates first: the $+z$ beam, at small $x$, most often")
    mark(a2, st.history[0]["pT"], "our event: first branching\nat %.1f GeV (ISR, $+z$)" % st.history[0]["pT"], ha="right")
    a2.text(0.03, 0.80, "nobody radiates at all: %.1f%%" % (100 * (winpt <= 0).mean()), transform=a2.transAxes, fontsize=10, color=TEXT2, va="top")
    a2.legend(loc="center left")
    fig.tight_layout()
    return fig


# --- page 4: final-state radiation ------------------------------------------------------------------------------
def page_fsr(st):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    fig.suptitle(r"Final-state radiation: the evolution variable is not the gluon's laboratory $p_{\mathrm{T}}$, and the beam-recoil damping", y=0.995)
    xs, ys, cs = [], [], []
    for h in st.history:
        if h["kind"] != "FSR":
            continue
        em = st.rows[h["new_rows"][1]]
        xs.append(h["pT"])
        ys.append(row_kin(em)[0])
        cs.append(KIND_COLOURS[h["branching"]])
    lim = [0.3, 200]
    a1.plot(lim, lim, color=TEXT2, lw=1, ls=":")
    a1.scatter(xs, ys, c=cs, s=36, edgecolor="white", zorder=5)
    for step, lab in ((4, r"$\bar b$, branching 4"), (5, "the mother, branching 5"), (13, "the mother, branching 13")):
        h = st.history[step - 1]
        em = st.rows[h["new_rows"][1]]
        off = {4: (12, 10), 5: (12, -36), 13: (-125, 30)}[step]
        a1.annotate("%s:\n%.1f $\\to$ %.1f GeV" % (lab, h["pT"], row_kin(em)[0]), xy=(h["pT"], row_kin(em)[0]), xytext=off,
                    textcoords="offset points", fontsize=9.5, color=COL_B if step != 4 else COL_BBAR,
                    arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.8))
    for k in ("q->qg", "g->gg", "g->qqbar"):
        a1.scatter([], [], c=KIND_COLOURS[k], s=36, label=KIND_TEX[k])
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlim(*lim)
    a1.set_ylim(*lim)
    a1.set_xlabel(r"$p_{\mathrm{T,evol}}$ of the branching  [GeV]")
    a1.set_ylabel(r"laboratory $p_{\mathrm{T}}$ of the emitted parton  [GeV]")
    a1.set_title(r"(a) the 33 final-state branchings: a factor 3 either way is common")
    a1.legend(loc="upper left")
    # (b) damping factor
    ptrad = 113.2
    pte = np.geomspace(0.6, 60, 300)
    for z, col, lab in ((0.98, COL_B, r"$z=0.98$ (the round-1 proposal)"), (0.86, "#8e5bb5", r"$z=0.86$"), (0.60, COL_BEAM, r"$z=0.60$ (branching 5)")):
        ma2 = MB**2 + pte**2 / (z * (1 - z))
        w = ptrad * pte / (ptrad * pte + ma2)
        a2.plot(pte, w, color=col, lw=1.8, label=lab)
    a2.plot(5.241, 0.2747, marker="*", ms=14, color=COL_EVENT, mec="white", zorder=5)
    a2.annotate("round 1: $p_{\\mathrm{T,evol}}=5.24$, $z=0.982$,\n$m_a=39.6\\,$GeV: $w_{\\rm damp}=0.275$", xy=(5.241, 0.2747), xytext=(12, 30), textcoords="offset points", fontsize=9.5, color=COL_EVENT)
    a2.set_xscale("log")
    a2.set_xlabel(r"$p_{\mathrm{T,evol}}$  [GeV]")
    a2.set_ylabel(r"$w_{\mathrm{damp}}=p_{\mathrm{T,rad}}\,p_{\mathrm{T,evol}}/(p_{\mathrm{T,rad}}\,p_{\mathrm{T,evol}}+m_a^2)$")
    a2.set_ylim(0, 1.05)
    a2.set_title(r"(b) TimeShower:dampenBeamRecoil for a radiator of $p_{\mathrm{T}}=%.0f\,$GeV" % ptrad)
    a2.text(0.03, 0.08, "a soft emission ($z\\to1$) at a large virtuality\nagainst a beam recoiler is damped the most", transform=a2.transAxes, fontsize=10, color=TEXT2)
    a2.legend(loc="upper left")
    fig.tight_layout()
    return fig


# --- page 5: initial-state radiation ------------------------------------------------------------------------------
def page_isr(st):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    fig.suptitle(r"Backward evolution: the kernel times the PDF ratio for the two beams of our event, and the infrared regularisation", y=0.995)
    q2 = st.pT2max
    z = np.linspace(0.02, 0.985, 300)
    xA, xB = 2.17429e-3, 0.390137
    fA = st.pdf.xf(-2, xA, q2)
    fB = st.pdf.xf(2, xB, q2)
    def ratio(pid_b, x, z, fa):
        return np.array([st.pdf.xf(pid_b, x / zz, q2) / fa if x / zz < 1 else 0.0 for zz in z])
    Pqq = CF * (1 + z**2) / (1 - z)
    Pqg = TR * (z**2 + (1 - z) ** 2)
    a1.plot(z, Pqq * ratio(-2, xA, z, fA), color=COL_A, lw=1.8, label=r"$+z$: $\bar u\leftarrow\bar u$ (emits $g$)")
    a1.plot(z, Pqg * ratio(21, xA, z, fA), color=COL_A, lw=1.8, ls="--", label=r"$+z$: $\bar u\leftarrow g$ (emits $u$)")
    a1.plot(z, Pqq * ratio(2, xB, z, fB), color=COL_BEAM, lw=1.8, label=r"$-z$: $u\leftarrow u$ (emits $g$)")
    a1.plot(z, Pqg * ratio(21, xB, z, fB), color=COL_BEAM, lw=1.8, ls="--", label=r"$-z$: $u\leftarrow g$ (emits $\bar u$)")
    for step, col in ((1, COL_A), (2, COL_BEAM)):
        h = st.history[step - 1]
        a1.axvline(h["z"], color=col, lw=1, ls=":")
        a1.annotate("branching %d:\n$z=%.3f$" % (step, h["z"]), xy=(h["z"], 0.75 if step == 1 else 0.55), xycoords=("data", "axes fraction"), xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=9.5, color=col)
    a1.set_yscale("log")
    a1.set_ylim(1e-2, 2e3)
    a1.set_xlabel(r"$z=x/x'$")
    a1.set_ylabel(r"$P_{ba}(z)\;x'f_b(x')/xf_a(x)$ at $Q^2=(p_{\mathrm{T}}^{\max})^2$")
    a1.set_title(r"(a) small $x$: $g\to q\bar q$; a valence quark: $q\to qg$")
    a1.legend(loc="upper center", ncol=2, fontsize=9)
    # (b) regularisation
    pt = np.geomspace(0.42, 120, 300)
    pt0 = 2.0
    a2.plot(pt, [st.alpha.alphaS(v * v) for v in pt], color=TEXT2, lw=1.4, ls="--", label=r"$\alpha_{\mathrm{s}}(p_{\mathrm{T}}^2)$, final state")
    a2.plot(pt, [st.alpha.alphaS(v * v + pt0**2) for v in pt], color=COL_ISR, lw=1.8, label=r"$\alpha_{\mathrm{s}}(p_{\mathrm{T}}^2+p_{\mathrm{T}0}^2)$, initial state")
    a2.plot(pt, pt**2 / (pt**2 + pt0**2), color=COL_BEAM, lw=1.8, label=r"measure factor $p_{\mathrm{T}}^2/(p_{\mathrm{T}}^2+p_{\mathrm{T}0}^2)$")
    a2.axvline(pt0, color=TEXT2, lw=0.8, ls=":")
    a2.text(pt0 * 1.08, 1.45, r"$p_{\mathrm{T}0}=2\,$GeV", fontsize=10, color=TEXT2)
    a2.axvline(1.0, color=COL_EVENT, lw=1, ls="--")
    a2.annotate(r"at 1 GeV: $\alpha_{\mathrm{s}}$ 0.29 instead of 0.53," "\n" r"measure $\times$ 0.2", xy=(1.0, 1.25), xytext=(-8, 0), textcoords="offset points", ha="right", fontsize=9.5, color=COL_EVENT)
    a2.axvline(st.pTmin_isr, color=TEXT2, lw=0.8, ls=":")
    a2.text(st.pTmin_isr * 1.05, 0.05, r"$p_{\mathrm{T}}^{\min,\mathrm{ISR}}$", fontsize=9.5, color=TEXT2)
    a2.set_xscale("log")
    a2.set_ylim(0, 1.65)
    a2.set_xlabel(r"$p_{\mathrm{T,evol}}$  [GeV]")
    a2.set_title(r"(b) pT0Ref tames coupling and measure near the cut-off")
    a2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


# --- page 6: interleaving and the scales ------------------------------------------------------------------------------
def page_interleaving(st, rounds):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    fig.suptitle(r"One ordered sequence: the ladder of 38 rungs of our event and the competitors of every round", y=0.995)
    steps = np.arange(1, len(st.history) + 1)
    for k in ("ISR", "q->qg", "g->gg", "g->qqbar"):
        sel = [i for i, h in enumerate(st.history) if kind_of(h) == k]
        a1.plot(steps[sel], [st.history[i]["pT"] for i in sel], ls="none", marker="s" if k == "ISR" else "o", ms=6, color=KIND_COLOURS[k], mec="white", label=KIND_TEX[k], zorder=5)
    a1.plot(steps, [h["pT"] for h in st.history], color=TEXT2, lw=0.8, zorder=1)
    for y, lab, col in ((st.pTmax, r"$p_{\mathrm{T}}^{\max}=113.3$", COL_EVENT), (MB, r"$m_b$", TEXT2), (MC, r"$m_c$", TEXT2), (2.0, r"$p_{\mathrm{T}0}$ (ISR)", COL_ISR),
                        (st.pTmin_fsr, r"$p_{\mathrm{T}}^{\min,\mathrm{FSR}}=0.611$", TEXT2), (st.pTmin_isr, r"$p_{\mathrm{T}}^{\min,\mathrm{ISR}}=0.420$", TEXT2)):
        a1.axhline(y, color=col, lw=0.9, ls="--" if col == COL_EVENT else ":")
        a1.text(38.8, y * 1.08, lab, ha="right", fontsize=9, color=col)
    a1.set_yscale("log")
    a1.set_xlim(0, 39.5)
    a1.set_ylim(0.3, 200)
    a1.set_xlabel("branching number (round)")
    a1.set_ylabel(r"$p_{\mathrm{T,evol}}$ of the winning proposal  [GeV]")
    a1.set_title(r"(a) steep at the top (3 octaves in 10 rungs), flat at the bottom (18 rungs in one)")
    a1.legend(loc="lower left", ncol=2, fontsize=9)
    if rounds:
        s = np.array([r["step"] for r in rounds])
        n_fsr = np.array([len(r["fsr"]) for r in rounds])
        n_isr = np.array([len(r["isr"]) for r in rounds])
        n_sil = np.array([r["silent"] for r in rounds])
        a2.bar(s, n_fsr, color=COL_FSR, alpha=0.85, lw=0, label="dipole ends proposing")
        a2.bar(s, n_isr, bottom=n_fsr, color=COL_ISR, alpha=0.85, lw=0, label="beams proposing")
        a2.bar(s, n_sil, bottom=n_fsr + n_isr, color="#cfcdc5", lw=0, label="silent (no emission above the cut-off)")
        a2.set_xlabel("round")
        a2.set_ylabel("competitors in the round")
        tot = n_fsr + n_isr + n_sil
        a2.set_title(r"(b) %d competitors in round 1, %d in round %d" % (tot[0], tot[-1], s[-1]))
        a2.text(0.03, 0.93, "every final-state branching adds\ntwo dipole ends; from the -v log", transform=a2.transAxes, fontsize=10, color=TEXT2, va="top")
        a2.legend(loc="center left", fontsize=9)
    else:
        a2.text(0.5, 0.5, "standalone_parton_showering.log not found", ha="center", transform=a2.transAxes)
    fig.tight_layout()
    return fig


# --- page 7: the coupling ---------------------------------------------------------------------------------------------
def page_coupling(st):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    fig.suptitle(r"The CP5 shower coupling: $\alpha_{\mathrm{s}}(m_Z)=0.118$ at second order with thresholds, and its first-order overestimate", y=0.995)
    q = np.geomspace(0.5, 200, 600)
    a2v = np.array([st.alpha.alphaS(v * v) for v in q])
    a1v = np.array([st.alpha.alphaS_1loop(v * v, st.alpha.nf(v * v)) for v in q])
    a1.plot(q, a2v, color=TEXT, lw=1.9, label="second order with thresholds (AlphaStrong)")
    a1.plot(q, a1v, color=TEXT2, lw=1.3, ls="--", label=r"first order, same $\Lambda_{n_f}$: the overestimate")
    for x, lab in ((MC, r"$m_c$: $n_f$ 3$\to$4"), (MB, r"$m_b$: $n_f$ 4$\to$5"), (91.1876, r"$m_Z$: 0.118")):
        a1.axvline(x, color=TEXT2, lw=0.8, ls=":")
        a1.text(x * 1.06, 1.35, lab, fontsize=9.5, color=TEXT2)
    for v, lab in ((0.611, "cut-off: 1.56"), (1.0, "1 GeV: 0.53"), (5.0, "5 GeV: 0.21"), (10.0, "10: 0.18"), (32.07, "32: 0.14"), (113.3, r"$p_{\mathrm{T}}^{\max}$: 0.114")):
        a1.plot(v, st.alpha.alphaS(v * v), marker="o", ms=5, color=COL_EVENT, mec="white", zorder=5)
        a1.annotate(lab, xy=(v, st.alpha.alphaS(v * v)), xytext=(6, 4), textcoords="offset points", fontsize=9, color=COL_EVENT)
    a1.set_xscale("log")
    a1.set_xlabel(r"$Q$  [GeV]")
    a1.set_ylabel(r"$\alpha_{\mathrm{s}}(Q^2)$")
    a1.set_ylim(0, 1.7)
    a1.set_title(r"(a) $\Lambda_5=0.226$, $\Lambda_4=0.328$, $\Lambda_3=0.382\,$GeV")
    a1.legend(loc="upper right", fontsize=9)
    a2.plot(q, a2v / a1v, color=TEXT, lw=1.9)
    a2.axhline(1, color=TEXT2, lw=0.8, ls=":")
    a2.axvline(st.pTmin_fsr, color=COL_EVENT, lw=1, ls="--")
    a2.annotate("cut-off 0.611 GeV: ratio 1.052,\nthe overestimate fails here", xy=(st.pTmin_fsr, 1.03), xytext=(8, 10), textcoords="offset points", fontsize=9.5, color=COL_EVENT)
    for v in (1.0, 5.2, 34.2):
        r = st.alpha.alphaS(v * v) / st.alpha.alphaS_1loop(v * v, st.alpha.nf(v * v))
        a2.plot(v, r, marker="o", ms=5, color=COL_EVENT, mec="white", zorder=5)
        a2.annotate("%.3g GeV: %.2f" % (v, r), xy=(v, r), xytext=(6, -12), textcoords="offset points", fontsize=9, color=COL_EVENT)
    a2.set_xscale("log")
    a2.set_ylim(0.65, 1.1)
    a2.set_xlabel(r"$Q$  [GeV]")
    a2.set_ylabel(r"$\alpha_{\mathrm{s}}^{(2)}/\alpha_{\mathrm{s}}^{(1)}$: the coupling factor of the veto weight")
    a2.set_title(r"(b) the price of the overestimate: 0.73 to 0.87")
    fig.tight_layout()
    return fig


# --- page 8: the competition, round by round -------------------------------------------------------------------------------
def page_competition(st, rounds):
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.6))
    fig.suptitle(r"Everybody proposes, the hardest wins: every proposal of every round of our event, from the -v log", y=0.995)
    if not rounds:
        ax.text(0.5, 0.5, "standalone_parton_showering.log not found", ha="center", transform=ax.transAxes)
        return fig
    for r in rounds:
        if r["fsr"]:
            ax.plot([r["step"]] * len(r["fsr"]), [p[1] for p in r["fsr"]], ls="none", marker="o", ms=3.5, color=COL_FSR, alpha=0.6)
        if r["isr"]:
            ax.plot([r["step"]] * len(r["isr"]), [p[1] for p in r["isr"]], ls="none", marker="s", ms=4.5, color=COL_ISR, alpha=0.9)
    ax.plot([r["step"] for r in rounds if r["winner"]], [r["winner"][1] for r in rounds if r["winner"]], color=TEXT, lw=1.4, marker="o", ms=6, mfc="white", mec=TEXT, label="the winner: the hardest proposal", zorder=5)
    ax.plot([r["step"] for r in rounds], [r["start"] for r in rounds], color=COL_EVENT, lw=1, ls="--", label="the scale the round starts from (the previous winner)")
    ax.plot([], [], ls="none", marker="o", ms=4, color=COL_FSR, label="proposals of dipole ends (FSR)")
    ax.plot([], [], ls="none", marker="s", ms=5, color=COL_ISR, label="proposals of the two beams (ISR)")
    ax.axhline(st.pTmin_fsr, color=TEXT2, lw=0.8, ls=":")
    ax.text(0.5, st.pTmin_fsr * 1.1, r"$p_{\mathrm{T}}^{\min,\mathrm{FSR}}=0.611\,$GeV", fontsize=9.5, color=TEXT2)
    ax.set_yscale("log")
    ax.set_xlim(0, 40)
    ax.set_ylim(0.3, 200)
    ax.set_xlabel("round")
    ax.set_ylabel(r"proposed $p_{\mathrm{T,evol}}$  [GeV]")
    r1 = rounds[0]
    ax.set_title(r"round 1: %s; %d rounds, %d proposals in all, %d winners" % (
        ", ".join("%.1f" % p[1] for p in r1["fsr"]) + " (FSR), " + ", ".join("%.1f" % p[1] for p in r1["isr"]) + " (ISR) GeV",
        len(rounds), sum(len(r["fsr"]) + len(r["isr"]) for r in rounds), sum(1 for r in rounds if r["winner"])), fontsize=10.5)
    ax.text(0.40, 0.93, "losers are thrown away and propose afresh\nbelow the winner's scale in the next round", transform=ax.transAxes, ha="center", va="top", fontsize=10, color=TEXT2)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


# --- page 9: the dead cone --------------------------------------------------------------------------------------------------
def mother_emissions(st):
    """(step, pT_evol, z, angle between the two daughters) for the mother's own branchings."""
    out = []
    cur = st.hard[1] if st.ev0.p[st.hard[1]].id == 5 else st.hard[0]
    b_rows = set()
    r = st.rows[6]
    chain = [6]
    while r["status"] < 0:
        d1, d2 = r["d1"], r["d2"]
        cands = [st.rows[d] for d in range(d1, max(d1, d2) + 1) if d in st.rows]
        same = [c for c in cands if c["id"] == 5]
        if not same:
            break
        r = same[0]
        chain.append(r["no"])
    for h in st.history:
        if h["kind"] == "FSR" and h["radiator"] == "b" and h["new_rows"][0] in chain:
            d1, d2 = st.rows[h["new_rows"][0]], st.rows[h["new_rows"][1]]
            p1 = np.array([d1["px"], d1["py"], d1["pz"]])
            p2 = np.array([d2["px"], d2["py"], d2["pz"]])
            ang = math.acos(np.dot(p1, p2) / np.linalg.norm(p1) / np.linalg.norm(p2))
            out.append((h["step"], h["pT"], h["z"], ang))
    return out, chain


def page_dead_cone(st):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    fig.suptitle(r"The dead cone of the heavy mother: the mass term of the $b\to bg$ kernel and the angular density of soft gluons", y=0.995)
    pte = np.geomspace(0.6, 60, 400)
    ems, _ = mother_emissions(st)
    for z, col in ((0.60, COL_BEAM), (0.80, "#2f8f5b"), (0.86, "#8e5bb5"), (0.98, COL_B)):
        r = 1 - 2 * MB**2 * z * (1 - z) ** 2 / ((1 + z**2) * pte**2)
        a1.plot(pte, np.clip(r, 0, None), lw=1.8, color=col, label=r"$z=%.2f$" % z)
    for step, pt_, z, ang in ems:
        r = 1 - 2 * MB**2 * z * (1 - z) ** 2 / ((1 + z**2) * pt_**2)
        a1.plot(pt_, r, marker="*", ms=13, color=COL_EVENT, mec="white", zorder=5)
        a1.annotate("branching %d: $p_{\\mathrm{T,evol}}=%.1f$, $z=%.2f$\nkernel reduced by %.1f%%" % (step, pt_, z, 100 * (1 - r)), xy=(pt_, r),
                    xytext=(-20, -70) if step == 5 else (12, -30), textcoords="offset points", ha="right" if step == 5 else "left", fontsize=9, color=COL_EVENT)
    a1.set_xscale("log")
    a1.set_ylim(0, 1.08)
    a1.set_xlabel(r"$p_{\mathrm{T,evol}}$  [GeV]")
    a1.set_ylabel(r"$P_{qq}(z,p_{\mathrm{T}}^2)/P_{qq}(z)=1-\frac{2m_b^2z(1-z)^2}{(1+z^2)p_{\mathrm{T,evol}}^2}$")
    a1.set_title(r"(a) only soft emissions near the cut-off notice $m_b$")
    a1.legend(loc="lower right")
    th = np.geomspace(1e-3, 1.0, 400)
    Eb = 252.059
    th0 = MB / Eb
    a2.plot(th / th0, th**4 / (th**2 + th0**2) ** 2, color=COL_B, lw=1.9, label=r"massive: $\theta^4/(\theta^2+\theta_0^2)^2$")
    a2.axhline(1, color=TEXT2, lw=1.2, ls="--", label="massless quark")
    a2.axvline(1, color=TEXT2, lw=0.8, ls=":")
    a2.text(1.05, 0.05, r"$\theta=\theta_0=m_b/E_b=%.3f$" % th0, fontsize=9.5, color=TEXT2)
    a2.text(0.6, 0.27, r"$\frac{1}{4}$ at $\theta_0$", fontsize=9.5, color=COL_B)
    for step, pt_, z, ang in ems:
        a2.plot(ang / th0, ang**4 / (ang**2 + th0**2) ** 2, marker="*", ms=13, color=COL_EVENT, mec="white", zorder=5)
        a2.annotate("branching %d: $\\theta=%.2f$ rad\n$=%.0f\\,\\theta_0$" % (step, ang, ang / th0), xy=(ang / th0, ang**4 / (ang**2 + th0**2) ** 2),
                    xytext=(-10, -34 if step == 5 else 12), textcoords="offset points", ha="right", fontsize=9, color=COL_EVENT)
    a2.set_xscale("log")
    a2.set_xlim(0.05, 60)
    a2.set_ylim(0, 1.1)
    a2.set_xlabel(r"$\theta/\theta_0$")
    a2.set_ylabel(r"soft-gluon density per $\ln\theta$, relative to massless")
    a2.set_title(r"(b) her two emissions sit far outside the dead cone")
    a2.legend(loc="center right")
    fig.tight_layout()
    return fig


# --- page 10: the runaways --------------------------------------------------------------------------------------------------
def page_runaways(st):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.5, 4.8))
    fig.suptitle(r"Children who run away: the mother's $p_{\mathrm{T}}$ copy by copy, her descendants, and the family inside a cone of radius $R$", y=0.995)
    ems, chain = mother_emissions(st)
    # (a) pT of the mother's copies against the step that created them
    created = {}
    for h in st.history:
        for no in h["new_rows"]:
            created[no] = h["step"]
    steps = [0] + [created.get(no, 0) for no in chain[1:]]
    pts = [row_kin(st.rows[no])[0] for no in chain]
    a1.step(steps + [38], pts + [pts[-1]], where="post", color=COL_B, lw=2)
    labels = {6: "#6: 113.2", 10: "#10: 93.7, #15: 91.7", 21: "#21: 112.5", 25: "#25: 99.9", 46: "#46: 99.5", 60: "#60: 80.9", 124: "#124: 78.1"}
    offs = {6: (4, 6), 10: (4, -14), 21: (6, 4), 25: (4, 6), 46: (4, 6), 60: (4, 6), 124: (4, -14)}
    for s_, p_, no in zip(steps, pts, chain):
        a1.plot(s_, p_, marker="o", ms=6, color=COL_B, mec="white", zorder=5)
        if no in labels:
            a1.annotate(labels[no], xy=(s_, p_), xytext=offs[no], textcoords="offset points", fontsize=8.5, color=COL_B)
    for step, pt_, z, ang in ems:
        a1.axvline(step, color=COL_EVENT, lw=0.9, ls=":")
    a1.text(0.03, 0.08, "dotted: her own emissions (branchings %s);\nthe other steps are recoils" % ", ".join(str(e[0]) for e in ems), transform=a1.transAxes, fontsize=9.5, color=TEXT2)
    a1.axhline(113.221, color=TEXT2, lw=0.8, ls="--")
    a1.set_xlim(-1, 39)
    a1.set_ylim(60, 125)
    a1.set_xlabel("branching number")
    a1.set_ylabel(r"$p_{\mathrm{T}}$ of the mother's current copy  [GeV]")
    a1.set_title(r"(a) $113.2\to93.7\to91.7\to112.5\to99.9\to80.9\to78.1\,$GeV", fontsize=10.5)
    # (b) descendants of the mother in (Delta R, pT)
    bfinal = [r for r in st.final if r["cat"] == "b"][0]
    ptb, etab, phib = row_kin(bfinal)
    desc = [r for r in st.final if r["family"] == "b" and r["no"] != bfinal["no"]]
    others = [r for r in st.final if r["family"] != "b"]
    def dr(r):
        pt_, eta_, phi_ = row_kin(r)
        dphi = (phi_ - phib + math.pi) % (2 * math.pi) - math.pi
        return math.hypot(eta_ - etab, dphi)
    a2.scatter([dr(r) for r in others], [row_kin(r)[0] for r in others], s=20, color="#cfcdc5", label="not her descendants (%d)" % len(others))
    a2.scatter([dr(r) for r in desc], [row_kin(r)[0] for r in desc], s=50, color=COL_B, edgecolor="white", zorder=5, label="her descendants (%d)" % len(desc))
    a2.plot(0, ptb, marker="*", ms=14, color=COL_B, mec="white", zorder=6)
    a2.annotate("the $b$ herself, %.1f GeV" % ptb, xy=(0.02, ptb), xytext=(6, 0), textcoords="offset points", fontsize=9.5, color=COL_B)
    for r in desc:
        a2.annotate("%s %.1f" % (journey.parton_name(r["id"]), row_kin(r)[0]), xy=(dr(r), row_kin(r)[0]), xytext=(5, 4), textcoords="offset points", fontsize=8.5, color=COL_B)
    for R, lab in ((0.4, r"$\Delta R=0.4$, the family"), (0.8, r"$\Delta R=0.8$")):
        a2.axvline(R, color=COL_EVENT if R == 0.4 else TEXT2, lw=1.1, ls="--")
        a2.text(R + 0.03, 0.4, lab, rotation=90, fontsize=9.5, color=COL_EVENT if R == 0.4 else TEXT2, va="bottom")
    a2.set_yscale("log")
    a2.set_ylim(0.3, 200)
    a2.set_xlim(-0.05, 3.2)
    a2.set_xlabel(r"$\Delta R$ from the mother's last copy")
    a2.set_ylabel(r"$p_{\mathrm{T}}$  [GeV]")
    a2.set_title(r"(b) her descendants: the $u\bar u$ pair at $\Delta R\approx1$", fontsize=10.5)
    a2.legend(loc="upper right", fontsize=9)
    # (c) family pT inside R
    Rs = np.linspace(0.0, 3.0, 301)
    fam = []
    allp = []
    for R in Rs:
        mem = [r for r in st.final if dr(r) < R] + ([bfinal] if R == 0 else [])
        mem = [r for r in st.final if dr(r) < R or r["no"] == bfinal["no"]]
        px = sum(r["px"] for r in mem)
        py = sum(r["py"] for r in mem)
        fam.append(math.hypot(px, py))
        memd = [r for r in mem if r["family"] == "b"]
        allp.append(math.hypot(sum(r["px"] for r in memd), sum(r["py"] for r in memd)))
    a3.plot(Rs, fam, color=TEXT, lw=1.9, label="all final partons within $R$")
    a3.plot(Rs, allp, color=COL_B, lw=1.9, ls="--", label="her descendants within $R$ only")
    a3.axhline(113.221, color=TEXT2, lw=0.8, ls=":")
    a3.text(2.95, 114, "Born: 113.2", ha="right", fontsize=9.5, color=TEXT2)
    for R in (0.4, 0.8):
        j = int(round(R / 0.01))
        a3.plot(R, fam[j], marker="o", ms=6, color=COL_EVENT, mec="white", zorder=5)
        a3.annotate("$R=%.1f$: %.1f GeV" % (R, fam[j]), xy=(R, fam[j]), xytext=(8, -14 if R == 0.4 else -30), textcoords="offset points", fontsize=9.5, color=COL_EVENT)
    a3.set_xlabel(r"cone radius $R$ around the mother")
    a3.set_ylabel(r"vector-summed $p_{\mathrm{T}}$ inside the cone  [GeV]")
    a3.set_title(r"(c) the ledger against the cone radius", fontsize=10.5)
    a3.set_ylim(60, 130)
    a3.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


PAGES = {1: "propagator", 2: "kernels", 3: "sudakov", 4: "fsr", 5: "isr", 6: "interleaving",
         7: "coupling", 8: "competition", 9: "dead cone", 10: "runaways"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, default=HERE / "theory_parton_showering.pdf")
    ap.add_argument("--page", type=int, default=None, help="render one page only (1-10)")
    ap.add_argument("--trials", type=int, default=1500, help="proposals per radiator for the survival curves")
    ap.add_argument("--dpi", type=int, default=160)
    a = ap.parse_args()
    st = Setup()
    rounds = parse_rounds(st.log)
    print(f"seed-1 shower replayed: {st.sh.n_isr} ISR + {st.sh.n_fsr} FSR branchings; {len(rounds)} rounds in the log")
    makers = {
        1: lambda: page_propagator(st), 2: lambda: page_kernels(st), 3: lambda: page_sudakov(st, a.trials),
        4: lambda: page_fsr(st), 5: lambda: page_isr(st), 6: lambda: page_interleaving(st, rounds),
        7: lambda: page_coupling(st), 8: lambda: page_competition(st, rounds), 9: lambda: page_dead_cone(st),
        10: lambda: page_runaways(st),
    }
    pages = [a.page] if a.page else sorted(makers)
    if a.output.suffix.lower() == ".pdf" and not a.page:
        with PdfPages(a.output) as pdf:
            for k in pages:
                fig = makers[k]()
                pdf.savefig(fig, dpi=a.dpi)
                plt.close(fig)
                print(f"  page {k}: {PAGES[k]}")
    else:
        for k in pages:
            fig = makers[k]()
            out = a.output if a.page else a.output.with_name(f"{a.output.stem}_p{k}{a.output.suffix}")
            fig.savefig(out, dpi=a.dpi)
            plt.close(fig)
            print(f"  page {k}: {PAGES[k]} -> {out}")
    print("wrote", a.output)


if __name__ == "__main__":
    main()
