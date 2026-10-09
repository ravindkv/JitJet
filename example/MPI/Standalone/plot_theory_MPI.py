#!/usr/bin/env python3
"""The plots behind the theory sections of chapter 4, drawn from the standalone MPI script.

One page per theory (sub)section of the chapter. Every curve is computed with
the classes of standalone_multiparton_interactions.py (SigmaInt, Overlap, Beam,
Generator), every marker is a number of the seed-1 event at the impact
parameter of the CMSSW event (b = 0.3136 <b>, the record ../event_MPI.lhe),
and the ensemble panels read the statistics files written by
--repeat (standalone_mpi_stats_*.json) and by validate_mpi_with_pythia.py
(validate_mpi_with_pythia*.json) when they are present.

  page 1  cross section   the bare and the regularised parton-parton cross section, sigma_int(pT0)
                          with the CP5 point against sigma_ND, and the channel composition
  page 2  overlap         the double-Gaussian proton, O(b) and f(b), and the b distributions of
                          minimum-bias and of hard events with our event and the CMSSW event marked
  page 3  chain           the no-interaction probability below the hard scale, the pT ladder of our
                          event against the 31 secondary scatterings of the CMSSW event, and the
                          multiplicity distribution against Poisson and PYTHIA
  page 4  flavours        the (x1, x2) of every interaction on the proton's density, the processes,
                          and the depletion of the two protons interaction by interaction
  page 5  remnants        the remnant momentum shapes, the primordial-kT width and the kT given, and
                          the light-cone bookkeeping of systems and remnants
  page 6  reconnection    the reconnection probability with every system of our event, the string
                          length before and after, and the strings attached to the b quark
  page 7  underlying      the underlying-event densities (toward, transverse, away) of the ensembles
                          against PYTHIA, their distribution, and <n> against pT0 (the tune knob)
  page 8  our jet         the pT the underlying event adds inside R = 0.4 and 0.8 around the b, its
                          distribution, and the family pT and mass against the cone radius

    python3 plot_theory_MPI.py                    # -> theory_multiparton_interactions.pdf
    python3 plot_theory_MPI.py --page 2 -o o.png  # one page as PNG

Needs numpy, scipy and matplotlib (on this Mac: /usr/bin/python3). About 20 s.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import standalone_multiparton_interactions as core  # noqa: E402
import plot_journey_MPI as journey  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

COL_B, COL_BBAR = "#ff0000", "#0000ff"
COL_MPI, COL_REM, COL_ISR, COL_FSR = "#009999", "#666666", "#eb6834", "#2f8f5b"
COL_A, COL_BEAM = "#8e5bb5", "#2a78d6"
COL_EVENT, COL_CMSSW, COL_PY = "#c0392b", "#1f1f1f", "#7f7f7f"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d6d3ca"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "legend.fontsize": 9.5,
    "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 100,
})

# the 31 secondary interactions of the CMSSW event (status-33 pairs of the ParticleListDrawer dump,
# example/PS/CMSSW/step1_GEN_PS_cfg.log; pT of the pair in GeV, hardest first)
CMSSW_MPI_PT = [6.654, 6.128, 4.391, 4.101, 4.013, 3.576, 2.570, 2.449, 2.226, 2.053, 1.955, 1.931, 1.025, 0.984,
                0.972, 0.874, 0.812, 0.803, 0.772, 0.726, 0.672, 0.557, 0.513, 0.512, 0.498, 0.475, 0.372, 0.369,
                0.302, 0.244, 0.241]
CMSSW_B, CMSSW_F, CMSSW_NMPI = 0.3136, 3.709, 32
CMSSW_SIGMA_INT, CMSSW_SIGMA_ND = 500.67, 55.42
STATS_SAMPLED = HERE / "standalone_mpi_stats_sampled.stats.json"
STATS_FIXED = HERE / "standalone_mpi_stats_fixedb.stats.json"
PY_SAMPLED = HERE / "validate_mpi_with_pythia_sampled.json"
PY_FIXED = HERE / "validate_mpi_with_pythia_fixedb.json"


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


class Setup:
    """The seed-1 event at the CMSSW impact parameter, replayed with the script's classes."""

    def __init__(self, impact=CMSSW_B, seed=1):
        self.args = SimpleNamespace(input=HERE.parent.parent / "PS" / "event_PS.lhe", ptmax=None, ptmin_isr=0.2,
                                    ptmin_fsr=0.5, pt0=2.0, no_isr=False, no_fsr=False, no_shower=False, no_cr=False,
                                    no_kt=False, check=False, cr_range=core.CR_RANGE, sigma_nd=core.sigma_nd(),
                                    radius=0.4, impact=impact)
        self.alpha = core.AlphaStrong(0.118, 2)
        self.pdf = core.TablePDF(core.TABLES)
        self.pt0 = core.pt0_at(core.ECM)
        self.si = core.SigmaInt(self.pt0, self.pdf, self.alpha)
        self.ov = core.Overlap(self.si.total / self.args.sigma_nd)
        self.ev, self.gen, self.hard = core.generate(self.args, seed, self.pdf, self.alpha, self.si, self.ov)
        self.summary = core.summarise(self.ev, self.gen, self.hard, 0.4)
        self.stats_s, self.stats_f = load_json(STATS_SAMPLED), load_json(STATS_FIXED)
        self.py_s, self.py_f = load_json(PY_SAMPLED), load_json(PY_FIXED)


def mark(ax, x, label, color=COL_EVENT, y=0.93, ha="left", dx=4, ls="--"):
    ax.axvline(x, color=color, lw=1.1, ls=ls)
    ax.annotate(label, xy=(x, y), xycoords=("data", "axes fraction"), xytext=(dx if ha == "left" else -dx, 0),
                textcoords="offset points", ha=ha, va="top", fontsize=9, color=color)


# ---------------------------------------------------------------------------------------------
def page_cross_section(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    si, pt0 = st.si, st.pt0
    pts = si.pt
    a1.loglog(pts, si.dsigma, color=COL_MPI, lw=2, label=r"regularised: $\alpha_s^2(p_T^2+p_{T0}^2)\,/\,(p_T^2+p_{T0}^2)^2$")
    sel = pts > 0.6                     # alpha_s(pT^2) exists only above Lambda_3 = 0.38 GeV
    als_ratio = np.array([st.alpha.alphaS(p * p) / st.alpha.alphaS(p * p + pt0 ** 2) for p in pts[sel]])
    bare = si.dsigma[sel] * ((pts[sel] ** 2 + pt0 ** 2) / pts[sel] ** 2) ** 2 * als_ratio ** 2
    a1.loglog(pts[sel], bare, color=TEXT2, lw=1.4, ls="--", label=r"bare: $\alpha_s^2(p_T^2)\,/\,p_T^4$")
    mark(a1, pt0, r"$p_{T0}$ = %.2f GeV" % pt0, y=0.5)
    mark(a1, core.PTMIN_MPI, r"$p_{T\min}$", color=TEXT2, y=0.3)
    a1.set_xlabel(r"$p_T$ of the 2 $\to$ 2 scattering  [GeV]")
    a1.set_ylabel(r"$d\sigma/dp_T^2$  [mb/GeV$^2$]")
    a1.set_title("(a) the parton-parton cross section", fontsize=10.5)
    a1.set_xlim(0.2, 200)
    a1.set_ylim(1e-7, 1e5)
    a1.legend(loc="upper right", fontsize=9)
    # sigma_int against pT0
    p0s = np.array([0.8, 1.0, 1.2, 1.44, 1.7, 2.0, 2.5, 3.0])
    vals = [core.SigmaInt(p, st.pdf, st.alpha, n=80, ny=41).total for p in p0s]
    a2.plot(p0s, vals, "o-", color=COL_MPI, lw=1.8, label=r"$\sigma_{\rm int}(p_{T0})$, this script")
    a2.plot([pt0], [si.total], "o", ms=9, color=COL_EVENT, mec="white", zorder=5)
    a2.annotate(f"CP5: {pt0:.2f} GeV -> {si.total:.0f} mb\n(CMSSW log: {CMSSW_SIGMA_INT:.2f} mb)", xy=(pt0, si.total),
                xytext=(1.75, 700), fontsize=9.5, color=COL_EVENT, arrowprops=dict(arrowstyle="-|>", color=COL_EVENT, lw=0.8))
    a2.axhline(st.args.sigma_nd, color=TEXT2, lw=1.2, ls=":")
    a2.text(2.95, st.args.sigma_nd * 1.08, r"$\sigma_{\rm ND}$ = %.2f mb" % st.args.sigma_nd, ha="right", fontsize=9.5, color=TEXT2)
    a2.set_yscale("log")
    a2.set_ylim(30, 4000)
    a2b = a2.twinx()
    a2b.set_yscale("log")
    a2b.set_ylim(30 / st.args.sigma_nd, 4000 / st.args.sigma_nd)
    a2b.set_ylabel(r"$\langle n \rangle = \sigma_{\rm int}/\sigma_{\rm ND}$", color=TEXT2)
    a2b.grid(False)
    a2.set_xlabel(r"$p_{T0}$  [GeV]")
    a2.set_ylabel(r"$\sigma_{\rm int}$  [mb]")
    a2.set_title("(b) the integrated cross section against the regulator", fontsize=10.5)
    a2.legend(loc="upper right", fontsize=9)
    # channel composition against pT
    comp_pts = np.array([0.3, 0.5, 1.0, 1.44, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
    fracs = {k: [] for k in ("gg", "qg", "qq")}
    for p in comp_pts:
        tot, parts = composition(st, p * p)
        for k in fracs:
            fracs[k].append(parts[k] / tot)
    bottom = np.zeros(len(comp_pts))
    for k, col, lab in (("gg", COL_MPI, r"$gg \to$ anything"), ("qg", COL_A, r"$qg \to qg$"), ("qq", COL_BEAM, r"$qq'$, $q\bar q$ initiated")):
        a3.bar(np.arange(len(comp_pts)), fracs[k], bottom=bottom, color=col, label=lab, width=0.8)
        bottom += np.array(fracs[k])
    a3.set_xticks(np.arange(len(comp_pts)))
    a3.set_xticklabels([f"{p:g}" for p in comp_pts])
    a3.set_xlabel(r"$p_T$  [GeV]")
    a3.set_ylabel("share of $d\\sigma/dp_T^2$")
    a3.set_title("(c) who scatters: initial states against $p_T$", fontsize=10.5)
    a3.set_ylim(0, 1)
    a3.legend(loc="lower left", fontsize=9)
    fig.suptitle("The rest of the protons does not stay quiet: the regularised QCD cross section (page 1)", fontsize=12)
    fig.tight_layout()
    return fig


def composition(st: Setup, pt2: float, ny: int = 41):
    """dsigma/dpT2 split into gg, qg and quark-quark initiated contributions (same grid as SigmaInt)."""
    pT = math.sqrt(pt2)
    ecm = core.ECM
    xT = 2.0 * pT / ecm
    ymax = math.acosh(1.0 / xT)
    y = np.linspace(-ymax, ymax, ny)
    y3, y4 = np.meshgrid(y, y, indexing="ij")
    x1 = (pT / ecm) * (np.exp(y3) + np.exp(y4))
    x2 = (pT / ecm) * (np.exp(-y3) + np.exp(-y4))
    ok = (x1 < 1) & (x2 < 1)
    x1c, x2c = np.where(ok, x1, 0.5), np.where(ok, x2, 0.5)
    s = x1c * x2c * ecm * ecm
    t = -pt2 * (1 + np.exp(y4 - y3))
    u = -pt2 * (1 + np.exp(y3 - y4))
    f1 = {p: st.pdf.xf_many(p, x1c, pt2) for p in core.ALL_IN}
    f2 = {p: st.pdf.xf_many(p, x2c, pt2) for p in core.ALL_IN}
    quarks = [p for p in core.ALL_IN if p != 21]
    gg = f1[21] * f2[21] * (core.me_light("gg->gg", s, t, u) + core.NQUARK_NEW * core.me_light("gg->qqbar", s, t, u)
                            + sum(core.me_heavy("gg->QQbar", s, t, u, m) for m in (core.MC, core.MB)))
    qg = (sum(f1[q] for q in quarks) * f2[21] + f1[21] * sum(f2[q] for q in quarks)) * 0.5 * (
        core.me_light("qg->qg", s, t, u) + core.me_light("qg->qg", s, u, t))
    tot = core.pair_weights(f1, f2, s, t, u)
    w = np.where(ok, 1.0, 0.0)
    return float((tot * w).sum()), dict(gg=float((gg * w).sum()), qg=float((qg * w).sum()), qq=float(((tot - gg - qg) * w).sum()))


# ---------------------------------------------------------------------------------------------
def page_overlap(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    ov = st.ov
    r = np.linspace(0, 3.0, 400)
    beta, a2r = core.CORE_FRACTION, core.CORE_RADIUS
    rho = (1 - beta) * np.exp(-r ** 2) + beta / a2r ** 3 * np.exp(-r ** 2 / a2r ** 2)
    rho1 = np.exp(-r ** 2)
    a1.plot(r, rho / rho[0], color=COL_MPI, lw=2, label="double Gaussian (CP5: core 63 %% of the matter in %.2f of the radius)" % a2r)
    a1.plot(r, rho1, color=TEXT2, lw=1.4, ls="--", label="single Gaussian")
    a1.set_xlabel(r"$r$  [units of the outer radius $a_1$]")
    a1.set_ylabel(r"$\rho(r)\,/\,\rho(0)$")
    a1.set_title("(a) matter inside the proton", fontsize=10.5)
    a1.legend(loc="upper right", fontsize=8.5)
    b = ov.b / ov.b_avg
    a2.plot(b, ov.O(ov.b) / ov.O(0.0), color=COL_MPI, lw=2, label=r"$O(b)/O(0)$ (overlap of the two protons)")
    a2.plot(b, ov.p_int, color=COL_A, lw=1.8, label=r"$P_{\rm int}(b) = 1 - e^{-n(b)}$")
    a2b = a2.twinx()
    a2b.plot(b, ov.O(ov.b) * ov.p_int_total, color=COL_EVENT, lw=1.6, ls="-.", label=r"$f(b) = n(b)/\langle n\rangle$")
    a2b.set_ylabel(r"enhancement factor $f(b)$", color=COL_EVENT)
    a2b.grid(False)
    a2b.set_ylim(0, 5)
    a2.set_xlim(0, 2.5)
    a2.set_ylim(0, 1.05)
    a2.set_xlabel(r"$b\,/\,\langle b\rangle$")
    a2.set_ylabel("overlap and interaction probability")
    a2.set_title("(b) overlap, interaction probability and enhancement", fontsize=10.5)
    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = a2b.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8.5)
    # b distributions
    dens_mb = ov.b * ov.p_int
    dens_hard = ov.b * ov.O(ov.b)
    a3.plot(b, dens_mb / np.trapz(dens_mb, b), color=COL_A, lw=1.8, label=r"minimum bias: $\propto b\,P_{\rm int}(b)$")
    a3.plot(b, dens_hard / np.trapz(dens_hard, b), color=COL_MPI, lw=2, label=r"events with a hard process: $\propto b\,O(b)$")
    a3.axvline(CMSSW_B, color=COL_EVENT, lw=1.3, ls="--")
    a3.annotate(f"our event and the CMSSW event:\n$b$ = {CMSSW_B} $\\langle b\\rangle$, $f$ = {st.gen.fb:.2f} (CMSSW {CMSSW_F})",
                xy=(CMSSW_B, 0.62), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=COL_EVENT, va="top")
    if st.stats_s and st.stats_s["counts"].get("b"):
        a3.hist(st.stats_s["counts"]["b"], bins=np.linspace(0, 2.5, 26), density=True, histtype="step", color=COL_EVENT, lw=1.2,
                label=f"this script, {len(st.stats_s['counts']['b'])} events, $b$ drawn")
    if st.py_s and st.py_s["counts"].get("b"):
        a3.hist(st.py_s["counts"]["b"], bins=np.linspace(0, 2.5, 26), density=True, histtype="step", color=COL_PY, lw=1.2, ls="--",
                label=f"PYTHIA 8, {len(st.py_s['counts']['b'])} events")
    a3.set_xlim(0, 2.5)
    a3.set_xlabel(r"$b\,/\,\langle b\rangle$")
    a3.set_ylabel("probability density")
    a3.set_title("(c) which impact parameters: the hard process picks central collisions", fontsize=10.5)
    a3.legend(loc="upper right", fontsize=8.5)
    fig.suptitle(r"The impact parameter: $\langle f\rangle_{\rm ND}$ = %.2f, $\langle f\rangle_{\rm hard}$ = %.2f, $f(0)$ = %.2f (page 2)"
                 % (ov.f_avg_minbias(), ov.f_avg_hard(), ov.f_max), fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
def page_chain(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    si, gen = st.si, st.gen
    pts = si.pt[si.pt < gen.pt_max]
    for f, col, lab in ((1.0, TEXT2, r"$f$ = 1 (average collision)"), (gen.fb, COL_EVENT, r"$f$ = %.2f (our event)" % gen.fb)):
        surv = [math.exp(-f * (si.above(p) - si.above(gen.pt_max)) / st.args.sigma_nd) for p in pts]
        a1.semilogx(pts, surv, color=col, lw=1.8, label=lab)
    for p in st.summary["mpi_pt"]:
        a1.axvline(p, color=COL_MPI, lw=0.6, alpha=0.7)
    a1.text(0.25, 0.02, "cyan lines: the %d secondary\ninteractions of our event" % len(st.summary["mpi_pt"]), fontsize=8.5, color=COL_MPI)
    a1.set_xlabel(r"$p_T$  [GeV]")
    a1.set_ylabel(r"$\exp[-f(b)\,\sigma(p_T' > p_T)/\sigma_{\rm ND}]$")
    a1.set_title("(a) no interaction between the hard scale and $p_T$", fontsize=10.5)
    a1.set_xlim(0.2, 120)
    a1.set_ylim(0, 1.02)
    a1.legend(loc="center left", fontsize=9)
    ours = sorted(st.summary["mpi_pt"], reverse=True)
    a2.semilogy(np.arange(1, len(ours) + 1), ours, "o-", color=COL_MPI, lw=1.4, ms=4, label=f"this script, seed 1 ({len(ours)} interactions)")
    a2.semilogy(np.arange(1, len(CMSSW_MPI_PT) + 1), CMSSW_MPI_PT, "s--", color=COL_CMSSW, lw=1.2, ms=4,
                label=f"CMSSW event ({len(CMSSW_MPI_PT)} interactions)")
    a2.axhline(st.pt0, color=TEXT2, lw=0.8, ls=":")
    a2.text(1.0, st.pt0 * 1.1, r"$p_{T0}$", fontsize=9, color=TEXT2)
    a2.set_xlabel("interaction (hardest first)")
    a2.set_ylabel(r"$p_T$  [GeV]")
    a2.set_title("(b) the ladder of secondary interactions", fontsize=10.5)
    a2.legend(loc="upper right", fontsize=9)
    nmax = 60
    ns = np.arange(0, nmax)
    mean_below = gen.n_mean_below
    pois = np.exp(-mean_below) * np.array([mean_below ** k / math.factorial(k) for k in ns])
    a3.plot(ns + 1, pois, color=TEXT2, lw=1.4, ls="--", label=r"Poisson with mean $1 + f\,\sigma(<p_{T\max})/\sigma_{\rm ND}$ = %.1f" % (1 + mean_below))
    if st.stats_f and st.stats_f["counts"].get("n_mpi"):
        v = st.stats_f["counts"]["n_mpi"]
        a3.hist(v, bins=np.arange(0.5, nmax + 0.5), density=True, histtype="step", color=COL_MPI, lw=1.8,
                label=f"this script at $b$ = {CMSSW_B}: {np.mean(v):.1f} $\\pm$ {np.std(v):.1f} ({len(v)} events)")
    if st.py_f and st.py_f["counts"].get("n_mpi"):
        v = st.py_f["counts"]["n_mpi"]
        a3.hist(v, bins=np.arange(0.5, nmax + 0.5), density=True, histtype="step", color=COL_PY, lw=1.6, ls="--",
                label=f"PYTHIA 8 at $b$ = {CMSSW_B}: {np.mean(v):.1f} $\\pm$ {np.std(v):.1f}")
    if st.stats_s and st.stats_s["counts"].get("n_mpi"):
        v = st.stats_s["counts"]["n_mpi"]
        a3.hist(v, bins=np.arange(0.5, nmax + 0.5), density=True, histtype="step", color=COL_A, lw=1.2,
                label=f"this script, $b$ drawn: {np.mean(v):.1f} $\\pm$ {np.std(v):.1f}")
    a3.axvline(st.summary["n_mpi"], color=COL_EVENT, lw=1.3)
    a3.axvline(CMSSW_NMPI, color=COL_CMSSW, lw=1.3, ls="--")
    a3.text(st.summary["n_mpi"] + 0.5, 0.95, "our event", color=COL_EVENT, fontsize=9, transform=a3.get_xaxis_transform(), va="top")
    a3.text(CMSSW_NMPI + 0.5, 0.86, "CMSSW event", color=COL_CMSSW, fontsize=9, transform=a3.get_xaxis_transform(), va="top")
    a3.set_xlabel("number of interactions (incl. the hard one)")
    a3.set_ylabel("probability")
    a3.set_title("(c) how many: the multiplicity of interactions", fontsize=10.5)
    a3.set_xlim(0, nmax)
    a3.legend(loc="upper left", fontsize=8)
    fig.suptitle("The chain of interactions below the hard scale (page 3)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
def page_flavours(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    gen, ev = st.gen, st.ev
    xs = np.logspace(-6, 0, 300)
    q2 = 2.0
    a1.loglog(xs, st.pdf.xf_many(21, xs, q2), color=COL_MPI, lw=1.6, label=r"$x g(x, Q = 1.65$ GeV$)$")
    a1.loglog(xs, st.pdf.xf_many(2, xs, q2) + st.pdf.xf_many(1, xs, q2), color=COL_A, lw=1.4, label=r"$x(u + d)$")
    a1.loglog(xs, st.pdf.xf_many(-2, xs, q2) + st.pdf.xf_many(-1, xs, q2), color=COL_BEAM, lw=1.4, ls="--", label=r"$x(\bar u + \bar d)$")
    x1s = [s["x1"] for s in gen.systems[1:]]
    x2s = [s["x2"] for s in gen.systems[1:]]
    a1b = a1.twinx()
    a1b.loglog(x1s, x2s, "s", color=COL_EVENT, ms=5, mec="white", label=r"$(x_1, x_2)$ of the %d secondary interactions" % len(x1s))
    a1b.loglog([gen.systems[0]["x1"]], [gen.systems[0]["x2"]], "o", color=COL_B, ms=9, mec="white", label=r"the hard process ($x_1$ = %.3f, $x_2$ = %.2f)" % (gen.systems[0]["x1"], gen.systems[0]["x2"]))
    a1b.set_ylabel(r"$x_2$ (proton B)", color=COL_EVENT)
    a1b.grid(False)
    a1b.set_ylim(1e-6, 1)
    a1.set_xlim(1e-6, 1)
    a1.set_ylim(1e-2, 1e3)
    a1.set_xlabel(r"$x$  (and $x_1$ of proton A)")
    a1.set_ylabel(r"$x f(x, Q^2)$")
    a1.set_title("(a) where in the proton the interactions happen", fontsize=10.5)
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a1b.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8.5)
    procs = {}
    for s in gen.systems[1:]:
        procs[s["process"]] = procs.get(s["process"], 0) + 1
    names = sorted(procs, key=lambda k: -procs[k])
    a2.barh(np.arange(len(names)), [procs[n] for n in names], color=COL_MPI)
    a2.set_yticks(np.arange(len(names)))
    a2.set_yticklabels([n.replace("->", r"$\to$").replace("qbar", r"$\bar q$").replace("QQbar", r"$Q\bar Q$") for n in names], fontsize=10)
    a2.invert_yaxis()
    a2.set_xlabel("secondary interactions of our event")
    a2.set_title("(b) which processes", fontsize=10.5)
    Xa, Xb = [1.0], [1.0]
    labels = []
    for s in gen.systems:
        Xa.append(Xa[-1] - ev.x(s["in_a_before_kt"]) if "in_a_before_kt" in s else Xa[-1] - s["x1"])
        Xb.append(Xb[-1] - ev.x(s["in_b_before_kt"]) if "in_b_before_kt" in s else Xb[-1] - s["x2"])
    a3.step(np.arange(len(Xa)), Xa, where="post", color=COL_A, lw=1.8, label=r"proton A: momentum fraction left")
    a3.step(np.arange(len(Xb)), Xb, where="post", color=COL_BEAM, lw=1.8, label=r"proton B")
    for side, beam, col in ((0, gen.beams[0], COL_A), (1, gen.beams[1], COL_BEAM)):
        for i, r in enumerate(beam.resolved[:beam.n_init]):
            if r.is_valence:
                a3.annotate(f"valence {core.pname(r.id)}", xy=(i + 1, (Xa if side == 0 else Xb)[i + 1]), xytext=(0, 8 if side == 0 else -14),
                            textcoords="offset points", fontsize=8, color=col, ha="center")
    a3.set_xlabel("interaction (0 = the hard process)")
    a3.set_ylabel(r"$X = 1 - \sum x_i$")
    a3.set_title("(c) the protons are being used up", fontsize=10.5)
    a3.set_ylim(0, 1.05)
    a3.legend(loc="lower left", fontsize=9)
    fig.suptitle("Flavours, momentum fractions and the depletion of the protons (page 4)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
def page_remnants(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    gen = st.gen
    x = np.linspace(1e-4, 1, 500)
    for power, col, lab in ((core.VALENCE_POWER_U, COL_A, r"valence $u$: $(1-x)^{3.5}/\sqrt{x}$"),
                            (core.VALENCE_POWER_D, COL_BEAM, r"valence $d$: $(1-x)^{2}/\sqrt{x}$")):
        y = (1 - x) ** power / np.sqrt(x)
        a1.plot(x, y / np.trapz(y, x), color=col, lw=1.8, label=lab)
    yg = (1 - x) ** core.GLUON_POWER / x
    sel = x > 1e-3
    a1.plot(x[sel], yg[sel] / np.trapz(yg[sel], x[sel]), color=COL_MPI, lw=1.4, ls="--", label=r"gluon: $(1-x)^4/x$")
    xs_c = 0.02
    yc = np.array([core.Beam.x_comp_dist(xc, xs_c) / max(xc, 1e-9) for xc in x])
    a1.plot(x, yc / np.trapz(yc, x), color=COL_EVENT, lw=1.4, ls="-.", label=r"companion of a sea quark at $x_s$ = 0.02")
    for side, beam in enumerate(gen.beams):
        for r in beam.resolved[beam.n_init:]:
            a1.plot([r.x], [0.2 + 0.3 * side], marker="v" if side == 0 else "^", color=COL_A if side == 0 else COL_BEAM, ms=7, mec="white")
    a1.text(0.55, 0.2, "markers: the remnant partons of our event\n(A down, B up; unrescaled $x$)", fontsize=8.5, color=TEXT2)
    a1.set_xscale("log")
    a1.set_xlim(1e-3, 1)
    a1.set_ylim(0, 12)
    a1.set_xlabel(r"unrescaled $x$ of a remnant parton")
    a1.set_ylabel("shape (normalised)")
    a1.set_title("(a) how the remnants share the leftover momentum", fontsize=10.5)
    a1.legend(loc="upper right", fontsize=8.5)
    q = np.logspace(-1, 2.2, 200)
    w = (core.HALF_SCALE_KT * core.KT_SOFT + q * core.KT_HARD) / (core.HALF_SCALE_KT + q)
    a2.semilogx(q, w, color=TEXT2, lw=1.8, label=r"$\sigma(Q) = (Q_{1/2}\sigma_{\rm soft} + Q\,\sigma_{\rm hard})/(Q_{1/2} + Q)$")
    for s in gen.systems:
        scale = gen.q_ren_hard if s["kind"] == "hard" else s["pt"]
        ra, rb = gen.beams[0].resolved[s["index"]], gen.beams[1].resolved[s["index"]]
        kt = 0.5 * (math.hypot(ra.p[1], ra.p[2]) + math.hypot(rb.p[1], rb.p[2]))
        a2.plot([scale], [s["kt_width"]], "o", color=COL_MPI, ms=5, mec="white")
        a2.plot([scale], [kt], "x", color=COL_EVENT, ms=6)
    a2.plot([], [], "o", color=COL_MPI, label=r"width used, after the mass damping $m/(m + m_{1/2} y_{\rm damp})$")
    a2.plot([], [], "x", color=COL_EVENT, label=r"mean $|k_T|$ actually given to the two initiators")
    a2.set_xlabel(r"scale $Q$ of the system  [GeV]  ($p_T$ of the MPI, $Q_{\rm ren}$ of the hard process)")
    a2.set_ylabel(r"primordial $k_T$ width  [GeV]")
    a2.set_title("(b) primordial transverse momentum", fontsize=10.5)
    a2.set_ylim(0, 2.0)
    a2.legend(loc="upper left", fontsize=8.5)
    ecm = gen.ecm
    pos = [[], []]
    for s in gen.systems:
        pa, pb = s["p_init_new"]
        pos[0].append((pa[0] + pa[3] + pb[0] + pb[3]) / ecm)
        pos[1].append((pa[0] - pa[3] + pb[0] - pb[3]) / ecm)
    rem = [[], []]
    for side in (0, 1):
        for r in gen.beams[side].resolved[gen.beams[side].n_init:]:
            rem[side].append((r.p[0] + r.p[3]) / ecm if side == 0 else (r.p[0] - r.p[3]) / ecm)
    for k, (lab, col) in enumerate((("systems", COL_MPI), ("remnants", COL_REM))):
        pass
    for side in (0, 1):
        bottom = 0.0
        for i, v in enumerate(pos[side]):
            a3.bar(side, v, bottom=bottom, color=COL_B if i == 0 else COL_MPI, edgecolor="white", lw=0.4, width=0.6)
            bottom += v
        for v in rem[side]:
            a3.bar(side, v, bottom=bottom, color=COL_REM, edgecolor="white", lw=0.4, width=0.6)
            bottom += v
    a3.bar([], [], color=COL_B, label="hard system")
    a3.bar([], [], color=COL_MPI, label="secondary systems")
    a3.bar([], [], color=COL_REM, label="beam remnants")
    a3.set_xticks([0, 1])
    a3.set_xticklabels([r"$p^+$ of proton A", r"$p^-$ of proton B"])
    a3.set_ylabel(r"share of the light-cone momentum $\sqrt{s}$")
    a3.set_title("(c) the bookkeeping must close: systems + remnants = the proton", fontsize=10.5)
    a3.set_ylim(0, 1.0)
    a3.legend(loc="upper center", fontsize=9)
    fig.suptitle("Beam remnants: flavours, momenta and primordial $k_T$ (page 5)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
def strings_touching(gen, rows):
    """The colour chains that contain any of the given rows (lists of rows)."""
    return [ch for ch in gen.strings() if any(r in ch for r in rows)]


def page_reconnection(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    gen, ev = st.gen, st.ev
    pt = np.logspace(-0.7, 2.1, 300)
    pt20 = (core.CR_RANGE * st.pt0) ** 2
    a1.semilogx(pt, pt20 / (pt20 + pt ** 2), color=COL_MPI, lw=2, label=r"$P = (R\,p_{T0})^2/((R\,p_{T0})^2 + p_T^2)$, $R$ = %.3f" % core.CR_RANGE)
    for irec, p, prob, merged in getattr(gen, "cr_decisions", []):
        a1.plot([p], [prob], "o" if merged is not None else "x", color=COL_EVENT if merged is not None else TEXT2, ms=6, mec="white" if merged is not None else None)
    a1.plot([], [], "o", color=COL_EVENT, label="merged into a harder system")
    a1.plot([], [], "x", color=TEXT2, label="not merged")
    a1.axvline(gen.pt_max, color=COL_B, lw=1.0, ls=":")
    a1.text(gen.pt_max, 0.5, "hard\nsystem", color=COL_B, fontsize=8.5, ha="right")
    a1.set_xlabel(r"$p_T$ of the system  [GeV]")
    a1.set_ylabel("reconnection probability")
    a1.set_title("(a) who gets reconnected: the %d systems of our event" % len(gen.systems), fontsize=10.5)
    a1.set_ylim(0, 1.05)
    a1.legend(loc="center left", fontsize=9)
    lam0, lam1 = getattr(gen, "lambda_before", None), getattr(gen, "lambda_after", None)
    bars = [("before", lam0, TEXT2), ("after", lam1, COL_MPI)]
    a2.bar([0, 1], [lam0 or 0, lam1 or 0], color=[c for _, _, c in bars], width=0.6)
    for i, (lab, v, c) in enumerate(bars):
        if v is not None:
            a2.text(i, v * 1.02, f"{v:.0f}", ha="center", fontsize=10, color=c)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["before reconnection", "after reconnection"])
    a2.set_ylabel(r"string length $\lambda = \sum \ln(1 + m_{ij}^2/m_0^2)$,  $m_0$ = %.1f GeV" % core.CR_M0)
    a2.set_title("(b) the strings get shorter: %d of %d systems merged" % (st.summary["n_reconnected"], len(gen.systems) - 1), fontsize=10.5)
    if st.stats_f and st.stats_f["counts"].get("lambda_cr"):
        a2.text(0.5, 0.5, "ensemble at $b$ = %.4f: %.0f $\\to$ %.0f" % (CMSSW_B, np.mean(st.stats_f["counts"]["lambda_nocr"]),
                                                                    np.mean(st.stats_f["counts"]["lambda_cr"])),
                transform=a2.transAxes, ha="center", fontsize=9.5, color=TEXT2)
    # strings attached to the b quark, in (y, phi)
    brow = next(p["row"] for p in st.summary["partons"] if p["name"] == "b")
    chains = strings_touching(gen, [brow])
    for ch in chains:
        ys = [0.5 * math.log((ev.p[i].p[0] + ev.p[i].p[3]) / max(ev.p[i].p[0] - ev.p[i].p[3], 1e-9)) for i in ch]
        phis = [core.phi(ev.p[i].p) for i in ch]
        for (y0, f0), (y1, f1) in zip(zip(ys, phis), zip(ys[1:], phis[1:])):
            d = f1 - f0
            if d > math.pi:
                f1 -= 2 * math.pi
            elif d < -math.pi:
                f1 += 2 * math.pi
            a3.plot([y0, y1], [f0, f1], color=COL_MPI, lw=1.2, alpha=0.8)
        orig = core.origins(ev, gen)
        for i, y, f in zip(ch, ys, phis):
            col = COL_B if i == brow else {"hard": COL_FSR, "MPI": COL_MPI, "remnant": COL_REM}.get(orig.get(i), COL_ISR)
            a3.plot([y], [f], "o", color=col, ms=max(3, min(12, 2 + 2 * math.log10(1 + core.pt(ev.p[i].p)))), mec="white")
    a3.set_xlabel("rapidity $y$")
    a3.set_ylabel(r"$\phi$")
    a3.set_title("(c) the %d string(s) that pass through the $b$ quark, after reconnection" % len(chains), fontsize=10.5)
    a3.set_xlim(-10, 10)
    a3.set_ylim(-math.pi - 0.3, math.pi + 0.3)
    a3.plot([], [], "o", color=COL_B, label="the $b$ quark")
    a3.plot([], [], "o", color=COL_FSR, label="partons of the hard system")
    a3.plot([], [], "o", color=COL_MPI, label="partons of secondary systems")
    a3.plot([], [], "o", color=COL_REM, label="beam remnants")
    a3.legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Colour reconnection: the MPI-based model (page 6)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
def page_underlying(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    sets = [("this script, $b$ drawn", st.stats_s, COL_A), ("this script, $b$ = %.4f" % CMSSW_B, st.stats_f, COL_MPI),
            ("PYTHIA 8, $b$ drawn", st.py_s, COL_PY), ("PYTHIA 8, $b$ = %.4f" % CMSSW_B, st.py_f, COL_CMSSW)]
    width = 0.2
    for k, (lab, data, col) in enumerate(sets):
        if not data:
            continue
        c = data["counts"]
        vals = [np.mean(c["ue_transverse"]), np.mean(c["ue_density_all"])]
        errs = [np.std(c["ue_transverse"]) / math.sqrt(len(c["ue_transverse"])), np.std(c["ue_density_all"]) / math.sqrt(len(c["ue_density_all"]))]
        a1.bar(np.arange(2) + (k - 1.5) * width, vals, width=width, yerr=errs, color=col, label=lab, capsize=2)
    u = st.summary.get("ue")
    if u:
        a1.plot([0 - 1.5 * width], [u["density_transverse"]], "*", color=COL_EVENT, ms=12, mec="white", label="our event", zorder=5)
    a1.set_xticks(np.arange(2))
    a1.set_xticklabels([r"transverse region" + "\n" + r"$60^\circ < |\Delta\phi| < 120^\circ$", "all azimuths"])
    a1.set_ylabel(r"$\sum p_T$ per unit $\eta\!-\!\phi$ area, $|\eta| < 2.5$  [GeV]")
    a1.set_title("(a) underlying-event density of partons not from the hard system", fontsize=10.5)
    a1.legend(loc="upper left", fontsize=8.5)
    bins = np.linspace(0, 8, 33)
    for lab, data, col, ls in (("this script, $b$ = %.4f" % CMSSW_B, st.stats_f, COL_MPI, "-"), ("PYTHIA 8, $b$ = %.4f" % CMSSW_B, st.py_f, COL_CMSSW, "--"),
                               ("this script, $b$ drawn", st.stats_s, COL_A, "-"), ("PYTHIA 8, $b$ drawn", st.py_s, COL_PY, "--")):
        if data and data["counts"].get("ue_transverse"):
            a2.hist(data["counts"]["ue_transverse"], bins=bins, density=True, histtype="step", color=col, lw=1.5, ls=ls, label=lab)
    if u:
        mark(a2, u["density_transverse"], "our event", y=0.95)
    a2.set_xlabel(r"transverse density  [GeV per unit $\eta\!-\!\phi$]")
    a2.set_ylabel("probability density")
    a2.set_title("(b) event-by-event spread of the transverse density", fontsize=10.5)
    a2.legend(loc="upper right", fontsize=8.5)
    p0s = np.array([1.0, 1.2, 1.44, 1.7, 2.0, 2.5])
    nav = [core.SigmaInt(p, st.pdf, st.alpha, n=80, ny=41).total / st.args.sigma_nd for p in p0s]
    a3.plot(p0s, nav, "o-", color=COL_MPI, lw=1.8, label=r"$\langle n\rangle = \sigma_{\rm int}/\sigma_{\rm ND}$")
    a3.plot(p0s, [st.gen.fb * v for v in nav], "s--", color=COL_EVENT, lw=1.4, label=r"$f(b)\,\langle n\rangle$ at our $b$")
    mark(a3, st.pt0, "CP5", y=0.9)
    a3.set_xlabel(r"$p_{T0}$  [GeV]")
    a3.set_ylabel("mean number of interactions")
    a3.set_title("(c) the tune knob: more screening, fewer interactions", fontsize=10.5)
    a3.legend(loc="upper right", fontsize=9)
    fig.suptitle("Measuring the underlying event: transverse-region observables (page 7)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
def page_our_jet(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    sets = [("this script, $b$ drawn", st.stats_s, COL_A), ("this script, $b$ = %.4f" % CMSSW_B, st.stats_f, COL_MPI),
            ("PYTHIA 8, $b$ drawn", st.py_s, COL_PY), ("PYTHIA 8, $b$ = %.4f" % CMSSW_B, st.py_f, COL_CMSSW)]
    width = 0.2
    for k, (lab, data, col) in enumerate(sets):
        if not data or "b" not in data["families"]:
            continue
        fam = data["families"]["b"]
        vals, errs = [], []
        for R in ("0.4", "0.8"):
            v = fam.get(R, fam.get(float(R), {})).get("ue_scalar", [])
            vals.append(np.mean(v) if len(v) else 0)
            errs.append(np.std(v) / math.sqrt(len(v)) if len(v) else 0)
        a1.bar(np.arange(2) + (k - 1.5) * width, vals, width=width, yerr=errs, color=col, label=lab, capsize=2)
    ours = next(p for p in st.summary["partons"] if p["name"] == "b")
    a1.plot([0 - 1.5 * width, 1 - 1.5 * width], [ours["cones"][0.4]["pt_ue_scalar"], ours["cones"][0.8]["pt_ue_scalar"]], "*",
            color=COL_EVENT, ms=12, mec="white", label="our event", zorder=5)
    if st.stats_f:
        rho = np.mean(st.stats_f["counts"]["ue_density_all"])
        a1.plot([0, 1], [rho * math.pi * 0.4 ** 2, rho * math.pi * 0.8 ** 2], "_", color=TEXT, ms=40, mew=1.5,
                label=r"$\rho\,\pi R^2$ with the all-azimuth density")
    a1.set_xticks(np.arange(2))
    a1.set_xticklabels(["$R$ = 0.4", "$R$ = 0.8"])
    a1.set_ylabel(r"scalar $\sum p_T$ of UE partons inside the cone  [GeV]")
    a1.set_title("(a) what the underlying event drops into the $b$ cone", fontsize=10.5)
    a1.legend(loc="upper left", fontsize=8.5)
    bins = np.linspace(0, 25, 26)
    for lab, data, col, ls in (("this script, $b$ = %.4f" % CMSSW_B, st.stats_f, COL_MPI, "-"), ("PYTHIA 8, $b$ = %.4f" % CMSSW_B, st.py_f, COL_CMSSW, "--")):
        if data and "b" in data["families"]:
            fam = data["families"]["b"]
            for R, lw in (("0.4", 1.8), ("0.8", 1.2)):
                v = fam.get(R, fam.get(float(R), {})).get("ue_scalar", [])
                if len(v):
                    a2.hist(v, bins=bins, density=True, histtype="step", color=col, lw=lw, ls=ls, label=f"{lab}, $R$ = {R}")
    a2.set_xlabel(r"UE $\sum p_T$ inside the cone  [GeV]")
    a2.set_ylabel("probability density")
    a2.set_title("(b) event-by-event: sometimes nothing, sometimes a lot", fontsize=10.5)
    a2.legend(loc="upper right", fontsize=8.5)
    ev, gen = st.ev, st.gen
    orig = core.origins(ev, gen)
    pb = ev.p[ours["row"]].p
    Rs = np.linspace(0.05, 2.0, 100)
    fin = ev.final_indices()
    pt_all, pt_hard, m_all, m_hard = [], [], [], []
    for R in Rs:
        mem = [i for i in fin if core.ps.delta_r(ev.p[i].p, pb) < R]
        pa = sum((ev.p[i].p for i in mem), np.zeros(4))
        ph = sum((ev.p[i].p for i in mem if orig.get(i) == "hard"), np.zeros(4))
        pt_all.append(core.pt(pa)); pt_hard.append(core.pt(ph))
        m_all.append(math.sqrt(max(core.mass2(pa), 0))); m_hard.append(math.sqrt(max(core.mass2(ph), 0)))
    a3.plot(Rs, pt_hard, color=COL_FSR, lw=1.8, ls="--", label=r"$p_T$ of the hard-system partons inside $R$")
    a3.plot(Rs, pt_all, color=COL_MPI, lw=2, label=r"$p_T$ of all partons inside $R$")
    a3.axhline(113.221, color=TEXT2, lw=0.8, ls=":")
    a3.text(1.95, 114, "Born: 113.2", ha="right", fontsize=9, color=TEXT2)
    a3b = a3.twinx()
    a3b.plot(Rs, m_hard, color=COL_FSR, lw=1.0, ls=":")
    a3b.plot(Rs, m_all, color=COL_MPI, lw=1.2, ls=":")
    a3b.set_ylabel("mass inside the cone  [GeV]  (dotted)", color=TEXT2)
    a3b.grid(False)
    for R in (0.4, 0.8):
        j = int(np.argmin(np.abs(Rs - R)))
        a3.plot(R, pt_all[j], "o", ms=6, color=COL_EVENT, mec="white", zorder=5)
        a3.annotate("$R$ = %.1f: %.1f GeV (%.1f without UE)" % (R, pt_all[j], pt_hard[j]), xy=(R, pt_all[j]), xytext=(8, -16 if R == 0.4 else -30),
                    textcoords="offset points", fontsize=9, color=COL_EVENT)
    a3.set_xlabel(r"cone radius $R$ around the $b$ quark")
    a3.set_ylabel(r"vector-summed $p_T$ inside the cone  [GeV]")
    a3.set_title("(c) the ledger of our event against the cone radius", fontsize=10.5)
    a3.set_ylim(60, 140)
    a3.legend(loc="upper left", fontsize=9)
    fig.suptitle("Our jet at this stage: UE energy inside $R$ = 0.4 and $R$ = 0.8 (page 8)", fontsize=12)
    fig.tight_layout()
    return fig


PAGES = {1: "cross section", 2: "overlap", 3: "chain", 4: "flavours", 5: "remnants", 6: "reconnection",
         7: "underlying event", 8: "our jet"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, default=HERE / "theory_multiparton_interactions.pdf")
    ap.add_argument("--page", type=int, default=None, help="render one page only (1-8)")
    ap.add_argument("--impact", type=float, default=CMSSW_B)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=160)
    a = ap.parse_args()
    st = Setup(a.impact, a.seed)
    print(f"seed-{a.seed} event at b = {a.impact} <b>: {st.summary['n_mpi']} systems, {st.summary['n_final']} final partons; "
          f"statistics files: sampled {'yes' if st.stats_s else 'no'}, fixed b {'yes' if st.stats_f else 'no'}; "
          f"PYTHIA: sampled {'yes' if st.py_s else 'no'}, fixed b {'yes' if st.py_f else 'no'}")
    makers = {1: lambda: page_cross_section(st), 2: lambda: page_overlap(st), 3: lambda: page_chain(st),
              4: lambda: page_flavours(st), 5: lambda: page_remnants(st), 6: lambda: page_reconnection(st),
              7: lambda: page_underlying(st), 8: lambda: page_our_jet(st)}
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
