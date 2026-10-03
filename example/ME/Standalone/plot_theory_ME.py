#!/usr/bin/env python3
"""The plots behind the theory sections of chapter 1, drawn from the standalone integrator.

One page per theory (sub)section of the chapter, every curve computed with the
tables, interpolators and born_weights() of standalone_qqbar_bbbar_xsec.py, so
that each figure can be checked against the -v log of that script
(standalone_qqbar_bbbar_xsec.log). The event followed through the book
(../event_ME.lhe) is marked on every page.

  page 1  factorisation   where the cross section comes from: d sigma / d log10 sqrt(sHat)
                          per incoming flavour, and d sigma / dY; our event at 396 GeV, Y = -2.59
  page 2  variables       the allowed triangle in (log x1, log x2) with lines of constant tau
                          and Y, and why ln tau is sampled: the integral per bin of u1 against
                          what a linear map in tau would give
  page 3  pdfs            x f(x, Q^2) at the event's scale with x1 (sea ubar) and x2 (valence u)
                          marked, and how the two PDF values move with Q^2 (the mu_F dependence)
  page 4  matrix element  d sigma_hat / d cos theta at the event's sHat with the pTHat cut shaded,
                          sigma_hat(sqrt sHat) with the threshold, and c_max and the surviving
                          fraction as a function of sqrt(sHat)
  page 5  flavours        the share of each incoming flavour as a function of the pTHat cut
  page 6  scales          alpha_s(Q^2) of the embedded table and the Q sampled by the integral
  page 7  convergence     the weight distribution of one Sobol pass, the running average, and the
                          error against N for scrambled Sobol and pseudo-random points
  page 8  scale variation sigma against the mu_R, mu_F and common scale factors from
                          scale_scan.json (written by scan_scales.py with the full integrator)

    python3 plot_theory_ME.py                        # -> theory_qqbar_bbbar.pdf (all pages)
    python3 plot_theory_ME.py --page 4 -o me.png     # one page as PNG
    python3 plot_theory_ME.py --power 16             # faster, coarser histograms

Needs numpy, scipy and matplotlib (on this Mac: /usr/bin/python3). About 40 s with
the default 2^18 points.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import standalone_qqbar_bbbar_xsec as core  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

# --- appearance: the colours of animate_standalone_qqbar_bbbar_xsec.py and plot_journey_ME.py ----
COL_B, COL_BBAR = "#ff0000", "#0000ff"
COL_U, COL_UBAR = "#2f8f5b", "#8e5bb5"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d6d3ca"
FLAVOUR_COLOURS = {"d": "#9a6b2f", "u": "#2a78d6", "s": "#2f8f5b", "c": "#eb6834", "b": "#8e5bb5"}
COL_EVENT = "#c0392b"
COL_CUT = "#f3c6c6"
COL_SOBOL, COL_RANDOM = "#2a78d6", "#9a6b2f"
COL_MUR, COL_MUF, COL_BOTH = "#c0392b", "#2a78d6", "#52514e"
PYTHIA_PB, PYTHIA_ERR_PB = 345.1, 6.1

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "legend.fontsize": 10,
    "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 100,
})

ECM, MB, PTMIN, SEED = 13600.0, 4.8, 100.0, 24680
NAMES = core.FLAVOUR_NAMES


# --- the integrand on the cube, exactly as born_weights maps it ------------------------------------
class Setup:
    def __init__(self, ecm=ECM, mb=MB, ptmin=PTMIN):
        self.ecm, self.mb, self.ptmin = ecm, mb, ptmin
        self.s, self.mb2 = ecm**2, mb**2
        self.lo = 4.0 * (self.mb2 + ptmin**2)
        self.taumin = self.lo / self.s
        self.logwidth = math.log(1.0 / self.taumin)
        self.pdf = core.TablePDF(core.TABLES)
        self.alpha = core.TableAlphaS(core.TABLES)

    def map_cube(self, u):
        logtau = math.log(self.taumin) + self.logwidth * u[:, 0]
        tau = np.exp(logtau)
        ymax = -0.5 * logtau
        y = (2.0 * u[:, 1] - 1.0) * ymax
        shat = tau * self.s
        p2 = shat / 4.0 - self.mb2
        cmax = np.sqrt(np.clip(1.0 - self.ptmin**2 / p2, 0.0, None))
        c = u[:, 2] * cmax
        pt2 = p2 * (1.0 - c * c)
        return dict(logtau=logtau, tau=tau, y=y, ymax=ymax, shat=shat, x1=np.exp(0.5 * logtau + y),
                    x2=np.exp(0.5 * logtau - y), cos=c, cmax=cmax, pt=np.sqrt(pt2), q2=pt2 + self.mb2)

    def weights(self, u):
        return core.born_weights(u, self.ecm, self.mb, self.ptmin, None, self.pdf, self.alpha)

    def sobol(self, power, seed=SEED):
        return qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(power)


def event_point(st: Setup):
    ev = core.read_event(HERE.parent / "event_ME.lhe")
    if ev is None:
        sys.exit("cannot read ../event_ME.lhe")
    return core.event_kinematics(ev, st.ecm, st.mb, st.ptmin)


def mark_event(ax, x, label, color=COL_EVENT, y=0.93, ha="left", dx=4):
    ax.axvline(x, color=color, lw=1.2, ls="--")
    ax.annotate(label, xy=(x, y), xycoords=("data", "axes fraction"), xytext=(dx if ha == "left" else -dx, 0),
                textcoords="offset points", ha=ha, va="top", fontsize=10, color=color)


def density(values, weights, bins, nsamples):
    """sum of weights / N per unit of the binning variable: the differential cross section."""
    h, edges = np.histogram(values, bins=bins, weights=weights)
    return h / nsamples / np.diff(edges), edges


def steps(ax, edges, h, **kw):
    return ax.step(edges, np.append(h, h[-1]), where="post", **kw)


# --- page 1: factorisation ------------------------------------------------------------------------
def page_factorisation(st, u, w, ev):
    m = st.map_cube(u)
    n = len(u)
    tot = w.sum(axis=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    fig.suptitle(r"Where the $345\,$pb come from: the integrand of the factorisation formula, "
                 r"one Sobol pass of $2^{%d}$ points" % int(math.log2(n)), y=0.995)
    # (a) d sigma / d log10 sqrt(sHat), stacked per flavour
    lsq = 0.5 * np.log10(m["shat"])
    bins = np.linspace(math.log10(math.sqrt(st.lo)), 3.5, 70)
    base = np.zeros(len(bins) - 1)
    for i, nm in enumerate(NAMES):
        h, edges = density(lsq, w[i], bins, n)
        a1.fill_between(edges[:-1], base, base + h, step="post", color=FLAVOUR_COLOURS[nm], alpha=0.85,
                        lw=0, label=rf"${nm}\bar {nm}$: {w[i].mean():.1f} pb")
        base += h
    a1.set_xlabel(r"$\log_{10}\,\sqrt{\hat s}\;[\mathrm{GeV}]$")
    a1.set_ylabel(r"$\mathrm{d}\sigma/\mathrm{d}\log_{10}\sqrt{\hat s}$  [pb]")
    a1.set_xlim(bins[0], 3.5)
    a1.set_title(r"(a) the pair mass: most of the rate sits just above the threshold")
    mark_event(a1, math.log10(math.sqrt(ev["shat"])), r"our event: $\sqrt{\hat s}=%.0f\,$GeV" % math.sqrt(ev["shat"]))
    a1.axvline(math.log10(math.sqrt(st.lo)), color=TEXT2, lw=1)
    a1.annotate(r"threshold $\sqrt{\hat s_{\min}}=%.0f\,$GeV" % math.sqrt(st.lo), xy=(bins[0], 0.6),
                xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points", fontsize=10,
                color=TEXT2, rotation=90, va="center")
    sel = m["shat"] < 305.3**2
    a1.text(0.98, 0.55, "%.0f%% of the integral below $305\\,$GeV\n(the first tenth of $u_1$)" % (100 * tot[sel].sum() / tot.sum()),
            transform=a1.transAxes, ha="right", va="top", fontsize=10, color=TEXT2)
    a1.legend(loc="upper right", title=r"total $%.2f\,$pb" % tot.mean(), title_fontsize=10)
    # (b) d sigma / dY
    bins = np.linspace(-4.5, 4.5, 73)
    base = np.zeros(len(bins) - 1)
    for i, nm in enumerate(NAMES):
        h, edges = density(m["y"], w[i], bins, n)
        a2.fill_between(edges[:-1], base, base + h, step="post", color=FLAVOUR_COLOURS[nm], alpha=0.85, lw=0)
        base += h
    a2.set_xlabel(r"pair rapidity $Y=\frac{1}{2}\ln(x_1/x_2)$")
    a2.set_ylabel(r"$\mathrm{d}\sigma/\mathrm{d}Y$  [pb]")
    a2.set_title(r"(b) the pair rapidity: where along the beam the pair is born")
    mark_event(a2, ev["y"], "our event: $Y=%.2f$\n($x_1=%.1f\\times10^{-3}$, $x_2=%.2f$)" % (ev["y"], 1e3 * ev["x1"], ev["x2"]), ha="left")
    sel = np.abs(m["y"]) > abs(ev["y"])
    a2.text(0.98, 0.62, "%.0f%% of the pairs are boosted\nmore than ours ($|Y|>%.2f$)" % (100 * tot[sel].sum() / tot.sum(), abs(ev["y"])),
            transform=a2.transAxes, ha="right", va="top", fontsize=10, color=TEXT2)
    fig.tight_layout()
    return fig


# --- page 2: better variables ----------------------------------------------------------------------
def page_variables(st, u, w, ev):
    m = st.map_cube(u)
    n = len(u)
    tot = w.sum(axis=0)
    fig = plt.figure(figsize=(10.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])
    a1, a2, a3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    fig.suptitle(r"Better variables: the triangle in $(\log x_1,\log x_2)$ and why $\ln\tau$ is sampled", y=0.995)
    # (a) the triangle with the integrand as a density
    lx1, lx2 = np.log10(m["x1"]), np.log10(m["x2"])
    lo = math.log10(st.taumin)
    bins = np.linspace(lo, 0, 90)
    h, xe, ye = np.histogram2d(lx1, lx2, bins=[bins, bins], weights=tot)
    h = h / n / (np.diff(bins)[0] ** 2)
    im = a1.pcolormesh(xe, ye, np.ma.masked_where(h.T <= 0, h.T), cmap="viridis",
                       norm=matplotlib.colors.LogNorm(vmin=h[h > 0].max() * 1e-5, vmax=h.max()), rasterized=True)
    fig.colorbar(im, ax=a1, pad=0.02, label=r"$\mathrm{d}^2\sigma/\mathrm{d}\log_{10}x_1\mathrm{d}\log_{10}x_2$ [pb]")
    a1.plot([lo, 0], [0, lo], color=TEXT, lw=1.2)
    a1.text(lo / 2 - 0.25, lo / 2 - 0.25, r"$x_1x_2=\tau_{\min}$", rotation=-45, ha="center", va="top", fontsize=10)
    for lt in (-3, -2, -1):
        a1.plot([lt, 0], [0, lt], color="white", lw=0.7, ls=":")
    for yy in (-2, 0, 2):
        c = yy / math.log(10)
        a1.plot([lo, 0], [lo - 2 * c, -2 * c], color="white", lw=0.7, ls="--")
    a1.text(-1.0, -0.3, r"$Y=0$", color="white", fontsize=9.5, rotation=45)
    a1.text(-3.2, -0.55, r"$Y=-2$", color="white", fontsize=9.5, rotation=45)
    a1.text(-0.8, -2.9, r"$Y=+2$", color="white", fontsize=9.5, rotation=45)
    a1.text(-2.5, -0.1, r"$\tau=10^{-2}$", color="white", fontsize=9.5, rotation=-45, ha="left")
    a1.plot(math.log10(ev["x1"]), math.log10(ev["x2"]), marker="*", ms=13, color=COL_EVENT, mec="white")
    a1.annotate("our event", xy=(math.log10(ev["x1"]), math.log10(ev["x2"])), xytext=(-8, -14),
                textcoords="offset points", color=COL_EVENT, fontsize=10, ha="right")
    a1.set_xlim(lo, 0)
    a1.set_ylim(lo, 0)
    a1.set_xlabel(r"$\log_{10}x_1$ (beam $+z$)")
    a1.set_ylabel(r"$\log_{10}x_2$ (beam $-z$)")
    a1.set_title(r"(a) the allowed region (dotted: $\tau$, dashed: $Y$)")
    # (b) integral per bin of u1 with the ln tau map
    bins = np.linspace(0, 1, 25)
    h, edges = np.histogram(u[:, 0], bins=bins, weights=tot)
    frac = h / tot.sum()
    a2.bar(edges[:-1], frac, width=np.diff(edges), align="edge", color=COL_SOBOL, alpha=0.85, lw=0)
    a2.set_xlabel(r"$u_1$   ($\ln\tau=\ln\tau_{\min}+u_1\ln(1/\tau_{\min})$)")
    a2.set_ylabel("fraction of the integral per bin")
    a2.set_title(r"(b) linear in $\ln\tau$: first bin %.0f%%" % (100 * frac[0]))
    a2.set_yscale("log")
    a2.set_ylim(1e-6, 1.5)
    mark_event(a2, (ev["logtau"] - math.log(st.taumin)) / st.logwidth, r"our event, $u_1=%.3f$" % ((ev["logtau"] - math.log(st.taumin)) / st.logwidth))
    # (c) the same integral per bin of a linear tau coordinate
    v = (m["tau"] - st.taumin) / (1 - st.taumin)
    h, edges = np.histogram(v, bins=bins, weights=tot)
    frac = h / tot.sum()
    a3.bar(edges[:-1], np.clip(frac, 1e-7, None), width=np.diff(edges), align="edge", color=COL_RANDOM, alpha=0.85, lw=0)
    a3.set_xlabel(r"$v=(\tau-\tau_{\min})/(1-\tau_{\min})$, a linear map in $\tau$")
    a3.set_title(r"(c) linear in $\tau$: first bin %.2f%%" % (100 * frac[0]))
    a3.set_yscale("log")
    a3.set_ylim(1e-6, 1.5)
    a3.text(0.97, 0.9, "a uniform sample in $\\tau$ would put\n%.1f%% of its points where\n%.3f%% of the integral is" % (100 * (1 - 1 / 24), 100 * (1 - frac[0])),
            transform=a3.transAxes, ha="right", va="top", fontsize=10, color=TEXT2)
    fig.tight_layout()
    return fig


# --- page 3: parton distribution functions ---------------------------------------------------------
def page_pdfs(st, ev):
    q2 = ev["q2"]
    x = np.geomspace(1.1e-6, 0.99, 400)
    q, qb = st.pdf.xf(x, np.full_like(x, q2))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    fig.suptitle(r"The PDFs of the embedded table (NNPDF3.1 NNLO, member 0) at the event's scale $Q^2=%.0f\,$GeV$^2$" % q2, y=0.995)
    for i, nm in enumerate(NAMES):
        a1.plot(x, q[i], color=FLAVOUR_COLOURS[nm], lw=1.6, label=rf"$xf_{nm}$")
        if nm in ("d", "u", "s"):
            a1.plot(x, qb[i], color=FLAVOUR_COLOURS[nm], lw=1.2, ls="--", label=rf"$xf_{{\bar {nm}}}$")
    a1.set_xscale("log")
    a1.set_xlabel(r"$x$")
    a1.set_ylabel(r"$x\,f(x,Q^2)$")
    a1.set_ylim(0, 2.6)
    a1.set_title(r"(a) $xf(x,Q^2)$: sea rising at small $x$, valence bump at $x\sim0.2$")
    i_u = NAMES.index("u")
    xf1 = float(st.pdf.xf(np.array([ev["x1"]]), np.array([q2]))[1][i_u, 0])
    xf2 = float(st.pdf.xf(np.array([ev["x2"]]), np.array([q2]))[0][i_u, 0])
    a1.plot(ev["x1"], xf1, marker="*", ms=13, color=COL_UBAR, mec="white", zorder=5)
    a1.plot(ev["x2"], xf2, marker="*", ms=13, color=COL_U, mec="white", zorder=5)
    a1.annotate(r"our $\bar u$ from the sea: $x_1=%.2f\times10^{-3}$, $xf=%.3f$" % (1e3 * ev["x1"], xf1), xy=(ev["x1"], xf1),
                xytext=(10, 22), textcoords="offset points", fontsize=10, color=COL_UBAR,
                arrowprops=dict(arrowstyle="-", color=COL_UBAR, lw=0.8))
    a1.annotate(r"our $u$, valence: $x_2=%.3f$, $xf=%.3f$" % (ev["x2"], xf2), xy=(ev["x2"], xf2),
                xytext=(-120, 40), textcoords="offset points", fontsize=10, color=COL_U,
                arrowprops=dict(arrowstyle="-", color=COL_U, lw=0.8))
    a1.text(0.02, 0.97, r"$\bar c=c$ and $\bar b=b$ (the symmetric heavy sea);" "\n" r"negative values are clipped to zero", transform=a1.transAxes, va="top", fontsize=10, color=TEXT2)
    a1.legend(ncol=2, loc="upper right")
    # (b) Q^2 dependence of the two values that enter our event
    q2s = np.geomspace(st.pdf.lq[0] and math.exp(st.pdf.lq[0]), 2.0e5, 200)
    xf1q = st.pdf.xf(np.full_like(q2s, ev["x1"]), q2s)[1][i_u]
    xf2q = st.pdf.xf(np.full_like(q2s, ev["x2"]), q2s)[0][i_u]
    a2.plot(q2s, xf1q / xf1, color=COL_UBAR, lw=1.8, label=r"$xf_{\bar u}(x_1,Q^2)$, sea: rises with $Q^2$")
    a2.plot(q2s, xf2q / xf2, color=COL_U, lw=1.8, label=r"$xf_u(x_2,Q^2)$, valence: falls with $Q^2$")
    prod = (xf1q * xf2q) / (xf1 * xf2)
    a2.plot(q2s, prod, color=TEXT, lw=1.2, ls="--", label="their product (what the weight sees)")
    a2.set_xscale("log")
    a2.set_xlabel(r"$Q^2=\mu_{\mathrm{F}}^2$  [GeV$^2$]")
    a2.set_ylabel(r"value relative to $Q^2=%.0f\,$GeV$^2$" % q2)
    a2.set_title(r"(b) the $\mu_{\mathrm{F}}$ dependence at our two $x$ values")
    for fac, lab in ((0.25, r"$Q^2/4$"), (1.0, r"$Q^2$"), (4.0, r"$4Q^2$")):
        a2.axvline(q2 * fac, color=TEXT2, lw=0.8, ls=":")
        j = np.searchsorted(q2s, q2 * fac)
        a2.annotate("%s\n%+.1f%%\n%+.1f%%" % (lab, 100 * (xf1q[j] / xf1 - 1), 100 * (xf2q[j] / xf2 - 1)),
                    xy=(q2 * fac, 0.97), xycoords=("data", "axes fraction"), xytext=(3, 0), textcoords="offset points",
                    fontsize=9.5, va="top", color=TEXT2)
    a2.set_ylim(0.8, 1.2)
    a2.legend(loc="lower right")
    fig.tight_layout()
    return fig


# --- page 4: the matrix element ----------------------------------------------------------------------
def page_matrix_element(st, ev):
    als = float(st.alpha(ev["q2"]))
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.5, 4.8))
    fig.suptitle(r"The Born matrix element $q\bar q\to g^*\to b\bar b$ and what the $\hat p_{\mathrm{T}}\geq%.0f\,$GeV cut does to it" % st.ptmin, y=0.995)
    # (a) angular distribution at the event's sHat
    c = np.linspace(-1, 1, 400)
    d = core.dsigma_dcostheta(np.full_like(c, ev["shat"]), c, st.mb, als)
    a1.plot(c, d, color=TEXT, lw=1.8)
    a1.plot(c, math.pi * als**2 * ev["beta"] / (9 * ev["shat"]) * (1 + c**2) * core.GEV2_TO_PB, color=TEXT2, lw=1, ls=":",
            label=r"$1+\cos^2\theta$ alone")
    cm = ev["cmax"]
    a1.axvspan(-1, -cm, color=COL_CUT, lw=0)
    a1.axvspan(cm, 1, color=COL_CUT, lw=0)
    a1.text(-0.995, 0.12 * d.max(), "removed by\nthe cut", fontsize=9.5, color=COL_EVENT, va="center", ha="left")
    a1.text(0.995, 0.12 * d.max(), "removed by\nthe cut", fontsize=9.5, color=COL_EVENT, va="center", ha="right")
    dev = float(core.dsigma_dcostheta(np.array([ev["shat"]]), np.array([ev["cos"]]), st.mb, als)[0])
    a1.plot(ev["cos"], dev, marker="*", ms=13, color=COL_EVENT, mec="white", zorder=5)
    a1.annotate(r"our event: $\cos\theta^*=%.3f$" "\n" r"$\mathrm{d}\hat\sigma/\mathrm{d}\cos\theta=%.1f\,$pb" % (ev["cos"], dev),
                xy=(ev["cos"], dev), xytext=(8, -34), textcoords="offset points", ha="left", fontsize=10, color=COL_EVENT)
    prim = lambda z: (1 + ev["rho"]) * z + (1 - ev["rho"]) * z**3 / 3
    a1.set_xlabel(r"$\cos\theta$ (partonic frame)")
    a1.set_ylabel(r"$\mathrm{d}\hat\sigma/\mathrm{d}\cos\theta$  [pb]")
    a1.set_ylim(0, 1.25 * d.max())
    a1.set_title(r"(a) at $\sqrt{\hat s}=%.0f\,$GeV: $|\cos\theta|\leq%.3f$ keeps %.1f%%" % (math.sqrt(ev["shat"]), cm, 100 * prim(cm) / prim(1)), fontsize=10.5)
    a1.legend(loc="upper center")
    # (b) sigma_hat against sqrt(sHat) at fixed alpha_s
    sq = np.geomspace(2 * st.mb * 1.0005, 3000, 600)
    sh = sq**2
    rho = 4 * st.mb**2 / sh
    sig = 4 * math.pi * als**2 * np.sqrt(1 - rho) * (2 + rho) / (27 * sh) * core.GEV2_TO_PB
    a2.plot(sq, sig, color=TEXT, lw=1.8, label=r"$\hat\sigma=4\pi\alpha_{\mathrm{s}}^2\beta(2+\rho)/(27\hat s)$")
    a2.plot(sq, 8 * math.pi * als**2 / (27 * sh) * core.GEV2_TO_PB, color=TEXT2, lw=1, ls=":", label=r"massless: $8\pi\alpha_{\mathrm{s}}^2/(27\hat s)$")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.axvline(math.sqrt(st.lo), color=TEXT2, lw=1, ls="--")
    a2.text(math.sqrt(st.lo) * 1.05, 2e4, r"$\sqrt{\hat s_{\min}}=%.0f\,$GeV" "\n" r"for $\hat p_{\mathrm{T}}\geq%.0f\,$GeV" % (math.sqrt(st.lo), st.ptmin), fontsize=10, color=TEXT2)
    a2.axvline(2 * st.mb, color=TEXT2, lw=1, ls="--")
    a2.text(2 * st.mb * 1.1, 3e-1, r"threshold $2m_b$: $\beta\to0$", fontsize=10, color=TEXT2)
    sig_ev = 4 * math.pi * als**2 * ev["beta"] * (2 + ev["rho"]) / (27 * ev["shat"]) * core.GEV2_TO_PB
    a2.plot(math.sqrt(ev["shat"]), sig_ev, marker="*", ms=13, color=COL_EVENT, mec="white", zorder=5)
    a2.annotate(r"our event: $\hat\sigma=%.1f\,$pb" % sig_ev, xy=(math.sqrt(ev["shat"]), sig_ev), xytext=(8, 8), textcoords="offset points", fontsize=10, color=COL_EVENT)
    a2.set_xlabel(r"$\sqrt{\hat s}$  [GeV]")
    a2.set_ylabel(r"$\hat\sigma(q\bar q\to b\bar b)$  [pb]")
    a2.set_title(r"(b) the partonic cross section at fixed $\alpha_{\mathrm{s}}=%.4f$" % als, fontsize=10.5)
    a2.legend(loc="lower left")
    # (c) c_max and the surviving fraction against sqrt(sHat)
    sq = np.geomspace(math.sqrt(st.lo) * 1.0001, 3000, 600)
    p2 = sq**2 / 4 - st.mb**2
    cmax = np.sqrt(1 - st.ptmin**2 / p2)
    rho = 4 * st.mb**2 / sq**2
    surv = ((1 + rho) * cmax + (1 - rho) * cmax**3 / 3) / ((1 + rho) + (1 - rho) / 3)
    a3.plot(sq, cmax, color=COL_SOBOL, lw=1.8, label=r"$c_{\max}=\sqrt{1-(\hat p_{\mathrm{T}}^{\min}/p^*)^2}$")
    a3.plot(sq, surv, color=COL_B, lw=1.8, label=r"fraction of $\hat\sigma$ with $|\cos\theta|\leq c_{\max}$")
    a3.plot(sq, 1 - cmax, color=TEXT2, lw=1.2, ls="--", label=r"solid-angle fraction removed, $1-c_{\max}$")
    a3.set_xscale("log")
    a3.set_xlabel(r"$\sqrt{\hat s}$  [GeV]")
    a3.set_ylim(0, 1.05)
    a3.set_title(r"(c) the cut on $\hat p_{\mathrm{T}}$ is a cut on the angle", fontsize=10.5)
    a3.axvline(math.sqrt(ev["shat"]), color=COL_EVENT, lw=1.2, ls="--")
    a3.plot([math.sqrt(ev["shat"])] * 2, [ev["cmax"], prim(ev["cmax"]) / prim(1)], ls="none", marker="*", ms=12, color=COL_EVENT, mec="white", zorder=5)
    a3.annotate(r"our event: $c_{\max}=%.3f$, %.1f%% kept" % (ev["cmax"], 100 * prim(ev["cmax"]) / prim(1)), xy=(math.sqrt(ev["shat"]), 0.15), xytext=(8, 0), textcoords="offset points", fontsize=10, color=COL_EVENT)
    a3.legend(loc="center right")
    fig.tight_layout()
    return fig


# --- page 5: the incoming flavours ----------------------------------------------------------------------
def page_flavours(st, power):
    ptmins = [30, 40, 50, 70, 100, 150, 200, 300, 400, 600, 800]
    shares, totals = [], []
    for pt in ptmins:
        mean, _, total, _ = core.integrate(st.ecm, st.mb, pt, None, min(power, 15), 1, SEED, st.pdf, st.alpha, progress=False)
        shares.append(mean / total)
        totals.append(total)
    shares = np.array(shares).T
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    fig.suptitle(r"Which quarks make the pair: the share of each incoming flavour (single Sobol passes of the standalone script)", y=0.995)
    base = np.zeros(len(ptmins))
    for i, nm in enumerate(NAMES):
        a1.fill_between(ptmins, base, base + shares[i], color=FLAVOUR_COLOURS[nm], alpha=0.85, lw=0, label=rf"${nm}\bar {nm}$")
        base += shares[i]
    a1.set_xscale("log")
    a1.set_xlim(ptmins[0], ptmins[-1])
    a1.set_ylim(0, 1)
    a1.set_xlabel(r"$\hat p_{\mathrm{T}}^{\min}$  [GeV]")
    a1.set_ylabel("share of the cross section")
    a1.set_title(r"(a) the valence $u$ wins as the cut rises; the incoming $b$ stays below 2%")
    j = ptmins.index(100)
    a1.axvline(100, color=COL_EVENT, lw=1.2, ls="--")
    cum = 0
    for i, nm in enumerate(NAMES):
        dy = {"c": -5, "b": 6}.get(nm, 0)
        a1.annotate("%.1f%%" % (100 * shares[i, j]), xy=(100, cum + shares[i, j] / 2), xytext=(5, dy), textcoords="offset points", fontsize=10, va="center", color="white" if shares[i, j] > 0.05 else TEXT)
        cum += shares[i, j]
    a1.legend(loc="lower left", ncol=5)
    sig = [t for t in totals]
    a2.plot(ptmins, sig, color=TEXT, lw=1.8, marker="o", ms=4)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel(r"$\hat p_{\mathrm{T}}^{\min}$  [GeV]")
    a2.set_ylabel(r"$\sigma(\hat p_{\mathrm{T}}\geq\hat p_{\mathrm{T}}^{\min})$  [pb]")
    a2.set_title(r"(b) the total: a factor %.0f between 30 and 100 GeV" % (totals[0] / totals[j]))
    for k in range(len(ptmins) - 1):
        nloc = -math.log(totals[k + 1] / totals[k]) / math.log(ptmins[k + 1] / ptmins[k])
        a2.annotate("$n=%.1f$" % nloc, xy=(math.sqrt(ptmins[k] * ptmins[k + 1]), math.sqrt(totals[k] * totals[k + 1])),
                    xytext=(6, 4), textcoords="offset points", fontsize=10, color=TEXT2)
    a2.text(0.03, 0.05, r"local slope $\sigma\propto(\hat p_{\mathrm{T}}^{\min})^{-n}$;" "\n" r"$n=4$ would be dimensional analysis alone", transform=a2.transAxes, fontsize=10, color=TEXT2)
    a2.axvline(100, color=COL_EVENT, lw=1.2, ls="--")
    a2.annotate("%.1f pb" % totals[j], xy=(100, totals[j]), xytext=(8, 8), textcoords="offset points", fontsize=10, color=COL_EVENT)
    fig.tight_layout()
    return fig


# --- page 6: scales and the coupling ------------------------------------------------------------------
def page_scales(st, u, w, ev):
    m = st.map_cube(u)
    n = len(u)
    tot = w.sum(axis=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    fig.suptitle(r"The scale of the hard process: $\mu_{\mathrm{R}}^2=\mu_{\mathrm{F}}^2=\hat p_{\mathrm{T}}^2+m_b^2$ and PYTHIA's second-order $\alpha_{\mathrm{s}}$", y=0.995)
    q2s = np.geomspace(st.alpha.q2[0], st.alpha.q2[-1], 400)
    a1.plot(q2s, st.alpha(q2s), color=TEXT, lw=1.8)
    a1.set_xscale("log")
    a1.set_xlabel(r"$Q^2=\mu_{\mathrm{R}}^2$  [GeV$^2$]")
    a1.set_ylabel(r"$\alpha_{\mathrm{s}}(Q^2)$")
    a1.set_title(r"(a) the embedded table (TableAlphaS): $\alpha_{\mathrm{s}}(m_Z^2)=0.118$, two loops")
    for q2, lab, col in ((91.1876**2, r"$m_Z^2$: %.4f" % float(st.alpha(91.1876**2)), TEXT2),
                         (ev["q2"] / 4, r"$Q^2/4$: %.4f" % float(st.alpha(ev["q2"] / 4)), COL_MUR),
                         (ev["q2"], r"our event $Q^2=%.0f$: %.4f" % (ev["q2"], float(st.alpha(ev["q2"]))), COL_EVENT),
                         (4 * ev["q2"], r"$4Q^2$: %.4f" % float(st.alpha(4 * ev["q2"])), COL_MUR)):
        a1.plot(q2, float(st.alpha(q2)), marker="o", ms=6, color=col, mec="white", zorder=5)
        a1.annotate(lab, xy=(q2, float(st.alpha(q2))), xytext=(8, 4), textcoords="offset points", fontsize=10, color=col)
    a1.text(0.03, 0.08, r"$\alpha_{\mathrm{s}}^2$ moves by $%+.0f\%%$ and $%+.0f\%%$ for $\mu_{\mathrm{R}}\to\mu_{\mathrm{R}}/2$ and $2\mu_{\mathrm{R}}$" % (
        100 * (float(st.alpha(ev["q2"] / 4)) ** 2 / float(st.alpha(ev["q2"])) ** 2 - 1),
        100 * (float(st.alpha(4 * ev["q2"])) ** 2 / float(st.alpha(ev["q2"])) ** 2 - 1)), transform=a1.transAxes, fontsize=10, color=TEXT2)
    # (b) the Q actually sampled by the integral
    lq = 0.5 * np.log10(m["q2"])
    bins = np.linspace(2.0, 3.6, 64)
    base = np.zeros(len(bins) - 1)
    for i, nm in enumerate(NAMES):
        h, edges = density(lq, w[i], bins, n)
        a2.fill_between(edges[:-1], base, base + h, step="post", color=FLAVOUR_COLOURS[nm], alpha=0.85, lw=0)
        base += h
    a2.set_xlabel(r"$\log_{10}\,Q\;[\mathrm{GeV}]$, $Q^2=\hat p_{\mathrm{T}}^2+m_b^2$")
    a2.set_ylabel(r"$\mathrm{d}\sigma/\mathrm{d}\log_{10}Q$  [pb]")
    a2.set_title(r"(b) the scales the integral actually visits")
    mark_event(a2, 0.5 * math.log10(ev["q2"]), r"our event: $Q=%.1f\,$GeV" % math.sqrt(ev["q2"]))
    a2.axvline(math.log10(st.ptmin), color=TEXT2, lw=1)
    med = np.sqrt(np.exp(np.interp(0.5, np.cumsum(tot[np.argsort(m["q2"])]) / tot.sum(), np.log(np.sort(m["q2"])))))
    a2.text(0.98, 0.93, "half of the cross section has\n$Q<%.0f\\,$GeV; the cut is at %.0f GeV" % (med, st.ptmin), transform=a2.transAxes, ha="right", va="top", fontsize=10, color=TEXT2)
    fig.tight_layout()
    return fig


# --- page 7: from formula to number ----------------------------------------------------------------------
def page_convergence(st, u, w, ev, power):
    tot = w.sum(axis=0)
    n = len(u)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.5, 4.8))
    fig.suptitle(r"From formula to number: the weights of one Sobol pass, the running average and the error against $N$", y=0.995)
    # (a) the weight distribution
    pos = tot[tot > 0]
    bins = np.linspace(math.log10(pos.min()), math.log10(pos.max()), 80)
    a1.hist(np.log10(pos), bins=bins, color=COL_SOBOL, alpha=0.85)
    a1.set_yscale("log")
    a1.set_xlabel(r"$\log_{10}\,(\sum_q w_q)$  [pb]")
    a1.set_ylabel("points per bin")
    a1.set_title(r"(a) one pass: median %.1f, mean %.1f, max %.0f pb" % (np.median(tot), tot.mean(), tot.max()), fontsize=10.5)
    a1.axvline(math.log10(tot.mean()), color=TEXT, lw=1.2, ls="--")
    a1.annotate(r"mean $=\sigma$", xy=(math.log10(tot.mean()), 0.93), xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points", fontsize=10)
    wev = 650.742
    a1.axvline(math.log10(wev), color=COL_EVENT, lw=1.2, ls="--")
    a1.annotate("our event's point:\n%.0f pb" % wev, xy=(math.log10(wev), 0.75), xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points", fontsize=10, color=COL_EVENT)
    srt = np.sort(tot)[::-1]
    a1.text(0.03, 0.05, "the largest 10%% of the weights\ncarry %.0f%% of the integral" % (100 * srt[: n // 10].sum() / tot.sum()), transform=a1.transAxes, fontsize=10, color=TEXT2)
    # (b) running average
    ks = np.arange(2, int(math.log2(n)) + 1)
    ns = 2**ks
    run = np.array([tot[:m].mean() for m in ns])
    rng = np.random.default_rng(SEED)
    wr = st.weights(rng.random((n, 3))).sum(axis=0)
    runr = np.array([wr[:m].mean() for m in ns])
    a2.axhspan(PYTHIA_PB - PYTHIA_ERR_PB, PYTHIA_PB + PYTHIA_ERR_PB, color="#e8e6df", lw=0, label=r"PYTHIA in CMSSW: $%.1f\pm%.1f\,$pb" % (PYTHIA_PB, PYTHIA_ERR_PB))
    a2.plot(ns, run, color=COL_SOBOL, lw=1.8, marker="o", ms=3.5, label="scrambled Sobol (the script)")
    a2.plot(ns, runr, color=COL_RANDOM, lw=1.2, marker="s", ms=3, label="pseudo-random points")
    a2.set_xscale("log", base=2)
    a2.set_xlabel(r"$N$ points")
    a2.set_ylabel(r"running average  [pb]")
    a2.set_ylim(300, 400)
    a2.set_title(r"(b) the running average settles at %.2f pb" % tot.mean(), fontsize=10.5)
    a2.legend(loc="upper right")
    # (c) error against N from R scrambles / seeds
    R = 8
    ests = np.array([[st.weights(st.sobol(int(ks[-1]), SEED + r)).sum(axis=0)[:m].mean() for m in ns] for r in range(R)])
    estr = np.array([[st.weights(np.random.default_rng(SEED + r).random((n, 3))).sum(axis=0)[:m].mean() for m in ns] for r in range(R)])
    err_s = ests.std(axis=0, ddof=1)
    err_r = estr.std(axis=0, ddof=1)
    a3.plot(ns, err_s, color=COL_SOBOL, lw=1.8, marker="o", ms=3.5, label="scrambled Sobol, %d scrambles" % R)
    a3.plot(ns, err_r, color=COL_RANDOM, lw=1.2, marker="s", ms=3, label="pseudo-random, %d seeds" % R)
    a3.plot(ns, err_r[3] * np.sqrt(ns[3] / ns), color=COL_RANDOM, lw=0.9, ls=":", label=r"$\propto N^{-1/2}$")
    a3.plot(ns, err_s[3] * (ns[3] / ns), color=COL_SOBOL, lw=0.9, ls=":", label=r"$\propto N^{-1}$")
    a3.set_xscale("log", base=2)
    a3.set_yscale("log")
    a3.set_xlabel(r"$N$ points")
    a3.set_ylabel(r"spread of the estimates  [pb]")
    a3.set_title(r"(c) spread at $2^{16}$: %.3f (Sobol), %.2f pb (random)" % (err_s[ks.tolist().index(16)] if 16 in ks else err_s[-1], err_r[ks.tolist().index(16)] if 16 in ks else err_r[-1]), fontsize=10)
    a3.legend(loc="lower left")
    fig.tight_layout()
    return fig


# --- page 8: scale variation (full integrator) -----------------------------------------------------------
def page_scale_variation(st, ev):
    path = HERE / "scale_scan.json"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    fig.suptitle(r"Scale variations of the leading-order cross section (qqbar_bbbar_xsec.py through scan_scales.py)", y=0.995)
    if not path.exists():
        a1.text(0.5, 0.5, "scale_scan.json missing:\nrun .venv/bin/python scan_scales.py", ha="center", va="center", transform=a1.transAxes)
        return fig
    data = json.loads(path.read_text())
    central = [p for p in data["points"] if p["mode"] == "both" and abs(p["factor"] - 1) < 1e-9][0]["sigma_pb"]
    for mode, col, lab in (("mur", COL_MUR, r"$\mu_{\mathrm{R}}$ only"), ("muf", COL_MUF, r"$\mu_{\mathrm{F}}$ only"), ("both", COL_BOTH, r"$\mu_{\mathrm{R}}=\mu_{\mathrm{F}}$ together")):
        pts = sorted((p for p in data["points"] if p["mode"] == mode), key=lambda p: p["factor"])
        f = np.array([p["factor"] for p in pts])
        sg = np.array([p["sigma_pb"] for p in pts])
        a1.plot(f, sg, color=col, lw=1.8, marker="o", ms=4, label=lab)
        for ff, ss in zip(f, sg):
            if abs(math.log2(ff)) == 1:
                a1.annotate("%.1f (%+.1f%%)" % (ss, 100 * (ss / central - 1)), xy=(ff, ss), xytext=(0, 7 if ss > central else -13),
                            textcoords="offset points", ha="center", fontsize=10, color=col)
    a1.set_xscale("log", base=2)
    a1.set_xticks([0.5, 2**-0.5, 1, 2**0.5, 2])
    a1.set_xticklabels([r"$\frac{1}{2}$", r"$\frac{1}{\sqrt{2}}$", "1", r"$\sqrt{2}$", "2"])
    a1.set_xlabel(r"scale factor $\mu/\mu_0$,  $\mu_0^2=\hat p_{\mathrm{T}}^2+m_b^2$")
    a1.set_ylabel(r"$\sigma$  [pb]")
    a1.axhline(central, color=TEXT2, lw=0.8, ls=":")
    a1.axhspan(PYTHIA_PB - PYTHIA_ERR_PB, PYTHIA_PB + PYTHIA_ERR_PB, color="#e8e6df", lw=0)
    a1.text(2 ** -0.95, PYTHIA_PB + 7, "PYTHIA in CMSSW\n$\\pm1.8\\%$", fontsize=9.5, color=TEXT2, ha="left")
    a1.set_title(r"(a) $\mu_{\mathrm{R}}$ moves the rate by $-16\%$ to $+23\%$, $\mu_{\mathrm{F}}$ by $\pm4\%$, in opposite directions")
    a1.legend(loc="upper right")
    # (b) the seven-point grid
    grid = {}
    for p in data["points"]:
        f = p["factor"]
        if abs(math.log2(f)) in (0.0, 1.0):
            r = f if p["mode"] in ("mur", "both") else 1.0
            ff = f if p["mode"] in ("muf", "both") else 1.0
            grid[(r, ff)] = p["sigma_pb"]
    vals = [v for k, v in grid.items()]
    for (r, ff), v in grid.items():
        col = plt.cm.RdBu_r((v - central) / 90 + 0.5)
        a2.add_patch(plt.Rectangle((math.log2(r) - 0.5, math.log2(ff) - 0.5), 1, 1, color=col))
        a2.text(math.log2(r), math.log2(ff), "%.1f\n%+.1f%%" % (v, 100 * (v / central - 1)), ha="center", va="center", fontsize=9)
    for r, ff in ((2.0, 0.5), (0.5, 2.0)):
        a2.add_patch(plt.Rectangle((math.log2(r) - 0.5, math.log2(ff) - 0.5), 1, 1, color="#f0eee8", hatch="//", ec="#cfcdc5"))
        a2.text(math.log2(r), math.log2(ff), "not used\n(opposite extremes)", ha="center", va="center", fontsize=10, color=TEXT2)
    a2.set_xlim(-1.5, 1.5)
    a2.set_ylim(-1.5, 1.5)
    a2.set_xticks([-1, 0, 1])
    a2.set_xticklabels([r"$\mu_{\mathrm{R}}/2$", r"$\mu_{\mathrm{R}}$", r"$2\mu_{\mathrm{R}}$"])
    a2.set_yticks([-1, 0, 1])
    a2.set_yticklabels([r"$\mu_{\mathrm{F}}/2$", r"$\mu_{\mathrm{F}}$", r"$2\mu_{\mathrm{F}}$"])
    a2.grid(False)
    a2.set_aspect("equal")
    a2.set_title(r"(b) the seven-point envelope: $%.0f$ to $%.0f\,$pb, $^{+%.0f\%%}_{-%.0f\%%}$" % (min(vals), max(vals), 100 * (max(vals) / central - 1), 100 * (1 - min(vals) / central)))
    fig.tight_layout()
    return fig


PAGES = {1: "factorisation", 2: "variables", 3: "pdfs", 4: "matrix element", 5: "flavours",
         6: "scales", 7: "convergence", 8: "scale variation"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, default=HERE / "theory_qqbar_bbbar.pdf")
    ap.add_argument("--page", type=int, default=None, help="render one page only (1-8)")
    ap.add_argument("--power", type=int, default=18, help="2**power Sobol points for the histograms")
    ap.add_argument("--dpi", type=int, default=160)
    a = ap.parse_args()
    st = Setup()
    ev = event_point(st)
    u = st.sobol(a.power)
    w = st.weights(u)
    print(f"{2**a.power} Sobol points: sigma = {w.sum(axis=0).mean():.3f} pb; event at sqrt(sHat) = {math.sqrt(ev['shat']):.1f} GeV, Y = {ev['y']:.3f}")
    makers = {
        1: lambda: page_factorisation(st, u, w, ev),
        2: lambda: page_variables(st, u, w, ev),
        3: lambda: page_pdfs(st, ev),
        4: lambda: page_matrix_element(st, ev),
        5: lambda: page_flavours(st, a.power),
        6: lambda: page_scales(st, u, w, ev),
        7: lambda: page_convergence(st, u[: 2**16], w[:, : 2**16], ev, 16),
        8: lambda: page_scale_variation(st, ev),
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
