#!/usr/bin/env python3
"""The plots behind the theory sections of chapter 8, drawn from the standalone pileup script.

One page per theory section of the chapter. Every curve is computed with the functions of
standalone_pileup.py (mu_of, the LuminousRegion, grid_rho, the fill profiles), every marker is a number of
the seed-1 record (../event_PU.lhe and standalone_pileup.json), the library panels read
../pileup_library.lhe.gz, and the ensemble panels read standalone_pileup_stats.json written by --repeat.

  page 1  the crowd's size     mu against the luminosity for the three scenarios; the Poisson distribution of
                               the collisions per crossing with the sixteen draws of our event; the flat profile
                               of the Run-3 production against a 2024-like year of fills, and the reweighting
  page 2  one member           the multiplicity of a minimum-bias collision by process type; the particles
                               per unit of eta; the transverse momentum per unit area per collision, the
                               slope of rho
  page 3  the beam spot        the collision density in (z, t) with the vertices of our crossing; the z of the
                               960 vertices against the model and the 2024 CMSSW library; the sixteen crossings
                               on the clock
  page 4  rho and the area     the grid cells and the kT jets with their medians; rho against the number of
                               collisions over the ensemble; the truth against rho A for our jet

    python3 plot_theory_PU.py                    # -> theory_pileup.pdf
    python3 plot_theory_PU.py --page 3 -o o.png  # one page as PNG

Needs numpy and matplotlib (on this Mac: /usr/bin/python3). A few seconds.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import standalone_pileup as core  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

COL_B, COL_BBAR, COL_EVENT, COL_CMSSW = "#ff0000", "#0000ff", "#c0392b", "#1f1f1f"
COL_EARLY, COL_IN, COL_LATE = "#b8860b", "#5aa9e6", "#fa8072"      # the colours of the CMSSW pages: early / in-time / late
COL_ND, COL_SD, COL_DD, COL_CD = "#2a78d6", "#2f8f5b", "#eb6834", "#8e5bb5"
COL_PU, COL_SIG, COL_MODEL = "#8a7f5a", "#ff0000", "#1b9e77"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d6d3ca"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "legend.fontsize": 9.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.edgecolor": TEXT2, "pdf.fonttype": 42, "mathtext.default": "regular", "figure.dpi": 100})


class Setup:
    def __init__(self, record: Path, library: Path, summary: Path, stats: Path):
        self.summary_lines, self.collisions, self.particles, self.jets, self.signal_of = core.read_pileup_tables(record)
        self.s = json.loads(summary.read_text()) if summary.exists() else None
        self.lib = core.Library(library)
        self.stats = json.loads(stats.read_text())["results"] if stats.exists() else None
        self.region = core.region_of(self.s.get("region_model", "run3") if self.s else "run3")
        self.mu = self.s["mu"] if self.s else 60.0
        self.n_bx = {int(k): v for k, v in self.s["n_bx"].items()} if self.s else {}
        self.scenario = core.SCENARIOS[self.s.get("scenario", "run3") if self.s else "run3"]


# the 2024 CMSSW library of the companion (PULib.pdf, 250 events): the z and t of its vertices in 4 cm and 100 ps bins
CMSSW_LIB_Z = (np.arange(-16, 17, 4.0), [0, 4, 29, 93, 92, 28, 4, 0])
CMSSW_LIB_T = (np.arange(-0.4, 0.41, 0.1), [4, 6, 46, 81, 67, 35, 8, 3])
CMSSW_LIB = dict(n=250, nstable_mean=194, nstable_median=132, sigma_z=3.56, sigma_t=0.122)


def mark(ax, x, label, color=COL_EVENT, y=0.93, ha="left", dx=4, ls="--"):
    ax.axvline(x, color=color, lw=1.1, ls=ls)
    ax.annotate(label, (x, y), xycoords=("data", "axes fraction"), xytext=(dx if ha == "left" else -dx, 0),
                textcoords="offset points", fontsize=8.5, color=color, ha=ha, va="top")


# ---------------------------------------------------------------------------
def page_scenario(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    # (a) mu against the luminosity
    L = np.logspace(33.5, 35.0, 200)
    for nb, col, ls in ((core.SCENARIOS["run3"]["n_bunch"], COL_MODEL, "-"), (core.SCENARIOS["ultimate"]["n_bunch"], COL_MODEL, "--")):
        a1.plot(L, core.mu_of(L, nb), ls, color=col, lw=2, label=rf"$\mu = L\,\sigma_{{inel}} / (n_b f_{{rev}})$, $\sigma_{{inel}}$ = {core.SIGMA_INEL_MB} mb, $n_b$ = {nb}")
    for key, sc in core.SCENARIOS.items():
        m = core.mu_of(sc["lumi"], sc["n_bunch"])
        a1.plot([sc["lumi"]], [m], "o", color=COL_EVENT, ms=7, mec="white")
        a1.plot([sc["lumi"]], [sc["mu"]], "s", color=COL_CMSSW, ms=6, mfc="none")
        a1.annotate(f"{sc['label'].split(',')[0]}: {m:.0f} (mixed at {sc['mu']:g})", (sc["lumi"], m), xytext=(-8, 8 if key != "ultimate" else -16), textcoords="offset points",
                    fontsize=8.5, ha="right", color=COL_EVENT)
    a1.set_xscale("log")
    a1.set_xlabel(r"instantaneous luminosity $L$  [cm$^{-2}$ s$^{-1}$]")
    a1.set_ylabel(r"$\mu$, collisions per bunch crossing")
    a1.set_title("(a) how many: the luminosity sets the crowd", fontsize=10.5)
    a1.legend(loc="upper left", fontsize=8)
    a1.plot([], [], "o", color=COL_EVENT, label="from the formula")
    a1.plot([], [], "s", color=COL_CMSSW, mfc="none", label="the CMSSW mixing scenario")
    a1.legend(loc="upper left", fontsize=8)
    a1.text(0.98, 0.04, f"2024: $L$ = 2 x 10$^{{34}}$, 2350 colliding bunches, levelled;\nthe simulation mixes at $\\mu$ = 60 (AVE_60_BX_25ns),\nthe production draws $\\mu_{{true}}$ flat in 55 ... 75",
            transform=a1.transAxes, ha="right", va="bottom", fontsize=8)
    # (b) Poisson
    n = np.arange(0, 320)
    other = 200.0 if st.mu < 150 else 60.0
    for mu, col, lab in ((st.mu, COL_MODEL, f"Poisson, $\\mu$ = {st.mu:g}" + (" (2024)" if st.mu == 60 else "")),
                         (other, COL_CMSSW, f"Poisson, $\\mu$ = {other:g}" + (" (HL-LHC)" if other == 200 else " (Run 3)"))):
        logp = n * math.log(mu) - mu - np.array([math.lgamma(k + 1) for k in n])
        a2.plot(n, np.exp(logp), "-", color=col, lw=1.8, label=lab)
    for bx, k in sorted(st.n_bx.items()):
        col = COL_EARLY if bx < 0 else COL_IN if bx == 0 else COL_LATE
        a2.axvline(k, color=col, lw=1.6 if bx == 0 else 0.8, ls="-" if bx == 0 else ":", alpha=1.0 if bx == 0 else 0.7)
    if 0 in st.n_bx:
        a2.annotate(f"in time, bx 0: {st.n_bx[0]}", (st.n_bx[0], 0.062), fontsize=8.5, color=COL_IN, rotation=90, ha="right", va="top")
    a2.set_xlabel("collisions in one bunch crossing, $n$")
    a2.set_ylabel("probability")
    a2.set_ylim(0, 0.065)
    a2.set_xlim(0, 260 if st.mu < 150 else 320)
    a2.set_title(f"(b) the draw: $n \\sim$ Poisson($\\mu$), $\\mu \\pm \\sqrt{{\\mu}}$ = {st.mu:g} $\\pm$ {math.sqrt(st.mu):.0f}", fontsize=10.5)
    a2.legend(loc="upper right", fontsize=8)
    if st.n_bx:
        tot = sum(st.n_bx.values())
        bxs = sorted(st.n_bx)
        a2.text(0.98, 0.80, f"our event: {tot} collisions in the {len(st.n_bx)} crossings\nbx = {bxs[0]:+d} ... {bxs[-1]:+d} (dotted: early dark yellow, late salmon)\nfor an expectation of {len(st.n_bx) * st.mu:g}; from {min(st.n_bx.values())} to {max(st.n_bx.values())}",
                transform=a2.transAxes, ha="right", va="top", fontsize=8.5)
    # (c) the profile of a fill, the production's flat profile, and the weight
    edges = np.arange(0, 101, 1.0)
    c = 0.5 * (edges[1:] + edges[:-1])
    mu_fill, w_fill = core.fill_profile(core.SCENARIOS["run3"]["mu"], 12.0, 8.0, 10.0)
    p_fill = core.profile_histogram(mu_fill, w_fill, edges)
    year = core.year_profile(np.arange(57, 68), 12.0, 8.0, 10.0, edges)
    flat = core.flat_profile(*core.FLAT_PROFILE, edges)
    a3.bar(c, year, width=1, color=COL_MODEL, alpha=0.55, label="a 2024-like year: fills levelled at 57 ... 67 for 8 h, then decaying (a model)")
    a3.step(edges[:-1], p_fill, where="post", color=COL_MODEL, lw=1.2, ls=":", label="one fill, levelled at 60")
    a3.step(edges[:-1], flat, where="post", color=COL_CMSSW, lw=1.8, label="the production's flat profile: $\\mu_{true}$ uniform in 55 ... 75")
    a3.set_xlim(25, 85)
    a3.set_ylim(0, max(year.max(), flat.max()) * 2.6)
    a3.set_xlabel(r"$\mu_{true}$ (luminosity-weighted: every collision counts once)")
    a3.set_ylabel("fraction of the collisions")
    a3.set_title("(c) the profiles, and the weight $w(\\mu_{true}) = P_{2024} / P_{sample}$", fontsize=10.5)
    a3.legend(loc="upper left", fontsize=7.4)
    aw = a3.twinx()
    w = np.array([core.reweight(m, year, flat, edges) for m in c])
    sel = (c >= core.FLAT_PROFILE[0]) & (c <= core.FLAT_PROFILE[1])
    aw.plot(c[sel], w[sel], "-", color=COL_EVENT, lw=2, label="$w$ of the flat sample")
    aw.plot([st.mu], [core.reweight(st.mu, year, flat, edges)], "*", color=COL_EVENT, ms=13, mec="white")
    aw.set_ylabel("weight $w$ of an event of the flat sample", color=COL_EVENT)
    aw.tick_params(axis="y", colors=COL_EVENT)
    aw.set_ylim(0, 3.2)
    aw.grid(False)
    a3.text(0.98, 0.70, f"mean $\\mu$: {float((c * year).sum()):.0f} (year), {float((mu_fill * w_fill).sum()):.0f} (one fill)\n$w$ = {core.reweight(st.mu, year, flat, edges):.2f} at $\\mu_{{true}}$ = {st.mu:g}, 0 where the year\nnever went; a sample at a fixed $\\mu$ = {st.mu:g}\ncovers one bin and cannot be reweighted",
            transform=a3.transAxes, ha="right", va="top", fontsize=8.2)
    fig.suptitle(f"The crowd's size: $\\mu$ = {core.mu_of(st.scenario['lumi'], st.scenario['n_bunch']):.0f} from the formula, {st.mu:g} in the mixing, {st.n_bx.get(0, '?')} in our crossing (page 1)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ---------------------------------------------------------------------------
def page_member(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    ev = st.lib.events
    kinds = ["ND", "SD", "DD", "CD"]
    cols = dict(ND=COL_ND, SD=COL_SD, DD=COL_DD, CD=COL_CD)
    # (a) multiplicity by process type
    edges = np.concatenate([[0], np.logspace(0, 3, 31)])
    a1.hist([[e["nfinal"] for e in ev if e["kind"] == k] for k in kinds], bins=edges, stacked=True, color=[cols[k] for k in kinds],
            label=[f"{k}: {sum(1 for e in ev if e['kind'] == k)} of {len(ev)}, mean {np.mean([e['nfinal'] for e in ev if e['kind'] == k]):.0f} particles" for k in kinds])
    a1.set_xscale("log")
    a1.axvline(np.mean([e["nfinal"] for e in ev]), color=COL_MODEL, lw=1.5, ls="--")
    a1.text(np.mean([e["nfinal"] for e in ev]) * 1.08, a1.get_ylim()[1] * 0.97, f"mean {np.mean([e['nfinal'] for e in ev]):.0f},\nmedian {np.median([e['nfinal'] for e in ev]):.0f}",
            fontsize=8.5, color=COL_MODEL, va="top")
    a1.axvline(CMSSW_LIB["nstable_mean"], color=COL_CMSSW, lw=1.2, ls=":")
    a1.text(CMSSW_LIB["nstable_mean"] * 1.08, a1.get_ylim()[1] * 0.80, f"CMSSW library (250):\nmean {CMSSW_LIB['nstable_mean']}, median {CMSSW_LIB['nstable_median']}", fontsize=8, color=COL_CMSSW, ha="left", va="top")
    a1.axvline(749, color=COL_SIG, lw=1.2, ls=":")
    a1.text(749 * 0.92, a1.get_ylim()[1] * 0.60, "our signal\nevent: 749", fontsize=8, color=COL_SIG, ha="right", va="top")
    a1.set_xlabel("final particles in one minimum-bias collision")
    a1.set_ylabel("collisions of the library")
    a1.set_title("(a) one member of the crowd: from 3 to 900 particles", fontsize=10.5)
    a1.legend(loc="upper left", fontsize=7.8)
    # (b) dN/deta per collision, all and charged, and the signal
    eb = np.linspace(-10, 10, 81)
    c = 0.5 * (eb[1:] + eb[:-1])
    all_eta = np.concatenate([e["eta"] for e in ev])
    ch_eta = np.concatenate([e["eta"][e["ch"]] for e in ev])
    h_all, _ = np.histogram(all_eta, bins=eb)
    h_ch, _ = np.histogram(ch_eta, bins=eb)
    h_all, h_ch = h_all / len(ev) / (eb[1] - eb[0]), h_ch / len(ev) / (eb[1] - eb[0])
    a2.step(c, h_all, where="mid", color=COL_PU, lw=1.8, label="all final particles, per collision")
    a2.step(c, h_ch, where="mid", color=COL_ND, lw=1.8, label="charged, per collision")
    sig_eta = np.array([math.asinh(p["pz"] / math.hypot(p["px"], p["py"])) if math.hypot(p["px"], p["py"]) > 0 else 9.9 * np.sign(p["pz"]) for p in st.particles]) if False else None
    for x in (-5, 5):
        a2.axvline(x, color=GRID, lw=2)
    for x in (-2.5, 2.5):
        a2.axvline(x, color=GRID, lw=1.2, ls="--")
    a2.text(0, h_ch[40] * 0.55, f"d$N_{{ch}}$/d$\\eta$ = {h_ch[39:41].mean():.1f} at $\\eta$ = 0\n({h_all[39:41].mean():.1f} with the neutrals)", ha="center", fontsize=8.5, color=COL_ND)
    a2.text(0.5, 0.97, "tracker |$\\eta$| < 2.5 (dashed), HF to 5 (grey);\nbeyond 5 the beam remnants, invisible", transform=a2.transAxes, ha="center", va="top", fontsize=8.5)
    a2.set_xlabel(r"$\eta$")
    a2.set_ylabel(r"particles per unit of $\eta$, per collision")
    a2.set_title("(b) where the crowd walks: the plateau", fontsize=10.5)
    a2.legend(loc="lower left", fontsize=8)
    # (c) sum pT per unit area per collision against eta
    all_pt = np.concatenate([e["pt"][~e["nu"]] for e in ev])
    all_eta_v = np.concatenate([e["eta"][~e["nu"]] for e in ev])
    hpt, _ = np.histogram(all_eta_v, bins=eb, weights=all_pt)
    dens = hpt / len(ev) / (eb[1] - eb[0]) / (2 * math.pi)
    a3.step(c, dens, where="mid", color=COL_PU, lw=2, label=r"$\Sigma p_T$ per unit of $\eta$ and $\phi$, per collision (no neutrinos)")
    rho1 = dens[np.abs(c) < 2.5].mean()
    a3.axhline(rho1, color=COL_MODEL, lw=1.2, ls="--")
    a3.text(-9.9, 0.04, f"{rho1:.2f} GeV per unit area per collision, |$\\eta$| < 2.5\n(x {st.mu:g} collisions = {st.mu * rho1:.0f} GeV per unit area; a jet of R = 0.4\ncovers {math.pi * 0.16:.2f}: {st.mu * rho1 * math.pi * 0.16:.0f} GeV under our 100 GeV jet)",
            fontsize=8.5, color=COL_MODEL, va="bottom", ha="left")
    if st.stats:
        rho = np.array([r["rho_grid"] for r in st.stats])
        nin = np.array([r["n_in_time"] for r in st.stats])
        a3.axhline(rho.mean() / nin.mean(), color=COL_EVENT, lw=1.2, ls=":")
        a3.text(9.9, 0.30, f"the grid median $\\rho$ per collision\nover {len(st.stats)} draws: {rho.mean() / nin.mean():.2f}", fontsize=8.5, color=COL_EVENT, ha="right", va="bottom")
    for x in (-5, 5):
        a3.axvline(x, color=GRID, lw=2)
    a3.set_xlabel(r"$\eta$")
    a3.set_ylabel(r"$p_T$ per unit area  [GeV], per collision")
    a3.set_title("(c) what the crowd carries: the slope of $\\rho$", fontsize=10.5)
    a3.legend(loc="upper right", fontsize=8)
    a3.set_ylim(0, max(dens) * 1.45)
    fig.suptitle(f"One member of the crowd: a minimum-bias collision has {np.mean([e['nfinal'] for e in ev]):.0f} particles on average, {h_ch[39:41].mean():.1f} charged per unit of $\\eta$, "
                 f"{rho1:.2f} GeV of $p_T$ per unit area (page 2)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ---------------------------------------------------------------------------
def page_region(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    r = st.region
    # (a) the (z, t) density with the vertices of the in-time crossing
    z = np.linspace(-20, 20, 241)
    t = np.linspace(-0.8, 0.8, 241)
    Z, T = np.meshgrid(z, t, indexing="ij")
    d = r.density(Z, T)
    d /= d.max()
    a1.contourf(Z, T * 1e3, d, levels=np.linspace(0.02, 1, 12), cmap="Greens", alpha=0.9)
    a1.contour(Z, T * 1e3, d, levels=[math.exp(-2), math.exp(-0.5)], colors=[COL_MODEL], linewidths=0.8)
    it = [c for c in st.collisions if c["bx"] == 0 and c["k"] > 0]
    a1.plot([c["z"] for c in it], [c["t"] * 1e3 for c in it], "o", color=COL_IN, ms=3.2, mec="none", label=f"the {len(it)} in-time collisions of our crossing")
    sig = [c for c in st.collisions if c["k"] == 0]
    if sig:
        a1.plot([sig[0]["z"]], [sig[0]["t"] * 1e3], "*", color=COL_SIG, ms=14, mec="white", label=f"the signal: z = {sig[0]['z']:.2f} cm, t = {sig[0]['t'] * 1e3:.0f} ps")
    a1.set_xlabel("z  [cm]")
    a1.set_ylabel("t  [ps]")
    a1.set_title("(a) where and when: the overlap of the two bunches", fontsize=10.5)
    a1.legend(loc="upper left", fontsize=8)
    m = r.moments()
    run3 = r.name == "run3"
    a1.text(0.98, 0.04, (f"model: $\\sigma_z$ = {m[0]:.2f} cm, $\\sigma_t$ = {m[1] * 1e3:.0f} ps\nBetafuncEvtVtxGenerator, 2024 payload: 3.6 cm, $\\sigma_z/c$ = 120 ps\nthe 250-event CMSSW library: {CMSSW_LIB['sigma_z']:.2f} cm, {CMSSW_LIB['sigma_t'] * 1e3:.0f} ps"
             if run3 else f"model: $\\sigma_z$ = {m[0]:.2f} cm, $\\sigma_t$ = {m[1] * 1e3:.0f} ps\nCMSSW HLLHCEvtVtxGenerator: 4.3 cm, 190 ps\nthe 50-event CMSSW library: 4.4 cm, 164 ps"),
            transform=a1.transAxes, ha="right", va="bottom", fontsize=8.5)
    # (b) the z marginal: the model, the vertices of our event, the CMSSW library
    zz = np.linspace(-25, 25, 501)
    tt = np.linspace(-1.5, 1.5, 301)
    Z2, T2 = np.meshgrid(zz, tt, indexing="ij")
    allv = [c for c in st.collisions if c["k"] > 0]
    nbx = len(set(c["bx"] for c in allv))
    if run3:
        pz = np.exp(-((zz - r.z0) ** 2) / (2 * r.sigma_z ** 2)) / (math.sqrt(2 * math.pi) * r.sigma_z)
        a2.plot(zz, pz, "-", color=COL_MODEL, lw=2, label=f"the 2024 payload: Gaussian, $z_0$ = {r.z0:.2f} cm, $\\sigma_z$ = {r.sigma_z:.1f} cm")
        ze, zh = CMSSW_LIB_Z
        zh = np.array(zh, dtype=float) / (CMSSW_LIB["n"] * (ze[1] - ze[0]))
        a2.step(ze, np.append(zh, zh[-1]), where="post", color=COL_CMSSW, lw=1.6, ls="--", label=f"the 2024 CMSSW library, 250 events: r.m.s. {CMSSW_LIB['sigma_z']:.2f} cm")
    else:
        for lab, sw, col, ls in (("head-on bunches: $\\sigma_s/\\sqrt{2}$", (False, False, False), TEXT2, ":"),
                                 ("+ crossing angle 510 $\\mu$rad", (True, False, False), COL_DD, "--"),
                                 ("+ crab cavities 380 $\\mu$rad", (True, True, False), COL_ND, "-."),
                                 ("+ hourglass, $\\beta^*$ = 20 cm", (True, True, True), COL_MODEL, "-")):
            dd = r.density(Z2, T2, *sw)
            pz = dd.sum(axis=1)
            pz /= pz.sum() * (zz[1] - zz[0])
            sz = r.moments(*sw)[0]
            a2.plot(zz, pz, ls, color=col, lw=1.8, label=f"{lab}: {sz:.2f} cm")
    if allv:
        a2.hist([c["z"] for c in allv], bins=np.arange(-25, 25.1, 1.0), density=True, color=COL_IN, alpha=0.45,
                label=f"the {len(allv)} vertices of our event, {nbx} crossings: r.m.s. {np.std([c['z'] for c in allv]):.2f} cm")
    a2.set_xlabel("z of the collision  [cm]")
    a2.set_ylabel("density  [1/cm]")
    a2.set_title("(b) along the beam: the model, our event, the CMSSW library" if run3 else "(b) why the region is shorter than the bunches", fontsize=10.5)
    a2.legend(loc="upper left", fontsize=8)
    a2.set_xlim(-25, 25)
    if run3 and allv:
        a2.text(0.98, 0.55, f"in time: $\\sigma_t$ = {r.sigma_t * 1e3:.0f} ps (model),\n{1e3 * np.std([c['t'] - c['bx'] * core.BUNCH_SPACING for c in allv]):.0f} ps (our event), {CMSSW_LIB['sigma_t'] * 1e3:.0f} ps (CMSSW library)",
                transform=a2.transAxes, ha="right", va="top", fontsize=8.5)
    # (c) the seven crossings on the clock
    for c in allv:
        col = COL_EARLY if c["bx"] < 0 else COL_IN if c["bx"] == 0 else COL_LATE
        a3.plot(c["t"], c["z"], "o", color=col, ms=2.6, mec="none", alpha=0.8)
    if sig:
        a3.plot([sig[0]["t"]], [sig[0]["z"]], "*", color=COL_SIG, ms=14, mec="white", zorder=5)
    bx_all = sorted(set(c["bx"] for c in allv))
    for bx in bx_all:
        n = sum(1 for c in allv if c["bx"] == bx)
        a3.text(bx * core.BUNCH_SPACING, 24.5, f"{bx:+d}\n{n}", ha="center", va="top", fontsize=7 if len(bx_all) > 8 else 8.5, color=COL_EARLY if bx < 0 else COL_IN if bx == 0 else COL_LATE)
    if bx_all:
        a3.set_xlim((bx_all[0] - 0.7) * core.BUNCH_SPACING, (bx_all[-1] + 0.7) * core.BUNCH_SPACING)
    a3.plot([], [], "o", color=COL_EARLY, label="early (bx < 0)")
    a3.plot([], [], "o", color=COL_IN, label="in time (bx = 0)")
    a3.plot([], [], "o", color=COL_LATE, label="late (bx > 0)")
    a3.plot([], [], "*", color=COL_SIG, ms=10, label="the signal")
    a3.set_xlabel("t  [ns]: 25 ns from one crossing to the next; labels: bx and its collisions")
    a3.set_ylabel("z  [cm]")
    a3.set_ylim(-26, 26)
    a3.set_title(f"(c) in time and out of time: the window bx = {bx_all[0] if bx_all else -12:+d} ... {bx_all[-1] if bx_all else 3:+d}", fontsize=10.5)
    a3.legend(loc="lower left", fontsize=8, ncol=2)
    if st.s:
        a3.text(0.98, 0.04, f"nearest crowd vertex to the signal's:\n{st.s['nearest']['dz_mm']:.1f} mm in z, {st.s['nearest']['dt_ps']:.0f} ps in t;\n{st.s['n_visible']} of {st.s['n_in_time']} in-time collisions visible to the tracker",
                transform=a3.transAxes, ha="right", va="bottom", fontsize=8.5)
    fig.suptitle(f"The beam spot of {'2024' if run3 else 'the HL-LHC'}: $\\sigma_z$ = {m[0]:.2f} cm and $\\sigma_t$ = {m[1] * 1e3:.0f} ps, and 25 ns between crossings (page 3)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ---------------------------------------------------------------------------
def page_rho(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    # (a) the grid cells and the kT jets of the record
    inp = []
    for p in st.particles:
        inp.append([p["e"], p["px"], p["py"], p["pz"]])
    sig_p4 = None
    rho_grid, cells, (dy, dphi) = (None, None, (None, None))
    if st.s and st.s.get("n_in_time"):
        # the mixed final state of the record: the in-time particles (no neutrinos, |eta| < 5) plus the signal particles from the chapter-7 record
        import standalone_genjets as gj
        ev, _, _ = gj.dec.read_record(HERE.parent.parent / "GenJet" / "event_GenJet.lhe")
        sig = [ev.p[i].p for i in ev.final_indices() if abs(ev.p[i].id) not in gj.IGNORE_IDS and not gj.is_neutrino(ev.p[i].id) and abs(gj.eta(ev.p[i].p)) < core.ETA_MAX]
        pu = [q for q, p in zip(inp, st.particles) if not core.is_neutrino(p["id"]) and abs(math.asinh(p["pz"] / max(math.hypot(p["px"], p["py"]), 1e-12))) < core.ETA_MAX]
        p4 = np.array(sig + pu)
        rho_grid, cells, (dy, dphi) = core.grid_rho(p4)
        vals = cells.ravel()
        a1.hist(vals, bins=np.linspace(0, max(vals) * 1.02, 40), color=COL_PU, alpha=0.6, label=f"the {cells.size} grid cells of {dy:.2f} x {dphi:.2f}: $\\Sigma p_T$ / area")
        a1.axvline(np.median(vals), color=COL_EVENT, lw=2, label=f"median = $\\rho$ = {np.median(vals):.1f} GeV per unit area")
        a1.axvline(vals.mean(), color=COL_CMSSW, lw=1.2, ls="--", label=f"mean {vals.mean():.1f}: pulled up by the tail")
        if st.jets:
            a1.text(0.98, 0.55, "the jets sit in the tail: the median\nignores them, the mean does not;\n" + f"jet median (kT, |y| < 4.4): {st.s['rho_kt']:.1f}" if st.s.get("rho_kt") else "",
                    transform=a1.transAxes, ha="right", va="top", fontsize=8.5)
    a1.set_xlabel(r"$p_T$ per unit area in a cell  [GeV]")
    a1.set_ylabel("cells")
    a1.set_title("(a) the median energy density of our crossing", fontsize=10.5)
    a1.legend(loc="upper right", fontsize=8)
    # (b) rho against n over the ensemble
    if st.stats:
        n = np.array([r["n_in_time"] for r in st.stats])
        rho = np.array([r["rho_grid"] for r in st.stats])
        a2.plot(n, rho, "o", color=COL_PU, ms=3.5, mec="none", alpha=0.7, label=f"{len(n)} draws of the crowd on the photograph")
        k = np.polyfit(n, rho, 1)
        xx = np.linspace(n.min() - 5, n.max() + 5, 10)
        a2.plot(xx, np.polyval(k, xx), "-", color=COL_MODEL, lw=2, label=f"fit: {k[0]:.3f} GeV per collision per unit area, offset {k[1]:+.1f}")
        if st.s:
            a2.plot([st.s["n_in_time"]], [st.s["rho_grid"]], "*", color=COL_SIG, ms=14, mec="white", label=f"our crossing: {st.s['n_in_time']} collisions, $\\rho$ = {st.s['rho_grid']:.1f}", zorder=5)
        alone = ""
        if all("rho_grid_pileup" in r for r in st.stats):
            rho_pu = np.array([r["rho_grid_pileup"] for r in st.stats])
            alone = f"\nthe crowd alone: {rho_pu.mean():.1f}; the signal raises the median by {(rho - rho_pu).mean():.1f}"
        a2.text(0.02, 0.97, f"$\\rho$ = {rho.mean():.1f} $\\pm$ {rho.std():.1f} over the draws\n$\\sqrt{{\\mu}}$ = {math.sqrt(st.mu):.0f} collisions of spread give {k[0] * math.sqrt(st.mu):.1f} of it;\nthe rest is the crowd's own fluctuation" + alone,
                transform=a2.transAxes, ha="left", va="top", fontsize=8.5)
    a2.set_xlabel("in-time collisions in the crossing, $n$")
    a2.set_ylabel(r"$\rho$ (grid median)  [GeV per unit area]")
    a2.set_title("(b) $\\rho$ grows with the crowd, one line for all", fontsize=10.5)
    a2.legend(loc="lower right", fontsize=8)
    # (c) the truth against rho A for our jet
    if st.stats:
        pu = np.array([r["circle"]["b"]["pt_pileup"] for r in st.stats])
        ra = np.array([r["circle"]["b"]["rho_area"] for r in st.stats])
        lim = (min(pu.min(), ra.min()) * 0.9, max(pu.max(), ra.max()) * 1.05)
        a3.plot(ra, pu, "o", color=COL_PU, ms=3.5, mec="none", alpha=0.7, label=f"the circle of R = 0.4 in (y, $\\phi$) around our jet, {len(pu)} draws")
        a3.plot(lim, lim, "-", color=TEXT2, lw=1)
        res = pu - ra
        alone = ""
        if all("rho_area_pileup" in r["circle"]["b"] for r in st.stats):
            rp = np.array([r["circle"]["b"]["rho_area_pileup"] for r in st.stats])
            alone = f"\n({(pu - rp).mean():+.1f} with the $\\rho$ of the crowd alone:\nthe signal raises the median)"
        a3.text(0.02, 0.84, f"crowd inside the circle: {pu.mean():.1f} $\\pm$ {pu.std():.1f} GeV\n$\\rho\\,\\pi R^2$ estimate: {ra.mean():.1f} $\\pm$ {ra.std():.1f}\nresidual {res.mean():+.1f} $\\pm$ {res.std():.1f} GeV:\nthe subtraction removes the mean,\nthe fluctuation stays" + alone,
                transform=a3.transAxes, ha="left", va="top", fontsize=8.5)
        if st.s and st.s.get("jets", {}).get("b"):
            j = st.s["jets"]["b"]
            a3.plot([j["rho_area"]], [j["pt_pileup"]], "*", color=COL_SIG, ms=14, mec="white", zorder=5,
                    label=f"our crossing, the mixed jet: {j['pt_pileup']:.1f} inside, $\\rho A$ = {j['rho_area']:.1f}")
        ins = a3.inset_axes([0.58, 0.08, 0.4, 0.36])
        ins.hist(res, bins=25, color=COL_MODEL, alpha=0.7)
        ins.set_xlabel("truth $-$ $\\rho A$  [GeV]", fontsize=8)
        ins.tick_params(labelsize=7)
        ins.set_yticks([])
    a3.set_xlabel(r"$\rho\,A$, the estimate  [GeV]")
    a3.set_ylabel("pileup $p_T$ inside the jet, the truth  [GeV]")
    a3.set_title("(c) what the subtraction can and cannot do", fontsize=10.5)
    a3.legend(loc="upper right", fontsize=8)
    title = "$\\rho$, the median energy density, and the jet area"
    if st.s and st.s.get("jets", {}).get("b"):
        j = st.s["jets"]["b"]
        title += f": our jet {j['ref_pt']:.1f} $\\to$ {j['pt']:.1f} with the crowd, $\\rho A$ = {j['rho_area']:.1f}, $p_T - \\rho A$ = {j['pt_sub']:.1f} GeV"
    fig.suptitle(title + " (page 4)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


PAGES = {1: page_scenario, 2: page_member, 3: page_region, 4: page_rho}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, default=HERE / "theory_pileup.pdf")
    ap.add_argument("--page", type=int, default=None)
    ap.add_argument("--record", type=Path, default=HERE.parent / "event_PU.lhe")
    ap.add_argument("--library", type=Path, default=HERE.parent / "pileup_library.lhe.gz")
    ap.add_argument("--summary", type=Path, default=HERE / "standalone_pileup.json")
    ap.add_argument("--stats", type=Path, default=HERE / "standalone_pileup_stats.json")
    args = ap.parse_args()
    st = Setup(args.record, args.library, args.summary, args.stats)
    if args.page:
        fig = PAGES[args.page](st)
        fig.savefig(args.output, dpi=150)
        print(f"wrote {args.output}")
        return
    with PdfPages(str(args.output)) as pdf:
        for n in sorted(PAGES):
            fig = PAGES[n](st)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {args.output} ({len(PAGES)} pages)")


if __name__ == "__main__":
    main()
