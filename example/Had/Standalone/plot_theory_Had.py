#!/usr/bin/env python3
"""The plots behind the theory sections of chapter 5, drawn from the standalone hadronisation script.

One page per theory (sub)section of the chapter. Every curve is computed with the
classes of standalone_hadronisation.py (StringZ, StringPT, StringFlav, StringSystem),
every marker is a number of the seed-1 hadronisation of the chapter-4 record
(../event_Had.lhe), and the ensemble panels read the statistics files written by
--repeat (standalone_had_stats*.stats.json) and by validate_hadronisation_with_pythia.py
(validate_hadronisation_with_pythia_*.json) when they are present.

  page 1  string        the linear potential, the space-time world sheet of the 1 TeV d dbar string with
                        its break-up vertices, and the hadron multiplicity of every string against its mass
  page 2  break-up      the tunnelling suppression of heavy flavours against the tune, the Gaussian pT of the
                        break-ups, and the species composition against PYTHIA
  page 3  z             the Lund symmetric fragmentation function, the z values drawn in the event against the
                        expectation, and the remaining mass of the b quark's string hadron by hadron
  page 4  kinks         the b quark's string in (eta, phi) with its hadrons, the region map of the world sheet
                        with the paths of the two ends, and the hadrons along the string in rapidity
  page 5  b-fragment.   Lund-Bowler against plain Lund and Peterson at m_B, the x_E distributions of the three
                        for the standalone script and for PYTHIA on the same partons, and the B species
  page 6  x_B           what x_B is in different settings: the B hadron pT against the quark pT, the share of
                        the jet, and the mean x_E from the same partons, the full chain, LEP and our event
  page 7  our jet       the hadron-level cones of R = 0.4 and 0.8 around the b against the parton-level ones,
                        mass and multiplicity, and the ledger from quark to B hadron to jet

    python3 plot_theory_Had.py                    # -> theory_hadronisation.pdf
    python3 plot_theory_Had.py --page 5 -o o.png  # one page as PNG

Needs numpy and matplotlib (on this Mac: /usr/bin/python3). About 15 s.
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
import standalone_hadronisation as core  # noqa: E402
import plot_journey_Had as journey  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

COL_B, COL_BBAR, COL_HAD = "#ff0000", "#0000ff", "#9b4dff"
COL_STR, COL_GLU, COL_MINI = "#2a78d6", "#2f8f5b", "#eb6834"
COL_A, COL_EVENT, COL_CMSSW, COL_PY, COL_PY2 = "#8e5bb5", "#c0392b", "#1f1f1f", "#7f7f7f", "#b0b0b0"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d6d3ca"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "legend.fontsize": 9.5,
    "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 100,
})
PAGES = {1: "the string", 2: "the break-up", 3: "the fragmentation function", 4: "gluons as kinks",
         5: "b-fragmentation", 6: "x_B in different settings", 7: "our jet"}

# the CMSSW event (page 4 of ../CMSSW/genPartAnalyzer_ME_PS_MPI_Had_out.pdf and the GEN log of chapter 2)
CMSSW = dict(n_hadrons=320, n_strings=26, b_pt=91.744, B_pt=94.279, B_name="B-", bbar_pt=116.356, Bbar_pt=114.174,
             cone04=114.45, cone08=141.10, n04=9, n08=28, m04=20.42, m08=49.76, hadrons_per_string=[42, 38, 33, 22, 21, 20, 18, 17, 15, 14, 10, 10, 8, 7, 7, 7, 6, 4, 4, 3, 3, 3, 2, 2, 2, 2])
ALEPH_XE = 0.716      # <x_E> of weakly decaying B hadrons in Z decays, ALEPH 2001 (Phys. Lett. B512 (2001) 30)
STATS = HERE / "standalone_had_stats.stats.json"
STATS_R0 = HERE / "standalone_had_stats_rfactb0.stats.json"
STATS_PET = HERE / "standalone_had_stats_peterson.stats.json"
PY_REC = HERE / "validate_hadronisation_with_pythia_record.json"
PY_REC_R0 = HERE / "validate_hadronisation_with_pythia_record_rfactb0.json"
PY_REC_PET = HERE / "validate_hadronisation_with_pythia_record_peterson.json"
PY_FULL = HERE / "validate_hadronisation_with_pythia_full.json"
PY_FULL_B = HERE / "validate_hadronisation_with_pythia_full_fixedb.json"


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


class Setup:
    """The seed-1 hadronisation of the chapter-4 record, replayed with the script's classes."""

    def __init__(self, seed=1):
        self.args = SimpleNamespace(input=HERE.parent.parent / "MPI" / "event_MPI.lhe", alund=core.A_LUND, blund=core.B_LUND,
                                    rfactb=core.R_FACT_B, rfactc=core.R_FACT_C, peterson=False, sigma=core.SIGMA_PT,
                                    prob_s=core.PROB_S_TO_UD, prob_qq=core.PROB_QQ_TO_Q, radius=0.4)
        self.ev, self.had = core.run(self.args, seed)
        self.summary = core.summarise(self.ev, self.had)
        self.zfun = self.had.zfun
        self.b = self.summary["b"]["b"]
        self.bstring = next(x for x in self.had.strings if self.b["quark_row"] in x["rows"])
        self.stats, self.stats_r0, self.stats_pet = load_json(STATS), load_json(STATS_R0), load_json(STATS_PET)
        self.py_rec, self.py_r0, self.py_pet = load_json(PY_REC), load_json(PY_REC_R0), load_json(PY_REC_PET)
        self.py_full, self.py_full_b = load_json(PY_FULL), load_json(PY_FULL_B)

    def all_infos(self, strings_only=True):
        out = []
        for x in self.had.strings:
            if strings_only and x["kind"] != "string":
                continue
            for hid, p, info in (x["hadrons"] or []):
                out.append((hid, p, info, x))
        return out


def norm_f(zfun, fo, fn, mt2):
    lf = zfun.logf(fo, fn, mt2)
    f = np.exp(lf - lf.max())
    return f / (f.sum() * zfun.dz)


def mark(ax, x, label, color=COL_EVENT, y=0.93, ha="left", dx=4, ls="--"):
    ax.axvline(x, color=color, lw=1.1, ls=ls)
    ax.annotate(label, xy=(x, y), xycoords=("data", "axes fraction"), xytext=(dx if ha == "left" else -dx, 0),
                textcoords="offset points", ha=ha, va="top", fontsize=9, color=color)


# ---------------------------------------------------------------------------------------------
def page_string(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    # (a) the potential
    r = np.linspace(0.05, 2.5, 300)
    kappa = core.KAPPA
    alpha_s = 0.3
    v_lin = kappa * r
    v_cornell = -4 * alpha_s / (3 * r) * core.HBARC + kappa * r
    a1.plot(r, v_lin, color=COL_STR, lw=2, label=r"$V = \kappa\,r$,  $\kappa$ = %.0f GeV/fm" % kappa)
    a1.plot(r, v_cornell, color=TEXT2, lw=1.2, ls="--", label=r"$-\frac{4}{3}\frac{\alpha_s \hbar c}{r} + \kappa r$ (Cornell)")
    for m, lab in ((2 * 0.33, r"$2 m_{u}$: a $u\bar u$ pair"), (2 * 0.5, r"$2 m_s$"), (2 * 4.8, None)):
        if m < v_lin.max():
            a1.axhline(m, color=COL_A, lw=0.8, ls=":")
            if lab:
                a1.text(0.07, m + 0.03, lab, fontsize=8.5, color=COL_A)
    a1.set_xlabel("$r$  [fm]")
    a1.set_ylabel("$V(r)$  [GeV]")
    a1.set_ylim(-1.0, 2.6)
    a1.set_title("(a) the string: energy grows linearly with length", fontsize=10.5)
    a1.legend(loc="upper left", fontsize=9)
    a1.text(0.98, 0.04, "a string of 1 fm holds 1 GeV:\nenough for a light quark pair\nevery fermi or so", transform=a1.transAxes,
            ha="right", va="bottom", fontsize=8.5, color=TEXT2)
    # (b) the world sheet of a two-parton string in light-cone coordinates: Gamma = x+ x- are straight lines
    two = [x for x in st.had.strings if x["kind"] == "string" and len(x["partons"]) == 2 and x.get("system") is not None]
    big = min(two, key=lambda x: abs(x["w"] - 30.0))      # a string light enough for the vertices to be visible
    sysm = big["system"]
    W = big["w"]
    lo = 1e-3 * W
    for g in (0.3, 1, 3, 10):
        xp = np.logspace(math.log10(lo), math.log10(W), 200)
        a2.plot(xp, g / xp, color=GRID, lw=0.9)
        if g >= 1:
            a2.text(W / 3, 3 * g / W * 1.2, rf"$\Gamma$ = {g:g} GeV$^2$", fontsize=7.5, color=TEXT2, ha="center")
    for end, col, lab in ((sysm.pos, COL_B, f"vertices of the {core.pname(sysm.ids[0])} end"), (sysm.neg, COL_BBAR, f"vertices of the {core.pname(sysm.ids[-1])} end")):
        pts = []
        for hid, p, info in end.hadrons:
            if "coords" in info:
                tp, tm = info["coords"]
                pts.append((max(tp * W, lo), max(tm * W, lo)))
        if pts:
            xs, ys = zip(*pts)
            a2.loglog(xs, ys, "o-", color=col, ms=5, lw=1.0, mec="white", label=lab + f" ({len(end.hadrons)} hadrons)")
            for j, (xv, yv) in enumerate(pts):
                a2.annotate(str(j + 1), (xv, yv), xytext=(3, 3), textcoords="offset points", fontsize=7, color=col)
    a2.set_xlim(lo, W * 1.1)
    a2.set_ylim(lo, W * 1.1)
    a2.set_xlabel(r"$\kappa\,x^+ = \kappa(t + z)$  [GeV]   (the %s moves along this axis)" % core.pname(sysm.ids[0]))
    a2.set_ylabel(r"$\kappa\,x^- = \kappa(t - z)$  [GeV]")
    a2.set_title(f"(b) the break-up vertices of the {W:.0f} GeV {core.pname(sysm.ids[0])} {core.pname(sysm.ids[-1])} string", fontsize=10.5)
    a2.legend(loc="upper right", fontsize=8.5)
    a2.text(0.03, 0.03, r"the area law puts every vertex near $\Gamma = \kappa^2 \tau^2 = x^+ x^- \sim$ 1-3 GeV$^2$;" "\n"
            r"a vertex at $(x^+, x^-)$ has used $x^+$ of the quark's and $x^-$ of the antiquark's light-cone momentum",
            transform=a2.transAxes, fontsize=8, color=TEXT2, va="bottom")
    # (c) multiplicity against the string mass
    ws = np.array([x["w"] for x in st.had.strings])
    ns = np.array([len(x["hadrons"] or []) for x in st.had.strings])
    two_p = np.array([len(x["partons"]) == 2 for x in st.had.strings])
    a3.semilogx(ws[two_p], ns[two_p], "o", color=COL_EVENT, ms=6, mec="white", label="our event: two-parton strings", zorder=5)
    a3.semilogx(ws[~two_p], ns[~two_p], "o", color=COL_EVENT, ms=6, mfc="none", label="our event: strings with gluons", zorder=5)
    if st.stats and st.stats.get("hadrons_per_string"):
        hps = np.array(st.stats["hadrons_per_string"], dtype=float)
        sw = np.array(st.stats["strings_w"])
        order = np.argsort(sw)
        a3.errorbar(sw[order], hps.mean(axis=0)[order], yerr=hps.std(axis=0)[order], fmt="s", color=COL_A, ms=4, lw=0.8,
                    label=f"this script, {hps.shape[0]} hadronisations (mean, std)")
        sel = two_p & (sw > 3)
        if sel.sum() > 2:
            c1, c0 = np.polyfit(np.log(sw[sel]), hps.mean(axis=0)[sel], 1)
            wline = np.logspace(0.3, 3.2, 100)
            a3.plot(wline, c0 + c1 * np.log(wline), color=COL_A, lw=1, ls=":", label=f"two-parton strings: {c0:.1f} + {c1:.2f} ln W")
    if st.py_rec and st.py_rec.get("strings"):
        acc = {}
        for ev_strings in st.py_rec["strings"]:
            for w, n in ev_strings:
                acc.setdefault(round(w, 1), []).append(n)
        wk = np.array(sorted(acc))
        a3.plot(wk, [np.mean(acc[w]) for w in wk], "D", color=COL_PY, ms=4, label="PYTHIA 8 on the same partons (mean)")
    a3.set_xlabel("string mass $W$  [GeV]")
    a3.set_ylabel("primary hadrons from the string")
    a3.set_title("(c) how many hadrons a string makes (gluons add)", fontsize=10.5)
    a3.legend(loc="upper left", fontsize=8.5)
    fig.suptitle("The Lund string: a linear potential, a world sheet that breaks, a logarithmic multiplicity (page 1)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
def page_breakup(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    kappa_gev2 = core.KAPPA * core.HBARC          # GeV^2
    m = np.linspace(0, 1.6, 300)
    sup = np.exp(-math.pi * (m ** 2 - 0.33 ** 2) / kappa_gev2)
    a1.semilogy(m, sup, color=COL_STR, lw=2, label=r"tunnelling: $\exp(-\pi (m_q^2 - m_u^2)/\kappa)$, $\kappa$ = %.3f GeV$^2$" % kappa_gev2)
    for mq, val, lab, col in ((0.5, core.PROB_S_TO_UD, "s : u = %.3f (tune)" % core.PROB_S_TO_UD, COL_A),
                              (0.58, core.PROB_QQ_TO_Q, "diquark : quark = %.3f (tune)" % core.PROB_QQ_TO_Q, COL_MINI)):
        a1.plot([mq], [val], "o", color=col, ms=7, mec="white")
        a1.annotate(lab, (mq, val), xytext=(6, 4), textcoords="offset points", fontsize=8.5, color=col)
    a1.set_xlabel("mass of the pair's quark (or diquark)  [GeV]")
    a1.set_ylabel("relative rate")
    a1.set_ylim(1e-6, 3)
    a1.set_xlim(0, 1.2)
    a1.set_title("(a) which flavour pops out of the vacuum", fontsize=10.5)
    a1.legend(loc="lower left", fontsize=8.5)
    a1.text(0.97, 0.75, "c and b: never from the vacuum\n($10^{-11}$ and $10^{-100}$); they come\nfrom the hard process and the showers",
            transform=a1.transAxes, ha="right", va="top", fontsize=8.5, color=TEXT2)
    # (b) the pT kicks of the event
    kicks = []
    for hid, p, info, x in st.all_infos():
        if "pt_new" in info and not info.get("final"):
            kicks += list(info["pt_new"])
    kicks = np.array(kicks)
    sig = st.args.sigma / math.sqrt(2)
    xx = np.linspace(-1.5, 1.5, 300)
    a2.hist(kicks, bins=40, range=(-1.5, 1.5), density=True, color=COL_HAD, alpha=0.55, label=f"{len(kicks)} kick components of our event")
    g = np.exp(-xx ** 2 / (2 * sig ** 2)) / (sig * math.sqrt(2 * math.pi))
    a2.plot(xx, (1 - core.ENHANCED_FRACTION) * g + core.ENHANCED_FRACTION * np.exp(-xx ** 2 / (2 * (2 * sig) ** 2)) / (2 * sig * math.sqrt(2 * math.pi)),
            color=COL_STR, lw=2, label=r"Gaussian, $\sigma_q = \sigma/\sqrt{2}$ = %.3f GeV (1%% with twice the width)" % sig)
    a2.set_xlabel("$p_x$ or $p_y$ of the new quark  [GeV]")
    a2.set_ylabel("density")
    a2.set_title(r"(b) the transverse kick of every break-up, $\sigma$ = %.3f GeV" % st.args.sigma, fontsize=10.5)
    a2.legend(loc="lower left", fontsize=8)
    a2.text(0.03, 0.95, f"rms of the drawn components: {kicks.std():.3f} GeV\nhadron $\\langle p_T^2\\rangle = 2\\sigma^2$ = {2 * st.args.sigma ** 2:.3f} GeV$^2$",
            transform=a2.transAxes, fontsize=8.5, color=TEXT2, va="top")
    # (c) species
    names = ["pi+", "pi-", "pi0", "rho+", "rho-", "rho0", "omega", "eta", "K+", "K-", "K0", "Kbar0", "K*+", "K*-", "p+", "n0", "eta'", "phi", "Lambda0", "Delta++"]
    srcs = [("our event", {n: st.summary["species"].get(n, 0) / st.summary["n_hadrons"] for n in names}, COL_EVENT)]
    if st.stats:
        tot = sum(st.stats["species"].values())
        srcs.append((f"this script, {st.stats['repeat']} hadronisations", {n: st.stats["species"].get(n, 0) / tot for n in names}, COL_A))
    if st.py_rec:
        tot = sum(st.py_rec["species"].values())
        srcs.append((f"PYTHIA 8, same partons, {st.py_rec['nevents']} events", {n: st.py_rec["species"].get(n, 0) / tot for n in names}, COL_PY))
    xpos = np.arange(len(names))
    wbar = 0.8 / len(srcs)
    for j, (lab, frac, col) in enumerate(srcs):
        a3.bar(xpos + (j - (len(srcs) - 1) / 2) * wbar, [100 * frac[n] for n in names], width=wbar, color=col, label=lab)
    a3.set_xticks(xpos)
    a3.set_xticklabels([journey.hadron_tex(n) for n in names], rotation=60, fontsize=8)
    a3.set_ylabel("% of all primary hadrons")
    a3.set_title("(c) what comes out: the species", fontsize=10.5)
    a3.legend(loc="upper right", fontsize=8.5)
    fig.suptitle("The break-up: flavour by tunnelling, transverse momentum by a Gaussian, species by spin counting (page 2)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
def page_z(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    z = st.zfun.z
    for fo, fn, mt2, lab, col in ((2, -1, core.hadron_mass(211) ** 2 + st.args.sigma ** 2, r"$u \to \pi$, $m_T$ = %.2f GeV" % math.sqrt(core.hadron_mass(211) ** 2 + st.args.sigma ** 2), COL_STR),
                                  (2, -1, core.hadron_mass(213) ** 2 + st.args.sigma ** 2, r"$u \to \rho$, $m_T$ = %.2f GeV" % math.sqrt(core.hadron_mass(213) ** 2 + st.args.sigma ** 2), COL_A),
                                  (2, 2101, core.hadron_mass(2212) ** 2 + st.args.sigma ** 2, r"$u \to p$ (new diquark, $a$ + %.2f)" % core.A_EXTRA_DIQUARK, COL_MINI),
                                  (3, -2, core.hadron_mass(321) ** 2 + st.args.sigma ** 2, r"$s \to K$", COL_GLU)):
        f = norm_f(st.zfun, fo, fn, mt2)
        a1.plot(z, f, color=col, lw=2, label=lab + f",  $\\langle z\\rangle$ = {(z * f).sum() * st.zfun.dz:.2f}")
    a1.set_xlabel("$z$")
    a1.set_ylabel("$f(z)$")
    a1.set_title(r"(a) Lund symmetric $f(z) = \frac{1}{z}(1-z)^a \exp(-b\,m_T^2/z)$, $a$ = %.2f, $b$ = %.2f" % (st.args.alund, st.args.blund), fontsize=10.5)
    a1.legend(loc="upper right", fontsize=8.5)
    # (b) the z drawn in the event against the expectation
    zs, expect = [], np.zeros_like(z)
    for hid, p, info, x in st.all_infos():
        if info.get("final") or abs(info["old"]) in (4, 5):
            continue
        zs.append(info["z"])
        expect += norm_f(st.zfun, info["old"], info["new"], info["mt2"])
    expect /= max(len(zs), 1)
    a2.hist(zs, bins=25, range=(0, 1), density=True, color=COL_HAD, alpha=0.55, label=f"{len(zs)} light-flavour hadrons of our event")
    a2.plot(z, expect, color=COL_STR, lw=2, label="expectation: mean of $f(z; m_T^2)$ over those hadrons")
    a2.set_xlabel("$z$ drawn")
    a2.set_ylabel("density")
    a2.set_title("(b) the fractions actually drawn", fontsize=10.5)
    a2.legend(loc="upper left", fontsize=8)
    a2.text(0.03, 0.05, f"mean z drawn {np.mean(zs):.3f}\nexpected {(z * expect).sum() * st.zfun.dz:.3f}", transform=a2.transAxes, fontsize=8.5, color=TEXT2, va="bottom")
    # (c) the b string: remaining mass hadron by hadron
    sysm = st.bstring["system"]
    wrem = [h[2].get("w_rem") for h in sysm.hadrons]
    wmin = [h[2].get("w_min") for h in sysm.hadrons]
    k = np.arange(len(wrem))
    a3.plot(k, wrem, "o-", color=COL_STR, ms=5, mec="white", label="remaining mass $W_{rem}$ before each hadron")
    a3.plot(k, wmin, "s--", color=COL_MINI, ms=4, label=r"$W_{min}$ = $m_{q_1}$ + $m_{q_2}$ + stopMass + stopNewFlav $m_{q,last}$, smeared")
    for i, (hid, p, info) in enumerate(sysm.hadrons):
        a3.annotate(journey.hadron_tex(core.pname(hid)), (i, wrem[i]), xytext=(0, 7 if i % 2 == 0 else -11), textcoords="offset points",
                    ha="center", fontsize=6.5, color=COL_B if info["end"] == "+" else COL_BBAR)
    a3.set_xlabel("hadrons in production order (red: from the $b$ end, blue: other end)")
    a3.set_ylabel("[GeV]")
    a3.set_yscale("log")
    a3.set_title(f"(c) the $b$ quark's string ({st.bstring['w']:.0f} GeV) runs out: {len(sysm.hadrons)} hadrons", fontsize=10.5)
    a3.legend(loc="upper right", fontsize=8.5)
    fig.suptitle("The fragmentation function and the iteration from both ends (page 3)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
def page_kinks(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    ev, x = st.ev, st.bstring
    sysm = x["system"]
    parts = x["partons"]
    # (a) the string and its hadrons in (eta, phi)
    etas = [core.eta(p) for _, p, _, _ in parts]
    phis = [core.phi(p) for _, p, _, _ in parts]
    for (e0, f0), (e1, f1) in zip(zip(etas, phis), zip(etas[1:], phis[1:])):
        d = f1 - f0
        f1u = f1 - 2 * math.pi if d > math.pi else f1 + 2 * math.pi if d < -math.pi else f1
        a1.plot([e0, e1], [f0, f1u], color=COL_STR, lw=1.3, alpha=0.8)
    for (pid, p, m, rows), e, f in zip(parts, etas, phis):
        a1.plot([e], [f], "o" if pid != 21 else "s", color=COL_B if abs(pid) == 5 else COL_GLU, ms=max(4, min(14, 3 + 2.5 * math.log10(1 + core.pt(p)))), mec="white")
        a1.annotate(f"{core.pname(pid)} {core.pt(p):.1f}", (e, f), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=TEXT2)
    for hid, p, info in x["hadrons"]:
        col = COL_B if info["end"] == "+" else COL_BBAR
        a1.plot([core.eta(p)], [core.phi(p)], "o", color=col, ms=max(3, min(12, 2 + 2.5 * math.log10(1 + core.pt(p)))), mfc="none", mew=1.2)
    a1.plot([], [], "o", color=COL_B, mfc="none", label="hadrons from the $b$ end")
    a1.plot([], [], "o", color=COL_BBAR, mfc="none", label="hadrons from the other end")
    a1.plot([], [], "s", color=COL_GLU, label="gluons (kinks)")
    a1.set_xlabel(r"$\eta$")
    a1.set_ylabel(r"$\phi$")
    a1.set_title(f"(a) the $b$ quark's string: {len(parts)} partons, {len(x['hadrons'])} hadrons", fontsize=10.5)
    a1.legend(loc="best", fontsize=8.5)
    # (b) the region map with the two paths
    n = sysm.n
    w2 = np.full((n, n + 1), np.nan)
    for i in range(n):
        for k in range(i + 1, n + 1):
            w2[i, k] = sysm.region(i, k).w2
    im = a2.imshow(np.log10(np.where(w2 > 0, w2, np.nan)).T, origin="lower", cmap="YlGnBu", aspect="auto",
                   extent=(-0.5, n - 0.5, 0.5, n + 0.5))
    fig.colorbar(im, ax=a2, label=r"$\log_{10} w^2_{ik}$  [GeV$^2$]  (the region's mass$^2$)")
    for end, col, lab in ((sysm.pos, COL_B, "path of the $b$ end"), (sysm.neg, COL_BBAR, "path of the other end")):
        path = [end.region_ik()]
        pts = [(0, 1)] if end.sign > 0 else [(n - 1, n)]
        for hid, p, info in end.hadrons:
            if "region" in info:
                pts.append(info["region"])
        ii, kk = zip(*pts)
        a2.plot(ii, kk, "o-", color=col, ms=5, lw=1.3, mec="white", label=lab)
    a2.set_xlabel("$i$: the piece from the $b$ side ($p_+$ of parton $i$)")
    a2.set_ylabel("$k$: the piece from the other side ($p_-$ of parton $k$)")
    a2.set_xticks(range(n))
    a2.set_yticks(range(1, n + 1))
    a2.set_title("(b) the regions of the world sheet and where the hadrons formed", fontsize=10.5)
    a2.legend(loc="lower right", fontsize=8.5)
    a2.grid(False)
    # (c) the hadrons along the string in rapidity
    axis = x["partons"][0][1][1:]
    axis = axis / np.linalg.norm(axis)
    def y_along(p):
        pl = float(np.dot(p[1:], axis))
        return 0.5 * math.log((p[0] + pl) / max(p[0] - pl, 1e-9))
    ys = [(y_along(p), core.pt(p), info["end"], core.pname(hid)) for hid, p, info in x["hadrons"]]
    yp = [(y_along(p), core.pt(p), pid) for pid, p, m, rows in parts]
    for yv, ptv, e, nm in ys:
        a3.plot([yv], [ptv], "o", color=COL_B if e == "+" else COL_BBAR, ms=6, mec="white")
        if ptv > 3:
            a3.annotate(journey.hadron_tex(nm), (yv, ptv), xytext=(3, 3), textcoords="offset points", fontsize=7.5, color=TEXT2)
    for yv, ptv, pid in yp:
        a3.plot([yv], [ptv], "s" if pid == 21 else "D", color=COL_GLU if pid == 21 else COL_STR, ms=7, mfc="none", mew=1.3)
    a3.plot([], [], "D", color=COL_STR, mfc="none", label="string ends (partons)")
    a3.plot([], [], "s", color=COL_GLU, mfc="none", label="gluons")
    a3.plot([], [], "o", color=COL_B, label="hadrons from the $b$ end")
    a3.plot([], [], "o", color=COL_BBAR, label="hadrons from the other end")
    a3.set_yscale("log")
    a3.set_xlabel("rapidity along the $b$ quark direction")
    a3.set_ylabel("$p_T$ in the lab  [GeV]")
    a3.set_title("(c) the hadrons fill the rapidity range between the partons", fontsize=10.5)
    a3.legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Gluons as kinks: regions of the world sheet, hadrons across the kinks (page 4)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
def page_bfrag(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    z = st.zfun.z
    mt2 = core.hadron_mass(513) ** 2 + st.args.sigma ** 2
    zb = core.StringZ(r_b=core.R_FACT_B)
    z0 = core.StringZ(r_b=0.0)
    zp = core.StringZ(peterson=True)
    for zf, lab, col in ((zb, r"Lund-Bowler, $1/z^{1 + r_b b m_b^2}$, $r_b$ = %.3f: exponent %.1f" % (core.R_FACT_B, 1 + core.R_FACT_B * core.B_LUND * 4.8 ** 2), COL_B),
                         (z0, r"plain Lund ($r_b$ = 0): the light-quark function at $m_T$ = $m_B$", COL_A),
                         (zp, r"Peterson, $\epsilon_b$ = %.3f" % core.EPSILON_B, COL_PY)):
        f = norm_f(zf, 5, -2, mt2)
        a1.plot(z, f, color=col, lw=2, label=lab + f"; $\\langle z\\rangle$ = {(z * f).sum() * zf.dz:.3f}")
    a1.set_xlabel("$z$")
    a1.set_ylabel("$f(z)$")
    a1.set_title(r"(a) the $b$ quark's fragmentation function at $m_T$ = $m_{B^*}$", fontsize=10.5)
    a1.legend(loc="upper left", fontsize=8.5)
    # (b) x_E distributions
    sets = [("this script, Lund-Bowler", st.stats, COL_B, "-"), ("this script, plain Lund", st.stats_r0, COL_A, "-"),
            ("this script, Peterson", st.stats_pet, COL_PY, "-"),
            ("PYTHIA, same partons, Lund-Bowler", st.py_rec, COL_B, "--"), ("PYTHIA, plain Lund", st.py_r0, COL_A, "--"),
            ("PYTHIA, Peterson", st.py_pet, COL_PY, "--")]
    bins = np.linspace(0.3, 1.3, 26)
    for lab, d, col, ls in sets:
        if not d or "b" not in d or "b" not in d["b"]:
            continue
        xe = np.array(d["b"]["b"]["x_e"], dtype=float)
        a2.hist(xe, bins=bins, density=True, histtype="step", color=col, ls=ls, lw=1.8, label=f"{lab}: $\\langle x_E\\rangle$ = {xe.mean():.3f}")
    mark(a2, st.b["x_e"], f"our event {st.b['x_e']:.3f}", COL_EVENT, y=0.45)
    a2.set_xlabel("$x_E = E_B / E_b$ of the $b$ quark's hadron")
    a2.set_ylabel("density")
    a2.set_xlim(0.4, 1.3)
    a2.set_title("(b) the energy the B hadron keeps, same partons every time", fontsize=10.5)
    a2.legend(loc="upper left", fontsize=7.5)
    # (c) species
    names = ["B*bar0", "B*-", "Bbar0", "B-", "B*_sbar0", "B_sbar0", "Lambda_b0", "Sigma*_b0", "Xi_b0", "Xi_b-"]
    srcs = []
    if st.stats:
        tot = sum(st.stats["b_species"]["b"].values())
        srcs.append((f"this script ({st.stats['repeat']})", {n: st.stats["b_species"]["b"].get(n, 0) / tot for n in names}, COL_A))
    if st.py_rec:
        tot = sum(st.py_rec["b_species"]["b"].values())
        srcs.append((f"PYTHIA, same partons ({st.py_rec['nevents']})", {n: st.py_rec["b_species"]["b"].get(n, 0) / tot for n in names}, COL_PY))
    if st.py_full:
        tot = sum(st.py_full["b_species"]["b"].values())
        srcs.append((f"PYTHIA, full chain ({st.py_full['nevents']})", {n: st.py_full["b_species"]["b"].get(n, 0) / tot for n in names}, COL_CMSSW))
    xpos = np.arange(len(names))
    wbar = 0.8 / max(len(srcs), 1)
    for j, (lab, frac, col) in enumerate(srcs):
        a3.bar(xpos + (j - (len(srcs) - 1) / 2) * wbar, [100 * frac[n] for n in names], width=wbar, color=col, label=lab)
    a3.set_xticks(xpos)
    a3.set_xticklabels([journey.hadron_tex(n) for n in names], rotation=45, fontsize=9)
    a3.set_ylabel("% of the $b$ quark's hadrons")
    a3.set_title("(c) which B hadron: vector 2.2 : 1, strange 0.217, baryon 0.081", fontsize=10.5)
    a3.legend(loc="upper right", fontsize=8.5)
    fig.suptitle(r"$b$-fragmentation: the Bowler factor and $x_B$ (page 5)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
def page_xb(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    # (a) B pT against the quark pT: same partons (fixed quark) and full chain (quark varies)
    if st.py_full:
        q = np.array(st.py_full["b"]["b"]["pt_hadron"]) / np.array(st.py_full["b"]["b"]["x_pt"])
        a1.plot(q, st.py_full["b"]["b"]["pt_hadron"], ".", color=COL_PY2, ms=3, label=f"PYTHIA full chain ({st.py_full['nevents']} events)")
    if st.stats:
        xs = np.array(st.stats["b"]["b"]["pt_hadron"])
        a1.plot(np.full_like(xs, st.b["pt_quark"]), xs, ".", color=COL_A, ms=4, label=f"this script, same quark ({st.stats['repeat']})")
    a1.plot([st.b["pt_quark"]], [st.b["pt_hadron"]], "o", color=COL_EVENT, ms=8, mec="white", label=f"our event: {st.b['pt_quark']:.1f} -> {st.b['pt_hadron']:.1f} GeV", zorder=5)
    a1.plot([CMSSW["b_pt"]], [CMSSW["B_pt"]], "s", color=COL_CMSSW, ms=8, mec="white", label=f"CMSSW event: {CMSSW['b_pt']:.1f} -> {CMSSW['B_pt']:.1f} GeV", zorder=5)
    lim = np.array([20, 260])
    for fr, ls in ((1.0, "-"), (0.9, "--"), (0.7, ":")):
        a1.plot(lim, fr * lim, color=TEXT2, lw=0.8, ls=ls)
        a1.text(lim[1], fr * lim[1], f"$x_{{p_T}}$ = {fr:g}", fontsize=8, color=TEXT2, ha="right", va="bottom")
    a1.set_xlabel("$p_T$ of the $b$ quark before hadronisation  [GeV]")
    a1.set_ylabel("$p_T$ of its B hadron  [GeV]")
    a1.set_xlim(20, 260)
    a1.set_ylim(0, 260)
    a1.set_title("(a) the B hadron follows its quark", fontsize=10.5)
    a1.legend(loc="upper left", fontsize=8.5)
    # (b) x_E against the share of the jet
    if st.stats:
        c = st.stats["b"]["b"]["cones"]["0.4"]
        a2.plot(st.stats["b"]["b"]["x_e"], c["frac_B"], ".", color=COL_A, ms=4, label="this script, same partons")
    if st.py_full:
        c = st.py_full["b"]["b"]["cones"]["0.4"]
        a2.plot(st.py_full["b"]["b"]["x_e"], c["frac_B"], ".", color=COL_PY2, ms=3, label="PYTHIA full chain")
    a2.plot([st.b["x_e"]], [st.b["cones"][0.4]["frac_B"]], "o", color=COL_EVENT, ms=8, mec="white", label="our event", zorder=5)
    a2.plot([CMSSW["B_pt"] / CMSSW["b_pt"]], [CMSSW["B_pt"] / CMSSW["cone04"]], "s", color=COL_CMSSW, ms=8, mec="white", label="CMSSW event", zorder=5)
    a2.set_xlabel("$x_E = E_B/E_b$ (the quark's energy kept by the hadron)")
    a2.set_ylabel("$p_T(B)\\,/\\,p_T$(hadrons within $\\Delta R < 0.4$ of the quark)")
    a2.set_xlim(0.3, 1.4)
    a2.set_ylim(0.2, 1.1)
    a2.set_title("(b) ... but the jet is more than the B hadron", fontsize=10.5)
    a2.legend(loc="lower right", fontsize=8.5)
    # (c) the mean x_E in different settings
    rows = []
    if st.stats:
        rows.append(("this script, same partons", np.mean(st.stats["b"]["b"]["x_e"]), np.std(st.stats["b"]["b"]["x_e"]), COL_A))
    if st.py_rec:
        rows.append(("PYTHIA, same partons", np.mean(st.py_rec["b"]["b"]["x_e"]), np.std(st.py_rec["b"]["b"]["x_e"]), COL_PY))
    if st.py_full:
        rows.append(("PYTHIA, full chain, $b$ drawn", np.mean(st.py_full["b"]["b"]["x_e"]), np.std(st.py_full["b"]["b"]["x_e"]), COL_CMSSW))
    if st.py_full_b:
        rows.append(("PYTHIA, full chain, $b$ = 0.3136", np.mean(st.py_full_b["b"]["b"]["x_e"]), np.std(st.py_full_b["b"]["b"]["x_e"]), COL_CMSSW))
    rows.append((r"ALEPH, $Z \to b\bar b$, weakly decaying B", ALEPH_XE, 0.0, COL_STR))
    rows.append(("our event (seed 1)", st.b["x_e"], 0.0, COL_EVENT))
    rows.append(("CMSSW event ($x_{p_T}$)", CMSSW["B_pt"] / CMSSW["b_pt"], 0.0, COL_CMSSW))
    for j, (lab, val, err, col) in enumerate(rows):
        a3.errorbar([val], [len(rows) - 1 - j], xerr=[err], fmt="o", color=col, ms=7, capsize=3)
        a3.text(0.32, len(rows) - 1 - j + 0.2, lab, fontsize=9, color=col)
    a3.set_yticks([])
    a3.set_xlim(0.3, 1.3)
    a3.set_ylim(-0.7, len(rows) - 0.2)
    a3.set_xlabel(r"$\langle x_E \rangle$ (bars: spread of the distribution)")
    a3.set_title("(c) the same word, different quantities", fontsize=10.5)
    fig.suptitle(r"$x_B$ in different settings: the hadron, the jet, the ensemble (page 6)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
def page_our_jet(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    c = st.b["cones"]
    for R, col, ls in ((0.4, COL_STR, "-"), (0.8, COL_A, "-")):
        for d, lab, ls2 in ((st.stats, "this script", "-"), (st.py_rec, "PYTHIA, same partons", "--")):
            if not d:
                continue
            v = np.array(d["b"]["b"]["cones"][str(R)]["pt_hadrons"])
            a1.hist(v, bins=30, range=(85, 120), density=True, histtype="step", color=col, ls=ls2, lw=1.8,
                    label=f"R = {R}, {lab}: {v.mean():.1f} $\\pm$ {v.std():.1f} GeV")
        mark(a1, c[R]["pt_partons"], f"partons R = {R}: {c[R]['pt_partons']:.1f}", col, y=0.42 if R == 0.4 else 0.32)
        a1.axvline(c[R]["pt_hadrons"], color=col, lw=1.5, ls=":")
    a1.set_xlabel("$p_T$ of the hadrons within $\\Delta R$ of the $b$ quark  [GeV]")
    a1.set_ylabel("density")
    a1.set_ylim(0, None)
    a1.set_title("(a) the cone $p_T$ before (dashed line) and after hadronisation", fontsize=10.5)
    a1.legend(loc="upper left", fontsize=7.5)
    a1.text(0.98, 0.95, "dotted: our event", transform=a1.transAxes, ha="right", va="top", fontsize=8.5, color=TEXT2)
    # (b) mass and multiplicity
    for R, col in ((0.4, COL_STR), (0.8, COL_A)):
        if st.stats:
            m = np.array(st.stats["b"]["b"]["cones"][str(R)]["m_hadrons"])
            n = np.array(st.stats["b"]["b"]["cones"][str(R)]["n_hadrons"])
            a2.plot(n, m, ".", color=col, ms=4, alpha=0.7, label=f"R = {R}: {n.mean():.1f} hadrons, mass {m.mean():.1f} GeV (partons: {c[R]['n_partons']}, {c[R]['m_partons']:.1f})")
        a2.plot([c[R]["n_hadrons"]], [c[R]["m_hadrons"]], "o", color=COL_EVENT, ms=8, mec="white", zorder=5)
        a2.plot([c[R]["n_partons"]], [c[R]["m_partons"]], "D", color=col, ms=8, mec="white", zorder=5)
    a2.plot([CMSSW["n04"]], [CMSSW["m04"]], "s", color=COL_CMSSW, ms=8, mec="white", label="CMSSW event (R = 0.4 and 0.8)", zorder=5)
    a2.plot([CMSSW["n08"]], [CMSSW["m08"]], "s", color=COL_CMSSW, ms=8, mec="white", zorder=5)
    a2.plot([], [], "o", color=COL_EVENT, label="our event (hadrons)")
    a2.plot([], [], "D", color=TEXT2, label="our event (partons)")
    a2.set_xlabel("hadrons in the cone")
    a2.set_ylabel("cone mass  [GeV]")
    a2.set_title("(b) more particles, more mass, same $p_T$", fontsize=10.5)
    a2.legend(loc="upper left", fontsize=8)
    # (c) the ledger
    labels = ["$b$ quark", "B hadron", "cone 0.4\npartons", "cone 0.4\nhadrons", "cone 0.8\npartons", "cone 0.8\nhadrons"]
    ours = [st.b["pt_quark"], st.b["pt_hadron"], c[0.4]["pt_partons"], c[0.4]["pt_hadrons"], c[0.8]["pt_partons"], c[0.8]["pt_hadrons"]]
    mean = None
    if st.stats:
        d = st.stats["b"]["b"]
        mean = [st.b["pt_quark"], np.mean(d["pt_hadron"]), c[0.4]["pt_partons"], np.mean(d["cones"]["0.4"]["pt_hadrons"]), c[0.8]["pt_partons"],
                np.mean(d["cones"]["0.8"]["pt_hadrons"])]
    cms = [CMSSW["b_pt"], CMSSW["B_pt"], None, CMSSW["cone04"], None, CMSSW["cone08"]]
    xpos = np.arange(len(labels))
    a3.bar(xpos - 0.27, ours, width=0.27, color=COL_EVENT, label="our event (seed 1)")
    if mean:
        a3.bar(xpos, mean, width=0.27, color=COL_A, label=f"this script, mean of {st.stats['repeat']}")
    a3.bar(xpos + 0.27, [v or 0 for v in cms], width=0.27, color=COL_CMSSW, label="CMSSW event")
    for i, v in enumerate(ours):
        a3.text(i - 0.27, v + 1, f"{v:.1f}", ha="center", fontsize=7.5, color=COL_EVENT)
    for i, v in enumerate(cms):
        if v:
            a3.text(i + 0.27, v + 1, f"{v:.1f}", ha="center", fontsize=7.5, color=COL_CMSSW)
    a3.set_xticks(xpos)
    a3.set_xticklabels(labels, fontsize=8.5)
    a3.set_ylabel("$p_T$  [GeV]")
    a3.set_title("(c) the ledger: quark, hadron, jet", fontsize=10.5)
    a3.legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Our jet at this stage: the hadron-level family and the B hadron's share (page 7)", fontsize=12)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, default=HERE / "theory_hadronisation.pdf")
    ap.add_argument("--page", type=int, default=None, help="render one page only (1-7)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=160)
    a = ap.parse_args()
    st = Setup(a.seed)
    print(f"seed-{a.seed} hadronisation: {st.summary['n_hadrons']} hadrons from {len(st.had.strings)} systems; the b quark's string has "
          f"{len(st.bstring['partons'])} partons and {len(st.bstring['hadrons'])} hadrons; statistics: "
          + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in (("standalone", st.stats), ("rfactb0", st.stats_r0), ("peterson", st.stats_pet),
                                                                   ("PYTHIA record", st.py_rec), ("PYTHIA full", st.py_full), ("PYTHIA fixed b", st.py_full_b))))
    makers = {1: lambda: page_string(st), 2: lambda: page_breakup(st), 3: lambda: page_z(st), 4: lambda: page_kinks(st),
              5: lambda: page_bfrag(st), 6: lambda: page_xb(st), 7: lambda: page_our_jet(st)}
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
