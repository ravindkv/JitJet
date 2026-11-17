#!/usr/bin/env python3
"""The plots behind the theory sections of chapter 9, drawn from the standalone hand-over script.

One page per theory subsection. Every curve is computed with the constants and functions of standalone_G4.py
(the Gate of Generator.cc, the PPS selection, the edges of the detector), every marker is a particle or a decay
of our window (the signal, the 63 in-time and the 897 out-of-time collisions of ../../PU/event_PU.lhe, rebuilt
from the library), and the panels by collision kind read the 2000 collisions of the library, each at a vertex
drawn from the 2025 beam spot.

  page 1  the edges          the particles and the energy per unit of eta of the signal and of the crowd, with the
                             edges of the detector and of the gate; how much of each is handed over below |eta|
  page 2  the line           the effective eta cut against the z of the vertex, with the border particles of
                             the window; the radius of the line at |z| = 354.8 cm; the effective cut over the
                             961 vertices
  page 3  soft, invisible    the momentum spectrum by kind with the 40 MeV cut; the chance that a photon of a
                             pi0 decay falls below it; where the energy that is not handed over goes
  page 4  the decays         the chance that a hadron decays outside r = 2.9 cm against its pT; the weak decays
                             of the signal in the transverse plane and in (z, r)
  page 5  the forward        the forward particles of the crowd in (|eta|, E) with the very-forward rule and the
                             PPS selection; xi of the forward protons; where the energy of a collision goes, by kind

    python3 plot_theory_G4.py                    # -> theory_G4.pdf
    python3 plot_theory_G4.py --page 2 -o p.png  # one page as PNG

Needs numpy and matplotlib (on this Mac: /usr/bin/python3). Ten seconds.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import standalone_G4 as core  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

COL_SIG, COL_PU, COL_OOT = "#d62728", "#8a7f5a", "#b8860b"
COL_KEPT, COL_DROP, COL_VF, COL_PPS, COL_NU = "#2a78d6", "#9a9a9a", "#2f8f5b", "#8e5bb5", "#eb6834"
COL_ND, COL_SD, COL_DD, COL_CD = "#2a78d6", "#2f8f5b", "#eb6834", "#8e5bb5"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d6d3ca"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "legend.fontsize": 9.2,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.edgecolor": TEXT2, "pdf.fonttype": 42, "mathtext.default": "regular", "figure.dpi": 100})
EDGES = [(2.5, "tracker", ":"), (3.0, "endcaps", ":"), (core.ETA_HF_FRONT, "HF hole", "--"), (core.ETA_HF_BACK, None, "--"),
         (core.ETA_CUT, "gate 5.5", "-"), (core.ETA_HECTOR, "8.2", "-")]
M_PI0 = 0.1349768


class Setup:
    def __init__(self, record: Path, library: Path, seed: int):
        args = argparse.Namespace(input=record, library=library, eta_cut=core.ETA_CUT, eta_hector=core.ETA_HECTOR, r_dec=core.R_DEC, p_min=core.P_MIN)
        self.st = core.Stage(args)
        self.R = self.st.compute()
        self.c, self.g, self.G = self.st.crowd, self.st.g, self.st.gate
        # the library at beam-spot vertices: one Crowd, with the kind of each collision
        lib = core.pu.Library(library)
        rng = np.random.default_rng(seed)
        vtx = core.pu.BeamSpot().draw(rng, len(lib.events))
        kk, ids, p4, vv = [], [], [], []
        for n, e in enumerate(lib.events):
            m = len(e["ids"])
            kk += [n] * m
            ids += list(e["ids"])
            p4 += list(e["p4"])
            vv += [[vtx[n, 0] * 1e-4, vtx[n, 1] * 1e-4, vtx[n, 2]]] * m
        self.lib_kind = np.array([e["kind"] for e in lib.events])
        self.lc = core.Crowd([1] * len(ids), kk, list(range(len(ids))), ids, p4, vv)
        self.lg = self.G.decide(self.lc)


def edges(ax, ymax_text=None, both=True, labels=True):
    for x, lab, ls in EDGES:
        for s in ((-1, 1) if both else (1,)):
            ax.axvline(s * x, color=TEXT2 if ls != "-" else TEXT, lw=0.9 if ls != "-" else 1.4, ls=ls, zorder=1)
        if labels and lab and ymax_text is not None:
            ax.text(x + 0.08, ymax_text, lab, rotation=90, va="top", ha="left", fontsize=8, color=TEXT2 if ls != "-" else TEXT)


# ---------------------------------------------------------------------------
# page 1: the edges
# ---------------------------------------------------------------------------
def page_edges(S: Setup):
    c, g = S.c, S.g
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    bins = np.linspace(-13, 13, 53)
    w = bins[1] - bins[0]
    sig, it = c.block == 0, c.block == 1
    nint = S.st.n_intime
    for m, col, lab, scale in ((it, COL_PU, f"in-time crowd, per collision ({nint})", 1.0 / nint), (sig, COL_SIG, "the signal", 1.0)):
        h, _ = np.histogram(c.eta[m & ~c.nu], bins=bins)
        a1.stairs(h * scale / w, bins, color=col, lw=1.6, label=lab)
        he, _ = np.histogram(c.eta[m & ~c.nu], bins=bins, weights=c.p4[m & ~c.nu, 0])
        a2.stairs(he * scale / w, bins, color=col, lw=1.6, label=lab)
    a1.set_yscale("log")
    a1.set_ylim(0.05, 300)
    a1.set_xlabel(r"$\eta$")
    a1.set_ylabel(r"stable particles per unit of $\eta$")
    a1.set_title("(a) where the particles go")
    edges(a1, 250, labels=True)
    a1.legend(loc="lower center", frameon=True, fontsize=8.6)
    a2.set_yscale("log")
    a2.set_ylim(0.5, 1e5)
    a2.set_xlabel(r"$\eta$")
    a2.set_ylabel(r"energy per unit of $\eta$ [GeV]")
    a2.set_title("(b) where the energy goes")
    edges(a2, 7e4, labels=False)
    a2.legend(loc="upper center", frameon=True, fontsize=8.6)
    # (c) the cumulative fraction below |eta|
    x = np.linspace(0, 13, 400)
    for m, col, lab in ((sig, COL_SIG, "signal"), (it, COL_PU, "in-time crowd")):
        a = np.abs(c.eta[m])
        E, pt = c.p4[m, 0], c.pt[m]
        fe = np.array([E[a < xx].sum() for xx in x]) / E.sum()
        fp = np.array([pt[a < xx].sum() for xx in x]) / pt.sum()
        a3.plot(x, fe, color=col, lw=1.8, label=f"{lab}: energy")
        a3.plot(x, fp, color=col, lw=1.4, ls="--", label=f"{lab}: $p_T$")
        f55e, f55p = E[a < core.ETA_CUT].sum() / E.sum(), pt[a < core.ETA_CUT].sum() / pt.sum()
        a3.plot([core.ETA_CUT], [f55e], "o", color=col, ms=5)
        a3.annotate(f"{100 * f55e:.0f} %", (core.ETA_CUT, f55e), xytext=(6, -12 if col == COL_PU else 4), textcoords="offset points", fontsize=8.5, color=col)
        a3.annotate(f"{100 * f55p:.0f} %", (core.ETA_CUT, f55p), xytext=(-30, -13 if col == COL_PU else 4), textcoords="offset points", fontsize=8.5, color=col)
    edges(a3, 0.62, both=False, labels=True)
    a3.set_xlim(0, 13)
    a3.set_ylim(0, 1.02)
    a3.set_xlabel(r"$|\eta|$ below which")
    a3.set_ylabel("fraction of the total")
    a3.set_title(r"(c) how much lies inside $|\eta|$")
    a3.legend(loc="lower right", frameon=True, fontsize=8.4)
    fig.suptitle("The edges of the country: the gate at |η| = 5.5 sits where the particles still are and the energy no longer is", fontsize=12.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# page 2: the eta cut is a line
# ---------------------------------------------------------------------------
def page_line(S: Setup):
    c, g, G = S.c, S.g, S.G
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    zv = np.linspace(-15, 15, 301)
    fwd = np.arcsinh((G.z_lmax - zv) / G.r)       # a particle flying towards +z from z_v
    bwd = np.arcsinh((G.z_lmax + zv) / G.r)       # towards -z
    a1.plot(zv, fwd, color=COL_KEPT, lw=1.8, label=r"towards $+z$: $\eta_c=\sinh^{-1}[(Z_{\max}-z_v)/R]$")
    a1.plot(zv, bwd, color=COL_NU, lw=1.8, label=r"towards $-z$: $\eta_c=\sinh^{-1}[(Z_{\max}+z_v)/R]$")
    a1.axhline(G.eta_cut, color=TEXT2, lw=0.8, ls=":")
    border = (c.block > 0) & (np.abs(c.eta) > G.eta_cut - 0.03) & (np.abs(c.eta) < G.eta_cut + 0.03) & ~g["very_forward"]
    kept = border & g["primary"]
    drop = border & ~g["primary"] & ~g["pps"]
    for m, mk, col, lab in ((kept, "o", COL_KEPT, "crowd, handed over"), (drop, "x", COL_DROP, "crowd, dropped")):
        a1.scatter(c.vtx[m, 2], np.abs(c.eta[m]), s=9, marker=mk, color=col, lw=0.8, label=f"{lab} ({int(m.sum())})", zorder=3)
    sb = (c.block == 0) & (np.abs(c.eta) > 5.3) & (np.abs(c.eta) < 5.7)
    a1.scatter(c.vtx[sb, 2], np.abs(c.eta[sb]), s=36, marker="*", color=COL_SIG, zorder=4, label=f"signal, $|\\eta|$ 5.3-5.7 ({int(sb.sum())})")
    a1.set_xlim(-15, 15)
    a1.set_ylim(G.eta_cut - 0.06, G.eta_cut + 0.045)
    a1.set_xlabel(r"$z$ of the production vertex $z_v$ [cm]")
    a1.set_ylabel(r"$|\eta|$ of the particle")
    a1.set_title(r"(a) the effective cut moves with the vertex")
    a1.legend(loc="lower center", fontsize=7.3, frameon=True, ncol=2)
    a1.text(14.5, G.eta_cut + 0.038, "above its own curve: dropped", ha="right", fontsize=8.5, color=TEXT2)
    # (b) the radius of the line at |z| = Z_lmax against |eta|, nominal and from the vertex
    m = (c.block > 0) & (np.abs(c.eta) > 5.40) & (np.abs(c.eta) < 5.60) & ~g["very_forward"]
    a2.scatter(np.abs(c.eta[m]), g["r_line"][m], s=4, color=np.where(g["primary"][m], COL_KEPT, COL_DROP), lw=0, zorder=3)
    ee = np.linspace(5.40, 5.60, 200)
    a2.plot(ee, G.z_lmax / np.sinh(ee), color=TEXT, lw=1.3, label=r"from $z_v=0$: $r=Z_{\max}/\sinh|\eta|$")
    for dz, ls in ((3.6, "--"), (-3.6, "--")):
        a2.plot(ee, (G.z_lmax - dz) / np.sinh(ee), color=TEXT2, lw=0.9, ls=ls)
    a2.text(5.405, (G.z_lmax + 3.6) / math.sinh(5.405) + 0.01, r"$z_v=\mp3.6$ cm", fontsize=8.2, color=TEXT2)
    a2.axhline(G.r, color="#cc0000", lw=1.2)
    a2.text(5.59, G.r + 0.01, "R = 2.9 cm", ha="right", va="bottom", fontsize=8.5, color="#cc0000")
    a2.axvline(G.eta_cut, color=TEXT2, lw=0.8, ls=":")
    a2.set_xlabel(r"$|\eta|$")
    a2.set_ylabel(r"$r$ of the line at $|z| = 354.8$ cm [cm]")
    a2.set_title("(b) the test: is the line still inside the cylinder?")
    a2.scatter([], [], s=12, color=COL_KEPT, label="crowd, handed over")
    a2.scatter([], [], s=12, color=COL_DROP, label="crowd, dropped")
    a2.legend(loc="upper right", fontsize=8.4, frameon=True)
    # (c) the effective cut over the vertices of the window
    zs = np.array([col["z"] for col in S.st.collisions])
    a3.hist(np.arcsinh((G.z_lmax - zs) / G.r), bins=np.linspace(5.465, 5.535, 36), color=COL_KEPT, alpha=0.6, label=r"towards $+z$")
    a3.hist(np.arcsinh((G.z_lmax + zs) / G.r), bins=np.linspace(5.465, 5.535, 36), histtype="step", color=COL_NU, lw=1.6, label=r"towards $-z$")
    a3.axvline(G.eta_cut, color=TEXT2, lw=0.8, ls=":")
    sz = float(np.std(zs))
    a3.set_xlabel(r"effective cut $\eta_c$ of a vertex")
    a3.set_ylabel(f"vertices of the window ({len(zs)})")
    a3.set_title(rf"(c) $\delta\eta_c \approx -z_v/Z_{{\max}}$: $\pm${sz / G.z_lmax:.3f} for $\sigma_z$ = {sz:.2f} cm")
    a3.legend(loc="upper right", fontsize=8.6, frameon=True)
    n_kb, n_di = len(S.R["line_vs_angle"]["kept_beyond"]), len(S.R["line_vs_angle"]["dropped_inside"])
    fig.suptitle(f"The eta cut is a line, not an angle: {n_kb} particles kept beyond |η| = 5.5 and {n_di} dropped inside it in our window", fontsize=12.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# page 3: the soft and the invisible
# ---------------------------------------------------------------------------
def p_below(E_pi, e_cut=core.P_MIN, m=M_PI0):
    """The chance that one photon of pi0 -> gamma gamma has an energy below e_cut: the photon energy is uniform between
    (E - p)/2 and (E + p)/2 in the laboratory (isotropic decay)."""
    E = np.asarray(E_pi, dtype=float)
    p = np.sqrt(np.maximum(E * E - m * m, 0.0))
    lo, hi = (E - p) / 2, (E + p) / 2
    return np.clip((e_cut - lo) / np.where(hi > lo, hi - lo, 1.0), 0.0, 1.0)


def page_soft(S: Setup):
    c, g = S.c, S.g
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    m0 = c.block < 2
    bins = np.logspace(-3.3, 3.5, 70)
    for kind, col in (("photon", "#2f8f5b"), ("charged hadron", TEXT), ("neutral hadron", "#6f6f6f"), ("electron", "#ff6600")):
        m = m0 & (c.kind == kind)
        h, _ = np.histogram(c.p[m], bins=bins)
        a1.stairs(h, bins, color=col, lw=1.5, label=f"{kind}s ({int(m.sum())}; {int((m & (c.p < core.P_MIN)).sum())} below)")
    a1.axvline(core.P_MIN, color="#cc0000", lw=1.3)
    a1.text(core.P_MIN * 1.1, 2e3, "40 MeV", color="#cc0000", fontsize=9)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_ylim(0.5, 5e3)
    a1.set_xlabel("|p| [GeV]")
    a1.set_ylabel("stable particles per bin (signal + in-time crowd)")
    a1.set_title("(a) the momentum cut: the soft tail")
    a1.legend(loc="upper left", fontsize=8.0, frameon=True)
    Epi = np.logspace(math.log10(M_PI0 * 1.0001), 2, 400)
    a2.plot(Epi, 2 * p_below(Epi), color="#2f8f5b", lw=1.8, label=r"$2\,P(E_\gamma<40$ MeV$)$ per $\pi^0$")
    a2.plot(Epi, 2 * core.P_MIN / Epi, color=TEXT2, lw=1.0, ls="--", label=r"$2E_c/E_\pi$ (high energy)")
    thr = (core.P_MIN ** 2 + (M_PI0 / 2) ** 2) / core.P_MIN
    a2.axvline(thr, color=TEXT2, lw=0.8, ls=":")
    a2.text(thr * 1.08, 0.015, f"threshold\n{thr * 1000:.0f} MeV", fontsize=8.2, color=TEXT2)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_ylim(1e-3, 2.2)
    a2.set_xlabel(r"$E_{\pi^0}$ [GeV]")
    a2.set_ylabel(r"photons below 40 MeV per $\pi^0$")
    a2.set_title(r"(b) $\pi^0\to\gamma\gamma$: the flat photon spectrum")
    a2.legend(loc="upper right", fontsize=8.6, frameon=True)
    # (c) the energy not handed over, by reason
    L = S.R["L"]
    labels = [r"$|\eta|>5.5$", "p < 40 MeV", "neutrinos", "other (line)", "to the\nRoman pots"]
    keys = ["eta", "p", "nu", "other"]
    x = np.arange(len(labels))
    for off, b, col, lab in ((-0.2, "signal", COL_SIG, "signal"), (0.2, "intime", COL_PU, f"in-time crowd ({S.st.n_intime})")):
        vals = [L[b]["drops"][k]["e"] for k in keys] + [L[b]["pps"]["e"]]
        vals = [max(v, 1e-3) for v in vals]
        a3.bar(x + off, vals, width=0.38, color=col, alpha=0.85, label=lab)
        for xi, v, k in zip(x + off, vals, keys + ["pps"]):
            n = L[b]["drops"][k]["n"] if k != "pps" else L[b]["pps"]["n"]
            a3.text(xi, v * 1.3 if n else 1.6e-2, f"{n}", ha="center", fontsize=8, color=col)
    a3.set_yscale("log")
    a3.set_ylim(1e-2, 3e6)
    a3.set_xticks(x)
    a3.set_xticklabels(labels, fontsize=9)
    a3.set_ylabel("energy [GeV] (number of particles above the bar)")
    a3.set_title("(c) what the gate does not hand over at the vertex")
    a3.legend(loc="upper right", fontsize=8.6, frameon=True)
    fig.suptitle("The soft and the invisible: 40 MeV removes a few soft photons, the StackingAction every neutrino", fontsize=12.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# page 4: the decays the generator made
# ---------------------------------------------------------------------------
LIFETIMES = [  # (name, mass [GeV], c tau [cm], colour, dashed): PDG values, as PYTHIA 8.315 has them
    (r"$B^\pm$", 5.2793, 0.04911, "#c0392b", False), (r"$B^0$", 5.2797, 0.04587, "#e67e22", False), (r"$\Lambda_b^0$", 5.6196, 0.0440, "#a04000", False),
    (r"$D^\pm$", 1.8697, 0.03118, "#2a78d6", False), (r"$D^0$", 1.8648, 0.01229, "#5dade2", False), (r"$\tau$", 1.7769, 0.00871, "#2f8f5b", False),
    (r"$K^0_S$ (if decayed)", 0.4976, 2.6844, "#7f7f7f", True), (r"$\Lambda$ (if decayed)", 1.1157, 7.89, "#b0b0b0", True)]


TEXH = {"Lambda_bbar0": r"$\bar\Lambda_b^0$", "Lambda_b0": r"$\Lambda_b^0$", "D-": r"$D^-$", "D+": r"$D^+$", "Bbar0": r"$\bar B^0$", "B0": r"$B^0$",
        "D_s+": r"$D_s^+$", "D_s-": r"$D_s^-$", "Dbar0": r"$\bar D^0$", "D0": r"$D^0$", "Xi_c0": r"$\Xi_c^0$", "Xi_cbar0": r"$\bar\Xi_c^0$",
        "tau+": r"$\tau^+$", "tau-": r"$\tau^-$", "B-": r"$B^-$", "B+": r"$B^+$"}


def texh(name: str) -> str:
    return TEXH.get(name, name)


def page_decays(S: Setup):
    G = S.G
    D = S.st.decays
    pv = S.st.pv
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    pt = np.logspace(-1, 3, 400)
    for name, m, ctau, col, dashed in LIFETIMES:
        a1.plot(pt, np.exp(-G.r * m / (pt * ctau)), color=col, lw=1.6, ls="--" if dashed else "-", label=name)
    for d in D.weak():
        lam = d["pt"] / d["m"] * d["ctau"]
        a1.plot(max(d["pt"], 0.1), math.exp(-G.r / lam) if lam > 0 else 0.0, "o", ms=6, mfc="white", mec=TEXT, mew=1.2, zorder=4)
    a1.plot([], [], "o", mfc="white", mec=TEXT, label=f"the {len(D.weak())} weak decays of our signal")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_ylim(1e-8, 1.5)
    a1.set_xlabel(r"$p_T$ of the hadron [GeV]")
    a1.set_ylabel(r"$P(r_{\rm dec}>2.9$ cm$) = e^{-Rm/(p_T c\tau)}$")
    a1.set_title("(a) who decays outside the cylinder?")
    a1.legend(loc="lower left", fontsize=7.6, frameon=True, ncol=2, bbox_to_anchor=(0.0, 0.0))
    a1.set_xlim(0.1, 1e3)
    # (b) the transverse plane
    th = np.linspace(0, 2 * math.pi, 300)
    a2.plot(G.r * np.cos(th), G.r * np.sin(th), color="#cc0000", lw=1.4, label="the cylinder, R = 2.9 cm")
    a2.plot(core.R_BEAMPIPE * np.cos(th), core.R_BEAMPIPE * np.sin(th), color=TEXT2, lw=1.0, ls="--", label="beam pipe, 2.25 cm")
    a2.plot(pv[0], pv[1], "+", color=TEXT, ms=9, mew=1.5, label="the collision")
    for d in D.weak():
        x, y = d["dec"][0], d["dec"][1]
        a2.plot([pv[0], x], [pv[1], y], color=TEXT2, lw=0.6)
        a2.plot(x, y, "o", ms=5, color=COL_KEPT)
        if math.hypot(x - pv[0], y - pv[1]) > 0.25:
            a2.annotate(texh(d["name"]), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8, color=TEXT)
    a2.set_aspect("equal")
    a2.set_xlim(-3.3, 3.3)
    a2.set_ylim(-3.3, 3.3)
    a2.set_xlabel("x [cm]")
    a2.set_ylabel("y [cm]")
    a2.set_title("(b) the weak decays of the signal, across the beam")
    a2.legend(loc="lower left", fontsize=7.8, frameon=True)
    # (c) z against r
    a3.axhline(G.r, color="#cc0000", lw=1.4)
    a3.axhline(core.R_BEAMPIPE, color=TEXT2, lw=1.0, ls="--")
    for d in D.weak():
        r = d["r_dec"]
        a3.plot(d["z_dec"], r, "o", ms=5, color=COL_KEPT)
        a3.plot([d["prod"][2], d["z_dec"]], [math.hypot(d["prod"][0], d["prod"][1]), r], color=TEXT2, lw=0.6)
        if d["pt"] > 5:
            a3.annotate(f"{texh(d['name'])} ({d['pt']:.0f})", (d["z_dec"], r), xytext=(3, 3), textcoords="offset points", fontsize=8, color=TEXT)
    a3.set_xlabel("z of the decay [cm]")
    a3.set_ylabel("r of the decay [cm]")
    a3.set_ylim(0, 3.2)
    a3.set_title(r"(c) the same along the beam ($p_T$ > 5 GeV labelled)")
    fig.suptitle("The decays the generator made: every one of them inside 2.9 cm, so every daughter starts as a primary", fontsize=12.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# page 5: the forward
# ---------------------------------------------------------------------------
def page_forward(S: Setup):
    c, g = S.c, S.g
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    crowd = c.block > 0
    a = np.abs(c.eta)
    fw = crowd & (a > 4.5)
    groups = [(fw & g["primary"] & ~g["very_forward"], COL_KEPT, "handed over at the vertex", 3),
              (fw & g["primary"] & g["very_forward"], COL_VF, r"very forward n, $\bar n$, $\gamma$: handed over (ZDC)", 5),
              (fw & g["pps"], COL_PPS, "protons to the Roman pots (PPS)", 9),
              (fw & ~g["primary"] & ~g["pps"], COL_DROP, "dropped", 2)]
    for m, col, lab, s in groups:
        a1.scatter(a[m], c.p4[m, 0], s=s, color=col, lw=0, label=f"{lab} ({int(m.sum())})", rasterized=True, zorder=3 if col != COL_DROP else 2)
    for x, ls in ((core.ETA_CUT, "-"), (core.ETA_HECTOR, "-"), (core.ETA_HF_FRONT, "--"), (core.ETA_HF_BACK, "--")):
        a1.axvline(x, color="#cc0000" if ls == "-" else TEXT2, lw=1.0, ls=ls)
    a1.set_yscale("log")
    a1.set_xlim(4.5, 13.5)
    a1.set_ylim(0.05, 1e4)
    a1.set_xlabel(r"$|\eta|$")
    a1.set_ylabel("E [GeV]")
    a1.set_title(f"(a) the forward crowd ({S.st.n_intime} + {S.st.n_oot} collisions)")
    a1.legend(loc="lower right", fontsize=7.8, frameon=True, markerscale=2)
    # (b) xi of the forward protons
    pr = crowd & (c.ids == 2212) & (a > core.PPS_ETA)
    xi = 1 - c.p[pr] / core.E_BEAM
    bins = np.linspace(0, 1, 41)
    a2.hist(xi[g["pps"][pr]], bins=bins, color=COL_PPS, alpha=0.75, label=f"to the Roman pots ({int(g['pps'][pr].sum())})")
    a2.hist(xi[~g["pps"][pr]], bins=bins, histtype="step", color=COL_DROP, lw=1.6, label=f"$|p_z|$ < 3 TeV: dropped ({int((~g['pps'][pr]).sum())})")
    a2.axvline(1 - core.PPS_PZ / core.E_BEAM, color="#cc0000", lw=1.2)
    a2.text(1 - core.PPS_PZ / core.E_BEAM + 0.015, 2.0, r"$\xi = 1 - 3000/6800$", fontsize=8.5, color="#cc0000", rotation=90, va="bottom")
    a2.set_yscale("log")
    a2.set_xlabel(r"$\xi = 1 - |p|/E_{\rm beam}$")
    a2.set_ylabel(r"protons beyond $|\eta| = 8.2$")
    a2.set_title("(b) the protons that keep most of the beam energy")
    a2.legend(loc="upper center", fontsize=8.4, frameon=True)
    # (c) where the energy of a collision goes, by kind (the library at beam-spot vertices)
    lc, lg = S.lc, S.lg
    kinds = ["ND", "SD", "DD", "CD"]
    parts = [("handed over at the vertex", lg["primary"] & ~lg["very_forward"], COL_KEPT), ("very forward (ZDC)", lg["primary"] & lg["very_forward"], COL_VF),
             ("Roman pots (PPS)", lg["pps"], COL_PPS), (r"dropped, $|\eta|>5.5$", ~lg["primary"] & ~lg["pps"] & (lg["reason"] == "eta"), COL_DROP),
             (r"neutrinos, soft, other", ~lg["primary"] & ~lg["pps"] & (lg["reason"] != "eta"), COL_NU)]
    kidx = np.asarray(lc.k)
    kind_of_particle = S.lib_kind[kidx]
    bottom = np.zeros(len(kinds))
    for lab, m, col in parts:
        vals = np.array([lc.p4[m & (kind_of_particle == k), 0].sum() / max((S.lib_kind == k).sum(), 1) / 13600.0 for k in kinds])
        a3.bar(kinds, vals, bottom=bottom, color=col, label=lab, width=0.6)
        for i, v in enumerate(vals):
            if v > 0.06:
                a3.text(i, bottom[i] + v / 2, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8.5, color="white")
        bottom += vals
    a3.set_xticks(range(len(kinds)))
    a3.set_xticklabels([f"{k}\n({int((S.lib_kind == k).sum())})" for k in kinds])
    a3.set_ylim(0, 1.0)
    a3.set_ylabel("share of the 13.6 TeV of a collision")
    a3.set_title("(c) the 2000 collisions of the library, by kind")
    a3.legend(loc="upper center", fontsize=7.6, frameon=True, ncol=2, bbox_to_anchor=(0.5, -0.15))
    fig.suptitle("The forward: neutrons and photons for the ZDC, protons for the Roman pots, the rest down the beam pipe", fontsize=12.5)
    fig.tight_layout()
    return fig


PAGES = [page_edges, page_line, page_soft, page_decays, page_forward]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=HERE.parent.parent / "PU" / "event_PU.lhe")
    ap.add_argument("-l", "--library", type=Path, default=HERE.parent.parent / "PU" / "pileup_library.lhe.gz")
    ap.add_argument("-o", "--output", type=Path, default=HERE / "theory_G4.pdf")
    ap.add_argument("--page", type=int, help="draw only this page (1-5)")
    ap.add_argument("--seed", type=int, default=9, help="the beam-spot seed of the library panels (as --library-scan)")
    args = ap.parse_args()
    S = Setup(args.input, args.library, args.seed)
    pages = [PAGES[args.page - 1]] if args.page else PAGES
    if args.output.suffix == ".png":
        fig = pages[0](S)
        fig.savefig(args.output, dpi=110)
    else:
        with PdfPages(str(args.output)) as pdf:
            for f in pages:
                fig = f(S)
                pdf.savefig(fig)
                plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
