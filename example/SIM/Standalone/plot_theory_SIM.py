#!/usr/bin/env python3
"""The theory figures of chapter 10: one page per equation, drawn with the functions of standalone_SIM.py.

  python3 plot_theory_SIM.py -o theory_SIM.pdf        (about a minute: pages 4 to 7 shoot single particles)

 page 1  the helix: R = pT / (0.3 B) and the looper thresholds, with the pT spectrum of our charged primaries
 page 2  decays in flight: the probability to decay inside the tracker, and where our event's decays happened
 page 3  the Bethe-Bloch curve in silicon and the energy loss of our tracker hits (Landau)
 page 4  the material in front of the calorimeters: the layer model against the CMS geometry (TGeo), and what it
         does to photons (conversions) and hadrons (nuclear interactions)
 page 5  a radiography of the detector model: the vertices of conversions and nuclear interactions
 page 6  the electromagnetic shower: longitudinal profile, and how many crystals share it
 page 7  the hadronic shower: the electromagnetic fraction and the response of ECAL + HCAL to pions
 page 8  the three crowds in time: the hits after the shift by 25 ns x bx, and the energy in the circle of our jet
 page 9  the crowd collision by collision: hits and energy against the number of primaries
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import gamma as gamma_fn

import add_hits_SIM as add
import standalone_SIM as sim

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100})
C = {"sig": "#cc0000", "it": "#3366cc", "early": "#999900", "late": "#ff6666", "cms": "black", "model": "#cc6600"}


def gun(pid, p, eta, n, stage="full", seed=7, first_only=False, flat_eta=None):
    out = []
    m = sim.dec.m0(pid)
    for i in range(n):
        rng = np.random.default_rng([seed, abs(pid), int(p * 1000), i])
        s = sim.Sim(rng, stage=stage, record=True)
        e = rng.uniform(-flat_eta, flat_eta) if flat_eta else eta
        phi = rng.uniform(-math.pi, math.pi)
        pt = p / math.cosh(e)
        s.push(pid, (0.0, 0.0, 0.0), np.array([pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(e)]), 0.0, 0, 0, primary=True)
        first = True
        while s.stack:
            s.track(s.stack.pop())
            if first_only and first:
                s.stack.clear()
            first = False
        s.eta0 = e
        out.append(s)
    return out


def outline(ax):
    g = dict(color="0.8", lw=0.5, zorder=0)
    for b in sim.BARREL:
        if b[0] != 0:
            for sg in (1, -1):
                ax.plot([sg * b[3], sg * b[4]], [b[2], b[2]], **g)
    for d in sim.DISKS:
        if d[0] != 0:
            ax.plot([d[2], d[2]], [d[3], d[4]], **g)


def page_field(pdf, S):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.6))
    t = np.linspace(0, 2 * math.pi, 300)
    for bl in sim.BARREL:
        if bl[0] in (sim.PXB, sim.TIB, sim.TOB):
            a.plot(bl[2] * np.cos(t), bl[2] * np.sin(t), color="0.8", lw=0.6)
    a.plot(sim.R_EB * np.cos(t), sim.R_EB * np.sin(t), color="#009900", lw=1.5)
    a.text(0, sim.R_EB + 4, "ECAL front face, r = 129 cm", ha="center", color="#009900")
    for pt, col in ((0.15, "#aa00aa"), (0.35, "#3366cc"), (0.735, "#cc0000"), (1.5, "#ff8800"), (5.0, "k")):
        R = sim.helix_radius(pt)
        s = np.linspace(0, min(2 * math.pi * R, 450), 600)
        x, y = R * np.sin(s / R), -R * (1 - np.cos(s / R))          # a positive particle starting along +x
        keep = np.hypot(x, y) <= sim.R_EB + 0.5
        if not keep.all():
            keep[np.argmin(keep):] = False
        a.plot(x[keep], y[keep], color=col, lw=1.6, label=f"$p_T$ = {pt:g} GeV: R = {R:.0f} cm")
    a.set_aspect("equal"); a.set_xlim(-150, 150); a.set_ylim(-150, 150)
    a.set_xlabel("x [cm]"); a.set_ylabel("y [cm]"); a.legend(fontsize=9, loc="lower left")
    a.set_title(r"$R = p_T/(0.3\,B)$ in 3.8 T: who reaches the calorimeter")
    sg = S["signal"]
    pt = np.hypot(sg["prim_p4"][:, 1], sg["prim_p4"][:, 2])
    eta = np.arcsinh(sg["prim_p4"][:, 3] / np.maximum(pt, 1e-9))
    ch = np.array([sim.dec.charge3(int(i)) != 0 for i in sg["prim_id"]])
    m = ch & (np.abs(eta) < 2.5)
    bins = np.logspace(-1.6, 1.4, 31)
    b.hist(pt[m], bins=bins, color="0.6", label=f"the {int(m.sum())} charged primaries of the signal, $|\\eta|<2.5$")
    b.hist(pt[m & (np.abs(eta) < 1.479)], bins=bins, color=C["sig"], alpha=0.8, label=f"of them in $|\\eta|<1.479$: {int((m & (np.abs(eta) < 1.479)).sum())}")
    for r, name, col in ((2.9, "pixel layer 1", "#aa00aa"), (25.5, "inner barrel", "#3366cc"), (60.8, "outer barrel", "#ff8800"), (129.0, "barrel ECAL", "#009900")):
        th = sim.looper_pt(r)
        b.axvline(th, color=col, ls="--")
        b.text(th * 1.05, b.get_ylim()[1] * (0.97 - 0.09 * [2.9, 25.5, 60.8, 129.0].index(r)), f"{name}\n{th * 1e3:.0f} MeV", color=col, fontsize=8.5, va="top")
    b.set_xscale("log"); b.set_xlabel(r"$p_T$ [GeV]"); b.set_ylabel("primaries")
    n_low = int((m & (pt < sim.looper_pt(129.0))).sum())
    b.set_title(f"$p_T > 0.3\\,B\\,r/2$: {n_low} of {int(m.sum())} cannot reach the barrel ECAL")
    b.legend(fontsize=9, loc="center right")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def page_decay(pdf, S):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.4))
    p = np.logspace(-1, 2, 200)
    L = sim.R_TRK
    for pid, name, col in ((211, r"$\pi^\pm$ ($c\tau$ = 7.8 m)", "k"), (321, r"$K^\pm$ (3.7 m)", "#3366cc"), (130, r"$K^0_L$ (15.3 m)", "0.5"),
                           (3122, r"$\Lambda$ (7.9 cm)", "#aa00aa"), (310, r"$K^0_S$ (2.7 cm)", C["sig"])):
        m = sim.dec.m0(pid)
        ct = sim.dec.tau0(pid) / 10.0
        a.plot(p, 1 - np.exp(-L * m / (p * ct)), color=col, lw=1.8, label=name)
    a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(1e-4, 1.5)
    a.set_xlabel("p [GeV]"); a.set_ylabel(f"probability to decay within L = {L:.0f} cm")
    a.set_title(r"$P = 1 - \exp(-L\,m / (p\,c\tau))$, straight flight")
    a.legend(fontsize=9)
    v = S["signal"]["vtx"]
    outline(b)
    d = v[v[:, 0] == sim.PROC["decay"]]
    for pid, col, name in ((211, "k", r"$\pi^\pm$"), (321, "#3366cc", r"$K^\pm$"), (310, C["sig"], r"$K^0_S$"), (3122, "#aa00aa", r"$\Lambda$"), (0, "#009900", "other")):
        mm = np.abs(d[:, 5]) == pid if pid else ~np.isin(np.abs(d[:, 5]), (211, 321, 310, 3122))
        if mm.any():
            b.scatter(d[mm, 4], np.hypot(d[mm, 2], d[mm, 3]), s=12 + 6 * np.sqrt(d[mm, 6]), color=col, label=f"{name}: {int(mm.sum())}", zorder=3)
    b.set_xlim(-300, 300); b.set_ylim(0, 125); b.set_xlabel("z [cm]"); b.set_ylabel("r [cm]")
    b.set_title(f"the {len(d)} decays in flight of the signal (marker grows with E)")
    b.legend(fontsize=9, ncol=5, loc="upper center")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def page_bethe(pdf, S):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2))
    bg = np.logspace(-1, 4, 300)
    a.plot(bg, [sim.bethe_si(x) for x in bg], "k", lw=1.8)
    a.axhline(sim.DEDX_MIN, color="0.5", ls=":")
    for name, m, p, col in ((r"$\pi$ 0.1 GeV", sim.MPI, 0.1, "#3366cc"), (r"$\pi$ 1 GeV", sim.MPI, 1.0, "#3366cc"), (r"$\pi$ 10 GeV", sim.MPI, 10.0, "#3366cc"),
                            ("p 0.3 GeV", sim.MP, 0.3, C["sig"]), ("p 1 GeV", sim.MP, 1.0, C["sig"]), ("K 0.3 GeV", 0.4937, 0.3, "#009900")):
        x = p / m
        y = sim.bethe_si(x)
        a.plot([x], [y], "o", color=col)
        a.annotate(name, (x, y), xytext=(4, 6), textcoords="offset points", fontsize=8.5, color=col)
    a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(1, 60)
    a.set_xlabel(r"$\beta\gamma = p/m$"); a.set_ylabel(r"$\langle -dE/dx\rangle$ in silicon [MeV cm$^2$/g]")
    a.set_title(f"Bethe-Bloch: minimum {sim.DEDX_MIN:.3f} MeV cm$^2$/g at $\\beta\\gamma\\approx3.5$")
    sg = S["signal"]
    bins = np.logspace(1.3, 3.7, 60)
    for det, name, col in ((sim.TIB, "inner barrel (320 $\\mu$m)", "#3366cc"), (sim.TOB, "outer barrel (500 $\\mu$m)", "#ff8800"), (sim.PXB, "pixel barrel (285 $\\mu$m)", "#aa00aa")):
        de = sg["trk_de"][sg["trk_det"] == det] * 1e6
        b.hist(de, bins=bins, histtype="step", lw=1.6, color=col, label=f"{name}: {len(de)} hits, median {np.median(de):.0f} keV")
    b.axvline(sim.landau_mpv(3.5, 0.032) * 1e6, color="#3366cc", ls="--")
    b.text(sim.landau_mpv(3.5, 0.032) * 1e6 * 0.93, b.get_ylim()[1] * 0.6, "most probable loss of a MIP\nat normal incidence, 320 $\\mu$m: 84 keV", ha="right", fontsize=8.5, color="#3366cc")
    b.set_xscale("log"); b.set_xlabel("energy lost in the sensor [keV]"); b.set_ylabel("tracker hits of the signal")
    b.set_title("Landau: a peak and a long tail (slow particles, grazing angles)")
    b.legend(fontsize=8.5)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def page_material(pdf):
    V = json.loads((HERE / "validate_SIM.json").read_text())
    sc = V["scan"]
    eta = np.array([s["eta"] for s in sc])
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.4))
    x0, rms = np.array([s["x0"] for s in sc]), np.array([s["x0_rms"] for s in sc])
    a.fill_between(eta, x0 - rms, x0 + rms, color="0.85", label="rms over azimuth")
    a.plot(eta, x0, color="k", lw=1.8, label="CMS simulation geometry (TGeo): $x/X_0$")
    a.plot(eta, [s["model_x0"] for s in sc], color=C["model"], lw=1.8, ls="--", label="layer model of the script: $x/X_0$")
    a.plot(eta, [s["lam"] for s in sc], color="#3366cc", lw=1.8, label=r"CMS: $x/\lambda_I$")
    a.plot(eta, [s["model_lam"] for s in sc], color="#3366cc", lw=1.8, ls="--", label=r"layer model: $x/\lambda_I$")
    a.set_xlabel(r"$\eta$"); a.set_ylabel("material before the calorimeters"); a.set_ylim(0, 2.6)
    a.set_title(f"tracker material: model / CMS = {V['mean_ratio_x0']:.2f} ($X_0$), {V['mean_ratio_lam']:.2f} ($\\lambda_I$) for $|\\eta|<2.5$")
    a.legend(fontsize=8.5, ncol=2, loc="upper center")
    # what it does: photons and pions shot through the model
    bins = np.linspace(-3, 3, 25)
    cen = 0.5 * (bins[1:] + bins[:-1])
    for pid, p, col, lab, fate in ((22, 5.0, "#009900", "5 GeV photons that convert", "conversion"), (211, 5.0, "#aa00aa", r"5 GeV $\pi^+$ with a nuclear interaction", "nuclear")):
        g = gun(pid, p, 0.0, 4000, first_only=True, flat_eta=3.0)
        e = np.array([s.eta0 for s in g])
        hit = np.array([s.fate.get(0) == fate for s in g])
        n, _ = np.histogram(e, bins)
        k, _ = np.histogram(e[hit], bins)
        f = k / np.maximum(n, 1)
        b.errorbar(cen, f, yerr=np.sqrt(f * (1 - f) / np.maximum(n, 1)), fmt="o", ms=4, color=col, label=f"{lab}: {100 * hit.mean():.0f} % on average")
    mx = np.array([s["x0"] for s in sc]); ml = np.array([s["lam"] for s in sc])
    b.plot(eta, 1 - np.exp(-7 / 9 * mx * sim.pair_fraction(5.0)), color="#009900", lw=1.5, label=r"$1-e^{-7t/9}$ with $t$ of the CMS geometry")
    b.plot(eta, 1 - np.exp(-ml), color="#aa00aa", lw=1.5, label=r"$1-e^{-x/\lambda_I}$ with the CMS geometry")
    b.set_xlabel(r"$\eta$"); b.set_ylabel("fraction lost before the calorimeters"); b.set_ylim(0, 1.0)
    b.set_title("what the material does to a photon and to a hadron")
    b.legend(fontsize=8.5, loc="upper center")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def page_radiography(pdf):
    fig, ax = plt.subplots(figsize=(13, 5.6))
    outline(ax)
    for pid, n, col, lab in ((22, 9000, "#009900", "conversions of 5 GeV photons"), (211, 6000, "#aa00aa", r"nuclear interactions of 5 GeV $\pi^+$")):
        g = gun(pid, 5.0, 0.0, n, first_only=True, flat_eta=3.2, seed=11)
        v = np.array([s.vtx[0][2:5] for s in g if s.vtx and s.vtx[0][0] in (sim.PROC["conv"], sim.PROC["had"])])
        ax.plot(v[:, 2], np.hypot(v[:, 0], v[:, 1]), ".", ms=2.5, color=col, label=f"{lab}: {len(v)} of {n}", rasterized=True)
    ax.set_xlim(-300, 300); ax.set_ylim(0, 125); ax.set_xlabel("z [cm]"); ax.set_ylabel("r [cm]")
    ax.set_title("a radiography of the detector model: where photons convert and hadrons interact (flat in $\\eta$, $|\\eta|<3.2$)")
    ax.legend(fontsize=9, loc="upper right", markerscale=5)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def page_em(pdf):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2))
    t = np.linspace(0.01, 28, 400)
    for e, col in ((1.0, "#3366cc"), (10.0, "#009900"), (100.0, C["sig"])):
        m_t, s_t, m_a, s_a, rho = sim.gp_parameters(e)
        al = math.exp(m_a)
        be = (al - 1) / math.exp(m_t)
        a.plot(t, be * (be * t) ** (al - 1) * np.exp(-be * t) / gamma_fn(al), color=col, lw=1.8,
               label=f"E = {e:g} GeV: $t_{{max}}$ = {math.exp(m_t):.1f} $X_0$ = {math.exp(m_t) * sim.X0_PBWO4:.1f} cm")
    a.axvline((sim.R_EB_BACK - sim.R_EB) / sim.X0_PBWO4, color="k", ls="--")
    a.text(25.5, a.get_ylim()[1] * 0.9, "end of a\nbarrel crystal\n25.8 $X_0$", ha="right", fontsize=9)
    a.set_xlabel("depth t [$X_0$ = 0.89 cm of PbWO$_4$]"); a.set_ylabel(r"$(1/E)\,dE/dt$")
    a.set_title(r"$dE/dt \propto (\beta t)^{\alpha-1}e^{-\beta t}$, $t_{max}=(\alpha-1)/\beta$", fontsize=11)
    a.legend(fontsize=9)
    g = gun(22, 10.0, 0.5, 400, seed=5)
    rows = {"unconverted": [], "converted in the tracker": []}
    for s in g:
        r = s.result(compress=True)
        c = r["cal"]
        m = (c["code"] // sim.CODE) == sim.EB
        e = np.sort(c["e"][m])[::-1]
        if len(e) == 0:
            continue
        tot = e.sum()
        cum = np.cumsum(e) / 10.0
        key = "converted in the tracker" if s.fate.get(0) == "conversion" else "unconverted"
        rows[key].append((e[0] / 10.0, e[:9].sum() / 10.0, e[:25].sum() / 10.0, tot / 10.0, int(np.searchsorted(cum, 0.9 * tot / 10.0) + 1), len(e)))
    w = 0.38
    for i, (key, col) in enumerate((("unconverted", "#009900"), ("converted in the tracker", "#ff8800"))):
        r_ = np.array(rows[key])
        mean = r_[:, :4].mean(axis=0)
        b.bar(np.arange(4) + (i - 0.5) * w, mean, w, color=col, yerr=r_[:, :4].std(axis=0), capsize=3,
              label=f"{key}: {len(r_)} photons; 90 % of the deposit in {np.median(r_[:, 4]):.0f} crystals (median)")
        for j, v in enumerate(mean):
            b.text(j + (i - 0.5) * w, 0.03, f"{v:.2f}", ha="center", color="w", fontsize=9, fontweight="bold")
    b.set_xticks(range(4)); b.set_xticklabels(["hottest crystal", "9 hottest", "25 hottest", "all barrel crystals"])
    b.set_ylabel("fraction of the photon's energy"); b.set_ylim(0, 1.25)
    b.set_title(r"400 photons of 10 GeV at $\eta=0.5$", fontsize=11)
    b.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def page_had(pdf):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2))
    e = np.logspace(-0.3, 3, 200)
    f = np.array([sim.f_em_mean(x) for x in e])
    fp = np.array([sim.f_em_mean(x, sim.E0_P) for x in e])
    a.plot(e, f, color=C["sig"], lw=1.8, label=r"pions: $f_{em} = 1-(E/0.8)^{k-1}$, $k=0.82$")
    a.plot(e, fp, color="#3366cc", lw=1.8, label=r"protons: $E_0 = 2.6$ GeV")
    a.plot(e, f + (1 - f) * sim.H_OVER_E_HCAL, color=C["sig"], ls="--", lw=1.5, label=r"seen by the HCAL: $f_{em} + (1-f_{em})\,h/e$, $h/e = 0.72$")
    a.plot(e, f + (1 - f) * sim.H_OVER_E_ECAL, color="#009900", ls="--", lw=1.5, label=r"seen by the crystals: $h/e = 0.62$")
    a.set_xscale("log"); a.set_ylim(0, 1.05); a.set_xlabel("E [GeV]"); a.set_ylabel("fraction")
    a.set_title("the electromagnetic fraction, and what a calorimeter sees", fontsize=11)
    a.legend(fontsize=8.5, loc="lower right")
    es = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 300.0]
    res, rms, fe, mipf = [], [], [], []
    for x in es:
        g = gun(211, x, 0.5, 250, seed=3)
        vals, ecal_frac, nm = [], [], 0
        for s in g:
            if s.fate.get(0) != "calorimeter":
                continue
            c = s.result(compress=True)["cal"]
            det = c["code"] // sim.CODE
            ee, eh = c["e"][det == sim.EB].sum(), c["e"][det == sim.HB].sum() * 125.0
            vals.append((ee + eh) / x)
            ecal_frac.append(ee / max(ee + eh, 1e-9))
            nm += ee < 0.6
        res.append(np.mean(vals)); rms.append(np.std(vals)); fe.append(np.mean(ecal_frac)); mipf.append(nm / max(len(vals), 1))
    b.errorbar(es, res, yerr=rms, fmt="o-", color=C["sig"], capsize=3, label=r"(crystals + 125 $\times$ HB scintillator) / E: mean, bars = rms")
    b.plot(es, fe, "s--", color="#009900", label="share of the crystals in that sum")
    b.plot(es, mipf, "^:", color="0.4", label="pions crossing the crystals as a MIP (< 0.6 GeV there)")
    b.set_xscale("log"); b.set_ylim(0, 1.3); b.set_xlabel(r"$E_\pi$ [GeV]"); b.set_ylabel("fraction")
    b.set_title(r"250 $\pi^+$ per energy at $\eta = 0.5$", fontsize=11)
    b.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return dict(e=es, response=[float(x) for x in res], rms=[float(x) for x in rms], ecal_share=[float(x) for x in fe], mip=[float(x) for x in mipf])


def page_time(pdf, S, N):
    M = add.mix(S)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2))
    t, o = M["trk"]["t"], M["trk"]["origin"]
    bins = np.arange(-305, 130, 1.0)
    for i, (name, col) in enumerate((("signal", "k"), ("early crowd, bx $-12\\ldots-1$", C["early"]), ("in-time crowd, bx 0", C["it"]), ("late crowd, bx $+1\\ldots+3$", C["late"]))):
        a.hist(t[o == i], bins=bins, histtype="stepfilled" if i else "step", color=col, alpha=0.75 if i else 1.0, lw=1.5, label=f"{name}: {int((o == i).sum()):,d} hits".replace(",", " "), zorder=5 if i == 0 else 2)
    a.set_yscale("log"); a.set_xlabel("time of the tracker hit [ns], 0 = our bunch crossing"); a.set_ylabel("hits per ns")
    a.set_title(r"$t = t_{flight} + 25\,\mathrm{ns}\times bx$: sixteen crossings on one clock")
    a.legend(fontsize=8.5, loc="upper left")
    et = np.array(N["circle"]["et_bx"])
    sig = N["circle"]["origins"]["signal"]["ecal_et"] + N["circle"]["origins"]["signal"]["hcal_et"]
    bx = np.arange(-12, 4)
    crowd = et.copy(); crowd[12] -= sig
    b.bar(bx, crowd, color=[C["early"] if x < 0 else (C["it"] if x == 0 else C["late"]) for x in bx], label="the crowd, by bunch crossing")
    b.bar([0], [sig], bottom=[crowd[12]], color="k", label=f"the family: {sig:.0f} GeV")
    b.axhline(crowd.mean(), color="0.4", ls=":")
    b.text(-12.3, crowd.mean() * 1.05, f"mean of the crowd {crowd.mean():.1f} GeV per crossing", fontsize=9)
    b.set_xlabel("bunch crossing"); b.set_ylabel(r"$\sum E_T$ of the cells in $\Delta R<0.4$ around our jet [GeV]")
    b.set_title(f"family {sig:.0f} GeV, crowd {crowd.sum():.0f} GeV ({crowd[12]:.0f} GeV in time)", fontsize=11)
    b.legend(fontsize=9, loc="upper left")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def page_scaling(pdf, S):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2))
    out = {}
    for name, col, lab in (("oot", C["early"], "out of time"), ("intime", C["it"], "in time")):
        d = S[name]
        kk = d["col_k"].astype(int)
        nk = int(kk.max()) + 1
        tr = np.bincount(d["trk_k"].astype(int), minlength=nk)[kk]
        det = d["cal_code"] // sim.CODE
        m = np.isin(det, (sim.EB, sim.EE))
        ee = np.bincount(d["cal_k"][m].astype(int), weights=d["cal_e"][m], minlength=nk)[kk]
        a.plot(d["col_nprim"], tr, ".", color=col, ms=4, label=f"{lab}: {len(kk)} collisions", rasterized=True)
        b.plot(d["col_nprim"], ee, ".", color=col, ms=4, label=lab, rasterized=True)
        out[name] = (d["col_nprim"], tr, ee)
    n = np.concatenate([out["oot"][0], out["intime"][0]]); tr = np.concatenate([out["oot"][1], out["intime"][1]]); ee = np.concatenate([out["oot"][2], out["intime"][2]])
    k1, k2 = tr.sum() / n.sum(), ee.sum() / n.sum()
    x = np.array([0, n.max()])
    a.plot(x, k1 * x, "k--", label=f"{k1:.1f} hits per primary; mean {tr.mean():.0f} hits per collision")
    b.plot(x, k2 * x, "k--", label=f"{k2:.2f} GeV per primary; mean {ee.mean():.0f} GeV per collision")
    a.set_xlabel("primaries of the collision"); a.set_ylabel("tracker hits"); a.set_title("every collision of the crowd: tracker hits"); a.legend(fontsize=9)
    b.set_xlabel("primaries of the collision"); b.set_ylabel("energy in the crystals [GeV]"); b.set_title("energy in the ECAL crystals"); b.legend(fontsize=9)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return dict(hits_per_primary=float(k1), gev_per_primary=float(k2), hits_per_collision=float(tr.mean()), gev_per_collision=float(ee.mean()),
                median_hits=float(np.median(tr)), max_hits=int(tr.max()), min_hits=int(tr.min()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, default=HERE / "theory_SIM.pdf")
    args = ap.parse_args()
    S = add.load(HERE.parent)
    N = json.loads((HERE / "add_hits_SIM.json").read_text())
    extra = {}
    with PdfPages(args.output) as pdf:
        page_field(pdf, S)
        page_decay(pdf, S)
        page_bethe(pdf, S)
        page_material(pdf)
        page_radiography(pdf)
        page_em(pdf)
        extra["pion_response"] = page_had(pdf)
        page_time(pdf, S, N)
        extra["scaling"] = page_scaling(pdf, S)
    (HERE / "plot_theory_SIM.json").write_text(json.dumps(extra, indent=1))
    print(f"{args.output} written (9 pages)")
    print(json.dumps(extra))


if __name__ == "__main__":
    main()
