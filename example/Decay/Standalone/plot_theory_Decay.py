#!/usr/bin/env python3
"""The plots behind the theory sections of chapter 6, drawn from the standalone hadron-decay script.

One page per theory (sub)section of the chapter. Every curve is computed with the functions of
standalone_hadron_decays.py (the Decayer, its M-generator and matrix elements, the decay tables),
every marker is a number of the seed-1 decays of the chapter-5 record (../event_Decay.lhe), and
the ensemble panels read the statistics written by --repeat (standalone_decay_stats.stats.json)
and by validate_decays_with_pythia.py (validate_decays_with_pythia_*.json) when they are present.

  page 1  lifetimes     the exponential decay law and the proper lifetimes drawn for the B hadrons, the
                        flight distance L = gamma beta c tau of the b quark's B against the ensemble and the
                        CMSSW event, and the c tau0 ladder of the hadrons with the 10 mm line of the generator
  page 2  vertices      the decay vertices of our event in the transverse plane with the tracks that leave
                        them, the same along the beam, and the impact parameters of the tracks from the B
                        (the birthmark the b tag of chapter 12 will look for)
  page 3  tables        what the decay tables hold: the classes of B0 and B+ decays, the decay chains of the
                        two B hadrons of our event, and the number of final particles per B hadron
  page 4  semileptonic  the lepton spectrum of the V-A matrix element against phase space and against PYTHIA,
                        the neutrino energy that leaves the B, and the pT of the lepton in the jet
  page 5  our jet       the cone around the b quark through the stages (partons, primary hadrons, everything,
                        the visible part, the charged part), the response after the neutrinos, the
                        multiplicity and the mass, and the ledger

    python3 plot_theory_Decay.py                    # -> theory_hadron_decays.pdf
    python3 plot_theory_Decay.py --page 4 -o o.png  # one page as PNG

Needs numpy and matplotlib (on this Mac: /usr/bin/python3). About 20 s.
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
sys.path.insert(0, str(HERE.parent.parent / "Had"))
import standalone_hadron_decays as core  # noqa: E402
import plot_journey_Had as jh  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

COL_B, COL_BBAR, COL_HAD = "#ff0000", "#0000ff", "#9b4dff"
COL_STR, COL_GLU, COL_MINI = "#2a78d6", "#2f8f5b", "#eb6834"
COL_A, COL_EVENT, COL_CMSSW, COL_PY, COL_PY2 = "#8e5bb5", "#c0392b", "#1f1f1f", "#7f7f7f", "#b0b0b0"
COL_NU, COL_LEP, COL_CH = "#1b9e77", "#d95f02", "#7570b3"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d6d3ca"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "legend.fontsize": 9.5,
    "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 100,
})
PAGES = {1: "lifetimes", 2: "vertices", 3: "the decay tables", 4: "semileptonic decays", 5: "our jet"}
C_MM_PER_PS = core.C_MM_PER_PS

# the CMSSW event (page 5 of ../CMSSW/genPartAnalyzer_ME_PS_MPI_Had_Decay_out.pdf and the GEN log of chapter 2,
# example/PS/CMSSW/step1_GEN_PS_cfg.log, rows 829-834 and 1434-1436 for the b, 968, 1503, 1752-1756 for the bbar;
# the vertices there are in cm)
CMSSW = dict(n_final=634, n_border=38, n_primary_stable=132, n_from_decays=477,
             b=dict(string="B-", weak="B-", pt=94.279, e=235.85, L=7.83, Lxy=3.13, channel="nu_ebar e- D*_00", cls="semileptonic (e)",
                    n_system=9, e_nu=96.2, pt_nu=37.457, pt_lepton=4.951, charm="D0", L_charm=1.0, system_pt=94.28),
             bbar=dict(string="B*+", weak="B+", pt=112.618, e=2385.5, L=117.3, Lxy=5.53, channel="D*bar0 pi+ pi0 K0 K*bar0", cls="hadronic",
                       n_system=16, e_nu=0.0, pt_nu=0.0, pt_lepton=0.0, charm="Dbar0", L_charm=140.6, system_pt=114.17))
STATS = HERE / "standalone_decay_stats.stats.json"
PY_REC = HERE / "validate_decays_with_pythia_record.json"
PY_FULL = HERE / "validate_decays_with_pythia_full.json"
PY_SINGLE = HERE / "validate_decays_with_pythia_single_511.json"
SA_SINGLE = HERE / "standalone_single_511.json"


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


class Setup:
    """The seed-1 decays of the chapter-5 record, replayed with the script's classes."""

    def __init__(self, seed=1):
        self.args = SimpleNamespace(input=HERE.parent.parent / "Had" / "event_Had.lhe", tau0max=core.TAU0_MAX, limit_tau0=True,
                                    no_mix=False, stable_b=False, radius=0.4)
        self.ev, self.dec = core.run(self.args, seed)
        self.summary = core.summarise(self.ev, self.dec)
        self.b, self.bbar = self.summary["b"]["b"], self.summary["b"]["bbar"]
        self.stats = load_json(STATS)
        self.py_rec, self.py_full = load_json(PY_REC), load_json(PY_FULL)
        self.py_single, self.sa_single = load_json(PY_SINGLE), load_json(SA_SINGLE)
        self.rng = np.random.default_rng(12345)


def mark(ax, x, label, color=COL_EVENT, y=0.93, ha="left", dx=4, ls="--"):
    ax.axvline(x, color=color, lw=1.1, ls=ls)
    ax.annotate(label, xy=(x, y), xycoords=("data", "axes fraction"), xytext=(dx if ha == "left" else -dx, 0),
                textcoords="offset points", ha=ha, va="top", fontsize=9, color=color)


def tex(name: str) -> str:
    return jh.hadron_tex(name)


# ---------------------------------------------------------------------------------------------
def page_lifetimes(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    # (a) the exponential decay law: the proper lifetimes drawn for the two B hadrons in the ensemble
    tau0 = core.tau0(511) / C_MM_PER_PS
    t = np.linspace(0, 8, 200)
    a1.plot(t, np.exp(-t / tau0) / tau0, color=COL_STR, lw=2, label=r"$\frac{1}{\tau_0} e^{-t/\tau_0}$,  $\tau_0(B^0)$ = %.3f ps" % tau0)
    if st.stats:
        for key, col, lab in (("b", COL_B, "the $b$ quark's B"), ("bbar", COL_BBAR, r"the $\bar b$ quark's B")):
            v = np.array(st.stats["b"][key]["tau_ps"])
            a1.hist(v, bins=32, range=(0, 8), density=True, histtype="step", color=col, lw=1.4, label=f"{lab}, {len(v)} decays: <t> = {v.mean():.2f} ps")
    for key, col in (("b", COL_B), ("bbar", COL_BBAR)):
        r = st.summary["b"][key]
        if r["decays"]:
            mark(a1, r["tau_ps"], f"our event: {r['tau_ps']:.2f} ps ({r['tau_over_tau0']:.2f} $\\tau_0$)", color=col, y=0.55 if key == "b" else 0.45)
    a1.set_xlabel("proper lifetime $t$ drawn for the B hadron  [ps]")
    a1.set_ylabel("probability density  [1/ps]")
    a1.set_title("(a) the decay law: $t = -\\tau_0 \\ln r$, the same at rest and at 1 TeV", fontsize=10.5)
    a1.legend(loc="upper right", fontsize=8.5)
    a1.text(0.98, 0.42, "a B hadron has no clock:\nthe probability to survive\nthe next ps never changes", transform=a1.transAxes,
            ha="right", va="top", fontsize=8.5, color=TEXT2)
    # (b) the flight distance of the b quark's B: L = gamma beta c tau
    r = st.b
    if st.stats:
        L = np.array(st.stats["b"]["b"]["L"])
        Lxy = np.array(st.stats["b"]["b"]["Lxy"])
        a2.hist(L, bins=40, range=(0, 80), density=True, color=COL_B, alpha=0.35, label=f"$L$ of the $b$ quark's B, {len(L)} decays: <L> = {L.mean():.1f} mm")
        a2.hist(Lxy, bins=40, range=(0, 80), density=True, histtype="step", color=COL_B, lw=1.4, label=f"$L_{{xy}}$: <$L_{{xy}}$> = {Lxy.mean():.1f} mm")
        gb = np.mean(st.stats["b"]["b"]["gamma"])
        x = np.linspace(0.01, 80, 300)
        lam = gb * core.tau0(511)
        a2.plot(x, np.exp(-x / lam) / lam, color=COL_STR, lw=1.5, ls="--", label=rf"$\gamma\beta\,c\tau_0$ = {gb:.1f} x {core.tau0(511):.3f} mm = {lam:.1f} mm")
    if st.py_rec:
        Lp = np.array(st.py_rec["b"]["b"]["L"])
        a2.hist(Lp, bins=40, range=(0, 80), density=True, histtype="step", color=COL_PY, lw=1.2, ls=":", label=f"PYTHIA 8 on the same hadrons: <L> = {Lp.mean():.1f} mm")
    if r["decays"]:
        mark(a2, r["L"], f"our event: {r['L']:.1f} mm", color=COL_EVENT, y=0.6)
    mark(a2, CMSSW["b"]["L"], f"CMSSW's $B^-$: {CMSSW['b']['L']:.1f} mm", color=COL_CMSSW, y=0.5, ls=":")
    a2.set_xlabel("flight distance of the B hadron  [mm]")
    a2.set_ylabel("probability density  [1/mm]")
    a2.set_title(rf"(b) the walk: $L = \gamma\beta\,c\tau$ for E = {r['e_hadron']:.0f} GeV ($\gamma$ = {r['gamma']:.0f})", fontsize=10.5)
    a2.legend(loc="upper right", fontsize=8)
    # (c) the c tau0 ladder
    ladder = [(511, "B^0"), (521, "B^+"), (531, "B_s^0"), (5122, r"\Lambda_b^0"), (421, "D^0"), (411, "D^+"), (431, "D_s^+"), (4122, r"\Lambda_c^+"),
              (15, r"\tau"), (310, "K_S^0"), (3122, r"\Lambda"), (3222, r"\Sigma^+"), (3312, r"\Xi^-"), (3334, r"\Omega^-"), (321, "K^+"), (211, r"\pi^+"),
              (130, "K_L^0"), (13, r"\mu"), (2112, "n")]
    ys = np.arange(len(ladder))
    vals = [core.tau0(pid) for pid, _ in ladder]
    cols = [COL_B if core.heavy_content(pid) == 5 else COL_CH if core.heavy_content(pid) == 4 else (COL_A if core.tau0(pid) < core.TAU0_MAX else COL_PY) for pid, _ in ladder]
    a3.barh(ys, vals, color=cols, height=0.7)
    a3.set_yticks(ys)
    a3.set_yticklabels([f"${n}$" for _, n in ladder], fontsize=9)
    a3.set_xscale("log")
    a3.set_xlim(1e-2, 1e15)
    a3.invert_yaxis()
    a3.axvline(core.TAU0_MAX, color=COL_EVENT, lw=1.3, ls="--")
    a3.text(150, 3.6, f"tau0Max = {core.TAU0_MAX:.0f} mm: the generator decays\nwhat lies to the left of the red line,\nthe detector simulation (GEANT) the rest", fontsize=8.5,
            color=COL_EVENT, va="top")
    a3.axvline(1000, color=TEXT2, lw=0.8, ls=":")
    a3.text(1500, 9.3, "1 m: PYTHIA's\nmayDecay default", fontsize=8, color=TEXT2, va="top")
    for y, v in zip(ys, vals):
        a3.text(max(v, 1.3e-2) * 1.15, y, f"{v:.3g} mm" if v < 1e4 else f"{v:.2g} mm", fontsize=7.5, va="center", color=TEXT2)
    a3.set_xlabel(r"$c\tau_0$  [mm]")
    a3.set_title("(c) who decays in the generator and who in the detector", fontsize=10.5)
    a3.grid(axis="y", visible=False)
    fig.suptitle("Lifetimes: an exponential clock, a walk of a centimetre at the LHC, and the 10 mm rule of the generator (page 1)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
def tracks_of(st: Setup, key: str):
    """(production vertex, momentum) of the charged final descendants of the b hadron, and the vertices of the chain."""
    ev, dec = st.ev, st.dec
    r = st.summary["b"][key]
    rw = r["hadron_row"]
    desc = core.descendants(ev, rw)
    fin = [d for d in desc if ev.p[d].status > 0 and core.is_charged(ev.p[d].id)]
    tracks = [(dec.vertex.get(d, np.zeros(4)), ev.p[d].p) for d in fin]
    chain = [(core.name(d["id"]), d["v_dec"], d["tau"]) for d in dec.decays if (d["row"] == rw or d["row"] in desc) and d["tau"] > 1e-4]
    return tracks, chain


def page_vertices(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    for key, col in (("b", COL_B), ("bbar", COL_BBAR)):
        r = st.summary["b"][key]
        if not r["decays"]:
            continue
        tracks, chain = tracks_of(st, key)
        for name, v, tau in chain:
            a1.plot([v[1]], [v[2]], "o", color=col, ms=7, mec="white", zorder=5)
            a1.annotate(f"{tex(name)}", (v[1], v[2]), xytext=(4, 4), textcoords="offset points", fontsize=8.5, color=col)
            a2.plot([v[3]], [math.hypot(v[1], v[2])], "o", color=col, ms=6, mec="white", zorder=5)
            a2.annotate(f"{tex(name)}", (v[3], math.hypot(v[1], v[2])), xytext=(4, 4), textcoords="offset points", fontsize=8.5, color=col)
        for v, p in tracks:
            n = p[1:] / np.linalg.norm(p[1:])
            s = np.linspace(0, 25, 2)
            a1.plot(v[1] + s * n[0], v[2] + s * n[1], color=col, lw=0.8, alpha=0.6)
            a2.plot(v[3] + s * n[2], np.hypot(v[1] + s * n[0], v[2] + s * n[1]), color=col, lw=0.8, alpha=0.6)
        # the B hadron's path from the origin
        v = r["vertex"]
        a1.plot([0, v[0]], [0, v[1]], color=col, lw=1.6, ls="--")
        a2.plot([0, v[2]], [0, math.hypot(v[0], v[1])], color=col, lw=1.6, ls="--")
    a1.plot([0], [0], "*", color=COL_EVENT, ms=14, zorder=6)
    a1.annotate("primary vertex", (0, 0), xytext=(6, -12), textcoords="offset points", fontsize=8.5, color=COL_EVENT)
    a1.set_xlabel("$x$  [mm]")
    a1.set_ylabel("$y$  [mm]")
    a1.set_aspect("equal")
    a1.set_title("(a) decay vertices of our event in (x, y) and the tracks leaving them", fontsize=10)
    a1.text(0.02, 0.98, "dashed: the B hadron's own path (invisible);\nthe tracks of the charged decay products\nstart millimetres away from the primary vertex",
            transform=a1.transAxes, va="top", fontsize=8.5, color=TEXT2)
    a2.set_xlabel("$z$  [mm]")
    a2.set_ylabel(r"$r = \sqrt{x^2 + y^2}$  [mm]")
    a2.set_title(r"(b) the same along the beam: the $\bar b$ hadron ($\eta$ = -3.4) walks far in $z$", fontsize=10)
    a2.axhspan(0, 0.0, color=TEXT2)
    # (c) the impact parameters
    if st.stats:
        d0b = 1e3 * np.array(st.stats["b"]["b"]["d0"])
        d0bb = 1e3 * np.array(st.stats["b"]["bbar"]["d0"])
        bins = np.logspace(0, 4, 41)
        a3.hist(d0b, bins=bins, color=COL_B, alpha=0.35, label=f"tracks from the $b$ quark's B ({len(d0b)} in {st.stats['repeat']} decays): median {np.median(d0b):.0f} um")
        a3.hist(d0bb, bins=bins, histtype="step", color=COL_BBAR, lw=1.3, label=rf"tracks from the $\bar b$ quark's B: median {np.median(d0bb):.0f} um")
        if st.py_rec:
            a3.hist(1e3 * np.array(st.py_rec["b"]["b"]["d0"]), bins=bins, histtype="step", color=COL_PY, lw=1.2, ls=":", label="PYTHIA 8, the $b$ quark's B")
        a3.set_xscale("log")
    for d in st.b["d0"]:
        a3.axvline(1e3 * d, color=COL_EVENT, lw=0.8, alpha=0.7)
    a3.axvline(100, color=COL_A, lw=1.2, ls="--")
    a3.text(110, 0.55, "100 um: a typical\ntrack resolution", transform=a3.get_xaxis_transform(), fontsize=8.5, color=COL_A, va="top")
    a3.set_xlabel(r"transverse impact parameter $|d_0|$ of the track  [um]")
    a3.set_ylabel("tracks")
    a3.set_title("(c) the birthmark: impact parameters of the tracks from the B", fontsize=10.5)
    a3.legend(loc="upper left", fontsize=8)
    a3.text(0.98, 0.35, f"red lines: the {st.b['n_tracks']} tracks of our event,\n{st.b['n_tracks_d0_100um']} above 100 um", transform=a3.transAxes, ha="right", fontsize=8.5, color=COL_EVENT)
    fig.suptitle("Displaced vertices: the B walks, its daughters' tracks miss the primary vertex by hundreds of microns (page 2)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
def class_of(prods, me):
    c = core.channel_class(prods, me)
    if c:
        return c.split(" (")[0] + " (" + c.split("(")[1] if "semi" in c else c
    return "partonic b -> c W, W -> q qbar'" if any(core.is_parton(p) for p in prods) else "explicit hadronic"


def page_tables(st: Setup):
    fig = plt.figure(figsize=(15, 5.4))
    a1 = fig.add_axes([0.05, 0.2, 0.26, 0.68])
    a2 = fig.add_axes([0.36, 0.05, 0.30, 0.83])
    a3 = fig.add_axes([0.71, 0.12, 0.27, 0.76])
    # (a) the classes of B decays
    order = ["partonic b -> c W, W -> q qbar'", "explicit hadronic", "semileptonic (e)", "semileptonic (mu)", "semileptonic (tau)"]
    cols = [COL_STR, COL_A, COL_LEP, COL_NU, COL_PY]
    names = [(511, "$B^0$"), (521, "$B^+$"), (531, "$B_s^0$"), (5122, r"$\Lambda_b^0$"), (421, "$D^0$"), (411, "$D^+$")]
    for i, (pid, lab) in enumerate(names):
        chans, total = core.open_channels(pid)
        frac = {k: 0.0 for k in order}
        for br, me, prods in chans:
            c = class_of(prods, me)
            frac[c if c in frac else "explicit hadronic"] += br / total
        left = 0.0
        for k, col in zip(order, cols):
            a1.barh(i, frac[k], left=left, color=col, height=0.7, label=k if i == 0 else None)
            if frac[k] > 0.06:
                a1.text(left + frac[k] / 2, i, f"{100 * frac[k]:.0f}%", ha="center", va="center", fontsize=8, color="white")
            left += frac[k]
    a1.set_yticks(range(len(names)))
    a1.set_yticklabels([lab for _, lab in names])
    a1.invert_yaxis()
    a1.set_xlim(0, 1)
    a1.set_xlabel("share of the decays (PYTHIA 8.315 tables)")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=7.5, ncol=2)
    a1.set_title("(a) what the tables say", fontsize=10.5)
    a1.grid(axis="y", visible=False)
    # (b) the decay chains of our event, as trees
    a2.axis("off")
    ev, dec = st.ev, st.dec
    y = 1.0
    for key, col in (("b", COL_B), ("bbar", COL_BBAR)):
        r = st.summary["b"][key]
        qlab = "$b$" if key == "b" else r"$\bar b$"
        a2.text(0.0, y, f"the {qlab} quark's hadron ({tex(r['string_hadron'])} row {r['string_row']}, $p_T$ = {r['pt_string']:.1f} GeV)",
                fontsize=9.5, color=col, weight="bold", va="top", transform=a2.transAxes)
        y -= 0.045
        lines = []

        def walk(row, depth):
            q = ev.p[row]
            d = next((x for x in dec.decays if x["row"] == row), None)
            if d is None:
                return
            prods = " ".join(tex(core.name(ev.p[k].id)) for k in d["rows"])
            flight = f"   $L$ = {d['L']:.2f} mm" if d["tau"] > 1e-4 else ""
            lines.append((depth, f"{tex(core.name(q.id))} $\\to$ {prods}   ({100 * d['br'] / core.open_channels(d['id_dec'])[1]:.1f}%){flight}"))
            for k in d["rows"]:
                if ev.p[k].status < 0 and (core.is_weak(ev.p[k].id) or core.heavy_content(ev.p[k].id) or abs(ev.p[k].id) in (313, 323, 10421)):
                    walk(k, depth + 1)

        walk(r["string_row"], 0)
        for depth, txt in lines[:9]:
            a2.text(0.02 + 0.05 * depth, y, txt, fontsize=8, color=TEXT, va="top", transform=a2.transAxes)
            y -= 0.04
        if len(lines) > 9:
            a2.text(0.07, y, f"... and {len(lines) - 9} more decays", fontsize=8, color=TEXT2, va="top", transform=a2.transAxes)
            y -= 0.04
        a2.text(0.02, y, f"{r['n_final']} final particles ({r['n_charged_final']} charged, {r['n_photons_final']} photons, {r['n_neutrinos']} neutrinos)",
                fontsize=8, color=TEXT2, va="top", transform=a2.transAxes)
        y -= 0.07
    a2.text(0.0, y, f"CMSSW's event: $B^-\\to \\bar\\nu_e e^- D_0^{{*0}}$ ({CMSSW['b']['n_system']} final, a {CMSSW['b']['pt_nu']:.1f} GeV neutrino);\n"
                    f"$B^{{*+}}\\to B^+\\gamma$, $B^+\\to \\bar D^{{*0}}\\pi^+\\pi^0 K^0\\bar K^{{*0}}$ ({CMSSW['bbar']['n_system']} final)",
            fontsize=8, color=COL_CMSSW, va="top", transform=a2.transAxes)
    a2.set_title("(b) the decay chains of our event (branching ratio of each step)", fontsize=10.5)
    # (c) the final multiplicity per B hadron
    if st.stats:
        n = np.array(st.stats["b"]["b"]["n_final"])
        a3.hist(n, bins=np.arange(-0.5, 30.5, 1), density=True, color=COL_B, alpha=0.35, label=f"this script, the $b$ quark's B: <n> = {n.mean():.1f}")
        nc = np.array(st.stats["b"]["b"]["n_charged_final"])
        a3.hist(nc, bins=np.arange(-0.5, 30.5, 1), density=True, histtype="step", color=COL_B, lw=1.0, ls="--", label=f"charged only: <n> = {nc.mean():.1f}")
    if st.py_rec:
        n = np.array(st.py_rec["b"]["b"]["n_final"])
        a3.hist(n, bins=np.arange(-0.5, 30.5, 1), density=True, histtype="step", color=COL_PY, lw=1.2, ls=":", label=f"PYTHIA 8 on the same hadrons: <n> = {n.mean():.1f}")
    mark(a3, st.b["n_final"], f"our event: {st.b['n_final']}", color=COL_EVENT, y=0.6)
    mark(a3, CMSSW["b"]["n_system"], f"CMSSW: {CMSSW['b']['n_system']}", color=COL_CMSSW, y=0.5, ls=":")
    a3.set_xlabel("final particles from one B hadron")
    a3.set_ylabel("probability")
    a3.set_title("(c) how many grandchildren a B hadron leaves", fontsize=10.5)
    a3.legend(loc="upper right", fontsize=8)
    fig.suptitle("The decay tables: 870 channels for a $B^0$, half of them written at quark level (page 3)", fontsize=12)
    return fig


# ---------------------------------------------------------------------------------------------
def lepton_spectrum(st: Setup, me: int, n=20000):
    """E_l in the rest frame of B0 -> D*- mu+ nu_mu generated with the script's M-generator and matrix element."""
    M = core.m0(511)
    masses = [0.0, 0.10566, core.m0(413)]
    ids = [14, -13, -413]
    P = np.array([M, 0, 0, 0])
    out = []
    for _ in range(n):
        p, info = st.dec.mgenerator(M, masses, me, P, ids)
        if p is not None:
            out.append(p[1][0])
    return np.array(out)


def page_semileptonic(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    # (a) the V-A lepton spectrum against phase space
    e_va = lepton_spectrum(st, 22, 20000)
    e_ps = lepton_spectrum(st, 0, 20000)
    bins = np.linspace(0, 2.6, 27)
    a1.hist(e_ps, bins=bins, density=True, color=COL_PY2, alpha=0.6, label=f"phase space (meMode 0): <E> = {e_ps.mean():.2f} GeV")
    a1.hist(e_va, bins=bins, density=True, histtype="step", color=COL_LEP, lw=2, label=f"V-A (meMode 22): <E> = {e_va.mean():.2f} GeV")
    if st.py_single:
        v = np.array(st.py_single["e_lepton_direct"], dtype=float)
        v = v[~np.isnan(v)]
        a1.hist(v, bins=bins, density=True, histtype="step", color=COL_PY, lw=1.2, ls=":", label=f"PYTHIA 8, all semileptonic $B^0$ channels: <E> = {v.mean():.2f}")
    if st.sa_single:
        v = np.array(st.sa_single["e_lepton_direct"], dtype=float)
        v = v[~np.isnan(v)]
        a1.hist(v, bins=bins, density=True, histtype="step", color=COL_EVENT, lw=1.2, ls="--", label=f"this script, all semileptonic channels: <E> = {v.mean():.2f}")
    a1.set_xlabel(r"lepton energy in the B rest frame  [GeV]   ($B^0\to D^{*-}\mu^+\nu_\mu$)")
    a1.set_ylabel("probability density  [1/GeV]")
    a1.set_title(r"(a) $|M|^2 \propto (p_B\cdot p_\nu)(p_{D^*}\cdot p_\mu)$ makes the lepton hard", fontsize=10.5)
    a1.legend(loc="upper left", fontsize=8)
    # (b) the neutrino energy fraction of the B
    if st.stats:
        f = np.array(st.stats["b"]["b"]["e_nu_frac"])
        fb = np.array(st.stats["b"]["bbar"]["e_nu_frac"])
        a2.hist(f, bins=np.linspace(0, 0.8, 33), color=COL_NU, alpha=0.4, label=f"the $b$ quark's B: <f> = {f.mean():.3f}, {100 * np.mean(f > 0):.0f}% with a neutrino")
        a2.hist(fb, bins=np.linspace(0, 0.8, 33), histtype="step", color=COL_BBAR, lw=1.3, label=rf"the $\bar b$ quark's B: <f> = {fb.mean():.3f}")
        if st.py_rec:
            fp = np.array(st.py_rec["b"]["b"]["e_nu_frac"])
            a2.hist(fp, bins=np.linspace(0, 0.8, 33), histtype="step", color=COL_PY, lw=1.2, ls=":", label=f"PYTHIA 8: <f> = {fp.mean():.3f}")
        a2.set_yscale("log")
    mark(a2, CMSSW["b"]["e_nu"] / CMSSW["b"]["e"], f"CMSSW's $B^-$: {CMSSW['b']['e_nu'] / CMSSW['b']['e']:.2f}", color=COL_CMSSW, y=0.6, ls=":")
    if st.b["decays"]:
        mark(a2, st.b["e_nu_frac"], f"our event: {st.b['e_nu_frac']:.2f}", color=COL_EVENT, y=0.5)
    a2.set_xlabel(r"$f_\nu = E_\nu / E_B$: energy that leaves the family for good")
    a2.set_ylabel("B decays")
    a2.set_title("(b) the emigrants: the neutrino's share of the B energy", fontsize=10.5)
    a2.legend(loc="upper right", fontsize=8)
    # (c) the lepton pT in the jet
    if st.stats:
        d = st.stats["b"]["b"]
        pt_b = np.array([p for p, f in zip(d["pt_lepton"], d["lepton_from_b"]) if p > 0 and f])
        pt_c = np.array([p for p, f in zip(d["pt_lepton"], d["lepton_from_b"]) if p > 0 and not f])
        bins = np.linspace(0, 60, 31)
        a3.hist(pt_b, bins=bins, color=COL_LEP, alpha=0.5, label=f"lepton from the B itself ({100 * len(pt_b) / st.stats['repeat']:.0f}% of the decays): <$p_T$> = {pt_b.mean():.1f} GeV")
        a3.hist(pt_c, bins=bins, histtype="step", color=COL_CH, lw=1.5, label=f"from the charm cascade ({100 * len(pt_c) / st.stats['repeat']:.0f}%): <$p_T$> = {pt_c.mean():.1f} GeV")
        if st.py_rec:
            dp = st.py_rec["b"]["b"]
            ptp = np.array([p for p in dp["pt_lepton"] if p > 0])
            a3.hist(ptp, bins=bins, histtype="step", color=COL_PY, lw=1.2, ls=":", label="PYTHIA 8, any lepton")
    mark(a3, CMSSW["b"]["pt_lepton"], f"CMSSW's $e^-$: {CMSSW['b']['pt_lepton']:.1f} GeV", color=COL_CMSSW, y=0.6, ls=":")
    a3.set_xlabel(r"$p_T$ of the hardest $e$ or $\mu$ among the B's descendants  [GeV]")
    a3.set_ylabel("B decays")
    a3.set_title(f"(c) the soft lepton in a {st.b['pt_quark']:.0f} GeV $b$ jet", fontsize=10.5)
    a3.legend(loc="upper right", fontsize=8)
    fig.suptitle("Semileptonic decays: a hard lepton, a harder neutrino, and the energy the jet will never recover (page 4)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
def page_jet(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    r = st.b
    c4, c8 = r["cones"][0.4], r["cones"][0.8]
    stages = ["partons", "primaries", "final", "visible", "charged"]
    labels = ["partons\n(stage 3)", "primary\nhadrons (4)", "all final\nparticles (5)", "visible\n(no $\\nu$)", "charged\nonly"]
    x = np.arange(len(stages))
    for R, c, col, off in ((0.4, c4, COL_B, -0.15), (0.8, c8, COL_BBAR, 0.15)):
        vals = [c[s]["pt"] for s in stages]
        a1.plot(x + off, vals, "o-", color=col, ms=7, mec="white", lw=1.4, label=f"our event, R = {R}")
        if st.stats:
            d = st.stats["b"]["b"]["cones"][str(R)]
            means = [c["partons"]["pt"]] + [np.mean(d[s]["pt"]) for s in stages[1:]]
            stds = [0] + [np.std(d[s]["pt"]) for s in stages[1:]]
            a1.errorbar(x + off + 0.05, means, yerr=stds, fmt="s", color=col, ms=4, lw=0.8, alpha=0.6, label=f"{st.stats['repeat']} decays, R = {R} (mean, std)")
    a1.set_xticks(x)
    a1.set_xticklabels(labels, fontsize=8.5)
    a1.set_ylabel("$p_T$ in the cone around the $b$ quark  [GeV]")
    a1.set_title("(a) the cone through the stages: decays keep the sum, neutrinos take theirs", fontsize=10)
    a1.legend(loc="lower left", fontsize=8)
    a1.text(0.98, 0.04, f"CMSSW's $b$ system: {CMSSW['b']['system_pt']:.1f} GeV, of which\nthe neutrino carries {CMSSW['b']['pt_nu']:.1f} GeV",
            transform=a1.transAxes, ha="right", va="bottom", fontsize=8.5, color=COL_CMSSW)
    # (b) the response after the neutrinos
    if st.stats:
        d = st.stats["b"]["b"]["cones"]["0.4"]
        ratio = np.array(d["visible"]["pt"]) / np.array(d["primaries"]["pt"])
        a2.hist(ratio, bins=np.linspace(0.3, 1.2, 46), density=True, color=COL_B, alpha=0.4, label=f"this script: <R> = {ratio.mean():.3f}, {100 * np.mean(ratio < 0.95):.0f}% below 0.95")
        if st.py_rec:
            dp = st.py_rec["b"]["b"]["cones"]["0.4"]
            rp = np.array(dp["visible"]["pt"]) / np.array(dp["primaries"]["pt"])
            a2.hist(rp, bins=np.linspace(0.3, 1.2, 46), density=True, histtype="step", color=COL_PY, lw=1.2, ls=":", label=f"PYTHIA 8 on the same hadrons: <R> = {rp.mean():.3f}")
        if st.py_full:
            dp = st.py_full["b"]["b"]["cones"]["0.4"]
            rp = np.array(dp["visible"]["pt"]) / np.array(dp["primaries"]["pt"])
            a2.hist(rp, bins=np.linspace(0.3, 1.2, 46), density=True, histtype="step", color=COL_PY2, lw=1.2, label=f"PYTHIA 8, full chain, {len(rp)} events: <R> = {rp.mean():.3f}")
        a2.set_yscale("log")
    mark(a2, c4["visible"]["pt"] / c4["primaries"]["pt"], f"our event: {c4['visible']['pt'] / c4['primaries']['pt']:.3f}", color=COL_EVENT, y=0.6)
    mark(a2, (CMSSW["b"]["system_pt"] - CMSSW["b"]["pt_nu"]) / CMSSW["b"]["system_pt"], "CMSSW's $B^-$ system: 0.60", color=COL_CMSSW, y=0.5, ls=":")
    a2.set_xlabel(r"visible $p_T$ / primary-hadron $p_T$ in the cone R = 0.4")
    a2.set_ylabel("probability density")
    a2.set_title("(b) the response of the jet after the decays", fontsize=10.5)
    a2.legend(loc="upper left", fontsize=8)
    # (c) multiplicity and mass through the stages
    vals_n = [c4[s]["n"] for s in stages]
    vals_m = [c4[s]["m"] for s in stages]
    a3.plot(x, vals_n, "o-", color=COL_STR, ms=7, mec="white", label="particles in the cone, R = 0.4 (our event)")
    a3b = a3.twinx()
    a3b.plot(x, vals_m, "s--", color=COL_MINI, ms=6, mec="white", label="mass of the cone [GeV], R = 0.4")
    if st.stats:
        d = st.stats["b"]["b"]["cones"]["0.4"]
        a3.errorbar(x[1:] + 0.08, [np.mean(d[s]["n"]) for s in stages[1:]], yerr=[np.std(d[s]["n"]) for s in stages[1:]], fmt="o", color=COL_STR, ms=4, alpha=0.5)
        a3b.errorbar(x[1:] + 0.08, [np.mean(d[s]["m"]) for s in stages[1:]], yerr=[np.std(d[s]["m"]) for s in stages[1:]], fmt="s", color=COL_MINI, ms=4, alpha=0.5)
    a3.set_xticks(x)
    a3.set_xticklabels(labels, fontsize=8.5)
    a3.set_ylabel("particles", color=COL_STR)
    a3b.set_ylabel("mass  [GeV]", color=COL_MINI)
    a3b.grid(False)
    a3.set_title("(c) four times more particles, the same mass", fontsize=10.5)
    h1, l1 = a3.get_legend_handles_labels()
    h2, l2 = a3b.get_legend_handles_labels()
    a3.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    fig.suptitle(f"Our jet at this stage: {c4['partons']['pt']:.2f} -> {c4['primaries']['pt']:.2f} -> {c4['final']['pt']:.2f} -> {c4['visible']['pt']:.2f} GeV within R = 0.4 (page 5)", fontsize=12)
    fig.tight_layout()
    return fig


PAGE_FUNCS = {1: page_lifetimes, 2: page_vertices, 3: page_tables, 4: page_semileptonic, 5: page_jet}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, default=HERE / "theory_hadron_decays.pdf")
    ap.add_argument("--page", type=int, default=None, help="one page only (PNG or PDF)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    st = Setup(args.seed)
    print(f"seed {args.seed}: {st.summary['n_final']} final particles; the b quark's {st.b['hadron']} flew {st.b['L']:.2f} mm and decayed to {st.b['channel']}")
    if args.page:
        fig = PAGE_FUNCS[args.page](st)
        fig.savefig(args.output, dpi=130)
        print(f"wrote {args.output}")
        return
    with PdfPages(str(args.output)) as pdf:
        for n in sorted(PAGE_FUNCS):
            fig = PAGE_FUNCS[n](st)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  page {n}: {PAGES[n]}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
