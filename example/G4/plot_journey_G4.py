#!/usr/bin/env python3
"""Draw the hand-over to Geant4 in the (eta, phi) plane, in the layout of the CMSSW SimAnalyzer page
"1. primaries" of the JitJetSW companion (page 6 of simAnalyzer_signal_09_G4.pdf): the stable particles of the
signal that become Geant4 primaries, by kind, the b and bbar systems (the stable descendants of the weakly
decaying b hadron of each side) in red and blue, their vector sum as a magenta star, and the stable particles
that the gate does not hand over as grey crosses (drawn on the border when they lie beyond |eta| = 6).

The input is the record of chapter 8 (../PU/event_PU.lhe); the decision of the gate is made again by
Standalone/standalone_G4.py (a second), so that the picture always shows what the script decides.

  python3 plot_journey_G4.py                       # -> journey_G4.pdf
  python3 plot_journey_G4.py --png

Dependency: matplotlib; on this Mac /usr/bin/python3.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "PS"))
sys.path.insert(0, str(HERE / "Standalone"))
import plot_journey_PS as ps  # noqa: E402
import standalone_G4 as core  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEXT, TEXT2, ETA_MAX, area = ps.TEXT, ps.TEXT2, ps.ETA_MAX, ps.area
COL_B, COL_BBAR, COL_SUM, COL_LOST = "#ff0000", "#0000ff", "#ff00ff", "#8c8c8c"
KIND_STYLE = {
    "charged hadron": dict(color="#000000", marker="o", hollow=False, label="charged hadron"),
    "neutral hadron": dict(color="#6f6f6f", marker="o", hollow=True, label="neutral hadron"),
    "photon": dict(color="#009900", marker="o", hollow=False, label="photon"),
    "electron": dict(color="#ff6600", marker="s", hollow=False, label=r"e$^\pm$"),
    "muon": dict(color="#9900cc", marker="D", hollow=False, label=r"$\mu^\pm$"),
}
GATE = ["the gate (Generator.cc):", r"  $|p|$ > 40 MeV", "  the line leaves r < 2.9 cm", "    before |z| = 354.8 cm", r"  beyond 52.8 m: n, $\bar n$, $\gamma$ only",
        r"  $\nu$ killed (StackingAction)", r"  p, $|p_z|$ > 3 TeV, $|\eta|$ > 8.2: PPS"]
TEX_NAMES = {"Bbar0": r"$\bar B^0$", "B0": r"$B^0$", "B*0": r"$B^{*0}$", "B*bar0": r"$\bar B^{*0}$", "B-": r"$B^-$", "B+": r"$B^+$",
             "B*+": r"$B^{*+}$", "B*-": r"$B^{*-}$", "Lambda_b0": r"$\Lambda_b^0$", "Lambda_bbar0": r"$\bar\Lambda_b^0$", "B_s0": r"$B_s^0$", "B_sbar0": r"$\bar B_s^0$"}


def tex(name: str) -> str:
    return TEX_NAMES.get(name, name)


def load(record: Path):
    args = argparse.Namespace(input=record, library=None, eta_cut=core.ETA_CUT, eta_hector=core.ETA_HECTOR, r_dec=core.R_DEC, p_min=core.P_MIN)
    st = core.Stage(args)
    R = st.compute()
    return st, R


def systems(st):
    """Index arrays of the b and bbar systems in the signal block of the crowd."""
    c = st.crowd
    row_index = {int(r): j for j, r in enumerate(c.row) if c.block[j] == 0}
    out = {}
    for key, b in core.b_systems(st.ev).items():
        out[key] = np.array([row_index[r] for r in b["finals"] if r in row_index], dtype=int)
    return out


def header(fig, st, R, sysidx, title, source):
    c, g = st.crowd, st.g
    sig = c.block == 0
    L = R["L"]["signal"]
    d = L["drops"]
    n_border = int((sig & (np.abs(c.eta) > ETA_MAX)).sum())
    x, y, dy = 0.06, 0.955, 0.029
    fig.text(x, y, title, fontsize=14, weight="bold", color=TEXT, va="center")
    fig.text(0.985, y, source, fontsize=9.5, color=TEXT, ha="right", va="center", weight="bold")
    y -= 0.036
    fig.text(x, y, f"{L['primary']['n']} primaries of {L['stable']['n']} stable particles; not handed over: "
                   f"{d['eta']['n']} $|\\eta|$ > 5.5, {d['p']['n']} p < 40 MeV, {d['nu']['n']} $\\nu$, {d['other']['n']} other; {n_border} beyond $|\\eta|$ = {ETA_MAX:g} (x)",
             fontsize=8.6, color=TEXT2, va="center", weight="bold")
    y -= dy
    pv = st.pv
    fig.text(x, y, f"primary vertex (smeared): x = {pv[0]:.4f}, y = {pv[1]:.4f}, z = {pv[2]:.3f} cm; {R['decays']['n_weak']} weak decays, "
                   f"all inside r = {core.R_DEC} cm (farthest r = {R['decays']['far']['r_dec']:.2f} cm)", fontsize=8.6, color=TEXT2, va="center", weight="bold")
    vec = np.zeros(4)
    scal = 0.0
    phis = {}
    for key, col, sym in (("b", COL_B, r"$b$"), ("bbar", COL_BBAR, r"$\bar b$")):
        y -= dy
        b = R["b"].get(key)
        if not b:
            continue
        idx = sysidx[key]
        m = np.zeros(len(c), bool)
        m[idx] = True
        mp = m & g["primary"]
        v = c.p4[mp].sum(axis=0)
        vec += v
        scal += math.hypot(v[1], v[2])
        phis[key] = math.atan2(v[2], v[1])
        chain = tex(b["first"]) + (r" $\to$ " + tex(b["hadron"]) if b["first"] != b["hadron"] else "")
        fig.text(x, y, f"{sym} system ({sym} hadron {chain} decay products, {b['n_primary']} of {b['n']} handed to Geant4): "
                       f"$p_T$ = {b['pt_after']:.2f} GeV,  $\\eta$ = {b['eta']:.2f},  $\\phi$ = {b['phi']:.2f}", fontsize=9.2, color=col, va="center", weight="bold")
    y -= dy
    if len(phis) == 2:
        dphi = abs(ps.dphi(phis["b"], phis["bbar"]))
        fig.text(x, y, rf"net $b$+$\bar b$: $|\vec p_T(b) + \vec p_T(\bar b)|$ = {math.hypot(vec[1], vec[2]):.2f} GeV,     $\Sigma|p_T|$ = {scal:.2f} GeV,     $\Delta\phi$ = {dphi:.2f}",
                 fontsize=9.2, color=COL_SUM, va="center", weight="bold")
    return y, vec


def draw_particles(ax, st, sysidx, vec, label_min):
    c, g = st.crowd, st.g
    sig = np.where(c.block == 0)[0]
    in_sys = {}
    for key, idx in sysidx.items():
        for j in idx:
            in_sys[int(j)] = key
    counts = {k: 0 for k in KIND_STYLE}
    counts.update(b=0, bbar=0, lost=0)
    order = sorted(sig, key=lambda j: (int(j) in in_sys, c.pt[j]))
    for j in order:
        e, f = c.eta[j], c.phi[j]
        border = abs(e) > ETA_MAX
        if border:
            e = math.copysign(ETA_MAX - 0.05, e)
        if not g["primary"][j]:
            counts["lost"] += 1
            ax.scatter(e, f, s=area(c.pt[j]) * 0.9, marker="x" if border else "+", color=COL_LOST, linewidth=0.9, zorder=3)
            continue
        key = in_sys.get(int(j))
        if key:
            counts[key] += 1
            ax.scatter(e, f, s=area(c.pt[j]), marker="o", color=COL_B if key == "b" else COL_BBAR, linewidth=0, zorder=6)
            if c.pt[j] >= label_min:
                ax.annotate(f"{c.pt[j]:.1f}", (e, f), xytext=(3, 2), textcoords="offset points", fontsize=6.5, color=COL_B if key == "b" else COL_BBAR, zorder=7)
            continue
        st_ = KIND_STYLE.get(c.kind[j], KIND_STYLE["charged hadron"])
        counts[c.kind[j]] = counts.get(c.kind[j], 0) + 1
        if border:
            ax.scatter(e, f, s=area(c.pt[j]), marker="x", color=st_["color"], linewidth=1.0, zorder=4)
        elif st_["hollow"]:
            ax.scatter(e, f, s=area(c.pt[j]), marker=st_["marker"], facecolors="none", edgecolors=st_["color"], linewidth=0.9, zorder=4)
        else:
            ax.scatter(e, f, s=area(c.pt[j]), marker=st_["marker"], color=st_["color"], linewidth=0, zorder=4)
        if c.pt[j] >= label_min:
            ax.annotate(f"{c.pt[j]:.1f}", (e, f), xytext=(3, 2), textcoords="offset points", fontsize=6.5, color=TEXT, zorder=7)
    if np.any(vec):
        pt = math.hypot(vec[1], vec[2])
        e, f = core.gj.eta(vec), math.atan2(vec[2], vec[1])
        e = max(-ETA_MAX + 0.05, min(ETA_MAX - 0.05, e))
        ax.scatter(e, f, s=area(pt) * 1.8, marker="*", color=COL_SUM, zorder=8, linewidth=0)
        ax.annotate(rf"$b$+$\bar b$ {pt:.1f}", (e, f), xytext=(4, -9), textcoords="offset points", fontsize=8, color=COL_SUM, weight="bold", zorder=8)
    return counts


def draw_legend(ax, counts):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y = 0.985
    entries = [("o", COL_B, False, rf"$b$ system ({counts['b']})"), ("o", COL_BBAR, False, rf"$\bar b$ system ({counts['bbar']})"),
               ("*", COL_SUM, False, r"$b$ + $\bar b$ (vector sum)")]
    for k, s in KIND_STYLE.items():
        if counts.get(k, 0):
            entries.append((s["marker"], s["color"], s["hollow"], f"{s['label']} ({counts[k]})"))
    entries.append(("+", COL_LOST, False, f"stable, not handed over ({counts['lost']})"))
    for marker, col, hollow, label in entries:
        y -= 0.052
        if hollow:
            ax.scatter(0.08, y, s=area(15.0), marker=marker, facecolors="none", edgecolors=col, linewidth=1.0)
        else:
            ax.scatter(0.08, y, s=area(15.0) * (1.6 if marker == "*" else 1.0), marker=marker, color=col, linewidth=1.0 if marker in "+x" else 0)
        ax.text(0.2, y, label, fontsize=8.6, color=TEXT, va="center")
    y -= 0.07
    ax.text(0.0, y, "marker size:", fontsize=8.6, color=TEXT, va="center", weight="bold")
    for pt in (1, 10, 100):
        y -= 0.047
        ax.scatter(0.08, y, s=area(pt), marker="o", color="#666666")
        ax.text(0.2, y, f"$p_T$ = {pt} GeV", fontsize=8.6, color=TEXT, va="center")
    y -= 0.065
    for i, line in enumerate(GATE):
        ax.text(0.0, y, line, fontsize=7.6, color="#cc0000" if i == 0 else TEXT2, va="center", weight="bold" if i == 0 else "normal")
        y -= 0.036


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=HERE.parent / "PU" / "event_PU.lhe")
    ap.add_argument("-o", "--output", type=Path, default=HERE / "journey_G4.pdf")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--label-min", type=float, default=5.0)
    a = ap.parse_args()
    st, R = load(a.input)
    sysidx = systems(st)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "mathtext.default": "regular"})
    fig = plt.figure(figsize=(10.0, 7.4))
    y_header, vec = header(fig, st, R, sysidx, "the gate: stable particles handed to Geant4", "standalone, seed 1 (event_PU.lhe)")
    top = y_header - 0.045
    ax = fig.add_axes([0.06, 0.08, 0.70, top - 0.08])
    ps.draw_axes(ax)
    counts = draw_particles(ax, st, sysidx, vec, a.label_min)
    side = fig.add_axes([0.79, 0.08, 0.20, top - 0.06])
    draw_legend(side, counts)
    fig.savefig(a.output)
    if a.png:
        fig.savefig(a.output.with_suffix(".png"), dpi=200)
    print(f"wrote {a.output}; drawn: " + ", ".join(f"{k} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
