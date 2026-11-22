#!/usr/bin/env python3
"""The displays of the simulated hits, station by station, in the layout of the CMSSW SimHitAnalyzer pages.

  python3 plot_simhits_SIM.py                          # the pages the chapter uses
  python3 plot_simhits_SIM.py --sample mixed --station ALL --view etaphi
  python3 plot_simhits_SIM.py --all                    # every sample x station x view (48 pages)

sample : signal | intime | oot | mixed (the three added by add_hits_SIM.mix)
station: TRK | ECAL | HCAL | MUON | ALL
view   : rz | xy (barrel) | etaphi (direction of the hit seen from the nominal collision point)

A tracker or muon hit is a dot; a calorimeter cell is a marker whose area grows with the logarithm of its
energy. Hits of the signal are coloured by the kind of the primary they descend from, hits of the crowd by
their bunch crossing (early, in time, late); a cell takes the colour of whoever deposited most of its energy.
The crowd is drawn first and the signal on top. The pages are written as PNG: with a million markers a
vector page would be tens of megabytes (matplotlib rasterises them in a few seconds).
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import add_hits_SIM as add
import standalone_SIM as sim

HERE = Path(__file__).resolve().parent
CAT = ["charged hadron, signal", "neutral hadron, signal", "photon, signal", r"e$^\pm$, signal", r"$\mu^\pm$, signal",
       "pileup, early (bx < 0)", "pileup, in-time (bx 0)", "pileup, late (bx > 0)"]
COL = ["black", "#990099", "#009900", "#ff6600", "#0033cc", "#999900", "#6699ff", "#ff6666"]
MRK = ["o", "s", "o", "s", "D", "o", "o", "o"]
ORDER = [5, 7, 6, 0, 1, 2, 3, 4]                       # drawing order: the crowd underneath, the signal on top
STATIONS = {"TRK": (sim.TRACKER, ()), "ECAL": ((), sim.ECAL), "HCAL": ((), sim.HCAL), "MUON": (sim.MUON, ()),
            "ALL": (sim.TRACKER + sim.MUON, sim.ECAL + sim.HCAL)}
BARREL_DETS = (sim.PXB, sim.TIB, sim.TOB, sim.DT, sim.EB, sim.HB, sim.HO)
TITLE = {"TRK": "tracker", "ECAL": "ECAL", "HCAL": "HCAL", "MUON": "muon system", "ALL": "ALL (all subdetectors)"}


def collect(sample: str, S=None):
    """Hits (x, y, z, det, cat) and cells (code, e, cat) of a sample."""
    S = S or add.load(HERE.parent)
    sg = S["signal"]
    kind = sg["prim_kind"]
    H = dict(x=[], y=[], z=[], det=[], cat=[])
    C = dict(code=[], e=[], cat=[])

    def hits(d, cat):
        for part in ("trk", "mu"):
            H["x"].append(d[f"{part}_x"]); H["y"].append(d[f"{part}_y"]); H["z"].append(d[f"{part}_z"]); H["det"].append(d[f"{part}_det"])
            H["cat"].append(cat(d, part))

    if sample in ("signal", "mixed"):
        hits(sg, lambda d, part: kind[d[f"{part}_prim"]])
    if sample in ("intime", "mixed"):
        hits(S["intime"], lambda d, part: np.full(len(d[f"{part}_det"]), 6, np.int8))
    if sample in ("oot", "mixed"):
        hits(S["oot"], lambda d, part: np.where(d[f"{part}_bx"] < 0, 5, 7).astype(np.int8))
    for k in H:
        H[k] = np.concatenate(H[k])
    # cells: the dominant origin; for the signal the kind of the dominant primary
    code, e, prim = sg["cal_code"], sg["cal_e"].astype(float), sg["cal_prim"]
    cells, inv = np.unique(code, return_inverse=True)
    Ek = np.zeros((len(cells), 5))
    np.add.at(Ek, (inv, kind[prim]), e)
    sig_kind = dict(zip(cells.tolist(), np.argmax(Ek, axis=1).tolist()))
    sig_e = dict(zip(cells.tolist(), Ek.sum(axis=1).tolist()))
    if sample == "signal":
        C = dict(code=cells, e=Ek.sum(axis=1), cat=np.argmax(Ek, axis=1))
    else:
        M = add.mix(S)
        E = M["E"].copy()
        if sample == "intime":
            E[:, [0, 1, 3]] = 0
        elif sample == "oot":
            E[:, [0, 2]] = 0
        keep = E.sum(axis=1) > 0
        dom = np.argmax(E, axis=1)
        cat = np.array([0, 5, 6, 7])[dom]
        sel = np.flatnonzero(keep & (dom == 0))
        cat[sel] = [sig_kind[c] for c in M["cells"][sel].tolist()]
        C = dict(code=M["cells"][keep], e=E.sum(axis=1)[keep], cat=cat[keep])
    return H, C, S


def draw_detector(ax, view, station):
    g = dict(color="0.72", lw=0.6, zorder=0)
    if view == "rz":
        if station in ("TRK", "ALL"):
            for b in sim.BARREL:
                if b[0] > 0:
                    for s in (1, -1):
                        ax.plot([s * b[3], s * b[4]], [b[2], b[2]], **g)
            for d in sim.DISKS:
                if d[0] > 0:
                    ax.plot([d[2], d[2]], [d[3], d[4]], **g)
        if station in ("ECAL", "ALL"):
            ax.plot([-sim.Z_EB, sim.Z_EB], [sim.R_EB, sim.R_EB], **g); ax.plot([-sim.Z_EB, sim.Z_EB], [sim.R_EB_BACK, sim.R_EB_BACK], **g)
            for s in (1, -1):
                ax.plot([s * sim.Z_EE, s * sim.Z_EE], [sim.R_EE_MIN, sim.R_EE_MAX], **g)
                ax.plot([s * sim.Z_EE_BACK, s * sim.Z_EE_BACK], [sim.R_EE_MIN * 342 / 320, sim.R_EE_MAX * 342 / 320], **g)
        if station in ("HCAL", "ALL"):
            zb = sim.R_HB * math.sinh(sim.ETA_HB)
            for r in sim.HB_R:
                ax.plot([-r * math.sinh(sim.ETA_HB), r * math.sinh(sim.ETA_HB)], [r, r], **g)
            for s in (1, -1):
                for z in sim.HE_Z:
                    ax.plot([s * z, s * z], [z / math.sinh(3.0), z / math.sinh(sim.ETA_HB)], **g)
                ax.plot([s * sim.Z_HF, s * sim.Z_HF], [sim.Z_HF / math.sinh(5.191), sim.Z_HF / math.sinh(2.853)], **g)
                ax.plot([s * 1280.0, s * 1280.0], [12.5, 130.0], **g)
            ax.plot([-407 * math.sinh(1.262), 407 * math.sinh(1.262)], [407, 407], **g)
        if station in ("MUON", "ALL"):
            for r, *_ in sim.MU_BARREL:
                ax.add_patch(plt.Rectangle((-661, r - 12), 1322, 24, fill=False, ec="0.72", lw=0.6, zorder=0))
            for z, rin, rout, *_ in sim.MU_ENDCAP:
                for s in (1, -1):
                    ax.plot([s * z, s * z], [rin, rout], **g)
    elif view == "xy":
        t = np.linspace(0, 2 * math.pi, 400)
        radii = []
        if station in ("TRK", "ALL"):
            radii += [b[2] for b in sim.BARREL if b[0] in (sim.PXB, sim.TIB, sim.TOB)]
        if station in ("ECAL", "ALL"):
            radii += [sim.R_EB, sim.R_EB_BACK]
        if station in ("HCAL", "ALL"):
            radii += list(sim.HB_R) + [407.0]
        if station in ("MUON", "ALL"):
            radii += [m[0] for m in sim.MU_BARREL]
        for r in radii:
            ax.plot(r * np.cos(t), r * np.sin(t), **g)


def size_of(e):
    return 1.5 + 9.0 * np.clip(np.log10(np.maximum(e, 1e-4) / 0.01) + 1.0, 0.0, 5.0) ** 1.6 / 3.0


def page(sample, station, view, H, C, S, out: Path, st=None):
    t0 = time.time()
    hd, cd = STATIONS[station]
    mh = np.isin(H["det"], hd)
    cdet = C["code"] // sim.CODE
    mc = np.isin(cdet, cd)
    _, cx, cy, cz = sim.cell_position(C["code"])
    if view == "xy":
        mh &= np.isin(H["det"], BARREL_DETS)
        mc &= np.isin(cdet, BARREL_DETS)
    fig = plt.figure(figsize=(13.96, 8.7))
    ax = fig.add_axes([0.07, 0.09, 0.69, 0.74])
    draw_detector(ax, view, station)
    def xy(x, y, z):
        r = np.hypot(x, y)
        if view == "rz":
            return z, r
        if view == "xy":
            return x, y
        return np.arcsinh(z / np.maximum(r, 1e-9)), np.arctan2(y, x)
    hx, hy = xy(H["x"][mh], H["y"][mh], H["z"][mh])
    hc = H["cat"][mh]
    ux, uy = xy(cx[mc], cy[mc], cz[mc])
    uc, ue = C["cat"][mc], C["e"][mc]
    counts = np.zeros(8, int)
    dense = len(hx) > 100000
    for k in ORDER:
        m = hc == k
        counts[k] += int(m.sum())
        if m.any():
            ax.plot(hx[m], hy[m], ls="none", marker="," if (dense and k >= 5) else ".", ms=1.0 if k >= 5 else 2.2, color=COL[k], rasterized=True, zorder=2 + ORDER.index(k))
    for k in ORDER:
        m = uc == k
        counts[k] += int(m.sum())
        if m.any():
            ax.scatter(ux[m], uy[m], s=size_of(ue[m]), marker=MRK[k], color=COL[k], lw=0, rasterized=True, zorder=12 + ORDER.index(k), alpha=0.95)
    # the followed primaries of the signal
    if sample in ("signal", "mixed"):
        sg = S["signal"]
        jm = np.flatnonzero(sg["prim_jet0"])
        pt = np.hypot(sg["prim_p4"][:, 1], sg["prim_p4"][:, 2])
        ch = np.array([sim.dec.charge3(int(i)) != 0 for i in sg["prim_id"]])
        eta = np.arcsinh(sg["prim_p4"][:, 3] / np.maximum(pt, 1e-9))
        follow = []
        if len(jm):
            follow.append(jm[np.argmax(np.where(ch[jm], pt[jm], 0))])
        g_ = np.flatnonzero((sg["prim_id"] == 22) & (np.abs(eta) < 3.0))
        if len(g_):
            follow.append(g_[np.argmax(pt[g_])])
        mu_ = np.flatnonzero(np.abs(sg["prim_id"]) == 13)
        if len(mu_):
            follow.append(mu_[np.argmax(pt[mu_])])
        for j in follow:
            pts = []
            for part, dets in (("trk", hd), ("mu", hd)):
                m = (sg[f"{part}_prim"] == j) & np.isin(sg[f"{part}_det"], dets)
                if view == "xy":
                    m &= np.isin(sg[f"{part}_det"], BARREL_DETS)
                if m.any():
                    o = np.argsort(sg[f"{part}_t"][m])
                    pts.append(np.column_stack(xy(sg[f"{part}_x"][m][o], sg[f"{part}_y"][m][o], sg[f"{part}_z"][m][o])))
            k = int(sg["prim_kind"][j])
            lab = None
            if pts and view != "etaphi":
                p = np.concatenate(pts)
                ax.plot(p[:, 0], p[:, 1], "-", color=COL[k], lw=0.9, zorder=30)
                lab = p[-1]
            m = (sg["cal_prim"] == j) & np.isin(sg["cal_code"] // sim.CODE, cd)
            if m.any():
                i = np.argmax(sg["cal_e"][m])
                _, qx, qy, qz = sim.cell_position(sg["cal_code"][m][i:i + 1])
                if not (view == "xy" and (sg["cal_code"][m][i] // sim.CODE) not in BARREL_DETS):
                    px_, py_ = xy(qx, qy, qz)
                    ax.scatter(px_, py_, s=120, facecolors="none", edgecolors="k", lw=1.2, zorder=31)
                    lab = (px_[0], py_[0])
            if lab is not None:
                ax.annotate(f"{sim.dec.name(int(sg['prim_id'][j]))} {pt[j]:.0f} GeV", lab, xytext=(8, 6), textcoords="offset points", color=COL[k], fontsize=11, fontweight="bold", zorder=32)
    lim = {"TRK": (300, 125), "ECAL": (360, 165), "HCAL": (1300, 430), "MUON": (1100, 760), "ALL": (1300, 800)}[station]
    if view == "rz":
        ax.set_xlim(-lim[0], lim[0]); ax.set_ylim(0, lim[1]); ax.set_xlabel("z [cm]", fontsize=16, loc="right"); ax.set_ylabel("r [cm]", fontsize=16, loc="top")
    elif view == "xy":
        ax.set_xlim(-lim[1] * 1.6, lim[1] * 1.6); ax.set_ylim(-lim[1], lim[1]); ax.set_xlabel("x [cm]", fontsize=16, loc="right"); ax.set_ylabel("y [cm]", fontsize=16, loc="top")
    else:
        em = 2.7 if station == "TRK" else (3.1 if station == "ECAL" else 5.3)
        ax.set_xlim(-em, em); ax.set_ylim(-math.pi, math.pi); ax.set_xlabel(r"$\eta$", fontsize=16, loc="right"); ax.set_ylabel(r"$\varphi$", fontsize=16, loc="top")
    ax.grid(True, ls=":", color="k", lw=0.6)
    ax.tick_params(labelsize=14)
    # the header
    name = {"signal": "SIM hits of the signal", "intime": "SIM hits of the in-time pileup", "oot": "SIM hits of the out-of-time pileup", "mixed": "SIM + pileup hits"}[sample]
    fig.text(0.07, 0.955, f"{name} at {TITLE[station]}", fontsize=16, fontweight="bold")
    fig.text(0.63, 0.955, "standalone, seed 1", fontsize=12, fontweight="bold")
    parts = [f"{sim.DET_NAME[d]} {int((H['det'][mh] == d).sum())}" for d in hd if (H["det"][mh] == d).any()]
    parts += [f"{sim.DET_NAME[d]} {int((cdet[mc] == d).sum())} cells ({C['e'][mc & (cdet == d)].sum():.3g} {'p.e.' if d == sim.HF else 'GeV'})" for d in cd if (cdet[mc] == d).any()]
    line, lines = "", []
    for p in parts:
        if len(line) + len(p) > 120:
            lines.append(line); line = ""
        line += ("" if not line else ", ") + p
    lines.append(line)
    fig.text(0.07, 0.93, f"{int(mh.sum()) + int(mc.sum())} hits and cells: " + lines[0], fontsize=8.5, fontweight="bold", color="0.25")
    for i, l in enumerate(lines[1:3]):
        fig.text(0.08, 0.912 - 0.018 * i, l, fontsize=8.5, fontweight="bold", color="0.25")
    n = add.numbers if False else None
    it, oo = S["intime"], S["oot"]
    desc = {"signal": f"the {len(S['signal']['prim_id'])} primaries of the signal through the full physics (field, decays, EM, hadronic); colour = kind of the primary the hit descends from",
            "intime": f"the {len(it['col_k'])} collisions of bunch crossing 0, simulated separately from the signal ({int(it['col_nprim'].sum())} primaries)",
            "oot": f"the {len(oo['col_k'])} collisions of the fifteen other crossings ({int(oo['col_nprim'].sum())} primaries): early bx -12..-1, late bx +1..+3",
            "mixed": f"signal + in-time + out-of-time, added by hand (add_hits_SIM.py): {len(it['col_k']) + len(oo['col_k'])} pileup collisions drawn underneath by bunch crossing; "
                     "a cell takes the colour of the origin that deposits most"}[sample]
    fig.text(0.07, 0.868, desc, fontsize=8.5, fontweight="bold", color="0.25")
    vtxt = {"rz": "r-z (longitudinal)", "xy": "x-y (barrel, transverse)", "etaphi": r"$\eta$-$\varphi$ (seen from the nominal collision point)"}
    fig.text(0.07, 0.85, "view: " + vtxt[view], fontsize=8.5, fontweight="bold", color="0.25")
    # the legend panel
    fig.text(0.775, 0.93, "chapter 10_SIM, standalone", fontsize=9.5, fontweight="bold", color="0.25")
    fig.text(0.775, 0.908, "PU = 60 per bx (Poisson), bx -12..3", fontsize=9.5, fontweight="bold", color="0.25")
    fig.text(0.775, 0.886, "2025 detector model, 13.6 TeV, B = 3.8 T", fontsize=9.5, fontweight="bold", color="0.25")
    hs = [Line2D([], [], ls="none", marker=MRK[k], color=COL[k], ms=8, label=f"{CAT[k]} ({counts[k]})") for k in range(8) if counts[k]]
    hs.append(Line2D([], [], ls="-", color="0.72", label="detector elements"))
    leg = fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.768, 0.82), frameon=False, fontsize=9.5, title=f"{station} hits by kind / bunch crossing", title_fontsize=9.5, labelspacing=1.1)
    leg._legend_box.align = "left"
    if cd:
        fig.text(0.775, 0.33, "marker size of calorimeter cells:", fontsize=10.5, fontweight="bold")
        axl = fig.add_axes([0.775, 0.19, 0.19, 0.13]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 3)
        for i, (e_, lab) in enumerate(((0.01, "E = 0.01 GeV"), (1.0, "E = 1 GeV"), (100.0, "E = 100 GeV"))):
            axl.scatter([0.08], [2.5 - i], s=size_of(np.array([e_])), color="0.4")
            axl.text(0.22, 2.5 - i, lab, va="center", fontsize=10.5, fontweight="bold")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", choices=["signal", "intime", "oot", "mixed"])
    ap.add_argument("--station", choices=list(STATIONS))
    ap.add_argument("--view", choices=["rz", "xy", "etaphi"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("-o", "--outdir", type=Path, default=HERE / "pages")
    args = ap.parse_args()
    args.outdir.mkdir(exist_ok=True)
    if args.all:
        todo = [(s, st, v) for s in ("signal", "intime", "oot", "mixed") for st in STATIONS for v in ("rz", "xy", "etaphi")]
    elif args.sample:
        todo = [(args.sample, args.station or "ALL", args.view or "etaphi")]
    else:   # the pages of the chapter
        todo = [("signal", "TRK", "rz"), ("signal", "TRK", "xy"), ("signal", "ECAL", "rz"), ("signal", "HCAL", "rz"), ("signal", "ALL", "etaphi"),
                ("intime", "ALL", "etaphi"), ("oot", "ALL", "etaphi"),
                ("mixed", "TRK", "xy"), ("mixed", "ECAL", "etaphi"), ("mixed", "HCAL", "rz"), ("mixed", "MUON", "rz"),
                ("mixed", "ALL", "rz"), ("mixed", "ALL", "xy"), ("mixed", "ALL", "etaphi")]
    cache = {}
    S = add.load(HERE.parent)
    for s, st, v in todo:
        if s not in cache:
            cache[s] = collect(s, S)
        H, C, _ = cache[s]
        out = args.outdir / f"simHits_{s}_10_SIM_{st}_{v}.png"
        dt = page(s, st, v, H, C, S, out)
        print(f"{out.name}: {dt:.1f} s")


if __name__ == "__main__":
    main()
