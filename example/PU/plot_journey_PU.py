#!/usr/bin/env python3
"""Draw the event with the crowd in the (eta, phi) plane, in the layout of the CMSSW analyser pages
(stage 7, "pileup"): the signal particles coloured by the mixed jet that holds them, the in-time pileup
particles in khaki, the circles of the mixed anti-kT jets, and the header with our jet before and after.

The input is the record written by Standalone/standalone_pileup.py (event_PU.lhe): the chapter-7 record
followed by the pileup tables (the collisions with their vertices, the in-time particles with the mixed
jet that holds each, the mixed jets with their areas and subtracted momenta, and the signal rows with
their mixed jet). The picture follows the GenJet page of chapter 7 so that the two can be compared: every
mixed jet above the threshold is a circle of radius R around its axis (a star), the leading eight in
colour with their signal constituents in the same colour, the b jet red and the bbar jet blue; the pileup
constituents of a jet are drawn in the jet's colour but small and pale, the pileup particles in no jet in
khaki; the signal particles in no jet are hollow grey circles; the last copy of each b quark is a black
star. The header gives the crowd (mu, collisions, particles), rho, and the quark -> photograph -> mixed
jet lines with the area, rho A and the subtracted pT; the right column lists the leading mixed jets.

  python3 plot_journey_PU.py                        # -> journey_PU.pdf
  python3 plot_journey_PU.py --signal-only          # the same frame without the crowd (for the film and the comparison)
  python3 plot_journey_PU.py --png

Dependency: matplotlib (any version from 3.3); on this Mac /usr/bin/python3.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "PS"))
sys.path.insert(0, str(HERE.parent / "GenJet"))
sys.path.insert(0, str(HERE.parent / "GenJet" / "Standalone"))
sys.path.insert(0, str(HERE / "Standalone"))
import plot_journey_PS as ps  # noqa: E402
import plot_journey_GenJet as pg  # noqa: E402
from standalone_genjets import read_jet_tables  # noqa: E402
from standalone_pileup import read_pileup_tables, ETA_MAX as ETA_IN  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL_B, COL_BBAR, COL_OTHER, COL_UNCL, COL_PU = pg.COL_B, pg.COL_BBAR, pg.COL_OTHER, pg.COL_UNCL, "#8a7f5a"
PALETTE = pg.PALETTE
SUM, TEXT, TEXT2, ETA_MAX = ps.SUM, ps.TEXT, ps.TEXT2, ps.ETA_MAX
STAGES = pg.STAGES + ["7 pileup"]
STAGE = 7
parse_listing, kinematics, dphi, area = ps.parse_listing, ps.kinematics, ps.dphi, ps.area


def analyse(rows, genjets, jets, signal_of, R: float):
    final = [r for r in rows.values() if r["status"] > 0]
    hard = {r["id"]: r for r in rows.values() if r["status"] == -23 and abs(r["id"]) == 5}
    quarks = {pid: rows[pg.last_copy(rows, r["no"])] for pid, r in hard.items()}
    for r in final:
        r["jet"] = signal_of.get(r["no"], -1)
    # the mixed jet of each quark: the one matched to the chapter-7 jet of the quark, else the nearest within R
    matched, roles = {}, {}
    for pid, q in quarks.items():
        k = kinematics(q)
        best = None
        for g in genjets:
            d = math.hypot(g["eta"] - k["eta"], dphi(g["phi"], k["phi"]))
            if d < R and (best is None or d < best[0]):
                best = (d, g)
        mixed = None
        if best is not None:
            cands = [j for j in jets if j["genjet"] == best[1]["rank"]]
            if cands:
                mixed = min(cands, key=lambda j: j["dr"] if j["dr"] is not None else 9)
        matched[pid] = (k, best, mixed)
        if mixed is not None:
            roles[mixed["rank"]] = "b" if pid > 0 else "bbar"
    s = dict(n_final=len(final), n_jets=len(jets), quarks=matched, roles=roles, n_border=sum(1 for r in final if abs(kinematics(r)["eta"]) > ETA_MAX),
             n_signal_clustered=sum(1 for r in final if r["jet"] >= 0), n_signal_in=len(signal_of))
    return final, s


def draw_particles(ax, final, particles, jets, s, R, label_min, signal_only):
    if not signal_only:
        # the crowd first, underneath: in no jet khaki, in a jet the jet's colour, pale
        xs, ys, cs, ss = [], [], [], []
        for p in particles:
            k = kinematics(p)
            if abs(k["eta"]) > ETA_MAX or k["pt"] <= 0:
                continue
            col = pg.jet_colour(p["jet"], s["roles"]) if p["jet"] >= 0 else COL_PU
            xs.append(k["eta"])
            ys.append(k["phi"])
            cs.append(col)
            ss.append(area(k["pt"]) * 0.55)
        ax.scatter(xs, ys, s=ss, c=cs, marker="o", alpha=0.35, linewidth=0, zorder=2, rasterized=True)
    n_border = 0
    for r in sorted(final, key=lambda r: (r["jet"] >= 0, -kinematics(r)["pt"])):
        k = kinematics(r)
        e, f, border = ps.place(k)
        col = pg.jet_colour(r["jet"], s["roles"]) if r["jet"] >= 0 else COL_UNCL
        if border:
            n_border += 1
            ax.scatter(e, f, s=area(k["pt"]), marker="x", color=col, linewidth=1.0, zorder=3)
            continue
        if r["jet"] < 0:
            ax.scatter(e, f, s=area(k["pt"]), marker="o", facecolors="none", edgecolors=col, linewidth=0.9, zorder=3)
        else:
            ax.scatter(e, f, s=area(k["pt"]), marker="o", color=col, alpha=0.95, linewidth=0, zorder=5 if col != COL_OTHER else 4)
    for j in sorted(jets, key=lambda j: j["rank"], reverse=True):
        col = pg.jet_colour(j["rank"], s["roles"])
        lead = j["rank"] in s["roles"] or col != COL_OTHER
        if not lead and j["pt"] < label_min:
            continue
        pg.circle(ax, j["eta"], j["phi"], R, col, 1.8 if lead else 0.8, "-" if lead else "--", zorder=6 if lead else 2)
        ax.scatter(j["eta"], j["phi"], s=area(j["pt"]) * 1.6, marker="*", color=col, zorder=7, linewidth=0)
        if j["pt"] >= label_min:
            ax.annotate(f"[{j['rank']}] {j['pt']:.1f}" + ("" if signal_only else f" $\\to$ {j['pt_sub']:.1f}"), (j["eta"], j["phi"]), xytext=(4, 3), textcoords="offset points",
                        fontsize=7.5 if lead else 6.5, color=col, weight="bold" if lead else "normal", zorder=8)
    for pid, (k, best, mixed) in s["quarks"].items():
        e, f, _ = ps.place(k)
        ax.scatter(e, f, s=area(k["pt"]) * 1.3, marker="*", facecolors="none", edgecolors="black", linewidth=1.0, zorder=8)
        ax.annotate(rf"{ps.STAR[pid]['name']} quark {k['pt']:.1f}", (e, f), xytext=(5, -9), textcoords="offset points", fontsize=7.5, color="black", zorder=8)
    return n_border


def draw_legend(ax, s, jets, R, summary, signal_only):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y = 0.99
    n_other = s["n_jets"] - len(s["roles"])
    entries = [("*", COL_B, False, rf"$b$ jet: axis, circle $R$ = {R:g}"), ("*", COL_BBAR, False, r"$\bar b$ jet"),
               ("*", PALETTE[0], False, f"other mixed jets ({n_other}), 6 in colour"),
               ("o", COL_OTHER, False, "signal constituents (jet colour)"), ("o", COL_UNCL, True, f"signal, in no jet ({s['n_signal_in'] - s['n_signal_clustered']})")]
    if not signal_only:
        entries += [("o", COL_PU, False, "pileup particle, in no jet"), ("o", PALETTE[0], False, "pileup constituent (pale)")]
    entries.append(("*", "black", True, r"$b$ / $\bar b$ quark, last copy"))
    for marker, col, hollow, label in entries:
        y -= 0.036
        if hollow:
            ax.scatter(0.07, y, s=area(15.0) * (1.5 if marker == "*" else 1.0), marker=marker, facecolors="none", edgecolors=col, linewidth=1.0)
        else:
            ax.scatter(0.07, y, s=area(15.0) * (1.5 if marker == "*" else 1.0), marker=marker, color=col, linewidth=0, alpha=0.4 if "pale" in label else 1.0)
        ax.text(0.16, y, label, fontsize=7.4, color=TEXT, va="center")
    y -= 0.04
    ax.text(0.0, y, r"leading mixed jets: [i] $p_T$  A  $\rho$A  $p_T$$-$$\rho$A  sig/pu", fontsize=7.4, color=TEXT, va="center", weight="bold")
    for j in jets[:9]:
        y -= 0.03
        col = pg.jet_colour(j["rank"], s["roles"])
        ax.text(0.0, y, f"[{j['rank']}] {j['pt']:5.1f} {j['area']:.2f} {j['rho_area']:5.1f} {j['pt_sub']:5.1f} {j['pt_signal']:5.1f}/{j['pt_pileup']:5.1f}", fontsize=6.9,
                color=col if col != COL_OTHER else TEXT2, va="center", family="monospace")
    if len(jets) > 9:
        y -= 0.03
        ax.text(0.0, y, f"... {len(jets) - 9} more", fontsize=7, color=TEXT2, va="center")
    y -= 0.045
    ax.text(0.0, y, "the crowd:", fontsize=8, color=TEXT, va="center", weight="bold")
    for line in summary:
        y -= 0.028
        ax.text(0.0, y, line, fontsize=6.6, color=TEXT2, va="center")
    y -= 0.045
    ax.text(0.0, y, "stages:", fontsize=8, color=TEXT, va="center", weight="bold")
    for i, nm in enumerate(STAGES, start=1):
        y -= 0.028
        ax.text(0.05, y, nm, fontsize=7.2, color="#ff0000" if i == STAGE else "#888888", va="center", weight="bold" if i == STAGE else "normal")


def header(fig, s, jets, R, title, source, label_min, info, signal_only):
    x, y, dy = 0.06, 0.955, 0.027
    fig.text(x, y, title, fontsize=12.5, weight="bold", color=TEXT, va="center")
    fig.text(0.985, y, source, fontsize=9, color=TEXT, ha="right", va="center")
    y -= 0.036
    fig.text(x, y, (f"$\\mu$ = {info['mu']}: {info['n_in_time']} in-time collisions, {info['n_pu_in']} pileup particles (|$\\eta$| < {ETA_IN:g}, no $\\nu$) + {s['n_signal_in']} of the signal; "
                    f"$\\rho$ = {info['rho']} GeV per unit area; {s['n_jets']} mixed jets; label = [jet] $p_T$ $\\to$ $p_T - \\rho A$ ($p_T$ > {label_min:g})")
             if not signal_only else f"the signal alone: {s['n_final']} particles, the chapter-7 jets; label = [jet] $p_T$ ($p_T$ > {label_min:g} GeV)", fontsize=7.8, color=TEXT2, va="center")
    for pid, col in ((5, COL_B), (-5, COL_BBAR)):
        y -= dy
        if pid not in s["quarks"]:
            continue
        k, best, mixed = s["quarks"][pid]
        line = rf"{ps.STAR[pid]['name']} quark (last copy): $p_T$ = {k['pt']:.2f}"
        if best is not None:
            line += rf"  $\rightarrow$ photograph [{best[1]['rank']}]: {best[1]['pt']:.2f} GeV, n = {best[1]['n']}"
        if mixed is not None and not signal_only:
            line += (rf"  $\rightarrow$ mixed jet [{mixed['rank']}]: {mixed['pt']:.2f} GeV ({mixed['n_signal']} + {mixed['n_pileup']} particles, "
                     rf"{mixed['pt_signal']:.2f} + {mixed['pt_pileup']:.2f}), A = {mixed['area']:.3f}, $\rho A$ = {mixed['rho_area']:.2f}, $p_T - \rho A$ = {mixed['pt_sub']:.2f}")
        fig.text(x, y, line, fontsize=7.6, color=col, va="center", weight="bold")
    y -= dy
    fig.text(x, y, info["vertex_line"], fontsize=7.8, color=TEXT, va="center")
    return y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=HERE / "event_PU.lhe")
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--signal-only", action="store_true")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--label-min", type=float, default=20.0)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    if args.output is None:
        args.output = HERE / ("journey_PU_signal.pdf" if args.signal_only else "journey_PU.pdf")
    lines = args.input.read_text().splitlines()
    rows = parse_listing(lines)
    genjets, genjets_of = read_jet_tables(args.input)
    summary, collisions, particles, jets, signal_of = read_pileup_tables(args.input)
    R = 0.4
    if args.signal_only:
        # the chapter-7 frame: the ak4 jets and their constituents, drawn with the same code
        jets_draw = [dict(g, pt_sub=g["pt"], area=0.0, rho_area=0.0, pt_signal=g["pt"], pt_pileup=0.0, n_signal=g["n"], n_pileup=0, genjet=g["rank"], dr=0.0) for g in genjets["ak4"]]
        signal_draw = dict(genjets_of["ak4"])
    else:
        jets_draw, signal_draw = jets, signal_of
    final, s = analyse(rows, genjets["ak4"], jets_draw, signal_draw, R)
    in_time = [c for c in collisions if c["bx"] == 0 and c["k"] > 0]
    sig = [c for c in collisions if c["k"] == 0]
    rho = next((l.split("grid median")[1].split()[0] for l in summary if "rho:" in l), "?")
    mu = next((l.split("mixed at mu =")[1].split()[0] for l in summary if "mixed at mu" in l), "?")
    n_pu_in = sum(1 for p in particles if abs(math.asinh(p["pz"] / max(math.hypot(p["px"], p["py"]), 1e-12))) < ETA_IN and abs(p["id"]) not in (12, 14, 16))
    vz = [c["z"] for c in in_time]
    info = dict(mu=mu, n_in_time=len(in_time), n_pu_in=n_pu_in, rho=rho,
                vertex_line=(f"vertices: the signal at z = {sig[0]['z']:.2f} cm, t = {sig[0]['t'] * 1e3:.0f} ps; the {len(in_time)} in-time collisions spread over "
                             f"{min(vz):.1f} to {max(vz):.1f} cm (r.m.s. {ps.math.sqrt(sum((v - sum(vz) / len(vz)) ** 2 for v in vz) / len(vz)):.1f} cm); "
                             f"{sum(c['visible'] for c in in_time)} of them visible to the tracker; {sum(1 for c in collisions if c['k'] > 0)} collisions in the 7 crossings")
                if sig and in_time else "")
    kinds = {}
    for c in in_time:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    side_summary = [f"in time: {len(in_time)} collisions (" + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) + ")",
                    f"{sum(c['nfinal'] for c in in_time)} particles, {n_pu_in} drawn",
                    f"out of time: {sum(1 for c in collisions if c['k'] > 0 and c['bx'] != 0)} collisions, not clustered",
                    f"rho = {rho} GeV per unit area"] if not args.signal_only else ["none: the signal alone"]
    title = args.title or ("stage 7: the crowd on the road (pileup)" if not args.signal_only else "stage 7: the family alone, before the crowd")
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "mathtext.default": "regular"})
    fig = plt.figure(figsize=(10.0, 7.4))
    y_header = header(fig, s, jets_draw, R, title, f"standalone, {args.input.name}", args.label_min, info, args.signal_only)
    top = y_header - 0.045
    ax = fig.add_axes([0.06, 0.08, 0.70, top - 0.08])
    ps.draw_axes(ax)
    draw_particles(ax, final, particles, jets_draw, s, R, args.label_min, args.signal_only)
    side = fig.add_axes([0.79, 0.08, 0.20, top - 0.08])
    draw_legend(side, s, jets_draw, R, side_summary, args.signal_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    if args.png:
        fig.savefig(args.output.with_suffix(".png"), dpi=200)
    print(f"wrote {args.output}" + (f" and {args.output.with_suffix('.png')}" if args.png else ""))
    for pid, (k, best, mixed) in s["quarks"].items():
        nm = "b" if pid > 0 else "bbar"
        if mixed:
            print(f"{nm} quark {k['pt']:.2f} -> photograph [{best[1]['rank']}] {best[1]['pt']:.2f} -> mixed jet [{mixed['rank']}] {mixed['pt']:.2f}, A {mixed['area']:.3f}, "
                  f"rho A {mixed['rho_area']:.2f}, pT - rho A {mixed['pt_sub']:.2f} (signal {mixed['pt_signal']:.2f}, pileup {mixed['pt_pileup']:.2f})")


if __name__ == "__main__":
    main()
