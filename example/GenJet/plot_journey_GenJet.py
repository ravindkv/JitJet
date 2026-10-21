#!/usr/bin/env python3
"""Draw the generator-level jets of the event in the (eta, phi) plane, in the layout of the CMSSW
genJetAnalyzer (stage 6, "GenJets": pages 6 to 8 of CMSSW/genJetAnalyzer_ME_PS_MPI_Had_Decay_GenJet_out.pdf).

The input is the record written by Standalone/standalone_genjets.py (event_GenJet.lhe): the listing of
chapter 6 followed by the jet table (one row per jet and collection, with its flavours) and the
constituent table (one row per final particle, saying which jet of ak4, ak4nonu and ak8 holds it). The
picture follows the CMSSW page of the chosen collection, so that the standalone and the CMSSW result
can be compared side by side: every jet above the threshold is a circle of radius R around its axis
(a star), the leading eight in colour with their constituents in the same colour, the b jet red and the
bbar jet blue, the other jets and their constituents grey, the final particles that ended in no jet
(or in a jet below 3 GeV) as hollow grey circles; the last copy of each b quark is a black star. The
header gives, as CMSSW does, the quark -> jet lines with dR and the response pT(jet)/pT(quark), the
b jet + bbar jet system, and the flavour labels of the b jets; the right column lists the leading jets.

  python3 plot_journey_GenJet.py                              # ak4GenJets -> journey_GenJet.pdf
  python3 plot_journey_GenJet.py --collection ak4nonu -o journey_GenJet_nonu.pdf
  python3 plot_journey_GenJet.py --collection ak8 -o journey_GenJet_ak8.pdf
  python3 plot_journey_GenJet.py --png

Dependency: matplotlib (any version from 3.3); on this Mac /usr/bin/python3.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "PS"))
sys.path.insert(0, str(HERE.parent / "Had"))
sys.path.insert(0, str(HERE / "Standalone"))
import plot_journey_PS as ps  # noqa: E402
import plot_journey_Had as jh  # noqa: E402
from standalone_genjets import COLLECTIONS, read_jet_tables  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

COL_B, COL_BBAR, COL_OTHER, COL_UNCL = "#ff0000", "#0000ff", "#8a8a8a", "#b4b4b4"
PALETTE = ["#00999a", "#9b59d6", "#3b9ae1", "#ff1493", "#2e9e2e", "#a3a300"]
SUM, TEXT, TEXT2, ETA_MAX = ps.SUM, ps.TEXT, ps.TEXT2, ps.ETA_MAX
STAGES = ps.STAGES + ["6 GenJets"]
STAGE = 6
parse_listing, kinematics, vsum, mass, dphi, area = ps.parse_listing, ps.kinematics, ps.vsum, ps.mass, ps.dphi, ps.area
hadron_tex = jh.hadron_tex


def last_copy(rows, no: int) -> int:
    """The last same-flavour copy of a hard parton before the hadronisation (status below 70)."""
    r = rows[no]
    while r["d1"] > 0:
        ds = [rows[d] for d in range(r["d1"], max(r["d1"], r["d2"]) + 1) if d in rows]
        same = [d for d in ds if d["id"] == r["id"] and abs(d["status"]) < 70]
        if not same:
            break
        r = same[0]
    return r["no"]


def analyse(rows, jets, jet_of, R: float):
    final = [r for r in rows.values() if r["status"] > 0]
    hard = {r["id"]: r for r in rows.values() if r["status"] == -23 and abs(r["id"]) == 5}
    quarks = {pid: rows[last_copy(rows, r["no"])] for pid, r in hard.items()}
    for r in final:
        r["jet"] = jet_of.get(r["no"], -1)
    matched = {}
    for pid, q in quarks.items():
        k = kinematics(q)
        best = None
        for j in jets:
            d = math.hypot(j["eta"] - k["eta"], dphi(j["phi"], k["phi"]))
            if d < R and (best is None or d < best[0]):
                best = (d, j)
        matched[pid] = (k, best)
    roles = {}
    for pid, (k, best) in matched.items():
        if best is not None:
            roles[best[1]["rank"]] = "b" if pid > 0 else "bbar"
    s = dict(n_final=len(final), n_clustered=sum(1 for r in final if r["jet"] >= 0), n_jets=len(jets), quarks=matched, roles=roles,
             n_border=sum(1 for r in final if abs(kinematics(r)["eta"]) > ETA_MAX))
    a, b = matched.get(5, (None, None))[1], matched.get(-5, (None, None))[1]
    if a and b:
        ja, jb = a[1], b[1]
        px = ja["pt"] * math.cos(ja["phi"]) + jb["pt"] * math.cos(jb["phi"])
        py = ja["pt"] * math.sin(ja["phi"]) + jb["pt"] * math.sin(jb["phi"])
        pa = dict(e=ja["e"], px=ja["pt"] * math.cos(ja["phi"]), py=ja["pt"] * math.sin(ja["phi"]), pz=ja["pt"] * math.sinh(ja["eta"]))
        pb = dict(e=jb["e"], px=jb["pt"] * math.cos(jb["phi"]), py=jb["pt"] * math.sin(jb["phi"]), pz=jb["pt"] * math.sinh(jb["eta"]))
        tot = {k: pa[k] + pb[k] for k in pa}
        s["bb"] = dict(pt=math.hypot(px, py), px=px, py=py, scalar=ja["pt"] + jb["pt"], dphi=dphi(ja["phi"], jb["phi"]), m=mass(tot))
    return final, s


def jet_colour(rank: int, roles: dict):
    if roles.get(rank) == "b":
        return COL_B
    if roles.get(rank) == "bbar":
        return COL_BBAR
    others = [r for r in range(rank + 1) if r not in roles]
    k = others.index(rank) if rank in others else len(PALETTE)
    return PALETTE[k] if k < len(PALETTE) else COL_OTHER


def circle(ax, eta0, phi0, R, color, lw, ls="-", zorder=3):
    for shift in (0.0, 2 * math.pi, -2 * math.pi):
        f = phi0 + shift
        if -math.pi - R < f < math.pi + R:
            ax.add_patch(Ellipse((eta0, f), 2 * R, 2 * R, fill=False, edgecolor=color, lw=lw, ls=ls, zorder=zorder))


def draw_particles(ax, final, jets, s, R, label_min, names):
    n_border = 0
    for r in sorted(final, key=lambda r: (r["jet"] >= 0, -kinematics(r)["pt"])):
        k = kinematics(r)
        e, f, border = ps.place(k)
        col = jet_colour(r["jet"], s["roles"]) if r["jet"] >= 0 else COL_UNCL
        if border:
            n_border += 1
            ax.scatter(e, f, s=area(k["pt"]), marker="x", color=col, linewidth=1.0, zorder=3)
            continue
        if r["jet"] < 0:
            ax.scatter(e, f, s=area(k["pt"]), marker="o", facecolors="none", edgecolors=col, linewidth=0.9, zorder=2)
        else:
            ax.scatter(e, f, s=area(k["pt"]), marker="o", color=col, alpha=0.9, linewidth=0, zorder=4 if col not in (COL_OTHER,) else 3)
    for j in sorted(jets, key=lambda j: j["rank"], reverse=True):
        col = jet_colour(j["rank"], s["roles"])
        lead = j["rank"] in s["roles"] or col != COL_OTHER
        circle(ax, j["eta"], j["phi"], R, col, 1.8 if lead else 0.9, "-" if lead else "--", zorder=5 if lead else 2)
        ax.scatter(j["eta"], j["phi"], s=area(j["pt"]) * 1.6, marker="*", color=col, zorder=6, linewidth=0)
        if j["pt"] >= label_min:
            ax.annotate(f"[{j['rank']}] {j['pt']:.1f}", (j["eta"], j["phi"]), xytext=(4, 3), textcoords="offset points", fontsize=7.5 if lead else 6.5,
                        color=col, weight="bold" if lead else "normal", zorder=7)
    for pid, (k, best) in s["quarks"].items():
        e, f, _ = ps.place(k)
        ax.scatter(e, f, s=area(k["pt"]) * 1.3, marker="*", facecolors="none", edgecolors="black", linewidth=1.0, zorder=7)
        ax.annotate(rf"{ps.STAR[pid]['name']} quark {k['pt']:.1f}", (e, f), xytext=(5, -9), textcoords="offset points", fontsize=7.5, color="black", zorder=7)
    return n_border


def draw_legend(ax, s, jets, R, coll_key):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y = 0.99
    n_other = s["n_jets"] - len(s["roles"])
    entries = [("*", COL_B, False, rf"$b$ jet: axis, circle $R$ = {R:g}"), ("*", COL_BBAR, False, r"$\bar b$ jet"),
               ("*", PALETTE[0], False, f"other jets ({n_other}), {min(6, n_other)} in colour"),
               ("o", COL_OTHER, False, "jet constituents (jet colour)"), ("o", COL_UNCL, True, f"status 1, in no jet ({s['n_final'] - s['n_clustered']})"),
               ("*", "black", True, r"$b$ / $\bar b$ quark, last copy")]
    for marker, col, hollow, label in entries:
        y -= 0.038
        if hollow:
            ax.scatter(0.07, y, s=area(15.0) * (1.5 if marker == "*" else 1.0), marker=marker, facecolors="none", edgecolors=col, linewidth=1.0)
        else:
            ax.scatter(0.07, y, s=area(15.0) * (1.5 if marker == "*" else 1.0), marker=marker, color=col, linewidth=0)
        ax.text(0.16, y, label, fontsize=7.6, color=TEXT, va="center")
    y -= 0.05
    ax.text(0.0, y, "marker size:", fontsize=8, color=TEXT, va="center", weight="bold")
    for pt in (1, 10, 100):
        y -= 0.036
        ax.scatter(0.07, y, s=area(pt), marker="o", color="#666666")
        ax.text(0.16, y, f"$p_T$ = {pt} GeV", fontsize=7.6, color=TEXT, va="center")
    y -= 0.05
    ax.text(0.0, y, r"leading jets: [i] $p_T$  $\eta$  $\phi$  n  had/part", fontsize=7.6, color=TEXT, va="center", weight="bold")
    for j in jets[:8]:
        y -= 0.031
        col = jet_colour(j["rank"], s["roles"])
        fl = f"{j['hadron_flavour']}/{j['parton_flavour']:+d}" if j["parton_flavour"] else f"{j['hadron_flavour']}/0"
        ax.text(0.0, y, f"[{j['rank']}] {j['pt']:.1f}  {j['eta']:.2f}  {j['phi']:.2f}  {j['n']}  {fl}", fontsize=7.2, color=col if col != COL_OTHER else TEXT2, va="center",
                family="monospace")
    if len(jets) > 8:
        y -= 0.031
        ax.text(0.0, y, f"... {len(jets) - 8} more", fontsize=7, color=TEXT2, va="center")
    y -= 0.045
    ax.text(0.0, y, "stages:", fontsize=8, color=TEXT, va="center", weight="bold")
    for i, nm in enumerate(STAGES, start=1):
        y -= 0.03
        ax.text(0.05, y, nm, fontsize=7.5, color="#ff0000" if i == STAGE else "#888888", va="center", weight="bold" if i == STAGE else "normal")


def header(fig, s, jets, coll_key, R, title, source, label_min, names):
    x, y, dy = 0.06, 0.955, 0.027
    fig.text(x, y, title, fontsize=12.5, weight="bold", color=TEXT, va="center")
    fig.text(0.985, y, source, fontsize=9, color=TEXT, ha="right", va="center")
    y -= 0.036
    border = f"; {s['n_border']} beyond $|\\eta|$ = {ETA_MAX:g} on the border (x)" if s["n_border"] else ""
    fig.text(x, y, f"{s['n_jets']} jets; {s['n_clustered']} of {s['n_final']} status-1 particles clustered; marker size ~ log $p_T$; label = [jet] $p_T$ [GeV] "
                   f"($p_T$ > {label_min:g} GeV)" + border, fontsize=8.5, color=TEXT2, va="center")
    for pid, key, col in ((5, "b", COL_B), (-5, "bbar", COL_BBAR)):
        y -= dy
        if pid not in s["quarks"]:
            continue
        k, best = s["quarks"][pid]
        line = rf"{ps.STAR[pid]['name']} quark (last copy): $p_T$ = {k['pt']:.2f},  $\eta$ = {k['eta']:.2f},  $\phi$ = {k['phi']:.2f}"
        if best is None:
            line += "   (no jet within R)"
        else:
            d, j = best
            line += (rf"  $\rightarrow$ jet [{j['rank']}]: $p_T$ = {j['pt']:.2f} GeV, $\eta$ = {j['eta']:.2f}, $\phi$ = {j['phi']:.2f}, n = {j['n']}, "
                     rf"$\Delta R$ = {d:.3f}, $p_T$(jet)/$p_T$(q) = {j['pt'] / k['pt']:.3f}")
        fig.text(x, y, line, fontsize=8.3, color=col, va="center", weight="bold")
    if "bb" in s:
        bb = s["bb"]
        y -= dy
        fig.text(x, y, rf"$b$ jet + $\bar b$ jet: $|\vec p_T(b) + \vec p_T(\bar b)|$ = {bb['pt']:.2f} GeV ($p_x$ = {bb['px']:.2f}, $p_y$ = {bb['py']:.2f}),   "
                       rf"$\Sigma|p_T|$ = {bb['scalar']:.2f} GeV,   $\Delta\phi$ = {bb['dphi']:.2f},  m = {bb['m']:.1f} GeV", fontsize=8.5, color=SUM["color"], va="center", weight="bold")
    y -= dy
    fl = []
    for j in jets:
        if j["hadron_flavour"]:
            fl.append(f"[{j['rank']}] " + ", ".join(hadron_tex(h.split("[")[0]) for h in j["heavy"].split(";")[0].split()))
    fig.text(x, y, "hadronFlavour 5 or 4 (ghost-associated heavy hadrons): " + ";  ".join(fl[:5]) + (" ..." if len(fl) > 5 else ""), fontsize=8.5, color=TEXT, va="center")
    return y


def draw(final, s, jets, coll_key, R, output: Path, title, source, png, label_min, names):
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "mathtext.default": "regular"})
    fig = plt.figure(figsize=(10.0, 7.4))
    y_header = header(fig, s, jets, coll_key, R, title, source, label_min, names)
    top = y_header - 0.045
    ax = fig.add_axes([0.06, 0.08, 0.70, top - 0.08])
    ps.draw_axes(ax)
    draw_particles(ax, final, jets, s, R, label_min, names)
    side = fig.add_axes([0.79, 0.08, 0.20, top - 0.08])
    draw_legend(side, s, jets, R, coll_key)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    if png:
        fig.savefig(output.with_suffix(".png"), dpi=200)
    print(f"wrote {output}" + (f" and {output.with_suffix('.png')}" if png else ""))


def print_table(s, jets, coll_key):
    c = COLLECTIONS[coll_key]
    print(f"{c['title']}: {s['n_jets']} jets, {s['n_clustered']} of {s['n_final']} final particles clustered")
    print(f"{'jet':>4s} {'pT [GeV]':>9s} {'eta':>7s} {'phi':>7s} {'m':>7s} {'n':>4s} {'had':>4s} {'part':>5s}  heavy hadrons; partons")
    for j in jets[:15]:
        print(f"{j['rank']:4d} {j['pt']:9.2f} {j['eta']:7.2f} {j['phi']:7.2f} {j['m']:7.2f} {j['n']:4d} {j['hadron_flavour']:4d} {j['parton_flavour']:5d}  {j['heavy']}")
    if len(jets) > 15:
        print(f"... and {len(jets) - 15} more")
    for pid, (k, best) in s["quarks"].items():
        nm = "b" if pid > 0 else "bbar"
        if best is None:
            print(f"{nm} quark: pT = {k['pt']:.2f}, eta = {k['eta']:.2f}, phi = {k['phi']:.2f}: no jet within R")
        else:
            d, j = best
            print(f"{nm} quark: pT = {k['pt']:.2f}, eta = {k['eta']:.2f}, phi = {k['phi']:.2f} -> jet [{j['rank']}] pT = {j['pt']:.2f}, n = {j['n']}, dR = {d:.3f}, response {j['pt'] / k['pt']:.3f}")
    if "bb" in s:
        bb = s["bb"]
        print(f"b jet + bbar jet: |pT sum| = {bb['pt']:.2f} GeV, Sigma|pT| = {bb['scalar']:.2f}, dphi = {bb['dphi']:.2f}, m = {bb['m']:.1f} GeV")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=HERE / "event_GenJet.lhe")
    ap.add_argument("-o", "--output", type=Path, default=None, help="default journey_GenJet.pdf (ak4), journey_GenJet_nonu.pdf, journey_GenJet_ak8.pdf")
    ap.add_argument("--collection", choices=sorted(COLLECTIONS), default="ak4")
    ap.add_argument("--title", default=None)
    ap.add_argument("--source", default=None, help="text at the top right (default: the input file name)")
    ap.add_argument("--label-min", type=float, default=5.0)
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    c = COLLECTIONS[args.collection]
    if args.output is None:
        args.output = HERE / {"ak4": "journey_GenJet.pdf", "ak4nonu": "journey_GenJet_nonu.pdf", "ak8": "journey_GenJet_ak8.pdf"}[args.collection]
    lines = args.input.read_text().splitlines()
    rows = parse_listing(lines)
    names = jh.read_names(lines)
    jets_all, jet_of_all = read_jet_tables(args.input)
    jets, jet_of = jets_all[args.collection], jet_of_all[args.collection]
    if not jets:
        sys.exit(f"no {args.collection} jets found in {args.input}: run Standalone/standalone_genjets.py first")
    final, s = analyse(rows, jets, jet_of, c["R"])
    print_table(s, jets, args.collection)
    title = args.title or (f"6. {c['title']}: anti-$k_T$ R = {c['R']:g}, jet $p_T$ > 3 GeV" + (", no neutrinos" if c["nonu"] else ""))
    source = args.source if args.source is not None else f"standalone: {args.input.name}"
    draw(final, s, jets, args.collection, c["R"], args.output, title, source, args.png, args.label_min, names)


if __name__ == "__main__":
    main()
