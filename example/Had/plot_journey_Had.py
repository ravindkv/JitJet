#!/usr/bin/env python3
"""Draw the event after hadronisation in the (eta, phi) plane, in the layout of the CMSSW
genPartAnalyzer (stage 4, primary hadrons).

The input is the hadron-level listing written by Standalone/standalone_hadronisation.py
(PYTHIA listing format, saved as event_Had.lhe): the primary hadrons are the rows with
status 81-84, every parton now has a negative status. The picture follows
example/Had/CMSSW/genPartAnalyzer_ME_PS_MPI_Had_out.pdf, page 4, so that the standalone and
the CMSSW result can be compared side by side. It reuses the frame, the marker sizes and
the parsing of ../PS/plot_journey_PS.py; the categories of this stage are

  * the b hadron (red) and the bbar hadron (blue): the primary hadron of each hard b quark's
    string that carries its flavour, labelled with its name and pT;
  * their vector sum (magenta star), as at every stage;
  * every other primary hadron (purple dots), filled because a hadron is what the detector
    will see; labelled with its pT above 5 GeV.

The header keeps the CMSSW lines (the two B hadrons, the net b + bbar) and adds the share
of the quark momentum each B hadron kept, x_E = E_B / E_b and x_pT, the number of strings,
and the vector sum of all hadrons, which must still vanish.

  python3 plot_journey_Had.py                              # -> journey_Had.pdf
  python3 plot_journey_Had.py -i event_Had.lhe -o journey_Had.pdf
  python3 plot_journey_Had.py --png

Dependency: matplotlib (any version from 3.3); on this Mac /usr/bin/python3.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "PS"))
import plot_journey_PS as ps  # noqa: E402  (../PS/plot_journey_PS.py)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAMILY = {
    "b": dict(label=r"$b$ hadron", color="#ff0000", marker="o"),
    "bbar": dict(label=r"$\bar b$ hadron", color="#0000ff", marker="o"),
    "hadron": dict(label="primary hadron", color="#9b4dff", marker="o"),
}
SUM, STAR, TEXT, TEXT2, ETA_MAX = ps.SUM, ps.STAR, ps.TEXT, ps.TEXT2, ps.ETA_MAX
STAGE = 4

parse_listing, kinematics, vsum, mass, dphi, area = ps.parse_listing, ps.kinematics, ps.vsum, ps.mass, ps.dphi, ps.area

GREEK = {"Lambda": r"\Lambda", "Sigma": r"\Sigma", "Xi": r"\Xi", "Omega": r"\Omega", "Delta": r"\Delta",
         "pi": r"\pi", "rho": r"\rho", "omega": r"\omega", "eta": r"\eta", "phi": r"\phi", "psi": r"\psi", "Upsilon": r"\Upsilon"}


def hadron_tex(name: str) -> str:
    """PYTHIA's hadron name as TeX: B*bar0 -> \\bar B^{*0}, Lambda_b0 -> \\Lambda_b^0, K*- -> K^{*-}."""
    m = re.match(r"^([A-Za-z]+)('?)(\*?)(?:_([a-z]+?))?(bar)?(\*?)([+\-0]*)$", name)
    if not m:
        return name
    base, prime, star1, sub, bar, star2, charge = m.groups()
    bar_flag = bar is not None
    if base.endswith("bar"):
        base, bar_flag = base[:-3], True
    if sub and sub.endswith("bar"):
        sub, bar_flag = sub[:-3], True
    sym = GREEK.get(base, base) + prime
    if bar_flag:
        sym = rf"\bar{{{sym}}}" if len(sym) > 1 else rf"\bar {sym}"
    sup = (star1 or star2) + charge.replace("+", "+").replace("-", "-")
    out = sym
    if sub:
        out += f"_{{{sub}}}"
    if sup:
        out += f"^{{{sup}}}"
    return f"${out}$"


def heavy_content(pid: int) -> int:
    a = abs(pid)
    if a < 100:
        return 0
    digits = [int(c) for c in str(a)]
    quarks = digits[-4:-1] if a >= 1000 else digits[-3:-1]
    return 5 if 5 in quarks else 4 if 4 in quarks else 0


def last_parton_copy(rows, no: int) -> int:
    """Follow the same-flavour daughter of a hard parton down to its last parton copy (status -71/-73)."""
    r = rows[no]
    while r["status"] < 0:
        d1, d2 = r["d1"], r["d2"]
        cands = [rows[d] for d in range(d1, (d2 if d2 >= d1 else d1) + 1) if d in rows]
        same = [c for c in cands if c["id"] == r["id"] and abs(c["status"]) < 80]
        if not same:
            break
        r = same[0]
    return r["no"]


def b_hadron(rows, final, no_copy: int, pid: int):
    """The hadron of the string that holds the parton copy no_copy with the flavour of the hard quark pid."""
    cands = [r for r in final if r["m1"] <= no_copy <= max(r["m1"], r["m2"])]
    want = (lambda h: (h < 0) if abs(h) < 1000 else (h > 0)) if pid > 0 else (lambda h: (h > 0) if abs(h) < 1000 else (h < 0))
    bh = [r for r in cands if heavy_content(r["id"]) == 5 and want(r["id"])]
    if not bh:
        return None
    k0 = kinematics(rows[no_copy])
    return min(bh, key=lambda r: math.hypot(kinematics(r)["eta"] - k0["eta"], dphi(kinematics(r)["phi"], k0["phi"])))


def classify(rows):
    final = [r for r in rows.values() if r["status"] > 0]
    hard = {r["id"]: r for r in rows.values() if r["status"] == -23 and abs(r["id"]) == 5}
    quarks = {}
    for r in final:
        r["star"], r["cat"] = None, "hadron"
    for pid, r in hard.items():
        copy = last_parton_copy(rows, r["no"])
        quark = next((rows[n] for n in (copy, rows[copy]["m1"]) if n in rows and abs(rows[n]["status"]) == 62), rows[copy])
        h = b_hadron(rows, final, copy, pid)
        if h is not None:
            h["star"] = pid
            h["cat"] = "b" if pid > 0 else "bbar"
            quarks[pid] = quark
    beams = [r for r in rows.values() if r["status"] == -12]
    n_strings = len({(r["m1"], r["m2"]) for r in final})
    return final, beams, quarks, n_strings


def summary(final, beams, quarks, n_strings):
    stars = {r["star"]: r for r in final if r["star"] is not None}
    cats = {k: [r for r in final if r["cat"] == k] for k in FAMILY}
    out = dict(stars=stars, cats=cats, n=len(final), n_strings=n_strings, quarks=quarks)
    if 5 in stars and -5 in stars:
        bb = vsum([stars[5], stars[-5]])
        out["bb"] = dict(**kinematics(bb), m=mass(bb), **bb,
                         scalar=kinematics(stars[5])["pt"] + kinematics(stars[-5])["pt"],
                         dphi=dphi(kinematics(stars[5])["phi"], kinematics(stars[-5])["phi"]))
    out["all"] = vsum(final)
    out["x"] = {}
    for pid, h in stars.items():
        q = quarks.get(pid)
        if q is not None:
            out["x"][pid] = dict(x_e=h["e"] / q["e"], x_pt=kinematics(h)["pt"] / kinematics(q)["pt"], pt_quark=kinematics(q)["pt"],
                                 e_quark=q["e"])
    return out


def print_table(final, s):
    print(f"{'no':>4s} {'hadron':12s} {'status':>6s} {'pT [GeV]':>9s} {'eta':>7s} {'phi':>7s} {'E [GeV]':>9s}")
    names = {r["no"]: r for r in final}
    for r in sorted(final, key=lambda r: -kinematics(r)["pt"])[:40]:
        k = kinematics(r)
        tag = " *" if r["star"] is not None else ""
        print(f"{r['no']:4d} {str(r['id']) + tag:12s} {r['status']:6d} {k['pt']:9.2f} {k['eta']:7.2f} {k['phi']:7.2f} {r['e']:9.2f}")
    if len(final) > 40:
        print(f"... and {len(final) - 40} more (sorted by pT)")
    for pid, x in s["x"].items():
        h = s["stars"][pid]
        print(f"{'b' if pid > 0 else 'bbar'} hadron (row {h['no']}, id {h['id']}): pT = {kinematics(h)['pt']:.2f} GeV from a quark of {x['pt_quark']:.2f}: "
              f"x_E = {x['x_e']:.3f}, x_pT = {x['x_pt']:.3f}")
    if "bb" in s:
        bb = s["bb"]
        print(f"b + bbar hadrons: |pT(b) + pT(bbar)| = {bb['pt']:.2f} GeV, m = {bb['m']:.1f} GeV, dphi = {bb['dphi']:.2f}")
    a = s["all"]
    print(f"all {s['n']} primary hadrons from {s['n_strings']} strings: |sum pT| = {math.hypot(a['px'], a['py']):.3f} GeV, "
          f"sum pz = {a['pz']:.1f}, sum E = {a['e']:.1f} GeV")


def draw_particles(ax, final, s, label_min: float, names: dict) -> int:
    n_border = 0
    for r in sorted(final, key=lambda r: -kinematics(r)["pt"]):
        k = kinematics(r)
        e, f, border = ps.place(k)
        style = FAMILY[r["cat"]]
        col = style["color"]
        if border:
            n_border += 1
            ax.scatter(e, f, s=area(k["pt"]), marker="x", color=col, linewidth=1.0, zorder=3)
            continue
        if r["star"] is not None:
            ax.scatter(e, f, s=area(k["pt"]), marker=style["marker"], color=col, zorder=4)
            ax.annotate(f"{hadron_tex(names.get(r['no'], str(r['id'])))} {k['pt']:.1f}", (e, f), xytext=(4, 4),
                        textcoords="offset points", fontsize=9, color=col, zorder=6)
        else:
            ax.scatter(e, f, s=area(k["pt"]), marker=style["marker"], color=col, alpha=0.85, linewidth=0, zorder=3)
            if k["pt"] >= label_min:
                ax.annotate(f"{k['pt']:.1f}", (e, f), xytext=(3, 3), textcoords="offset points", fontsize=6.5, color=col, zorder=5)
    if "bb" in s and s["bb"]["pt"] > 0.05:
        bb = s["bb"]
        e, f, _ = ps.place(bb)
        ax.scatter(e, f, s=area(bb["pt"]) * 2.2, marker=SUM["marker"], color=SUM["color"], zorder=5)
        ax.annotate(rf"$b$+$\bar b$ {bb['pt']:.1f}", (e, f), xytext=(4, -10), textcoords="offset points",
                    fontsize=8.5, color=SUM["color"], zorder=6)
    return n_border


def draw_legend(ax, counts: dict, stage: int, n_strings: int):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y = 0.98
    ax.text(0.0, y, "particle origin (number drawn)", fontsize=8.5, color=TEXT, va="top")
    entries = [(FAMILY["b"], counts.get("b", 0)), (FAMILY["bbar"], counts.get("bbar", 0)), (SUM, None),
               (FAMILY["hadron"], counts.get("hadron", 0))]
    for style, n in entries:
        if n == 0 and style is not SUM:
            continue
        y -= 0.055
        ax.scatter(0.08, y, s=area(15.0) * (1.8 if style is SUM else 1.0), marker=style["marker"], color=style["color"], linewidth=0)
        ax.text(0.2, y, style["label"] + (f" ({n})" if n is not None else ""), fontsize=8.5, color=TEXT, va="center")
    y -= 0.06
    ax.text(0.0, y, f"{n_strings} colour-singlet strings", fontsize=8, color=TEXT2, va="center")
    y -= 0.09
    ax.text(0.0, y, "marker size:", fontsize=8.5, color=TEXT, va="center", weight="bold")
    for pt in (1, 10, 100):
        y -= 0.055
        ax.scatter(0.08, y, s=area(pt), marker="o", color="#666666")
        ax.text(0.2, y, f"$p_T$ = {pt} GeV", fontsize=8.5, color=TEXT, va="center")
    y -= 0.09
    ax.text(0.0, y, "stages:", fontsize=8.5, color=TEXT, va="center", weight="bold")
    for i, name in enumerate(ps.STAGES, start=1):
        y -= 0.045
        ax.text(0.05, y, name, fontsize=8, color="#ff0000" if i == stage else "#888888", va="center",
                weight="bold" if i == stage else "normal")


def header(fig, s, title: str, source: str, n_drawn: int, n_border: int, label_min: float, names: dict):
    x, y, dy = 0.06, 0.955, 0.029
    fig.text(x, y, title, fontsize=14, weight="bold", color=TEXT, va="center")
    fig.text(0.985, y, source, fontsize=9, color=TEXT, ha="right", va="center")
    y -= 0.038
    border = f", {n_border} with $|\\eta|$ > {ETA_MAX:g} drawn on the border (x)" if n_border else ""
    fig.text(x, y, f"{n_drawn} particles drawn (marker size ~ log $p_T$; label = $p_T$ [GeV] for $p_T$ > {label_min:g} GeV)"
             + border, fontsize=8.5, color=TEXT2, va="center")
    for pid, key in ((5, "b"), (-5, "bbar")):
        y -= dy
        if pid not in s["stars"]:
            continue
        r = s["stars"][pid]
        k = kinematics(r)
        xx = s["x"].get(pid)
        extra = (rf";   from a quark of $p_T$ = {xx['pt_quark']:.2f}: $x_E$ = $E_B/E_b$ = {xx['x_e']:.3f}, $x_{{p_T}}$ = {xx['x_pt']:.3f}"
                 if xx else "")
        fig.text(x, y, rf"{STAR[pid]['name']} hadron {hadron_tex(names.get(r['no'], str(r['id'])))}: $p_T$ = {k['pt']:.2f} GeV,   "
                    rf"$\eta$ = {k['eta']:.2f},   $\phi$ = {k['phi']:.2f}" + extra, fontsize=9.5, color=FAMILY[key]["color"],
                 va="center", weight="bold")
    if "bb" in s:
        bb = s["bb"]
        y -= dy
        fig.text(x, y, rf"net $b$+$\bar b$: $|\vec p_T(b) + \vec p_T(\bar b)|$ = {bb['pt']:.2f} GeV  "
                       rf"($p_x$ = {bb['px']:.2f}, $p_y$ = {bb['py']:.2f}),   $\Sigma|p_T|$ = {bb['scalar']:.2f} GeV,   "
                       rf"$\Delta\phi(b,\bar b)$ = {bb['dphi']:.2f}", fontsize=9.5, color=SUM["color"], va="center",
                 weight="bold")
    a = s["all"]
    y -= dy
    fig.text(x, y, rf"all {s['n']} primary hadrons from {s['n_strings']} strings: $|\Sigma\vec p_T|$ = {math.hypot(a['px'], a['py']):.2f} GeV  "
                   rf"(hadronisation creates no momentum: the sum must still vanish)", fontsize=9.5, color=TEXT, va="center",
             weight="bold")
    return y


def draw(final, s, output: Path, title: str, source: str, png: bool, label_min: float, names: dict):
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "mathtext.default": "regular"})
    fig = plt.figure(figsize=(10.0, 7.4))
    n_border = sum(1 for r in final if abs(kinematics(r)["eta"]) > ETA_MAX)
    y_header = header(fig, s, title, source, len(final), n_border, label_min, names)
    top = y_header - 0.05
    ax = fig.add_axes([0.06, 0.08, 0.70, top - 0.08])
    ps.draw_axes(ax)
    draw_particles(ax, final, s, label_min, names)
    side = fig.add_axes([0.79, 0.08, 0.20, top - 0.08])
    draw_legend(side, {k: len(v) for k, v in s["cats"].items()}, STAGE, s["n_strings"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    if png:
        fig.savefig(output.with_suffix(".png"), dpi=200)
    print(f"wrote {output}" + (f" and {output.with_suffix('.png')}" if png else ""))


def read_names(lines):
    """Row number -> name column of the listing (the hadron names, for the labels)."""
    names = {}
    for line in lines:
        f = line.split()
        if len(f) >= 15 and f[0].isdigit():
            names[int(f[0])] = f[2].strip("()")
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=HERE / "event_Had.lhe")
    ap.add_argument("-o", "--output", type=Path, default=HERE / "journey_Had.pdf")
    ap.add_argument("--title", default="4. + hadronisation (primary hadrons)")
    ap.add_argument("--source", default=None, help="text at the top right (default: the input file name)")
    ap.add_argument("--label-min", type=float, default=5.0)
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    lines = args.input.read_text().splitlines()
    rows = parse_listing(lines)
    names = read_names(lines)
    final, beams, quarks, n_strings = classify(rows)
    if not final:
        sys.exit(f"no final-state particles found in {args.input}")
    s = summary(final, beams, quarks, n_strings)
    print_table(final, s)
    source = args.source if args.source is not None else f"standalone hadronisation: {args.input.name}"
    draw(final, s, args.output, args.title, source, args.png, args.label_min, names)


if __name__ == "__main__":
    main()
