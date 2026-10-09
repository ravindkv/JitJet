#!/usr/bin/env python3
"""Draw the event after multiparton interactions, beam remnants and colour reconnection in
the (eta, phi) plane, in the layout of the CMSSW genPartAnalyzer (stage 3).

The input is the parton-level listing written by Standalone/standalone_multiparton_interactions.py
(PYTHIA listing format, saved as event_MPI.lhe). The picture follows
example/MPI/CMSSW/genPartAnalyzer_ME_PS_MPI_out.pdf, page 3, so that the standalone and
the CMSSW result can be compared side by side. It reuses the frame, the marker sizes and
the parsing of ../PS/plot_journey_PS.py and adds the two origins of this stage:

  * MPI (hollow cyan squares): the outgoing partons of the secondary scatterings and their
    copies, as in the CMSSW analyzer (a copy or an FSR radiator inherits the origin of its
    mother; a gluon emitted by an MPI parton is FSR, an ISR emission of a secondary system
    is ISR);
  * beam remnant (hollow grey diamonds): the status-63 partons that carry the rest of the
    two protons, mostly beyond |eta| = 6 and so drawn on the border.

The header gains lines with the number and the summed pT of the MPI partons and of the
remnants; the vector sum of all final-state partons must still vanish.

  python3 plot_journey_MPI.py                              # -> journey_MPI.pdf
  python3 plot_journey_MPI.py -i event_MPI.lhe -o journey_MPI.pdf
  python3 plot_journey_MPI.py --png

Dependency: matplotlib (any version from 3.3); on this Mac /usr/bin/python3.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "PS"))
import plot_journey_PS as ps  # noqa: E402  (../PS/plot_journey_PS.py)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAMILY = dict(ps.FAMILY)
FAMILY["MPI"] = dict(label="MPI", color="#009999", marker="s")
FAMILY["remnant"] = dict(label="beam remnant", color="#666666", marker="d")
SUM, STAR, TEXT, TEXT2, ETA_MAX = ps.SUM, ps.STAR, ps.TEXT, ps.TEXT2, ps.ETA_MAX
STAGE = 3
HOLLOW = ("^", "v", "s", "d")

parse_listing, kinematics, vsum, mass, dphi, area = ps.parse_listing, ps.kinematics, ps.vsum, ps.mass, ps.dphi, ps.area
parton_name, parton_tex, descendant = ps.parton_name, ps.parton_tex, ps.descendant


def origin_of(rows, no: int) -> str:
    """The process that created a final-state parton, walking copies and FSR radiators up the record.

    Copies (44, 52, 62, -53 ...) and the first daughter of an FSR branching inherit their
    mother's origin; the second daughter of an FSR branching is FSR; an ISR emission (43) is
    ISR; an MPI outgoing parton (33) is MPI; a hard outgoing parton (23) is the hard process;
    a beam remnant (63) is a remnant.
    """
    seen = set()
    while no in rows and no not in seen:
        seen.add(no)
        r = rows[no]
        st = abs(r["status"])
        if st == 63:
            return "remnant"
        if st == 33:
            return "MPI"
        if st == 23:
            return "b" if r["id"] == 5 else "bbar" if r["id"] == -5 else "hard"
        if st == 43:
            return "ISR"
        if st in (41, 53, 21, 31, 12, 11, 61):
            return "ISR"
        m = rows.get(r["m1"])
        if m is None:
            return "ISR"
        if st == 51 and not (no == m["d1"] and r["id"] == m["id"]):
            return "FSR"           # the emitted parton; the radiator (first daughter, same flavour) inherits
        no = r["m1"]
    return "ISR"


def classify(rows):
    final = [r for r in rows.values() if r["status"] > 0]
    hard = {r["id"]: r for r in rows.values() if r["status"] == -23 and abs(r["id"]) == 5}
    stars = {pid: descendant(rows, r["no"]) for pid, r in hard.items()}
    for r in final:
        r["star"] = next((pid for pid, no in stars.items() if no == r["no"]), None)
        o = origin_of(rows, r["no"])
        r["family"] = o
        r["cat"] = ("b" if r["star"] == 5 else "bbar" if r["star"] == -5
                    else o if o in ("ISR", "MPI", "remnant") else "FSR")
    beams = [r for r in rows.values() if r["status"] == -12]
    incoming0 = [r for r in rows.values() if r["status"] == -21]
    incoming = [r for r in rows.values() if r["status"] == -61 and r["m1"] in
                [q["no"] for q in rows.values() if q["status"] in (-41, -53, -21)]]
    hard_in = []
    for side in (1, -1):
        cands = [r for r in incoming if (r["pz"] > 0) == (side > 0)]
        if cands:
            hard_in.append(min(cands, key=lambda r: r["no"]))     # the first -61 copy belongs to the hard system
    return final, beams, incoming0, hard_in or incoming0


def summary(final, beams, incoming0, incoming):
    stars = {r["star"]: r for r in final if r["star"] is not None}
    cats = {k: [r for r in final if r["cat"] == k] for k in FAMILY}
    out = dict(stars=stars, cats=cats, n=len(final))
    if 5 in stars and -5 in stars:
        bb = vsum([stars[5], stars[-5]])
        out["bb"] = dict(**kinematics(bb), m=mass(bb), **bb,
                         scalar=kinematics(stars[5])["pt"] + kinematics(stars[-5])["pt"],
                         dphi=dphi(kinematics(stars[5])["phi"], kinematics(stars[-5])["phi"]))
    out["all"] = vsum(final)
    out["extra_b"] = [r for r in final if abs(r["id"]) == 5 and r["star"] is None]
    if beams:
        ebeam = beams[0]["e"]
        out["x0"] = [(r, abs(r["pz"]) / ebeam) for r in incoming0]
        out["x1"] = [(r, abs(r["pz"]) / ebeam) for r in incoming]
    out["n_mpi_systems"] = len({r["m1"] for r in final if r["cat"] == "MPI"})
    return out


def print_table(final, s):
    print(f"{'no':>4s} {'particle':9s} {'status':>6s} {'origin':8s} {'pT [GeV]':>9s} {'eta':>7s} {'phi':>7s} {'E [GeV]':>9s}")
    for r in sorted(final, key=lambda r: -kinematics(r)["pt"])[:40]:
        k = kinematics(r)
        tag = " *" if r["star"] is not None else ""
        print(f"{r['no']:4d} {parton_name(r['id']) + tag:9s} {r['status']:6d} {r['cat']:8s} {k['pt']:9.2f} "
              f"{k['eta']:7.2f} {k['phi']:7.2f} {r['e']:9.2f}")
    if len(final) > 40:
        print(f"... and {len(final) - 40} more (sorted by pT)")
    for k in ("ISR", "FSR", "MPI", "remnant"):
        parts = s["cats"][k]
        v = vsum(parts) if parts else dict(px=0.0, py=0.0)
        print(f"{k:8s}: n = {len(parts):3d}   |sum pT| = {math.hypot(v['px'], v['py']):7.2f} GeV   "
              f"scalar sum pT = {sum(kinematics(r)['pt'] for r in parts):7.2f} GeV")
    if "bb" in s:
        bb = s["bb"]
        print(f"b + bbar: |pT(b) + pT(bbar)| = {bb['pt']:.2f} GeV, m = {bb['m']:.1f} GeV, dphi = {bb['dphi']:.2f}")
    a = s["all"]
    print(f"all {s['n']} final-state partons: |sum pT| = {math.hypot(a['px'], a['py']):.3f} GeV, sum pz = {a['pz']:.1f}, "
          f"sum E = {a['e']:.1f} GeV")


def draw_particles(ax, final, s, label_min: float) -> int:
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
            ax.annotate(f"{STAR[r['star']]['name']} {k['pt']:.1f}", (e, f), xytext=(4, 4), textcoords="offset points",
                        fontsize=9, color=col, zorder=6)
        else:
            ax.scatter(e, f, s=area(k["pt"]), marker=style["marker"], facecolor="none", edgecolor=col,
                       linewidth=1.0, zorder=3)
            if k["pt"] >= label_min:
                txt = (parton_tex(r["id"]) + " " if r["id"] != 21 else "") + f"{k['pt']:.1f}"
                ax.annotate(txt, (e, f), xytext=(3, 3), textcoords="offset points", fontsize=6.5, color=col, zorder=5)
    if "bb" in s and s["bb"]["pt"] > 0.05:
        bb = s["bb"]
        e, f, _ = ps.place(bb)
        ax.scatter(e, f, s=area(bb["pt"]) * 2.2, marker=SUM["marker"], color=SUM["color"], zorder=5)
        ax.annotate(rf"$b$+$\bar b$ {bb['pt']:.1f}", (e, f), xytext=(4, -10), textcoords="offset points",
                    fontsize=8.5, color=SUM["color"], zorder=6)
    return n_border


def draw_legend(ax, counts: dict, stage: int, n_systems: int):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y = 0.98
    ax.text(0.0, y, "particle origin (number drawn)", fontsize=8.5, color=TEXT, va="top")
    entries = [(FAMILY["b"], counts.get("b", 0)), (FAMILY["bbar"], counts.get("bbar", 0)), (SUM, None),
               (FAMILY["ISR"], counts.get("ISR", 0)), (FAMILY["FSR"], counts.get("FSR", 0)),
               (FAMILY["MPI"], counts.get("MPI", 0)), (FAMILY["remnant"], counts.get("remnant", 0))]
    for style, n in entries:
        if n == 0 and style is not SUM:
            continue
        y -= 0.055
        hollow = style["marker"] in HOLLOW
        ax.scatter(0.08, y, s=area(15.0) * (1.8 if style is SUM else 1.0), marker=style["marker"],
                   facecolor="none" if hollow else style["color"], edgecolor=style["color"], linewidth=1.0)
        ax.text(0.2, y, style["label"] + (f" ({n})" if n is not None else ""), fontsize=8.5, color=TEXT, va="center")
    y -= 0.06
    ax.text(0.0, y, f"{n_systems} secondary scatterings", fontsize=8, color=TEXT2, va="center")
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


def header(fig, s, title: str, source: str, n_drawn: int, n_border: int, label_min: float):
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
        fig.text(x, y, rf"{STAR[pid]['name']} quark (status {r['status']}): $p_T$ = {k['pt']:.2f} GeV,   "
                    rf"$\eta$ = {k['eta']:.2f},   $\phi$ = {k['phi']:.2f}", fontsize=9.5, color=FAMILY[key]["color"],
                 va="center", weight="bold")
    if "bb" in s:
        bb = s["bb"]
        y -= dy
        fig.text(x, y, rf"net $b$+$\bar b$: $|\vec p_T(b) + \vec p_T(\bar b)|$ = {bb['pt']:.2f} GeV  "
                       rf"($p_x$ = {bb['px']:.2f}, $p_y$ = {bb['py']:.2f}),   $\Sigma|p_T|$ = {bb['scalar']:.2f} GeV,   "
                       rf"$\Delta\phi(b,\bar b)$ = {bb['dphi']:.2f}", fontsize=9.5, color=SUM["color"], va="center",
                 weight="bold")
    for key in ("ISR", "FSR", "MPI", "remnant"):
        parts = s["cats"][key]
        if not parts:
            continue
        y -= dy
        v = vsum(parts)
        scalar = sum(kinematics(r)["pt"] for r in parts)
        fig.text(x, y, rf"{FAMILY[key]['label']} ({len(parts)} partons): $|\Sigma\vec p_T|$ = {math.hypot(v['px'], v['py']):.2f} GeV,  "
                       rf"$\Sigma|p_T|$ = {scalar:.2f} GeV", fontsize=9.5, color=FAMILY[key]["color"], va="center", weight="bold")
    a = s["all"]
    y -= dy
    fig.text(x, y, rf"all {s['n']} final-state partons: $|\Sigma\vec p_T|$ = {math.hypot(a['px'], a['py']):.2f} GeV  "
                   rf"(the two protons carry no $p_T$, so the sum must vanish)", fontsize=9.5, color=TEXT, va="center",
             weight="bold")
    return y


def draw(final, s, output: Path, title: str, source: str, png: bool, label_min: float):
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "mathtext.default": "regular"})
    fig = plt.figure(figsize=(10.0, 7.4))
    n_border = sum(1 for r in final if abs(kinematics(r)["eta"]) > ETA_MAX)
    y_header = header(fig, s, title, source, len(final), n_border, label_min)
    top = y_header - 0.05
    ax = fig.add_axes([0.06, 0.08, 0.70, top - 0.08])
    ps.draw_axes(ax)
    draw_particles(ax, final, s, label_min)
    ps.draw_incoming(ax, s)
    side = fig.add_axes([0.79, 0.08, 0.20, top - 0.08])
    draw_legend(side, {k: len(v) for k, v in s["cats"].items()}, STAGE, s["n_mpi_systems"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    if png:
        fig.savefig(output.with_suffix(".png"), dpi=200)
    print(f"wrote {output}" + (f" and {output.with_suffix('.png')}" if png else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=HERE / "event_MPI.lhe")
    ap.add_argument("-o", "--output", type=Path, default=HERE / "journey_MPI.pdf")
    ap.add_argument("--title", default="3. + MPI / beam remnants")
    ap.add_argument("--source", default=None, help="text at the top right (default: the input file name)")
    ap.add_argument("--label-min", type=float, default=5.0)
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    rows = parse_listing(args.input.read_text().splitlines())
    final, beams, incoming0, incoming = classify(rows)
    if not final:
        sys.exit(f"no final-state particles found in {args.input}")
    s = summary(final, beams, incoming0, incoming)
    print_table(final, s)
    source = args.source if args.source is not None else f"standalone MPI: {args.input.name}"
    draw(final, s, args.output, args.title, source, args.png, args.label_min)


if __name__ == "__main__":
    main()
