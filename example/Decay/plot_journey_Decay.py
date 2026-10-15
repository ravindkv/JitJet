#!/usr/bin/env python3
"""Draw the event after the hadron decays in the (eta, phi) plane, in the layout of the CMSSW
genPartAnalyzer (stage 5, "hadron decays" = the genParticles with status 1).

The input is the listing written by Standalone/standalone_hadron_decays.py (PYTHIA listing
format plus a vertex table, saved as event_Decay.lhe): the final particles are the rows with
a positive status (the primary hadrons that are stable for the detector keep their status
81-84, the decay products carry 91 or 92). The picture follows
example/Decay/CMSSW/genPartAnalyzer_ME_PS_MPI_Had_Decay_out.pdf, page 5, so that the
standalone and the CMSSW result can be compared side by side. It reuses the frame, the
marker sizes and the parsing of ../PS/plot_journey_PS.py and ../Had/plot_journey_Had.py;
the categories of this stage are

  * the b system (red) and the bbar system (blue): every final particle that descends from
    the primary hadron of the hard b (bbar) quark's string, the B* of chapter 5; neutrinos
    among them are drawn hollow and labelled nu, since the detector will not see them;
  * their vector sum (magenta star), as at every stage;
  * the primary hadrons that do not decay (purple): pi+-, K+-, K_S0, K_L0, p, n, Lambda ...;
  * every other decay product (grey), labelled with its pT above 5 GeV.

The header keeps the CMSSW lines (the decay products of the two B hadrons with their count,
pT, eta, phi, the net b + bbar) and adds the weakly decaying B hadron of each system with its
flight distance and decay vertex, the neutrinos of the event (count, energy, missing pT),
and the vector sum of all final particles, which must still vanish.

  python3 plot_journey_Decay.py                              # -> journey_Decay.pdf
  python3 plot_journey_Decay.py -i event_Decay.lhe -o journey_Decay.pdf
  python3 plot_journey_Decay.py --png

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
import plot_journey_PS as ps  # noqa: E402
import plot_journey_Had as jh  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAMILY = {
    "b": dict(label=r"$b$ system", color="#ff0000", marker="o"),
    "bbar": dict(label=r"$\bar b$ system", color="#0000ff", marker="o"),
    "primary": dict(label="primary hadron", color="#9b4dff", marker="o"),
    "decay": dict(label="decay product", color="#999999", marker="o"),
}
SUM, STAR, TEXT, TEXT2, ETA_MAX = ps.SUM, ps.STAR, ps.TEXT, ps.TEXT2, ps.ETA_MAX
STAGE = 5
NEUTRINOS = (12, 14, 16)

parse_listing, kinematics, vsum, mass, dphi, area = ps.parse_listing, ps.kinematics, ps.vsum, ps.mass, ps.dphi, ps.area
hadron_tex, heavy_content, last_parton_copy = jh.hadron_tex, jh.heavy_content, jh.last_parton_copy


def read_vertices(lines):
    """Row -> (x, y, z, t, tau) of the vertex table at the end of the listing."""
    out = {}
    for line in lines:
        f = line.split()
        if len(f) == 6 and f[0].isdigit():
            try:
                out[int(f[0])] = tuple(float(v) for v in f[1:])
            except ValueError:
                pass
    return out


def descendants(rows, no: int):
    out, stack = [], [no]
    while stack:
        r = rows[stack.pop()]
        if r["d1"] > 0:
            for d in range(r["d1"], max(r["d1"], r["d2"]) + 1):
                if d in rows:
                    out.append(d)
                    stack.append(d)
    return out


def string_hadron(rows, no_copy: int, pid: int):
    """The primary hadron (status +-81..84) of the string that holds the parton copy, with the flavour of the hard quark."""
    cands = [r for r in rows.values() if abs(r["status"]) in (81, 82, 83, 84) and r["m1"] <= no_copy <= max(r["m1"], r["m2"])]
    want = (lambda h: (h < 0) if abs(h) < 1000 else (h > 0)) if pid > 0 else (lambda h: (h > 0) if abs(h) < 1000 else (h < 0))
    bh = [r for r in cands if heavy_content(r["id"]) == 5 and want(r["id"])]
    if not bh:
        return None
    k0 = kinematics(rows[no_copy])
    return min(bh, key=lambda r: math.hypot(kinematics(r)["eta"] - k0["eta"], dphi(kinematics(r)["phi"], k0["phi"])))


def weak_hadron(rows, no: int, vertices):
    """Down the b flavour from the string hadron to the first one with a lifetime (a B rather than a B*):
    the vertex table lists the proper lifetime drawn for every row that flew."""
    r = rows[no]
    while r["d1"] > 0 and vertices.get(r["no"], (0, 0, 0, 0, 0.0))[4] <= 1e-4:
        ds = [rows[d] for d in range(r["d1"], max(r["d1"], r["d2"]) + 1) if d in rows and heavy_content(rows[d]["id"]) == 5]
        if not ds:
            break
        r = ds[0]
    return r


def classify(rows, vertices):
    final = [r for r in rows.values() if r["status"] > 0]
    for r in final:
        r["cat"] = "primary" if r["status"] < 90 else "decay"
        r["nu"] = abs(r["id"]) in NEUTRINOS
    hard = {r["id"]: r for r in rows.values() if r["status"] == -23 and abs(r["id"]) == 5}
    systems = {}
    for pid, r in hard.items():
        copy = last_parton_copy(rows, r["no"])
        h = string_hadron(rows, copy, pid)
        if h is None:
            continue
        desc = [d for d in descendants(rows, h["no"]) if rows[d]["status"] > 0]
        key = "b" if pid > 0 else "bbar"
        for d in desc:
            rows[d]["cat"] = key
        systems[pid] = dict(string=h, weak=weak_hadron(rows, h["no"], vertices), members=[rows[d] for d in desc])
    n_strings = len({(r["m1"], r["m2"]) for r in rows.values() if abs(r["status"]) in (81, 82, 83, 84)})
    return final, systems, n_strings


def summary(final, systems, vertices, n_strings):
    cats = {k: [r for r in final if r["cat"] == k] for k in FAMILY}
    out = dict(cats=cats, n=len(final), n_strings=n_strings, systems={})
    for pid, sy in systems.items():
        v = vsum(sy["members"]) if sy["members"] else dict(px=0.0, py=0.0, pz=0.0, e=0.0)
        w = sy["weak"]
        vx = vertices.get(w["d1"], (0.0, 0.0, 0.0, 0.0, 0.0)) if w["d1"] > 0 else (0.0, 0.0, 0.0, 0.0, 0.0)
        nus = [r for r in sy["members"] if r["nu"]]
        out["systems"][pid] = dict(**kinematics(v), **v, m=mass(v), n=len(sy["members"]), string=sy["string"], weak=w,
                                   L=math.sqrt(vx[0] ** 2 + vx[1] ** 2 + vx[2] ** 2), Lxy=math.hypot(vx[0], vx[1]), vertex=vx[:3],
                                   tau=vertices.get(w["no"], (0, 0, 0, 0, 0.0))[4], n_nu=len(nus), e_nu=sum(r["e"] for r in nus),
                                   pt_nu=kinematics(vsum(nus))["pt"] if nus else 0.0)
    if 5 in out["systems"] and -5 in out["systems"]:
        a, b = out["systems"][5], out["systems"][-5]
        bb = dict(px=a["px"] + b["px"], py=a["py"] + b["py"], pz=a["pz"] + b["pz"], e=a["e"] + b["e"])
        out["bb"] = dict(**kinematics(bb), m=mass(bb), **bb, scalar=a["pt"] + b["pt"], dphi=dphi(a["phi"], b["phi"]))
    out["all"] = vsum(final)
    nus = [r for r in final if r["nu"]]
    out["nu"] = dict(n=len(nus), e=sum(r["e"] for r in nus), pt=kinematics(vsum(nus))["pt"] if nus else 0.0)
    return out


def print_table(final, s, names):
    print(f"{'no':>4s} {'particle':12s} {'status':>6s} {'origin':8s} {'pT [GeV]':>9s} {'eta':>7s} {'phi':>7s} {'E [GeV]':>9s}")
    for r in sorted(final, key=lambda r: -kinematics(r)["pt"])[:40]:
        k = kinematics(r)
        print(f"{r['no']:4d} {names.get(r['no'], str(r['id'])):12s} {r['status']:6d} {r['cat']:8s} {k['pt']:9.2f} {k['eta']:7.2f} {k['phi']:7.2f} {r['e']:9.2f}")
    if len(final) > 40:
        print(f"... and {len(final) - 40} more (sorted by pT)")
    for pid, sy in s["systems"].items():
        w = sy["weak"]
        print(f"{'b' if pid > 0 else 'bbar'} system: {sy['n']} final particles from {names.get(sy['string']['no'])} (row {sy['string']['no']}): pT = {sy['pt']:.2f} GeV, "
              f"eta = {sy['eta']:.2f}, phi = {sy['phi']:.2f}; the {names.get(w['no'])} (row {w['no']}) flew L = {sy['L']:.2f} mm (L_xy = {sy['Lxy']:.2f}) "
              f"to ({sy['vertex'][0]:.2f}, {sy['vertex'][1]:.2f}, {sy['vertex'][2]:.2f}) mm; {sy['n_nu']} neutrinos with {sy['e_nu']:.2f} GeV")
    if "bb" in s:
        bb = s["bb"]
        print(f"b + bbar systems: |pT(b) + pT(bbar)| = {bb['pt']:.2f} GeV, m = {bb['m']:.1f} GeV, dphi = {bb['dphi']:.2f}")
    a = s["all"]
    print(f"all {s['n']} final particles: |sum pT| = {math.hypot(a['px'], a['py']):.3f} GeV, sum pz = {a['pz']:.1f}, sum E = {a['e']:.1f} GeV; "
          f"{s['nu']['n']} neutrinos carry {s['nu']['e']:.1f} GeV, missing pT = {s['nu']['pt']:.2f} GeV")


def draw_particles(ax, final, s, label_min: float, names: dict) -> int:
    n_border = 0
    order = {"decay": 0, "primary": 1, "bbar": 2, "b": 3}
    for r in sorted(final, key=lambda r: (order[r["cat"]], -kinematics(r)["pt"])):
        k = kinematics(r)
        e, f, border = ps.place(k)
        style = FAMILY[r["cat"]]
        col = style["color"]
        if border:
            n_border += 1
            ax.scatter(e, f, s=area(k["pt"]), marker="x", color=col, linewidth=1.0, zorder=3)
            continue
        z = 3 + order[r["cat"]]
        if r["nu"]:
            ax.scatter(e, f, s=area(k["pt"]), marker="o", facecolors="none", edgecolors=col, linewidth=1.2, zorder=z + 1)
            ax.annotate(rf"$\nu$ {k['pt']:.1f}", (e, f), xytext=(3, 3), textcoords="offset points", fontsize=7, color=col, zorder=6)
            continue
        ax.scatter(e, f, s=area(k["pt"]), marker=style["marker"], color=col, alpha=0.9 if r["cat"] in ("b", "bbar") else 0.8, linewidth=0, zorder=z)
        if k["pt"] >= label_min:
            ax.annotate(f"{k['pt']:.1f}", (e, f), xytext=(3, 3), textcoords="offset points", fontsize=6.5, color=col, zorder=5)
    if "bb" in s and s["bb"]["pt"] > 0.05:
        bb = s["bb"]
        e, f, _ = ps.place(bb)
        ax.scatter(e, f, s=area(bb["pt"]) * 2.2, marker=SUM["marker"], color=SUM["color"], zorder=7)
        ax.annotate(rf"$b$+$\bar b$ {bb['pt']:.1f}", (e, f), xytext=(4, -10), textcoords="offset points", fontsize=8.5, color=SUM["color"], zorder=7)
    return n_border


def draw_legend(ax, counts: dict, stage: int, n_strings: int):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y = 0.98
    ax.text(0.0, y, "particle origin (number drawn)", fontsize=8.5, color=TEXT, va="top")
    entries = [(FAMILY["b"], counts.get("b", 0)), (FAMILY["bbar"], counts.get("bbar", 0)), (SUM, None),
               (FAMILY["primary"], counts.get("primary", 0)), (FAMILY["decay"], counts.get("decay", 0))]
    for style, n in entries:
        y -= 0.055
        ax.scatter(0.08, y, s=area(15.0) * (1.8 if style is SUM else 1.0), marker=style["marker"], color=style["color"], linewidth=0)
        ax.text(0.2, y, style["label"] + (f" ({n})" if n is not None else ""), fontsize=8.5, color=TEXT, va="center")
    y -= 0.05
    ax.scatter(0.08, y, s=area(15.0), marker="o", facecolors="none", edgecolors="#666666", linewidth=1.2)
    ax.text(0.2, y, r"neutrino (hollow, $\nu$)", fontsize=8.5, color=TEXT, va="center")
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
    x, y, dy = 0.06, 0.955, 0.027
    fig.text(x, y, title, fontsize=14, weight="bold", color=TEXT, va="center")
    fig.text(0.985, y, source, fontsize=9, color=TEXT, ha="right", va="center")
    y -= 0.036
    border = f", {n_border} with $|\\eta|$ > {ETA_MAX:g} drawn on the border (x)" if n_border else ""
    fig.text(x, y, f"{n_drawn} particles drawn (marker size ~ log $p_T$; label = $p_T$ [GeV] for $p_T$ > {label_min:g} GeV)" + border,
             fontsize=8.5, color=TEXT2, va="center")
    for pid, key in ((5, "b"), (-5, "bbar")):
        y -= dy
        if pid not in s["systems"]:
            continue
        sy = s["systems"][pid]
        w = sy["weak"]
        fig.text(x, y, rf"{STAR[pid]['name']} hadron {hadron_tex(names.get(sy['string']['no'], ''))} decay products (n={sy['n']}): $p_T$ = {sy['pt']:.2f} GeV,   "
                    rf"$\eta$ = {sy['eta']:.2f},   $\phi$ = {sy['phi']:.2f};   the {hadron_tex(names.get(w['no'], ''))} flew $L$ = {sy['L']:.1f} mm "
                    rf"($L_{{xy}}$ = {sy['Lxy']:.1f} mm)" + (rf", {sy['n_nu']} $\nu$ with {sy['e_nu']:.1f} GeV" if sy["n_nu"] else ""),
                 fontsize=9, color=FAMILY[key]["color"], va="center", weight="bold")
    if "bb" in s:
        bb = s["bb"]
        y -= dy
        fig.text(x, y, rf"net $b$+$\bar b$: $|\vec p_T(b) + \vec p_T(\bar b)|$ = {bb['pt']:.2f} GeV  "
                       rf"($p_x$ = {bb['px']:.2f}, $p_y$ = {bb['py']:.2f}),   $\Sigma|p_T|$ = {bb['scalar']:.2f} GeV,   "
                       rf"$\Delta\phi(b,\bar b)$ = {bb['dphi']:.2f}", fontsize=9.5, color=SUM["color"], va="center", weight="bold")
    a = s["all"]
    y -= dy
    fig.text(x, y, rf"all {s['n']} final particles: $|\Sigma\vec p_T|$ = {math.hypot(a['px'], a['py']):.2f} GeV (decays create no momentum);   "
                   rf"{s['nu']['n']} neutrinos carry {s['nu']['e']:.1f} GeV, missing $p_T$ = {s['nu']['pt']:.2f} GeV",
             fontsize=9.5, color=TEXT, va="center", weight="bold")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=HERE / "event_Decay.lhe")
    ap.add_argument("-o", "--output", type=Path, default=HERE / "journey_Decay.pdf")
    ap.add_argument("--title", default="5. + hadron decays (what the detector will see)")
    ap.add_argument("--source", default=None, help="text at the top right (default: the input file name)")
    ap.add_argument("--label-min", type=float, default=5.0)
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    lines = args.input.read_text().splitlines()
    rows = parse_listing(lines)
    names = jh.read_names(lines)
    vertices = read_vertices(lines)
    final, systems, n_strings = classify(rows, vertices)
    if not final:
        sys.exit(f"no final-state particles found in {args.input}")
    s = summary(final, systems, vertices, n_strings)
    print_table(final, s, names)
    source = args.source if args.source is not None else f"standalone hadron decays: {args.input.name}"
    draw(final, s, args.output, args.title, source, args.png, args.label_min, names)


if __name__ == "__main__":
    main()
