#!/usr/bin/env python3
"""Draw the b and bbar of the hard process in the (eta, phi) plane, in the layout of the CMSSW genPartAnalyzer.

The input is the hard-process record of the event followed through the book:
either PYTHIA's "Event Listing (hard process)" table as saved in event_ME.lhe,
or a genuine Les Houches <event> block. The picture follows
example/PS/CMSSW/genPartAnalyzer_stage1_hardProcess_run1_event1.pdf, and
example/PS/plot_journey_PS.py draws the showered event the same way, so the
stages of the journey can be compared at a glance:

  * header lines with the kinematics of the b (red) and bbar (blue) quark and
    of their vector sum (magenta): net pT, scalar sum of pT and Delta phi;
  * the final-state partons at their (eta, phi), marker size growing with
    log pT and labelled with the pT in GeV;
  * a legend column with the particle origin, the marker-size scale and the
    list of generator stages with the current one highlighted.

The faint bands are the CMS detector regions (tracker + barrel/endcap
calorimeters, endcap, HF) that the chapter refers to, and the incoming
partons with their momentum fractions x are noted at the two eta edges.

  python3 plot_journey_ME.py                             # -> journey_ME.pdf
  python3 plot_journey_ME.py -i event_ME.lhe -o journey_ME.pdf
  python3 plot_journey_ME.py --png                       # also write a .png

Dependency: matplotlib (any version from 3.3). Tested with /usr/bin/python3
(3.9, matplotlib 3.8) on macOS.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- appearance (colours and markers of the CMSSW genPartAnalyzer figures) -----
SERIES = {
    5: dict(name=r"$b$", label=r"$b$ quark", color="#ff0000", marker="o"),
    -5: dict(name=r"$\bar b$", label=r"$\bar b$ quark", color="#0000ff", marker="o"),
}
OTHER = dict(name="other", label="other", color="#666666", marker="D")
SUM = dict(label=r"$b$ + $\bar b$ (vector sum)", color="#ff00ff", marker="*")
TEXT, TEXT2 = "#0b0b0b", "#52514e"
REGIONS = [
    (r"tracker + calorimeters ($|\eta|<2.5$)", 0.0, 2.5, "#f7f6f2"),
    ("endcap", 2.5, 3.0, "#eeece6"),
    ("HF", 3.0, 5.2, "#e4e2da"),
]
ETA_MAX = 6.0
STAGES = ["1 hard process", "2 ISR + FSR", "3 MPI / remnants", "4 hadronisation", "5 hadron decays"]
STAGE = 1

PARTON_NAMES = {1: "d", 2: "u", 3: "s", 4: "c", 5: "b", 6: "t", 21: "g"}


def parton_name(pid: int) -> str:
    base = PARTON_NAMES.get(abs(pid), str(pid))
    return base + "bar" if pid < 0 and abs(pid) != 21 else base


def radius(pt: float) -> float:
    """Marker radius in points, growing with log10 pT: 1, 10, 100 GeV -> 2.6, 5.2, 7.8 pt."""
    return 2.6 * (1.0 + math.log10(max(pt, 0.3)))


def area(pt: float) -> float:
    """Marker area (points^2) for matplotlib's scatter."""
    return math.pi * radius(pt) ** 2


# --- parsing ------------------------------------------------------------------
def parse_pythia_listing(lines):
    """Rows of PYTHIA's event listing: no id name status m1 m2 d1 d2 col acol px py pz e m."""
    beams, incoming, outgoing = [], [], []
    for line in lines:
        f = line.split()
        if len(f) < 14 or not f[0].isdigit():
            continue
        try:
            pid, status = int(f[1]), int(f[3])
            px, py, pz, e, m = (float(v) for v in f[-5:])
        except ValueError:
            continue
        rec = dict(id=pid, status=status, px=px, py=py, pz=pz, e=e, m=m)
        if status == -12:
            beams.append(rec)
        elif status == -21:
            incoming.append(rec)
        elif status > 0:
            outgoing.append(rec)
    return beams, incoming, outgoing


def parse_lhe(lines):
    """First <event> block of a Les Houches file. Beams come from the <init> block."""
    beams, incoming, outgoing = [], [], []
    text = "\n".join(lines)
    if "<init>" in text:
        init = text.split("<init>")[1].split("</init>")[0].strip().splitlines()
        f = init[0].split()
        for pid, e in ((int(f[0]), float(f[2])), (int(f[1]), float(f[3]))):
            sign = 1.0 if not beams else -1.0
            beams.append(dict(id=pid, status=-12, px=0.0, py=0.0, pz=sign * e, e=e, m=0.0))
    if "<event>" not in text:
        raise ValueError("no <event> block found")
    body = text.split("<event>")[1].split("</event>")[0].strip().splitlines()
    nup = int(body[0].split()[0])
    for line in body[1:1 + nup]:
        f = line.split()
        pid, status = int(f[0]), int(f[1])
        px, py, pz, e, m = (float(v) for v in f[6:11])
        rec = dict(id=pid, status=status, px=px, py=py, pz=pz, e=e, m=m)
        (incoming if status == -1 else outgoing if status == 1 else []).append(rec)
    return beams, incoming, outgoing


def load(path: Path):
    lines = path.read_text().splitlines()
    if any("<event>" in ln for ln in lines):
        return parse_lhe(lines)
    return parse_pythia_listing(lines)


# --- kinematics ---------------------------------------------------------------
def kinematics(p):
    pt = math.hypot(p["px"], p["py"])
    eta = math.asinh(p["pz"] / pt) if pt > 0 else math.copysign(math.inf, p["pz"])
    y = 0.5 * math.log((p["e"] + p["pz"]) / (p["e"] - p["pz"])) if p["e"] > abs(p["pz"]) else eta
    phi = math.atan2(p["py"], p["px"])
    return dict(pt=pt, eta=eta, y=y, phi=phi)


def vsum(parts):
    return dict(px=sum(p["px"] for p in parts), py=sum(p["py"] for p in parts),
                pz=sum(p["pz"] for p in parts), e=sum(p["e"] for p in parts))


def mass(v):
    m2 = v["e"] ** 2 - v["px"] ** 2 - v["py"] ** 2 - v["pz"] ** 2
    return math.copysign(math.sqrt(abs(m2)), m2)


def dphi(a: float, b: float) -> float:
    d = abs(a - b) % (2 * math.pi)
    return 2 * math.pi - d if d > math.pi else d


def summary(beams, incoming, outgoing):
    """Numbers shown in the header and printed to the terminal."""
    stars = {p["id"]: p for p in outgoing if p["id"] in SERIES}
    out = dict(stars=stars, n=len(outgoing))
    if 5 in stars and -5 in stars:
        bb = vsum([stars[5], stars[-5]])
        out["bb"] = dict(**kinematics(bb), m=mass(bb), **bb,
                         scalar=kinematics(stars[5])["pt"] + kinematics(stars[-5])["pt"],
                         dphi=dphi(kinematics(stars[5])["phi"], kinematics(stars[-5])["phi"]))
    if beams and len(incoming) == 2:
        ebeam = beams[0]["e"]
        out["x"] = [(p, abs(p["pz"]) / ebeam) for p in incoming]
        x1, x2 = (x for _, x in out["x"])
        out["shat"] = x1 * x2 * (2 * ebeam) ** 2
        out["Y"] = 0.5 * math.log(x1 / x2) if incoming[0]["pz"] > 0 else 0.5 * math.log(x2 / x1)
    return out


def print_table(outgoing, s):
    print(f"{'particle':10s} {'status':>6s} {'pT [GeV]':>10s} {'eta':>8s} {'y':>8s} {'phi':>8s} {'E [GeV]':>10s} {'m [GeV]':>8s}")
    for p in outgoing:
        k = kinematics(p)
        print(f"{parton_name(p['id']):10s} {p['status']:6d} {k['pt']:10.3f} {k['eta']:8.3f} {k['y']:8.3f} "
              f"{k['phi']:8.3f} {p['e']:10.3f} {p['m']:8.3f}")
    if "bb" in s:
        bb = s["bb"]
        print(f"b + bbar: |pT(b) + pT(bbar)| = {bb['pt']:.2f} GeV, m = {bb['m']:.1f} GeV, "
              f"sum |pT| = {bb['scalar']:.2f} GeV, dphi(b, bbar) = {bb['dphi']:.2f}")
    if "x" in s:
        for p, x in s["x"]:
            print(f"incoming {parton_name(p['id']):6s} x = {x:.4e}  (pz = {p['pz']:+.3f} GeV)")
        print(f"sHat = x1 x2 s = {s['shat']:.1f} GeV^2, sqrt(sHat) = {math.sqrt(s['shat']):.2f} GeV, "
              f"Y = 0.5 ln(x1/x2) = {s['Y']:+.3f}")


# --- drawing ------------------------------------------------------------------
def draw_axes(ax):
    """The (eta, phi) frame: CMSSW-style dotted grid, eta from -6 to 6, faint detector bands."""
    for label, lo, hi, shade in REGIONS:
        for sgn in (-1, 1):
            ax.axvspan(sgn * lo, sgn * hi, color=shade, lw=0, zorder=0)
        for x in ({0.0} if lo == 0 else {0.5 * (lo + hi), -0.5 * (lo + hi)}):
            ax.text(x, math.pi + 0.06, label, ha="center", va="bottom", fontsize=6.5, color=TEXT2)
    ax.set_xlim(-ETA_MAX, ETA_MAX)
    ax.set_ylim(-math.pi, math.pi)
    ax.set_xticks(range(-6, 7, 2))
    ax.set_yticks(range(-3, 4))
    ax.set_xlabel(r"$\eta$", loc="right", fontsize=12)
    ax.set_ylabel(r"$\phi$", loc="top", fontsize=12, rotation=0, labelpad=8)
    ax.grid(color=TEXT, lw=0.5, ls=":", zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT, labelsize=10, direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_color(TEXT)


def place(k):
    """(eta, phi) on the canvas and whether the parton had to be moved to the border."""
    if abs(k["eta"]) > ETA_MAX:
        return math.copysign(ETA_MAX - 0.05, k["eta"]), k["phi"], True
    return k["eta"], k["phi"], False


def draw_particles(ax, outgoing, s) -> int:
    """The outgoing partons, biggest first; returns the number drawn on the border."""
    n_border = 0
    for p in sorted(outgoing, key=lambda p: -kinematics(p)["pt"]):
        k = kinematics(p)
        e, f, border = place(k)
        style = SERIES.get(p["id"], OTHER)
        if border:
            n_border += 1
            ax.scatter(e, f, s=area(k["pt"]), marker="x", color=style["color"], linewidth=1.0, zorder=3)
            continue
        ax.scatter(e, f, s=area(k["pt"]), marker=style["marker"], color=style["color"], zorder=4)
        ax.annotate(f"{style['name']} {k['pt']:.1f}", (e, f), xytext=(4, 4), textcoords="offset points",
                    fontsize=9, color=style["color"], zorder=6)
    if "bb" in s and s["bb"]["pt"] > 0.05:
        bb = s["bb"]
        e, f, _ = place(bb)
        ax.scatter(e, f, s=area(bb["pt"]) * 2.2, marker=SUM["marker"], color=SUM["color"], zorder=5)
        ax.annotate(rf"$b$+$\bar b$ {bb['pt']:.1f}", (e, f), xytext=(4, -10), textcoords="offset points",
                    fontsize=8.5, color=SUM["color"], zorder=6)
    return n_border


def draw_incoming(ax, s):
    """Incoming partons and their momentum fractions, noted at the two eta edges."""
    for p, x in s.get("x", []):
        side = 1 if p["pz"] > 0 else -1
        ax.annotate(f"incoming {parton_name(p['id'])}\n$x$ = {x:.3g}", (side * ETA_MAX, -math.pi + 0.3),
                    xytext=(-side * 0.6, 0), textcoords="offset fontsize", ha="right" if side > 0 else "left",
                    va="center", fontsize=6.5, color=TEXT2, arrowprops=dict(arrowstyle="-|>", color=TEXT2, lw=0.7),
                    zorder=4)


def draw_legend(ax, counts: dict, stage: int):
    """The legend column: particle origin, marker-size scale, generator stages."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y = 0.98
    ax.text(0.0, y, "particle origin (number drawn)", fontsize=8.5, color=TEXT, va="top")
    entries = [(SERIES[5], counts.get(5, 0)), (SERIES[-5], counts.get(-5, 0)), (SUM, None),
               (OTHER, counts.get("other", 0))]
    for style, n in entries:
        if n == 0 and style is not SUM:
            continue
        y -= 0.065
        ax.scatter(0.08, y, s=area(15.0) * (1.8 if style is SUM else 1.0), marker=style["marker"], color=style["color"])
        ax.text(0.2, y, style["label"] + (f" ({n})" if n is not None else ""), fontsize=8.5, color=TEXT, va="center")
    y -= 0.11
    ax.text(0.0, y, "marker size:", fontsize=8.5, color=TEXT, va="center", weight="bold")
    for pt in (1, 10, 100):
        y -= 0.06
        ax.scatter(0.08, y, s=area(pt), marker="o", color="#666666")
        ax.text(0.2, y, f"$p_T$ = {pt} GeV", fontsize=8.5, color=TEXT, va="center")
    y -= 0.11
    ax.text(0.0, y, "stages:", fontsize=8.5, color=TEXT, va="center", weight="bold")
    for i, name in enumerate(STAGES, start=1):
        y -= 0.05
        ax.text(0.05, y, name, fontsize=8, color="#ff0000" if i == stage else "#888888", va="center",
                weight="bold" if i == stage else "normal")


def header(fig, s, title: str, source: str, n_drawn: int, n_border: int):
    """The CMSSW-style header: title, drawing rule, b, bbar and their vector sum."""
    x, y, dy = 0.06, 0.955, 0.031
    fig.text(x, y, title, fontsize=14, weight="bold", color=TEXT, va="center")
    fig.text(0.985, y, source, fontsize=9, color=TEXT, ha="right", va="center")
    y -= 0.04
    border = f", {n_border} with $|\\eta|$ > {ETA_MAX:g} drawn on the border (x)" if n_border else ""
    fig.text(x, y, f"{n_drawn} particles drawn (marker size ~ log $p_T$; label = $p_T$ [GeV])" + border,
             fontsize=8.5, color=TEXT2, va="center")
    for pid in (5, -5):
        y -= dy
        if pid not in s["stars"]:
            continue
        p = s["stars"][pid]
        k = kinematics(p)
        fig.text(x, y, rf"{SERIES[pid]['name']} quark (status {p['status']}): $p_T$ = {k['pt']:.2f} GeV,   "
                    rf"$\eta$ = {k['eta']:.2f},   $\phi$ = {k['phi']:.2f}", fontsize=9.5, color=SERIES[pid]["color"],
                 va="center", weight="bold")
    if "bb" in s:
        bb = s["bb"]
        y -= dy
        fig.text(x, y, rf"net $b$+$\bar b$: $|\vec p_T(b) + \vec p_T(\bar b)|$ = {bb['pt']:.2f} GeV  "
                       rf"($p_x$ = {bb['px']:.2f}, $p_y$ = {bb['py']:.2f}),   $\Sigma|p_T|$ = {bb['scalar']:.2f} GeV,   "
                       rf"$\Delta\phi(b,\bar b)$ = {bb['dphi']:.2f}", fontsize=9.5, color=SUM["color"], va="center",
                 weight="bold")
        y -= dy
        fig.text(x, y, rf"pair: $m_{{b\bar b}}$ = {bb['m']:.1f} GeV = $\sqrt{{\hat s}}$,   $Y$ = {bb['y']:.2f}"
                 + (rf"   ($x_1 x_2 s$ = {s['shat']:.0f} GeV$^2$)" if "shat" in s else ""),
                 fontsize=9.5, color=TEXT, va="center", weight="bold")
    return y


def draw(beams, incoming, outgoing, s, output: Path, title: str, source: str, png: bool):
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "mathtext.default": "regular"})
    fig = plt.figure(figsize=(10.0, 7.0))
    y_header = header(fig, s, title, source, len(outgoing),
                      sum(1 for p in outgoing if abs(kinematics(p)["eta"]) > ETA_MAX))
    top = y_header - 0.05
    ax = fig.add_axes([0.06, 0.08, 0.70, top - 0.08])
    draw_axes(ax)
    draw_particles(ax, outgoing, s)
    draw_incoming(ax, s)
    side = fig.add_axes([0.79, 0.08, 0.20, top - 0.08])
    counts = {5: sum(1 for p in outgoing if p["id"] == 5), -5: sum(1 for p in outgoing if p["id"] == -5)}
    counts["other"] = len(outgoing) - counts[5] - counts[-5]
    draw_legend(side, counts, STAGE)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    if png:
        fig.savefig(output.with_suffix(".png"), dpi=200)
    print(f"wrote {output}" + (f" and {output.with_suffix('.png')}" if png else ""))


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=here / "event_ME.lhe")
    ap.add_argument("-o", "--output", type=Path, default=here / "journey_ME.pdf")
    ap.add_argument("--title", default=r"1. hard process (ME): $q\bar q \to b\bar b$")
    ap.add_argument("--source", default=None, help="text at the top right (default: the input file name)")
    ap.add_argument("--png", action="store_true", help="also write a PNG next to the PDF")
    args = ap.parse_args()
    beams, incoming, outgoing = load(args.input)
    if not outgoing:
        sys.exit(f"no final-state particles found in {args.input}")
    s = summary(beams, incoming, outgoing)
    print_table(outgoing, s)
    source = args.source if args.source is not None else f"hard process: {args.input.name}"
    draw(beams, incoming, outgoing, s, args.output, args.title, source, args.png)


if __name__ == "__main__":
    main()
