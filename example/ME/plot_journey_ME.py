#!/usr/bin/env python3
"""Draw the b and bbar of the hard process in the (eta, phi) plane.

The input is the hard-process record of the event followed through the book:
either PYTHIA's "Event Listing (hard process)" table as saved in event_ME.lhe,
or a genuine Les Houches <event> block. Final-state particles (PYTHIA status
21..29 outgoing, or LHE status 1) are drawn at their (eta, phi) with a marker
whose area grows with pT; the pT value is written next to each marker. CMS
detector regions (tracker, endcap calorimeters, forward calorimeter) are shaded
in the background so the reader can see immediately which part of the detector
the mother quark is heading for. The incoming partons and their momentum
fractions x are noted at the two eta edges.

  python3 plot_journey_ME.py                             # -> journey_ME.pdf
  python3 plot_journey_ME.py -i event_ME.lhe -o journey_ME.pdf
  python3 plot_journey_ME.py --png                       # also write a .png

Dependency: matplotlib (any version from 3.x). Tested with /usr/bin/python3
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
from matplotlib.lines import Line2D  # noqa: E402

# --- appearance --------------------------------------------------------------
# Categorical colours in fixed order (b = slot 1, bbar = slot 2); the marker
# shape is a second, colour-independent encoding of the same identity.
SERIES = {
    5: dict(name=r"$b$", color="#2a78d6", marker="o"),
    -5: dict(name=r"$\bar b$", color="#eb6834", marker="s"),
}
OTHER = dict(name="other", color="#52514e", marker="D")
TEXT, TEXT2 = "#0b0b0b", "#52514e"
# CMS regions in |eta|: (label, eta_min, eta_max, grey shade). Barrel/endcap
# boundaries are rounded; the tracker edge is the one that matters for b-tagging.
REGIONS = [
    (r"tracker + barrel/endcap calorimeters ($|\eta|<2.5$)", 0.0, 2.5, "#f4f3ef"),
    ("endcap calo.", 2.5, 3.0, "#e6e4de"),
    ("HF", 3.0, 5.2, "#d6d3ca"),
]
ETA_MAX = 5.2

PARTON_NAMES = {1: "d", 2: "u", 3: "s", 4: "c", 5: "b", 6: "t", 21: "g"}


def parton_name(pid: int) -> str:
    base = PARTON_NAMES.get(abs(pid), str(pid))
    if pid < 0 and abs(pid) != 21:
        return base + "bar"
    return base


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


def print_table(beams, incoming, outgoing):
    ebeam = beams[0]["e"] if beams else None
    print(f"{'particle':10s} {'pT [GeV]':>10s} {'eta':>8s} {'y':>8s} {'phi':>8s} {'E [GeV]':>10s} {'m [GeV]':>8s}")
    for p in outgoing:
        k = kinematics(p)
        print(f"{parton_name(p['id']):10s} {k['pt']:10.3f} {k['eta']:8.3f} {k['y']:8.3f} "
              f"{k['phi']:8.3f} {p['e']:10.3f} {p['m']:8.3f}")
    if ebeam:
        for p in incoming:
            x = abs(p["pz"]) / ebeam
            print(f"incoming {parton_name(p['id']):6s} x = {x:.4e}  (pz = {p['pz']:+.3f} GeV)")
        if len(incoming) == 2:
            x1, x2 = (abs(p["pz"]) / ebeam for p in incoming)
            shat = x1 * x2 * (2 * ebeam) ** 2
            print(f"sHat = x1 x2 s = {shat:.1f} GeV^2, sqrt(sHat) = {math.sqrt(shat):.2f} GeV, "
                  f"Y = 0.5 ln(x1/x2) = {0.5 * math.log(x1 / x2):+.3f}")


# --- drawing ------------------------------------------------------------------
def draw(beams, incoming, outgoing, output: Path, title: str, png: bool):
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.titlecolor": TEXT,
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    # detector regions as a backdrop, mirrored in eta
    for label, lo, hi, shade in REGIONS:
        for sgn in (-1, 1):
            ax.axvspan(sgn * lo, sgn * hi, color=shade, lw=0, zorder=0)
        ax.text(0.5 * (lo + hi) if lo > 0 else 0.0, math.pi + 0.12, label, ha="center",
                va="bottom", fontsize=7, color=TEXT2)
        if lo > 0:
            ax.text(-0.5 * (lo + hi), math.pi + 0.12, label, ha="center", va="bottom",
                    fontsize=7, color=TEXT2)
    for edge in (2.5, 3.0):
        for sgn in (-1, 1):
            ax.axvline(sgn * edge, color="#ffffff", lw=1.0, zorder=1)
    ax.axhline(0.0, color=TEXT2, lw=0.5, ls=":", zorder=1)

    # outgoing partons: marker area ~ pT, direct label with the pT value
    pts = [kinematics(p)["pt"] for p in outgoing] or [1.0]
    ptmax = max(pts)
    handles = []
    for p in outgoing:
        k = kinematics(p)
        style = SERIES.get(p["id"], OTHER)
        size = 60 + 260 * k["pt"] / ptmax
        ax.scatter(k["eta"], k["phi"], s=size, marker=style["marker"], color=style["color"],
                   edgecolor="#ffffff", linewidth=1.5, zorder=3)
        dx = 0.18 if k["eta"] < 0 else -0.18
        ax.annotate(f"{style['name']}\n$p_T$ = {k['pt']:.1f} GeV\n"
                    rf"$\eta$ = {k['eta']:.2f}, $\phi$ = {k['phi']:.2f}",
                    (k["eta"], k["phi"]), xytext=(dx, 0), textcoords="offset fontsize",
                    ha="left" if k["eta"] < 0 else "right", va="center", fontsize=8,
                    color=TEXT, zorder=4)
        handles.append(Line2D([], [], marker=style["marker"], color=style["color"], ls="",
                              markersize=7, label=f"{style['name']} ({parton_name(p['id'])})"))

    # incoming partons at the eta edges
    if beams and len(incoming) == 2:
        ebeam = beams[0]["e"]
        for p in incoming:
            x = abs(p["pz"]) / ebeam
            side = 1 if p["pz"] > 0 else -1
            ax.annotate(f"incoming {parton_name(p['id'])}\n$x$ = {x:.3g}",
                        (side * ETA_MAX, -math.pi + 0.35), xytext=(-side * 0.6, 0),
                        textcoords="offset fontsize", ha="right" if side > 0 else "left",
                        va="center", fontsize=7.5, color=TEXT2,
                        arrowprops=dict(arrowstyle="-|>", color=TEXT2, lw=0.8), zorder=4)

    ax.set_xlim(-ETA_MAX, ETA_MAX)
    ax.set_ylim(-math.pi, math.pi)
    ax.set_yticks([-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.set_xlabel(r"pseudorapidity $\eta$")
    ax.set_ylabel(r"azimuth $\phi$")
    ax.set_title(title, loc="left", fontsize=10, pad=22)
    ax.grid(color="#ffffff", lw=0.6, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
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
    ap.add_argument("--title", default=r"Our event at the ancestral home: $q\bar q \to b\bar b$ "
                                        r"at $\sqrt{s}$ = 13.6 TeV (hard process)")
    ap.add_argument("--png", action="store_true", help="also write a PNG next to the PDF")
    args = ap.parse_args()
    beams, incoming, outgoing = load(args.input)
    if not outgoing:
        sys.exit(f"no final-state particles found in {args.input}")
    print_table(beams, incoming, outgoing)
    draw(beams, incoming, outgoing, args.output, args.title, args.png)


if __name__ == "__main__":
    main()
