#!/usr/bin/env python3
"""Draw the showered event in the (eta, phi) plane, in the layout of the CMSSW genPartAnalyzer.

The input is the parton-level listing written by standalone_parton_showering.py
(PYTHIA listing format, saved as event_PS.lhe). The picture follows
example/PS/CMSSW/genPartAnalyzer_02_PS.pdf, the page of the CMSSW analyser,
so that the standalone and the CMSSW result can be compared one above the
other on the same page of the book:

  * header lines with the kinematics of the b (red) and bbar (blue) quark and
    of their vector sum (magenta), then the net pT of the ISR partons (orange),
    of the FSR partons (green) and of the whole final state (black);
  * every final-state parton at its (eta, phi), marker size growing with
    log pT, labelled with its pT when it is above 5 GeV; partons beyond
    |eta| = 6 are drawn on the border with a cross;
  * a legend column with the particle origin, the marker-size scale and the
    list of generator stages with the current one highlighted;
  * the cone of the energy ledger, Delta R < 0.4 around the last copy of each
    hard quark, as a dashed circle with its vector pT and the number of
    partons inside, and an inset that magnifies the cone of the b with the
    name and pT of its members (cone_of, draw_cones, draw_zoom; the journey
    scripts of the MPI, hadronisation and decay stages reuse them).

"b quark" and "bbar quark" are the final-state copies of the two hard quarks,
found by following the same-flavour daughter down the record (CMSSW's status
52 copies). Every other final-state parton is classified by walking its
mother chain back: to the hard b or bbar (status -23) it is FSR, to an ISR
emission (status 43) it is ISR. The faint bands are the CMS detector regions
(tracker + barrel/endcap calorimeters, endcap, HF) that the chapter refers to.

  python3 plot_journey_PS.py                             # -> journey_PS.pdf
  python3 plot_journey_PS.py -i event_PS.lhe -o journey_PS.pdf
  python3 plot_journey_PS.py --png                       # also write a .png
  python3 plot_journey_PS.py --label-min 10              # label partons above 10 GeV

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
from matplotlib.patches import Ellipse  # noqa: E402

# --- appearance (colours and markers of the CMSSW genPartAnalyzer figures) -----
FAMILY = {
    "b": dict(label=r"$b$ quark", color="#ff0000", marker="o"),
    "bbar": dict(label=r"$\bar b$ quark", color="#0000ff", marker="o"),
    "ISR": dict(label="ISR", color="#ff7f00", marker="^"),
    "FSR": dict(label="FSR", color="#009900", marker="v"),
}
SUM = dict(label=r"$b$ + $\bar b$ (vector sum)", color="#ff00ff", marker="*")
STAR = {5: dict(name=r"$b$", marker="o"), -5: dict(name=r"$\bar b$", marker="o")}
TEXT, TEXT2 = "#0b0b0b", "#52514e"
REGIONS = [
    (r"tracker + calorimeters ($|\eta|<2.5$)", 0.0, 2.5, "#f7f6f2"),
    ("endcap", 2.5, 3.0, "#eeece6"),
    ("HF", 3.0, 5.2, "#e4e2da"),
]
ETA_MAX = 6.0
CONE_R = 0.4  # the cone of the energy ledger: what a jet of radius 0.4 around the quark would collect
STAGES = ["1 hard process", "2 ISR + FSR", "3 MPI / remnants", "4 hadronisation", "5 hadron decays"]
STAGE = 2

PARTON_NAMES = {1: "d", 2: "u", 3: "s", 4: "c", 5: "b", 6: "t", 21: "g"}


def parton_name(pid: int) -> str:
    base = PARTON_NAMES.get(abs(pid), str(pid))
    return base + "bar" if pid < 0 and abs(pid) != 21 else base


def parton_tex(pid: int) -> str:
    base = PARTON_NAMES.get(abs(pid), str(pid))
    return rf"$\bar {base}$" if pid < 0 and abs(pid) != 21 else f"${base}$"


def radius(pt: float) -> float:
    """Marker radius in points, growing with log10 pT: 1, 10, 100 GeV -> 2.6, 5.2, 7.8 pt."""
    return 2.6 * (1.0 + math.log10(max(pt, 0.3)))


def area(pt: float) -> float:
    """Marker area (points^2) for matplotlib's scatter."""
    return math.pi * radius(pt) ** 2


# --- parsing ------------------------------------------------------------------
def parse_listing(lines):
    """Rows of a PYTHIA-style listing, keyed by their number.

    Columns: no id name status m1 m2 d1 d2 col acol px py pz e m. The name
    column contains no blanks, so the row can be split on whitespace.
    """
    rows = {}
    for line in lines:
        f = line.split()
        if len(f) < 15 or not f[0].isdigit():
            continue
        try:
            no, pid, status = int(f[0]), int(f[1]), int(f[3])
            m1, m2, d1, d2, col, acol = (int(v) for v in f[4:10])
            px, py, pz, e, m = (float(v) for v in f[-5:])
        except ValueError:
            continue
        rows[no] = dict(no=no, id=pid, status=status, m1=m1, m2=m2, d1=d1, d2=d2,
                        col=col, acol=acol, px=px, py=py, pz=pz, e=e, m=m)
    if not rows:
        raise ValueError("no event-listing rows found")
    return rows


def family_of(rows, no: int) -> str:
    """Walk mother1 until a hard outgoing parton (-23) or an ISR emission (43) is met."""
    seen = set()
    while no in rows and no not in seen:
        seen.add(no)
        r = rows[no]
        if abs(r["status"]) == 23:
            return "b" if r["id"] == 5 else "bbar" if r["id"] == -5 else "hard"
        if abs(r["status"]) in (41, 43, 53, 21, 12, 11):
            return "ISR"
        no = r["m1"]
    return "ISR"


def descendant(rows, no: int) -> int:
    """Follow the same-flavour daughter from a hard parton to its final-state copy."""
    r = rows[no]
    while r["status"] < 0:
        d1, d2 = r["d1"], r["d2"]
        cands = [rows[d] for d in range(d1, (d2 if d2 >= d1 else d1) + 1) if d in rows]
        same = [c for c in cands if c["id"] == r["id"]] or cands
        if not same:
            break
        r = same[0]
    return r["no"]


def current_incoming(rows, r):
    """The last copy of the incoming parton that entered the hard process as row r (status -21).

    Its x grows twice over: by backward evolution (new mothers with status -41, the other side copied
    with -42) and whenever a final-state emission recoils against it (a copy with status -53). PYTHIA
    makes every new copy the mother of the old one; the standalone record does so for -41 but appends
    a -53 copy as the single daughter of the parton it replaces. Both links are followed."""
    seen = {r["no"]}
    while True:
        up = rows.get(r["m1"])
        down = rows.get(r["d1"]) if r["d1"] == r["d2"] else None
        if up is not None and up["no"] not in seen and up["status"] in (-41, -42, -53):
            r = up
        elif down is not None and down["no"] not in seen and down["status"] == -53:
            r = down
        else:
            return r
        seen.add(r["no"])


def classify(rows):
    """Final-state rows with a 'family' (b, bbar, ISR) and a 'cat' (b, bbar, ISR, FSR)."""
    final = [r for r in rows.values() if r["status"] > 0]
    hard = {r["id"]: r for r in rows.values() if r["status"] == -23 and abs(r["id"]) == 5}
    stars = {pid: descendant(rows, r["no"]) for pid, r in hard.items()}
    for r in final:
        r["family"] = family_of(rows, r["no"])
        r["star"] = next((pid for pid, no in stars.items() if no == r["no"]), None)
        r["cat"] = ("b" if r["star"] == 5 else "bbar" if r["star"] == -5
                    else "ISR" if r["family"] == "ISR" else "FSR")
    beams = [r for r in rows.values() if r["status"] == -12]
    incoming0 = [r for r in rows.values() if r["status"] == -21]
    incoming = [current_incoming(rows, r) for r in incoming0]
    return final, beams, incoming0, incoming


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


# --- the cone of the ledger (shared by the journey scripts of the later stages) -----
def wrap(d: float) -> float:
    """An azimuthal difference brought to (-pi, pi]."""
    return (d + math.pi) % (2 * math.pi) - math.pi


def cone_of(final, axis, R: float = CONE_R, keep=None):
    """The final-state rows within Delta R < R of the row `axis` (in eta, phi) and their vector sum.

    This is the "family" row of the energy ledger and the dashed circle of the CMSSW genPartAnalyzer pages
    (its option coneR). `keep` can drop rows from the sum, e.g. the neutrinos."""
    ka = kinematics(axis)
    members = []
    for r in final:
        if keep is not None and not keep(r):
            continue
        k = kinematics(r)
        if math.hypot(k["eta"] - ka["eta"], wrap(k["phi"] - ka["phi"])) < R:
            members.append(r)
    v = vsum(members) if members else dict(px=0.0, py=0.0, pz=0.0, e=0.0)
    return dict(eta=ka["eta"], phi=ka["phi"], R=R, members=members, n=len(members),
                pt=math.hypot(v["px"], v["py"]), m=mass(v) if members else 0.0, e=v["e"])


def print_cones(cones: dict):
    for key, c in cones.items():
        print(f"cone dR < {c['R']:g} around the {key} axis (eta = {c['eta']:.3f}, phi = {c['phi']:.3f}): n = {c['n']}, "
              f"|sum pT| = {c['pt']:.2f} GeV, m = {c['m']:.1f} GeV (from the rounded listing), E = {c['e']:.2f} GeV")


def draw_cones(ax, cones: dict):
    """A dashed circle Delta R < R around each axis with the vector pT and the number of particles inside."""
    for key, c in cones.items():
        col, R = FAMILY[key]["color"], c["R"]
        for shift in (0.0, 2 * math.pi, -2 * math.pi):          # a circle that crosses phi = +-pi continues on the other side
            ax.add_patch(Ellipse((c["eta"], c["phi"] + shift), 2 * R, 2 * R, fill=False, ls=(0, (4, 2)), lw=1.5,
                                 edgecolor=col, zorder=7, clip_on=True))
        below = c["phi"] - R > -math.pi + 0.30
        ax.annotate(f"cone {R:g}: {c['pt']:.1f} GeV (n = {c['n']})", (c["eta"], c["phi"] - R if below else c["phi"] + R),
                    xytext=(0, -3 if below else 3), textcoords="offset points", ha="center", va="top" if below else "bottom",
                    fontsize=7, color=col, zorder=8, annotation_clip=False)


def legend_cone(side, y: float = -0.085):
    """The cone entry of the legend column, below the list of stages."""
    side.plot([0.02, 0.14], [y, y], ls=(0, (4, 2)), lw=1.5, color=TEXT, clip_on=False)
    side.text(0.2, y, rf"cone $\Delta R$ < {CONE_R:g} (the ledger)", fontsize=8.5, color=TEXT, va="center", clip_on=False)


def zoom_rects(ax, fig, final, cones: dict, n: int = 1, height: float = 0.42):
    """Where to put n insets: the corners of the frame that cover the fewest particles and no cone."""
    R = max(c["R"] for c in cones.values())
    box = ax.get_position()
    w_in, h_in = box.width * fig.get_figwidth(), box.height * fig.get_figheight()
    width = height * h_in * (ZOOM_X[1] - ZOOM_X[0]) / (ZOOM_Y[1] - ZOOM_Y[0]) / w_in
    pad_x, pad_y = 0.012, 0.018
    spots = [(1 - width - pad_x, 1 - height - pad_y), (1 - width - pad_x, pad_y), (pad_x, 1 - height - pad_y), (pad_x, pad_y)]
    pts = [place(kinematics(r))[:2] for r in final]
    scored = []
    for order, (x0, y0) in enumerate(spots):
        e0, e1 = -ETA_MAX + 2 * ETA_MAX * x0, -ETA_MAX + 2 * ETA_MAX * (x0 + width)
        f0, f1 = -math.pi + 2 * math.pi * y0, -math.pi + 2 * math.pi * (y0 + height)
        hits = sum(1 for e, f in pts if e0 < e < e1 and f0 < f < f1) + (2 if y0 < 0.5 else 0)   # the bottom corners hold the notes
        for c in cones.values():
            if e0 - R < c["eta"] < e1 + R and f0 - R - 0.25 < c["phi"] < f1 + R + 0.25:
                hits += 10 ** 6
        scored.append((hits, order, [x0, y0, width, height]))
    return [rect for _, _, rect in sorted(scored)[:n]]


ZOOM_X = (-1.55, 2.55)   # the inset frame in units of R around the cone axis: tick labels, the cone, a column for the labels
ZOOM_Y = (-1.45, 1.55)   # ... tick labels below and a strip for the title above


def draw_zoom(ax, final, cone, rect, color, title: str, style_of, label_of, n_label: int = 10):
    """The cone once more, magnified, with the name and pT of its hardest members (as on the CMSSW pages).

    style_of(row) -> dict(marker, color, filled); label_of(row) -> the text of its label."""
    R, e0, f0 = cone["R"], cone["eta"], cone["phi"]
    axin = ax.inset_axes(rect, zorder=9)
    axin.set_facecolor("white")
    axin.set_xlim(e0 + ZOOM_X[0] * R, e0 + ZOOM_X[1] * R)
    axin.set_ylim(f0 + ZOOM_Y[0] * R, f0 + ZOOM_Y[1] * R)
    for spine in axin.spines.values():
        spine.set_color(color)
        spine.set_linewidth(1.8)
    axin.tick_params(axis="x", labelsize=5.8, direction="in", length=2.5, colors=TEXT, pad=-9)      # tick labels inside the frame
    axin.tick_params(axis="y", labelsize=5.8, direction="in", length=2.5, colors=TEXT, pad=-4)
    step = 0.2 if R <= 0.5 else 0.4
    axin.set_xticks([x for x in (round((e0 - 1.2 * R) / step) * step + i * step for i in range(8)) if abs(x - e0) <= 1.05 * R])
    axin.set_yticks([y for y in (round((f0 - 1.2 * R) / step) * step + i * step for i in range(8)) if abs(y - f0) <= 1.05 * R])
    for lab in axin.get_yticklabels():
        lab.set_horizontalalignment("left")
    axin.grid(color=TEXT, lw=0.3, ls=":")
    axin.set_axisbelow(True)
    axin.add_patch(Ellipse((e0, f0), 2 * R, 2 * R, fill=False, ls=(0, (4, 2)), lw=1.5, edgecolor=color, zorder=2))
    axin.text(0.02, 0.975, title, transform=axin.transAxes, fontsize=6.6, color=color, va="top", ha="left", weight="bold")
    inside = {id(r) for r in cone["members"]}
    shown = []
    for r in sorted(final, key=lambda r: -kinematics(r)["pt"]):
        k = kinematics(r)
        de, df = k["eta"] - e0, wrap(k["phi"] - f0)
        if abs(de) > 1.22 * R or abs(df) > 1.22 * R:
            continue
        st = style_of(r)
        member = id(r) in inside
        axin.scatter(e0 + de, f0 + df, s=area(k["pt"]) * 1.6, marker=st["marker"],
                     facecolor=st["color"] if st.get("filled", True) else "none", edgecolor=st["color"],
                     linewidth=0.9, alpha=0.95 if member else 0.35, zorder=4 if member else 3)
        if member:
            shown.append((f0 + df, e0 + de, r, st))
    labelled = sorted(sorted(shown, key=lambda t: -kinematics(t[2])["pt"])[:n_label], key=lambda t: -t[0])
    if labelled:
        gap = 0.21 * R
        top, bottom = f0 + 1.12 * R, f0 - 1.12 * R
        ys = []
        for f, _, _, _ in labelled:
            ys.append(min(f, ys[-1] - gap) if ys else min(f, top))
        if ys[-1] < bottom:                                   # push the column up if it ran out of the frame
            shift = bottom - ys[-1]
            ys = [y + shift for y in ys]
            for i in range(len(ys)):
                ys[i] = min(ys[i], top - i * gap)
        x_lab = e0 + 1.33 * R
        for y, (f, e, r, st) in zip(ys, labelled):
            axin.plot([e, x_lab - 0.03 * R], [f, y], lw=0.4, color="#777777", zorder=3)
            axin.text(x_lab, y, label_of(r), fontsize=6.2, color=st["color"], va="center", ha="left", zorder=5)
    if len(shown) > len(labelled):
        axin.text(0.98, 0.02, f"+{len(shown) - len(labelled)} more", transform=axin.transAxes, fontsize=5.8, color=TEXT2,
                  ha="right", va="bottom")
    return axin


def summary(final, beams, incoming0, incoming):
    """Numbers shown in the header and printed to the terminal."""
    stars = {r["star"]: r for r in final if r["star"] is not None}
    cats = {k: [r for r in final if r["cat"] == k] for k in FAMILY}
    out = dict(stars=stars, cats=cats, n=len(final))
    if 5 in stars and -5 in stars:
        bb = vsum([stars[5], stars[-5]])
        out["bb"] = dict(**kinematics(bb), m=mass(bb), **bb,
                         scalar=kinematics(stars[5])["pt"] + kinematics(stars[-5])["pt"],
                         dphi=dphi(kinematics(stars[5])["phi"], kinematics(stars[-5])["phi"]))
    out["all"] = vsum(final)
    out["cones"] = {key: cone_of(final, stars[pid]) for pid, key in ((5, "b"), (-5, "bbar")) if pid in stars}
    out["extra_b"] = [r for r in final if abs(r["id"]) == 5 and r["star"] is None]
    if beams:
        ebeam = beams[0]["e"]
        out["x0"] = [(r, abs(r["pz"]) / ebeam) for r in incoming0]
        out["x1"] = [(r, abs(r["pz"]) / ebeam) for r in incoming]
    return out


def print_table(final, s):
    print(f"{'no':>4s} {'particle':9s} {'status':>6s} {'origin':7s} {'pT [GeV]':>9s} {'eta':>7s} {'y':>7s} "
          f"{'phi':>7s} {'E [GeV]':>9s}")
    for r in sorted(final, key=lambda r: -kinematics(r)["pt"]):
        k = kinematics(r)
        tag = " *" if r["star"] is not None else ""
        print(f"{r['no']:4d} {parton_name(r['id']) + tag:9s} {r['status']:6d} {r['cat']:7s} {k['pt']:9.2f} "
              f"{k['eta']:7.2f} {k['y']:7.2f} {k['phi']:7.2f} {r['e']:9.2f}")
    print("(* = final-state copy of the hard b / bbar)")
    for k in ("ISR", "FSR"):
        v = vsum(s["cats"][k]) if s["cats"][k] else dict(px=0.0, py=0.0)
        print(f"{k} partons: n = {len(s['cats'][k]):2d}   sum pT = ({v['px']:+8.2f}, {v['py']:+8.2f}) GeV  "
              f"|sum pT| = {math.hypot(v['px'], v['py']):.2f}")
    if "bb" in s:
        bb = s["bb"]
        print(f"b + bbar: |pT(b) + pT(bbar)| = {bb['pt']:.2f} GeV, phi = {bb['phi']:+.2f}, y = {bb['y']:+.2f}, "
              f"m = {bb['m']:.1f} GeV, sum |pT| = {bb['scalar']:.2f} GeV, dphi(b, bbar) = {bb['dphi']:.2f}")
    print_cones(s["cones"])
    a = s["all"]
    print(f"all {s['n']} final-state partons: |sum pT| = {math.hypot(a['px'], a['py']):.3f} GeV, "
          f"sum pz = {a['pz']:.1f} GeV, sum E = {a['e']:.1f} GeV")
    for r in s["extra_b"]:
        k = kinematics(r)
        print(f"extra {parton_name(r['id'])} from g -> b bbar in the shower: no {r['no']}, "
              f"pT = {k['pt']:.2f} GeV, eta = {k['eta']:.2f}")
    if "x0" in s:
        for (r0, x0), (r1, x1) in zip(sorted(s["x0"], key=lambda t: -t[0]["pz"]),
                                      sorted(s["x1"], key=lambda t: -t[0]["pz"])):
            print(f"incoming {parton_name(r0['id']):4s} x = {x0:.4e}  -> after ISR "
                  f"{parton_name(r1['id']):4s} x = {x1:.4e}")


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


def draw_particles(ax, final, s, label_min: float) -> int:
    """All final-state partons, biggest first; returns the number drawn on the border."""
    n_border = 0
    for r in sorted(final, key=lambda r: -kinematics(r)["pt"]):
        k = kinematics(r)
        e, f, border = place(k)
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
        e, f, _ = place(bb)
        ax.scatter(e, f, s=area(bb["pt"]) * 2.2, marker=SUM["marker"], color=SUM["color"], zorder=5)
        ax.annotate(rf"$b$+$\bar b$ {bb['pt']:.1f}", (e, f), xytext=(4, -10), textcoords="offset points",
                    fontsize=8.5, color=SUM["color"], zorder=6)
    return n_border


def draw_incoming(ax, s):
    """Incoming partons before and after ISR, noted at the two eta edges."""
    if "x0" not in s:
        return
    for (r0, x0), (r1, x1) in zip(sorted(s["x0"], key=lambda t: -t[0]["pz"]),
                                  sorted(s["x1"], key=lambda t: -t[0]["pz"])):
        side = 1 if r0["pz"] > 0 else -1
        txt = f"incoming {parton_name(r0['id'])}, $x$ = {x0:.3g}"
        if r1 is not r0:
            txt += f"\nafter ISR: {parton_name(r1['id'])}, $x$ = {x1:.3g}"
        ax.annotate(txt, (side * ETA_MAX, -math.pi + 0.3), xytext=(-side * 0.6, 0), textcoords="offset fontsize",
                    ha="right" if side > 0 else "left", va="center", fontsize=6.5, color=TEXT2,
                    arrowprops=dict(arrowstyle="-|>", color=TEXT2, lw=0.7), zorder=4)


def draw_legend(ax, counts: dict, stage: int):
    """The legend column: particle origin, marker-size scale, generator stages."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y = 0.98
    ax.text(0.0, y, "particle origin (number drawn)", fontsize=8.5, color=TEXT, va="top")
    entries = [(FAMILY["b"], counts.get("b", 0)), (FAMILY["bbar"], counts.get("bbar", 0)), (SUM, None),
               (FAMILY["ISR"], counts.get("ISR", 0)), (FAMILY["FSR"], counts.get("FSR", 0))]
    for style, n in entries:
        if n == 0 and style is not SUM:
            continue
        y -= 0.065
        hollow = style["marker"] in ("^", "v")
        ax.scatter(0.08, y, s=area(15.0) * (1.8 if style is SUM else 1.0), marker=style["marker"],
                   facecolor="none" if hollow else style["color"], edgecolor=style["color"], linewidth=1.0)
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


def header(fig, s, title: str, source: str, n_drawn: int, n_border: int, label_min: float):
    """The CMSSW-style header: title, drawing rule, b, bbar, vector sum, ISR, FSR and total."""
    x, y, dy = 0.06, 0.955, 0.031
    fig.text(x, y, title, fontsize=14, weight="bold", color=TEXT, va="center")
    fig.text(0.985, y, source, fontsize=9, color=TEXT, ha="right", va="center")
    y -= 0.04
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
    for key in ("ISR", "FSR"):
        parts = s["cats"][key]
        if not parts:
            continue
        y -= dy
        v = vsum(parts)
        extra = ""
        if key == "FSR" and s["extra_b"]:
            extra = rf" (incl. a second $b\bar b$ pair from $g \to b\bar b$)"
        fig.text(x, y, rf"{key} ({len(parts)} partons{extra}): $|\Sigma\vec p_T|$ = {math.hypot(v['px'], v['py']):.2f} GeV  "
                       rf"($p_x$ = {v['px']:.2f}, $p_y$ = {v['py']:.2f})", fontsize=9.5, color=FAMILY[key]["color"],
                 va="center", weight="bold")
    a = s["all"]
    y -= dy
    fig.text(x, y, rf"all {s['n']} final-state partons: $|\Sigma\vec p_T|$ = {math.hypot(a['px'], a['py']):.2f} GeV  "
                   rf"(the incoming partons carry no $p_T$, so the sum must vanish)", fontsize=9.5, color=TEXT,
             va="center", weight="bold")
    return y


def parton_style(r):
    """How draw_particles draws a parton, for the inset of the cone."""
    return dict(marker=FAMILY[r["cat"]]["marker"], color=FAMILY[r["cat"]]["color"], filled=r["star"] is not None)


def parton_label(r):
    return f"{parton_tex(r['id'])} {kinematics(r)['pt']:.1f}"


def cone_title(name: str, c) -> str:
    return rf"{name} cone $\Delta R$ < {c['R']:g}: n = {c['n']}, $p_T$ = {c['pt']:.2f} GeV"


def draw_cone_layer(ax, fig, final, s, style_of, label_of, zoom=("b",), names=None):
    """The dashed cones of the ledger and the inset(s) that magnify them; used by every journey script from here on."""
    cones = s.get("cones", {})
    if not cones:
        return
    draw_cones(ax, cones)
    keys = [k for k in zoom if k in cones]
    names = names or {"b": r"$b$", "bbar": r"$\bar b$"}
    for key, rect in zip(keys, zoom_rects(ax, fig, final, cones, n=len(keys))):
        draw_zoom(ax, final, cones[key], rect, FAMILY[key]["color"], cone_title(names[key], cones[key]), style_of, label_of)


def draw(final, s, output: Path, title: str, source: str, png: bool, label_min: float):
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "mathtext.default": "regular"})
    fig = plt.figure(figsize=(10.0, 7.0))
    y_header = header(fig, s, title, source, len(final), sum(1 for r in final if abs(kinematics(r)["eta"]) > ETA_MAX),
                      label_min)
    top = y_header - 0.05
    ax = fig.add_axes([0.06, 0.08, 0.70, top - 0.08])
    draw_axes(ax)
    draw_particles(ax, final, s, label_min)
    draw_incoming(ax, s)
    draw_cone_layer(ax, fig, final, s, parton_style, parton_label)
    side = fig.add_axes([0.79, 0.08, 0.20, top - 0.08])
    draw_legend(side, {k: len(v) for k, v in s["cats"].items()}, STAGE)
    legend_cone(side)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    if png:
        fig.savefig(output.with_suffix(".png"), dpi=200)
    print(f"wrote {output}" + (f" and {output.with_suffix('.png')}" if png else ""))


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=here / "event_PS.lhe")
    ap.add_argument("-o", "--output", type=Path, default=here / "journey_PS.pdf")
    ap.add_argument("--title", default="2. + parton shower (ISR + FSR)")
    ap.add_argument("--source", default=None, help="text at the top right (default: the input file name)")
    ap.add_argument("--label-min", type=float, default=5.0,
                    help="write the pT next to every ISR/FSR parton above this value in GeV")
    ap.add_argument("--png", action="store_true", help="also write a PNG next to the PDF")
    args = ap.parse_args()
    rows = parse_listing(args.input.read_text().splitlines())
    final, beams, incoming0, incoming = classify(rows)
    if not final:
        sys.exit(f"no final-state particles found in {args.input}")
    s = summary(final, beams, incoming0, incoming)
    print_table(final, s)
    source = args.source if args.source is not None else f"standalone shower: {args.input.name}"
    draw(final, s, args.output, args.title, source, args.png, args.label_min)


if __name__ == "__main__":
    main()
