#!/usr/bin/env python3
"""The CMS detector of Run 3 in three dimensions, with the quadrant that faces the reader cut away: the drawing of
the picture to carry of chapter 9 (chapter/tikz/g4_cms3d.tex) and of the last scene of its film.

The detector is a set of solids of revolution around the beam, each an annulus r1 < r < r2 between z1 and z2, with
the dimensions of the release's geometry where the chapter quotes them (cms.xml: the tracker volume r < 123.3 cm,
|z| < 293.2 cm; CALO up to r = 295 cm and |z| = 554.1 cm; the HF between |z| = 11.15 and 12.80 m, 12.5 to 159.5 cm
from the beam) and the round numbers of the CMS detector paper for the rest (the solenoid to 3.8 m, the iron
yoke to 7.4 m with its four muon stations, the endcap disks to |z| = 10.6 m). The quadrant x > 0, y > 0 is
removed, the faces turned towards the viewer are cut into small quads, shaded with a fixed light, sorted from
back to front (the painter's algorithm) and written as TikZ \\filldraw commands, or drawn with matplotlib.

  python3 draw_cms3d.py                        # -> ../../../chapter/tikz/g4_cms3d.tex
  python3 draw_cms3d.py --png cms3d.png        # a quick look with matplotlib

Lengths in metres. A schematic for the eye: the bands are the subdetectors, not their material.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# (name, r1, r2, z1, z2, rgb): z ranges are symmetric unless given twice; the order does not matter (sorted by depth)
SOLIDS = [
    ("tracker", 0.04, 1.233, -2.932, 2.932, (150, 190, 235)),
    ("ECAL barrel", 1.29, 1.77, -3.00, 3.00, (90, 175, 95)),
    ("ECAL endcap -", 0.32, 1.71, -3.55, -3.15, (90, 175, 95)),
    ("ECAL endcap +", 0.32, 1.71, 3.15, 3.55, (90, 175, 95)),
    ("HCAL barrel", 1.77, 2.95, -4.33, 4.33, (235, 160, 70)),
    ("HCAL endcap -", 0.40, 2.95, -5.54, -3.90, (235, 160, 70)),
    ("HCAL endcap +", 0.40, 2.95, 3.90, 5.54, (235, 160, 70)),
    ("solenoid", 2.95, 3.80, -6.30, 6.30, (175, 175, 180)),
    # the barrel yoke: iron (red) with the four muon stations (pale) between and inside its layers, iron outside
    ("MB1", 3.80, 4.20, -6.60, 6.60, (238, 232, 222)),
    ("yoke 1", 4.20, 4.80, -6.60, 6.60, (200, 45, 45)),
    ("MB2", 4.80, 5.20, -6.60, 6.60, (238, 232, 222)),
    ("yoke 2", 5.20, 5.80, -6.60, 6.60, (200, 45, 45)),
    ("MB3", 5.80, 6.20, -6.60, 6.60, (238, 232, 222)),
    ("yoke 3", 6.20, 6.70, -6.60, 6.60, (200, 45, 45)),
    ("MB4", 6.70, 7.05, -6.60, 6.60, (238, 232, 222)),
    ("yoke 4", 7.05, 7.40, -6.60, 6.60, (200, 45, 45)),
    ("ME1 -", 0.95, 6.95, -7.00, -6.60, (238, 232, 222)),
    ("YE1 -", 0.95, 6.95, -7.60, -7.00, (200, 45, 45)),
    ("ME2 -", 0.95, 6.95, -8.00, -7.60, (238, 232, 222)),
    ("YE2 -", 0.95, 6.95, -8.60, -8.00, (200, 45, 45)),
    ("ME3 -", 0.95, 6.95, -9.00, -8.60, (238, 232, 222)),
    ("YE3 -", 0.95, 6.95, -9.60, -9.00, (200, 45, 45)),
    ("ME4 -", 0.95, 6.95, -10.00, -9.60, (238, 232, 222)),
    ("YE4 -", 0.95, 6.95, -10.60, -10.00, (200, 45, 45)),
    ("ME1 +", 0.95, 6.95, 6.60, 7.00, (238, 232, 222)),
    ("YE1 +", 0.95, 6.95, 7.00, 7.60, (200, 45, 45)),
    ("ME2 +", 0.95, 6.95, 7.60, 8.00, (238, 232, 222)),
    ("YE2 +", 0.95, 6.95, 8.00, 8.60, (200, 45, 45)),
    ("ME3 +", 0.95, 6.95, 8.60, 9.00, (238, 232, 222)),
    ("YE3 +", 0.95, 6.95, 9.00, 9.60, (200, 45, 45)),
    ("ME4 +", 0.95, 6.95, 9.60, 10.00, (238, 232, 222)),
    ("YE4 +", 0.95, 6.95, 10.00, 10.60, (200, 45, 45)),
    # the forward calorimeter HF (VCAL): 12.495 to 159.5 cm, |z| = 11.1495 to 12.8005 m
    ("HF -", 0.125, 1.595, -12.80, -11.15, (150, 120, 190)),
    ("HF +", 0.125, 1.595, 11.15, 12.80, (150, 120, 190)),
]
LABELS = {"tracker": "tracker", "ECAL barrel": "ECAL", "HCAL barrel": "HCAL", "solenoid": "solenoid 3.8 T", "yoke 2": "iron yoke + muon chambers", "HF +": "HF"}


class View:
    """An orthographic view from the side of +x, a little from -z and from above: screen = (e1 . P, e2 . P), depth = v . P."""

    def __init__(self, azimuth=28.0, elevation=16.0):
        a, b = math.radians(azimuth), math.radians(elevation)
        self.v = np.array([math.cos(b) * math.cos(a), math.sin(b), -math.cos(b) * math.sin(a)])      # towards the viewer
        self.e1 = np.array([math.sin(a), 0.0, math.cos(a)])                                           # screen right ~ +z
        self.e2 = np.cross(self.e1, self.v)                                                            # screen up ~ +y
        self.light = np.array([0.55, 0.75, -0.35])
        self.light /= np.linalg.norm(self.light)

    def project(self, P):
        P = np.asarray(P, dtype=float)
        return np.stack([P @ self.e1, P @ self.e2], axis=-1)

    def depth(self, P):
        return np.asarray(P, dtype=float) @ self.v


def cyl(r, phi, z):
    return np.array([r * math.cos(phi), r * math.sin(phi), z])


def quads(solid, view: View, phi_a=90.0, phi_b=360.0, dphi=10.0, dz=0.9, dr=0.8):
    """The visible quads of one solid with the quadrant 0 < phi < 90 degrees removed: (points 4x3, normal, rgb)."""
    name, r1, r2, z1, z2, rgb = solid
    out = []
    nphi = max(1, int(round((phi_b - phi_a) / dphi)))
    phis = np.radians(np.linspace(phi_a, phi_b, nphi + 1))
    nz = max(1, int(math.ceil((z2 - z1) / dz)))
    zs = np.linspace(z1, z2, nz + 1)
    nr = max(1, int(math.ceil((r2 - r1) / dr)))
    rs = np.linspace(r1, r2, nr + 1)

    def add(pts, n):
        n = np.asarray(n, dtype=float)
        if n @ view.v > 1e-9:                     # facing the viewer
            out.append((np.array(pts), n, rgb))

    for i in range(nphi):                         # the outer surface
        p0, p1 = phis[i], phis[i + 1]
        pm = 0.5 * (p0 + p1)
        for j in range(nz):
            add([cyl(r2, p0, zs[j]), cyl(r2, p1, zs[j]), cyl(r2, p1, zs[j + 1]), cyl(r2, p0, zs[j + 1])], [math.cos(pm), math.sin(pm), 0])
    for i in range(nphi):                         # the two end faces
        p0, p1 = phis[i], phis[i + 1]
        for k in range(nr):
            for z, nz_ in ((z1, -1.0), (z2, 1.0)):
                add([cyl(rs[k], p0, z), cyl(rs[k + 1], p0, z), cyl(rs[k + 1], p1, z), cyl(rs[k], p1, z)], [0, 0, nz_])
    for phi, n in ((math.radians(phi_a), [1.0, 0.0, 0.0]), (math.radians(phi_b), [0.0, 1.0, 0.0])):   # the two cut faces
        for j in range(nz):
            for k in range(nr):
                add([cyl(rs[k], phi, zs[j]), cyl(rs[k + 1], phi, zs[j]), cyl(rs[k + 1], phi, zs[j + 1]), cyl(rs[k], phi, zs[j + 1])], n)
    return out


def shade(rgb, n, view: View, cut=False):
    lam = 0.55 + 0.45 * max(0.0, float(np.asarray(n) @ view.light))
    if cut:
        lam = min(1.0, lam + 0.08)
    return tuple(int(max(0, min(255, c * lam))) for c in rgb)


def scene(view: View, z_range=(-13.0, 13.0)):
    faces = []
    for s in SOLIDS:
        if s[4] < z_range[0] or s[3] > z_range[1]:
            continue
        faces += quads(s, view)
    # painter's algorithm: the farthest first
    faces.sort(key=lambda f: float(view.depth(f[0].mean(axis=0))))
    return faces


def tikz(faces, view: View, scale=1.0) -> str:
    lines = ["% written by example/G4/Standalone/draw_cms3d.py (do not edit): the CMS detector of Run 3, the quadrant x > 0, y > 0 cut away,",
             f"% {len(faces)} quads sorted from back to front; 1 unit = 1 m times the scale of the enclosing scope",
             "\\begin{scope}[line width=0.15pt, line join=round, c/.style args={#1/#2/#3}{fill={rgb,255:red,#1;green,#2;blue,#3},draw={rgb,255:red,#1;green,#2;blue,#3}}]"]
    for pts, n, rgb in faces:
        cut = abs(n[0] - 1.0) < 1e-9 or abs(n[1] - 1.0) < 1e-9
        r, g, b = shade(rgb, n, view, cut)
        q = view.project(pts) * scale
        path = "--".join(f"({x:.2f},{y:.2f})" for x, y in q)
        lines.append(f"\\filldraw[c={r}/{g}/{b}]{path}--cycle;")
    lines.append("\\end{scope}")
    # the anchors the picture uses: the centre, the ends of the beam line, label points
    anchors = {"cmsO": (0, 0, 0), "cmsBeamL": (0, 0, -15.0), "cmsBeamR": (0, 0, 15.0), "cmsTop": (0, 7.4, 0), "cmsBottom": (0, -7.4, 0),
               "cmsHFm": (0, 1.6, -12.0), "cmsHFp": (0, -1.6, 12.0), "cmsTracker": (0, 0.6, -1.5), "cmsECAL": (0, 1.53, -1.0),
               "cmsHCAL": (0, 2.35, -0.5), "cmsSol": (0, 3.37, 0.0), "cmsYoke": (0, 5.6, 0.5), "cmsEndcap": (0, 4.0, -8.6)}
    for k, P in anchors.items():
        x, y = view.project(P) * scale
        lines.append(f"\\coordinate ({k}) at ({x:.3f},{y:.3f});")
    return "\n".join(lines) + "\n"


def draw_matplotlib(ax, faces, view: View, scale=1.0, alpha=1.0):
    from matplotlib.collections import PolyCollection
    polys, cols, edges = [], [], []
    for pts, n, rgb in faces:
        cut = abs(n[0] - 1.0) < 1e-9 or abs(n[1] - 1.0) < 1e-9
        r, g, b = shade(rgb, n, view, cut)
        polys.append(view.project(pts) * scale)
        cols.append((r / 255, g / 255, b / 255, alpha))
        edges.append((r / 255, g / 255, b / 255, alpha))
    pc = PolyCollection(polys, facecolors=cols, edgecolors=edges, linewidths=0.2)
    ax.add_collection(pc)
    return pc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, default=HERE.parent.parent.parent / "chapter" / "tikz" / "g4_cms3d.tex")
    ap.add_argument("--png", type=Path, help="draw with matplotlib instead")
    ap.add_argument("--azimuth", type=float, default=28.0)
    ap.add_argument("--elevation", type=float, default=16.0)
    a = ap.parse_args()
    view = View(a.azimuth, a.elevation)
    faces = scene(view)
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        draw_matplotlib(ax, faces, view)
        bl, br = view.project([0, 0, -15]), view.project([0, 0, 15])
        ax.plot([bl[0], br[0]], [bl[1], br[1]], color="#444444", lw=1.0, zorder=0)
        ax.set_aspect("equal")
        ax.autoscale()
        ax.axis("off")
        fig.savefig(a.png, dpi=150, bbox_inches="tight")
        print(f"wrote {a.png} ({len(faces)} quads)")
        return
    a.output.write_text(tikz(faces, view))
    print(f"wrote {a.output} ({len(faces)} quads)")


if __name__ == "__main__":
    main()
