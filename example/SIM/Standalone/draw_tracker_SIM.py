#!/usr/bin/env python3
"""The tracker in the r-z plane, one quadrant, as a TikZ picture: chapter/tikz/sim_tracker.tex.

The layers and disks are those of standalone_SIM.py (BARREL, DISKS_1), which follow the CMS simulation geometry:
the pixel barrel and disks, the inner barrel and disks, the outer barrel and the endcap of the strip tracker,
with the pseudorapidities a straight line from the collision point crosses them at.

  python3 draw_tracker_SIM.py          # writes ../../../chapter/tikz/sim_tracker.tex
"""
from __future__ import annotations

import math
from pathlib import Path

import standalone_SIM as sim

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent.parent / "chapter" / "tikz" / "sim_tracker.tex"
SX, SY = 13.6 / 300.0, 6.0 / 125.0            # cm on the page per cm of detector, along z and along r
COL = {sim.PXB: "blue!75!black", sim.PXF: "cyan!60!black", sim.TIB: "red!75!black", sim.TID: "orange!85!black", sim.TOB: "green!45!black", sim.TEC: "violet!80!black"}


def main():
    L = ["% written by example/SIM/Standalone/draw_tracker_SIM.py from the layers of standalone_SIM.py: do not edit",
         "\\begin{tikzpicture}[x=1cm,y=1cm,font=\\footnotesize,>={Stealth[length=2mm]}]"]
    X, Y = lambda z: f"{z * SX:.3f}", lambda r: f"{r * SY:.3f}"
    zmax, rmax = sim.Z_TRK, sim.R_TRK
    # lines of constant pseudorapidity, clipped to the tracker volume
    for eta in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        t = 1.0 / math.sinh(eta)                # r / z
        z, r = (rmax / t, rmax) if rmax / t <= zmax else (zmax, zmax * t)
        L.append(f"\\draw[black!30,densely dashed] (0,0)--({X(z)},{Y(r)});")
        if r >= rmax - 1e-6:
            L.append(f"\\node[black!60,above,font=\\scriptsize] at ({X(z)},{Y(r)}) {{$\\eta={eta:g}$}};")
        else:
            L.append(f"\\node[black!60,right,font=\\scriptsize] at ({X(z)},{Y(r)}) {{$\\eta={eta:g}$}};")
    L.append(f"\\node[black!60,above,font=\\scriptsize] at (0,{Y(rmax)}) {{$\\eta=0$}};")
    # the volume, the beam pipe
    L.append(f"\\draw[black!55] (0,0) rectangle ({X(zmax)},{Y(rmax)});")
    L.append(f"\\draw[black!45,line width=0.3pt] (0,{Y(2.25)})--({X(190.0)},{Y(2.25)});")
    for det, lay, r, z0, z1, *_ in sim.BARREL:
        if det > 0:
            L.append(f"\\draw[{COL[det]},line width=1.1pt] ({X(z0)},{Y(r)})--({X(z1)},{Y(r)});")
    for det, lay, z, r0, r1, *_ in sim.DISKS_1:
        if det > 0:
            L.append(f"\\draw[{COL[det]},line width=1.1pt] ({X(z)},{Y(r0)})--({X(z)},{Y(r1)});")
    # axes
    L.append(f"\\draw[->] (0,0)--({X(zmax + 14)},0) node[right]{{$z$ [cm]}};")
    L.append(f"\\draw[->] (0,0)--(0,{Y(rmax + 13)}) node[above]{{$r$ [cm]}};")
    for z in (0, 50, 100, 150, 200, 250):
        L.append(f"\\draw ({X(z)},0)--({X(z)},-0.07) node[below,font=\\scriptsize]{{{z}}};")
    for r in (20, 40, 60, 80, 100, 120):
        L.append(f"\\draw (0,{Y(r)})--(-0.07,{Y(r)}) node[left,font=\\scriptsize]{{{r}}};")
    # the names, each in the colour of its layers
    lab = [(sim.PXB, "PXB, 4 layers", 55.0, 16.5, "right"), (sim.PXF, "PXF, 3 disks", 55.0, 9.0, "right"), (sim.TIB, "TIB, 4 layers", 72.0, 29.5, "right"),
           (sim.TID, "TID, 3 disks", 78.0, 54.5, "right"), (sim.TOB, "TOB, 6 layers", 3.0, 115.5, "right"), (sim.TEC, "TEC, 9 disks", 131.0, 14.0, "right")]
    for det, text, z, r, pos in lab:
        L.append(f"\\node[{COL[det]},{pos}=-1pt,font=\\scriptsize\\bfseries,fill=white,inner sep=1pt] at ({X(z)},{Y(r)}) {{{text}}};")
    L.append(f"\\node[black!60,font=\\scriptsize,anchor=south west] at ({X(195.0)},{Y(1.0)}) {{beam pipe}};")
    L.append("\\end{tikzpicture}")
    OUT.write_text("\n".join(L) + "\n")
    print(f"{OUT} written")


if __name__ == "__main__":
    main()
