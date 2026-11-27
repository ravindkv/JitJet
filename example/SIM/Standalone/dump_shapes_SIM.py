#!/usr/bin/env python3
"""The shapes of the CMS detector for the 3D displays, read out of the Geant4 geometry.

The displays of display_geant4_SIM.py and the film draw the detector the particles were tracked through, not
a sketch of it: this script walks the tree of placed volumes of ../../G4/geometry/cmsSimGeo2021.gdml inside
Geant4, multiplies the rotations and translations from the world down to every volume of interest, and writes
the eight corners of each in the CMS frame.

  kind      what                                              volumes
  trk       the active silicon of every tracker module        about 16 000
  eb, ee    every lead-tungstate crystal                      61 200 + 14 648
  es        every preshower sensor                            about 4 300
  hb he ho  every scintillator tile                           tens of thousands
  mu        the gas volumes of the muon chambers              DT, CSC, RPC, GEM
  yoke      the steel of the return yoke
  coil      the superconducting coil and its cryostat (tubes: inner radius, outer radius, half length)
  hf        the steel wedges of the forward calorimeter

A box, a trapezoid (G4Trap) and a generic trapezoid (the crystals) give their true corners; any other solid
gives the corners of its bounding box. Run with the Geant4 interpreter, about a minute:

  .venv/bin/python dump_shapes_SIM.py          # writes ../cms_shapes_SIM.npz (not in git: make shapes)

Lengths in cm.
"""
from __future__ import annotations

import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import standalone_SIM as sim  # noqa: E402

GDML = Path(__import__("os").environ.get("JITJET_SIM_GDML", HERE.parent.parent / "G4" / "geometry" / "cmsSimGeo2021.gdml"))
OUT = HERE.parent / "cms_shapes_SIM.npz"
FERRO = ("materialsM_Steel-008", "materialsIron", "materialsYokeSteel")
TRK_PREFIX = {"pixbar": sim.PXB, "pixfwd": sim.PXF, "tib": sim.TIB, "tid": sim.TID, "tob": sim.TOB, "tec": sim.TEC}
SIGNS = np.array([(-1, -1, -1), (1, -1, -1), (-1, 1, -1), (1, 1, -1), (-1, -1, 1), (1, -1, 1), (-1, 1, 1), (1, 1, 1)], dtype=float)


def kind_of(lv):
    """(kind, sub) of a logical volume, or None."""
    n, m = lv.GetName(), lv.GetMaterial().GetName()
    if m == "materialsE_PbWO4":
        return ("eb", 0) if n.startswith("ebalgo") else ("ee", 0)
    if m == "materialsE_Silicon" and n.startswith("esalgoSFS"):
        return ("es", 0 if n.endswith("X") else 1)
    if m == "materialsScintillator":
        for pre, k in (("hcalbarrel", "hb"), ("hcalendcap", "he"), ("hcalouter", "ho")):
            if n.startswith(pre):
                return (k, 0)
        return None
    if m == "materialsSilicon" and ("Active" in n or "ForwardSensor" in n):
        for pre, det in TRK_PREFIX.items():
            if n.startswith(pre):
                return ("trk", det)
    if m == "materialsM_DTBX_Gas":
        return ("mu", sim.DT)
    if m == "materialsM_Argon_CF_4_CO_2" and "ActiveGas" in n:
        return ("mu", sim.CSC)
    if m == "materialsM_RPC_Gas":
        return ("mu", sim.RPC)
    if m == "gemfM_GEM_Gas":
        return ("mu", sim.GEM)
    if n == "hcalforwardalgoHVQX":
        return ("hf", 0)
    if n.startswith("muonYoke") or (m in FERRO and (n.startswith("mf") or n.startswith("muon"))):
        return ("yoke", 0)
    if n.startswith("mgnt") and lv.GetSolid().GetEntityType() == "G4Tubs":
        return ("coil", 0)
    return None


def corners(solid, g):
    """The eight corners of a solid in its own frame [mm]; reflected solids are mirrored in z."""
    t = str(solid.GetEntityType())
    if t == "G4ReflectedSolid":
        c = corners(solid.GetConstituentMovedSolid(), g)
        c = c.copy()
        c[:, 2] *= -1
        return c
    if t == "G4Box":
        return SIGNS * np.array([solid.GetXHalfLength(), solid.GetYHalfLength(), solid.GetZHalfLength()])
    if t == "G4Trap":
        dz = solid.GetZHalfLength()
        ax = solid.GetSymAxis()
        tx, ty = ax.x / ax.z, ax.y / ax.z                    # tan(theta) cos(phi), tan(theta) sin(phi)
        out = []
        for s, dy, dxa, dxb, ta in ((-1, solid.GetYHalfLength1(), solid.GetXHalfLength1(), solid.GetXHalfLength2(), solid.GetTanAlpha1()),
                                    (1, solid.GetYHalfLength2(), solid.GetXHalfLength3(), solid.GetXHalfLength4(), solid.GetTanAlpha2())):
            cx, cy = s * dz * tx, s * dz * ty
            out += [(cx - dy * ta - dxa, cy - dy, s * dz), (cx - dy * ta + dxa, cy - dy, s * dz),
                    (cx + dy * ta - dxb, cy + dy, s * dz), (cx + dy * ta + dxb, cy + dy, s * dz)]
        return np.array(out)
    if t == "G4GenericTrap":
        dz = solid.GetZHalfLength()
        v = solid.GetVertices()
        o = [0, 1, 3, 2]                                     # around the face -> the order of SIGNS
        return np.array([(v[i].x, v[i].y, -dz) for i in o] + [(v[4 + i].x, v[4 + i].y, dz) for i in o])
    lo, hi = g.G4ThreeVector(), g.G4ThreeVector()
    solid.BoundingLimits(lo, hi)
    c = np.array([(lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2])
    h = np.array([(hi.x - lo.x) / 2, (hi.y - lo.y) / 2, (hi.z - lo.z) / 2])
    return c + SIGNS * h


def main():
    import geant4_pybind as g
    t0 = time.time()
    parser = g.G4GDMLParser()
    parser.Read(str(GDML), False)
    world = parser.GetWorldVolume().GetLogicalVolume()
    info, kids, has = {}, {}, {}

    def prepare(lv):
        n = lv.GetName()
        if n in has:
            return has[n]
        k = kind_of(lv)
        info[n] = (k, corners(lv.GetSolid(), g) if k else None, lv)
        ch = []
        any_ = k is not None
        for i in range(lv.GetNoDaughters()):
            pv = lv.GetDaughter(i)
            dl = pv.GetLogicalVolume()
            if prepare(dl):
                r = pv.GetObjectRotationValue()
                t = pv.GetObjectTranslation()
                R = np.array([[r.xx, r.xy, r.xz], [r.yx, r.yy, r.yz], [r.zx, r.zy, r.zz]])
                ch.append((R, np.array([t.x, t.y, t.z]), dl.GetName()))
                any_ = True
        kids[n] = ch
        has[n] = any_
        return any_

    prepare(world)
    out = {}
    tubes = []
    count = Counter()

    def walk(n, R, t):
        k, c, lv = info[n]
        if k is not None:
            if k[0] == "coil":
                s = lv.GetSolid()
                tubes.append((s.GetInnerRadius() / 10, s.GetOuterRadius() / 10, s.GetZHalfLength() / 10, t[2] / 10))
            else:
                out.setdefault(k[0], []).append((k[1], (c @ R.T + t) / 10.0))
            count[k[0]] += 1
        for Rc, tc, dn in kids[n]:
            walk(dn, R @ Rc, R @ tc + t)

    sys.setrecursionlimit(10000)
    walk(world.GetName(), np.eye(3), np.zeros(3))
    d = {}
    for k, rows in out.items():
        d[k] = np.array([r[1] for r in rows], dtype=np.float32)
        d[k + "_sub"] = np.array([r[0] for r in rows], dtype=np.int8)
    d["coil"] = np.array(tubes, dtype=np.float32)
    np.savez_compressed(OUT, **d)
    print("volumes: " + ", ".join(f"{k} {v}" for k, v in sorted(count.items())))
    for k in sorted(out):
        v = d[k]
        r = np.hypot(v[..., 0], v[..., 1])
        print(f"   {k:5s} r {r.min():7.1f} .. {r.max():7.1f} cm   |z| up to {np.abs(v[..., 2]).max():7.1f} cm")
    print("   coil  " + "; ".join(f"r {a:.0f}-{b:.0f} half length {c:.0f}" for a, b, c, _ in tubes))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
