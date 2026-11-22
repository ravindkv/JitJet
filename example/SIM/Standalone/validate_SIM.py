#!/usr/bin/env python3
"""Validation of the detector of standalone_SIM.py against the CMS simulation geometry, with ROOT's TGeo.

The official package here is the geometry itself: ../../G4/geometry/cmsSimGeo2021.root is the TGeo dump of the
geometry CMSSW gives to Geant4 (every volume with its material, radiation length and interaction length; the
file of chapter 9). ROOT's navigator follows straight lines through it, and three things are compared with the
one-page detector of standalone_SIM.py:

 1. the material in front of the calorimeters, in radiation lengths and in interaction lengths, against
    pseudorapidity (mean over azimuth), up to the tracker / calorimeter boundary r = 123.3 cm, |z| = 293.5 cm;
 2. the silicon sensors crossed by the straight lines of the charged primaries of the signal, per subdetector,
    against the hits of the stage "transport" of the standalone script;
 3. the formulas against the values of the Particle Data Group: the minimum of the Bethe-Bloch curve in silicon,
    the radiation lengths, the looper threshold.

Run with an interpreter that has ROOT (here the ME venv with PYTHONPATH=$(root-config --libdir)):

  PYTHONPATH=$(root-config --libdir) ../../ME/Standalone/.venv/bin/python validate_SIM.py

Writes validate_SIM.json and prints the comparison (validate_SIM.log is this output).
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import standalone_SIM as sim  # noqa: E402

GEO = HERE.parent.parent / "G4" / "geometry" / "cmsSimGeo2021.root"
SUB = {"pixbar": "PXB", "pixfwd": "PXF", "tib": "TIB", "tid": "TID", "tob": "TOB", "tec": "TEC"}


def model_material(eta: float):
    """x/X0 and x/lambda_I of the standalone layer model along a straight line from the origin."""
    th = 2 * math.atan(math.exp(-eta))
    ux, uz = math.sin(th), math.cos(th)
    x0 = lam = 0.0
    for j, b in enumerate(sim.BARREL):
        det, r, zmin, zmax = b[0], b[2], b[3], b[4]
        if det == 0 or ux < 1e-9:
            continue
        z = abs(r / ux * uz)
        if zmin <= z <= zmax and z <= sim.Z_TRK:
            f = min(1.0 / ux, sim.F_CAP if det != -1 else 80.0)
            t = sim.layer_t(True, j, z, r)
            x0 += t * f
            lam += t * f * (0.84 if det == -1 else sim.LAMBDA_PER_X0)
    for j, d in enumerate(sim.DISKS):
        det, z, rin, rout = d[0], d[2], d[3], d[4]
        if det == 0 or z * uz <= 0:
            continue
        r = abs(z / uz * ux)
        if rin <= r <= rout and r <= sim.R_TRK:
            f = min(1.0 / abs(uz), sim.F_CAP)
            t = sim.layer_t(False, j, z, r)
            x0 += t * f
            lam += t * f * sim.LAMBDA_PER_X0
    return x0, lam


def main():
    import ROOT
    ROOT.gErrorIgnoreLevel = ROOT.kError
    f = ROOT.TFile.Open(str(GEO))
    g = f.Get("cmsGeo")
    out = {}

    def walk(x, y, z, d, sensors=None):
        g.InitTrack(x, y, z, *d)
        x0 = lam = 0.0
        last = None
        for _ in range(20000):
            node = g.GetCurrentNode()
            vol = node.GetVolume()
            mat = vol.GetMaterial()
            name = vol.GetName()
            g.FindNextBoundaryAndStep(1e5)
            s = g.GetStep()
            p = g.GetCurrentPoint()
            if mat.GetRadLen() > 0:
                x0 += s / mat.GetRadLen()
            if mat.GetIntLen() > 0:
                lam += s / mat.GetIntLen()
            if sensors is not None and ("Active" in name or "PixelForwardSensor" in name) and s > 1e-4:
                path = g.GetPath()
                if path != last:
                    ns = name.split(":")[0]
                    for k, v in SUB.items():
                        if ns.startswith(k):
                            sensors[v] += 1
                            break
                last = path
            if math.hypot(p[0], p[1]) >= sim.R_TRK or abs(p[2]) >= sim.Z_TRK or g.IsOutside():
                break
        return x0, lam

    # 1. the material scan
    etas = np.round(np.arange(-3.0, 3.01, 0.1), 2)
    phis = np.linspace(0.0, 2 * math.pi, 24, endpoint=False) + 0.013
    scan = []
    for eta in etas:
        th = 2 * math.atan(math.exp(-eta))
        a = np.array([walk(0.0, 0.0, 0.0, (math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th))) for ph in phis])
        mx, ml = model_material(float(eta))
        scan.append(dict(eta=float(eta), x0=float(a[:, 0].mean()), x0_rms=float(a[:, 0].std()), lam=float(a[:, 1].mean()), model_x0=mx, model_lam=ml))
    out["scan"] = scan
    print("== 1. material in front of the calorimeters: CMS geometry (TGeo, mean over 24 azimuths) against the layer model ==")
    print(f"   {'eta':>5s} {'x/X0 CMS':>9s} {'model':>7s} {'x/lambda CMS':>13s} {'model':>7s}")
    for s in scan:
        if abs(round(s["eta"] * 10)) % 5 == 0 and s["eta"] >= 0:
            print(f"   {s['eta']:5.1f} {s['x0']:9.3f} {s['model_x0']:7.3f} {s['lam']:13.3f} {s['model_lam']:7.3f}")
    a = np.array([(s["x0"], s["model_x0"], s["lam"], s["model_lam"]) for s in scan if abs(s["eta"]) <= 2.5])
    out["mean_ratio_x0"] = float(a[:, 1].mean() / a[:, 0].mean())
    out["mean_ratio_lam"] = float(a[:, 3].mean() / a[:, 2].mean())
    out["rms_dev_x0"] = float(np.sqrt(np.mean((a[:, 1] / a[:, 0] - 1) ** 2)))
    print(f"   |eta| < 2.5: mean x/X0 {a[:, 0].mean():.3f} (CMS) against {a[:, 1].mean():.3f} (model), ratio {out['mean_ratio_x0']:.2f}, point-by-point rms {100 * out['rms_dev_x0']:.0f} %;"
          f" mean x/lambda {a[:, 2].mean():.3f} against {a[:, 3].mean():.3f}, ratio {out['mean_ratio_lam']:.2f}; lambda per X0 in CMS: {a[:, 2].mean() / a[:, 0].mean():.2f}")
    imax = int(np.argmax([s["x0"] for s in scan]))
    print(f"   the maximum of the CMS curve: {scan[imax]['x0']:.2f} X0 at eta = {scan[imax]['eta']:.1f} (model there: {scan[imax]['model_x0']:.2f})")

    # 2. the sensors crossed by the primaries on straight lines
    st = sim.load_stage()
    c, gg = st.crowd, st.g
    charged = np.array([sim.dec.charge3(int(i)) != 0 for i in c.ids])
    sel = np.flatnonzero(gg["primary"] & (c.block == 0) & charged)
    sens = Counter()
    for j in sel:
        p = c.p4[j, 1:]
        d = p / np.linalg.norm(p)
        walk(float(c.vtx[j, 0]), float(c.vtx[j, 1]), float(c.vtx[j, 2]), (float(d[0]), float(d[1]), float(d[2])), sensors=sens)
    r = sim.run_signal(st, 1, "transport", sim.B_FIELD)
    mine = {sim.DET_NAME[d]: int((r["trk"]["det"] == d).sum()) for d in sim.TRACKER}
    out["sensors_tgeo"], out["sensors_model"] = dict(sens), mine
    print()
    out["n_charged"] = int(len(sel))
    print(f"== 2. silicon sensors on the straight lines of the {len(sel)} charged primaries of the signal (the hits of the stage 'transport') ==")
    print(f"   {'':6s} {'CMS geometry':>13s} {'layer model':>12s}")
    for k in ("PXB", "PXF", "TIB", "TID", "TOB", "TEC"):
        print(f"   {k:6s} {sens[k]:13d} {mine[k]:12d}")
    print(f"   {'total':6s} {sum(sens.values()):13d} {sum(mine.values()):12d}   ratio {sum(mine.values()) / max(sum(sens.values()), 1):.2f}")

    # 3. the formulas
    print()
    print("== 3. the formulas against the Particle Data Group ==")
    chk = [("minimum of dE/dx in silicon [MeV cm2/g]", sim.DEDX_MIN, 1.664),
           ("most probable loss of a MIP in 320 um of silicon [keV]", sim.landau_mpv(3.5, 0.032) * 1e6, 84.0),
           ("helix radius at 1 GeV in 3.8 T [cm]", sim.helix_radius(1.0), 100.0 / (0.299792458 * 3.8)),
           ("looper threshold for r = 129 cm [GeV]", sim.looper_pt(129.0), 0.299792458 * 3.8 * 1.29 / 2),
           ("conversion probability in 1 X0", sim.conversion_probability(1.0), 1 - math.exp(-7 / 9)),
           ("Highland angle, 1 GeV pion, 10 % X0 [mrad]", 1e3 * sim.highland(1.0, 0.9904, 0.1), 1e3 * 0.0136 / 0.9904 * math.sqrt(0.1) * (1 + 0.038 * math.log(0.1)))]
    out["formulas"] = [dict(what=w, script=float(a_), reference=float(b_)) for w, a_, b_ in chk]
    for w, a_, b_ in chk:
        print(f"   {w:56s} script {a_:9.4f}   reference {b_:9.4f}   ({100 * (a_ / b_ - 1):+.1f} %)")
    (HERE / "validate_SIM.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
