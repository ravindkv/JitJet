#!/usr/bin/env python3
"""Validation of the hand-written simulation standalone_SIM.py against Geant4 in the CMS geometry.

The official package of this stop is Geant4 itself, run by geant4_SIM.py in the CMS simulation geometry
(../../G4/geometry/cmsSimGeo2021.gdml). Five comparisons with the one-page detector and the textbook formulas
of standalone_SIM.py:

 1. the material in front of the calorimeters, in radiation lengths and in interaction lengths, against
    pseudorapidity: geantinos (Geant4's particle that feels nothing) on straight lines from the origin, 97
    azimuths per pseudorapidity, up to the tracker / calorimeter boundary r = 123.3 cm, |z| = 293.5 cm;
 2. the silicon sensors crossed by the straight lines of the charged primaries of the signal, per
    subdetector (geantinos again), against the hits of the stage "transport" of the hand-written script;
 3. the formulas against the values of the Particle Data Group: the minimum of the Bethe-Bloch curve in
    silicon, the radiation lengths, the looper threshold;
 4. the family: every count and energy of the full hand-written simulation next to Geant4's;
 5. single particles: 10 GeV photons and pions of 5, 20 and 50 GeV at eta = 0.5, the same guns the theory
    plots shoot through the hand-written showers.

Run with the interpreter that has Geant4, after geant4_SIM.py --sample signal (about three minutes):

  .venv/bin/python validate_SIM.py

Writes validate_SIM.json and prints the comparison (validate_SIM.log is this output).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import standalone_SIM as sim  # noqa: E402
import geant4_SIM as g4sim  # noqa: E402

SUB = {sim.PXB: "PXB", sim.PXF: "PXF", sim.TIB: "TIB", sim.TID: "TID", sim.TOB: "TOB", sim.TEC: "TEC"}
N_PHI = 97                        # a prime: the tracker repeats itself 12, 24, 36 ... times around the beam


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


def fast_gun(pid, e, eta, n, seed):
    """The same gun through the hand-written simulation."""
    m = sim.dec.m0(pid)
    p = math.sqrt(max(e * e - m * m, 0.0))
    out = []
    for i in range(n):
        rng = np.random.default_rng([seed, abs(pid), int(e * 1000), i])
        s = sim.Sim(rng, stage="full", record=True)
        phi = rng.uniform(-math.pi, math.pi)
        pt = p / math.cosh(eta)
        s.push(pid, (0.0, 0.0, 0.0), np.array([pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta)]), 0.0, 0, 0, primary=True)
        while s.stack:
            s.track(s.stack.pop())
        c = s.result(compress=True)["cal"]
        det = c["code"] // sim.CODE
        out.append((float(c["e"][np.isin(det, (sim.EB, sim.EE))].sum()), float(c["e"][np.isin(det, (sim.HB, sim.HE, sim.HO))].sum()),
                    np.sort(c["e"][det == sim.EB])[::-1], s.fate.get(0, "stopped")))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=max(1, min(10, (mp.cpu_count() or 2) - 2)))
    ap.add_argument("--physics", default="FTFP_BERT")
    ap.add_argument("--n-gun", type=int, default=100)
    args = ap.parse_args()
    args.no_field, args.worker_logs, args.no_birks, args.no_regions = False, None, False, False
    out = {}
    summ = json.loads((HERE / "geant4_SIM_signal.json").read_text())
    print(f"Geant4: {summ['version']}, physics list {summ['physics']}, geometry {g4sim.GDML.name}")
    print()

    # 1. the material scan
    etas = np.round(np.arange(-3.0, 3.01, 0.1), 2)
    x0, rms, lam = g4sim.material_scan(args, etas, nphi=N_PHI)
    scan = []
    for i, eta in enumerate(etas):
        mx, ml = model_material(float(eta))
        scan.append(dict(eta=float(eta), x0=float(x0[i]), x0_rms=float(rms[i]), lam=float(lam[i]), model_x0=mx, model_lam=ml))
    out["scan"] = scan
    print(f"== 1. material in front of the calorimeters: CMS geometry (Geant4 geantinos, mean over {N_PHI} azimuths) against the layer model ==")
    print(f"   {'eta':>5s} {'x/X0 CMS':>9s} {'model':>7s} {'x/lambda CMS':>13s} {'model':>7s}")
    for s in scan:
        if abs(round(s["eta"] * 10)) % 5 == 0 and s["eta"] >= 0:
            print(f"   {s['eta']:5.1f} {s['x0']:9.3f} {s['model_x0']:7.3f} {s['lam']:13.3f} {s['model_lam']:7.3f}")
    a = np.array([(s["x0"], s["model_x0"], s["lam"], s["model_lam"]) for s in scan if abs(s["eta"]) <= 2.5])
    out["mean_ratio_x0"] = float(a[:, 1].mean() / a[:, 0].mean())
    out["mean_ratio_lam"] = float(a[:, 3].mean() / a[:, 2].mean())
    out["rms_dev_x0"] = float(np.sqrt(np.mean((a[:, 1] / a[:, 0] - 1) ** 2)))
    out["lam_per_x0"] = float(a[:, 2].mean() / a[:, 0].mean())
    print(f"   |eta| < 2.5: mean x/X0 {a[:, 0].mean():.3f} (CMS) against {a[:, 1].mean():.3f} (model), ratio {out['mean_ratio_x0']:.2f}, point-by-point rms {100 * out['rms_dev_x0']:.0f} %;"
          f" mean x/lambda {a[:, 2].mean():.3f} against {a[:, 3].mean():.3f}, ratio {out['mean_ratio_lam']:.2f}; lambda per X0 in CMS: {out['lam_per_x0']:.2f}")
    imax = int(np.argmax([s["x0"] for s in scan]))
    out["x0_max"], out["x0_max_eta"] = scan[imax]["x0"], scan[imax]["eta"]
    print(f"   the maximum of the CMS curve: {scan[imax]['x0']:.2f} X0 at eta = {scan[imax]['eta']:.1f} (model there: {scan[imax]['model_x0']:.2f})")
    print("   (the layer model was tuned to a scan of the same geometry with ROOT's TGeo, whose radiation lengths are a few per cent longer)")

    # 2. the sensors crossed by the primaries on straight lines
    st = sim.load_stage()
    c, gg = st.crowd, st.g
    charged = np.array([sim.dec.charge3(int(i)) != 0 for i in c.ids])
    sel = np.flatnonzero(gg["primary"] & (c.block == 0) & charged)
    rows = [(0, float(c.p4[j, 1]), float(c.p4[j, 2]), float(c.p4[j, 3]), float(c.vtx[j, 0]), float(c.vtx[j, 1]), float(c.vtx[j, 2]), 0.0, 1) for j in sel]
    sens = {v: 0 for v in SUB.values()}
    for _, _, _, d in g4sim.run_scan(args, rows, list(range(len(rows)))):
        for k, v in d.items():
            sens[SUB[int(k)]] += v
    r = sim.run_signal(st, 1, "transport", sim.B_FIELD)
    mine = {sim.DET_NAME[d]: int((r["trk"]["det"] == d).sum()) for d in sim.TRACKER}
    out["sensors_geant4"], out["sensors_model"] = sens, mine
    out["n_charged"] = int(len(sel))
    print()
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

    # 4. the family: the full hand-written simulation next to Geant4
    rf = sim.run_signal(st, 1, "full", sim.B_FIELD)
    fast = sim.signal_row(st, rf, "full")
    g4 = summ["stages"][0]
    nf, ng = rf["n"], summ["n"]
    fam = [("primaries that cross the tracker boundary as themselves", fast["boundary"], g4["boundary"] + g4.get("boundary_fwd", 0), "d"),
           ("decays in flight in the tracker volume", nf.get("decays", 0), ng["decays"], "d"),
           ("photon conversions in the tracker volume", nf.get("conversions", 0), ng["conversions"], "d"),
           ("nuclear interactions in the tracker volume", nf.get("nuclear", 0), ng["nuclear"], "d"),
           ("secondaries above 10 MeV born in the tracker volume", fast["secondaries"], g4["secondaries"], "d"),
           ("tracker hits", fast["trk"], g4["trk"], "d"),
           ("energy left in the silicon [MeV]", fast["de"] * 1e3, g4["de"] * 1e3, ".0f"),
           ("ECAL barrel energy [GeV]", fast["EB"], g4["EB"], ".1f"),
           ("ECAL endcap energy [GeV]", fast["EE"], g4["EE"], ".1f"),
           ("HB + HE scintillator energy [GeV]", fast["HB"] + fast["HE"], g4["HB"] + g4["HE"], ".2f"),
           ("HF photoelectrons", fast["HF"], g4["HF"], ".0f"),
           ("muon-chamber hits", fast["mu"], g4["mu"], "d"),
           ("jet [0]: members reaching the calorimeters as primaries", fast["jet_bnd_n"], g4["jet_bnd_n"], "d"),
           ("jet [0]: energy of its descendants in the crystals [GeV]", fast["jet_ecal"], g4["jet_ecal"], ".1f")]
    out["family"] = [dict(what=w, standalone=float(a_), geant4=float(b_)) for w, a_, b_, _ in fam]
    out["hits_per_primary"] = dict(standalone=fast["trk"] / fast["primaries"], geant4=g4["trk"] / g4["primaries"])
    print()
    print("== 4. the family (672 primaries): hand-written simulation against Geant4 ==")
    print(f"   {'':58s} {'hand-written':>13s} {'Geant4':>10s} {'ratio':>7s}")
    for w, a_, b_, f in fam:
        print(f"   {w:58s} {format(a_, f):>13s} {format(b_, f):>10s} {a_ / b_ if b_ else float('nan'):7.2f}")
    print("   Geant4 also makes what the hand-written script leaves out: "
          f"{ng['delta_rays']} delta rays, {ng['compton']} Compton scatterings, {ng['photoelectric']} photoelectric absorptions and {ng['brems']} bremsstrahlung photons down to 1 keV")

    # 5. single particles
    print()
    print(f"== 5. single particles at eta = 0.5, {args.n_gun} per point (mean +- rms) ==")
    guns = {}
    g = g4sim.run_gun(args, 22, 10.0, 0.5, args.n_gun)
    f = fast_gun(22, 10.0, 0.5, args.n_gun, 5)
    def frac(cells_list, etot):
        v = np.array([(c[0], c[:9].sum(), c[:25].sum(), c.sum()) for c in cells_list if len(c)]) / etot
        return v.mean(axis=0), v.std(axis=0)
    for label, cells, fates in (("geant4", g["eb_cells"], g["fate"]), ("standalone", [x[2] for x in f], [x[3] for x in f])):
        for conv in (False, True):
            sel_ = [c for c, ft in zip(cells, fates) if (ft == "conversion") == conv]
            if sel_:
                mean, std = frac(sel_, 10.0)
                guns[f"photon10_{label}_{'converted' if conv else 'unconverted'}"] = dict(n=len(sel_), mean=mean.tolist(), rms=std.tolist())
    print("   10 GeV photons, fraction of the energy in the hottest crystal / 9 hottest / 25 hottest / all barrel crystals")
    for conv in ("unconverted", "converted"):
        for label in ("standalone", "geant4"):
            k = f"photon10_{label}_{conv}"
            if k in guns:
                v = guns[k]
                print(f"   {conv:11s} {('hand-written' if label == 'standalone' else 'Geant4'):12s} {v['n']:4d} photons   " + "  ".join(f"{m:.3f}" for m in v["mean"]))
    print("   pions: (crystals + 125 x HB scintillator) / E for those that reach the calorimeters, and the share of the crystals in that sum")
    for e in (5.0, 20.0, 50.0):
        g = g4sim.run_gun(args, 211, e, 0.5, args.n_gun)
        ok = np.array([ft == "calorimeter" for ft in g["fate"]])
        rg = (g["ecal"] + 125.0 * g["hcal"])[ok] / e
        sg = (g["ecal"][ok] / np.maximum(g["ecal"] + 125.0 * g["hcal"], 1e-9)[ok])
        f = [x for x in fast_gun(211, e, 0.5, args.n_gun, 3) if x[3] == "calorimeter"]
        rf_ = np.array([(x[0] + 125.0 * x[1]) / e for x in f])
        sf = np.array([x[0] / max(x[0] + 125.0 * x[1], 1e-9) for x in f])
        guns[f"pion{e:g}"] = dict(geant4=dict(n=int(ok.sum()), response=float(rg.mean()), rms=float(rg.std()), ecal_share=float(sg.mean())),
                                  standalone=dict(n=len(f), response=float(rf_.mean()), rms=float(rf_.std()), ecal_share=float(sf.mean())))
        print(f"   pi+ {e:5.0f} GeV   hand-written {len(f):3d}: {rf_.mean():.2f} +- {rf_.std():.2f} (crystals {100 * sf.mean():.0f} %)"
              f"   Geant4 {int(ok.sum()):3d}: {rg.mean():.2f} +- {rg.std():.2f} (crystals {100 * sg.mean():.0f} %)")
    out["guns"] = guns
    out["geant4"] = dict(version=summ["version"], physics=summ["physics"])
    (HERE / "validate_SIM.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
