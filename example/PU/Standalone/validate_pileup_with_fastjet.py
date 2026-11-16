#!/usr/bin/env python3
"""Validation of the standalone pileup stage against FastJet and PYTHIA.

Two modes.

record  The mixed final state of the book's record (../event_PU.lhe: the signal particles of chapter 7 and
        the in-time pileup particles, within |eta| < 5 and without neutrinos, exactly as standalone_pileup.py
        clustered them) is clustered by FastJet 3.5.2: anti-kT R = 0.4 with the active area (explicit ghosts,
        ghost area 0.01, |y| < 5), the GridMedianBackgroundEstimator(5.0, 0.55) that CMSSW's
        fixedGridRhoFastjetAll wraps, and the JetMedianBackgroundEstimator on kT jets of R = 0.4 with areas in
        |y| < 4.4; then every jet is subtracted with rho A. The jets must agree exactly (the same particles,
        the same algorithm); the areas agree within the ghost placement (FastJet's ghosts are scattered
        differently from ours), rho within a fraction of a percent (the grid is identical; the jet median
        depends on the ghost jets), and the subtracted momenta accordingly.

library The cross sections and the multiplicities of the minimum-bias collisions: PYTHIA's own numbers at
        initialisation (sigma for ND, SD, DD, CD at 13.6 TeV) against the shares of the library, and the
        per-collision particle and charged multiplicities, with and without the |eta| cuts, against the 2025
        CMSSW library of the companion (1000 events at 13.6 TeV, CMSSW_15_0_5, PYTHIA 8.311).

Needs the environment of example/ME/Standalone (fastjet, pythia8mc):
  ../../ME/Standalone/.venv/bin/python validate_pileup_with_fastjet.py --mode record
  ../../ME/Standalone/.venv/bin/python validate_pileup_with_fastjet.py --mode library
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def mode_record(args, fastjet):
    sp = load("sp", HERE / "standalone_pileup.py")
    gj = sp.gj
    summary, collisions, particles, jets, signal_of = sp.read_pileup_tables(args.input)
    ev, _, _ = gj.dec.read_record(args.input)
    # the same input as the standalone stage: the signal without neutrinos within |eta| < 5, the in-time pileup likewise
    sig_rows = [i for i in ev.final_indices() if abs(ev.p[i].id) not in gj.IGNORE_IDS and not gj.is_neutrino(ev.p[i].id) and abs(gj.eta(ev.p[i].p)) < sp.ETA_MAX]
    p4 = [ev.p[i].p for i in sig_rows]
    tags = [0] * len(sig_rows)
    for p in particles:
        ptv = math.hypot(p["px"], p["py"])
        if gj.is_neutrino(p["id"]) or ptv <= 0 or abs(math.asinh(p["pz"] / ptv)) >= sp.ETA_MAX:
            continue
        p4.append(np.array([p["e"], p["px"], p["py"], p["pz"]]))
        tags.append(p["k"])
    p4 = np.array(p4)
    tags = np.array(tags)
    print(f"the mixed final state of the record: {len(sig_rows)} signal + {len(p4) - len(sig_rows)} pileup particles")
    pjs = [fastjet.PseudoJet(float(p[1]), float(p[2]), float(p[3]), float(p[0])) for p in p4]
    for n, pj in enumerate(pjs):
        pj.set_user_index(n)
    jd = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    ad = fastjet.AreaDefinition(fastjet.active_area_explicit_ghosts, fastjet.GhostedAreaSpec(sp.GHOST_MAXRAP, 1, sp.GHOST_AREA))
    cs = fastjet.ClusterSequenceArea(pjs, jd, ad)
    fj_jets = sorted([j for j in cs.inclusive_jets(sp.JET_PT_MIN) if j.has_constituents() and any(not c.is_pure_ghost() for c in j.constituents())],
                     key=lambda j: -j.pt())
    # rho: the grid and the jet median
    grid = fastjet.GridMedianBackgroundEstimator(sp.GRID_RAP_MAX, sp.GRID_SPACING)
    grid.set_particles(pjs)
    rho_grid_fj = grid.rho()
    jd_kt = fastjet.JetDefinition(fastjet.kt_algorithm, 0.4)
    sel = fastjet.SelectorAbsRapMax(sp.KT_RHO_RAP_MAX)
    jme = fastjet.JetMedianBackgroundEstimator(sel, jd_kt, ad)
    jme.set_particles(pjs)
    rho_kt_fj = jme.rho()
    # the standalone numbers of the record
    rho_grid_sa = float(next(l.split("grid median")[1].split()[0] for l in summary if "rho:" in l))
    rho_kt_line = next((l for l in summary if "jet median" in l), None)
    rho_kt_sa = float(rho_kt_line.split("jet median")[1].split()[0].rstrip(";")) if rho_kt_line else None
    grid_y, cells, _ = sp.grid_rho(p4)
    srt = np.sort(cells.ravel())
    print(f"rho (grid median, |y| < 5, 0.55, phi from 0): FastJet {rho_grid_fj:.4f}, standalone {grid_y:.4f} (record: {rho_grid_sa:.4f}) GeV per unit area; "
          f"the two middle cells of the {srt.size}: {srt[srt.size // 2 - 1]:.4f}, {srt[srt.size // 2]:.4f}")
    print(f"   the same grid in eta (the older fixedGridRhoAll): {sp.grid_rho(p4, use_rapidity=False)[0]:.4f}")
    if rho_kt_sa is not None:
        print(f"rho (jet median, kT R = 0.4, |y| < 4.4): FastJet {rho_kt_fj:.4f}, standalone {rho_kt_sa:.4f}")
    # jet by jet
    out = []
    print(f"{'fj':>4s} {'pT':>9s} {'A_fj':>7s} {'sa':>4s} {'pT_sa':>9s} {'A_sa':>7s} {'dpT':>8s} {'dA':>7s} {'n':>4s} {'n_sa':>5s}  same")
    n_same, max_dpt, max_da = 0, 0.0, 0.0
    sa_by_const = {}
    # the standalone constituents from the record: signal rows and (k, index) of the pileup particles
    sa_const = {j["rank"]: set() for j in jets}
    for r, k in signal_of.items():
        if k >= 0:
            sa_const.setdefault(k, set()).add(("s", r))
    idx_in_coll = {}
    for p in particles:
        idx_in_coll[p["k"]] = idx_in_coll.get(p["k"], -1) + 1
        if p["jet"] >= 0:
            sa_const.setdefault(p["jet"], set()).add((p["k"], idx_in_coll[p["k"]]))
    # map the FastJet constituent user indices back to those keys
    keys = [("s", r) for r in sig_rows]
    counters = {}
    for p in particles:
        ptv = math.hypot(p["px"], p["py"])
        counters[p["k"]] = counters.get(p["k"], -1) + 1
        if gj.is_neutrino(p["id"]) or ptv <= 0 or abs(math.asinh(p["pz"] / ptv)) >= sp.ETA_MAX:
            continue
        keys.append((p["k"], counters[p["k"]]))
    sa_jets = {j["rank"]: j for j in jets}
    for n, j in enumerate(fj_jets):
        const = set(keys[c.user_index()] for c in j.constituents() if not c.is_pure_ghost())
        match = next((r for r, cset in sa_const.items() if cset == const), None)
        sa = sa_jets.get(match) if match is not None else None
        if sa is None:
            # the nearest standalone jet
            d, match = min(((math.hypot(sj["eta"] - j.eta(), (sj["phi"] - j.phi_std() + math.pi) % (2 * math.pi) - math.pi), r) for r, sj in sa_jets.items()), default=(9, None))
            sa = sa_jets.get(match)
            same = False
        else:
            same = True
            n_same += 1
        if sa is not None:
            dpt, da = j.pt() - sa["pt"], j.area() - sa["area"]
            max_dpt, max_da = max(max_dpt, abs(dpt)), max(max_da, abs(da))
            if n < 15:
                print(f"{n:4d} {j.pt():9.3f} {j.area():7.3f} {match:4d} {sa['pt']:9.3f} {sa['area']:7.3f} {dpt:8.4f} {da:7.3f} {len(const):4d} {sa['n']:5d}  {'yes' if same else 'NO'}")
            out.append(dict(fj=n, pt=j.pt(), area=j.area(), eta=j.eta(), phi=j.phi_std(), n=len(const), sa=match, pt_sa=sa["pt"], area_sa=sa["area"], same=same,
                            pt_sub_fj=j.pt() - rho_grid_fj * j.area(), pt_sub_sa=sa["pt_sub"]))
    print(f"{len(fj_jets)} FastJet jets above {sp.JET_PT_MIN:g} GeV against {len(jets)} standalone: {n_same} with identical constituents; max |dpT| = {max_dpt:.4f} GeV, "
          f"max |dA| = {max_da:.3f} (the ghosts are placed differently)")
    areas = np.array([o["area"] - o["area_sa"] for o in out if o["same"]])
    print(f"area differences over the identical jets: mean {areas.mean():+.4f}, r.m.s. {areas.std():.4f}")
    res = dict(n_fj=len(fj_jets), n_sa=len(jets), n_same=n_same, max_dpt=max_dpt, max_da=max_da, rho_grid_fj=rho_grid_fj, rho_grid_sa=rho_grid_sa, rho_grid_sa_y=grid_y, middle_cells=[float(srt[srt.size // 2 - 1]), float(srt[srt.size // 2])],
               rho_kt_fj=rho_kt_fj, rho_kt_sa=rho_kt_sa, jets=out)
    args.output.write_text(json.dumps(res, indent=1))
    print(f"wrote {args.output}")


def mode_library(args, pythia8):
    sp = load("sp", HERE / "standalone_pileup.py")
    lib = sp.Library(args.library)
    kinds = {}
    for e in lib.events:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    sig = dict(ND=55.42, SD=2 * 6.43, DD=8.85, CD=1.29)           # PYTHIA 8.315 with CP5 at 13.6 TeV (pileup_library.log); the 2025 CMSSW library prints the same
    tot = sum(sig.values())
    print(f"the library: {len(lib)} events; shares against PYTHIA's cross sections at 13.6 TeV (sigma_inel = {tot:.2f} mb):")
    print("CMSSW library (1000 events, companion): ND 722 = 72.2%, SD 161 = 16.1%, DD 105 = 10.5%, CD 12 = 1.2%")
    for k in ("ND", "SD", "DD", "CD"):
        n = kinds.get(k, 0)
        print(f"  {k}: {n:5d} = {100 * n / len(lib):5.1f}%  expected {100 * sig[k] / tot:5.1f}%  ({sig[k]:.2f} mb)")
    nf = np.array([e["nfinal"] for e in lib.events])
    nc = np.array([e["n_central"] for e in lib.events])
    nch = np.array([e["nch"] for e in lib.events])
    ptmax = np.array([e["pt_max"] for e in lib.events])
    sumpt = np.array([e["sumpt"] for e in lib.events])
    print(f"per collision: final particles {nf.mean():.1f} (median {np.median(nf):.0f}, min {nf.min()}, max {nf.max()}); charged {nch.mean():.1f}; "
          f"central charged (|eta| < 2.5, pT > 0.5) {nc.mean():.1f} (median {np.median(nc):.0f}, max {nc.max()}); hardest particle {ptmax.mean():.2f} GeV (max {ptmax.max():.1f}); "
          f"sum pT {sumpt.mean():.1f} (max {sumpt.max():.0f})")
    nc_all = np.array([int(((e["pt"] > sp.VISIBLE_PT) & (np.abs(e["eta"]) < sp.VISIBLE_ETA)).sum()) for e in lib.events])
    print(f"               central all particles (|eta| < 2.5, pT > 0.5) {nc_all.mean():.1f} (median {np.median(nc_all):.0f}, max {nc_all.max()})")
    print("CMSSW library (1000 events, companion): stable particles 187.7 (median 119, 3 to 857), charged 86.6, central all particles 20.4 (median 9, max 167), "
          "hardest 2.21 GeV (max 19.0), sum pT 76.5 (0.3 to 498)")
    for k in ("ND", "SD", "DD", "CD"):
        sel = np.array([e["kind"] == k for e in lib.events])
        if sel.any():
            print(f"  {k}: {nf[sel].mean():6.1f} particles, {nc[sel].mean():5.1f} central charged, {sumpt[sel].mean():6.1f} GeV of pT")
    print("  (CMSSW library per type, stable particles / central all species: ND 236 / 26, SD 70 / 7, DD 55 / 3, CD 24 / 2)")
    res = dict(n=len(lib), kinds=kinds, nfinal=dict(mean=float(nf.mean()), median=float(np.median(nf)), min=int(nf.min()), max=int(nf.max())),
               central=dict(mean=float(nc.mean()), median=float(np.median(nc)), max=int(nc.max())),
               central_all=dict(mean=float(nc_all.mean()), median=float(np.median(nc_all)), max=int(nc_all.max())), ptmax=float(ptmax.mean()), ptmax_max=float(ptmax.max()),
               sumpt=float(sumpt.mean()), sumpt_max=float(sumpt.max()), sumpt_min=float(sumpt.min()))
    args.output.write_text(json.dumps(res, indent=1))
    print(f"wrote {args.output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("record", "library"), default="record")
    ap.add_argument("-i", "--input", type=Path, default=HERE.parent / "event_PU.lhe")
    ap.add_argument("--library", type=Path, default=HERE.parent / "pileup_library.lhe.gz")
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()
    if args.output is None:
        args.output = HERE / f"validate_pileup_with_fastjet_{args.mode}.json"
    if args.mode == "record":
        import fastjet
        mode_record(args, fastjet)
    else:
        try:
            import pythia8mc as pythia8
        except ImportError:
            pythia8 = None
        mode_library(args, pythia8)


if __name__ == "__main__":
    main()
