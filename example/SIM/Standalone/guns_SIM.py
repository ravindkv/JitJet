#!/usr/bin/env python3
"""Single particles through Geant4 in the CMS geometry: the numbers behind three theory plots of chapter 10.

 1. radiography   photons and pi+ of 5 GeV, flat in |eta| < 3, stopped when they reach a calorimeter: which of
                  them converted or interacted in the tracker, and where. The vertices draw the material of the
                  tracker, as the vertices of data do in the papers of the experiments.
 2. photons       10 GeV photons at eta = 0.5: the share of the hottest, the 9 and the 25 hottest crystals, for
                  those that arrive unconverted and for those that converted in the tracker.
 3. pions         pi+ of 2 to 100 GeV at eta = 0.5: the light of the crystals plus the light of the scintillator
                  times the sampling factor of the barrel (125), over the energy of the pion.

Run with the interpreter that has Geant4 (about twenty minutes on 10 processes; the pions are the expensive part):

  .venv/bin/python guns_SIM.py            # writes guns_SIM.json, prints a summary (guns_SIM.log)
  .venv/bin/python guns_SIM.py --quick    # a tenth of the particles
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import standalone_SIM as sim  # noqa: E402
import geant4_SIM as g4sim  # noqa: E402

SF_HB = 125.0
PIONS = [(2.0, 80), (5.0, 80), (10.0, 60), (20.0, 50), (50.0, 30), (100.0, 20)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=max(1, min(10, (mp.cpu_count() or 2) - 2)))
    ap.add_argument("--physics", default="FTFP_BERT")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    args.no_field, args.worker_logs, args.no_birks, args.no_regions = False, None, False, False
    f = 0.1 if args.quick else 1.0
    out = {}
    t0 = time.time()

    # 1. the radiography: everything is stopped at the calorimeters
    args.trk_only = True
    out["radiography"] = {}
    for pid, name, fate in ((22, "photon", "conversion"), (211, "pion", "nuclear")):
        n = int(4000 * f)
        g = g4sim.run_gun(args, pid, 5.0, 0.0, n, seed=11, flat_eta=3.0)
        hit = np.array([ft == fate for ft in g["fate"]])
        v = np.array([e for e, h in zip(g["end"], hit) if h and e is not None], dtype=float).reshape(-1, 3)
        out["radiography"][name] = dict(n=n, energy=5.0, eta=np.round(g["eta"], 4).tolist(), lost=hit.astype(int).tolist(),
                                        vertices=np.round(v, 2).tolist(), fates={k: int(sum(ft == k for ft in g["fate"])) for k in set(g["fate"])})
        inside = np.abs(g["eta"]) < 2.5
        print(f"radiography: {n} {name}s of 5 GeV, flat in |eta| < 3: {int(hit.sum())} end with a {fate} before the calorimeters "
              f"({100 * hit.mean():.1f} %; {100 * hit[inside].mean():.1f} % inside |eta| < 2.5); fates {out['radiography'][name]['fates']}   [{time.time() - t0:.0f} s]")
    args.trk_only = False

    # 2. photons of 10 GeV into the barrel crystals
    n = int(300 * f)
    g = g4sim.run_gun(args, 22, 10.0, 0.5, n, seed=5)
    rows = {"unconverted": [], "converted": []}
    for cells, ft in zip(g["eb_cells"], g["fate"]):
        if len(cells) == 0:
            continue
        tot = cells.sum()
        cum = np.cumsum(cells)
        rows["converted" if ft == "conversion" else "unconverted"].append(
            [cells[0] / 10.0, cells[:9].sum() / 10.0, cells[:25].sum() / 10.0, tot / 10.0, int(np.searchsorted(cum, 0.9 * tot) + 1), len(cells)])
    out["photon10"] = {}
    for k, r in rows.items():
        r = np.array(r)
        out["photon10"][k] = dict(n=len(r), mean=r[:, :4].mean(axis=0).tolist(), rms=r[:, :4].std(axis=0).tolist(), n90=float(np.median(r[:, 4])), cells=float(np.median(r[:, 5])))
        print(f"10 GeV photons at eta 0.5, {k}: {len(r)}; hottest crystal {r[:, 0].mean():.3f}, 9 hottest {r[:, 1].mean():.3f}, 25 hottest {r[:, 2].mean():.3f}, "
              f"all barrel crystals {r[:, 3].mean():.3f}; 90 % of the deposit in {np.median(r[:, 4]):.0f} crystals   [{time.time() - t0:.0f} s]")

    # 3. pions into the barrel calorimeters
    out["pions"] = []
    for e, n in PIONS:
        n = max(4, int(n * f))
        g = g4sim.run_gun(args, 211, e, 0.5, n, seed=3)
        ok = np.array([ft == "calorimeter" for ft in g["fate"]])
        tot = g["ecal"] + SF_HB * g["hcal"]
        resp = tot[ok] / e
        share = g["ecal"][ok] / np.maximum(tot[ok], 1e-9)
        mip = g["ecal"][ok] < 0.6
        out["pions"].append(dict(e=e, n=n, reach=int(ok.sum()), response=float(resp.mean()), rms=float(resp.std()), ecal_share=float(share.mean()), mip=float(mip.mean()),
                                 ecal=float(g["ecal"][ok].mean()), scint=float(g["hcal"][ok].mean()), fates={k: int(sum(ft == k for ft in g["fate"])) for k in set(g["fate"])}))
        print(f"pi+ of {e:5.0f} GeV at eta 0.5: {int(ok.sum())} of {n} reach the calorimeters; (crystals + {SF_HB:.0f} x scintillator) / E = {resp.mean():.2f} +- {resp.std():.2f}; "
              f"crystals {100 * share.mean():.0f} % of it; {100 * mip.mean():.0f} % cross the crystals with less than 0.6 GeV; scintillator {1e3 * g['hcal'][ok].mean():.0f} MeV   [{time.time() - t0:.0f} s]")
    summ = json.loads((HERE / "geant4_SIM_signal.json").read_text())
    out["geant4"] = dict(version=summ["version"], physics=summ["physics"], geometry=g4sim.GDML.name)
    (HERE / "guns_SIM.json").write_text(json.dumps(out, indent=1))
    print(f"guns_SIM.json written, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
