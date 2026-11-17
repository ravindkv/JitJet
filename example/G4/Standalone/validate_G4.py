#!/usr/bin/env python3
"""Validation of the standalone hand-over (standalone_G4.py) with two official packages, on exactly our particles.

1. ROOT's geometry package (TGeo, ROOT 6.32), the navigator of the virtual Monte Carlo: the two cylinders of
   Generator.cc are built as TGeo volumes, a tube of radius RDecLenCut = 2.9 cm and half-length Z_lmax = 354.8 cm
   inside a tube of the same radius and half-length Z_hector = 52.8 m, and every stable particle of the window
   (the signal, the 63 in-time and the 897 out-of-time collisions) is started at its production vertex in the
   direction of its momentum. TGeo's navigator steps to the next boundary and says where the straight line
   leaves each tube: through its end cap (the line is still inside r < 2.9 cm at that |z|: the eta cut drops
   the particle, or the very-forward rule applies) or through its side. With the momentum cut, the list of
   very-forward species (n, nbar, gamma), the neutrinos of the StackingAction and the PPS selection written
   again here, the decision of the gate is rebuilt for every particle and compared with the script's code.
2. PYTHIA 8.315 (pythia8mc), ParticleData: which species the generator decays with the CMSSW settings
   (ParticleDecays:limitTau0 = on, tau0Max = 10 mm): every species the record keeps stable must be one PYTHIA
   leaves to Geant4 (no decay, or tau0 > 10 mm), every species it decayed one PYTHIA decays.

Run with the ME virtual environment and ROOT's Python path:

  PYTHONPATH=$(root-config --libdir) ../../ME/Standalone/.venv/bin/python validate_G4.py --log validate_G4.log --summary validate_G4.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import standalone_G4 as core  # noqa: E402

dec = core.dec


def tgeo_world(r: float, z_lmax: float, z_hector: float):
    import ROOT
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    geo = ROOT.TGeoManager("gate", "the two cylinders of Generator.cc")
    ROOT.SetOwnership(geo, False)                     # ROOT owns the manager (gGeoManager); Python must not delete it
    mat = ROOT.TGeoMaterial("vacuum", 0, 0, 0)
    vac = ROOT.TGeoMedium("vacuum", 1, mat)
    for obj in (mat, vac):
        ROOT.SetOwnership(obj, False)
    world = geo.MakeBox("WORLD", vac, 3 * r, 3 * r, 2 * z_hector)
    geo.SetTopVolume(world)
    hector = geo.MakeTube("HECTOR", vac, 0.0, r, z_hector)       # r < 2.9 cm, |z| < 52.8 m
    cut = geo.MakeTube("ETACUT", vac, 0.0, r, z_lmax)            # r < 2.9 cm, |z| < 354.8 cm
    hector.AddNode(cut, 1)
    world.AddNode(hector, 1)
    geo.CloseGeometry()
    geo.SetVerboseLevel(0)
    return geo


def exits(geo, x, y, z, px, py, pz):
    """Step the straight line out of ETACUT and then out of HECTOR; return the two exit points (x, y, z)."""
    p = math.sqrt(px * px + py * py + pz * pz)
    geo.InitTrack(x, y, z, px / p, py / p, pz / p)
    out = []
    for name in ("ETACUT", "HECTOR"):
        # step until we leave the named volume
        for _ in range(4):
            cur = geo.GetCurrentVolume().GetName()
            geo.FindNextBoundaryAndStep()
            if cur == name:
                break
        pt = geo.GetCurrentPoint()
        out.append((pt[0], pt[1], pt[2]))
    return out


def run_tgeo(st, log):
    c, g, G = st.crowd, st.g, st.gate
    geo = tgeo_world(G.r, G.z_lmax, G.z_hector)
    n = len(c)
    eta_fail = np.zeros(n, bool)
    very_forward = np.zeros(n, bool)
    t0 = time.time()
    tol = 1e-6
    for j in range(n):
        x, y, z = c.vtx[j]
        E, px, py, pz = c.p4[j]
        (x1, y1, z1), (x2, y2, z2) = exits(geo, x, y, z, px, py, pz)
        eta_fail[j] = abs(abs(z1) - G.z_lmax) < tol and math.hypot(x1, y1) < G.r
        very_forward[j] = abs(abs(z2) - G.z_hector) < tol * 100 and math.hypot(x2, y2) <= G.r
    dt = time.time() - t0
    # the gate rebuilt from TGeo's two answers
    p_fail = ~very_forward & ((c.p < G.p_min) | (c.p > core.P_MAX))
    vf_ok = very_forward & np.isin(c.ids, (2112, -2112, 22))
    std = ~very_forward
    cut = std & ~p_fail & eta_fail
    added = vf_ok | (std & ~p_fail & ~cut)
    primary = added & ~c.nu
    pps = (c.ids == 2212) & (np.abs(c.eta) > core.PPS_ETA) & (np.abs(c.p4[:, 3]) > core.PPS_PZ)
    code = np.full(n, "?", dtype="<U1")
    code[primary & ~very_forward] = "P"
    code[primary & very_forward] = "V"
    code[added & c.nu] = "N"
    code[very_forward & ~vf_ok] = "F"
    code[p_fail] = "M"
    code[cut] = "E"
    code[pps] = "R"
    same = code == g["code"]
    res = dict(n=n, seconds=dt, identical=int(same.sum()), different=int((~same).sum()),
               eta_line_tgeo=int(cut.sum()), eta_line_script=int(g["eta_fail"].sum()),
               very_forward_tgeo=int(very_forward.sum()), very_forward_script=int(g["very_forward"].sum()),
               codes={k: int(v) for k, v in Counter(code.tolist()).items()},
               by_block={b: dict(n=int((c.block == i).sum()), identical=int((same & (c.block == i)).sum()),
                                 primaries=int((primary & (c.block == i)).sum())) for i, b in enumerate(core.BLOCKS)})
    # how close does a line come to the edge? the smallest distance |r - R| at |z| = Z_lmax among the decided particles
    near = np.abs(g["r_line"] - G.r)
    j = int(np.argmin(near))
    res["closest"] = dict(block=core.BLOCKS[c.block[j]], k=int(c.k[j]), id=int(c.ids[j]), eta=float(c.eta[j]), r_line=float(g["r_line"][j]), code=str(g["code"][j]))
    print(f"-- 1. TGeo (ROOT {__import__('ROOT').gROOT.GetVersion()}): tubes r < {G.r} cm, |z| < {G.z_lmax:.1f} cm inside |z| < {G.z_hector:.1f} cm; "
          f"{n} stable particles navigated in {dt:.1f} s", file=log)
    print(f"   eta cut (line leaves through the end cap at {G.z_lmax:.1f} cm): TGeo {res['eta_line_tgeo']}, script {res['eta_line_script']}", file=log)
    print(f"   very forward (line leaves through the end cap at {G.z_hector / 100:.1f} m): TGeo {res['very_forward_tgeo']}, script {res['very_forward_script']}", file=log)
    for b, v in res["by_block"].items():
        print(f"   {b:<7s}: {v['n']:7d} particles, decision identical for {v['identical']}, {v['primaries']} primaries", file=log)
    print(f"   decisions: {res['identical']} identical, {res['different']} different; codes " + ", ".join(f"{k} {v}" for k, v in sorted(res['codes'].items())), file=log)
    diff = np.where(~same)[0]
    res["differences"] = [dict(block=core.BLOCKS[c.block[j]], k=int(c.k[j]), id=int(c.ids[j]), px=float(c.p4[j, 1]), py=float(c.p4[j, 2]), pz=float(c.p4[j, 3]),
                               script=str(g["code"][j]), tgeo=str(code[j])) for j in diff]
    for d in res["differences"]:
        print(f"     different: {d['block']} k={d['k']} id {d['id']} p = ({d['px']:+.4f}, {d['py']:+.4f}, {d['pz']:+.4f}) GeV: script {d['script']}, TGeo {d['tgeo']}"
              + (" (pz rounded to zero in the library: Generator.cc keeps the line at its vertex, minTan, and its eta test drops it; TGeo leaves through the side)" if d["pz"] == 0 else ""), file=log)
    cl = res["closest"]
    print(f"   the closest call: {cl['block']} k={cl['k']} id {cl['id']} eta {cl['eta']:+.5f}: the line at r = {cl['r_line']:.6f} cm ({abs(cl['r_line'] - G.r) * 1e4:.2f} um from the edge), {cl['code']}", file=log)
    return res


def run_pythia(st, log):
    import pythia8mc as pythia8
    py = pythia8.Pythia("", False)
    pd = py.particleData
    tau0_max = core.TAU0_MAX

    def decays(pid):
        # ParticleDecays: a particle is decayed if it has a decay table (canDecay), may decay, and, with limitTau0 on, tau0 <= tau0Max
        return bool(pd.canDecay(pid)) and bool(pd.mayDecay(pid)) and pd.tau0(pid) <= tau0_max

    c = st.crowd
    stable = Counter(int(i) for i in c.ids)
    decayed = Counter(st.ev.p[r].id for r in core.hepmc_status2(st.ev))
    bad_stable = {pid: n for pid, n in stable.items() if decays(pid)}
    bad_decayed = {pid: n for pid, n in decayed.items() if not decays(pid)}
    left = sorted({abs(p) for p in stable}, key=lambda a: -pd.tau0(a) if pd.tau0(a) > 0 else 0)
    rows = []
    for a in sorted({abs(p) for p in stable}):
        rows.append(dict(id=a, name=pd.name(a), tau0=float(pd.tau0(a)), mayDecay=bool(pd.mayDecay(a)), canDecay=bool(pd.canDecay(a)), n=sum(n for p, n in stable.items() if abs(p) == a)))
    drows = []
    for a in sorted({abs(p) for p in decayed}):
        drows.append(dict(id=a, name=pd.name(a), tau0=float(pd.tau0(a)), n=sum(n for p, n in decayed.items() if abs(p) == a)))
    res = dict(version="8.315", tau0_max=tau0_max, stable_species=rows, decayed_species=drows, bad_stable=bad_stable, bad_decayed=bad_decayed)
    print(f"-- 2. PYTHIA 8.315 ParticleData with ParticleDecays:limitTau0 = on, tau0Max = {tau0_max:.0f} mm", file=log)
    print(f"   {len(rows)} stable species in the window ({len(c)} particles), {len(drows)} species decayed by the generator in the signal "
          f"({sum(decayed.values())} decays)", file=log)
    print("   left to Geant4 (stable in the record): " + ", ".join(f"{r['name']} {r['tau0']:.4g} mm" for r in sorted(rows, key=lambda r: r['tau0']) if r['tau0'] > 0), file=log)
    print("   decayed by the generator (signal): " + ", ".join(f"{r['name']} {r['tau0']:.3g}" for r in sorted(drows, key=lambda r: -r['tau0']) if r['tau0'] > 1e-4), file=log)
    print(f"   stable species PYTHIA would decay: {len(bad_stable)}; decayed species PYTHIA would not: {len(bad_decayed)}", file=log)
    longest = max((r for r in drows), key=lambda r: r["tau0"])
    shortest = min((r for r in rows if r["tau0"] > 0), key=lambda r: r["tau0"])
    print(f"   the longest-lived decayed: {longest['name']} ({longest['tau0']:.3g} mm); the shortest-lived left to Geant4: {shortest['name']} ({shortest['tau0']:.4g} mm)", file=log)
    res["longest_decayed"], res["shortest_left"] = longest, shortest
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=HERE.parent.parent / "PU" / "event_PU.lhe")
    ap.add_argument("-l", "--library", type=Path, default=HERE.parent.parent / "PU" / "pileup_library.lhe.gz")
    ap.add_argument("--log", type=Path)
    ap.add_argument("--summary", type=Path)
    a = ap.parse_args()
    args = argparse.Namespace(input=a.input, library=a.library, eta_cut=core.ETA_CUT, eta_hector=core.ETA_HECTOR, r_dec=core.R_DEC, p_min=core.P_MIN)
    log = open(a.log, "w") if a.log else sys.stdout
    st = core.Stage(args, verbose=0)
    print(f"== validate_G4: the hand-over of {a.input.name} (+ the out-of-time collisions of {a.library.name}) against official packages", file=log)
    res = dict(tgeo=run_tgeo(st, log), pythia=run_pythia(st, log))
    if a.summary:
        a.summary.write_text(json.dumps(core.to_jsonable(res), indent=1))
    if a.log:
        log.close()
        print(Path(a.log).read_text())


if __name__ == "__main__":
    main()
