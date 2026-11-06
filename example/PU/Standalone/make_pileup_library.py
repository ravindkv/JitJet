#!/usr/bin/env python3
"""The pileup library: minimum-bias proton-proton collisions at 13.6 TeV, generated once with PYTHIA 8 and
stored as a record, to be stacked on the b bbar event by standalone_pileup.py.

This is the generator-level twin of the GEN-SIM library that CMSSW's MixingModule reads (chapter 8):
the fragment of the Run-3 RelVal libraries, Configuration/Generator/MinBias_14TeV_pythia8_TuneCP5_cfi.py,
i.e. SoftQCD:inelastic (non-diffractive, single-, double- and central-diffractive collisions in their
natural proportions) with the CP5 tune and the CMS common settings (every particle with c tau < 10 mm
decayed, the longer-lived ones left for the detector), with the energy set to the sqrt(s) = 13.6 TeV of
the Run-3 signal as the 2024 library of the CMSSW companion does (process.generator.comEnergy = 13600).
Every collision is generated at the origin; the beam spot (where and when it happened) is drawn later by
the mixing, as CMSSW's VtxSmeared does. Nothing is ported here: this script only runs the official
package and writes what it produced, so that the standalone mixing needs NumPy and nothing else.

Output (../pileup_library.lhe.gz, read with gzip): a header with the settings and the cross sections
PYTHIA printed, then one block per collision,

  <event 1  code 101 ND  nfinal 174  nch 101  sumpt 32.1  nmpi 9  pthat 4.1>
      id         px         py         pz          e
     211      0.123     -0.456     12.345     12.357
  ...
  </event>

with every final particle (status 1: what the detector will see, neutrinos included). The process codes
are PYTHIA's: 101 ND, 103/104 SD (AB -> XB / AX), 105 DD, 106 CD; nmpi is the number of parton-parton
interactions of the collision (the hardest one plus the MPIs, PYTHIA's Info::nMPI) and pthat the transverse
momentum of the hardest, in GeV.

Needs the environment of example/ME/Standalone (pythia8mc, lhapdf):
  ../../ME/Standalone/.venv/bin/python make_pileup_library.py                 # 2000 events at 13.6 TeV, seed 20261
  ../../ME/Standalone/.venv/bin/python make_pileup_library.py --ecm 14000 -o lib14.lhe.gz   # the 14 TeV library of the RelVal workflows
"""
from __future__ import annotations

import argparse
import gzip
import math
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROCESS = {101: "ND", 103: "SD", 104: "SD", 105: "DD", 106: "CD"}


def quiet(fn):
    sys.stdout.flush()
    saved = os.dup(1)
    tmp = tempfile.TemporaryFile(mode="w+")
    os.dup2(tmp.fileno(), 1)
    try:
        res = fn()
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
    tmp.seek(0)
    return res, tmp.read()


def pdf_set() -> str:
    try:
        import lhapdf
        for d in lhapdf.paths():
            dat = Path(d) / "NNPDF31_nnlo_as_0118" / "NNPDF31_nnlo_as_0118_0000.dat"
            if dat.exists():
                return f"LHAGrid1:{dat}"
    except ImportError:
        pass
    return "13"       # PYTHIA's internal NNPDF2.3 QCD+QED LO: a fallback, not CP5's set


def settings(ecm: float, seed: int, pset: str):
    """MinBias_14TeV_pythia8_TuneCP5_cfi: the CMS common settings, the CP5 tune and SoftQCD:inelastic (Beams:eCM from comEnergy)."""
    return [
        f"Beams:eCM = {ecm}", "Beams:idA = 2212", "Beams:idB = 2212",
        "Random:setSeed = on", f"Random:seed = {seed}",
        # pythia8CommonSettings
        "Tune:preferLHAPDF = 2", "Main:timesAllowErrors = 10000", "Check:epTolErr = 0.01",
        "Beams:setProductionScalesFromLHEF = off", "SLHA:minMassSM = 1000.",
        "ParticleDecays:limitTau0 = on", "ParticleDecays:tau0Max = 10",
        # pythia8CP5Settings
        "Tune:pp 14", "Tune:ee 7", "MultipartonInteractions:ecmPow = 0.03344", "MultipartonInteractions:bProfile = 2",
        "MultipartonInteractions:pT0Ref = 1.41", "MultipartonInteractions:coreRadius = 0.7634",
        "MultipartonInteractions:coreFraction = 0.63", "ColourReconnection:range = 5.176", "SigmaTotal:zeroAXB = off",
        "SpaceShower:alphaSorder = 2", "SpaceShower:alphaSvalue = 0.118", "SigmaProcess:alphaSvalue = 0.118",
        "SigmaProcess:alphaSorder = 2", "MultipartonInteractions:alphaSvalue = 0.118", "MultipartonInteractions:alphaSorder = 2",
        "TimeShower:alphaSorder = 2", "TimeShower:alphaSvalue = 0.118", "SigmaTotal:mode = 0",
        "SigmaTotal:sigmaEl = 21.89", "SigmaTotal:sigmaTot = 100.309", f"PDF:pSet = {pset}",
        # processParameters
        "SoftQCD:inelastic = on",
        "Print:quiet = on", "Next:numberShowEvent = 0", "Next:numberCount = 0",
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--nevents", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261, help="PYTHIA's seed (the CMSSW library uses PUSEED = 20241)")
    ap.add_argument("--ecm", type=float, default=13600.0, help="sqrt(s) in GeV (13.6 TeV, the Run-3 energy of the signal)")
    ap.add_argument("-o", "--output", type=Path, default=HERE.parent / "pileup_library.lhe.gz")
    ap.add_argument("--log", type=Path, default=None, help="PYTHIA's initialisation and statistics (default <output stem>.log)")
    args = ap.parse_args()
    try:
        import pythia8mc as pythia8
    except ImportError:
        import pythia8
    pset = pdf_set()
    py = pythia8.Pythia("", False)
    for s in settings(args.ecm, args.seed, pset):
        py.readString(s)
    py.readString("Print:quiet = off")
    py.readString("Init:showChangedSettings = on")
    py.readString("Init:showProcesses = on")
    ok, init_text = quiet(lambda: py.init())
    if not ok:
        print(init_text)
        sys.exit("PYTHIA initialisation failed")
    info = py.infoPython()
    log = args.log or args.output.with_name(args.output.name.split(".")[0] + ".log")
    t0 = time.time()
    counts = {}
    nfinal_all = []
    opener = gzip.open if args.output.suffix == ".gz" else open
    with opener(args.output, "wt") as out:
        out.write(f"# Standalone Pileup Library: {args.nevents} minimum-bias pp collisions at sqrt(s) = {args.ecm / 1000:g} TeV, PYTHIA {py.parm('Pythia:versionNumber'):.3f}, "
                  f"SoftQCD:inelastic, tune CP5, PDF {pset.split(':')[0]}, seed {args.seed}; every collision at the origin, final particles (status 1) only\n")
        out.write("# process codes: 101 ND (non-diffractive), 103/104 SD (single diffractive), 105 DD (double diffractive), 106 CD (central diffractive)\n")
        out.write("# columns: id px py pz e [GeV]\n")
        for n in range(args.nevents):
            good, text = quiet(lambda: py.next())
            if not good:
                continue
            ev = py.event
            info = py.infoPython()          # a snapshot: ask again for every event
            code = info.code()
            kind = PROCESS.get(code, str(code))
            rows = []
            nch, sumpt = 0, 0.0
            for i in range(ev.size()):
                q = ev[i]
                if not q.isFinal():
                    continue
                rows.append((q.id(), q.px(), q.py(), q.pz(), q.e()))
                if q.isCharged():
                    nch += 1
                sumpt += q.pT()
            counts[kind] = counts.get(kind, 0) + 1
            nfinal_all.append(len(rows))
            out.write(f"<event {n + 1}  code {code} {kind}  nfinal {len(rows)}  nch {nch}  sumpt {sumpt:.3f}  nmpi {info.nMPI()}  pthat {info.pTHat():.2f}>\n")
            for pid, px, py_, pz, e in rows:
                out.write(f"{pid:8d} {px:11.4f} {py_:11.4f} {pz:11.4f} {e:11.4f}\n")
            out.write("</event>\n")
            if (n + 1) % 500 == 0:
                print(f"  {n + 1} events, {time.time() - t0:.0f} s", flush=True)
    stat, stat_text = quiet(lambda: py.stat())
    info = py.infoPython()
    sig = {}
    for code in (101, 103, 104, 105, 106):
        try:
            sig[code] = (info.sigmaGen(code), info.sigmaErr(code))
        except Exception:
            pass
    with open(log, "w") as f:
        f.write(init_text)
        f.write(stat_text)
        f.write("\n# per-process cross sections [mb] as PYTHIA estimated them from the generated events (sigmaGen, sigmaErr)\n")
        for code, (s, e) in sig.items():
            f.write(f"  code {code} {PROCESS.get(code, '')}: {s:.4f} +- {e:.4f} mb\n")
        f.write(f"  all: {info.sigmaGen():.4f} +- {info.sigmaErr():.4f} mb\n")
        f.write(f"\n# {sum(counts.values())} events written to {args.output}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + "\n")
        f.write(f"# final particles per event: mean {sum(nfinal_all) / max(1, len(nfinal_all)):.1f}, min {min(nfinal_all)}, max {max(nfinal_all)}\n")
    print(f"wrote {args.output} ({sum(counts.values())} events: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f") and {log}; "
          f"sigma_inel = {info.sigmaGen():.3f} +- {info.sigmaErr():.3f} mb; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
