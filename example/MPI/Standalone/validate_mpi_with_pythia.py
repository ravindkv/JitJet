#!/usr/bin/env python3
"""Add multiparton interactions, beam remnants and colour reconnection to the same hard event
with PYTHIA 8 and compare with the standalone script.

Needs the full environment of example/ME/Standalone (pythia8mc + LHAPDF):
  ../../ME/Standalone/.venv/bin/python validate_mpi_with_pythia.py -n 2000
  ../../ME/Standalone/.venv/bin/python validate_mpi_with_pythia.py -n 2000 --standalone-stats stats.stats.json
  ../../ME/Standalone/.venv/bin/python validate_mpi_with_pythia.py -n 500 --no-cr
  ../../ME/Standalone/.venv/bin/python validate_mpi_with_pythia.py --impact 0.3136 -n 2000

The hard process of ../../ME/event_ME.lhe is written N times into a Les Houches
file and handed to PYTHIA with the CP5 settings of the CMSSW configuration
(multiparton interactions, showers and colour reconnection on, hadronisation
off, QED radiation off). PYTHIA's own initialisation numbers (pT0, sigma_int,
sigma_ND) are read from its printout, and per event the number of
interactions, the impact parameter and its enhancement factor, the
underlying-event pT density in the transverse region of the leading b quark,
the pT that partons not belonging to the hard system add inside cones of
R = 0.4 and 0.8 around each b quark, the number of beam remnants and the
summed pT of the MPI partons are recorded and averaged. With
--standalone-stats FILE (the .stats.json written by
standalone_multiparton_interactions.py --repeat N --history FILE.json) both
sets of averages are printed side by side.

--impact B fixes the impact parameter b/<b> through a UserHooks object, as the
standalone script's --impact does, so that both can be compared at the same
collision geometry (the CMSSW event: b = 0.3136).

Two things worth knowing about this script: the tune must be set before the
PDF, because Tune:pp resets PDF:pSet; and PYTHIA writes its initialisation
through C++ streams, which Python's sys.stdout does not see, so the file
descriptor is redirected while init() runs.
"""
import argparse, importlib.util, json, math, os, sys, tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_lhef(ev, path: Path, n: int, scalup: float, sigma=345.1, dsigma=6.1):
    ebeam = ev.ebeam
    hard_in = [ev.inA, ev.inB]
    hard_out = [i for i, q in enumerate(ev.p) if q.status == 23]
    rows = []
    for i in hard_in + hard_out:
        q = ev.p[i]
        st = -1 if i in hard_in else 1
        m1, m2 = (0, 0) if st == -1 else (1, 2)
        rows.append(f"{q.id:5d} {st:3d} {m1:3d} {m2:3d} {q.col:4d} {q.acol:4d} "
                    f"{q.p[1]: .9e} {q.p[2]: .9e} {q.p[3]: .9e} {q.p[0]: .9e} {q.m: .9e} 0. 9.")
    event = (f"<event>\n{len(rows)} 1 1.0 {scalup:.6f} 7.837e-03 1.143e-01\n" + "\n".join(rows) + "\n</event>\n")
    with path.open("w") as f:
        f.write('<LesHouchesEvents version="3.0">\n<init>\n')
        f.write(f"2212 2212 {ebeam:.6e} {ebeam:.6e} 0 0 303600 303600 3 1\n{sigma:.4e} {dsigma:.4e} 1.0 1\n</init>\n")
        for _ in range(n):
            f.write(event)
        f.write("</LesHouchesEvents>\n")


def pythia_listing(e) -> str:
    title = "PYTHIA 8 Event Listing  (parton level: LHEF hard process + showers + MPI + beam remnants + CR)"
    out = [f" --------  {title}  " + "-" * max(10, 133 - len(title) - 14), " ",
           "    no         id  name            status     mothers   daughters     colours"
           "      p_x        p_y        p_z         e          m "]
    for i in range(e.size()):
        q = e[i]
        out.append(f"{i:6d}{q.id():11d}  {q.nameWithStatus(18):<18s}{q.status():4d}{q.mother1():6d}{q.mother2():6d}"
                   f"{q.daughter1():6d}{q.daughter2():6d}{q.col():6d}{q.acol():6d}{q.px():11.3f}{q.py():11.3f}"
                   f"{q.pz():11.3f}{q.e():11.3f}{q.m():11.3f}")
    out.append("")
    out.append(f" --------  End {title}  " + "-" * max(10, 133 - len(title) - 18))
    return "\n".join(out) + "\n"


def delta_r(a, b):
    dphi = abs(a.phi() - b.phi())
    dphi = 2 * math.pi - dphi if dphi > math.pi else dphi
    return math.hypot(a.eta() - b.eta(), dphi)


def origin(e, i: int) -> str:
    """The system a final parton belongs to: 'hard' (the b bbar system with all its ISR and FSR),
    'MPI' (a secondary scattering with its shower) or 'remnant' (status 63).

    Copies and shower products are walked up through their mothers. An ISR
    emission (43) hangs on the new incoming parton (-41) whose first daughter
    is the previous incoming parton of the same side, so the walk continues
    there; a -42 copy of the other incoming parton points to its original in
    the same way.
    """
    seen = set()
    while 0 < i < e.size() and i not in seen:
        seen.add(i)
        st = e[i].statusAbs()
        if st in (21, 23, 24):
            return "hard"
        if st in (31, 33, 34):
            return "MPI"
        if st == 63:
            return "remnant"
        if st == 43:
            m = e[i].mother1()                      # the new incoming parton; its other daughter is the old one
            d1, d2 = e[m].daughter1(), e[m].daughter2()
            i = d2 if d1 == i else d1
            continue
        if st in (41, 42, 45, 46):
            d1, d2 = e[i].daughter1(), e[i].daughter2()
            i = d1 if (d2 <= 0 or e[d1].statusAbs() != 43) else d2
            continue
        i = e[i].mother1()
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=HERE.parent.parent / "ME" / "event_ME.lhe")
    ap.add_argument("-n", "--nevents", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--impact", type=float, default=None,
                    help="keep only events whose b/<b> lies within --window of this value (CMSSW's event: 0.3136)")
    ap.add_argument("--window", type=float, default=0.03, help="half width of the b/<b> selection around --impact")
    ap.add_argument("--no-cr", action="store_true", help="ColourReconnection:reconnect = off")
    ap.add_argument("--no-mpi", action="store_true", help="PartonLevel:MPI = off (the chapter-2 situation)")
    ap.add_argument("--standalone-stats", type=Path, help=".stats.json from standalone_multiparton_interactions.py --repeat")
    ap.add_argument("--output", type=Path, default=HERE / "validate_mpi_with_pythia.json")
    ap.add_argument("--lhef", type=Path, default=HERE / "validate_mpi_with_pythia.lhe")
    ap.add_argument("--listing", type=Path, default=HERE / "validate_mpi_with_pythia_event1.txt")
    args = ap.parse_args()

    try:
        import pythia8mc as pythia8
    except ImportError:
        import pythia8
    sps = load("sps", HERE.parent.parent / "PS" / "Standalone" / "standalone_parton_showering.py")
    ev = sps.read_event(args.input)
    hard = [i for i, q in enumerate(ev.p) if q.status == 23]
    m2avg = sum(ev.p[i].m ** 2 for i in hard) / len(hard)
    scalup = math.sqrt(sps.pt(ev.p[hard[0]].p) ** 2 + m2avg)
    write_lhef(ev, args.lhef, args.nevents, scalup)

    pset = "20"
    try:
        import lhapdf
        for d in lhapdf.paths():
            dat = Path(d) / "NNPDF31_nnlo_as_0118" / "NNPDF31_nnlo_as_0118_0000.dat"
            if dat.exists():
                pset = f"LHAGrid1:{dat}"
    except ImportError:
        pass
    print(f"PDF:pSet = {pset}")

    py = pythia8.Pythia("", False)
    settings = ["Beams:frameType = 4", f"Beams:LHEF = {args.lhef}",
                # CP5 (CMSSW PythiaCP5Settings_cfi): the tune first, the PDF after it
                "Tune:pp = 14", "Tune:ee = 7",
                "MultipartonInteractions:ecmPow = 0.03344", "MultipartonInteractions:bProfile = 2",
                "MultipartonInteractions:pT0Ref = 1.41", "MultipartonInteractions:coreRadius = 0.7634",
                "MultipartonInteractions:coreFraction = 0.63", "ColourReconnection:range = 5.176",
                "SigmaTotal:zeroAXB = off", "SpaceShower:alphaSorder = 2", "SpaceShower:alphaSvalue = 0.118",
                "SigmaProcess:alphaSvalue = 0.118", "SigmaProcess:alphaSorder = 2",
                "MultipartonInteractions:alphaSvalue = 0.118", "MultipartonInteractions:alphaSorder = 2",
                "TimeShower:alphaSorder = 2", "TimeShower:alphaSvalue = 0.118",
                "SigmaTotal:mode = 0", "SigmaTotal:sigmaEl = 21.89", "SigmaTotal:sigmaTot = 100.309",
                f"PDF:pSet = {pset}",
                # this comparison: no hadronisation, no QED; MPI on, CR as asked
                "HadronLevel:all = off", f"PartonLevel:MPI = {'off' if args.no_mpi else 'on'}",
                f"ColourReconnection:reconnect = {'off' if args.no_cr else 'on'}",
                "TimeShower:QEDshowerByQ = off", "TimeShower:QEDshowerByL = off",
                "SpaceShower:QEDshowerByQ = off", "SpaceShower:QEDshowerByL = off",
                "Check:epTolErr = 0.01", "Next:numberCount = 0", "Next:numberShowEvent = 0",
                "Next:numberShowProcess = 0", "Next:numberShowInfo = 0", "Next:numberShowLHA = 0", "Init:showChangedSettings = off",
                "Random:setSeed = on", f"Random:seed = {args.seed}"]
    for s in settings:
        py.readString(s)
    # PYTHIA prints its MPI initialisation through C++ cout: catch it at the file-descriptor level
    saved = os.dup(1)
    tmp = tempfile.TemporaryFile(mode="w+")
    os.dup2(tmp.fileno(), 1)
    ok = py.init()
    sys.stdout.flush()
    os.dup2(saved, 1)
    tmp.seek(0)
    init_text = tmp.read()
    if not ok:
        print(init_text)
        sys.exit("PYTHIA initialisation failed")
    init = {}
    for ln in init_text.splitlines():
        if "sigmaNonDiffractive" in ln:
            init["sigma_nd"] = float(ln.split("=")[1].split("mb")[0])
        if "sigmaInteraction" in ln:
            init["pt0"] = float(ln.split("pT0 =")[1].split("gives")[0])
            init["sigma_int"] = float(ln.split("=")[-1].split("mb")[0])
    print("PYTHIA initialisation: " + ", ".join(f"{k} = {v}" for k, v in init.items()))

    keys = ["n_mpi", "b", "f", "n_final", "n_mpi_partons", "n_remnants", "sum_pt_mpi", "ue_transverse",
            "ue_density_all", "mpi_pt_max"]
    acc = {k: [] for k in keys}
    fam = {}
    for n in range(args.nevents):
        if not py.next():
            if py.infoPython().atEndOfFile():
                break
            continue
        e = py.event
        info = py.infoPython()
        if args.impact is not None and abs(info.bMPI() - args.impact) > args.window:
            continue
        finals = [i for i in range(e.size()) if e[i].isFinal()]
        org = {i: origin(e, i) for i in finals}
        acc["n_mpi"].append(info.nMPI())
        acc["b"].append(info.bMPI())
        acc["f"].append(info.enhanceMPI())
        acc["n_final"].append(len(finals))
        acc["n_mpi_partons"].append(sum(1 for i in finals if org[i] == "MPI"))
        acc["n_remnants"].append(sum(1 for i in finals if org[i] == "remnant"))
        acc["sum_pt_mpi"].append(sum(e[i].pT() for i in finals if org[i] == "MPI"))
        pts = [info.pTMPI(k) for k in range(1, info.nMPI())]
        acc["mpi_pt_max"].append(max(pts) if pts else 0.0)
        # the two b quarks: final-state copies of the hard ones
        stars = {}
        for i in range(e.size()):
            if e[i].statusAbs() == 23 and abs(e[i].id()) == 5:
                stars[e[i].id()] = e[i].iBotCopyId()
        lead = max(stars.values(), key=lambda j: e[j].pT())
        phi0 = e[lead].phi()
        regions = dict(toward=0.0, transverse=0.0, away=0.0)
        for i in finals:
            if org[i] == "hard" or abs(e[i].eta()) > 2.5:
                continue
            d = abs(e[i].phi() - phi0)
            d = 2 * math.pi - d if d > math.pi else d
            key = "toward" if d < math.pi / 3 else "transverse" if d < 2 * math.pi / 3 else "away"
            regions[key] += e[i].pT()
        acc["ue_transverse"].append(regions["transverse"] / (2 * 2.5 * 2 * math.pi / 3))
        acc["ue_density_all"].append(sum(regions.values()) / (2 * 2.5 * 2 * math.pi))
        for pid, j in stars.items():
            name = sps.pname(pid)
            d = fam.setdefault(name, {R: dict(pt_all=[], pt_hard=[], ue_scalar=[], n_ue=[], m_all=[], m_hard=[])
                                      for R in (0.4, 0.8)})
            for R in (0.4, 0.8):
                mem = [k for k in finals if delta_r(e[k], e[j]) < R]
                p_all = sum((np.array([e[k].e(), e[k].px(), e[k].py(), e[k].pz()]) for k in mem), np.zeros(4))
                p_hard = sum((np.array([e[k].e(), e[k].px(), e[k].py(), e[k].pz()]) for k in mem if org[k] == "hard"),
                             np.zeros(4))
                d[R]["pt_all"].append(math.hypot(p_all[1], p_all[2]))
                d[R]["pt_hard"].append(math.hypot(p_hard[1], p_hard[2]))
                d[R]["ue_scalar"].append(sum(e[k].pT() for k in mem if org[k] != "hard"))
                d[R]["n_ue"].append(sum(1 for k in mem if org[k] != "hard"))
                d[R]["m_all"].append(math.sqrt(max(p_all[0] ** 2 - p_all[1] ** 2 - p_all[2] ** 2 - p_all[3] ** 2, 0.0)))
                d[R]["m_hard"].append(math.sqrt(max(p_hard[0] ** 2 - p_hard[1] ** 2 - p_hard[2] ** 2 - p_hard[3] ** 2, 0.0)))
        if n == 0 and args.listing:
            args.listing.write_text(pythia_listing(e))
    saved = os.dup(1)
    os.dup2(tmp.fileno(), 1)
    py.stat()
    sys.stdout.flush()
    os.dup2(saved, 1)
    ndone = len(acc["n_mpi"])
    print(f"\nPYTHIA 8 processed the hard event {ndone} times (MPI {'off' if args.no_mpi else 'on'}, "
          f"CR {'off' if args.no_cr else 'on'}, b {'sampled' if args.impact is None else f'= {args.impact} <b>'}, "
          f"SCALUP = {scalup:.2f} GeV)")
    st = None
    if args.standalone_stats and args.standalone_stats.exists():
        st = json.loads(args.standalone_stats.read_text())
        print(f"standalone script: {st['repeat']} events from {args.standalone_stats}")
    head = f"{'quantity':22s}{'PYTHIA mean':>14s}{'std':>10s}"
    if st:
        head += f"{'standalone mean':>18s}{'std':>10s}{'diff/err':>10s}"
    print(head)

    def row(label, v_py, v_st):
        v_py = np.asarray(v_py, dtype=float)
        line = f"{label:22s}{v_py.mean():14.4f}{v_py.std():10.4f}"
        if v_st is not None and len(v_st):
            v_st = np.asarray(v_st, dtype=float)
            err = math.hypot(v_py.std() / math.sqrt(len(v_py)), v_st.std() / math.sqrt(len(v_st)))
            line += f"{v_st.mean():18.4f}{v_st.std():10.4f}{(v_st.mean() - v_py.mean()) / err if err > 0 else 0:10.2f}"
        print(line)

    for k in keys:
        row(k, acc[k], st["counts"].get(k) if st else None)
    for name, byR in fam.items():
        for R, d in byR.items():
            for q, label in (("pt_hard", "pT hard system"), ("pt_all", "pT all partons"), ("ue_scalar", "UE scalar pT"),
                             ("n_ue", "UE partons"), ("m_hard", "mass hard"), ("m_all", "mass all")):
                stv = None
                if st and name in st["families"]:
                    stv = st["families"][name].get(str(R), st["families"][name].get(R, {})).get(q)
                row(f"{name} R={R} {label}", d[q], stv)
    args.output.write_text(json.dumps(dict(nevents=ndone, seed=args.seed, scalup=scalup, init=init, impact=args.impact,
                                           no_cr=args.no_cr, counts=acc, families={n: {str(R): v for R, v in d.items()}
                                                                                   for n, d in fam.items()}),
                                      default=float) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
