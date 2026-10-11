#!/usr/bin/env python3
"""Hadronise with PYTHIA 8 and compare with the standalone Lund string script.

Needs the environment of example/ME/Standalone (pythia8mc, LHAPDF for --mode full):
  ../../ME/Standalone/.venv/bin/python validate_hadronisation_with_pythia.py -n 400                      # mode record
  ../../ME/Standalone/.venv/bin/python validate_hadronisation_with_pythia.py -n 400 --standalone-stats standalone_had_stats.stats.json
  ../../ME/Standalone/.venv/bin/python validate_hadronisation_with_pythia.py --mode full -n 2000
  ../../ME/Standalone/.venv/bin/python validate_hadronisation_with_pythia.py --mode full -n 8000 --impact 0.3136 --window 0.03

Two modes.

  record  PYTHIA hadronises exactly the parton configuration of the standalone record
          (../../MPI/event_MPI.lhe: the status-62 and status-63 partons with their colour
          tags) N times with different random seeds, with ProcessLevel:all = off, the Monash
          fragmentation parameters that CP5 keeps, and hadron decays off, so that only the
          hadronisation differs from standalone_hadronisation.py --repeat N on the same
          input. This is the sharpest comparison of this chapter: same strings, two Lund
          implementations.
  full    the hard event of ../../ME/event_ME.lhe is written into a Les Houches file N times
          and PYTHIA runs its full chain (showers, MPI, remnants, colour reconnection,
          hadronisation; decays off) with the CP5 settings of the CMSSW configuration, as the
          chapter-4 validation did at parton level. With --impact the events are selected in
          a window of b/<b> around the CMSSW value.

Per event the number of primary hadrons (all, charged, baryons, strange, central), the
species, and for each hard b quark its hadron (species, pT, E, the fractions x_E = E_B/E_b,
x_pT = pT_B/pT_b and the light-cone fraction x_+ along the quark), the distance between
quark and hadron, and the hadron-level cones of R = 0.4 and 0.8 around the quark direction
(pT, mass, multiplicity, the hadron's share) next to the parton-level cones are recorded and
averaged, in the same JSON layout as the standalone --repeat statistics, so that
plot_theory_Had.py can draw both.

Two things worth knowing: with ProcessLevel:all = off PYTHIA's next() runs the hadron level on
whatever the event record holds, so the record is filled by hand (main21.cpp style) and the
partons are given status 23; and PYTHIA's initialisation and its error messages go through
C++ streams, invisible to Python's sys.stdout, so the file descriptor is redirected while
init() and stat() run.
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


def quiet(fn):
    """Run fn() with the C++ stdout captured; returns (result, text)."""
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


def pythia_listing(e, title) -> str:
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


def delta_r(eta1, phi1, eta2, phi2):
    d = abs(phi1 - phi2)
    d = 2 * math.pi - d if d > math.pi else d
    return math.hypot(eta1 - eta2, d)


def heavy_content(pid: int) -> int:
    a = abs(pid)
    if a < 100:
        return 0
    digits = [int(c) for c in str(a)]
    quarks = digits[-4:-1] if a >= 1000 else digits[-3:-1]
    return 5 if 5 in quarks else 4 if 4 in quarks else 0


def is_strange(pid: int) -> bool:
    a = abs(pid)
    return a >= 100 and 3 in [int(c) for c in str(a)[:-1]]


def kin(q):
    return dict(pt=q.pT(), eta=q.eta(), phi=q.phi(), e=q.e(), p=np.array([q.e(), q.px(), q.py(), q.pz()]), m=q.m())


def cone(e, rows, k0, R):
    mem = [i for i in rows if delta_r(e[i].eta(), e[i].phi(), k0["eta"], k0["phi"]) < R]
    p = sum((np.array([e[i].e(), e[i].px(), e[i].py(), e[i].pz()]) for i in mem), np.zeros(4))
    m2 = p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2
    return mem, math.hypot(p[1], p[2]), math.sqrt(max(m2, 0.0))


def analyse(e, pd, b_rows, parton_rows):
    """One hadronised event: the summary in the layout of standalone_hadronisation.summarise()."""
    had = [i for i in range(e.size()) if e[i].isFinal() and e[i].isHadron()]
    s = dict(n_hadrons=len(had), n_charged=sum(1 for i in had if e[i].isCharged()),
             n_baryons=sum(1 for i in had if abs(e[i].id()) >= 1000),
             n_strange=sum(1 for i in had if is_strange(e[i].id())),
             n_hadrons_central=sum(1 for i in had if abs(e[i].eta()) < 2.5),
             n_charged_central=sum(1 for i in had if abs(e[i].eta()) < 2.5 and e[i].isCharged()),
             n_failed=0, species={}, b={})
    for i in had:
        n = e[i].name()
        s["species"][n] = s["species"].get(n, 0) + 1
    # the strings: hadrons grouped by their mother range (the 7x copies of the string's partons), with the mass
    groups = {}
    for i in had:
        groups.setdefault((e[i].mother1(), e[i].mother2()), []).append(i)
    s["strings"] = []
    for (a, b), hs in groups.items():
        p = np.zeros(4)
        for j in range(a, max(a, b) + 1):
            if e[j].statusAbs() in (71, 72, 73):
                p += np.array([e[j].e(), e[j].px(), e[j].py(), e[j].pz()])
        s["strings"].append((math.sqrt(max(p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2, 0.0)), len(hs)))
    for pid, jb in b_rows.items():
        qb = kin(e[jb])
        # the hadrons of the string that holds this quark: mothers span the 7x copies of the string
        jcopy = e[jb].iBotCopyId()
        cands = [i for i in had if e[i].mother1() <= jcopy <= max(e[i].mother2(), e[i].mother1())]
        want = (lambda h: (h < 0) if abs(h) < 1000 else (h > 0)) if pid > 0 else (lambda h: (h > 0) if abs(h) < 1000 else (h < 0))
        bh = [i for i in cands if heavy_content(e[i].id()) == 5 and want(e[i].id())]
        if not bh:
            bh = [i for i in had if heavy_content(e[i].id()) == 5 and want(e[i].id())]
        if not bh:
            continue
        ih = min(bh, key=lambda i: delta_r(e[i].eta(), e[i].phi(), qb["eta"], qb["phi"]))
        hb = kin(e[ih])
        nb = qb["p"][1:] / np.linalg.norm(qb["p"][1:])
        lc_b = qb["p"][0] + float(np.dot(qb["p"][1:], nb))
        lc_h = hb["p"][0] + float(np.dot(hb["p"][1:], nb))
        name = "b" if pid > 0 else "bbar"
        rec = dict(quark=name, pt_quark=qb["pt"], e_quark=qb["e"], hadron=e[ih].name(), hadron_id=e[ih].id(),
                   pt_hadron=hb["pt"], e_hadron=hb["e"], m_hadron=hb["m"],
                   dr=delta_r(hb["eta"], hb["phi"], qb["eta"], qb["phi"]),
                   x_e=hb["e"] / qb["e"], x_pt=hb["pt"] / qb["pt"], x_plus=lc_h / lc_b, z_drawn=None,
                   rank=1 if cands and ih in cands else 0, string_n=len(cands), cones={})
        for R in (0.4, 0.8):
            mem_h, pt_h, m_h = cone(e, had, qb, R)
            mem_p, pt_p, m_p = cone(e, parton_rows, qb, R)
            mem_hb, pt_hb, m_hb = cone(e, had, hb, R)
            rec["cones"][R] = dict(n_hadrons=len(mem_h), pt_hadrons=pt_h, m_hadrons=m_h, n_partons=len(mem_p), pt_partons=pt_p,
                                   m_partons=m_p, n_hadrons_B=len(mem_hb), pt_hadrons_B=pt_hb, m_hadrons_B=m_hb,
                                   n_charged=sum(1 for i in mem_h if e[i].isCharged()),
                                   frac_B=hb["pt"] / pt_h if pt_h > 0 else 0.0)
        s["b"][name] = rec
    return s


def accumulate(acc, fam, species, bspecies, s, strings=None):
    for k in acc:
        acc[k].append(s[k])
    if strings is not None:
        strings.append(s["strings"])
    for name, k in s["species"].items():
        species[name] = species.get(name, 0) + k
    for name, r in s["b"].items():
        d = fam.setdefault(name, dict(x_e=[], x_pt=[], x_plus=[], pt_hadron=[], e_hadron=[], dr=[], z_drawn=[], rank=[],
                                      cones={R: dict(pt_hadrons=[], m_hadrons=[], n_hadrons=[], pt_partons=[], frac_B=[],
                                                     pt_hadrons_B=[], n_charged=[]) for R in (0.4, 0.8)}))
        for k in ("x_e", "x_pt", "x_plus", "pt_hadron", "e_hadron", "dr", "rank"):
            d[k].append(r[k])
        d["z_drawn"].append(float("nan"))
        bspecies.setdefault(name, {})
        bspecies[name][r["hadron"]] = bspecies[name].get(r["hadron"], 0) + 1
        for R in (0.4, 0.8):
            for k in d["cones"][R]:
                d["cones"][R][k].append(r["cones"][R][k])


COMMON = ["Tune:pp = 14", "Tune:ee = 7", "HadronLevel:Decay = off",
          "Next:numberCount = 0", "Next:numberShowEvent = 0", "Next:numberShowProcess = 0", "Next:numberShowInfo = 0",
          "Next:numberShowLHA = 0", "Init:showChangedSettings = off", "Init:showProcesses = off",
          "Init:showMultipartonInteractions = off", "Random:setSeed = on"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("record", "full"), default="record")
    ap.add_argument("-i", "--input", type=Path, default=None, help="record: ../../MPI/event_MPI.lhe; full: ../../ME/event_ME.lhe")
    ap.add_argument("-n", "--nevents", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--impact", type=float, default=None, help="full mode: keep events with |b/<b> - impact| < window")
    ap.add_argument("--window", type=float, default=0.03)
    ap.add_argument("--no-bowler", action="store_true", help="StringZ:rFactB = 0 (plain Lund for the b quark)")
    ap.add_argument("--peterson", action="store_true", help="StringZ:usePetersonB/C = on")
    ap.add_argument("--standalone-stats", type=Path, help=".stats.json from standalone_hadronisation.py --repeat")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--listing", type=Path, default=None)
    args = ap.parse_args()
    if args.input is None:
        args.input = HERE.parent.parent / ("MPI" if args.mode == "record" else "ME") / ("event_MPI.lhe" if args.mode == "record" else "event_ME.lhe")
    if args.output is None:
        args.output = HERE / f"validate_hadronisation_with_pythia_{args.mode}.json"
    if args.listing is None:
        args.listing = HERE / f"validate_hadronisation_with_pythia_{args.mode}_event1.txt"

    try:
        import pythia8mc as pythia8
    except ImportError:
        import pythia8
    sh = load("sh", HERE / "standalone_hadronisation.py")

    py = pythia8.Pythia("", False)
    settings = list(COMMON) + [f"Random:seed = {args.seed}"]
    if args.no_bowler:
        settings.append("StringZ:rFactB = 0.")
    if args.peterson:
        settings += ["StringZ:usePetersonB = on", "StringZ:usePetersonC = on"]
    keys = ["n_hadrons", "n_charged", "n_baryons", "n_strange", "n_hadrons_central", "n_charged_central", "n_failed"]
    acc = {k: [] for k in keys}
    fam, species, bspecies = {}, {}, {}
    strings = []
    extra = {}

    if args.mode == "record":
        ev = sh.read_record(args.input)
        b_in = sh.b_quark_rows(ev)                        # id -> row in the standalone record
        fin = ev.final_indices()
        settings += ["ProcessLevel:all = off", "Check:event = on"]
        for s_ in settings:
            py.readString(s_)
        ok, init_text = quiet(lambda: py.init())
        if not ok:
            print(init_text)
            sys.exit("PYTHIA initialisation failed")
        print(f"PYTHIA 8 hadronises the {len(fin)} final partons of {args.input.name} {args.nevents} times "
              f"(Monash fragmentation, decays off{', rFactB = 0' if args.no_bowler else ''}{', Peterson' if args.peterson else ''})")
        ndone = 0
        for n in range(args.nevents):
            py.event.reset()
            index = {}
            for i in fin:
                q = ev.p[i]
                k = py.event.append(q.id, 23, 0, 0, 0, 0, q.col, q.acol, float(q.p[1]), float(q.p[2]), float(q.p[3]), float(q.p[0]), q.m)
                index[i] = k
            good, text = quiet(lambda: py.next())
            if not good:
                print(f"  event {n}: PYTHIA could not hadronise the record: " + " ".join(text.split("\n")[-3:])[:200])
                continue
            e = py.event
            b_rows = {pid: index[row] for pid, row in b_in.items()}
            parton_rows = list(index.values())
            s = analyse(e, py.particleData, b_rows, parton_rows)
            accumulate(acc, fam, species, bspecies, s, strings)
            ndone += 1
            if n == 0:
                args.listing.write_text(pythia_listing(e, "PYTHIA 8 Event Listing  (the standalone partons of event_MPI.lhe hadronised by PYTHIA)"))
        extra = dict(n_partons=len(fin))
    else:
        mpi = load("mpi_val", HERE.parent.parent / "MPI" / "Standalone" / "validate_mpi_with_pythia.py")
        sps = load("sps", HERE.parent.parent / "PS" / "Standalone" / "standalone_parton_showering.py")
        ev = sps.read_event(args.input)
        hard = [i for i, q in enumerate(ev.p) if q.status == 23]
        m2avg = sum(ev.p[i].m ** 2 for i in hard) / len(hard)
        scalup = math.sqrt(sps.pt(ev.p[hard[0]].p) ** 2 + m2avg)
        lhef = HERE / "validate_hadronisation_with_pythia.lhe"
        mpi.write_lhef(ev, lhef, args.nevents, scalup)
        pset = "20"
        try:
            import lhapdf
            for d in lhapdf.paths():
                dat = Path(d) / "NNPDF31_nnlo_as_0118" / "NNPDF31_nnlo_as_0118_0000.dat"
                if dat.exists():
                    pset = f"LHAGrid1:{dat}"
        except ImportError:
            pass
        settings += ["Beams:frameType = 4", f"Beams:LHEF = {lhef}",
                     "MultipartonInteractions:ecmPow = 0.03344", "MultipartonInteractions:bProfile = 2",
                     "MultipartonInteractions:pT0Ref = 1.41", "MultipartonInteractions:coreRadius = 0.7634",
                     "MultipartonInteractions:coreFraction = 0.63", "ColourReconnection:range = 5.176",
                     "SigmaTotal:zeroAXB = off", "SpaceShower:alphaSorder = 2", "SpaceShower:alphaSvalue = 0.118",
                     "SigmaProcess:alphaSvalue = 0.118", "SigmaProcess:alphaSorder = 2",
                     "MultipartonInteractions:alphaSvalue = 0.118", "MultipartonInteractions:alphaSorder = 2",
                     "TimeShower:alphaSorder = 2", "TimeShower:alphaSvalue = 0.118",
                     "SigmaTotal:mode = 0", "SigmaTotal:sigmaEl = 21.89", "SigmaTotal:sigmaTot = 100.309",
                     f"PDF:pSet = {pset}", "TimeShower:QEDshowerByQ = off", "TimeShower:QEDshowerByL = off",
                     "SpaceShower:QEDshowerByQ = off", "SpaceShower:QEDshowerByL = off", "Check:epTolErr = 0.01"]
        for s_ in settings:
            py.readString(s_)
        ok, init_text = quiet(lambda: py.init())
        if not ok:
            print(init_text)
            sys.exit("PYTHIA initialisation failed")
        print(f"PYTHIA 8 runs the full chain on the hard event {args.nevents} times (CP5, decays off, PDF {pset.split(':')[0]})")
        ndone = 0
        for n in range(args.nevents):
            good, text = quiet(lambda: py.next())
            if not good:
                if py.infoPython().atEndOfFile():
                    break
                continue
            info = py.infoPython()
            if args.impact is not None and abs(info.bMPI() - args.impact) > args.window:
                continue
            e = py.event
            b_rows = {}
            for i in range(e.size()):
                if e[i].statusAbs() == 23 and abs(e[i].id()) == 5:
                    j = i
                    while e[j].daughter1() > 0 and e[e[j].daughter1()].id() == e[j].id() and e[e[j].daughter1()].statusAbs() < 70:
                        j = e[j].daughter1()
                    b_rows[e[i].id()] = j
            parton_rows = [i for i in range(e.size()) if e[i].statusAbs() in (62, 63)]
            s = analyse(e, py.particleData, b_rows, parton_rows)
            accumulate(acc, fam, species, bspecies, s)
            ndone += 1
            if ndone == 1:
                args.listing.write_text(pythia_listing(e, "PYTHIA 8 Event Listing  (full chain: LHEF hard process + showers + MPI + CR + hadronisation)"))
        extra = dict(scalup=scalup, impact=args.impact, window=args.window)
    _, stat_text = quiet(lambda: py.stat())

    print(f"\n{ndone} events analysed")
    st = None
    if args.standalone_stats and args.standalone_stats.exists():
        st = json.loads(args.standalone_stats.read_text())
        print(f"standalone script: {st['repeat']} hadronisations from {args.standalone_stats}")
    head = f"{'quantity':30s}{'PYTHIA mean':>14s}{'std':>10s}"
    if st:
        head += f"{'standalone mean':>18s}{'std':>10s}{'diff/err':>10s}"
    print(head)

    def row(label, v_py, v_st):
        v_py = np.asarray(v_py, dtype=float)
        v_py = v_py[~np.isnan(v_py)]
        line = f"{label:30s}{v_py.mean():14.4f}{v_py.std():10.4f}"
        if v_st is not None and len(v_st):
            v_st = np.asarray(v_st, dtype=float)
            v_st = v_st[~np.isnan(v_st)]
            err = math.hypot(v_py.std() / math.sqrt(len(v_py)), v_st.std() / math.sqrt(len(v_st)))
            line += f"{v_st.mean():18.4f}{v_st.std():10.4f}{(v_st.mean() - v_py.mean()) / err if err > 0 else 0:10.2f}"
        print(line)

    for k in keys:
        row(k, acc[k], st["counts"].get(k) if st else None)
    for name, d in fam.items():
        for q in ("x_e", "x_pt", "x_plus", "pt_hadron", "dr"):
            row(f"{name} {q}", d[q], st["b"][name][q] if st and name in st["b"] else None)
        for R in (0.4, 0.8):
            for q in ("pt_partons", "pt_hadrons", "m_hadrons", "n_hadrons", "n_charged", "frac_B"):
                stv = st["b"][name]["cones"][str(R)][q] if st and name in st["b"] else None
                row(f"{name} R={R} {q}", d["cones"][R][q], stv)
        tot = sum(bspecies[name].values())
        print(f"  {name} hadrons (PYTHIA): " + ", ".join(f"{h} {100 * k / tot:.1f}%" for h, k in sorted(bspecies[name].items(), key=lambda kv: -kv[1])))
        if st and name in st.get("b_species", {}):
            tot_s = sum(st["b_species"][name].values())
            print(f"  {name} hadrons (standalone): " + ", ".join(f"{h} {100 * k / tot_s:.1f}%" for h, k in sorted(st["b_species"][name].items(), key=lambda kv: -kv[1])))
    tot = sum(species.values())
    print("species (PYTHIA, % of all primary hadrons): " + ", ".join(f"{h} {100 * k / tot:.1f}" for h, k in sorted(species.items(), key=lambda kv: -kv[1])[:16]))
    if st:
        tot_s = sum(st["species"].values())
        print("species (standalone): " + ", ".join(f"{h} {100 * k / tot_s:.1f}" for h, k in sorted(st["species"].items(), key=lambda kv: -kv[1])[:16]))
    args.output.write_text(json.dumps(dict(mode=args.mode, nevents=ndone, seed=args.seed, no_bowler=args.no_bowler, peterson=args.peterson,
                                           counts=acc, species=species, b=fam, b_species=bspecies, strings=strings, **extra), default=float) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
