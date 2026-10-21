#!/usr/bin/env python3
"""Cluster with FastJet and with PYTHIA's SlowJet, and compare with the standalone GenJet script.

Needs the environment of example/ME/Standalone (fastjet, pythia8mc; LHAPDF for --mode full):
  PY=../../ME/Standalone/.venv/bin/python
  $PY validate_genjets_with_fastjet.py                                   # mode record: the 749 particles of event_Decay.lhe
  $PY validate_genjets_with_fastjet.py --mode full -n 2000 --standalone-stats standalone_genjet_stats.stats.json
  $PY validate_genjets_with_fastjet.py --mode full -n 8000 --impact 0.3136 --window 0.03 --output validate_genjets_with_fastjet_full_fixedb.json

Two modes.

  record  FastJet 3 (what CMSSW's FastjetJetProducer calls) and PYTHIA 8's SlowJet (its own anti-kT, with the
          fjcore of FastJet inside) cluster exactly the final particles of the standalone record
          (../../Decay/event_Decay.lhe), with the ghosts of the b hadrons, c hadrons and partons added as the
          standalone script adds them, for the three collections ak4GenJets, ak4GenJetsNoNu and ak8GenJets
          (and the kT and Cambridge/Aachen variants). Jet by jet the momenta, the constituent lists and
          the flavour labels must be identical to those of standalone_genjets.py: this is the sharpest test
          of the chapter, two implementations of the same algorithm on the same 749 particles.
  full    the hard event of ../../ME/event_ME.lhe is written into a Les Houches file N times and PYTHIA runs
          its full chain (showers, MPI, remnants, colour reconnection, hadronisation, decays) with the CP5
          settings of the CMSSW configuration, as the validations of chapters 4 to 6 did; FastJet clusters
          every event into the three collections, the ghosts give hadronFlavour and partonFlavour, and
          for each hard b quark the matched jet is recorded (pT, response, dR, neutrinos, flavours, the
          cone of the same radius, the B hadron's share) in the layout of the standalone --repeat
          statistics, so that plot_theory_GenJet.py draws both. With --impact the events are selected in a
          window of b/<b> around the CMSSW value.

PYTHIA's initialisation and its error messages go through C++ streams, invisible to Python's sys.stdout,
so the file descriptor is redirected while init() and stat() run.
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


def delta_r(eta1, phi1, eta2, phi2):
    d = abs(phi1 - phi2)
    d = 2 * math.pi - d if d > math.pi else d
    return math.hypot(eta1 - eta2, d)


def pj(p4, index, scale=1.0):
    """A FastJet PseudoJet from (E, px, py, pz), tagged with an index."""
    import fastjet
    q = fastjet.PseudoJet(float(p4[1]) * scale, float(p4[2]) * scale, float(p4[3]) * scale, float(p4[0]) * scale)
    q.set_user_index(index)
    return q


def cluster(fastjet, pjs, R, algo, ptmin):
    jd = fastjet.JetDefinition({"antikt": fastjet.antikt_algorithm, "kt": fastjet.kt_algorithm, "ca": fastjet.cambridge_algorithm}[algo], R)
    cs = fastjet.ClusterSequence(pjs, jd)
    jets = fastjet.sorted_by_pt(cs.inclusive_jets(ptmin))
    return cs, jets


def jet_record(j, tags, ev, sg, extra_flav=None):
    """Kinematics and flavours of a FastJet jet from its tagged constituents, with the rules of standalone_genjets.py."""
    c = [k.user_index() for k in j.constituents()]
    rows = [tags[i][1] for i in c if tags[i][0] == "particle"]
    ghosts = {k: [tags[i][1] for i in c if tags[i][0] == k] for k in ("bhad", "chad", "parton", "physics")}
    p = sum((ev.p[r].p for r in rows), np.zeros(4))

    def hardest_rule(rs):
        if not rs:
            return 0
        b = [r for r in rs if abs(ev.p[r].id) == 5]
        cc = [r for r in rs if abs(ev.p[r].id) == 4]
        pool = b or cc or rs
        return int(ev.p[max(pool, key=lambda r: sg.pt(ev.p[r].p))].id)
    return dict(pt=sg.pt(p), eta=sg.eta(p), phi=sg.phi(p), m=math.sqrt(max(sg.mass2(p), 0.0)), e=float(p[0]), y=sg.rapidity(p), n=len(rows), rows=sorted(rows),
                n_charged=sum(1 for r in rows if sg.is_charged(ev.p[r].id)),
                hadron_flavour=5 if ghosts["bhad"] else 4 if ghosts["chad"] else 0, parton_flavour=hardest_rule(ghosts["parton"]),
                physics_flavour=hardest_rule(ghosts["physics"]), b_hadrons=[sg.name(ev.p[r].id) for r in ghosts["bhad"]],
                pt_nu=sg.pt(sum((ev.p[r].p for r in rows if sg.is_neutrino(ev.p[r].id)), np.zeros(4))))


# ---------------------------------------------------------------------------------------------------------
def mode_record(args, fastjet, pythia8, sg):
    ev, vertex, tau = sg.dec.read_record(args.input)
    ghosts = sg.ghost_rows(ev)
    fargs = sg.SimpleNamespace(algorithm=args.algorithm, ptmin=args.ptmin, radius=None, no_flavour_kt=True, input=args.input)
    st = sg.Stage(ev, fargs, verbose=0)
    out = dict(mode="record", algorithm=args.algorithm, collections={})
    print(f"{len(ev.final_indices())} final particles of {args.input.name}; ghosts: {len(ghosts['bhad'])} b hadrons, {len(ghosts['chad'])} c hadrons, "
          f"{len(ghosts['parton'])} partons, {len(ghosts['physics'])} hard-process partons")
    print(f"FastJet {fastjet.__version__ if hasattr(fastjet, '__version__') else ''}: {fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4).description()}")
    listing = []
    for key, c in sg.COLLECTIONS.items():
        rows = sg.input_rows(ev, c["nonu"])
        tags = [("particle", r) for r in rows]
        pjs = [pj(ev.p[r].p, i) for i, r in enumerate(rows)]
        for kind in ("bhad", "chad", "parton", "physics"):
            for r in ghosts[kind]:
                tags.append((kind, r))
                pjs.append(pj(ev.p[r].p, len(pjs), sg.GHOST_RESCALING))
        cs, jets = cluster(fastjet, pjs, c["R"], args.algorithm, args.ptmin)
        fj = [jet_record(j, tags, ev, sg) for j in jets]
        sa = st.coll[key].info
        # the same particles through PYTHIA's SlowJet (select 2 = no neutrinos; massSet 2 = true masses; no ghosts)
        py = pythia8.Pythia("", False)
        for s_ in ("ProcessLevel:all = off", "Next:numberCount = 0", "Print:quiet = on"):
            py.readString(s_)
        quiet(lambda: py.init())
        py.event.reset()
        index = {}
        for r in sg.input_rows(ev, False):
            q = ev.p[r]
            index[py.event.append(q.id, 1, 0, 0, float(q.p[1]), float(q.p[2]), float(q.p[3]), float(q.p[0]), q.m)] = r
        sj = pythia8.SlowJet(-1 if args.algorithm == "antikt" else (1 if args.algorithm == "kt" else 0), c["R"], args.ptmin, 1000.0, 2 if c["nonu"] else 1, 2)
        sj.analyze(py.event)
        slow = sorted([dict(pt=sj.pT(i), y=sj.y(i), phi=sj.phi(i), m=sj.m(i), n=sj.multiplicity(i), rows=sorted(index[k] for k in sj.constituents(i)))
                       for i in range(sj.sizeJet())], key=lambda j: -j["pt"])
        # compare
        n = min(len(fj), len(sa))
        dpt = max((abs(fj[i]["pt"] - sa[i]["pt"]) for i in range(n)), default=0.0)
        dm = max((abs(fj[i]["m"] - sa[i]["m"]) for i in range(n)), default=0.0)
        same_const = sum(1 for i in range(n) if fj[i]["rows"] == sorted(sa[i]["rows"]))
        same_flav = sum(1 for i in range(n) if (fj[i]["hadron_flavour"], fj[i]["parton_flavour"], fj[i]["physics_flavour"]) ==
                        (sa[i]["hadron_flavour"], sa[i]["parton_flavour"], sa[i]["physics_flavour"]))
        ns = min(len(slow), len(sa))
        dpt_slow = max((abs(slow[i]["pt"] - sa[i]["pt"]) for i in range(ns)), default=0.0)
        same_slow = sum(1 for i in range(ns) if slow[i]["rows"] == sorted(sa[i]["rows"]))
        print(f"\n{c['title']} ({args.algorithm}, R = {c['R']}, pT > {args.ptmin:g}{', no neutrinos' if c['nonu'] else ''}):")
        print(f"  jets: standalone {len(sa)}, FastJet {len(fj)}, SlowJet {len(slow)}")
        print(f"  FastJet vs standalone, jet by jet: max |dpT| = {dpt:.2e} GeV, max |dm| = {dm:.2e} GeV, identical constituent lists {same_const}/{n}, "
              f"identical (hadron, parton, physics) flavours {same_flav}/{n}")
        print(f"  SlowJet vs standalone: max |dpT| = {dpt_slow:.2e} GeV, identical constituent lists {same_slow}/{ns}")
        print(f"  {'jet':>4s} {'standalone pT':>14s} {'FastJet pT':>11s} {'SlowJet pT':>11s} {'eta':>7s} {'phi':>7s} {'n':>4s} {'had/part/phys':>14s}  b hadrons (FastJet ghosts)")
        for i in range(min(12, n)):
            s_, f_ = sa[i], fj[i]
            print(f"  {i:4d} {s_['pt']:14.4f} {f_['pt']:11.4f} {(slow[i]['pt'] if i < len(slow) else float('nan')):11.4f} {f_['eta']:7.3f} {f_['phi']:7.3f} {f_['n']:4d} "
                  f"{f_['hadron_flavour']:>4d}/{f_['parton_flavour']:+d}/{f_['physics_flavour']:+d}".ljust(60) + "  " + " ".join(f_["b_hadrons"]))
        out["collections"][key] = dict(n_standalone=len(sa), n_fastjet=len(fj), n_slowjet=len(slow), max_dpt=dpt, max_dm=dm, same_constituents=same_const,
                                       same_flavours=same_flav, max_dpt_slowjet=dpt_slow, same_constituents_slowjet=same_slow,
                                       fastjet=[{k: v for k, v in j.items() if k != "rows"} for j in fj],
                                       slowjet=[{k: v for k, v in j.items() if k != "rows"} for j in slow])
        listing.append(f"{c['title']}: {len(fj)} FastJet jets (anti-kT R = {c['R']}, pT > {args.ptmin:g} GeV, E scheme)")
        listing.append(f"{'jet':>4s}{'pT':>11s}{'y':>9s}{'eta':>9s}{'phi':>9s}{'m':>9s}{'n':>5s}{'had':>5s}{'part':>6s}{'phys':>6s}  constituents (rows of event_Decay.lhe)")
        for i, j in enumerate(fj):
            listing.append(f"{i:4d}{j['pt']:11.3f}{j['y']:9.3f}{j['eta']:9.3f}{j['phi']:9.3f}{j['m']:9.3f}{j['n']:5d}{j['hadron_flavour']:5d}{j['parton_flavour']:6d}{j['physics_flavour']:6d}  "
                           + " ".join(str(r) for r in j["rows"]))
        listing.append("")
    args.listing.write_text("\n".join(listing) + "\n")
    args.output.write_text(json.dumps(out, default=float) + "\n")
    print(f"\nwrote {args.output} and {args.listing}")


# ---------------------------------------------------------------------------------------------------------
JKEYS = ["pt", "eta", "phi", "m", "n", "n_charged", "dr", "response", "pt_cone", "n_cone", "pt_from_hadron", "n_from_hadron", "n_hadron_outside", "pt_nu", "n_nu",
         "hadron_flavour", "parton_flavour", "physics_flavour", "legacy_flavour", "dr_hadron"]


def heavy_content(pid: int) -> int:
    a = abs(pid)
    if a < 100 or a in (130, 310):
        return 0
    digits = [int(c) for c in str(a)]
    quarks = digits[-4:-1] if a >= 1000 else digits[-3:-1]
    return 5 if 5 in quarks else 4 if 4 in quarks else 0


def descendants(e, i):
    out, stack = set(), [i]
    while stack:
        k = stack.pop()
        d1, d2 = e[k].daughter1(), e[k].daughter2()
        if d1 > 0:
            for d in (range(d1, max(d1, d2) + 1) if d2 >= d1 else [d1, d2] if d2 > 0 else [d1]):
                if d not in out:
                    out.add(d)
                    stack.append(d)
    return out


def analyse_full(e, pd, fastjet, sg, ptmin):
    """One PYTHIA event: the three collections with ghosts, the two b quarks' jets."""
    fin = [i for i in range(e.size()) if e[i].isFinal()]
    # ghosts: last b hadrons, last c hadrons, partons of status 71-79, the hard-process partons
    def last(flav):
        out = []
        for i in range(e.size()):
            if e[i].isHadron() and heavy_content(e[i].id()) == flav:
                ds = e[i].daughterList()
                if not any(heavy_content(e[d].id()) == flav for d in ds):
                    out.append(i)
        return out
    bhad, chad = last(5), last(4)
    partons = [i for i in range(e.size()) if 71 <= e[i].statusAbs() <= 79 and (abs(e[i].id()) <= 6 or e[i].id() == 21)]
    physics = [i for i in range(e.size()) if e[i].statusAbs() == 23 and (abs(e[i].id()) <= 6 or e[i].id() == 21)]
    kin = {i: (e[i].pT(), e[i].eta(), e[i].phi()) for i in fin + bhad + chad + partons + physics}
    quarks = {}
    for i in range(e.size()):
        if e[i].statusAbs() == 23 and abs(e[i].id()) == 5:
            j = i
            while e[j].daughter1() > 0 and e[e[j].daughter1()].id() == e[j].id() and e[e[j].daughter1()].statusAbs() < 80:
                j = e[j].daughter1()
            quarks[e[i].id()] = j
    # the weakly decaying B of each quark: the last b hadron nearest to the quark with the right flavour
    bq = {}
    for pid, j in quarks.items():
        want = (lambda h: (h < 0) if abs(h) < 1000 else (h > 0)) if pid > 0 else (lambda h: (h > 0) if abs(h) < 1000 else (h < 0))
        cands = [k for k in bhad if want(e[k].id())] or bhad
        if cands:
            bq[pid] = min(cands, key=lambda k: delta_r(kin[k][1], kin[k][2], e[j].eta(), e[j].phi()))
    res = dict(collections={}, b={})
    for key, c in sg.COLLECTIONS.items():
        rows = [i for i in fin if not (c["nonu"] and abs(e[i].id()) in (12, 14, 16)) and abs(e[i].id()) not in sg.IGNORE_IDS]
        tags = [("particle", r) for r in rows]
        pjs = [fastjet.PseudoJet(e[r].px(), e[r].py(), e[r].pz(), e[r].e()) for r in rows]
        for kind, lst in (("bhad", bhad), ("chad", chad), ("parton", partons), ("physics", physics)):
            for r in lst:
                tags.append((kind, r))
                pjs.append(fastjet.PseudoJet(e[r].px() * sg.GHOST_RESCALING, e[r].py() * sg.GHOST_RESCALING, e[r].pz() * sg.GHOST_RESCALING, e[r].e() * sg.GHOST_RESCALING))
        for i, q in enumerate(pjs):
            q.set_user_index(i)
        cs, jets = cluster(fastjet, pjs, c["R"], "antikt", ptmin)
        info = []
        for rank, j in enumerate(jets):
            cc = [k.user_index() for k in j.constituents()]
            prow = [tags[i][1] for i in cc if tags[i][0] == "particle"]
            g = {k: [tags[i][1] for i in cc if tags[i][0] == k] for k in ("bhad", "chad", "parton", "physics")}
            px = sum(e[r].px() for r in prow)
            py_ = sum(e[r].py() for r in prow)
            pz = sum(e[r].pz() for r in prow)
            en = sum(e[r].e() for r in prow)
            pt_ = math.hypot(px, py_)
            p = math.sqrt(px * px + py_ * py_ + pz * pz)
            eta_ = 0.5 * math.log((p + pz) / (p - pz)) if p > abs(pz) else 0.0

            def hardest(rs):
                if not rs:
                    return 0
                b = [r for r in rs if abs(e[r].id()) == 5]
                cq = [r for r in rs if abs(e[r].id()) == 4]
                pool = b or cq or rs
                return e[max(pool, key=lambda r: e[r].pT())].id()
            pnu = [e[r] for r in prow if abs(e[r].id()) in (12, 14, 16)]
            info.append(dict(rank=rank, pt=pt_, eta=eta_, phi=math.atan2(py_, px), m=math.sqrt(max(en * en - p * p, 0.0)), n=len(prow), rows=prow,
                             n_charged=sum(1 for r in prow if e[r].isCharged()), hadron_flavour=5 if g["bhad"] else 4 if g["chad"] else 0,
                             parton_flavour=hardest(g["parton"]), physics_flavour=hardest(g["physics"]), bhad=g["bhad"],
                             pt_nu=math.hypot(sum(q.px() for q in pnu), sum(q.py() for q in pnu)), n_nu=len(pnu),
                             legacy_flavour=hardest([r for r in partons if delta_r(kin[r][1], kin[r][2], eta_, math.atan2(py_, px)) < sg.DR_LEGACY])))
        res["collections"][key] = dict(n_jets=len(info), n_clustered=sum(j["n"] for j in info), n_jets_10=sum(1 for j in info if j["pt"] > 10),
                                       n_jets_30=sum(1 for j in info if j["pt"] > 30), n_double_b=sum(1 for j in info if len(j["bhad"]) >= 2))
        for pid, jq in quarks.items():
            qn = "b" if pid > 0 else "bbar"
            rec = res["b"].setdefault(qn, dict(pt_quark=e[jq].pT(), eta_quark=e[jq].eta(), phi_quark=e[jq].phi(), jets={}))
            best = None
            for j in info:
                d = delta_r(j["eta"], j["phi"], e[jq].eta(), e[jq].phi())
                if d < c["R"] and (best is None or d < best[0]):
                    best = (d, j)
            if best is None:
                rec["jets"][key] = None
                continue
            d, j = best
            cone = [r for r in rows if delta_r(kin[r][1], kin[r][2], e[jq].eta(), e[jq].phi()) < c["R"]]
            pt_cone = math.hypot(sum(e[r].px() for r in cone), sum(e[r].py() for r in cone))
            hb = bq.get(pid)
            desc = set(r for r in descendants(e, hb) if e[r].isFinal()) if hb is not None else set()
            inside = [r for r in j["rows"] if r in desc]
            rec["jets"][key] = dict(pt=j["pt"], eta=j["eta"], phi=j["phi"], m=j["m"], n=j["n"], n_charged=j["n_charged"], dr=d, response=j["pt"] / e[jq].pT(),
                                    pt_cone=pt_cone, n_cone=len(cone), pt_from_hadron=math.hypot(sum(e[r].px() for r in inside), sum(e[r].py() for r in inside)),
                                    n_from_hadron=len(inside), n_hadron_outside=len(desc - set(j["rows"])), pt_nu=j["pt_nu"], n_nu=j["n_nu"],
                                    hadron_flavour=j["hadron_flavour"], parton_flavour=j["parton_flavour"], physics_flavour=j["physics_flavour"],
                                    legacy_flavour=j["legacy_flavour"], dr_hadron=(delta_r(kin[hb][1], kin[hb][2], j["eta"], j["phi"]) if hb is not None else -1.0),
                                    n_bhad=len(j["bhad"]), composition=composition(e, j["rows"]))
    return res


def composition(e, rows):
    out = dict(charged_hadron=0.0, photon=0.0, neutral_hadron=0.0, electron=0.0, muon=0.0, neutrino=0.0)
    acc = {k: [0.0, 0.0] for k in out}
    for r in rows:
        a = abs(e[r].id())
        k = "photon" if a == 22 else "electron" if a == 11 else "muon" if a == 13 else "neutrino" if a in (12, 14, 16) else "charged_hadron" if e[r].isCharged() else "neutral_hadron"
        acc[k][0] += e[r].px()
        acc[k][1] += e[r].py()
    return {k: math.hypot(*v) for k, v in acc.items()}


def mode_full(args, fastjet, pythia8, sg):
    dv = load("dv", HERE.parent.parent / "Decay" / "Standalone" / "validate_decays_with_pythia.py")
    mpi = load("mpi_val", HERE.parent.parent / "MPI" / "Standalone" / "validate_mpi_with_pythia.py")
    sps = load("sps", HERE.parent.parent / "PS" / "Standalone" / "standalone_parton_showering.py")
    ev = sps.read_event(args.input)
    hard = [i for i, q in enumerate(ev.p) if q.status == 23]
    m2avg = sum(ev.p[i].m ** 2 for i in hard) / len(hard)
    scalup = math.sqrt(sps.pt(ev.p[hard[0]].p) ** 2 + m2avg)
    lhef = HERE / "validate_genjets_with_fastjet.lhe"
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
    py = pythia8.Pythia("", False)
    settings = list(dv.COMMON) + [f"Random:seed = {args.seed}"]
    settings += ["Tune:pp = 14", "Tune:ee = 7", "Beams:frameType = 4", f"Beams:LHEF = {lhef}",
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
    print(f"PYTHIA 8 runs the full chain on the hard event {args.nevents} times (CP5, decays on, PDF {pset.split(':')[0]}); FastJet clusters every event")
    keys = ["n_jets", "n_clustered", "n_jets_10", "n_jets_30", "n_double_b"]
    acc = {k: {c: [] for c in sg.COLLECTIONS} for k in keys}
    fam = {}
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
        s = analyse_full(py.event, py.particleData, fastjet, sg, args.ptmin)
        for c in sg.COLLECTIONS:
            for k in keys:
                acc[k][c].append(s["collections"][c][k])
        for q, r in s["b"].items():
            d = fam.setdefault(q, dict(pt_quark=[], jets={c: {k: [] for k in JKEYS + ["found", "composition", "n_bhad"]} for c in sg.COLLECTIONS}))
            d["pt_quark"].append(r["pt_quark"])
            for c in sg.COLLECTIONS:
                j = r["jets"][c]
                d["jets"][c]["found"].append(j is not None)
                for k in JKEYS + ["n_bhad"]:
                    d["jets"][c][k].append(j[k] if j is not None else float("nan"))
                d["jets"][c]["composition"].append(j["composition"] if j is not None else None)
        ndone += 1
        if ndone % 200 == 0:
            print(f"  {ndone} events", flush=True)
        if ndone == 1:
            args.listing.write_text(dv.pythia_listing(py.event, "PYTHIA 8 Event Listing  (full chain: LHEF hard process + showers + MPI + CR + hadronisation + decays)"))
    _, stat_text = quiet(lambda: py.stat())
    st = json.loads(args.standalone_stats.read_text()) if args.standalone_stats and args.standalone_stats.exists() else None
    print(f"\n{ndone} events analysed" + (f"; standalone script: {st['repeat']} decays of the record from {args.standalone_stats}" if st else ""))
    head = f"{'quantity':40s}{'PYTHIA+FastJet':>16s}{'std':>10s}" + (f"{'standalone':>14s}{'std':>10s}" if st else "")
    print(head)
    for c in sg.COLLECTIONS:
        for k in keys:
            v = np.asarray(acc[k][c], dtype=float)
            line = f"{c + ' ' + k:40s}{v.mean():16.3f}{v.std():10.3f}"
            if st and k in st["counts"]:
                w = np.asarray(st["counts"][k][c], dtype=float)
                line += f"{w.mean():14.3f}{w.std():10.3f}"
            print(line)
    for q, d in fam.items():
        for c in sg.COLLECTIONS:
            jj = d["jets"][c]
            for k in ("pt", "response", "dr", "n", "n_charged", "m", "pt_nu", "pt_cone", "pt_from_hadron", "n_hadron_outside", "dr_hadron"):
                v = np.asarray(jj[k], dtype=float)
                v = v[~np.isnan(v)]
                line = f"{q + ' ' + c + ' ' + k:40s}{v.mean():16.3f}{v.std():10.3f}"
                if st and q in st["b"]:
                    w = np.asarray(st["b"][q]["jets"][c][k], dtype=float)
                    w = w[~np.isnan(w)]
                    line += f"{w.mean():14.3f}{w.std():10.3f}"
                print(line)
            hf = np.asarray(jj["hadron_flavour"], dtype=float)
            pf = np.asarray(jj["parton_flavour"], dtype=float)
            line = f"{q + ' ' + c + ' hadronFlavour 5 / partonFlavour 5 [%]':40s}{100 * np.nanmean(hf == 5):8.1f} {100 * np.nanmean(np.abs(pf) == 5):7.1f}"
            if st and q in st["b"]:
                w1 = np.asarray(st["b"][q]["jets"][c]["hadron_flavour"], dtype=float)
                w2 = np.asarray(st["b"][q]["jets"][c]["parton_flavour"], dtype=float)
                line += f"{'':10s}{100 * np.nanmean(w1 == 5):8.1f} {100 * np.nanmean(np.abs(w2) == 5):7.1f}"
            print(line)
            print(f"{q + ' ' + c + ' found [%]':40s}{100 * np.mean(jj['found']):16.1f}")
    args.output.write_text(json.dumps(dict(mode="full", nevents=ndone, seed=args.seed, scalup=scalup, impact=args.impact, window=args.window, counts=acc, b=fam),
                                      default=float) + "\n")
    print(f"wrote {args.output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("record", "full"), default="record")
    ap.add_argument("-i", "--input", type=Path, default=None, help="record: ../../Decay/event_Decay.lhe; full: ../../ME/event_ME.lhe")
    ap.add_argument("-n", "--nevents", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--algorithm", choices=("antikt", "kt", "ca"), default="antikt")
    ap.add_argument("--ptmin", type=float, default=3.0)
    ap.add_argument("--impact", type=float, default=None, help="full mode: keep events with |b/<b> - impact| < window")
    ap.add_argument("--window", type=float, default=0.03)
    ap.add_argument("--standalone-stats", type=Path, help="full mode: .stats.json from standalone_genjets.py --repeat")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--listing", type=Path, default=None)
    args = ap.parse_args()
    if args.input is None:
        args.input = HERE.parent.parent / ("Decay" if args.mode == "record" else "ME") / ("event_Decay.lhe" if args.mode == "record" else "event_ME.lhe")
    tag = args.mode + ("" if args.algorithm == "antikt" else "_" + args.algorithm)
    if args.output is None:
        args.output = HERE / f"validate_genjets_with_fastjet_{tag}.json"
    if args.listing is None:
        args.listing = HERE / (f"validate_genjets_with_fastjet_{tag}_jets.txt" if args.mode == "record" else "validate_genjets_with_fastjet_full_event1.txt")
    import fastjet
    try:
        import pythia8mc as pythia8
    except ImportError:
        import pythia8
    sg = load("sg", HERE / "standalone_genjets.py")
    if args.mode == "record":
        mode_record(args, fastjet, pythia8, sg)
    else:
        mode_full(args, fastjet, pythia8, sg)


if __name__ == "__main__":
    main()
