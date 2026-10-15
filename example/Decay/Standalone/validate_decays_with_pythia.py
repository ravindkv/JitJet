#!/usr/bin/env python3
"""Decay with PYTHIA 8 and compare with the standalone hadron-decay script.

Needs the environment of example/ME/Standalone (pythia8mc, LHAPDF for --mode full):
  PY=../../ME/Standalone/.venv/bin/python
  $PY validate_decays_with_pythia.py -n 400 --standalone-stats standalone_decay_stats.stats.json   # mode record
  $PY validate_decays_with_pythia.py --mode single --id 511 -n 20000 --standalone-stats standalone_single_511.json
  $PY validate_decays_with_pythia.py --mode full -n 2000
  $PY validate_decays_with_pythia.py --mode full -n 8000 --impact 0.3136 --window 0.03 --output validate_decays_with_pythia_full_fixedb.json

Three modes.

  record  PYTHIA decays exactly the 365 primary hadrons of the standalone record
          (../../Had/event_Had.lhe, the rows with status 81-84) N times with different
          random seeds, with ProcessLevel:all = off and ParticleDecays:limitTau0 = on as
          CMSSW sets it, so that only the decays differ from
          standalone_hadron_decays.py --repeat N on the same input. This is the sharpest
          comparison of this chapter: same hadrons, two implementations of the decay tables.
  single  one particle at rest (--id, e.g. 511 for a B0) decayed N times: the multiplicities,
          the semileptonic fractions, the lepton and neutrino energies of the decay table
          itself, next to standalone_hadron_decays.py --single ID --repeat N.
  full    the hard event of ../../ME/event_ME.lhe is written into a Les Houches file N times
          and PYTHIA runs its full chain (showers, MPI, remnants, colour reconnection,
          hadronisation, decays) with the CP5 settings of the CMSSW configuration, as the
          chapter-5 validation did with decays off. With --impact the events are selected in
          a window of b/<b> around the CMSSW value.

Per event: the number of final particles (all, charged, photons, neutrinos, leptons,
hadrons, central), the energy and the missing pT of the neutrinos, and for each hard b
quark the weakly decaying B hadron with its lifetime, flight distance and vertex, its
decay class, the number and the charge of its final descendants, the neutrino energy
they carry, the hardest lepton, the charm hadron, the impact parameters of its charged
tracks, and the cones of R = 0.4 and 0.8 around the quark direction (primary hadrons, all
final particles, the visible ones, the charged ones), in the same JSON layout as the
standalone --repeat statistics, so that plot_theory_Decay.py can draw both.

PYTHIA's initialisation and its error messages go through C++ streams, invisible to
Python's sys.stdout, so the file descriptor is redirected while init() and stat() run.
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
    t2 = "PYTHIA 8 Vertex Listing  (production vertex [mm] and time [mm/c] of every row born more than 0.1 um from the origin; tau = proper lifetime [mm/c])"
    out += ["", f" --------  {t2}  " + "-" * max(10, 133 - len(t2) - 14), " ", "    no      x_prod      y_prod      z_prod      t_prod         tau"]
    for i in range(e.size()):
        q = e[i]
        if max(abs(q.xProd()), abs(q.yProd()), abs(q.zProd())) > 1e-4 or (q.tau() > 1e-4 and q.status() < 0):
            out.append(f"{i:6d}{q.xProd():12.5f}{q.yProd():12.5f}{q.zProd():12.5f}{q.tProd():12.5f}{q.tau() if q.status() < 0 else 0.0:12.5f}")
    out.append("")
    out.append(f" --------  End {t2}  " + "-" * max(10, 133 - len(t2) - 18))
    return "\n".join(out) + "\n"


def delta_r(eta1, phi1, eta2, phi2):
    d = abs(phi1 - phi2)
    d = 2 * math.pi - d if d > math.pi else d
    return math.hypot(eta1 - eta2, d)


def heavy_content(pid: int) -> int:
    a = abs(pid)
    if a < 100 or a in (130, 310):
        return 0
    digits = [int(c) for c in str(a)]
    quarks = digits[-4:-1] if a >= 1000 else digits[-3:-1]
    return 5 if 5 in quarks else 4 if 4 in quarks else 0


def is_neutrino(pid):
    return abs(pid) in (12, 14, 16)


def is_lepton(pid):
    return abs(pid) in (11, 13, 15)


def p4(q):
    return np.array([q.e(), q.px(), q.py(), q.pz()])


def cone_kin(e, rows):
    p = sum((p4(e[i]) for i in rows), np.zeros(4))
    m2 = p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2
    return dict(n=len(rows), pt=math.hypot(p[1], p[2]), m=math.sqrt(max(m2, 0.0)), e=float(p[0]),
                n_charged=sum(1 for i in rows if e[i].isCharged()))


def channel_class(e, i):
    ds = e[i].daughterList()
    ids = [e[d].id() for d in ds]
    leps = [x for x in ids if is_lepton(x)]
    nus = [x for x in ids if is_neutrino(x)]
    if leps and nus:
        lep = {11: "e", 13: "mu", 15: "tau"}[abs(leps[0])]
        return f"semileptonic ({lep})" if any(not is_lepton(x) and not is_neutrino(x) for x in ids) else f"leptonic ({lep})"
    if leps:
        return "with a lepton pair"
    return "hadronic"


def weak_b(e, i):
    """From a b hadron row down the b flavour to the first one with a lifetime."""
    r = i
    while e[r].tau0() <= 1e-4 and e[r].daughter1() > 0:
        ds = [d for d in e[r].daughterList() if heavy_content(e[d].id()) == 5]
        if not ds:
            break
        r = ds[0]
    return r


def impact_parameter(q):
    ptv = q.pT()
    return (q.xProd() * q.py() - q.yProd() * q.px()) / ptv if ptv > 0 else 0.0


def analyse(e, pd, b_rows, primaries, parton_rows, quark_axes, parton_cones=None):
    """One decayed event: the summary in the layout of standalone_hadron_decays.summarise().
    b_rows: pid -> row of the primary b hadron of the hard quark's string; quark_axes: pid -> (pt, e, eta, phi) of the quark."""
    fin = [i for i in range(e.size()) if e[i].isFinal()]
    prim = set(primaries)
    s = dict(n_final=len(fin), n_charged=sum(1 for i in fin if e[i].isCharged()),
             n_photons=sum(1 for i in fin if e[i].id() == 22), n_neutrinos=sum(1 for i in fin if is_neutrino(e[i].id())),
             n_electrons=sum(1 for i in fin if abs(e[i].id()) == 11), n_muons=sum(1 for i in fin if abs(e[i].id()) == 13),
             n_hadrons_final=sum(1 for i in fin if e[i].isHadron()),
             n_primary_stable=sum(1 for i in fin if i in prim),
             n_final_central=sum(1 for i in fin if abs(e[i].eta()) < 2.5),
             n_charged_central=sum(1 for i in fin if abs(e[i].eta()) < 2.5 and e[i].isCharged()),
             e_neutrinos=float(sum(e[i].e() for i in fin if is_neutrino(e[i].id()))),
             n_decays=sum(1 for i in range(e.size()) if e[i].status() < 0 and e[i].statusAbs() in (81, 82, 83, 84, 91, 92)),
             n_oscillated=len({e[i].mother1() for i in range(e.size()) if e[i].status() == 92}),
             species={}, b={})
    p_nu = sum((p4(e[i]) for i in fin if is_neutrino(e[i].id())), np.zeros(4))
    s["pt_neutrinos"] = math.hypot(p_nu[1], p_nu[2])
    for i in fin:
        n = e[i].name()
        s["species"][n] = s["species"].get(n, 0) + 1
    for pid, r0 in b_rows.items():
        rw = weak_b(e, r0)
        hw = e[rw]
        desc = hw.daughterListRecursive()
        fin_desc = [d for d in desc if e[d].isFinal()]
        decays = hw.daughter1() > 0
        nus = [d for d in fin_desc if is_neutrino(e[d].id())]
        pnu = sum((p4(e[d]) for d in nus), np.zeros(4))
        leps = [d for d in fin_desc if abs(e[d].id()) in (11, 13)]
        lep = max(leps, key=lambda d: e[d].pT()) if leps else None
        charm = [d for d in desc if heavy_content(e[d].id()) == 4 and e[d].tau0() > 1e-4]
        rc = charm[0] if charm else None
        tracks = [d for d in fin_desc if e[d].isCharged()]
        d0 = [abs(impact_parameter(e[d])) for d in tracks]
        tau = hw.tau() if decays else 0.0
        vdec = np.array([hw.xDec(), hw.yDec(), hw.zDec()]) if decays else np.zeros(3)
        vprod = np.array([hw.xProd(), hw.yProd(), hw.zProd()])
        cls = channel_class(e, rw) if decays else "stable"
        name = "b" if pid > 0 else "bbar"
        qa = quark_axes[pid]
        rec = dict(quark=name, pt_quark=qa["pt"], e_quark=qa["e"], string_hadron=e[r0].name(), pt_string=e[r0].pT(),
                   hadron=hw.name(), hadron_id=hw.id(), pt_hadron=hw.pT(), e_hadron=hw.e(), m_hadron=hw.m(), gamma=hw.e() / hw.m(),
                   tau=tau, tau_ps=tau / 0.299792458, tau0=hw.tau0(), tau_over_tau0=(tau / hw.tau0() if hw.tau0() > 0 else 0.0),
                   L=float(np.linalg.norm(vdec - vprod)), Lxy=float(math.hypot(vdec[0] - vprod[0], vdec[1] - vprod[1])), vertex=[float(v) for v in vdec],
                   oscillated=bool(decays and e[hw.daughter1()].status() == 92), decays=bool(decays), cls=cls,
                   channel=" ".join(e[d].name() for d in hw.daughterList()) if decays else "",
                   n_direct=len(hw.daughterList()) if decays else 0, n_final=len(fin_desc),
                   n_charged_final=sum(1 for d in fin_desc if e[d].isCharged()), n_photons_final=sum(1 for d in fin_desc if e[d].id() == 22),
                   n_neutrinos=len(nus), e_nu=float(pnu[0]), pt_nu=math.hypot(pnu[1], pnu[2]), e_nu_frac=float(pnu[0] / hw.e()),
                   lepton=(e[lep].name() if lep is not None else ""), pt_lepton=(e[lep].pT() if lep is not None else 0.0),
                   lepton_from_b=bool(lep is not None and e[lep].mother1() == rw),
                   charm=(e[rc].name() if rc is not None else ""), pt_charm=(e[rc].pT() if rc is not None else 0.0),
                   L_charm=(math.sqrt((e[rc].xDec() - e[rc].xProd()) ** 2 + (e[rc].yDec() - e[rc].yProd()) ** 2 + (e[rc].zDec() - e[rc].zProd()) ** 2)
                            if rc is not None and e[rc].daughter1() > 0 else 0.0),
                   n_tracks=len(tracks), d0=d0, max_d0=(max(d0) if d0 else 0.0), n_tracks_d0_100um=sum(1 for x in d0 if x > 0.1), cones={})
        psys = sum((p4(e[d]) for d in fin_desc), np.zeros(4))
        pvis = psys - pnu
        rec["system"] = dict(n=len(fin_desc), pt=math.hypot(psys[1], psys[2]), pt_visible=math.hypot(pvis[1], pvis[2]))
        fset = set(fin_desc)
        for R in (0.4, 0.8):
            mem_h = [i for i in primaries if delta_r(e[i].eta(), e[i].phi(), qa["eta"], qa["phi"]) < R]
            mem_f = [i for i in fin if delta_r(e[i].eta(), e[i].phi(), qa["eta"], qa["phi"]) < R]
            mem_v = [i for i in mem_f if not is_neutrino(e[i].id())]
            mem_c = [i for i in mem_f if e[i].isCharged()]
            mem_b = [i for i in mem_f if i in fset]
            c = dict(primaries=cone_kin(e, mem_h), final=cone_kin(e, mem_f), visible=cone_kin(e, mem_v), charged=cone_kin(e, mem_c), from_b=cone_kin(e, mem_b))
            if parton_cones is not None:
                c["partons"] = parton_cones[pid][R]
            elif parton_rows:
                mem_p = [i for i in parton_rows if delta_r(e[i].eta(), e[i].phi(), qa["eta"], qa["phi"]) < R]
                c["partons"] = cone_kin(e, mem_p)
            c["frac_B"] = c["from_b"]["pt"] / c["final"]["pt"] if c["final"]["pt"] > 0 else 0.0
            c["nu_pt_in_cone"] = math.hypot(*sum((p4(e[i]) for i in mem_f if is_neutrino(e[i].id())), np.zeros(4))[1:3])
            rec["cones"][R] = c
        s["b"][name] = rec
    return s


KEYS = ["n_final", "n_charged", "n_photons", "n_neutrinos", "n_electrons", "n_muons", "n_hadrons_final", "n_primary_stable",
        "n_final_central", "n_charged_central", "e_neutrinos", "pt_neutrinos", "n_decays", "n_oscillated"]
BKEYS = ["tau_ps", "tau_over_tau0", "L", "Lxy", "n_direct", "n_final", "n_charged_final", "n_photons_final", "n_neutrinos", "e_nu", "pt_nu", "e_nu_frac",
         "pt_lepton", "pt_charm", "L_charm", "n_tracks", "max_d0", "n_tracks_d0_100um", "gamma"]
CKEYS = ["pt", "m", "n", "n_charged"]


def accumulate(acc, fam, species, s):
    for k in KEYS:
        acc[k].append(s[k])
    for nm, k in s["species"].items():
        species[nm] = species.get(nm, 0) + k
    for nm, r in s["b"].items():
        d = fam.setdefault(nm, dict({k: [] for k in BKEYS}, cls=[], hadron=[], lepton=[], lepton_from_b=[], oscillated=[], channel=[], charm=[], d0=[],
                                    system_pt=[], system_pt_visible=[],
                                    cones={R: {c: {k: [] for k in CKEYS} for c in ("primaries", "final", "visible", "charged", "from_b")} for R in (0.4, 0.8)}))
        for k in BKEYS:
            d[k].append(r[k])
        for k in ("cls", "hadron", "lepton", "lepton_from_b", "oscillated", "channel", "charm"):
            d[k].append(r[k])
        d["d0"] += r["d0"]
        d["system_pt"].append(r["system"]["pt"])
        d["system_pt_visible"].append(r["system"]["pt_visible"])
        for R in (0.4, 0.8):
            for c in ("primaries", "final", "visible", "charged", "from_b"):
                for k in CKEYS:
                    d["cones"][R][c][k].append(r["cones"][R][c][k])
            d["cones"][R].setdefault("frac_B", []).append(r["cones"][R]["frac_B"])
            d["cones"][R].setdefault("nu_pt_in_cone", []).append(r["cones"][R]["nu_pt_in_cone"])
            if "partons" in r["cones"][R]:
                d["cones"][R].setdefault("partons_pt", []).append(r["cones"][R]["partons"]["pt"])


COMMON = ["Next:numberCount = 0", "Next:numberShowEvent = 0", "Next:numberShowProcess = 0", "Next:numberShowInfo = 0",
          "Next:numberShowLHA = 0", "Init:showChangedSettings = off", "Init:showProcesses = off",
          "Init:showMultipartonInteractions = off", "Random:setSeed = on", "ParticleDecays:limitTau0 = on"]


def count(it):
    out = {}
    for i in it:
        out[i] = out.get(i, 0) + 1
    return out


def row(label, v_py, v_st):
    v_py = np.asarray(v_py, dtype=float)
    v_py = v_py[~np.isnan(v_py)]
    line = f"{label:34s}{v_py.mean():14.4f}{v_py.std():10.4f}"
    if v_st is not None and len(v_st):
        v_st = np.asarray(v_st, dtype=float)
        v_st = v_st[~np.isnan(v_st)]
        err = math.hypot(v_py.std() / math.sqrt(len(v_py)), v_st.std() / math.sqrt(len(v_st)))
        line += f"{v_st.mean():18.4f}{v_st.std():10.4f}{(v_st.mean() - v_py.mean()) / err if err > 0 else 0:10.2f}"
    print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("record", "single", "full"), default="record")
    ap.add_argument("-i", "--input", type=Path, default=None, help="record: ../../Had/event_Had.lhe; full: ../../ME/event_ME.lhe")
    ap.add_argument("-n", "--nevents", type=int, default=400)
    ap.add_argument("--id", type=int, default=511, help="single: the particle to decay at rest")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--impact", type=float, default=None, help="full mode: keep events with |b/<b> - impact| < window")
    ap.add_argument("--window", type=float, default=0.03)
    ap.add_argument("--no-mix", action="store_true", help="ParticleDecays:mixB = off")
    ap.add_argument("--standalone-stats", type=Path, help=".stats.json from standalone_hadron_decays.py --repeat (or the --single --history file)")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--listing", type=Path, default=None)
    args = ap.parse_args()
    if args.input is None:
        args.input = HERE.parent.parent / ("Had" if args.mode == "record" else "ME") / ("event_Had.lhe" if args.mode == "record" else "event_ME.lhe")
    if args.output is None:
        args.output = HERE / f"validate_decays_with_pythia_{args.mode}{'_' + str(args.id) if args.mode == 'single' else ''}.json"
    if args.listing is None:
        args.listing = HERE / f"validate_decays_with_pythia_{args.mode}_event1.txt"

    try:
        import pythia8mc as pythia8
    except ImportError:
        import pythia8
    sd = load("sd", HERE / "standalone_hadron_decays.py")
    had = sd.had

    py = pythia8.Pythia("", False)
    settings = list(COMMON) + [f"Random:seed = {args.seed}"]
    if args.no_mix:
        settings.append("ParticleDecays:mixB = off")
    st = json.loads(args.standalone_stats.read_text()) if args.standalone_stats and args.standalone_stats.exists() else None

    # ------------------------------------------------------------------ single particle at rest
    if args.mode == "single":
        settings += ["ProcessLevel:all = off"]
        for s_ in settings:
            py.readString(s_)
        ok, init_text = quiet(lambda: py.init())
        if not ok:
            print(init_text)
            sys.exit("PYTHIA initialisation failed")
        pid = args.id
        m = py.particleData.m0(pid)
        acc = dict(n_final=[], n_charged=[], n_photons=[], n_neutrinos=[], e_lepton=[], e_nu=[], cls=[], n_direct=[], L=[], charm=[], n_kaons=[], n_pions=[],
                   e_lepton_direct=[])
        print(f"PYTHIA 8 decays a {py.particleData.name(pid)} at rest {args.nevents} times")
        for n in range(args.nevents):
            py.event.reset()
            i0 = py.event.append(pid, 81, 0, 0, 0.0, 0.0, 0.0, m, m)
            good, text = quiet(lambda: py.next())
            if not good:
                continue
            e = py.event
            fin = [i for i in range(e.size()) if e[i].isFinal()]
            acc["n_final"].append(len(fin))
            acc["n_charged"].append(sum(1 for i in fin if e[i].isCharged()))
            acc["n_photons"].append(sum(1 for i in fin if e[i].id() == 22))
            acc["n_neutrinos"].append(sum(1 for i in fin if is_neutrino(e[i].id())))
            acc["n_kaons"].append(sum(1 for i in fin if abs(e[i].id()) in (321, 130, 310)))
            acc["n_pions"].append(sum(1 for i in fin if abs(e[i].id()) == 211))
            acc["e_nu"].append(sum(e[i].e() for i in fin if is_neutrino(e[i].id())))
            leps = [i for i in fin if abs(e[i].id()) in (11, 13)]
            acc["e_lepton"].append(max((e[i].e() for i in leps), default=0.0))
            ds = e[i0].daughterList()
            direct = [d for d in ds if abs(e[d].id()) in (11, 13)]
            acc["e_lepton_direct"].append(e[direct[0]].e() if direct else float("nan"))
            acc["cls"].append(channel_class(e, i0) if ds else "stable")
            acc["n_direct"].append(len(ds))
            acc["L"].append(e[i0].tau())
            charm = [i for i in range(e.size()) if heavy_content(e[i].id()) == 4 and e[i].tau0() > 1e-4]
            acc["charm"].append(e[charm[0]].name() if charm else "")
            if n == 0:
                args.listing.write_text(pythia_listing(e, f"PYTHIA 8 Event Listing  (a {py.particleData.name(pid)} at rest decayed by PYTHIA)"))
        _, stat_text = quiet(lambda: py.stat())
        n = len(acc["n_final"])
        head = f"{'quantity':34s}{'PYTHIA mean':>14s}{'std':>10s}" + (f"{'standalone mean':>18s}{'std':>10s}{'diff/err':>10s}" if st else "")
        print(head)
        for k in ("n_final", "n_charged", "n_photons", "n_neutrinos", "n_kaons", "n_pions", "n_direct", "e_nu", "e_lepton", "e_lepton_direct"):
            row(k, acc[k], st[k] if st else None)
        v = np.asarray(acc["e_lepton_direct"], dtype=float)
        print(f"  direct semileptonic (e, mu) decays: PYTHIA {100 * np.mean(~np.isnan(v)):.2f}%" +
              (f", standalone {100 * np.mean(~np.isnan(np.asarray(st['e_lepton_direct'], dtype=float))):.2f}%" if st else ""))
        cls = count(acc["cls"])
        print("  classes (PYTHIA): " + ", ".join(f"{k} {100 * v_ / n:.2f}%" for k, v_ in sorted(cls.items(), key=lambda kv: -kv[1])))
        if st:
            c2 = count(st["cls"])
            print("  classes (standalone): " + ", ".join(f"{k} {100 * v_ / st['n']:.2f}%" for k, v_ in sorted(c2.items(), key=lambda kv: -kv[1])))
        ch = count(acc["charm"])
        print("  charm hadron (PYTHIA): " + ", ".join(f"{k or 'none'} {100 * v_ / n:.1f}%" for k, v_ in sorted(ch.items(), key=lambda kv: -kv[1])[:8]))
        if st:
            c2 = count(st["charm"])
            print("  charm hadron (standalone): " + ", ".join(f"{k or 'none'} {100 * v_ / st['n']:.1f}%" for k, v_ in sorted(c2.items(), key=lambda kv: -kv[1])[:8]))
        # the lepton energy spectrum of the direct semileptonic decays (the V-A matrix element at work)
        edges = np.linspace(0, 2.6, 14)
        hp, _ = np.histogram(v[~np.isnan(v)], bins=edges, density=True)
        line = "  E_lepton spectrum (direct, rest frame), PYTHIA:     " + " ".join(f"{x:5.2f}" for x in hp)
        print(line)
        if st:
            vs = np.asarray(st["e_lepton_direct"], dtype=float)
            hs, _ = np.histogram(vs[~np.isnan(vs)], bins=edges, density=True)
            print("  E_lepton spectrum (direct, rest frame), standalone: " + " ".join(f"{x:5.2f}" for x in hs))
        args.output.write_text(json.dumps(dict(mode="single", id=pid, n=n, seed=args.seed, **acc), default=float) + "\n")
        print(f"wrote {args.output}")
        return

    # ------------------------------------------------------------------ the record, or the full chain
    acc = {k: [] for k in KEYS}
    fam, species = {}, {}
    extra = {}
    if args.mode == "record":
        ev, _, _ = sd.read_record(args.input)
        bq = had.b_quark_rows(ev)
        bstr = sd.string_b_hadrons(ev)
        quark_axes = {pid: dict(pt=had.pt(ev.p[j].p), e=float(ev.p[j].p[0]), eta=had.eta(ev.p[j].p), phi=had.phi(ev.p[j].p)) for pid, j in bq.items()}
        parton_rows_sa = [i for i, q in enumerate(ev.p) if abs(q.status) in (62, 63)]
        parton_cones = {pid: {R: sd.cone_kin(ev, sd.cone_members(ev, parton_rows_sa, ev.p[j].p, R)) for R in (0.4, 0.8)} for pid, j in bq.items()}
        prim = [i for i, q in enumerate(ev.p) if q.status in (81, 82, 83, 84)]
        settings += ["ProcessLevel:all = off", "HadronLevel:Hadronize = off"]
        for s_ in settings:
            py.readString(s_)
        ok, init_text = quiet(lambda: py.init())
        if not ok:
            print(init_text)
            sys.exit("PYTHIA initialisation failed")
        print(f"PYTHIA 8 decays the {len(prim)} primary hadrons of {args.input.name} {args.nevents} times (limitTau0 on, mixing {'off' if args.no_mix else 'on'})")
        ndone = 0
        for n in range(args.nevents):
            py.event.reset()
            index = {}
            for i in prim:
                q = ev.p[i]
                k = py.event.append(q.id, q.status, 0, 0, 0, 0, 0, 0, float(q.p[1]), float(q.p[2]), float(q.p[3]), float(q.p[0]), q.m)
                index[i] = k
            good, text = quiet(lambda: py.next())
            if not good:
                print(f"  event {n}: PYTHIA could not decay the record: " + " ".join(text.split("\n")[-3:])[:200])
                continue
            e = py.event
            b_rows = {pid: index[r] for pid, r in bstr.items()}
            s = analyse(e, py.particleData, b_rows, list(index.values()), [], quark_axes, parton_cones)
            accumulate(acc, fam, species, s)
            ndone += 1
            if n == 0:
                args.listing.write_text(pythia_listing(e, "PYTHIA 8 Event Listing  (the standalone primary hadrons of event_Had.lhe decayed by PYTHIA)"))
        extra = dict(n_primaries=len(prim))
    else:
        mpi = load("mpi_val", HERE.parent.parent / "MPI" / "Standalone" / "validate_mpi_with_pythia.py")
        sps = load("sps", HERE.parent.parent / "PS" / "Standalone" / "standalone_parton_showering.py")
        ev = sps.read_event(args.input)
        hard = [i for i, q in enumerate(ev.p) if q.status == 23]
        m2avg = sum(ev.p[i].m ** 2 for i in hard) / len(hard)
        scalup = math.sqrt(sps.pt(ev.p[hard[0]].p) ** 2 + m2avg)
        lhef = HERE / "validate_decays_with_pythia.lhe"
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
        print(f"PYTHIA 8 runs the full chain on the hard event {args.nevents} times (CP5, decays on, PDF {pset.split(':')[0]})")
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
            quark_axes, b_rows = {}, {}
            prim = [i for i in range(e.size()) if e[i].statusAbs() in (81, 82, 83, 84)]
            for i in range(e.size()):
                if e[i].statusAbs() == 23 and abs(e[i].id()) == 5:
                    j = i
                    while e[j].daughter1() > 0 and e[e[j].daughter1()].id() == e[j].id() and e[e[j].daughter1()].statusAbs() < 70:
                        j = e[j].daughter1()
                    pid = e[i].id()
                    quark_axes[pid] = dict(pt=e[j].pT(), e=e[j].e(), eta=e[j].eta(), phi=e[j].phi())
                    jcopy = e[j].iBotCopyId()
                    cands = [k for k in prim if e[k].mother1() <= jcopy <= max(e[k].mother2(), e[k].mother1())]
                    want = (lambda h: (h < 0) if abs(h) < 1000 else (h > 0)) if pid > 0 else (lambda h: (h > 0) if abs(h) < 1000 else (h < 0))
                    bh = [k for k in cands if heavy_content(e[k].id()) == 5 and want(e[k].id())] or \
                         [k for k in prim if heavy_content(e[k].id()) == 5 and want(e[k].id())]
                    if bh:
                        b_rows[pid] = min(bh, key=lambda k: delta_r(e[k].eta(), e[k].phi(), e[j].eta(), e[j].phi()))
            parton_rows = [i for i in range(e.size()) if e[i].statusAbs() in (62, 63)]
            s = analyse(e, py.particleData, b_rows, prim, parton_rows, quark_axes)
            accumulate(acc, fam, species, s)
            ndone += 1
            if ndone == 1:
                args.listing.write_text(pythia_listing(e, "PYTHIA 8 Event Listing  (full chain: LHEF hard process + showers + MPI + CR + hadronisation + decays)"))
        extra = dict(scalup=scalup, impact=args.impact, window=args.window)
    _, stat_text = quiet(lambda: py.stat())

    print(f"\n{ndone} events analysed")
    if st:
        print(f"standalone script: {st['repeat']} decays of the record from {args.standalone_stats}")
    head = f"{'quantity':34s}{'PYTHIA mean':>14s}{'std':>10s}"
    if st:
        head += f"{'standalone mean':>18s}{'std':>10s}{'diff/err':>10s}"
    print(head)
    for k in KEYS:
        row(k, acc[k], st["counts"].get(k) if st else None)
    for name, d in fam.items():
        for q in ("tau_ps", "L", "Lxy", "gamma", "n_direct", "n_final", "n_charged_final", "n_photons_final", "n_neutrinos", "e_nu", "e_nu_frac", "pt_lepton",
                  "n_tracks", "n_tracks_d0_100um", "max_d0", "L_charm"):
            row(f"{name} {q}", d[q], st["b"][name][q] if st and name in st["b"] else None)
        row(f"{name} d0 of the tracks", d["d0"], st["b"][name]["d0"] if st and name in st["b"] else None)
        row(f"{name} oscillated", [float(x) for x in d["oscillated"]], [float(x) for x in st["b"][name]["oscillated"]] if st and name in st["b"] else None)
        row(f"{name} lepton from the B", [float(x) for x in d["lepton_from_b"]], [float(x) for x in st["b"][name]["lepton_from_b"]] if st and name in st["b"] else None)
        for R in (0.4, 0.8):
            for c in ("primaries", "final", "visible", "charged"):
                for q in ("pt", "m", "n"):
                    stv = st["b"][name]["cones"][str(R)][c][q] if st and name in st["b"] else None
                    row(f"{name} R={R} {c} {q}", d["cones"][R][c][q], stv)
            row(f"{name} R={R} B share", d["cones"][R]["frac_B"], st["b"][name]["cones"][str(R)]["frac_B"] if st and name in st["b"] else None)
            row(f"{name} R={R} nu pT in cone", d["cones"][R]["nu_pt_in_cone"], st["b"][name]["cones"][str(R)]["nu_pt_in_cone"] if st and name in st["b"] else None)
        tot = len(d["cls"])
        cls = count(d["cls"])
        print(f"  {name} decay classes (PYTHIA): " + ", ".join(f"{k} {100 * v_ / tot:.1f}%" for k, v_ in sorted(cls.items(), key=lambda kv: -kv[1])))
        if st and name in st["b"]:
            c2 = count(st["b"][name]["cls"])
            print(f"  {name} decay classes (standalone): " + ", ".join(f"{k} {100 * v_ / len(st['b'][name]['cls']):.1f}%" for k, v_ in sorted(c2.items(), key=lambda kv: -kv[1])))
        hd = count(d["hadron"])
        print(f"  {name} weakly decaying hadron (PYTHIA): " + ", ".join(f"{k} {100 * v_ / tot:.1f}%" for k, v_ in sorted(hd.items(), key=lambda kv: -kv[1])[:6]))
        ch = count(d["charm"])
        print(f"  {name} charm hadron (PYTHIA): " + ", ".join(f"{k or 'none'} {100 * v_ / tot:.1f}%" for k, v_ in sorted(ch.items(), key=lambda kv: -kv[1])[:8]))
        if st and name in st["b"]:
            c2 = count(st["b"][name]["charm"])
            print(f"  {name} charm hadron (standalone): " + ", ".join(f"{k or 'none'} {100 * v_ / len(st['b'][name]['charm']):.1f}%" for k, v_ in sorted(c2.items(), key=lambda kv: -kv[1])[:8]))
    tot = sum(species.values())
    print("species (PYTHIA, % of all final particles): " + ", ".join(f"{h} {100 * k / tot:.1f}" for h, k in sorted(species.items(), key=lambda kv: -kv[1])[:16]))
    if st:
        tot_s = sum(st["species"].values())
        print("species (standalone): " + ", ".join(f"{h} {100 * k / tot_s:.1f}" for h, k in sorted(st["species"].items(), key=lambda kv: -kv[1])[:16]))
    args.output.write_text(json.dumps(dict(mode=args.mode, nevents=ndone, seed=args.seed, no_mix=args.no_mix, counts=acc, species=species, b=fam, **extra),
                                      default=float) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
