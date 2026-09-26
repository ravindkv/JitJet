#!/usr/bin/env python3
"""Shower the same hard event with PYTHIA 8 and compare with the standalone shower.

Needs the full environment of example/ME/Standalone (pythia8mc + LHAPDF):
  ../../ME/Standalone/.venv/bin/python validate_shower_with_pythia.py -n 2000
  ../../ME/Standalone/.venv/bin/python validate_shower_with_pythia.py -n 2000 --rapidity-order

The hard process of ../../ME/event_ME.lhe is written N times into a Les Houches
file and handed to PYTHIA with MPI, hadronisation and QED radiation switched
off, so only the interleaved ISR + FSR shower acts, with the CP5 shower
settings of the CMSSW configuration (alpha_s(MZ) = 0.118 at second order for
both showers, NNPDF31_nnlo_as_0118) and the starting scale SCALUP =
sqrt(pTHat^2 + mb^2) = 113.32 GeV. The same per-event quantities as the
standalone script's --repeat mode are averaged: number of ISR and FSR
branchings, final-state parton multiplicity, the pT of each hard b quark after
the shower and the pT, multiplicity and mass of its family (final partons
within dR < 0.4), and the x of the incoming partons after ISR.

With --standalone-stats FILE (the .stats.json written by
standalone_parton_showering.py --repeat N --history FILE.json) both sets of
averages are printed side by side.

PYTHIA's default SpaceShower:rapidityOrder = on is switched off unless
--rapidity-order is given, because the standalone shower does not implement
that extra ordering; both numbers are worth knowing.
"""
import argparse, importlib.util, json, math, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_standalone():
    spec = importlib.util.spec_from_file_location("sps", HERE / "standalone_parton_showering.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_lhef(ev, sps, path: Path, n: int, scalup: float, sigma=345.1, dsigma=6.1):
    ebeam = ev.ebeam
    hard_in = [ev.inA, ev.inB]
    hard_out = [i for i, q in enumerate(ev.p) if q.status == 23]
    rows = []
    for k, i in enumerate(hard_in + hard_out):
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


def pythia_listing(e, sps) -> str:
    """PYTHIA's event record in the column format of the standalone listing."""
    title = "PYTHIA 8 Event Listing  (parton level: LHEF hard process + interleaved ISR/FSR shower)"
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=HERE.parent.parent / "ME" / "event_ME.lhe")
    ap.add_argument("-n", "--nevents", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--radius", type=float, default=0.4)
    ap.add_argument("--rapidity-order", action="store_true", help="keep PYTHIA's SpaceShower:rapidityOrder = on")
    ap.add_argument("--no-isr", action="store_true", help="PartonLevel:ISR = off (compare with the standalone --no-isr)")
    ap.add_argument("--ptmin-fsr", type=float, default=0.5, help="TimeShower:pTmin")
    ap.add_argument("--standalone-stats", type=Path, help=".stats.json from standalone_parton_showering.py --repeat")
    ap.add_argument("--output", type=Path, default=HERE / "validate_shower_with_pythia.json")
    ap.add_argument("--lhef", type=Path, default=HERE / "validate_shower_with_pythia.lhe")
    ap.add_argument("--listing", type=Path, default=HERE / "validate_shower_with_pythia_event1.txt",
                    help="PYTHIA's own record of the first showered event")
    args = ap.parse_args()

    try:
        import pythia8mc as pythia8
    except ImportError:
        import pythia8
    sps = load_standalone()
    ev = sps.read_event(args.input)
    hard = [i for i, q in enumerate(ev.p) if q.status == 23]
    m2avg = sum(ev.p[i].m ** 2 for i in hard) / len(hard)
    scalup = math.sqrt(sps.pt(ev.p[hard[0]].p) ** 2 + m2avg)
    write_lhef(ev, sps, args.lhef, args.nevents, scalup)

    # the pythia8mc wheel has no LHAPDF plugin: read the LHAPDF grid file with PYTHIA's own LHAGrid1 loader
    pset = "20"                                   # fallback: internal NNPDF3.1 NNLO (QCD+LUXQED)
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
    for s in ["Beams:frameType = 4", f"Beams:LHEF = {args.lhef}", f"PDF:pSet = {pset}",
              "Tune:pp = 14", "Tune:ee = 7",
              "SpaceShower:alphaSorder = 2", "SpaceShower:alphaSvalue = 0.118",
              "TimeShower:alphaSorder = 2", "TimeShower:alphaSvalue = 0.118",
              f"SpaceShower:rapidityOrder = {'on' if args.rapidity_order else 'off'}",
              "PartonLevel:MPI = off", "HadronLevel:all = off", "ColourReconnection:reconnect = off",
              f"PartonLevel:ISR = {'off' if args.no_isr else 'on'}", f"TimeShower:pTmin = {args.ptmin_fsr}",
              "TimeShower:QEDshowerByQ = off", "TimeShower:QEDshowerByL = off",
              "SpaceShower:QEDshowerByQ = off", "SpaceShower:QEDshowerByL = off",
              "Check:epTolErr = 0.01", "Next:numberCount = 0", "Next:numberShowEvent = 0",
              "Next:numberShowProcess = 0", "Next:numberShowInfo = 0",
              "Random:setSeed = on", f"Random:seed = {args.seed}"]:
        py.readString(s)
    if not py.init():
        sys.exit("PYTHIA initialisation failed")

    keys = ["n_isr", "n_fsr", "n_final", "n_gluons", "n_pt1", "n_pt5", "n_pt20", "x_A", "x_B"]
    acc = {k: [] for k in keys}
    fam = {}
    ebeam = ev.ebeam
    for n in range(args.nevents):
        if not py.next():
            if py.info.atEndOfFile():
                break
            continue
        e = py.event
        finals = [i for i in range(e.size()) if e[i].isFinal() and e[i].status() != 63]
        psum = sum((np.array([e[i].e(), e[i].px(), e[i].py(), e[i].pz()]) for i in finals), np.zeros(4))
        acc["n_isr"].append(sum(1 for i in range(e.size()) if abs(e[i].status()) == 43))
        acc["n_fsr"].append(sum(1 for i in range(e.size()) if abs(e[i].status()) == 51) // 2)
        acc["n_final"].append(len(finals))
        acc["n_gluons"].append(sum(1 for i in finals if e[i].id() == 21))
        for cut, key in ((1.0, "n_pt1"), (5.0, "n_pt5"), (20.0, "n_pt20")):
            acc[key].append(sum(1 for i in finals if e[i].pT() > cut))
        acc["x_A"].append(0.5 * (psum[0] + psum[3]) / ebeam)
        acc["x_B"].append(0.5 * (psum[0] - psum[3]) / ebeam)
        for i in range(e.size()):
            if abs(e[i].status()) == 23 and abs(e[i].id()) == 5:
                name = sps.pname(e[i].id())
                j = e[i].iBotCopyId()
                d = fam.setdefault(name, dict(pt_after=[], fam_pt=[], fam_n=[], fam_m=[]))
                members = [k for k in finals if delta_r(e[k], e[j]) < args.radius]
                pf = sum((np.array([e[k].e(), e[k].px(), e[k].py(), e[k].pz()]) for k in members), np.zeros(4))
                d["pt_after"].append(e[j].pT())
                d["fam_pt"].append(math.hypot(pf[1], pf[2]))
                d["fam_n"].append(len(members))
                d["fam_m"].append(math.sqrt(max(pf[0] ** 2 - pf[1] ** 2 - pf[2] ** 2 - pf[3] ** 2, 0.0)))
        if n == 0 and args.listing:
            args.listing.write_text(pythia_listing(e, sps))
    py.stat()
    ndone = len(acc["n_isr"])
    print(f"\nPYTHIA 8 showered the hard event {ndone} times (ISR {'off' if args.no_isr else 'on'}, "
          f"rapidityOrder {'on' if args.rapidity_order else 'off'}, SCALUP = {scalup:.2f} GeV)")
    st = None
    if args.standalone_stats and args.standalone_stats.exists():
        st = json.loads(args.standalone_stats.read_text())
        print(f"standalone shower: {st['repeat']} showers from {args.standalone_stats}")
    head = f"{'quantity':22s}{'PYTHIA mean':>14s}{'std':>10s}"
    if st:
        head += f"{'standalone mean':>18s}{'std':>10s}{'diff/err':>10s}"
    print(head)

    def row(label, v_py, v_st):
        v_py = np.asarray(v_py, dtype=float)
        line = f"{label:22s}{v_py.mean():14.4f}{v_py.std():10.4f}"
        if v_st is not None:
            v_st = np.asarray(v_st, dtype=float)
            err = math.hypot(v_py.std() / math.sqrt(len(v_py)), v_st.std() / math.sqrt(len(v_st)))
            line += f"{v_st.mean():18.4f}{v_st.std():10.4f}{(v_st.mean() - v_py.mean()) / err if err > 0 else 0:10.2f}"
        print(line)

    for k in keys:
        row(k, acc[k], st["counts"][k] if st and k in st["counts"] else None)
    for name, d in fam.items():
        for q, label in (("pt_after", "quark pT after"), ("fam_pt", "family pT"), ("fam_n", "family n"),
                         ("fam_m", "family mass")):
            row(f"{name} {label}", d[q], st["families"][name][q] if st and name in st["families"] else None)
    args.output.write_text(json.dumps(dict(nevents=ndone, seed=args.seed, scalup=scalup,
                                           rapidity_order=args.rapidity_order, counts=acc, families=fam),
                                      default=float) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
