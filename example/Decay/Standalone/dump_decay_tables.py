#!/usr/bin/env python3
"""Write decay_tables.json, the particle properties and decay tables that
standalone_hadron_decays.py needs, taken from PYTHIA 8's ParticleData, so that the
standalone script needs no PYTHIA.

Needs the environment of example/ME/Standalone (pythia8mc):
  ../../ME/Standalone/.venv/bin/python dump_decay_tables.py            # -> decay_tables.json
  ../../ME/Standalone/.venv/bin/python dump_decay_tables.py --print    # summary only

The set of particles is the closure of "everything the chapter-5 string can make" (the
78 hadrons embedded in ../../Had/Standalone/standalone_hadronisation.py plus whatever
appears in ../../Had/event_Had.lhe) under "appears among the decay products of": every
hadron, lepton, photon, quark and diquark that can show up in the decay chains, with
    id, name, antiName, m0, mWidth, mMin, mMax [GeV], tau0 [mm/c], 3 x charge, spinType,
    mayDecay (PYTHIA's default: tau0 < 1000 mm), constituent mass [GeV], and the list of
    channels (bRatio, meMode, onMode, products), exactly as PYTHIA 8.315 holds them
    (the branching ratios of a particle need not sum to one; PYTHIA picks a channel in
    proportion to its bRatio among the open ones, and so does the standalone script).
Also the settings of ParticleDecays that the script uses (tau0Max, the B-mixing x_d, x_s,
sigmaSoft, multIncrease, multIncreaseWeak, multRefMass, multGoffset, colRearrange, mSafety).
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "decay_tables.json"
HAD = HERE.parent.parent / "Had"


def is_diquark(pid: int) -> bool:
    a = abs(pid)
    return 1000 < a < 10000 and (a // 10) % 10 == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", action="store_true")
    ap.add_argument("-o", "--output", type=Path, default=TARGET)
    args = ap.parse_args()
    try:
        import pythia8mc as pythia8
    except ImportError:
        import pythia8
    py = pythia8.Pythia("", False)
    pd = py.particleData
    st = py.settings

    # the seeds of the closure: the chapter-5 hadron table and the record
    sys.path.insert(0, str(HAD / "Standalone"))
    import standalone_hadronisation as had  # noqa: E402
    seeds = set(had.HADRONS) | set(had.DIQUARK_M0) | {1, 2, 3, 4, 5, 21, 22, 11, 12, 13, 14, 15, 16}
    rec = HAD / "event_Had.lhe"
    if rec.exists():
        for line in rec.read_text().splitlines():
            f = line.split()
            if len(f) >= 15 and f[0].isdigit():
                seeds.add(abs(int(f[1])))
    todo, seen = sorted(seeds), set()
    while todo:
        a = abs(todo.pop())
        if a in seen or not pd.isParticle(a):
            continue
        seen.add(a)
        e = pd.findParticle(a)
        for c in range(e.sizeChannels()):
            ch = e.channel(c)
            for k in range(ch.multiplicity()):
                q = abs(ch.product(k))
                if q not in seen and pd.isParticle(q):
                    todo.append(q)
    particles = {}
    nch = 0
    for a in sorted(seen):
        e = pd.findParticle(a)
        channels = []
        for c in range(e.sizeChannels()):
            ch = e.channel(c)
            channels.append([ch.bRatio(), ch.meMode(), ch.onMode(), [ch.product(k) for k in range(ch.multiplicity())]])
        nch += len(channels)
        particles[str(a)] = dict(name=e.name(), anti=pd.name(-a) if e.hasAnti() else e.name(), m0=e.m0(), mWidth=e.mWidth(),
                                 mMin=e.mMin(), mMax=e.mMax(), tau0=e.tau0(), charge3=int(round(3 * e.charge())),
                                 spinType=e.spinType(), mayDecay=bool(e.mayDecay()), isResonance=bool(e.isResonance()),
                                 mConst=e.constituentMass(), channels=channels)
    settings = {k: st.parm(k) for k in ("ParticleDecays:tau0Max", "ParticleDecays:xBdMix", "ParticleDecays:xBsMix",
                                          "ParticleDecays:mSafety", "ParticleDecays:sigmaSoft", "ParticleDecays:multIncrease",
                                          "ParticleDecays:multIncreaseWeak", "ParticleDecays:multRefMass",
                                          "ParticleDecays:multGoffset", "ParticleDecays:colRearrange")}
    settings["ParticleDecays:limitTau0"] = True        # what CMSSW sets (the GEN log of chapter 2); PYTHIA's default is off
    settings["ParticleDecays:mixB"] = st.flag("ParticleDecays:mixB")
    version = getattr(pythia8, "__version__", "8.315")
    out = dict(source=f"PYTHIA {version} ParticleData, dumped by dump_decay_tables.py", settings=settings, particles=particles)
    if args.print:
        dec = [a for a in sorted(seen) if particles[str(a)]["channels"] and particles[str(a)]["mayDecay"] and particles[str(a)]["tau0"] < 10]
        print(f"{len(particles)} particles, {nch} channels; {len(dec)} decay with limitTau0 (tau0 < 10 mm)")
        for a in sorted(seen):
            p = particles[str(a)]
            print(f"{a:7d} {p['name']:16s} m0 = {p['m0']:.4f}  tau0 = {p['tau0']:.4g} mm  {len(p['channels'])} channels")
        return
    args.output.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote {args.output}: {len(particles)} particles, {nch} channels")


if __name__ == "__main__":
    main()
