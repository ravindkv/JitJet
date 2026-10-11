#!/usr/bin/env python3
"""Write the particle tables that standalone_hadronisation.py embeds, taken from PYTHIA 8's
ParticleData (names, masses, charges of the hadrons the Lund string can make, and the
constituent masses of quarks and diquarks), so that the standalone script needs no PYTHIA.

Needs the environment of example/ME/Standalone (pythia8mc):
  ../../ME/Standalone/.venv/bin/python dump_hadron_tables.py            # rewrites the block in standalone_hadronisation.py
  ../../ME/Standalone/.venv/bin/python dump_hadron_tables.py --print    # show the table only

The table holds the lowest-lying multiplets that the flavour selection of the string model
produces: the pseudoscalar and vector meson nonets with u, d, s plus the open-charm, open-bottom
and quarkonium ground states, and the octet and decuplet baryons with up to one heavy quark.
Every entry: id -> (name, name of the antiparticle, m0 [GeV], 3 x charge).
"""
import argparse, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "standalone_hadronisation.py"

LIGHT = (1, 2, 3)
HEAVY = (4, 5)


def meson_ids():
    ids = set()
    flavs = LIGHT + HEAVY
    for q in flavs:
        for a in flavs:
            for spin in (1, 3):
                if q == a:
                    ids.add(100 * q + 10 * q + spin)        # diagonal: pi0/eta/eta'/rho0/omega/phi, cc, bb
                else:
                    hi, lo = max(q, a), min(q, a)
                    ids.add(100 * hi + 10 * lo + spin)      # the id of the meson with |content|; sign added at use
    ids |= {221, 331, 223, 333}
    return sorted(ids)


def baryon_ids():
    ids = set()
    flavs = LIGHT + HEAVY
    for q1 in flavs:
        for q2 in flavs:
            for q3 in flavs:
                trio = sorted((q1, q2, q3), reverse=True)
                if sum(1 for q in trio if q in HEAVY) > 1:
                    continue
                a, b, c = trio
                # octet / "Lambda-like" (light pair antisymmetric) and decuplet
                if a == b == c:
                    ids.add(1000 * a + 100 * b + 10 * c + 4)
                    continue
                ids.add(1000 * a + 100 * b + 10 * c + 2)
                ids.add(1000 * a + 100 * b + 10 * c + 4)
                if b != c and a != b:
                    ids.add(1000 * a + 100 * c + 10 * b + 2)   # Lambda-like: the two lighter quarks in spin 0
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    try:
        import pythia8mc as pythia8
    except ImportError:
        import pythia8
    py = pythia8.Pythia("", False)
    pd = py.particleData
    table = {}
    for pid in meson_ids() + baryon_ids():
        if not pd.isParticle(pid):
            continue
        table[pid] = (pd.name(pid), pd.name(-pid) if pd.hasAnti(pid) else pd.name(pid), round(pd.m0(pid), 5),
                      int(round(3 * pd.charge(pid))))
    quarks = {q: round(pd.m0(q), 4) for q in (1, 2, 3, 4, 5)}
    diquarks = {}
    for pid in (1103, 2101, 2103, 2203, 3101, 3103, 3201, 3203, 3303):
        diquarks[pid] = (pd.name(pid), round(pd.m0(pid), 5))
    out = dict(source="PYTHIA 8 ParticleData (pythia8mc " + pythia8.__version__ if hasattr(pythia8, "__version__") else "PYTHIA 8 ParticleData",
               hadrons={str(k): v for k, v in table.items()}, quarks={str(k): v for k, v in quarks.items()},
               diquarks={str(k): v for k, v in diquarks.items()})
    text = json.dumps(out, separators=(",", ":"))
    if args.print or not TARGET.exists():
        for k, v in table.items():
            print(k, v)
        print(quarks, diquarks)
        print(f"{len(table)} hadrons")
        return
    src = TARGET.read_text()
    new = re.sub(r"(# BEGIN GENERATED TABLES\n)(.*?)(# END GENERATED TABLES)",
                 lambda m: m.group(1) + "# produced by dump_hadron_tables.py from PYTHIA 8's ParticleData; do not edit by hand\n"
                 + "TABLES = " + text + "\n" + m.group(3), src, flags=re.S)
    if new == src:
        sys.exit("no BEGIN/END GENERATED TABLES block found in " + str(TARGET))
    TARGET.write_text(new)
    print(f"wrote {len(table)} hadrons, {len(quarks)} quarks and {len(diquarks)} diquarks into {TARGET}")


if __name__ == "__main__":
    main()
