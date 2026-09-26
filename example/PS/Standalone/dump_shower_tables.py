#!/usr/bin/env python3
"""Dump the PDF grid embedded in standalone_parton_showering.py.

Needs the full environment of example/ME/Standalone (LHAPDF + pythia8mc):
  ../../ME/Standalone/.venv/bin/python dump_shower_tables.py          # rewrite the standalone file
  ../../ME/Standalone/.venv/bin/python dump_shower_tables.py --json t.json

The initial-state (backward) shower needs x*f(x, Q^2) for every flavour and
the gluon from the shower cut-off (Q ~ 1 GeV, frozen at the grid edge
Q_min = 1.65 GeV) up to the hard scale pTHat ~ 113 GeV, and for every x from
the smallest incoming x of the event (2e-3) up to 1. The knots are LHAPDF's own
grid knots, as in example/ME/Standalone/dump_standalone_tables.py, so the
bicubic interpolation of the standalone script passes through the same points
LHAPDF interpolates between.

The script also prints PYTHIA's Lambda_3,4,5 and a few alpha_s values so the
analytic AlphaStrong class of the standalone script can be checked against the
generator (it reproduces them to better than 1e-4).
"""
import argparse, json, re, sys
from pathlib import Path
import lhapdf

PDF, MEMBER = "NNPDF31_nnlo_as_0118", 0
XMIN_KEEP, Q2MAX_KEEP = 1e-3, 2.0e4       # x only grows in backward evolution; pTmax^2 ~ 1.3e4
FLAVOURS = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 21]
BEGIN, END = "# BEGIN GENERATED TABLES", "# END GENERATED TABLES"


def knots_from_dat(path):
    xs, qs = set(), set()
    for block in path.read_text().split("---\n")[1:]:
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        xs.update(float(v) for v in lines[0].split())
        qs.update(float(v) for v in lines[1].split())
    return sorted(xs), sorted(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    ap.add_argument("--target", type=Path, default=Path(__file__).with_name("standalone_parton_showering.py"))
    a = ap.parse_args()
    pdf = lhapdf.mkPDF(PDF, MEMBER)
    dat = None
    for p in lhapdf.paths():
        cand = Path(p) / PDF / f"{PDF}_{MEMBER:04d}.dat"
        if cand.exists():
            dat = cand
    if dat is None:
        sys.exit("PDF grid file not found in lhapdf.paths()")
    xk, qk = knots_from_dat(dat)
    xk = [x for x in xk if x >= XMIN_KEEP]
    q2k = [q * q for q in qk]
    ihi = min(i for i, v in enumerate(q2k) if v >= Q2MAX_KEEP)
    q2k = q2k[: ihi + 1]
    grid = [[[float(f"{pdf.xfxQ2(f, x, q2):.6g}") for q2 in q2k] for x in xk] for f in FLAVOURS]
    tables = {"pdf_set": PDF, "member": MEMBER, "lhapdf_version": lhapdf.version(),
              "x_knots": xk, "q2_knots": q2k, "flavours": FLAVOURS, "xf": grid}
    try:
        import pythia8mc as pythia8
        al = pythia8.AlphaStrong(); al.init(0.118, 2, 6, False)
        print(f"PYTHIA AlphaStrong(0.118, order 2): Lambda3 = {al.Lambda3():.6f}, Lambda4 = {al.Lambda4():.6f}, "
              f"Lambda5 = {al.Lambda5():.6f}")
        for q2 in (1.0, 4.0, 25.0, 100.0, 1.0e4):
            print(f"  alpha_s({q2:g} GeV^2) = {al.alphaS(q2):.6f}")
    except ImportError:
        print("pythia8mc not available: skipping the alpha_s reference values")
    if a.json:
        a.json.write_text(json.dumps(tables) + "\n"); print("wrote", a.json); return
    text = a.target.read_text()
    block = (f"{BEGIN}\n# produced by dump_shower_tables.py; do not edit by hand\n"
             f"TABLES = {json.dumps(tables, separators=(',', ':'))}\n{END}")
    new, n = re.subn(rf"{BEGIN}.*?{END}", lambda m: block, text, flags=re.S)
    if n != 1:
        sys.exit(f"markers not found in {a.target}")
    a.target.write_text(new)
    print(f"updated {a.target}: {len(xk)} x knots, {len(q2k)} Q2 knots ({q2k[0]:.3g}..{q2k[-1]:.3g} GeV^2), "
          f"{len(FLAVOURS)} flavours")


if __name__ == "__main__":
    main()
