#!/usr/bin/env python3
"""Dump the PDF grid embedded in standalone_multiparton_interactions.py.

Needs the full environment of example/ME/Standalone (LHAPDF + pythia8mc):
  ../../ME/Standalone/.venv/bin/python dump_mpi_tables.py          # rewrite the standalone file
  ../../ME/Standalone/.venv/bin/python dump_mpi_tables.py --json t.json

Multiparton interactions probe the proton at much smaller x than the hard
process or its shower: a scattering at pT = 0.2 GeV and rapidity 9 takes
x ~ 1e-8 from one proton. The grid of chapter 2 (x >= 1e-3) is therefore not
enough, and this file embeds x f(x, Q^2) for every flavour and the gluon from
x = 1e-9 (the lowest knot of the set) to 1, and
from the lowest Q of the set (1.65 GeV, frozen below) up to the first knot
above the hard scale 113 GeV, which is the largest scale a secondary
interaction of our event can have. The knots are LHAPDF's own grid knots, so
the bicubic interpolation of the standalone script passes through the same
points LHAPDF interpolates between.
"""
import argparse, json, re, sys
from pathlib import Path
import lhapdf

PDF, MEMBER = "NNPDF31_nnlo_as_0118", 0
XMIN_KEEP, Q2MAX_KEEP = 1e-9, 1.3e4       # all x knots of the set; pTmax^2 of our event = 113.3^2
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
    ap.add_argument("--target", type=Path, default=Path(__file__).with_name("standalone_multiparton_interactions.py"))
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
    grid = [[[float(f"{pdf.xfxQ2(f, x, q2):.5g}") for q2 in q2k] for x in xk] for f in FLAVOURS]
    tables = {"pdf_set": PDF, "member": MEMBER, "lhapdf_version": lhapdf.version(),
              "x_knots": xk, "q2_knots": q2k, "flavours": FLAVOURS, "xf": grid}
    if a.json:
        a.json.write_text(json.dumps(tables) + "\n"); print("wrote", a.json); return
    text = a.target.read_text()
    block = (f"{BEGIN}\n# produced by dump_mpi_tables.py; do not edit by hand\n"
             f"TABLES = {json.dumps(tables, separators=(',', ':'))}\n{END}")
    new, n = re.subn(rf"{BEGIN}.*?{END}", lambda m: block, text, flags=re.S)
    if n != 1:
        sys.exit(f"markers not found in {a.target}")
    a.target.write_text(new)
    print(f"updated {a.target}: {len(xk)} x knots ({xk[0]:.3g}..1), {len(q2k)} Q2 knots "
          f"({q2k[0]:.3g}..{q2k[-1]:.3g} GeV^2), {len(FLAVOURS)} flavours")


if __name__ == "__main__":
    main()
