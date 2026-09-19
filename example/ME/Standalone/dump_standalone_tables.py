#!/usr/bin/env python3
"""Dump the PDF and alpha_s tables embedded in standalone_qqbar_bbbar_xsec.py.

Needs the full environment (LHAPDF + pythia8/pythia8mc), see README.md:
  .venv/bin/python dump_standalone_tables.py            # rewrite the standalone file
  .venv/bin/python dump_standalone_tables.py --json t.json

x knots and Q knots are LHAPDF's own grid knots (subset covering the default
phase space), so the standalone cubic interpolation passes through the same
points LHAPDF interpolates between.
"""
import argparse, importlib, json, math, re, sys
from pathlib import Path
import numpy as np
import lhapdf

PDF, MEMBER = "NNPDF31_nnlo_as_0118", 0
ECM, MB, PTMIN = 13600.0, 4.8, 30.0
ALPHAS_MZ, ALPHA_ORDER, ALPHA_NFMAX = 0.118, 2, 6
XMIN_KEEP = 1e-6          # tau_min ~2e-5 for pTHat>=30; keep some margin below
BEGIN, END = "# BEGIN GENERATED TABLES", "# END GENERATED TABLES"


def knots_from_dat(path):
    xs, qs = set(), set()
    blocks = path.read_text().split("---\n")[1:]
    for b in blocks:
        lines = b.splitlines()
        if len(lines) < 3:
            continue
        xs.update(float(v) for v in lines[0].split())
        qs.update(float(v) for v in lines[1].split())
    return sorted(xs), sorted(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    ap.add_argument("--target", type=Path, default=Path("standalone_qqbar_bbbar_xsec.py"))
    a = ap.parse_args()
    pdf = lhapdf.mkPDF(PDF, MEMBER)
    dat = Path(lhapdf.paths()[0]) / PDF / f"{PDF}_{MEMBER:04d}.dat"
    for p in lhapdf.paths():
        if (Path(p) / PDF / dat.name).exists():
            dat = Path(p) / PDF / dat.name
    xk, qk = knots_from_dat(dat)
    s = ECM**2
    q2lo, q2hi = PTMIN**2 + MB**2, s / 4
    # keep knots covering [q2lo, q2hi] with one flanking knot each side
    q2k = [q * q for q in qk]
    ilo = max(i for i, v in enumerate(q2k) if v <= q2lo)
    ihi = min(i for i, v in enumerate(q2k) if v >= q2hi)
    q2k = q2k[max(ilo - 1, 0): ihi + 2]
    xk = [x for x in xk if x >= XMIN_KEEP]
    flav = [1, 2, 3, 4, 5, -1, -2, -3]      # -4,-5 symmetrised in the integrand
    grid = [[[pdf.xfxQ2(f, x, q2) for q2 in q2k] for x in xk] for f in flav]
    mod = None
    for name in ("pythia8", "pythia8mc"):
        try:
            mod = importlib.import_module(name); break
        except ImportError:
            pass
    if mod is None:
        sys.exit("need pythia8 or pythia8mc for the alpha_s table")
    al = mod.AlphaStrong(); al.init(ALPHAS_MZ, ALPHA_ORDER, ALPHA_NFMAX, False)
    aq2 = np.geomspace(q2lo * 0.9, q2hi * 1.1, 120)
    tables = {
        "pdf_set": PDF, "member": MEMBER, "lhapdf_version": lhapdf.version(),
        "pythia_alpha_s": {"alphas_mz": ALPHAS_MZ, "order": ALPHA_ORDER, "nfmax": ALPHA_NFMAX},
        "x_knots": xk, "q2_knots": q2k, "flavours": flav,
        "xf": [[[round(v, 8) for v in row] for row in g] for g in grid],
        "alpha_q2": [float(v) for v in aq2],
        "alpha_s": [round(al.alphaS(float(v)), 10) for v in aq2],
    }
    if a.json:
        a.json.write_text(json.dumps(tables) + "\n"); print("wrote", a.json); return
    text = a.target.read_text()
    block = f"{BEGIN}\n# produced by dump_standalone_tables.py; do not edit by hand\nTABLES = {json.dumps(tables, separators=(',', ':'))}\n{END}"
    new, n = re.subn(rf"{BEGIN}.*?{END}", lambda m: block, text, flags=re.S)
    if n != 1:
        sys.exit(f"markers not found in {a.target}")
    a.target.write_text(new); print(f"updated {a.target}: {len(xk)} x knots, {len(q2k)} Q2 knots, {len(aq2)} alpha_s points")


if __name__ == "__main__":
    main()
