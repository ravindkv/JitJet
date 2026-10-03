#!/usr/bin/env python3
"""Scale-variation scan of the q qbar -> b bbar Born cross section with the full integrator.

Runs qqbar_bbbar_xsec.py (LHAPDF + PYTHIA alpha_s, so it needs the .venv of
README.md) for a grid of renormalisation and factorisation scale factors and
writes scale_scan.json, which plot_theory_ME.py turns into the scale-variation
figure of chapter 1. The standalone script cannot do this scan itself: its
embedded alpha_s table does not cover mu_R x 2 at the top of the phase space.

    .venv/bin/python scan_scales.py                  # -> scale_scan.json (about 2 min)
    .venv/bin/python scan_scales.py --power 13 --repeats 2 --quick

Modes: "mur" varies mu_R only, "muf" mu_F only, "both" the two together; the
factors are 2^(k/4) for k = -4 .. 4, so that 1/2, 1 and 2 are on the grid and
the seven-point envelope can be read off the same file.
"""
import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qqbar_bbbar_xsec import Config, integrate  # noqa: E402

FACTORS = [2 ** (k / 4) for k in range(-4, 5)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--power", type=int, default=14, help="2**power Sobol points per repeat")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="only the factors 1/2, 1, 2")
    ap.add_argument("-o", "--output", type=Path, default=HERE / "scale_scan.json")
    a = ap.parse_args()
    factors = [0.5, 1.0, 2.0] if a.quick else FACTORS
    base = Config(power=a.power, repeats=a.repeats)
    out = {"config": {"power": a.power, "repeats": a.repeats, "ptmin": base.ptmin, "ecm": base.ecm, "mb": base.mb},
           "points": []}
    for mode in ("mur", "muf", "both"):
        for f in factors:
            cfg = replace(base)
            if mode in ("mur", "both"):
                cfg.renorm_mult_fac = f * f
            if mode in ("muf", "both"):
                cfg.factor_mult_fac = f * f
            res = integrate(cfg, progress=False)
            out["points"].append({"mode": mode, "factor": f, "log2_factor": math.log2(f),
                                  "sigma_pb": res["sigma_pb"], "error_pb": res["error_pb"],
                                  "by_flavour_pb": {k: v["sigma"] for k, v in res["by_flavour_pb"].items()}})
            print(f"{mode:5s} x{f:6.3f}: {res['sigma_pb']:9.3f} +/- {res['error_pb']:.3f} pb", flush=True)
    a.output.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", a.output)


if __name__ == "__main__":
    main()
