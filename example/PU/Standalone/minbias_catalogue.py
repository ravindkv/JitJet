#!/usr/bin/env python3
"""The catalogue of the minimum-bias library: one line per collision, as minBiasGenStats.py of the CMSSW
companion prints it for the 250 GEN events of the 2024 library, plus the summary of the sample and the
means per process type.

For every collision of ../pileup_library.lhe.gz: the process type; the number of parton-parton
interactions (the hardest one plus the MPIs, PYTHIA's Info::nMPI, written by make_pileup_library.py; at least one, since
PYTHIA reports no MPI for a diffractive system that had only its hardest scattering);
the stable particles, the charged ones, the central ones (|eta| < 2.5 and pT > 0.5 GeV, all species, as
the CMSSW catalogue counts them), the ones beyond |eta| = 5.5 (where no CMS subdetector is); the pT of the
hardest particle and the scalar sum of pT; the anti-kT R = 0.4 jets above 10 GeV (chapter 7's clustering
on the collision alone, without neutrinos); the forward protons (final protons that keep most of the beam
energy, xi = 1 - E/E_beam < 0.55, the ones CMSSW's LHCTransport carries to the PPS Roman pots); and, for
the collisions of our in-time crossing, the vertex z and t the mixing drew for them (../event_PU.lhe).

Writes minbias_catalogue.json (every collision), minbias_catalogue.txt (the three tables as text) and
minbias_catalogue.tex (the same three tables as LaTeX tabulars, \\input by the chapter).

  /usr/bin/python3 minbias_catalogue.py              # 2000 collisions, about a minute (the jets)
  /usr/bin/python3 minbias_catalogue.py --no-jets    # a few seconds
  /usr/bin/python3 minbias_catalogue.py --first 20   # print twenty library events instead of the in-time ten
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import standalone_pileup as core  # noqa: E402

gj = core.gj
E_BEAM = 6800.0          # GeV, the Run-3 beam energy
XI_MAX = 0.55            # LHCTransport: protons that keep most of the beam energy (xi up to 0.55 in the CMSSW catalogue)
ETA_FAR = 5.5            # beyond the HF: nothing of CMS sees them (except the PPS, for the protons)
JET_MIN = 10.0
KINDS = ["ND", "SD", "DD", "CD"]
LONG = dict(ND="non-diffractive", SD="single diffractive", DD="double diffractive", CD="central diffractive")


def describe(e: dict, jets: bool) -> dict:
    ids, p4, ptv, etav = e["ids"], e["p4"], e["pt"], e["eta"]
    nu = e["nu"]
    central = (np.abs(etav) < 2.5) & (ptv > 0.5)
    far = np.abs(etav) > ETA_FAR
    prot = (np.abs(ids) == 2212) & (p4[:, 0] > (1 - XI_MAX) * E_BEAM)
    row = dict(event=e["no"], kind=e["kind"], nmpi=max(1, e["nmpi"]), pthat=e["pthat"], stable=int(len(ids)), charged=int(e["nch"]), central=int(central.sum()),
               far=int(far.sum()), lead_pt=float(ptv.max()), sum_pt=float(ptv.sum()), protons=int(prot.sum()), jets10=None)
    if jets:
        sel = ~nu & (ptv > 0)
        cl = gj.Clustering(p4[sel], gj.R_AK4, -1)
        row["jets10"] = int(sum(1 for j in cl.jets(JET_MIN)))
    return row


def stats(v):
    v = np.asarray(v, dtype=float)
    return dict(mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()), max=float(v.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", type=Path, default=HERE.parent / "pileup_library.lhe.gz")
    ap.add_argument("--record", type=Path, default=HERE.parent / "event_PU.lhe")
    ap.add_argument("--no-jets", action="store_true")
    ap.add_argument("--first", type=int, default=0, help="list the first N library events instead of the first ten in-time collisions of the record")
    ap.add_argument("-o", "--output", type=Path, default=HERE / "minbias_catalogue")
    args = ap.parse_args()
    t0 = time.time()
    lib = core.Library(args.library)
    rows = [describe(e, not args.no_jets) for e in lib.events]
    print(f"{len(rows)} collisions described in {time.time() - t0:.0f} s")
    # the vertices of the record: the in-time collisions with their library event and vertex
    summary, collisions, _, _, _ = core.read_pileup_tables(args.record) if args.record.exists() else ([], [], [], [], {})
    in_time = [c for c in collisions if c["bx"] == 0 and c["k"] > 0]
    all_vtx = [c for c in collisions if c["k"] > 0]
    by_event = {r["event"]: r for r in rows}
    if args.first:
        listed = [dict(by_event[n + 1], k=None, z=None, t=None) for n in range(min(args.first, len(rows)))]
        head = f"the first {len(listed)} collisions of the library"
    else:
        listed = [dict(by_event[c["lib"]], k=c["k"], z=c["z"], t=c["t"]) for c in in_time[:10]]
        head = "the first ten in-time collisions of our crossing (bx = 0), with the vertex the mixing drew for each"
    # the summary of the sample
    keys = [("nmpi", "interactions"), ("stable", "stable particles"), ("charged", "charged stable particles"), ("central", "central (|eta| < 2.5, pT > 0.5 GeV)"),
            ("far", "beyond |eta| = 5.5"), ("lead_pt", "pT of the hardest particle [GeV]"), ("sum_pt", "scalar sum of pT [GeV]")]
    if not args.no_jets:
        keys.append(("jets10", "ak4 jets above 10 GeV"))
    keys.append(("protons", "forward protons (xi < 0.55)"))
    summ = {k: stats([r[k] for r in rows]) for k, _ in keys}
    if all_vtx:
        summ["z"] = stats([c["z"] for c in all_vtx])
        summ["t"] = stats([c["t"] - c["bx"] * core.BUNCH_SPACING for c in all_vtx])
    per = {}
    for kd in KINDS:
        sel = [r for r in rows if r["kind"] == kd]
        per[kd] = dict(events=len(sel), nmpi=np.mean([r["nmpi"] for r in sel]), stable=np.mean([r["stable"] for r in sel]), charged=np.mean([r["charged"] for r in sel]),
                       central=np.mean([r["central"] for r in sel]), lead_pt=np.mean([r["lead_pt"] for r in sel]), sum_pt=np.mean([r["sum_pt"] for r in sel]),
                       sigma=core.SIGMA_INEL_MB * 0 + {"ND": 55.42, "SD": 12.86, "DD": 8.85, "CD": 1.29}[kd])
    # ---- text
    fmt = lambda v, d=1: ("--" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v)))
    lines = [f"# {head}", "event  proc  interactions  stable  charged  central  |eta|>5.5  lead pT  sum pT  jets>10  fwd p     z[cm]    t[ns]"]
    for r in listed:
        lines.append(f"{r['event']:5d}  {r['kind']:4s}  {r['nmpi']:12d}  {r['stable']:6d}  {r['charged']:7d}  {r['central']:7d}  {r['far']:9d}  {r['lead_pt']:7.2f}  {r['sum_pt']:6.1f}  "
                     f"{fmt(r['jets10']):>7s}  {r['protons']:5d}  {fmt(r['z'], 2):>8s}  {fmt(r['t'], 3):>7s}")
    lines += ["", "# per collision over the library           mean   median      min      max"]
    for k, lab in keys:
        s = summ[k]
        lines.append(f"{lab:40s} {s['mean']:8.2f} {s['median']:8.1f} {s['min']:8.1f} {s['max']:8.1f}")
    if "z" in summ:
        lines.append(f"{'vertex z [cm] (the 960 drawn)':40s} {summ['z']['mean']:8.2f} {summ['z']['median']:8.2f} {summ['z']['min']:8.1f} {summ['z']['max']:8.1f}")
        lines.append(f"{'vertex t [ns] (the 960 drawn)':40s} {summ['t']['mean']:8.3f} {summ['t']['median']:8.3f} {summ['t']['min']:8.2f} {summ['t']['max']:8.2f}")
    lines += ["", "# per process                   events  sigma[mb]  interactions  stable  charged  central  lead pT  sum pT"]
    for kd in KINDS:
        p = per[kd]
        lines.append(f"{LONG[kd] + ' (' + kd + ')':30s} {p['events']:6d}  {p['sigma']:8.2f}  {p['nmpi']:12.1f}  {p['stable']:6.0f}  {p['charged']:7.0f}  {p['central']:7.0f}  {p['lead_pt']:7.1f}  {p['sum_pt']:6.0f}")
    text = "\n".join(lines) + "\n"
    args.output.with_suffix(".txt").write_text(text)
    print(text)
    # ---- JSON
    args.output.with_suffix(".json").write_text(json.dumps(dict(library=str(args.library.name), listed=listed, summary=summ, per_process=per, rows=rows), indent=1, default=float))
    # ---- LaTeX
    tex = ["% written by minbias_catalogue.py: three tables of the minimum-bias library (do not edit)",
           "\\newcommand{\\minbiasCatalogueTen}{%",
           "\\begin{tabular}{rlrrrrrrrrrrr}\\toprule",
           "event & proc. & interactions & stable & charged & central & $|\\eta|>5.5$ & lead \\pt & $\\Sigma\\pt$ & jets $>10$ & fwd $p$ & $z$ [cm] & $t$ [ns] \\\\ \\midrule"]
    for r in listed:
        tex.append(f"{r['event']} & {r['kind']} & {r['nmpi']} & {r['stable']} & {r['charged']} & {r['central']} & {r['far']} & {r['lead_pt']:.2f} & {r['sum_pt']:.1f} & "
                   f"{fmt(r['jets10'])} & {r['protons']} & ${r['z']:+.2f}$ & ${r['t']:+.3f}$ \\\\" if r["z"] is not None else
                   f"{r['event']} & {r['kind']} & {r['nmpi']} & {r['stable']} & {r['charged']} & {r['central']} & {r['far']} & {r['lead_pt']:.2f} & {r['sum_pt']:.1f} & {fmt(r['jets10'])} & {r['protons']} & -- & -- \\\\")
    tex += ["\\bottomrule\\end{tabular}}", "\\newcommand{\\minbiasCatalogueSummary}{%", "\\begin{tabular}{lrrrr}\\toprule", "per collision & mean & median & min & max \\\\ \\midrule"]
    tex_lab = {"nmpi": "interactions", "stable": "stable particles", "charged": "charged stable particles", "central": "central ($|\\eta|<2.5$, $\\pt>0.5\\GeV$)",
               "far": "beyond $|\\eta|=5.5$", "lead_pt": "\\pt of the hardest particle [GeV]", "sum_pt": "scalar sum of \\pt [GeV]", "jets10": "\\antikt $R=0.4$ jets above $10\\GeV$",
               "protons": "forward protons ($\\xi<0.55$)"}
    for k, _ in keys:
        s = summ[k]
        d = 2 if k in ("jets10", "protons", "lead_pt") else (1 if k in ("nmpi",) else 0)
        tex.append(f"{tex_lab[k]} & {s['mean']:.{d}f} & {s['median']:.{1 if k == 'lead_pt' else 0}f} & {s['min']:.{1 if k == 'lead_pt' else 0}f} & {s['max']:.{1 if k == 'lead_pt' else 0}f} \\\\")
    if "z" in summ:
        tex.append(f"vertex $z$ [cm], the 960 drawn & ${summ['z']['mean']:.2f}$ & ${summ['z']['median']:.2f}$ & ${summ['z']['min']:.1f}$ & ${summ['z']['max']:.1f}$ \\\\")
        tex.append(f"vertex $t$ [ns], the 960 drawn & ${summ['t']['mean']:.3f}$ & ${summ['t']['median']:.3f}$ & ${summ['t']['min']:.2f}$ & ${summ['t']['max']:.2f}$ \\\\")
    tex += ["\\bottomrule\\end{tabular}}", "\\newcommand{\\minbiasCataloguePerProcess}{%", "\\begin{tabular}{lrrrrrrrr}\\toprule",
            "per process & events & $\\sigma$ [mb] & interactions & stable & charged & central & lead \\pt [GeV] & $\\Sigma\\pt$ [GeV] \\\\ \\midrule"]
    for kd in KINDS:
        p = per[kd]
        tex.append(f"{LONG[kd]} ({kd}) & {p['events']} & {p['sigma']:.2f} & {p['nmpi']:.1f} & {p['stable']:.0f} & {p['charged']:.0f} & {p['central']:.0f} & {p['lead_pt']:.1f} & {p['sum_pt']:.0f} \\\\")
    tex += ["\\bottomrule\\end{tabular}}", ""]
    args.output.with_suffix(".tex").write_text("\n".join(tex))
    print(f"wrote {args.output.with_suffix('.txt')}, .json, .tex ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
