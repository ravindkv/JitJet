#!/usr/bin/env python3
"""The ledger of the overlay: the twin of genLedger.py --pu of the CMSSW companion, on the standalone record.

Reads ../event_PU.lhe, the record standalone_pileup.py wrote (the signal of chapter 7 with its jets, the
vertex listing of every collision of the window, and the in-time crossing: every final particle of its
collisions and the mixed anti-kT jets), and ../pileup_library.lhe.gz, from which the out-of-time collisions
are rebuilt by their library events (the record does not list their particles), and books, as the JitJetSW
ledger_08_PU.txt does for the CMSSW overlay:

  * the final state of the signal, of the in-time pileup (bx = 0), of the out-of-time pileup (the other
    fifteen crossings, never overlaid: a list of particles has no time axis, and the out-of-time collisions
    reach the event only through the pulses of the detector) and of the signal with its in-time pileup, by
    kind (charged hadrons, neutral hadrons, photons, electrons, muons, neutrinos) and by region of |eta| (2.5,
    3.0, 5.2, 5.5: the tracker, the end of the endcap calorimeters, the end of HF, the CASTOR edge), with the
    visible part (|eta| < 5.2, no neutrinos) and, for the overlaid blocks, the cone Delta R < 0.4 around the
    last copy of each b quark;
  * the jets before (chapter 7's ak4GenJets, the photograph) and after the overlay (the mixed jets of the
    record): the counts above 3 / 20 / 30 / 50 GeV, the signal particles inside a jet, the b and bbar jets with
    their signal and pileup constituents, their response and their distance to the quark, the dijet mass and
    the imbalance of the pair, and the jets without a single signal particle.

The standalone mixing clusters only the particles within |eta| < 5 and without neutrinos (standalone_pileup.py,
ETA_MAX), where CMSSW clusters every stable particle of genParticlesForJets; the ledger counts every particle.

Writes overlay_ledger.txt (the ledger as text, with energies) and overlay_ledger.tex (the two tables as LaTeX
tabulars, n and scalar pT only, as the CMSSW table; \\input by the chapter).

  /usr/bin/python3 overlay_ledger.py            # a second
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import standalone_pileup as core  # noqa: E402

gj = core.gj
dec = core.dec
KINDS = ["charged hadron", "neutral hadron", "photon", "electron", "muon", "neutrino"]
REGIONS = [("|eta|<2.5", 0.0, 2.5), ("2.5-3.0", 2.5, 3.0), ("3.0-5.2", 3.0, 5.2), ("5.2-5.5", 5.2, 5.5), (">5.5", 5.5, math.inf)]
ETA_VISIBLE = 5.2
R = core.R_JET


def kind_of(pid: int) -> str:
    a = abs(pid)
    if a == 22:
        return "photon"
    if a == 11:
        return "electron"
    if a == 13:
        return "muon"
    if a in gj.NEUTRINOS:
        return "neutrino"
    return "charged hadron" if gj.is_charged(pid) else "neutral hadron"


class Block:
    """The final state of one part of the event: ids and four-momenta (E, px, py, pz)."""

    def __init__(self, ids, p4):
        self.ids = np.asarray(ids, dtype=int)
        self.p4 = np.asarray(p4, dtype=float).reshape(-1, 4)
        self.pt = np.hypot(self.p4[:, 1], self.p4[:, 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            self.eta = np.where(self.pt > 0, np.arcsinh(self.p4[:, 3] / np.where(self.pt > 0, self.pt, 1)), np.sign(self.p4[:, 3]) * 1e5)
        self.phi = np.arctan2(self.p4[:, 2], self.p4[:, 1])
        self.kind = np.array([kind_of(int(i)) for i in self.ids])
        self.nu = self.kind == "neutrino"

    def __add__(self, other):
        return Block(np.concatenate([self.ids, other.ids]), np.vstack([self.p4, other.p4]))

    def book(self, mask):
        return dict(n=int(mask.sum()), e=float(self.p4[mask, 0].sum()), pt=float(self.pt[mask].sum()), e_nu=float(self.p4[mask & self.nu, 0].sum()))

    def ledger(self) -> dict:
        abseta = np.abs(self.eta)
        out = dict(all=self.book(np.ones(len(self.ids), bool)),
                   kinds={k: self.book(self.kind == k) for k in KINDS},
                   regions={name: self.book((abseta >= lo) & (abseta < hi)) for name, lo, hi in REGIONS},
                   visible=self.book((abseta < ETA_VISIBLE) & ~self.nu),
                   neutrinos=self.book(self.nu))
        return out

    def cone(self, eta0, phi0, radius=R):
        d = np.hypot(self.eta - eta0, np.vectorize(gj.dphi)(self.phi, phi0)) if len(self.ids) else np.zeros(0)
        m = (d < radius) & ~self.nu
        v = self.p4[m].sum(axis=0) if m.any() else np.zeros(4)
        return dict(n=int(m.sum()), e=float(v[0]), vector_pt=float(math.hypot(v[1], v[2])), scalar_pt=float(self.pt[m].sum()))


def last_copy(ev, row: int) -> int:
    """The last same-flavour copy of a hard parton before the hadronisation (status below 70), as the journey pages take it."""
    while True:
        q = ev.p[row]
        if q.d1 <= 0:
            return row
        same = [d for d in range(q.d1, max(q.d1, q.d2) + 1) if d < len(ev.p) and ev.p[d].id == q.id and abs(ev.p[d].status) < 70]
        if not same:
            return row
        row = same[0]


def four(j) -> np.ndarray:
    px, py = j["pt"] * math.cos(j["phi"]), j["pt"] * math.sin(j["phi"])
    pz = j["pt"] * math.sinh(j["eta"])
    return np.array([math.sqrt(px * px + py * py + pz * pz + j["m"] ** 2), px, py, pz])


def inv_mass(a, b) -> float:
    v = four(a) + four(b)
    return math.sqrt(max(v[0] ** 2 - v[1] ** 2 - v[2] ** 2 - v[3] ** 2, 0.0))


def imbalance(a, b) -> float:
    v = four(a) + four(b)
    return math.hypot(v[1], v[2])


def out_of_time(collisions, library: Path):
    """The final state of the collisions of the other crossings, rebuilt from their library events (the record
    lists only the particles of the in-time crossing). Returns the block, the number of collisions and the
    in-time block rebuilt the same way, to check the rebuilding against the record's particle listing."""
    lib = core.Library(library)
    by_no = {e["no"]: e for e in lib.events}
    pick = lambda sel: [by_no[c["lib"]] for c in collisions if c["k"] > 0 and sel(c["bx"])]
    oot, intime = pick(lambda bx: bx != 0), pick(lambda bx: bx == 0)
    stack = lambda evs: Block(np.concatenate([e["ids"] for e in evs]), np.vstack([e["p4"] for e in evs]))
    return stack(oot), len(oot), stack(intime), sorted({c["bx"] for c in collisions if c["k"] > 0})


def analyse(record: Path, library: Path) -> dict:
    ev, _, _ = dec.read_record(record)
    genjets, genjets_of = gj.read_jet_tables(record)
    summary, collisions, particles, mixed, signal_of = core.read_pileup_tables(record)
    finals = ev.final_indices()
    sig = Block([ev.p[i].id for i in finals], [ev.p[i].p for i in finals])
    pu = Block([q["id"] for q in particles], [[q["e"], q["px"], q["py"], q["pz"]] for q in particles])
    both = sig + pu
    n_intime = len({q["k"] for q in particles})
    oot, n_oot, intime_rebuilt, bxs = out_of_time(collisions, library)
    # the rebuilding must give back the record's in-time crossing exactly (the mixing moves vertices, not momenta)
    # (the listing rounds px and py to 1e-4 GeV, so the scalar sums may differ by up to ~1e-4 GeV per particle)
    assert len(intime_rebuilt.ids) == len(pu.ids) and abs(intime_rebuilt.pt.sum() - pu.pt.sum()) < 1e-4 * len(pu.ids), \
        (len(intime_rebuilt.ids), len(pu.ids), intime_rebuilt.pt.sum(), pu.pt.sum())
    # the two quarks, their chapter-7 jet (nearest within R in (eta, phi)) and the mixed jet matched to it
    photo = genjets["ak4"]
    quarks = {}
    for i, q in enumerate(ev.p):
        if q.status == -23 and abs(q.id) == 5:
            lc = ev.p[last_copy(ev, i)]
            k = dict(row=last_copy(ev, i), pt=gj.pt(lc.p), eta=gj.eta(lc.p), phi=gj.phi(lc.p))
            near = [(math.hypot(g["eta"] - k["eta"], gj.dphi(g["phi"], k["phi"])), g) for g in photo]
            near = [x for x in near if x[0] < R]
            g = min(near, key=lambda x: x[0])[1] if near else None
            cands = [j for j in mixed if g is not None and j["genjet"] == g["rank"]]
            mj = min(cands, key=lambda j: j["dr"] if j["dr"] is not None else 9) if cands else None
            quarks["b" if q.id > 0 else "bbar"] = dict(quark=k, photo=g, mixed=mj)
    sig_rows_in_mixed = len(signal_of)
    jets = dict(
        photo_counts=[sum(1 for j in photo if j["pt"] > c) for c in (core.JET_PT_MIN, 20, 30, 50)],
        mixed_counts=[sum(1 for j in mixed if j["pt"] > c) for c in (core.JET_PT_MIN, 20, 30, 50)],
        photo_signal_in_jet=len(genjets_of["ak4"]), n_signal=len(finals),
        mixed_signal_in_jet=sum(1 for v in signal_of.values() if v >= 0), n_signal_entered=sig_rows_in_mixed,
        mixed_no_signal=sum(1 for j in mixed if j["n_signal"] == 0), n_mixed=len(mixed),
        above30_with_signal=sum(1 for j in mixed if j["pt"] > 30 and j["n_signal"] > 0), n_above30=sum(1 for j in mixed if j["pt"] > 30))
    b, bb = quarks.get("b"), quarks.get("bbar")
    if b and bb and b["photo"] and bb["photo"] and b["mixed"] and bb["mixed"]:
        jets["m_photo"], jets["m_mixed"] = inv_mass(b["photo"], bb["photo"]), inv_mass(b["mixed"], bb["mixed"])
        jets["imb_photo"], jets["imb_mixed"] = imbalance(b["photo"], bb["photo"]), imbalance(b["mixed"], bb["mixed"])
    for key, qd in quarks.items():
        for which in ("photo", "mixed"):
            j = qd[which]
            if j is not None:
                qd[f"dr_{which}"] = math.hypot(j["eta"] - qd["quark"]["eta"], gj.dphi(j["phi"], qd["quark"]["phi"]))
                qd[f"resp_{which}"] = j["pt"] / qd["quark"]["pt"]
        qd["cones"] = {name: blk.cone(qd["quark"]["eta"], qd["quark"]["phi"]) for name, blk in (("signal", sig), ("pileup", pu), ("all", both))}
    return dict(summary=summary, n_intime=n_intime, n_oot=n_oot, bx_range=(bxs[0], bxs[-1]), signal=sig.ledger(), pileup=pu.ledger(),
                oot=oot.ledger(), all=both.ledger(), quarks=quarks, jets=jets, record=record.name, library=library.name)


# ---------------------------------------------------------------------------
# the text ledger (the layout of the CMSSW companion's ledger_08_PU.txt)
# ---------------------------------------------------------------------------
def text(L: dict) -> str:
    lo, hi = L["bx_range"]
    out = [f"== overlay_ledger {L['record']}: signal {L['signal']['all']['n']} + in-time pileup {L['pileup']['all']['n']} final particles "
           f"from {L['n_intime']} collisions (bx 0); out-of-time pileup {L['oot']['all']['n']} final particles from {L['n_oot']} collisions "
           f"(bx {lo}..-1 and 1..{hi}, rebuilt from {L['library']}, not overlaid)"]
    for part, label in (("signal", "the signal"), ("pileup", f"the in-time pileup ({L['n_intime']} collisions, bx 0)"),
                        ("oot", f"the out-of-time pileup ({L['n_oot']} collisions, bx {lo}..-1 and 1..{hi})"),
                        ("all", "signal + in-time pileup")):
        B = L[part]
        out.append(f"final state of {label}: n={B['all']['n']} sumE={B['all']['e']:.1f} GeV sumPt={B['all']['pt']:.1f} GeV")
        out.append(f"  {'by kind':<20s}{'n':>6s}{'E[GeV]':>12s}{'pT[GeV]':>10s}")
        for k in KINDS:
            r = B["kinds"][k]
            if r["n"]:
                out.append(f"  {k:<20s}{r['n']:6d}{r['e']:12.1f}{r['pt']:10.1f}")
        out.append(f"  {'by region':<20s}{'n':>6s}{'E[GeV]':>12s}{'pT[GeV]':>10s}{'E_nu':>10s}")
        for name, _, _ in REGIONS:
            r = B["regions"][name]
            out.append(f"  {name:<20s}{r['n']:6d}{r['e']:12.1f}{r['pt']:10.1f}{r['e_nu']:10.1f}")
        v, nu = B["visible"], B["neutrinos"]
        out.append(f"  visible (|eta|<{ETA_VISIBLE}, no neutrinos): n={v['n']} E={v['e']:.1f} GeV pT={v['pt']:.1f} GeV; "
                   f"neutrinos: n={nu['n']} E={nu['e']:.1f} GeV pT={nu['pt']:.1f} GeV")
        if part == "oot":
            continue                                  # not overlaid: no cone around the quarks
        out.append(f"  cone Delta R < {R} around the last copy of each b quark (no neutrinos):")
        for key in ("bbar", "b"):
            qd = L["quarks"].get(key)
            if qd:
                q, c = qd["quark"], qd["cones"][part]
                out.append(f"    {key:<4s} axis [{q['row']}] pT={q['pt']:.1f} eta={q['eta']:.2f} phi={q['phi']:.2f}: n={c['n']} E={c['e']:.1f} GeV "
                           f"vector pT={c['vector_pt']:.1f} GeV scalar pT={c['scalar_pt']:.1f} GeV")
    J = L["jets"]
    out.append("jets: chapter 7's ak4GenJets (the photograph) -> the mixed anti-kT R = 0.4 jets (signal + in-time pileup, |eta| < 5, no neutrinos)")
    out.append(f"  jets above 3 / 20 / 30 / 50 GeV: {' / '.join(map(str, J['photo_counts']))} -> {' / '.join(map(str, J['mixed_counts']))}")
    out.append(f"  signal particles inside a jet: {J['photo_signal_in_jet']} of {J['n_signal']} -> {J['mixed_signal_in_jet']} of {J['n_signal']} "
               f"({J['n_signal_entered']} entered the mixed clustering)")
    for key in ("b", "bbar"):
        qd = L["quarks"].get(key)
        if not qd or qd["photo"] is None or qd["mixed"] is None:
            continue
        g, m, q = qd["photo"], qd["mixed"], qd["quark"]
        out.append(f"  {key:<4s} quark pT={q['pt']:.2f}: jet [{g['rank']}] pt={g['pt']:.2f} eta={g['eta']:.3f} phi={g['phi']:.3f} m={g['m']:.2f} n={g['n']} "
                   f"dR={qd['dr_photo']:.4f} pt(jet)/pt(quark)={qd['resp_photo']:.4f}")
        out.append(f"       -> mixed jet [{m['rank']}] pt={m['pt']:.2f} eta={m['eta']:.3f} phi={m['phi']:.3f} m={m['m']:.2f} n={m['n']} "
                   f"({m['n_signal']} signal + {m['n_pileup']} pileup from {m['n_collisions']} collisions) pileupPt={m['pt_pileup']:.2f} "
                   f"({100 * m['pt_pileup'] / m['pt']:.1f}%) dR={qd['dr_mixed']:.4f} pt(jet)/pt(quark)={qd['resp_mixed']:.4f} "
                   f"A={m['area']:.3f} rhoA={m['rho_area']:.2f} pTsub={m['pt_sub']:.2f}")
    if "m_photo" in J:
        out.append(f"  m(b jet + bbar jet): {J['m_photo']:.1f} -> {J['m_mixed']:.1f} GeV; |pT(b jet) + pT(bbar jet)|: {J['imb_photo']:.1f} -> {J['imb_mixed']:.1f} GeV")
    out.append(f"  mixed jets without a single signal particle: {J['mixed_no_signal']} of {J['n_mixed']}; "
               f"jets above 30 GeV with at least one: {J['above30_with_signal']} of {J['n_above30']}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# the LaTeX tables (the layout of the ledger table of the CMSSW companion's chapter 08)
# ---------------------------------------------------------------------------
def thou(x: float, nd: int = 0) -> str:
    """A thin space between the thousands from five digits on (7168, 13\\,600), as the CMSSW companion's table."""
    s = f"{x:,.{nd}f}"
    return s.replace(",", "\\,") if abs(x) >= 9999.5 else s.replace(",", "")


def cells(r: dict) -> str:
    """n and the scalar sum of pT (the CMSSW table dropped the energy column on 16 November 2026)."""
    return f"{thou(r['n'])} & {thou(r['pt'], 1)}"


def latex(L: dict) -> str:
    blocks = [L["signal"], L["pileup"], L["oot"], L["all"]]
    n, n_oot = L["n_intime"], L["n_oot"]
    rows = []
    kinds = [("charged hadrons", ["charged hadron"]), ("neutral hadrons", ["neutral hadron"]), ("photons", ["photon"]),
             ("electrons, muons", ["electron", "muon"]), ("neutrinos", ["neutrino"])]

    def merge(B, ks):
        return dict(n=sum(B["kinds"][k]["n"] for k in ks), e=sum(B["kinds"][k]["e"] for k in ks), pt=sum(B["kinds"][k]["pt"] for k in ks))

    def row(label, get):
        return f"{label} & " + " & ".join(cells(get(B)) for B in blocks) + " \\\\"

    for label, ks in kinds:
        rows.append(row(label, lambda B: merge(B, ks)))
    rows.append("\\midrule")
    names = {"|eta|<2.5": "$|\\eta|<2.5$", "2.5-3.0": "$2.5<|\\eta|<3.0$", "3.0-5.2": "$3.0<|\\eta|<5.2$",
             "5.2-5.5": "$5.2<|\\eta|<5.5$", ">5.5": "$|\\eta|>5.5$"}
    for name, _, _ in REGIONS:
        rows.append(row(names[name], lambda B: B["regions"][name]))
    rows.append("\\midrule")
    rows.append(row("all", lambda B: B["all"]))
    rows.append(row("visible ($|\\eta|<5.2$, no $\\nu$)", lambda B: B["visible"]))
    top = ("\\begin{tabular}{lrrrrrrrr}\\toprule\n"
           " & \\multicolumn{2}{c}{signal} & \\multicolumn{2}{c}{in-time pileup} & \\multicolumn{2}{c}{out-of-time pileup} & \\multicolumn{2}{c}{signal +} \\\\\n"
           f" & \\multicolumn{{2}}{{c}}{{}} & \\multicolumn{{2}}{{c}}{{({n} collisions)}} & \\multicolumn{{2}}{{c}}{{({n_oot} collisions)}} & \\multicolumn{{2}}{{c}}{{in-time pileup}} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
           "final state & $n$ & \\pt{} [GeV] & $n$ & \\pt{} [GeV] & $n$ & \\pt{} [GeV] & $n$ & \\pt{} [GeV] \\\\ \\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
    J, Q = L["jets"], L["quarks"]
    b, bb = Q["b"], Q["bbar"]
    pc, mc = J["photo_counts"], J["mixed_counts"]

    def jetrow(label, qd):
        g, m = qd["photo"], qd["mixed"]
        share = 100 * m["pt_pileup"] / m["pt"]
        return (f"{label}: \\pt{{}} [GeV], constituents & {g['pt']:.2f}, {g['n']} & {m['pt']:.2f}, {m['n']} ({m['n_signal']} + {m['n_pileup']}) & "
                f"${m['pt'] - g['pt']:+.2f}$ (pileup \\pt{{}} {m['pt_pileup']:.2f}, {share:.0f}\\,\\%) \\\\")

    jrows = [
        f"jets with $\\pt>3\\GeV$ & {pc[0]} & {mc[0]} & ${mc[0] - pc[0]:+d}$ \\\\",
        f"jets with $\\pt>20$ / $>30$ / $>50\\GeV$ & {pc[1]} / {pc[2]} / {pc[3]} & {mc[1]} / {mc[2]} / {mc[3]} & \\\\",
        f"signal particles inside a jet & {J['photo_signal_in_jet']} of {J['n_signal']} & {J['mixed_signal_in_jet']} of {J['n_signal']} & "
        f"${J['mixed_signal_in_jet'] - J['photo_signal_in_jet']:+d}$ \\\\",
        jetrow("$b$ jet", b),
        jetrow("$\\bar b$ jet", bb),
        f"$\\pt(\\mathrm{{jet}})/\\pt(\\mathrm{{quark}})$, $b$ / $\\bar b$ & {b['resp_photo']:.3f} / {bb['resp_photo']:.3f} & "
        f"{b['resp_mixed']:.3f} / {bb['resp_mixed']:.3f} & \\\\",
        f"$\\Delta R$(jet axis, quark), $b$ / $\\bar b$ & {b['dr_photo']:.3f} / {bb['dr_photo']:.3f} & {b['dr_mixed']:.3f} / {bb['dr_mixed']:.3f} & \\\\",
        f"$m(b\\,\\mathrm{{jet}} + \\bar b\\,\\mathrm{{jet}})$ [GeV] & {J['m_photo']:.1f} & {J['m_mixed']:.1f} & ${J['m_mixed'] - J['m_photo']:+.1f}$ \\\\",
        f"$|\\vec\\pt(b) + \\vec\\pt(\\bar b)|$ [GeV] & {J['imb_photo']:.1f} & {J['imb_mixed']:.1f} & \\\\",
        f"jets without a single signal particle & -- & {J['mixed_no_signal']} of {J['n_mixed']} & \\\\",
    ]
    bottom = ("\\begin{tabular}{lrrr}\\toprule\n"
              "jets \\code{ak4GenJets} $\\to$ mixed jets & signal (ch.~7) & signal + in-time pileup & difference \\\\ \\midrule\n"
              + "\n".join(jrows) + "\n\\bottomrule\\end{tabular}")
    return ("% written by overlay_ledger.py from event_PU.lhe: the ledger of the overlay (do not edit)\n"
            f"\\newcommand{{\\puOverlayLedgerFinal}}{{%\n{top}}}\n"
            f"\\newcommand{{\\puOverlayLedgerJets}}{{%\n{bottom}}}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=HERE.parent / "event_PU.lhe")
    ap.add_argument("-l", "--library", type=Path, default=HERE.parent / "pileup_library.lhe.gz", help="the library the record was mixed from")
    ap.add_argument("-o", "--output", type=Path, default=HERE / "overlay_ledger", help="writes OUTPUT.txt and OUTPUT.tex")
    args = ap.parse_args()
    L = analyse(args.input, args.library)
    t = text(L)
    sys.stdout.write(t)
    args.output.with_suffix(".txt").write_text(t)
    args.output.with_suffix(".tex").write_text(latex(L))


if __name__ == "__main__":
    main()
