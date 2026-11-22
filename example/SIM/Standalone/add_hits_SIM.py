#!/usr/bin/env python3
"""Add the hits of the signal, the in-time crowd and the out-of-time crowd by hand.

standalone_SIM.py simulated the three samples separately (../simhits_signal.npz, ../simhits_intime.npz,
../simhits_oot.npz). Here they are laid on top of each other, which is all that CMSSW's MixingModule does at
this level: every hit of a collision of bunch crossing bx is shifted in time by 25 ns x bx, the tracker and
muon hits are put in one list, and the energies of the calorimeter cells are added cell by cell. No physics
happens in this script; a hit never learns that the others exist. What the electronics make of the sum, and
which bunch crossings they still see, is the next stop (chapter 11).

Dependencies: numpy and standalone_SIM.py (for the cell codes). Output:
  ../event_SIM.lhe            the record of chapter 9 followed by the tables of this stop
  sim_ledger.tex / .txt       the tables of the chapter as LaTeX macros and as text
  add_hits_SIM.json           every number of the tables

  python3 add_hits_SIM.py          # a few seconds
  python3 add_hits_SIM.py -v       # narrates the addition
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import standalone_SIM as sim

HERE = Path(__file__).resolve().parent
BX_NS = 25.0
ORIGINS = ["signal", "early", "intime", "late"]          # early: bx < 0, late: bx > 0
SF = {sim.HB: 125.0, sim.HE: 190.0}                       # the sampling factors the digitisation applies (chapter 11)
R_JET = 0.4


def load(folder: Path) -> dict:
    return {k: dict(np.load(folder / f"simhits_{k}.npz", allow_pickle=False)) for k in ("signal", "intime", "oot")}


def origin_of_bx(bx):
    return np.where(bx < 0, 1, np.where(bx > 0, 3, 2)).astype(np.int8)


def mix(S: dict) -> dict:
    """The mixed event: all tracker and muon hits with absolute times and their origin (0 signal, 1 early, 2 in time,
    3 late), and one row per calorimeter cell with the energy of each origin."""
    sg, it, oo = S["signal"], S["intime"], S["oot"]
    M = {}
    for part in ("trk", "mu"):
        d = {}
        for key in ("det", "lay", "x", "y", "z", "de"):
            d[key] = np.concatenate([sg[f"{part}_{key}"], it[f"{part}_{key}"], oo[f"{part}_{key}"]])
        d["t"] = np.concatenate([sg[f"{part}_t"], it[f"{part}_t"], oo[f"{part}_t"] + BX_NS * oo[f"{part}_bx"]])      # the shift
        d["bx"] = np.concatenate([np.zeros(len(sg[f"{part}_det"]), np.int8), it[f"{part}_bx"], oo[f"{part}_bx"]])
        d["origin"] = np.concatenate([np.zeros(len(sg[f"{part}_det"]), np.int8), origin_of_bx(it[f"{part}_bx"]), origin_of_bx(oo[f"{part}_bx"])])
        M[part] = d
    code = np.concatenate([sg["cal_code"], it["cal_code"], oo["cal_code"]])
    e = np.concatenate([sg["cal_e"], it["cal_e"], oo["cal_e"]]).astype(float)
    bx = np.concatenate([np.zeros(len(sg["cal_code"]), np.int8), it["cal_bx"], oo["cal_bx"]])
    org = np.concatenate([np.zeros(len(sg["cal_code"]), np.int8), origin_of_bx(it["cal_bx"]), origin_of_bx(oo["cal_bx"])])
    cells, inv = np.unique(code, return_inverse=True)
    E = np.zeros((len(cells), 4))
    np.add.at(E, (inv, org), e)
    Ebx = np.zeros((len(cells), 16))
    np.add.at(Ebx, (inv, bx.astype(int) + 12), e)
    M["cells"], M["E"], M["Ebx"] = cells, E, Ebx
    M["det"] = cells // sim.CODE
    M["dom"] = np.argmax(E, axis=1)
    return M


def jet_axis(st):
    j = st.genjets["ak4"][0]
    return j["eta"], j["phi"], j


def in_circle(codes, eta0, phi0, r=R_JET):
    det, x, y, z = sim.cell_position(codes)
    eta = np.arcsinh(z / np.maximum(np.hypot(x, y), 1e-9))
    phi = np.arctan2(y, x)
    dphi = (phi - phi0 + math.pi) % (2 * math.pi) - math.pi
    return (eta - eta0) ** 2 + dphi ** 2 < r * r, eta, phi


def numbers(S, M, st) -> dict:
    sg, it, oo = S["signal"], S["intime"], S["oot"]
    J = {s: json.loads((HERE / f"standalone_SIM_{s}.json").read_text()) for s in ("signal", "intime", "oot")}
    early_k = oo["col_bx"] < 0
    late_k = oo["col_bx"] > 0
    N = {"collisions": dict(signal=1, intime=int(len(it["col_k"])), early=int(early_k.sum()), late=int(late_k.sum())),
         "primaries": dict(signal=int(len(sg["prim_id"])), intime=int(it["col_nprim"].sum()), early=int(oo["col_nprim"][early_k].sum()), late=int(oo["col_nprim"][late_k].sum())),
         "tracks": dict(signal=J["signal"]["simtracks"], intime=J["intime"]["simtracks"],
                        early=int((oo["col_nprim"] + oo["col_nsec"])[early_k].sum()), late=int((oo["col_nprim"] + oo["col_nsec"])[late_k].sum()))}
    # per bunch crossing
    bxs = list(range(-12, 4))
    ncol = {b: int((oo["col_bx"] == b).sum()) for b in bxs}
    ncol[0] = N["collisions"]["intime"]
    N["per_bx"] = dict(bx=bxs, collisions=[ncol[b] for b in bxs],
                       trk=[int(((M["trk"]["bx"] == b) & (M["trk"]["origin"] > 0)).sum()) for b in bxs])
    # tracker and muon hits by subdetector and origin
    for part, dets, name in (("trk", sim.TRACKER, "tracker"), ("mu", sim.MUON, "muon")):
        d = M[part]
        T = {}
        for dd in dets:
            m = d["det"] == dd
            T[sim.DET_NAME[dd]] = {o: int((m & (d["origin"] == i)).sum()) for i, o in enumerate(ORIGINS)}
            T[sim.DET_NAME[dd]]["de"] = {o: float(d["de"][m & (d["origin"] == i)].sum()) for i, o in enumerate(ORIGINS)}
            if part == "trk":
                T[sim.DET_NAME[dd]]["layers"] = {int(l): [int((m & (d["lay"] == l) & (d["origin"] == i)).sum()) for i in range(4)] for l in np.unique(d["lay"][m])}
        T["total"] = {o: int((d["origin"] == i).sum()) for i, o in enumerate(ORIGINS)}
        N[name] = T
    # calorimeter cells
    ncell = sim.n_cells()
    C = {}
    for dd in sim.ECAL + sim.HCAL:
        m = M["det"] == dd
        nm = sim.DET_NAME[dd]
        C[nm] = dict(total=ncell.get(nm), hit={o: int((M["E"][m, i] > 0).sum()) for i, o in enumerate(ORIGINS)}, hit_all=int(m.sum()),
                     e={o: float(M["E"][m, i].sum()) for i, o in enumerate(ORIGINS)},
                     dom={o: int((M["dom"][m] == i).sum()) for i, o in enumerate(ORIGINS)},
                     e_dom={o: float(M["E"][m][M["dom"][m] == i].sum()) for i, o in enumerate(ORIGINS)},
                     e_bx=[float(x) for x in M["Ebx"][m].sum(axis=0)])
        own = M["E"][m, 0]
        C[nm]["signal_cells_kept"] = int(((own > 0) & (M["dom"][m] == 0)).sum())
        C[nm]["signal_e_kept"] = float(own[(own > 0) & (M["dom"][m] == 0)].sum())
    N["calo"] = C
    for grp, dets in (("ECAL", ("EB", "EE", "ES")), ("HBHEHO", ("HB", "HE", "HO")), ("HF", ("HF",))):
        N[grp] = dict(hit={o: sum(C[d]["hit"][o] for d in dets) for o in ORIGINS}, hit_all=sum(C[d]["hit_all"] for d in dets),
                      e={o: sum(C[d]["e"][o] for d in dets) for o in ORIGINS}, dom={o: sum(C[d]["dom"][o] for d in dets) for o in ORIGINS},
                      e_dom={o: sum(C[d]["e_dom"][o] for d in dets) for o in ORIGINS})
    # primaries of the signal that left a hit, alone and still visible in the mixture
    prim_hit = set(np.unique(sg["trk_prim"])) | set(np.unique(sg["cal_prim"])) | set(np.unique(sg["mu_prim"]))
    N["signal_primaries_with_hit"] = len(prim_hit)
    # the circle of our jet
    eta0, phi0, jet = jet_axis(st)
    inc, eta, phi = in_circle(M["cells"], eta0, phi0)
    w = np.where(M["det"] == sim.HB, SF[sim.HB], np.where(M["det"] == sim.HE, SF[sim.HE], 1.0))
    ecal = np.isin(M["det"], (sim.EB, sim.EE))
    hcal = np.isin(M["det"], (sim.HB, sim.HE))
    circ = {}
    for i, o in enumerate(ORIGINS):
        et = M["E"][:, i] / np.cosh(eta)
        circ[o] = dict(ecal_e=float(M["E"][inc & ecal, i].sum()), ecal_et=float(et[inc & ecal].sum()), ecal_cells=int((M["E"][inc & ecal, i] > 0).sum()),
                       hcal_e=float(M["E"][inc & hcal, i].sum()), hcal_et=float((et * w)[inc & hcal].sum()), hcal_cells=int((M["E"][inc & hcal, i] > 0).sum()),
                       trk=int(0))
        px = (et * np.where(ecal | hcal, w, 0.0) * np.cos(phi))[inc & (ecal | hcal)].sum()
        py = (et * np.where(ecal | hcal, w, 0.0) * np.sin(phi))[inc & (ecal | hcal)].sum()
        circ[o]["px"], circ[o]["py"] = float(px), float(py)
    ebx = (M["Ebx"] / np.cosh(eta)[:, None] * w[:, None])[inc & (ecal | hcal)].sum(axis=0)
    N["circle"] = dict(eta=eta0, phi=phi0, origins=circ, et_bx=[float(x) for x in ebx], cells_ecal=int((inc & ecal).sum()), cells_hcal=int((inc & hcal).sum()))
    # tracker hits in the circle (direction of the hit seen from the origin)
    d = M["trk"]
    r = np.hypot(d["x"], d["y"])
    he = np.arcsinh(d["z"] / np.maximum(r, 1e-9))
    hp = np.arctan2(d["y"], d["x"])
    hin = (he - eta0) ** 2 + ((hp - phi0 + math.pi) % (2 * math.pi) - math.pi) ** 2 < R_JET ** 2
    for i, o in enumerate(ORIGINS):
        circ[o]["trk"] = int((hin & (d["origin"] == i)).sum())
    # the members of the jet
    jm = np.flatnonzero(sg["prim_jet0"])
    cdet = sg["cal_code"] // sim.CODE
    mem = np.isin(sg["cal_prim"], jm)
    me_, mh_ = mem & np.isin(cdet, (sim.EB, sim.EE)), mem & np.isin(cdet, (sim.HB, sim.HE))
    px1, py1, _, _ = sim.cell_pt(sg["cal_code"][me_], sg["cal_e"][me_])
    wj = np.where(cdet[mh_] == sim.HB, SF[sim.HB], SF[sim.HE])
    px2, py2, _, _ = sim.cell_pt(sg["cal_code"][mh_], sg["cal_e"][mh_] * wj)
    b = sg["prim_bnd"][jm]
    ok = np.isfinite(b[:, 0])
    p4 = sg["prim_p4"][jm]
    N["jet"] = dict(pt=jet["pt"], m=jet["m"], n=int(len(jm)), e=float(p4[:, 0].sum()), eta=eta0, phi=phi0,
                    bnd_n=int(ok.sum()), bnd_pt=float(math.hypot(b[ok, 3].sum(), b[ok, 4].sum())),
                    ecal_e=float(sg["cal_e"][me_].sum()), ecal_cells=int(len(np.unique(sg["cal_code"][me_]))), ecal_pt=sim.vec_pt(px1, py1),
                    hcal_e=float(sg["cal_e"][mh_].sum()), hcal_cells=int(len(np.unique(sg["cal_code"][mh_]))), hcal_pt=sim.vec_pt(px2, py2),
                    hcal_e_sf=float((sg["cal_e"][mh_] * wj).sum()),
                    calo_pt=sim.vec_pt(np.concatenate([px1, px2]), np.concatenate([py1, py2])),
                    fates={str(n): int((sg["prim_fate"][jm] == i).sum()) for i, n in enumerate(sg["fate_names"])})
    acc = np.zeros(2)
    N["circle_ledger"] = []
    for o in ("signal", "intime", "early", "late"):
        acc += np.array([circ[o]["px"], circ[o]["py"]])
        N["circle_ledger"].append(dict(origin=o, pt=float(np.hypot(*acc)), scalar=circ[o]["ecal_et"] + circ[o]["hcal_et"], cells=circ[o]["ecal_cells"] + circ[o]["hcal_cells"]))
    N["stages"] = J["signal"]["stages"]
    N["counters"] = {s: J[s]["n"] for s in J}
    N["energies"] = {s: J[s]["e"] for s in J}
    N["seconds"] = {s: J[s]["seconds"] for s in J}
    N["jobs"] = J["oot"].get("jobs")
    # timing of the tracker hits (per origin: fraction later than 25 ns after their own crossing)
    t_loc = d["t"] - BX_NS * d["bx"]
    N["late_hits"] = {o: int(((t_loc > 25.0) & (d["origin"] == i)).sum()) for i, o in enumerate(ORIGINS)}
    return N


# ---------------------------------------------------------------------------
# LaTeX and text
# ---------------------------------------------------------------------------
def th(x, nd=0):
    s = f"{x:,.{nd}f}" if nd else f"{int(round(x)):,}"
    return s.replace(",", "\\,")


def tot(d):
    return sum(d[o] for o in ORIGINS)


def latex(N) -> str:
    O = ["signal", "intime", "early", "late"]
    nc = N["collisions"]
    L = []
    L.append("% written by example/SIM/Standalone/add_hits_SIM.py: do not edit")
    def row(label, d, nd=0, total=True, extra=None):
        cells = [th(d[o], nd) for o in O] + ([th(extra if extra is not None else tot(d), nd)] if total else ["--"])
        return f"{label} & " + " & ".join(cells) + " \\\\"
    b = []
    b.append(row("collisions", nc))
    b.append(row("primaries handed to the simulation", N["primaries"]))
    b.append(row("tracks: primaries and stored secondaries", N["tracks"]))
    b.append(row("tracker hits", N["tracker"]["total"]))
    b.append(row("muon-chamber hits", N["muon"]["total"]))
    b.append(row("ECAL cells with energy", N["ECAL"]["hit"], extra=N["ECAL"]["hit_all"]))
    b.append(row("ECAL energy [GeV]", N["ECAL"]["e"], nd=1))
    b.append(row("HB, HE, HO cells with energy", N["HBHEHO"]["hit"], extra=N["HBHEHO"]["hit_all"]))
    b.append(row("HB, HE, HO scintillator energy [GeV]", N["HBHEHO"]["e"], nd=2))
    b.append(row("HF cells with light", N["HF"]["hit"], extra=N["HF"]["hit_all"]))
    b.append(row("HF photoelectrons", N["HF"]["e"]))
    b.append("\\midrule\\multicolumn{6}{@{}l}{\\emph{after the addition, each cell given to the origin that deposits most in it}}\\\\")
    b.append(row("ECAL cells", N["ECAL"]["dom"]))
    b.append(row("\\quad their energy [GeV]", N["ECAL"]["e_dom"], nd=1))
    b.append(row("HB, HE, HO cells", N["HBHEHO"]["dom"]))
    b.append(row("HF cells", N["HF"]["dom"]))
    L.append("\\newcommand{\\simLedgerRows}{%\n" + "\n".join(b) + "}")
    # tracker
    b = []
    nlay = {"PXB": "4 layers", "PXF": "$2\\times3$ disks", "TIB": "4 layers", "TID": "$2\\times3$ disks", "TOB": "6 layers", "TEC": "$2\\times9$ disks"}
    T = N["tracker"]
    for k in ("PXB", "PXF", "TIB", "TID", "TOB", "TEC", "total"):
        d = T[k]
        a = tot(d)
        pre = "\\midrule " if k == "total" else ""
        de = f"{1e3 * d['de']['signal'] / max(d['signal'], 1):.2f}" if k != "total" else f"{1e3 * sum(T[x]['de']['signal'] for x in nlay) / max(d['signal'], 1):.2f}"
        b.append(f"{pre}{k if k != 'total' else 'tracker'} & {nlay.get(k, '')} & {th(d['signal'])} & {de} & {th(d['intime'])} & {th(d['early'])} & {th(d['late'])} & {th(a)} & {100 * d['signal'] / a:.2f} & {100 * d['signal'] / (d['signal'] + d['intime']):.1f} \\\\")
    L.append("\\newcommand{\\simTrackerRows}{%\n" + "\n".join(b) + "}")
    # calorimeters
    C = N["calo"]
    def calo_rows(names, nd_e, unit=1.0):
        out = []
        for k in names:
            c = C[k]
            ntot = th(c["total"]) if c["total"] else "--"
            occ = f"{100 * c['hit_all'] / c['total']:.0f}" if c["total"] else "--"
            out.append(f"{k} & {ntot} & {th(c['hit']['signal'])} & {th(c['e']['signal'], nd_e)} & {th(c['hit']['intime'])} & {th(c['e']['intime'], nd_e)} & "
                       f"{th(c['e']['early'], nd_e)} & {th(c['e']['late'], nd_e)} & {th(c['hit_all'])} & {occ} & {th(c['signal_cells_kept'])} \\\\")
        return "\n".join(out)
    L.append("\\newcommand{\\simEcalRows}{%\n" + calo_rows(("EB", "EE", "ES"), 1) + "}")
    L.append("\\newcommand{\\simHcalRows}{%\n" + calo_rows(("HB", "HE", "HO"), 2) + "\n" + calo_rows(("HF",), 0) + "}")
    b = []
    for k in ("DT", "CSC", "RPC", "GEM", "total"):
        d = N["muon"][k]
        b.append(("\\midrule " if k == "total" else "") + f"{k if k != 'total' else 'muon system'} & " + " & ".join(th(d[o]) for o in O) + f" & {th(tot(d))} \\\\")
    L.append("\\newcommand{\\simMuonRows}{%\n" + "\n".join(b) + "}")
    # stages
    names = {"transport": "transport only (chapter 9)", "B": "+ magnetic field", "B_Decay": "+ decays in flight", "B_Decay_EM": "+ electromagnetic physics", "full": "+ hadronic physics = full"}
    b = []
    for w in N["stages"]:
        b.append(f"{names[w['stage']]} & {w['boundary']} & {th(w['secondaries'])} & {w['sec_decay']}/{w['sec_em']}/{w['sec_had']} & {th(w['trk'])} & {w['mu']} & {w['de']:.2f} & {th(w['cells'])} & "
                 f"{w['EB']:.1f} & {w['EE']:.1f} & {w['HB']:.2f} & {w['HE']:.2f} & {th(w['HF'])} & {w['jet_bnd_pt']:.1f} ({w['jet_bnd_n']}) & {w['jet_ecal']:.1f} \\\\")
    L.append("\\newcommand{\\simStageRows}{%\n" + "\n".join(b) + "}")
    # the circle of our jet
    c = N["circle"]["origins"]
    b = []
    lab = {"signal": "the family", "intime": "in-time crowd (bx $=0$)", "early": "early crowd (bx $-12\\ldots-1$)", "late": "late crowd (bx $+1\\ldots+3$)"}
    for o in O:
        x = c[o]
        b.append(f"{lab[o]} & {th(x['trk'])} & {th(x['ecal_cells'])} & {x['ecal_e']:.1f} & {x['ecal_et']:.1f} & {th(x['hcal_cells'])} & {x['hcal_e']:.3f} & {x['hcal_et']:.1f} & {x['ecal_et'] + x['hcal_et']:.1f} \\\\")
    L.append("\\newcommand{\\simCircleRows}{%\n" + "\n".join(b) + "}")
    return "\n".join(L) + "\n"


def text(N) -> str:
    O = ["signal", "intime", "early", "late"]
    out = []
    out.append("The ledger of the hits (standalone): three samples simulated separately, then added")
    out.append(f"{'':44s}{'signal':>12s}{'in time':>12s}{'early':>12s}{'late':>12s}{'added':>12s}")
    def row(label, d, nd=0, extra=None):
        out.append(f"{label:44s}" + "".join(f"{d[o]:12,.{nd}f}" for o in O) + f"{(extra if extra is not None else tot(d)):12,.{nd}f}")
    row("collisions", N["collisions"]); row("primaries", N["primaries"]); row("tracks", N["tracks"])
    row("tracker hits", N["tracker"]["total"]); row("muon hits", N["muon"]["total"])
    row("ECAL cells with energy", N["ECAL"]["hit"], extra=N["ECAL"]["hit_all"]); row("ECAL energy [GeV]", N["ECAL"]["e"], 1)
    row("HB+HE+HO cells", N["HBHEHO"]["hit"], extra=N["HBHEHO"]["hit_all"]); row("HB+HE+HO scintillator energy [GeV]", N["HBHEHO"]["e"], 2)
    row("HF cells", N["HF"]["hit"], extra=N["HF"]["hit_all"]); row("HF photoelectrons", N["HF"]["e"])
    out.append("after the addition, cells by the origin that deposits most:")
    row("ECAL cells", N["ECAL"]["dom"]); row("  their energy [GeV]", N["ECAL"]["e_dom"], 1); row("HB+HE+HO cells", N["HBHEHO"]["dom"]); row("HF cells", N["HF"]["dom"])
    out.append("")
    out.append("tracker, hits by subdetector")
    for k in ("PXB", "PXF", "TIB", "TID", "TOB", "TEC", "total"):
        d = N["tracker"][k]
        out.append(f"  {k:6s}" + "".join(f"{d[o]:12,d}" for o in O) + f"{tot(d):12,d}   signal {100 * d['signal'] / tot(d):.2f} % of all, {100 * d['signal'] / (d['signal'] + d['intime']):.1f} % of its own crossing")
    for k in ("PXB", "TIB", "TOB"):
        out.append(f"  {k} layers (signal, early, in time, late): " + "; ".join(f"L{l}: {v[0]}, {v[1]}, {v[2]}, {v[3]}" for l, v in sorted(N["tracker"][k]["layers"].items())))
    out.append("muon system")
    for k in ("DT", "CSC", "RPC", "GEM", "total"):
        d = N["muon"][k]
        out.append(f"  {k:6s}" + "".join(f"{d[o]:12,d}" for o in O) + f"{tot(d):12,d}")
    out.append("calorimeters: cells in the detector; cells hit and energy by origin; cells hit after the addition; signal cells still dominated by the signal")
    for k in ("EB", "EE", "ES", "HB", "HE", "HO", "HF"):
        c = N["calo"][k]
        out.append(f"  {k:3s} {str(c['total']):>7s} | " + " ".join(f"{o} {c['hit'][o]:,d} cells {c['e'][o]:,.2f}" for o in O) + f" | hit {c['hit_all']:,d} | signal keeps {c['signal_cells_kept']:,d} cells, {c['signal_e_kept']:.1f}")
    out.append(f"per bunch crossing (bx: collisions, tracker hits): " + ", ".join(f"{b}: {n}, {t}" for b, n, t in zip(N['per_bx']['bx'], N['per_bx']['collisions'], N['per_bx']['trk'])))
    j = N["jet"]
    out.append("")
    out.append(f"our jet: ak4 jet [0], pT {j['pt']:.2f}, {j['n']} members, E {j['e']:.1f} GeV at eta {j['eta']:.3f}, phi {j['phi']:.3f}")
    out.append(f"  members reaching the calorimeters as primaries: {j['bnd_n']}, vector pT {j['bnd_pt']:.2f}; fates {j['fates']}")
    out.append(f"  their deposits: crystals {j['ecal_e']:.2f} GeV in {j['ecal_cells']} cells (pT {j['ecal_pt']:.2f}); HCAL {j['hcal_e']:.3f} GeV scintillator in {j['hcal_cells']} cells, x factor {j['hcal_e_sf']:.1f} GeV (pT {j['hcal_pt']:.2f}); together pT {j['calo_pt']:.2f}")
    out.append(f"  the circle dR < {R_JET} around the jet: {N['circle']['cells_ecal']} crystals and {N['circle']['cells_hcal']} HCAL cells hit after the addition")
    for o in O:
        x = N["circle"]["origins"][o]
        out.append(f"    {o:7s}: tracker hits {x['trk']:7,d}; crystals {x['ecal_cells']:5d} cells {x['ecal_e']:8.2f} GeV, ET {x['ecal_et']:7.2f}; HCAL {x['hcal_cells']:4d} cells {x['hcal_e']:7.3f} GeV, x factor ET {x['hcal_et']:7.2f}; sum ET {x['ecal_et'] + x['hcal_et']:7.2f}")
    out.append("  vector pT of the cells of the circle, adding one origin after the other: " + ", ".join(f"+{x['origin']} {x['pt']:.2f}" for x in N["circle_ledger"]))
    out.append("  ET in the circle per bunch crossing (every deposit, the signal's in bx 0): " + ", ".join(f"{b}: {e:.1f}" for b, e in zip(range(-12, 4), N["circle"]["et_bx"])))
    return "\n".join(out) + "\n"


def record_tables(N, S, st) -> str:
    """The tables this stop appends to the record of chapter 9."""
    sg = S["signal"]
    out = []
    hdr = " --------  Standalone SimHit Summary  (three samples simulated separately by standalone_SIM.py, seed 1, added by add_hits_SIM.py)  ----------"
    out.append(hdr)
    out.append(text(N))
    out.append(hdr.replace("Standalone SimHit Summary", "End Standalone SimHit Summary"))
    out.append("")
    hdr = (" --------  Standalone SimHit Primary Listing  (the signal: row of the chapter-7 record, id, E, pT, eta; fate as a primary; position [cm] and"
           " pT at the tracker boundary; tracker hits; ECAL [GeV]; HB+HE scintillator [GeV]; HF [p.e.]; jet0 = member of ak4 jet [0])  ----------")
    out.append(hdr)
    out.append(f"{'row':>6s} {'id':>6s} {'name':>10s} {'E':>9s} {'pT':>8s} {'eta':>7s} {'fate':>12s} {'r_b':>7s} {'z_b':>8s} {'pT_b':>8s} {'trk':>5s} {'ECAL':>9s} {'HCAL':>8s} {'HF':>6s} jet0")
    cdet = sg["cal_code"] // sim.CODE
    n = len(sg["prim_id"])
    trk_n = np.bincount(sg["trk_prim"], minlength=n)
    def per(mask):
        return np.bincount(sg["cal_prim"][mask], weights=sg["cal_e"][mask], minlength=n)
    ecal, hcal, hf = per(np.isin(cdet, sim.ECAL)), per(np.isin(cdet, (sim.HB, sim.HE, sim.HO))), per(cdet == sim.HF)
    for j in range(n):
        E, px, py, pz = sg["prim_p4"][j]
        pt = math.hypot(px, py)
        b = sg["prim_bnd"][j]
        bb = f"{math.hypot(b[0], b[1]):7.1f} {b[2]:8.1f} {math.hypot(b[3], b[4]):8.3f}" if np.isfinite(b[0]) else f"{'-':>7s} {'-':>8s} {'-':>8s}"
        out.append(f"{int(sg['prim_row'][j]):6d} {int(sg['prim_id'][j]):6d} {sim.dec.name(int(sg['prim_id'][j])):>10s} {E:9.3f} {pt:8.3f} {math.asinh(pz / max(pt, 1e-9)):7.3f} "
                   f"{str(sg['fate_names'][sg['prim_fate'][j]]):>12s} {bb} {int(trk_n[j]):5d} {ecal[j]:9.3f} {hcal[j]:8.4f} {hf[j]:6.0f} {'*' if sg['prim_jet0'][j] else ''}")
    out.append(hdr.replace("Standalone SimHit Primary Listing", "End Standalone SimHit Primary Listing"))
    out.append("")
    hdr = " --------  Standalone SimHit Collision Listing  (the crowd: collision k, bx, primaries, tracks, tracker hits, muon hits, ECAL [GeV], HB+HE+HO scintillator [GeV], HF [p.e.])  ----------"
    out.append(hdr)
    for name in ("intime", "oot"):
        d = S[name]
        kk = d["col_k"].astype(int)
        nk = int(kk.max()) + 1
        tr = np.bincount(d["trk_k"].astype(int), minlength=nk)
        mu = np.bincount(d["mu_k"].astype(int), minlength=nk) if len(d["mu_k"]) else np.zeros(nk, int)
        det = d["cal_code"] // sim.CODE
        def per_k(mask):
            return np.bincount(d["cal_k"][mask].astype(int), weights=d["cal_e"][mask], minlength=nk)
        e1, e2, e3 = per_k(np.isin(det, sim.ECAL)), per_k(np.isin(det, (sim.HB, sim.HE, sim.HO))), per_k(det == sim.HF)
        for i, k in enumerate(kk):
            out.append(f"{k:5d} {int(d['col_bx'][i]):4d} {int(d['col_nprim'][i]):6d} {int(d['col_nprim'][i] + d['col_nsec'][i]):6d} {int(tr[k]):7d} {int(mu[k]):4d} {e1[k]:9.2f} {e2[k]:8.3f} {e3[k]:7.0f}")
    out.append(hdr.replace("Standalone SimHit Collision Listing", "End Standalone SimHit Collision Listing"))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--dir", type=Path, default=HERE.parent, help="where the three simhits_*.npz are")
    ap.add_argument("-i", "--input", type=Path, default=HERE.parent.parent / "G4" / "event_G4.lhe", help="the record of chapter 9")
    ap.add_argument("-o", "--output", type=Path, default=HERE.parent / "event_SIM.lhe")
    ap.add_argument("--tables", type=Path, default=HERE / "sim_ledger")
    ap.add_argument("--summary", type=Path, default=HERE / "add_hits_SIM.json")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    args = ap.parse_args()
    S = load(args.dir)
    st = sim.load_stage()
    M = mix(S)
    N = numbers(S, M, st)
    txt = text(N)
    args.tables.with_suffix(".tex").write_text(latex(N))
    args.tables.with_suffix(".txt").write_text(txt)
    args.summary.write_text(json.dumps(sim.to_jsonable(N), indent=1))
    rec = args.input.read_text()
    if not rec.endswith("\n"):
        rec += "\n"
    args.output.write_text(rec + "\n" + record_tables(N, S, st))
    if args.verbose:
        print("== adding the hits by hand ==")
        print(f"   every hit of a collision of bunch crossing bx is shifted by {BX_NS:.0f} ns x bx: -300 ns for bx = -12, +75 ns for bx = +3")
        print(f"   tracker hits: {' + '.join(str(N['tracker']['total'][o]) for o in ORIGINS)} = {tot(N['tracker']['total'])} (signal, early, in time, late)")
        print(f"   calorimeter cells: {len(M['cells'])} distinct cells receive energy; each sums up to 961 collisions")
        print()
    print(txt if args.verbose else f"add_hits_SIM: {args.output} written; tracker hits {tot(N['tracker']['total']):,d}, cells {len(M['cells']):,d}")


if __name__ == "__main__":
    main()
