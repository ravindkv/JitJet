#!/usr/bin/env python3
"""Standalone hand-over to Geant4 for the b bbar event of chapter 8: the gate at the border of the detector.

Dependencies: numpy, plus chapter 8's ../../PU/Standalone/standalone_pileup.py (the record of the crowd, the
library, the beam spot; chapters 7, 6 and 5 are imported through it). No CMSSW, no Geant4: the decision of
CMSSW's SimG4Core/Generators/src/Generator.cc (CMSSW_15_0_5), which generator particles become Geant4
primaries, is written out here line by line, together with the two pieces around it that also decide who
enters the simulation: the neutrinos the StackingAction kills before their first step, and the forward
protons that the PPS transport (SimTransport/PPSProtonTransport, LHCTransport) carries along the beam line
to the Roman pots and hands to Geant4 there.

Input is the record of chapter 8 (../../PU/event_PU.lhe: the signal of chapter 7 with its vertex table, the
vertex of every collision of the sixteen bunch crossings bx = -12 ... +3 and the particles of the in-time
crossing) and the pileup library the crowd was drawn from (../../PU/pileup_library.lhe.gz), from which the
897 out-of-time collisions are rebuilt by their library events (the record lists only the in-time particles).
Output is the chapter-8 record followed by the tables of this stop (../event_G4.lhe): the summary, every
stable particle of the signal and of the in-time crowd with the decision of the gate, one row per collision
of the window, and the decays the generator made.

What is implemented (the physics of chapter 9)
----------------------------------------------
* The detector the decision is made for: its edges only, the beam pipe (2.25 cm), the hole of the forward
  calorimeter HF (|eta| = 5.18 at its front face, 5.32 at its back: ForwdBeamR1 = 12.495 cm at |z| = 11.15 and
  12.80 m, cms.xml), the Zero Degree Calorimeter at 140 m and the Roman pots at 203 m. The cuts below are
  parameters of g4SimHits_cfi.py written by hand; they read none of these numbers.
* The decays the generator made: PYTHIA decays every hadron with tau0 < 10 mm (ParticleDecays:limitTau0 = on,
  tau0Max = 10 of Pythia8CommonSettings_cfi.py); a decayed hadron (HepMC status 2) whose decay vertex is
  inside the cylinder r < RDecLenCut = 2.9 cm keeps its decay, and its daughters start as primaries at the
  decay point; outside it, the parent itself is the primary, with the generator's decay attached as a
  predefined channel (none of our events has such a decay; the farthest is at r = 1.5 cm).
* The hand-over cuts of Generator.cc, for every stable particle (status 1), in the order of the code:
  - the very-forward rule: the straight line from the production vertex is followed to |z| = Z_hector =
    2.9 cm x sinh(8.2) = 52.8 m (EtaCutForHector); if it is still inside r < 2.9 cm there, the particle is
    given to Geant4 only if it is a neutron, an antineutron or a photon (for the Zero Degree Calorimeter);
  - otherwise the momentum cut, 40 MeV < |p| < 99 999 GeV (MinPCut, MaxPCut, on the total momentum);
  - and the eta cut, which is a line, not an angle: the line is followed to |z| = Z_lmax = 2.9 cm x
    sinh(5.5) = 354.8 cm (MinEtaCut, MaxEtaCut) and the particle is dropped if it is still inside r < 2.9 cm
    there; from the nominal collision point this is |eta| > 5.5, from a displaced vertex it is not.
* The StackingAction (TrackNeutrino = False): the neutrinos that pass the cuts are killed before their first
  step and get no SimTrack.
* The PPS transport (LHCTransport, switched on by run3_common): every stable proton with |eta| > 8.2 and |pz| >
  3000 GeV (CommonParameters_cfi.py) is carried along the beam line by the optical functions and handed to
  Geant4 at |z| = 203 m (useEmpiricalApertures = False in the Run-3 setup: no proton is lost on the way).
* The ledger of the hand-over, in the layout of the CMSSW companion's g4Ledger.py: the stable particles before
  and the primaries after, by region and by kind, for the signal, the in-time and the out-of-time pileup and
  the signal with its in-time pileup; the drops by reason (|eta| > 5.5, p < 40 MeV, neutrino, other: tested
  in this order with the eta of the momentum, so that "other" is the line at work); the protons at the
  Roman pots; and what the hand-over does to the b and bbar systems and to our jet.
* --library-scan: the same gate on all 2000 collisions of the library, each at a vertex drawn from the 2025
  beam spot, for the per-collision averages.

What is left out, on purpose: the PDG filter (it removes status-1 quarks and gluons, of which a hadron-level
record has none); the phi and pT cuts (off in CMSSW); the luminosity-monitor filter (off); the exotic and
slepton special cases; Geant4 itself, the field, the material, the showers and the hits, which are the next
stops (chapters 10 and 11).

Learning mode (--verbose / -v, -vv)
------------------------------------
  -v   narrates the rules and their numbers, the three crowds arriving, the decays the generator made and
       where they happened, the gate rule by rule for each crowd, the border particles of the signal and of
       the in-time crowd where the line and the angle disagree, the soft photons, the neutrinos and the
       missing pT they leave, the forward neutrals for the ZDC and the protons for the Roman pots, the b and
       bbar systems and our jet as Geant4 receives them, and the ledger.
  -vv  adds every stable particle of the signal that does not become a primary, and one line per in-time
       collision.

Examples
--------
  python3 standalone_G4.py                          # -> ../event_G4.lhe (a few seconds)
  python3 standalone_G4.py -v                       # learning mode
  python3 standalone_G4.py --library-scan           # the gate on the 2000 library collisions
  python3 standalone_G4.py --eta-cut 7.0            # the CASTOR-era cut of SimWithCastor_cff.py (exercise)

All energies and momenta in GeV, lengths in cm, times in ns.
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
sys.path.insert(0, str(HERE.parent.parent / "PU" / "Standalone"))
import standalone_pileup as pu  # noqa: E402  (chapter 8: the crowd, the library, the beam spot)

gj = pu.gj
dec = pu.dec

# ---------------------------------------------------------------------------
# the gate: the Generator PSet of SimG4Core/Application/python/g4SimHits_cfi.py (CMSSW_15_0_5)
# ---------------------------------------------------------------------------
R_DEC = 2.9                       # cm, RDecLenCut: the cylinder of the eta cut and of the decays
ETA_CUT = 5.5                     # MinEtaCut = -5.5, MaxEtaCut = 5.5
ETA_HECTOR = 8.2                  # EtaCutForHector (HectorEtaCut of hectorParameter_cfi.py)
P_MIN, P_MAX = 0.04, 99999.0      # GeV, MinPCut, MaxPCut (on the total momentum)
MIN_TAN = 1e-20                   # Generator.cc: protection against pz = 0
VERY_FORWARD_IDS = (2112, -2112, 22)   # n, nbar, gamma: Generator.cc, "very forward n, nbar, gamma are allowed"
TAU0_MAX = 10.0                   # mm, ParticleDecays:tau0Max of Pythia8CommonSettings_cfi.py (limitTau0 = on)
# the PPS transport (SimTransport/PPSProtonTransport/python/CommonParameters_cfi.py, OpticalFunctionsTransport)
PPS_ETA, PPS_PZ, PPS_Z = 8.2, 3000.0, 20300.0   # EtaCut, MomentumCut [GeV], PPSRegionStart_45/56 = 203 m
E_BEAM = 6800.0                   # GeV
# the edges of the detector (cms.xml, hcalforwardalgo.xml, beampipe.xml of the release; for the narration)
R_BEAMPIPE = 2.25                 # cm, BeampipeRout1
HF_R_MIN, HF_Z1, HF_Z2 = 12.495, 1114.95, 1280.05   # cm: ForwdBeamR1, ForwdVcalZ1, ForwdVcalZ2
Z_ZDC = 14000.0                   # cm, the ZDC (cmszdc.xml)
TRACKER_R, TRACKER_Z = 123.3, 293.2                 # cm: TrackCalorR, TrackBeamZ2


def sinh_cut(eta: float, r: float = R_DEC) -> float:
    """Generator.cc's Z = R (1 - exp(-2 eta)) / (2 exp(-eta)) = R sinh(eta)."""
    return r * (1.0 - math.exp(-2.0 * eta)) / (2.0 * math.exp(-eta))


Z_LMAX = sinh_cut(ETA_CUT)        # 354.8 cm
Z_HECTOR = sinh_cut(ETA_HECTOR)   # 5279.4 cm
ETA_HF_FRONT = math.asinh(HF_Z1 / HF_R_MIN)   # 5.18
ETA_HF_BACK = math.asinh(HF_Z2 / HF_R_MIN)    # 5.32

KINDS = ["charged hadron", "neutral hadron", "photon", "electron", "muon", "neutrino"]
REGIONS = [("|eta|<2.5", 0.0, 2.5), ("2.5-3.0", 2.5, 3.0), ("3.0-5.2", 3.0, 5.2), ("5.2-5.5", 5.2, 5.5), (">5.5", 5.5, math.inf)]
REASONS = ["eta", "p", "nu", "other"]          # the order of g4Ledger.py
BLOCKS = ["signal", "intime", "oot"]
# the decision codes of the hand-over listing
CODE = {"primary": "P", "very_forward": "V", "neutrino": "N", "eta_line": "E", "forward": "F", "momentum": "M", "pps": "R"}


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


# ---------------------------------------------------------------------------
# the crowds that arrive at the gate: one flat table of stable particles with their production vertices
# ---------------------------------------------------------------------------
class Crowd:
    """Every stable particle of the window: block (0 signal, 1 in time, 2 out of time), collision k, row in the
    signal record or index in the in-time listing, id, (E, px, py, pz), production vertex (x, y, z) [cm]."""

    def __init__(self, block, k, row, ids, p4, vtx):
        self.block = np.asarray(block, dtype=int)
        self.k = np.asarray(k, dtype=int)
        self.row = np.asarray(row, dtype=int)
        self.ids = np.asarray(ids, dtype=int)
        self.p4 = np.asarray(p4, dtype=float).reshape(-1, 4)
        self.vtx = np.asarray(vtx, dtype=float).reshape(-1, 3)
        E, px, py, pz = self.p4.T
        self.p = np.sqrt(px * px + py * py + pz * pz)
        self.pt = np.hypot(px, py)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.eta = np.where(self.pt > 0, np.arcsinh(pz / np.where(self.pt > 0, self.pt, 1.0)), np.sign(pz) * 99.0)
        self.phi = np.arctan2(py, px)
        self.kind = np.array([kind_of(int(i)) for i in self.ids])
        self.nu = self.kind == "neutrino"

    def __len__(self):
        return len(self.ids)


def signal_vertex(collisions) -> np.ndarray:
    """The smeared primary vertex of the signal (x, y in um and z in cm in the record) in cm."""
    s = collisions[0]
    return np.array([s["x"] * 1e-4, s["y"] * 1e-4, s["z"]])


def build_crowd(ev, vertex, collisions, particles, library: Path | None):
    """The three crowds as one Crowd. The signal's particles start at the smeared vertex plus their own displacement
    (the vertex table of chapter 6, in mm); a pileup particle starts at the vertex of its collision (the library
    keeps no displaced vertices: its few heavy-flavour decays happen within a millimetre of the collision)."""
    pv = signal_vertex(collisions)
    finals = ev.final_indices()
    blk, kk, rows, ids, p4, vtx = [], [], [], [], [], []
    for i in finals:
        d = vertex.get(i)
        v = pv + (np.array(d[1:4]) / 10.0 if d is not None else 0.0)
        blk.append(0); kk.append(0); rows.append(i); ids.append(ev.p[i].id); p4.append(ev.p[i].p); vtx.append(v)
    by_k = {c["k"]: c for c in collisions}
    for j, q in enumerate(particles):
        c = by_k[q["k"]]
        blk.append(1); kk.append(q["k"]); rows.append(j); ids.append(q["id"]); p4.append([q["e"], q["px"], q["py"], q["pz"]])
        vtx.append([c["x"] * 1e-4, c["y"] * 1e-4, c["z"]])
    n_oot = 0
    if library is not None:
        lib = pu.Library(library)
        by_no = {e["no"]: e for e in lib.events}
        for c in collisions:
            if c["k"] > 0 and c["bx"] != 0:
                e = by_no[c["lib"]]
                n = len(e["ids"])
                blk += [2] * n; kk += [c["k"]] * n; rows += list(range(n)); ids += list(e["ids"]); p4 += list(e["p4"])
                vtx += [[c["x"] * 1e-4, c["y"] * 1e-4, c["z"]]] * n
                n_oot += 1
    return Crowd(blk, kk, rows, ids, p4, vtx), n_oot


def hepmc_status2(ev):
    """The rows PYTHIA's statusHepMC() would write with status 2: hadrons, muons and taus decayed normally
    (first daughter with status 91 ... 99 and another identity)."""
    out = []
    for i, q in enumerate(ev.p):
        if q.status >= 0 or q.d1 <= 0 or q.d1 >= len(ev.p):
            continue
        a = abs(q.id)
        if not (a >= 100 or a in (13, 15)):
            continue
        d = ev.p[q.d1]
        if d.id != q.id and 90 < abs(d.status) < 100:
            out.append(i)
    return out


class Decays:
    """The decays the generator made in the signal: for every status-2 row its production and decay vertex [cm]."""

    def __init__(self, ev, vertex, pv):
        self.rows = hepmc_status2(ev)
        self.items = []
        for i in self.rows:
            q = ev.p[i]
            d0 = vertex.get(i)
            d1 = vertex.get(q.d1)
            prod = pv + (np.array(d0[1:4]) / 10.0 if d0 is not None else 0.0)
            decv = pv + (np.array(d1[1:4]) / 10.0 if d1 is not None else 0.0)
            E, px, py, pz = q.p
            p = math.sqrt(px * px + py * py + pz * pz)
            m = math.sqrt(max(E * E - p * p, 0.0)) or dec.m0(q.id)
            flight = float(np.linalg.norm(decv - prod))
            ctau = dec.tau0(q.id) / 10.0                     # mm -> cm
            r_dec = float(math.hypot(decv[0], decv[1]))
            self.items.append(dict(row=i, id=q.id, name=dec.name(q.id), E=E, p=p, pt=math.hypot(px, py), eta=gj.eta(q.p), m=m,
                                   betagamma=p / m if m > 0 else math.inf, ctau=ctau, flight=flight, prod=prod, dec=decv,
                                   r_dec=r_dec, z_dec=float(decv[2]), inside=r_dec * r_dec < R_DEC * R_DEC,
                                   weak=dec.tau0(q.id) > 1e-4, n_daughters=max(q.d1, q.d2) - q.d1 + 1))

    def weak(self):
        return [d for d in self.items if d["weak"]]

    def outside(self):
        return [d for d in self.items if not d["inside"] and abs(d["z_dec"]) < Z_HECTOR]


# ---------------------------------------------------------------------------
# the gate: Generator.cc, StackingAction, PPS
# ---------------------------------------------------------------------------
class Gate:
    """The hand-over of Generator::HepMC2G4 for status-1 particles, vectorised. Every array is one entry per particle."""

    def __init__(self, eta_cut=ETA_CUT, eta_hector=ETA_HECTOR, r_dec=R_DEC, p_min=P_MIN, p_max=P_MAX):
        self.eta_cut, self.eta_hector, self.r = eta_cut, eta_hector, r_dec
        self.p_min, self.p_max = p_min, p_max
        self.z_lmax = sinh_cut(eta_cut, r_dec)
        self.z_lmin = -self.z_lmax
        self.z_hector = sinh_cut(eta_hector, r_dec)

    def line_at(self, c: Crowd, z_target: float):
        """The point (x, y) where the straight line from the production vertex reaches |z| = z_target on its own side, and r."""
        E, px, py, pz = c.p4.T
        x1, y1, z1 = c.vtx.T
        zt = np.where(pz > 0, z_target, -z_target)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(pz != 0, (zt - z1) / pz, 0.0)
        xi, yi = x1 + d * px, y1 + d * py
        return xi, yi, np.hypot(xi, yi)

    def decide(self, c: Crowd) -> dict:
        E, px, py, pz = c.p4.T
        x1, y1, z1 = c.vtx.T
        ptot = c.p
        pz_ok = np.abs(pz) >= MIN_TAN * ptot
        # the impact point at the transition to the forward transport, |z| = Z_hector
        to_h = (np.abs(z1) < self.z_hector) & pz_ok
        zimp = np.where(to_h, np.where(pz > 0, self.z_hector, -self.z_hector), z1)
        with np.errstate(divide="ignore", invalid="ignore"):
            dl = np.where(to_h, (zimp - z1) / np.where(pz != 0, pz, 1.0), 0.0)
        ximp, yimp = x1 + dl * px, y1 + dl * py
        rimp2 = ximp * ximp + yimp * yimp
        very_forward = (np.abs(zimp) >= self.z_hector) & (rimp2 <= self.r * self.r)
        vf_ok = very_forward & np.isin(c.ids, VERY_FORWARD_IDS) & (c.ids != -22)
        std = ~very_forward
        p_fail = std & ((ptot < self.p_min) | (ptot > self.p_max))
        # the eta cut: the line to Z_lmax (or Z_lmin), with the cases of Generator.cc for vertices beyond them
        zi = np.where((z1 >= self.z_lmax) & (pz < 0), self.z_lmax,
                      np.where((z1 <= self.z_lmin) & (pz > 0), self.z_lmin, np.where(pz > 0, self.z_lmax, self.z_lmin)))
        zi = np.where(pz_ok, zi, z1)
        with np.errstate(divide="ignore", invalid="ignore"):
            de = np.where(pz_ok, (zi - z1) / np.where(pz != 0, pz, 1.0), 0.0)
        xi, yi = x1 + de * px, y1 + de * py
        r_line = np.hypot(xi, yi)
        eta_fail = std & ~p_fail & (zi >= self.z_lmin) & (zi <= self.z_lmax) & (xi * xi + yi * yi < self.r * self.r)
        added = vf_ok | (std & ~p_fail & ~eta_fail)
        killed = added & c.nu                                   # StackingAction, TrackNeutrino = False
        primary = added & ~killed
        pps = (c.ids == 2212) & (np.abs(c.eta) > PPS_ETA) & (np.abs(pz) > PPS_PZ)
        # the decision of the gate, one code per particle
        code = np.full(len(c), "?", dtype="<U1")
        code[primary & ~very_forward] = CODE["primary"]
        code[primary & very_forward] = CODE["very_forward"]
        code[killed] = CODE["neutrino"]
        code[very_forward & ~vf_ok] = CODE["forward"]
        code[p_fail] = CODE["momentum"]
        code[eta_fail] = CODE["eta_line"]
        code[pps] = CODE["pps"]
        # the reason of the ledger (g4Ledger.py: the eta of the momentum first, then p, then neutrino, then 'other')
        lost = ~primary & ~pps
        reason = np.full(len(c), "", dtype="<U5")
        r_eta = lost & (np.abs(c.eta) > self.eta_cut)
        r_p = lost & ~r_eta & (ptot < self.p_min)
        r_nu = lost & ~r_eta & ~r_p & c.nu
        r_other = lost & ~r_eta & ~r_p & ~r_nu
        reason[r_eta], reason[r_p], reason[r_nu], reason[r_other] = "eta", "p", "nu", "other"
        return dict(primary=primary, very_forward=very_forward, vf_ok=vf_ok, p_fail=p_fail, eta_fail=eta_fail, killed=killed,
                    pps=pps, code=code, reason=reason, r_line=r_line, r_hector=np.sqrt(rimp2), lost=lost)


# ---------------------------------------------------------------------------
# the books
# ---------------------------------------------------------------------------
def book(c: Crowd, m) -> dict:
    return dict(n=int(m.sum()), e=float(c.p4[m, 0].sum()), pt=float(c.pt[m].sum()))


def ledger(c: Crowd, g: dict, sel) -> dict:
    """Stable particles before, primaries after (from the vertex) and PPS protons, by region and by kind; drops by reason."""
    a = np.abs(c.eta)
    out = dict(stable=book(c, sel), primary=book(c, sel & g["primary"]), pps=book(c, sel & g["pps"]),
               drops={r: book(c, sel & (g["reason"] == r)) for r in REASONS},
               regions={}, kinds={})
    for name, lo, hi in REGIONS:
        m = sel & (a >= lo) & (a < hi)
        out["regions"][name] = dict(stable=book(c, m), primary=book(c, m & g["primary"]), pps=book(c, m & g["pps"]))
    for k in KINDS:
        m = sel & (c.kind == k)
        out["kinds"][k] = dict(stable=book(c, m), primary=book(c, m & g["primary"]), pps=book(c, m & g["pps"]))
    st, pr = out["stable"], out["primary"]
    out["fractions"] = dict(n=pr["n"] / st["n"] if st["n"] else 0.0, e=pr["e"] / st["e"] if st["e"] else 0.0,
                            pt=pr["pt"] / st["pt"] if st["pt"] else 0.0)
    out["very_forward"] = book(c, sel & g["primary"] & g["very_forward"])
    out["beyond"] = book(c, sel & g["primary"] & (a > ETA_CUT))
    out["killed_nu"] = book(c, sel & g["killed"])
    return out


def vector_pt(c: Crowd, m) -> tuple:
    v = c.p4[m].sum(axis=0) if m.any() else np.zeros(4)
    pt = math.hypot(v[1], v[2])
    m2 = v[0] ** 2 - v[1] ** 2 - v[2] ** 2 - v[3] ** 2
    return pt, math.sqrt(max(m2, 0.0)), (gj.eta(v) if pt > 0 else 0.0), (math.atan2(v[2], v[1]) if pt > 0 else 0.0)


def descendants_final(ev, row: int):
    out, stack = [], [row]
    while stack:
        r = stack.pop()
        q = ev.p[r]
        if q.final:
            out.append(r)
        elif q.d1 > 0:
            stack.extend(range(q.d1, max(q.d1, q.d2) + 1))
    return sorted(set(out))


def b_systems(ev):
    """The weakly decaying b hadron of each side and its stable descendants (the 'b system' of the CMSSW companion):
    the b hadron that carries the flavour of the hard b (chapter 6's string_b_hadrons) followed down to its last copy
    before the weak decay."""
    out = {}
    try:
        firsts = dec.string_b_hadrons(ev)
    except Exception:
        firsts = {}
    for pid, row in firsts.items():
        r = row
        # follow the chain down to the hadron that decays weakly (a B* first decays to a B and a photon)
        while True:
            q = ev.p[r]
            kids = list(range(q.d1, max(q.d1, q.d2) + 1)) if q.d1 > 0 else []
            nxt = [k for k in kids if dec.heavy_content(ev.p[k].id) == 5 and abs(ev.p[k].id) > 100]
            if len(nxt) == 1 and not dec.is_weak(q.id):
                r = nxt[0]
                continue
            if len(nxt) == 1 and nxt[0] != r and ev.p[nxt[0]].id != q.id and dec.is_weak(q.id) is False:
                r = nxt[0]
                continue
            # a mixing copy (B0 -> Bbar0) has the same row structure: follow the single b-hadron daughter
            if len(nxt) == 1 and len(kids) == 1:
                r = nxt[0]
                continue
            break
        out["b" if pid > 0 else "bbar"] = dict(first=row, weak=r, finals=descendants_final(ev, r))
    return out


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------
class Stage:
    def __init__(self, args, verbose=0, log=None):
        self.args, self.verbose = args, verbose
        self.out = log if log is not None else sys.stdout
        t0 = time.time()
        self.ev, self.vertex, self.tau = dec.read_record(args.input)
        self.genjets, self.genjets_of = gj.read_jet_tables(args.input)
        self.summary, self.collisions, self.particles, self.mixed, self.signal_of = pu.read_pileup_tables(args.input)
        self.pv = signal_vertex(self.collisions)
        self.crowd, self.n_oot = build_crowd(self.ev, self.vertex, self.collisions, self.particles, args.library)
        self.n_intime = len({q["k"] for q in self.particles})
        self.decays = Decays(self.ev, self.vertex, self.pv)
        self.gate = Gate(eta_cut=args.eta_cut, eta_hector=args.eta_hector, r_dec=args.r_dec, p_min=args.p_min)
        self.g = self.gate.decide(self.crowd)
        self.t_read = time.time() - t0

    def say(self, text="", level=1):
        if self.verbose >= level:
            print(text, file=self.out)

    # ---------------------------------------------------------------- the numbers
    def compute(self) -> dict:
        c, g = self.crowd, self.g
        sel = {b: c.block == i for i, b in enumerate(BLOCKS)}
        sel["all"] = sel["signal"] | sel["intime"]
        L = {b: ledger(c, g, sel[b]) for b in ("signal", "intime", "oot", "all")}
        # per collision
        per = []
        for col in self.collisions:
            m = (c.k == col["k"]) & (c.block == (0 if col["k"] == 0 else (1 if col["bx"] == 0 else 2)))
            if col["k"] > 0 and not m.any() and col["bx"] != 0 and self.n_oot == 0:
                continue
            per.append(dict(k=col["k"], bx=col["bx"], lib=col["lib"], kind=col["kind"], z=col["z"], stable=book(c, m), primary=book(c, m & g["primary"]),
                            pps=book(c, m & g["pps"]), drops={r: int((m & (g["reason"] == r)).sum()) for r in REASONS}))
        # the b systems and our jet
        bs = {}
        row_index = {int(r): j for j, r in enumerate(c.row) if c.block[j] == 0}
        for key, b in b_systems(self.ev).items():
            idx = np.array([row_index[r] for r in b["finals"] if r in row_index], dtype=int)
            m = np.zeros(len(c), bool)
            m[idx] = True
            q = self.ev.p[b["weak"]]
            before, after = vector_pt(c, m), vector_pt(c, m & g["primary"])
            q0 = self.ev.p[b["first"]]
            bs[key] = dict(first=dec.name(q0.id), first_pt=gj.pt(q0.p), hadron=dec.name(q.id), id=q.id, row=b["weak"], pt_hadron=gj.pt(q.p), eta_hadron=gj.eta(q.p), phi_hadron=gj.phi(q.p),
                           n=int(m.sum()), n_primary=int((m & g["primary"]).sum()), pt_before=before[0], pt_after=after[0],
                           eta=before[2], phi=before[3], lost=[dict(id=int(c.ids[j]), name=dec.name(int(c.ids[j])), code=str(g["code"][j]),
                                                                   pt=float(c.pt[j]), e=float(c.p4[j, 0]), eta=float(c.eta[j])) for j in idx if not g["primary"][j]])
        net = {}
        if "b" in bs and "bbar" in bs:
            for when, key in (("before", None), ("after", "primary")):
                v = np.zeros(4)
                for k2 in ("b", "bbar"):
                    idx = np.array([row_index[r] for r in b_systems(self.ev)[k2]["finals"] if r in row_index], dtype=int)
                    mm = np.zeros(len(c), bool)
                    mm[idx] = True
                    if key:
                        mm &= g["primary"]
                    v += c.p4[mm].sum(axis=0)
                net[when] = math.hypot(v[1], v[2])
        jets = {}
        for coll in ("ak4", "ak4nonu"):
            js = self.genjets.get(coll, [])
            if not js:
                continue
            j0 = js[0]
            rows = [r for r, rank in self.genjets_of[coll].items() if rank == 0]
            idx = np.array([row_index[r] for r in rows if r in row_index], dtype=int)
            m = np.zeros(len(c), bool)
            m[idx] = True
            before, after = vector_pt(c, m), vector_pt(c, m & g["primary"])
            jets[coll] = dict(rank=0, pt=j0["pt"], m=j0["m"], n=j0["n"], eta=j0["eta"], phi=j0["phi"], pt_rebuilt=before[0], m_rebuilt=before[1],
                              n_rebuilt=int(m.sum()), pt_after=after[0], m_after=after[1], n_after=int((m & g["primary"]).sum()),
                              lost=[dict(id=int(c.ids[j]), name=dec.name(int(c.ids[j])), code=str(g["code"][j]), pt=float(c.pt[j]), eta=float(c.eta[j]))
                                    for j in idx if not g["primary"][j]])
        # our jet with the crowd: the mixed jet that matches the photograph's jet [0]
        mixed = [j for j in self.mixed if j["genjet"] == 0]
        if mixed:
            mj = min(mixed, key=lambda j: j["dr"] if j["dr"] is not None else 9.0)
            sig_rows = [r for r, rank in self.signal_of.items() if rank == mj["rank"]]
            idx_s = [row_index[r] for r in sig_rows if r in row_index]
            pu_idx = np.where((c.block == 1))[0]
            idx_p = [pu_idx[j] for j, q in enumerate(self.particles) if q["jet"] == mj["rank"]]
            idx = np.array(idx_s + idx_p, dtype=int)
            m = np.zeros(len(c), bool)
            m[idx] = True
            before, after = vector_pt(c, m), vector_pt(c, m & g["primary"])
            jets["mixed"] = dict(rank=mj["rank"], pt=mj["pt"], m=mj["m"], n=mj["n"], pt_sub=mj["pt_sub"], rho_area=mj["rho_area"],
                                 pt_rebuilt=before[0], n_rebuilt=int(m.sum()), pt_after=after[0], m_after=after[1], n_after=int((m & g["primary"]).sum()))
        # border particles: where the line and the angle disagree
        a = np.abs(c.eta)
        line_vs_angle = dict(
            kept_beyond=[int(j) for j in np.where(g["primary"] & (a > self.gate.eta_cut) & ~g["very_forward"])[0]],
            dropped_inside=[int(j) for j in np.where(g["eta_fail"] & (a <= self.gate.eta_cut))[0]])
        # forward: the neutrals for the ZDC and the protons for the Roman pots
        vf = np.where(g["primary"] & g["very_forward"])[0]
        pps = np.where(g["pps"])[0]
        # the decays
        D = self.decays
        weak = D.weak()
        r_max = max((d["r_dec"] - math.hypot(*self.pv[:2]) for d in D.items), default=0.0)
        far = max(weak, key=lambda d: d["r_dec"]) if weak else None
        # distinct displaced decay points (more than 1 um from the vertex and from each other)
        pts = []
        for d in D.items:
            dv = d["dec"]
            if np.linalg.norm(dv - self.pv) > 1e-4 and all(np.linalg.norm(dv - p) > 1e-4 for p in pts):
                pts.append(dv)
        return dict(L=L, per=per, b=bs, net=net, jets=jets, line_vs_angle=line_vs_angle, vf=vf, pps=pps,
                    decays=dict(n=len(D.items), n_weak=len(weak), n_outside=len(D.outside()), n_points=len(pts), r_max=r_max, far=far))

    # ---------------------------------------------------------------- narration
    def narrate(self, R: dict):
        c, g, G = self.crowd, self.g, self.gate
        say = self.say
        say(gj.banner("JitJet chapter 9: the gate at the border, the hand-over to Geant4") if hasattr(gj, "banner") else "== JitJet chapter 9: the hand-over to Geant4")
        say(f"input {self.args.input.name}: the signal of chapter 7 and the crowd of chapter 8; library {self.args.library.name if self.args.library else '-'}")
        say("")
        say("-- 1. the rules of the gate (g4SimHits_cfi.py, Generator PSet; Generator.cc of CMSSW_15_0_5)")
        say(f"   RDecLenCut = {G.r:.1f} cm: the cylinder around the beam (the beam pipe ends at r = {R_BEAMPIPE} cm)")
        say(f"   eta cut {G.eta_cut}: Z_lmax = R sinh({G.eta_cut}) = {G.r:.1f} x {math.sinh(G.eta_cut):.2f} = {G.z_lmax:.1f} cm: a stable particle whose line is still")
        say(f"       inside r < {G.r:.1f} cm at |z| = {G.z_lmax:.1f} cm is dropped")
        say(f"   very-forward rule {G.eta_hector}: Z_hector = R sinh({G.eta_hector}) = {G.r:.1f} x {math.sinh(G.eta_hector):.1f} = {G.z_hector:.1f} cm = {G.z_hector / 100:.1f} m:")
        say("       a line still inside there is given to Geant4 only for n, nbar and photons (the ZDC at 140 m)")
        say(f"   momentum cut: {G.p_min * 1000:.0f} MeV < |p| < {P_MAX:.0f} GeV (MinPCut, MaxPCut)")
        say("   StackingAction: TrackNeutrino = False (the neutrinos are killed before their first step)")
        say(f"   PPS transport (LHCTransport): protons with |eta| > {PPS_ETA} and |pz| > {PPS_PZ:.0f} GeV, handed to Geant4 at |z| = {PPS_Z / 100:.0f} m")
        say(f"   the edges the cuts follow: HF hole |eta| = asinh({HF_Z1:.2f}/{HF_R_MIN}) = {ETA_HF_FRONT:.2f} (front face) to asinh({HF_Z2:.2f}/{HF_R_MIN}) = {ETA_HF_BACK:.2f} (back)")
        say("")
        L = R["L"]
        say("-- 2. the crowds arrive at the gate")
        sv = self.pv
        say(f"   the signal: {L['signal']['stable']['n']} stable particles, {L['signal']['stable']['e']:.1f} GeV, scalar pT {L['signal']['stable']['pt']:.1f} GeV; "
            f"vertex ({sv[0] * 1e4:.1f} um, {sv[1] * 1e4:.1f} um, {sv[2]:.3f} cm)")
        say(f"   the in-time crowd: {self.n_intime} collisions, {L['intime']['stable']['n']} stable particles, {L['intime']['stable']['e']:.1f} GeV, scalar pT {L['intime']['stable']['pt']:.1f} GeV")
        if self.n_oot:
            say(f"   the out-of-time crowd: {self.n_oot} collisions, {L['oot']['stable']['n']} stable particles, {L['oot']['stable']['e']:.1f} GeV, scalar pT {L['oot']['stable']['pt']:.1f} GeV")
        zs = [col["z"] for col in self.collisions if col["k"] > 0]
        say(f"   the vertices along the beam: z from {min(zs):+.2f} to {max(zs):+.2f} cm (r.m.s. {np.std(zs):.2f} cm)")
        say("")
        D, Rd = self.decays, R["decays"]
        say("-- 3. the decays the generator made (signal; PYTHIA decays what has tau0 < 10 mm, ParticleDecays:tau0Max)")
        say(f"   {Rd['n']} hadrons and leptons decayed by the generator (HepMC status 2), {Rd['n_weak']} of them weakly (tau0 > 0.1 um);")
        say(f"   {Rd['n_points']} distinct decay points more than 1 um from the collision; {Rd['n_outside']} decay outside r = {G.r} cm")
        say(f"   {'hadron':<14s}{'E':>8s}{'pT':>7s}{'eta':>7s}{'bg':>7s}{'c tau':>8s}{'<r_T>':>8s}{'flight':>8s}{'r_dec':>8s}{'z_dec':>9s}{'P(r<R)':>8s}")
        for d in D.weak():
            rt = d["pt"] / d["m"] * d["ctau"] if d["m"] > 0 else math.inf
            pin = 1.0 - math.exp(-G.r / rt) if rt > 0 else 1.0
            say(f"   {d['name']:<14s}{d['E']:8.1f}{d['pt']:7.1f}{d['eta']:7.2f}{d['betagamma']:7.1f}{d['ctau'] * 1e4:6.0f}um{rt:8.3f}{d['flight']:8.3f}{d['r_dec']:8.3f}{d['z_dec']:9.3f}{pin:8.4f}")
        say("   (<r_T> = (pT/m) c tau, the mean transverse decay length; P(r<R) = 1 - exp(-R/<r_T>) from the beam line; lengths in cm)")
        if Rd["far"]:
            f = Rd["far"]
            say(f"   the farthest from the beam line: the {f['name']} at r = {f['r_dec']:.3f} cm, z = {f['z_dec']:+.2f} cm ({f['flight']:.2f} cm from where it was made): inside; every daughter starts there")
        say("")
        say("-- 4. the gate, rule by rule")
        for b, label in (("signal", "signal"), ("intime", f"in-time crowd ({self.n_intime} collisions)"), ("oot", f"out-of-time crowd ({self.n_oot} collisions)")):
            if b == "oot" and not self.n_oot:
                continue
            m = c.block == BLOCKS.index(b)
            say(f"   {label}: {int(m.sum())} stable")
            say(f"     very forward (line inside r < {G.r} cm at {G.z_hector / 100:.1f} m): {int((m & g['very_forward']).sum())}; "
                f"kept (n, nbar, gamma) {int((m & g['vf_ok']).sum())} with {c.p4[m & g['vf_ok'], 0].sum():.1f} GeV; dropped {int((m & g['very_forward'] & ~g['vf_ok']).sum())}")
            say(f"     momentum cut: {int((m & g['p_fail']).sum())} dropped ({c.p4[m & g['p_fail'], 0].sum():.2f} GeV)")
            say(f"     eta cut (the line at {G.z_lmax:.1f} cm): {int((m & g['eta_fail']).sum())} dropped ({c.p4[m & g['eta_fail'], 0].sum():.1f} GeV)")
            say(f"     neutrinos killed by the StackingAction: {int((m & g['killed']).sum())} ({c.p4[m & g['killed'], 0].sum():.1f} GeV)")
            say(f"     protons to the Roman pots: {int((m & g['pps']).sum())} ({c.p4[m & g['pps'], 0].sum():.1f} GeV)")
            Lb = L[b]
            say(f"     -> {Lb['primary']['n']} primaries, {Lb['primary']['e']:.1f} GeV, pT {Lb['primary']['pt']:.1f} GeV: "
                f"{100 * Lb['fractions']['n']:.1f} % of the particles, {100 * Lb['fractions']['e']:.1f} % of the energy, {100 * Lb['fractions']['pt']:.1f} % of the pT")
        say("")
        say(f"-- 5. the line, not the angle: stable signal particles with 5.3 < |eta| < 5.7 (r of the line at |z| = {G.z_lmax:.1f} cm)")
        say(f"   {'eta':>9s}{'id':>7s}{'E':>8s}{'pT':>7s}{'z_vtx':>8s}{'r(nominal)':>12s}{'r(vertex)':>11s}  decision")
        m = (c.block == 0) & (np.abs(c.eta) > 5.3) & (np.abs(c.eta) < 5.7)
        for j in sorted(np.where(m)[0], key=lambda j: c.eta[j]):
            rn = G.z_lmax / math.sinh(abs(c.eta[j]))
            say(f"   {c.eta[j]:+9.4f}{c.ids[j]:7d}{c.p4[j, 0]:8.1f}{c.pt[j]:7.3f}{c.vtx[j, 2]:8.2f}{rn:12.3f}{g['r_line'][j]:11.3f}  {g['code'][j]}")
        lva = R["line_vs_angle"]
        for key, text in (("kept_beyond", f"kept although |eta| > {G.eta_cut} (the line from the vertex leaves the cylinder)"),
                          ("dropped_inside", f"dropped although |eta| < {G.eta_cut} (the line from the vertex stays inside)")):
            lst = lva[key]
            say(f"   {text}: {len(lst)}" + (": " + ", ".join(f"{BLOCKS[c.block[j]]} k={c.k[j]} {dec.name(int(c.ids[j]))} eta {c.eta[j]:+.4f} z {c.vtx[j, 2]:+.2f} cm r {g['r_line'][j]:.3f}"
                                                         for j in lst[:12]) if lst else ""))
        say("")
        say(f"-- 6. the soft and the invisible (signal)")
        m = (c.block == 0) & g["p_fail"]
        say(f"   below {G.p_min * 1000:.0f} MeV: {int(m.sum())}: " + ", ".join(f"{dec.name(int(c.ids[j]))} {c.p[j] * 1000:.1f} MeV" for j in np.where(m)[0]))
        m = (c.block == 0) & c.nu
        for j in np.where(m)[0]:
            say(f"   neutrino {dec.name(int(c.ids[j]))}: E {c.p4[j, 0]:.2f} GeV, pT {c.pt[j]:.2f}, eta {c.eta[j]:+.2f}, phi {c.phi[j]:+.2f}: {g['code'][j]}")
        if R["net"]:
            say(f"   |sum pT| of the b and bbar systems: {R['net']['before']:.2f} GeV before the gate, {R['net']['after']:.2f} GeV among the primaries")
        nu_all = (c.block == 0) & c.nu
        vx, vy = c.p4[nu_all, 1].sum(), c.p4[nu_all, 2].sum()
        say(f"   all neutrinos of the signal: |sum pT| = {math.hypot(vx, vy):.2f} GeV, the missing pT the gate decides")
        say("")
        say("-- 7. the forward: neutrals for the ZDC, protons for the Roman pots")
        for b in ("signal", "intime", "oot"):
            if b == "oot" and not self.n_oot:
                continue
            vf = [j for j in R["vf"] if c.block[j] == BLOCKS.index(b)]
            pp = [j for j in R["pps"] if c.block[j] == BLOCKS.index(b)]
            say(f"   {b}: {len(vf)} very-forward n, nbar, gamma kept ({sum(c.p4[j, 0] for j in vf):.1f} GeV); "
                f"{len(pp)} protons to the Roman pots ({sum(c.p4[j, 0] for j in pp):.1f} GeV" + (f", |eta| {min(abs(c.eta[j]) for j in pp):.2f} to {max(abs(c.eta[j]) for j in pp):.2f}" if pp else "") + ")")
            if b != "oot":
                for j in pp[:30]:
                    say(f"      p k={c.k[j]:4d} E {c.p4[j, 0]:8.1f} xi {1 - c.p[j] / E_BEAM:.4f} eta {c.eta[j]:+7.2f}", level=1)
        say("")
        say("-- 8. the b systems and our jet as Geant4 receives them")
        for key in ("b", "bbar"):
            b = R["b"].get(key)
            if b:
                say(f"   {key}: {b['first']} -> {b['hadron']} pT {b['pt_hadron']:.2f} eta {b['eta_hadron']:+.2f}: {b['n']} stable descendants, pT {b['pt_before']:.2f} -> "
                    f"{b['n_primary']} primaries, pT {b['pt_after']:.2f}" + ("; not handed over: " + ", ".join(f"{x['name']} ({x['code']}, pT {x['pt']:.2f})" for x in b["lost"]) if b["lost"] else ""))
        for coll, label in (("ak4", "the photograph, ak4 jet [0]"), ("ak4nonu", "ak4 NoNu jet [0]")):
            j = R["jets"].get(coll)
            if j:
                say(f"   {label}: pT {j['pt']:.2f}, m {j['m']:.2f}, {j['n']} constituents -> handed over {j['n_after']}: pT {j['pt_after']:.2f}, m {j['m_after']:.2f}"
                    + ("; not handed over: " + ", ".join(f"{x['name']} ({x['code']}, pT {x['pt']:.3f})" for x in j["lost"]) if j["lost"] else ""))
        mj = R["jets"].get("mixed")
        if mj:
            say(f"   with the crowd, mixed jet [{mj['rank']}]: pT {mj['pt']:.2f}, {mj['n']} constituents -> handed over {mj['n_after']}: pT {mj['pt_after']:.2f}, m {mj['m_after']:.2f}")
        say("")
        say("-- 9. the ledger: stable before -> primaries after (n / E / pT), and the drops")
        for b, label in (("signal", "signal"), ("intime", "in-time crowd"), ("oot", "out-of-time crowd"), ("all", "signal + in-time crowd")):
            if b == "oot" and not self.n_oot:
                continue
            Lb = L[b]
            say(f"   {label}: {Lb['stable']['n']} / {Lb['stable']['e']:.1f} / {Lb['stable']['pt']:.1f} -> {Lb['primary']['n']} / {Lb['primary']['e']:.1f} / {Lb['primary']['pt']:.1f}"
                f" (+ PPS {Lb['pps']['n']} / {Lb['pps']['e']:.1f}); dropped " + ", ".join(f"{r} {Lb['drops'][r]['n']} ({Lb['drops'][r]['e']:.1f})" for r in REASONS))
            for name, _, _ in REGIONS:
                x = Lb["regions"][name]
                say(f"      {name:<10s} {x['stable']['n']:6d} / {x['stable']['e']:11.1f} / {x['stable']['pt']:8.1f}  -> {x['primary']['n']:6d} / {x['primary']['e']:11.1f} / {x['primary']['pt']:8.1f}"
                    + (f"  PPS {x['pps']['n']}" if x["pps"]["n"] else ""))
            for k in KINDS:
                x = Lb["kinds"][k]
                if x["stable"]["n"]:
                    say(f"      {k:<15s} {x['stable']['n']:6d} / {x['stable']['e']:11.1f} / {x['stable']['pt']:8.1f}  -> {x['primary']['n']:6d} / {x['primary']['e']:11.1f} / {x['primary']['pt']:8.1f}")
        if self.verbose >= 2:
            say("", 2)
            say("-- every stable particle of the signal that does not become a primary", 2)
            for j in np.where((c.block == 0) & ~g["primary"])[0]:
                say(f"   row {c.row[j]:5d} {dec.name(int(c.ids[j])):<10s} E {c.p4[j, 0]:9.2f} pT {c.pt[j]:7.3f} eta {c.eta[j]:+8.3f} z {c.vtx[j, 2]:+6.2f}  {g['code'][j]} ({g['reason'][j]})", 2)
            say("-- the in-time collisions", 2)
            for p in R["per"]:
                if p["bx"] == 0 and p["k"] > 0:
                    say(f"   k {p['k']:4d} lib {p['lib']:5d} {p['kind']:<3s} z {p['z']:+6.2f}: {p['stable']['n']:4d} stable -> {p['primary']['n']:4d} primaries; "
                        f"{p['primary']['e']:8.1f} of {p['stable']['e']:.0f} GeV; PPS {p['pps']['n']}; drops " + "/".join(str(p['drops'][r]) for r in REASONS), 2)
        s = L["signal"]["fractions"]
        p = L["intime"]["fractions"]
        say("")
        say(f"== the number to take away: Geant4 receives {100 * s['n']:.0f} % of the family's particles, {100 * s['pt']:.0f} % of its pT, "
            f"but {100 * s['e']:.0f} % of its energy; of the crowd, {100 * p['e']:.0f} % of the energy")

    # ---------------------------------------------------------------- the record
    def record_tables(self, R: dict) -> str:
        c, g, G = self.crowd, self.g, self.gate
        L = R["L"]
        tag = (f"RDecLenCut = {G.r} cm, eta cut {G.eta_cut} (Z_lmax = {G.z_lmax:.1f} cm), EtaCutForHector {G.eta_hector} (Z_hector = {G.z_hector / 100:.1f} m), "
               f"p > {G.p_min * 1000:.0f} MeV; TrackNeutrino = False; PPS |eta| > {PPS_ETA}, |pz| > {PPS_PZ:.0f} GeV")
        out = []

        def section(title, lines):
            head = f" --------  Standalone Hand-over {title}  ----------"
            out.append(head)
            out.append(" ")
            out.extend(lines)
            out.append(head.replace("--------  Standalone", "--------  End Standalone"))
            out.append("")

        lines = [f"  the gate (Generator.cc of CMSSW_15_0_5): {tag}"]
        for b, label in (("signal", "signal"), ("intime", f"in-time pileup ({self.n_intime} collisions)"), ("oot", f"out-of-time pileup ({self.n_oot} collisions)"),
                         ("all", "signal + in-time pileup")):
            if b == "oot" and not self.n_oot:
                continue
            x = L[b]
            lines.append(f"  {label}: stable {x['stable']['n']} ({x['stable']['e']:.1f} GeV, pT {x['stable']['pt']:.1f}) -> primaries {x['primary']['n']} "
                         f"({x['primary']['e']:.1f} GeV, pT {x['primary']['pt']:.1f}); PPS {x['pps']['n']} ({x['pps']['e']:.1f} GeV); dropped "
                         + ", ".join(f"{r} {x['drops'][r]['n']} ({x['drops'][r]['e']:.1f} GeV)" for r in REASONS))
        Rd = R["decays"]
        lines.append(f"  decays by the generator in the signal: {Rd['n']} ({Rd['n_weak']} weak), {Rd['n_outside']} outside r = {G.r} cm; the farthest at r = {Rd['r_max']:.3f} cm from the beam line through the vertex")
        section("Summary", lines)
        lines = ["     blk     k    row        id  name             E [GeV]   pT [GeV]       eta    z_vtx [cm]  r_line [cm]  code  reason"]
        for j in np.where(c.block < 2)[0]:
            lines.append(f"  {BLOCKS[c.block[j]]:>6s} {c.k[j]:5d} {c.row[j]:6d} {c.ids[j]:9d}  {dec.name(int(c.ids[j])):<12s} {c.p4[j, 0]:11.4f} {c.pt[j]:10.4f} {c.eta[j]:+9.4f} "
                         f"{c.vtx[j, 2]:+13.4f} {g['r_line'][j]:12.4f}     {g['code'][j]}  {g['reason'][j] or '-'}")
        section("Listing  (the signal and the in-time crowd: every stable particle; code P primary, V very-forward primary, N neutrino killed, "
                "E eta cut (the line), F very forward not n/nbar/gamma, M momentum cut, R proton to the Roman pots; reason as g4Ledger.py)", lines)
        lines = ["     k   bx    lib  proc   z [cm]  stable   E [GeV]  pT [GeV]  primary   E [GeV]  pT [GeV]   pps   eta     p    nu  other"]
        for p in R["per"]:
            lines.append(f"  {p['k']:4d} {p['bx']:4d} {p['lib']:6d}  {p['kind']:>4s} {p['z']:+8.3f} {p['stable']['n']:7d} {p['stable']['e']:9.1f} {p['stable']['pt']:9.2f} "
                         f"{p['primary']['n']:8d} {p['primary']['e']:9.1f} {p['primary']['pt']:9.2f} {p['pps']['n']:5d} " + " ".join(f"{p['drops'][r]:5d}" for r in REASONS))
        section("Collision Listing  (one row per collision of the window, k = 0 the signal: stable particles before, primaries after, PPS protons, drops by reason)", lines)
        lines = ["    row        id  name              E [GeV]   pT [GeV]      eta     bg   ctau [um]  flight [cm]   r_dec [cm]   z_dec [cm]  inside"]
        for d in self.decays.items:
            lines.append(f"  {d['row']:5d} {d['id']:9d}  {d['name']:<14s} {d['E']:10.3f} {d['pt']:10.3f} {d['eta']:+8.3f} {d['betagamma']:7.1f} {d['ctau'] * 1e4:10.3g} "
                         f"{d['flight']:12.5f} {d['r_dec']:12.5f} {d['z_dec']:+12.5f}  {'yes' if d['inside'] else 'no'}")
        section(f"Decay Listing  (the decays the generator made in the signal, HepMC status 2: r_dec from the beam line, inside = r_dec < {G.r} cm)", lines)
        return "\n".join(out) + "\n"

    def summary_json(self, R: dict) -> dict:
        c, g = self.crowd, self.g
        L = R["L"]
        lva = R["line_vs_angle"]

        def plist(idx):
            return [dict(block=BLOCKS[c.block[j]], k=int(c.k[j]), id=int(c.ids[j]), name=dec.name(int(c.ids[j])), E=float(c.p4[j, 0]), pt=float(c.pt[j]),
                         eta=float(c.eta[j]), z=float(c.vtx[j, 2]), r_line=float(g["r_line"][j]), code=str(g["code"][j])) for j in idx]

        sig = c.block == 0
        border = np.where(sig & (np.abs(c.eta) > 5.3) & (np.abs(c.eta) < 5.7))[0]
        return dict(
            gate=dict(r_dec=self.gate.r, eta_cut=self.gate.eta_cut, z_lmax=self.gate.z_lmax, eta_hector=self.gate.eta_hector, z_hector=self.gate.z_hector,
                      p_min=self.gate.p_min, pps_eta=PPS_ETA, pps_pz=PPS_PZ, eta_hf=[ETA_HF_FRONT, ETA_HF_BACK]),
            n_intime=self.n_intime, n_oot=self.n_oot, pv=self.pv.tolist(), ledger=L,
            per_collision=[dict(k=p["k"], bx=p["bx"], lib=p["lib"], kind=p["kind"], z=p["z"], n_stable=p["stable"]["n"], n_primary=p["primary"]["n"],
                                e_stable=p["stable"]["e"], e_primary=p["primary"]["e"], pt_primary=p["primary"]["pt"], n_pps=p["pps"]["n"], drops=p["drops"]) for p in R["per"]],
            b=R["b"], net=R["net"], jets=R["jets"],
            soft=plist(np.where(sig & g["p_fail"])[0]), neutrinos=plist(np.where(sig & c.nu)[0]), border=plist(border),
            kept_beyond=plist(lva["kept_beyond"]), dropped_inside=plist(lva["dropped_inside"]),
            very_forward=plist(R["vf"][c.block[R["vf"]] < 2]) if len(R["vf"]) else [], pps=plist(R["pps"][c.block[R["pps"]] < 2]) if len(R["pps"]) else [],
            decays=dict(R["decays"], far=({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in R["decays"]["far"].items()} if R["decays"]["far"] else None),
                        weak=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()} for d in self.decays.weak()]))


# ---------------------------------------------------------------------------
# the library scan: the gate on all 2000 collisions of the library
# ---------------------------------------------------------------------------
def library_scan(library: Path, seed: int, gate: Gate) -> dict:
    lib = pu.Library(library)
    rng = np.random.default_rng(seed)
    vtx = pu.BeamSpot().draw(rng, len(lib.events))                         # x, y [um], z [cm], t [ns]
    blk, kk, rows, ids, p4, vv = [], [], [], [], [], []
    for n, e in enumerate(lib.events):
        m = len(e["ids"])
        kk += [n] * m; ids += list(e["ids"]); p4 += list(e["p4"]); vv += [[vtx[n, 0] * 1e-4, vtx[n, 1] * 1e-4, vtx[n, 2]]] * m
    c = Crowd([1] * len(ids), kk, list(range(len(ids))), ids, p4, vv)
    g = gate.decide(c)
    per = dict(n=[], prim=[], e=[], pt=[], pps=[], kind=[])
    kidx = np.asarray(c.k)
    for n, e in enumerate(lib.events):
        m = kidx == n
        per["n"].append(int(m.sum())); per["prim"].append(int((m & g["primary"]).sum()))
        per["e"].append(float(c.p4[m & g["primary"], 0].sum())); per["pt"].append(float(c.pt[m & g["primary"]].sum()))
        per["pps"].append(int((m & g["pps"]).sum())); per["kind"].append(e["kind"])
    L = ledger(c, g, np.ones(len(c), bool))
    arr = {k: np.array(v) for k, v in per.items() if k != "kind"}
    kinds = np.array(per["kind"])
    by_kind = {k: dict(n=int((kinds == k).sum()), prim=float(arr["prim"][kinds == k].mean()), e=float(arr["e"][kinds == k].mean()),
                       pps=float(arr["pps"][kinds == k].mean())) for k in sorted(set(per["kind"]))}
    return dict(n_collisions=len(lib.events), seed=seed, ledger=L, mean=dict(n=float(arr["n"].mean()), prim=float(arr["prim"].mean()), e=float(arr["e"].mean()),
                pt=float(arr["pt"].mean()), pps=float(arr["pps"].mean())), median_prim=float(np.median(arr["prim"])), zero_prim=int((arr["prim"] == 0).sum()),
                max_prim=int(arr["prim"].max()), by_kind=by_kind, per=dict(prim=per["prim"], e=per["e"], pps=per["pps"], kind=per["kind"]))


# ---------------------------------------------------------------------------
# the LaTeX ledger of the hand-over (the layout of the CMSSW companion's table)
# ---------------------------------------------------------------------------
def thou(x: float, nd: int = 0) -> str:
    s = f"{x:,.{nd}f}"
    return s.replace(",", "\\,") if abs(x) >= 9999.5 else s.replace(",", "")


def latex(L: dict, n_intime: int, n_oot: int) -> str:
    blocks = [L["signal"], L["intime"], L["oot"]]

    def cells(x):
        return f"{thou(x['stable']['n'])} & {thou(x['stable']['pt'], 1)} & {thou(x['primary']['n'])} & {thou(x['primary']['pt'], 1)}"

    rows = []
    names = {"|eta|<2.5": "$|\\eta|<2.5$", "2.5-3.0": "$2.5<|\\eta|<3.0$", "3.0-5.2": "$3.0<|\\eta|<5.2$", "5.2-5.5": "$5.2<|\\eta|<5.5$", ">5.5": "$|\\eta|>5.5$"}
    for name, _, _ in REGIONS:
        rows.append(f"{names[name]} & " + " & & ".join(cells(B["regions"][name]) for B in blocks) + " \\\\")
    rows.append("\\midrule")
    kinds = [("charged hadrons", ["charged hadron"]), ("neutral hadrons", ["neutral hadron"]), ("photons", ["photon"]),
             ("electrons, muons", ["electron", "muon"]), ("neutrinos", ["neutrino"])]

    def merged(B, ks):
        out = {}
        for part in ("stable", "primary"):
            out[part] = dict(n=sum(B["kinds"][k][part]["n"] for k in ks), pt=sum(B["kinds"][k][part]["pt"] for k in ks))
        return out

    for label, ks in kinds:
        rows.append(f"{label} & " + " & & ".join(cells(merged(B, ks)) for B in blocks) + " \\\\")
    rows.append("\\midrule")
    rows.append("all & " + " & & ".join(cells(B) for B in blocks) + " \\\\")
    rows.append("\\multicolumn{5}{l}{\\quad and at the Roman pots (PPS): protons} & & & & "
                f"{L['intime']['pps']['n']} & {thou(L['intime']['pps']['pt'], 1)} & & & & {L['oot']['pps']['n']} & {thou(L['oot']['pps']['pt'], 1)} \\\\")
    top = ("\\begin{tabular}{lrrrrcrrrrcrrrr}\\toprule\n"
           f" & \\multicolumn{{4}}{{c}}{{signal}} & & \\multicolumn{{4}}{{c}}{{in-time pileup ({n_intime} collisions)}} & & \\multicolumn{{4}}{{c}}{{out-of-time pileup ({n_oot} collisions)}} \\\\\n"
           "\\cmidrule(lr){2-5}\\cmidrule(lr){7-10}\\cmidrule(lr){12-15}\n"
           " & \\multicolumn{2}{c}{before: stable} & \\multicolumn{2}{c}{after: primaries} & & \\multicolumn{2}{c}{before: stable} & \\multicolumn{2}{c}{after: primaries} & & "
           "\\multicolumn{2}{c}{before: stable} & \\multicolumn{2}{c}{after: primaries} \\\\\n"
           " & $n$ & \\pt{} & $n$ & \\pt{} & & $n$ & \\pt{} & $n$ & \\pt{} & & $n$ & \\pt{} & $n$ & \\pt{} \\\\ \\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")

    def drops(B):
        return ", ".join(f"{thou(B['drops'][r]['n'])} {lab} ({thou(B['drops'][r]['e'], 1)})" for r, lab in
                         (("eta", "beyond $|\\eta|=5.5$"), ("p", "below $40\\MeV$"), ("nu", "neutrinos"), ("other", "other")))

    def frac(B):
        f = B["fractions"]
        return f"{100 * f['n']:.1f}\\,\\% of $n$, {100 * f['e']:.1f}\\,\\% of $E$, {100 * f['pt']:.1f}\\,\\% of \\pt{{}}"

    A = L["all"]
    bottom = ("\\begin{tabular}{ll}\\toprule\n"
              f"signal & handed over: {frac(L['signal'])}; \\\\\n & dropped ($E$ in GeV): {drops(L['signal'])} \\\\\n"
              f"in-time pileup & at the vertex: {frac(L['intime'])}; at the Roman pots: {L['intime']['pps']['n']} protons ({thou(L['intime']['pps']['e'], 0)}); \\\\\n"
              f" & dropped: {drops(L['intime'])} \\\\\n"
              f"out-of-time pileup & at the vertex: {frac(L['oot'])}; at the Roman pots: {L['oot']['pps']['n']} protons ({thou(L['oot']['pps']['e'], 0)}); \\\\\n"
              f" & dropped: {drops(L['oot'])} \\\\\n"
              f"signal + in-time pileup & before: {thou(A['stable']['n'])} stable particles, {thou(A['stable']['e'], 0)}\\GeV, {thou(A['stable']['pt'], 1)}\\GeV of \\pt{{}}; \\\\\n"
              f" & after: {thou(A['primary']['n'])} primaries, {thou(A['primary']['e'], 0)}\\GeV, {thou(A['primary']['pt'], 1)}\\GeV of \\pt{{}} ({frac(A)}), "
              f"and {A['pps']['n']} PPS protons \\\\\n"
              "\\bottomrule\\end{tabular}")
    return ("% written by standalone_G4.py from event_PU.lhe and the library: the ledger of the hand-over (do not edit)\n"
            f"\\newcommand{{\\gfourLedgerTable}}{{%\n{top}}}\n"
            f"\\newcommand{{\\gfourLedgerDrops}}{{%\n{bottom}}}\n")


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=HERE.parent.parent / "PU" / "event_PU.lhe", help="the record of chapter 8")
    ap.add_argument("-l", "--library", type=Path, default=HERE.parent.parent / "PU" / "pileup_library.lhe.gz", help="the library the crowd was drawn from")
    ap.add_argument("--no-oot", action="store_true", help="leave the out-of-time collisions out (no library needed)")
    ap.add_argument("-o", "--output", type=Path, default=HERE.parent / "event_G4.lhe", help="the record of this stop")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    ap.add_argument("--log", type=Path, help="write the narration here instead of the terminal")
    ap.add_argument("--summary", type=Path, help="write the numbers as JSON")
    ap.add_argument("--tables", type=Path, help="write the LaTeX ledger (OUTPUT.tex) and its text twin (OUTPUT.txt)")
    ap.add_argument("--library-scan", type=Path, nargs="?", const=HERE / "standalone_G4_library.json", help="the gate on the 2000 library collisions")
    ap.add_argument("--seed", type=int, default=9, help="the seed of the beam-spot vertices of the library scan")
    ap.add_argument("--eta-cut", type=float, default=ETA_CUT, help="MaxEtaCut = -MinEtaCut")
    ap.add_argument("--eta-hector", type=float, default=ETA_HECTOR, help="EtaCutForHector")
    ap.add_argument("--r-dec", type=float, default=R_DEC, help="RDecLenCut [cm]")
    ap.add_argument("--p-min", type=float, default=P_MIN, help="MinPCut [GeV]")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.no_oot:
        args.library = None
    gate = Gate(eta_cut=args.eta_cut, eta_hector=args.eta_hector, r_dec=args.r_dec, p_min=args.p_min)
    if args.library_scan:
        t0 = time.time()
        S = library_scan(HERE.parent.parent / "PU" / "pileup_library.lhe.gz", args.seed, gate)
        args.library_scan.write_text(json.dumps(to_jsonable(S), indent=1))
        m, Lx = S["mean"], S["ledger"]
        print(f"library scan: {S['n_collisions']} collisions, vertices from the 2025 beam spot (seed {args.seed}), {time.time() - t0:.1f} s")
        print(f"  per collision: {m['n']:.1f} stable -> {m['prim']:.1f} primaries (median {S['median_prim']:.0f}, {S['zero_prim']} with none, max {S['max_prim']}), "
              f"{m['e']:.0f} GeV, pT {m['pt']:.1f} GeV; {m['pps']:.3f} PPS protons")
        print(f"  handed over: {100 * Lx['fractions']['n']:.1f} % of n, {100 * Lx['fractions']['e']:.1f} % of E, {100 * Lx['fractions']['pt']:.1f} % of pT")
        for k, v in S["by_kind"].items():
            print(f"  {k}: {v['n']} collisions, {v['prim']:.1f} primaries, {v['e']:.0f} GeV, {v['pps']:.3f} PPS protons each")
        return
    log = open(args.log, "w") if args.log else None
    t0 = time.time()
    st = Stage(args, verbose=args.verbose, log=log)
    R = st.compute()
    st.narrate(R)
    text = args.input.read_text()
    if not text.endswith("\n"):
        text += "\n"
    args.output.write_text(text + "\n" + st.record_tables(R))
    if args.summary:
        args.summary.write_text(json.dumps(to_jsonable(st.summary_json(R)), indent=1))
    if args.tables:
        args.tables.with_suffix(".tex").write_text(latex(R["L"], st.n_intime, st.n_oot))
    if log:
        log.close()
    if not args.quiet:
        L = R["L"]
        print(f"standalone_G4: {args.output} written in {time.time() - t0:.1f} s")
        for b in ("signal", "intime", "oot", "all"):
            if b == "oot" and not st.n_oot:
                continue
            x = L[b]
            print(f"  {b:<7s}: {x['stable']['n']:7d} stable -> {x['primary']['n']:7d} primaries ({100 * x['fractions']['e']:.1f} % of E, {100 * x['fractions']['pt']:.1f} % of pT), "
                  f"PPS {x['pps']['n']}; drops " + ", ".join(f"{r} {x['drops'][r]['n']}" for r in REASONS))


if __name__ == "__main__":
    main()
