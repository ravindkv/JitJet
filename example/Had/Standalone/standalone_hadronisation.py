#!/usr/bin/env python3
"""Standalone PYTHIA-like hadronisation (the Lund string model) for the parton-level
b bbar event of chapter 4.

Dependencies: numpy, plus the event-record classes of chapter 2's
../../PS/Standalone/standalone_parton_showering.py (imported, same repository).
No PYTHIA: the hadron table (names, masses, charges) is embedded below, dumped
once from PYTHIA 8's ParticleData by dump_hadron_tables.py.

Input is the parton-level record after multiparton interactions, beam remnants
and colour reconnection (example/MPI/event_MPI.lhe: the final partons are the
status-62 rows of every system plus the status-63 beam remnants, and every
colour tag appears once as colour and once as anticolour). Output is the same
table format with the primary hadrons appended (example/Had/event_Had.lhe),
with PYTHIA's status codes: -71 for the copies of the partons collected into a
colour-singlet string, -73 for very nearby partons joined into one, -72 for a
string that absorbed the recoil of a collapsing ministring, 81/82 for the one
or two hadrons of a ministring, and 83/84 for the hadrons of a normal string,
produced from its colour end (83) or its anticolour end (84). Hadron decays
are chapter 6; here every hadron is a primary one, as in the CMSSW figure of
../CMSSW/genPartAnalyzer_ME_PS_MPI_Had_out.pdf (page 4, "primary hadrons").

What is implemented (the physics of chapter 5)
-----------------------------------------------
* Colour-singlet strings. The colour tags of the record are followed from
  every quark or antidiquark through the gluons to an antiquark or diquark
  (open strings) and the remaining gluons close into loops. A string whose
  invariant mass is below m_q1 + m_q2 + mStringMin = 1 GeV cannot fragment and
  is treated as a ministring (one or two hadrons). Two colour-connected
  partons closer than mJoin = 0.3 GeV (gluons counted with half their
  momentum) are joined, as in PYTHIA.
* The Lund string (Andersson, Gustafson, Ingelman, Sjostrand, Phys. Rept. 97
  (1983) 31). A q qbar pair with the string tension kappa = 1 GeV/fm between
  them: the string breaks by q' qbar' pair creation, the hadrons are produced
  iteratively from both ends, each taking a fraction z of the remaining
  light-cone momentum of its end with the Lund symmetric fragmentation
  function
        f(z) = (1/z) (1-z)^a exp(-b mT^2 / z),   mT^2 = m_had^2 + pT^2,
  a = 0.68, b = 0.98 GeV^-2 (the Monash values used by CP5), and the two
  chains are joined in the middle when the remaining mass falls below
  W_min = m_q1 + m_q2 + stopMass + stopNewFlav m_qlast, smeared by +-20%.
* Flavour and pT at the break-up: the tunnelling picture gives a Gaussian
  transverse momentum of width sigma = 0.335 GeV for the pair and the
  suppression of heavy flavours, s : u : d = 0.217 : 1 : 1, diquarks (baryons)
  with probQQtoQ = 0.081 and the strange-diquark and spin-1 suppressions; the
  hadron multiplet is chosen with the vector/pseudoscalar ratios of the tune
  (0.5 for u, d; 0.55 for s; 0.88 for c; 2.2 for b), the eta/eta' mixing and
  suppression, and the SU(6) spin-flavour weights for octet and decuplet
  baryons.
* Gluons as kinks. A string q g1 g2 ... qbar is a sequence of string pieces
  (regions) spanned by the light-cone momenta of neighbouring partons, each
  gluon carrying half its momentum into the region on either side; the
  hadrons from one end eat through the regions one after the other, with the
  momentum bookkeeping of Sjostrand, Nucl. Phys. B248 (1984) 469, so that a
  hadron produced across a kink takes momentum from both sides.
* Heavy quarks. The b quark (and c) at a string end fragments with the
  Lund-Bowler function
        f(z) = (1/z^(1 + r_b b m_b^2)) (1-z)^a exp(-b mT^2 / z),  r_b = 0.855,
  which is what puts the B hadron at x_B ~ 0.8 instead of the light-quark
  0.3: this is the one number of this chapter that the detector will see as
  the b-jet response. The Peterson function (epsilon_b = 0.005) is available
  with --peterson for the comparison in the chapter.

What is left out, on purpose: popcorn baryon production (a baryon and its
antibaryon are always neighbours here), junction strings (the chapter-4
record has none), the thermal pT model, the closely-related rope and
close-packing effects, hadron decays (chapter 6), and the exact details of
PYTHIA's final-two joining (here the last two hadrons share the remaining
momentum with the pT of the final break and the direction of the positive
end, the orientation chosen with the area law).

How close is it to PYTHIA? validate_hadronisation_with_pythia.py lets
PYTHIA 8.315 hadronise exactly the same parton configuration (the strings of
example/MPI/event_MPI.lhe) many times and also runs the full chain from the
hard event; the hadron multiplicities, the species, the x_B of the B hadrons
and the hadron-level jet pT are compared in README.md.

Learning mode (--verbose / -v, -vv)
------------------------------------
  -v   narrates everything for the string that contains the b quark (and
       one line per hadron for every other string): the string's partons,
       its mass and the regions, then for every hadron the end it comes
       from, the flavour draw with its weights, the hadron and its mass,
       the pT kick, the fragmentation function and the z drawn, the
       regions stepped through, the hadron momentum, the proper time of
       the break-up vertex, and the remaining mass against W_min; then the
       final two hadrons, the ministrings and the conservation checks.
  -vv  narrates every string in full.
  The random numbers are the same with and without -v.

Examples
--------
  python3 standalone_hadronisation.py                      # -> ../event_Had.lhe (the book's record)
  python3 standalone_hadronisation.py -v                   # learning mode
  python3 standalone_hadronisation.py --peterson           # Peterson instead of Lund-Bowler for b and c
  python3 standalone_hadronisation.py --rfactb 0 --seed 7  # plain Lund for the b quark
  python3 standalone_hadronisation.py --repeat 400 --quiet --history stats.json

All energies in GeV.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "PS" / "Standalone"))
import standalone_parton_showering as ps  # noqa: E402  (chapter 2: Particle, Event, frames, listing helpers)

Event, Particle = ps.Event, ps.Particle
pt, eta, phi, mass2, dot, fmt_p, banner = ps.pt, ps.eta, ps.phi, ps.mass2, ps.dot, ps.fmt_p, ps.banner
boost_matrix, rotation_to_z, to_cm = ps.boost_matrix, ps.rotation_to_z, ps.to_cm

# BEGIN GENERATED TABLES
# produced by dump_hadron_tables.py from PYTHIA 8's ParticleData; do not edit by hand
TABLES = {"source":"PYTHIA 8 ParticleData","hadrons":{"111":["pi0","pi0",0.13498,0],"113":["rho0","rho0",0.77549,0],"211":["pi+","pi-",0.13957,3],"213":["rho+","rho-",0.77549,3],"221":["eta","eta",0.54785,0],"223":["omega","omega",0.78265,0],"311":["K0","Kbar0",0.49761,0],"313":["K*0","K*bar0",0.89594,0],"321":["K+","K-",0.49368,3],"323":["K*+","K*-",0.89166,3],"331":["eta'","eta'",0.95778,0],"333":["phi","phi",1.01946,0],"411":["D+","D-",1.86962,3],"413":["D*+","D*-",2.01028,3],"421":["D0","Dbar0",1.86486,0],"423":["D*0","D*bar0",2.00698,0],"431":["D_s+","D_s-",1.96849,3],"433":["D*_s+","D*_s-",2.1123,3],"441":["eta_c","eta_c",2.981,0],"443":["J/psi","J/psi",3.09692,0],"511":["B0","Bbar0",5.27958,0],"513":["B*0","B*bar0",5.3252,0],"521":["B+","B-",5.27925,3],"523":["B*+","B*-",5.3252,3],"531":["B_s0","B_sbar0",5.36677,0],"533":["B*_s0","B*_sbar0",5.4154,0],"541":["B_c+","B_c-",6.277,3],"543":["B*_c+","B*_c-",6.34,3],"551":["eta_b","eta_b",9.4,0],"553":["Upsilon","Upsilon",9.4603,0],"1114":["Delta-","Deltabar+",1.232,-3],"2112":["n0","nbar0",0.93957,0],"2114":["Delta0","Deltabar0",1.232,0],"2212":["p+","pbar-",0.93827,3],"2214":["Delta+","Deltabar-",1.232,3],"2224":["Delta++","Deltabar--",1.232,6],"3112":["Sigma-","Sigmabar+",1.19745,-3],"3114":["Sigma*-","Sigma*bar+",1.3872,-3],"3122":["Lambda0","Lambdabar0",1.11568,0],"3212":["Sigma0","Sigmabar0",1.19264,0],"3214":["Sigma*0","Sigma*bar0",1.3837,0],"3222":["Sigma+","Sigmabar-",1.18937,3],"3224":["Sigma*+","Sigma*bar-",1.3828,3],"3312":["Xi-","Xibar+",1.32171,-3],"3314":["Xi*-","Xi*bar+",1.535,-3],"3322":["Xi0","Xibar0",1.31486,0],"3324":["Xi*0","Xi*bar0",1.5318,0],"3334":["Omega-","Omegabar+",1.67245,-3],"4112":["Sigma_c0","Sigma_cbar0",2.45374,0],"4114":["Sigma*_c0","Sigma*_cbar0",2.5188,0],"4122":["Lambda_c+","Lambda_cbar-",2.28646,3],"4132":["Xi_c0","Xi_cbar0",2.47088,0],"4212":["Sigma_c+","Sigma_cbar-",2.4529,3],"4214":["Sigma*_c+","Sigma*_cbar-",2.5175,3],"4222":["Sigma_c++","Sigma_cbar--",2.45398,6],"4224":["Sigma*_c++","Sigma*_cbar--",2.5179,6],"4232":["Xi_c+","Xi_cbar-",2.4678,3],"4312":["Xi'_c0","Xi'_cbar0",2.5779,0],"4314":["Xi*_c0","Xi*_cbar0",2.6459,0],"4322":["Xi'_c+","Xi'_cbar-",2.5756,3],"4324":["Xi*_c+","Xi*_cbar-",2.6459,3],"4332":["Omega_c0","Omega_cbar0",2.6952,0],"4334":["Omega*_c0","Omega*_cbar0",2.7659,0],"5112":["Sigma_b-","Sigma_bbar+",5.8155,-3],"5114":["Sigma*_b-","Sigma*_bbar+",5.8351,-3],"5122":["Lambda_b0","Lambda_bbar0",5.6194,0],"5132":["Xi_b-","Xi_bbar+",5.7911,-3],"5212":["Sigma_b0","Sigma_bbar0",5.8,0],"5214":["Sigma*_b0","Sigma*_bbar0",5.81,0],"5222":["Sigma_b+","Sigma_bbar-",5.8113,3],"5224":["Sigma*_b+","Sigma*_bbar-",5.8321,3],"5232":["Xi_b0","Xi_bbar0",5.788,0],"5312":["Xi'_b-","Xi'_bbar+",5.96,-3],"5314":["Xi*_b-","Xi*_bbar+",5.97,-3],"5322":["Xi'_b0","Xi'_bbar0",5.96,0],"5324":["Xi*_b0","Xi*_bbar0",5.97,0],"5332":["Omega_b-","Omega_bbar+",6.07,-3],"5334":["Omega*_b-","Omega*_bbar+",6.13,-3]},"quarks":{"1":0.33,"2":0.33,"3":0.5,"4":1.5,"5":4.8},"diquarks":{"1103":["dd_1",0.77133],"2101":["ud_0",0.57933],"2103":["ud_1",0.77133],"2203":["uu_1",0.77133],"3101":["sd_0",0.80473],"3103":["sd_1",0.92953],"3201":["su_0",0.80473],"3203":["su_1",0.92953],"3303":["ss_1",1.09361]}}
# END GENERATED TABLES

# ---------------------------------------------------------------------------
# the parameters: Monash 2013 (Tune:ee = 7), which CP5 keeps for the hadronisation
# ---------------------------------------------------------------------------
KAPPA = 1.0                          # string tension [GeV/fm]
A_LUND, B_LUND = 0.68, 0.98          # StringZ:aLund, StringZ:bLund [GeV^-2]
A_EXTRA_DIQUARK, A_EXTRA_SQUARK = 0.97, 0.0
R_FACT_C, R_FACT_B = 1.32, 0.855     # StringZ:rFactC, StringZ:rFactB (Bowler)
EPSILON_C, EPSILON_B = 0.05, 0.005   # StringZ:epsilonC/B (Peterson, optional)
SIGMA_PT = 0.335                     # StringPT:sigma (width of the pair pT; px, py each sigma/sqrt 2)
ENHANCED_FRACTION, ENHANCED_WIDTH = 0.01, 2.0
PROB_S_TO_UD = 0.217                 # StringFlav:probStoUD
PROB_QQ_TO_Q = 0.081                 # StringFlav:probQQtoQ
PROB_SQ_TO_QQ = 0.915                # StringFlav:probSQtoQQ
PROB_QQ1_TO_QQ0 = 0.0275             # StringFlav:probQQ1toQQ0
MESON_VECTOR = {1: 0.5, 2: 0.5, 3: 0.55, 4: 0.88, 5: 2.2}     # StringFlav:mesonUDvector, S, C, B
ETA_SUP, ETA_PRIME_SUP = 0.6, 0.12
THETA_PS, THETA_V = -15.0, 36.0      # mixing angles [degrees]
DECUPLET_SUP = 1.0
STOP_MASS, STOP_NEW_FLAV, STOP_SMEAR = 1.0, 2.0, 0.2    # StringFragmentation:stopMass/stopNewFlav/stopSmear
M_STRING_MIN = 1.0                   # HadronLevel:mStringMin
M_JOIN = 0.3                         # FragmentationSystems:mJoin
NTRY_STRING = 10                     # attempts before a string is given up and fragmented as a ministring
NTRY_HADRON = 20                     # attempts to place one hadron (new flavour, pT, z) before the string is restarted

QUARK_M0 = {1: 0.33, 2: 0.33, 3: 0.5, 4: 1.5, 5: 4.8}      # constituent masses for W_min
DIQUARK_M0 = {1103: 0.77133, 2101: 0.57933, 2103: 0.77133, 2203: 0.77133, 3101: 0.80473, 3103: 0.92953,
              3201: 0.80473, 3203: 0.92953, 3303: 1.09361}
DIQUARK_NAME = {1103: "dd_1", 2101: "ud_0", 2103: "ud_1", 2203: "uu_1", 3101: "sd_0", 3103: "sd_1",
                3201: "su_0", 3203: "su_1", 3303: "ss_1"}
QUARK_CHARGE3 = {1: -1, 2: 2, 3: -1, 4: 2, 5: -1, 6: 2}
QUARK_NAME = {1: "d", 2: "u", 3: "s", 4: "c", 5: "b"}
HBARC = 0.19732698                   # GeV fm


def _load_tables():
    had = {int(k): tuple(v) for k, v in TABLES.get("hadrons", {}).items()}
    q = {int(k): v for k, v in TABLES.get("quarks", {}).items()}
    dq = {int(k): v for k, v in TABLES.get("diquarks", {}).items()}
    return had, q, dq


HADRONS, _Q, _DQ = _load_tables()
if _Q:
    QUARK_M0.update(_Q)
for _k, _v in _DQ.items():
    DIQUARK_NAME[_k], DIQUARK_M0[_k] = _v[0], _v[1]


def is_diquark(pid: int) -> bool:
    a = abs(pid)
    return 1000 < a < 10000 and (a // 10) % 10 == 0


def is_quark(pid: int) -> bool:
    return 0 < abs(pid) < 10


def diquark_flavours(pid: int):
    a = abs(pid)
    return a // 1000, (a // 100) % 10, a % 10      # q1, q2, 2 s + 1


def flav_m0(pid: int) -> float:
    a = abs(pid)
    return DIQUARK_M0[a] if is_diquark(pid) else QUARK_M0[a]


def charge3(pid: int) -> int:
    a = abs(pid)
    if a == 21 or a == 90:
        return 0
    if a < 10:
        c = QUARK_CHARGE3[a]
    elif a == 2212:
        c = 3
    elif is_diquark(pid):
        q1, q2, _ = diquark_flavours(pid)
        c = QUARK_CHARGE3[q1] + QUARK_CHARGE3[q2]
    elif a in HADRONS:
        c = HADRONS[a][3]
    else:
        c = 0
    return c if pid > 0 else -c


def pname(pid: int) -> str:
    a = abs(pid)
    if a in HADRONS:
        return HADRONS[a][0] if pid > 0 else HADRONS[a][1]
    if is_diquark(pid):
        n = DIQUARK_NAME.get(a, str(a))
        return n if pid > 0 else n + "bar"
    if a in QUARK_NAME:
        return QUARK_NAME[a] if pid > 0 else QUARK_NAME[a] + "bar"
    return ps.pname(pid)


def hadron_mass(pid: int) -> float:
    return HADRONS[abs(pid)][2]


def meson_id(q: int, abar: int, spin: int) -> int:
    """PDG id of the meson (quark q, antiquark abar; spin = 1 pseudoscalar, 3 vector), q != abar."""
    hi, lo = max(q, abar), min(q, abar)
    base = 100 * hi + 10 * lo + spin
    heavy_is_quark = (hi == q)
    up_type = hi in (2, 4, 6)
    return base if heavy_is_quark == up_type else -base


def baryon_id(trio, spin: int, lambda_like: bool) -> int:
    """PDG id of the baryon with the three (positive) flavours; spin 2 (octet) or 4 (decuplet);
    lambda_like puts the two lightest quarks in spin 0 (Lambda, Xi_c, Lambda_b ...)."""
    a, b, c = sorted(trio, reverse=True)
    if lambda_like and b != c and a != b:
        return 1000 * a + 100 * c + 10 * b + spin
    return 1000 * a + 100 * b + 10 * c + spin


def diquark_id(q1: int, q2: int, spin1: bool) -> int:
    hi, lo = max(q1, q2), min(q1, q2)
    return 1000 * hi + 100 * lo + (3 if spin1 else 1)


# ---------------------------------------------------------------------------
# flavour selection at a string break (StringFlav)
# ---------------------------------------------------------------------------
class StringFlav:
    def __init__(self, rng, prob_s=PROB_S_TO_UD, prob_qq=PROB_QQ_TO_Q, prob_sq=PROB_SQ_TO_QQ,
                 prob_qq1=PROB_QQ1_TO_QQ0, vector=None, eta_sup=ETA_SUP, etap_sup=ETA_PRIME_SUP,
                 decuplet_sup=DECUPLET_SUP):
        self.rng = rng
        self.prob_s, self.prob_qq, self.prob_sq, self.prob_qq1 = prob_s, prob_qq, prob_sq, prob_qq1
        self.vector = dict(MESON_VECTOR if vector is None else vector)
        self.eta_sup, self.etap_sup, self.decuplet_sup = eta_sup, etap_sup, decuplet_sup
        self.quark_w = np.array([1.0, 1.0, prob_s])               # d, u, s
        self.dq_quark_w = np.array([1.0, 1.0, prob_sq * prob_s])  # d, u, s inside a diquark
        self.spin1_prob = 3 * prob_qq1 / (1 + 3 * prob_qq1)       # spin 1 : spin 0 = 3 probQQ1toQQ0 : 1
        # octet-singlet mixing: probabilities that u ubar (or d dbar) and s sbar project on the two isoscalars
        self.mix = {1: self._mixing(THETA_PS), 3: self._mixing(THETA_V)}
        self.log = []                                            # (kind, text) of the last draw, for the narration

    @staticmethod
    def _mixing(theta_deg):
        alpha = math.radians(90.0 - (theta_deg + 54.7))          # PYTHIA's alpha -> 90 - alpha so that eta <-> eta'
        s2, c2 = math.sin(alpha) ** 2, math.cos(alpha) ** 2
        # [P(first isoscalar | uubar), P(second | uubar), P(first | ssbar), P(second | ssbar)]
        # first = eta / phi-like? no: first = the state with |ssbar|^2 = cos^2 alpha (eta, phi), second = eta', omega
        return dict(uu_first=0.5 * s2, uu_second=0.5 * c2, ss_first=c2, ss_second=s2)

    def pick_quark(self, weights):
        u = self.rng.random() * weights.sum()
        c = np.cumsum(weights)
        return 1 + int(np.searchsorted(c, u))                    # 1 = d, 2 = u, 3 = s

    def pick(self, old: int):
        """The new flavour that goes into the hadron with `old`: an antiquark or a diquark for a quark end,
        a quark or an antidiquark for an antiquark end, a quark for a diquark end (no diquark-antidiquark
        neighbours). Returns the signed id."""
        sign = 1 if old > 0 else -1
        if is_diquark(old):
            q = self.pick_quark(self.quark_w)
            self.log = [("flavour", f"diquark end {pname(old)}: a quark must follow (d : u : s = 1 : 1 : {self.prob_s}) -> {pname(sign * q)}")]
            return sign * q
        r = self.rng.random() * (1 + self.prob_qq)
        if r < 1:
            q = self.pick_quark(self.quark_w)
            self.log = [("flavour", f"quark (P = 1/(1 + probQQtoQ) = {1 / (1 + self.prob_qq):.3f}), d : u : s = 1 : 1 : {self.prob_s} -> {pname(-sign * q)}")]
            return -sign * q
        while True:
            q1 = self.pick_quark(self.dq_quark_w)
            q2 = self.pick_quark(self.dq_quark_w)
            if q1 == q2:
                if self.rng.random() < self.spin1_prob:
                    spin1 = True
                    break
                continue
            spin1 = self.rng.random() < self.spin1_prob
            break
        dq = diquark_id(q1, q2, spin1)
        self.log = [("flavour", f"diquark (P = probQQtoQ/(1 + probQQtoQ) = {self.prob_qq / (1 + self.prob_qq):.3f}); "
                                f"flavours d : u : s = 1 : 1 : {self.dq_quark_w[2]:.3f}, spin 1 with P = 3 probQQ1toQQ0/(1 + 3 probQQ1toQQ0) = "
                                f"{self.spin1_prob:.3f} -> {pname(sign * dq)}")]
        return sign * dq

    def combine(self, f1: int, f2: int) -> int:
        """The hadron made of two flavours, or 0 if the combination is rejected (then draw a new flavour)."""
        if is_quark(f1) and is_quark(f2):
            q, abar = (f1, -f2) if f1 > 0 else (f2, -f1)
            if q <= 0 or abar <= 0:
                return 0
            heavy = max(q, abar)
            pv = self.vector[min(heavy, 5)]
            spin = 3 if self.rng.random() * (1 + pv) < pv else 1
            if q != abar:
                hid = meson_id(q, abar, spin)
                self.log.append(("hadron", f"meson {pname(q)} {pname(-abar)}: vector with P = {pv}/(1 + {pv}) = {pv / (1 + pv):.3f} -> "
                                           f"{'vector' if spin == 3 else 'pseudoscalar'} {pname(hid)} (m = {hadron_mass(hid):.3f})"))
                return hid
            if q >= 4:
                hid = 100 * q + 10 * q + spin
                self.log.append(("hadron", f"{pname(q)} {pname(-q)} -> {pname(hid)}"))
                return hid
            mix = self.mix[spin]
            r = self.rng.random()
            if q == 3:
                hid = (221 if spin == 1 else 333) if r < mix["ss_first"] else (331 if spin == 1 else 223)
            else:
                if r < 0.5:
                    hid = 111 if spin == 1 else 113
                elif r < 0.5 + mix["uu_first"]:
                    hid = 221 if spin == 1 else 333
                else:
                    hid = 331 if spin == 1 else 223
            sup = self.eta_sup if hid == 221 else self.etap_sup if hid == 331 else 1.0
            if sup < 1.0 and self.rng.random() > sup:
                self.log.append(("hadron", f"{pname(q)} {pname(-q)} -> {pname(hid)} rejected (suppression {sup}): new flavour"))
                return 0
            self.log.append(("hadron", f"{pname(q)} {pname(-q)} -> {pname(hid)} (m = {hadron_mass(hid):.3f}; "
                                       f"pi0/rho0 1/2, eta/phi-like {mix['uu_first']:.3f}, eta'/omega-like {mix['uu_second']:.3f} for {pname(q)} {pname(-q)})"
                                       if q != 3 else f"eta/phi-like {mix['ss_first']:.3f}, eta'/omega-like {mix['ss_second']:.3f} for s sbar)"))
            return hid
        # baryon: a quark and a diquark of the same sign
        if is_diquark(f1) and is_quark(f2):
            dq, q = f1, f2
        elif is_quark(f1) and is_diquark(f2):
            q, dq = f1, f2
        else:
            return 0
        if (q > 0) != (dq > 0):
            return 0
        sign = 1 if q > 0 else -1
        q1, q2, s2 = diquark_flavours(dq)
        q3 = abs(q)
        spin1 = s2 == 3
        trio = sorted((q1, q2, q3), reverse=True)
        a, b, c = trio
        pair = sorted((q1, q2))
        lightest_pair = (pair == [c, b])
        w = {}            # (spin, lambda_like) -> weight, PYTHIA's normalisation (max = 1 for Delta++ from uu_1 + u)
        if not spin1:
            if q3 in (q1, q2):
                w[(2, False)] = 0.75
            elif lightest_pair:
                w[(2, True)] = 0.5
            else:
                w[(2, True)], w[(2, False)] = 0.125, 0.375
        else:
            if q1 == q2 == q3:
                w[(4, False)] = 1.0
            elif q1 == q2:
                w[(2, False)], w[(4, False)] = 1 / 6, 1 / 3
            elif q3 in (q1, q2):
                w[(2, False)], w[(4, False)] = 1 / 12, 2 / 3
            elif lightest_pair:
                w[(2, False)], w[(4, False)] = 1 / 6, 1 / 3
            else:
                w[(2, True)], w[(2, False)], w[(4, False)] = 1 / 8, 1 / 24, 1 / 3
        for k in list(w):
            if k[0] == 4:
                w[k] *= self.decuplet_sup
        total = sum(w.values())
        r = self.rng.random()
        if r > total:
            self.log.append(("hadron", f"{pname(dq)} + {pname(q)}: SU(6) weights {self._wtxt(w, trio)} sum to {total:.3f} < r = {r:.3f}: rejected, new flavour"))
            return 0
        acc = 0.0
        for k, v in w.items():
            acc += v
            if r <= acc:
                hid = sign * baryon_id(trio, k[0], k[1])
                self.log.append(("hadron", f"{pname(dq)} + {pname(q)}: SU(6) weights {self._wtxt(w, trio)} -> {pname(hid)} (m = {hadron_mass(hid):.3f})"))
                return hid
        return 0

    @staticmethod
    def _wtxt(w, trio):
        return ", ".join(f"{pname(baryon_id(trio, k[0], k[1]))} {v:.3f}" for k, v in w.items())


# ---------------------------------------------------------------------------
# the longitudinal fraction z (StringZ) and the transverse momentum (StringPT)
# ---------------------------------------------------------------------------
class StringZ:
    def __init__(self, a=A_LUND, b=B_LUND, a_dq=A_EXTRA_DIQUARK, a_s=A_EXTRA_SQUARK, r_c=R_FACT_C, r_b=R_FACT_B,
                 peterson=False, eps_c=EPSILON_C, eps_b=EPSILON_B, n_grid=800):
        self.a, self.b, self.a_dq, self.a_s = a, b, a_dq, a_s
        self.r = {4: r_c, 5: r_b}
        self.peterson, self.eps = peterson, {4: eps_c, 5: eps_b}
        self.z = np.linspace(0.0, 1.0, n_grid + 1)[1:-1] - 0.5 / n_grid     # cell centres in (0, 1)
        self.dz = 1.0 / n_grid
        self.last = {}

    def a_of(self, flav: int) -> float:
        a = self.a
        if is_diquark(flav):
            a += self.a_dq
        elif abs(flav) == 3:
            a += self.a_s
        return a

    def logf(self, flav_old: int, flav_new: int, mt2: float, z=None):
        """ln f(z) of the fragmentation function for the hadron that contains flav_old, with the new flavour flav_new."""
        z = self.z if z is None else np.asarray(z, dtype=float)
        a_old, a_new = self.a_of(flav_old), self.a_of(flav_new)
        heavy = abs(flav_old) if is_quark(flav_old) and abs(flav_old) in (4, 5) else 0
        if heavy and self.peterson:
            eps = self.eps[heavy]
            return -np.log(z) - 2 * np.log(np.abs(1 - 1 / z - eps / (1 - z)))
        lf = (a_old - a_new - 1.0) * np.log(z) + a_new * np.log(1 - z) - self.b * mt2 / z
        if heavy:
            lf -= self.r[heavy] * self.b * QUARK_M0[heavy] ** 2 * np.log(z)
        return lf

    def sample(self, flav_old: int, flav_new: int, mt2: float, rng) -> float:
        lf = self.logf(flav_old, flav_new, mt2)
        f = np.exp(lf - lf.max())
        cdf = np.cumsum(f)
        u = rng.random() * cdf[-1]
        k = int(np.searchsorted(cdf, u))
        z0 = self.z[k] - 0.5 * self.dz
        frac = (u - (cdf[k - 1] if k > 0 else 0.0)) / f[k]
        z = min(max(z0 + frac * self.dz, 1e-6), 1 - 1e-6)
        self.last = dict(z=z, mean=float((self.z * f).sum() / f.sum()), a_old=self.a_of(flav_old), a_new=self.a_of(flav_new),
                         heavy=abs(flav_old) if abs(flav_old) in (4, 5) and is_quark(flav_old) else 0, mt2=mt2)
        return z


class StringPT:
    def __init__(self, sigma=SIGMA_PT, enh_frac=ENHANCED_FRACTION, enh_width=ENHANCED_WIDTH):
        self.sigma, self.enh_frac, self.enh_width = sigma, enh_frac, enh_width

    def sample(self, rng):
        s = self.sigma / math.sqrt(2.0)
        if rng.random() < self.enh_frac:
            s *= self.enh_width
        return s * rng.standard_normal(), s * rng.standard_normal()


# ---------------------------------------------------------------------------
# the string: regions, ends, fragmentation (StringSystem, StringRegion, StringEnd, StringFragmentation)
# ---------------------------------------------------------------------------
G_METRIC = np.diag([1.0, -1.0, -1.0, -1.0])


def lorentz_inverse(M: np.ndarray) -> np.ndarray:
    return G_METRIC @ M.T @ G_METRIC


class Region:
    """The string piece spanned by the massless light-cone momenta p_plus (from the colour end side) and
    p_minus (from the anticolour end side): w2 = (p+ + p-)^2 and the transverse unit vectors eX, eY."""

    def __init__(self, i: int, k: int, p_plus, p_minus):
        self.i, self.k = i, k
        self.p_plus, self.p_minus = p_plus, p_minus
        self.w2 = 2.0 * dot(p_plus, p_minus)
        if self.w2 > 1e-12:
            M = to_cm(p_plus, p_minus)
            Minv = lorentz_inverse(M)
            self.eX, self.eY = Minv @ np.array([0.0, 1.0, 0.0, 0.0]), Minv @ np.array([0.0, 0.0, 1.0, 0.0])
        else:
            self.eX = self.eY = np.zeros(4)


def massless_pair(p1, p2):
    """Massless light-cone vectors a (along p1) and b (along p2) with a + b = p1 + p2 (StringRegion::setUp)."""
    P = p1 + p2
    w2 = mass2(P)
    if w2 <= 1e-12:
        return p1.copy(), p2.copy()
    M = to_cm(p1, p2)
    Minv = lorentz_inverse(M)
    w = math.sqrt(w2)
    a = Minv @ np.array([0.5 * w, 0.0, 0.0, 0.5 * w])
    b = Minv @ np.array([0.5 * w, 0.0, 0.0, -0.5 * w])
    return a, b


class End:
    """One end of the string: its flavour, the pieces it is eating (dir = its own light-cone pieces,
    inv = the pieces of the other side), the pT left from the last break-up and the last vertex."""

    def __init__(self, sign: int, flav: int, i_dir: int, i_inv: int):
        self.sign, self.flav = sign, flav
        self.i_dir, self.i_inv = i_dir, i_inv
        self.px = self.py = 0.0
        self.used_dir = 0.0           # fraction of the current dir piece consumed by this end
        self.used_inv = 0.0           # fraction of the current inv piece consumed by this end
        self.gamma = 0.0              # kappa^2 tau^2 of the last vertex
        self.vertex = None            # space-time point (times kappa) of the last vertex
        self.hadrons = []             # (id, p, info) in rank order from this end
        self.last_new = None

    def region_ik(self):
        return (self.i_dir, self.i_inv) if self.sign > 0 else (self.i_inv, self.i_dir)


class StringSystem:
    def __init__(self, partons, rng, flav: StringFlav, zfun: StringZ, ptfun: StringPT, verbose=0, label="",
                 stop_mass=STOP_MASS, stop_new=STOP_NEW_FLAV, stop_smear=STOP_SMEAR):
        """partons: list of (id, p4, m) in colour order, from a quark or antidiquark to an antiquark or diquark;
        for a closed gluon loop all partons are gluons (the loop is opened at one of them)."""
        self.rng, self.flav, self.zfun, self.ptfun = rng, flav, zfun, ptfun
        self.verbose, self.label = verbose, label
        self.stop_mass, self.stop_new, self.stop_smear = stop_mass, stop_new, stop_smear
        self.closed = all(pid == 21 for pid, _, _ in partons)
        self.partons = list(partons)
        self.p_tot = sum((p for _, p, _ in partons), np.zeros(4))
        self.w = math.sqrt(max(mass2(self.p_tot), 0.0))
        self.opened_at = None
        self.notes = []

    # --- set-up ------------------------------------------------------------------------------------------
    def open_loop(self):
        """A closed gluon loop is opened at one gluon, chosen with weight E: its two halves become a q and a
        qbar end of a light flavour (the first break-up of PYTHIA's closed-loop treatment, simplified)."""
        es = np.array([p[0] for _, p, _ in self.partons])
        j = int(self.rng.choice(len(self.partons), p=es / es.sum()))
        q = self.flav.pick_quark(self.flav.quark_w)
        pid_g, p_g, _ = self.partons[j]
        order = self.partons[j + 1:] + self.partons[:j]
        self.partons = [(q, 0.5 * p_g, 0.0)] + order + [(-q, 0.5 * p_g, 0.0)]
        self.opened_at = (j, q)
        self.notes.append(f"closed loop of {len(order) + 1} gluons opened at gluon {j} (E = {p_g[0]:.2f} GeV) into {pname(q)} {pname(-q)}")

    def setup(self):
        if self.closed:
            self.open_loop()
        n = len(self.partons) - 1
        self.n = n
        ids = [pid for pid, _, _ in self.partons]
        p = [pp for _, pp, _ in self.partons]
        self.p_plus = [None] * n            # p_plus[i], i = 0 .. n-1
        self.p_minus = [None] * (n + 1)     # p_minus[k], k = 1 .. n
        # every piece is massless: the two partons of each adjacent region (a gluon with half its momentum,
        # possibly massive after joining) are split into massless light-cone vectors with the same sum
        for j in range(n):
            pa = p[j] if j == 0 else 0.5 * p[j]
            pb = p[j + 1] if j + 1 == n else 0.5 * p[j + 1]
            self.p_plus[j], self.p_minus[j + 1] = massless_pair(pa, pb)
        self.ids = ids
        self.regions = {}
        self.rem_plus = [1.0] * n
        self.rem_minus = [None] + [1.0] * n
        self.pos = End(+1, ids[0], 0, 1)
        self.neg = End(-1, ids[n], n, n - 1)
        self.hadrons = []
        self.n_fail = 0
        self.w_rem_now = self.w_min_now = self.w

    def region(self, i: int, k: int) -> Region:
        key = (i, k)
        if key not in self.regions:
            self.regions[key] = Region(i, k, self.p_plus[i], self.p_minus[k])
        return self.regions[key]

    def vertex_point(self, i: int, k: int, t_plus: float, t_minus: float):
        """kappa times the space-time point in region (i, k): the origin of the region (the gluons between
        have turned around) plus t_plus p_plus[i] + t_minus p_minus[k]."""
        return self.origin(i, k) + t_plus * self.p_plus[i] + t_minus * self.p_minus[k]

    # --- one hadron from one end -------------------------------------------------------------------------
    def origin(self, i: int, k: int):
        """kappa times the space-time origin of region (i, k): the turning points of the gluons in between."""
        return sum((0.5 * (self.p_plus[j] + self.p_minus[j]) for j in range(i + 1, k)), np.zeros(4))

    def solve_vertex(self, reg: Region, D, pT, m_had: float, gamma_new: float, tP0: float, tN0: float, sP: int, sN: int, a_ref: float):
        """The fractions (aP, aN) of the region's two pieces that the hadron takes so that it is on shell and the
        new vertex O + tP p+ + tN p- has proper time Gamma_new (Sjostrand 1984): two bilinear equations, hence a
        quadratic; the root with both fractions positive and aP closest to a_ref is returned, else None."""
        P, N, w2 = reg.p_plus, reg.p_minus, reg.w2
        O = self.origin(reg.i, reg.k)
        PN = 0.5 * w2
        oP, oN = dot(O, P), dot(O, N)
        DT = D + pT
        c0 = mass2(DT) - m_had ** 2
        alpha, beta = dot(DT, P), dot(DT, N)
        g0 = mass2(O) + 2 * tP0 * oP + 2 * tN0 * oN + 2 * tP0 * tN0 * PN - gamma_new
        gP = 2 * sP * (oP + tN0 * PN)
        gN = 2 * sN * (oN + tP0 * PN)
        A2 = -w2 * (gP + 2 * alpha)
        A1 = 2 * alpha * (gN + 2 * beta) - 2 * beta * (gP + 2 * alpha) - w2 * (g0 + c0)
        A0 = c0 * (gN + 2 * beta) - 2 * beta * (g0 + c0)
        roots = []
        if abs(A2) < 1e-14 * max(1.0, abs(A1)):
            if abs(A1) > 1e-14:
                roots = [-A0 / A1]
        else:
            disc = A1 * A1 - 4 * A2 * A0
            if disc >= 0:
                sq = math.sqrt(disc)
                roots = [(-A1 + sq) / (2 * A2), (-A1 - sq) / (2 * A2)]
        sols = []
        for aP in roots:
            # aN from whichever of the two equations is better conditioned at this aP
            denG, denM = gN - w2 * aP, 2 * beta + w2 * aP
            if abs(denG) >= abs(denM):
                if abs(denG) < 1e-14:
                    continue
                aN = -(g0 + gP * aP) / denG
            else:
                aN = -(c0 + 2 * alpha * aP) / denM
            if aP >= -1e-12 and aN >= -1e-12:
                sols.append((max(aP, 0.0), max(aN, 0.0)))
        if not sols:
            return None
        return min(sols, key=lambda ab: abs(ab[0] - a_ref))

    def produce(self, end: End, say):
        """Try to produce one hadron from `end`; returns True on success, False if there is no room left.

        The hadron takes a fraction z of the remaining momentum of the end's own piece and as much of the
        other side as its mass needs (the single-region Lund rule). If that does not fit into the current
        region, the new vertex is searched in the regions further along the string, swallowing the rest of
        the pieces in between, as the point with the proper time Gamma_new = (1 - z)(Gamma_old + mT^2/z) at
        which the hadron is on shell (Sjostrand 1984); the first region, scanning the end's own side
        outwards and then the other side, in which that point lies inside the region is taken.
        """
        n = self.n
        pos = end.sign > 0
        if pos:
            dir_pieces, inv_pieces, rem_dir, rem_inv = self.p_plus, self.p_minus, self.rem_plus, self.rem_minus
            dir_range = range(end.i_dir, n)
            inv_range = lambda: range(end.i_inv, n + 1)
        else:
            dir_pieces, inv_pieces, rem_dir, rem_inv = self.p_minus, self.p_plus, self.rem_minus, self.rem_plus
            dir_range = range(end.i_dir, 0, -1)
            inv_range = lambda: range(end.i_inv, -1, -1)
        for attempt in range(NTRY_HADRON):
            # flavour and hadron
            for _ in range(100):
                new = self.flav.pick(end.flav)
                hid = self.flav.combine(end.flav, new)
                if hid:
                    break
            else:
                return False
            px_new, py_new = self.ptfun.sample(self.rng)
            px_had, py_had = end.px + px_new, end.py + py_new
            m_had = hadron_mass(hid)
            mt2 = m_had ** 2 + px_had ** 2 + py_had ** 2
            z = self.zfun.sample(end.flav, new, mt2, self.rng)
            gamma_new = (1.0 - z) * (end.gamma + mt2 / z)
            a_ref = z * rem_dir[end.i_dir]
            placed = None
            n_solved = 0
            D_dir = np.zeros(4)
            swallowed_dir = []
            for i_dir in dir_range:
                if i_dir != end.i_dir:
                    D_dir = D_dir + rem_dir[i_dir - (1 if pos else -1)] * dir_pieces[i_dir - (1 if pos else -1)]
                    swallowed_dir.append(i_dir - (1 if pos else -1))
                u_start = end.used_dir if i_dir == end.i_dir else 0.0
                D = D_dir.copy()
                swallowed_inv = []
                for i_inv in inv_range():
                    if (pos and i_inv <= i_dir) or (not pos and i_inv >= i_dir):
                        continue                       # not a region (the piece of the other side lies behind)
                    v_start = end.used_inv if i_inv == end.i_inv else 0.0
                    i, k = (i_dir, i_inv) if pos else (i_inv, i_dir)
                    reg = self.region(i, k)
                    if reg.w2 > 1e-9:
                        P, N = reg.p_plus, reg.p_minus
                        pT = px_had * reg.eX + py_had * reg.eY
                        if i_dir == end.i_dir and i_inv == end.i_inv:
                            a_dir = z * rem_dir[i_dir]
                            A = a_dir * dir_pieces[i_dir]
                            den = 2.0 * dot(A, inv_pieces[i_inv])
                            a_inv = (m_had ** 2 - mass2(A + pT)) / den if den > 1e-12 else -1.0
                            sol = (a_dir, a_inv) if a_inv >= 0 else None
                        else:
                            if pos:
                                r = self.solve_vertex(reg, D, pT, m_had, gamma_new, 1.0 - u_start, v_start, -1, +1, a_ref)
                            else:
                                r = self.solve_vertex(reg, D, pT, m_had, gamma_new, v_start, 1.0 - u_start, +1, -1, a_ref)
                                r = None if r is None else (r[1], r[0])
                            sol = r
                        n_solved += 1
                        if sol is not None and sol[0] <= rem_dir[i_dir] + 1e-12 and sol[1] <= rem_inv[i_inv] + 1e-12:
                            a_dir, a_inv = sol
                            p_had = D + a_dir * dir_pieces[i_dir] + a_inv * inv_pieces[i_inv] + pT
                            placed = (i_dir, i_inv, a_dir, a_inv, u_start, v_start, list(swallowed_dir), list(swallowed_inv))
                            break
                    # swallow this piece of the other side and look further
                    D = D + rem_inv[i_inv] * inv_pieces[i_inv]
                    swallowed_inv.append(i_inv)
                if placed is not None:
                    break
            if placed is None:
                if n_solved == 0 or attempt == NTRY_HADRON - 1:
                    self.n_fail += 1
                    return False                       # the string is used up: no room
                continue                               # redraw flavour, pT and z
            i_dir, i_inv, a_dir, a_inv, u_start, v_start, swallowed_dir, swallowed_inv = placed
            # bookkeeping of the shared pieces and of this end
            for kk in swallowed_dir:
                rem_dir[kk] = 0.0
            for kk in swallowed_inv:
                rem_inv[kk] = 0.0
            rem_dir[i_dir] = max(rem_dir[i_dir] - a_dir, 0.0)
            rem_inv[i_inv] = max(rem_inv[i_inv] - a_inv, 0.0)
            end.i_dir, end.i_inv = i_dir, i_inv
            end.used_dir, end.used_inv = u_start + a_dir, v_start + a_inv
            i, k = end.region_ik()
            if pos:
                vtx = self.vertex_point(i, k, 1.0 - end.used_dir, end.used_inv)
            else:
                vtx = self.vertex_point(i, k, end.used_inv, 1.0 - end.used_dir)
            gamma_old, end.gamma = end.gamma, max(mass2(vtx), 0.0)
            end.vertex = vtx
            info = dict(end="+" if pos else "-", old=end.flav, new=new, z=z, mt2=mt2, pt=(px_had, py_had), pt_new=(px_new, py_new),
                        region=(i, k), steps=swallowed_inv, steps_dir=swallowed_dir, x_dir=a_dir, x_inv=a_inv,
                        gamma=end.gamma, gamma_lund=gamma_new, gamma_old=gamma_old, rank=len(end.hadrons) + 1,
                        zinfo=dict(self.zfun.last), attempts=attempt + 1, m=math.sqrt(max(mass2(p_had), 0.0)),
                        coords=(1.0 - end.used_dir, end.used_inv) if pos else (end.used_inv, 1.0 - end.used_dir),
                        vertex=vtx.copy(), w_rem=self.w_rem_now, w_min=self.w_min_now)
            end.hadrons.append((hid, p_had, info))
            self.hadrons.append((hid, p_had, info))
            end.flav = -new
            end.px, end.py = -px_new, -py_new
            end.last_new = new
            if say:
                self.narrate_hadron(end, hid, p_had, info, px_new, py_new, say)
            return True
        return False

    def narrate_hadron(self, end, hid, p_had, info, px_new, py_new, say):
        zi = info["zinfo"]
        for kind, txt in self.flav.log:
            say(f"      {kind:8s} {txt}")
        say(f"      pT kick  ({px_new:+.3f}, {py_new:+.3f}) GeV from the Gaussian of width sigma/sqrt2 = {self.ptfun.sigma / math.sqrt(2):.3f} per component; "
            f"hadron pT = old + new = ({info['pt'][0]:+.3f}, {info['pt'][1]:+.3f}), mT = {math.sqrt(info['mt2']):.3f} GeV")
        heavy = zi.get("heavy", 0)
        ftxt = (f"Lund-Bowler, 1/z^(1 + r b m^2) with r = {self.zfun.r[heavy]}, m = {QUARK_M0[heavy]}: exponent {1 + self.zfun.r[heavy] * self.zfun.b * QUARK_M0[heavy] ** 2:.3f}"
                if heavy and not self.zfun.peterson else f"Peterson, epsilon = {self.zfun.eps[heavy]}" if heavy else
                f"Lund symmetric, a_old = {zi['a_old']:.2f}, a_new = {zi['a_new']:.2f}, b = {self.zfun.b}")
        say(f"      z        {ftxt}; b mT^2 = {self.zfun.b * info['mt2']:.3f}; <z> = {zi['mean']:.3f}; drawn z = {info['z']:.4f}")
        stp = ""
        if info["steps"]:
            stp += f", after using up the other side's piece(s) {info['steps']}"
        if info["steps_dir"]:
            stp += f" and its own piece(s) {info['steps_dir']}"
        say(f"      Gamma    new vertex at Gamma = (1 - z)(Gamma_old + mT^2/z) = {info['gamma_lund']:.3f} GeV^2 "
            f"(tau = sqrt(Gamma)/kappa = {math.sqrt(info['gamma_lund']) / KAPPA:.3f} fm)")
        say(f"      region   ({info['region'][0]}, {info['region'][1]}){stp}: fraction {info['x_dir']:.4f} of its own piece and "
            f"{info['x_inv']:.4f} of the other side's piece")
        say(f"   -> {pname(hid):10s} rank {info['rank']} from the {info['end']} end: {fmt_p(p_had)}")

    # --- the last two hadrons ----------------------------------------------------------------------------
    def final_two(self, say):
        pos, neg = self.pos, self.neg
        for _ in range(NTRY_HADRON):
            new = self.flav.pick(pos.flav)
            h1 = self.flav.combine(pos.flav, new)
            h2 = self.flav.combine(-new, neg.flav)
            if h1 and h2:
                break
        else:
            return False
        p_rem = self.p_tot - sum((p for _, p, _ in self.hadrons), np.zeros(4))
        w2 = mass2(p_rem)
        if w2 <= 0:
            return False
        w = math.sqrt(w2)
        m1, m2 = hadron_mass(h1), hadron_mass(h2)
        # frame: p_rem at rest, z along the colour end's piece
        B = boost_matrix(-p_rem[1:] / p_rem[0])
        axis = (B @ self.p_plus[pos.i_dir])[1:]
        if np.linalg.norm(axis) < 1e-9:
            axis = np.array([0.0, 0.0, 1.0])
        R = rotation_to_z(axis)
        M = R @ B
        Minv = lorentz_inverse(M)
        px_new, py_new = self.ptfun.sample(self.rng)
        qx = 0.5 * (pos.px - neg.px) + px_new
        qy = 0.5 * (pos.py - neg.py) + py_new
        mt1 = math.sqrt(m1 ** 2 + qx ** 2 + qy ** 2)
        mt2 = math.sqrt(m2 ** 2 + qx ** 2 + qy ** 2)
        if w < mt1 + mt2:
            return False
        e1 = (w2 + mt1 ** 2 - mt2 ** 2) / (2 * w)
        e2 = w - e1
        pz = math.sqrt(max(e1 ** 2 - mt1 ** 2, 0.0))
        # orientation by the area law: the colour-end hadron forward (along the colour end's piece) or
        # backward; each assignment is weighted with the fragmentation functions of both ends at the
        # light-cone fractions the two hadrons then take, which keeps a heavy hadron with its quark
        def weight(sign):
            z1 = min(max((e1 + sign * pz) / w, 1e-6), 1 - 1e-6)
            z2 = min(max((e2 + sign * pz) / w, 1e-6), 1 - 1e-6)
            return (float(self.zfun.logf(pos.flav, new, mt1 ** 2, np.array([z1]))[0])
                    + float(self.zfun.logf(neg.flav, -new, mt2 ** 2, np.array([z2]))[0]))
        lw_fwd, lw_rev = weight(+1), weight(-1)
        p_rev = 1.0 / (1.0 + math.exp(min(50.0, max(-50.0, lw_fwd - lw_rev))))
        reverse = self.rng.random() < p_rev
        z_fwd, z_rev = (e1 + pz) / w, (e1 - pz) / w
        g_fwd, g_rev = -lw_fwd, -lw_rev
        s = -1.0 if reverse else 1.0
        p1 = Minv @ np.array([e1, qx, qy, s * pz])
        p2 = p_rem - p1
        info1 = dict(end="+", old=pos.flav, new=new, final=True, rank=len(pos.hadrons) + 1, z=z_rev if reverse else z_fwd,
                     reverse=reverse, p_reverse=p_rev, w_rem=w, w_min=self.w_min_now, mt2=mt1 ** 2, pt=(qx, qy), pt_new=(px_new, py_new))
        info2 = dict(end="-", old=neg.flav, new=-new, final=True, rank=len(neg.hadrons) + 1, reverse=reverse, w_rem=w,
                     w_min=self.w_min_now, mt2=mt2 ** 2, pt=(-qx, -qy), pt_new=(px_new, py_new))
        pos.hadrons.append((h1, p1, info1))
        neg.hadrons.append((h2, p2, info2))
        self.hadrons += [(h1, p1, info1), (h2, p2, info2)]
        if say:
            for kind, txt in self.flav.log:
                say(f"      {kind:8s} {txt}")
            say(f"      the remaining momentum has W = {w:.3f} GeV; the two hadrons {pname(h1)} ({pos.flav and pname(pos.flav)} + {pname(new)}) and "
                f"{pname(h2)} ({pname(-new)} + {pname(neg.flav)}) share it with pT = ({qx:+.3f}, {qy:+.3f}) GeV")
            say(f"      colour-end hadron forward: z = ({z_fwd:.3f}, {(e2 + pz) / w:.3f}), ln(f+ f-) = {lw_fwd:.2f}; reversed: "
                f"z = ({z_rev:.3f}, {(e2 - pz) / w:.3f}), ln(f+ f-) = {lw_rev:.2f}; P(reverse) = {p_rev:.3f} -> {'reversed' if reverse else 'forward'}")
            say(f"   -> {pname(h1):10s} rank {info1['rank']} from the + end: {fmt_p(p1)}")
            say(f"   -> {pname(h2):10s} rank {info2['rank']} from the - end: {fmt_p(p2)}")
        return True

    # --- the whole string --------------------------------------------------------------------------------
    def w_min(self):
        m_last = flav_m0(self.pos.last_new) if self.pos.last_new else 0.0
        if self.neg.last_new:
            m_last = max(m_last, flav_m0(self.neg.last_new)) if self.pos.last_new else flav_m0(self.neg.last_new)
        base = (math.hypot(flav_m0(self.pos.flav), math.hypot(self.pos.px, self.pos.py))
                + math.hypot(flav_m0(self.neg.flav), math.hypot(self.neg.px, self.neg.py))
                + self.stop_mass + self.stop_new * m_last)
        return base * (1.0 + (2 * self.rng.random() - 1) * self.stop_smear), base

    def fragment(self, say=None):
        """Fragment the string; returns the list of (id, p4, info) or None if it failed NTRY_STRING times."""
        for ntry in range(NTRY_STRING):
            self.setup()
            self.attempt = ntry + 1
            if say:
                say(f"   attempt {ntry + 1}: W = {self.w:.3f} GeV, {self.n + 1} partons -> {self.n} light-cone pieces on each side")
            while True:
                p_rem = self.p_tot - sum((p for _, p, _ in self.hadrons), np.zeros(4))
                w_rem = math.sqrt(max(mass2(p_rem), 0.0))
                wmin, wmin0 = self.w_min()
                self.w_rem_now, self.w_min_now = w_rem, wmin
                if w_rem < wmin:
                    if say:
                        say(f"   remaining W = {w_rem:.3f} < W_min = {wmin0:.3f} smeared to {wmin:.3f} GeV: the two chains are joined")
                    break
                from_pos = self.rng.random() < 0.5
                end = self.pos if from_pos else self.neg
                if say:
                    say(f"   remaining W = {w_rem:.3f} GeV > W_min = {wmin:.3f}; next hadron from the {'+' if from_pos else '-'} end "
                        f"({pname(end.flav)} with pT ({end.px:+.3f}, {end.py:+.3f}))")
                if not self.produce(end, say):
                    if say:
                        say(f"   no room for another hadron from the {'+' if from_pos else '-'} end: the two chains are joined now")
                    break
            if self.final_two(say):
                return self.ordered()
            if say:
                say("   the final two hadrons do not fit: restart the string")
        return None

    def ordered(self):
        return list(self.pos.hadrons) + list(reversed(self.neg.hadrons))


# ---------------------------------------------------------------------------
# ministrings
# ---------------------------------------------------------------------------
def two_body(p_tot, m1, m2, axis3, px, py, rng=None):
    """Two hadrons from a system at rest with the transverse momentum (px, py) relative to axis3 (lab 3-vector)."""
    w2 = mass2(p_tot)
    w = math.sqrt(w2)
    mt1, mt2 = math.hypot(m1, math.hypot(px, py)), math.hypot(m2, math.hypot(px, py))
    if w < mt1 + mt2:
        return None
    B = boost_matrix(-p_tot[1:] / p_tot[0])
    a = (B @ np.array([math.sqrt(1 + float(np.dot(axis3, axis3))), *axis3]))[1:]
    if np.linalg.norm(a) < 1e-9:
        a = np.array([0.0, 0.0, 1.0])
    M = rotation_to_z(a) @ B
    Minv = lorentz_inverse(M)
    e1 = (w2 + mt1 ** 2 - mt2 ** 2) / (2 * w)
    pz = math.sqrt(max(e1 ** 2 - mt1 ** 2, 0.0))
    p1 = Minv @ np.array([e1, px, py, pz])
    return p1, p_tot - p1


class MiniString:
    def __init__(self, partons, rng, flav, ptfun, verbose=0):
        self.partons, self.rng, self.flav, self.ptfun = list(partons), rng, flav, ptfun
        self.p_tot = sum((p for _, p, _ in partons), np.zeros(4))
        self.w = math.sqrt(max(mass2(self.p_tot), 0.0))
        self.closed = all(pid == 21 for pid, _, _ in partons)
        self.notes = []

    def ends(self):
        if self.closed:
            q = self.flav.pick_quark(self.flav.quark_w)
            return q, -q
        return self.partons[0][0], self.partons[-1][0]

    def to_two(self, say=None):
        f1, f2 = self.ends()
        for _ in range(NTRY_HADRON):
            new = self.flav.pick(f1)
            h1, h2 = self.flav.combine(f1, new), self.flav.combine(-new, f2)
            if not (h1 and h2):
                continue
            px, py = self.ptfun.sample(self.rng)
            res = two_body(self.p_tot, hadron_mass(h1), hadron_mass(h2), self.partons[0][1][1:], px, py)
            if res is None:
                continue
            if say:
                say(f"   ministring W = {self.w:.3f} GeV ({pname(f1)} ... {pname(f2)}): one break ({pname(new)}) -> {pname(h1)} + {pname(h2)}, "
                    f"pT = ({px:+.3f}, {py:+.3f}) along the first parton")
            return [(h1, res[0], dict(end="mini", old=f1, new=new, rank=1)), (h2, res[1], dict(end="mini", old=f2, new=-new, rank=1))]
        return None

    def to_one(self, others, say=None):
        """One hadron; the mass difference is taken from the other string system that gives the smallest
        invariant mass together with this one (PYTHIA's ministring2one), whose partons are boosted."""
        f1, f2 = self.ends()
        hid = 0
        for _ in range(NTRY_HADRON):
            hid = self.flav.combine(f1, f2)
            if hid:
                break
        if not hid or not others:
            return None
        m_had = hadron_mass(hid)
        best, best_m2 = None, math.inf
        for j, (p_other) in enumerate(others):
            m2 = mass2(self.p_tot + p_other)
            if m2 < best_m2 and m2 > 0:
                best, best_m2 = j, m2
        if best is None:
            return None
        p_other = others[best]
        m_other = math.sqrt(max(mass2(p_other), 0.0))
        s = best_m2
        if math.sqrt(s) < m_had + m_other:
            return None
        P = self.p_tot + p_other
        B = boost_matrix(-P[1:] / P[0])
        p1c, p2c = B @ self.p_tot, B @ p_other
        axis = p1c[1:]
        if np.linalg.norm(axis) < 1e-9:
            axis = np.array([0.0, 0.0, 1.0])
        axis = axis / np.linalg.norm(axis)
        pstar = math.sqrt(max((s - (m_had + m_other) ** 2) * (s - (m_had - m_other) ** 2), 0.0)) / (2 * math.sqrt(s))
        p1n = np.array([math.hypot(m_had, pstar), *(pstar * axis)])
        p2n = np.array([math.hypot(m_other, pstar), *(-pstar * axis)])
        Binv = lorentz_inverse(B)
        p_had = Binv @ p1n
        # the other system: the boost from its old to its new momentum (in the pair frame), applied to every parton
        T = Binv @ boost_matrix(p2n[1:] / p2n[0]) @ boost_matrix(-p2c[1:] / p2c[0]) @ B
        if say:
            say(f"   ministring W = {self.w:.3f} GeV ({pname(f1)} ... {pname(f2)}) -> one hadron {pname(hid)} (m = {m_had:.3f}); "
                f"the {m_had - self.w:+.3f} GeV of mass are taken from the string with the smallest pair mass "
                f"(m_pair = {math.sqrt(s):.2f}, its own mass {m_other:.2f} GeV)")
        return (hid, p_had, dict(end="mini", old=f1, new=f2, rank=1)), best, T


# ---------------------------------------------------------------------------
# the event: strings from the colour tags, joining, fragmentation, record
# ---------------------------------------------------------------------------
def read_record(path: Path) -> Event:
    lines = path.read_text().splitlines()
    parts = []
    for line in lines:
        f = line.split()
        if len(f) < 14 or not f[0].isdigit():
            continue
        try:
            pid, status = int(f[1]), int(f[3])
            m1, m2, d1, d2, col, acol = (int(v) for v in f[4:10])
            px, py, pz, e, m = (float(v) for v in f[-5:])
        except ValueError:
            continue
        parts.append(Particle(pid, status, m1, m2, d1, d2, col, acol, [e, px, py, pz], m))
    beams = [q for q in parts if q.status == -12]
    for q in parts:
        if q.status > 0:
            q.p[0] = math.sqrt(float(q.p[1] ** 2 + q.p[2] ** 2 + q.p[3] ** 2) + q.m * q.m)
    inc = [i for i, q in enumerate(parts) if q.status == -21]
    in_a, in_b = (sorted(inc, key=lambda i: -parts[i].p[3]) + [0, 0])[:2]
    return Event(parts, beams[0].p[0] if beams else 6800.0, in_a, in_b)


def listing(ev: Event, title: str) -> str:
    out = [f" --------  {title}  " + "-" * max(10, 133 - len(title) - 14), " ",
           "    no         id  name            status     mothers   daughters     colours"
           "      p_x        p_y        p_z         e          m "]
    for i, q in enumerate(ev.p):
        name = pname(q.id) if q.final else f"({pname(q.id)})"
        out.append(f"{i:6d}{q.id:11d}  {name:<18s}{q.status:4d}{q.m1:6d}{q.m2:6d}{q.d1:6d}{q.d2:6d}"
                   f"{q.col:6d}{q.acol:6d}{q.p[1]:11.3f}{q.p[2]:11.3f}{q.p[3]:11.3f}{q.p[0]:11.3f}{q.m:11.3f}")
    fin = ev.final_indices()
    psum = ev.momentum_sum(fin)
    charge = sum(charge3(ev.p[i].id) for i in fin) / 3.0
    out.append(f"{'':35s}Charge sum: {charge:6.3f}           Momentum sum:{psum[1]:11.3f}{psum[2]:11.3f}"
               f"{psum[3]:11.3f}{psum[0]:11.3f}{math.sqrt(max(mass2(psum), 0)):11.3f}")
    out.append("")
    out.append(f" --------  End {title}  " + "-" * max(10, 133 - len(title) - 18))
    return "\n".join(out) + "\n"


def colour_singlets(ev: Event):
    """The colour-singlet systems of the final state: lists of row indices in colour order (open strings from
    a quark/antidiquark to an antiquark/diquark; closed gluon loops), as PYTHIA's ColourTracing finds them."""
    fin = ev.final_indices()
    by_acol = {ev.p[i].acol: i for i in fin if ev.p[i].acol}
    used = set()
    chains = []
    for i in fin:
        q = ev.p[i]
        if q.col and not q.acol:            # a colour end
            chain = [i]
            used.add(i)
            cur = q
            while cur.col:
                j = by_acol.get(cur.col)
                if j is None or j in used:
                    raise RuntimeError(f"colour tag {cur.col} of row {i} has no free partner")
                chain.append(j)
                used.add(j)
                cur = ev.p[j]
            chains.append(chain)
    for i in fin:
        if i in used:
            continue
        q = ev.p[i]
        if q.col and q.acol:                # a gluon of a closed loop
            loop = [i]
            used.add(i)
            cur = q
            while True:
                j = by_acol.get(cur.col)
                if j is None or j == i:
                    break
                if j in used:
                    raise RuntimeError(f"gluon loop through row {i} is not closed")
                loop.append(j)
                used.add(j)
                cur = ev.p[j]
            chains.append(loop)
    left = [i for i in fin if i not in used and ev.p[i].coloured]
    if left:
        raise RuntimeError(f"coloured partons without a string: {left}")
    return chains


def join_nearby(partons, closed: bool, notes: list):
    """Join two colour-connected partons, at least one a gluon, whose invariant mass (gluons at half momentum)
    exceeds the sum of their masses by less than mJoin; repeated until none is left
    (FragmentationSystems:mJoin). Returns (partons, joined pairs)."""
    partons = list(partons)
    joined = []
    while True:
        best, best_m = None, M_JOIN
        n = len(partons)
        pairs = [(j, j + 1) for j in range(n - 1)] + ([(n - 1, 0)] if closed and n > 2 else [])
        for a, b in pairs:
            pa, pb = partons[a], partons[b]
            if pa[0] != 21 and pb[0] != 21:
                continue
            fa, fb = (0.5 if pa[0] == 21 else 1.0), (0.5 if pb[0] == 21 else 1.0)
            m = math.sqrt(max(mass2(fa * pa[1] + fb * pb[1]), 0.0)) - (0.0 if pa[0] == 21 else pa[2]) - (0.0 if pb[0] == 21 else pb[2])
            if m < best_m:
                best, best_m = (a, b), m
        if best is None or len(partons) <= (2 if not closed else 2):
            return partons, joined
        a, b = best
        pa, pb = partons[a], partons[b]
        keep, drop = (a, b) if pa[0] != 21 else (b, a) if pb[0] != 21 else (a, b)
        pk, pd = partons[keep], partons[drop]
        p_new = pk[1] + pd[1]
        new = (pk[0], p_new, math.sqrt(max(mass2(p_new), 0.0)), (pk[3] + pd[3]))
        notes.append(f"joined {pname(pd[0])} (row {pd[3][0]}) into {pname(pk[0])} (row {pk[3][0]}): pair mass above the quark masses {best_m:.3f} < mJoin = {M_JOIN} GeV")
        joined.append((pk[3], pd[3], new))
        partons[keep] = new
        del partons[drop]


class Hadroniser:
    def __init__(self, ev: Event, rng, args, verbose=0):
        self.ev, self.rng, self.args, self.verbose = ev, rng, args, verbose
        self.flav = StringFlav(rng, prob_s=args.prob_s, prob_qq=args.prob_qq)
        self.zfun = StringZ(a=args.alund, b=args.blund, r_b=args.rfactb, r_c=args.rfactc, peterson=args.peterson)
        self.ptfun = StringPT(sigma=args.sigma)
        self.strings = []       # dicts: rows, partons [(id, p, m, (rows))], closed, kind, hadrons, copies
        self.hadron_rows = []
        self.notes = []
        self.n_failed = 0

    def say(self, *a):
        print(*a)

    def collect(self):
        ev = self.ev
        self.b_rows = set(b_quark_rows(ev).values())
        self.p_in = ev.momentum_sum(ev.final_indices())
        chains = colour_singlets(ev)
        for rows in chains:
            closed = all(ev.p[i].id == 21 for i in rows)
            partons = [(ev.p[i].id, ev.p[i].p.copy(), ev.p[i].m, (i,)) for i in rows]
            notes = []
            partons, joined = join_nearby(partons, closed, notes)
            p_tot = sum((p for _, p, _, _ in partons), np.zeros(4))
            w = math.sqrt(max(mass2(p_tot), 0.0))
            m_ends = 0.0 if closed else flav_m0(partons[0][0]) + flav_m0(partons[-1][0])
            kind = "mini" if w < m_ends + M_STRING_MIN else "string"
            self.strings.append(dict(rows=rows, partons=partons, closed=closed, w=w, kind=kind, notes=notes, joined=joined,
                                     p_tot=p_tot, hadrons=None, has_b=any(i in self.b_rows for i in rows)))
        # the string with the b quark first in the narration, otherwise by mass
        order = sorted(range(len(self.strings)), key=lambda j: (-self.strings[j]["w"]))
        self.order = order

    def lambda_measure(self, partons):
        """The string-length measure of chapter 4: sum ln(1 + m_ij^2/m0^2) over neighbouring pairs."""
        lam = 0.0
        for a, b in zip(partons, partons[1:]):
            fa, fb = (0.5 if a[0] == 21 else 1.0), (0.5 if b[0] == 21 else 1.0)
            lam += math.log(1 + max(mass2(fa * a[1] + fb * b[1]), 0) / 0.3 ** 2)
        return lam

    def run(self):
        self.collect()
        v = self.verbose
        if v:
            self.say(banner("the colour-singlet strings of the record"))
            self.say(f"  {len(self.strings)} systems: {sum(1 for s in self.strings if s['kind'] == 'string')} strings and "
                     f"{sum(1 for s in self.strings if s['kind'] == 'mini')} ministrings (W < m_q1 + m_q2 + mStringMin = ... + {M_STRING_MIN} GeV); "
                     f"total string mass {sum(s['w'] for s in self.strings):.1f} GeV")
            self.say(f"  {'#':>3s} {'partons':>8s} {'W [GeV]':>9s} {'lambda':>7s} {'kind':6s}  colour order")
            for j in self.order:
                s = self.strings[j]
                ptxt = " ".join(pname(pid) for pid, _, _, _ in s["partons"])
                if len(ptxt) > 70:
                    ptxt = ptxt[:67] + "..."
                self.say(f"  {j:3d} {len(s['partons']):8d} {s['w']:9.2f} {self.lambda_measure(s['partons']):7.1f} {s['kind']:6s}  "
                         f"{'(closed) ' if s['closed'] else ''}{ptxt}{'   <- the b quark' if s['has_b'] else ''}")
                for n in s["notes"]:
                    self.say(f"        {n}")
        # ministrings first (their recoil goes to a string that is still unfragmented), then the strings
        minis = [j for j in self.order if self.strings[j]["kind"] == "mini"]
        normal = [j for j in self.order if self.strings[j]["kind"] == "string"]
        for j in minis:
            self.fragment_mini(j)
        for j in normal:
            self.fragment_string(j)
        self.write_record()

    def fragment_mini(self, j):
        s = self.strings[j]
        v = self.verbose
        say = self.say if v else None
        ms = MiniString([(pid, p, m) for pid, p, m, _ in s["partons"]], self.rng, self.flav, self.ptfun)
        had = ms.to_two(say)
        if had is not None:
            s["hadrons"], s["status"] = had, 82
            return
        others = [self.strings[k]["p_tot"] for k in range(len(self.strings)) if k != j and self.strings[k]["hadrons"] is None]
        idx = [k for k in range(len(self.strings)) if k != j and self.strings[k]["hadrons"] is None]
        res = ms.to_one(others, say)
        if res is None:
            self.n_failed += 1
            self.notes.append(f"ministring {j} (W = {s['w']:.3f}) could not be hadronised; its partons are kept")
            s["hadrons"], s["status"] = [], 0
            return
        had, best, T = res
        k = idx[best]
        other = self.strings[k]
        other["partons"] = [(pid, T @ p, m, rows) for pid, p, m, rows in other["partons"]]
        other["p_tot"] = sum((p for _, p, _, _ in other["partons"]), np.zeros(4))
        other["w"] = math.sqrt(max(mass2(other["p_tot"]), 0.0))
        other["recoil"] = True
        s["hadrons"], s["status"] = [had], 81
        s["partner"] = k

    def fragment_string(self, j):
        s = self.strings[j]
        v = self.verbose
        narrate = v >= 2 or (v == 1 and s["has_b"])
        say = self.say if narrate else None
        if v:
            self.say(banner(f"string {j}: W = {s['w']:.2f} GeV, {len(s['partons'])} partons" + (" (the b quark's string)" if s["has_b"] else ""), "-"))
        st = StringSystem([(pid, p, m) for pid, p, m, _ in s["partons"]], self.rng, self.flav, self.zfun, self.ptfun,
                          verbose=v, label=str(j))
        had = st.fragment(say)
        if had is None:
            # fall back: split into two hadrons like a ministring
            ms = MiniString([(pid, p, m) for pid, p, m, _ in s["partons"]], self.rng, self.flav, self.ptfun)
            had = ms.to_two(say)
            self.n_failed += 1
            self.notes.append(f"string {j} (W = {s['w']:.2f}) failed {NTRY_STRING} times and was split into two hadrons")
            s["hadrons"], s["status"], s["system"] = (had or []), 82, st
            return
        s["hadrons"], s["status"], s["system"] = had, 83, st
        for n in st.notes:
            s["notes"].append(n)
        if v:
            self.say(f"   {len(had)} hadrons: " + " ".join(pname(h) for h, _, _ in had))
            if not narrate:
                for h, p, info in had:
                    self.say(f"     {pname(h):10s} rank {info.get('rank', 0):2d} {info['end']:4s} z = {info.get('z', float('nan')):.3f}  {fmt_p(p)}")

    def write_record(self):
        ev = self.ev
        for s in self.strings:
            if s["hadrons"] is None:
                continue
            # copies of the partons (status -71; -72 if the string took a ministring's recoil; -73 joined)
            first = len(ev.p)
            for pid, p, m, rows in s["partons"]:
                st = -73 if len(rows) > 1 else (-72 if s.get("recoil") else -71)
                r0 = rows[0]
                new = Particle(pid, st, r0, rows[-1] if len(rows) > 1 else 0, 0, 0, ev.p[r0].col, ev.p[r0].acol, p, m)
                k = ev.add(new)
                for r in rows:
                    q = ev.p[r]
                    if q.status > 0:
                        q.status = -q.status
                    q.d1, q.d2 = k, k
            last = len(ev.p) - 1
            s["copies"] = (first, last)
            hfirst = len(ev.p)
            for hid, p, info in s["hadrons"]:
                status = s["status"]
                if status == 83 and info.get("end") == "-":
                    status = 84
                k = ev.add(Particle(hid, status, first, last, 0, 0, 0, 0, p, hadron_mass(hid)))
                info["row"] = k
                self.hadron_rows.append(k)
            for r in range(first, last + 1):
                ev.p[r].d1, ev.p[r].d2 = hfirst, len(ev.p) - 1

    def verify(self):
        """Largest deviation between the final state now and the partons that were handed in."""
        ev = self.ev
        psum = ev.momentum_sum(ev.final_indices())
        return float(np.max(np.abs(psum - self.p_in))), psum


# ---------------------------------------------------------------------------
# what the chapter looks at
# ---------------------------------------------------------------------------
def b_quark_rows(ev: Event):
    """The status-(-)62 copies of the two hard b quarks (the partons handed to hadronisation)."""
    out = {}
    hard = [i for i, q in enumerate(ev.p) if q.status == -23 and abs(q.id) == 5]
    for i in hard:
        j = i
        while True:
            q = ev.p[j]
            d1, d2 = q.d1, q.d2
            cands = [k for k in range(d1, max(d1, d2) + 1) if 0 < k < len(ev.p)]
            same = [k for k in cands if ev.p[k].id == q.id and abs(ev.p[k].status) < 70]
            if not same:
                break
            j = same[0]
        out[ev.p[i].id] = j
    return out


def heavy_content(pid: int) -> int:
    """5 for a b hadron, 4 for a charm hadron, 0 otherwise."""
    a = abs(pid)
    if a < 100:
        return 0
    digits = [int(c) for c in str(a)]
    quarks = digits[-4:-1] if a >= 1000 else digits[-3:-1]
    return 5 if 5 in quarks else 4 if 4 in quarks else 0


def is_charged(pid: int) -> bool:
    return charge3(pid) != 0


def cone_sum(ev, rows, axis, R):
    mem = [i for i in rows if ps.delta_r(ev.p[i].p, axis) < R]
    p = sum((ev.p[i].p for i in mem), np.zeros(4))
    return mem, p


def summarise(ev: Event, had: Hadroniser, radius=0.4) -> dict:
    hrows = had.hadron_rows
    s = dict(n_strings=sum(1 for x in had.strings if x["kind"] == "string"),
             n_mini=sum(1 for x in had.strings if x["kind"] == "mini"),
             n_closed=sum(1 for x in had.strings if x["closed"]),
             n_joined=sum(len(x["joined"]) for x in had.strings),
             n_hadrons=len(hrows), n_charged=sum(1 for i in hrows if is_charged(ev.p[i].id)),
             n_baryons=sum(1 for i in hrows if abs(ev.p[i].id) >= 1000),
             n_strange=sum(1 for i in hrows if 3 in [int(c) for c in str(abs(ev.p[i].id))[:-1]] and abs(ev.p[i].id) >= 100),
             n_failed=had.n_failed, sum_w=sum(x["w"] for x in had.strings),
             species={}, n_hadrons_central=sum(1 for i in hrows if abs(eta(ev.p[i].p)) < 2.5),
             n_charged_central=sum(1 for i in hrows if abs(eta(ev.p[i].p)) < 2.5 and is_charged(ev.p[i].id)))
    for i in hrows:
        n = pname(ev.p[i].id)
        s["species"][n] = s["species"].get(n, 0) + 1
    s["strings_w"] = sorted((x["w"] for x in had.strings), reverse=True)
    s["hadrons_per_string"] = [len(x["hadrons"] or []) for x in had.strings]
    # the b quarks and their hadrons
    bq = b_quark_rows(ev)
    s["b"] = {}
    for pid, jb in bq.items():
        qb = ev.p[jb]
        string = next((x for x in had.strings if jb in x["rows"]), None)
        if string is None or not string["hadrons"]:
            continue
        # the hadron that holds this b: the one with the b content among the string's hadrons, nearest in angle
        bh = [i for i in [h[2].get("row") for h in string["hadrons"]] if i is not None and heavy_content(ev.p[i].id) == 5
              and (ev.p[i].id < 0) == (pid > 0) if abs(ev.p[i].id) < 1000] \
            + [i for i in [h[2].get("row") for h in string["hadrons"]] if i is not None and heavy_content(ev.p[i].id) == 5
               and (ev.p[i].id > 0) == (pid > 0) and abs(ev.p[i].id) >= 1000]
        if not bh:
            continue
        ih = min(bh, key=lambda i: ps.delta_r(ev.p[i].p, qb.p))
        hb = ev.p[ih]
        info = next(h[2] for h in string["hadrons"] if h[2].get("row") == ih)
        pb, ph = qb.p, hb.p
        nb = pb[1:] / np.linalg.norm(pb[1:])
        lc_b = pb[0] + float(np.dot(pb[1:], nb))
        lc_h = ph[0] + float(np.dot(ph[1:], nb))
        rec = dict(quark=pname(pid), quark_row=jb, pt_quark=pt(pb), e_quark=pb[0], eta_quark=eta(pb), phi_quark=phi(pb),
                   hadron=pname(hb.id), hadron_id=hb.id, hadron_row=ih, pt_hadron=pt(ph), e_hadron=ph[0], m_hadron=hb.m,
                   eta_hadron=eta(ph), phi_hadron=phi(ph), dr=ps.delta_r(ph, pb),
                   x_e=ph[0] / pb[0], x_pt=pt(ph) / pt(pb), x_plus=lc_h / lc_b, z_drawn=info.get("z"),
                   rank=info.get("rank"), end=info.get("end"), string_w=string["w"], string_n=len(string["hadrons"]),
                   string_partons=len(string["partons"]), cones={})
        rows_string = [h[2]["row"] for h in string["hadrons"] if h[2].get("row") is not None]
        for R in (0.4, 0.8):
            mem_h, p_h = cone_sum(ev, hrows, pb, R)
            mem_hb, p_hb = cone_sum(ev, hrows, ph, R)
            mem_str = [i for i in mem_h if i in rows_string]
            p_str = sum((ev.p[i].p for i in mem_str), np.zeros(4))
            parton_rows = [i for i, q in enumerate(ev.p) if abs(q.status) in (62, 63)]
            mem_p, p_p = cone_sum(ev, parton_rows, pb, R)
            rec["cones"][R] = dict(n_hadrons=len(mem_h), pt_hadrons=pt(p_h), m_hadrons=math.sqrt(max(mass2(p_h), 0)),
                                   n_partons=len(mem_p), pt_partons=pt(p_p), m_partons=math.sqrt(max(mass2(p_p), 0)),
                                   n_string=len(mem_str), pt_string=pt(p_str),
                                   n_hadrons_B=len(mem_hb), pt_hadrons_B=pt(p_hb), m_hadrons_B=math.sqrt(max(mass2(p_hb), 0)),
                                   n_charged=sum(1 for i in mem_h if is_charged(ev.p[i].id)),
                                   frac_B=pt(ph) / pt(p_h) if pt(p_h) > 0 else 0.0)
        s["b"][pname(pid)] = rec
    return s


def print_summary(s: dict):
    print(f"  {s['n_strings']} strings and {s['n_mini']} ministrings ({s['n_closed']} closed gluon loops, {s['n_joined']} partons joined), "
          f"total string mass {s['sum_w']:.0f} GeV -> {s['n_hadrons']} primary hadrons, {s['n_charged']} charged, "
          f"{s['n_baryons']} baryons, {s['n_strange']} with strangeness; {s['n_hadrons_central']} hadrons ({s['n_charged_central']} charged) at |eta| < 2.5"
          + (f"; {s['n_failed']} system(s) needed the fallback" if s["n_failed"] else ""))
    top = sorted(s["species"].items(), key=lambda kv: -kv[1])[:12]
    print("  most common: " + ", ".join(f"{n} {k}" for n, k in top))
    for name, r in s["b"].items():
        c4, c8 = r["cones"][0.4], r["cones"][0.8]
        print(f"  {name} (row {r['quark_row']}, pT = {r['pt_quark']:.2f}, E = {r['e_quark']:.2f} GeV) -> {r['hadron']} (row {r['hadron_row']}): "
              f"pT = {r['pt_hadron']:.2f}, E = {r['e_hadron']:.2f} GeV, dR = {r['dr']:.3f}; x_E = {r['x_e']:.3f}, x_pT = {r['x_pt']:.3f}, "
              f"x_+ = {r['x_plus']:.3f} (z drawn {r['z_drawn'] if r['z_drawn'] is None else round(r['z_drawn'], 3)}, rank {r['rank']} from the {r['end']} end of a "
              f"{r['string_w']:.0f} GeV string with {r['string_partons']} partons and {r['string_n']} hadrons)")
        print(f"     cone R = 0.4 around the quark: {c4['n_partons']} partons, pT {c4['pt_partons']:.2f}, m {c4['m_partons']:.2f} -> "
              f"{c4['n_hadrons']} hadrons ({c4['n_charged']} charged), pT {c4['pt_hadrons']:.2f}, m {c4['m_hadrons']:.2f} GeV; "
              f"the {name} hadron carries {100 * c4['frac_B']:.0f}% of the cone pT; R = 0.8: pT {c8['pt_partons']:.2f} -> {c8['pt_hadrons']:.2f}, "
              f"m {c8['m_partons']:.2f} -> {c8['m_hadrons']:.2f}, {c8['n_hadrons']} hadrons")


def explain_setup(args, had: Hadroniser):
    say = print
    say(banner("setting the scene: from partons to hadrons"))
    say(f"  string tension kappa = {KAPPA} GeV/fm; Lund a = {args.alund}, b = {args.blund} GeV^-2; a_diquark = a + {A_EXTRA_DIQUARK}; "
        f"Bowler r_b = {args.rfactb}, r_c = {args.rfactc}" + (" (Peterson used instead)" if args.peterson else ""))
    say(f"  pT at a break: Gaussian with sigma = {args.sigma} GeV for the pair (each component sigma/sqrt2 = {args.sigma / math.sqrt(2):.3f}); "
        f"{100 * ENHANCED_FRACTION:.0f}% of the breaks with {ENHANCED_WIDTH} x the width")
    say(f"  flavours: s/u = {args.prob_s} (tunnelling exp(-pi m^2/kappa): m_s = 0.5 GeV would give "
        f"{math.exp(-math.pi * (0.5 ** 2 - 0.33 ** 2) / (KAPPA * HBARC)):.3f}), diquark/quark = {args.prob_qq}, "
        f"strange diquark x {PROB_SQ_TO_QQ}, spin-1 diquark x 3 x {PROB_QQ1_TO_QQ0}")
    say(f"  vector/pseudoscalar: u,d {MESON_VECTOR[1]}, s {MESON_VECTOR[3]}, c {MESON_VECTOR[4]}, b {MESON_VECTOR[5]}; eta x {ETA_SUP}, eta' x {ETA_PRIME_SUP}")
    say(f"  stop: W_min = m_q1 + m_q2 + {STOP_MASS} + {STOP_NEW_FLAV} m_qlast, smeared +-{100 * STOP_SMEAR:.0f}%; ministring below "
        f"m_q1 + m_q2 + {M_STRING_MIN} GeV; partons joined below {M_JOIN} GeV")
    mt2 = hadron_mass(-521) ** 2 + 2 * (args.sigma / math.sqrt(2)) ** 2
    zf = had.zfun
    for lab, fo, fn, m2 in (("u -> pi (mT = m_pi)", 2, -1, hadron_mass(211) ** 2 + args.sigma ** 2),
                            ("u -> rho", 2, -1, hadron_mass(213) ** 2 + args.sigma ** 2),
                            ("b -> B (Lund-Bowler)", 5, -2, mt2)):
        lf = zf.logf(fo, fn, m2)
        f = np.exp(lf - lf.max())
        mean = float((zf.z * f).sum() / f.sum())
        zmax = float(zf.z[np.argmax(f)])
        say(f"  f(z) for {lab:22s}: <z> = {mean:.3f}, maximum at z = {zmax:.3f}")


def run(args, seed: int, verbose=0):
    ev = read_record(args.input)
    rng = np.random.default_rng(seed)
    had = Hadroniser(ev, rng, args, verbose=verbose)
    if verbose:
        explain_setup(args, had)
    had.run()
    dev, psum = had.verify()
    if verbose:
        print(banner("checks"))
        print(f"  final state: {len(had.hadron_rows)} hadrons" + (f" + {sum(1 for i in ev.final_indices() if ev.p[i].coloured)} partons left" if any(ev.p[i].coloured for i in ev.final_indices()) else "")
              + f"; their momentum sum deviates from that of the partons by {dev:.2e} GeV; charge {sum(charge3(ev.p[i].id) for i in ev.final_indices()) / 3:.1f}")
        for n in had.notes:
            print("  " + n)
        print("  final event record (also written to the output file):")
        print(listing(ev, "Standalone Event Listing after hadronisation").rstrip())
    return ev, had


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=HERE.parent.parent / "MPI" / "event_MPI.lhe")
    ap.add_argument("-o", "--output", type=Path, default=HERE.parent / "event_Had.lhe")
    ap.add_argument("--history", type=Path, help="write the summary (and with --repeat the per-event statistics) as JSON")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--alund", type=float, default=A_LUND, help="StringZ:aLund")
    ap.add_argument("--blund", type=float, default=B_LUND, help="StringZ:bLund [GeV^-2]")
    ap.add_argument("--rfactb", type=float, default=R_FACT_B, help="StringZ:rFactB (0: plain Lund for the b quark)")
    ap.add_argument("--rfactc", type=float, default=R_FACT_C, help="StringZ:rFactC")
    ap.add_argument("--peterson", action="store_true", help="Peterson fragmentation function for c and b")
    ap.add_argument("--sigma", type=float, default=SIGMA_PT, help="StringPT:sigma [GeV]")
    ap.add_argument("--prob-s", type=float, default=PROB_S_TO_UD, help="StringFlav:probStoUD")
    ap.add_argument("--prob-qq", type=float, default=PROB_QQ_TO_Q, help="StringFlav:probQQtoQ")
    ap.add_argument("--radius", type=float, default=0.4)
    ap.add_argument("--repeat", type=int, default=0, help="hadronise N times (seeds seed..seed+N-1) and print averages")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("-v", "--verbose", action="count", default=0,
                    help="learning mode: -v narrates the b quark's string hadron by hadron, -vv every string")
    args = ap.parse_args()

    ev, had = run(args, args.seed, verbose=args.verbose)
    summary = summarise(ev, had, args.radius)
    if args.verbose:
        print(banner("summary"))
    if not args.quiet:
        print(f"input {args.input}: {len(ev.p)} rows after hadronisation; seed = {args.seed}")
        print_summary(summary)
    title = "Standalone Event Listing  (hadron level: hard process + showers + MPI + beam remnants + colour reconnection + hadronisation)"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(listing(ev, title))
    print(f"wrote {args.output} ({len(ev.p)} rows, {summary['n_hadrons']} primary hadrons)")
    if args.history:
        args.history.write_text(json.dumps(dict(seed=args.seed, summary=summary), indent=1, default=float) + "\n")
        print(f"wrote {args.history}")

    if args.repeat > 1:
        keys = ["n_hadrons", "n_charged", "n_baryons", "n_strange", "n_hadrons_central", "n_charged_central", "n_failed"]
        acc = {k: [] for k in keys}
        fam = {}
        species = {}
        bspecies = {}
        per_string = []
        strings_w = None
        for n in range(args.repeat):
            try:
                e2, h2 = run(args, args.seed + n, verbose=0)
            except RuntimeError as err:
                print(f"  seed {args.seed + n}: {err}")
                continue
            s = summarise(e2, h2, args.radius)
            for k in keys:
                acc[k].append(s[k])
            per_string.append(s["hadrons_per_string"])
            strings_w = [x["w"] for x in h2.strings]
            for name, k in s["species"].items():
                species[name] = species.get(name, 0) + k
            for name, r in s["b"].items():
                d = fam.setdefault(name, dict(x_e=[], x_pt=[], x_plus=[], pt_hadron=[], e_hadron=[], dr=[], z_drawn=[], rank=[],
                                              cones={R: dict(pt_hadrons=[], m_hadrons=[], n_hadrons=[], pt_partons=[], frac_B=[],
                                                             pt_hadrons_B=[], n_charged=[]) for R in (0.4, 0.8)}))
                for k in ("x_e", "x_pt", "x_plus", "pt_hadron", "e_hadron", "dr", "rank"):
                    d[k].append(r[k])
                d["z_drawn"].append(r["z_drawn"] if r["z_drawn"] is not None else float("nan"))
                bspecies.setdefault(name, {})
                bspecies[name][r["hadron"]] = bspecies[name].get(r["hadron"], 0) + 1
                for R in (0.4, 0.8):
                    for k in d["cones"][R]:
                        d["cones"][R][k].append(r["cones"][R][k])
        ndone = len(acc["n_hadrons"])
        print(f"averages over {ndone} hadronisations of the same parton configuration (seeds {args.seed}..{args.seed + args.repeat - 1}):")
        for k, v in acc.items():
            v = np.asarray(v, dtype=float)
            print(f"  {k:18s} mean = {v.mean():9.3f}  std = {v.std():8.3f}")
        for name, d in fam.items():
            print(f"  {name}: x_E = {np.mean(d['x_e']):.3f} +- {np.std(d['x_e']):.3f}, x_pT = {np.mean(d['x_pt']):.3f}, x_+ = {np.mean(d['x_plus']):.3f}, "
                  f"<z drawn> = {np.nanmean(d['z_drawn']):.3f}, rank 1 in {100 * np.mean(np.asarray(d['rank']) == 1):.0f}%; "
                  f"hadrons: " + ", ".join(f"{h} {100 * k / ndone:.0f}%" for h, k in sorted(bspecies[name].items(), key=lambda kv: -kv[1])))
            for R in (0.4, 0.8):
                c = d["cones"][R]
                print(f"     cone R = {R}: partons pT {np.mean(c['pt_partons']):.2f} -> hadrons pT {np.mean(c['pt_hadrons']):.2f} +- {np.std(c['pt_hadrons']):.2f}, "
                      f"m {np.mean(c['m_hadrons']):.2f}, {np.mean(c['n_hadrons']):.1f} hadrons ({np.mean(c['n_charged']):.1f} charged); "
                      f"B share {np.mean(c['frac_B']):.3f}")
        if args.history:
            stats = dict(repeat=args.repeat, seed0=args.seed, counts=acc, species=species, b=fam, b_species=bspecies,
                         strings_w=strings_w, hadrons_per_string=per_string,
                         params=dict(alund=args.alund, blund=args.blund, rfactb=args.rfactb, peterson=args.peterson, sigma=args.sigma))
            out = args.history.with_suffix(".stats.json")
            out.write_text(json.dumps(stats, default=float) + "\n")
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
