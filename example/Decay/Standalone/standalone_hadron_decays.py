#!/usr/bin/env python3
"""Standalone PYTHIA-like hadron decays for the hadron-level b bbar event of chapter 5.

Dependencies: numpy, plus the classes of chapter 5's ../../Had/Standalone/standalone_hadronisation.py
(the event record, the StringFlav flavour selection, the Lorentz frames) and of chapter 2
(imported through it). No PYTHIA: the particle properties and the decay tables (161
particles, 3114 channels) are read from decay_tables.json, dumped once from PYTHIA 8's
ParticleData by dump_decay_tables.py.

Input is the hadron-level record after the Lund string fragmentation
(../../Had/event_Had.lhe: the 365 primary hadrons are the rows with status 81-84). Output is
the same table format with every unstable hadron decayed (../event_Decay.lhe), with PYTHIA's
status codes: a decayed particle keeps its row with a negative status, its products follow
with status 91 (92 when a B0 or B_s0 oscillated before it decayed), and the intermediate
partons of a partonic decay are listed with status -91 before the hadrons they became.
A second table at the end of the file holds the production vertex (mm) and the proper
lifetime drawn for every row that did not start at the origin: the displaced vertices of
this chapter. Everything the detector will see is a row with a positive status, as in
../CMSSW/genPartAnalyzer_ME_PS_MPI_Had_Decay_out.pdf (page 5, "hadron decays").

What is implemented (the physics of chapter 6, following PYTHIA 8's ParticleDecays)
-----------------------------------------------------------------------------------
* Which particles decay. Every final particle with a decay table and tau0 < tau0Max =
  10 mm (ParticleDecays:limitTau0 = on, as CMSSW sets it): pi0, eta, the vector and
  tensor mesons, the Delta and Sigma* baryons, the charm and bottom hadrons and the tau
  lepton decay; pi+-, K+-, K_S0, K_L0, Lambda, Sigma+-, Xi, Omega, n, p, mu and the
  photons, electrons and neutrinos are left for the detector (or for GEANT).
* Lifetime and flight. tau = -tau0 ln r, the decay vertex x_dec = x_prod + tau p/m: a
  B hadron with gamma = E/m = 30 and c tau0 = 0.46 mm flies 14 mm on average; this
  flight, seen through the impact parameters of the tracks, is the b tag of chapter 12.
* B0 and B_s0 mixing. With probability sin^2(x tau / 2 tau0), x_d = 0.776, x_s = 26.05,
  the meson decays as its antiparticle (status 92 on the products).
* The channel. Picked in proportion to its branching ratio among the open channels (as
  PYTHIA; a channel can be open for the particle only or for the antiparticle only); the masses of broad products (rho, K*, D*_0, a_1 ...) are drawn from
  a Breit-Wigner within [mMin, mMax]; a channel whose products do not fit is drawn again.
* The kinematics. Two-body decays isotropic in the rest frame; three and more bodies
  with the M-generator (Raubold-Lynch phase space with accept/reject on the product of
  two-body momenta); matrix elements as in PYTHIA's meMode: the V-A weight
        |M|^2 ~ (p_B . p_nu) (p_X . p_l)
  for semileptonic decays (meMode 22, 23), which makes the lepton hard and the neutrino
  softer than phase space (the mother is contracted with the antifermion of the lepton pair, the
  soft one); |p_1 x p_2|^2 for omega, phi -> 3 pi (meMode 1); cos^2 theta
  for a vector meson from a pseudoscalar decaying to two pseudoscalars (meMode 2); the
  Kroll-Wada spectrum of the lepton pair in Dalitz decays pi0, eta -> gamma l+ l-
  (meMode 11); a soft damping exp(-p^2/sigma^2), sigma = 0.5 GeV, of the extra hadrons
  in semileptonic decays with more than three bodies.
* Partonic channels. Two thirds of PYTHIA's B decays are written at quark level,
  b -> c W, W -> q qbar' plus the spectator (meMode 23, 42-70), and so are the charmonium
  decays to light hadrons (J/psi -> g g g, "rndmflavg"). The partons are paired into one
  or two colour singlets (with probability colRearrange = 0.5 the other pairing), the
  number of hadrons is drawn from a Poissonian with mean
        n_known/2 + n_partons/4 + multIncrease ln(mDiff / multRefMass),
  multIncrease = 4 (2.5 for the weak meMode 23), at least 2 (3 for meMode 23, the code's
  own minimum for 42-70, fixed for 62-70) and at most 10, and each singlet becomes a chain
  of hadrons with chapter 5's StringFlav (new flavours, vector/pseudoscalar, SU(6) baryon
  weights); the products are then distributed in phase space. A semileptonic channel with
  a q qbar' spectator system (b -> u l nu) collapses it to one hadron (meMode 22).

What is left out, on purpose: QED radiation in decays (PHOTOS-like photons, CMSSW has
them on), the polarised tau decays of PYTHIA's TauDecays (taus decay in phase space here),
the parton shower before the hadronisation of J/psi -> g g g (the Poissonian multiplicity
is used instead), the mass-dependent widths of meMode 3-7 (the Breit-Wigner is used), the
Bose-Einstein effects, and EvtGen, the decay package CMSSW uses for b and c hadrons in many
samples (this sample used PYTHIA's own tables; chapter 6 compares them).

How close is it to PYTHIA? validate_decays_with_pythia.py lets PYTHIA 8.315 decay exactly
the same 365 primary hadrons many times, decays single B mesons at rest in both, and runs
the full chain; the multiplicities, the flight distances, the semileptonic fractions, the
neutrino energy and the hadron-level jet are compared in README.md.

Learning mode (--verbose / -v, -vv)
------------------------------------
  -v   narrates the decay chains of the two B hadrons step by step (lifetime draw and
       flight, mixing, the channel draw with its branching ratio, the partons turned
       into hadrons, the masses, the phase-space and matrix-element weights, the
       momenta in the rest frame and in the laboratory, the vertex), one line for every
       other decay, then the checks, the record and the summary.
  -vv  narrates every decay in full.
  The random numbers are the same with and without -v.

Examples
--------
  python3 standalone_hadron_decays.py                      # -> ../event_Decay.lhe (the book's record)
  python3 standalone_hadron_decays.py -v                   # learning mode
  python3 standalone_hadron_decays.py --no-mix             # no B0 oscillations
  python3 standalone_hadron_decays.py --no-limit-tau0      # also decay K_S0, Lambda, ... (PYTHIA's default)
  python3 standalone_hadron_decays.py --repeat 400 --quiet --history standalone_decay_stats.json
  python3 standalone_hadron_decays.py --single 511 --repeat 20000   # B0 at rest, the decay table in action

All energies in GeV, lengths in mm, times in mm/c (1 ps = 0.2998 mm/c).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "Had" / "Standalone"))
import standalone_hadronisation as had  # noqa: E402  (chapter 5: Event, StringFlav, frames, record reader)

ps = had.ps
Event, Particle = had.Event, had.Particle
pt, eta, phi, mass2, dot, fmt_p, banner = had.pt, had.eta, had.phi, had.mass2, had.dot, had.fmt_p, had.banner
boost_matrix, rotation_to_z = had.boost_matrix, had.rotation_to_z
is_quark, is_diquark = had.is_quark, had.is_diquark

# ---------------------------------------------------------------------------
# the tables: particle properties and decay channels from PYTHIA 8 (decay_tables.json)
# ---------------------------------------------------------------------------
TABLES_FILE = HERE / "decay_tables.json"
TABLES = json.loads(TABLES_FILE.read_text())
PART = {int(k): v for k, v in TABLES["particles"].items()}
SET = TABLES["settings"]
TAU0_MAX = SET["ParticleDecays:tau0Max"]            # mm/c
LIMIT_TAU0 = SET["ParticleDecays:limitTau0"]
MIX_B = SET["ParticleDecays:mixB"]
X_BD, X_BS = SET["ParticleDecays:xBdMix"], SET["ParticleDecays:xBsMix"]
M_SAFETY = SET["ParticleDecays:mSafety"]
SIGMA_SOFT = SET["ParticleDecays:sigmaSoft"]
MULT_INCREASE, MULT_INCREASE_WEAK = SET["ParticleDecays:multIncrease"], SET["ParticleDecays:multIncreaseWeak"]
MULT_REF_MASS, MULT_G_OFFSET = SET["ParticleDecays:multRefMass"], SET["ParticleDecays:multGoffset"]
COL_REARRANGE = SET["ParticleDecays:colRearrange"]
C_MM_PER_PS = 0.299792458                           # c in mm/ps
NARROW_WIDTH = 1e-6                                 # below this a product mass is its nominal one
NTRY_CHANNEL, NTRY_MASS, NTRY_PICK, NTRY_ME = 10, 100, 100, 2000
M_RHO = 0.7755                                      # vector-meson form factor of the Dalitz decays
LEPTONS, NEUTRINOS = (11, 13, 15), (12, 14, 16)


def name(pid: int) -> str:
    a = abs(pid)
    if a in PART:
        return PART[a]["name"] if pid > 0 else PART[a]["anti"]
    return had.pname(pid)


def has_anti(pid: int) -> bool:
    a = abs(pid)
    return a in PART and PART[a]["name"] != PART[a]["anti"]


def m0(pid: int) -> float:
    a = abs(pid)
    return PART[a]["m0"] if a in PART else had.hadron_mass(pid)


def tau0(pid: int) -> float:
    return PART[abs(pid)]["tau0"] if abs(pid) in PART else 0.0


def is_weak(pid: int) -> bool:
    """A particle with a measurable lifetime (the D*+ has tau0 = 2e-9 mm: not one of them)."""
    return tau0(pid) > 1e-4


def charge3(pid: int) -> int:
    a = abs(pid)
    if a in PART:
        c = PART[a]["charge3"]
        return c if pid > 0 else -c
    return had.charge3(pid)


def is_parton(pid: int) -> bool:
    a = abs(pid)
    return a < 10 or a == 21 or a in (81, 82, 83) or is_diquark(pid)


def is_lepton(pid: int) -> bool:
    return abs(pid) in LEPTONS


def is_neutrino(pid: int) -> bool:
    return abs(pid) in NEUTRINOS


def is_hadron(pid: int) -> bool:
    a = abs(pid)
    return a >= 100 and a not in (130, 310) or a in (130, 310)


def is_charged(pid: int) -> bool:
    return charge3(pid) != 0


def heavy_content(pid: int) -> int:
    """5 for a b hadron, 4 for a charm hadron, 0 otherwise (also for the quarks themselves)."""
    a = abs(pid)
    if a < 100 or a in (130, 310):
        return 0
    digits = [int(c) for c in str(a)]
    quarks = digits[-4:-1] if a >= 1000 else digits[-3:-1]
    return 5 if 5 in quarks else 4 if 4 in quarks else 0


def const_mass(pid: int) -> float:
    a = abs(pid)
    if a in (81, 82, 83):
        return 2 * 0.325
    return PART[a]["mConst"] if a in PART else had.flav_m0(pid)


def open_channels(pid: int):
    """[(bRatio, meMode, products)] of the channels open for this particle (onMode 1 = both, 2 = the particle
    only, 3 = the antiparticle only) and their sum (cached); the products are those of the positive id."""
    key = (abs(pid), pid > 0)
    if key not in _CHANNELS:
        allowed = (1, 2) if pid > 0 else (1, 3)
        chans = [(br, me, tuple(prods)) for br, me, on, prods in PART[abs(pid)]["channels"] if on in allowed and br > 0]
        _CHANNELS[key] = (chans, sum(c[0] for c in chans))
    return _CHANNELS[key]


_CHANNELS = {}


def can_decay(pid: int) -> bool:
    return abs(pid) in PART and open_channels(pid)[1] > 0


def lifetime_ps(tau_mm: float) -> float:
    return tau_mm / C_MM_PER_PS


def pstar(M: float, m1: float, m2: float) -> float:
    """Momentum of the two bodies in the rest frame of M (0 if they do not fit)."""
    if M <= m1 + m2:
        return 0.0
    return math.sqrt((M * M - (m1 + m2) ** 2) * (M * M - (m1 - m2) ** 2)) / (2 * M)


def isotropic(rng) -> np.ndarray:
    c = 2 * rng.random() - 1
    s = math.sqrt(1 - c * c)
    ph = 2 * math.pi * rng.random()
    return np.array([s * math.cos(ph), s * math.sin(ph), c])


def four(m: float, p3) -> np.ndarray:
    return np.array([math.sqrt(m * m + float(np.dot(p3, p3))), *p3])


def boost_all(ps_list, P: np.ndarray):
    """Boost 4-vectors given in the rest frame of P into the frame where P has that momentum."""
    if P[0] <= 0:
        return list(ps_list)
    B = boost_matrix(P[1:] / P[0])
    return [B @ p for p in ps_list]


# ---------------------------------------------------------------------------
# masses, lifetimes, channels
# ---------------------------------------------------------------------------
class Decayer:
    """Decays every unstable particle of an event (PYTHIA's ParticleDecays, simplified)."""

    def __init__(self, ev: Event, rng, args, verbose=0, narrate_rows=None):
        self.ev, self.rng, self.args, self.verbose = ev, rng, args, verbose
        self.flav = had.StringFlav(rng)
        self.vertex = {}            # row -> production vertex (t, x, y, z) in mm
        self.tau = {}               # row -> proper lifetime drawn (mm/c), only for rows with tau0 > 0 that decayed
        self.decays = []            # dicts, one per decay performed (row, channel, products, ...)
        self.notes = []
        self.n_osc = 0
        self.narrate_rows = set(narrate_rows or [])   # rows whose descendants are narrated in full with -v
        self.full_rows = set(self.narrate_rows)
        self.explicit = {}          # |id| -> set of product multisets of the explicit (non-partonic) channels

    def say(self, *a):
        print(*a)

    # --- the decision -------------------------------------------------------------------------------------
    def decides_to_decay(self, q: Particle) -> bool:
        a = abs(q.id)
        if a not in PART or not can_decay(a):
            return False
        if not PART[a]["mayDecay"]:
            return False
        if self.args.stable_b and heavy_content(q.id) == 5:
            return False
        if self.args.limit_tau0 and PART[a]["tau0"] > self.args.tau0max:
            return False
        return True

    # --- lifetime and vertex -------------------------------------------------------------------------------
    def draw_lifetime(self, pid: int) -> float:
        t0 = tau0(pid)
        if t0 <= 0:
            return 0.0
        return -t0 * math.log(1 - self.rng.random())

    def decay_vertex(self, row: int, q: Particle, tau: float):
        v = self.vertex.get(row, np.zeros(4))
        if tau <= 0 or q.m <= 0:
            return v
        return v + tau * q.p / q.m

    # --- the channel ---------------------------------------------------------------------------------------
    def pick_channel(self, pid: int):
        chans, total = open_channels(pid)
        r = self.rng.random() * total
        acc = 0.0
        for k, (br, me, prods) in enumerate(chans):
            acc += br
            if r < acc:
                return k, br, me, self.conjugate(prods, pid < 0), r, total
        br, me, prods = chans[-1]
        return len(chans) - 1, br, me, self.conjugate(prods, pid < 0), r, total

    @staticmethod
    def conjugate(prods, anti: bool):
        if not anti:
            return list(prods)
        return [-p if has_anti(p) or is_parton(p) else p for p in prods]

    def mass_sel(self, pid: int) -> float:
        """A product mass: nominal, or a Breit-Wigner between mMin and mMax for a broad resonance."""
        a = abs(pid)
        if a not in PART:
            return m0(pid)
        e = PART[a]
        w, lo, hi = e["mWidth"], e["mMin"], e["mMax"]
        if w < NARROW_WIDTH or lo >= hi:
            return e["m0"]
        a_lo, a_hi = math.atan(2 * (lo - e["m0"]) / w), math.atan(2 * (hi - e["m0"]) / w)
        return e["m0"] + 0.5 * w * math.tan(a_lo + self.rng.random() * (a_hi - a_lo))

    # --- partons -> hadrons (PYTHIA's pickHadrons) ------------------------------------------------------------
    @staticmethod
    def can_pair(a: int, b: int) -> bool:
        if is_quark(a) and is_quark(b):
            return a * b < 0
        if is_quark(a) and is_diquark(b) or is_diquark(a) and is_quark(b):
            return a * b > 0
        if is_diquark(a) and is_diquark(b):
            return a * b < 0
        return False

    def pick_hadrons(self, mother_id: int, m_mother: float, prods, me: int, say=None):
        """Replace the partons among the products by hadrons; returns (ids, text) or (None, reason)."""
        partons = [p for p in prods if is_parton(p)]
        known = [p for p in prods if not is_parton(p)]
        if not partons:
            return list(prods), ""
        from_loop = False
        if any(abs(p) == 83 for p in partons):          # a random light flavour pair from a closed gluon loop
            q = self.flav.pick_quark(self.flav.quark_w)
            partons = [q if p == 83 else -q if p == -83 else p for p in partons]
            from_loop = True
        if any(abs(p) == 82 for p in partons):
            q = self.flav.pick_quark(self.flav.quark_w)
            partons = [q if p == 82 else -q if p == -82 else p for p in partons]
        if any(p == 21 for p in partons):               # g g (g): one closed loop -> a light pair
            q = self.flav.pick_quark(self.flav.quark_w)
            partons = [q, -q]
            from_loop = True
        if len(partons) not in (2, 4):
            return None, f"{len(partons)} partons: not handled"
        if len(partons) == 2:
            pairs = [(partons[0], partons[1])]
            if not self.can_pair(*pairs[0]):
                return None, "the two partons cannot form a singlet"
        else:
            a, b, c, d = partons
            orig = [(a, b), (c, d)]
            alts = [[(a, c), (b, d)], [(a, d), (b, c)]]
            alts = [alt for alt in alts if all(self.can_pair(*pr) for pr in alt)]
            pairs = orig
            if alts and self.rng.random() < COL_REARRANGE:
                pairs = alts[0]
            if not all(self.can_pair(*pr) for pr in pairs):
                pairs = alts[0] if alts else None
                if pairs is None:
                    return None, "no colour-singlet pairing of the four partons"
        m_known = sum(m0(p) for p in known)
        n_known, n_partons, n_pairs = len(known), len(partons), len(pairs)
        # a weak decay (meMode 22, 23) written as W partons + spectator: the spectator pair (the last two
        # partons, or the rearranged pair that holds the spectator) collapses to one hadron, as PYTHIA's
        # pickHadrons does ("nSpec = 1"); only the W system makes a variable number of hadrons
        spec_pair = None
        if me in (22, 23) and n_pairs == 2:
            spec_pair = pairs[1]
            pairs = [pairs[0]]
            n_pairs = 1
        m_diff = m_mother - m_known - sum(const_mass(p) for p in partons)
        if m_diff <= 0:
            return None, f"no mass left for the partons (mDiff = {m_diff:.3f})"
        fixed = None
        n_spec = 1 if spec_pair else 0
        if me in (0, 22, 11, 12, 13, 31) or me < 0:
            n_min = max(2, n_known + n_pairs + n_spec)
            mode = "minimal"
        elif me == 23:
            mean = (n_known + n_spec) / 2 + n_partons / 4 + MULT_INCREASE_WEAK * math.log(max(1.1, m_diff / MULT_REF_MASS))
            n_min, mode = 3, "poisson"
        elif 42 <= me <= 60 or 91 <= me <= 94:
            mean = n_known / 2 + n_partons / 4 + MULT_INCREASE * math.log(max(1.1, m_diff / MULT_REF_MASS)) + (MULT_G_OFFSET if from_loop else 0)
            n_min = (me - 40) if me <= 50 else (me - 50) if me <= 60 else max(2, n_known + n_pairs)
            mode = "poisson"
        elif 62 <= me <= 80:
            fixed = (me - 60) if me <= 70 else (me - 70)
            n_min, mode = max(2, n_known + n_pairs), "fixed"
        else:
            n_min, mode = max(2, n_known + n_pairs), "minimal"
        avoid = (52 <= me <= 60) or (72 <= me <= 80)
        if avoid and abs(mother_id) not in self.explicit:
            chans, _ = open_channels(mother_id)
            self.explicit[abs(mother_id)] = {tuple(sorted(abs(p) for p in pr)) for _, m_, pr in chans if not any(is_parton(p) for p in pr)}
        text = []
        for ntry in range(NTRY_PICK):
            if mode == "minimal":
                n_tot = max(n_min, n_known + n_pairs + n_spec)
            elif mode == "fixed":
                n_tot = fixed
                if n_tot < n_known + n_pairs:
                    return None, f"fixed multiplicity {fixed} below the minimum {n_known + n_pairs}"
            else:
                n_tot = int(self.rng.poisson(mean))
                if n_tot < max(n_min, n_known + n_pairs + n_spec) or n_tot >= 10:
                    continue
            n_new = n_tot - n_known - n_spec
            # distribute the new hadrons over the singlets: at least one each, the rest at random
            per = [1] * n_pairs
            for _ in range(n_new - n_pairs):
                per[int(self.rng.integers(n_pairs))] += 1
            new = []
            ok = True
            for (f1, f2), n_i in zip(pairs, per):
                chain = self.chain(f1, f2, n_i)
                if chain is None:
                    ok = False
                    break
                new += chain
            if ok and spec_pair:
                h = self.chain(spec_pair[0], spec_pair[1], 1)
                if h is None:
                    ok = False
                else:
                    new += h
            if not ok:
                continue
            ids = known + new
            if avoid and tuple(sorted(abs(p) for p in ids)) in self.explicit[abs(mother_id)]:
                continue
            if sum(m0(p) for p in ids) > m_mother - M_SAFETY:
                continue
            if say:
                pairs_txt = " ".join("(" + " ".join(name(f) for f in pr) + ")" for pr in pairs) + (
                    f" and the spectator pair ({name(spec_pair[0])} {name(spec_pair[1])}) -> one hadron" if spec_pair else "")
                if mode == "poisson":
                    text.append(f"partons {' '.join(name(p) for p in partons)} as the singlets {pairs_txt}; mDiff = {m_mother:.3f} - {m_known:.3f} (known) - "
                                f"{sum(const_mass(p) for p in partons):.3f} (constituent masses) = {m_diff:.3f} GeV: "
                                f"<n> = {n_known}/2 + {n_partons}/4 + {MULT_INCREASE_WEAK if me == 23 else MULT_INCREASE:g} ln({m_diff:.3f}/{MULT_REF_MASS}) "
                                f"{'+ ' + str(MULT_G_OFFSET) + ' ' if from_loop else ''}= {mean:.2f}; Poisson -> {n_tot} hadrons (at least {n_min})"
                                + (f" after {ntry + 1} tries" if ntry else ""))
                else:
                    text.append(f"partons {' '.join(name(p) for p in partons)} as the singlets {pairs_txt} -> {mode} multiplicity: {n_tot} hadrons"
                                + (f" after {ntry + 1} tries" if ntry else ""))
                for (f1, f2), n_i in zip(list(pairs) + ([spec_pair] if spec_pair else []), per + ([1] if spec_pair else [])):
                    text.append(f"   {name(f1)} ... {name(f2)} -> {n_i} hadron(s): " + " ".join(name(h) for h in new[:n_i]))
                    new = new[n_i:]
            return ids, "\n".join(text)
        return None, f"no hadron set fitted into {m_mother:.3f} GeV in {NTRY_PICK} tries"

    def chain(self, f1: int, f2: int, n: int):
        """n hadrons along a colour singlet from flavour f1 to f2 with StringFlav; None if a draw is rejected."""
        for _ in range(20):
            hadrons, old, good = [], f1, True
            for _k in range(n - 1):
                new = self.flav.pick(old)
                h = self.flav.combine(old, new)
                if not h:
                    good = False
                    break
                hadrons.append(h)
                old = -new
            if not good:
                continue
            h = self.flav.combine(old, f2)
            if not h:
                continue
            hadrons.append(h)
            return hadrons
        return None

    # --- kinematics ---------------------------------------------------------------------------------------
    def mgenerator(self, M: float, masses, me: int, mother_p, ids, say=None):
        """n >= 3 products in the rest frame of the mother: the M-generator phase space with the matrix-element
        weight of meMode; returns (momenta, info)."""
        n = len(masses)
        msum = sum(masses)
        mdiff = M - msum
        if mdiff <= 0:
            return None, "no phase space"
        # the maximum of the phase-space weight: every intermediate system as heavy as it can be
        wt_max, mmax, mmin = 1.0, mdiff + masses[-1], 0.0
        for i in range(n - 2, -1, -1):
            mmax += masses[i]
            mmin += masses[i + 1]
            wt_max *= pstar(mmax, mmin, masses[i])
        info = dict(tries=0, me_tries=0)
        tail = [sum(masses[k:]) for k in range(n)]           # sum of masses of products k .. n-1
        for ntry in range(NTRY_ME):
            r = np.sort(self.rng.random(n - 2))[::-1]
            Mk = [M] + [tail[k] + r[k - 1] * mdiff for k in range(1, n - 1)] + [masses[-1]]
            wt = 1.0
            for k in range(n - 1):
                wt *= pstar(Mk[k], Mk[k + 1], masses[k])
            info["tries"] += 1
            if wt < self.rng.random() * wt_max:
                continue
            # momenta: build from the last pair outwards
            p = [None] * n
            pk = pstar(Mk[n - 2], masses[n - 2], masses[n - 1])
            d = isotropic(self.rng)
            p[n - 2], p[n - 1] = four(masses[n - 2], pk * d), four(masses[n - 1], -pk * d)
            for k in range(n - 3, -1, -1):
                pk = pstar(Mk[k], masses[k], Mk[k + 1])
                d = isotropic(self.rng)
                p[k] = four(masses[k], pk * d)
                Psys = four(Mk[k + 1], -pk * d)
                for j in range(k + 1, n):
                    p[j] = boost_matrix(Psys[1:] / Psys[0]) @ p[j]
            # matrix element
            wt_me, wt_me_max = self.me_weight(me, M, masses, p, ids, mother_p)
            info["me_tries"] += 1
            if wt_me_max > 0 and wt_me < self.rng.random() * wt_me_max:
                continue
            info.update(wt_ps=wt / wt_max, wt_me=(wt_me / wt_me_max if wt_me_max > 0 else 1.0))
            return p, info
        return None, f"no configuration accepted in {NTRY_ME} tries"

    def me_weight(self, me: int, M: float, masses, p, ids, mother_p):
        """The matrix-element weight of the configuration p (mother at rest) and its maximum (0: none)."""
        P = np.array([M, 0.0, 0.0, 0.0])
        if me == 1 and len(p) == 3:                                    # omega, phi -> pi+ pi- pi0
            c = np.cross(p[0][1:], p[1][1:])
            pmax = pstar(M, masses[0], masses[1] + masses[2])
            return float(np.dot(c, c)), pmax ** 4
        if me in (22, 23) and any(is_neutrino(i) for i in ids) and any(is_lepton(i) for i in ids):
            # PYTHIA's convention: the first product listed in the channel is the one contracted with the mother
            # (the neutrino for b -> c l nu, the charged lepton for c -> s l nu: the table's order encodes the V-A
            # pairing), the second with the rest of the system
            p_rest = sum((p[k] for k in range(2, len(p))), np.zeros(4))
            wt = M * p[0][0] * dot(p[1], p_rest)
            # the soft damping of hadrons beyond the first three products
            for k in range(3, len(p)):
                p2 = float(np.dot(p[k][1:], p[k][1:]))
                wt *= math.exp(-p2 / SIGMA_SOFT ** 2)
            return wt, 0.375 * M ** 4
        if me in (22, 23) and len(p) == 3:
            # a hadronic weak three-body decay: PYTHIA weights the first product's momentum with x (3 - 2x)
            x1 = 2 * math.sqrt(float(np.dot(p[0][1:], p[0][1:]))) / M
            return x1 * (3 - 2 * x1), 1.125
        return 1.0, 0.0

    def two_body(self, M: float, masses, me: int, mother_row: int, ids):
        """Two products in the rest frame; cos^2 theta weight (meMode 2) for a vector meson that came from a
        pseudoscalar's two-body decay, with theta measured from the direction of its sister."""
        pk = pstar(M, masses[0], masses[1])
        axis = None
        if me == 2:
            q = self.ev.p[mother_row]
            gm = q.m1
            if 0 < gm < len(self.ev.p):
                g = self.ev.p[gm]
                sisters = [k for k in range(g.d1, max(g.d1, g.d2) + 1) if k != mother_row]
                if len(sisters) == 1 and (abs(g.id) % 10 == 1 or abs(g.id) in (130, 310)) and g.d2 - g.d1 == 1:
                    # the sister's momentum in the rest frame of the vector meson
                    Bm = boost_matrix(-q.p[1:] / q.p[0])
                    ps_ = Bm @ self.ev.p[sisters[0]].p
                    if np.linalg.norm(ps_[1:]) > 1e-9:
                        axis = ps_[1:] / np.linalg.norm(ps_[1:])
        for _ in range(NTRY_ME):
            d = isotropic(self.rng)
            if axis is None:
                break
            c = float(np.dot(d, axis))
            if c * c >= self.rng.random():
                break
        return [four(masses[0], pk * d), four(masses[1], -pk * d)], dict(pstar=pk, polarised=axis is not None, cos=float(np.dot(d, axis)) if axis is not None else None)

    def dalitz(self, M: float, ids, masses):
        """gamma (or a hadron) + l+ l-: the lepton-pair mass from the Kroll-Wada spectrum with a rho form factor,
        then two two-body decays, the pair with the 1 + cos^2 theta distribution of a virtual photon."""
        i_l = [k for k, i in enumerate(ids) if is_lepton(i)]
        i_o = [k for k in range(len(ids)) if k not in i_l]
        if len(i_l) != 2 or len(i_o) != 1:
            return None, "not a Dalitz configuration"
        ml, mo = masses[i_l[0]], masses[i_o[0]]
        mlo, mhi = 2 * ml, M - mo
        if mhi <= mlo:
            return None, "no phase space"
        s_min, s_max = mlo * mlo, mhi * mhi
        s_rho, w_rho = M_RHO ** 2, (0.1491) ** 2

        def kw(s):          # PYTHIA's wtGam: Kroll-Wada times the rho propagator, normalised to be < 1 near s_min
            return ((1 + 0.5 * s_min / s) * math.sqrt(max(1 - s_min / s, 0)) * (1 - s / s_max) ** 3
                    * s_rho * (s_rho + w_rho) / ((s - s_rho) ** 2 + s_rho * w_rho))

        grid = np.exp(np.linspace(math.log(s_min * 1.0001), math.log(s_max * 0.9999), 80))
        wmax = max(kw(float(x)) for x in grid)
        for _ in range(NTRY_ME):
            sg = s_min * (s_max / s_min) ** self.rng.random()
            if kw(sg) >= self.rng.random() * wmax:
                break
        m = math.sqrt(sg)
        pk = pstar(M, mo, m)
        d = isotropic(self.rng)
        p_o, P_ll = four(mo, pk * d), four(m, -pk * d)
        pl = pstar(m, ml, ml)
        for _ in range(NTRY_ME):
            dl = isotropic(self.rng)
            c = float(np.dot(dl, -d))
            if 1 + c * c >= 2 * self.rng.random():
                break
        pls = boost_all([four(ml, pl * dl), four(ml, -pl * dl)], P_ll)
        out = [None] * 3
        out[i_o[0]], out[i_l[0]], out[i_l[1]] = p_o, pls[0], pls[1]
        return out, dict(m_ll=m)

    # --- one decay ------------------------------------------------------------------------------------------
    def decay(self, row: int):
        ev, q = self.ev, self.ev.p[row]
        pid = q.id
        full = self.verbose >= 2 or (self.verbose == 1 and row in self.full_rows)
        say = self.say if full else None
        brief = self.verbose >= 1 and not full
        k_eta, k_phi = eta(q.p), phi(q.p)
        t0 = tau0(pid)
        tau = self.draw_lifetime(pid)
        v_prod = self.vertex.get(row, np.zeros(4))
        v_dec = self.decay_vertex(row, q, tau)
        if say:
            say(banner(f"row {row}: {name(pid)} decays  (m = {q.m:.4f}, E = {q.p[0]:.3f}, pT = {pt(q.p):.3f} GeV, eta = {k_eta:.3f}, phi = {k_phi:.3f})", "-"))
            if t0 > 0:
                gb = math.sqrt(max(q.p[0] ** 2 - q.m ** 2, 0)) / q.m
                L = tau * gb
                say(f"  lifetime: c tau0 = {t0:.4f} mm ({lifetime_ps(t0):.3f} ps); r = {1 - math.exp(-tau / t0):.4f} -> tau = -tau0 ln(1 - r) = {tau:.4f} mm/c "
                    f"({lifetime_ps(tau):.3f} ps, {tau / t0:.2f} lifetimes); gamma beta = p/m = {gb:.2f} -> flight L = {L:.3f} mm "
                    f"(L_xy = {L * pt(q.p) / math.sqrt(max(q.p[0] ** 2 - q.m ** 2, 1e-12)):.3f} mm)")
                say(f"  born at ({v_prod[1]:.4f}, {v_prod[2]:.4f}, {v_prod[3]:.4f}) mm, decays at ({v_dec[1]:.4f}, {v_dec[2]:.4f}, {v_dec[3]:.4f}) mm, "
                    f"t = {v_dec[0]:.4f} mm/c")
            else:
                say(f"  lifetime: tau0 = 0 (a strong or electromagnetic decay, width {PART[abs(pid)]['mWidth'] * 1e3:.3g} MeV): "
                    f"it decays where it was born, at ({v_prod[1]:.4f}, {v_prod[2]:.4f}, {v_prod[3]:.4f}) mm")
        # oscillation
        pid_dec, status = pid, 91
        oscillated = False
        if MIX_B and not self.args.no_mix and abs(pid) in (511, 531) and t0 > 0:
            x = X_BD if abs(pid) == 511 else X_BS
            p_osc = math.sin(0.5 * x * tau / t0) ** 2
            r = self.rng.random()
            oscillated = r < p_osc
            if say:
                say(f"  mixing: x = Delta m / Gamma = {x}: P(oscillated) = sin^2(x tau / 2 tau0) = sin^2({0.5 * x * tau / t0:.3f}) = {p_osc:.3f}; "
                    f"r = {r:.3f} -> " + (f"oscillated: it decays as a {name(-pid)}" if oscillated else "no oscillation"))
            if oscillated:
                pid_dec, status = -pid, 92
                self.n_osc += 1
                q.id = pid_dec
        # the channel, the products, the masses, the kinematics
        chans, total = open_channels(pid_dec)
        result = None
        for ntry_ch in range(NTRY_CHANNEL):
            k, br, me, prods = self.pick_channel(pid_dec)[:4]
            if say:
                cls = channel_class(prods, me)
                say(f"  channel: r x (BR sum {total:.4f}) -> channel {k + 1} of {len(chans)}: {name(pid_dec)} -> {' '.join(name(p) for p in prods)} "
                    f"(BR {br:.5f} = {100 * br / total:.2f}% of the decays; meMode {me}: {ME_TEXT.get(me, me_text(me))}{'; ' + cls if cls else ''})"
                    + (f"   [try {ntry_ch + 1}]" if ntry_ch else ""))
            ids, text = self.pick_hadrons(pid_dec, q.m, prods, me, say)
            if ids is None:
                if say:
                    say(f"    {text}: another channel")
                continue
            if say and text:
                say("  " + text.replace("\n", "\n  "))
            n = len(ids)
            momenta, info, masses = None, None, None
            for ntry_m in range(NTRY_MASS):
                masses = [self.mass_sel(i) for i in ids]
                if sum(masses) > q.m - M_SAFETY and n > 1:
                    continue
                if n == 1:
                    momenta, info = [q.p.copy()], dict(kind="one body")
                    masses = [q.m]
                elif n == 2:
                    momenta, info = self.two_body(q.m, masses, me, row, ids)
                    info["kind"] = "two-body"
                elif me == 11 and n == 3:
                    momenta, info = self.dalitz(q.m, ids, masses)
                    if momenta is None:
                        continue
                    info["kind"] = "Dalitz"
                else:
                    momenta, info = self.mgenerator(q.m, masses, me, q.p, ids, say)
                    if momenta is None:
                        continue
                    info["kind"] = f"{n}-body"
                break
            if momenta is None:
                if say:
                    say(f"    the products do not fit into {q.m:.3f} GeV: another channel")
                continue
            result = (k, br, me, prods, ids, masses, momenta, info)
            break
        if result is None:
            self.notes.append(f"row {row} ({name(pid)}, m = {q.m:.3f}) could not be decayed in {NTRY_CHANNEL} channel draws; kept")
            return []
        k, br, me, prods, ids, masses, momenta, info = result
        if say:
            if any(abs(masses[j] - m0(ids[j])) > 1e-6 for j in range(len(ids))):
                say("  masses: " + ", ".join(f"{name(i)} {m:.3f}" + (" (Breit-Wigner)" if abs(m - m0(i)) > 1e-6 else "") for i, m in zip(ids, masses)))
            if info.get("kind") == "two-body":
                say(f"  two-body decay: p* = {info['pstar']:.4f} GeV in the rest frame, "
                    + (f"cos^2 theta weight (vector meson from a pseudoscalar): cos theta = {info['cos']:.3f}" if info["polarised"] else "isotropic"))
            elif info.get("kind") == "Dalitz":
                say(f"  Dalitz decay: lepton pair mass {info['m_ll']:.4f} GeV from the Kroll-Wada spectrum, 1 + cos^2 theta for the pair")
            elif "tries" in info:
                say(f"  {info['kind']} phase space (M-generator): accepted configuration {info['tries']} "
                    f"(phase-space weight {info['wt_ps']:.3f} of its maximum" + (f", matrix-element weight {info['wt_me']:.3f}" if me in (1, 22, 23) else "") + ")")
        # append the products
        p_lab = boost_all(momenta, q.p) if len(ids) > 1 else momenta
        first = len(ev.p)
        rows = []
        for i, m_, p_ in zip(ids, masses, p_lab):
            r_ = ev.add(Particle(i, status, row, 0, 0, 0, 0, 0, p_, m_))
            rows.append(r_)
            self.vertex[r_] = v_dec.copy()
        q.status = -abs(q.status)
        q.d1, q.d2 = first, len(ev.p) - 1
        if tau > 0:
            self.tau[row] = tau
        rec = dict(row=row, id=pid, id_dec=pid_dec, oscillated=oscillated, channel=k, br=br, me=me, prods=list(prods), ids=list(ids),
                   rows=rows, tau=tau, v_dec=v_dec.copy(), L=float(np.linalg.norm(v_dec[1:] - v_prod[1:])),
                   cls=channel_class(ids, me))
        self.decays.append(rec)
        if say:
            say(f"  products (lab frame), rows {first}-{len(ev.p) - 1}:")
            for r_ in rows:
                qq = ev.p[r_]
                say(f"    {r_:5d} {name(qq.id):12s} m = {qq.m:7.4f}  {fmt_p(qq.p)}")
            psum = sum(p_lab, np.zeros(4))
            say(f"  check: sum of the products - mother = ({psum[0] - q.p[0]:+.1e}, {psum[1] - q.p[1]:+.1e}, {psum[2] - q.p[2]:+.1e}, {psum[3] - q.p[3]:+.1e}) GeV")
        elif brief:
            flight = f", L = {rec['L']:.3f} mm" if tau > 0 else ""
            self.say(f"  row {row:5d} {name(pid):12s} (E = {q.p[0]:8.3f}) -> {' '.join(name(i) for i in ids):40s}  BR {100 * br / total:5.2f}%{flight}"
                     + ("  [oscillated]" if oscillated else ""))
        if row in self.full_rows:
            self.full_rows.update(rows)
        return rows

    # --- the loop over the event ----------------------------------------------------------------------------
    def run(self):
        ev = self.ev
        self.p_in = ev.momentum_sum(ev.final_indices())
        self.primaries = [i for i, q in enumerate(ev.p) if q.status in (81, 82, 83, 84)]
        if self.verbose:
            self.say(banner("which particles decay"))
            n_dec = sum(1 for i in self.primaries if self.decides_to_decay(ev.p[i]))
            self.say(f"  {len(self.primaries)} primary hadrons; {n_dec} of them decay (a decay table, tau0 < {self.args.tau0max} mm"
                     + (", B hadrons kept stable" if self.args.stable_b else "") + "), "
                     f"{len(self.primaries) - n_dec} are stable for the detector: "
                     + ", ".join(f"{name(k)} {v}" for k, v in sorted(count_ids(ev.p[i].id for i in self.primaries if not self.decides_to_decay(ev.p[i])).items(), key=lambda kv: -kv[1])))
            if self.verbose == 1:
                self.say(f"  the decay chains of rows {sorted(self.narrate_rows)} (the two B hadrons) are narrated in full, every other decay in one line")
        i = 0
        while i < len(ev.p):
            q = ev.p[i]
            if q.status > 0 and self.decides_to_decay(q):
                self.decay(i)
            i += 1

    def verify(self):
        psum = self.ev.momentum_sum(self.ev.final_indices())
        return float(np.max(np.abs(psum - self.p_in))), psum


ME_TEXT = {0: "phase space", 1: "|p1 x p2|^2 for omega/phi -> 3 pi", 2: "cos^2 theta if the vector came from a pseudoscalar",
           3: "two-body, L = 0", 4: "two-body, L = 1", 5: "two-body, L = 2", 6: "two-body, L = 3", 7: "two-body, L = 4",
           11: "Dalitz decay, Kroll-Wada lepton pair", 12: "Dalitz decay (phase space here)", 13: "double Dalitz (phase space here)",
           21: "tau decay (phase space here)", 22: "weak decay, V-A for the lepton pair", 23: "weak decay to partons, V-A, at least three hadrons",
           31: "b -> s gamma, hadrons from the quarks", 42: "partons -> hadrons, Poisson multiplicity", 43: "partons -> hadrons, Poisson, at least 3",
           52: "partons -> hadrons, Poisson, not an explicit channel", 53: "partons -> hadrons, Poisson, at least 3, not explicit",
           55: "partons -> hadrons, Poisson, at least 5, not explicit", 63: "partons -> exactly 3 hadrons", 91: "q qbar -> hadrons (no shower here)",
           92: "onium -> g g g -> hadrons (no shower here)", 93: "q qbar + singlet -> hadrons"}


def me_text(me: int) -> str:
    if 1500 <= me < 1600:
        return "tau decay model (phase space here)"
    if 42 <= me <= 50:
        return f"partons -> hadrons, Poisson, at least {me - 40}"
    if 52 <= me <= 60:
        return f"partons -> hadrons, Poisson, at least {me - 50}, not an explicit channel"
    if 62 <= me <= 70:
        return f"partons -> exactly {me - 60} hadrons"
    if 72 <= me <= 80:
        return f"partons -> exactly {me - 70} hadrons, not an explicit channel"
    return "phase space"


def channel_class(ids, me: int) -> str:
    """semileptonic (e, mu, tau), leptonic, or '' for hadronic."""
    leps = [i for i in ids if is_lepton(i)]
    nus = [i for i in ids if is_neutrino(i)]
    if leps and nus:
        lep = name(abs(leps[0])).rstrip("-+")
        return f"semileptonic ({lep})" if any(not is_lepton(i) and not is_neutrino(i) for i in ids) else f"leptonic ({lep})"
    if leps:
        return "with a lepton pair"
    return ""


def count_ids(it):
    out = {}
    for i in it:
        out[i] = out.get(i, 0) + 1
    return out


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------
def listing(ev: Event, title: str, dec: Decayer = None) -> str:
    out = [f" --------  {title}  " + "-" * max(10, 133 - len(title) - 14), " ",
           "    no         id  name            status     mothers   daughters     colours"
           "      p_x        p_y        p_z         e          m "]
    for i, q in enumerate(ev.p):
        nm = name(q.id) if q.final else f"({name(q.id)})"
        out.append(f"{i:6d}{q.id:11d}  {nm:<18s}{q.status:4d}{q.m1:6d}{q.m2:6d}{q.d1:6d}{q.d2:6d}"
                   f"{q.col:6d}{q.acol:6d}{q.p[1]:11.3f}{q.p[2]:11.3f}{q.p[3]:11.3f}{q.p[0]:11.3f}{q.m:11.3f}")
    fin = ev.final_indices()
    psum = ev.momentum_sum(fin)
    charge = sum(charge3(ev.p[i].id) for i in fin) / 3.0
    out.append(f"{'':35s}Charge sum: {charge:6.3f}           Momentum sum:{psum[1]:11.3f}{psum[2]:11.3f}"
               f"{psum[3]:11.3f}{psum[0]:11.3f}{math.sqrt(max(mass2(psum), 0)):11.3f}")
    out.append("")
    out.append(f" --------  End {title}  " + "-" * max(10, 133 - len(title) - 18))
    if dec is not None:
        t2 = "Standalone Vertex Listing  (production vertex [mm] and time [mm/c] of every row born more than 0.1 um from the origin; tau = proper lifetime drawn [mm/c])"
        out += ["", f" --------  {t2}  " + "-" * max(10, 133 - len(t2) - 14), " ",
                "    no      x_prod      y_prod      z_prod      t_prod         tau"]
        for i in range(len(ev.p)):
            v = dec.vertex.get(i)
            tau = dec.tau.get(i, 0.0)
            if (v is not None and float(np.max(np.abs(v[1:]))) > 1e-4) or tau > 1e-4:
                v = v if v is not None else np.zeros(4)
                out.append(f"{i:6d}{v[1]:12.5f}{v[2]:12.5f}{v[3]:12.5f}{v[0]:12.5f}{tau:12.5f}")
        out.append("")
        out.append(f" --------  End {t2}  " + "-" * max(10, 133 - len(t2) - 18))
    return "\n".join(out) + "\n"


def read_record(path: Path):
    """The event and, when present, the vertex table (row -> (t, x, y, z), row -> tau)."""
    ev = had.read_record(path)
    vertex, tau = {}, {}
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) == 6 and f[0].isdigit():
            try:
                x, y, z, t, ta = (float(v) for v in f[1:])
            except ValueError:
                continue
            vertex[int(f[0])] = np.array([t, x, y, z])
            if ta > 0:
                tau[int(f[0])] = ta
    return ev, vertex, tau


# ---------------------------------------------------------------------------
# what the chapter looks at
# ---------------------------------------------------------------------------
def descendants(ev: Event, row: int):
    out = []
    stack = [row]
    while stack:
        r = stack.pop()
        q = ev.p[r]
        if q.d1 > 0:
            for d in range(q.d1, max(q.d1, q.d2) + 1):
                out.append(d)
                stack.append(d)
    return out


def string_b_hadrons(ev: Event):
    """pid of the hard quark -> row of the primary hadron that carries its flavour (chapter 5's B* mesons)."""
    out = {}
    for pid, jb in had.b_quark_rows(ev).items():
        copy = ev.p[jb].d1
        cands = [i for i, q in enumerate(ev.p) if abs(q.status) in (81, 82, 83, 84) and q.m1 <= copy <= max(q.m1, q.m2)]
        want = (lambda h: (h < 0) if abs(h) < 1000 else (h > 0)) if pid > 0 else (lambda h: (h > 0) if abs(h) < 1000 else (h < 0))
        bh = [i for i in cands if heavy_content(ev.p[i].id) == 5 and want(ev.p[i].id)]
        if not bh:
            bh = [i for i, q in enumerate(ev.p) if abs(q.status) in (81, 82, 83, 84) and heavy_content(q.id) == 5 and want(q.id)]
        if bh:
            out[pid] = min(bh, key=lambda i: ps.delta_r(ev.p[i].p, ev.p[jb].p))
    return out


def weak_b_hadron(ev: Event, row: int):
    """From the string's B* (or B) down to the b hadron with a lifetime, following the b flavour."""
    r = row
    while not is_weak(ev.p[r].id) and ev.p[r].d1 > 0:
        ds = [d for d in range(ev.p[r].d1, max(ev.p[r].d1, ev.p[r].d2) + 1) if heavy_content(ev.p[d].id) == 5]
        if not ds:
            break
        r = ds[0]
    return r


def cone_members(ev: Event, rows, axis, R):
    return [i for i in rows if ps.delta_r(ev.p[i].p, axis) < R]


def cone_kin(ev: Event, rows):
    p = sum((ev.p[i].p for i in rows), np.zeros(4))
    return dict(n=len(rows), pt=pt(p), m=math.sqrt(max(mass2(p), 0.0)), e=float(p[0]),
                n_charged=sum(1 for i in rows if is_charged(ev.p[i].id)))


def impact_parameter(v, p):
    """Transverse impact parameter (mm) of a straight track from vertex v = (t, x, y, z) with momentum p:
    the distance of closest approach to the beam line in the transverse plane, signed by (x x p) . z."""
    ptv = math.hypot(p[1], p[2])
    if ptv <= 0:
        return 0.0
    return (v[1] * p[2] - v[2] * p[1]) / ptv


def summarise(ev: Event, dec: Decayer, radius=0.4) -> dict:
    fin = ev.final_indices()
    primaries = dec.primaries
    s = dict(n_rows=len(ev.p), n_primaries=len(primaries), n_decays=len(dec.decays), n_oscillated=dec.n_osc,
             n_final=len(fin), n_charged=sum(1 for i in fin if is_charged(ev.p[i].id)),
             n_photons=sum(1 for i in fin if ev.p[i].id == 22), n_neutrinos=sum(1 for i in fin if is_neutrino(ev.p[i].id)),
             n_electrons=sum(1 for i in fin if abs(ev.p[i].id) == 11), n_muons=sum(1 for i in fin if abs(ev.p[i].id) == 13),
             n_hadrons_final=sum(1 for i in fin if is_hadron(ev.p[i].id)),
             n_primary_stable=sum(1 for i in primaries if ev.p[i].status > 0),
             n_final_central=sum(1 for i in fin if abs(eta(ev.p[i].p)) < 2.5),
             n_charged_central=sum(1 for i in fin if abs(eta(ev.p[i].p)) < 2.5 and is_charged(ev.p[i].id)),
             e_neutrinos=float(sum(ev.p[i].p[0] for i in fin if is_neutrino(ev.p[i].id))),
             species={}, b={})
    p_nu = sum((ev.p[i].p for i in fin if is_neutrino(ev.p[i].id)), np.zeros(4))
    s["pt_neutrinos"] = pt(p_nu)
    for i in fin:
        n = name(ev.p[i].id)
        s["species"][n] = s["species"].get(n, 0) + 1
    s["n_from_decays"] = sum(1 for i in fin if ev.p[i].status >= 91)
    s["classes"] = count_ids(d["cls"] or "hadronic" for d in dec.decays if heavy_content(d["id"]) == 5)
    bq = had.b_quark_rows(ev)
    bstr = string_b_hadrons(ev)
    for pid, jb in bq.items():
        if pid not in bstr:
            continue
        qb = ev.p[jb]
        r0 = bstr[pid]
        rw = weak_b_hadron(ev, r0)
        hw = ev.p[rw]
        desc = descendants(ev, rw)
        fin_desc = [d for d in desc if ev.p[d].status > 0]
        drec = next((d for d in dec.decays if d["row"] == rw), None)
        gb = math.sqrt(max(hw.p[0] ** 2 - hw.m ** 2, 0)) / hw.m
        tau = dec.tau.get(rw, 0.0)
        v_dec = drec["v_dec"] if drec else dec.vertex.get(rw, np.zeros(4))
        nus = [d for d in fin_desc if is_neutrino(ev.p[d].id)]
        p_nu = sum((ev.p[d].p for d in nus), np.zeros(4))
        leps = [d for d in fin_desc if abs(ev.p[d].id) in (11, 13)]
        lep = max(leps, key=lambda d: pt(ev.p[d].p)) if leps else None
        charm = [d for d in desc if heavy_content(ev.p[d].id) == 4 and is_weak(ev.p[d].id)]
        rc = charm[0] if charm else None
        # the chain of the b flavour: B* -> B -> (D) ...
        chain = []
        r = r0
        while True:
            chain.append(r)
            q = ev.p[r]
            if q.d1 <= 0:
                break
            ds = [d for d in range(q.d1, max(q.d1, q.d2) + 1)]
            nxt = [d for d in ds if heavy_content(ev.p[d].id) == 5] or [d for d in ds if heavy_content(ev.p[d].id) == 4]
            if not nxt:
                break
            r = nxt[0]
        tracks = [d for d in fin_desc if is_charged(ev.p[d].id)]
        d0 = [abs(impact_parameter(dec.vertex.get(d, np.zeros(4)), ev.p[d].p)) for d in tracks]
        rec = dict(quark=had.pname(pid), quark_row=jb, pt_quark=pt(qb.p), e_quark=float(qb.p[0]),
                   string_hadron=name(ev.p[r0].id), string_row=r0, pt_string=pt(ev.p[r0].p), e_string=float(ev.p[r0].p[0]),
                   hadron=name(hw.id), hadron_id=int(hw.id), hadron_row=rw, pt_hadron=pt(hw.p), e_hadron=float(hw.p[0]), m_hadron=hw.m,
                   gamma=float(hw.p[0] / hw.m), gamma_beta=gb, tau=tau, tau_ps=lifetime_ps(tau), tau0=tau0(hw.id),
                   tau_over_tau0=(tau / tau0(hw.id) if tau0(hw.id) > 0 else 0.0),
                   L=float(np.linalg.norm(v_dec[1:])), Lxy=float(math.hypot(v_dec[1], v_dec[2])), vertex=[float(v) for v in v_dec[1:]],
                   oscillated=bool(drec and drec["oscillated"]), decays=bool(drec),
                   channel=(" ".join(name(i) for i in drec["ids"]) if drec else ""), br=(drec["br"] if drec else 0.0),
                   me=(drec["me"] if drec else -1), cls=(drec["cls"] or "hadronic") if drec else "stable",
                   partonic=bool(drec and any(is_parton(p) for p in drec["prods"])),
                   n_direct=(len(drec["ids"]) if drec else 0), n_final=len(fin_desc),
                   n_charged_final=sum(1 for d in fin_desc if is_charged(ev.p[d].id)),
                   n_photons_final=sum(1 for d in fin_desc if ev.p[d].id == 22),
                   n_neutrinos=len(nus), e_nu=float(p_nu[0]), pt_nu=pt(p_nu), e_nu_frac=float(p_nu[0] / hw.p[0]),
                   lepton=(name(ev.p[lep].id) if lep is not None else ""), pt_lepton=(pt(ev.p[lep].p) if lep is not None else 0.0),
                   lepton_from_b=(lep is not None and ev.p[lep].m1 == rw),
                   charm=(name(ev.p[rc].id) if rc is not None else ""), pt_charm=(pt(ev.p[rc].p) if rc is not None else 0.0),
                   L_charm=(float(np.linalg.norm(dec.decays[[d["row"] for d in dec.decays].index(rc)]["v_dec"][1:] - dec.vertex.get(rc, np.zeros(4))[1:]))
                            if rc is not None and rc in [d["row"] for d in dec.decays] else 0.0),
                   chain=[name(ev.p[r].id) for r in chain], chain_rows=chain,
                   n_tracks=len(tracks), d0=d0, max_d0=(max(d0) if d0 else 0.0), n_tracks_d0_100um=sum(1 for x in d0 if x > 0.1),
                   cones={})
        if drec and drec["me"] in (22, 23) and rec["cls"].startswith("semi"):
            rec["lepton_from_b"] = True
        # the "b system" of CMSSW: the vector sum of the B hadron's final descendants
        p_sys = sum((ev.p[d].p for d in fin_desc), np.zeros(4))
        p_vis = p_sys - p_nu
        rec["system"] = dict(n=len(fin_desc), pt=pt(p_sys), eta=eta(p_sys) if pt(p_sys) > 0 else 0.0, phi=phi(p_sys), m=math.sqrt(max(mass2(p_sys), 0)),
                             pt_visible=pt(p_vis), m_visible=math.sqrt(max(mass2(p_vis), 0)))
        parton_rows = [i for i, q in enumerate(ev.p) if abs(q.status) in (62, 63)]
        for R in (0.4, 0.8):
            mem_p = cone_members(ev, parton_rows, qb.p, R)
            mem_h = cone_members(ev, primaries, qb.p, R)
            mem_f = cone_members(ev, fin, qb.p, R)
            mem_v = [i for i in mem_f if not is_neutrino(ev.p[i].id)]
            mem_c = [i for i in mem_f if is_charged(ev.p[i].id)]
            mem_b = [i for i in mem_f if i in set(fin_desc)]
            c = dict(partons=cone_kin(ev, mem_p), primaries=cone_kin(ev, mem_h), final=cone_kin(ev, mem_f),
                     visible=cone_kin(ev, mem_v), charged=cone_kin(ev, mem_c), from_b=cone_kin(ev, mem_b))
            c["frac_B"] = c["from_b"]["pt"] / c["final"]["pt"] if c["final"]["pt"] > 0 else 0.0
            c["nu_pt_in_cone"] = pt(sum((ev.p[i].p for i in mem_f if is_neutrino(ev.p[i].id)), np.zeros(4)))
            c["n_b_outside"] = sum(1 for i in fin_desc if i not in set(mem_f))
            rec["cones"][R] = c
        s["b"][had.pname(pid)] = rec
    return s


def print_summary(s: dict):
    print(f"  {s['n_primaries']} primary hadrons -> {s['n_decays']} decays ({s['n_oscillated']} B oscillations) -> {s['n_final']} final particles: "
          f"{s['n_charged']} charged, {s['n_photons']} photons, {s['n_neutrinos']} neutrinos ({s['e_neutrinos']:.1f} GeV, "
          f"|sum pT| = {s['pt_neutrinos']:.2f} GeV), {s['n_electrons']} electrons, {s['n_muons']} muons, {s['n_hadrons_final']} hadrons; "
          f"{s['n_primary_stable']} primary hadrons survive, {s['n_from_decays']} particles come from decays; "
          f"{s['n_final_central']} final particles ({s['n_charged_central']} charged) at |eta| < 2.5")
    top = sorted(s["species"].items(), key=lambda kv: -kv[1])[:12]
    print("  most common: " + ", ".join(f"{n} {k}" for n, k in top))
    for nm, r in s["b"].items():
        print(f"  {nm} (row {r['quark_row']}, pT = {r['pt_quark']:.2f}) -> {r['string_hadron']} (row {r['string_row']}, pT = {r['pt_string']:.2f}) -> "
              + " -> ".join(r["chain"][1:]) )
        if r["decays"]:
            print(f"     {r['hadron']} (row {r['hadron_row']}): pT = {r['pt_hadron']:.2f}, E = {r['e_hadron']:.2f} GeV, gamma = {r['gamma']:.1f}; "
                  f"tau = {r['tau_ps']:.3f} ps = {r['tau_over_tau0']:.2f} tau0 -> L = {r['L']:.2f} mm (L_xy = {r['Lxy']:.2f}), "
                  f"vertex ({r['vertex'][0]:.2f}, {r['vertex'][1]:.2f}, {r['vertex'][2]:.2f}) mm{'; oscillated' if r['oscillated'] else ''}")
            print(f"     channel: {r['hadron']} -> {r['channel']} (BR {r['br']:.4f}, meMode {r['me']}, {r['cls']}{', partonic' if r['partonic'] else ''}); "
                  f"{r['n_direct']} direct products, {r['n_final']} final particles ({r['n_charged_final']} charged, {r['n_photons_final']} photons, "
                  f"{r['n_neutrinos']} neutrinos with {r['e_nu']:.2f} GeV = {100 * r['e_nu_frac']:.1f}% of the B energy)"
                  + (f"; lepton {r['lepton']} pT = {r['pt_lepton']:.2f}" + (" from the B" if r["lepton_from_b"] else " from the cascade") if r["lepton"] else "")
                  + (f"; charm hadron {r['charm']} pT = {r['pt_charm']:.2f}, flies {r['L_charm']:.2f} mm" if r["charm"] else ""))
            print(f"     {r['n_tracks']} charged tracks from the B: largest impact parameter {1e3 * r['max_d0']:.0f} um, {r['n_tracks_d0_100um']} above 100 um; "
                  f"the b system: {r['system']['n']} particles, pT = {r['system']['pt']:.2f} GeV (visible {r['system']['pt_visible']:.2f})")
        else:
            print(f"     {r['hadron']} (row {r['hadron_row']}) is kept stable")
        for R in (0.4, 0.8):
            c = r["cones"][R]
            print(f"     cone R = {R} around the quark: partons pT {c['partons']['pt']:.2f} -> {c['primaries']['n']} primary hadrons pT {c['primaries']['pt']:.2f} -> "
                  f"{c['final']['n']} final particles pT {c['final']['pt']:.2f} (m {c['primaries']['m']:.2f} -> {c['final']['m']:.2f}), "
                  f"visible {c['visible']['n']} with pT {c['visible']['pt']:.2f}, charged {c['charged']['n']} with pT {c['charged']['pt']:.2f}; "
                  f"neutrino pT in the cone {c['nu_pt_in_cone']:.2f}; the B's products carry {100 * c['frac_B']:.0f}%, {c['n_b_outside']} of them left the cone")


def explain_setup(args):
    say = print
    say(banner("setting the scene: from primary hadrons to what the detector sees"))
    say(f"  decay tables: {TABLES['source']}: {len(PART)} particles, {sum(len(p['channels']) for p in PART.values())} channels")
    say(f"  a particle decays if it has a table, mayDecay (PYTHIA: tau0 < 1000 mm) and tau0 < tau0Max = {args.tau0max} mm" + ("" if args.limit_tau0 else " (limit off)")
        + (", B hadrons kept stable" if args.stable_b else ""))
    say(f"  stable for the detector: " + ", ".join(f"{name(a)} ({PART[a]['tau0']:.3g} mm)" for a in (211, 321, 130, 310, 3122, 3222, 3112, 3312, 3322, 3334, 2112, 13)))
    say(f"  lifetimes: c tau0 = " + ", ".join(f"{name(a)} {PART[a]['tau0']:.4f} mm ({lifetime_ps(PART[a]['tau0']):.3f} ps)" for a in (511, 521, 531, 5122, 421, 411, 431, 4122, 15)))
    say(f"  B mixing {'on' if MIX_B and not args.no_mix else 'off'}: x_d = {X_BD}, x_s = {X_BS}; time-integrated P(oscillated) = x^2/(2(1 + x^2)) = "
        f"{X_BD ** 2 / (2 * (1 + X_BD ** 2)):.3f} (B0), {X_BS ** 2 / (2 * (1 + X_BS ** 2)):.3f} (B_s0)")
    for a in (511, 521, 5122, 421):
        chans, total = open_channels(a)
        cls = {}
        for br, me, prods in chans:
            c = channel_class(prods, me) or ("partonic" if any(is_parton(p) for p in prods) else "hadronic (explicit)")
            cls[c] = cls.get(c, 0) + br
        say(f"  {name(a)}: {len(chans)} open channels, BR sum {total:.4f}: " + ", ".join(f"{k} {100 * v / total:.1f}%" for k, v in sorted(cls.items(), key=lambda kv: -kv[1])))
    say(f"  partonic channels: hadrons from a Poissonian with mean n_known/2 + n_partons/4 + {MULT_INCREASE:g} ln(mDiff/{MULT_REF_MASS}) "
        f"({MULT_INCREASE_WEAK:g} for the weak meMode 23), colour rearrangement with P = {COL_REARRANGE}; semileptonic V-A: |M|^2 ~ (p_B.p_nu)(p_X.p_l), "
        f"soft damping sigma = {SIGMA_SOFT} GeV beyond three bodies")


def run(args, seed: int, verbose=0):
    ev, vertex0, tau0_ = read_record(args.input)
    rng = np.random.default_rng(seed)
    narrate = list(string_b_hadrons(ev).values())
    dec = Decayer(ev, rng, args, verbose=verbose, narrate_rows=narrate)
    dec.vertex.update(vertex0)
    if verbose:
        explain_setup(args)
    dec.run()
    dev, psum = dec.verify()
    if verbose:
        print(banner("checks"))
        fin = ev.final_indices()
        print(f"  final state: {len(fin)} particles; their momentum sum deviates from that of the primary hadrons by {dev:.2e} GeV; "
              f"charge {sum(charge3(ev.p[i].id) for i in fin) / 3:.1f}")
        for n in dec.notes:
            print("  " + n)
        print("  final event record (also written to the output file):")
        print(listing(ev, "Standalone Event Listing after hadron decays", dec).rstrip())
    return ev, dec


# ---------------------------------------------------------------------------
# a single particle at rest, many times (the decay table in action; used by the validation)
# ---------------------------------------------------------------------------
def single(args):
    pid = args.single
    n = max(args.repeat, 1)
    rng = np.random.default_rng(args.seed)
    acc = dict(n_final=[], n_charged=[], n_photons=[], n_neutrinos=[], e_lepton=[], e_nu=[], cls=[], n_direct=[], L=[], charm=[], n_kaons=[], n_pions=[],
               p_lepton=[], e_lepton_direct=[])
    for k in range(n):
        ev = Event([Particle(pid, 81, 0, 0, 0, 0, 0, 0, [m0(pid), 0.0, 0.0, 0.0], m0(pid))], 6800.0, 0, 0)
        dec = Decayer(ev, rng, args, verbose=(args.verbose if k == 0 else 0), narrate_rows=[0])
        dec.run()
        fin = ev.final_indices()
        d = dec.decays[0] if dec.decays else None
        acc["n_final"].append(len(fin))
        acc["n_charged"].append(sum(1 for i in fin if is_charged(ev.p[i].id)))
        acc["n_photons"].append(sum(1 for i in fin if ev.p[i].id == 22))
        acc["n_neutrinos"].append(sum(1 for i in fin if is_neutrino(ev.p[i].id)))
        acc["n_kaons"].append(sum(1 for i in fin if abs(ev.p[i].id) in (321, 130, 310)))
        acc["n_pions"].append(sum(1 for i in fin if abs(ev.p[i].id) == 211))
        acc["e_nu"].append(sum(ev.p[i].p[0] for i in fin if is_neutrino(ev.p[i].id)))
        leps = [i for i in fin if abs(ev.p[i].id) in (11, 13)]
        acc["e_lepton"].append(max((ev.p[i].p[0] for i in leps), default=0.0))
        direct = [i for i in (range(d["rows"][0], d["rows"][-1] + 1) if d else []) if abs(ev.p[i].id) in (11, 13)]
        acc["e_lepton_direct"].append(ev.p[direct[0]].p[0] if direct else float("nan"))
        acc["cls"].append((d["cls"] or "hadronic") if d else "stable")
        acc["n_direct"].append(len(d["ids"]) if d else 0)
        acc["L"].append(d["tau"] if d else 0.0)
        charm = [i for i in range(len(ev.p)) if heavy_content(ev.p[i].id) == 4 and is_weak(ev.p[i].id)]
        acc["charm"].append(name(ev.p[charm[0]].id) if charm else "")
    cls = count_ids(acc["cls"])
    print(f"{name(pid)} at rest decayed {n} times (seed {args.seed}):")
    for k in ("n_final", "n_charged", "n_photons", "n_neutrinos", "n_kaons", "n_pions", "n_direct", "e_nu", "e_lepton"):
        v = np.asarray(acc[k], dtype=float)
        print(f"  {k:12s} mean = {v.mean():8.3f}  std = {v.std():7.3f}")
    v = np.asarray(acc["e_lepton_direct"], dtype=float)
    print(f"  direct semileptonic decays: {np.sum(~np.isnan(v))} ({100 * np.mean(~np.isnan(v)):.2f}%), <E_lepton> = {np.nanmean(v):.3f} GeV in the rest frame")
    print("  classes: " + ", ".join(f"{k} {100 * v / n:.2f}%" for k, v in sorted(cls.items(), key=lambda kv: -kv[1])))
    ch = count_ids(acc["charm"])
    print("  charm hadron: " + ", ".join(f"{k or 'none'} {100 * v / n:.1f}%" for k, v in sorted(ch.items(), key=lambda kv: -kv[1])[:8]))
    if args.history:
        args.history.write_text(json.dumps(dict(id=pid, n=n, seed=args.seed, **{k: v for k, v in acc.items()}), default=float) + "\n")
        print(f"wrote {args.history}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=HERE.parent.parent / "Had" / "event_Had.lhe")
    ap.add_argument("-o", "--output", type=Path, default=HERE.parent / "event_Decay.lhe")
    ap.add_argument("--history", type=Path, help="write the summary (and with --repeat the per-event statistics) as JSON")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tau0max", type=float, default=TAU0_MAX, help="ParticleDecays:tau0Max [mm]")
    ap.add_argument("--no-limit-tau0", dest="limit_tau0", action="store_false", help="decay every particle with a table (K_S0, Lambda, ...)")
    ap.add_argument("--no-mix", action="store_true", help="no B0 / B_s0 oscillations")
    ap.add_argument("--stable-b", action="store_true", help="keep the b hadrons stable (everything else decays)")
    ap.add_argument("--radius", type=float, default=0.4)
    ap.add_argument("--repeat", type=int, default=0, help="decay the event N times (seeds seed..seed+N-1) and print averages")
    ap.add_argument("--single", type=int, default=None, help="decay this particle at rest --repeat times instead (e.g. 511)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("-v", "--verbose", action="count", default=0,
                    help="learning mode: -v narrates the decay chains of the two B hadrons, -vv every decay")
    args = ap.parse_args()
    if args.single is not None:
        single(args)
        return

    ev, dec = run(args, args.seed, verbose=args.verbose)
    summary = summarise(ev, dec, args.radius)
    if args.verbose:
        print(banner("summary"))
    if not args.quiet:
        print(f"input {args.input}: {len(ev.p)} rows after the decays; seed = {args.seed}")
        print_summary(summary)
    title = "Standalone Event Listing  (hadron decays: hard process + showers + MPI + beam remnants + colour reconnection + hadronisation + decays)"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(listing(ev, title, dec))
    print(f"wrote {args.output} ({len(ev.p)} rows, {summary['n_final']} final particles)")
    if args.history:
        args.history.write_text(json.dumps(dict(seed=args.seed, summary=summary), indent=1, default=float) + "\n")
        print(f"wrote {args.history}")

    if args.repeat > 1:
        keys = ["n_final", "n_charged", "n_photons", "n_neutrinos", "n_electrons", "n_muons", "n_hadrons_final", "n_primary_stable",
                "n_final_central", "n_charged_central", "e_neutrinos", "pt_neutrinos", "n_decays", "n_oscillated"]
        acc = {k: [] for k in keys}
        fam, species, classes = {}, {}, {}
        bkeys = ["tau_ps", "tau_over_tau0", "L", "Lxy", "n_direct", "n_final", "n_charged_final", "n_photons_final", "n_neutrinos", "e_nu", "pt_nu", "e_nu_frac",
                 "pt_lepton", "pt_charm", "L_charm", "n_tracks", "max_d0", "n_tracks_d0_100um", "gamma"]
        ckeys = ["pt", "m", "n", "n_charged"]
        for n in range(args.repeat):
            e2, d2 = run(args, args.seed + n, verbose=0)
            s = summarise(e2, d2, args.radius)
            for k in keys:
                acc[k].append(s[k])
            for nm, k in s["species"].items():
                species[nm] = species.get(nm, 0) + k
            for nm, r in s["b"].items():
                d = fam.setdefault(nm, dict({k: [] for k in bkeys}, cls=[], hadron=[], lepton=[], lepton_from_b=[], oscillated=[], channel=[], charm=[], d0=[],
                                            system_pt=[], system_pt_visible=[],
                                            cones={R: {c: {k: [] for k in ckeys} for c in ("primaries", "final", "visible", "charged", "from_b")} for R in (0.4, 0.8)}))
                for k in bkeys:
                    d[k].append(r[k])
                d["cls"].append(r["cls"])
                d["hadron"].append(r["hadron"])
                d["lepton"].append(r["lepton"])
                d["lepton_from_b"].append(bool(r["lepton_from_b"]))
                d["oscillated"].append(bool(r["oscillated"]))
                d["channel"].append(r["channel"])
                d["charm"].append(r["charm"])
                d["d0"] += r["d0"]
                d["system_pt"].append(r["system"]["pt"])
                d["system_pt_visible"].append(r["system"]["pt_visible"])
                for R in (0.4, 0.8):
                    for c in ("primaries", "final", "visible", "charged", "from_b"):
                        for k in ckeys:
                            d["cones"][R][c][k].append(r["cones"][R][c][k])
                    d["cones"][R].setdefault("frac_B", []).append(r["cones"][R]["frac_B"])
                    d["cones"][R].setdefault("nu_pt_in_cone", []).append(r["cones"][R]["nu_pt_in_cone"])
        ndone = len(acc["n_final"])
        print(f"averages over {ndone} decays of the same {summary['n_primaries']} primary hadrons (seeds {args.seed}..{args.seed + args.repeat - 1}):")
        for k, v in acc.items():
            v = np.asarray(v, dtype=float)
            print(f"  {k:18s} mean = {v.mean():9.3f}  std = {v.std():8.3f}")
        for nm, d in fam.items():
            cls = count_ids(d["cls"])
            print(f"  {nm}: hadrons " + ", ".join(f"{h} {100 * k / ndone:.0f}%" for h, k in sorted(count_ids(d['hadron']).items(), key=lambda kv: -kv[1]))
                  + f"; oscillated {100 * np.mean(d['oscillated']):.1f}%")
            print(f"     tau = {np.mean(d['tau_ps']):.3f} +- {np.std(d['tau_ps']):.3f} ps, L = {np.mean(d['L']):.2f} +- {np.std(d['L']):.2f} mm "
                  f"(L_xy {np.mean(d['Lxy']):.2f}), gamma = {np.mean(d['gamma']):.1f}; {np.mean(d['n_direct']):.2f} direct products, "
                  f"{np.mean(d['n_final']):.1f} +- {np.std(d['n_final']):.1f} final ({np.mean(d['n_charged_final']):.1f} charged, {np.mean(d['n_neutrinos']):.2f} neutrinos "
                  f"with {np.mean(d['e_nu']):.2f} GeV = {100 * np.mean(d['e_nu_frac']):.1f}% of the B energy)")
            print("     classes: " + ", ".join(f"{k} {100 * v / ndone:.1f}%" for k, v in sorted(cls.items(), key=lambda kv: -kv[1]))
                  + f"; a lepton (e or mu) among the final particles in {100 * np.mean([bool(x) for x in d['lepton']]):.1f}% "
                  f"(from the B itself {100 * np.mean(d['lepton_from_b']):.1f}%), pT = {np.mean([x for x in d['pt_lepton'] if x > 0] or [0]):.2f} GeV")
            print(f"     tracks: {np.mean(d['n_tracks']):.1f} per B, {np.mean(d['n_tracks_d0_100um']):.1f} with d0 > 100 um, largest d0 {1e3 * np.mean(d['max_d0']):.0f} um; "
                  f"median d0 {1e3 * np.median(d['d0']):.0f} um")
            for R in (0.4, 0.8):
                c = d["cones"][R]
                print(f"     cone R = {R}: primaries pT {np.mean(c['primaries']['pt']):.2f} -> final {np.mean(c['final']['pt']):.2f} +- {np.std(c['final']['pt']):.2f}, "
                      f"visible {np.mean(c['visible']['pt']):.2f} +- {np.std(c['visible']['pt']):.2f}, charged {np.mean(c['charged']['pt']):.2f}; "
                      f"n {np.mean(c['primaries']['n']):.1f} -> {np.mean(c['final']['n']):.1f} ({np.mean(c['charged']['n']):.1f} charged); "
                      f"m {np.mean(c['primaries']['m']):.2f} -> {np.mean(c['final']['m']):.2f}; B share {np.mean(c['frac_B']):.3f}")
        if args.history:
            stats = dict(repeat=args.repeat, seed0=args.seed, counts=acc, species=species, b=fam,
                         params=dict(tau0max=args.tau0max, limit_tau0=args.limit_tau0, no_mix=args.no_mix, stable_b=args.stable_b))
            out = args.history.with_suffix(".stats.json")
            out.write_text(json.dumps(stats, default=float) + "\n")
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
