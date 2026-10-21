#!/usr/bin/env python3
"""Standalone generator-level jets (GenJets) and jet-flavour definitions for the b bbar event of chapter 6.

Dependencies: numpy, plus the classes of chapter 6's ../../Decay/Standalone/standalone_hadron_decays.py
(the event record, the particle names and properties, the vertex table; chapter 5 and chapter 2 are
imported through it). No FastJet, no CMSSW: the sequential-recombination algorithms, the ghost
association and the infrared-safe flavour definitions are written out here, step by step.

Input is the final state after the hadron decays (../../Decay/event_Decay.lhe: the 749 rows with a
positive status are what the detector will see, the "genParticles with status 1" of CMSSW). Output is
the same record followed by two more tables (../event_GenJet.lhe): the jets of every collection with
their kinematics and their flavours under each definition, and one row per particle saying which jet
holds it. This is the family photograph: the reference the reconstructed jets of chapters 15 to 21
will be compared against.

What is implemented (the physics of chapter 7, following CMSSW's GenJet and JetFlavour producers)
-------------------------------------------------------------------------------------------------
* The input particles (RecoJets/Configuration GenJetParticles_cff): every final particle except the
  ones in ignoreParticleIDs (neutralinos, gravitinos, ... none in this event) for genParticlesForJets,
  and in addition no neutrinos (12, 14, 16) for genParticlesForJetsNoNu. No eta cut, no pT cut.
* Sequential recombination (the inn of chapter 15, used here in its generator-level form): the
  generalised kT family with
        d_ij = min(pT_i^2p, pT_j^2p) dR_ij^2 / R^2,     d_iB = pT_i^2p,     dR^2 = dy^2 + dphi^2,
  p = -1 anti-kT (the GenJets), p = 0 Cambridge/Aachen, p = 1 kT; four-momenta are added (the E
  scheme, as FastJet and CMSSW do), rapidity y is used in the distance (not pseudorapidity), and the
  inclusive jets with pT > 3 GeV are kept (GenJetParameters.jetPtMin = 3). The collections are the
  CMSSW ones: ak4GenJets (R = 0.4, with neutrinos), ak4GenJetsNoNu (R = 0.4, without) and ak8GenJets
  (R = 0.8, with). The comparison with FastJet and with PYTHIA's SlowJet on the same particles is in
  validate_genjets_with_fastjet.py; chapter 15 derives the algorithms.
* Ghost association (PhysicsTools/JetMCAlgos: HadronAndPartonSelector + JetFlavourClustering): the
  last b hadrons and the last c hadrons of every decay chain, the partons handed to the hadronisation
  (PYTHIA status 71-79, here the rows the strings were built from) and the leptons from heavy-hadron
  decays are added to the particle list with their momentum scaled by ghostRescaling = 1e-18, so that
  they follow the clustering without changing a single jet momentum. Then
    hadronFlavour  = 5 if a b-hadron ghost is in the jet, else 4 if a c-hadron ghost is, else 0;
    partonFlavour  = the id of the hardest b-parton ghost in the jet, else of the hardest c parton,
                     else of the hardest parton (a gluon or a light quark), 0 if none;
    physics flavour: the same rule with the hard-process partons (status 23) as ghosts;
    dR < 0.3       : the legacy matching of Run 1 (the hardest parton within 0.3 of the axis, b and c first).
* Net flavour and its infrared problem: a jet's flavour as the sum of the flavours of its constituents
  (+1 for a b quark or a hadron containing b, -1 for a bbar), counted with the b and c hadrons kept
  undecayed (their final descendants replaced by the hadron, as in the hadron-level tests of the
  flavour papers). A soft g -> b bbar whose two hadrons fall into two different jets flips both labels:
  the definition is not infrared safe, and neither is hadronFlavour.
* Flavour-kT (Banfi, Salam, Zanderighi 2006): the kT algorithm with the flavour-aware distances
        d_ij = dR_ij^2/R^2 x { max(kT_i, kT_j)^alpha min(kT_i, kT_j)^(2-alpha)  if the softer is flavoured,
                              min(kT_i, kT_j)^2                                 otherwise },
        d_iB = the same with kT_B(y_i) = sum_j kT_j [Theta(y_j - y_i) + Theta(y_i - y_j) e^(y_j - y_i)]
  (and d_iBbar with the mirror sum), alpha = 2 by default; the jets are different jets, kT-like.
* Interleaved flavour neutralisation (IFN; Caola, Grabarczyk, Hutt, Salam, Scyboz, Thaler 2023): the
  anti-kT clustering unchanged, and before every recombination i + j in which i (or j) carries flavour,
  the flavour of i is neutralised against the pseudojets k whose neutralisation distance
        u_ik = max(pT_i, pT_k)^alpha min(pT_i, pT_k)^(2-alpha) Omega_ik^2,
        Omega_ik^2 = 2 [ (cosh(omega dy) - 1) / omega^2 - (cos dphi - 1) ]
  is smaller than u_ij, closest first, each candidate first neutralised against its own closer partners
  (the recursion of the paper); alpha = 2, omega = 1 for anti-kT. The jets are the anti-kT jets; only
  the labels change. The candidates k are all pseudojets alive at that step, those already declared jets
  included (an assumption of this port: the FastJet contrib IFNPlugin is the reference implementation
  and could not be run here).

What is left out, on purpose: the CMSSW ghost "leptons" collection is built but not used for a label;
the sign conventions of partonFlavour for a jet with a b and a bbar parton (CMSSW keeps the harder
one, so do we); jet areas (chapter 15); the flavour-dressing and CMP variants of the infrared-safe
definitions (the further reading of chapter 7).

Learning mode (--verbose / -v, -vv)
------------------------------------
  -v   narrates the anti-kT clustering of the b quark's jet step by step (each merge with its d_ij,
       dR and the pT of the growing pseudojet), the ghosts that landed in it, every flavour decision
       for the two b jets, the IFN neutralisations of the event and the flavour-kT jets.
  -vv  narrates every merge of the ak4 clustering and every IFN candidate.

Examples
--------
  python3 standalone_genjets.py                       # -> ../event_GenJet.lhe (the book's record)
  python3 standalone_genjets.py -v                    # learning mode
  python3 standalone_genjets.py --algorithm kt        # the kT jets instead of anti-kT (also ca)
  python3 standalone_genjets.py --ptmin 10            # a higher jet threshold
  python3 standalone_genjets.py --soft-pair 1.0       # add a soft g -> b bbar pair between two light jets
  python3 standalone_genjets.py --repeat 400 --quiet --history standalone_genjet_stats.json
                                                      # decay the chapter-5 hadrons again and again, cluster each time

All energies in GeV, angles in radians, lengths in mm.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "Decay" / "Standalone"))
import standalone_hadron_decays as dec  # noqa: E402  (chapter 6: the record, names, properties; had and ps through it)

had, ps = dec.had, dec.ps
Event, Particle = dec.Event, dec.Particle
name, heavy_content, is_charged, is_neutrino, is_lepton, is_hadron, is_parton = (
    dec.name, dec.heavy_content, dec.is_charged, dec.is_neutrino, dec.is_lepton, dec.is_hadron, dec.is_parton)
pt, eta, phi, mass2, banner = had.pt, had.eta, had.phi, had.mass2, had.banner

# ---------------------------------------------------------------------------
# the CMSSW configuration this chapter follows
# ---------------------------------------------------------------------------
JET_PT_MIN = 3.0                 # GenJetParameters.jetPtMin
R_AK4, R_AK8 = 0.4, 0.8          # ak4GenJets, ak8GenJets rParam
NEUTRINOS = (12, 14, 16)
IGNORE_IDS = {1000022, 1000012, 1000014, 1000016, 2000012, 2000014, 2000016, 1000039, 5100039,
              4000012, 4000014, 4000016, 9900012, 9900014, 9900016, 39}   # genParticlesForJets.ignoreParticleIDs
GHOST_RESCALING = 1e-18          # JetFlavourClustering.ghostRescaling
DR_LEGACY = 0.3                  # JetFlavourIdentifier.coneSizeToAssociate (Run 1)
PARTON_STATUS = range(71, 80)    # Pythia8PartonSelector: partons with status 71-79
IFN_ALPHA, IFN_OMEGA = 2.0, 1.0  # IFN with anti-kT: alpha = 2, omega = 3 - alpha
FKT_ALPHA = 2.0                  # flavour-kT alpha (1 or 2 in the paper)
COLLECTIONS = {"ak4": dict(R=R_AK4, nonu=False, title="ak4GenJets"),
               "ak4nonu": dict(R=R_AK4, nonu=True, title="ak4GenJetsNoNu"),
               "ak8": dict(R=R_AK8, nonu=False, title="ak8GenJets")}
POWER = {"antikt": -1, "ca": 0, "kt": 1}


# ---------------------------------------------------------------------------
# kinematics
# ---------------------------------------------------------------------------
def rapidity(p) -> float:
    """y = 1/2 ln((E + pz)/(E - pz)), with FastJet's guard for particles along the beam."""
    E, pz = float(p[0]), float(p[3])
    m2 = max(float(p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2), 0.0)
    pt2 = float(p[1] ** 2 + p[2] ** 2)
    if pt2 + m2 <= 0:
        return 1e5 if pz > 0 else -1e5
    if pz >= 0:
        return 0.5 * math.log((E + pz) ** 2 / (pt2 + m2))
    return -0.5 * math.log((E - pz) ** 2 / (pt2 + m2))


def dphi(a: float, b: float) -> float:
    d = abs(a - b)
    return 2 * math.pi - d if d > math.pi else d


def dr2(y1, phi1, y2, phi2) -> float:
    return (y1 - y2) ** 2 + dphi(phi1, phi2) ** 2


def flavour_of(pid: int) -> dict:
    """Net heavy flavour of a particle: {5: +-1, 4: +-1} for the b and c quarks it contains (quark +1, antiquark -1).
    PDG mesons q1 qbar2 with q1 the heavier: for a positive id q1 is a quark when it is up-type (D+ = c dbar) and an
    antiquark when it is down-type (B+ = u bbar, B0 = d bbar, B_c+ = c bbar); baryons with a positive id hold quarks."""
    a = abs(pid)
    if a in (4, 5):
        return {a: 1 if pid > 0 else -1}
    if a < 100 or a in (130, 310):
        return {}
    digits = [int(c) for c in str(a)]
    sign = 1 if pid > 0 else -1
    out = {}
    if a >= 1000:
        for q in (5, 4):
            n = digits[-4:-1].count(q)
            if n:
                out[q] = sign * n
        return out
    q1, q2 = digits[-3], digits[-2]
    up = q1 in (2, 4, 6)
    for q, s_ in ((q1, 1 if up else -1), (q2, -1 if up else 1)):
        if q in (4, 5):
            out[q] = out.get(q, 0) + sign * s_
    return {q: n for q, n in out.items() if n}


def flavour_text(f: dict) -> str:
    if not f:
        return "none"
    out = []
    for q, w in (("b", 5), ("c", 4)):
        if f.get(w):
            n = f[w]
            out.append(("+" if n > 0 else "") + f"{n} {q}")
    return ", ".join(out) if out else "none"


# ---------------------------------------------------------------------------
# the generalised-kT sequential recombination with the E scheme
# ---------------------------------------------------------------------------
class Clustering:
    """Sequential recombination of a list of four-vectors (rows E, px, py, pz) with the generalised-kT
    distances of power p and radius R. Every pseudojet, original or merged, has an index: the originals are
    0..N-1, a merge makes index N, N+1, ... The history is a list of steps ('ij', i, j, d, k) and
    ('iB', i, 0, d, i); jets(ptmin) returns the inclusive jets. Ghosts are ordinary entries with a tiny pT.
    """

    def __init__(self, p4, R: float, power: int = -1, log=None, names=None):
        p4 = np.asarray(p4, dtype=float)
        n = len(p4)
        self.n0, self.R, self.p = n, R, power
        cap = 2 * n + 2
        self.p4 = np.zeros((cap, 4))
        self.p4[:n] = p4
        self.pt2 = np.zeros(cap)
        self.y = np.zeros(cap)
        self.phi = np.zeros(cap)
        self.parents = {}
        self.const = {}                           # merged index -> list of original indices (lazily for originals)
        for i in range(n):
            self._kin(i)
        self.alive = np.zeros(cap, dtype=bool)
        self.alive[:n] = True
        self.is_jet = np.zeros(cap, dtype=bool)
        self.size = n
        self.history = []
        self.log, self.names = log, names
        self._run()

    def _kin(self, i):
        p = self.p4[i]
        self.pt2[i] = p[1] ** 2 + p[2] ** 2
        self.y[i] = rapidity(p)
        self.phi[i] = math.atan2(p[2], p[1])

    def _row(self, i, idx):
        """d_ij of pseudojet i against the pseudojets idx (an index array)."""
        pt2p_i = self.pt2[i] ** self.p
        pt2p = self.pt2[idx] ** self.p
        dy = self.y[idx] - self.y[i]
        dp = np.abs(self.phi[idx] - self.phi[i])
        dp = np.where(dp > math.pi, 2 * math.pi - dp, dp)
        return np.minimum(pt2p_i, pt2p) * (dy * dy + dp * dp) / (self.R * self.R)

    def _run(self):
        cap = len(self.pt2)
        nn = -np.ones(cap, dtype=int)
        nnd = np.full(cap, np.inf)
        diB = np.full(cap, np.inf)
        alive = self.alive
        for i in range(self.n0):
            diB[i] = self.pt2[i] ** self.p

        def recompute(i):
            idx = np.flatnonzero(alive)
            idx = idx[idx != i]
            if idx.size == 0:
                nn[i], nnd[i] = -1, np.inf
                return
            d = self._row(i, idx)
            k = int(np.argmin(d))
            nn[i], nnd[i] = idx[k], d[k]

        for i in range(self.n0):
            recompute(i)
        while alive.any():
            idx = np.flatnonzero(alive)
            best = np.minimum(nnd[idx], diB[idx])
            a = int(np.argmin(best))
            i = int(idx[a])
            if diB[i] <= nnd[i]:
                alive[i] = False
                self.is_jet[i] = True
                self.history.append(("iB", i, 0, float(diB[i]), i))
                if self.log:
                    self.log(2, f"    step {len(self.history):4d}: pseudojet {i} is a jet, d_iB = {diB[i]:.4g} (pT = {math.sqrt(self.pt2[i]):.3f})")
                touched = np.flatnonzero(alive & (nn == i))
            else:
                j = int(nn[i])
                k = self.size
                self.size += 1
                self.p4[k] = self.p4[i] + self.p4[j]
                self._kin(k)
                self.parents[k] = (i, j)
                alive[i] = alive[j] = False
                alive[k] = True
                diB[k] = self.pt2[k] ** self.p
                self.history.append(("ij", i, j, float(nnd[i]), k))
                if self.log:
                    self.log(2, f"    step {len(self.history):4d}: {i} + {j} -> {k}, d_ij = {nnd[i]:.4g}, dR = {math.sqrt(dr2(self.y[i], self.phi[i], self.y[j], self.phi[j])):.3f}, "
                                f"pT {math.sqrt(self.pt2[i]):.3f} + {math.sqrt(self.pt2[j]):.3f} -> {math.sqrt(self.pt2[k]):.3f}")
                touched = np.flatnonzero(alive & ((nn == i) | (nn == j)))
                recompute(k)
                touched = touched[touched != k]
            for m in touched:
                recompute(int(m))
            if self.size and alive.any():
                # the new pseudojet may be the nearest neighbour of others: check them cheaply
                if self.history[-1][0] == "ij":
                    k = self.history[-1][4]
                    idx = np.flatnonzero(alive)
                    idx = idx[idx != k]
                    if idx.size:
                        d = self._row(k, idx)
                        closer = idx[d < nnd[idx]]
                        for m in closer:
                            nn[m], nnd[m] = k, float(d[idx == m][0])

    # -- results ------------------------------------------------------------------------------------
    def constituents(self, i: int):
        if i < self.n0:
            return [i]
        if i not in self.const:
            a, b = self.parents[i]
            self.const[i] = self.constituents(a) + self.constituents(b)
        return self.const[i]

    def jets(self, ptmin: float):
        js = [i for i in range(self.size) if self.is_jet[i] and math.sqrt(self.pt2[i]) >= ptmin]
        return sorted(js, key=lambda i: -self.pt2[i])

    def merges_into(self, jet: int):
        """The merge steps that built pseudojet `jet`, in clustering order: (step no, i, j, d, k)."""
        members = set()
        stack = [jet]
        while stack:
            i = stack.pop()
            members.add(i)
            if i in self.parents:
                stack.extend(self.parents[i])
        return [(n + 1, s[1], s[2], s[3], s[4]) for n, s in enumerate(self.history) if s[0] == "ij" and s[4] in members]

    def kin(self, i: int) -> dict:
        p = self.p4[i]
        return dict(pt=math.sqrt(self.pt2[i]), y=self.y[i], eta=eta(p), phi=self.phi[i], m=math.sqrt(max(mass2(p), 0.0)), e=float(p[0]),
                    px=float(p[1]), py=float(p[2]), pz=float(p[3]))


# ---------------------------------------------------------------------------
# the particles of the event: what goes into the jets, and the ghosts
# ---------------------------------------------------------------------------
def last_in_chain(ev: Event, rows, flavour: int):
    """Those rows whose daughters contain no hadron of the same heavy flavour (the 'last' b or c hadrons)."""
    out = []
    for r in rows:
        q = ev.p[r]
        ds = range(q.d1, max(q.d1, q.d2) + 1) if q.d1 > 0 else []
        if not any(heavy_content(ev.p[d].id) == flavour for d in ds if d < len(ev.p)):
            out.append(r)
    return out


def ghost_rows(ev: Event) -> dict:
    """The CMSSW selectedHadronsAndPartons collections, as rows of the record."""
    b_had = last_in_chain(ev, [i for i, q in enumerate(ev.p) if is_hadron(q.id) and heavy_content(q.id) == 5], 5)
    c_had = last_in_chain(ev, [i for i, q in enumerate(ev.p) if is_hadron(q.id) and heavy_content(q.id) == 4], 4)
    partons = [i for i, q in enumerate(ev.p) if abs(q.status) in PARTON_STATUS and is_parton(q.id)]
    physics = [i for i, q in enumerate(ev.p) if abs(q.status) == 23 and is_parton(q.id)]
    heavy = set()
    for r in b_had + c_had:
        heavy.update(dec.descendants(ev, r))
    leptons = [i for i in ev.final_indices() if abs(ev.p[i].id) in (11, 13) and i in heavy]
    return dict(bhad=b_had, chad=c_had, parton=partons, physics=physics, lepton=leptons)


def stable_heavy_particles(ev: Event):
    """The final particles with every last b hadron and every last c hadron (not from a b) put back in place of
    its descendants: the hadron level of the flavour papers. Returns (rows, flavour dicts)."""
    g = ghost_rows(ev)
    replaced = {}
    for r in g["bhad"]:
        for d in dec.descendants(ev, r):
            replaced[d] = r
    for r in g["chad"]:
        if r in replaced:
            continue
        for d in dec.descendants(ev, r):
            replaced.setdefault(d, r)
    rows, seen = [], set()
    for i in ev.final_indices():
        r = replaced.get(i, i)
        if r in seen:
            continue
        seen.add(r)
        rows.append(r)
    flav = [flavour_of(ev.p[r].id) if (r in g["bhad"] or r in g["chad"]) else {} for r in rows]
    return rows, flav


def input_rows(ev: Event, nonu: bool):
    return [i for i in ev.final_indices() if abs(ev.p[i].id) not in IGNORE_IDS and not (nonu and is_neutrino(ev.p[i].id))]


# ---------------------------------------------------------------------------
# a collection: the clustering with ghosts and the flavours of every jet
# ---------------------------------------------------------------------------
class JetCollection:
    def __init__(self, ev: Event, key: str, R: float, nonu: bool, power: int, ptmin: float, ghosts: dict, log=None, extra=None):
        self.ev, self.key, self.R, self.nonu, self.power, self.ptmin = ev, key, R, nonu, power, ptmin
        self.rows = input_rows(ev, nonu)
        extra = extra or []                     # (p4, flavour dict, label): the soft pair of the infrared test
        self.extra = extra
        tags = [("particle", r) for r in self.rows] + [("extra", n) for n in range(len(extra))]
        p4 = [ev.p[r].p for r in self.rows] + [e[0] for e in extra]
        for kind in ("bhad", "chad", "parton", "physics", "lepton"):
            for r in ghosts[kind]:
                tags.append((kind, r))
                p4.append(ev.p[r].p * GHOST_RESCALING)
        self.tags = tags
        self.n_real = len(self.rows) + len(extra)
        self.cl = Clustering(np.array(p4), R, power, log=log if key == "ak4" else None)
        self.jets = self.cl.jets(ptmin)
        self.jet_of = {}                        # original row -> jet rank
        self.info = []
        for rank, j in enumerate(self.jets):
            c = self.cl.constituents(j)
            real = [i for i in c if i < self.n_real]
            rows = [tags[i][1] for i in real if tags[i][0] == "particle"]
            ghost = {k: [tags[i][1] for i in c if tags[i][0] == k] for k in ("bhad", "chad", "parton", "physics", "lepton")}
            for r in rows:
                self.jet_of[r] = rank
            k = self.cl.kin(j)
            # a cleaner momentum: the sum of the real constituents (the ghosts shift it by 1e-18)
            p = sum((ev.p[r].p for r in rows), np.zeros(4)) + sum((extra[tags[i][1]][0] for i in real if tags[i][0] == "extra"), np.zeros(4))
            k.update(pt=pt(p), eta=eta(p), phi=phi(p), m=math.sqrt(max(mass2(p), 0.0)), e=float(p[0]), px=float(p[1]), py=float(p[2]), pz=float(p[3]), y=rapidity(p))
            extras_in = [tags[i][1] for i in real if tags[i][0] == "extra"]
            info = dict(rank=rank, index=j, rows=rows, n=len(rows) + len(extras_in), extras=extras_in,
                        n_charged=sum(1 for r in rows if is_charged(ev.p[r].id)), ghosts=ghost, **k)
            info.update(self.flavours(info))
            info["composition"] = composition(ev, rows)
            self.info.append(info)

    def flavours(self, info) -> dict:
        ev, g = self.ev, info["ghosts"]
        extra_flav = [self.extra[n][1] for n in info["extras"]]
        has_b = bool(g["bhad"]) or any(5 in f for f in extra_flav)
        has_c = bool(g["chad"]) or any(4 in f for f in extra_flav)
        out = dict(hadron_flavour=5 if has_b else 4 if has_c else 0)

        def hardest_rule(rows):
            if not rows:
                return 0
            b = [r for r in rows if abs(ev.p[r].id) == 5]
            c = [r for r in rows if abs(ev.p[r].id) == 4]
            pool = b or c or rows
            return int(ev.p[max(pool, key=lambda r: pt(ev.p[r].p))].id)
        out["parton_flavour"] = hardest_rule(g["parton"])
        out["physics_flavour"] = hardest_rule(g["physics"])
        # the legacy dR matching: the partons of the final partonic state within DR_LEGACY of the axis
        near = [r for r, q in enumerate(ev.p) if abs(q.status) in PARTON_STATUS and is_parton(q.id)
                and dr2(rapidity(q.p), phi(q.p), info["y"], info["phi"]) < DR_LEGACY ** 2]
        out["legacy_flavour"] = hardest_rule(near)
        # the net flavour of the b and c hadrons in the jet, as the record has them (after the oscillations)
        net = {}
        for f in [flavour_of(ev.p[r].id) for r in g["bhad"] + g["chad"]] + extra_flav:
            for q, n in f.items():
                net[q] = net.get(q, 0) + n
        out["net_flavour"] = {q: n for q, n in net.items() if n}
        out["b_hadrons"] = [name(ev.p[r].id) for r in g["bhad"]] + [self.extra[n][2] for n in info["extras"] if 5 in self.extra[n][1]]
        out["c_hadrons"] = [name(ev.p[r].id) for r in g["chad"]]
        out["partons"] = [name(ev.p[r].id) for r in sorted(g["parton"], key=lambda r: -pt(ev.p[r].p))]
        out["leptons"] = [name(ev.p[r].id) for r in g["lepton"]]
        return out

    def matched(self, axis, maxdr=None):
        """The jet nearest in dR (eta, phi: what CMSSW's analyzer and the figures use) to the axis, within maxdr (default R)."""
        maxdr = self.R if maxdr is None else maxdr
        y0, f0 = eta(axis), phi(axis)
        best = None
        for info in self.info:
            d = math.sqrt(dr2(info["eta"], info["phi"], y0, f0))
            if d < maxdr and (best is None or d < best[0]):
                best = (d, info)
        return best


def composition(ev: Event, rows) -> dict:
    """pT and count of the jet's constituents by the classes the detector will separate."""
    cls = dict(charged_hadron=[], photon=[], neutral_hadron=[], electron=[], muon=[], neutrino=[])
    for r in rows:
        pid = ev.p[r].id
        a = abs(pid)
        if a == 22:
            cls["photon"].append(r)
        elif a == 11:
            cls["electron"].append(r)
        elif a == 13:
            cls["muon"].append(r)
        elif a in NEUTRINOS:
            cls["neutrino"].append(r)
        elif is_charged(pid):
            cls["charged_hadron"].append(r)
        else:
            cls["neutral_hadron"].append(r)
    out = {}
    for k, rs in cls.items():
        p = sum((ev.p[r].p for r in rs), np.zeros(4))
        out[k] = dict(n=len(rs), pt=pt(p), e=float(p[0]))
    return out


# ---------------------------------------------------------------------------
# the infrared-safe flavours: net flavour, IFN on the anti-kT sequence, flavour-kT
# ---------------------------------------------------------------------------
def omega2(y1, phi1, y2, phi2, omega=IFN_OMEGA):
    dy = y1 - y2
    dp = np.abs(phi1 - phi2)
    dp = np.where(dp > math.pi, 2 * math.pi - dp, dp)
    if omega == 0:
        return dy * dy + dp * dp
    return 2 * ((np.cosh(omega * dy) - 1) / omega ** 2 - (np.cos(dp) - 1))


def can_cancel(a: dict, b: dict) -> bool:
    return any(a.get(q, 0) * b.get(q, 0) < 0 for q in (5, 4))


def cancel(a: dict, b: dict):
    """Remove the common opposite flavour from both; returns what was cancelled."""
    done = {}
    for q in (5, 4):
        if a.get(q, 0) * b.get(q, 0) < 0:
            n = min(abs(a[q]), abs(b[q]))
            s = 1 if a[q] > 0 else -1
            a[q] -= s * n
            b[q] += s * n
            done[q] = s * n
            for d in (a, b):
                if d.get(q) == 0:
                    del d[q]
    return done


class IFN:
    """Interleaved flavour neutralisation replayed on the clustering sequence of an anti-kT Clustering
    whose first entries are the particles with the flavours `flav` (a list of dicts, {} for none)."""

    def __init__(self, cl: Clustering, flav, alpha=IFN_ALPHA, omega=IFN_OMEGA, log=None, label=None):
        self.cl, self.alpha, self.omega, self.log = cl, alpha, omega, log
        self.label = label or (lambda i: str(i))
        self.flav = {i: dict(f) for i, f in enumerate(flav)}
        self.events = []                   # (step, i, k, u_ik, u_threshold, cancelled)
        alive = set(range(cl.n0))
        declared = set()
        for step, (kind, i, j, d, k) in enumerate(cl.history, start=1):
            if kind == "iB":
                alive.discard(i)
                declared.add(i)
                continue
            fi, fj = self.flav.get(i, {}), self.flav.get(j, {})
            if fi or fj:
                pool = sorted(alive | declared)
                for a, b in ((i, j), (j, i)):
                    if self.flav.get(a):
                        u_ab = float(self.u(a, np.array([b]))[0])
                        self.neutralise(a, u_ab, pool, step, depth=0)
            alive.discard(i)
            alive.discard(j)
            alive.add(k)
            merged = {}
            for q in (5, 4):
                n = self.flav.get(i, {}).get(q, 0) + self.flav.get(j, {}).get(q, 0)
                if n:
                    merged[q] = n
            self.flav[k] = merged

    def u(self, i, idx):
        pti = math.sqrt(self.cl.pt2[i])
        ptk = np.sqrt(self.cl.pt2[idx])
        hi, lo = np.maximum(pti, ptk), np.minimum(pti, ptk)
        return hi ** self.alpha * lo ** (2 - self.alpha) * omega2(self.cl.y[i], self.cl.phi[i], self.cl.y[idx], self.cl.phi[idx], self.omega)

    def neutralise(self, i, threshold, others, step, depth):
        if not self.flav.get(i) or depth > 50:
            return
        idx = np.array([k for k in others if k != i and self.flav.get(k)], dtype=int)
        if idx.size == 0:
            return
        u = self.u(i, idx)
        order = np.argsort(u)
        for o in order:
            k, uik = int(idx[o]), float(u[o])
            if uik >= threshold:
                break
            if not self.flav.get(i):
                break
            if not can_cancel(self.flav[i], self.flav.get(k, {})):
                if self.log:
                    self.log(2, f"      {'  ' * depth}IFN step {step}: candidate {self.label(k)} for {self.label(i)} (u = {uik:.4g} < {threshold:.4g}) cannot cancel {flavour_text(self.flav[i])}")
                continue
            # k's own closer partners come first (the recursion of the paper)
            self.neutralise(k, uik, others, step, depth + 1)
            if can_cancel(self.flav[i], self.flav.get(k, {})):
                done = cancel(self.flav[i], self.flav[k])
                self.events.append((step, i, k, uik, threshold, done))
                if self.log:
                    self.log(1, f"      {'  ' * depth}IFN step {step}: {self.label(i)} neutralised against {self.label(k)}: {flavour_text(done)} cancelled "
                                f"(u_ik = {uik:.4g} < u_ij = {threshold:.4g}); {self.label(i)} now {flavour_text(self.flav[i])}, {self.label(k)} now {flavour_text(self.flav[k])}")


def flavour_kt(p4, flav, R: float, alpha=FKT_ALPHA, ptmin=JET_PT_MIN, log=None):
    """The flavour-kT algorithm of Banfi, Salam and Zanderighi (hadron-collider version, two beams), with the
    net flavour carried along. Returns a list of jets: dict(pt, eta, phi, y, m, e, flavour, const), sorted by pT."""
    p4 = np.asarray(p4, dtype=float)
    n = len(p4)
    P = {i: p4[i].copy() for i in range(n)}
    F = {i: dict(f) for i, f in enumerate(flav)}
    C = {i: [i] for i in range(n)}
    nxt = n
    jets = []
    R2 = R * R
    while P:
        idx = list(P)
        A = np.array([P[i] for i in idx])
        kt = np.sqrt(A[:, 1] ** 2 + A[:, 2] ** 2)
        y = np.array([rapidity(P[i]) for i in idx])
        ph = np.arctan2(A[:, 2], A[:, 1])
        fl = np.array([bool(F[i]) for i in idx])
        m = len(idx)
        # beam "transverse momenta" seen from rapidity y_i
        dy = y[None, :] - y[:, None]                       # y_j - y_i
        kB = np.sum(kt[None, :] * np.where(dy > 0, 1.0, np.exp(np.minimum(dy, 0.0))), axis=1)
        kBbar = np.sum(kt[None, :] * np.where(dy < 0, 1.0, np.exp(np.minimum(-dy, 0.0))), axis=1)

        def fdist(ka, kb, flavoured):
            hi, lo = np.maximum(ka, kb), np.minimum(ka, kb)
            return np.where(flavoured, hi ** alpha * lo ** (2 - alpha), lo ** 2)
        diB = fdist(kt, kB, fl)
        diBb = fdist(kt, kBbar, fl)
        dp = np.abs(ph[None, :] - ph[:, None])
        dp = np.where(dp > math.pi, 2 * math.pi - dp, dp)
        dR2 = dy * dy + dp * dp
        softer_flavoured = np.where(kt[:, None] < kt[None, :], fl[:, None], fl[None, :]) | ((kt[:, None] == kt[None, :]) & (fl[:, None] | fl[None, :]))
        dij = fdist(kt[:, None], kt[None, :], softer_flavoured) * dR2 / R2
        np.fill_diagonal(dij, np.inf)
        a = int(np.argmin(dij))
        i_, j_ = divmod(a, m)
        dmin_ij = dij[i_, j_]
        b = int(np.argmin(np.minimum(diB, diBb)))
        dmin_B = min(diB[b], diBb[b])
        if dmin_B <= dmin_ij:
            i = idx[b]
            p = P.pop(i)
            if pt(p) >= ptmin:
                jets.append(dict(pt=pt(p), eta=eta(p), phi=phi(p), y=rapidity(p), m=math.sqrt(max(mass2(p), 0.0)), e=float(p[0]),
                                 flavour=F[i], const=C[i], beam="B" if diB[b] <= diBb[b] else "Bbar", d=float(dmin_B)))
            continue
        i, j = idx[i_], idx[j_]
        P[nxt] = P.pop(i) + P.pop(j)
        merged = {}
        for q in (5, 4):
            s = F[i].get(q, 0) + F[j].get(q, 0)
            if s:
                merged[q] = s
        if log and (F[i] or F[j]):
            log(1, f"      flavour-kT: {i} ({flavour_text(F[i])}, pT {kt[i_]:.2f}) + {j} ({flavour_text(F[j])}, pT {kt[j_]:.2f}) -> {nxt} ({flavour_text(merged)}), "
                   f"d_ij = {dmin_ij:.4g}, dR = {math.sqrt(dR2[i_, j_]):.3f}")
        F[nxt] = merged
        C[nxt] = C[i] + C[j]
        nxt += 1
    return sorted(jets, key=lambda j: -j["pt"])


# ---------------------------------------------------------------------------
# the stage: all collections, the flavours, the summary
# ---------------------------------------------------------------------------
class Stage:
    def __init__(self, ev: Event, args, verbose=0, soft_pair=None):
        self.ev, self.args, self.verbose = ev, args, verbose
        self.lines = []
        self.power = POWER[args.algorithm]
        self.ghosts = ghost_rows(ev)
        self.bq = had.b_quark_rows(ev)                      # pid -> row of the b quark copy (the axis of chapters 5 and 6)
        self.extra = soft_pair or []
        self.coll = {}
        for key, c in COLLECTIONS.items():
            R = args.radius if (args.radius and key.startswith("ak4")) else c["R"]
            self.coll[key] = JetCollection(ev, key, R, c["nonu"], self.power, args.ptmin, self.ghosts,
                                           log=lambda lv, t: self.say(lv, t, "merge"), extra=self.extra)
        # the hadron level with the heavy hadrons kept: net flavour, IFN and flavour-kT
        rows, flav = stable_heavy_particles(ev)
        self.stable_rows = rows
        p4 = [ev.p[r].p for r in rows] + [e[0] for e in self.extra]
        flav = flav + [e[1] for e in self.extra]
        self.stable_flav = flav
        self.labels = [name(ev.p[r].id) + f"[{r}]" for r in rows] + [e[2] for e in self.extra]
        self.cl_stable = Clustering(np.array(p4), R_AK4, -1)
        lab = lambda i: self.labels[i] if i < len(self.labels) else f"pseudojet {i}"  # noqa: E731
        self.ifn = IFN(self.cl_stable, flav, log=lambda lv, t: self.say(lv, t, "ifn"), label=lab)
        self.stable_jets = []
        for rank, j in enumerate(self.cl_stable.jets(args.ptmin)):
            c = self.cl_stable.constituents(j)
            net = {}
            for i in c:
                for q, nq in flav[i].items():
                    net[q] = net.get(q, 0) + nq
            k = self.cl_stable.kin(j)
            self.stable_jets.append(dict(rank=rank, index=j, const=c, net_flavour={q: v for q, v in net.items() if v}, ifn_flavour=dict(self.ifn.flav.get(j, {})),
                                         flavoured=[self.labels[i] for i in c if flav[i]], **k))
        self.fkt_jets = flavour_kt(np.array(p4), flav, R_AK4, FKT_ALPHA, args.ptmin, log=lambda lv, t: self.say(lv, t, "fkt")) if not args.no_flavour_kt else []
        for j in self.fkt_jets:
            j["flavoured"] = [self.labels[i] for i in j["const"] if flav[i]]

    def say(self, level, text, tag="misc"):
        """Learning-mode lines are kept and printed by explain() under their banner."""
        if self.verbose >= level:
            self.lines.append((tag, text))

    def said(self, tag):
        return [t for g, t in self.lines if g == tag]


def stage_summary(st: Stage) -> dict:
    ev = st.ev
    s = dict(n_final=len(ev.final_indices()), n_neutrinos=sum(1 for i in ev.final_indices() if is_neutrino(ev.p[i].id)),
             algorithm=st.args.algorithm, ptmin=st.args.ptmin, collections={}, b={}, ghosts={k: len(v) for k, v in st.ghosts.items()},
             ifn_events=len(st.ifn.events))
    for key, c in st.coll.items():
        s["collections"][key] = dict(R=c.R, nonu=c.nonu, n_jets=len(c.info), n_input=len(c.rows), n_clustered=len(c.jet_of),
                                     n_jets_10=sum(1 for j in c.info if j["pt"] > 10), n_jets_30=sum(1 for j in c.info if j["pt"] > 30),
                                     flavours=dict(hadron=dec.count_ids(abs(j["hadron_flavour"]) for j in c.info),
                                                   parton=dec.count_ids(abs(j["parton_flavour"]) for j in c.info)),
                                     jets=[{k: v for k, v in j.items() if k not in ("rows", "ghosts")} for j in c.info])
    # the two b quarks' jets
    for pid, jb in st.bq.items():
        qname = had.pname(pid)
        axis = ev.p[jb].p
        rec = dict(quark_row=jb, pt_quark=pt(axis), eta_quark=eta(axis), phi_quark=phi(axis), y_quark=rapidity(axis), jets={})
        bstr = dec.string_b_hadrons(ev)
        rw = dec.weak_b_hadron(ev, bstr[pid]) if pid in bstr else None
        rec["hadron"] = name(ev.p[rw].id) if rw is not None else ""
        rec["hadron_row"] = rw
        rec["pt_hadron"] = pt(ev.p[rw].p) if rw is not None else 0.0
        fin_desc = set(d for d in dec.descendants(ev, rw) if ev.p[d].status > 0) if rw is not None else set()
        for key, c in st.coll.items():
            m = c.matched(axis)
            if m is None:
                rec["jets"][key] = None
                continue
            d, info = m
            cone = dec.cone_members(ev, c.rows, axis, c.R)
            p_cone = sum((ev.p[i].p for i in cone), np.zeros(4))
            inside = [r for r in info["rows"] if r in fin_desc]
            p_in = sum((ev.p[r].p for r in inside), np.zeros(4))
            p_nu = sum((ev.p[r].p for r in info["rows"] if is_neutrino(ev.p[r].id)), np.zeros(4))
            rec["jets"][key] = dict(rank=info["rank"], pt=info["pt"], eta=info["eta"], phi=info["phi"], y=info["y"], m=info["m"], e=info["e"], n=info["n"],
                                    n_charged=info["n_charged"], dr=d, response=info["pt"] / pt(axis), pt_cone=pt(p_cone), n_cone=len(cone),
                                    in_cone_not_jet=sorted(set(cone) - set(info["rows"])), in_jet_not_cone=sorted(set(info["rows"]) - set(cone)),
                                    pt_from_hadron=pt(p_in), n_from_hadron=len(inside), n_hadron_outside=len(fin_desc - set(info["rows"])),
                                    pt_nu=pt(p_nu), n_nu=sum(1 for r in info["rows"] if is_neutrino(ev.p[r].id)),
                                    hadron_flavour=info["hadron_flavour"], parton_flavour=info["parton_flavour"], physics_flavour=info["physics_flavour"],
                                    legacy_flavour=info["legacy_flavour"], net_flavour=info["net_flavour"], b_hadrons=info["b_hadrons"], c_hadrons=info["c_hadrons"],
                                    partons=info["partons"], composition=info["composition"],
                                    dr_hadron=(math.sqrt(dr2(eta(ev.p[rw].p), phi(ev.p[rw].p), info["eta"], info["phi"])) if rw is not None else -1.0))
        # the stable-hadron jets: net, IFN and flavour-kT
        y0, f0 = eta(axis), phi(axis)
        sj = min(st.stable_jets, key=lambda j: dr2(j["eta"], j["phi"], y0, f0), default=None)
        if sj is not None and dr2(sj["eta"], sj["phi"], y0, f0) < R_AK4 ** 2:
            rec["stable_jet"] = dict(rank=sj["rank"], pt=sj["pt"], net_flavour=sj["net_flavour"], ifn_flavour=sj["ifn_flavour"], flavoured=sj["flavoured"])
        else:
            rec["stable_jet"] = None
        fj = min(st.fkt_jets, key=lambda j: dr2(j["eta"], j["phi"], y0, f0), default=None)
        if fj is not None and dr2(fj["eta"], fj["phi"], y0, f0) < R_AK4 ** 2:
            rec["fkt_jet"] = dict(pt=fj["pt"], eta=fj["eta"], phi=fj["phi"], m=fj["m"], n=len(fj["const"]), flavour=fj["flavour"], flavoured=fj["flavoured"],
                                  dr=math.sqrt(dr2(fj["eta"], fj["phi"], y0, f0)), response=fj["pt"] / pt(axis))
        else:
            rec["fkt_jet"] = None
        s["b"][qname] = rec
    # the b jet + bbar jet system per collection
    for key, c in st.coll.items():
        a, b = s["b"].get("b", {}).get("jets", {}).get(key), s["b"].get("bbar", {}).get("jets", {}).get(key)
        if a and b:
            px, py = a["pt"] * math.cos(a["phi"]) + b["pt"] * math.cos(b["phi"]), a["pt"] * math.sin(a["phi"]) + b["pt"] * math.sin(b["phi"])
            pa = np.array([a["e"], a["pt"] * math.cos(a["phi"]), a["pt"] * math.sin(a["phi"]), a["pt"] * math.sinh(a["eta"])])
            pb = np.array([b["e"], b["pt"] * math.cos(b["phi"]), b["pt"] * math.sin(b["phi"]), b["pt"] * math.sinh(b["eta"])])
            P = pa + pb
            s["collections"][key]["bb"] = dict(pt=math.hypot(px, py), px=px, py=py, scalar=a["pt"] + b["pt"], dphi=dphi(a["phi"], b["phi"]), m=math.sqrt(max(mass2(P), 0.0)))
    s["stable_jets"] = [{k: v for k, v in j.items() if k != "const"} | dict(n=len(j["const"])) for j in st.stable_jets[:12]]
    s["fkt_jets"] = [{k: v for k, v in j.items() if k != "const"} | dict(n=len(j["const"])) for j in st.fkt_jets[:12]]
    s["ifn"] = [dict(step=e[0], i=st.labels[e[1]] if e[1] < len(st.labels) else f"pseudojet {e[1]}", k=st.labels[e[2]] if e[2] < len(st.labels) else f"pseudojet {e[2]}",
                     u=e[3], threshold=e[4], cancelled=e[5]) for e in st.ifn.events]
    return s


def fl_text(f: int) -> str:
    return "0" if f == 0 else f"{f:+d}"


def print_summary(st: Stage, s: dict):
    ev = st.ev
    print(f"{s['n_final']} final particles ({s['n_neutrinos']} neutrinos); ghosts: {s['ghosts']['bhad']} b hadrons, {s['ghosts']['chad']} c hadrons, "
          f"{s['ghosts']['parton']} partons (status 71-79), {s['ghosts']['physics']} hard-process partons, {s['ghosts']['lepton']} leptons from heavy hadrons")
    for key, c in s["collections"].items():
        cc = COLLECTIONS[key]
        print(f"{cc['title']} ({st.args.algorithm}, R = {c['R']}, pT > {s['ptmin']:g} GeV{', no neutrinos' if c['nonu'] else ''}): {c['n_jets']} jets, "
              f"{c['n_clustered']} of {c['n_input']} particles clustered; {c['n_jets_10']} jets above 10 GeV, {c['n_jets_30']} above 30; "
              f"hadronFlavour 5: {c['flavours']['hadron'].get(5, 0)}, 4: {c['flavours']['hadron'].get(4, 0)}")
        print(f"  {'jet':>4s} {'pT':>8s} {'eta':>7s} {'phi':>7s} {'m':>7s} {'n':>4s} {'nch':>4s} {'had':>4s} {'part':>5s} {'phys':>5s} {'dR.3':>5s}  heavy hadrons / partons in the jet")
        for j in c["jets"][:10]:
            print(f"  {j['rank']:4d} {j['pt']:8.2f} {j['eta']:7.2f} {j['phi']:7.2f} {j['m']:7.2f} {j['n']:4d} {j['n_charged']:4d} {j['hadron_flavour']:4d} "
                  f"{fl_text(j['parton_flavour']):>5s} {fl_text(j['physics_flavour']):>5s} {fl_text(j['legacy_flavour']):>5s}  "
                  + " ".join(j["b_hadrons"] + j["c_hadrons"]) + (" | " + " ".join(j["partons"][:4]) if j["partons"] else ""))
        if c["n_jets"] > 10:
            print(f"  ... and {c['n_jets'] - 10} more")
    for q, r in s["b"].items():
        print(f"{q} quark (row {r['quark_row']}): pT = {r['pt_quark']:.2f} GeV, eta = {r['eta_quark']:.2f}, phi = {r['phi_quark']:.2f}; its {r['hadron']} (row {r['hadron_row']}) pT = {r['pt_hadron']:.2f}")
        for key, j in r["jets"].items():
            if j is None:
                print(f"  {COLLECTIONS[key]['title']:16s}: no jet within R of the quark")
                continue
            comp = j["composition"]
            print(f"  {COLLECTIONS[key]['title']:16s}: jet [{j['rank']}] pT = {j['pt']:.2f} GeV, eta = {j['eta']:.2f}, phi = {j['phi']:.2f}, m = {j['m']:.2f}, n = {j['n']} ({j['n_charged']} charged); "
                  f"dR(jet, quark) = {j['dr']:.3f}, pT(jet)/pT(quark) = {j['response']:.3f}; cone of the same R: {j['pt_cone']:.2f} GeV ({j['n_cone']}), "
                  f"{len(j['in_cone_not_jet'])} in the cone but not the jet, {len(j['in_jet_not_cone'])} in the jet but not the cone")
            print(f"  {'':16s}  the {r['hadron']}'s descendants in it: {j['n_from_hadron']} with {j['pt_from_hadron']:.2f} GeV ({j['n_hadron_outside']} outside), dR(jet, hadron) = {j['dr_hadron']:.3f}; "
                  f"neutrinos {j['n_nu']} with {j['pt_nu']:.2f} GeV; charged hadrons {comp['charged_hadron']['pt']:.2f}, photons {comp['photon']['pt']:.2f}, "
                  f"neutral hadrons {comp['neutral_hadron']['pt']:.2f}, leptons {comp['electron']['pt'] + comp['muon']['pt']:.2f} GeV")
            print(f"  {'':16s}  hadronFlavour {j['hadron_flavour']}, partonFlavour {j['parton_flavour']}, physics {j['physics_flavour']}, dR<0.3 {j['legacy_flavour']}, "
                  f"net {flavour_text(j['net_flavour'])}; b hadrons: {' '.join(j['b_hadrons']) or 'none'}; c hadrons: {' '.join(j['c_hadrons']) or 'none'}")
        if r.get("stable_jet"):
            sj = r["stable_jet"]
            print(f"  heavy hadrons kept : anti-kT jet [{sj['rank']}] pT = {sj['pt']:.2f}: net flavour {flavour_text(sj['net_flavour'])}, IFN flavour {flavour_text(sj['ifn_flavour'])} "
                  f"(flavoured constituents: {' '.join(sj['flavoured']) or 'none'})")
        if r.get("fkt_jet"):
            fj = r["fkt_jet"]
            print(f"  flavour-kT         : jet pT = {fj['pt']:.2f}, eta = {fj['eta']:.2f}, phi = {fj['phi']:.2f}, n = {fj['n']}, dR = {fj['dr']:.3f}: flavour {flavour_text(fj['flavour'])} "
                  f"({' '.join(fj['flavoured']) or 'none'})")
    for key, c in s["collections"].items():
        if "bb" in c:
            bb = c["bb"]
            print(f"{COLLECTIONS[key]['title']}: b jet + bbar jet: |pT(b) + pT(bbar)| = {bb['pt']:.2f} GeV, Sigma|pT| = {bb['scalar']:.2f}, dphi = {bb['dphi']:.2f}, m = {bb['m']:.1f} GeV")
    if s["ifn"]:
        print(f"IFN: {len(s['ifn'])} neutralisation(s):")
        for e in s["ifn"]:
            print(f"  step {e['step']}: {e['i']} with {e['k']}: {flavour_text(e['cancelled'])} (u = {e['u']:.4g} < {e['threshold']:.4g})")
    else:
        print("IFN: no neutralisation in this event (every flavoured hadron keeps its flavour)")
    print("anti-kT jets with the heavy hadrons kept, net and IFN flavour (pT > 10 GeV or flavoured):")
    for j in s["stable_jets"]:
        if j["pt"] > 10 or j["net_flavour"] or j["ifn_flavour"] or j["flavoured"]:
            print(f"  [{j['rank']}] pT = {j['pt']:.2f}, eta = {j['eta']:.2f}, phi = {j['phi']:.2f}: net {flavour_text(j['net_flavour'])}, IFN {flavour_text(j['ifn_flavour'])}"
                  + (f"  ({' '.join(j['flavoured'])})" if j["flavoured"] else ""))
    if s["fkt_jets"]:
        print("flavour-kT jets (pT > 10 GeV or flavoured):")
        for j in s["fkt_jets"]:
            if j["pt"] > 10 or j["flavour"]:
                print(f"  pT = {j['pt']:.2f}, eta = {j['eta']:.2f}, phi = {j['phi']:.2f}, n = {j['n']}: flavour {flavour_text(j['flavour'])}" + (f"  ({' '.join(j['flavoured'])})" if j["flavoured"] else ""))


# ---------------------------------------------------------------------------
# learning mode: the b jet step by step
# ---------------------------------------------------------------------------
def explain(st: Stage, s: dict):
    ev = st.ev
    print(banner("set-up"))
    print(f"  input {st.args.input}: {len(ev.p)} rows, {s['n_final']} with a positive status (CMSSW: genParticles with status 1)")
    print(f"  {st.args.algorithm}: d_ij = min(pT_i^2p, pT_j^2p) dR^2/R^2 with p = {st.power}, d_iB = pT_i^2p, dR^2 = dy^2 + dphi^2 (rapidity), E-scheme sums")
    print(f"  collections: " + "; ".join(f"{COLLECTIONS[k]['title']} R = {c.R}{', no neutrinos' if c.nonu else ''} ({len(c.rows)} particles)" for k, c in st.coll.items())
          + f"; jets kept above pT = {st.args.ptmin:g} GeV")
    g = st.ghosts
    print(f"  ghosts (momentum x {GHOST_RESCALING:g}): b hadrons " + ", ".join(f"{name(ev.p[r].id)} [{r}] pT {pt(ev.p[r].p):.2f} eta {eta(ev.p[r].p):.2f} phi {phi(ev.p[r].p):.2f}" for r in g["bhad"]))
    print(f"          c hadrons " + ", ".join(f"{name(ev.p[r].id)} [{r}] {pt(ev.p[r].p):.2f}" for r in g["chad"]))
    print(f"          {len(g['parton'])} partons with status 71-79 (the strings' partons), {len(g['physics'])} hard-process partons: "
          + ", ".join(f"{name(ev.p[r].id)} [{r}] pT {pt(ev.p[r].p):.2f}" for r in g["physics"]) + f"; {len(g['lepton'])} leptons from heavy hadrons")
    print(banner("the b quark's ak4 jet, merge by merge"))
    c = st.coll["ak4"]
    r = s["b"]["b"]
    j = r["jets"]["ak4"]
    if j is None:
        print("  no jet")
        return
    jet = c.info[j["rank"]]
    cl = c.cl
    steps = cl.merges_into(jet["index"])
    if st.verbose >= 2:
        print(f"  every merge of the ak4 clustering ({len(cl.history)} steps):")
        for t in st.said("merge"):
            print(t)
    print(f"  the jet is pseudojet {jet['index']}; {len(steps)} merges built it (of {len(cl.history)} steps of the whole event); step = position in the event's sequence")
    print(f"  {'step':>5s} {'merge':>13s} {'d_ij':>11s} {'dR':>6s} {'pT_i':>8s} {'pT_j':>8s} {'pT_k':>8s}  what joined")

    def tagname(i):
        if i >= cl.n0:
            return f"pseudojet {i}"
        kind, r = c.tags[i]
        if kind == "particle":
            return f"{name(ev.p[r].id)}[{r}] {pt(ev.p[r].p):.2f} GeV"
        if kind == "extra":
            return st.extra[r][2]
        return f"ghost {kind} {name(ev.p[r].id)}[{r}]"
    for n, i, jj, d, k in steps:
        soft = jj if cl.pt2[jj] < cl.pt2[i] else i
        hard = i if soft == jj else jj
        print(f"  {n:5d} {i:5d} + {jj:5d} {d:11.4g} {math.sqrt(dr2(cl.y[i], cl.phi[i], cl.y[jj], cl.phi[jj])):6.3f} {math.sqrt(cl.pt2[i]):8.3f} {math.sqrt(cl.pt2[jj]):8.3f} {math.sqrt(cl.pt2[k]):8.3f}  "
              + tagname(soft) + f" joins {tagname(hard)}")
    print(f"  then d_iB = 1/pT^2 = {1 / jet['pt'] ** 2:.4g} is the smallest distance left for it: the jet is declared, pT = {jet['pt']:.3f} GeV, "
          f"eta = {jet['eta']:.3f}, phi = {jet['phi']:.3f}, m = {jet['m']:.2f}, {jet['n']} constituents")
    print(f"  the cone of R = 0.4 around the quark holds {j['pt_cone']:.2f} GeV in {j['n_cone']} particles; the jet's axis is {j['dr']:.3f} from the quark (dR in eta, phi);")
    for lab, rows in (("in the cone but not in the jet", j["in_cone_not_jet"]), ("in the jet but not in the cone", j["in_jet_not_cone"])):
        print(f"    {lab}: " + (", ".join(f"{name(ev.p[x].id)}[{x}] {pt(ev.p[x].p):.2f} GeV" for x in rows) or "none"))
    print(banner("the flavour of the two b jets"))
    for q, rr in s["b"].items():
        jj = rr["jets"]["ak4"]
        if jj is None:
            continue
        print(f"  {q} jet [{jj['rank']}]: ghosts clustered: b hadrons {' '.join(jj['b_hadrons']) or 'none'}; c hadrons {' '.join(jj['c_hadrons']) or 'none'}; "
              f"partons {' '.join(jj['partons']) or 'none'}")
        print(f"    hadronFlavour = {jj['hadron_flavour']} (a b-hadron ghost {'is' if jj['hadron_flavour'] == 5 else 'is not'} in the jet)")
        print(f"    partonFlavour = {jj['parton_flavour']} (the hardest b parton among the status-71 ghosts, else c, else the hardest)")
        print(f"    physics flavour = {jj['physics_flavour']} (the status-23 partons as ghosts); dR < 0.3 matching: {jj['legacy_flavour']}")
        print(f"    net flavour of the b/c hadrons in the jet, as recorded: {flavour_text(jj['net_flavour'])}")
        if rr.get("stable_jet"):
            sj = rr["stable_jet"]
            print(f"    with the heavy hadrons undecayed: net {flavour_text(sj['net_flavour'])}, IFN {flavour_text(sj['ifn_flavour'])}; flavoured constituents {' '.join(sj['flavoured'])}")
    print(banner("IFN over the whole event (heavy hadrons kept, anti-kT R = 0.4)"))
    print(f"  u_ik = max(pT_i, pT_k)^{IFN_ALPHA:g} min(pT_i, pT_k)^{2 - IFN_ALPHA:g} Omega_ik^2, Omega^2 = 2[(cosh(omega dy) - 1)/omega^2 - (cos dphi - 1)], omega = {IFN_OMEGA:g}")
    lines = st.said("ifn")
    for t in lines:
        print(t)
    if not lines:
        print("  no flavoured pseudojet found a partner with u_ik < u_ij: no neutralisation in this event")
    if not st.args.no_flavour_kt:
        print(banner("flavour-kT over the whole event"))
        lines = st.said("fkt")
        for t in lines[:60]:
            print(t)
        if len(lines) > 60:
            print(f"  ... {len(lines) - 60} more flavoured merges")


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------
def jet_tables(st: Stage, s: dict) -> str:
    ev = st.ev
    t1 = (f"Standalone Jet Listing  ({st.args.algorithm}, E scheme, jet pT > {st.args.ptmin:g} GeV; ak4 R = {st.coll['ak4'].R}, ak4nonu R = {st.coll['ak4nonu'].R} without neutrinos, "
          f"ak8 R = {st.coll['ak8'].R}; flavours: had = hadronFlavour, part = partonFlavour, phys = physics definition, dR3 = legacy dR < 0.3, net = net b flavour of the heavy hadrons)")
    out = ["", f" --------  {t1}  " + "-" * max(10, 133 - len(t1) - 14), " ",
           f"{'coll':>8s}{'jet':>5s}{'pT':>11s}{'y':>9s}{'eta':>9s}{'phi':>9s}{'m':>9s}{'E':>11s}{'n':>5s}{'nch':>5s}{'had':>5s}{'part':>6s}{'phys':>6s}{'dR3':>6s}{'net':>5s}  heavy hadrons; partons (rows)"]
    for key, c in st.coll.items():
        for j in c.info:
            heavy = " ".join(f"{name(ev.p[r].id)}[{r}]" for r in j["ghosts"]["bhad"] + j["ghosts"]["chad"])
            parts = " ".join(f"{name(ev.p[r].id)}[{r}]" for r in sorted(j["ghosts"]["parton"], key=lambda r: -pt(ev.p[r].p))[:6])
            out.append(f"{key:>8s}{j['rank']:5d}{j['pt']:11.3f}{j['y']:9.3f}{j['eta']:9.3f}{j['phi']:9.3f}{j['m']:9.3f}{j['e']:11.3f}{j['n']:5d}{j['n_charged']:5d}"
                       f"{j['hadron_flavour']:5d}{j['parton_flavour']:6d}{j['physics_flavour']:6d}{j['legacy_flavour']:6d}{j['net_flavour'].get(5, 0):5d}  {heavy}" + (f"; {parts}" if parts else ""))
        out.append("")
    out.append(f" --------  End {t1}  " + "-" * max(10, 133 - len(t1) - 18))
    t2 = "Standalone Jet Constituent Listing  (row -> jet index in ak4, ak4nonu, ak8; -1 = not clustered or in a jet below the threshold)"
    out += ["", f" --------  {t2}  " + "-" * max(10, 133 - len(t2) - 14), " ", f"{'no':>6s}{'ak4':>6s}{'ak4nonu':>9s}{'ak8':>6s}"]
    for i in ev.final_indices():
        out.append(f"{i:6d}{st.coll['ak4'].jet_of.get(i, -1):6d}{st.coll['ak4nonu'].jet_of.get(i, -1):9d}{st.coll['ak8'].jet_of.get(i, -1):6d}")
    out.append("")
    out.append(f" --------  End {t2}  " + "-" * max(10, 133 - len(t2) - 18))
    return "\n".join(out) + "\n"


def read_jet_tables(path: Path):
    """The jet table and the constituent table of a record written by this script: (jets[coll] = list of dicts, jet_of[coll] = {row: rank})."""
    jets, jet_of = {k: [] for k in COLLECTIONS}, {k: {} for k in COLLECTIONS}
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) >= 15 and f[0] in COLLECTIONS and f[1].isdigit():
            rest = line.split(None, 15)
            jets[f[0]].append(dict(rank=int(f[1]), pt=float(f[2]), y=float(f[3]), eta=float(f[4]), phi=float(f[5]), m=float(f[6]), e=float(f[7]), n=int(f[8]),
                                   n_charged=int(f[9]), hadron_flavour=int(f[10]), parton_flavour=int(f[11]), physics_flavour=int(f[12]), legacy_flavour=int(f[13]),
                                   net_b=int(f[14]), heavy=(rest[15].strip() if len(rest) > 15 else "")))
        elif len(f) == 4 and all(x.lstrip("-").isdigit() for x in f):
            no = int(f[0])
            for k, v in zip(("ak4", "ak4nonu", "ak8"), f[1:]):
                if int(v) >= 0:
                    jet_of[k][no] = int(v)
    return jets, jet_of


# ---------------------------------------------------------------------------
# the soft g -> b bbar pair of the infrared test
# ---------------------------------------------------------------------------
def soft_pair(ev: Event, st_jets, pt_pair: float, sep: float = 0.5):
    """Two soft b hadrons (a B0 and a Bbar0, pt_pair each) placed between the two nearest light jets of pT > 5 GeV,
    `sep` apart in dR, one on each side of the mid-point: a g -> b bbar at the edge of two jets."""
    light = [j for j in st_jets if j["pt"] > 5 and j["hadron_flavour"] == 0]
    best = None
    for a in light:
        for b in light:
            if a["rank"] < b["rank"]:
                d = math.sqrt(dr2(a["y"], a["phi"], b["y"], b["phi"]))
                if 0.6 < d < 1.6 and (best is None or d < best[0]):
                    best = (d, a, b)
    if best is None:
        return None, None
    d, a, b = best
    ym, fm = 0.5 * (a["y"] + b["y"]), a["phi"] + 0.5 * ((b["phi"] - a["phi"] + math.pi) % (2 * math.pi) - math.pi)
    ux, uy = (b["y"] - a["y"]) / d, ((b["phi"] - a["phi"] + math.pi) % (2 * math.pi) - math.pi) / d
    mB = dec.m0(511)
    out = []
    for sgn, pid, lab in ((-1, 511, "soft B0"), (+1, -511, "soft Bbar0")):
        y, f = ym + sgn * 0.5 * sep * ux, fm + sgn * 0.5 * sep * uy
        mt = math.sqrt(mB * mB + pt_pair * pt_pair)
        p4 = np.array([mt * math.cosh(y), pt_pair * math.cos(f), pt_pair * math.sin(f), mt * math.sinh(y)])
        out.append((p4, flavour_of(pid), lab))
    return out, (a["rank"], b["rank"])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(args, verbose=0, extra=None):
    ev, vertex, tau = dec.read_record(args.input)
    st = Stage(ev, args, verbose=verbose, soft_pair=extra)
    return ev, st, SimpleNamespace(vertex=vertex, tau=tau)


_REPEAT_ARGS = None


def one_repeat(seed: int) -> dict:
    """Decay the chapter-5 primary hadrons with this seed and cluster the result (a worker of --repeat)."""
    dargs = SimpleNamespace(input=HERE.parent.parent / "Had" / "event_Had.lhe", tau0max=dec.TAU0_MAX, limit_tau0=True, no_mix=False, stable_b=False, radius=0.4)
    e2, d2 = dec.run(dargs, seed, verbose=0)
    st2 = Stage(e2, _REPEAT_ARGS, verbose=0)
    return stage_summary(st2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=HERE.parent.parent / "Decay" / "event_Decay.lhe")
    ap.add_argument("-o", "--output", type=Path, default=HERE.parent / "event_GenJet.lhe")
    ap.add_argument("--history", type=Path, help="write the summary (and with --repeat the per-event statistics) as JSON")
    ap.add_argument("--algorithm", choices=sorted(POWER), default="antikt")
    ap.add_argument("--ptmin", type=float, default=JET_PT_MIN, help="jets below this pT are dropped (CMSSW: 3 GeV)")
    ap.add_argument("--radius", type=float, default=None, help="use this R for the ak4 collections instead of 0.4")
    ap.add_argument("--no-flavour-kt", action="store_true", help="skip the flavour-kT clustering (slow for many repeats)")
    ap.add_argument("--soft-pair", type=float, default=None, metavar="PT", help="add a soft B0 Bbar0 pair of this pT between two light jets (the infrared test)")
    ap.add_argument("--soft-sep", type=float, default=0.5, help="dR between the two soft hadrons")
    ap.add_argument("--repeat", type=int, default=0, help="decay the chapter-5 hadrons N times (seeds seed..seed+N-1), cluster each, average")
    ap.add_argument("--jobs", type=int, default=1, help="--repeat: run this many worker processes")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="learning mode: -v the b jet step by step, -vv every merge")
    args = ap.parse_args()

    extra = None
    if args.soft_pair is not None:
        ev0, st0, _ = run(args, 0)
        extra, between = soft_pair(ev0, st0.coll["ak4"].info, args.soft_pair, args.soft_sep)
        if extra is None:
            sys.exit("no pair of light jets found for the soft pair")
        print(f"soft pair: a B0 and a Bbar0 with pT = {args.soft_pair:g} GeV, {args.soft_sep:g} apart, between ak4 jets {between[0]} and {between[1]} "
              f"(at y = {rapidity(extra[0][0]):.2f}, {rapidity(extra[1][0]):.2f}; phi = {phi(extra[0][0]):.2f}, {phi(extra[1][0]):.2f})")
    ev, st, vt = run(args, args.verbose, extra)
    s = stage_summary(st)
    if args.verbose:
        explain(st, s)
        print(banner("summary"))
    if not args.quiet:
        print_summary(st, s)
    if args.soft_pair is not None:
        print("with the soft pair: jets that hold one of the two soft hadrons:")
        for key, c in st.coll.items():
            for j in c.info:
                for i in c.cl.constituents(j["index"]):
                    if c.tags[i][0] == "extra":
                        print(f"  {key} jet [{j['rank']}] pT = {j['pt']:.2f}: holds the {st.extra[c.tags[i][1]][2]}; hadronFlavour would be 5 by the ghost rule")
        for j in st.stable_jets:
            if any(lab.startswith("soft") for lab in j["flavoured"]):
                print(f"  heavy hadrons kept, anti-kT jet [{j['rank']}] pT = {j['pt']:.2f}: net {flavour_text(j['net_flavour'])}, IFN {flavour_text(j['ifn_flavour'])} ({' '.join(j['flavoured'])})")
        for j in st.fkt_jets:
            if any(lab.startswith("soft") for lab in j["flavoured"]):
                print(f"  flavour-kT jet pT = {j['pt']:.2f}: flavour {flavour_text(j['flavour'])} ({' '.join(j['flavoured'])})")
    write_record = (args.soft_pair is None and args.repeat == 0) or args.output != HERE.parent / "event_GenJet.lhe"
    if write_record:
        title = "Standalone Event Listing  (generator-level jets: hard process + showers + MPI + beam remnants + colour reconnection + hadronisation + decays, clustered)"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(dec.listing(ev, title, vt) + jet_tables(st, s))
        print(f"wrote {args.output} ({len(ev.p)} rows, {s['n_final']} final particles, {sum(c['n_jets'] for c in s['collections'].values())} jets in three collections)")
    if args.history and args.repeat == 0:
        args.history.write_text(json.dumps(dict(seed=args.seed, summary=s), indent=1, default=float) + "\n")
        print(f"wrote {args.history}")

    if args.repeat > 1:
        global _REPEAT_ARGS
        _REPEAT_ARGS = args
        seeds = list(range(args.seed, args.seed + args.repeat))
        if args.jobs > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(args.jobs) as pool:
                results = pool.imap(one_repeat, seeds, chunksize=1)
                summaries = []
                for n, s2 in enumerate(results, start=1):
                    summaries.append(s2)
                    if not args.quiet and n % 20 == 0:
                        print(f"  {n} / {args.repeat} decays clustered", flush=True)
        else:
            summaries = []
            for n, seed in enumerate(seeds, start=1):
                summaries.append(one_repeat(seed))
                if not args.quiet and n % 20 == 0:
                    print(f"  {n} / {args.repeat} decays clustered", flush=True)
        keys = ["n_jets", "n_clustered", "n_jets_10", "n_jets_30"]
        acc = {k: {c: [] for c in COLLECTIONS} for k in keys}
        jkeys = ["pt", "eta", "phi", "m", "n", "n_charged", "dr", "response", "pt_cone", "n_cone", "pt_from_hadron", "n_from_hadron", "n_hadron_outside", "pt_nu", "n_nu",
                 "hadron_flavour", "parton_flavour", "physics_flavour", "legacy_flavour", "dr_hadron"]
        fam = {}
        n_ifn, n_double_b = [], []
        for s2 in summaries:
            for c in COLLECTIONS:
                for k in keys:
                    acc[k][c].append(s2["collections"][c][k])
            n_ifn.append(len(s2["ifn"]))
            n_double_b.append(sum(1 for j in s2["collections"]["ak4"]["jets"] if len(j["b_hadrons"]) >= 2))
            for q, r in s2["b"].items():
                d = fam.setdefault(q, dict(pt_quark=r["pt_quark"], hadron=[], jets={c: {k: [] for k in jkeys + ["found", "composition"]} for c in COLLECTIONS},
                                           net=[], ifn=[], fkt_flavour=[], fkt_pt=[], fkt_response=[]))
                d["hadron"].append(r["hadron"])
                for c in COLLECTIONS:
                    j = r["jets"][c]
                    d["jets"][c]["found"].append(j is not None)
                    for k in jkeys:
                        d["jets"][c][k].append(j[k] if j is not None else float("nan"))
                    d["jets"][c]["composition"].append({k: v["pt"] for k, v in j["composition"].items()} if j is not None else None)
                sj, fj = r.get("stable_jet"), r.get("fkt_jet")
                d["net"].append(sj["net_flavour"].get(5, 0) if sj else 0)
                d["ifn"].append(sj["ifn_flavour"].get(5, 0) if sj else 0)
                d["fkt_flavour"].append(fj["flavour"].get(5, 0) if fj else 0)
                d["fkt_pt"].append(fj["pt"] if fj else float("nan"))
                d["fkt_response"].append(fj["response"] if fj else float("nan"))
        print(f"averages over {args.repeat} decays of the same primary hadrons (seeds {args.seed}..{args.seed + args.repeat - 1}), clustered each time:")
        for c in COLLECTIONS:
            print(f"  {COLLECTIONS[c]['title']:16s}: " + ", ".join(f"{k} {np.mean(acc[k][c]):.2f} +- {np.std(acc[k][c]):.2f}" for k in keys))
        print(f"  IFN neutralisations per event {np.mean(n_ifn):.2f}; ak4 jets with two or more b hadrons {np.mean(n_double_b):.2f} per event")
        for q, d in fam.items():
            print(f"  {q} quark (pT {d['pt_quark']:.2f}):")
            for c in COLLECTIONS:
                jj = d["jets"][c]
                resp = np.array(jj["response"], dtype=float)
                ok = ~np.isnan(resp)
                print(f"    {COLLECTIONS[c]['title']:16s}: found {100 * np.mean(jj['found']):.1f}%, pT {np.nanmean(jj['pt']):.2f} +- {np.nanstd(jj['pt']):.2f}, response {np.nanmean(resp):.3f} +- {np.nanstd(resp):.3f}, "
                      f"dR {np.nanmean(jj['dr']):.3f}, n {np.nanmean(jj['n']):.1f}, nu pT {np.nanmean(jj['pt_nu']):.2f}; hadronFlavour 5 in {100 * np.mean(np.array(jj['hadron_flavour'])[ok] == 5):.1f}%, "
                      f"partonFlavour +-5 in {100 * np.mean(np.abs(np.array(jj['parton_flavour'])[ok]) == 5):.1f}%, cone of the same R {np.nanmean(jj['pt_cone']):.2f}")
            print(f"    net b flavour of the anti-kT jet (heavy hadrons kept): " + ", ".join(f"{k:+d}: {100 * v / len(d['net']):.1f}%" for k, v in sorted(dec.count_ids(d['net']).items()))
                  + "; IFN: " + ", ".join(f"{k:+d}: {100 * v / len(d['ifn']):.1f}%" for k, v in sorted(dec.count_ids(d['ifn']).items()))
                  + ("" if args.no_flavour_kt else "; flavour-kT: " + ", ".join(f"{k:+d}: {100 * v / len(d['fkt_flavour']):.1f}%" for k, v in sorted(dec.count_ids(d['fkt_flavour']).items()))
                     + f", pT {np.nanmean(d['fkt_pt']):.2f}"))
        if args.history:
            stats = dict(repeat=args.repeat, seed0=args.seed, counts=acc, n_ifn=n_ifn, n_double_b=n_double_b, b=fam,
                         params=dict(algorithm=args.algorithm, ptmin=args.ptmin, flavour_kt=not args.no_flavour_kt))
            out = args.history.with_suffix(".stats.json")
            out.write_text(json.dumps(stats, default=float) + "\n")
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
