#!/usr/bin/env python3
"""The crossing of the border inside Geant4: the b bbar event of chapter 9 through the CMS detector.

This is the one stop of the road where the "standalone" script is not written from the textbook: it runs the
official Geant4 toolkit (through its Python bindings, geant4-pybind) in the official CMS simulation geometry.
standalone_SIM.py next to it stays the readable version of the same physics, with one formula per effect;
this script is the accurate one, and its hits are the record the road continues with.

What is official and what is ours
---------------------------------
* The detector is the CMS simulation geometry itself: ../../G4/geometry/cmsSimGeo2021.gdml, written by
  export_gdml_SIM.py from the TGeo dump of chapter 9 (3398 volume types, 490 materials, 2.1 million placed
  volumes: every pixel module, every crystal, every scintillator tile, every drift cell).
* The physics is Geant4's reference list FTFP_BERT: the standard electromagnetic processes (ionisation with
  delta rays, multiple scattering, bremsstrahlung, pair production, Compton scattering, the photoelectric
  effect, annihilation), decays in flight, and for the hadrons the Bertini cascade below about 5 GeV and the
  Fritiof string model with the Precompound de-excitation above. CMSSW runs FTFP_BERT_EMM, the same hadronic
  physics with a faster variant of the electromagnetic one in the calorimeters.
* The magnetic field is ours: field_map_SIM.py solves Ampere's law for the coil and the iron of this geometry
  (3.8 T at the centre, about 2 T the other way in the return yoke). Inside the tracker and the calorimeters,
  where a shower makes millions of steps, Geant4 is given a uniform 3.8 T (a compiled class; the map differs
  from it by 3 % at the end of the tracker and 13 % at the end of the endcap calorimeter); everywhere else it
  asks the map.
* The sensitive detectors are ours, and simple: a volume is sensitive by its material and its name (the
  "Active" silicon of the tracker, the lead-tungstate crystals, the preshower silicon, the scintillator
  tiles, the gas of the muon chambers). A tracker or muon hit is one crossing of one sensor by one charged
  particle (entry point, time, energy lost, momentum). A calorimeter hit is the energy left in one cell by the
  descendants of one primary; the cell is found from the position of the deposit with the cell grid of
  standalone_SIM.py (CMSSW finds it from the copy numbers of the crystal, a few millimetres more exact).
* The forward calorimeter HF is not showered, exactly as in CMSSW: a particle that enters its steel is
  stopped and its energy is turned into photoelectrons by the parametrisation of standalone_SIM.py (CMSSW
  uses a library of pre-simulated showers there). Particles are also stopped behind the HF, |z| > 13 m,
  where CMSSW has its dead region towards the ZDC, and 500 ns after their bunch crossing.
* Not done: Birks' law in the scintillator, the Roman pots and the ZDC, the production cuts per region of
  CMSSW (Geant4's default of 0.7 mm is used everywhere).

Every primary is one Geant4 event, so every hit knows the primary it descends from without any bookkeeping
of track parentage, and the events are spread over --jobs processes. The random seed of an event depends
only on (--seed, collision, primary): the result does not depend on the number of processes.

  --sample signal    the family, 672 primaries (about two minutes on 10 processes); also writes the
                     trajectories of every particle in the tracker and in the muon system and the energy
                     deposits of the showers on a 3D grid (../g4tracks_signal.npz), for the 3D displays
  --sample intime    the 63 collisions of bunch crossing 0
  --sample oot       the 897 collisions of the fifteen other crossings (hours: see the chapter)

Each writes ../simhits_<sample>.npz in the format of standalone_SIM.py and geant4_SIM_<sample>.json.

Run with the interpreter that has Geant4 (make venv-sim: python -m venv .venv; pip install geant4-pybind
numpy scipy; the first import downloads Geant4's data tables, 2 GB, to ~/.geant4_pybind):

  .venv/bin/python geant4_SIM.py --sample signal -v          # the family, learning mode
  .venv/bin/python geant4_SIM.py --sample signal --only 590  # one primary (row of the list), with every step group printed
  .venv/bin/python geant4_SIM.py --scan                      # geantinos: the material in front of the calorimeters

All energies in GeV, lengths in cm, times in ns (Geant4 itself thinks in MeV and mm).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import standalone_SIM as sim  # noqa: E402  (the primaries of chapter 9, the cell codes, the file format)

# the Run-3 simulation geometry of CMS; another dump (a 2025 one written from CMSSW, see the companion) is used with
# JITJET_SIM_GDML=/path/to/cmsSimGeo2025.gdml
GDML = Path(os.environ.get("JITJET_SIM_GDML", HERE.parent.parent / "G4" / "geometry" / "cmsSimGeo2021.gdml"))
B_INNER = 3.8                     # tesla, the uniform field inside the tracker and the calorimeters
T_MAX = 500.0                     # ns after the bunch crossing (SteppingAction.MaxTrackTime of CMSSW)
Z_DEAD = 1300.0                   # cm: behind the HF everything is stopped
E_SEC = 0.010                     # GeV: a secondary born in the tracker is "stored" above this kinetic energy
E_REC_IN, E_REC_OUT = 0.0002, 0.020   # GeV: trajectories are kept above this kinetic energy (tracker; muon system)
VOX_ECAL, VOX_HCAL = 0.5, 1.0     # cm: the 3D grid of the shower deposits kept for the displays
HF_MIN = 0.3                      # GeV: below this a particle entering the HF leaves no light

# Birks' law as CMSSW applies it (SimG4Core/Application/python/g4SimHits_cfi.py; CaloSD::getAttenuation and
# ECalSD::getBirkL3): the light of a step is its energy deposit times a weight that falls with dE/dx.
# Scintillator (HCalSD): weight = 1 / (1 + kB dE/dx + C (kB dE/dx)^2), kB = BirkC1 / density, C = BirkC2, and kB
# divided by BirkC3 for a particle of charge 2 or more. Crystals (ECalSD, the "L3" form): weight = 1 - BirkSlope
# x ln(kB dE/dx), kept between BirkCut and 1.
BIRK_HCAL = (0.0060, 0.142, 1.75)        # BirkC1 in g / (MeV cm2), BirkC2, BirkC3
BIRK_ECAL = (0.03333, 0.253694, 0.1)     # BirkC1 in g / (MeV cm2), BirkSlope, BirkCut

# The production cuts of CMSSW, region by region (the *ProdCuts.xml files of the 2025 geometry; in mm): a
# secondary electron, positron or photon is created only if it would travel further than this in the material
# where it is born; below, its energy is left on the spot. (name, cut, pattern of the volume names at the root)
# CMSSW gives photons and electrons different cuts in the muon system; here one number per region, the electron's.
CUT_WORLD = 10.0
REGIONS = [
    ("TrackerDeadRegion", 1.0, r"^trackerTracker$"),
    ("TrackerPixelDeadRegion", 1.0, r"PixelBarrel$|PixelForwardZ(plus|minus)$"),
    ("TrackerSensRegion", 0.1, r"(TIB|TOB)(Active|Wafer)|(TID|TEC)Module\d(Rphi|Stereo)(Active|Wafer)"),
    ("TrackerPixelSensRegion", 0.01, r"PixelBarrelActiveFull\d$|PixelForwardSensor\d?$"),
    ("BeamPipe", 0.01, r"BeamTube1$"),
    ("BeamPipeOutside", 1.0, r"^beampipeBEAM\d*$"),
    ("CaloRegion", 1.0, r"^caloBaseCALO$|algoMBAT$|algoMBBT|algoVCAL$"),          # EcalRegion and HcalRegion: both 1 mm
    ("PreshowerSensRegion", 0.1, r"algoSFS[XY]$"),
    ("Muon", 10.0, r"^muonBaseMUON$"),
    ("MuonIron", 1.0, r"MGNT$|YBSepar|[a-z]COI|COCF$|[a-z]YB|ZStopSpace$|SupportTubeForHE|[a-z]YN|[a-z]YE[1-4]|MB4Shield|ShieldingME4|ShieldingAnti|ShieldingBoronPoly|ME11SupportDisk$"),
    ("MuonChamber", 0.002, r"[a-z0-9]MB\w*[NP]$|[a-z0-9]ME\w*Space$|[a-z0-9]RR\w*$|[a-z0-9]GE\d"),
    ("MuonSensitive_DT-CSC", 1.0, r"MB\w*PhiGas$|MB\w*ZGas$|ME\w*ActiveGasVol$"),
    ("MuonSensitive_GEM", 0.002, r"[a-z0-9]GHA\d"),
    ("MuonSensitive_RPC", 8.0, r"MB\w*RPC_\w*Gas|[a-z0-9]RE[BCEFGH]\d"),
]
# what is added up for every primary (GeV): the energy left in all the material of the tracker volume and of the
# muon system, in the crystals, the preshower silicon and the scintillator before Birks' law (the light, after
# it, is the sum of the hits), the energy stopped at the HF and behind it
TALLY = ["n", "e_gate", "trkvol", "muvol", "eb", "ee", "es", "hb", "he", "ho", "eb_vis", "ee_vis", "es_vis", "hb_vis", "he_vis", "ho_vis",
         "si", "gas", "hf_in", "dead"]

INNER = ("trackerTracker", "beampipeBEAM", "trackerpixelnoseTrackerPixelNose")      # trajectories of everything
OUTER = ("muonBaseMUON",)                                                           # trajectories of charged particles
UNIFORM = ("trackerTracker", "caloBaseCALO", "beampipeBEAM", "trackerpixelnoseTrackerPixelNose")
DEAD = ("cmsextentCMStoZDC", "cmsextentZDCtoFP420", "zdcZDC", "forwardshieldOQUA", "brmBSC2",
        "beampipeBEAM1", "beampipeBEAM2", "beampipeBEAM3", "beampipeBEAM4")
FERRO = ("materialsM_Steel-008", "materialsSteel-008", "materialsIron", "materialsYokeSteel", "materialsAISI-1018-Steel")
TRK_PREFIX = {"pixbar": sim.PXB, "pixfwd": sim.PXF, "tib": sim.TIB, "tid": sim.TID, "tob": sim.TOB, "tec": sim.TEC}

# Geant4's process names -> the words of the chapter
PROC_WORD = {"Decay": "decay", "conv": "conversion", "eBrem": "bremsstrahlung", "compt": "Compton scattering",
             "phot": "photoelectric effect", "annihil": "annihilation", "eIoni": "delta ray", "hIoni": "delta ray",
             "muIoni": "delta ray", "ionIoni": "delta ray", "hBrems": "bremsstrahlung", "muBrems": "bremsstrahlung",
             "hPairProd": "pair production", "muPairProd": "pair production", "hadElastic": "elastic scattering",
             "CoulombScat": "elastic scattering", "photonNuclear": "photonuclear", "electronNuclear": "electronuclear",
             "positronNuclear": "electronuclear", "muonNuclear": "muon-nuclear", "nCapture": "neutron capture",
             "Rayl": "Rayleigh scattering", "hBertiniCaptureAtRest": "capture at rest", "muMinusCaptureAtRest": "capture at rest"}
PROC_CODE = {"primary": 0, "decay": 1, "conversion": 2, "bremsstrahlung": 3, "nuclear interaction": 4, "delta ray": 5,
             "Compton scattering": 6, "photoelectric effect": 7, "annihilation": 8, "elastic scattering": 9, "other": 10}
PROC_OF_CODE = {v: k for k, v in PROC_CODE.items()}
FATES = ["calorimeter", "decay", "conversion", "nuclear", "stopped", "beam pipe"]
# the physics switched on one family of processes at a time (--stage): field, decays, electromagnetic, hadronic
STAGES = {"transport": (0, 0, 0, 0), "B": (1, 0, 0, 0), "B_Decay": (1, 1, 0, 0), "B_Decay_EM": (1, 1, 1, 0), "full": (1, 1, 1, 1)}
STAGE_LABEL = {"transport": "transport only (chapter 9)", "B": "+ magnetic field", "B_Decay": "+ decays in flight",
               "B_Decay_EM": "+ electromagnetic physics", "full": "+ hadronic physics (the record)"}


def word(proc: str) -> str:
    w = PROC_WORD.get(proc)
    if w is None:
        w = "nuclear interaction" if proc.endswith("Inelastic") else "other"
    return w


def code_of(proc: str) -> int:
    return PROC_CODE.get(word(proc), PROC_CODE["other"])


# ---------------------------------------------------------------------------
# cells from positions (the grid of standalone_SIM.py; the volume already says which subdetector it is)
# ---------------------------------------------------------------------------
def eb_codes(P):
    r = np.hypot(P[:, 0], P[:, 1])
    eta = np.arcsinh(P[:, 2] / np.maximum(r, 1e-9))
    ie = np.clip(np.floor(eta / sim.D_ETA_EB).astype(np.int64) + 85, 0, 169)
    ip = np.floor((np.arctan2(P[:, 1], P[:, 0]) + math.pi) / (2 * math.pi / 360)).astype(np.int64) % 360
    return sim.EB * sim.CODE + ie * 360 + ip


def ee_codes(P):
    n = sim.NX_EE
    sc = sim.Z_EE / np.maximum(np.abs(P[:, 2]), 1e-9)
    ix = np.clip(np.floor(P[:, 0] * sc / sim.X_EE).astype(np.int64) + n // 2, 0, n - 1)
    iy = np.clip(np.floor(P[:, 1] * sc / sim.X_EE).astype(np.int64) + n // 2, 0, n - 1)
    return sim.EE * sim.CODE + ((P[:, 2] > 0).astype(np.int64) * n + ix) * n + iy


def es_codes(P, plane):
    side = (P[:, 2] > 0).astype(np.int64)
    out = np.zeros(len(P), np.int64)
    for pl in (0, 1):
        m = plane == pl
        if m.any():
            out[m] = sim.es_codes(P[m, 0], P[m, 1], side[m], pl)
    return out


def _tower(P, lo, hi):
    r = np.hypot(P[:, 0], P[:, 1])
    ae = np.abs(np.arcsinh(P[:, 2] / np.maximum(r, 1e-9)))
    ieta = np.clip(np.searchsorted(sim.ETA_EDGES, ae, side="right"), lo, hi)
    nphi = np.where(ieta <= 20, 72, 36)
    ip = np.floor((np.arctan2(P[:, 1], P[:, 0]) + math.pi) / (2 * math.pi) * nphi).astype(np.int64) % nphi
    return r, ieta, ip, (P[:, 2] > 0).astype(np.int64)


def hb_codes(P):
    r, ieta, ip, side = _tower(P, 1, 16)
    dep = np.minimum(np.clip(np.searchsorted(sim.HB_R, r, side="right"), 1, 4), sim.ND_HBHE[ieta])
    return sim.HB * sim.CODE + ((side * 32 + ieta) * 8 + dep) * 72 + ip


def he_codes(P):
    r, ieta, ip, side = _tower(P, 17, 29)
    dep = np.minimum(np.clip(np.searchsorted(sim.HE_Z, np.abs(P[:, 2]), side="right"), 1, 7), sim.ND_HBHE[ieta])
    return sim.HE * sim.CODE + ((side * 32 + ieta) * 8 + dep) * 72 + ip


def ho_codes(P):
    r, ieta, ip, side = _tower(P, 1, 15)
    return sim.HO * sim.CODE + (side * 32 + ieta) * 72 + ip


CAL_CODES = {"eb": eb_codes, "ee": ee_codes, "hb": hb_codes, "he": he_codes, "ho": ho_codes}
CAL_DET = {"eb": sim.EB, "ee": sim.EE, "es": sim.ES, "hb": sim.HB, "he": sim.HE, "ho": sim.HO}


def tracker_layer(det, x, y, z):
    """Layer or disk number of a tracker hit, from its position (the radii and positions of standalone_SIM.py)."""
    r = np.hypot(x, y)
    lay = np.zeros(len(det), np.int8)
    for d in sim.TRACKER:
        m = det == d
        if not m.any():
            continue
        if d in (sim.PXB, sim.TIB, sim.TOB):
            rr = np.array([b[2] for b in sim.BARREL if b[0] == d])
            lay[m] = np.argmin(np.abs(r[m][:, None] - rr[None, :]), axis=1) + 1
        else:
            zz = np.array([b[2] for b in sim.DISKS_1 if b[0] == d])
            lay[m] = np.argmin(np.abs(np.abs(z[m])[:, None] - zz[None, :]), axis=1) + 1
    return lay


def muon_station(det, x, y, z):
    r, az = np.hypot(x, y), np.abs(z)
    st = np.zeros(len(det), np.int8)
    m = np.isin(det, (sim.DT,)) | ((det == sim.RPC) & (az < 670.0))
    st[m] = np.clip(np.searchsorted([470.0, 560.0, 660.0], r[m]) + 1, 1, 4)
    m2 = ~m
    st[m2] = np.clip(np.searchsorted([750.0, 880.0, 980.0], az[m2]) + 1, 1, 4)
    return st


# ---------------------------------------------------------------------------
# the Geant4 application (one per process)
# ---------------------------------------------------------------------------
_APP = None
_KEEP = []


class App:
    """Geometry, field, physics list, sensitive detectors and the event loop. Imported lazily: only the worker
    processes load Geant4."""

    def __init__(self, record_tracks=False, field=True, quiet=True, physics="FTFP_BERT", scan=False, birks=True, regions=True, stage="full", trk_only=False):
        import geant4_pybind as g
        self.trk_only = trk_only                         # stop everything at the calorimeters (for the radiography)
        self.stage = stage
        field = field and bool(STAGES[stage][0])
        self.count_all = not STAGES[stage][2]            # without ionisation a hit is a crossing of a sensor
        self.g = g
        self.record_tracks = record_tracks
        self.scan = scan
        self.birks, self.regions, self.region_roots = birks, regions and not scan, {}
        self.cur = Counter()
        self.MeV, self.cm, self.ns, self.GeV = g.MeV, g.cm, g.ns, g.GeV
        ui = g.G4UImanager.GetUIpointer()
        self.rm = g.G4RunManagerFactory.CreateRunManager(g.G4RunManagerType.Serial)
        app = self

        # ---------------- sensitive detectors
        class CalSD(g.G4VSensitiveDetector):
            """Crystals, preshower silicon, scintillator: every step with an energy deposit is kept."""
            def __init__(self, name, plane=-1):
                super().__init__(name)
                self.buf = []
                self.plane = plane
                self.key = name[:2]
                self.birk = 0 if not app.birks else (1 if name in ("hb", "he", "ho") else (2 if name in ("eb", "ee") else 0))
                self.rkb = None

            def ProcessHits(self, step, hist):
                if app.trk_only:
                    step.GetTrack().SetTrackStatus(g.fKillTrackAndSecondaries)
                    return True
                e = step.GetTotalEnergyDeposit()
                if e > 0.0:
                    pre = step.GetPreStepPoint()
                    t = pre.GetGlobalTime()
                    if t > app.t_kill:                              # later than 500 ns: nobody is listening any more
                        step.GetTrack().SetTrackStatus(g.fStopAndKill)
                        app.n["time_cut"] += 1
                        return True
                    app.cur[self.key] += e
                    if self.birk:
                        q = pre.GetCharge()
                        L = step.GetStepLength()
                        if q != 0.0 and L > 0.0:
                            if self.rkb is None:
                                self.rkb = (BIRK_HCAL if self.birk == 1 else BIRK_ECAL)[0] * g.g / (g.MeV * g.cm2) / pre.GetMaterial().GetDensity()
                            x = self.rkb * e / L
                            if self.birk == 1:
                                c = BIRK_HCAL[1] * x * x
                                if abs(q) >= 2.0:
                                    x /= BIRK_HCAL[2]
                                e /= 1.0 + x + c
                            else:
                                e *= min(1.0, max(BIRK_ECAL[2], 1.0 - BIRK_ECAL[1] * math.log(x)))
                    p = pre.GetPosition()
                    self.buf.append((p.x, p.y, p.z, t, e))
                return True

        class RecSD(g.G4VSensitiveDetector):
            """Every volume of the tracker and of the muon system: the trajectory, the vertices, and a hit where
            the volume is a sensor (det > 0)."""
            def __init__(self, name, det, emin, charged_only):
                super().__init__(name)
                self.det, self.emin, self.charged_only = det, emin * g.GeV, charged_only
                self.vol = "muvol" if charged_only else "trkvol"

            def ProcessHits(self, step, hist):
                tr = step.GetTrack()
                d = tr.GetDefinition()
                q = d.GetPDGCharge()
                post = step.GetPostStepPoint()
                t = post.GetGlobalTime()
                tid = tr.GetTrackID()
                if t > app.t_kill:
                    tr.SetTrackStatus(g.fStopAndKill)
                    app.n["time_cut"] += 1
                    return True
                e = step.GetTotalEnergyDeposit()
                if e > 0.0:
                    app.cur[self.vol] += e
                if self.det and q != 0.0:
                    if e > 0.0 or app.count_all:
                        pre = step.GetPreStepPoint()
                        p = pre.GetPosition()
                        app.hits.append((tid, self.det, p.x, p.y, p.z, pre.GetGlobalTime(), e, pre.GetMomentum().mag(),
                                         d.GetPDGEncoding(), post.GetStepStatus() == g.fGeomBoundary))
                if self.charged_only and q == 0.0:
                    return True
                ek = post.GetKineticEnergy()
                known = app.known
                if tid not in known:
                    pre = step.GetPreStepPoint()
                    ek0 = pre.GetKineticEnergy()
                    if ek0 < self.emin and tid != 1:
                        return True
                    cp = tr.GetCreatorProcess()
                    first = tr.GetCurrentStepNumber() == 1
                    known[tid] = len(app.births)
                    p = pre.GetPosition()
                    app.births.append((tid, tr.GetParentID(), d.GetPDGEncoding(), cp.GetProcessName() if cp is not None else "primary",
                                       p.x, p.y, p.z, pre.GetGlobalTime(), ek0, first, self.charged_only))
                    app.pts.append((tid, p.x, p.y, p.z, pre.GetGlobalTime(), ek0))
                p = post.GetPosition()
                app.pts.append((tid, p.x, p.y, p.z, t, ek))
                ns = step.GetNumberOfSecondariesInCurrentStep()
                dead = tr.GetTrackStatus() != g.fAlive
                if ns or dead or tid == 1:
                    pd = post.GetProcessDefinedStep()
                    proc = pd.GetProcessName() if pd is not None else "none"
                    if ns or dead:
                        app.n[proc if not self.charged_only else "out_" + proc] += 1      # out_: behind the calorimeters
                        if not self.charged_only and proc not in ("eIoni", "hIoni", "muIoni", "ionIoni", "msc", "Transportation"):
                            app.verts.append((proc, tid, p.x, p.y, p.z, d.GetPDGEncoding(), step.GetPreStepPoint().GetKineticEnergy(), ns, t))
                    if tid == 1:
                        m = post.GetMomentum()
                        app.last1 = (proc, dead, p.x, p.y, p.z, m.x, m.y, m.z, self.charged_only)
                return True

        class HFSD(g.G4VSensitiveDetector):
            """The steel of the forward calorimeter: the particle stops here and the parametrisation answers."""
            def ProcessHits(self, step, hist):
                tr = step.GetTrack()
                d = tr.GetDefinition()
                pid = d.GetPDGEncoding()
                a = abs(pid)
                if a == 13:
                    return True
                pre = step.GetPreStepPoint()
                p = pre.GetPosition()
                ek = pre.GetKineticEnergy()
                e = ek if a in (2212, 2112) or a > 1000000000 else (ek + 2 * d.GetPDGMass() if pid in (-2212, -2112) else pre.GetTotalEnergy())
                app.hf.append((p.x, p.y, p.z, pre.GetGlobalTime(), e, 1.0 if a in (11, 22, 111) else 0.0))
                app.cur["hf_in"] += e
                tr.SetTrackStatus(g.fKillTrackAndSecondaries)       # and what this very step produced: its energy is counted above
                return True

        class DeadSD(g.G4VSensitiveDetector):
            def ProcessHits(self, step, hist):
                pre = step.GetPreStepPoint()
                if abs(pre.GetPosition().z) < app.z_dead:
                    return True
                tr = step.GetTrack()
                app.e["dead_region"] += pre.GetKineticEnergy()
                app.cur["dead"] += pre.GetKineticEnergy()
                app.n["dead_region"] += 1
                tr.SetTrackStatus(g.fKillTrackAndSecondaries)
                return True

        class ScanSD(g.G4VSensitiveDetector):
            """For the geantino scan: the material of every step."""
            def ProcessHits(self, step, hist):
                pre = step.GetPreStepPoint()
                p = pre.GetPosition()
                if abs(p.z) >= app.z_trk or p.x * p.x + p.y * p.y >= app.r2_trk:      # only up to the calorimeters
                    step.GetTrack().SetTrackStatus(g.fStopAndKill)
                    return True
                mat = pre.GetMaterial()
                L = step.GetStepLength()
                app.scan_x0 += L / mat.GetRadlen()
                app.scan_li += L / mat.GetNuclearInterLength()
                lv = pre.GetTouchable().GetVolume().GetLogicalVolume().GetName()
                if lv in app.sensor_det and L > 0:
                    app.scan_sensors[app.sensor_det[lv]] += 1
                return True

        class MapField(g.G4MagneticField):
            def __init__(self, fmap, iron):
                super().__init__()
                self.f, self.iron = fmap.b, iron

            def GetFieldValue(self, point, B):
                bx, by, bz = self.f(point[0] * 0.1, point[1] * 0.1, point[2] * 0.1, self.iron)
                B[0] = bx * g.tesla
                B[1] = by * g.tesla
                B[2] = bz * g.tesla

        class Detector(g.G4VUserDetectorConstruction):
            def Construct(self):
                self.parser = g.G4GDMLParser()
                self.parser.Read(str(GDML), False)
                world = self.parser.GetWorldVolume()
                if app.regions:
                    import re
                    spec = [(nm, re.compile(pat)) for nm, _, pat in REGIONS]
                    reg = {nm: g.G4Region(nm) for nm, _, _ in REGIONS}
                    _KEEP.append(reg)                                  # Geant4 owns them: never let Python free them
                    seen, count = set(), Counter()

                    def walk(lv, cur):
                        n = lv.GetName()
                        if n in seen:
                            return
                        seen.add(n)
                        for nm, rx in spec:
                            if rx.search(n):
                                if nm != cur:
                                    reg[nm].AddRootLogicalVolume(lv)
                                    count[nm] += 1
                                    cur = nm
                                break
                        for i in range(lv.GetNoDaughters()):
                            walk(lv.GetDaughter(i).GetLogicalVolume(), cur)

                    walk(world.GetLogicalVolume(), None)
                    app.region_roots = dict(count)
                return world

            def ConstructSDandField(self):
                sdm = g.G4SDManager.GetSDMpointer()
                world = self.parser.GetWorldVolume().GetLogicalVolume()
                keep = app.keep
                sd = {}

                def make(key, obj):
                    sd[key] = obj
                    sdm.AddNewDetector(obj)
                    return obj

                for k in ("eb", "ee", "hb", "he", "ho"):
                    make(k, CalSD(k))
                make("esx", CalSD("esx", 0)); make("esy", CalSD("esy", 1))
                make("hf", HFSD("hf")); make("dead", DeadSD("dead"))
                make("in", RecSD("in", 0, E_REC_IN, False)); make("out", RecSD("out", 0, E_REC_OUT, True))
                for nm, det in list(TRK_PREFIX.items()):
                    make("trk%d" % det, RecSD("trk%d" % det, det, E_REC_IN, False))
                for det in sim.MUON:
                    make("mu%d" % det, RecSD("mu%d" % det, det, E_REC_OUT, True))
                app.sd = sd
                counts = Counter()
                app.sensor_det = {}

                def sensor(lv):
                    n, m = lv.GetName(), lv.GetMaterial().GetName()
                    if m == "materialsE_PbWO4":
                        return "eb" if n.startswith("ebalgo") else "ee"
                    if m == "materialsE_Silicon" and n.startswith("esalgoSFS"):
                        return "esx" if n.endswith("X") else "esy"
                    if m == "materialsScintillator":
                        return "hb" if n.startswith("hcalbarrel") else ("he" if n.startswith("hcalendcap") else ("ho" if n.startswith("hcalouter") else None))
                    if m == "materialsSilicon" and ("Active" in n or "ForwardSensor" in n):
                        for pre, det in TRK_PREFIX.items():
                            if n.startswith(pre):
                                return "trk%d" % det
                    if m == "materialsM_DTBX_Gas":
                        return "mu%d" % sim.DT
                    if m == "materialsM_Argon_CF_4_CO_2" and "ActiveGas" in n:
                        return "mu%d" % sim.CSC
                    if m == "materialsM_RPC_Gas":
                        return "mu%d" % sim.RPC
                    if m == "gemfM_GEM_Gas":
                        return "mu%d" % sim.GEM
                    if n == "hcalforwardalgoHVQX":
                        return "hf"
                    return None

                seen = set()

                def walk(lv, region):
                    n = lv.GetName()
                    if n in INNER:
                        region = "in"
                    elif n in OUTER:
                        region = "out"
                    elif n in DEAD:
                        region = "dead"
                    if (n, region) in seen:
                        return
                    seen.add((n, region))
                    if app.scan:
                        lv.SetSensitiveDetector(sd_scan)
                        s = sensor(lv)
                        if s and s.startswith("trk"):
                            app.sensor_det[n] = int(s[3:])
                    else:
                        s = "dead" if region == "dead" else sensor(lv)
                        if s is None and region in ("in", "out"):
                            s = region
                        if s is not None:
                            lv.SetSensitiveDetector(sd[s])
                            counts[s] += 1
                    for i in range(lv.GetNoDaughters()):
                        walk(lv.GetDaughter(i).GetLogicalVolume(), region)

                if app.scan:
                    sd_scan = make("scan", ScanSD("scan"))
                walk(world, None)
                app.sd_counts = dict(counts)
                if not app.field or app.scan:
                    return
                # the field: the map everywhere, its iron version inside the steel, and a uniform field
                # (compiled, fast) in the tracker and the calorimeters
                from field_map_SIM import FieldMap
                fmap = FieldMap()
                tm = g.G4TransportationManager.GetTransportationManager()
                fm = tm.GetFieldManager()
                f_air = MapField(fmap, False)
                fm.SetDetectorField(f_air)
                fm.CreateChordFinder(f_air)
                f_iron = MapField(fmap, True)
                fm_iron = g.G4FieldManager(f_iron)
                fm_iron.CreateChordFinder(f_iron)
                f_uni = g.G4UniformMagField(g.G4ThreeVector(0.0, 0.0, B_INNER * g.tesla))
                fm_uni = g.G4FieldManager(f_uni)
                fm_uni.CreateChordFinder(f_uni)
                keep += [fmap, f_air, f_iron, fm_iron, f_uni, fm_uni]
                done = set()

                def assign(lv):
                    n = lv.GetName()
                    if n in done:
                        return
                    done.add(n)
                    if n in UNIFORM:
                        lv.SetFieldManager(fm_uni, True)
                        return
                    if lv.GetMaterial().GetName() in FERRO:
                        lv.SetFieldManager(fm_iron, False)
                    for i in range(lv.GetNoDaughters()):
                        assign(lv.GetDaughter(i).GetLogicalVolume())

                assign(world)

        class Gun(g.G4VUserPrimaryGeneratorAction):
            def __init__(self):
                super().__init__()
                self.gun = g.G4ParticleGun(1)
                self.table = g.G4ParticleTable.GetParticleTable()

            def GeneratePrimaries(self, ev):
                j = ev.GetEventID()
                pid, px, py, pz, x, y, z, t, seed = app.todo[j]
                g.G4Random.setTheSeed(int(seed))
                d = self.table.FindParticle("geantino") if app.scan else self.table.FindParticle(int(pid))
                if d is None:
                    app.n["unknown_particle"] += 1
                    d = self.table.FindParticle("geantino")
                pp = math.sqrt(px * px + py * py + pz * pz)
                m = d.GetPDGMass()
                self.gun.SetParticleDefinition(d)
                self.gun.SetParticleMomentumDirection(g.G4ThreeVector(px / pp, py / pp, pz / pp))
                self.gun.SetParticleEnergy(math.sqrt((pp * g.GeV) ** 2 + m * m) - m)         # kinetic energy, Geant4's mass
                self.gun.SetParticlePosition(g.G4ThreeVector(x * g.cm, y * g.cm, z * g.cm))
                self.gun.SetParticleTime(t * g.ns)
                self.gun.GeneratePrimaryVertex(ev)

        class EventAction(g.G4UserEventAction):
            def BeginOfEventAction(self, ev):
                app.begin(ev.GetEventID())

            def EndOfEventAction(self, ev):
                app.end(ev.GetEventID())

        class Init(g.G4VUserActionInitialization):
            def Build(self):
                self.gun, self.ea = Gun(), EventAction()
                self.SetUserAction(self.gun)
                self.SetUserAction(self.ea)

        self.keep = []
        self.field = field
        self.det = Detector()
        self.pl = getattr(g, physics)()
        self.init = Init()
        self.rm.SetUserInitialization(self.det)
        self.rm.SetUserInitialization(self.pl)
        self.rm.SetUserInitialization(self.init)
        for cmd in ("/run/verbose 0", "/event/verbose 0", "/tracking/verbose 0", "/process/em/verbose 0", "/process/had/verbose 0"):
            ui.ApplyCommand(cmd)
        cut = self.pl.GetPhysics("neutronTrackingCut")          # slow neutrons live for microseconds: stop them at 500 ns too
        if cut is not None:
            cut.SetTimeLimit(T_MAX * g.ns)
        self.z_dead = Z_DEAD * g.cm
        self.z_trk, self.r2_trk = sim.Z_TRK * g.cm, (sim.R_TRK * g.cm) ** 2
        if self.regions:
            ui.ApplyCommand(f"/run/setCut {CUT_WORLD} mm")
        self.rm.Initialize()
        if stage != "full":
            pt = g.G4ProcessTable.GetProcessTable()
            _, dec_on, em_on, had_on = STAGES[stage]
            for on, types in ((dec_on, (g.fDecay,)), (em_on, (g.fElectromagnetic,)), (had_on, (g.fHadronic, g.fPhotolepton_hadron))):
                if not on:
                    for ty in types:
                        pt.SetProcessActivation(ty, False)
        if self.regions:
            for nm, cut, _ in REGIONS:
                if self.region_roots.get(nm):
                    ui.ApplyCommand(f"/run/setCutForRegion {nm} {cut} mm")
        self.version = str(g.G4Version).replace("$Name:", "").replace("$", "").strip()
        self.physics = physics
        self.reset()

    # ---------------------------------------------------------------- per task
    def reset(self):
        self.n, self.e = Counter(), Counter()
        self.out = dict(trk=[], mu=[], cal_c=[], cal_e=[], cal_t=[], cal_p=[], births=[], pts=[], verts=[], vox=[], fate={}, bnd={}, end={}, scan=[], tal=[])
        self.ncalhits = 0
        self.nsec = 0

    def begin(self, j):
        self.hits, self.births, self.pts, self.verts, self.hf = [], [], [], [], []
        self.known = {}
        self.last1 = None
        self.t_kill = (self.todo[j][7] + T_MAX) * self.ns
        self.rng = np.random.default_rng(self.todo[j][8])     # for the HF parametrisation: one stream per primary
        self.scan_x0 = self.scan_li = 0.0
        self.scan_sensors = Counter()
        self.cur = Counter()

    def end(self, j):
        o = self.out
        prim = self.prims[j]
        MeV, cm = self.MeV, self.cm
        if self.scan:
            o["scan"].append((prim, self.scan_x0, self.scan_li, dict(self.scan_sensors)))
            return
        # tracker and muon hits: the steps of one particle in one sensor are one hit
        if self.hits:
            h = self.hits
            tid = np.array([r[0] for r in h]); det = np.array([r[1] for r in h])
            a = np.array([r[2:8] for r in h], dtype=float)
            pid = np.array([r[8] for r in h]); end = np.array([r[9] for r in h], dtype=bool)
            new = np.ones(len(h), bool)
            new[1:] = (tid[1:] != tid[:-1]) | (det[1:] != det[:-1]) | end[:-1]
            st = np.flatnonzero(new)
            de = np.add.reduceat(a[:, 4], st) / 1e3
            x, y, z = a[st, 0] / cm, a[st, 1] / cm, a[st, 2] / cm
            d = det[st]
            for part, m in (("trk", d < 10), ("mu", d >= 10)):
                if m.any():
                    self.cur["si" if part == "trk" else "gas"] += float(de[m].sum()) * 1e3
                    lay = tracker_layer(d[m], x[m], y[m], z[m]) if part == "trk" else muon_station(d[m], x[m], y[m], z[m])
                    o[part].append(dict(det=d[m].astype(np.int8), lay=lay, x=x[m].astype(np.float32), y=y[m].astype(np.float32), z=z[m].astype(np.float32),
                                        t=a[st, 3][m].astype(np.float32), de=de[m].astype(np.float32), p=(a[st, 5][m] / 1e3).astype(np.float32),
                                        pid=pid[st][m].astype(np.int32), prim=np.full(int(m.sum()), prim, np.int32)))
        # calorimeters: energy per cell
        for key, s in self.sd.items():
            if not isinstance(getattr(s, "buf", None), list) or not s.buf:
                continue
            b = np.array(s.buf, dtype=float)
            s.buf.clear()
            P = b[:, :3] / cm
            e = b[:, 4] / 1e3
            self.ncalhits += len(b)
            self.cur[key[:2] + "_vis"] += float(b[:, 4].sum())
            codes = es_codes(P, np.full(len(b), s.plane)) if key in ("esx", "esy") else CAL_CODES[key](P)
            uc, inv = np.unique(codes, return_inverse=True)
            es = np.bincount(inv, weights=e)
            ts = np.bincount(inv, weights=e * b[:, 3]) / np.maximum(es, 1e-30)
            o["cal_c"].append(uc); o["cal_e"].append(es.astype(np.float32)); o["cal_t"].append(ts.astype(np.float32))
            o["cal_p"].append(np.full(len(uc), prim, np.int32))
            if self.record_tracks and key not in ("esx", "esy"):
                v = VOX_ECAL if key in ("eb", "ee") else VOX_HCAL
                q = np.floor(P / v).astype(np.int64)
                uq, inv = np.unique(q, axis=0, return_inverse=True)
                ev = np.bincount(inv.ravel(), weights=e)
                tv = np.bincount(inv.ravel(), weights=e * b[:, 3]) / np.maximum(ev, 1e-30)
                o["vox"].append(np.column_stack([np.full(len(uq), CAL_DET[key]), (uq + 0.5) * v, ev, tv, np.full(len(uq), prim)]))
        # the forward calorimeter: photoelectrons from the parametrisation
        for (x, y, z, t, e, em) in self.hf:
            e /= 1e3
            self.e["hf_in"] += e
            self.n["hf_in"] += 1
            if e < HF_MIN:
                continue
            kind = "em" if em else "had"
            kl, ks = sim.PE_HF[kind]
            fl = 1.0 if em else float(np.exp(0.35 * self.rng.standard_normal() - 0.06))
            side = 1 if z > 0 else 0
            for depth, k in ((0, kl), (1, ks)):
                npe = int(self.rng.poisson(k * e * fl))
                if npe == 0:
                    continue
                npe = min(npe, 200000)
                rr = self.rng.exponential(1.5 if em else 5.0, npe)
                ps = 2 * math.pi * self.rng.random(npe)
                codes = sim.hf_codes(x / cm + rr * np.cos(ps), y / cm + rr * np.sin(ps), side, depth)
                uc, cnt = np.unique(codes, return_counts=True)
                o["cal_c"].append(uc); o["cal_e"].append(cnt.astype(np.float32)); o["cal_t"].append(np.full(len(uc), t + 0.5, np.float32))
                o["cal_p"].append(np.full(len(uc), prim, np.int32))
        # the energy budget of this primary
        row = self.todo[j]
        cur = self.cur
        tal = np.array([cur.get(k, 0.0) for k in TALLY]) / 1e3
        tal[0] = 1.0
        tal[1] = math.sqrt(row[1] ** 2 + row[2] ** 2 + row[3] ** 2 + sim.dec.m0(int(row[0])) ** 2)
        o["tal"].append((prim, sim.kind_of(int(row[0])), tal))
        # secondaries stored, vertices, the fate of the primary
        for b in self.births:
            if b[1] > 0 and b[9] and not b[10] and b[8] >= E_SEC * 1e3:
                self.nsec += 1
                self.n["sec_" + word(b[3])] += 1
        if self.last1 is not None:
            proc, dead, x, y, z, px, py, pz, outer = self.last1
            x, y, z = x / cm, y / cm, z / cm
            r = math.hypot(x, y)
            if dead and not outer:
                w = word(proc)
                o["fate"][prim] = {"decay": "decay", "conversion": "conversion", "nuclear interaction": "nuclear"}.get(w, "stopped")
                o["end"][prim] = (x, y, z)
            elif not outer and r < 30.0 and abs(z) > 280.0:
                o["fate"][prim] = "beam pipe"
            else:
                o["fate"][prim] = "calorimeter"
                o["bnd"][prim] = (x, y, z, px / 1e3, py / 1e3, pz / 1e3)
        if self.record_tracks:
            if self.births:
                o["births"].append((prim, self.births))
                o["pts"].append((prim, np.array(self.pts, dtype=float)))
            if self.verts:
                o["verts"].append((prim, self.verts))
        elif self.verts:
            pass

    def run(self, todo, prims, seed_rng):
        """todo: rows (pid, px, py, pz, x, y, z, t, seed); prims: the label of each row."""
        self.reset()
        self.todo, self.prims = todo, prims
        self.rm.BeamOn(len(todo))
        return self.result()

    def result(self):
        o = self.out
        f4 = np.float32

        def cat(part):
            keys = (("det", np.int8), ("lay", np.int8), ("x", f4), ("y", f4), ("z", f4), ("t", f4), ("de", f4), ("p", f4), ("pid", np.int32), ("prim", np.int32))
            return {k: (np.concatenate([d[k] for d in o[part]]) if o[part] else np.zeros(0, dtype=dt)) for k, dt in keys}

        if o["cal_c"]:
            cal = dict(code=np.concatenate(o["cal_c"]), e=np.concatenate(o["cal_e"]), t=np.concatenate(o["cal_t"]), prim=np.concatenate(o["cal_p"]))
        else:
            cal = dict(code=np.zeros(0, np.int64), e=np.zeros(0, f4), t=np.zeros(0, f4), prim=np.zeros(0, np.int32))
        r = dict(trk=cat("trk"), mu=cat("mu"), cal=cal, n=dict(self.n), e={k: float(v) / (1e3 if k == "dead_region" else 1.0) for k, v in self.e.items()},
                 ncalhits=self.ncalhits, nsec=self.nsec, fate=o["fate"], bnd=o["bnd"], end=o["end"], scan=o["scan"])
        r["tal_prim"] = np.array([x[0] for x in o["tal"]], dtype=np.int64)
        r["tal_kind"] = np.array([x[1] for x in o["tal"]], dtype=np.int8)
        r["tal"] = np.array([x[2] for x in o["tal"]], dtype=float).reshape(-1, len(TALLY))
        if self.record_tracks:
            r["tracks"] = pack_tracks(o["births"], o["pts"], self.cm)
            r["verts"] = [(code_of(v[0]), prim, v[2] / self.cm, v[3] / self.cm, v[4] / self.cm, v[5], v[6] / 1e3, v[7], v[8], v[1]) for prim, vs in o["verts"] for v in vs]
            r["vox"] = np.concatenate(o["vox"]) if o["vox"] else np.zeros((0, 7))
        return r


def pack_tracks(births, pts, cm):
    """Trajectories of one task: a table of tracks and a table of points, thinned where the path is straight."""
    T, Pn = [], []
    base = 0
    for (prim, bs), (_, P) in zip(births, pts):
        tids = np.array([b[0] for b in bs])
        order = np.argsort(tids)
        pos = np.searchsorted(tids[order], P[:, 0])
        ok = (pos < len(tids)) & (tids[order][np.minimum(pos, len(tids) - 1)] == P[:, 0])
        P = P[ok]
        idx = order[pos[ok]]                                  # index of the track in bs
        srt = np.argsort(idx, kind="stable")
        P, idx = P[srt], idx[srt]
        xyz = P[:, 1:4] / cm
        # keep the first and last point of a track and every point where the direction has turned by 2 degrees
        # or the path has grown by 20 cm since the last kept point
        seg = np.diff(xyz, axis=0)
        L = np.linalg.norm(seg, axis=1)
        u = seg / np.maximum(L, 1e-9)[:, None]
        same = idx[1:] == idx[:-1]
        turn = np.zeros(len(xyz))
        if len(u) > 1:
            cosang = np.clip(np.sum(u[1:] * u[:-1], axis=1), -1, 1)
            turn[1:-1] = np.where(same[1:] & same[:-1], np.degrees(np.arccos(cosang)), 0.0)
        path = np.zeros(len(xyz))
        path[1:] = np.where(same, L, 0.0)
        first = np.ones(len(xyz), bool); first[1:] = ~same
        last = np.ones(len(xyz), bool); last[:-1] = ~same
        cum_t, cum_l = np.cumsum(turn), np.cumsum(path)
        st = np.flatnonzero(first)
        grp = np.cumsum(first) - 1
        ct, cl = cum_t - cum_t[st][grp], cum_l - cum_l[st][grp]
        kt, kl = np.floor(ct / 2.0), np.floor(cl / 20.0)
        keep = first | last
        keep[1:] |= (kt[1:] != kt[:-1]) | (kl[1:] != kl[:-1])
        tid_to_idx = {int(t): i for i, t in enumerate(tids)}
        for i, b in enumerate(bs):
            T.append((prim, b[2], base + tid_to_idx.get(int(b[1]), -1 - base) if b[1] > 0 else -1, code_of(b[3]) if b[1] > 0 else 0,
                      b[4] / cm, b[5] / cm, b[6] / cm, b[7], b[8] / 1e3, 1 if b[10] else 0))
        Pn.append(np.column_stack([idx[keep] + base, xyz[keep], P[keep, 4], P[keep, 5] / 1e3]))
        base += len(bs)
    if not T:
        return dict(T=np.zeros((0, 10)), P=np.zeros((0, 6)))
    return dict(T=np.array(T, dtype=float), P=np.concatenate(Pn))


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------
def _init_worker(record_tracks, field, physics, scan, logdir, birks=True, regions=True, stage="full", trk_only=False):
    global _APP
    if logdir:
        fd = os.open(str(Path(logdir) / f"geant4_worker_{os.getpid()}.out"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    else:
        fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(fd, 1)                                            # Geant4 talks a lot on stdout
    _APP = App(record_tracks=record_tracks, field=field, physics=physics, scan=scan, birks=birks, regions=regions, stage=stage, trk_only=trk_only)


def _task(task):
    key, todo, prims, seed = task
    t0 = time.time()
    r = _APP.run(todo, prims, seed)
    r["key"], r["seconds"], r["version"], r["physics"], r["sd"] = key, time.time() - t0, _APP.version, _APP.physics, _APP.sd_counts
    r["regions"] = _APP.region_roots
    return r


def event_seed(seed, k, j):
    return (seed * 1000003 + k * 10007 + j * 7919 + 12345) % 2147483647


def rows_of(c, sel, t0, seed, k):
    return [(int(c.ids[i]), float(c.p4[i, 1]), float(c.p4[i, 2]), float(c.p4[i, 3]), float(c.vtx[i, 0]), float(c.vtx[i, 1]), float(c.vtx[i, 2]),
             float(t0), event_seed(seed, k, j)) for j, i in enumerate(sel)]


def pool(args, record_tracks=False, scan=False):
    ctx = mp.get_context("spawn")
    return ctx.Pool(args.jobs, initializer=_init_worker,
                    initargs=(record_tracks, not args.no_field, args.physics, scan, args.worker_logs, not getattr(args, "no_birks", False), not getattr(args, "no_regions", False), getattr(args, "stage", "full"), getattr(args, "trk_only", False)))


def budget(kind, tal):
    """The energy budget per group of primaries: {group: {column of TALLY: sum in GeV}}."""
    out = {}
    for k, name in enumerate(sim.KINDS):
        s = tal[kind == k].sum(axis=0) if len(tal) else np.zeros(len(TALLY))
        out[name] = {c: float(v) for c, v in zip(TALLY, s)}
    out["all"] = {c: float(v) for c, v in zip(TALLY, tal.sum(axis=0) if len(tal) else np.zeros(len(TALLY)))}
    return out


def by_kind(kind, tal):
    return np.array([tal[kind == k].sum(axis=0) for k in range(len(sim.KINDS))])


# ---------------------------------------------------------------------------
# the signal
# ---------------------------------------------------------------------------
def merge_cal(parts, per_prim=True):
    c = np.concatenate([p["cal"]["code"] for p in parts]); e = np.concatenate([p["cal"]["e"] for p in parts]).astype(float)
    t = np.concatenate([p["cal"]["t"] for p in parts]).astype(float); pr = np.concatenate([p["cal"]["prim"] for p in parts])
    if len(c) == 0:
        return dict(code=c, e=e.astype(np.float32), t=t.astype(np.float32), prim=pr)
    order = np.lexsort((pr, c)) if per_prim else np.argsort(c, kind="stable")
    c, e, t, pr = c[order], e[order], t[order], pr[order]
    new = np.ones(len(c), bool)
    new[1:] = (c[1:] != c[:-1]) | ((pr[1:] != pr[:-1]) if per_prim else False)
    st = np.flatnonzero(new)
    es = np.add.reduceat(e, st)
    ts = np.add.reduceat(e * t, st) / np.maximum(es, 1e-30)
    return dict(code=c[st], e=es.astype(np.float32), t=ts.astype(np.float32), prim=pr[st].astype(np.int32))


def merge_hits(parts, part):
    keys = parts[0][part].keys()
    d = {k: np.concatenate([p[part][k] for p in parts]) for k in keys}
    order = np.argsort(d["prim"], kind="stable")
    return {k: v[order] for k, v in d.items()}


def run_signal(args, st, say):
    c, g = st.crowd, st.g
    sel = np.flatnonzero(g["primary"] & (c.block == 0))
    t0 = st.collisions[0]["t"]
    rows = rows_of(c, sel, t0, args.seed, 0)
    idx = list(range(len(sel)))
    if args.only is not None:
        idx = [j for j in idx if int(c.row[sel[j]]) in args.only or j in args.only]
    # heavy primaries first, a handful per task, so that the processes finish together
    idx.sort(key=lambda j: -c.p4[sel[j], 0])
    nt = max(1, min(len(idx), args.jobs * 6))
    tasks = [(("signal", b), [rows[j] for j in idx[b::nt]], idx[b::nt], [args.seed, 7, b]) for b in range(nt)]
    t1 = time.time()
    with pool(args, record_tracks=True) as p:
        parts = list(p.imap_unordered(_task, tasks))
    parts.sort(key=lambda r: r["key"][1])
    dt = time.time() - t1
    r = dict(sel=sel, trk=merge_hits(parts, "trk"), mu=merge_hits(parts, "mu"), cal=merge_cal(parts), seconds=dt,
             cpu=sum(p["seconds"] for p in parts), version=parts[0]["version"], physics=parts[0]["physics"], sd=parts[0]["sd"])
    n, e = Counter(), Counter()
    fate, bnd = {}, {}
    for p in parts:
        n.update(p["n"]); e.update(p["e"]); fate.update(p["fate"]); bnd.update(p["bnd"])
    r.update(n=dict(n), e=dict(e), fate=fate, bnd=bnd, ncalhits=sum(p["ncalhits"] for p in parts), nsec=sum(p["nsec"] for p in parts))
    tp = np.concatenate([p["tal_prim"] for p in parts]); order = np.argsort(tp)
    r["tal_prim"] = tp[order]
    r["tal_kind"] = np.concatenate([p["tal_kind"] for p in parts])[order]
    r["tal"] = np.concatenate([p["tal"] for p in parts])[order]
    r["regions"] = parts[0]["regions"]
    # trajectories: one table
    T, P, base = [], [], 0
    for p in parts:
        t = p["tracks"]
        if len(t["T"]) == 0:
            continue
        tt = t["T"].copy()
        par = tt[:, 2] >= 0
        tt[par, 2] += base
        pp = t["P"].copy()
        pp[:, 0] += base
        T.append(tt); P.append(pp)
        base += len(tt)
    r["T"] = np.concatenate(T) if T else np.zeros((0, 10))
    r["P"] = np.concatenate(P) if P else np.zeros((0, 6))
    r["verts"] = [v for p in parts for v in p["verts"]]
    r["vox"] = np.concatenate([p["vox"] for p in parts])
    return r


def save_signal(args, st, r):
    c = st.crowd
    sel = r["sel"]
    d = {}
    for part in ("trk", "mu"):
        for key, v in r[part].items():
            d[f"{part}_{key}"] = v
    for key, v in r["cal"].items():
        d[f"cal_{key}"] = v
    d["prim_id"] = c.ids[sel].astype(np.int32)
    d["prim_p4"] = c.p4[sel]
    d["prim_vtx"] = c.vtx[sel]
    d["prim_row"] = c.row[sel].astype(np.int32)
    d["prim_kind"] = np.array([sim.kind_of(int(i)) for i in c.ids[sel]], np.int8)
    d["prim_jet0"] = np.isin(np.arange(len(sel)), sim.jet_members(st, sel))
    d["prim_fate"] = np.array([FATES.index(r["fate"].get(j, "stopped")) for j in range(len(sel))], np.int8)
    d["fate_names"] = np.array(FATES)
    b = np.full((len(sel), 6), np.nan)
    for j, v in r["bnd"].items():
        b[j] = v
    d["prim_bnd"] = b
    tal = np.zeros((len(sel), len(TALLY)))
    tal[r["tal_prim"]] = r["tal"]
    d["prim_tal"] = tal.astype(np.float32)
    d["tal_names"] = np.array(TALLY)
    T = r["T"]
    stored = (T[:, 2] >= 0) & (T[:, 9] == 0) & (T[:, 8] >= E_SEC) if len(T) else np.zeros(0, bool)
    par = T[stored, 2].astype(int)
    d["sec"] = np.column_stack([np.minimum(T[stored, 3], 4), T[stored, 1], T[stored, 0], T[stored, 4], T[stored, 5], T[stored, 6],
                                T[stored, 8], T[par, 1]]) if stored.any() else np.zeros((0, 8))
    V = np.array([v[:8] for v in r["verts"]], dtype=float).reshape(-1, 8)
    d["vtx"] = V
    np.savez_compressed(args.outdir / "simhits_signal.npz", **d)
    Vfull = np.array(r["verts"], dtype=float).reshape(-1, 10)
    np.savez_compressed(args.outdir / "g4tracks_signal.npz", T=T.astype(np.float32), P_trk=r["P"][:, 0].astype(np.int32), P=r["P"][:, 1:].astype(np.float32),
                        V=Vfull.astype(np.float32), vox=r["vox"].astype(np.float32), proc_names=np.array([PROC_OF_CODE[i] for i in range(len(PROC_OF_CODE))]),
                        T_columns=np.array(["prim", "pid", "parent", "proc", "x", "y", "z", "t", "ekin", "outer"]),
                        V_columns=np.array(["proc", "prim", "x", "y", "z", "pid", "ekin", "nsec", "t", "g4track"]),
                        vox_columns=np.array(["det", "x", "y", "z", "e", "t", "prim"]))


def summary_signal(args, st, r):
    sel = r["sel"]
    t = r["trk"]
    n = Counter(r["n"])
    jm = sim.jet_members(st, sel)
    cal = r["cal"]
    det = cal["code"] // sim.CODE
    bnd = r["bnd"]
    jb = [j for j in jm if j in bnd]
    in_jet = np.isin(cal["prim"], jm)
    row = dict(stage=STAGE_LABEL[args.stage], primaries=int(len(sel)), boundary=len(bnd), secondaries=int(r["nsec"]),
               sec_decay=n["sec_decay"], sec_em=n["sec_conversion"] + n["sec_bremsstrahlung"] + n["sec_delta ray"] + n["sec_Compton scattering"] + n["sec_photoelectric effect"],
               sec_had=n["sec_nuclear interaction"], trk=int(len(t["det"])), mu=int(len(r["mu"]["det"])), de=float(t["de"].sum()),
               cells=int(len(np.unique(cal["code"]))), calhits=int(r["ncalhits"]), jet_n=len(jm), jet_bnd_n=len(jb),
               jet_bnd_pt=sim.vec_pt([bnd[j][3] for j in jb], [bnd[j][4] for j in jb]) if jb else 0.0,
               jet_ecal=float(cal["e"][in_jet & np.isin(det, sim.ECAL)].sum()), jet_hcal=float(cal["e"][in_jet & np.isin(det, (sim.HB, sim.HE, sim.HO))].sum()))
    for d in sim.ECAL + sim.HCAL:
        row[sim.DET_NAME[d]] = float(cal["e"][det == d].sum())
    words = Counter()
    for k, v in n.items():
        if not k.startswith(("sec_", "out_")) and k not in ("time_cut", "dead_region", "hf_in", "unknown_particle"):
            words[word(k)] += v
    nn = dict(n)
    nn.update(decays=words["decay"], conversions=words["conversion"], brems=words["bremsstrahlung"], nuclear=words["nuclear interaction"],
              delta_rays=words["delta ray"], compton=words["Compton scattering"], photoelectric=words["photoelectric effect"],
              annihilations=words["annihilation"], elastic=words["elastic scattering"])
    return dict(sample="signal", engine="Geant4", version=r["version"], physics=r["physics"], seed=args.seed, field=B_INNER, stages=[row], n=nn, e=r["e"],
                stations=sim.station_summary(t["det"], t["lay"], t["de"], r["mu"]["det"], cal["code"], cal["e"]),
                primaries=int(len(sel)), simtracks=int(len(sel) + r["nsec"]), calhits=int(r["ncalhits"]), cells=sim.n_cells(),
                seconds=r["seconds"], cpu_seconds=r["cpu"], jobs=args.jobs, sensitive_volume_types=r["sd"],
                budget=budget(r["tal_kind"], r["tal"]), jet_budget=budget(r["tal_kind"][np.isin(r["tal_prim"], jm)], r["tal"][np.isin(r["tal_prim"], jm)]),
                regions=r["regions"], birks=not args.no_birks, geometry=GDML.name,
                trajectories=int(len(r["T"])), trajectory_points=int(len(r["P"])), vertices=len(r["verts"]), shower_voxels=int(len(r["vox"])))


# ---------------------------------------------------------------------------
# the crowd
# ---------------------------------------------------------------------------
def run_crowd(args, st, which, say):
    c, g = st.crowd, st.g
    block = 1 if which == "intime" else 2
    cols = {col["k"]: col for col in st.collisions if col["k"] > 0 and ((col["bx"] == 0) == (which == "intime"))}
    idx = np.flatnonzero(g["primary"] & (c.block == block))
    ks = c.k[idx]
    keys = sorted(cols)
    if args.max_collisions:
        keys = keys[:args.max_collisions]
    tasks = []
    for k in keys:
        sel = idx[ks == k]
        t0 = cols[k]["t"] - 25.0 * cols[k]["bx"]
        tasks.append(((which, k), rows_of(c, sel, t0, args.seed, k), list(range(len(sel))), [args.seed, 1000 + k]))
    tasks.sort(key=lambda t: -sum(abs(r[3]) for r in t[1]))
    t1 = time.time()
    out = []
    with pool(args) as p:
        for i, r in enumerate(p.imap_unordered(_task, tasks, chunksize=1)):
            k = r["key"][1]
            r["cal"] = merge_cal([r], per_prim=False)
            r["k"], r["bx"], r["nprim"] = k, cols[k]["bx"], len(r["trk"]["det"]) * 0 + int((ks == k).sum())
            r["tal"] = by_kind(r["tal_kind"], r["tal"])
            out.append(r)
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                say(f"   {which}: {i + 1}/{len(tasks)} collisions, {time.time() - t1:.0f} s")
    out.sort(key=lambda r: r["k"])
    return out, time.time() - t1


def pack_crowd(out):
    d = {}
    for part in ("trk", "mu"):
        for key in ("det", "lay", "x", "y", "z", "t", "de", "p", "pid"):
            d[f"{part}_{key}"] = np.concatenate([r[part][key] for r in out])
        d[f"{part}_k"] = np.concatenate([np.full(len(r[part]["det"]), r["k"], np.int16) for r in out])
        d[f"{part}_bx"] = np.concatenate([np.full(len(r[part]["det"]), r["bx"], np.int8) for r in out])
    for key in ("code", "e", "t"):
        d[f"cal_{key}"] = np.concatenate([r["cal"][key] for r in out])
    d["cal_k"] = np.concatenate([np.full(len(r["cal"]["code"]), r["k"], np.int16) for r in out])
    d["cal_bx"] = np.concatenate([np.full(len(r["cal"]["code"]), r["bx"], np.int8) for r in out])
    d["col_k"] = np.array([r["k"] for r in out], np.int16)
    d["col_bx"] = np.array([r["bx"] for r in out], np.int8)
    d["col_nprim"] = np.array([r["nprim"] for r in out], np.int32)
    d["col_nsec"] = np.array([r["nsec"] for r in out], np.int32)
    d["col_ncalhits"] = np.array([r["ncalhits"] for r in out], np.int32)
    d["col_seconds"] = np.array([r["seconds"] for r in out], np.float32)
    d["col_tal"] = np.array([r["tal"] for r in out], np.float32)          # collision x group x column of TALLY, GeV
    d["tal_names"] = np.array(TALLY)
    d["kind_names"] = np.array(sim.KINDS)
    return d


# ---------------------------------------------------------------------------
# the material scan with geantinos
# ---------------------------------------------------------------------------
def run_scan(args, rows, prims):
    """Geantinos along straight lines, up to the tracker / calorimeter boundary: the material on the way and the
    silicon sensors crossed. rows as for the gun (the particle type is ignored); returns one (label, x/X0,
    x/lambda_I, {detector: sensors}) per row."""
    nt = max(1, min(len(rows), args.jobs * 2))
    tasks = [(("scan", b), rows[b::nt], prims[b::nt], [1]) for b in range(nt)]
    with pool(args, scan=True) as p:
        parts = list(p.imap_unordered(_task, tasks))
    return [x for p in parts for x in p["scan"]]


def material_scan(args, etas, nphi=72):
    """Mean over azimuth of the material seen from the origin, per pseudorapidity."""
    phis = (np.arange(nphi) + 0.5) * 2 * math.pi / nphi
    rows, prims = [], []
    for ie, eta in enumerate(etas):
        for phi in phis:
            rows.append((0, math.cos(phi), math.sin(phi), math.sinh(eta), 0.0, 0.0, 0.0, 0.0, 1))
            prims.append(ie)
    x0 = [[] for _ in etas]
    li = [[] for _ in etas]
    for prim, a, b, _ in run_scan(args, rows, prims):
        x0[prim].append(a); li[prim].append(b)
    return np.array([np.mean(v) for v in x0]), np.array([np.std(v) for v in x0]), np.array([np.mean(v) for v in li])


def run_gun(args, pid, energy, eta, n, seed=1, flat_eta=None):
    """n particles of one kind and energy from the origin at pseudorapidity eta (azimuth random; flat_eta: flat in
    |eta| < flat_eta instead): for each the energy in the crystals, in the scintillator, the number of tracker
    hits, what ended it (fate) and where (end)."""
    rng = np.random.default_rng([seed, abs(pid), int(energy * 1000)])
    m = sim.dec.m0(pid)
    p = math.sqrt(max(energy * energy - m * m, 0.0))
    rows = []
    etas = np.full(n, float(eta)) if flat_eta is None else rng.uniform(-flat_eta, flat_eta, n)
    for j in range(n):
        phi = 2 * math.pi * rng.random()
        eta = float(etas[j])
        pt = p / math.cosh(eta)
        rows.append((pid, pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta), 0.0, 0.0, 0.0, 0.0, event_seed(seed, 90000 + abs(pid), j)))
    nt = max(1, min(n, args.jobs * 3))
    tasks = [(("gun", b), rows[b::nt], list(range(n))[b::nt], [1]) for b in range(nt)]
    with pool(args) as p_:
        parts = list(p_.imap_unordered(_task, tasks))
    ecal, hcal, trk = np.zeros(n), np.zeros(n), np.zeros(n)
    fate, cells, end = {}, {}, {}
    for r in parts:
        fate.update(r["fate"])
        end.update(r["end"])
        eb = (r["cal"]["code"] // sim.CODE) == sim.EB
        for j in np.unique(r["cal"]["prim"][eb]):
            cells[int(j)] = np.sort(r["cal"]["e"][eb & (r["cal"]["prim"] == j)])[::-1]
        det = r["cal"]["code"] // sim.CODE
        np.add.at(ecal, r["cal"]["prim"][np.isin(det, (sim.EB, sim.EE))], r["cal"]["e"][np.isin(det, (sim.EB, sim.EE))])
        np.add.at(hcal, r["cal"]["prim"][np.isin(det, (sim.HB, sim.HE, sim.HO))], r["cal"]["e"][np.isin(det, (sim.HB, sim.HE, sim.HO))])
        np.add.at(trk, r["trk"]["prim"], 1)
    return dict(ecal=ecal, hcal=hcal, trk=trk, fate=[fate.get(j, "stopped") for j in range(n)], eb_cells=[cells.get(j, np.zeros(0)) for j in range(n)],
                eta=etas, end=[end.get(j) for j in range(n)])


# ---------------------------------------------------------------------------
# narration
# ---------------------------------------------------------------------------
def narrate(st, r, summ, say):
    row = summ["stages"][0]
    n = Counter(summ["n"])
    sel = r["sel"]
    c = st.crowd
    say("=" * 100)
    say(f"Geant4 in the CMS detector: the family of the b bbar event, {len(sel)} primaries ({c.p4[sel, 0].sum():.0f} GeV)")
    say("=" * 100)
    say(f"1. The toolkit: {summ['version']}, physics list {summ['physics']}; {summ['jobs']} processes, {summ['seconds']:.0f} s "
        f"({summ['cpu_seconds']:.0f} s of processor time)")
    sd = summ["sensitive_volume_types"]
    say(f"2. The detector: {GDML.name}, 3398 volume types, 2.1 million placed volumes. Sensitive volume types:")
    say("   " + ", ".join(f"{k} {v}" for k, v in sorted(sd.items())))
    say("   (trk1..6 = PXB PXF TIB TID TOB TEC, mu11..14 = DT CSC RPC GEM, in / out = the other volumes of the tracker and of")
    say("   the muon system, where only the trajectory is recorded)")
    say(f"3. The field: uniform {B_INNER} T in the tracker and the calorimeters, the map of field_map_SIM.py elsewhere")
    reg = summ.get("regions", {})
    say("   The settings of CMSSW: production cuts by region [mm] (number of root volumes): " + ", ".join(f"{nm} {cut:g} ({reg.get(nm, 0)})" for nm, cut, _ in REGIONS))
    say(f"   elsewhere {CUT_WORLD:g} mm; Birks' law " + (f"on (scintillator: kB = {BIRK_HCAL[0]} g/MeV/cm2, C = {BIRK_HCAL[1]}, kB / {BIRK_HCAL[2]} for charge >= 2; "
        f"crystals: weight = 1 - {BIRK_ECAL[1]} ln(kB dE/dx), kB = {BIRK_ECAL[0]} g/MeV/cm2, at least {BIRK_ECAL[2]})" if summ.get("birks") else "off") + f"; particles stopped {T_MAX:.0f} ns after the bunch crossing")
    say("4. What happened in the tracker volume (Geant4 process: number of steps that ended with it and made a secondary or stopped the particle)")
    procs = sorted(((k, v) for k, v in summ["n"].items() if not k.startswith(("sec_", "out_")) and k[0].islower() or k in ("Decay",)), key=lambda kv: -kv[1])
    for k, v in procs:
        if k in ("decays", "conversions", "brems", "nuclear", "delta_rays", "compton", "photoelectric", "annihilations", "elastic", "time_cut", "dead_region", "hf_in", "msc", "none"):
            continue
        say(f"   {k:28s} {v:8d}   {word(k)}")
    say(f"   in words: {n['decays']} decays, {n['conversions']} conversions, {n['brems']} bremsstrahlung photons, {n['nuclear']} nuclear interactions,")
    say(f"   {n['delta_rays']} delta rays, {n['compton']} Compton scatterings, {n['photoelectric']} photoelectric absorptions, {n['annihilations']} annihilations")
    outs = {k[4:]: v for k, v in summ["n"].items() if k.startswith("out_")}
    say("   behind the calorimeters (charged particles in the muon system): " + ", ".join(f"{k} {v}" for k, v in sorted(outs.items(), key=lambda kv: -kv[1])[:8]))
    say(f"5. The primaries: {row['boundary']} of {row['primaries']} leave the tracker volume towards the calorimeters; the fates of the others:")
    fates = Counter(r["fate"].values())
    say("   " + ", ".join(f"{k} {v}" for k, v in fates.most_common()))
    say(f"   secondaries born in the tracker above {E_SEC * 1e3:.0f} MeV: {row['secondaries']} ({row['sec_decay']} from decays, {row['sec_em']} electromagnetic, {row['sec_had']} from nuclear interactions)")
    say(f"6. The hits: {row['trk']} tracker hits ({row['de'] * 1e3:.1f} MeV left in the silicon), {row['mu']} muon-chamber hits")
    for d, v in summ["stations"]["tracker"].items():
        say(f"   {d:4s} {v['hits']:6d} hits  {v['de'] * 1e3:8.2f} MeV   layers " + " ".join(f"{k}:{w}" for k, w in sorted(v['layers'].items())))
    say("   " + ", ".join(f"{d} {v}" for d, v in summ["stations"]["muon"].items()))
    say(f"7. The calorimeters: {row['calhits']} energy deposits (steps) in {row['cells']} cells")
    for d, v in summ["stations"]["calo"].items():
        unit = "photoelectrons" if d == "HF" else "GeV"
        say(f"   {d:3s} {v['cells']:6d} cells  {v['e']:10.3f} {unit}")
    e = summ["e"]
    say(f"   forward calorimeter: {n['hf_in']} particles with {e.get('hf_in', 0):.0f} GeV stopped at its steel and parametrised; "
        f"{n['dead_region']} particles with {e.get('dead_region', 0):.0f} GeV of kinetic energy stopped in the dead region behind it")
    say(f"8. Our jet [0]: {row['jet_n']} members, {row['jet_bnd_n']} reach the calorimeters as primaries (vector pT {row['jet_bnd_pt']:.2f} GeV); "
        f"its descendants leave {row['jet_ecal']:.1f} GeV in the crystals and {row['jet_hcal']:.3f} GeV in the scintillator")
    say(f"9. For the displays: {summ['trajectories']} trajectories with {summ['trajectory_points']} points, {summ['vertices']} vertices, {summ['shower_voxels']} shower voxels")
    say("10. Who leaves how much where, in GeV (all primaries of a group; the light is the deposit after Birks' law):")
    say(f"   {'group':15s} {'number':>6s} {'energy':>9s} {'tracker vol.':>12s} {'silicon':>8s} {'crystals':>9s} {'light':>9s} {'scint.':>8s} {'light':>8s} {'at the HF':>10s} {'behind':>8s} {'muon sys.':>9s}")
    for k, b in summ["budget"].items():
        say(f"   {k:15s} {b['n']:6.0f} {b['e_gate']:9.1f} {b['trkvol']:12.2f} {b['si']:8.4f} {b['eb'] + b['ee']:9.2f} {b['eb_vis'] + b['ee_vis']:9.2f} {b['hb'] + b['he'] + b['ho']:8.4f} "
            f"{b['hb_vis'] + b['he_vis'] + b['ho_vis']:8.4f} {b['hf_in']:10.1f} {b['dead']:8.1f} {b['muvol']:9.3f}")


def to_jsonable(x):
    return sim.to_jsonable(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", choices=["signal", "intime", "oot"], default="signal")
    ap.add_argument("-o", "--outdir", type=Path, default=HERE.parent)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=max(1, min(10, (mp.cpu_count() or 2) - 2)))
    ap.add_argument("--physics", default="FTFP_BERT", help="a Geant4 reference physics list (FTFP_BERT, QGSP_BERT, FTFP_BERT_EMZ ...)")
    ap.add_argument("--no-field", action="store_true", help="exercise: switch the magnet off")
    ap.add_argument("--stage", choices=list(STAGES), default="full",
                    help="signal: the physics switched on step by step; anything but full writes only geant4_SIM_signal_<stage>.json")
    ap.add_argument("--no-birks", action="store_true", help="exercise: every MeV in the scintillator and the crystals gives the same light")
    ap.add_argument("--no-regions", action="store_true", help="exercise: Geant4's default production cut of 0.7 mm everywhere instead of CMSSW's regions")
    ap.add_argument("--only", type=int, nargs="+", help="signal: only these primaries (index in the list of primaries)")
    ap.add_argument("--max-collisions", type=int, default=0, help="crowd: only the first N collisions (for timing)")
    ap.add_argument("--scan", action="store_true", help="geantino scan of the material; writes geant4_SIM_scan.json")
    ap.add_argument("--tag", default="", help="suffix of the output files (so that a test does not overwrite the record)")
    ap.add_argument("--worker-logs", type=Path, help="keep what Geant4 prints, one file per process, in this folder")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    ap.add_argument("--log", type=Path)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.stage != "full":
        if args.sample != "signal":
            sys.exit("--stage is for the signal")
        args.tag = args.tag or "_" + args.stage
    if not GDML.exists():
        sys.exit(f"{GDML} is missing: make gdml (export_gdml_SIM.py, needs ROOT)")
    log = open(args.log, "w") if args.log else None

    def say(text=""):
        if args.verbose:
            print(text, file=log or sys.stdout, flush=True)

    def note(text):
        if not args.quiet:
            print(text, flush=True)

    if args.scan:
        etas = np.round(np.arange(-3.0, 3.0001, 0.1), 2)
        x0, rms, li = material_scan(args, etas)
        (HERE / "geant4_SIM_scan.json").write_text(json.dumps(dict(eta=etas.tolist(), x0=x0.tolist(), x0_rms=rms.tolist(), lambda_i=li.tolist()), indent=1))
        note("geantino scan up to the calorimeters: x/X0 at eta 0, 1, 1.5, 2, 2.5 = " + ", ".join(f"{x0[np.argmin(abs(etas - e))]:.2f}" for e in (0, 1, 1.5, 2, 2.5)))
        return
    st = sim.load_stage()
    t0 = time.time()
    if args.sample == "signal":
        r = run_signal(args, st, note)
        summ = summary_signal(args, st, r)
        if args.only is None and not args.tag:
            save_signal(args, st, r)
            (HERE / "geant4_SIM_signal.json").write_text(json.dumps(to_jsonable(summ), indent=1))
        elif args.tag:
            (HERE / f"geant4_SIM_signal{args.tag}.json").write_text(json.dumps(to_jsonable(summ), indent=1))
        narrate(st, r, summ, say)
        w = summ["stages"][0]
        note(f"geant4_SIM signal: {w['primaries']} primaries -> {w['boundary']} at the boundary, {w['secondaries']} secondaries, {w['trk']} tracker hits, "
             f"{w['cells']} cells (EB {w['EB']:.1f} EE {w['EE']:.1f} GeV, HB {w['HB']:.2f} HE {w['HE']:.2f} GeV, HF {w['HF']:.0f} pe), {w['mu']} muon hits; "
             f"{summ['seconds']:.0f} s on {args.jobs} processes")
    else:
        which = args.sample
        out, dt = run_crowd(args, st, which, note)
        d = pack_crowd(out)
        np.savez_compressed(args.outdir / f"simhits_{which}{args.tag}.npz", **d)
        n, e = Counter(), Counter()
        for r in out:
            n.update(r["n"]); e.update(r["e"])
        words = Counter()
        for k, v in n.items():
            if not k.startswith(("sec_", "out_")) and k not in ("time_cut", "dead_region", "hf_in", "unknown_particle"):
                words[word(k)] += v
        nn = dict(n)
        nn.update(decays=words["decay"], conversions=words["conversion"], brems=words["bremsstrahlung"], nuclear=words["nuclear interaction"], delta_rays=words["delta ray"])
        summ = dict(sample=which, engine="Geant4", version=out[0]["version"], physics=out[0]["physics"], seed=args.seed, field=B_INNER, collisions=len(out),
                    primaries=int(d["col_nprim"].sum()), simtracks=int(d["col_nprim"].sum() + d["col_nsec"].sum()), calhits=int(d["col_ncalhits"].sum()),
                    n=nn, e=dict(e), stations=sim.station_summary(d["trk_det"], d["trk_lay"], d["trk_de"], d["mu_det"], d["cal_code"], d["cal_e"]),
                    seconds=dt, cpu_seconds=float(d["col_seconds"].sum()), jobs=args.jobs, birks=not args.no_birks, geometry=GDML.name,
                    budget={**{name: {c: float(v) for c, v in zip(TALLY, d["col_tal"][:, k].sum(axis=0))} for k, name in enumerate(sim.KINDS)},
                            "all": {c: float(v) for c, v in zip(TALLY, d["col_tal"].sum(axis=(0, 1)))}},
                    regions=out[0]["regions"], fates=dict(Counter(f for r in out for f in r["fate"].values())),
                    sensitive_volume_types=out[0]["sd"])
        (HERE / f"geant4_SIM_{which}{args.tag}.json").write_text(json.dumps(to_jsonable(summ), indent=1))
        note(f"geant4_SIM {which}: {len(out)} collisions, {summ['primaries']} primaries -> {summ['simtracks']} tracks, {len(d['trk_det'])} tracker hits, "
             f"{len(d['mu_det'])} muon hits, {len(d['cal_code'])} (cell, collision) deposits; {dt:.0f} s on {args.jobs} processes")
    if log:
        log.close()
    note(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
