#!/usr/bin/env python3
"""Standalone detector simulation for the b bbar event of chapter 9: the crossing of the border.

Dependencies: numpy, plus chapter 9's ../../G4/Standalone/standalone_G4.py (the primaries handed to Geant4;
chapters 8, 7, 6 and 5 are imported through it). No CMSSW, no Geant4. Geant4 follows every particle step by
step through a million volumes; this script follows the same primaries through a detector written on one
page, with the physics of each step written as the formula of the textbook. It is a simulation in the
spirit of Geant4 and with the granularity of CMS, not a port of either: where Geant4 tracks the thousands
of particles of a shower, a parametrisation places the energy of the shower in the cells.

Input is the record of chapter 9 (../../G4/event_G4.lhe is the same event; the primaries are rebuilt from
../../PU/event_PU.lhe and the library ../../PU/pileup_library.lhe.gz exactly as chapter 9 did). Three samples are
simulated separately, each collision with its own random stream:

  --sample signal    the family: 672 primaries of one collision (a few seconds; this is the learning mode)
  --sample intime    the in-time crowd: the 63 collisions of bunch crossing 0
  --sample oot       the out-of-time crowd: the 897 collisions of the fifteen other crossings, each simulated
                     at its own bunch crossing as if it were alone; its hit times are relative to that crossing
  --sample all       the three, one after the other (default)

Each writes ../simhits_<sample>.npz (tracker and muon hits, calorimeter deposits per cell) and
<sample>.json next to this script. Nothing is added here: add_hits_SIM.py reads the three files, shifts the
out-of-time hits by 25 ns x bx and adds them cell by cell, which is all the MixingModule of CMSSW does.

What is implemented (the physics of chapter 10), each switchable so that the signal can be run stage by stage
(--stage transport | B | B_Decay | B_Decay_EM | full):

* The detector: a beam pipe, the 4 + 4 + 6 barrel layers and the 3 + 3 + 9 pairs of disks of the pixel and
  strip tracker with the radiation and interaction lengths a particle crosses in each (sensor, support,
  cables, cooling), the dead material of the services, the crystals of the ECAL barrel (61 200, front face at
  r = 129 cm) and endcaps (a 2.862 cm grid at |z| = 320 cm), the preshower, the brass and scintillator towers
  of the HCAL barrel and endcap with their depths, the outer HCAL behind the coil, the forward calorimeter HF
  at |z| = 11.15 m, and the muon stations behind the calorimeters and between the iron of the yoke.
* The magnetic field: a uniform 3.8 T along the beam inside the coil. A charged particle moves on a helix of
  radius R = pT / (0.3 B) from surface to surface; below pT = 0.3 B r / 2 it cannot reach radius r (a looper).
* Decays in flight: every pi+-, K+-, K0L, K0S, Lambda, Sigma, Xi, Omega is given a flight path beta gamma c tau
  x (-ln u) when it is born; if it is reached before the particle interacts or leaves the tracker the particle
  decays there, by the branching ratios of chapter 6's decay table and flat phase space.
* Electromagnetic physics: the Bethe-Bloch energy loss in the material of every layer and its Landau
  fluctuation in the silicon of the sensor (the dE of a tracker hit); multiple scattering by the Highland
  formula; bremsstrahlung of electrons by the Bethe-Heitler distribution; conversion of photons with the
  probability 1 - exp(-7 t / 9) per t radiation lengths; the electromagnetic shower in the crystals with the
  longitudinal profile of Grindhammer and Peters and a two-component lateral profile in Moliere radii.
* Hadronic physics: an inelastic nuclear interaction with probability 1 - exp(-x / lambda_I) per layer of the
  tracker (a toy model of its secondaries: pions, protons, neutrons) and in the calorimeters, where the
  hadronic shower is the sum of an electromagnetic part (fraction 1 - (E/E0)^(k-1), fluctuating) and a
  hadronic part with a longitudinal profile in interaction lengths; the hadronic part is seen with h/e < 1
  (the rest is binding energy, slow neutrons, neutrinos: invisible), and the HCAL records only the energy in
  its scintillator (a sampling fraction of about 1 %). The HF answers in photoelectrons.
* The hits: a tracker or muon hit per sensor crossed (position, time, energy loss, the primary it descends
  from), a calorimeter hit per cell and primary (energy, energy-weighted time). Times are since the bunch
  crossing of the collision.

What is left out, on purpose: delta rays as particles, Compton scattering and the photoelectric effect (all
photons here are above 40 MeV), the real field map (the return field in the yoke), the real shapes of the
modules (a layer is a cylinder or a disk), Birks' law, the shower-to-shower detail that only a full cascade
gives, the Roman pots and the ZDC. Each is discussed in the chapter.

Learning mode (--verbose / -v, -vv) for --sample signal
-----------------------------------------------------
  -v   narrates the detector, then the four ingredients one by one with the numbers of our event (the field
       and the loopers, the decays in flight, the conversions, the bremsstrahlung and the energy loss, the
       nuclear interactions and the showers), the stage-by-stage table, the stations and our jet.
  -vv  adds one line per decay, conversion and nuclear interaction.

Examples
--------
  python3 standalone_SIM.py --sample signal -v          # the family, learning mode
  python3 standalone_SIM.py --sample signal --stage B   # the field alone
  python3 standalone_SIM.py --jobs 8                    # the three samples (a few minutes)
  python3 standalone_SIM.py --sample signal --field 2   # exercise: a 2 T solenoid

All energies and momenta in GeV, lengths in cm, times in ns.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "G4" / "Standalone"))
import standalone_G4 as g4  # noqa: E402

dec = g4.dec

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
C = 29.9792458                    # cm/ns
B_FIELD = 3.8                     # tesla
ME = 0.000510999                  # GeV
MPI = 0.13957
MP = 0.93827
MN = 0.93957

# the tracker volume: where CMSSW's CaloTrkProcessing sees a particle pass from the tracker to the calorimeters
R_TRK, Z_TRK = 123.3, 293.5
T_MAX = 500.0                     # ns, SteppingAction.MaxTrackTime
T_CUT = 0.010                     # GeV: a charged particle below this kinetic energy stops where it is
E_SEC = 0.010                     # GeV: a bremsstrahlung photon or a nuclear secondary softer than this is not created
X0_SI_G = 21.82                   # g/cm2, radiation length of silicon
RHO_SI = 2.329                    # g/cm3
LAMBDA_PER_X0 = 0.28              # tracker: interaction lengths per radiation length of the average material (TGeo scan)
F_CAP = 40.0                      # cap of 1 / cos(incidence) for a grazing crossing

# calorimeters
R_EB, R_EB_BACK, ETA_EB = 129.0, 152.0, 1.479
Z_EB = R_EB * math.sinh(ETA_EB)                 # 268.4 cm
Z_EE, Z_EE_BACK, ETA_EE = 320.0, 342.0, 3.0
X_EE = 2.862                                    # cm, endcap crystal at the front face
NX_EE = 120
R_EE_MIN, R_EE_MAX = Z_EE / math.sinh(ETA_EE), Z_EE / math.sinh(ETA_EB)
Z_ES1, Z_ES2, ETA_ES = 303.2, 307.7, (1.653, 2.6)
X0_PBWO4, RM_PBWO4, EC_PBWO4, LAMBDA_PBWO4, Z_PBWO4 = 0.89, 2.19, 0.0093, 20.3, 68.4
MIP_PBWO4 = 0.0102                              # GeV/cm
D_ETA_EB = 0.0174
R_HB, R_HB_BACK, ETA_HB = 177.7, 287.6, 1.392
Z_HE, Z_HE_BACK = 388.0, 570.0
LAMBDA_HCAL, X0_HCAL, RM_HCAL = 19.5, 1.9, 2.3  # brass and scintillator, averaged
MIP_HCAL = 0.0105                               # GeV/cm, averaged
HB_R = np.array([177.7, 190.5, 214.2, 244.8, 287.6])            # the four depths of the barrel towers
HE_Z = np.array([388.0, 400.0, 418.0, 436.0, 462.0, 488.0, 522.0, 570.0])   # up to seven depths in the endcap
ETA_EDGES = np.array([0.0, 0.087, 0.174, 0.261, 0.348, 0.435, 0.522, 0.609, 0.696, 0.783, 0.870, 0.957, 1.044, 1.131,
                      1.218, 1.305, 1.392, 1.479, 1.566, 1.653, 1.740, 1.830, 1.930, 2.043, 2.172, 2.322, 2.500, 2.650,
                      2.868, 3.000])            # towers ieta = 1 ... 29
HB_DEPTHS = {**{i: 4 for i in range(1, 16)}, 16: 3}
HE_DEPTHS = {17: 3, 18: 5, 19: 6, 20: 6, 21: 6, 22: 6, 23: 6, 24: 6, 25: 6, 26: 7, 27: 7, 28: 7, 29: 3}
ND_HBHE = np.array([0] + [HB_DEPTHS.get(i, HE_DEPTHS.get(i, 0)) for i in range(1, 30)])
S_HB, S_HE, S_HO = 1.0 / 108.0, 1.0 / 165.0, 1.0 / 250.0        # scintillator energy per GeV of electromagnetic shower
H_OVER_E_ECAL, H_OVER_E_HCAL = 0.62, 0.72
R_HO = (382.0, 420.0)
ETA_HO = 1.262
Z_HF, Z_FIELD_END = 1115.0, 650.0
HF_EDGES = np.array([2.853, 2.964, 3.139, 3.314, 3.489, 3.664, 3.839, 4.013, 4.191, 4.363, 4.538, 4.716, 4.889, 5.191])
PE_HF = {"em": (0.33, 0.10), "had": (0.21, 0.15)}   # photoelectrons per GeV in the long and the short fibres
E0_PI, E0_P, K_FEM = 0.8, 2.6, 0.82                 # electromagnetic fraction f = 1 - (E/E0)^(k-1)

# detector codes
PXB, PXF, TIB, TID, TOB, TEC = 1, 2, 3, 4, 5, 6
DT, CSC, RPC, GEM = 11, 12, 13, 14
EB, EE, ES, HB, HE, HO, HF = 21, 22, 23, 24, 25, 26, 27
DET_NAME = {0: "boundary", PXB: "PXB", PXF: "PXF", TIB: "TIB", TID: "TID", TOB: "TOB", TEC: "TEC", DT: "DT", CSC: "CSC", RPC: "RPC",
            GEM: "GEM", EB: "EB", EE: "EE", ES: "ES", HB: "HB", HE: "HE", HO: "HO", HF: "HF", -1: "beam pipe", -2: "services"}
TRACKER, MUON = (PXB, PXF, TIB, TID, TOB, TEC), (DT, CSC, RPC, GEM)
ECAL, HCAL = (EB, EE, ES), (HB, HE, HO, HF)
CODE = 10 ** 9                    # cell code = detector x 10^9 + index

# the tracker. The thickness of a layer in radiation lengths at normal incidence is t = t0 + t1 u, with u = |z| / zmax
# on a barrel layer and u = (r - rin) / (rout - rin) on a disk: the cables and cooling pipes of a layer run to its
# end and pile up there. The numbers t0, t1 of the sensitive layers and of the dead pieces (det = -2) are tuned to
# the material of the CMS simulation geometry, scanned with ROOT's TGeo (validate_SIM.py).
# barrel layers (det, layer, r, zmin, zmax, t0, sensor thickness, stereo, t1)
BARREL = [
    (-1, 0, 2.25, 0.0, 190.0, 0.0023, 0.0, 0, 0.0),     # beryllium beam pipe, 0.8 mm
    (PXB, 1, 2.9, 0.0, 27.0, 0.0091, 0.0285, 0, 0.0057), (PXB, 2, 6.8, 0.0, 27.0, 0.0091, 0.0285, 0, 0.0057),
    (PXB, 3, 10.9, 0.0, 27.0, 0.0091, 0.0285, 0, 0.0057), (PXB, 4, 16.0, 0.0, 27.0, 0.0091, 0.0285, 0, 0.0057),
    (-2, 0, 20.0, 0.0, 280.0, 0.088, 0.0, 0, 0.0),      # pixel support tube and services
    (TIB, 1, 25.5, 0.0, 70.0, 0.0245, 0.032, 1, 0.0196), (TIB, 2, 33.9, 0.0, 70.0, 0.0245, 0.032, 1, 0.0196),
    (TIB, 3, 41.85, 0.0, 70.0, 0.0175, 0.032, 0, 0.0140), (TIB, 4, 49.8, 0.0, 70.0, 0.0175, 0.032, 0, 0.0140),
    (-2, 0, 55.0, 70.0, 120.0, 0.197, 0.0, 0, 0.0),     # services between the inner and the outer barrel
    (TOB, 1, 60.8, 0.0, 109.0, 0.028, 0.050, 1, 0.066), (TOB, 2, 69.2, 0.0, 109.0, 0.028, 0.050, 1, 0.066),
    (TOB, 3, 78.0, 0.0, 109.0, 0.021, 0.050, 0, 0.0495), (TOB, 4, 86.8, 0.0, 109.0, 0.021, 0.050, 0, 0.0495),
    (TOB, 5, 96.5, 0.0, 109.0, 0.021, 0.050, 0, 0.0495), (TOB, 6, 108.0, 0.0, 109.0, 0.021, 0.050, 0, 0.0495),
    (-2, 0, 118.0, 0.0, 280.0, 0.017, 0.0, 0, 0.0),     # outer support tube and thermal screen
    (0, 0, R_TRK, 0.0, Z_TRK, 0.0, 0.0, 0, 0.0),        # the tracker / calorimeter boundary
]
# disks, one side (det, layer, z, rin, rout, t0, sensor thickness, t1)
DISKS_1 = [
    (PXF, 1, 29.1, 4.5, 16.1, 0.020, 0.0285, 0.0), (PXF, 2, 39.6, 4.5, 16.1, 0.020, 0.0285, 0.0), (PXF, 3, 51.6, 4.5, 16.1, 0.020, 0.0285, 0.0),
    (-2, 0, 65.0, 5.0, 22.0, 0.113, 0.0, 0.0),          # pixel services
    (TID, 1, 80.0, 22.0, 50.0, 0.025, 0.032, 0.159), (TID, 2, 93.0, 22.0, 50.0, 0.025, 0.032, 0.159), (TID, 3, 106.0, 22.0, 50.0, 0.025, 0.032, 0.159),
    (-2, 0, 121.0, 58.0, 116.0, 0.357, 0.0, 0.0),       # services at the end of the outer barrel
    (TEC, 1, 131.0, 22.0, 113.5, 0.030, 0.032, 0.0), (TEC, 2, 145.0, 22.0, 113.5, 0.030, 0.032, 0.0), (TEC, 3, 159.0, 22.0, 113.5, 0.030, 0.032, 0.0),
    (TEC, 4, 173.0, 30.0, 113.5, 0.030, 0.032, 0.0), (TEC, 5, 187.0, 30.0, 113.5, 0.030, 0.032, 0.0), (TEC, 6, 205.0, 30.0, 113.5, 0.030, 0.032, 0.0),
    (TEC, 7, 224.0, 40.0, 113.5, 0.030, 0.032, 0.0), (TEC, 8, 244.0, 40.0, 113.5, 0.030, 0.032, 0.0), (TEC, 9, 266.0, 50.0, 113.5, 0.030, 0.032, 0.0),
    (-2, 0, 282.0, 20.0, 120.0, 0.368, 0.0, 0.0),       # end flange of the tracker and the services leaving it
    (0, 0, Z_TRK, 0.0, R_TRK, 0.0, 0.0, 0.0),           # the boundary
]
DISKS = [(d, l, s * z, a, b, t, th, t1) for s in (1.0, -1.0) for (d, l, z, a, b, t, th, t1) in DISKS_1]
B_DET = np.array([b[0] for b in BARREL]); B_LAY = np.array([b[1] for b in BARREL]); B_R = np.array([b[2] for b in BARREL])
B_ZMIN = np.array([b[3] for b in BARREL]); B_ZMAX = np.array([b[4] for b in BARREL]); B_T = np.array([b[5] for b in BARREL])
B_TH = np.array([b[6] for b in BARREL]); B_ST = np.array([b[7] for b in BARREL]); B_T1 = np.array([b[8] for b in BARREL]); B_R2 = B_R * B_R
D_DET = np.array([d[0] for d in DISKS]); D_LAY = np.array([d[1] for d in DISKS]); D_Z = np.array([d[2] for d in DISKS])
D_RIN = np.array([d[3] for d in DISKS]); D_ROUT = np.array([d[4] for d in DISKS]); D_T = np.array([d[5] for d in DISKS])
D_TH = np.array([d[6] for d in DISKS]); D_T1 = np.array([d[7] for d in DISKS])


def layer_t(barrel: bool, j: int, z: float, r: float) -> float:
    """Radiation lengths of surface j at normal incidence, where the particle crosses it."""
    if barrel:
        return B_T[j] + B_T1[j] * abs(z) / B_ZMAX[j]
    return D_T[j] + D_T1[j] * (r - D_RIN[j]) / max(D_ROUT[j] - D_RIN[j], 1e-9)


NB = len(BARREL)

# the muon stations behind the calorimeters: (kind, r or |z|, lo, hi, energy lost on the way from the previous one, hits)
MU_BARREL = [(425.0, 0.40, 12, 2), (510.0, 0.34, 12, 2), (615.0, 0.72, 12, 1), (720.0, 0.72, 8, 1)]      # DT layers, RPC layers; |z| < 661
MU_ENDCAP = [(572.0, 130.0, 255.0, 0.00, GEM, 2), (600.0, 100.0, 260.0, 0.00, CSC, 6), (700.0, 275.0, 695.0, 0.00, CSC, 6),
             (830.0, 140.0, 695.0, 0.68, CSC, 6), (935.0, 160.0, 695.0, 0.68, CSC, 6), (1025.0, 180.0, 695.0, 0.29, CSC, 6)]

FLY = {211, 321, 130, 310, 3122, 3222, 3112, 3312, 3322, 3334}     # these fly and may decay in flight
KINDS = ["charged hadron", "neutral hadron", "photon", "electron", "muon"]
STAGES = {"transport": (0, 0, 0, 0), "B": (1, 0, 0, 0), "B_Decay": (1, 1, 0, 0), "B_Decay_EM": (1, 1, 1, 0), "full": (1, 1, 1, 1)}
PROC = {"decay": 1, "conv": 2, "brem": 3, "had": 4}
PROC_NAME = {0: "primary", 1: "decay", 2: "conversion", 3: "bremsstrahlung", 4: "nuclear interaction"}


# ---------------------------------------------------------------------------
# the formulas
# ---------------------------------------------------------------------------
def helix_radius(pt: float, b: float = B_FIELD) -> float:
    """R [cm] = pT [GeV] / (0.3 B [T]) x 100."""
    return pt / (0.0029979 * b)


def looper_pt(r: float, b: float = B_FIELD) -> float:
    """The transverse momentum below which a track from the beam line cannot reach radius r [cm]: 2 R = r."""
    return 0.0029979 * b * r / 2.0


def bethe_si(bg: float) -> float:
    """Mean energy loss of a heavy charged particle in silicon [MeV cm2/g] (PDG, with Sternheimer's density effect)."""
    bg = max(bg, 0.05)
    g = math.sqrt(1.0 + bg * bg)
    b2 = bg * bg / (g * g)
    tmax = 2.0 * 0.511 * bg * bg                      # MeV, for M >> m_e
    x = math.log10(bg)
    if x >= 2.8716:
        delta = 2.0 * math.log(10.0) * x - 4.4355
    elif x >= 0.2015:
        delta = 2.0 * math.log(10.0) * x - 4.4355 + 0.1492 * (2.8716 - x) ** 3.2546
    else:
        delta = 0.14 * 10.0 ** (2.0 * (x - 0.2015))
    arg = 2.0 * 0.511 * bg * bg * tmax / (173e-6) ** 2
    return 0.307075 * 0.4985 / b2 * (0.5 * math.log(arg) - b2 - 0.5 * delta)


DEDX_MIN = None


def landau_si(bg: float, x_cm: float, rng) -> float:
    """Energy lost in x cm of silicon [GeV]: the most probable value of Landau and Vavilov plus a Moyal-distributed tail."""
    bg = max(bg, 0.05)
    b2 = bg * bg / (1.0 + bg * bg)
    xi = 0.5 * 0.307075 * 0.4985 * x_cm * RHO_SI / b2                 # MeV
    xl = math.log10(bg)
    delta = 2.0 * math.log(10.0) * xl - 4.4355 if xl >= 2.8716 else (2.0 * math.log(10.0) * xl - 4.4355 + 0.1492 * (2.8716 - xl) ** 3.2546 if xl >= 0.2015 else 0.0)
    mpv = xi * (math.log(2.0 * 0.511 * bg * bg / 173e-6) + math.log(xi / 173e-6) + 0.2 - b2 - delta)
    z = rng.standard_normal()
    lam = -math.log(z * z + 1e-12)                                    # Moyal: its mode is at 0
    return max(mpv + xi * lam, 0.1 * mpv) * 1e-3


def highland(p: float, beta: float, t: float) -> float:
    """The width of the multiple-scattering angle in a plane after t radiation lengths (Highland, Lynch and Dahl)."""
    if t <= 0:
        return 0.0
    return 0.0136 / (beta * p) * math.sqrt(t) * (1.0 + 0.038 * math.log(max(t, 1e-4)))


PAIR_LOGE = np.log10([0.001, 0.002, 0.005, 0.01, 0.03, 0.1, 0.3, 1.0, 10.0, 100.0])
PAIR_FRAC = np.array([0.0, 0.02, 0.20, 0.36, 0.60, 0.78, 0.89, 0.94, 0.99, 1.0])


def pair_fraction(e: float) -> float:
    """The pair-production cross section of a photon of energy e [GeV] in silicon, relative to its high-energy
    limit 7 / (9 X0) (approximate values read from the NIST XCOM tables)."""
    return float(np.interp(math.log10(max(e, 1e-3)), PAIR_LOGE, PAIR_FRAC))


def conversion_probability(t: float, e: float = 1e3) -> float:
    return 1.0 - math.exp(-7.0 / 9.0 * t * pair_fraction(e))


def f_em_mean(e: float, e0: float = E0_PI) -> float:
    """The mean electromagnetic fraction of a hadronic shower (Gabriel, Groom et al.): 1 - (E/E0)^(k-1)."""
    return max(0.0, 1.0 - (max(e, 1e-6) / e0) ** (K_FEM - 1.0))


def gp_parameters(e: float, ec: float = EC_PBWO4, z: float = Z_PBWO4):
    """Grindhammer and Peters, homogeneous media: mean and width of ln(t_max) and ln(alpha) and their correlation,
    for an electron of energy e; t_max = (alpha - 1) / beta in radiation lengths."""
    ly = math.log(max(e / ec, 4.0))
    m_t, s_t = math.log(max(ly - 0.812, 0.3)), 1.0 / max(-1.4 + 1.26 * ly, 1.0)
    m_a, s_a = math.log(0.81 + (0.458 + 2.26 / z) * ly), 1.0 / max(-0.58 + 0.86 * ly, 1.0)
    return m_t, s_t, m_a, s_a, max(0.705 - 0.023 * ly, 0.0)


def bock_parameters(e: float):
    """The longitudinal profile of the hadronic part of a shower, a Gamma distribution in interaction lengths (Bock et al.)."""
    le = math.log(max(e, 0.5))
    return max(0.62 + 0.32 * le, 0.5), max(0.91 - 0.02 * le, 0.6)


def kind_of(pid: int) -> int:
    a = abs(pid)
    if a == 22:
        return 2
    if a == 11:
        return 3
    if a == 13:
        return 4
    return 0 if dec.charge3(pid) != 0 else 1


def two_body(M, m1, m2, rng):
    e1 = (M * M + m1 * m1 - m2 * m2) / (2 * M)
    p = math.sqrt(max(e1 * e1 - m1 * m1, 0.0))
    ct = 2 * rng.random() - 1
    st = math.sqrt(1 - ct * ct)
    ph = 2 * math.pi * rng.random()
    v = np.array([p * st * math.cos(ph), p * st * math.sin(ph), p * ct])
    return np.array([e1, *v]), np.array([M - e1, *(-v)])


def boost(p4, P):
    """p4 given in the rest frame of P (E, px, py, pz) -> the frame where P has that momentum."""
    M = math.sqrt(max(P[0] ** 2 - P[1] ** 2 - P[2] ** 2 - P[3] ** 2, 1e-18))
    b = P[1:] / M
    g = P[0] / M
    bp = float(b @ p4[1:])
    e = g * p4[0] + bp
    return np.array([e, *(p4[1:] + b * (bp / (g + 1.0) + p4[0]))])


def phase_space(M, masses, rng):
    """Flat n-body phase space in the rest frame of M (the M-generator of GENBOD, n = 2 ... 4)."""
    n = len(masses)
    if n == 2:
        return list(two_body(M, masses[0], masses[1], rng))
    tk = M - sum(masses)
    wmax = None
    for _ in range(200):
        u = np.sort(rng.random(n - 2))
        inv = [sum(masses[: i + 2]) + (u[i] * tk) for i in range(n - 2)] + [M]      # masses of the first 2, 3, ... n
        inv = [masses[0]] + inv
        w, ok = 1.0, True
        for i in range(1, n):
            a, b_, c_ = inv[i], inv[i - 1], masses[i]
            q2 = (a * a - (b_ + c_) ** 2) * (a * a - (b_ - c_) ** 2)
            if q2 <= 0:
                ok = False
                break
            w *= math.sqrt(q2) / (2 * a)
        if not ok:
            continue
        if wmax is None:
            wmax = (0.5 * tk) ** (n - 1) + 1e-12
        if rng.random() * wmax <= w:
            break
    out = [None] * n
    P = np.array([M, 0.0, 0.0, 0.0])
    for i in range(n - 1, 0, -1):
        rest, last = two_body(inv[i], inv[i - 1], masses[i], rng)
        out[i] = boost(last, P)
        P = boost(rest, P)
    out[0] = P
    return out


# ---------------------------------------------------------------------------
# the cells of the calorimeters: position -> code, code -> position
# ---------------------------------------------------------------------------
def ecal_codes(P):
    """Cell code of every point of P (n, 3) inside the crystals: EB by (eta, phi) of the point, EE by its (x, y)
    projected on the front face; 0 where there is no crystal."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    r = np.hypot(x, y)
    eta = np.arcsinh(z / np.maximum(r, 1e-9))
    phi = np.arctan2(y, x)
    code = np.zeros(len(P), dtype=np.int64)
    eb = (r >= R_EB) & (r <= R_EB_BACK) & (np.abs(eta) < ETA_EB)
    ie = np.clip(np.floor(eta / D_ETA_EB).astype(np.int64) + 85, 0, 169)
    ip = np.floor((phi + math.pi) / (2 * math.pi / 360)).astype(np.int64) % 360
    code[eb] = EB * CODE + ie[eb] * 360 + ip[eb]
    az = np.abs(z)
    sc = Z_EE / np.maximum(az, 1e-9)
    ix = np.clip(np.floor(x * sc / X_EE).astype(np.int64) + NX_EE // 2, 0, NX_EE - 1)
    iy = np.clip(np.floor(y * sc / X_EE).astype(np.int64) + NX_EE // 2, 0, NX_EE - 1)
    rf = np.hypot((ix - NX_EE // 2 + 0.5) * X_EE, (iy - NX_EE // 2 + 0.5) * X_EE)      # a crystal exists if its centre is in the ring
    ee = (az >= Z_EE) & (az <= Z_EE_BACK) & (rf >= R_EE_MIN) & (rf <= R_EE_MAX)
    code[ee] = EE * CODE + ((z[ee] > 0).astype(np.int64) * NX_EE + ix[ee]) * NX_EE + iy[ee]
    return code


def es_codes(x, y, side, plane):
    """Preshower strips, 1.9 mm x 6.1 cm: plane 0 measures x, plane 1 measures y."""
    if plane == 0:
        a, b_ = np.floor(x / 0.19).astype(np.int64) + 1024, np.floor(y / 6.1).astype(np.int64) + 32
    else:
        a, b_ = np.floor(y / 0.19).astype(np.int64) + 1024, np.floor(x / 6.1).astype(np.int64) + 32
    return ES * CODE + ((side * 2 + plane) * 2048 + a) * 64 + b_


def hcal_codes(P):
    """Cell code of every point inside the scintillator towers of HB, HE or HO; 0 elsewhere."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    r = np.hypot(x, y)
    eta = np.arcsinh(z / np.maximum(r, 1e-9))
    ae = np.abs(eta)
    phi = np.arctan2(y, x)
    side = (z > 0).astype(np.int64)
    ieta = np.clip(np.searchsorted(ETA_EDGES, ae, side="right"), 1, 29)
    code = np.zeros(len(P), dtype=np.int64)
    nphi = np.where(ieta <= 20, 72, 36)
    ip = np.floor((phi + math.pi) / (2 * math.pi) * nphi).astype(np.int64) % nphi
    nd = ND_HBHE[ieta]
    hb = (ieta <= 16) & (r >= R_HB) & (r < R_HB_BACK)
    dep = np.minimum(np.clip(np.searchsorted(HB_R, r, side="right"), 1, 4), nd)
    code[hb] = HB * CODE + ((side[hb] * 32 + ieta[hb]) * 8 + dep[hb]) * 72 + ip[hb]
    az = np.abs(z)
    he = (ieta >= 17) & (ae < 3.0) & (az >= Z_HE) & (az < Z_HE_BACK)
    dep = np.minimum(np.clip(np.searchsorted(HE_Z, az, side="right"), 1, 7), nd)
    code[he] = HE * CODE + ((side[he] * 32 + ieta[he]) * 8 + dep[he]) * 72 + ip[he]
    ho = (ae < ETA_HO) & (r >= R_HO[0]) & (r <= R_HO[1])
    iho = np.clip(np.searchsorted(ETA_EDGES, ae, side="right"), 1, 15)
    code[ho] = HO * CODE + (side[ho] * 32 + iho[ho]) * 72 + ip[ho]
    return code


def hf_codes(x, y, side, depth):
    r = np.hypot(x, y)
    ae = np.arcsinh(Z_HF / np.maximum(r, 1e-9))
    it = np.clip(np.searchsorted(HF_EDGES, ae, side="right") - 1, 0, 12)
    nphi = np.where(it >= 11, 18, 36)
    ip = np.floor((np.arctan2(y, x) + math.pi) / (2 * math.pi) * nphi).astype(np.int64) % nphi
    return HF * CODE + ((side * 16 + it) * 2 + depth) * 36 + ip


def n_cells() -> dict:
    """How many cells the detector of this script has."""
    ix, iy = np.meshgrid(np.arange(NX_EE), np.arange(NX_EE), indexing="ij")
    rf = np.hypot((ix - NX_EE // 2 + 0.5) * X_EE, (iy - NX_EE // 2 + 0.5) * X_EE)
    nee = 2 * int(((rf >= R_EE_MIN) & (rf <= R_EE_MAX)).sum())
    return {"EB": 170 * 360, "EE": nee, "HB": 2 * 72 * sum(HB_DEPTHS.values()),
            "HE": 2 * sum(v * (72 if k <= 20 else 36) for k, v in HE_DEPTHS.items()), "HO": 2 * 15 * 72, "HF": 2 * 2 * (11 * 36 + 2 * 18)}


def cell_position(codes):
    """(det, x, y, z) of the centre of every cell: for the displays."""
    codes = np.asarray(codes, dtype=np.int64)
    det = codes // CODE
    idx = codes % CODE
    x = np.zeros(len(codes)); y = np.zeros(len(codes)); z = np.zeros(len(codes))
    m = det == EB
    eta = ((idx[m] // 360) - 85 + 0.5) * D_ETA_EB
    phi = ((idx[m] % 360) + 0.5) * 2 * math.pi / 360 - math.pi
    rr = R_EB + 8.0
    x[m], y[m], z[m] = rr * np.cos(phi), rr * np.sin(phi), rr * np.sinh(eta)
    m = det == EE
    side = idx[m] // (NX_EE * NX_EE)
    x[m] = ((idx[m] // NX_EE) % NX_EE - NX_EE // 2 + 0.5) * X_EE
    y[m] = (idx[m] % NX_EE - NX_EE // 2 + 0.5) * X_EE
    z[m] = np.where(side == 1, 1.0, -1.0) * (Z_EE + 8.0)
    m = det == ES
    sp = idx[m] // (2048 * 64)
    a = ((idx[m] // 64) % 2048 - 1024 + 0.5) * 0.19
    b_ = (idx[m] % 64 - 32 + 0.5) * 6.1
    plane = sp % 2
    x[m], y[m] = np.where(plane == 0, a, b_), np.where(plane == 0, b_, a)
    z[m] = np.where(sp // 2 == 1, 1.0, -1.0) * np.where(plane == 0, Z_ES1, Z_ES2)
    for d in (HB, HE):
        m = det == d
        if not m.any():
            continue
        ip = idx[m] % 72
        dep = (idx[m] // 72) % 8
        ieta = (idx[m] // (72 * 8)) % 32
        side = np.where(idx[m] // (72 * 8 * 32) == 1, 1.0, -1.0)
        nphi = np.where(ieta <= 20, 72, 36)
        phi = (ip + 0.5) * 2 * math.pi / nphi - math.pi
        eta = 0.5 * (ETA_EDGES[ieta - 1] + ETA_EDGES[ieta])
        if d == HB:
            rr = 0.5 * (HB_R[dep - 1] + HB_R[np.minimum(dep, 4)])
            x[m], y[m], z[m] = rr * np.cos(phi), rr * np.sin(phi), side * rr * np.sinh(eta)
        else:
            zz = 0.5 * (HE_Z[dep - 1] + HE_Z[np.minimum(dep, 7)])
            rr = zz / np.sinh(eta)
            x[m], y[m], z[m] = rr * np.cos(phi), rr * np.sin(phi), side * zz
    m = det == HO
    ieta = (idx[m] // 72) % 32
    side = np.where(idx[m] // (72 * 32) == 1, 1.0, -1.0)
    phi = (idx[m] % 72 + 0.5) * 2 * math.pi / 72 - math.pi
    eta = 0.5 * (ETA_EDGES[ieta - 1] + ETA_EDGES[ieta])
    x[m], y[m], z[m] = 407.0 * np.cos(phi), 407.0 * np.sin(phi), side * 407.0 * np.sinh(eta)
    m = det == HF
    ip = idx[m] % 36
    it = (idx[m] // 72) % 16
    side = np.where(idx[m] // (72 * 16) == 1, 1.0, -1.0)
    nphi = np.where(it >= 11, 18, 36)
    phi = (ip + 0.5) * 2 * math.pi / nphi - math.pi
    eta = 0.5 * (HF_EDGES[it] + HF_EDGES[it + 1])
    rr = Z_HF / np.sinh(eta)
    x[m], y[m], z[m] = rr * np.cos(phi), rr * np.sin(phi), side * (Z_HF + 20.0)
    return det, x, y, z


# ---------------------------------------------------------------------------
# one collision through the detector
# ---------------------------------------------------------------------------
class Sim:
    def __init__(self, rng, stage="full", field=B_FIELD, t0=0.0, record=False, vv=None):
        self.rng = rng
        self.FIELD, self.DECAY, self.EM, self.HAD = STAGES[stage]
        self.b = field if self.FIELD else 0.0
        self.t0 = t0
        self.record = record            # keep the stories (signal only)
        self.vv = vv
        self.trk = []                   # (det, layer, x, y, z, t, dE, p, pid, prim)
        self.mu = []
        self.cal_c, self.cal_e, self.cal_t, self.cal_p = [], [], [], []
        self.n = Counter()
        self.sec = []                   # stored secondaries: (proc, pid, prim, x, y, z, e, parent pid)
        self.vtx = []                   # (proc, prim, x, y, z, parent pid, parent e, n products)
        self.bnd = {}                   # prim -> (x, y, z, px, py, pz) at the tracker / calorimeter boundary
        self.fate = {}                  # prim -> what ended it as a primary
        self.e = Counter()              # energy sums (lost, invisible ...)
        self.stack = []

    # ---------------------------------------------------------------- bookkeeping
    def hit(self, det, lay, x, y, z, t, de, p, pid, prim, store=None):
        (self.trk if store is None else store).append((det, lay, x, y, z, t, de, p, pid, prim))

    def deposit(self, codes, e, t, prim):
        m = codes > 0
        if not m.all():
            self.e["calo_dead"] += float(np.sum(e[~m])) if isinstance(e, np.ndarray) else 0.0
            codes, e, t = codes[m], e[m], t[m]
        if len(codes):
            self.cal_c.append(codes); self.cal_e.append(e.astype(np.float32)); self.cal_t.append(t.astype(np.float32))
            self.cal_p.append(np.full(len(codes), prim, dtype=np.int32))

    def push(self, pid, x, p3, t, prim, proc, parent=0, primary=False):
        """A new particle on the stack: (pid, x y z, px py pz, t, prim, is_primary, decay path left)."""
        a = abs(pid)
        if a in (12, 14, 16):
            self.n["neutrinos"] += 1
            self.e["neutrino"] += float(np.linalg.norm(p3))
            return
        m = dec.m0(pid)
        pp = float(np.linalg.norm(p3))
        if not primary:
            if a not in FLY and a != 2112 and a != 22 and a != 11 and a != 13 and a != 2212:
                # anything else decays at once (pi0, eta, Sigma0 ...)
                self.decay(pid, np.asarray(x, float), np.array([math.sqrt(pp * pp + m * m), *p3]), t, prim, prompt=True)
                return
            self.sec.append((proc, pid, prim, x[0], x[1], x[2], math.sqrt(pp * pp + m * m), parent))
            self.n["sec_" + PROC_NAME[proc]] += 1
        ldec = math.inf
        if self.DECAY and a in FLY and m > 0:
            ldec = -math.log(1.0 - self.rng.random()) * pp / m * dec.tau0(pid) / 10.0
        self.stack.append((pid, float(x[0]), float(x[1]), float(x[2]), float(p3[0]), float(p3[1]), float(p3[2]), t, prim, primary, ldec))

    def end(self, prim, primary, what):
        if primary:
            self.fate[prim] = what

    # ---------------------------------------------------------------- decays
    def decay(self, pid, x, P, t, prim, prompt=False, primary=False):
        chans, total = dec.open_channels(pid)
        if total <= 0:
            return
        M = dec.m0(pid)
        for _ in range(20):
            r, acc = self.rng.random() * total, 0.0
            for br, me, prods in chans:
                acc += br
                if r <= acc:
                    break
            ids = [(-i if (pid < 0 and dec.has_anti(i)) else i) for i in prods]
            masses = [dec.m0(i) for i in ids]
            if sum(masses) < M:
                break
        else:
            return
        ps = phase_space(M, masses, self.rng)
        if not prompt:
            self.n["decays"] += 1
            self.n["decay_%d" % abs(pid)] += 1
            self.vtx.append((PROC["decay"], prim, x[0], x[1], x[2], pid, P[0], len(ids)))
            if self.vv:
                self.vv(f"      decay  {dec.name(pid):>10s} E {P[0]:7.2f} at r {math.hypot(x[0], x[1]):6.1f} z {x[2]:7.1f} cm -> " + " ".join(dec.name(i) for i in ids))
        for i, q in zip(ids, ps):
            q = boost(q, P)
            self.push(i, x, q[1:], t, prim, PROC["decay"], parent=pid)

    # ---------------------------------------------------------------- the tracker
    def next_surface(self, x, y, z, phi, cot, kap):
        """Transverse path to the next surface of the tracker along the helix (or the line, kap = 0)."""
        eps = 1e-7
        if abs(kap) > 1e-9:
            rho = 1.0 / abs(kap)
            xc, yc = x - math.sin(phi) / kap, y + math.cos(phi) / kap
            d = math.hypot(xc, yc)
            if d < 1e-9:
                sb = np.full(NB, np.inf)
            else:
                c0 = (B_R2 - d * d - rho * rho) / (2.0 * d * rho)
                ok = np.abs(c0) <= 1.0
                da = np.arccos(np.clip(c0, -1.0, 1.0))
                ac = math.atan2(yc, xc)
                a0 = math.atan2(y - yc, x - xc)
                sg = 1.0 if kap > 0 else -1.0
                d1 = ((ac + da - a0) * sg) % (2 * math.pi)
                d2 = ((ac - da - a0) * sg) % (2 * math.pi)
                d1[d1 < eps] += 2 * math.pi
                d2[d2 < eps] += 2 * math.pi
                d1[d1 > 2 * math.pi - eps] = np.inf
                d2[d2 > 2 * math.pi - eps] = np.inf
                sb = np.minimum(d1, d2) * rho
                sb[~ok] = np.inf
        else:
            cx, cy = math.cos(phi), math.sin(phi)
            bb = x * cx + y * cy
            disc = bb * bb - (x * x + y * y - B_R2)
            sq = np.sqrt(np.maximum(disc, 0.0))
            s1, s2 = -bb - sq, -bb + sq
            s1[s1 < 1e-6] = np.inf
            s2[s2 < 1e-6] = np.inf
            sb = np.minimum(s1, s2)
            sb[disc < 0] = np.inf
        zb = np.abs(z + sb * cot)
        with np.errstate(invalid="ignore"):
            sb[(zb < B_ZMIN) | (zb > B_ZMAX) | ~np.isfinite(zb)] = np.inf
        ib = int(np.argmin(sb))
        if abs(cot) > 1e-12:
            sd = (D_Z - z) / cot
            sd[sd < 1e-6] = np.inf
        else:
            sd = np.full(len(D_Z), np.inf)
        # the nearest disks first: check the radius where the helix meets them
        best_d, jd = np.inf, -1
        for j in np.argsort(sd):
            s = sd[j]
            if not np.isfinite(s) or s >= sb[ib]:
                break
            xx, yy, _ = self.move_xy(x, y, phi, kap, s)
            rr = math.hypot(xx, yy)
            if D_RIN[j] <= rr <= D_ROUT[j]:
                best_d, jd = s, int(j)
                break
        if jd >= 0:
            return 1, jd, best_d
        return 0, ib, float(sb[ib])

    @staticmethod
    def move_xy(x, y, phi, kap, s):
        if abs(kap) > 1e-9:
            p2 = phi + kap * s
            return x + (math.sin(p2) - math.sin(phi)) / kap, y - (math.cos(p2) - math.cos(phi)) / kap, p2
        return x + s * math.cos(phi), y + s * math.sin(phi), phi

    def track(self, item):
        pid, x, y, z, px, py, pz, t, prim, primary, ldec = item
        a = abs(pid)
        q = dec.charge3(pid) / 3.0
        m = dec.m0(pid)
        rng = self.rng
        hadron = a > 100
        r0 = math.hypot(x, y)
        if r0 >= R_TRK or abs(z) >= Z_TRK:
            return self.calorimeter(pid, np.array([x, y, z]), np.array([px, py, pz]), t, prim, primary, m)
        steps = 0
        while True:
            steps += 1
            pt = math.hypot(px, py)
            p = math.sqrt(pt * pt + pz * pz)
            if pt < 1e-9:
                pt = 1e-9
            e = math.sqrt(p * p + m * m)
            beta = p / e
            phi = math.atan2(py, px)
            cot = pz / pt
            kap = -q * 0.0029979 * self.b / pt if q != 0 else 0.0
            kindsurf, js, sT = self.next_surface(x, y, z, phi, cot, kap)
            if not np.isfinite(sT) or steps > 4000:
                self.n["lost_in_tracker"] += 1
                return self.end(prim, primary, "stopped")
            path = sT * p / pt
            if path >= ldec:                                   # it decays before the next surface
                sT = ldec * pt / p
                x, y, phi = self.move_xy(x, y, phi, kap, sT)
                z += sT * cot
                t += ldec / (beta * C)
                self.end(prim, primary, "decay")
                return self.decay(pid, np.array([x, y, z]), np.array([e, pt * math.cos(phi), pt * math.sin(phi), pz]), t, prim, primary=primary)
            ldec -= path
            x, y, phi = self.move_xy(x, y, phi, kap, sT)
            z += sT * cot
            t += path / (beta * C)
            px, py = pt * math.cos(phi), pt * math.sin(phi)
            if t - self.t0 > T_MAX:
                self.n["time_cut"] += 1
                return self.end(prim, primary, "stopped")
            if kindsurf == 0:
                det, lay, th, st = int(B_DET[js]), int(B_LAY[js]), B_TH[js], int(B_ST[js])
                r = math.hypot(x, y)
                t0x = layer_t(True, js, z, r)
                cosi = abs(px * x + py * y) / (r * p)
            else:
                det, lay, th = int(D_DET[js]), int(D_LAY[js]), D_TH[js]
                r = math.hypot(x, y)
                t0x = layer_t(False, js, z, r)
                cosi = abs(pz) / p
                st = 1 if (det == TID and r < 40.0) or (det == TEC and (r < 41.0 or 60.0 < r < 76.0)) else 0
            if det == 0:                                       # the boundary: on to the calorimeters
                if primary:
                    self.bnd[prim] = (x, y, z, px, py, pz)
                return self.calorimeter(pid, np.array([x, y, z]), np.array([px, py, pz]), t, prim, primary, m)
            f = min(1.0 / max(cosi, 1e-6), F_CAP if det != -1 else 80.0)
            tt = t0x * f
            if q != 0:
                bg = p / m
                if th > 0:                                     # a sensor: one hit, two on a stereo layer
                    for _ in range(1 + st):
                        de = landau_si(bg, th * f, rng) if self.EM else 0.0
                        de = min(de, max(e - m, 0.0))
                        self.trk.append((det, lay, x, y, z, t, de, p, pid, prim))
                if self.EM:
                    # the mean energy loss in the whole layer, sensor and dead material alike
                    loss = tt * X0_SI_G * (bethe_si(bg) if a != 11 else 1.2 * DEDX_MIN) * 1e-3
                    if a == 11:                                # bremsstrahlung (Bethe and Heitler)
                        zf = math.exp(-rng.gamma(tt / math.log(2.0)))
                        eg = (e - m) * (1.0 - zf)
                        if eg > E_SEC:
                            self.n["brems"] += 1
                            self.push(22, (x, y, z), np.array([px, py, pz]) * (eg / p), t, prim, PROC["brem"], parent=pid)
                        loss += eg
                    ekin = e - m - loss
                    if ekin < T_CUT:
                        self.n["stopped"] += 1
                        self.e["stopped"] += max(e - m, 0.0)
                        return self.end(prim, primary, "stopped")
                    e = ekin + m
                    pn = math.sqrt(e * e - m * m)
                    th0 = highland(pn, pn / e, tt)
                    if th0 > 0:
                        a1, a2 = rng.standard_normal(2) * th0
                        ux, uy, uz = px / p, py / p, pz / p
                        # two unit vectors perpendicular to the direction
                        if abs(uz) < 0.999:
                            e1 = np.array([-uy, ux, 0.0]) / math.hypot(ux, uy)
                        else:
                            e1 = np.array([1.0, 0.0, 0.0])
                        u = np.array([ux, uy, uz])
                        e2 = np.cross(u, e1)
                        u = u + a1 * e1 + a2 * e2
                        u /= np.linalg.norm(u)
                        px, py, pz = pn * u
                    else:
                        sc = pn / p
                        px, py, pz = px * sc, py * sc, pz * sc
                    p = pn
            elif a == 22 and self.EM:
                if rng.random() < conversion_probability(tt, p):
                    self.n["conversions"] += 1
                    self.n["conv_" + DET_NAME.get(det, "?")] += 1
                    self.vtx.append((PROC["conv"], prim, x, y, z, pid, p, 2))
                    if self.vv:
                        self.vv(f"      conversion  gamma E {p:7.2f} at r {r:6.1f} z {z:7.1f} cm ({DET_NAME[det]} {lay})")
                    ee = ME + rng.random() * (p - 2 * ME)
                    u = np.array([px, py, pz]) / p
                    self.end(prim, primary, "conversion")
                    self.push(-11, (x, y, z), u * math.sqrt(max(ee * ee - ME * ME, 0.0)), t, prim, PROC["conv"], parent=22)
                    e2_ = p - ee
                    self.push(11, (x, y, z), u * math.sqrt(max(e2_ * e2_ - ME * ME, 0.0)), t, prim, PROC["conv"], parent=22)
                    return
            if hadron and self.HAD:
                lam = tt * (0.84 if det == -1 else LAMBDA_PER_X0)
                if rng.random() < 1.0 - math.exp(-lam):
                    self.end(prim, primary, "nuclear")
                    return self.nuclear(pid, np.array([x, y, z]), np.array([px, py, pz]), t, prim, m, det, lay)

    def nuclear(self, pid, x, p3, t, prim, m, det, lay):
        """An inelastic nuclear interaction in the material of the tracker: a toy model of what comes out.
        The projectile's energy goes to a few pions (one third neutral) along its direction, with a transverse
        momentum of 0.35 GeV on average, and to a few slow nucleons knocked out of the nucleus."""
        rng = self.rng
        p = float(np.linalg.norm(p3))
        e = math.sqrt(p * p + m * m)
        a = abs(pid)
        baryon = a > 1000
        w = (e - m) if (baryon and pid > 0) else (e + m if baryon else e)
        self.n["nuclear"] += 1
        self.n["nucl_" + DET_NAME.get(det, "?")] += 1
        n_p, n_n = rng.poisson(0.6), rng.poisson(0.8)
        nucl = []
        for i in range(n_p + n_n):
            tk = rng.exponential(0.08)
            nucl.append((2212 if i < n_p else 2112, tk))
        w_n = sum(tk for _, tk in nucl) + 0.008 * len(nucl)
        if w_n > 0.5 * w:
            nucl, w_n = [], 0.0
        w -= w_n
        self.e["binding"] += 0.008 * len(nucl)
        n_pi = 1 + rng.poisson(1.2 * math.log(1.0 + p))
        while n_pi > 1 and w < n_pi * (MPI + 0.03):
            n_pi -= 1
        out = []
        if w > MPI + 0.01:
            share = rng.dirichlet([2.0] + [1.0] * (n_pi - 1)) if n_pi > 1 else np.array([1.0])
            u = p3 / p
            e1 = np.array([-u[1], u[0], 0.0]) / max(math.hypot(u[0], u[1]), 1e-9) if abs(u[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
            e2 = np.cross(u, e1)
            sign = 1 if dec.charge3(pid) >= 0 else -1
            for i, fsh in enumerate(share):
                ei = max(fsh * w, MPI + 1e-4)
                if i == 0 and not baryon and a in (211, 321, 130, 310):
                    pidi = pid if a != 310 else 130            # the leading particle keeps its identity
                elif rng.random() < 1.0 / 3.0:
                    pidi = 111
                else:
                    pidi = 211 * sign
                    sign = -sign
                mi = dec.m0(pidi)
                ei = max(ei, mi + 1e-4)
                pi_ = math.sqrt(ei * ei - mi * mi)
                ptr = min(rng.exponential(0.35), 0.95 * pi_)
                ps = 2 * math.pi * rng.random()
                pl = math.sqrt(pi_ * pi_ - ptr * ptr)
                out.append((pidi, pl * u + ptr * (math.cos(ps) * e1 + math.sin(ps) * e2)))
        else:
            self.e["absorbed"] += max(w, 0.0)
        for pidn, tk in nucl:
            if tk < E_SEC:
                continue
            mn = MP if pidn == 2212 else MN
            pn = math.sqrt(tk * (tk + 2 * mn))
            ct = 2 * rng.random() - 1
            stt = math.sqrt(1 - ct * ct)
            ps = 2 * math.pi * rng.random()
            out.append((pidn, pn * np.array([stt * math.cos(ps), stt * math.sin(ps), ct])))
        self.vtx.append((PROC["had"], prim, x[0], x[1], x[2], pid, e, len(out)))
        if self.vv:
            self.vv(f"      nuclear  {dec.name(pid):>10s} E {e:7.2f} at r {math.hypot(x[0], x[1]):6.1f} z {x[2]:7.1f} cm ({DET_NAME[det]} {lay}) -> " + " ".join(dec.name(i) for i, _ in out))
        for pidi, q3 in out:
            if float(np.linalg.norm(q3)) > 0:
                self.push(pidi, x, q3, t, prim, PROC["had"], parent=pid)

    # ---------------------------------------------------------------- from the tracker to the calorimeters
    def to_face(self, pid, x, p3, t):
        """Follow the particle from the boundary of the tracker to the front face of a calorimeter.
        Returns (region, position, direction, time): region EB, EE, HF or None (lost along the beam pipe)."""
        q = dec.charge3(pid) / 3.0
        pt = math.hypot(p3[0], p3[1])
        p = float(np.linalg.norm(p3))
        m = dec.m0(pid)
        beta = p / math.sqrt(p * p + m * m) if p > 0 else 1.0
        phi = math.atan2(p3[1], p3[0])
        cot = p3[2] / max(pt, 1e-9)
        kap = -q * 0.0029979 * self.b / max(pt, 1e-9) if q != 0 else 0.0
        X, Y, Z = x

        def cyl(R):
            if abs(kap) > 1e-9:
                rho = 1.0 / abs(kap)
                xc, yc = X - math.sin(phi) / kap, Y + math.cos(phi) / kap
                d = math.hypot(xc, yc)
                c0 = (R * R - d * d - rho * rho) / (2 * d * rho) if d > 1e-9 else 2.0
                if abs(c0) > 1:
                    return math.inf
                da, ac, a0 = math.acos(c0), math.atan2(yc, xc), math.atan2(Y - yc, X - xc)
                sg = 1.0 if kap > 0 else -1.0
                c = [((ac + s_ * da - a0) * sg) % (2 * math.pi) for s_ in (1, -1)]
                c = [v for v in c if 1e-7 < v < 2 * math.pi - 1e-7]
                return min(c) * rho if c else math.inf
            cx, cy = math.cos(phi), math.sin(phi)
            bb = X * cx + Y * cy
            disc = bb * bb - (X * X + Y * Y - R * R)
            if disc < 0:
                return math.inf
            c = [v for v in (-bb - math.sqrt(disc), -bb + math.sqrt(disc)) if v > 1e-6]
            return min(c) if c else math.inf

        def at(s):
            xx, yy, ph = self.move_xy(X, Y, phi, kap, s)
            return np.array([xx, yy, Z + s * cot]), np.array([math.cos(ph), math.sin(ph), cot]) / math.sqrt(1 + cot * cot), t + s * p / max(pt, 1e-9) / (beta * C)

        s_eb = cyl(R_EB)
        if np.isfinite(s_eb) and abs(Z + s_eb * cot) <= Z_EB:
            return (EB, *at(s_eb))
        if abs(cot) < 1e-9:
            return (None, x, p3 / max(p, 1e-12), t)
        zt = Z_EE if cot > 0 else -Z_EE
        s = (zt - Z) / cot
        if s <= 0:
            return (None, x, p3 / max(p, 1e-12), t)
        pos, u, tt = at(s)
        r = math.hypot(pos[0], pos[1])
        if R_EE_MIN <= r <= R_EE_MAX:
            return (EE, pos, u, tt)
        if r > R_EE_MAX:
            return (None, pos, u, tt)                          # outside every crystal: the gap between barrel and endcap
        # forward: the helix to the end of the field, then a straight line to the HF
        s = ((Z_FIELD_END if cot > 0 else -Z_FIELD_END) - Z) / cot
        pos, u, tt = at(s)
        dz = (Z_HF - Z_FIELD_END) / abs(u[2])
        pos = pos + u * dz
        tt += dz / (beta * C)
        r = math.hypot(pos[0], pos[1])
        if Z_HF / math.sinh(HF_EDGES[-1]) <= r <= Z_HF / math.sinh(HF_EDGES[0]):
            return (HF, pos, u, tt)
        return (None, pos, u, tt)

    def calorimeter(self, pid, x, p3, t, prim, primary, m):
        a = abs(pid)
        p = float(np.linalg.norm(p3))
        e = math.sqrt(p * p + m * m)
        if a in FLY or a in (2112, 2212) or a > 100:
            baryon = a > 1000
            avail = (e - m) if (baryon and pid > 0) else (e + m if baryon else e)
        else:
            avail = e
        region, pos, u, tt = self.to_face(pid, x, p3, t)
        if primary and prim not in self.fate:
            self.fate[prim] = "calorimeter" if region else "beam pipe"
        if region is None:
            self.n["beside_calorimeters"] += 1
            self.e["beside"] += avail
            return
        if a in (22, 11):
            if not self.EM:
                return
            if region == HF:
                return self.hf_shower(avail, pos, tt, prim, "em")
            return self.em_shower(avail, pos, u, tt, prim, photon=(a == 22), region=region)
        if a == 13 or not self.HAD:
            if dec.charge3(pid) == 0 or not self.EM:
                return
            return self.mip(pid, e, m, region, pos, u, tt, prim, nlam=math.inf)
        if region == HF:
            return self.hf_shower(avail, pos, tt, prim, "had")
        return self.hadron(pid, e, m, avail, region, pos, u, tt, prim)

    # ---------------------------------------------------------------- showers
    @staticmethod
    def basis(u):
        e1 = np.array([-u[1], u[0], 0.0]) / max(math.hypot(u[0], u[1]), 1e-9) if abs(u[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
        return e1, np.cross(u, e1)

    def em_spots(self, e, photon, n=None):
        """Depths [X0] and radii [Moliere radii] of the n equal-energy spots of an electromagnetic shower."""
        rng = self.rng
        if n is None:
            n = int(min(max(e / 0.004, 6), 250))
        m_t, s_t, m_a, s_a, rho = gp_parameters(e)
        z1, z2 = rng.standard_normal(2)
        ln_t = m_t + s_t * z1
        ln_a = m_a + s_a * (rho * z1 + math.sqrt(1 - rho * rho) * z2)
        alpha = max(math.exp(ln_a), 1.05)
        beta = (alpha - 1.0) / max(math.exp(ln_t), 0.2)
        depth = rng.gamma(alpha, 1.0 / beta, n)
        if photon:
            depth += rng.exponential(9.0 / 7.0)
        v = rng.random(n)
        rad = np.where(rng.random(n) < 0.85, 0.17, 1.2) * np.sqrt(v / (1.0 - v + 1e-12))
        return depth, rad, alpha, beta

    def em_shower(self, e, pos, u, t, prim, photon, region, scale=1.0, x0=X0_PBWO4, rm=RM_PBWO4, codes=ecal_codes):
        rng = self.rng
        if e <= 0:
            return
        depth, rad, alpha, beta = self.em_spots(e, photon)
        n = len(depth)
        es = np.full(n, e * scale / n)
        e1, e2 = self.basis(u)
        psi = 2 * math.pi * rng.random(n)
        shift = 0.0
        if region == EE:
            eta = abs(math.asinh(pos[2] / max(math.hypot(pos[0], pos[1]), 1e-9)))
            if ETA_ES[0] < eta < ETA_ES[1]:                    # the preshower: 2 X0 of lead, silicon, 1 X0, silicon
                f = 1.0 / abs(u[2])
                side = 1 if pos[2] > 0 else 0
                for plane, (tpl, zpl) in enumerate(((2.0 * f, Z_ES1), (3.0 * f, Z_ES2))):
                    w = 0.032 / 9.37 * f
                    frac = float(np.mean((depth > tpl - 0.25) & (depth < tpl + 0.25))) / 0.5 * w
                    if frac > 0:
                        k = 6
                        back = (abs(pos[2]) - zpl) / abs(u[2])
                        c = pos - u * back
                        vv_ = rng.random(k)
                        rr = 0.3 * np.sqrt(vv_ / (1 - vv_ + 1e-12))
                        ps = 2 * math.pi * rng.random(k)
                        P = c[None, :] + (rr * np.cos(ps))[:, None] * e1[None, :] + (rr * np.sin(ps))[:, None] * e2[None, :]
                        self.deposit(es_codes(P[:, 0], P[:, 1], side, plane), np.full(k, e * scale * frac / k), np.full(k, t), prim)
                shift = 3.0 * f
                lost = depth < shift
                self.e["es_absorber"] += float(es[lost].sum())
                depth, rad, psi, es = depth[~lost] - shift, rad[~lost], psi[~lost], es[~lost]
        P = pos[None, :] + (depth * x0)[:, None] * u[None, :] + (rad * rm * np.cos(psi))[:, None] * e1[None, :] + (rad * rm * np.sin(psi))[:, None] * e2[None, :]
        self.deposit(codes(P), es, t + depth * x0 / C, prim)

    def mip(self, pid, e, m, region, pos, u, t, prim, nlam):
        """A charged particle that only ionises: through the crystals, the HCAL, the coil and the iron, hit by hit."""
        rng = self.rng
        ekin = e - m
        if region == HF:
            return
        ur = (u[0] * pos[0] + u[1] * pos[1]) / max(math.hypot(pos[0], pos[1]), 1e-9)
        if region == EB:
            l_ecal = (R_EB_BACK - R_EB) / max(ur, 0.05)
            d_h = (R_HB - R_EB) / max(ur, 0.05)
            l_hcal = (R_HB_BACK - R_HB) / max(ur, 0.05)
            sfac = S_HB
        else:
            l_ecal = (Z_EE_BACK - Z_EE) / abs(u[2])
            d_h = (Z_HE - Z_EE) / abs(u[2])
            l_hcal = (Z_HE_BACK - Z_HE) / abs(u[2])
            sfac = S_HE
            eta = abs(math.asinh(pos[2] / max(math.hypot(pos[0], pos[1]), 1e-9)))
            if ETA_ES[0] < eta < ETA_ES[1]:
                side = 1 if pos[2] > 0 else 0
                for plane, zpl in enumerate((Z_ES1, Z_ES2)):
                    c = pos - u * ((abs(pos[2]) - zpl) / abs(u[2]))
                    self.deposit(es_codes(np.array([c[0]]), np.array([c[1]]), side, plane), np.array([landau_si(max((e * e - m * m) ** 0.5 / m, 0.1), 0.032 / abs(u[2]), rng)]), np.array([t]), prim)
        de = min(MIP_PBWO4 * l_ecal * (1.0 + 0.1 * abs(rng.standard_normal())), ekin)
        k = 3
        s = (np.arange(k) + 0.5) / k * l_ecal
        self.deposit(ecal_codes(pos[None, :] + s[:, None] * u[None, :]), np.full(k, de / k), t + s / C, prim)
        ekin -= de
        if ekin <= 0:
            return
        de = min(MIP_HCAL * l_hcal, ekin)
        k = 8
        s = d_h + (np.arange(k) + 0.5) / k * l_hcal
        # the scintillator sees the 6 % of the path that is plastic
        self.deposit(hcal_codes(pos[None, :] + s[:, None] * u[None, :]), np.full(k, 0.063 * 0.002 * l_hcal / k), t + s / C, prim)
        ekin -= de
        if ekin <= 0:
            return
        self.muon_stations(pid, ekin, pos + u * (d_h + l_hcal), u, t + (d_h + l_hcal) / C, prim, region)

    def muon_stations(self, pid, ekin, pos, u, t, prim, region):
        """Straight on from the back of the HCAL: the stations a charged particle still reaches."""
        if region == EB:
            for r, loss, ndt, nrpc in MU_BARREL:
                ekin -= loss
                if ekin <= 0:
                    return
                bb = pos[0] * u[0] + pos[1] * u[1]
                aa = u[0] ** 2 + u[1] ** 2
                disc = bb * bb - aa * (pos[0] ** 2 + pos[1] ** 2 - r * r)
                if disc < 0 or aa < 1e-9:
                    return
                s = (-bb + math.sqrt(disc)) / aa
                P = pos + u * s
                if abs(P[2]) > 661.0:
                    break
                for i in range(ndt):
                    self.mu.append((DT, i, P[0], P[1], P[2], t + s / C, 3e-6, ekin, pid, prim))
                for i in range(nrpc):
                    self.mu.append((RPC, i, P[0], P[1], P[2], t + s / C, 1e-6, ekin, pid, prim))
            return
        for z, rin, rout, loss, det, nh in MU_ENDCAP:
            ekin -= loss
            if ekin <= 0:
                return
            s = (z - abs(pos[2])) / abs(u[2])
            if s < 0:
                continue
            P = pos + u * s
            r = math.hypot(P[0], P[1])
            if rin <= r <= rout:
                for i in range(nh):
                    self.mu.append((det, i, P[0], P[1], P[2], t + s / C, 2e-6, ekin, pid, prim))
                if det == CSC and r > 275.0:
                    self.mu.append((RPC, 0, P[0], P[1], P[2], t + s / C, 1e-6, ekin, pid, prim))

    def hadron(self, pid, e, m, avail, region, pos, u, t, prim):
        """A hadron in the calorimeters: it ionises until its first inelastic interaction, drawn in interaction
        lengths (the crystals are about one), and showers from there."""
        rng = self.rng
        charged = dec.charge3(pid) != 0
        ur = (u[0] * pos[0] + u[1] * pos[1]) / max(math.hypot(pos[0], pos[1]), 1e-9)
        if region == EB:
            l_ecal = (R_EB_BACK - R_EB) / max(ur, 0.05)
            d_h = (R_HB - R_EB) / max(ur, 0.05)
            l_hcal = (R_HB_BACK - R_HB) / max(ur, 0.05)
            s_h = S_HB
        else:
            l_ecal = (Z_EE_BACK - Z_EE) / abs(u[2])
            d_h = (Z_HE - Z_EE) / abs(u[2])
            l_hcal = (Z_HE_BACK - Z_HE) / abs(u[2])
            s_h = S_HE
        nlam = rng.exponential(1.0)
        n_ecal = l_ecal / LAMBDA_PBWO4
        ekin = e - m
        if nlam < n_ecal:
            s0 = nlam * LAMBDA_PBWO4
            in_ecal = True
            self.n["had_start_ecal"] += 1
        else:
            s0 = d_h + (nlam - n_ecal) * LAMBDA_HCAL
            in_ecal = False
            if s0 > d_h + l_hcal:                              # punch-through
                self.n["punch_through"] += 1
                if charged:
                    return self.mip(pid, e, m, region, pos, u, t, prim, nlam)
                self.e["beside"] += avail
                return
            self.n["had_start_hcal"] += 1
        if charged:                                            # ionisation on the way to the interaction
            if region == EE:
                eta = abs(math.asinh(pos[2] / max(math.hypot(pos[0], pos[1]), 1e-9)))
                if ETA_ES[0] < eta < ETA_ES[1]:
                    side = 1 if pos[2] > 0 else 0
                    for plane, zpl in enumerate((Z_ES1, Z_ES2)):
                        c = pos - u * ((abs(pos[2]) - zpl) / abs(u[2]))
                        self.deposit(es_codes(np.array([c[0]]), np.array([c[1]]), side, plane), np.array([9e-5 / abs(u[2])]), np.array([t]), prim)
            le = min(s0, l_ecal)
            de = min(MIP_PBWO4 * le, 0.9 * ekin)
            k = 3
            s = (np.arange(k) + 0.5) / k * le
            self.deposit(ecal_codes(pos[None, :] + s[:, None] * u[None, :]), np.full(k, de / k), t + s / C, prim)
            avail -= de
            if not in_ecal:
                lh = s0 - d_h
                k = max(1, int(lh / 15.0))
                s = d_h + (np.arange(k) + 0.5) / k * lh
                self.deposit(hcal_codes(pos[None, :] + s[:, None] * u[None, :]), np.full(k, 0.063 * 0.002 * lh / k), t + s / C, prim)
                avail -= min(MIP_HCAL * lh, 0.5 * avail)
        if avail <= 0:
            return
        # the shower: an electromagnetic part and a hadronic part
        f0 = f_em_mean(avail, E0_P if abs(pid) > 1000 else E0_PI)
        fem = min(max(f0 + 0.6 * math.sqrt(f0 * (1 - f0) + 1e-9) * rng.standard_normal() * min(1.0, 2.0 / math.sqrt(avail + 1.0)) , 0.0), 1.0) if f0 > 0 else 0.0
        e_em, e_h = fem * avail, (1 - fem) * avail
        p0 = pos + u * s0
        e1, e2 = self.basis(u)
        t0 = t + s0 / C
        if e_em > 0.002:
            if in_ecal:
                self.em_shower(e_em, p0, u, t0, prim, photon=False, region=None)
            else:
                self.em_shower(e_em, p0, u, t0, prim, photon=False, region=None, scale=s_h, x0=X0_HCAL, rm=RM_HCAL, codes=hcal_codes)
        if e_h > 0.002:
            n = int(min(max(e_h / 0.05, 4), 120))
            al, be = bock_parameters(avail)
            dl = rng.gamma(al, 1.0 / be, n)                    # depth in interaction lengths
            core = rng.random(n) < 0.6
            rl = rng.exponential(np.where(core, 0.08, 0.35))   # radius in interaction lengths
            psi = 2 * math.pi * rng.random(n)
            if in_ecal:
                left = (l_ecal - s0) / LAMBDA_PBWO4            # interaction lengths left in the crystals
                d_cm = np.where(dl < left, dl * LAMBDA_PBWO4, (l_ecal - s0) + (d_h - l_ecal) + (dl - left) * LAMBDA_HCAL)
                lam_loc = np.where(dl < left, LAMBDA_PBWO4, LAMBDA_HCAL)
            else:
                d_cm, lam_loc = dl * LAMBDA_HCAL, np.full(n, LAMBDA_HCAL)
            P = p0[None, :] + d_cm[:, None] * u[None, :] + (rl * lam_loc * np.cos(psi))[:, None] * e1[None, :] + (rl * lam_loc * np.sin(psi))[:, None] * e2[None, :]
            es = np.full(n, e_h / n)
            ce = ecal_codes(P)
            ch = hcal_codes(P)
            tt = t0 + d_cm / C
            me = ce > 0
            mh = (ch > 0) & ~me
            sc = np.where(ch // CODE == HB, S_HB, np.where(ch // CODE == HE, S_HE, S_HO))
            self.e["invisible"] += float(es[me].sum()) * (1 - H_OVER_E_ECAL) + float(es[mh].sum()) * (1 - H_OVER_E_HCAL)
            self.e["calo_dead"] += float(es[~me & ~mh].sum())
            if me.any():
                self.deposit(ce[me], es[me] * H_OVER_E_ECAL, tt[me], prim)
            if mh.any():
                self.deposit(ch[mh], es[mh] * H_OVER_E_HCAL * sc[mh], tt[mh], prim)

    def hf_shower(self, e, pos, t, prim, kind):
        """The forward calorimeter answers in photoelectrons: Cherenkov light of the shower in the quartz fibres,
        long fibres from the front, short ones from 22 cm on. One unit per photoelectron, as in CMSSW's library."""
        rng = self.rng
        kl, ks = PE_HF[kind]
        fl = 1.0 if kind == "em" else float(np.exp(0.35 * rng.standard_normal() - 0.06))
        side = 1 if pos[2] > 0 else 0
        for depth, k in ((0, kl), (1, ks)):
            n = int(rng.poisson(k * e * fl))
            if n == 0:
                continue
            n = min(n, 200000)
            rr = rng.exponential(1.5 if kind == "em" else 5.0, n)
            ps = 2 * math.pi * rng.random(n)
            codes = hf_codes(pos[0] + rr * np.cos(ps), pos[1] + rr * np.sin(ps), side, depth)
            uc, cnt = np.unique(codes, return_counts=True)
            self.deposit(uc, cnt.astype(float), np.full(len(uc), t + 0.5), prim)

    # ---------------------------------------------------------------- the event loop
    def run(self, ids, p4, vtx, t_vtx):
        for j in range(len(ids)):
            self.push(int(ids[j]), vtx[j], p4[j, 1:], t_vtx, j, 0, primary=True)
            while self.stack:
                self.track(self.stack.pop())
        return self

    def result(self, compress=True):
        f4 = np.float32
        def arr(rows):
            if not rows:
                return {k: np.zeros(0, dtype=d) for k, d in (("det", np.int8), ("lay", np.int8), ("x", f4), ("y", f4), ("z", f4), ("t", f4), ("de", f4), ("p", f4), ("pid", np.int32), ("prim", np.int32))}
            a = np.array(rows, dtype=float)
            return dict(det=a[:, 0].astype(np.int8), lay=a[:, 1].astype(np.int8), x=a[:, 2].astype(f4), y=a[:, 3].astype(f4), z=a[:, 4].astype(f4),
                        t=a[:, 5].astype(f4), de=a[:, 6].astype(f4), p=a[:, 7].astype(f4), pid=a[:, 8].astype(np.int32), prim=a[:, 9].astype(np.int32))
        if self.cal_c:
            c, e, t, p = np.concatenate(self.cal_c), np.concatenate(self.cal_e).astype(float), np.concatenate(self.cal_t).astype(float), np.concatenate(self.cal_p)
            nhits = len(c)
            key = c * 4096 + (p % 4096) if not compress else c          # signal: per cell and primary; crowd: per cell
            if not compress:
                # exact (cell, primary) grouping
                order = np.lexsort((p, c))
                c, e, t, p = c[order], e[order], t[order], p[order]
                new = np.ones(len(c), bool)
                new[1:] = (c[1:] != c[:-1]) | (p[1:] != p[:-1])
            else:
                order = np.argsort(c, kind="stable")
                c, e, t, p = c[order], e[order], t[order], p[order]
                new = np.ones(len(c), bool)
                new[1:] = c[1:] != c[:-1]
            start = np.flatnonzero(new)
            es = np.add.reduceat(e, start)
            ts = np.add.reduceat(e * t, start) / np.maximum(es, 1e-12)
            cal = dict(code=c[start], e=es.astype(f4), t=ts.astype(f4), prim=p[start].astype(np.int32))
        else:
            nhits = 0
            cal = dict(code=np.zeros(0, np.int64), e=np.zeros(0, f4), t=np.zeros(0, f4), prim=np.zeros(0, np.int32))
        return dict(trk=arr(self.trk), mu=arr(self.mu), cal=cal, n=dict(self.n), e=dict(self.e), ncalhits=nhits,
                    sec=self.sec if self.record else len(self.sec), vtx=self.vtx if self.record else len(self.vtx),
                    bnd=self.bnd if self.record else len(self.bnd), fate=self.fate if self.record else None)


# ---------------------------------------------------------------------------
# the three samples
# ---------------------------------------------------------------------------
_W = {}


def load_stage():
    a = types.SimpleNamespace(input=g4.HERE.parent.parent / "PU" / "event_PU.lhe", library=g4.HERE.parent.parent / "PU" / "pileup_library.lhe.gz",
                              eta_cut=g4.ETA_CUT, eta_hector=g4.ETA_HECTOR, r_dec=g4.R_DEC, p_min=g4.P_MIN)
    return g4.Stage(a)


def _one(k):
    """One collision of the crowd (worker)."""
    c, sel, col, seed, stage, field = _W["crowd"], _W["sel"][k], _W["col"][k], _W["seed"], _W["stage"], _W["field"]
    rng = np.random.default_rng([seed, 1000 + k])
    t0 = col["t"] - 25.0 * col["bx"]                       # the time of the collision inside its own bunch crossing
    s = Sim(rng, stage=stage, field=field, t0=t0)
    s.run(c.ids[sel], c.p4[sel], c.vtx[sel], t0)
    r = s.result()
    r["k"], r["bx"], r["nprim"] = k, col["bx"], int(len(sel))
    return r


def run_crowd(st, which, seed, stage, field, jobs, progress=True):
    c, g = st.crowd, st.g
    block = 1 if which == "intime" else 2
    cols = {col["k"]: col for col in st.collisions if col["k"] > 0 and ((col["bx"] == 0) == (which == "intime"))}
    order = np.argsort(c.k, kind="stable")
    sel = {}
    mask = g["primary"] & (c.block == block)
    idx = np.flatnonzero(mask)
    ks = c.k[idx]
    for k in cols:
        sel[k] = idx[ks == k]
    _W.update(crowd=c, sel=sel, col=cols, seed=seed, stage=stage, field=field)
    keys = sorted(cols)
    t0 = time.time()
    out = []
    if jobs > 1:
        with mp.get_context("fork").Pool(jobs) as pool:
            for i, r in enumerate(pool.imap(_one, keys, chunksize=4)):
                out.append(r)
                if progress and (i + 1) % 100 == 0:
                    print(f"   {which}: {i + 1}/{len(keys)} collisions, {time.time() - t0:.0f} s", flush=True)
    else:
        out = [_one(k) for k in keys]
    return out, time.time() - t0


def pack_crowd(out):
    """One flat file for a crowd: every hit knows its collision k and its bunch crossing; times are relative to that crossing."""
    def cat(part, key, extra=None):
        return np.concatenate([r[part][key] for r in out]) if out else np.zeros(0)
    d = {}
    for part in ("trk", "mu"):
        for key in ("det", "lay", "x", "y", "z", "t", "de", "p", "pid"):
            d[f"{part}_{key}"] = cat(part, key)
        d[f"{part}_k"] = np.concatenate([np.full(len(r[part]["det"]), r["k"], np.int16) for r in out])
        d[f"{part}_bx"] = np.concatenate([np.full(len(r[part]["det"]), r["bx"], np.int8) for r in out])
    for key in ("code", "e", "t"):
        d[f"cal_{key}"] = cat("cal", key)
    d["cal_k"] = np.concatenate([np.full(len(r["cal"]["code"]), r["k"], np.int16) for r in out])
    d["cal_bx"] = np.concatenate([np.full(len(r["cal"]["code"]), r["bx"], np.int8) for r in out])
    d["col_k"] = np.array([r["k"] for r in out], np.int16)
    d["col_bx"] = np.array([r["bx"] for r in out], np.int8)
    d["col_nprim"] = np.array([r["nprim"] for r in out], np.int32)
    d["col_nsec"] = np.array([r["sec"] for r in out], np.int32)
    d["col_ncalhits"] = np.array([r["ncalhits"] for r in out], np.int32)
    return d


def counters(out):
    n, e = Counter(), Counter()
    for r in out:
        n.update(r["n"]); e.update(r["e"])
    return dict(n), {k: float(v) for k, v in e.items()}


def station_summary(trk_det, trk_lay, trk_de, mu_det, cal_code, cal_e) -> dict:
    s = {"tracker": {}, "muon": {}, "calo": {}}
    for d in TRACKER:
        m = trk_det == d
        s["tracker"][DET_NAME[d]] = dict(hits=int(m.sum()), de=float(trk_de[m].sum()), layers={int(l): int((m & (trk_lay == l)).sum()) for l in np.unique(trk_lay[m])})
    for d in MUON:
        s["muon"][DET_NAME[d]] = int((mu_det == d).sum())
    det = cal_code // CODE
    for d in ECAL + HCAL:
        m = det == d
        s["calo"][DET_NAME[d]] = dict(cells=int(len(np.unique(cal_code[m]))), e=float(cal_e[m].sum()))
    return s


# ---------------------------------------------------------------------------
# the signal: stage by stage, with the narration
# ---------------------------------------------------------------------------
def run_signal(st, seed, stage, field, vv=None):
    c, g = st.crowd, st.g
    sel = np.flatnonzero(g["primary"] & (c.block == 0))
    rng = np.random.default_rng([seed, 0])
    t0 = st.collisions[0]["t"]
    s = Sim(rng, stage=stage, field=field, t0=t0, record=True, vv=vv)
    s.run(c.ids[sel], c.p4[sel], c.vtx[sel], t0)
    r = s.result(compress=False)
    r["sel"] = sel
    return r


def jet_members(st, sel):
    """Index (in the list of signal primaries) of the members of ak4 jet [0], the photograph of chapter 7."""
    of = st.genjets_of["ak4"]
    rows = st.crowd.row[sel]
    return np.array([j for j, r in enumerate(rows) if of.get(int(r), -1) == 0], dtype=int)


def vec_pt(px, py):
    return float(math.hypot(np.sum(px), np.sum(py)))


def cell_pt(codes, e, sf=None):
    """Vector transverse momentum of calorimeter cells seen from the origin: E / cosh(eta) along phi of the cell."""
    det, x, y, z = cell_position(codes)
    r = np.hypot(x, y)
    eta = np.arcsinh(z / np.maximum(r, 1e-9))
    phi = np.arctan2(y, x)
    et = e / np.cosh(eta)
    return et * np.cos(phi), et * np.sin(phi), eta, phi


def signal_row(st, r, stage) -> dict:
    """One line of the stage table."""
    c = st.crowd
    sel = r["sel"]
    jm = jet_members(st, sel)
    n = Counter(r["n"])
    cal = r["cal"]
    det = cal["code"] // CODE
    bnd = r["bnd"]
    jb = [j for j in jm if j in bnd]
    in_jet = np.isin(cal["prim"], jm)
    row = dict(stage=stage, primaries=int(len(sel)), boundary=len(bnd), secondaries=len(r["sec"]),
               sec_decay=n["sec_decay"], sec_em=n["sec_conversion"] + n["sec_bremsstrahlung"], sec_had=n["sec_nuclear interaction"],
               trk=int(len(r["trk"]["det"])), mu=int(len(r["mu"]["det"])), de=float(r["trk"]["de"].sum()),
               cells=int(len(np.unique(cal["code"]))), calhits=int(r["ncalhits"]),
               jet_n=len(jm), jet_bnd_n=len(jb), jet_bnd_pt=vec_pt([bnd[j][3] for j in jb], [bnd[j][4] for j in jb]) if jb else 0.0,
               jet_ecal=float(cal["e"][in_jet & np.isin(det, ECAL)].sum()), jet_hcal=float(cal["e"][in_jet & np.isin(det, (HB, HE, HO))].sum()))
    for d in ECAL + HCAL:
        row[DET_NAME[d]] = float(cal["e"][det == d].sum())
    return row


def save_signal(path, st, r):
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
    d["prim_kind"] = np.array([kind_of(int(i)) for i in c.ids[sel]], np.int8)
    d["prim_jet0"] = np.isin(np.arange(len(sel)), jet_members(st, sel))
    fate = r["fate"]
    names = ["calorimeter", "decay", "conversion", "nuclear", "stopped", "beam pipe"]
    d["prim_fate"] = np.array([names.index(fate.get(j, "stopped")) for j in range(len(sel))], np.int8)
    d["fate_names"] = np.array(names)
    b = np.full((len(sel), 6), np.nan)
    for j, v in r["bnd"].items():
        b[j] = v
    d["prim_bnd"] = b
    sec = r["sec"]
    d["sec"] = np.array([(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]) for s in sec], dtype=float).reshape(-1, 8)
    d["vtx"] = np.array([(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]) for v in r["vtx"]], dtype=float).reshape(-1, 8)
    np.savez_compressed(path, **d)


def narrate(st, rows, r, say, field):
    c = st.crowd
    sel = r["sel"]
    ids, p4 = c.ids[sel], c.p4[sel]
    pt, eta = c.pt[sel], c.eta[sel]
    q = np.array([dec.charge3(int(i)) for i in ids]) != 0
    n = Counter(r["n"])
    full = rows[-1]
    say("== The crossing of the border: the family through the detector (standalone_SIM.py, learning mode) ==")
    say()
    say("-- 0. the detector of this script")
    nc = n_cells()
    say(f"   tracker: beam pipe r = 2.25 cm; pixel barrel r = 2.9, 6.8, 10.9, 16.0 cm; inner barrel 25.5 ... 49.8; outer barrel 60.8 ... 108.0;")
    say(f"            3 + 3 + 9 pairs of disks at |z| = 29 ... 266 cm; the boundary to the calorimeters at r = {R_TRK} cm, |z| = {Z_TRK} cm")
    say(f"   ECAL: {nc['EB']} barrel crystals from r = {R_EB:.0f} cm, |eta| < {ETA_EB}; {nc['EE']} endcap crystals at |z| = {Z_EE:.0f} cm, to |eta| = {ETA_EE}; preshower {ETA_ES[0]} < |eta| < {ETA_ES[1]}")
    say(f"   HCAL: HB {nc['HB']} cells, HE {nc['HE']}, HO {nc['HO']}, HF {nc['HF']} (long and short fibres) at |z| = {Z_HF:.0f} cm")
    say(f"   lead tungstate: X0 = {X0_PBWO4} cm, Moliere radius {RM_PBWO4} cm, critical energy {EC_PBWO4 * 1e3:.1f} MeV, interaction length {LAMBDA_PBWO4} cm;"
        f" a crystal is {(R_EB_BACK - R_EB) / X0_PBWO4:.1f} X0 and {(R_EB_BACK - R_EB) / LAMBDA_PBWO4:.2f} lambda_I deep")
    say()
    say("-- 1. the magnetic field")
    say(f"   B = {field} T: R = pT / (0.3 B) = {helix_radius(1.0, field):.1f} cm x pT[GeV]; a track from the beam line reaches radius r only if pT > 0.3 B r / 2:")
    say(f"   pT > {looper_pt(R_EB, field):.3f} GeV for the barrel ECAL (r = {R_EB:.0f} cm), {looper_pt(R_TRK, field):.3f} for the edge of the tracker, {looper_pt(2.9, field) * 1e3:.1f} MeV for the first pixel layer")
    trk_reg = q & (np.abs(eta) < 2.5)
    low = trk_reg & (pt < looper_pt(R_EB, field))
    say(f"   {int(q.sum())} of the {len(sel)} primaries are charged, {int(trk_reg.sum())} of them in |eta| < 2.5; {int(low.sum())} of those have pT < {looper_pt(R_EB, field):.3f} GeV"
        f" (median pT {np.median(pt[trk_reg]):.2f} GeV): they cannot reach the barrel ECAL")
    brl = q & (np.abs(eta) < 1.479)
    say(f"   in |eta| < 1.479: {int(brl.sum())} charged, {int((brl & (pt < looper_pt(R_EB, field))).sum())} of them loopers")
    for a_, b_ in zip(rows[:-1], rows[1:]):
        if a_["stage"] == "transport" and b_["stage"] == "B":
            say(f"   sensor crossings: {a_['trk']} on straight lines -> {b_['trk']} on helices; primaries at the boundary {a_['boundary']} -> {b_['boundary']}")
    jm = jet_members(st, sel)
    lead = jm[np.argmax(np.where(q[jm], pt[jm], 0.0))]
    R = helix_radius(pt[lead], field)
    say(f"   the leading charged member of our jet: {dec.name(int(ids[lead]))} pT {pt[lead]:.2f} GeV, eta {eta[lead]:.3f}: R = {R / 100:.1f} m;"
        f" over L = 1.1 m its sagitta is L^2 / 8R = {110.0 ** 2 / (8 * R):.2f} cm, its azimuth turns by asin(L / 2R) = {math.asin(110.0 / (2 * R)):.4f} rad")
    say()
    say("-- 2. decays in flight")
    for pid_, nm in ((211, "pi+-"), (321, "K+-"), (130, "K0L"), (310, "K0S"), (3122, "Lambda"), (3222, "Sigma+"), (3112, "Sigma-")):
        m_ = dec.m0(pid_)
        say(f"   {nm:7s} c tau = {dec.tau0(pid_) / 10.0:8.2f} cm: at p = 1 GeV it flies beta gamma c tau = {dec.tau0(pid_) / 10.0 / m_:8.1f} cm on average")
    dk = {k: v for k, v in n.items() if k.startswith("decay_")}
    say(f"   decays in flight in the full simulation: {n['decays']} ("
        + ", ".join(f"{dec.name(int(k[6:]))} {v}" for k, v in sorted(dk.items(), key=lambda kv: -kv[1])) + f"); {n['sec_decay']} daughters stored, {n['neutrinos']} neutrinos not tracked")
    vt = np.array([(v[0], math.hypot(v[2], v[3]), v[4]) for v in r["vtx"]]).reshape(-1, 3)
    def where(proc):
        m_ = vt[:, 0] == proc
        rr, zz = vt[m_, 1], np.abs(vt[m_, 2])
        return f"beam pipe and pixel volume (r < 20 cm) {int(((rr < 20) & (zz < 70)).sum())}, among the strips {int((~((rr < 20) & (zz < 70))).sum())}"
    say(f"   where: {where(PROC['decay'])}")
    say()
    say("-- 3. electromagnetic physics")
    say(f"   silicon: minimum of the Bethe-Bloch curve {DEDX_MIN:.3f} MeV cm2/g at beta gamma = 3.5; a minimum-ionising particle loses on average"
        f" {DEDX_MIN * 0.032 * RHO_SI * 1e3:.0f} keV in 320 um, most probably {landau_mpv(3.5, 0.032) * 1e6:.0f} keV")
    de = r["trk"]["de"]
    say(f"   tracker hits: {full['trk']} with {full['de']:.3f} GeV of energy loss in the sensors, {1e3 * full['de'] / max(full['trk'], 1):.3f} MeV per hit (median {1e6 * float(np.median(de)) if len(de) else 0:.0f} keV)")
    say(f"   Highland: a 1 GeV pion crossing 2.5 % X0 is deflected by {1e3 * highland(1.0, 0.99, 0.025):.2f} mrad (rms, one plane), a 10 GeV one by {1e3 * highland(10.0, 1.0, 0.025):.3f}")
    cv = {k[5:]: v for k, v in n.items() if k.startswith("conv_")}
    say(f"   conversions: {n['conversions']} ({', '.join(f'{k} {v}' for k, v in sorted(cv.items(), key=lambda kv: -kv[1]))}); bremsstrahlung photons above {E_SEC * 1e3:.0f} MeV: {n['brems']}")
    gam = (ids == 22) & (np.abs(eta) < 2.5) & (p4[:, 0] > 1.0)
    fate = r["fate"]
    conv = sum(1 for j in np.flatnonzero(gam) if fate.get(j) == "conversion")
    if gam.sum():
        frac = conv / gam.sum()
        say(f"   of the {int(gam.sum())} primary photons above 1 GeV in |eta| < 2.5, {conv} converted in the tracker: 1 - exp(-7t/9) = {frac:.2f} -> t = {-9 / 7 * math.log(max(1 - frac, 1e-6)):.2f} X0 on average")
    say(f"   particles stopped in the tracker (kinetic energy below {T_CUT * 1e3:.0f} MeV): {n['stopped']}")
    say()
    say("-- 4. hadronic physics")
    nv = {k[5:]: v for k, v in n.items() if k.startswith("nucl_")}
    say(f"   nuclear interactions in the tracker: {n['nuclear']} ({', '.join(f'{k} {v}' for k, v in sorted(nv.items(), key=lambda kv: -kv[1]))}), {n['sec_nuclear interaction']} secondaries stored")
    had = q & (np.abs(ids) > 100) & (np.abs(eta) < 2.5) & (pt > 0.5)
    nint = sum(1 for j in np.flatnonzero(had) if fate.get(j) == "nuclear")
    if had.sum():
        say(f"   of the {int(had.sum())} charged hadrons with pT > 0.5 GeV in |eta| < 2.5, {nint} ({100 * nint / had.sum():.0f} %) interact before the calorimeters:"
            f" as if the tracker were {-math.log(max(1 - nint / had.sum(), 1e-6)):.2f} lambda_I thick")
    say(f"   hadrons whose shower starts in the crystals: {n['had_start_ecal']}, in the HCAL: {n['had_start_hcal']}, punch-through: {n['punch_through']}")
    for e_ in (1.0, 10.0, 100.0):
        say(f"   a pion of {e_:5.0f} GeV: electromagnetic fraction 1 - (E/{E0_PI})^({K_FEM} - 1) = {f_em_mean(e_):.2f}; seen by the HCAL as {f_em_mean(e_) + (1 - f_em_mean(e_)) * H_OVER_E_HCAL:.2f} of an electron")
    say(f"   invisible energy of the hadronic showers: {r['e'].get('invisible', 0.0):.1f} GeV; in dead material between and behind the calorimeters: {r['e'].get('calo_dead', 0.0):.1f} GeV;"
        f" beside every calorimeter (|eta| > {HF_EDGES[-1]}): {r['e'].get('beside', 0.0):.1f} GeV in {n['beside_calorimeters']} particles")
    say()
    say("-- 5. the stages: one ingredient more each time")
    say(f"   {'stage':<12s} {'bound.':>6s} {'second.':>7s} {'dec/EM/had':>13s} {'trk hits':>8s} {'mu':>5s} {'sum dE':>7s} {'cells':>6s} {'EB':>7s} {'EE':>7s} {'HB':>6s} {'HE':>6s} {'HF pe':>6s}  jet: pT bnd, E_ECAL")
    for w in rows:
        say(f"   {w['stage']:<12s} {w['boundary']:6d} {w['secondaries']:7d} {str(w['sec_decay']) + '/' + str(w['sec_em']) + '/' + str(w['sec_had']):>13s} {w['trk']:8d} {w['mu']:5d} {w['de']:7.3f} {w['cells']:6d}"
            f" {w['EB']:7.1f} {w['EE']:7.1f} {w['HB']:6.3f} {w['HE']:6.3f} {w['HF']:6.0f}  {w['jet_bnd_pt']:6.2f} ({w['jet_bnd_n']}/{w['jet_n']}), {w['jet_ecal']:6.2f}")
    say()
    say("-- 6. the stations of the full simulation")
    t = r["trk"]
    s = station_summary(t["det"], t["lay"], t["de"], r["mu"]["det"], r["cal"]["code"], r["cal"]["e"])
    say("   tracker: " + ", ".join(f"{k} {v['hits']}" for k, v in s["tracker"].items()) + f" = {full['trk']} hits")
    for k in ("PXB", "TIB", "TOB"):
        say(f"      {k} by layer: " + ", ".join(str(v) for _, v in sorted(s["tracker"][k]["layers"].items())))
    late = int((t["t"] - st.collisions[0]["t"] > 25.0).sum())
    say(f"      hits later than 25 ns after the collision: {late}")
    say("   calorimeters: " + ", ".join(f"{k} {v['cells']} cells {v['e']:.2f}" for k, v in s["calo"].items()) + "  (GeV; HB, HE, HO scintillator energy; HF photoelectrons)")
    say("   muon chambers: " + ", ".join(f"{k} {v}" for k, v in s["muon"].items()))
    fates = Counter(fate.get(j, "stopped") for j in range(len(sel)))
    say("   what ended each primary as a primary: " + ", ".join(f"{k} {v}" for k, v in fates.most_common()))
    withhit = len(set(np.unique(t["prim"])) | set(np.unique(r["cal"]["prim"])) | set(np.unique(r["mu"]["prim"])))
    say(f"   primaries that left at least one hit: {withhit} of {len(sel)}")
    say()
    say("-- 7. our jet (the members of ak4 jet [0], the photograph of chapter 7)")
    mem = sorted(jm, key=lambda j: -p4[j, 0])
    say(f"   {len(jm)} members handed to the simulation, E = {p4[jm, 0].sum():.1f} GeV, vector pT {vec_pt(p4[jm, 1], p4[jm, 2]):.2f} GeV")
    cal = r["cal"]
    det = cal["code"] // CODE
    for j in mem[:10]:
        mj = cal["prim"] == j
        say(f"      {dec.name(int(ids[j])):>8s} E {p4[j, 0]:7.2f} pT {pt[j]:6.2f} eta {eta[j]:6.3f}: {fate.get(j, 'stopped'):<11s} tracker hits {int((t['prim'] == j).sum()):4d},"
            f" ECAL {cal['e'][mj & np.isin(det, ECAL)].sum():7.2f} GeV in {int((mj & np.isin(det, ECAL)).sum()):4d} cells, HCAL {cal['e'][mj & np.isin(det, (HB, HE, HO))].sum():6.3f}")
    inj = np.isin(cal["prim"], jm)
    me_, mh_ = inj & np.isin(det, (EB, EE)), inj & np.isin(det, (HB, HE))
    px1, py1, _, _ = cell_pt(cal["code"][me_], cal["e"][me_])
    sf = np.where(det[mh_] == HB, 125.0, 190.0)
    px2, py2, _, _ = cell_pt(cal["code"][mh_], cal["e"][mh_] * sf)
    say(f"   reaching the boundary as primaries: {full['jet_bnd_n']} of {len(jm)}, vector pT {full['jet_bnd_pt']:.2f} GeV")
    say(f"   in the crystals: {cal['e'][me_].sum():.2f} GeV in {int(me_.sum())} cells, vector pT {vec_pt(px1, py1):.2f} GeV")
    say(f"   in the HCAL scintillator: {cal['e'][mh_].sum():.3f} GeV in {int(mh_.sum())} cells; x sampling factor (125 HB, 190 HE) = {float((cal['e'][mh_] * sf).sum()):.1f} GeV, vector pT {vec_pt(px2, py2):.2f} GeV")
    say(f"   crystals + HCAL x factor: vector pT {vec_pt(np.concatenate([px1, px2]), np.concatenate([py1, py2])):.2f} GeV")


def landau_mpv(bg, x_cm):
    b2 = bg * bg / (1.0 + bg * bg)
    xi = 0.5 * 0.307075 * 0.4985 * x_cm * RHO_SI / b2
    xl = math.log10(bg)
    delta = 2.0 * math.log(10.0) * xl - 4.4355 if xl >= 2.8716 else (2.0 * math.log(10.0) * xl - 4.4355 + 0.1492 * (2.8716 - xl) ** 3.2546 if xl >= 0.2015 else 0.0)
    return xi * (math.log(2.0 * 0.511 * bg * bg / 173e-6) + math.log(xi / 173e-6) + 0.2 - b2 - delta) * 1e-3


DEDX_MIN = min(bethe_si(x) for x in np.linspace(2.5, 4.5, 81))


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", choices=["signal", "intime", "oot", "all"], default="all")
    ap.add_argument("--stage", choices=list(STAGES), default="full", help="the physics switched on (default: all of it)")
    ap.add_argument("-o", "--outdir", type=Path, default=HERE.parent, help="where simhits_<sample>.npz go")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--field", type=float, default=B_FIELD, help="the field of the solenoid [T]")
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (mp.cpu_count() or 2) - 2)))
    ap.add_argument("-v", "--verbose", action="count", default=0)
    ap.add_argument("--log", type=Path, help="write the narration here instead of the terminal")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    t00 = time.time()
    st = load_stage()
    samples = ["signal", "intime", "oot"] if args.sample == "all" else [args.sample]
    tag = "" if args.stage == "full" else "_" + args.stage
    for which in samples:
        t0 = time.time()
        if which == "signal":
            log = open(args.log, "w") if args.log else None
            out = log if log else sys.stdout
            def say(text=""):
                if args.verbose:
                    print(text, file=out)
            stages = list(STAGES) if (args.stage == "full") else [args.stage]
            rows, r = [], None
            for sg in stages:
                vv = None
                if args.verbose >= 2 and sg == stages[-1]:
                    vv = lambda s_: print(s_, file=out)
                r = run_signal(st, args.seed, sg, args.field, vv=vv)
                rows.append(signal_row(st, r, sg))
            narrate(st, rows, r, say, args.field) if args.stage == "full" else None
            save_signal(args.outdir / f"simhits_signal{tag}.npz", st, r)
            t = r["trk"]
            summ = dict(sample="signal", stage=args.stage, seed=args.seed, field=args.field, stages=rows, n=r["n"], e=r["e"],
                        stations=station_summary(t["det"], t["lay"], t["de"], r["mu"]["det"], r["cal"]["code"], r["cal"]["e"]),
                        primaries=int(len(r["sel"])), simtracks=int(len(r["sel"]) + len(r["sec"])), calhits=int(r["ncalhits"]), cells=n_cells(), seconds=time.time() - t0)
            (HERE / f"standalone_SIM_signal{tag}.json").write_text(json.dumps(to_jsonable(summ), indent=1))
            if log:
                log.close()
            if not args.quiet:
                w = rows[-1]
                print(f"standalone_SIM signal ({args.stage}): {w['primaries']} primaries -> {w['boundary']} at the boundary, {w['secondaries']} secondaries, {w['trk']} tracker hits,"
                      f" {w['cells']} cells (EB {w['EB']:.1f} EE {w['EE']:.1f} GeV, HF {w['HF']:.0f} pe), {w['mu']} muon hits; {time.time() - t0:.1f} s")
        else:
            out, dt = run_crowd(st, which, args.seed, args.stage, args.field, args.jobs, progress=not args.quiet)
            d = pack_crowd(out)
            np.savez_compressed(args.outdir / f"simhits_{which}{tag}.npz", **d)
            n, e = counters(out)
            summ = dict(sample=which, stage=args.stage, seed=args.seed, field=args.field, collisions=len(out), primaries=int(d["col_nprim"].sum()),
                        simtracks=int(d["col_nprim"].sum() + d["col_nsec"].sum()), calhits=int(d["col_ncalhits"].sum()), n=n, e=e,
                        stations=station_summary(d["trk_det"], d["trk_lay"], d["trk_de"], d["mu_det"], d["cal_code"], d["cal_e"]),
                        seconds=dt, jobs=args.jobs)
            (HERE / f"standalone_SIM_{which}{tag}.json").write_text(json.dumps(to_jsonable(summ), indent=1))
            if not args.quiet:
                print(f"standalone_SIM {which}: {len(out)} collisions, {summ['primaries']} primaries -> {summ['simtracks']} tracks, {len(d['trk_det'])} tracker hits, "
                      f"{len(d['mu_det'])} muon hits, {len(d['cal_code'])} (cell, collision) deposits; {dt:.0f} s on {args.jobs} processes")
    if not args.quiet:
        print(f"total {time.time() - t00:.0f} s")


if __name__ == "__main__":
    main()
