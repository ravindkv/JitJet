#!/usr/bin/env python3
"""Standalone pileup for the b bbar event of chapter 7: the crowd on the road, in Run 3 (2025).

Dependencies: numpy, plus chapter 7's ../../GenJet/Standalone/standalone_genjets.py (the record, the
anti-kT clustering, the jet tables; chapters 6, 5 and 2 are imported through it). No CMSSW, no FastJet:
the mixing of CMSSW's MixingModule, the luminous region of its HLLHCEvtVtxGenerator, the jet areas and
the median energy density of FastJet are written out here, step by step.

Input is the final state of chapter 7 (../../GenJet/event_GenJet.lhe: the 749 particles the detector
will see and the generator-level jets) and the pileup library (../pileup_library.lhe.gz: 2000
minimum-bias collisions at 13.6 TeV made once by make_pileup_library.py with PYTHIA, the generator-level
twin of the GEN-SIM library the CMSSW mixing reads). Output is the chapter-7 record followed by the
tables of this stop (../event_PU.lhe): the collisions of every bunch crossing with their vertices, the
particles of the in-time crossing, and the anti-kT jets of signal + in-time pileup with their areas,
the energy density and the subtracted momenta.

What is implemented (the physics of chapter 8)
----------------------------------------------
* The mean number of collisions per bunch crossing, mu = L sigma_inel / (n_b f_rev), from the
  luminosity scenario (Run 3 in 2025, HL-LHC nominal, HL-LHC ultimate) and PYTHIA's sigma_inel = 78.42 mb;
  the 2025 numbers are the header of the release's 2025 profile (2340 colliding bunches, levelled at
  2.23e34 for 9 h of a 12.5 h fill, 80 mb): mu = 66.5 at the levelled peak, a mean of 64.3 over the
  profile. The book mixes at mu = 60, the AVE_60_BX_25ns of the CMSSW companion (the production draws the
  in-time mean from the flat profile Run3_Flat55To75_PoissonOOTPU), the default here.
* The draw of the MixingModule: for every bunch crossing of the window (bx = -12 ... +3 in the Run-3
  production, 25 ns apart) a Poisson number of collisions with mean mu, the library read in order from a
  random position, wrapping at its end, so that no collision is used twice in an event (CMSSW's
  sequential = False, RootEmbeddedFileSequence::readOneRandom; --sequential reads from the first event
  on, as the exercise of the CMSSW companion does).
* The beam spot of 2025 (CMSSW's BetafuncEvtVtxGenerator with the SimBeamSpot payload of the 2025
  global tag, the VtxSmeared of the GEN step; the payload is a fit to early-2024 collisions, which the
  2025 conditions keep unchanged): every collision, the signal included, gets a position and
  a time drawn from a Gaussian of sigma_z = 3.6 cm around z0 = 0.42 cm, a Gaussian in time of
  sigma_z / c = 120 ps, and a transverse Gaussian of 9 um that widens with the beta function away from
  the focus (beta* = 30 cm, the hourglass). The out-of-time collisions are shifted by 25 ns x bx
  (CMSSW's Adjuster). The HL-LHC luminous region of the Phase-2 generator (two 9 cm bunches crossing at
  510 urad, crab-rotated, beta* = 20 cm) is kept as --region hllhc for the open problems.
* The stacking: the signal and the in-time collisions are one final state; the out-of-time collisions
  are recorded with their time offset and left to the detector chapters (at the generator level nothing
  integrates over 25 ns; the calorimeter pulses of chapter 10 will).
* The anti-kT jets of signal + in-time pileup (chapter 7's clustering with FastJet's tiled search, R = 0.4,
  E scheme, pT > 3 GeV, the particles within |eta| < 5 without the neutrinos, as ak4GenJetsNoNu), with the active area of every jet measured with ghosts (area 0.01 each,
  |y| < 5, FastJet's active_area_explicit_ghosts) and every constituent tagged by its collision.
* The median energy density rho two ways: the grid median of CMSSW's fixedGridRhoFastjetAll (FastJet's grid
  of 0.56 x 0.57 cells in |y| < 5) and the jet median of Cacciari and Salam (kT jets of R = 0.4 with ghosts,
  the rapidity range of CMSSW's kT-jet rho), and the area subtraction pT - rho A of every jet (the L1FastJet
  correction of chapter 17 in its generator-level form), compared with the truth: the pT of the signal
  constituents of the jet (the residual of the record), and the pT of its pileup constituents against rho A
  (the residual of the ensemble, scalar sums on both sides).
* The vertices the tracker will see: the collisions with at least four charged particles of pT > 0.5
  GeV within |eta| < 2.5, and how far from the signal vertex each one is in z and in t.
* The pileup profile and its reweighting: the flat profile the Run-3 production mixes with
  (mu_true uniform in 55 ... 75), the 2025 profile of the release (Run3_2025_LHC_Simulation_fill12p5h_
  levelling9h_nb2340_xSec80mb_il2p23_cfi, the distribution of mu over the fills of 2025), one levelled fill
  of its header, and the weight w(mu_true) = P_target(mu_true) / P_sample(mu_true) of the pileup reweighting.

What is left out, on purpose: the premixing bookkeeping (the physics is the same); the four-vector
area subtraction (the scalar one is what chapter 17 uses); the detector's time windows (chapters 9 to
11); the charged-hadron subtraction and PUPPI (chapter 14), which need tracks.

Learning mode (--verbose / -v, -vv)
------------------------------------
  -v   narrates the scenario and the arithmetic of mu, the Poisson draws per bunch crossing, the library
       events read, the luminous-region model and the vertices drawn, the stacking, the clustering of
       our jet with signal and pileup constituents, the two rho estimates, the areas and the subtraction
       of the leading jets, the visible vertices and the profile weights.
  -vv  adds every collision with its vertex and every constituent of our jet.

Examples
--------
  python3 standalone_pileup.py                       # -> ../event_PU.lhe (the book's record: mu = 60, bx = -12 ... +3)
  python3 standalone_pileup.py -v                    # learning mode
  python3 standalone_pileup.py --window 0            # the in-time crossing only
  python3 standalone_pileup.py --window 1 --sequential   # the conditions of the CMSSW companion (PU 60 in bx = -1 ... +1)
  python3 standalone_pileup.py --scenario ultimate   # the HL-LHC crowd: mu = 200 in bx = -3 ... +3, the Phase-2 luminous region
  python3 standalone_pileup.py --repeat 400 --quiet --history standalone_pileup_stats.json
                                                     # draw the crowd again and again on the same jet

All energies in GeV, angles in radians, lengths in cm (vertices) and um (transverse), times in ns.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "GenJet" / "Standalone"))
import standalone_genjets as gj  # noqa: E402  (chapter 7: the record, the clustering, the jet tables)

dec = gj.dec
pt, eta, phi, mass2, rapidity, banner = gj.pt, gj.eta, gj.phi, gj.mass2, gj.rapidity, gj.banner
name, is_charged, is_neutrino = gj.name, gj.is_charged, gj.is_neutrino

# ---------------------------------------------------------------------------
# the accelerator and the scenarios (CMSSW's AVE_<mu>_BX_25ns and the LHC numbers behind them)
# ---------------------------------------------------------------------------
F_REV = 11245.5                   # Hz, the revolution frequency of the LHC (26659 m at c)
BUNCH_SPACING = 25.0              # ns
SIGMA_INEL_MB = 78.42             # mb, PYTHIA 8.315 with CP5 (SigmaTotal:mode = 0: the tune fixes it, at 13.6 as at 14 TeV): what the library was made with
MB_TO_CM2 = 1e-27
SCENARIOS = {                     # instantaneous luminosity [cm^-2 s^-1], colliding bunch pairs, the window of bunch crossings, the rounded mu CMSSW mixes, the beam spot
    "run3": dict(lumi=2.23e34, n_bunch=2340, bunches=(-12, 3), mu=60.0, region="run3", label="Run 3 (2025, levelled)",
                 production="Run3_Flat55To75_PoissonOOTPU"),   # L and n_b: the header of the release's 2025 profile (PROFILE_2025)
    "nominal": dict(lumi=5.0e34, n_bunch=2748, bunches=(-3, 3), mu=140.0, region="hllhc", label="HL-LHC nominal, levelled", production="AVE_140_BX_25ns"),
    "ultimate": dict(lumi=7.5e34, n_bunch=2748, bunches=(-3, 3), mu=200.0, region="hllhc", label="HL-LHC ultimate, levelled (Phase-2 production)",
                     production="AVE_200_BX_25ns"),
}
FLAT_PROFILE = (55, 75)           # the Run-3 production: mu_true of the in-time crossing uniform in 55 ... 75 (mix_Run3_Flat55To75_PoissonOOTPU_cfi)

# the pileup profile of 2025 in CMSSW_15_0_5: SimGeneral/MixingModule/python/Run3_2025_LHC_Simulation_fill12p5h_levelling9h_nb2340_xSec80mb_il2p23_cfi.py
# (the "2025_LHC_Simulation_..." scenario of Configuration/StandardSequences/python/Mixing.py). Its header: "A distribution for Run3 studies
# (2023) derived from LHC simulations. Colliding bunches: 2340, Hadronic x-sec: 80mb, Fill length: 750min, Levelling time: 540min,
# Initial lumi: 2.23e38Hz/m2". probValue[n] is the probability of mu_true = n, n = 0 ... 99 (mean 64.34, r.m.s. 11.70, median 65).
PROFILE_2025 = dict(
    name="Run3_2025_LHC_Simulation_fill12p5h_levelling9h_nb2340_xSec80mb_il2p23", n_bunch=2340, xsec_mb=80.0, lumi=2.23e34,
    fill_hours=12.5, levelled_hours=9.0,
    prob=[0.0000010538773679583783, 0.0000014453031026961327, 0.000001970363090883917, 0.000002670281107593809,
          0.000003597462798567677, 0.000004818033267765735, 0.000006414803175506918, 0.00000849069485055825, 0.000011172650635919335,
          0.000014616032654130512, 0.000019009506082787833, 0.00002458037676190198, 0.00003160032867400924, 0.0000403914780629409,
          0.0000513326296393648, 0.00006486558791741853, 0.00008150134525564022, 0.00010182594025209081, 0.00012650575898492516,
          0.0001562920409839073, 0.00019202435605314403, 0.00023463284180945632, 0.0002851390399338541, 0.0003446552465146764,
          0.0004143824030220098, 0.0004956067031895889, 0.0005896952798974523, 0.0006980915654970719, 0.0008223111862412097,
          0.0009639395493586345, 0.0011246325961596164, 0.001306122503779093, 0.0015102303874063258, 0.0017388882349847804,
          0.0019941723308908393, 0.0022783502088846506, 0.0025939426153526782, 0.002943800947840044, 0.0033311990480591035,
          0.0037599359803347204, 0.004234443471816578, 0.0047598880701177565, 0.005342253951931872, 0.005988388019207414,
          0.0067059849622914496, 0.0075034870576089615, 0.008389872464109042, 0.00937430762932423, 0.01046564500185625,
          0.011671757233613201, 0.012998713670709948, 0.01444982377096003, 0.0160245939526876, 0.017717667226787342,
          0.0195178359692353, 0.0214072340014225, 0.02336082128919122, 0.025346270004274998, 0.02732434240367562,
          0.029249818558649544, 0.03107298698415659, 0.03274165737659939, 0.03420359754458777, 0.035409243087022646,
          0.03631448568932437, 0.036883320614776005, 0.03709013079903057, 0.03692140594496318, 0.03637673898184205,
          0.03546900479726313, 0.0342237001823724, 0.03267750066270309, 0.030876160122826117, 0.028871934631968292,
          0.0267207465479145, 0.024479315695643617, 0.02220247134867907, 0.01994082511453152, 0.017738936250727092,
          0.01563404440428138, 0.01365538752456454, 0.011824071217400022, 0.010153414981995273, 0.00864967347023981,
          0.007313017889518202, 0.0061386628307422315, 0.0051180346517136965, 0.0042398957981439585, 0.0034913616164943813,
          0.0028587691188455234, 0.002328378260551149, 0.001886903867739054, 0.0015218895230552547, 0.0012219433370943896,
          0.0009768600068832721, 0.0007776546470029057, 0.0006165324821007001, 0.0004868155154920938, 0.000382843527372107,
          0.0002998628250636046])

# ---------------------------------------------------------------------------
# the beam spot of 2025: VtxSmearedRealistic_cfi (BetafuncEvtVtxGenerator, readDB = True), the SimBeamSpotObjectsRcd payload
# SimBeamSpot_Realistic25ns13p6TeVEarly2024CollisionVtxSmearingParameters_v1_mc of the global tag 150X_mcRun3_2025_realistic_v1
# (auto:phase1_2025_realistic): a fit to early-2024 collisions, the same tag as in the 2024 global tag 150X_mcRun3_2024_realistic_v2
# ---------------------------------------------------------------------------
C_CM_PER_NS = 29.9792458
BEAMSPOT_2025 = dict(x0=0.0285, y0=-0.0166, z0=0.416, sigma_z=3.6, beta_star=30.0, emittance=5.725e-8, time_offset=0.0,
                     tag="SimBeamSpot_Realistic25ns13p6TeVEarly2024CollisionVtxSmearingParameters_v1_mc")   # cm, cm, cm, cm, cm, cm, ns

# ---------------------------------------------------------------------------
# the luminous region of the HL-LHC: VtxSmearedHLLHC_cfi (HLLHCEvtVtxGenerator), nominal Phase-2 parameters (--region hllhc)
# ---------------------------------------------------------------------------
LUMINOUS = dict(e_proton=7000.0, crossing_angle=510e-6, crab_angle=380e-6, crab_frequency_mhz=400.0,
                beta_star=0.20, emittance_x=2.5e-6, emittance_y=2.05e-6, bunch_length=0.090)
M_PROTON = 0.93827

# ---------------------------------------------------------------------------
# the jets, the ghosts and rho (FastJet and CMSSW defaults)
# ---------------------------------------------------------------------------
R_JET = gj.R_AK4                  # 0.4
JET_PT_MIN = gj.JET_PT_MIN        # 3 GeV
ETA_MAX = 5.0                     # the particles that enter the mixed clustering (the detector's reach, HF included)
GHOST_AREA = 0.01                 # FastJet GhostedAreaSpec default; CMSSW GhostArea = 0.01
GHOST_MAXRAP = 5.0                # CMSSW Ghost_EtaMax = 5.0
GHOST_PT = 1e-18                  # the ghosts of chapter 7 (FastJet uses 1e-100; any value far below every particle will do)
GRID_RAP_MAX, GRID_SPACING = 5.0, 0.55    # fixedGridRhoFastjetAll: GridMedianBackgroundEstimator(5.0, 0.55)
KT_RHO_RAP_MAX = 4.4              # the jet median of kT jets: rapidity range |y| < 4.4 (CMSSW Rho_EtaMax = 4.4)
VISIBLE_NCH, VISIBLE_PT, VISIBLE_ETA = 4, 0.5, 2.5   # a vertex the tracker can reconstruct: >= 4 charged tracks of pT > 0.5 within |eta| < 2.5
MATCH_DR = 0.2                    # the matching of a mixed jet to the chapter-7 jet


# ---------------------------------------------------------------------------
# the library
# ---------------------------------------------------------------------------
class Library:
    """The minimum-bias collisions of make_pileup_library.py: event k -> (code, kind, p4 array (E, px, py, pz), ids)."""

    def __init__(self, path: Path):
        self.path = path
        self.header, self.events = [], []
        opener = gzip.open if path.suffix == ".gz" else open
        cur = None
        with opener(path, "rt") as f:
            for line in f:
                if line.startswith("#"):
                    self.header.append(line.rstrip())
                elif line.startswith("<event"):
                    fld = line.strip("<>\n").split()
                    kv = {fld[i]: fld[i + 1] for i in range(5, len(fld) - 1, 2)}      # nfinal, nch, sumpt, and since 2026-11-06 nmpi, pthat
                    cur = dict(no=int(fld[1]), code=int(fld[3]), kind=fld[4], rows=[], nmpi=int(kv.get("nmpi", 0)), pthat=float(kv.get("pthat", 0.0)))
                elif line.startswith("</event"):
                    ids = np.array([r[0] for r in cur["rows"]], dtype=int)
                    p4 = np.array([[r[4], r[1], r[2], r[3]] for r in cur["rows"]], dtype=float)
                    cur["ids"], cur["p4"] = ids, p4
                    del cur["rows"]
                    self.events.append(cur)
                    cur = None
                elif cur is not None:
                    fld = line.split()
                    cur["rows"].append((int(fld[0]), float(fld[1]), float(fld[2]), float(fld[3]), float(fld[4])))
        for e in self.events:
            e.update(self.describe(e["ids"], e["p4"]))

    @staticmethod
    def describe(ids, p4) -> dict:
        ptv = np.hypot(p4[:, 1], p4[:, 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            etav = np.arcsinh(np.where(ptv > 0, p4[:, 3] / np.where(ptv > 0, ptv, 1.0), 0.0))
        etav = np.where(ptv > 0, etav, np.sign(p4[:, 3]) * 99.0)
        ch = np.array([is_charged(int(i)) for i in ids], dtype=bool)
        nu = np.array([is_neutrino(int(i)) for i in ids], dtype=bool)
        central = ch & (ptv > VISIBLE_PT) & (np.abs(etav) < VISIBLE_ETA)
        inacc = np.abs(etav) < ETA_MAX
        return dict(nfinal=len(ids), nch=int(ch.sum()), n_central=int(central.sum()), sumpt=float(ptv.sum()),
                    sumpt_acc=float(ptv[inacc & ~nu].sum()), pt_max=float(ptv.max()) if len(ptv) else 0.0, pt=ptv, eta=etav, nu=nu, ch=ch,
                    visible=bool(central.sum() >= VISIBLE_NCH))

    def __len__(self):
        return len(self.events)


# ---------------------------------------------------------------------------
# the scenario: mu from the luminosity
# ---------------------------------------------------------------------------
def mu_of(lumi: float, n_bunch: int = SCENARIOS["run3"]["n_bunch"], sigma_mb: float = SIGMA_INEL_MB, f_rev: float = F_REV) -> float:
    """mu = L sigma_inel / (n_b f_rev): inelastic collisions per second over bunch crossings per second."""
    return lumi * sigma_mb * MB_TO_CM2 / (n_bunch * f_rev)


def parse_window(text) -> tuple:
    """'3' -> (-3, 3); '-12,3' -> (-12, 3)."""
    if isinstance(text, (tuple, list)):
        return int(text[0]), int(text[1])
    if "," in str(text):
        a, b = str(text).split(",")
        return int(a), int(b)
    n = int(text)
    return -n, n


# ---------------------------------------------------------------------------
# the beam spot of 2025 (BetafuncEvtVtxGenerator)
# ---------------------------------------------------------------------------
class BeamSpot:
    """CMSSW's BetafuncEvtVtxGenerator with the 2025 payload: the luminous region as the database describes it.

    z is a Gaussian of width SigmaZ around Z0; t an independent Gaussian of the same width in mm/c (sigma_t = sigma_z / c);
    x and y are Gaussians around X0, Y0 of width sqrt(epsilon beta* (1 + ((z - z0)/beta*)^2)) / sqrt(2), the beam size at that z
    (the hourglass) divided by sqrt(2) for the overlap of two beams. Two Gaussian bunches of r.m.s. length sigma_s meeting head-on
    give exactly this, with sigma_z = sigma_s / sqrt(2) and z and t independent; the payload is a fit to early-2024 data (kept for 2025) rather than a
    model of the bunches, whose length and crossing angle it absorbs (the HL-LHC model, LuminousRegion, builds the region from them).
    """

    name = "run3"

    def __init__(self, p=BEAMSPOT_2025):
        self.p = dict(p)
        self.x0, self.y0, self.z0 = p["x0"], p["y0"], p["z0"]
        self.sigma_z = p["sigma_z"]
        self.sigma_t = p["sigma_z"] / C_CM_PER_NS                 # ns
        self.beta = p["beta_star"]
        self.emittance = p["emittance"]
        self.sigma_x0 = self.sigma_y0 = math.sqrt(self.emittance * self.beta) / math.sqrt(2.0)   # cm, at the focus
        self.sigma_s = self.sigma_z * math.sqrt(2.0)              # the bunch length two head-on bunches would need
        self.t0 = p["time_offset"]

    def sigma_x(self, z):
        return self.sigma_x0 * np.sqrt(1 + ((np.asarray(z, dtype=float) - self.z0) / self.beta) ** 2)

    sigma_y = sigma_x

    def density(self, z, t, **unused):
        """The collision density in (z, t) up to a constant: a Gaussian in z around z0 times a Gaussian in t."""
        z, t = np.asarray(z, dtype=float), np.asarray(t, dtype=float)
        return np.exp(-((z - self.z0) ** 2) / (2 * self.sigma_z ** 2) - ((t - self.t0) ** 2) / (2 * self.sigma_t ** 2))

    def moments(self, **unused):
        """(sigma_z [cm], sigma_t [ns]) of the model."""
        return self.sigma_z, self.sigma_t

    def draw(self, rng, n: int, **unused):
        """n vertices (x [um], y [um], z [cm], t [ns]) as BetafuncEvtVtxGenerator::vertexShift draws them."""
        zz = self.z0 + rng.normal(0.0, self.sigma_z, n)
        tt = self.t0 + rng.normal(0.0, self.sigma_t, n)
        sx = self.sigma_x(zz)
        xx = self.x0 + rng.normal(0.0, 1.0, n) * sx
        yy = self.y0 + rng.normal(0.0, 1.0, n) * sx
        return np.column_stack([xx * 1e4, yy * 1e4, zz, tt])


# ---------------------------------------------------------------------------
# the luminous region of the HL-LHC (HLLHCEvtVtxGenerator): kept for --region hllhc and the open problems
# ---------------------------------------------------------------------------
class LuminousRegion:
    """The overlap of two crossing, crabbed, focused Gaussian bunches: the density of collisions in (x, y, z, t).

    Bunch 1 moves along +z, bunch 2 along -z; at lab time t their centres are at z = +ct and z = -ct, and a
    particle at longitudinal offset s from its bunch centre sits at z = ct + s (bunch 1) or z = -ct + s
    (bunch 2). In the crossing plane the centre lines are tilted by +-alpha/2 and the crab cavities displace
    the head and the tail of each bunch in the opposite sense, x_crab = -+ (alpha_c/2) sin(k s)/k with
    k = omega_crab / c, so that the residual tilt at the collision point is (alpha - alpha_c)/2. The transverse
    width grows away from the focus (the hourglass), sigma(z) = sigma_0 sqrt(1 + z^2/beta*^2), with
    sigma_0 = sqrt(beta* epsilon_n / gamma). Integrating the product of the two bunch densities over x and y
    leaves the density in (z, t) that the method density() returns; x and y follow as Gaussians around the
    mid-point of the two centre lines.
    """

    name = "hllhc"

    def __init__(self, p=LUMINOUS):
        self.p = dict(p)
        self.z0, self.t0 = 0.0, 0.0
        gamma = p["e_proton"] / M_PROTON
        self.sigma_x0 = math.sqrt(p["beta_star"] * p["emittance_x"] / gamma) * 100.0     # cm
        self.sigma_y0 = math.sqrt(p["beta_star"] * p["emittance_y"] / gamma) * 100.0     # cm
        self.sigma_s = p["bunch_length"] * 100.0                                         # cm
        self.beta = p["beta_star"] * 100.0                                               # cm
        self.k_crab = 2 * math.pi * p["crab_frequency_mhz"] * 1e6 / (C_CM_PER_NS * 1e9) # 1/cm
        self.alpha, self.alpha_c = p["crossing_angle"], p["crab_angle"]

    def sigma_x(self, z):
        return self.sigma_x0 * np.sqrt(1 + (z / self.beta) ** 2)

    def sigma_y(self, z):
        return self.sigma_y0 * np.sqrt(1 + (z / self.beta) ** 2)

    def f_crab(self, s):
        return np.sin(self.k_crab * s) / self.k_crab

    def centres(self, z, t):
        """x of the two centre lines at (z, t), in cm."""
        s1, s2 = z - C_CM_PER_NS * t, z + C_CM_PER_NS * t
        x1 = 0.5 * self.alpha * z - 0.5 * self.alpha_c * self.f_crab(s1)
        x2 = -0.5 * self.alpha * z + 0.5 * self.alpha_c * self.f_crab(s2)
        return x1, x2

    def density(self, z, t, crossing=True, crab=True, hourglass=True):
        """The collision density in (z, t) up to a constant; the switches turn the three effects off for the theory plot."""
        z, t = np.asarray(z, dtype=float), np.asarray(t, dtype=float)
        s1, s2 = z - C_CM_PER_NS * t, z + C_CM_PER_NS * t
        d = np.exp(-(s1 * s1 + s2 * s2) / (2 * self.sigma_s ** 2))
        sx = self.sigma_x(z) if hourglass else self.sigma_x0
        sy = self.sigma_y(z) if hourglass else self.sigma_y0
        if crossing:
            x1 = 0.5 * self.alpha * z - (0.5 * self.alpha_c * self.f_crab(s1) if crab else 0.0)
            x2 = -0.5 * self.alpha * z + (0.5 * self.alpha_c * self.f_crab(s2) if crab else 0.0)
            d = d * np.exp(-((x1 - x2) ** 2) / (4 * sx * sx))
        return d / (sx * sy)

    def moments(self, crossing=True, crab=True, hourglass=True, zmax=40.0, tmax=1.5, n=801):
        """(sigma_z [cm], sigma_t [ns]) of the density on a grid."""
        z = np.linspace(-zmax, zmax, n)
        t = np.linspace(-tmax, tmax, n)
        Z, T = np.meshgrid(z, t, indexing="ij")
        d = self.density(Z, T, crossing, crab, hourglass)
        d /= d.sum()
        return float(np.sqrt((d * Z * Z).sum())), float(np.sqrt((d * T * T).sum()))

    def draw(self, rng, n: int, zmax=40.0, tmax=1.5, ngrid=801):
        """n vertices (x [um], y [um], z [cm], t [ns]) from the density: a cell of a fine (z, t) grid by its weight, jittered within the cell."""
        z = np.linspace(-zmax, zmax, ngrid)
        t = np.linspace(-tmax, tmax, ngrid)
        Z, T = np.meshgrid(z, t, indexing="ij")
        w = self.density(Z, T).ravel()
        w /= w.sum()
        cells = rng.choice(w.size, size=n, p=w)
        dz, dt = z[1] - z[0], t[1] - t[0]
        zz = Z.ravel()[cells] + rng.uniform(-0.5, 0.5, n) * dz
        tt = T.ravel()[cells] + rng.uniform(-0.5, 0.5, n) * dt
        x1, x2 = self.centres(zz, tt)
        xx = 0.5 * (x1 + x2) + rng.normal(0.0, 1.0, n) * self.sigma_x(zz) / math.sqrt(2)
        yy = rng.normal(0.0, 1.0, n) * self.sigma_y(zz) / math.sqrt(2)
        return np.column_stack([xx * 1e4, yy * 1e4, zz, tt])


def region_of(name: str):
    """The vertex generator of a scenario: 'run3' (the 2025 beam spot) or 'hllhc' (the Phase-2 luminous region)."""
    return BeamSpot() if name == "run3" else LuminousRegion()


# ---------------------------------------------------------------------------
# the pileup profile of a fill, and the reweighting
# ---------------------------------------------------------------------------
def fill_profile(mu0: float, hours: float, levelled_hours: float, tau_hours: float, n_steps: int = 2000):
    """mu over a fill: mu(t) = mu0 while levelled, then mu0 exp(-(t - t_lev)/tau), mu being proportional to the
    luminosity; each moment weighted by its luminosity (every collision counts once). Returns (mu values, weights)."""
    t = np.linspace(0.0, hours, n_steps)
    mu = np.where(t < levelled_hours, mu0, mu0 * np.exp(-(t - levelled_hours) / tau_hours))
    return mu, mu / mu.sum()


def profile_histogram(mu, w, edges):
    h, _ = np.histogram(mu, bins=edges, weights=w)
    return h / max(h.sum(), 1e-300)


def year_profile(mu_levels, hours: float, levelled_hours: float, tau_hours: float, edges):
    """A year of fills, each levelled at one of mu_levels for levelled_hours and then decaying: the luminosity-weighted
    distribution of mu over the year, as a histogram on `edges` (a model of a Run-3 profile, not the measured one)."""
    h = np.zeros(len(edges) - 1)
    for mu0 in mu_levels:
        mu, w = fill_profile(mu0, hours, levelled_hours, tau_hours)
        h += np.histogram(mu, bins=edges, weights=w)[0]
    return h / max(h.sum(), 1e-300)


def release_profile(edges, prof=PROFILE_2025):
    """The pileup profile of a CMSSW scenario (type probFunction): probValue[n] for mu_true = n, as a histogram on `edges`
    (bin [n, n + 1) holds probValue[n], as the flat profile below)."""
    c = 0.5 * (edges[1:] + edges[:-1])
    p = np.asarray(prof["prob"], dtype=float)
    idx = np.floor(c).astype(int)
    h = np.where((idx >= 0) & (idx < len(p)), p[np.clip(idx, 0, len(p) - 1)], 0.0)
    return h / max(h.sum(), 1e-300)


def profile_stats(prof=PROFILE_2025) -> dict:
    """Mean, r.m.s., median and the shares below, inside and above the flat band of the production, of a probFunction profile."""
    p = np.asarray(prof["prob"], dtype=float)
    p = p / p.sum()
    n = np.arange(len(p))
    mean = float((n * p).sum())
    cum = np.cumsum(p)
    return dict(mean=mean, rms=float(math.sqrt(((n - mean) ** 2 * p).sum())), median=int(n[np.searchsorted(cum, 0.5)]), peak=int(n[p.argmax()]),
                q16=int(n[np.searchsorted(cum, 0.16)]), q84=int(n[np.searchsorted(cum, 0.84)]),
                below=float(p[:FLAT_PROFILE[0]].sum()), inside=float(p[FLAT_PROFILE[0]:FLAT_PROFILE[1] + 1].sum()), above=float(p[FLAT_PROFILE[1] + 1:].sum()))


def flat_profile(lo: int, hi: int, edges):
    """The probFunction of the Run-3 production (mix_Run3_Flat55To75_PoissonOOTPU_cfi): mu_true uniform over the integers lo ... hi."""
    c = 0.5 * (edges[1:] + edges[:-1])
    h = ((c >= lo) & (c < hi + 1)).astype(float)
    return h / max(h.sum(), 1e-300)


def reweight(mu_true: float, p_target, p_sample, edges) -> float:
    """w(mu) = P_target(mu) / P_sample(mu), the per-event weight of the pileup reweighting."""
    k = int(np.clip(np.searchsorted(edges, mu_true, side="right") - 1, 0, len(p_target) - 1))
    return float(p_target[k] / p_sample[k]) if p_sample[k] > 0 else 0.0


# ---------------------------------------------------------------------------
# the median energy density and the areas
# ---------------------------------------------------------------------------
def grid_rho(p4, rap_max=GRID_RAP_MAX, spacing=GRID_SPACING, use_rapidity=True):
    """fixedGridRhoFastjetAll: CMSSW's FixedGridRhoProducerFastjet wraps FastJet's GridMedianBackgroundEstimator(5.0, 0.55),
    the median over the cells of a grid in (y, phi) of the summed pT per unit area: rapidity, phi from 0 to 2 pi, and as
    many cells as come closest to the requested spacing (18 x 11 here). use_rapidity=False fills the grid in eta instead
    (the older FixedGridRhoProducer, fixedGridRhoAll)."""
    ny = max(1, int(2 * rap_max / spacing + 0.5))          # FastJet's RectangularGrid: the nearest integer, not the floor
    nphi = max(1, int(2 * math.pi / spacing + 0.5))
    dy, dphi = 2 * rap_max / ny, 2 * math.pi / nphi
    ptv = np.hypot(p4[:, 1], p4[:, 2])
    yv = np.array([rapidity(p) if use_rapidity else eta(p) for p in p4])
    phiv = np.arctan2(p4[:, 2], p4[:, 1]) % (2 * math.pi)
    sel = (np.abs(yv) < rap_max) & (ptv > 0)
    iy = np.floor((yv[sel] + rap_max) / dy).astype(int)
    ip = np.floor(phiv[sel] / dphi).astype(int) % nphi
    cells = np.zeros((ny, nphi))
    np.add.at(cells, (iy, ip), ptv[sel])
    dens = cells / (dy * dphi)
    return float(np.median(dens)), dens, (dy, dphi)


def ghost_grid(rng, area=GHOST_AREA, maxrap=GHOST_MAXRAP):
    """FastJet's GhostedAreaSpec: a grid of cells of `area` in (y, phi) over |y| < maxrap, one ghost per cell,
    scattered uniformly within its cell (grid_scatter = 1), with pT = GHOST_PT (pt_scatter ignored). Returns (E, px, py, pz) rows."""
    d = math.sqrt(area)
    ny, nphi = int(math.ceil(2 * maxrap / d)), int(math.ceil(2 * math.pi / d))
    dy, dphi = 2 * maxrap / ny, 2 * math.pi / nphi
    y = -maxrap + (np.arange(ny) + 0.5) * dy
    f = -math.pi + (np.arange(nphi) + 0.5) * dphi
    Y, F = np.meshgrid(y, f, indexing="ij")
    Y = Y.ravel() + rng.uniform(-0.5, 0.5, Y.size) * dy
    F = F.ravel() + rng.uniform(-0.5, 0.5, F.size) * dphi
    ptg = GHOST_PT
    p4 = np.column_stack([ptg * np.cosh(Y), ptg * np.cos(F), ptg * np.sin(F), ptg * np.sinh(Y)])
    return p4, dy * dphi


class TiledClustering(gj.Clustering):
    """Chapter 7's sequential recombination with FastJet's tiled search (the N2Tiled strategy).

    A pair with Delta R > R is never merged: d_ij = min(pT_i^2p, pT_j^2p) Delta R^2 / R^2 is then larger than the
    smaller of the two beam distances (that of the softer for kT, of the harder for anti-kT), and that one becomes a jet first. So the nearest neighbour of a pseudojet lies within
    R of it, and the (y, phi) plane is tiled in cells of side R: a nearest neighbour is searched in the 3 x 3 tiles
    around the pseudojet, and only the pseudojets of those tiles need a new neighbour when it merges. The result is
    the same as the plain search of chapter 7, step by step; the cost is what makes thirty thousand particles and
    ghosts affordable (anti-kT makes every ghost's nearest neighbour a hard particle, and a plain search had to
    revisit all of them at every merge).
    """

    def _run(self):
        R, n0 = self.R, self.n0
        cap = len(self.pt2)
        nn = -np.ones(cap, dtype=int)
        nnd = np.full(cap, np.inf)
        diB = np.full(cap, np.inf)
        alive = self.alive
        ymin, ymax = float(self.y[:n0].min()) - R, float(self.y[:n0].max()) + R
        ny = max(1, int(math.ceil((ymax - ymin) / R)))
        nphi = max(3, int(math.floor(2 * math.pi / R)))
        dphi = 2 * math.pi / nphi
        tile_of = np.zeros((cap, 2), dtype=int)
        tiles = {}

        def tile(i):
            iy = min(max(int((self.y[i] - ymin) / R), 0), ny - 1)
            ip = int((self.phi[i] + math.pi) / dphi) % nphi
            return iy, ip

        def add(i):
            t = tile(i)
            tile_of[i] = t
            tiles.setdefault(t, set()).add(i)

        def remove(i):
            tiles[tuple(tile_of[i])].discard(i)

        def around(i):
            iy, ip = tile_of[i]
            out = []
            for a in (iy - 1, iy, iy + 1):
                if 0 <= a < ny:
                    for b in (ip - 1, ip, ip + 1):
                        out.extend(tiles.get((a, b % nphi), ()))
            return out

        for i in range(n0):
            add(i)
            diB[i] = self.pt2[i] ** self.p

        def recompute(i):
            idx = np.array([m for m in around(i) if m != i], dtype=int)
            if idx.size == 0:
                nn[i], nnd[i] = -1, np.inf
                return
            d = self._row(i, idx)
            k = int(np.argmin(d))
            nn[i], nnd[i] = idx[k], d[k]

        for i in range(n0):
            recompute(i)
        alive_list = set(range(n0))
        while alive_list:
            idx = np.fromiter(alive_list, dtype=int)
            best = np.minimum(nnd[idx], diB[idx])
            i = int(idx[int(np.argmin(best))])
            if diB[i] <= nnd[i]:
                alive[i] = False
                alive_list.discard(i)
                self.is_jet[i] = True
                self.history.append(("iB", i, 0, float(diB[i]), i))
                if self.log:
                    self.log(2, f"    step {len(self.history):4d}: pseudojet {i} is a jet, d_iB = {diB[i]:.4g} (pT = {math.sqrt(self.pt2[i]):.3f})")
                touched = [m for m in around(i) if alive[m] and nn[m] == i]
                remove(i)
                for m in touched:
                    recompute(m)
            else:
                j = int(nn[i])
                k = self.size
                self.size += 1
                self.p4[k] = self.p4[i] + self.p4[j]
                self._kin(k)
                self.parents[k] = (i, j)
                alive[i] = alive[j] = False
                alive[k] = True
                alive_list.discard(i)
                alive_list.discard(j)
                alive_list.add(k)
                diB[k] = self.pt2[k] ** self.p
                self.history.append(("ij", i, j, float(nnd[i]), k))
                if self.log:
                    self.log(2, f"    step {len(self.history):4d}: {i} + {j} -> {k}, d_ij = {nnd[i]:.4g}, dR = {math.sqrt(gj.dr2(self.y[i], self.phi[i], self.y[j], self.phi[j])):.3f}, "
                                f"pT {math.sqrt(self.pt2[i]):.3f} + {math.sqrt(self.pt2[j]):.3f} -> {math.sqrt(self.pt2[k]):.3f}")
                touched = set(m for m in around(i) + around(j) if alive[m] and (nn[m] == i or nn[m] == j))
                remove(i)
                remove(j)
                add(k)
                for m in touched:
                    recompute(m)
                recompute(k)
                near = np.array([m for m in around(k) if m != k and alive[m]], dtype=int)
                if near.size:
                    d = self._row(k, near)
                    closer = d < nnd[near]
                    nn[near[closer]] = k
                    nnd[near[closer]] = d[closer]


class MixedJets:
    """The jets of a list of tagged particles (tag = collision index, 0 the signal) with ghosts for the areas."""

    def __init__(self, p4, tags, R=R_JET, power=-1, ptmin=JET_PT_MIN, rng=None, ghosts=True, log=None):
        self.R, self.ptmin = R, ptmin
        self.n_real = len(p4)
        self.tags = np.asarray(tags)
        if ghosts:
            g, self.ghost_area = ghost_grid(rng if rng is not None else np.random.default_rng(1))
            allp4 = np.vstack([p4, g])
        else:
            allp4, self.ghost_area = np.asarray(p4), 0.0
        self.n_ghosts = len(allp4) - self.n_real
        t0 = time.time()
        self.cl = TiledClustering(allp4, R, power, log=log)
        self.seconds = time.time() - t0
        self.jets = []
        self.jet_of = {}
        for rank, j in enumerate(self.cl.jets(ptmin)):
            c = self.cl.constituents(j)
            real = [i for i in c if i < self.n_real]
            if not real and ptmin > 0:
                continue                        # a pure-ghost jet: no particle in it (kept, as a zero, when ptmin = 0: the kT jets of the median)
            ng = len(c) - len(real)
            p = allp4[real].sum(axis=0) if real else allp4[c].sum(axis=0)
            info = dict(rank=rank, index=j, const=real, n=len(real), n_ghosts=ng, area=ng * self.ghost_area,
                        pt=pt(p) if real else 0.0, y=rapidity(p), eta=eta(p), phi=phi(p), m=math.sqrt(max(mass2(p), 0.0)) if real else 0.0, e=float(p[0]), p4=p)
            tg = self.tags[real] if real else np.zeros(0, dtype=int)
            ptc = np.hypot(allp4[real, 1], allp4[real, 2]) if real else np.zeros(0)
            info["pt_signal"] = float(ptc[tg == 0].sum())
            info["pt_pileup"] = float(ptc[tg != 0].sum())
            info["n_signal"] = int((tg == 0).sum())
            info["n_pileup"] = int((tg != 0).sum())
            info["collisions"] = sorted(set(int(t) for t in tg if t != 0))
            for i in real:
                self.jet_of[i] = rank
            self.jets.append(info)
        self.jets = [dict(j, rank=r) for r, j in enumerate(sorted(self.jets, key=lambda j: -j["pt"]))]
        self.jet_of = {i: j["rank"] for j in self.jets for i in j["const"]}

    def matched(self, y0, phi0, maxdr=MATCH_DR):
        """The mixed jet nearest to (y0, phi0), in rapidity: the plane the clustering lives in. In pseudorapidity a jet
        that only gained mass (14.8 -> 45 GeV for ours) would seem to have moved, eta - y growing with m/pT."""
        best = None
        for j in self.jets:
            d = math.sqrt(gj.dr2(j["y"], j["phi"], y0, phi0))
            if d < maxdr and (best is None or d < best[0]):
                best = (d, j)
        return best


def jet_median_rho(jets, rap_max=KT_RHO_RAP_MAX):
    """Cacciari-Salam: the median of pT/A over the (kT) jets with |y| < rap_max, the pure-ghost jets counted as zero
    (FastJet's JetMedianBackgroundEstimator with explicit ghosts and use_area_4vector off)."""
    vals = [j["pt"] / j["area"] for j in jets if abs(j["y"]) < rap_max and j["area"] > 0]
    return float(np.median(vals)) if vals else 0.0, vals


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------
class Stage:
    def __init__(self, ev, genjets, genjets_of, lib: Library, args, verbose=0, rng=None, log=None, full=True):
        self.ev, self.genjets, self.genjets_of, self.lib, self.args, self.verbose = ev, genjets, genjets_of, lib, args, verbose
        self.rng = rng if rng is not None else np.random.default_rng(args.seed)
        self._log = log
        self.t0 = time.time()
        self.scenario()
        self.draw_crowd()
        self.vertices()
        self.stack()
        if full:
            self.cluster()
            self.density()
            self.subtract()
        else:
            self.quick_density()
        self.visible_vertices()
        self.profile()

    def log(self, level, text=""):
        if self.verbose >= level:
            if self._log:
                self._log(text)
            else:
                print(text)

    # -- 1 the scenario --------------------------------------------------------------------------
    def scenario(self):
        a = self.args
        sc = SCENARIOS[a.scenario]
        self.sc = sc
        self.n_bunch = sc["n_bunch"]
        self.mu_formula = mu_of(sc["lumi"], self.n_bunch)
        self.mu = a.mu if a.mu is not None else sc["mu"]
        self.bunches = parse_window(a.window) if a.window is not None else tuple(sc["bunches"])
        self.window = max(abs(self.bunches[0]), abs(self.bunches[1]))
        self.bxs = list(range(self.bunches[0], self.bunches[1] + 1))
        self.region_name = a.region or sc["region"]
        self.log(1, banner("1  the scenario: how many collisions per bunch crossing"))
        self.log(1, f"  {sc['label']}: L = {sc['lumi']:.2e} cm^-2 s^-1, sigma_inel = {SIGMA_INEL_MB} mb (PYTHIA, CP5, the tune fixes it),")
        self.log(1, f"  n_b = {self.n_bunch} colliding bunches, f_rev = {F_REV:.1f} Hz: {sc['lumi'] * SIGMA_INEL_MB * MB_TO_CM2:.3e} collisions/s over "
                    f"{self.n_bunch * F_REV:.3e} crossings/s")
        self.log(1, f"  mu = L sigma / (n_b f_rev) = {self.mu_formula:.1f} ({mu_of(sc['lumi'], self.n_bunch, 80.0):.1f} with the 80 mb of the bookkeeping); "
                    f"mixed at mu = {self.mu:g} (CMSSW AVE_{self.mu:g}_BX_{BUNCH_SPACING:g}ns, as the companion), window bx = {self.bxs[0]:+d} ... {self.bxs[-1]:+d}, "
                    f"{len(self.bxs) * self.mu:g} collisions expected per signal event")
        if sc.get("production", "").startswith("Run3_Flat"):
            self.log(1, f"  the Run-3 production mixes {sc['production']}: mu_true of the in-time crossing drawn flat in {FLAT_PROFILE[0]} ... {FLAT_PROFILE[1]} "
                        f"(mean {0.5 * (FLAT_PROFILE[0] + FLAT_PROFILE[1]):g}), Poisson draws of the same mean in the other {len(self.bxs) - 1} crossings, "
                        f"about {len(self.bxs) * 0.5 * (FLAT_PROFILE[0] + FLAT_PROFILE[1]):.0f} collisions per signal event; this record fixes mu_true = {self.mu:g}")

    # -- 2 the draw -------------------------------------------------------------------------------
    def draw_crowd(self):
        a = self.args
        self.n_bx = {bx: int(self.rng.poisson(self.mu)) for bx in self.bxs}
        n_total = sum(self.n_bx.values())
        # CMSSW's RootEmbeddedFileSequence::readOneRandom opens the file at a random entry and reads on from
        # there in order, wrapping at the end: no collision is used twice until the whole file has been read.
        # Drawing every position at random instead would stack the same collision twice in most crossings
        # of a library of this size.
        self.lib_start = 0 if a.sequential else int(self.rng.integers(0, len(self.lib)))
        idx = (self.lib_start + np.arange(n_total)) % len(self.lib)
        self.collisions = []            # dicts: k (1..), bx, event (library index), lib event dict
        k = 0
        pos = 0
        for bx in self.bxs:
            for j in range(self.n_bx[bx]):
                k += 1
                e = self.lib.events[int(idx[pos])]
                self.collisions.append(dict(k=k, bx=bx, j=j + 1, lib=int(idx[pos]) + 1, ev=e))
                pos += 1
        self.log(1, banner("2  the draw of the MixingModule: Poisson(mu) collisions per bunch crossing"))
        self.log(1, "  " + ", ".join(f"bx {bx:+d}: {n}" for bx, n in self.n_bx.items()) + f"; {n_total} collisions for an expectation of {len(self.bxs) * self.mu:g} "
                    f"(Poisson: {self.mu:g} +- {math.sqrt(self.mu):.1f} per crossing)")
        self.log(1, f"  library: {len(self.lib)} events, read in order from {'its first event' if a.sequential else 'a random position (event ' + str(self.lib_start + 1) + '), wrapping at the end'}; "
                    f"{len(set(c['lib'] for c in self.collisions))} distinct events used")
        kinds = {}
        for c in self.collisions:
            kinds[c["ev"]["kind"]] = kinds.get(c["ev"]["kind"], 0) + 1
        self.log(1, "  process types: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) + f" (library shares: {self.lib_shares()})")
        self.log(2, "  the collisions (k, bx, library event, type, final particles, central charged):")
        for c in self.collisions:
            self.log(2, f"    {c['k']:5d}  bx {c['bx']:+d}  event {c['lib']:5d}  {c['ev']['kind']}  {c['ev']['nfinal']:4d}  {c['ev']['n_central']:3d}")

    def lib_shares(self) -> str:
        kinds = {}
        for e in self.lib.events:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        return ", ".join(f"{k} {100 * v / len(self.lib):.1f}%" for k, v in sorted(kinds.items()))

    # -- 3 the luminous region -----------------------------------------------------------------------
    def vertices(self):
        self.region = region_of(self.region_name)
        v = self.region.draw(self.rng, len(self.collisions) + 1)
        self.signal_vertex = v[0]
        for c, row in zip(self.collisions, v[1:]):
            c["vertex"] = row.copy()
            c["vertex"][3] += c["bx"] * BUNCH_SPACING
        r = self.region
        self.log(1, banner("3  the beam spot: where and when each collision happens"))
        if r.name == "run3":
            self.log(1, f"  BetafuncEvtVtxGenerator with the payload of the 2025 global tag ({r.p['tag']}, fitted to early-2024 data):")
            self.log(1, f"  centre ({r.x0:.4f}, {r.y0:.4f}, {r.z0:.3f}) cm, sigma_z = {r.sigma_z:.1f} cm, beta* = {r.beta:.0f} cm, emittance {r.emittance:.3e} cm: "
                        f"sigma_xy = {r.sigma_x0 * 1e4:.1f} um at the focus, x sqrt(1 + ((z - z0)/beta*)^2) away from it (the hourglass)")
            m = dict(model=r.moments())
            self.region_moments = m
            self.log(1, f"  sigma_z, sigma_t of the model: {m['model'][0]:.2f} cm, {m['model'][1] * 1e3:.0f} ps (sigma_t = sigma_z / c, z and t independent: what two head-on "
                        f"Gaussian bunches of {r.sigma_s:.1f} cm would give; the payload is a fit to the data, the crossing angle folded in)")
        else:
            self.log(1, f"  HLLHCEvtVtxGenerator: two bunches of sigma_s = {r.sigma_s:.1f} cm, crossing at {r.alpha * 1e6:.0f} urad, crabbed by {r.alpha_c * 1e6:.0f} urad "
                        f"({r.p['crab_frequency_mhz']:.0f} MHz), beta* = {r.beta:.0f} cm, sigma_x0 = {r.sigma_x0 * 1e4:.1f} um, sigma_y0 = {r.sigma_y0 * 1e4:.1f} um")
            m = dict(headon=r.moments(False, False, False), crossing=r.moments(True, False, False), crab=r.moments(True, True, False), hourglass=r.moments(True, True, True))
            m["model"] = m["hourglass"]
            self.region_moments = m
            self.log(1, f"  sigma_z, sigma_t of the model: head-on {m['headon'][0]:.2f} cm, {m['headon'][1] * 1e3:.0f} ps (sigma_s/sqrt2); + crossing angle "
                        f"{m['crossing'][0]:.2f} cm; + crab cavities {m['crab'][0]:.2f} cm; + hourglass {m['hourglass'][0]:.2f} cm, {m['hourglass'][1] * 1e3:.0f} ps")
        vz = np.array([c["vertex"][2] for c in self.collisions])
        vt = np.array([c["vertex"][3] - c["bx"] * BUNCH_SPACING for c in self.collisions])
        vx = np.array([c["vertex"][0] for c in self.collisions])
        vy = np.array([c["vertex"][1] for c in self.collisions])
        self.vertex_rms = dict(x=float(vx.std()), y=float(vy.std()), z=float(vz.std()), t=float(vt.std()), z_mean=float(vz.mean()))
        self.log(1, f"  drawn: {len(self.collisions)} vertices with r.m.s. x {vx.std():.1f} um, y {vy.std():.1f} um, z {vz.std():.2f} cm (mean {vz.mean():.2f}), t {vt.std() * 1e3:.0f} ps; "
                    f"z from {vz.min():.1f} to {vz.max():.1f} cm")
        s = self.signal_vertex
        self.log(1, f"  the signal vertex: x = {s[0]:.1f} um, y = {s[1]:.1f} um, z = {s[2]:.2f} cm, t = {s[3] * 1e3:.0f} ps; the out-of-time crossings are shifted by "
                    f"{BUNCH_SPACING:g} ns x bx (the CMSSW signal event of the companion sits at z = 3.11 cm, t = +121 ps)")

    # -- 4 the stacking -------------------------------------------------------------------------------
    def stack(self):
        ev = self.ev
        self.signal_rows = [i for i in ev.final_indices() if abs(ev.p[i].id) not in gj.IGNORE_IDS and not is_neutrino(ev.p[i].id)]
        sig_p4 = np.array([ev.p[i].p for i in self.signal_rows])
        sig_eta = np.array([eta(p) for p in sig_p4])
        keep = np.abs(sig_eta) < ETA_MAX
        self.input_p4 = [sig_p4[keep]]
        self.input_tags = [np.zeros(int(keep.sum()), dtype=int)]
        self.input_rows = [np.array(self.signal_rows)[keep]]
        self.n_signal_in = int(keep.sum())
        self.in_time = [c for c in self.collisions if c["bx"] == 0]
        n_pu, n_pu_in = 0, 0
        for c in self.in_time:
            e = c["ev"]
            sel = (np.abs(e["eta"]) < ETA_MAX) & ~e["nu"]
            n_pu += e["nfinal"]
            n_pu_in += int(sel.sum())
            self.input_p4.append(e["p4"][sel])
            self.input_tags.append(np.full(int(sel.sum()), c["k"], dtype=int))
            self.input_rows.append(np.flatnonzero(sel))
        self.input_p4 = np.vstack(self.input_p4) if self.input_p4 else np.zeros((0, 4))
        self.input_tags = np.concatenate(self.input_tags)
        self.input_rows = np.concatenate(self.input_rows)
        self.n_pu_particles, self.n_pu_in = n_pu, n_pu_in
        per_bx = {bx: sum(c["ev"]["nfinal"] for c in self.collisions if c["bx"] == bx) for bx in self.bxs}
        self.log(1, banner("4  the stacking: one final state out of the signal and the in-time crowd"))
        self.log(1, f"  the signal: {len(self.signal_rows)} final particles without the neutrinos, {self.n_signal_in} within |eta| < {ETA_MAX:g}")
        self.log(1, f"  in-time (bx = 0): {len(self.in_time)} collisions, {n_pu} final particles, {n_pu_in} within |eta| < {ETA_MAX:g} and not neutrinos; "
                    f"the mixed final state has {len(self.input_p4)} particles")
        self.log(1, "  particles per crossing: " + ", ".join(f"bx {bx:+d}: {n}" for bx, n in per_bx.items()) + "; the out-of-time ones are recorded with their time offset and not clustered")
        central = sum(c["ev"]["n_central"] for c in self.in_time)
        self.log(1, f"  the in-time crowd in the tracker (|eta| < {VISIBLE_ETA:g}, pT > {VISIBLE_PT:g}): {central} charged particles against "
                    f"{sum(1 for i in self.signal_rows if is_charged(ev.p[i].id) and pt(ev.p[i].p) > VISIBLE_PT and abs(eta(ev.p[i].p)) < VISIBLE_ETA)} of the signal")

    # -- 5 the clustering --------------------------------------------------------------------------------
    def cluster(self):
        a = self.args
        self.log(1, banner("5  anti-kT on the mixed final state, with ghosts for the areas"))
        self.mixed = MixedJets(self.input_p4, self.input_tags, R=a.radius, ptmin=a.ptmin, rng=np.random.default_rng(a.seed + 1), ghosts=not a.no_area)
        m = self.mixed
        self.log(1, f"  {m.n_real} particles + {m.n_ghosts} ghosts of area {m.ghost_area:.4f} (|y| < {GHOST_MAXRAP:g}): {len(m.cl.history)} steps in {m.seconds:.0f} s; "
                    f"{len(m.jets)} jets above {a.ptmin:g} GeV, {sum(1 for j in m.jets if j['pt'] > 30)} above 30")
        # our jet: the chapter-7 jet [0] of ak4 (the b quark's), matched within MATCH_DR
        self.ref = {key: self.genjets["ak4"][k] for key, k in (("b", 0), ("bbar", 1))}
        self.ours = {}
        for key, ref in self.ref.items():
            best = m.matched(ref["y"], ref["phi"])
            self.ours[key] = best
            if best is None:
                self.log(1, f"  the {key} jet: no mixed jet within {MATCH_DR} of the chapter-7 jet [{ref['rank']}]")
                continue
            d, j = best
            self.log(1, f"  the {key} jet: chapter-7 jet [{ref['rank']}] {ref['pt']:.2f} GeV, {ref['n']} particles -> mixed jet [{j['rank']}] {j['pt']:.2f} GeV, {j['n']} particles "
                        f"({j['n_signal']} signal, {j['n_pileup']} pileup from {len(j['collisions'])} collisions), dR = {d:.3f}, area {j['area']:.3f} "
                        f"(pi R^2 = {math.pi * a.radius ** 2:.3f}); signal pT inside {j['pt_signal']:.2f}, pileup pT inside {j['pt_pileup']:.2f}")
            if key == "b":
                # which signal particles of the chapter-7 jet are in the mixed jet
                ref_rows = set(self.ref_rows("ak4", ref["rank"]))
                in_rows = set(int(self.input_rows[i]) for i in j["const"] if self.input_tags[i] == 0)
                lost, gained = ref_rows - in_rows, in_rows - ref_rows
                self.log(1, f"    of the {len(ref_rows)} particles of the photograph {len(ref_rows & in_rows)} are in the mixed jet, {len(lost)} fell out, {len(gained)} signal particles joined"
                            + (f" (lost: {', '.join(name(self.ev.p[r].id) + '[' + str(r) + '] ' + format(pt(self.ev.p[r].p), '.2f') for r in sorted(lost))})" if lost else ""))
                self.log(2, "    the constituents of the mixed jet (tag 0 = signal, else the collision k; pT, dR to the axis):")
                for i in sorted(j["const"], key=lambda i: -pt(self.input_p4[i])):
                    p = self.input_p4[i]
                    tg = int(self.input_tags[i])
                    lab = name(self.ev.p[int(self.input_rows[i])].id) + f"[{int(self.input_rows[i])}]" if tg == 0 else name(int(self.in_time[tg - self.in_time[0]['k']]['ev']['ids'][int(self.input_rows[i])]))
                    self.log(2, f"      {'signal' if tg == 0 else 'k=' + str(tg):>8s}  {lab:>14s}  {pt(p):8.3f}  {math.sqrt(gj.dr2(rapidity(p), phi(p), j['y'], j['phi'])):.3f}")

    def ref_rows(self, coll, rank):
        return [r for r, k in self.genjets_of[coll].items() if k == rank]

    # -- 6 rho --------------------------------------------------------------------------------------------
    def density(self):
        a = self.args
        self.log(1, banner("6  the median energy density rho, two ways"))
        self.rho_grid, self.grid_cells, (dy, dphi) = grid_rho(self.input_p4)
        ny, nphi = self.grid_cells.shape
        self.log(1, f"  grid median (fixedGridRhoFastjetAll): {ny} x {nphi} cells of {dy:.3f} x {dphi:.3f} = {dy * dphi:.3f} in |y| < {GRID_RAP_MAX:g}, phi from 0; "
                    f"rho = {self.rho_grid:.2f} GeV per unit area (cells from {self.grid_cells.min():.1f} to {self.grid_cells.max():.1f}, mean {self.grid_cells.mean():.2f})")
        self.rho_grid_signal, cells_sig, _ = grid_rho(self.input_p4[self.input_tags == 0])
        self.rho_grid_pileup = grid_rho(self.input_p4[self.input_tags != 0])[0]
        self.log(1, f"  the same grid on the signal alone: rho = {self.rho_grid_signal:.2f} (median; mean {cells_sig.mean():.2f}, {int((cells_sig == 0).sum())} of {cells_sig.size} cells empty); "
                    f"on the crowd alone: rho = {self.rho_grid_pileup:.2f}, so the signal raises the median by {self.rho_grid - self.rho_grid_pileup:.2f}")
        if a.no_area:
            self.rho_kt, self.kt = None, None
            return
        self.kt = MixedJets(self.input_p4, self.input_tags, R=a.radius, power=1, ptmin=0.0, rng=np.random.default_rng(a.seed + 2), ghosts=True)
        self.rho_kt, vals = jet_median_rho(self.kt.jets)
        self.log(1, f"  jet median (Cacciari-Salam): kT jets of R = {a.radius:g} with ghosts, {len(self.kt.jets)} jets, {len(vals)} within |y| < {KT_RHO_RAP_MAX:g}; "
                    f"rho = {self.rho_kt:.2f} GeV per unit area (pT/A from {min(vals):.1f} to {max(vals):.1f}); {self.kt.seconds:.0f} s")
        self.log(1, f"  per collision: {self.rho_grid / max(1, len(self.in_time)):.3f} (grid) and {self.rho_kt / max(1, len(self.in_time)):.3f} (jets) GeV per unit area")

    def quick_density(self):
        """The ensemble: the grid rho and the pileup inside the circle of the photograph, no clustering. The circle is
        drawn in (y, phi) around the rapidity of the chapter-7 jet, the plane of the clustering, of rho and of pi R^2."""
        self.rho_grid = grid_rho(self.input_p4)[0]
        self.rho_grid_pileup = grid_rho(self.input_p4[self.input_tags != 0])[0]
        self.rho_kt, self.mixed, self.kt = None, None, None
        self.ref = {key: self.genjets["ak4"][k] for key, k in (("b", 0), ("bbar", 1))}
        self.circle = {}
        for key, ref in self.ref.items():
            d2 = np.array([gj.dr2(rapidity(p), phi(p), ref["y"], ref["phi"]) for p in self.input_p4])
            inside = d2 < self.args.radius ** 2
            ptv = np.hypot(self.input_p4[:, 1], self.input_p4[:, 2])
            pu = inside & (self.input_tags != 0)
            self.circle[key] = dict(pt_pileup=float(ptv[pu].sum()), n_pileup=int(pu.sum()), pt_signal=float(ptv[inside & (self.input_tags == 0)].sum()),
                                    n_collisions=len(set(int(t) for t in self.input_tags[pu])),
                                    rho_area=self.rho_grid * math.pi * self.args.radius ** 2,
                                    rho_area_pileup=self.rho_grid_pileup * math.pi * self.args.radius ** 2)

    # -- 7 the subtraction ------------------------------------------------------------------------------------
    def subtract(self):
        a = self.args
        rho = self.rho_grid
        self.log(1, banner("7  the area subtraction, pT - rho A, against the truth"))
        for j in self.mixed.jets:
            j["pt_sub"] = j["pt"] - rho * j["area"]
            j["rho_area"] = rho * j["area"]
            j["residual"] = j["pt_sub"] - j["pt_signal"]            # the record's: the jet's (vector) pT - rho A against the scalar sum of the signal inside
            j["residual_pileup"] = j["pt_pileup"] - j["rho_area"]   # the ensemble's: the scalar sum of the pileup inside against rho A
        self.log(1, f"  rho = {rho:.2f} (grid); for every jet pT_sub = pT - rho A, and the truth is the signal pT inside the jet")
        self.log(1, f"  {'jet':>4s} {'pT':>8s} {'A':>6s} {'rho A':>7s} {'pT_sub':>8s} {'signal':>8s} {'pileup':>8s} {'resid':>7s}  {'n':>4s} {'n_pu':>5s}  eta    phi")
        for j in self.mixed.jets[:12]:
            self.log(1, f"  [{j['rank']:2d}] {j['pt']:8.2f} {j['area']:6.3f} {j['rho_area']:7.2f} {j['pt_sub']:8.2f} {j['pt_signal']:8.2f} {j['pt_pileup']:8.2f} {j['residual']:+7.2f}  "
                        f"{j['n']:4d} {j['n_pileup']:5d}  {j['eta']:5.2f} {j['phi']:5.2f}")
        n_pu_jets = sum(1 for j in self.mixed.jets if j["pt_signal"] < 0.5 * j["pt"])
        n_pu_jets30 = sum(1 for j in self.mixed.jets if j["pt_signal"] < 0.5 * j["pt"] and j["pt"] > 30)
        n_pu_jets30_sub = sum(1 for j in self.mixed.jets if j["pt_signal"] < 0.5 * j["pt"] and j["pt_sub"] > 30)
        self.log(1, f"  pileup jets (more than half their pT from the crowd): {n_pu_jets} of {len(self.mixed.jets)}, {n_pu_jets30} above 30 GeV before and {n_pu_jets30_sub} after the subtraction")
        self.n_pu_jets = dict(all=n_pu_jets, above30=n_pu_jets30, above30_sub=n_pu_jets30_sub)
        if self.ours.get("b"):
            d, j = self.ours["b"]
            ref = self.ref["b"]
            self.log(1, f"  our jet: {ref['pt']:.2f} (photograph) -> {j['pt']:.2f} with the crowd; rho A = {rho:.2f} x {j['area']:.3f} = {j['rho_area']:.2f}; "
                        f"pT_sub = {j['pt_sub']:.2f}; the crowd put {j['pt_pileup']:.2f} inside, the subtraction took {j['rho_area']:.2f} out: residual {j['residual']:+.2f} GeV "
                        f"on the signal's {j['pt_signal']:.2f} (pileup inside - rho A = {j['residual_pileup']:+.2f}, the definition of the ensemble: scalar sums on both sides; "
                        f"with the rho of the crowd alone, {self.rho_grid_pileup:.2f}, the two read {j['pt'] - self.rho_grid_pileup * j['area'] - j['pt_signal']:+.2f} and "
                        f"{j['pt_pileup'] - self.rho_grid_pileup * j['area']:+.2f})")

    # -- 8 the visible vertices ---------------------------------------------------------------------------
    def visible_vertices(self):
        it = self.in_time
        vis = [c for c in it if c["ev"]["visible"]]
        s = self.signal_vertex
        dz = np.array([abs(c["vertex"][2] - s[2]) for c in it]) if it else np.zeros(0)
        dt = np.array([abs(c["vertex"][3] - s[3]) for c in it]) if it else np.zeros(0)
        self.n_visible = len(vis)
        self.nearest = dict(dz_mm=float(dz.min() * 10) if len(dz) else None, dt_ps=float(dt.min() * 1e3) if len(dt) else None,
                            n_within_1mm=int((dz < 0.1).sum()), n_within_30ps=int((dt < 0.03).sum()))
        self.log(1, banner("8  the vertices the tracker will see"))
        self.log(1, f"  in-time collisions with >= {VISIBLE_NCH} charged particles of pT > {VISIBLE_PT:g} GeV within |eta| < {VISIBLE_ETA:g}: {len(vis)} of {len(it)} "
                    f"({100 * len(vis) / max(1, len(it)):.0f}%); CMS reconstructs about 0.7 mu primary vertices")
        if len(it):
            self.log(1, f"  the nearest crowd vertex to the signal's: {dz.min() * 10:.2f} mm away in z, {dt.min() * 1e3:.0f} ps in t; {self.nearest['n_within_1mm']} within 1 mm "
                        f"(the vertex finder's reach), {self.nearest['n_within_30ps']} within 30 ps (what the timing layer of Phase 2 would resolve; Run 3 has none)")

    # -- 9 the profile and the reweighting ---------------------------------------------------------------------
    def profile(self):
        edges = np.arange(0, 301, 1.0)
        prof = PROFILE_2025
        # one fill of the 2025 profile's header: levelled at 2.23e34 (mu = 67.8 with its 80 mb) for 9 h, then decaying with tau = 10 h
        # (a model of the burn-off), 12.5 h in all; the beams are separated at the start to hold the luminosity
        self.mu_fill0 = mu_of(prof["lumi"], prof["n_bunch"], prof["xsec_mb"])
        mu_fill, w_fill = fill_profile(self.mu_fill0, prof["fill_hours"], prof["levelled_hours"], 10.0)
        # the year: the release's 2025 profile, the distribution of mu_true over the fills of 2025 "derived from LHC simulations"
        year = release_profile(edges, prof)
        ys = profile_stats(prof)
        flat = flat_profile(*FLAT_PROFILE, edges)
        mu_hl, w_hl = fill_profile(SCENARIOS["ultimate"]["mu"], 12.0, 8.0, 5.0)
        hl = profile_histogram(mu_hl, w_hl, edges)
        sample = np.zeros(len(edges) - 1)
        sample[int(np.clip(np.searchsorted(edges, self.mu, side="right") - 1, 0, len(sample) - 1))] = 1.0
        self.profiles = dict(edges=edges, fill=profile_histogram(mu_fill, w_fill, edges), year=year, flat=flat, hllhc=hl, sample=sample)
        self.mu_fill_mean, self.mu_year_mean, self.mu_hl_mean = float((mu_fill * w_fill).sum()), ys["mean"], float((mu_hl * w_hl).sum())
        self.year_stats = ys
        self.w_fixed = reweight(self.mu, year, sample, edges)
        self.w_flat = reweight(self.mu, year, flat, edges)
        self.w_flat_range = [reweight(m, year, flat, edges) for m in range(FLAT_PROFILE[0], FLAT_PROFILE[1] + 1)]
        self.log(1, banner("9  the pileup profile of a fill, and the reweighting"))
        if self.sc["region"] == "hllhc":
            w = reweight(self.mu, hl, sample, edges)
            self.w_fixed = w
            self.log(1, f"  a levelled HL-LHC fill (mu = {SCENARIOS['ultimate']['mu']:g} for 8 h, then decaying with tau = 5 h, 12 h in all): mu from {mu_hl.min():.0f} to {mu_hl.max():.0f}, "
                        f"luminosity-weighted mean {self.mu_hl_mean:.0f}, {100 * w_hl[mu_hl > 0.99 * mu_hl.max()].sum():.0f}% of the collisions at the levelled value")
            self.log(1, f"  a sample mixed at a fixed mu = {self.mu:g} reweighted to that profile: w = P_target({self.mu:g}) / P_sample({self.mu:g}) = {w:.2f}, and no event at all for the tail; "
                        f"a production mixed with the profile itself (type probFunction) needs no reweighting")
            self.log(1, f"  for comparison the Run-3 production's flat profile (mu_true uniform in {FLAT_PROFILE[0]} ... {FLAT_PROFILE[1]}) reweighted to the 2025 profile (mean {self.mu_year_mean:.1f}): "
                        f"w from {min(self.w_flat_range):.2f} to {max(self.w_flat_range):.2f}")
            return
        self.log(1, f"  one fill of the 2025 profile's header (L = {prof['lumi']:.2e} for {prof['levelled_hours']:g} h, mu = {self.mu_fill0:.1f} with its {prof['xsec_mb']:g} mb and "
                    f"{prof['n_bunch']} bunches, then decaying with tau = 10 h, {prof['fill_hours']:g} h in all): mu from {mu_fill.min():.0f} to {mu_fill.max():.0f}, "
                    f"luminosity-weighted mean {self.mu_fill_mean:.0f}, {100 * w_fill[mu_fill > 0.99 * mu_fill.max()].sum():.0f}% of the collisions at the levelled value")
        self.log(1, f"  the 2025 profile of the release ({prof['name']}_cfi, \"derived from LHC simulations\"): mean {ys['mean']:.1f}, r.m.s. {ys['rms']:.1f}, "
                    f"median {ys['median']}, 68% of the collisions between {ys['q16']} and {ys['q84']}; {100 * ys['below']:.0f}% below {FLAT_PROFILE[0]}, "
                    f"{100 * ys['inside']:.0f}% in {FLAT_PROFILE[0]} ... {FLAT_PROFILE[1]}, {100 * ys['above']:.0f}% above {FLAT_PROFILE[1]} (an analysis reweights to the measured profile of its data)")
        self.log(1, f"  the production's flat profile (probFunction, mu_true uniform in {FLAT_PROFILE[0]} ... {FLAT_PROFILE[1]}): the weight w = P_2025(mu_true) / P_flat(mu_true) "
                    f"runs from {min(self.w_flat_range):.2f} to {max(self.w_flat_range):.2f} over the flat range, {self.w_flat:.2f} at mu_true = {self.mu:g}; the "
                    f"{100 * (ys['below'] + ys['above']):.0f}% of the 2025 collisions outside the band have no simulated event to carry them")
        self.log(1, f"  a sample mixed at a fixed mu = {self.mu:g} reweighted to the same profile: one bin (w = {self.w_fixed:.3f} for every event) and no event at all for the rest "
                    f"of the profile: it cannot be reweighted")
        self.log(1, f"  for comparison a levelled HL-LHC fill (mu = {SCENARIOS['ultimate']['mu']:g} for 8 h, then tau = 5 h, 12 h): mean {self.mu_hl_mean:.0f}, "
                    f"{100 * w_hl[mu_hl > 0.99 * mu_hl.max()].sum():.0f}% of the collisions at the levelled value")


# ---------------------------------------------------------------------------
# the summary and the record
# ---------------------------------------------------------------------------
def stage_summary(st: Stage) -> dict:
    s = dict(scenario=st.args.scenario, mu=st.mu, mu_formula=st.mu_formula, n_bunch=st.n_bunch, window=st.window, bunches=list(st.bunches), n_bx=dict(st.n_bx),
             n_total=len(st.collisions), n_in_time=len(st.in_time),
             n_pu_particles=st.n_pu_particles, n_pu_in=st.n_pu_in, n_signal_in=st.n_signal_in, rho_grid=st.rho_grid, rho_kt=st.rho_kt,
             rho_grid_pileup=st.rho_grid_pileup,
             signal_vertex=[float(v) for v in st.signal_vertex], vertex_rms=st.vertex_rms, region_model=st.region.name, region=st.region_moments,
             n_visible=st.n_visible, nearest=st.nearest, kinds={}, library=len(st.lib),
             profile=dict(mu_fill_mean=st.mu_fill_mean, mu_year_mean=st.mu_year_mean, mu_hllhc_mean=st.mu_hl_mean, w_fixed=st.w_fixed, w_flat=st.w_flat,
                          w_flat_range=[min(st.w_flat_range), max(st.w_flat_range)], year=PROFILE_2025["name"], year_stats=st.year_stats,
                          mu_fill_levelled=st.mu_fill0))
    for c in st.collisions:
        s["kinds"][c["ev"]["kind"]] = s["kinds"].get(c["ev"]["kind"], 0) + 1
    if st.mixed is not None:
        s["n_jets"] = len(st.mixed.jets)
        s["n_jets30"] = sum(1 for j in st.mixed.jets if j["pt"] > 30)
        s["n_jets30_sub"] = sum(1 for j in st.mixed.jets if j["pt_sub"] > 30)
        s["pu_jets"] = st.n_pu_jets
        s["jets"] = {}
        for key in ("b", "bbar"):
            ref, best = st.ref[key], st.ours.get(key)
            if best is None:
                s["jets"][key] = None
                continue
            d, j = best
            s["jets"][key] = dict(rank=j["rank"], dr=d, pt=j["pt"], area=j["area"], rho_area=j["rho_area"], pt_sub=j["pt_sub"], pt_signal=j["pt_signal"],
                                  pt_pileup=j["pt_pileup"], n=j["n"], n_signal=j["n_signal"], n_pileup=j["n_pileup"], n_collisions=len(j["collisions"]),
                                  residual=j["residual"], residual_pileup=j["residual_pileup"], m=j["m"], y=j["y"], eta=j["eta"], phi=j["phi"], ref_pt=ref["pt"], ref_n=ref["n"])
    else:
        s["circle"] = st.circle
    return s


def pileup_tables(st: Stage) -> str:
    a = st.args
    out = []
    t0 = (f"Standalone Pileup Summary  (mu = {st.mu:g}, window bx = {st.bxs[0]:+d}..{st.bxs[-1]:+d}, {BUNCH_SPACING:g} ns; library {st.lib.path.name} "
          f"({len(st.lib)} events, {'sequential' if a.sequential else 'random'}); beam spot {'2025' if st.region.name == 'run3' else 'HL-LHC'}; seed {a.seed})")
    out += ["", f" --------  {t0}  " + "-" * max(10, 133 - len(t0) - 14), " "]
    out.append(f"  mu from L sigma/(n_b f_rev): {st.mu_formula:.2f}; mixed at mu = {st.mu:g}")
    out.append("  collisions per bunch crossing: " + ", ".join(f"bx {bx:+d}: {n}" for bx, n in st.n_bx.items()) + f"; {len(st.collisions)} in all, {len(st.in_time)} in time")
    r = st.region_moments["model"]
    out.append(f"  beam spot: sigma_z = {r[0]:.2f} cm, sigma_t = {r[1] * 1e3:.0f} ps (model); drawn r.m.s. z {st.vertex_rms['z']:.2f} cm, t {st.vertex_rms['t'] * 1e3:.0f} ps, "
               f"x {st.vertex_rms['x']:.1f} um, y {st.vertex_rms['y']:.1f} um")
    s = st.signal_vertex
    out.append(f"  signal vertex: x = {s[0]:.2f} um, y = {s[1]:.2f} um, z = {s[2]:.3f} cm, t = {s[3]:.4f} ns")
    out.append(f"  in-time final particles: {st.n_pu_particles} ({st.n_pu_in} within |eta| < {ETA_MAX:g}); signal {len(st.signal_rows)} ({st.n_signal_in})")
    out.append(f"  rho: grid median {st.rho_grid:.3f} GeV per unit area" + (f", jet median {st.rho_kt:.3f}" if st.rho_kt is not None else "") + f"; signal alone {st.rho_grid_signal:.3f}, crowd alone {st.rho_grid_pileup:.3f}")
    out.append(f"  visible in-time vertices (>= {VISIBLE_NCH} charged, pT > {VISIBLE_PT:g}, |eta| < {VISIBLE_ETA:g}): {st.n_visible} of {len(st.in_time)}")
    out.append(f" --------  End {t0}  " + "-" * max(10, 133 - len(t0) - 18))
    t1 = "Standalone Pileup Vertex Listing  (k = 0 the signal; bx, library event, process, final particles, central charged, visible; vertex x y [um], z [cm], t [ns] with the 25 ns x bx offset)"
    out += ["", f" --------  {t1}  " + "-" * max(10, 133 - len(t1) - 14), " ",
            f"{'k':>6s}{'bx':>5s}{'lib':>7s}{'proc':>6s}{'nfin':>6s}{'nch':>5s}{'cen':>5s}{'vis':>5s}{'x[um]':>10s}{'y[um]':>10s}{'z[cm]':>10s}{'t[ns]':>10s}"]
    n_cen = sum(1 for i in st.signal_rows if is_charged(st.ev.p[i].id) and pt(st.ev.p[i].p) > VISIBLE_PT and abs(eta(st.ev.p[i].p)) < VISIBLE_ETA)
    out.append(f"{0:6d}{0:5d}{0:7d}{'sig':>6s}{len(st.signal_rows):6d}{sum(1 for i in st.signal_rows if is_charged(st.ev.p[i].id)):5d}{n_cen:5d}{'1':>5s}"
               f"{s[0]:10.2f}{s[1]:10.2f}{s[2]:10.3f}{s[3]:10.4f}")
    for c in st.collisions:
        e, v = c["ev"], c["vertex"]
        out.append(f"{c['k']:6d}{c['bx']:5d}{c['lib']:7d}{e['kind']:>6s}{e['nfinal']:6d}{e['nch']:5d}{e['n_central']:5d}{int(e['visible']):5d}{v[0]:10.2f}{v[1]:10.2f}{v[2]:10.3f}{v[3]:10.4f}")
    out.append(f" --------  End {t1}  " + "-" * max(10, 133 - len(t1) - 18))
    t2 = ("Standalone Pileup Particle Listing  (the in-time crossing, bx = 0: collision k, id, name, px py pz e [GeV], every final particle of the collision; "
          "mixjet = the mixed anti-kT jet that holds it, -1 if none or not clustered)")
    out += ["", f" --------  {t2}  " + "-" * max(10, 133 - len(t2) - 14), " ", f"{'k':>6s}{'id':>9s}  {'name':<14s}{'px':>12s}{'py':>12s}{'pz':>12s}{'e':>12s}{'mixjet':>8s}"]
    jet_of = {}
    if st.mixed is not None:
        for i, r in enumerate(st.input_rows):
            jet_of[(int(st.input_tags[i]), int(r))] = st.mixed.jet_of.get(i, -1)
    for c in st.in_time:
        e = c["ev"]
        for n_, (pid, p) in enumerate(zip(e["ids"], e["p4"])):
            out.append(f"{c['k']:6d}{int(pid):9d}  {name(int(pid)):<14s}{p[1]:12.4f}{p[2]:12.4f}{p[3]:12.4f}{p[0]:12.4f}{jet_of.get((c['k'], n_), -1):8d}")
    out.append(f" --------  End {t2}  " + "-" * max(10, 133 - len(t2) - 18))
    if st.mixed is not None:
        t3 = (f"Standalone Pileup Jet Listing  (anti-kT R = {a.radius:g} on signal + in-time pileup within |eta| < {ETA_MAX:g}, pT > {a.ptmin:g}; A = active area (ghosts); "
              f"rho = {st.rho_grid:.3f} (grid); pTsub = pT - rho A; sig/pu = pT of the signal / pileup constituents; genjet = matched chapter-7 ak4 jet, dR in (y, phi))")
        out += ["", f" --------  {t3}  " + "-" * max(10, 133 - len(t3) - 14), " ",
                f"{'mixjet':>8s}{'pT':>10s}{'A':>8s}{'rhoA':>9s}{'pTsub':>10s}{'sig':>10s}{'pu':>10s}{'y':>8s}{'eta':>8s}{'phi':>8s}{'m':>8s}{'n':>5s}{'nsig':>6s}{'npu':>5s}{'ncol':>5s}{'genjet':>8s}{'dR':>7s}"]
        gen = st.genjets["ak4"]
        for j in st.mixed.jets:
            best = None
            for g in gen:
                d = math.sqrt(gj.dr2(g["y"], g["phi"], j["y"], j["phi"]))        # in rapidity, as Stage.cluster matches
                if d < MATCH_DR and (best is None or d < best[0]):
                    best = (d, g)
            out.append(f"{'mix':>4s} {j['rank']:3d}{j['pt']:10.3f}{j['area']:8.3f}{j['rho_area']:9.3f}{j['pt_sub']:10.3f}{j['pt_signal']:10.3f}{j['pt_pileup']:10.3f}{j['y']:8.3f}{j['eta']:8.3f}"
                       f"{j['phi']:8.3f}{j['m']:8.3f}{j['n']:5d}{j['n_signal']:6d}{j['n_pileup']:5d}{len(j['collisions']):5d}"
                       + (f"{best[1]['rank']:8d}{best[0]:7.3f}" if best else f"{-1:8d}{'':>7s}"))
        out.append(f" --------  End {t3}  " + "-" * max(10, 133 - len(t3) - 18))
        t4 = "Standalone Pileup Signal Constituent Listing  (row of the chapter-7 record -> mixed jet index, -1 if none; only the signal particles that entered the mixed clustering)"
        out += ["", f" --------  {t4}  " + "-" * max(10, 133 - len(t4) - 14), " ", f"{'no':>6s}{'mixjet':>8s}"]
        for i, r in enumerate(st.input_rows):
            if st.input_tags[i] == 0:
                out.append(f"{int(r):6d}{st.mixed.jet_of.get(i, -1):8d}")
        out.append(f" --------  End {t4}  " + "-" * max(10, 133 - len(t4) - 18))
    return "\n".join(out) + "\n"


def read_pileup_tables(path: Path):
    """The tables of a record written by this script: (summary lines, collisions, in-time particles, mixed jets, signal row -> mixed jet)."""
    summary, collisions, particles, jets, signal_of = [], [], [], [], {}
    section = None
    for line in path.read_text().splitlines():
        if "Standalone Pileup Summary" in line:
            section = None if line.lstrip().startswith("--------  End") else "summary"
            continue
        if "Standalone Pileup Vertex Listing" in line:
            section = None if line.lstrip().startswith("--------  End") else "vertex"
            continue
        if "Standalone Pileup Particle Listing" in line:
            section = None if line.lstrip().startswith("--------  End") else "particle"
            continue
        if "Standalone Pileup Jet Listing" in line:
            section = None if line.lstrip().startswith("--------  End") else "jet"
            continue
        if "Standalone Pileup Signal Constituent Listing" in line:
            section = None if line.lstrip().startswith("--------  End") else "signal"
            continue
        f = line.split()
        if section == "summary" and line.strip():
            summary.append(line.strip())
        elif section == "vertex" and len(f) == 11 and f[0] == "0" and f[3] == "sig":
            f = f[:6] + ["93"] + f[6:]            # an older record left the signal's central count blank (93 charged in the tracker's reach)
            collisions.append(dict(k=0, bx=0, lib=0, kind="sig", nfinal=int(f[4]), nch=int(f[5]), n_central=int(f[6]), visible=1, x=float(f[8]), y=float(f[9]), z=float(f[10]), t=float(f[11])))
        elif section == "vertex" and len(f) == 12 and f[0].lstrip("-").isdigit():
            collisions.append(dict(k=int(f[0]), bx=int(f[1]), lib=int(f[2]), kind=f[3], nfinal=int(f[4]), nch=int(f[5]), n_central=int(f[6]) if f[6].isdigit() else None,
                                   visible=int(f[7]), x=float(f[8]), y=float(f[9]), z=float(f[10]), t=float(f[11])))
        elif section == "particle" and len(f) in (7, 8) and f[0].isdigit():
            particles.append(dict(k=int(f[0]), id=int(f[1]), name=f[2], px=float(f[3]), py=float(f[4]), pz=float(f[5]), e=float(f[6]), jet=int(f[7]) if len(f) == 8 else -1))
        elif section == "signal" and len(f) == 2 and f[0].isdigit():
            signal_of[int(f[0])] = int(f[1])
        elif section == "jet" and len(f) >= 16 and (f[0] == "mix" or (f[0].startswith("mix") and f[0][3:].isdigit())):
            if f[0] != "mix":                     # an older record wrote 'mix100' for the ranks above 99
                f = ["mix", f[0][3:]] + f[1:]
            jets.append(dict(rank=int(f[1]), pt=float(f[2]), area=float(f[3]), rho_area=float(f[4]), pt_sub=float(f[5]), pt_signal=float(f[6]), pt_pileup=float(f[7]),
                             y=float(f[8]), eta=float(f[9]), phi=float(f[10]), m=float(f[11]), n=int(f[12]), n_signal=int(f[13]), n_pileup=int(f[14]),
                             n_collisions=int(f[15]), genjet=int(f[16]), dr=float(f[17]) if len(f) > 17 else None))
    return summary, collisions, particles, jets, signal_of


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def load_inputs(args):
    ev, vertex, tau = dec.read_record(args.input)
    genjets, genjets_of = gj.read_jet_tables(args.input)
    lib = Library(args.library)
    return ev, genjets, genjets_of, lib


def run(args, verbose=0, full=True, lib=None, log=None):
    if lib is None:
        ev, genjets, genjets_of, lib = load_inputs(args)
    else:
        ev, vertex, tau = dec.read_record(args.input)
        genjets, genjets_of = gj.read_jet_tables(args.input)
    st = Stage(ev, genjets, genjets_of, lib, args, verbose=verbose, rng=np.random.default_rng(args.seed), log=log, full=full)
    return ev, st


_REPEAT = None


def one_repeat(seed: int) -> dict:
    args = SimpleNamespace(**vars(_REPEAT))
    args.seed = seed
    ev, st = run(args, verbose=0, full=False, lib=_REPEAT.lib_obj)
    s = stage_summary(st)
    return dict(seed=seed, n_in_time=s["n_in_time"], rho_grid=s["rho_grid"], rho_grid_pileup=s["rho_grid_pileup"], n_visible=s["n_visible"], nearest=s["nearest"], kinds=s["kinds"],
                circle=s["circle"], n_pu_in=s["n_pu_in"], signal_vertex=s["signal_vertex"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=HERE.parent.parent / "GenJet" / "event_GenJet.lhe")
    ap.add_argument("-l", "--library", type=Path, default=HERE.parent / "pileup_library.lhe.gz")
    ap.add_argument("-o", "--output", type=Path, default=HERE.parent / "event_PU.lhe")
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="run3", help="run3 (2025: mu = 60, bx = -12 ... +3, the 2025 beam spot), nominal or ultimate (HL-LHC)")
    ap.add_argument("--mu", type=float, default=None, help="the mean number of collisions per crossing (default: the scenario's)")
    ap.add_argument("--window", default=None, help="the bunch crossings around the signal's: N for bx = -N ... +N, or MIN,MAX (default: the scenario's, -12,3 for Run 3)")
    ap.add_argument("--region", choices=("run3", "hllhc"), default=None, help="the vertex generator: the 2025 beam spot or the HL-LHC luminous region (default: the scenario's)")
    ap.add_argument("--sequential", action="store_true", help="read the library from its first event on (CMSSW sequential = True)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--radius", type=float, default=R_JET)
    ap.add_argument("--ptmin", type=float, default=JET_PT_MIN)
    ap.add_argument("--no-area", action="store_true", help="no ghosts: faster, no areas, no jet-median rho")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    ap.add_argument("--log", type=Path, default=None, help="write the learning-mode narration to this file as well")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--repeat", type=int, default=0, help="draw the in-time crowd this many times on the photograph (seeds 1..N), no clustering")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--history", type=Path, default=None, help="--repeat: write the per-draw results to this JSON")
    ap.add_argument("--summary", type=Path, default=None, help="write the stage summary as JSON")
    args = ap.parse_args()
    lines = []
    logf = (lambda text: (lines.append(text), None if args.quiet else print(text))) if args.log else None
    if args.repeat:
        global _REPEAT
        _REPEAT = args
        _REPEAT.lib_obj = Library(args.library)
        seeds = list(range(1, args.repeat + 1))
        t0 = time.time()
        if args.jobs > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(args.jobs) as pool:
                results = pool.map(one_repeat, seeds)
        else:
            results = [one_repeat(s) for s in seeds]
        hist = args.history or HERE / "standalone_pileup_stats.json"
        hist.write_text(json.dumps(dict(args={k: str(v) for k, v in vars(args).items() if k != "lib_obj"}, results=results), indent=1))
        rho = np.array([r["rho_grid"] for r in results])
        n = np.array([r["n_in_time"] for r in results])
        rho_pu = np.array([r["rho_grid_pileup"] for r in results])
        print(f"{len(results)} draws in {time.time() - t0:.0f} s: in-time collisions {n.mean():.1f} +- {n.std():.1f}; rho {rho.mean():.2f} +- {rho.std():.2f} "
              f"({rho.mean() / n.mean():.3f} per collision); rho of the crowd alone {rho_pu.mean():.2f} +- {rho_pu.std():.2f}")
        for key in ("b", "bbar"):
            pu = np.array([r["circle"][key]["pt_pileup"] for r in results])
            ra = np.array([r["circle"][key]["rho_area"] for r in results])
            rp = np.array([r["circle"][key]["rho_area_pileup"] for r in results])
            npu = np.array([r["circle"][key]["n_pileup"] for r in results])
            nco = np.array([r["circle"][key]["n_collisions"] for r in results])
            sig = np.array([r["circle"][key]["pt_signal"] for r in results])
            print(f"  the circle of the {key} jet (R = {args.radius:g} in y-phi, signal inside {sig.mean():.2f}): pileup {pu.mean():.2f} +- {pu.std():.2f} GeV in "
                  f"{npu.mean():.1f} particles from {nco.mean():.1f} collisions; rho pi R^2 {ra.mean():.2f} +- {ra.std():.2f}; "
                  f"residual {(pu - ra).mean():+.2f} +- {(pu - ra).std():.2f} GeV ({(pu - rp).mean():+.2f} +- {(pu - rp).std():.2f} with the rho of the crowd alone)")
        print(f"wrote {hist}")
        return
    ev, st = run(args, verbose=args.verbose, log=logf)
    s = stage_summary(st)
    if args.log:
        args.log.write_text("\n".join(lines) + "\n")
    if args.summary:
        args.summary.write_text(json.dumps(s, indent=1, default=float))
    text = args.input.read_text()
    args.output.write_text(text.rstrip("\n") + "\n" + pileup_tables(st))
    if not args.quiet:
        j = s.get("jets", {}).get("b")
        print(f"mu = {st.mu:g}: {len(st.collisions)} collisions ({len(st.in_time)} in time, {st.n_pu_in} particles within |eta| < {ETA_MAX:g}); rho = {st.rho_grid:.2f} GeV per unit area"
              + (f"; our jet {j['ref_pt']:.2f} -> {j['pt']:.2f} with the crowd, A = {j['area']:.3f}, pT - rho A = {j['pt_sub']:.2f} (signal inside {j['pt_signal']:.2f}, pileup {j['pt_pileup']:.2f})" if j else "")
              + f"; wrote {args.output} ({time.time() - st.t0:.0f} s)")


if __name__ == "__main__":
    main()
