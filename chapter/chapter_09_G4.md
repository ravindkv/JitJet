# Chapter 9 companion: The hand-over drop at the Border (G4)

This is the technical companion of chapter 9, *The hand-over drop at the Border: Geant4 primaries*. The chapter gives
the physics; this page says how to run the scripts, how to read what they write, how each number of the official
packages was obtained, and lists every table the chapter quotes with the command that produced it. All files are in
[`example/G4/`](../example/G4/) of this repository unless stated otherwise; the CMSSW side (the hand-over of the signal
event and of the 1000-collision pileup library to Geant4 in the 2025 detector, with the physics reduced to the
transportation) is in [JitJetSW](https://github.com/ravindkv/JitJetSW) and is written up there, see section 4.3.

The conditions are those of Run 3 in 2025, as in chapter 8: 13.6 TeV, mu = 60 in bx = -12 ... +3, the 2025 beam spot
and the CMS detector of 2025 (`CMSSW_15_0_5`, `auto:phase1_2025_realistic`, era `Run3_2025`; the geometry is the
`Extended2024` one, which the 2025 global tag reads unchanged). The HL-LHC appears only in the exercises and the open
problems.

| | |
|---|---|
| Stage | the gate in front of the detector: which stable particles of the family and of the crowd become Geant4 primaries (`Generator.cc`), which are killed (`StackingAction`) and which are carried to the Roman pots (PPS transport) |
| Standalone script | [`example/G4/Standalone/standalone_G4.py`](../example/G4/Standalone/standalone_G4.py) |
| Input | [`example/PU/event_PU.lhe`](../example/PU/event_PU.lhe), the record of chapter 8 (the signal with its vertex table, the 961 collisions of the window with their vertices, the in-time particles), and [`example/PU/pileup_library.lhe.gz`](../example/PU/pileup_library.lhe.gz), from which the 897 out-of-time collisions are rebuilt |
| Event record | [`example/G4/event_G4.lhe`](../example/G4/event_G4.lhe) (the chapter-8 record plus the summary, the hand-over listing, the collision listing and the decay listing; 3.5 MB) |
| Learning-mode log | [`example/G4/Standalone/standalone_G4.log`](../example/G4/Standalone/standalone_G4.log) (`-v`, seed 1) and its summary [`standalone_G4.json`](../example/G4/Standalone/standalone_G4.json) |
| The library scan | [`standalone_G4_library.log`](../example/G4/Standalone/standalone_G4_library.log) / `.json`: the gate on the 2000 collisions of the library |
| The ledger of the hand-over | [`example/G4/Standalone/handover_ledger.tex`](../example/G4/Standalone/handover_ledger.tex), written with the record (`--tables`); the chapter `\input`s it as `tab:crossing_the_border:handover_ledger` |
| Validation with official packages | [`example/G4/Standalone/validate_G4.py`](../example/G4/Standalone/validate_G4.py): ROOT's TGeo navigator and PYTHIA 8.315's particle data on exactly our particles; log [`validate_G4.log`](../example/G4/Standalone/validate_G4.log), numbers [`validate_G4.json`](../example/G4/Standalone/validate_G4.json) |
| The detector in 3D | [`example/G4/Standalone/draw_cms3d.py`](../example/G4/Standalone/draw_cms3d.py) -> [`chapter/tikz/g4_cms3d.tex`](tikz/g4_cms3d.tex) (the picture to carry) and the last scene of the film |
| CMSSW figure | [`example/G4/CMSSW/simAnalyzer_09_G4.pdf`](../example/G4/CMSSW/simAnalyzer_09_G4.pdf), page 6 of the companion's `simAnalyzer_signal_09_G4.pdf`, the lower panel of the chapter's display |
| CMSSW write-up (JitJetSW) | physics details: [`chapter_09_G4.tex`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_09_G4.tex); technical companion: [`chapter_09_G4.md`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_09_G4.md) |
| Film | scene 9 of the [JitJet playlist](https://www.youtube.com/playlist?list=PLbSW20iyZFd4) |

Contents: [1. Running the standalone script](#1-running-the-standalone-script) ·
[2. Reading the record](#2-reading-the-record) ·
[3. Drawing the event and rendering the film](#3-drawing-the-event-and-rendering-the-film) ·
[4. Validation from official packages](#4-validation-from-official-packages) ·
[5. The full tables](#5-the-full-tables)

## 1. Running the standalone script

`standalone_G4.py` writes out the decision that CMSSW makes before Geant4 takes its first step, on the record of
chapter 8. It is a port of three pieces of `CMSSW_15_0_5`:

* `SimG4Core/Generators/src/Generator.cc`, `Generator::HepMC2G4`, with the `Generator` parameter set of
  `SimG4Core/Application/python/g4SimHits_cfi.py`: `RDecLenCut = 2.9` cm, `MinEtaCut = -5.5`, `MaxEtaCut = 5.5`,
  `EtaCutForHector = 8.2`, `MinPCut = 0.04` GeV, `MaxPCut = 99999` GeV (`ApplyPhiCuts`, `ApplyPtransCut` off; the PDG
  filter of quarks and gluons has nothing to filter in a hadron-level record). The order of the tests is the order of
  the code: for a status-1 particle, the line from its vertex is followed to `|z| = Z_hector = R sinh(8.2) = 52.8 m`;
  if it is still inside `r <= R` there (the very-forward case) the particle is a primary only if it is a neutron, an
  antineutron or a photon; otherwise the momentum cut, then the eta cut: the line is followed to `|z| = Z_lmax =
  R sinh(5.5) = 354.8 cm` (the code's `R (1 - e^{-2 eta}) / (2 e^{-eta})`) and the particle is dropped if
  `x^2 + y^2 < R^2` there. A status-2 particle (decayed by the generator) with its decay vertex at `r >= R` and
  `|z| < Z_hector` would be a primary with its decay predefined; inside, its daughters are processed at the decay
  vertex. The `minTan = 1e-20` protection is ported as it is: a particle with `p_z = 0` exactly stays at its vertex,
  inside the cylinder, and is dropped (section 4.1).
* `SimG4Core/Application/src/StackingAction.cc` with `TrackNeutrino = False`: the neutrinos among the primaries are
  killed (`fKill`) before their first step.
* `SimTransport/PPSProtonTransport` (`LHCTransport`, switched on by `run3_common`): `CommonParameters_cfi.py`
  (`EtaCut = 8.2`, `MomentumCut = 3000`, `PPSRegionStart_45/56 = 203` m) and `OpticalFunctionsTransport::process`,
  which takes every status-1 particle with `pdg_id == 2212`, `|eta| > 8.2` and `|pz| > 3000` GeV; the Run-3 setup
  (`opticalfunctionsTransportSetup_2021`) keeps `useEmpiricalApertures = False`, so the script books every such proton
  at the Roman pots and does not propagate it through the optical functions.

The three crowds are one flat table (`Crowd`): the 749 stable particles of the signal at the smeared vertex of
chapter 8 plus their own displacement (the vertex table of chapter 6, in mm), the 13 456 particles of the 63 in-time
collisions at their collision's vertex, and the 161 696 particles of the 897 out-of-time collisions, rebuilt from their
library events (the record of chapter 8 lists only the in-time particles; `overlay_ledger.py` of chapter 8 rebuilds
them the same way). The library stores the final particles of each collision at its vertex, so a pileup particle from
a heavy-flavour decay starts at the collision rather than at the decay point.

```bash
cd example/G4/Standalone
python3 standalone_G4.py                                   # -> ../event_G4.lhe (6 s)
python3 standalone_G4.py -v --log standalone_G4.log --summary standalone_G4.json --tables handover_ledger
python3 standalone_G4.py --library-scan standalone_G4_library.json > standalone_G4_library.log   # 5 s
python3 standalone_G4.py --no-oot -v                       # the family and the in-time crowd only (no library)
python3 standalone_G4.py --eta-cut 7.0                     # MinEtaCut of SimWithCastor_cff.py (exercise 2)
python3 standalone_G4.py --eta-cut 5.2 --r-dec 2.25 --p-min 0.1   # the knobs: MaxEtaCut, RDecLenCut, MinPCut
```

On this Mac use `/usr/bin/python3` (NumPy); the script imports chapter 8's `standalone_pileup.py` (the record, the
library, the beam spot) and through it chapters 7, 6 and 5, and needs nothing else. `make records` writes the record,
the log, the summary, the LaTeX ledger and the library scan in one go.

### The learning modes

`-v` prints nine blocks, every number of the chapter's theory is read from them:

1. **the rules of the gate**: `RDecLenCut`, `Z_lmax = 2.9 x 122.34 = 354.8 cm`, `Z_hector = 2.9 x 1820.5 = 5279.4 cm`,
   the momentum cut, the StackingAction, the PPS selection, and the edges of the detector the cuts follow (the hole
   of HF from `asinh(1114.95/12.495) = 5.18` to `asinh(1280.05/12.495) = 5.32`);
2. **the crowds arrive**: the signal (749 stable, 13 600 GeV, scalar pT 743.0 GeV, vertex 287.1 um, -155.1 um,
   0.157 cm), the 63 in-time collisions (13 456, 856 800 GeV, 5725.4 GeV) and the 897 out-of-time ones (161 696,
   12 199 200 GeV, 65 501.2 GeV), the vertices from -12.36 to +13.92 cm (r.m.s. 3.56 cm);
3. **the decays the generator made** in the signal: 376 status-2 decays, 14 weak (tau0 > 0.1 um), none outside the
   cylinder; for each weak decay E, pT, eta, beta gamma, c tau, the mean transverse decay length `(pT/m) c tau`, the
   flight, `r_dec`, `z_dec` and `P(r < R)`; the farthest from the beam line, the D0 at r = 1.485 cm, z = -2.67 cm;
4. **the gate, rule by rule**, for each crowd: very forward (kept, dropped), momentum cut, eta cut, neutrinos killed,
   protons to the Roman pots, and the primaries with their shares of n, E and pT;
5. **the line, not the angle**: every stable signal particle with 5.3 < |eta| < 5.7 with the radius of its line at
   354.8 cm from the nominal point and from its vertex; the particles of the window kept beyond 5.5 (114) and dropped
   inside it (93);
6. **the soft and the invisible**: the signal's particles below 40 MeV, its neutrinos and the |sum pT| of the two b
   systems and of all neutrinos (9.13 GeV);
7. **the forward**: the very-forward neutrals and the Roman-pot protons of each crowd, every in-time proton with its xi;
8. **the b systems and our jet**: the stable descendants of the weakly decaying b hadron of each side and the
   constituents of ak4 jet [0], of the NoNu jet [0] and of the mixed jet [1] of chapter 8, before and after;
9. **the ledger**: stable before and primaries after by region and by kind for the signal, the in-time crowd, the
   out-of-time crowd and the signal + in-time crowd, with the drops by reason.

`-vv` adds every stable particle of the signal that does not become a primary (77 lines, with its code and reason)
and one line per in-time collision (stable, primaries, energy, PPS, drops).

The decision codes of the record and of `-vv`: `P` primary, `V` very-forward primary, `N` neutrino killed by the
StackingAction, `E` dropped by the eta cut (the line), `F` very forward but not n / nbar / gamma, `M` momentum cut,
`R` proton carried to the Roman pots. The *reason* is the classification of the CMSSW companion's `g4Ledger.py`, tested
in this order with the eta of the momentum: `eta` (|eta| > 5.5), `p` (p < 40 MeV), `nu`, `other` (the line at work).

## 2. Reading the record

`event_G4.lhe` is the record of chapter 8, unchanged, followed by four sections:

* **Summary**: the gate and its counts for each crowd. The last line, "the farthest at r = 1.452 cm from the beam line
  through the vertex", measures the decay radius from the line through the smeared vertex parallel to the beam; from
  the beam line itself (the `x^2 + y^2` of `Generator.cc`) the same D0 is at 1.485 cm.
```
  signal: stable 749 (13600.0 GeV, pT 743.0) -> primaries 672 (7139.6 GeV, pT 709.5); PPS 0 (0.0 GeV); dropped eta 67 (6230.9 GeV), p 5 (0.1 GeV), nu 5 (229.4 GeV), other 0 (0.0 GeV)
  in-time pileup (63 collisions): stable 13456 (856800.0 GeV, pT 5725.4) -> primaries 10825 (216366.0 GeV, pT 4869.5); PPS 27 (132170.2 GeV); dropped eta 2442 (507722.9 GeV), p 143 (3.5 GeV), nu 10 (120.9 GeV), other 9 (416.5 GeV)
  out-of-time pileup (897 collisions): stable 161696 (12199200.0 GeV, pT 65501.2) -> primaries 128894 (3068509.6 GeV, pT 55072.9); PPS 363 (1953625.7 GeV); dropped eta 30601 (7172245.1 GeV), p 1706 (44.6 GeV), nu 48 (727.6 GeV), other 84 (4047.4 GeV)
  signal + in-time pileup: stable 14205 (870400.0 GeV, pT 6468.3) -> primaries 11497 (223505.6 GeV, pT 5579.1); PPS 27 (132170.2 GeV); dropped eta 2509 (513953.8 GeV), p 148 (3.7 GeV), nu 15 (350.3 GeV), other 9 (416.5 GeV)
```
* **Listing**: one row per stable particle of the signal (`blk signal`, `row` = its row in the record) and of the
  in-time crowd (`blk intime`, `k` = the collision of chapter 8's vertex listing, `row` = its index in chapter 8's
  particle listing): id, name, E, pT, eta, the z of its vertex, `r_line` (the radius of its line at |z| = 354.8 cm),
  the code and the reason. 14 205 rows.
* **Collision Listing**: one row per collision of the window (961, the signal first), with its crossing, library
  event, process, vertex z, stable particles (n, E, pT), primaries (n, E, pT), PPS protons and the drops by reason.
* **Decay Listing**: the 376 rows the generator decayed in the signal (HepMC status 2: hadrons, muons and taus whose
  first daughter has status 91-99), with E, pT, eta, beta gamma, c tau, the flight from the production point, the
  decay radius from the beam line, the decay z and whether it is inside the cylinder (all are).

## 3. Drawing the event and rendering the film

### The (eta, phi) displays

```bash
python3 example/G4/plot_journey_G4.py --png      # -> example/G4/journey_G4.pdf (+ .png)
```

The layout is the CMSSW companion's `SimAnalyzer` page "1. primaries": the primaries by kind (charged hadrons black,
neutral hadrons open grey, photons green, e± orange squares, μ± purple diamonds), the b and bbar systems in red and
blue with their vector sum (magenta star), and the stable particles that are not handed over as grey crosses (x on
the border beyond |eta| = 6). The header: 672 primaries of 749; 67 / 5 / 5 / 0 not handed over; the vertex and the
14 weak decays; the b system (B\*bar0 -> Bbar0, 15 of 15, pT 78.26 GeV, eta -1.38), the bbar system (B\*0 -> Bbar0 after
mixing, 9 of 9, pT 81.97, eta -3.43) and their net pT 14.28 GeV.

### The theory plots

```bash
python3 example/G4/Standalone/plot_theory_G4.py                 # -> theory_G4.pdf, 5 pages, 15 s
python3 example/G4/Standalone/plot_theory_G4.py --page 2 -o p.png
```

| page | subsection | panels |
|---|---|---|
| 1 | the edges | dN/deta and dE/deta of the signal and of an average in-time collision; the fraction of E and pT inside \|eta\| |
| 2 | the line | the effective cut against the vertex z with the border particles; r of the line at 354.8 cm; the effective cut over the 961 vertices |
| 3 | soft, invisible | the momentum spectrum by kind with the 40 MeV cut; photons below 40 MeV per pi0; the energy not handed over by reason |
| 4 | the decays | P(r_dec > 2.9 cm) against pT; the 14 weak decays across the beam and in (z, r) |
| 5 | the forward | the forward crowd in (\|eta\|, E); xi of the protons beyond 8.2; where the 13.6 TeV of a collision go, by kind (the library at beam-spot vertices, seed 9) |

### The detector in three dimensions

```bash
python3 example/G4/Standalone/draw_cms3d.py                   # -> chapter/tikz/g4_cms3d.tex (6639 quads)
python3 example/G4/Standalone/draw_cms3d.py --png cms3d.png   # a look with matplotlib
```

The detector is a list of solids of revolution `(r1, r2, z1, z2, colour)`: the tracker volume (r < 123.3 cm,
|z| < 293.2 cm of `cms.xml`), ECAL barrel and endcaps, HCAL barrel and endcaps (CALO to r = 295 cm, |z| = 554.1 cm), the
solenoid to 3.8 m, the barrel yoke in four iron layers with the four muon stations between them to 7.4 m, the endcap
disks to |z| = 10.6 m and HF between |z| = 11.15 and 12.80 m (12.5 to 159.5 cm from the beam). The quadrant
0 < phi < 90 degrees, the one facing the viewer, is removed; every face turned towards the viewer is cut into quads of
10 degrees and up to 0.9 m by 0.8 m, shaded with a fixed light and sorted from back to front (the painter's algorithm).
The TikZ file draws in metres. Drawn inside the book its quads cost about 15 s of every pdflatex pass, so `make`
renders it once, scaled by 0.245, into `chapter/tikz/g4_cms3d.pdf` (`chapter/tikz/g4_cms3d_page.tex`, remade only when
`g4_cms3d.tex` changes), and `chapter/tikz/g4_gate.tex` places that PDF centred on the origin of the detector. A
schematic for the eye: the bands are the subdetectors, not their material.

### The film

```bash
python3 example/G4/Standalone/animate_standalone_G4.py --jobs 8          # -> animate_standalone_G4.mp4 + .srt
python3 example/G4/Standalone/animate_standalone_G4.py --stills animate_standalone_G4_stills.pdf --no-video --no-narration --no-music
python3 example/G4/Standalone/animate_standalone_G4.py --print-script
```

Six scenes (road, line, soft, decays, forward, ready), one narrated line per phase (16 lines and the outro), voice
edge-tts `en-GB-RyanNeural` cached in `build/narration/G4/`, the score of `example/soundtrack.py`. The last scene
draws the detector of `draw_cms3d.py` with matplotlib.

## 4. Validation from official packages

### 4.1 The geometry of the cuts with ROOT's TGeo

```bash
cd example/G4/Standalone
PYTHONPATH=$(root-config --libdir) ../../ME/Standalone/.venv/bin/python validate_G4.py --log validate_G4.log --summary validate_G4.json
```

(ROOT 6.32.06 from Homebrew, whose PyROOT is built for Python 3.13: the ME virtual environment, which has PYTHIA too.
Ten seconds.) The two cylinders of `Generator.cc` are TGeo tubes, `ETACUT` (r < 2.9 cm, |z| < 354.8 cm) inside
`HECTOR` (r < 2.9 cm, |z| < 5279.4 cm), in a vacuum box. Every stable particle of the window is started at its vertex
in the direction of its momentum (`InitTrack`), and `FindNextBoundaryAndStep` is called until it leaves `ETACUT` and
then `HECTOR`; a particle whose exit point is on the end cap of `ETACUT` (|z| = 354.8 cm, r < 2.9 cm) fails the eta cut,
one whose exit from `HECTOR` is on its end cap is very forward. With the momentum cut, the n / nbar / gamma list, the
neutrinos and the PPS selection written again in the validation, the code of every particle is rebuilt and compared:

```
-- 1. TGeo (ROOT 6.32.06): tubes r < 2.9 cm, |z| < 354.8 cm inside |z| < 5279.4 cm; 175901 stable particles navigated in 5.7 s
   eta cut (line leaves through the end cap at 354.8 cm): TGeo 30400, script 30406
   very forward (line leaves through the end cap at 52.8 m): TGeo 5382, script 5382
   signal :     749 particles, decision identical for 749, 672 primaries
   intime :   13456 particles, decision identical for 13456, 10825 primaries
   oot    :  161696 particles, decision identical for 161690, 128900 primaries
   decisions: 175895 identical, 6 different; codes E 30400, F 2797, M 1854, N 63, P 138202, R 390, V 2195
     different: oot k=125 id -211 p = (+0.0968, -0.1581, -0.0000) GeV: script E, TGeo P (pz rounded to zero in the library: ...)
     ... (six lines, all with pz = 0)
   the closest call: oot k=801 id 310 eta +5.48628: the line at r = 2.899949 cm (0.51 um from the edge), E
```

The six differences are particles of the out-of-time crowd whose `pz` the library (four decimals) rounded to zero.
`Generator.cc` does not move such a particle (`std::abs(pz) >= minTan * ptot` is false), so its "line" stays at the
vertex, inside the cylinder, and the eta test drops it; TGeo's straight line leaves through the side of the tube. The
script follows the code; in CMSSW, with double-precision momenta, the case does not arise. They are 6 of the 84 "other"
drops of the out-of-time crowd.

### 4.2 The decays the generator made with PYTHIA 8.315

The same script asks `pythia8mc`'s `ParticleData` which species PYTHIA decays with the settings of CMSSW
(`Pythia8CommonSettings_cfi.py`: `ParticleDecays:limitTau0 = on`, `tau0Max = 10`): a species is decayed if it has a
decay table (`canDecay`), may decay (`mayDecay`) and has `tau0 <= 10` mm.

```
-- 2. PYTHIA 8.315 ParticleData with ParticleDecays:limitTau0 = on, tau0Max = 10 mm
   18 stable species in the window (175901 particles), 28 species decayed by the generator in the signal (376 decays)
   left to Geant4 (stable in the record): Sigma+ 24.04 mm, Omega- 24.61 mm, K_S0 26.84 mm, Sigma- 44.34 mm, Xi- 49.1 mm, Lambda0 78.9 mm, Xi0 87.1 mm, K+ 3713 mm, pi+ 7804 mm, K_L0 1.533e+04 mm, mu- 6.587e+05 mm, n0 2.655e+14 mm
   decayed by the generator (signal): B0 0.459, Lambda_b0 0.44, D+ 0.312, D_s+ 0.15, D0 0.123, tau- 0.0871, Xi_c0 0.0336
   stable species PYTHIA would decay: 0; decayed species PYTHIA would not: 0
   the longest-lived decayed: B0 (0.459 mm); the shortest-lived left to Geant4: Sigma+ (24.04 mm)
```

### 4.3 CMSSW

The CMSSW side of this stop is one job of [JitJetSW](https://github.com/ravindkv/JitJetSW), `test/09_G4/run_09_G4.sh`
(CMSSW_15_0_5, `auto:phase1_2025_realistic`, `Run3_2025`, `DB:Extended`): `cmsDriver.py step2 -s SIM` on the signal
(`test/07_GenJet/out/step2_GENJET_07_GenJet.root`) and on the 1000 collisions of the library of chapter 8, with the
physics list `JourneyTransportPhysics` (transportation only), no magnetic field and `TrackerSD.ZeroEnergyLoss = True`;
`g4Ledger.py` and `g4Extras_09_G4.py` count what Geant4 received, `simAnalyzer_cfg.py` draws it. Its commands,
configurations, logs and tables are documented there, and this page does not repeat them:

- physics details: [`chapter_09_G4.tex`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_09_G4.tex);
- technical companion: [`chapter_09_G4.md`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_09_G4.md).

The links are also in [`example/G4/CMSSW/README.md`](../example/G4/CMSSW/README.md). The figure
`example/G4/CMSSW/simAnalyzer_09_G4.pdf` is page 6 of `simAnalyzer_signal_09_G4.pdf` (the 1256 x 784 PNG of the
JitJetSW write-up, as a one-page PDF; its right margin is cut in the original).

What the chapter reads from the companion (`test/09_G4/out/`):

| number in the chapter | value | file |
|---|---|---|
| signal: stable -> primaries; drops eta / p / nu / other | 634 -> 571; 56 (6581.3 GeV) / 6 (0.2) / 1 (95.5) / 0 | `ledger_09_G4.txt` |
| signal: handed over n / E / pT | 90.1 / 50.9 / 89.2 % | `ledger_09_G4.txt` |
| signal: primaries beyond 5.5 | neutron 151.2 GeV at eta 9.72 (very forward); pi+ 84.6 GeV at eta -5.5002 (line: 2.940 cm from the vertex at z = +3.11 cm, 2.899 cm from the centre) | `extras_09_G4.txt` (2), chapter 9 of JitJetSW |
| SimVertices | 29: the primary vertex and 28 generator decay vertices, all inside 2.9 cm; the farthest decay the Dbar0 at 25.8 cm from the collision, r = 1.21 cm | JitJetSW md 4.4 |
| b system, bbar system | 8 of 9 (pT 56.82 of 94.28 GeV; the anti-nu_e of 95.5 GeV, pT 37.46), 16 of 16 (114.17); net 20.3 -> 57.45 GeV | `simAnalyzer_signal_09_G4.log` |
| in time (64): stable -> primaries; n / E / pT | 10 893 -> 8786; 80.7 / 28.2 / 84.6 % | `ledger_09_G4.txt` |
| in time: drops; PPS; very forward; border | 1981 / 102 / 0 / 6; 18 protons (104.9 TeV); 147 n, nbar, gamma (179.7 TeV); 7 kept at 5.50-5.52 | `ledger_09_G4.txt`, JitJetSW md 4.5 |
| out of time (931): stable -> primaries; PPS | 176 266 -> 140 749 (79.9 / 25.4 / 84.1 %); 341 | `ledger_09_G4.txt` |
| per collision | 137.3 primaries (in time), 140 749 / 931 = 151.2 (out of time) | JitJetSW md 4.1 |
| run times | signal 53 s, library 3 min 18 s | JitJetSW md 2 |
| 2024 vs 2025 | identical primaries, vertices and hits | JitJetSW md 4.7 |
| the ZDC | `SteppingAction.CMStoZDCtransport = False`: the very-forward neutrals are killed entering `CMStoZDC` (\|z\| 27.5-139 m, `InterimRegion`); `Phase2SteppingAction` does not read the flag | JitJetSW chapter 9, open problems |
| Phase 2 | `Generator` and `StackingAction` parameters the same as `Run3_2025` under `Phase2C17I13M9`; `phase2_common` sets `LHCTransport = False`; ITLayer1 rods at r = 2.75 and 3.25 cm | JitJetSW md 4.6 |

The CMSSW ledger rows of the chapter: the photograph, `ak4GenJets` jet [0], 117.79 GeV, m 16.72, 24 constituents
(chapter 7); `ak4GenJetsNoNu` jet [1], 79.82 GeV, m 13.36, 22 constituents (chapter 7), quoted as the jet without its
neutrino, the nearest number the companion has to the jet's members after the gate (the companion counts the b system
of the B-, not the jet constituents, one by one; the six soft photons, 0.2 GeV in all, are scattered over the centre
and may or may not be in the jet); the b system before (9, 94.28 GeV) and after (8, 56.82 GeV).

## 5. The full tables

### 5.1 The agreement table (`tab:crossing_the_border:numbers`)

| row | standalone | command / source |
|---|---|---|
| cylinder, Z_lmax, Z_hector; momentum cut; PPS | 2.9 cm, 354.8 cm, 52.8 m; 40 MeV; \|eta\| > 8.2, \|pz\| > 3 TeV | `-v`, block 1 |
| longest-lived decayed / shortest-lived left | B0 0.459 mm / Sigma+ 24.04 mm | `validate_G4.log`, block 2 |
| weak decays; farthest | 14, all inside; D0 r = 1.485 cm | `-v`, block 3 |
| family: stable -> primaries; drops; shares | 749 -> 672; 67 / 5 / 5 / 0; 89.7 / 52.5 / 95.5 % | `-v`, blocks 4 and 9 |
| b systems; net pT | 15/15, 9/9; 14.28 -> 14.28 GeV | `-v`, blocks 6 and 8 |
| our jet | 31 of 31, 99.83 GeV (m 14.83) | `-v`, block 8 |
| in time: stable -> primaries; shares; drops | 13 456 -> 10 825; 80.4 / 25.3 / 85.1 %; 2442 / 143 / 10 / 9 | `-v`, blocks 4 and 9 |
| in time: very forward; PPS | 124 (124 087 GeV); 27 (132 170 GeV) | `-v`, block 4 and 7 |
| in time: kept beyond / dropped inside | 6 / 9 | `standalone_G4.json`, `kept_beyond`, `dropped_inside` |
| out of time: stable -> primaries; PPS | 161 696 -> 128 894; 363 | `-v`, block 4 |
| per collision (library) | 142.7 primaries, 25.5 % of E | `standalone_G4_library.log` |

### 5.2 The energy ledger (`tab:crossing_the_border:ledger`)

| row | pT [GeV] | m [GeV] | n | source |
|---|---|---|---|---|
| the photograph, ak4GenJets jet [0] | 99.83 | 14.83 | 31 | chapter 7 |
| its members handed to Geant4 | 99.83 | 14.83 | 31 | `-v`, block 8 (`jets.ak4` of the JSON: `pt_after` 99.830, `n_after` 31, `lost` empty) |
| with the crowd: mixed jet [1] | 133.53 | 43.21 | 115 | chapter 8 |
| its members handed to Geant4 | 133.52 | 43.20 | 114 | `-v`, block 8 (`jets.mixed`: `pt_after` 133.517) |
| CMSSW: photograph; NoNu jet [1]; b system before / after | 117.79; 79.82; 94.28 / 56.82 | 16.72; 13.36 | 24; 22; 9 / 8 | section 4.3 |

### 5.3 The ledger of the hand-over (`tab:crossing_the_border:handover_ledger`)

Written by `standalone_G4.py --tables handover_ledger` (the macros `\gfourLedgerTable` and `\gfourLedgerDrops`). The
numbers, from block 9 of the log (n / E [GeV] / pT [GeV], stable -> primaries):

```
signal          749 / 13600.0 / 743.0   ->   672 / 7139.6 / 709.5
  |eta|<2.5     396 / 982.8 / 418.6     ->   389 / 979.8 / 417.6
  2.5-3.0        55 / 378.0 / 51.3      ->    55 / 378.0 / 51.3
  3.0-5.2       216 / 5180.7 / 241.7    ->   215 / 5058.6 / 233.6
  5.2-5.5        15 / 827.7 / 8.2       ->    13 / 723.3 / 7.1
  >5.5           67 / 6230.9 / 23.1     ->     0
in time        13456 / 856800.0 / 5725.4 ->  10825 / 216366.0 / 4869.5   (+ PPS 27 / 132170.2)
out of time   161696 / 12199200.0 / 65501.2 -> 128894 / 3068509.6 / 55072.9 (+ PPS 363 / 1953625.7)
signal + in time 14205 / 870400.0 / 6468.3 -> 11497 / 223505.6 / 5579.1   (25.7 % of E, 86.3 % of pT)
```

### 5.4 The library scan

```
library scan: 2000 collisions, vertices from the 2025 beam spot (seed 9), 4.8 s
  per collision: 179.2 stable -> 142.7 primaries (median 86, 21 with none, max 757), 3463 GeV, pT 61.3 GeV; 0.393 PPS protons
  handed over: 79.6 % of n, 25.5 % of E, 84.0 % of pT
  CD: 37 collisions, 22.5 primaries, 138 GeV, 2.000 PPS protons each
  DD: 219 collisions, 42.3 primaries, 4289 GeV, 0.452 PPS protons each
  ND: 1425 collisions, 184.9 primaries, 3750 GeV, 0.168 PPS protons each
  SD: 319 collisions, 36.9 primaries, 1997 GeV, 1.169 PPS protons each
```

The 21 collisions with no primary are 20 single- and one double-diffractive collision whose every particle flew beyond
|eta| = 5.5; each gave a proton to the Roman pots.

### 5.5 Two worked examples of the line (chapter, section "The eta cut is a line")

From `standalone_G4.json` (`kept_beyond`, `dropped_inside`), `sinh(5.505) = 122.957`:

* in-time pi+, E 150.4 GeV, eta -5.5050, vertex z = +6.39 cm: flight to z = -354.8 cm is 361.19 cm, r = 361.19 / 122.96
  = 2.938 cm > 2.9: kept;
* in-time proton, E 71.9 GeV, eta -5.4905, vertex z = -4.84 cm: flight 349.96 cm, sinh(5.4905) = 121.18, r = 2.888 cm
  < 2.9: dropped.
