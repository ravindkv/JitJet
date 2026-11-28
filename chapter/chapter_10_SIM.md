# Chapter 10 companion: The Footprints at the Crossings (SIM)

This is the technical companion of chapter 10, *The Footprints at the Crossings: The Simulated RAW Hits*
([`chapter_10_SIM.tex`](chapter_10_SIM.tex)). The chapter gives the physics; this page says how to run the
scripts, how to read what they write, how each number of the official packages was obtained, and lists every
table the chapter quotes with the command that produced it. All files are in [`example/SIM/`](../example/SIM/)
unless stated otherwise; the CMSSW side (the full Geant4 physics stage by stage, the pileup library and the
`MixingModule` at SimHit level) is in [JitJetSW](https://github.com/ravindkv/JitJetSW), see section 4.3.

The conditions are those of Run 3 in 2025, as in chapters 8 and 9: 13.6 TeV, mu = 60 in the sixteen bunch
crossings bx = -12 ... +3, the 2025 beam spot, a 3.8 T field.

**Every number of the chapter is Geant4's**, run on the laptop: the signal, the in-time and the out-of-time
pileup, the physics stage by stage, the single particles. The one exception is the table "The crossing
obtained three ways", which compares Geant4 with the hand-written script and with CMSSW.

| | |
|---|---|
| Stage | the primaries of chapter 9 through the detector: field, decays in flight, electromagnetic and hadronic physics; the signal, the in-time and the out-of-time pileup simulated separately, each in its own file, and their hits added by hand |
| Standalone scripts | [`geant4_SIM.py`](../example/SIM/Standalone/geant4_SIM.py) (the three samples through the official Geant4 in the CMS geometry: the only standalone script of the book that runs an official package for accuracy), [`add_hits_SIM.py`](../example/SIM/Standalone/add_hits_SIM.py) (the addition), [`guns_SIM.py`](../example/SIM/Standalone/guns_SIM.py) (single particles for the theory plots), [`standalone_SIM.py`](../example/SIM/Standalone/standalone_SIM.py) (the hand-written simulation: the readable version, used only in the comparison) |
| Input | the primaries of chapter 9, rebuilt from [`example/PU/event_PU.lhe`](../example/PU/event_PU.lhe) and [`example/PU/pileup_library.lhe.gz`](../example/PU/pileup_library.lhe.gz) through `example/G4/Standalone/standalone_G4.py` |
| Hits | `example/SIM/simhits_signal.npz` (2 MB), `simhits_intime.npz` (7 MB), `simhits_oot.npz` (85 MB, not in git: `make sim-oot`), all three from Geant4; `g4tracks_signal.npz` (Geant4's trajectories and shower deposits for the 3D displays, 13 MB, not in git); the hand-written ones are `simhits_<sample>_fast.npz` |
| Event record | [`example/SIM/event_SIM.lhe`](../example/SIM/event_SIM.lhe): the chapter-9 record plus three tables |
| Logs | [`geant4_SIM.log`](../example/SIM/Standalone/geant4_SIM.log) (`-v`, the signal), `geant4_SIM_<sample>.json`, `geant4_SIM_signal_<stage>.json`, [`guns_SIM.log`](../example/SIM/Standalone/guns_SIM.log), [`add_hits_SIM.log`](../example/SIM/Standalone/add_hits_SIM.log), `plot_theory_SIM.json`, [`geant4_SIM_physics.log`](../example/SIM/Standalone/geant4_SIM_physics.log) (the models of the physics list) |
| Validation with official packages | Geant4 is the official package. [`validate_SIM.py`](../example/SIM/Standalone/validate_SIM.py) compares the hand-written script with it (material, sensors, the family, single particles); log `validate_SIM.log` |
| Displays | [`plot_simhits_SIM.py`](../example/SIM/Standalone/plot_simhits_SIM.py) -> `Standalone/pages/simHits_<sample>_10_SIM_<station>_<view>.png` |
| CMSSW figures | `example/SIM/CMSSW/simHits_mixed_10_SIM_ALL_{etaphi,rz,xy}_run1_event1.png` |
| CMSSW write-up (JitJetSW) | physics: [`chapter_10_SIM.tex`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_10_SIM.tex); technical: [`chapter_10_SIM.md`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_10_SIM.md) |
| 3D displays | [`display_geant4_SIM.py`](../example/SIM/Standalone/display_geant4_SIM.py) -> `Standalone/pages/geant4_{magnet,tracker,ecal,hcal,muon}.png`, drawn with [`g4view_SIM.py`](../example/SIM/Standalone/g4view_SIM.py) |
| Film | [`animate_standalone_SIM.py`](../example/SIM/Standalone/animate_standalone_SIM.py): six scenes inside the Geant4 detector, narrated, no music; key frames in `animate_standalone_SIM_stills.pdf` |

Contents: [0. Geant4 on the laptop](#0-geant4-on-the-laptop) · [1. The hand-written script](#1-the-hand-written-script) ·
[2. Reading the record](#2-reading-the-record) ·
[3. Drawing the event and rendering the film](#3-drawing-the-event-and-rendering-the-film) ·
[4. Validation from official packages](#4-validation-from-official-packages) ·
[5. The full tables](#5-the-full-tables)


## 0. Geant4 on the laptop

Everything needed is public and installs without a compiler:

```bash
make venv-sim     # python -m venv example/SIM/Standalone/.venv; pip install numpy scipy geant4-pybind;
                  # the first import downloads Geant4's data tables (2 GB) to ~/.geant4_pybind
make gdml         # export_gdml_SIM.py: cmsSimGeo2021.root (TGeo, chapter 9) -> cmsSimGeo2021.gdml, 10 MB (needs ROOT)
cd example/SIM/Standalone
.venv/bin/python field_map_SIM.py              # the field map from the coil and the iron of the geometry, 80 s
.venv/bin/python geant4_SIM.py --sample signal -v --log geant4_SIM.log     # the family, 70 s on 10 processes
.venv/bin/python geant4_SIM.py --sample intime                             # 63 collisions of bx 0, 5 min
.venv/bin/python geant4_SIM.py --sample oot                                # 897 collisions of the 15 other crossings, 51 min
for s in transport B B_Decay B_Decay_EM; do .venv/bin/python geant4_SIM.py --sample signal --stage $s; done   # the stage table
.venv/bin/python guns_SIM.py > guns_SIM.log    # single particles for the theory plots, 10 min
python3 add_hits_SIM.py -v > add_hits_SIM.log  # the addition and every table, 10 s
.venv/bin/python dump_shapes_SIM.py            # the corners of every sensor, crystal, tile, chamber: ../cms_shapes_SIM.npz
python3 display_geant4_SIM.py                  # the five 3D pages
```

**Three samples, three files.** `simhits_signal.npz`, `simhits_intime.npz` and `simhits_oot.npz` are written
by three separate runs, and the Makefile makes the two crowd files only when they are missing (`make sim-intime
sim-oot` forces them). If the signal or the script changes, the hour of the crowd is not spent again, and the
same crowd can be laid over another signal. This is a freedom of the simulation only: in data the signal, the
in-time and the out-of-time pileup arrive together and cannot be taken apart.

| script | what it does |
|---|---|
| `export_gdml_SIM.py` | ROOT writes the TGeo geometry (3398 volume types, 490 materials, 2 108 646 placed volumes) as GDML; Geant4's `G4GDMLParser` reads it in one second |
| `field_map_SIM.py` | Geant4's navigator says which material is at each point of a 5 cm grid in (r, z), 16 azimuths; Ampere's law for the flux function psi = r A_phi with a saturating B(H) curve in the steel (Bs = 2.05 T, mu_i = 2000), Picard iteration with scipy's sparse solver. Result: 39.6 million ampere-turns for 3.8 T at the centre, -2.06 / -1.66 / -1.63 T in the three barrel layers of the yoke, 2.30 T radial in the first endcap disk (`field_map_SIM.log`) |
| `geant4_SIM.py` | the application: geometry from GDML, physics list `FTFP_BERT`, CMSSW's production cuts by region and Birks' law, the field (the map everywhere, its iron version in the steel, a compiled uniform 3.8 T in the tracker and calorimeters), sensitive detectors by material and name, one Geant4 event per primary, `--jobs` processes. Writes `../simhits_<sample>.npz`, for the signal also `../g4tracks_signal.npz`, and `geant4_SIM_<sample>.json`. Options: `--sample signal\|intime\|oot`, `--stage` (signal: `transport`, `B`, `B_Decay`, `B_Decay_EM`, `full`; anything but `full` writes only `geant4_SIM_signal_<stage>.json`), `--only`, `--no-field`, `--no-birks`, `--no-regions`, `--physics`, `--scan`, `--tag`, `--max-collisions` |
| `guns_SIM.py` | single particles from the origin: 4000 photons and 4000 pi+ of 5 GeV flat in abs(eta) < 3, stopped at the calorimeters (which converted or interacted, and where); 300 photons of 10 GeV at eta = 0.5; pi+ of 2 to 100 GeV at eta = 0.5. Writes `guns_SIM.json` |
| `dump_shapes_SIM.py` | walks the tree of placed volumes and writes the eight corners of the 17 004 tracker sensors, 61 200 + 14 648 crystals, scintillator layers, muon gas volumes, yoke plates |
| `g4view_SIM.py`, `display_geant4_SIM.py` | a perspective camera and the painter's algorithm in matplotlib; the five pages |
| `draw_tracker_SIM.py` | the layers and disks of the tracker in the r-z plane, from `BARREL` and `DISKS_1` of `standalone_SIM.py`, as TikZ: `chapter/tikz/sim_tracker.tex` |

### How close this is to CMSSW 2025

**Geometry.** `cmsSimGeo2021.root` is the public dump of the Run-3 simulation geometry of CMS
([cms-data/Fireworks-Geometry](https://github.com/cms-data/Fireworks-Geometry); the only Run-3 dump there, next
to Run 1, Run 2 and Phase 2). In the scenario list of CMSSW (`Configuration/Geometry/README.md`) the two years
are `2021 = T3+C3+M13+P7+O7+F1` and `2025 = T3+C2+M17+P8+O10+F5`: the same tracker `T3`; calorimeter versions
that differ in details; muon system `M13 -> M17` (more GE2/1 and iRPC chambers, a modified DT shield, a
corrected GE0 shape); `P8` the rotated PPS detectors; `O10` vacuum between the interaction point and the ZDC;
`F5` the ZDC itself. For the particles of this chapter (the tracker, the calorimeters, the first muon stations)
the 2021 dump is the 2025 detector. A 2025 dump can be dropped in: CMSSW's
`Fireworks/Geometry/python/dumpSimGeometry_cfg.py` knows the tags `Run1 ... 2021, Run4`; with the line
`process.load("Geometry.CMSCommonData.cmsExtendedGeometry2021XML_cfi")` changed to `...2025XML_cfi` it writes
the 2025 tree, and then

```bash
PYTHONPATH=$(root-config --libdir) ../../ME/Standalone/.venv/bin/python export_gdml_SIM.py /path/to/cmsSimGeom-2025.root
JITJET_SIM_GDML=/path/to/cmsSimGeom-2025.gdml .venv/bin/python geant4_SIM.py --sample signal -v
```

(`field_map_SIM.py` and `dump_shapes_SIM.py` read the same variable).

**Production cuts**, from the `*ProdCuts.xml` files the 2025 geometry loads (`Geometry/TrackerSimData`,
`EcalSimData`, `HcalSimData`, `MuonSimData`), as regions of Geant4 (`REGIONS` in `geant4_SIM.py`; the last
column is what `/run/dumpCouples` prints for an electron):

| region | root volumes | cut | energy threshold for an electron |
|---|---|---|---|
| `TrackerPixelSensRegion` | the pixel sensors | 0.01 mm | 31.7 keV in silicon |
| `TrackerSensRegion` | the strip sensors and wafers | 0.1 mm | 122 keV in silicon |
| `TrackerDeadRegion`, `TrackerPixelDeadRegion` | the rest of the tracker | 1 mm | 546 keV in silicon |
| `BeamPipe` / `BeamPipeOutside` | the beryllium section / the rest | 0.01 mm / 1 mm | 27.6 keV in beryllium |
| `CaloRegion` (CMSSW: `EcalRegion`, `HcalRegion`) | ECAL, HCAL, HO, HF | 1 mm | 1.13 MeV in PbWO4, 1.34 MeV in brass, 357 keV in scintillator |
| `PreshowerSensRegion` | the preshower silicon | 0.1 mm | 122 keV |
| `Muon` / `MuonIron` | the muon system / yoke, coil, shielding | 10 mm / 1 mm | 1.29 MeV in iron |
| `MuonChamber`, `MuonSensitive_GEM` | chambers | 0.002 mm | |
| `MuonSensitive_DT-CSC` / `MuonSensitive_RPC` | the gas | 1 mm / 8 mm | 1 keV in the CSC gas |
| everything else | | 10 mm (CMSSW: `DefaultCutValue`) | |

CMSSW gives photons and electrons different cuts in the muon regions; here one number per region (the
electron's), because the Python bindings set a region's cut through `/run/setCutForRegion`. Not taken over: the
APD region of the ECAL (0.005 mm).

**Birks' law**, from `SimG4Core/Application/python/g4SimHits_cfi.py` and `SimG4CMS/Calo/src/{CaloSD,ECalSD}.cc`:
scintillator (HB, HE, HO; `HCalSD`) weight = 1 / (1 + kB dE/dx + C (kB dE/dx)^2) with kB = `BirkC1` / density
= 0.0060 g/(MeV cm2) / density, C = `BirkC2` = 0.142, kB / `BirkC3` = kB / 1.75 for charge >= 2; crystals (`ECalSD`,
`BirkL3Parametrization`) weight = 1 - `BirkSlope` ln(kB dE/dx) = 1 - 0.253694 ln(kB dE/dx) with kB = 0.03333
g/(MeV cm2) / density, kept between `BirkCut` = 0.1 and 1. For the family the scintillator light is
74.1 % of the deposit and the crystal light
98.85 %. `--no-birks` switches both off.

**Also as in CMSSW:** particles are stopped 500 ns after their bunch crossing (`MaxTrackTime`); the forward
calorimeter is not showered (a particle entering its steel is stopped and answered in photoelectrons by the
parametrisation of `standalone_SIM.py`; CMSSW uses a shower library); everything behind the HF (abs(z) > 13 m) is
a dead region.

**Different from CMSSW:** the physics list is `FTFP_BERT` (CMSSW: `FTFP_BERT_EMM`, the same hadronic physics
with a faster electromagnetic configuration in the calorimeters); the models it uses are listed in
`geant4_SIM_physics.log` (photoelectric `LivermorePhElectric`, Compton `Klein-Nishina`, conversion
`BetheHeitlerLPM`, multiple scattering `UrbanMsc` below 100 MeV and `WentzelVIUni` above for e+-, `WentzelVIUni`
for hadrons and muons, ionisation `MollerBhabha`, `Bragg` / `BetheBloch` (2 MeV), `MuBetheBloch`, bremsstrahlung
`eBremSB` below 1 GeV and `eBremLPM` above; hadrons: `BertiniCascade` up to 6 GeV, `FTFP` from 3 GeV, cross
sections `BarashenkovGlauberGribov`, `G4NeutronInelasticXS`); the field is our own 2D map (uniform 3.8 T inside
the calorimeter volume); a calorimeter cell is found from the position of the deposit with the grid of
`standalone_SIM.py`, not from the copy numbers; no light-collection curve along the crystal (`SlopeLightYield`),
no Russian roulette for slow neutrons and soft photons.

**The energy tallies.** For every primary `geant4_SIM.py` adds up (`TALLY`, array `prim_tal` of the signal file,
`col_tal` per collision and group of the crowd files): the energy left in all material of the tracker volume
and of the muon system, in the crystals, the preshower silicon and the scintillator before and after Birks'
law, in the silicon sensors and the chamber gas, and the energy that arrives at the HF and behind it. The
tables "Who leaves how much" of the chapter are sums of these by group of primary (charged hadrons, neutral
hadrons, photons, electrons, muons).

**What is counted.** A *delta ray* is an ionisation step (`eIoni`, `hIoni`, `muIoni`, `ionIoni`) that made a
secondary; a charged particle that loses its last energy by ionisation and stops is counted apart (`ranged_out`:
14 297 for the family, against 45 759 delta rays). Neutrinos are followed, so that the displays can draw them, but
nobody sees them: one that reaches the steel of the HF or the dead region is removed without light and without
an entry in the energy tallies (`neutrinos_hf`, `neutrinos_dead`: 75 and 2 for the family), and a neutrino is not
a stored secondary. At the HF a baryon or a nucleus brings its kinetic energy, an antibaryon its kinetic energy
plus twice its mass, everything else its total energy.

**The boundary of a primary.** A primary "reaches the calorimeters" when it crosses the envelope of the volume
`trackerTracker` as itself: r = 123.3 cm or abs(z) = 293.2 cm (`R_BND`, `Z_BND`; CMSSW: `CaloTrkProcessing`). The
first crossing counts (`App.crossing`): the point where the chord of the step meets the surface and the momentum
at the end of that step go to `prim_bnd`. A crossing of the end face inside r = 293.2 / sinh 3 = 29.3 cm is the
fate `beam pipe` (beyond abs(eta) = 3, past the endcap calorimeters), any other crossing `calorimeter`; a primary
ended before by a decay, a conversion or a nuclear interaction has that fate, and one that ranges out or is still
inside after 500 ns is `stopped`. For the family: 192 + 183 = 375 cross, 63 decay, 139 convert, 86 interact, 9 stop.
Between the beam pipe and the tracker, from abs(z) = 150 cm on, and behind the end of the tracker near the beam,
the geometry has gaps that belong to the world volume and to no subdetector; the sensitive detector `gap` on the
world volume follows the primary there, or a forward primary would be lost from sight at r = 2.5 cm.

Traps met: a `G4PrimaryVertex` built in Python is freed twice (use `G4ParticleGun`); killing a particle
in a sensitive detector needs `fKillTrackAndSecondaries`, or the secondaries of that step are counted again; the
neutron time limit is set through `GetPhysics("neutronTrackingCut").SetTimeLimit`, the UI command arrives too late;
a multiprocessing pool must be `spawn`, and the script a file (not stdin); `G4ProductionCuts` is not in the
bindings (create the `G4Region` in `Construct`, then `/run/setCutForRegion` after `Initialize`); a `G4Region`
made in Python must be kept alive by the script; `/run/dumpCouples` crashes at exit but prints first.

## 1. The hand-written script

`standalone_SIM.py` is the same crossing written by hand, with `/usr/bin/python3` and numpy only: a detector on
one page and one formula per effect. The chapter no longer quotes it, except in the table "three ways"; it
stays as the readable version of what Geant4 does and for the exercises.

```bash
cd example/SIM/Standalone
python3 standalone_SIM.py --sample signal -v --log standalone_SIM.log   # the family, 6 s; ../simhits_signal_fast.npz
python3 standalone_SIM.py --sample intime --jobs 10                     # 63 collisions, 5 s; ../simhits_intime_fast.npz
python3 standalone_SIM.py --sample oot --jobs 10                        # 897 collisions, 75 s; ../simhits_oot_fast.npz
```

| option | meaning |
|---|---|
| `--sample signal\|intime\|oot\|all` | which sample |
| `--stage transport\|B\|B_Decay\|B_Decay_EM\|full` | the physics switched on |
| `--field 3.8` | the field in tesla |
| `--seed 1`, `--jobs N`, `-o DIR` | seed, processes for the crowd, output folder |
| `-v`, `-vv`, `--log FILE` | the learning mode (signal only): eight blocks, one per ingredient |

What the script contains, in the order of the file: the constants of the detector (`BARREL`, `DISKS_1`: one
tuple per surface with radius or z, extent, radiation lengths `t0 + t1 u`, sensor thickness; the calorimeter
edges; the muon stations); the formulas (`helix_radius`, `looper_pt`, `bethe_si`, `landau_si`, `highland`,
`pair_fraction`, `conversion_probability`, `f_em_mean`, `gp_parameters`, `bock_parameters`, `phase_space`);
the cell codes (`ecal_codes`, `es_codes`, `hcal_codes`, `hf_codes`, `cell_position`), which `geant4_SIM.py`
uses too; the class `Sim` with `track` (the tracker loop), `decay`, `nuclear`, `to_face`, `calorimeter`,
`em_shower`, `hadron`, `mip`, `muon_stations`, `hf_shower`; and the drivers of the three samples.

`add_hits_SIM.py` has one function that matters, `mix()`: it concatenates the hit arrays of the three files,
adds `25 ns x bx` to the times of the out-of-time hits, and sums the calorimeter energies per cell and origin
(`E[cell, origin]`, origins 0 signal, 1 early, 2 in time, 3 late) and per bunch crossing (`Ebx[cell, 16]`).
Everything else in it counts.

## 2. Reading the record

`example/SIM/event_SIM.lhe` is the record of chapter 9 followed by:

- `Standalone SimHit Summary`: the text of `sim_ledger.txt` (section 5);
- `Standalone SimHit Primary Listing`: one row per primary of the family. Columns: row of the chapter-7
  record, id, name, E, pT, eta; `fate` (what ended it as a primary: `calorimeter`, `conversion`, `nuclear`,
  `decay`, `stopped`, `beam pipe`); radius, z and pT where it crossed the boundary of the tracker volume (r = 123.3 cm or abs(z) = 293.2 cm; `beam pipe`: through the end inside r = 29.3 cm, beyond abs(eta) = 3); its tracker hits; the
  energy it and its descendants left in the ECAL, in the scintillator of HB+HE+HO, and the photoelectrons in
  the HF; `*` if it is a member of ak4 jet [0];
- `Standalone SimHit Collision Listing`: one row per collision of the crowd: k, bx, primaries, tracks, tracker
  hits, muon hits, ECAL energy, HCAL scintillator energy, HF photoelectrons.

The hits are in the `.npz` files (`numpy.load`). Arrays of the signal file: `trk_*` and `mu_*`
(`det, lay, x, y, z, t, de, p, pid, prim`: detector code, layer, position [cm], time [ns], energy lost in the
sensor [GeV], momentum, id of the particle that made the hit, index of the primary it descends from); `cal_*`
(`code, e, t, prim`: one row per cell and primary, the energy after Birks' law; cell code = detector x 10^9 +
index, decoded by `standalone_SIM.cell_position`); `prim_*` (id, four-momentum, vertex, row, kind, fate,
boundary crossing, jet membership, and `prim_tal`, the energy tallies with their names in `tal_names`); `sec`
(stored secondaries: process, id, primary, vertex, energy, parent id) and `vtx` (decay, conversion and
interaction vertices). The crowd files have `k` and `bx` instead of `prim`, one calorimeter row per cell and
collision, `col_*` per collision and `col_tal[collision, group, column]`. Detector codes: 1 PXB, 2 PXF, 3 TIB,
4 TID, 5 TOB, 6 TEC, 11 DT, 12 CSC, 13 RPC, 14 GEM, 21 EB, 22 EE, 23 ES, 24 HB, 25 HE, 26 HO, 27 HF.

Excerpt of the primary listing, the hardest members of our jet:

```text
   row     id       name         E       pT     eta         fate     r_b      z_b     pT_b   trk      ECAL     HCAL     HF jet0
  2181    211        pi+    29.007   14.269  -1.336  calorimeter   123.3   -218.3   14.220    24    10.095   0.1173      0 *
  2211   -321         K-    22.521   10.567  -1.389      nuclear       -        -        -    79     9.364   0.0860      0 *
  2238    321         K+    19.114    8.546  -1.444      nuclear       -        -        -    73     6.842   0.0606      0 *
  2245    310       K_S0    18.646    8.987  -1.359        decay       -        -        -   175    10.858   0.0295      0 *
  1973   -211        pi-    11.406    4.491  -1.584  calorimeter   123.3   -287.5    4.488    32     3.656   0.0616      0 *
```

A member whose fate is `nuclear`, `decay` or `conversion` has no boundary crossing, and its tracker hits are
those of its descendants (`grep ' \*$' example/SIM/event_SIM.lhe` lists the 31 members).

## 3. Drawing the event and rendering the film

### The displays

```bash
python3 plot_simhits_SIM.py                 # the 4 pages of the chapter and the film, 10 s
python3 plot_simhits_SIM.py --sample mixed --station TRK --view rz
python3 plot_simhits_SIM.py --all           # 4 samples x 5 stations x 3 views
../../compress_png.sh pages/simHits_*.png   # 256 colours: 442 kB for the three pages of the book
```

The layout follows the `SimHitAnalyzer` pages of JitJetSW (`example/SIM/CMSSW/*.png`): header with the counts,
the detector elements in grey, the signal by kind of primary, cells sized by energy. The crowd is two million
hits and is drawn as a histogram (the TH2 of ROOT): 480 x 320 bins in r-z, 320 x 320 in x-y, 0.035 x 0.035 in
(eta, phi); every hit and cell of the crowd is counted in its bin, the bin takes the colour of the group
(early, in time, late) that fills it most per bunch crossing, and its opacity grows with the logarithm of the
count. The r-z page shows r < 450 cm (the tracker, the calorimeters and the first muon station of the barrel;
the header and the tables count all stations), the x-y page the barrel in +-450 cm. Red dashed lines and circles:
the directions of the generator jets with a B hadron (chapter 7).

**Sizes.** `example/compress_png.sh` (ffmpeg `palettegen` / `paletteuse`, no dithering, at most 2000 pixels
wide; `-w 1600` for the 3D pages) rewrites a PNG with an indexed palette of 256 colours; `make` runs it after
each display script. The five 3D pages went from 7.7 MB to 1.8 MB, the stills page of the film from
5.8 MB to 0.9 MB (a JPEG inside the PDF: it is a photograph, not a drawing), and the pages
`simHits_*` that the book does not use are no longer written.

### The theory plots

```bash
python3 plot_theory_SIM.py -o theory_SIM.pdf     # 10 pages, a few seconds; needs guns_SIM.json, validate_SIM.json, add_hits_SIM.json
```

Pages: 1 helix and loopers, 2 decays, 3 Bethe-Bloch and Landau, 4 material (geantinos) and what it does to
photons and pions, 5 a radiography of the tracker in Geant4 (not in the book), 6 electromagnetic shower,
7 hadronic shower, 8 the three crowds in time, 9 hits per collision (not in the book), 10 Birks' law (the weight
against dE/dx, and the light of the family by group). The curves are the
formulas; every point and histogram is Geant4's. `plot_theory_SIM.json` holds the numbers the theory sections
quote for the event (`event`: the charged primaries and the loopers, the fates by group, the decays by
particle, the hits, the photons that convert, the hadrons that interact, the muon-chamber hits, the members of
the jet).

### The film

```bash
python3 animate_standalone_SIM.py --stills animate_standalone_SIM_stills.pdf --no-video
python3 animate_standalone_SIM.py --print-script                     # the narration, one line per phase
python3 animate_standalone_SIM.py --fast -o preview.mp4              # every fourth frame at 60 dpi
python3 animate_standalone_SIM.py --jobs 8                           # the film, 1920x1080 (edge-tts needs the network)
```

Six scenes: 1 the detector and the field, 2 the tracker (and five close-ups), 3 the crystals, 4 the HCAL,
5 the muon system, all for the signal (in scene 5 the tracker, the crystals and the HCAL inside the yoke keep the
colours of their own scenes: blue silicon with yellow hits, green crystals, gold scintillator, deposits coloured by
energy); 6 the crowd in the same detector, first the 63 collisions of our bunch crossing, then all sixteen crossings
(drawn from `../g4pileup_display.npz`, a sample of the hits cut once from the two crowd files). The r-z, x-y and
(eta, phi) pages are in the book, not in the film. 17 lines of narration, every number read from
`geant4_SIM_signal.json` and `add_hits_SIM.json`. Large numbers are written short, 13K and 1.4M, on the screen and in the subtitles, and
spoken as thousand and million (`soundtrack.short_number`, `SPOKEN_RE`). Pronunciations fixed in
`example/soundtrack.py` for all films: "Geant4" is spoken "Jee-antt-four" (joined, so that the number is a stressed syllable), "ampere" "am-pee-ar", "looper"
"loo-per", and a lone "4" or "four" is spoken "fore" (the voices swallow "four" to an unstressed "for"). The
`.mp4` and `.srt` are not in git; the narration clips are cached in `build/narration/SIM/`.

## 4. Validation from official packages

### 4.1 The hand-written script against Geant4

```bash
.venv/bin/python validate_SIM.py > validate_SIM.log      # 5 min; needs geant4_SIM_signal.json
```

1. **Material.** Geantinos from the origin, 61 pseudorapidities x 97 azimuths (a prime: the tracker repeats itself 12,
   24, 36 times around the beam, and 24 azimuths alias by 10 %), summing step / radiation length and step / nuclear
   interaction length up to r = 123.3 cm or |z| = 293.5 cm. Mean x/X0 in |eta| < 2.5: 1.236 in Geant4, 1.164 in the layer
   model; x/lambda 0.367 against 0.329; maximum 2.04 X0 at eta = -1.4. The layer
   model was fitted to a scan of the same geometry with ROOT's TGeo: Geant4 computes the radiation length of a
   mixture from its elements, TGeo keeps the stored value, and they differ by a few per cent.
2. **Sensors on straight lines** of the 307 charged primaries: 3636 in Geant4,
   3419 in the layer model.
3. **Formulas** against the Particle Data Group (the Bethe-Bloch minimum, the Landau peak, the helix, the looper
   threshold, the conversion probability, the Highland angle): all within a per cent.
4. **The family**, hand-written against Geant4:

   | | hand-written | Geant4 |
   |---|---|---|
   | primaries that cross the tracker boundary as themselves | 407 | 375 |
   | decays in flight in the tracker volume | 87 | 369 |
   | photon conversions in the tracker volume | 313 | 655 |
   | nuclear interactions in the tracker volume | 163 | 447 |
   | secondaries above 10 MeV born in the tracker volume | 2008 | 4298 |
   | tracker hits | 8585 | 12623 |
   | energy left in the silicon [MeV] | 2102 | 2979 |
   | ECAL barrel energy [GeV] | 206 | 181 |
   | ECAL endcap energy [GeV] | 411 | 457 |
   | HB + HE scintillator energy [GeV] | 2.51 | 3.48 |
   | HF photoelectrons | 1936 | 1651 |
   | muon-chamber hits | 0 | 82 |
   | jet [0]: members reaching the calorimeters as primaries | 13 | 18 |
   | jet [0]: energy of its descendants in the crystals [GeV] | 134 | 99.1 |

   The counts of decays, conversions and nuclear interactions are not like for like: Geant4 counts every step of
   every secondary in the tracker volume, pi0 decays and slow neutrons included.
5. **Single particles** at eta = 0.5, hand-written / Geant4: 10 GeV photons, unconverted, 9 hottest crystals
   0.952 / 0.933; pions,
   (crystals + 125 x scintillator light) / E: 0.81 +- 0.12 / 0.94 +- 0.38 at 5 GeV, 0.83 +- 0.11 / 1.04 +- 0.24 at 20 GeV, 0.86 +- 0.15 / 1.05 +- 0.15 at 50 GeV.

### 4.2 The same physics on another event: CMSSW

The comparison with CMSSW is per primary or per collision, because the events differ (section 4.3).

| quantity | standalone (Geant4 on the laptop) | CMSSW | how |
|---|---|---|---|
| sensors crossed per charged primary, no field | 3636 / 307 = 11.8 | (6734 - 3283) / 277 = 12.5 | the stage `transport`; theirs minus the crossings of neutral primaries, which its stage table counts as hits in the rows without electromagnetic physics |
| secondaries per primary | 4298 / 672 = 6.4 | 1355 / 571 = 2.4 | not the same count: born in the tracker above 10 MeV here, kept as `SimTrack` there |
| tracker hits per primary, full | 12623 / 672 = 18.8 | 10 272 / 571 = 18.0 | |
| energy loss per hit | 0.236 MeV | 0.23 MeV | |
| primaries crossing the tracker boundary as themselves | 375 of 672 (192 towards the calorimeters, 183 beyond abs(eta) = 3) | 310 of 571 | first crossing of r = 123.3 cm or abs(z) = 293.2 cm, forward ones included on both sides |
| photons > 1 GeV converting, abs(eta) < 2.5 | 39 of 61 | 15 of 29 | |
| charged hadrons p > 0.5 GeV interacting | 44 of 126 | 19 of 73 | |
| HF photoelectrons, signal | 1651 | 1849 | the HF constants of the parametrisation were chosen to give this scale |
| tracker hits per pileup collision | 2040322 / 960 = 2125 | 2 096 409 / 995 = 2107 | |
| ECAL energy per pileup collision, all crossings | 50 294 / 960 = 52 GeV | (56 598 - 362) / 995 = 57 GeV | in time alone 4147 / 63 = 66 against 3430 / 64 = 54: our in-time draw is a busy one (172 primaries per collision against 144 out of time), CMSSW's a quiet one (138 against 151) |
| HF photoelectrons per pileup collision | 293 172 / 960 = 305 | (246 009 - 1849) / 995 = 245 | the two constants of the formula were chosen on the signal |
| muon-chamber hits per pileup collision | 1320 / 960 = 1.4 | 838 / 995 = 0.8 | one production cut per region here, the electron's, also for photons |
| HB / HE cells hit in the sum | 8632 / 6147 | 6856 / 5464 | no Russian roulette here: more slow neutrons, more cells, the same energy (HB 0.043 against 0.044 GeV per collision, HE 0.26 against 0.28) |
| signal share of tracker hits | 0.61 % | 0.49 % | |

### 4.3 CMSSW

The CMSSW side is one job of [JitJetSW](https://github.com/ravindkv/JitJetSW), `test/10_SIM/run_10_SIM.sh`
(CMSSW_15_0_5, 13.6 TeV, run 1, event 1): Geant4 with `FTFP_BERT_EMM` in four stages on the signal, the 1000
collisions of the library, and the `MixingModule` without digitisers on 995 of them in bx = -12 ... +3. Its
commands, configurations, logs and tables are documented there:

- physics details: [`chapter_10_SIM.tex`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_10_SIM.tex);
- technical companion: [`chapter_10_SIM.md`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_10_SIM.md).

What the chapter reads from it (its tables `tab:10_stages`, `tab:10_ledger`, `tab:10_bbudget`, `tab:10_photon`):

| number in the chapter | value |
|---|---|
| primaries at the boundary, transport -> full | 557 -> 310 of 571 |
| secondaries, tracker hits, energy loss, full | 1355; 10 272; 2.33 GeV |
| muon hits with EM physics only | 522 |
| b system: primaries, energy, at the boundary | 8, 140.3 GeV; 3 of 8 with pT 24.4 GeV |
| b system: ECAL, HCAL scintillator, x sampling factor | 63.6 GeV; 0.39 GeV; about 73 GeV |
| the 44 GeV photon: crystals unconverted / converted | two thirds in one crystal / 466 crystals |
| mixture: tracker hits, signal share, signal-dominated ECAL cells | 2 106 681; 0.49 %; 699 of 12 804 |

## 5. The full tables

Every table of the chapter is a macro of [`sim_ledger.tex`](../example/SIM/Standalone/sim_ledger.tex), written
by `add_hits_SIM.py`; the same numbers as text are in `sim_ledger.txt` and `add_hits_SIM.log`, and as JSON in
`add_hits_SIM.json`.

| table of the chapter | macro | source |
|---|---|---|
| who is at the gate (`tab:...:fates`) | `\simFateRows` | `simhits_signal.npz` (`prim_fate`), `geant4_SIM.log` block 5 |
| the stages (`tab:...:stages`) | `\simStageRows` | `geant4_SIM_signal_<stage>.json`, `geant4_SIM_signal.json` |
| the ledger of the hits (`tab:...:hits_ledger`) | `\simLedgerRows` | `add_hits_SIM.log`, first block |
| tracker, ECAL, HCAL, muon system by subdetector | `\simTrackerRows`, `\simEcalRows`, `\simHcalRows`, `\simMuonRows` | `add_hits_SIM.log` |
| who leaves how much in the tracker, the ECAL, the HCAL, behind the calorimeters | `\simBudgetTrkRows`, `\simBudgetEcalRows`, `\simBudgetHcalRows`, `\simBudgetMuonRows` | `add_hits_SIM.log`, block "who leaves how much where"; `geant4_SIM.log` block 10 |
| the circle of our jet (`tab:...:circle`) | `\simCircleRows` | `add_hits_SIM.log`, last block |
| the numbers of the picture to carry | `\simChainSignal`, `\simChainIntime`, `\simChainOot`, `\simChainSum` | `add_hits_SIM.json` |

### 5.1 The agreement table

Column hand-written: `validate_SIM.log` (blocks 1, 2, 4, 5) and, for the crowd rows, `standalone_SIM_intime.json`
and `standalone_SIM_oot.json`; column Geant4: `geant4_SIM.log`, `add_hits_SIM.log`, `plot_theory_SIM.json`;
column CMSSW: section 4.2.

### 5.2 The energy ledger

From the last block of `add_hits_SIM.log`:

| row | value | where |
|---|---|---|
| members reaching the calorimeters as primaries | 17 of 31, vector pT 42.48 (two more leave along the beam pipe) | `members reaching the calorimeters as primaries` |
| what the cells record of the members | crystals 99.04 GeV in 4262 cells (pT 40.35); HCAL 0.587 GeV in 477 cells, x factor 97.9 GeV (pT 38.84); together 79.16 | `their deposits` |
| family, circle | 68.28 (vector), 68.99 (scalar ET) in 1316 cells | `signal :` and `vector pT of the cells` |
| + in time, + early, + late | 92.21, 304.09, 364.35 | `vector pT of the cells of the circle` |

A cell's transverse momentum is E / cosh(eta) along the phi of its centre, seen from the origin; HB and HE
cells are multiplied by 125 and 190, the sampling factors the CMSSW companion quotes for the digitisation.
The circle is dR < 0.4 around the jet axis (eta, phi) = (-1.439, -2.712) in pseudorapidity.

### 5.3 Runtimes

On 10 processes of a laptop with 12 cores:

| step | time |
|---|---|
| Geant4, the family (672 primaries) | 70 s (10 min of processor time) |
| Geant4, the in-time crowd (63 collisions, 10 825 primaries) | 4.9 min (47 min of processor time) |
| Geant4, the out-of-time crowd (897 collisions, 128 894 primaries) | 51 min (8.4 h of processor time) |
| Geant4, the four reduced stages of the family | 208 s |
| Geant4, single particles (`guns_SIM.py`) / validation | 10 min / 5 min |
| addition and tables | 10 s |
| 4 display pages / 5 3D pages / theory plots / stills | 10 s / 1 min / 10 s / 40 s |
| hand-written script: signal / in time / out of time | 6 s / 5 s / 75 s |
| field map / shapes | 80 s / 10 s |
