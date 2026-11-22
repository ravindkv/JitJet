# Chapter 10 companion: The Crossing of the Border (SIM)

This is the technical companion of chapter 10, *The Crossing of the Border: The Simulated Hits*
([`chapter_10_SIM.tex`](chapter_10_SIM.tex)). The chapter gives the physics; this page says how to run the
scripts, how to read what they write, how each number of the official packages was obtained, and lists every
table the chapter quotes with the command that produced it. All files are in [`example/SIM/`](../example/SIM/)
unless stated otherwise; the CMSSW side (the full Geant4 physics stage by stage, the pileup library and the
`MixingModule` at SimHit level) is in [JitJetSW](https://github.com/ravindkv/JitJetSW), see section 4.3.

The conditions are those of Run 3 in 2025, as in chapters 8 and 9: 13.6 TeV, mu = 60 in the sixteen bunch
crossings bx = -12 ... +3, the 2025 beam spot, a 3.8 T field.

| | |
|---|---|
| Stage | the primaries of chapter 9 through the detector: field, decays in flight, electromagnetic and hadronic physics; the signal, the in-time and the out-of-time pileup simulated separately and their hits added by hand |
| Standalone scripts | [`standalone_SIM.py`](../example/SIM/Standalone/standalone_SIM.py) (the simulation), [`add_hits_SIM.py`](../example/SIM/Standalone/add_hits_SIM.py) (the addition) |
| Input | the primaries of chapter 9, rebuilt from [`example/PU/event_PU.lhe`](../example/PU/event_PU.lhe) and [`example/PU/pileup_library.lhe.gz`](../example/PU/pileup_library.lhe.gz) through `example/G4/Standalone/standalone_G4.py` |
| Hits | `example/SIM/simhits_signal.npz` (0.3 MB), `simhits_intime.npz` (2.4 MB), `simhits_oot.npz` (27 MB, not in git: `make records` writes it) |
| Event record | [`example/SIM/event_SIM.lhe`](../example/SIM/event_SIM.lhe): the chapter-9 record plus three tables |
| Learning-mode log | [`standalone_SIM.log`](../example/SIM/Standalone/standalone_SIM.log) (`-v`, seed 1), [`add_hits_SIM.log`](../example/SIM/Standalone/add_hits_SIM.log); numbers as JSON next to them |
| Validation with official packages | [`validate_SIM.py`](../example/SIM/Standalone/validate_SIM.py): the CMS simulation geometry scanned with ROOT's TGeo; log `validate_SIM.log` |
| Displays | [`plot_simhits_SIM.py`](../example/SIM/Standalone/plot_simhits_SIM.py) -> `Standalone/pages/simHits_<sample>_10_SIM_<station>_<view>.png` |
| CMSSW figures | `example/SIM/CMSSW/simHits_mixed_10_SIM_ALL_{etaphi,rz,xy}_run1_event1.png` |
| CMSSW write-up (JitJetSW) | physics: [`chapter_10_SIM.tex`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_10_SIM.tex); technical: [`chapter_10_SIM.md`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_10_SIM.md) |
| Film | [`animate_standalone_SIM.py`](../example/SIM/Standalone/animate_standalone_SIM.py): narrated film (four scenes, edge-tts voice and numpy score through `example/soundtrack.py`); not yet on the [playlist](https://www.youtube.com/playlist?list=PLbSW20iyZFd4) |

Contents: [1. Running the standalone script](#1-running-the-standalone-script) ·
[2. Reading the record](#2-reading-the-record) ·
[3. Drawing the event and rendering the film](#3-drawing-the-event-and-rendering-the-film) ·
[4. Validation from official packages](#4-validation-from-official-packages) ·
[5. The full tables](#5-the-full-tables)

## 1. Running the standalone script

Two scripts, both with `/usr/bin/python3` (numpy; the plots also need matplotlib and scipy).

```bash
cd example/SIM/Standalone
python3 standalone_SIM.py --sample signal -v --log standalone_SIM.log   # the family, 6 s
python3 standalone_SIM.py --sample intime --jobs 10                     # 63 collisions, 5 s
python3 standalone_SIM.py --sample oot --jobs 10                        # 897 collisions, 75 s
python3 add_hits_SIM.py -v > add_hits_SIM.log                           # the addition, 4 s
```

`standalone_SIM.py` simulates one sample at a time (`--sample all` runs the three). Every collision has its
own random stream, `numpy.random.default_rng([seed, 1000 + k])` for collision `k` of the crowd and
`[seed, 0]` for the family, so the result does not depend on `--jobs`. A collision of the crowd is simulated
on the clock of its own bunch crossing; its hits carry `k` and `bx`, and times relative to that crossing.

| option | meaning |
|---|---|
| `--sample signal\|intime\|oot\|all` | which sample |
| `--stage transport\|B\|B_Decay\|B_Decay_EM\|full` | the physics switched on. With `full` (default) the signal is run in all five stages, for the stage table; the file written is the full one. Any other stage writes `simhits_<sample>_<stage>.npz` |
| `--field 3.8` | the field in tesla |
| `--seed 1`, `--jobs N`, `-o DIR` | seed, processes for the crowd, output folder |
| `-v`, `-vv`, `--log FILE` | the learning mode (signal only) |

What the script contains, in the order of the file: the constants of the detector (`BARREL`, `DISKS_1`: one
tuple per surface with radius or z, extent, radiation lengths `t0 + t1 u`, sensor thickness; the calorimeter
edges; the muon stations); the formulas (`helix_radius`, `looper_pt`, `bethe_si`, `landau_si`, `highland`,
`pair_fraction`, `conversion_probability`, `f_em_mean`, `gp_parameters`, `bock_parameters`, `phase_space`);
the cell codes (`ecal_codes`, `es_codes`, `hcal_codes`, `hf_codes`, `cell_position`); the class `Sim` with
`track` (the tracker loop), `decay`, `nuclear`, `to_face`, `calorimeter`, `em_shower`, `hadron`, `mip`,
`muon_stations`, `hf_shower`; and the drivers of the three samples.

`add_hits_SIM.py` has one function that matters, `mix()`: it concatenates the hit arrays of the three files,
adds `25 ns x bx` to the times of the out-of-time hits, and sums the calorimeter energies per cell and origin
(`E[cell, origin]`, origins 0 signal, 1 early, 2 in time, 3 late) and per bunch crossing (`Ebx[cell, 16]`).
Everything else in it counts.

### The learning modes

`-v` prints eight blocks, the source of every number in the chapter's theory section:

- **0** the detector: layers, cells (61 200 EB, 17 376 EE, 9072 HB, 6768 HE, 2160 HO, 1728 HF), the constants
  of lead tungstate;
- **1** the field: the thresholds of equation (looper), the charged primaries below them, the sensor crossings
  without and with field, the leading charged member of the jet;
- **2** decays in flight: c tau and mean flight per species, the decays by species and where they happened;
- **3** electromagnetic physics: the Bethe-Bloch minimum, the hits and their energy loss, the Highland angle,
  conversions by subdetector, bremsstrahlung photons, the photons above 1 GeV that convert;
- **4** hadronic physics: nuclear interactions by subdetector, the charged hadrons that interact, where the
  showers start, the electromagnetic fraction at 1, 10, 100 GeV, the invisible and the dead energy;
- **5** the stage table; **6** the stations of the full simulation; **7** our jet, member by member.

`-vv` adds one line per decay, conversion and nuclear interaction of the full simulation.

## 2. Reading the record

`example/SIM/event_SIM.lhe` is the record of chapter 9 followed by:

- `Standalone SimHit Summary`: the text of `sim_ledger.txt` (section 5);
- `Standalone SimHit Primary Listing`: one row per primary of the family. Columns: row of the chapter-7
  record, id, name, E, pT, eta; `fate` (what ended it as a primary: `calorimeter`, `conversion`, `nuclear`,
  `decay`, `stopped`, `beam pipe`); radius, z and pT where it left the tracker volume (r = 123.3 cm or
  |z| = 293.5 cm); its tracker hits; the energy it and its descendants left in the ECAL, in the scintillator of
  HB+HE+HO, and the photoelectrons in the HF; `*` if it is a member of ak4 jet [0];
- `Standalone SimHit Collision Listing`: one row per collision of the crowd: k, bx, primaries, tracks, tracker
  hits, muon hits, ECAL energy, HCAL scintillator energy, HF photoelectrons.

The hits are in the `.npz` files (`numpy.load`). Arrays of the signal file: `trk_*` and `mu_*`
(`det, lay, x, y, z, t, de, p, pid, prim`: detector code, layer, position [cm], time [ns], energy lost in the
sensor [GeV], momentum, id of the particle that made the hit, index of the primary it descends from); `cal_*`
(`code, e, t, prim`: one row per cell and primary; cell code = detector x 10^9 + index, decoded by
`standalone_SIM.cell_position`); `prim_*` (id, four-momentum, vertex, row, kind, fate, boundary crossing, jet
membership); `sec` (stored secondaries: process, id, primary, vertex, energy, parent id) and `vtx` (decay,
conversion and interaction vertices). The crowd files have `k` and `bx` instead of `prim`, one calorimeter
row per cell and collision, and `col_*` per collision. Detector codes: 1 PXB, 2 PXF, 3 TIB, 4 TID, 5 TOB,
6 TEC, 11 DT, 12 CSC, 13 RPC, 14 GEM, 21 EB, 22 EE, 23 ES, 24 HB, 25 HE, 26 HO, 27 HF.

Excerpt of the primary listing, the members of our jet the chapter names:

```text
   row     id       name         E       pT     eta         fate     r_b      z_b     pT_b   trk      ECAL     HCAL     HF jet0
  2181    211        pi+    29.007   14.269  -1.336  calorimeter   123.3   -218.5   14.211    16    13.700   0.0865      0 *
  2211   -321         K-    22.521   10.567  -1.389      nuclear       -        -        -    63    15.314   0.0087      0 *
  2238    321         K+    19.114    8.546  -1.444  calorimeter   123.3   -246.6    8.504    19    12.487   0.0045      0 *
  2245    310       K_S0    18.646    8.987  -1.359        decay       -        -        -   102    12.627   0.0140      0 *
```

The leading pi+ leaves the tracker through its side at z = -218.5 cm with pT 14.21 of its 14.27 GeV; the K- had a
nuclear interaction and the K0S decayed, so neither has a boundary crossing, and their 63 and 102 tracker hits
are those of their descendants (`grep ' \*$' example/SIM/event_SIM.lhe` lists the 31 members).

## 3. Drawing the event and rendering the film

### The displays

```bash
python3 plot_simhits_SIM.py                 # the 14 pages the chapter uses, 12 s
python3 plot_simhits_SIM.py --sample mixed --station TRK --view rz
python3 plot_simhits_SIM.py --all           # 4 samples x 5 stations x 3 views
```

The layout follows the `SimHitAnalyzer` pages of JitJetSW (`example/SIM/CMSSW/*.png`): header with the counts,
the detector elements in grey, the signal by kind of primary, the crowd by bunch crossing, cells sized by
energy. The pages are PNG. A page with 1.2 million markers takes 1.4 s in matplotlib; the slow ROOT PNG
writing reported for the CMSSW pages (50 min per page, against half a minute through PDF) does not occur here.
The (eta, phi) page of the sum, `pages/simHits_mixed_10_SIM_ALL_etaphi.png`, is the upper panel of the
chapter's display; there is no separate `journey_SIM` script at this stop.

### The theory plots

```bash
python3 plot_theory_SIM.py -o theory_SIM.pdf     # 9 pages, 1 min; needs validate_SIM.json and add_hits_SIM.json
```

Pages: 1 helix and loopers, 2 decays, 3 Bethe-Bloch and Landau, 4 material (TGeo against the model) and what
it does to photons and pions, 5 a radiography of the model (not in the book), 6 electromagnetic shower,
7 hadronic shower, 8 the three crowds in time, 9 hits per collision (not in the book). Pages 4 to 7 shoot
single particles through `Sim` (function `gun`); their numbers are written to `plot_theory_SIM.json`.

### The film

```bash
python3 animate_standalone_SIM.py --stills animate_standalone_SIM_stills.pdf --no-video
python3 animate_standalone_SIM.py --print-script                     # the narration, one line per phase
python3 animate_standalone_SIM.py --fast -o preview.mp4              # every fourth frame at 60 dpi
python3 animate_standalone_SIM.py --jobs 8                           # the film, 1920x1080 (edge-tts needs the network)
```

Four scenes in the r-z view; the hits appear as the clock runs, with counters of the hits and of the ECAL energy
so far. 13 lines of narration (12 phases and the outro), every number read from `add_hits_SIM.json`; 253 s,
1920x1080, about 8 min to render with `--jobs 8`. In the film one out-of-time hit in four is drawn (`--thin`);
the counters count all. The `.mp4` and `.srt` are not in git; the narration clips are cached in
`build/narration/SIM/`.

## 4. Validation from official packages

### 4.1 The detector against the CMS simulation geometry (TGeo)

There is no Geant4 on the machine the book is built on. The official package of this stop is the geometry
Geant4 runs in: `example/G4/geometry/cmsSimGeo2021.root`, the TGeo dump of the CMS simulation geometry (tag
2021; the tracker is the one of 2025), with the material of every volume.

```bash
PYTHONPATH=$(root-config --libdir) ../../ME/Standalone/.venv/bin/python validate_SIM.py > validate_SIM.log
```

1. **Material.** Straight lines from the origin, 61 values of eta and 24 azimuths, followed with
   `TGeoManager::FindNextBoundaryAndStep` to r = 123.3 cm or |z| = 293.5 cm, summing step / radiation length and
   step / interaction length of each volume. Result: 0.342 X0 and 0.134 lambda at eta = 0, 1.40 and 0.38 at 1.0,
   a maximum of 2.04 X0 at |eta| = 1.4, about 1.3 beyond 2. The layer model: mean ratio 0.99 in both units for
   |eta| < 2.5, rms deviation 8 % point by point. **The model was tuned to this scan**: the thickness
   coefficients `t0`, `t1` of the layers and of the dead pieces were fitted (bounded least squares) to the TGeo
   curve, and `LAMBDA_PER_X0 = 0.28` is the ratio of the two TGeo means. Before the tuning the model, written
   from the published numbers of the tracker, was 14 % low on average and 20 % off point by point.
2. **Sensors.** The straight lines of the 307 charged primaries of the family, counting the volumes whose name
   contains `Active` (or `PixelForwardSensor`): 3636 (PXB 565, PXF 349, TIB 687, TID 299, TOB 676, TEC 1060).
   The stage `transport` of the script: 3419 (558, 231, 640, 277, 657, 1056). Not tuned. The forward pixel
   disks of the model, flat rings, have a third fewer crossings than the real turbine blades.
3. **Formulas** against the PDG: minimum dE/dx in silicon 1.666 (1.664) MeV cm2/g; most probable loss in
   320 um 84.1 (84) keV; helix radius, looper threshold, conversion probability and Highland angle identical.

What is *not* validated against an official package: the showers. Their closure tests are the single-particle
pages of `plot_theory_SIM.py`: photons of 10 GeV (71 % in the hottest crystal unconverted), pions of 2 to
300 GeV (response 0.77 to 0.85, to be compared with the test-beam shape of EPJC 60 (2009) 359, not fitted to
it). The resolution of the script for pions does not improve with energy as a real calorimeter's does.

### 4.2 The same physics on another event: CMSSW

The comparison with Geant4 is per primary or per collision, because the events differ (section 4.3).

| quantity | standalone | CMSSW | how |
|---|---|---|---|
| sensors crossed per charged primary, no field | 3419 / 307 = 11.1 | (6734 - 3283) / 277 = 12.5 | their transport-only crossings minus those of neutral primaries |
| tracker hits per primary, full | 8585 / 672 = 12.8 | 10 272 / 571 = 18.0 | |
| energy loss per hit | 0.245 MeV | 0.23 MeV | |
| decays in flight | 87 | 41 | ours has more loopers decaying |
| photons > 1 GeV converting, abs(eta) < 2.5 | 40 of 61 | 15 of 29 | |
| charged hadrons pT > 0.5 interacting | 32 of 93 | 19 of 73 | |
| HF photoelectrons, signal | 1936 | 1849 | the HF constants of the script were chosen to give this scale |
| tracker hits per pileup collision | 1 197 428 / 960 = 1247 | 2 096 409 / 995 = 2107 | |
| ECAL energy per in-time collision | 4290 / 63 = 68 GeV | 3430 / 64 = 54 GeV | |
| signal share of tracker hits | 0.71 % | 0.49 % | |

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
| the stages (`tab:...:stages`) | `\simStageRows` | `standalone_SIM.log`, block 5 |
| the ledger of the hits (`tab:...:hits_ledger`) | `\simLedgerRows` | `add_hits_SIM.log`, first block |
| tracker, calorimeters, muon system | `\simTrackerRows`, `\simEcalRows`, `\simHcalRows`, `\simMuonRows` | `add_hits_SIM.log` |
| the circle of our jet (`tab:...:circle`) | `\simCircleRows` | `add_hits_SIM.log`, last block |

### 5.1 The agreement table

Column standalone: `standalone_SIM.log` (blocks 1, 3, 4, 6), `add_hits_SIM.log`; column TGeo:
`validate_SIM.log`; column CMSSW: section 4.2.

### 5.2 The energy ledger

From the last block of `add_hits_SIM.log`:

| row | value | where |
|---|---|---|
| members reaching the calorimeters as primaries | 13 of 31, vector pT 44.32 | `members reaching the calorimeters as primaries` |
| what the cells record of the members | crystals 133.50 GeV in 942 cells (pT 56.33); HCAL 0.306 GeV in 151 cells, x factor 46.2 GeV (pT 17.24); together 73.52 | `their deposits` |
| family, circle | 65.21 (vector), 65.87 (scalar ET) in 491 + 69 cells | `signal :` and `vector pT of the cells` |
| + in time, + early, + late | 80.41, 247.03, 291.97 | `vector pT of the cells of the circle` |

A cell's transverse momentum is E / cosh(eta) along the phi of its centre, seen from the origin; HB and HE
cells are multiplied by 125 and 190, the sampling factors the CMSSW companion quotes for the digitisation.
The circle is dR < 0.4 around the jet axis (eta, phi) = (-1.439, -2.712) in pseudorapidity.

### 5.3 Runtimes

| step | time |
|---|---|
| signal, five stages, `-v` | 6 s |
| in-time crowd, 63 collisions, 10 processes | 5 s |
| out-of-time crowd, 897 collisions, 10 processes | 75 s |
| addition and tables | 4 s |
| 14 display pages | 12 s |
| theory plots | 60 s |
| TGeo validation | 18 s |
