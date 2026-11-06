# Chapter 8 companion: The Crowd on the Road (PU)

This is the technical companion of chapter 8, *The Crowd on the Road: Pileup*. The
chapter gives the physics; this page says how to run the scripts, how to read what
they write, how each number of the official packages was obtained, and lists every
table the chapter quotes with the command that produced it. All files are in
[`example/PU/`](../example/PU/) of this repository unless stated otherwise; the CMSSW
side (the 2024 pileup library and the mixing of one signal event at PU 60) is in
[JitJetSW](https://github.com/ravindkv/JitJetSW), with its write-ups copied to
[`example/PU/CMSSW/2024/`](../example/PU/CMSSW/2024/) (and the Phase-2 exercise at
PU 10 to [`example/PU/CMSSW/Phase2/`](../example/PU/CMSSW/Phase2/)).

The crowd is the crowd of Run 3 in 2024: the event of chapters 1 to 7 was born at
13.6 TeV under Run-3 conditions and stays there: 60 collisions per crossing, the 2024
beam spot, the Run-3 production window bx = -12 ... +3, and, from chapter 9, the 2024
CMS detector. The HL-LHC crowd (mu = 200 at 14 TeV, the Phase-2 luminous region) is one
option of the script, `--scenario ultimate`, kept for the exercises and the open problems.

| | |
|---|---|
| Stage | the signal of chapter 7 in a bunch crossing of 2024: the crowd, where and when it walks, rho, the areas and the subtraction |
| Standalone script | [`example/PU/Standalone/standalone_pileup.py`](../example/PU/Standalone/standalone_pileup.py) |
| The library | [`example/PU/pileup_library.lhe.gz`](../example/PU/pileup_library.lhe.gz), 2000 minimum-bias collisions at 13.6 TeV made once by [`make_pileup_library.py`](../example/PU/Standalone/make_pileup_library.py) with PYTHIA (its log: [`pileup_library.log`](../example/PU/pileup_library.log)) |
| Input | [`example/GenJet/event_GenJet.lhe`](../example/GenJet/event_GenJet.lhe), the final state and the generator-level jets of chapter 7 |
| Event record | [`example/PU/event_PU.lhe`](../example/PU/event_PU.lhe) (the chapter-7 record plus the summary, the vertex listing of the 960 collisions, the particles of the in-time crossing, the mixed jets and the signal constituents; 1.5 MB) |
| Learning-mode log | [`example/PU/Standalone/standalone_pileup.log`](../example/PU/Standalone/standalone_pileup.log) (`-v`, seed 1) and its summary [`standalone_pileup.json`](../example/PU/Standalone/standalone_pileup.json) |
| Validation with FastJet and PYTHIA | [`example/PU/Standalone/validate_pileup_with_fastjet.py`](../example/PU/Standalone/validate_pileup_with_fastjet.py) |
| The catalogue of the library | [`example/PU/Standalone/minbias_catalogue.py`](../example/PU/Standalone/minbias_catalogue.py), its tables [`minbias_catalogue.txt`](../example/PU/Standalone/minbias_catalogue.txt) / `.json` / `.tex` (the chapter `\input`s the last) |
| CMSSW write-ups | [`example/PU/CMSSW/2024/PULib.pdf`](../example/PU/CMSSW/2024/PULib.pdf) (the library) and [`PUSIM.pdf`](../example/PU/CMSSW/2024/PUSIM.pdf) (the mixing), CMSSW_15_0_5, 2024 conditions (`auto:phase1_2024_realistic`, era `Run3_2024`), 13.6 TeV |
| Film | scene 8 of the [JitJet playlist](https://www.youtube.com/playlist?list=PLbSW20iyZFd4) |

Contents: [1. Running the standalone script](#1-running-the-standalone-script) ·
[2. Reading the record](#2-reading-the-record) ·
[3. Drawing the event and rendering the film](#3-drawing-the-event-and-rendering-the-film) ·
[4. Validation from official packages](#4-validation-from-official-packages) ·
[5. The full tables](#5-the-full-tables)

## 1. Running the standalone script

`standalone_pileup.py` takes the final state of chapter 7 and puts it into a bunch
crossing of 2024, with every piece of CMSSW's mixing written out: mu from the
luminosity, mu = L sigma_inel / (n_b f_rev); the Poisson draw of the MixingModule for
every crossing of the production window bx = -12 ... +3; the library of minimum-bias
collisions it reads at random positions; the beam spot of CMSSW's BetafuncEvtVtxGenerator
(the SimBeamSpot payload of the 2024 global tag: a Gaussian of 3.6 cm along the beam, an
independent Gaussian of 120 ps in time, 9 um across with the hourglass of beta* = 30 cm)
and the 25 ns offsets of the out-of-time crossings; the stacking of the in-time collisions
onto the signal; anti-kT on the mixed final state with the active area of every jet
measured with ghosts, using FastJet's tiled search; the median energy density rho two
ways (FastJet's grid, which CMSSW's fixedGridRhoFastjetAll wraps, and the kT-jet median of
Cacciari and Salam); the subtraction pT - rho A against the truth; the vertices the
tracker will see; the flat pileup profile of the Run-3 production, a 2024-like year and
the reweighting. Seven hundred lines of Python that need NumPy and nothing else; it
imports chapter 7's `standalone_genjets.py` for the record, the clustering and the jet
tables.

A map of the file, with the steps of the chapter's flow chart:

| Function or class | What it does |
|---|---|
| `mu_of`, `SCENARIOS`, `parse_window` | step 1: mu = L sigma_inel / (n_b f_rev), the three scenarios (`run3`, `nominal`, `ultimate`) with their bunches, windows and vertex generators |
| `Library` | the minimum-bias collisions of the library file, with their multiplicities |
| `Stage.draw_crowd` | step 2: the Poisson draw per crossing and the library events read |
| `BeamSpot` (`LuminousRegion` for `--region hllhc`) | step 3: the 2024 beam spot of BetafuncEvtVtxGenerator (the HL-LHC overlap of two crabbed bunches), its moments, and the drawing of vertices |
| `Stage.stack` | step 4: the mixed final state (signal + in-time crowd, abs(eta) < 5, no neutrinos) |
| `TiledClustering`, `MixedJets`, `ghost_grid` | step 5: chapter 7's anti-kT with FastJet's tiled search, the ghosts, every constituent tagged by its collision |
| `grid_rho`, `jet_median_rho` | step 6: the two estimators of rho |
| `Stage.subtract` | step 7: pT - rho A for every jet against the signal pT inside it |
| `Stage.visible_vertices`, `fill_profile`, `year_profile`, `flat_profile`, `reweight` | step 8: the collisions the tracker can see; the profiles and the weight |
| `pileup_tables`, `read_pileup_tables` | step 9: the tables of the record and their reader |
| `stage_summary`, `one_repeat` | the summary JSON and the worker of `--repeat` |

```bash
cd example/PU/Standalone
/usr/bin/python3 standalone_pileup.py -v --log standalone_pileup.log --summary standalone_pileup.json
      # -> ../event_PU.lhe, the book's record (mu = 60, bx = -12 ... +3, seed 1): 2 min, almost all of it the anti-kT clustering
/usr/bin/python3 standalone_pileup.py --no-area              # 20 s: no ghosts, no areas, no jet-median rho
/usr/bin/python3 standalone_pileup.py --window 0             # the in-time crossing only
/usr/bin/python3 standalone_pileup.py --window 1 --sequential   # the conditions of the CMSSW exercise (PUSIM: PU 60 in bx = -1 ... +1, the library read in order)
/usr/bin/python3 standalone_pileup.py --scenario ultimate    # the HL-LHC crowd: mu = 200 in bx = -3 ... +3, the Phase-2 luminous region (15 min)
/usr/bin/python3 standalone_pileup.py --repeat 400 --jobs 4 --quiet --history standalone_pileup_stats.json   # 15 s
```

Options: `--scenario run3|nominal|ultimate` (the luminosity, the colliding bunches, the
window, the vertex generator and the rounded mu CMSSW mixes at: 60 in bx = -12 ... +3 with
the 2024 beam spot; 140 and 200 in bx = -3 ... +3 with the HL-LHC luminous region), `--mu`,
`--window N|MIN,MAX` and `--region run3|hllhc` to override them, `--sequential` to read
the library from its first event on (CMSSW's `sequential = True`), `--seed`, `--radius`,
`--ptmin`, `--no-area`, `--repeat N --jobs J --history FILE` for the ensemble (the
in-time crowd drawn N times on the photograph; no clustering: the grid rho and the
pileup inside the circle of R = 0.4 around our jet and the bbar jet), `--summary FILE`
for the JSON the plots read. The library is made once:

```bash
cd example/PU/Standalone
../../ME/Standalone/.venv/bin/python make_pileup_library.py        # -> ../pileup_library.lhe.gz, 2000 events at 13.6 TeV, 15 s
```

Every collision of the library carries, in its header line, its process code, its final particles, its
charged ones, its scalar sum of pT, the number of parton-parton interactions (`nmpi`, PYTHIA's
`Info::nMPI`: the hardest scattering plus the MPIs) and the pT of the hardest (`pthat`).

### The catalogue of the library

```bash
cd example/PU/Standalone
/usr/bin/python3 minbias_catalogue.py              # -> minbias_catalogue.txt, .json, .tex (two minutes: the anti-kT jets of every collision)
/usr/bin/python3 minbias_catalogue.py --no-jets    # a few seconds
/usr/bin/python3 minbias_catalogue.py --first 20   # the first twenty library events instead of the in-time ten of the record
```

The twin of the companion's `minBiasGenStats.py`, which prints one line per GEN event of the 2024 CMSSW
library: for every collision the kind, the interactions, the stable particles, the charged ones, the
central ones (|eta| < 2.5 and pT > 0.5 GeV, all species, as the CMSSW catalogue counts them), the ones
beyond |eta| = 5.5, the pT of the hardest particle, the scalar sum of pT, the anti-kT R = 0.4 jets above
10 GeV (chapter 7's clustering on the collision alone, without neutrinos) and the forward protons (final
protons with xi = 1 - E/E_beam < 0.55, the ones CMSSW's `LHCTransport` carries to the PPS). The listing
of the chapter is the first ten in-time collisions of our crossing with the vertex the mixing drew for
each (the record's vertex listing); the summary is over the two thousand collisions (the vertex rows over
the 960 drawn); the means per kind close it. The `.tex` holds the three tables as the macros
`\minbiasCatalogueTen`, `\minbiasCatalogueSummary` and `\minbiasCataloguePerProcess`.

### The learning modes

`-v` narrates the nine steps: the scenario and the arithmetic of mu (1.57e9 collisions
per second over 2.64e7 crossings per second, 59.3, or 60.5 with 80 mb; mixed at 60; the
production's flat 55 ... 75 profile and its 1040 collisions per signal event); the
sixteen Poisson draws (60, 56, 68, 61, 66, 56, 55, 50, 53, 66, 61, 60, **63**, 63, 59,
63) and the library events read (763 distinct of 2000; ND 667, SD 162, DD 115, CD 16);
the payload of the beam spot (centre (0.0285, -0.0166, 0.416) cm, sigma_z = 3.6 cm,
beta* = 30 cm, emittance 5.725e-8 cm: sigma_xy = 9.3 um, sigma_t = 120 ps) and the
vertices drawn (r.m.s. 3.79 cm, 119 ps, 9 um; the signal at z = -0.17 cm, t = -121 ps);
the stacking crossing by crossing (651 signal particles, 9033 of the crowd within
abs(eta) < 5 without neutrinos; 860 charged in the tracker against 93); the clustering
(9684 particles + 6300 ghosts, 15 984 steps, 117 s; 173 jets, 62 above 30 GeV) and our
jet (112 particles, 31 + 81 from 34 collisions, A = 0.529); the two medians (68.28 grid,
69.87 kT; the signal alone 3.42); the subtraction of the twelve leading jets against the
truth; the visible vertices (43 of 63; the nearest 0.06 mm and 5 ps away; 4 within 1 mm,
7 within 30 ps); the profiles (one fill: mean 57, 72% levelled; the year: mean 59; the
flat profile reweighted: w from 0 to 1.62, 1.54 at 60; a fixed-mu sample: 0.07) and, for
comparison, the levelled HL-LHC fill. `-vv` adds every collision with its vertex and
every constituent of our jet with its collision, pT and distance to the axis. The seed-1
log is committed as `standalone_pileup.log`; every number quoted in the chapter's theory
is in it.

## 2. Reading the record

`event_PU.lhe` is the chapter-7 record (the listing with its sums, the vertex table, the
jet listing and the constituent listing: see the
[chapter 7 companion](chapter_07_GenJet.md)) followed by four tables of this stop.

### The summary

```
 --------  Standalone Pileup Summary  (mu = 60, window bx = -12..+3, 25 ns; library pileup_library.lhe.gz (2000 events, random); beam spot 2024; seed 1)  ----------
  mu from L sigma/(n_b f_rev): 59.35; mixed at mu = 60
  collisions per bunch crossing: bx -12: 60, bx -11: 56, bx -10: 68, bx -9: 61, bx -8: 66, bx -7: 56, bx -6: 55, bx -5: 50, bx -4: 53, bx -3: 66, bx -2: 61, bx -1: 60, bx +0: 63, bx +1: 63, bx +2: 59, bx +3: 63; 960 in all, 63 in time
  beam spot: sigma_z = 3.60 cm, sigma_t = 120 ps (model); drawn r.m.s. z 3.79 cm, t 119 ps, x 9.2 um, y 9.4 um
  signal vertex: x = 285.90 um, y = -164.13 um, z = -0.168 cm, t = -0.1214 ns
  in-time final particles: 12113 (9033 within |eta| < 5); signal 744 (651)
  rho: grid median 68.276 GeV per unit area, jet median 69.873; signal alone 3.420
  visible in-time vertices (>= 4 charged, pT > 0.5, |eta| < 2.5): 43 of 63
```

### The vertex listing

One row per collision, the signal first (k = 0), 961 rows:

```
     k   bx    lib  proc  nfin  nch  cen  vis     x[um]     y[um]     z[cm]     t[ns]
     0    0      0   sig   744  338   93    1    285.90   -164.13    -0.168   -0.1214
     1  -12    430    ND   381  192   36    1    287.81   -155.01     3.328 -299.9956
   ...
```

`k` numbers the collisions through the window (the CMSSW `EncodedEventId(bx, k)` counts
within a crossing instead), `bx` the crossing, `lib` the library event, `proc` its type,
`nfin` its final particles, `nch` its charged ones, `cen` its charged ones within
abs(eta) < 2.5 with pT > 0.5 GeV, `vis` whether it has at least four of them (the proxy
of a vertex the tracker can reconstruct), then the vertex: x and y in microns (around
the centre of the beam spot, (285, -166) um), z in centimetres, t in nanoseconds with
25 ns x bx added (the `Adjuster`'s offset). The signal row has nch = 338 charged particles
of its 744 (no neutrinos).

### The particle listing

Every final particle of the 63 in-time collisions, 12 113 rows:

```
     k       id  name                px          py          pz           e  mixjet
   608      211  pi+             0.1234     -0.4567     12.3456     12.3578      -1
```

with `mixjet` the mixed anti-kT jet that holds the particle (-1: none, or outside
abs(eta) < 5, or a neutrino). The out-of-time collisions are not listed; their particles
are reproducible from the library events of the vertex listing, and nothing at this stop
uses them.

### The jet listing and the signal constituent listing

One row per mixed jet, 173 rows, pT-ordered:

```
   mixjet        pT       A     rhoA     pTsub       sig        pu       y     eta     phi       m    n  nsig  npu ncol  genjet     dR
     mix  0   132.747   0.529   36.090    96.657   100.070    33.611  -1.431  -1.480  -2.718  45.269  112    31   81   34       0  0.042
```

`A` is the active area (ghosts x 0.01), `rhoA` the product with the grid median,
`pTsub = pT - rhoA`, `sig` and `pu` the scalar sums of the pT of the signal and of the
pileup constituents (the truth), `n`, `nsig`, `npu` their numbers, `ncol` the number of
collisions the pileup constituents came from, `genjet` the chapter-7 ak4 jet matched
within dR < 0.2 (-1 if none) and `dR` the distance to it. The signal constituent listing
has one row per signal particle that entered the clustering (651 rows): its row in the
chapter-7 listing and its mixed jet. `read_pileup_tables(path)` reads the four tables back
(the summary lines, the collisions, the particles, the jets and the signal map).

## 3. Drawing the event and rendering the film

### The (eta, phi) displays

```bash
cd example/PU
/usr/bin/python3 plot_journey_PU.py                         # -> journey_PU.pdf: the event with the crowd
/usr/bin/python3 plot_journey_PU.py --signal-only           # -> journey_PU_signal.pdf: the same frame, the family alone
/usr/bin/python3 plot_journey_PU.py --png --label-min 30
```

The layout is that of the CMSSW analyser pages and of chapter 7's `plot_journey_GenJet.py`
(stage 7, "pileup"): the signal particles coloured by the mixed jet that holds them, the
pileup particles in khaki (pale in the jet's colour when they are in a jet), the circles
of the mixed jets labelled "[i] pT -> pT - rho A", the quark -> photograph -> mixed jet
lines in the header, and the leading mixed jets with A, rho A, pT - rho A and the
signal / pileup split in the right column.

### The theory plots

```bash
cd example/PU/Standalone
/usr/bin/python3 plot_theory_PU.py                          # -> theory_pileup.pdf (4 pages, a few seconds)
/usr/bin/python3 plot_theory_PU.py --page 3 -o beamspot.png
```

Page 1: mu against the luminosity for the three scenarios, the Poisson draw with the
sixteen draws of our event, the profile of a 2024-like year against the flat 55 ... 75
profile of the production with the weight. Page 2: the multiplicity of the library by
process type, the plateau dN/deta, the pT per unit area per collision. Page 3: the
density p(z, t) with the in-time vertices and the signal, the z of the 960 vertices
against the Gaussian of the payload and the 250 vertices of the CMSSW library, the sixteen
crossings on the clock. Page 4: the grid cells of our crossing with their median, rho
against the number of collisions over the ensemble, the truth against rho A. The script
reads the record, the library, `standalone_pileup.json` and `standalone_pileup_stats.json`
(the ensemble, remade in 15 s with `--repeat 400`).

### The film

```bash
cd example/PU/Standalone
/usr/bin/python3 animate_standalone_pileup.py --print-script           # the narration, one line per phase
/usr/bin/python3 animate_standalone_pileup.py --stills animate_standalone_pileup_stills.pdf --no-video --no-narration --no-music   # the six key frames (a build product)
/usr/bin/python3 animate_standalone_pileup.py --fast -o preview.mp4    # a low-resolution preview
/usr/bin/python3 animate_standalone_pileup.py --jobs 4                 # the film, 1920x1080, narrated (edge-tts, needs the network)
```

Six scenes (crowd, road, member, median, ourjet, ensemble), 19 narrated phases and an
outro, the clips cached in `build/narration/PU/`, the subtitles next to the .mp4; the
score follows `example/soundtrack.py` with a chord per scene and a thud when the crowd is
complete. Scene 3 shows the four kinds of collision one after the other (ND, SD, DD, CD,
one library event each, the surviving protons marked), the table of their cross sections
and multiplicities filling row by row, then the multiplicity of the whole library; the
still the book prints for it is the library histogram, since the table is in the theory
section. The film picks up where chapter 7's ends and ends at the border of the detector.
The 2024 film (`--jobs 4`, about 25 minutes to render) is the one on the playlist.

## 4. Validation from official packages

### 4.1 The library: PYTHIA's cross sections and the 2024 CMSSW library

The catalogue of section 1 is the first comparison: against the summary of the companion's 250 GEN
events (interactions 7.0 mean / 3 median / 1 to 39; stable 194 / 132 / 3 to 723; charged 89 / 59;
central 21 / 10 / 0 to 126; beyond |eta| = 5.5 39 / 37; hardest 2.2 / 1.8 / 0.2 to 8.1 GeV; sum pT 79 /
45 / 0.5 to 376; jets above 10 GeV 0.15 per event; protons at the PPS 0.37 per event; per kind ND
9.0 interactions / 241 stable / 27 central, SD 2.0 / 70 / 7, DD 1.2 / 57 / 2, CD 2.0 / 35 / 2) the
library gives 6.0 / 2 / 1 to 45; 179 / 116 / 3 to 829; 83 / 54; 19 / 8 / 0 to 167; 37 / 34; 2.2 / 1.7 /
0.1 to 17.9; 73 / 38 / 0.5 to 518; 0.15; 0.38; ND 8.1 / 230 / 25, SD 1.0 / 51 / 4, DD 1.0 / 62 / 4, CD
1.0 / 28 / 2. The interactions differ in the counting (CMSSW counts one plus the status-31 pairs, the
library PYTHIA's `nMPI`, at least one), the diffractive means to what 45, 19 and 3 events allow, and the
rest within the statistics of 250 against 2000.

The library is itself PYTHIA's product (`make_pileup_library.py`: `SoftQCD:inelastic`,
the CP5 tune, the CMS common settings, `Beams:eCM = 13600` as the companion's
`process.generator.comEnergy = 13600` does with the 14 TeV fragment, seed 20261), so the
check is that its composition follows the cross sections PYTHIA printed at initialisation
and that its multiplicities agree with the 250-event GEN-SIM library of the CMSSW
exercise:

```bash
cd example/PU/Standalone
../../ME/Standalone/.venv/bin/python validate_pileup_with_fastjet.py --mode library
```

| | PYTHIA sigma [mb] at 13.6 TeV | expected share | library (2000) | CMSSW library (250) |
|---|---|---|---|---|
| non-diffractive (101) | 55.42 | 70.7% | 1425 = 71.2% | 183 = 73.2% |
| single diffractive (103 + 104) | 2 x 6.43 | 16.4% | 319 = 16.0% | 45 = 18.0% |
| double diffractive (105) | 8.85 | 11.3% | 219 = 11.0% | 19 = 7.6% |
| central diffractive (106) | 1.29 | 1.6% | 37 = 1.9% | 3 = 1.2% |
| inelastic | 78.42 | | 2000 | 250 |

| per collision | library (2000) | CMSSW library (250) |
|---|---|---|
| final particles: mean, median, min, max | 179.2, 116, 3, 829 | 194, 132, 3, 723 |
| charged | 82.6 | |
| central (abs(eta) < 2.5, pT > 0.5 GeV): charged / all | 12.2 (median 6, max 97) / 19.3 (median 8, max 167) | 21 all particles (median 10, max 126) |
| hardest particle [GeV] | 2.17 (max 17.9) | 2.2 (max 8.1) |
| sum pT [GeV] | 73 (max 518) | 0.5 to 376 |
| by type: particles (central charged) | ND 230 (16.1), SD 51 (2.5), DD 62 (2.7), CD 28 (1.6) | stable ND 241, SD 70, DD 57, CD 35; primaries handed to GEANT ND 195, SD 55, DD 36, CD 32 |

The CMSSW numbers are those of `PULib.pdf` (sections 3 and 5), whose cross sections are
exactly the ones above (the same fragment, the same tune, the same energy; PYTHIA 8.311
there, 8.315 here; the 14 TeV fragment of the Phase-2 exercise gives 55.36, 6.44, 8.88,
1.29 mb: the tune, not the energy, sets them). The CMSSW library's central count includes
the neutrals and its "primaries" are the stable particles handed to GEANT after the
beam-pipe and momentum cuts; the medians and the extremes agree to what 250 events allow
(the 19 double-diffractive events are 1.7 sigma low; the 723 of its largest event against
829 here is the tail of two thousand against 250).

### 4.2 The mixed final state with FastJet

```bash
cd example/PU/Standalone
../../ME/Standalone/.venv/bin/python validate_pileup_with_fastjet.py --mode record
```

The mixed final state of the record (the 651 signal particles and the 9033 in-time
pileup particles within abs(eta) < 5 without neutrinos, exactly the input of the
standalone clustering) is clustered by FastJet 3.5.2 with anti-kT R = 0.4 and the active
area (`active_area_explicit_ghosts`, ghost area 0.01, abs(y) < 5), the
`GridMedianBackgroundEstimator(5.0, 0.55)` that CMSSW's `fixedGridRhoFastjetAll` wraps,
and the `JetMedianBackgroundEstimator` on kT jets of R = 0.4 with areas within abs(y) <
4.4. The jets agree exactly (the same particles, the same algorithm: all 173 jets with
identical constituents, the momenta to the last digit, max |dpT| = 0.0005 GeV); the grid
median agrees to the last digit (68.2765 both: the standalone grid is FastJet's, in
rapidity with phi from 0; the same grid filled in eta, the older `fixedGridRhoAll`, gives
68.84); the areas differ by the placement of the ghosts (FastJet scatters its own: mean
difference +0.001, r.m.s. 0.025 over the 173 jets, max 0.07; our jet 0.529 both); the jet
median, which is measured on the ghost jets themselves, moves by a percent with the
ghosts (FastJet 70.43 against the standalone's 69.87). The result is
`validate_pileup_with_fastjet_record.json`.

### 4.3 CMSSW: the 2024 mixing

The CMSSW side of this stop is the exercise of JitJetSW (`test/PU_2024/`,
`runPileup_mc2024_BBbar.sh`, CMSSW_15_0_5, conditions `auto:phase1_2024_realistic`, era
`Run3_2024`, geometry `DB:Extended`, beam spot `DBrealistic`, 13.6 TeV), written up in
`PULib.pdf` (the library: 250 minimum-bias GEN-SIM events, the fragment, the 2024 vertex
smearing, the seeds 20241 / 20242 / 20243, the products, the statistics, one event drawn
on the detector) and `PUSIM.pdf` (the mixing: the MixingModule, the CrossingFrames, one
signal event mixed at mu = 60 in bx = -1 ... +1 with the library read in sequence, the
three samples drawn station by station, the scaling to the production window). The
numbers the chapter quotes from them: the beam spot (BetafuncEvtVtxGenerator with the
payload `SimBeamSpot_Realistic25ns13p6TeVEarly2024CollisionVtxSmearingParameters_v1_mc`:
sigma_z = 3.6 cm, beta* = 30 cm, emittance 5.725e-8 cm, sigma_t = sigma_z / c = 120 ps;
the 250 events: 3.56 cm, 122 ps, z from -9.52 to +10.08 cm); the draws of the mixed event
(66, 55, 77 collisions for an expectation of 60 per crossing, 198 in all; 31 597 primaries
against 571 of the signal, 160 per collision; 98 693 SimTracks; 45 890 TrackingParticles
of which 910 of the signal); the signal vertex (z = 3.11 cm, t = -0.14 ns; three pileup
vertices within 1 mm of it); the hits (454 311 tracker PSimHits of which 9776 of the
signal; 106 581 ECAL cells of which 2304 signal-dominated; 11 049 HCAL cells; 204 muon
hits, none of the signal); the production scenario (`Run3_Flat55To75_PoissonOOTPU`,
bx = -12 ... +3, about 1040 collisions per signal event, 0.25 GB of library read per
signal event, which is why the production premixes); the cost (2.45 s of GEANT per
minimum-bias event, 60 MB for the 250, 82 minutes for the mixing job with the drawing);
and the energy estimate (62 GeV of ECAL-station energy per collision, about 55 GeV of
in-time pileup under a jet of area 0.5). The standalone script reproduces the conditions
of the exercise with `--window 1 --sequential`; the draws differ (different random
numbers) and the vertex spread, the particles per collision and the per-collision density
are what can be compared.

## 5. The full tables

### 5.1 The crowd on the road obtained four ways

The table of the chapter's first section, column by column: PYTHIA's cross sections from
`pileup_library.log` (the "PYTHIA Process Initialization" block and the sigmaGen lines);
the library from `validate_pileup_with_fastjet.py --mode library`; the standalone numbers
from `standalone_pileup.log` and `standalone_pileup.json`; the FastJet numbers from
`validate_pileup_with_fastjet.py --mode record`; the 400 draws from
`standalone_pileup_stats.json` (`--repeat 400`); the CMSSW numbers from `PULib.pdf`
and `PUSIM.pdf`.

### 5.2 Our jet and the bbar jet before and after the crowd

From the jet listing of the record (rows `mix 0` and `mix 1`), the chapter-7 jet listing
(rows `ak4 0` and `ak4 1`), and the ensemble:

| | pT | m | n | A | rho A | pT - rho A | residual |
|---|---|---|---|---|---|---|---|
| b: photograph (ak4 [0]) | 99.830 | 14.826 | 31 | | | | |
| b: mixed jet [0] | 132.747 | 45.269 | 112 (31 + 81 from 34 collisions) | 0.529 | 36.090 | 96.657 | -3.41 (against 100.070 of signal inside; 33.611 of pileup) |
| b: the circle, 400 draws | | | 65 pileup | 0.503 | 29.67 +- 5.79 | | -1.87 +- 6.65 (pileup inside 27.80 +- 8.38) |
| bbar: photograph (ak4 [1]) | 94.093 | 17.825 | 18 | | | | |
| bbar: mixed jet [1] | 124.706 | 44.328 | 98 (18 + 80 from 36) | 0.499 | 34.047 | 90.659 | -3.64 (94.295 of signal inside; 31.256 of pileup) |
| bbar: the circle, 400 draws | | | | 0.503 | 29.67 | | -1.49 +- 5.90 (pileup inside 28.18 +- 7.70) |

### 5.3 The ensemble

`standalone_pileup.py --repeat 400 --jobs 4 --quiet --history standalone_pileup_stats.json`
(13 s): in-time collisions 59.9 +- 7.4, 7949 particles within abs(eta) < 5; rho (grid)
59.03 +- 11.51, 0.986 per collision (a linear fit: 0.939 per collision, offset +2.8, the
signal's own underlying event; sqrt(60) collisions of Poisson spread give 7.3 of the
11.5); visible vertices 37.4 +- 5.8 (62%); the nearest crowd vertex to the signal's:
median 0.98 mm in z and 2.6 ps in t, at least one within 1 mm in 51% of the draws, 8.6
within 30 ps on average; the signal vertex r.m.s. 3.74 cm and 119 ps; our jet's circle as
in 5.2, 0.46 GeV of pileup per collision inside it; the pileup inside exceeds 40 GeV in
7% of the draws and the residual exceeds 10 GeV in magnitude in 12%.

### 5.4 The energy ledger

| stage | pT [GeV] | m [GeV] | n | delta pT |
|---|---|---|---|---|
| the photograph, ak4 jet [0] (chapter 7) | 99.83 | 14.83 | 31 | |
| with the crowd: mixed jet [0], mu = 60 | 132.75 | 45.27 | 112 | +32.92 |
| after the subtraction, pT - rho A | 96.66 | | 112 | -36.09 |

The 32.92 GeV of the second line are the 33.61 GeV of pileup constituents less the
difference between the vector sum of the jet and the scalar sums of its constituents;
the third line is the scalar subtraction, which leaves the mass at its inflated value.

### 5.5 The HL-LHC crowd, for the open problems

`standalone_pileup.py --scenario ultimate` mixes the same family at mu = 200 in
bx = -3 ... +3 with the luminous region of CMSSW's HLLHCEvtVtxGenerator (two 9 cm bunches
crossing at 510 urad, crab-rotated by 380 urad at 400 MHz, beta* = 20 cm: sigma_z =
4.27 cm, sigma_t = 193 ps against the generator's nominal 4.3 cm and 190 ps); with the
14 TeV library of the Phase-2 production (`make_pileup_library.py --ecm 14000`) and seed 1
it gave, in the first version of this chapter, 1405 collisions, rho = 181.9 GeV per unit
area, our jet 99.83 -> 195.87 -> 101.53 and a residual of +7.9 +- 13.7 GeV over 400
draws; the record takes 15 minutes (`--no-area` one).
