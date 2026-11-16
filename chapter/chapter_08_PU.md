# Chapter 8 companion: The Crowd on the Road (PU)

This is the technical companion of chapter 8, *The Crowd on the Road: Pileup*. The
chapter gives the physics; this page says how to run the scripts, how to read what
they write, how each number of the official packages was obtained, and lists every
table the chapter quotes with the command that produced it. All files are in
[`example/PU/`](../example/PU/) of this repository unless stated otherwise; the CMSSW
side (the pileup library and the overlay of the in-time collisions on the signal event
at PU 60) is in [JitJetSW](https://github.com/ravindkv/JitJetSW) and is written up
there, see section 4.3.

The crowd is the crowd of Run 3 in 2025: the event of chapters 1 to 7 was born at
13.6 TeV under Run-3 conditions and stays there: 60 collisions per crossing, the 2025
beam spot (the payload of the 2025 global tag, fitted to early-2024 collisions), the
Run-3 production window bx = -12 ... +3, and, from chapter 9, the CMS detector of 2025.
The luminosity and the bunches of 2025 (L = 2.23e34 levelled, 2340 colliding bunches,
mu = 66.5 at the peak) and the year's profile (mean 64.3) are those of the release's
2025 pileup profile, `Run3_2025_LHC_Simulation_fill12p5h_levelling9h_nb2340_xSec80mb_il2p23_cfi`. The HL-LHC crowd (mu = 200 at 14 TeV, the Phase-2 luminous region) is one
option of the script, `--scenario ultimate`, kept for the exercises and the open problems.

| | |
|---|---|
| Stage | the signal of chapter 7 in a bunch crossing of 2025: the crowd, where and when it walks, rho, the areas and the subtraction |
| Standalone script | [`example/PU/Standalone/standalone_pileup.py`](../example/PU/Standalone/standalone_pileup.py) |
| The library | [`example/PU/pileup_library.lhe.gz`](../example/PU/pileup_library.lhe.gz), 2000 minimum-bias collisions at 13.6 TeV made once by [`make_pileup_library.py`](../example/PU/Standalone/make_pileup_library.py) with PYTHIA (its log: [`pileup_library.log`](../example/PU/pileup_library.log)) |
| Input | [`example/GenJet/event_GenJet.lhe`](../example/GenJet/event_GenJet.lhe), the final state and the generator-level jets of chapter 7 |
| Event record | [`example/PU/event_PU.lhe`](../example/PU/event_PU.lhe) (the chapter-7 record plus the summary, the vertex listing of the 960 collisions, the particles of the in-time crossing, the mixed jets and the signal constituents; 1.7 MB) |
| Learning-mode log | [`example/PU/Standalone/standalone_pileup.log`](../example/PU/Standalone/standalone_pileup.log) (`-v`, seed 1) and its summary [`standalone_pileup.json`](../example/PU/Standalone/standalone_pileup.json) |
| Validation with FastJet and PYTHIA | [`example/PU/Standalone/validate_pileup_with_fastjet.py`](../example/PU/Standalone/validate_pileup_with_fastjet.py) |
| The catalogue of the library | [`example/PU/Standalone/minbias_catalogue.py`](../example/PU/Standalone/minbias_catalogue.py), its tables [`minbias_catalogue.txt`](../example/PU/Standalone/minbias_catalogue.txt) / `.json` / `.tex` (the chapter `\input`s the last) |
| The ledger of the overlay | [`example/PU/Standalone/overlay_ledger.py`](../example/PU/Standalone/overlay_ledger.py), the twin of the companion's `genLedger.py --pu`: [`overlay_ledger.txt`](../example/PU/Standalone/overlay_ledger.txt) / `.tex` (the chapter `\input`s the last as `tab:crowd_on_the_road:overlay_ledger`) |
| CMSSW figure | [`example/PU/CMSSW/genJetAnalyzer_08_PU.pdf`](../example/PU/CMSSW/genJetAnalyzer_08_PU.pdf), the lower panel of the chapter's display |
| CMSSW write-up (JitJetSW) | physics details: [`chapter_08_PU.tex`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_08_PU.tex); technical companion: [`chapter_08_PU.md`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_08_PU.md) |
| Film | scene 8 of the [JitJet playlist](https://www.youtube.com/playlist?list=PLbSW20iyZFd4) |

Contents: [1. Running the standalone script](#1-running-the-standalone-script) ·
[2. Reading the record](#2-reading-the-record) ·
[3. Drawing the event and rendering the film](#3-drawing-the-event-and-rendering-the-film) ·
[4. Validation from official packages](#4-validation-from-official-packages) ·
[5. The full tables](#5-the-full-tables)

## 1. Running the standalone script

`standalone_pileup.py` takes the final state of chapter 7 and puts it into a bunch
crossing of 2025, with every piece of CMSSW's mixing written out: mu from the
luminosity, mu = L sigma_inel / (n_b f_rev); the Poisson draw of the MixingModule for
every crossing of the production window bx = -12 ... +3; the library of minimum-bias
collisions it reads in order from a random position; the beam spot of CMSSW's BetafuncEvtVtxGenerator
(the SimBeamSpot payload of the 2025 global tag: a Gaussian of 3.6 cm along the beam, an
independent Gaussian of 120 ps in time, 9 um across with the hourglass of beta* = 30 cm)
and the 25 ns offsets of the out-of-time crossings; the stacking of the in-time collisions
onto the signal; anti-kT on the mixed final state with the active area of every jet
measured with ghosts, using FastJet's tiled search; the median energy density rho two
ways (FastJet's grid, which CMSSW's fixedGridRhoFastjetAll wraps, and the kT-jet median of
Cacciari and Salam); the subtraction pT - rho A against the truth; the vertices the
tracker will see; the flat pileup profile of the Run-3 production, the 2025 profile of
CMSSW (`PROFILE_2025`, the probValue of the release's cfi) and the reweighting. Seven hundred lines of Python that need NumPy and nothing else; it
imports chapter 7's `standalone_genjets.py` for the record, the clustering and the jet
tables.

A map of the file, with the steps of the chapter's flow chart:

| Function or class | What it does |
|---|---|
| `mu_of`, `SCENARIOS`, `parse_window` | step 1: mu = L sigma_inel / (n_b f_rev), the three scenarios (`run3`, `nominal`, `ultimate`) with their bunches, windows and vertex generators |
| `Library` | the minimum-bias collisions of the library file, with their multiplicities |
| `Stage.draw_crowd` | step 2: the Poisson draw per crossing and the library events read, in order from a random position, wrapping at the end (CMSSW's `readOneRandom`: no collision twice in an event) |
| `BeamSpot` (`LuminousRegion` for `--region hllhc`) | step 3: the 2025 beam spot of BetafuncEvtVtxGenerator (the HL-LHC overlap of two crabbed bunches), its moments, and the drawing of vertices |
| `Stage.stack` | step 4: the mixed final state (signal + in-time crowd, abs(eta) < 5, no neutrinos) |
| `TiledClustering`, `MixedJets`, `ghost_grid` | step 5: chapter 7's anti-kT with FastJet's tiled search, the ghosts, every constituent tagged by its collision |
| `grid_rho`, `jet_median_rho` | step 6: the two estimators of rho |
| `Stage.subtract` | step 7: pT - rho A for every jet against the signal pT inside it (`residual`), and the pileup pT inside against rho A (`residual_pileup`, the ensemble's definition) |
| `Stage.visible_vertices`, `release_profile`, `profile_stats`, `fill_profile`, `flat_profile`, `reweight` | step 8: the collisions the tracker can see; the profiles (the 2025 profile of CMSSW, one fill of its header, the flat band) and the weight; `year_profile`, a year of modelled fills, is kept for the exercises |
| `pileup_tables`, `read_pileup_tables` | step 9: the tables of the record and their reader |
| `stage_summary`, `one_repeat` | the summary JSON and the worker of `--repeat` |

```bash
cd example/PU/Standalone
/usr/bin/python3 standalone_pileup.py -v --log standalone_pileup.log --summary standalone_pileup.json
      # -> ../event_PU.lhe, the book's record (mu = 60, bx = -12 ... +3, seed 1): 2 min, almost all of it the anti-kT clustering
/usr/bin/python3 standalone_pileup.py --no-area              # 20 s: no ghosts, no areas, no jet-median rho
/usr/bin/python3 standalone_pileup.py --window 0             # the in-time crossing only
/usr/bin/python3 standalone_pileup.py --sequential           # the conditions of the CMSSW draw (PU 60 in bx = -12 ... +3, the library read from its first event)
/usr/bin/python3 standalone_pileup.py --scenario ultimate    # the HL-LHC crowd: mu = 200 in bx = -3 ... +3, the Phase-2 luminous region (15 min)
/usr/bin/python3 standalone_pileup.py --repeat 400 --jobs 4 --quiet --history standalone_pileup_stats.json   # 15 s
```

Options: `--scenario run3|nominal|ultimate` (the luminosity, the colliding bunches, the
window, the vertex generator and the rounded mu CMSSW mixes at: 60 in bx = -12 ... +3 with
the 2025 beam spot; 140 and 200 in bx = -3 ... +3 with the HL-LHC luminous region), `--mu`,
`--window N|MIN,MAX` and `--region run3|hllhc` to override them, `--sequential` to read
the library from its first event on (CMSSW's `sequential = True`), `--seed`, `--radius`,
`--ptmin`, `--no-area`, `--repeat N --jobs J --history FILE` for the ensemble (the
in-time crowd drawn N times on the photograph; no clustering: the grid rho, with and
without the signal, and the pileup inside the circle of R = 0.4 in (y, phi) around our jet
and the bbar jet), `--summary FILE`
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

The twin of the companion's `minBiasGenStats.py`, which prints one line per GEN event of the 2025 CMSSW
library: for every collision the kind, the interactions, the stable particles, the charged ones, the
central ones (|eta| < 2.5 and pT > 0.5 GeV, all species, as the CMSSW catalogue counts them), the ones
beyond |eta| = 5.5, the pT of the hardest particle, the scalar sum of pT, the anti-kT R = 0.4 jets above
10 GeV (chapter 7's clustering on the collision alone, without neutrinos) and the forward protons (final
protons with xi = 1 - E/E_beam < 0.55, the ones CMSSW's `LHCTransport` carries to the PPS). The listing
of the chapter is the first ten in-time collisions of our crossing with the vertex the mixing drew for
each (the record's vertex listing); the summary is over the two thousand collisions (the vertex rows over
the 960 drawn); the means per kind close it. The `.tex` holds the three tables as the macros
`\minbiasCatalogueTen`, `\minbiasCatalogueSummary` and `\minbiasCataloguePerProcess`.

### The ledger of the overlay

```bash
cd example/PU/Standalone
/usr/bin/python3 overlay_ledger.py                 # -> overlay_ledger.txt, .tex (3 s; reads ../event_PU.lhe and ../pileup_library.lhe.gz)
```

The twin of `genLedger.py --pu` of the CMSSW companion (its `ledger_08_PU.txt`): the
final state of the signal, of the 63 in-time collisions, of the 897 out-of-time collisions
(never overlaid; rebuilt from their library events) and of the signal with its in-time
pileup, by kind (charged
hadrons, neutral hadrons, photons, electrons, muons, neutrinos) and by region of |eta|
(2.5, 3.0, 5.2, 5.5), with the visible part (|eta| < 5.2, no neutrinos) and, for the
overlaid blocks, the cone of Delta R < 0.4 around the last copy of each b quark; then the jets before (chapter 7's
ak4GenJets) and after (the mixed jets of the record): the counts above 3 / 20 / 30 / 50
GeV, the signal particles inside a jet, the b and bbar jets with their signal and pileup
constituents, their response, their distance to the quark in (eta, phi), the dijet mass,
the imbalance of the pair and the jets without a signal particle. The signal is read from
the chapter-7 listing the record keeps, the in-time crowd from its particle listing, the
out-of-time crowd from `../pileup_library.lhe.gz` by the library events of its vertex
listing, the jets from the two jet listings. The text keeps the energies; the `.tex`, like
the CMSSW table since 16 November 2026, gives n and scalar pT only and holds the two tables of the
chapter as the macros `\puOverlayLedgerFinal` and `\puOverlayLedgerJets`; the text is in
section 5.5.

### The learning modes

`-v` narrates the nine steps: the scenario and the arithmetic of mu (L = 2.23e34 and
2340 bunches from the 2025 profile's header: 1.75e9 collisions per second over 2.63e7
crossings per second, 66.5, or 67.8 with 80 mb; mixed at 60, as the companion; the
production's flat 55 ... 75 profile and its 1040 collisions per signal event); the
sixteen Poisson draws (60, 56, 68, 61, 66, 56, 55, 50, 53, 66, 61, 60, **63**, 63, 59,
63) and the library events read (in order from event 430, 960 distinct; ND 698, SD 158,
DD 86, CD 18);
the payload of the beam spot (centre (0.0285, -0.0166, 0.416) cm, sigma_z = 3.6 cm,
beta* = 30 cm, emittance 5.725e-8 cm: sigma_xy = 9.3 um, sigma_t = 120 ps) and the
vertices drawn (r.m.s. 3.56 cm, 124 ps, 9 um; the signal at z = +0.16 cm, t = +94 ps);
the stacking crossing by crossing (651 signal particles, 10 045 of the crowd within
abs(eta) < 5 without neutrinos; 995 charged in the tracker against 93); the clustering
(10 696 particles + 6300 ghosts, 16 996 steps, 148 s; 175 jets, 82 above 30 GeV)
and our jet (mixed jet [1], 115 particles, 31 + 84 from 33 collisions, A = 0.529, 0.012
from the photograph's axis in (y, phi)); the medians (75.47 grid, 78.50 kT; the signal
alone 3.42 with a mean of 11.08 and 17 of 198 cells empty; the crowd alone 69.93); the
subtraction of the twelve leading jets against the truth, with both residuals of our jet
(-6.43 on the signal's sum, -5.32 as pileup inside minus rho A; -3.50 and -2.39 with the
rho of the crowd alone); the visible vertices (40 of 63; the nearest 0.01 mm and 0.3 ps
away; 3 within 1 mm, 7 within 30 ps); the profiles (one fill of the 2025 header,
levelled at 67.8 for 9 h of 12.5: mean 65, 76% levelled; the 2025 profile of CMSSW:
mean 64.3, r.m.s. 11.7, median 65, 18% below 55 and 16% above 75; the flat profile
reweighted to it: w from 0.45 to 0.78, 0.65 at 60, nothing for the 34% outside the band;
a fixed-mu sample: one bin, it cannot be reweighted) and, for comparison, the levelled
HL-LHC fill. `-vv` adds every collision
with its vertex and every constituent of our jet with its collision, pT and distance to
the axis in (y, phi). The seed-1 log is committed as `standalone_pileup.log`; every number
quoted in the chapter's theory is in it.

## 2. Reading the record

`event_PU.lhe` is the chapter-7 record (the listing with its sums, the vertex table, the
jet listing and the constituent listing: see the
[chapter 7 companion](chapter_07_GenJet.md)) followed by four tables of this stop.

### The summary

```
 --------  Standalone Pileup Summary  (mu = 60, window bx = -12..+3, 25 ns; library pileup_library.lhe.gz (2000 events, random); beam spot 2025; seed 1)  ----------
  mu from L sigma/(n_b f_rev): 66.46; mixed at mu = 60
  collisions per bunch crossing: bx -12: 60, bx -11: 56, bx -10: 68, bx -9: 61, bx -8: 66, bx -7: 56, bx -6: 55, bx -5: 50, bx -4: 53, bx -3: 66, bx -2: 61, bx -1: 60, bx +0: 63, bx +1: 63, bx +2: 59, bx +3: 63; 960 in all, 63 in time
  beam spot: sigma_z = 3.60 cm, sigma_t = 120 ps (model); drawn r.m.s. z 3.56 cm, t 124 ps, x 8.9 um, y 9.6 um
  signal vertex: x = 287.13 um, y = -155.09 um, z = 0.157 cm, t = 0.0940 ns
  in-time final particles: 13456 (10045 within |eta| < 5); signal 744 (651)
  rho: grid median 75.468 GeV per unit area, jet median 78.499; signal alone 3.420, crowd alone 69.933
  visible in-time vertices (>= 4 charged, pT > 0.5, |eta| < 2.5): 40 of 63
```

### The vertex listing

One row per collision, the signal first (k = 0), 961 rows:

```
     k   bx    lib  proc  nfin  nch  cen  vis     x[um]     y[um]     z[cm]     t[ns]
     0    0      0   sig   744  338   93    1    287.13   -155.09     0.157    0.0940
     1  -12    430    ND   381  192   36    1    283.49   -171.07    -2.985 -300.3093
   ...
```

`k` numbers the collisions through the window (the CMSSW `EncodedEventId(bx, k)` counts
within a crossing instead), `bx` the crossing, `lib` the library event (430 to 1389 in
order: the library is opened at a random event and read on), `proc` its type,
`nfin` its final particles, `nch` its charged ones, `cen` its charged ones within
abs(eta) < 2.5 with pT > 0.5 GeV, `vis` whether it has at least four of them (the proxy
of a vertex the tracker can reconstruct), then the vertex: x and y in microns (around
the centre of the beam spot, (285, -166) um), z in centimetres, t in nanoseconds with
25 ns x bx added (the `Adjuster`'s offset). The signal row has nch = 338 charged particles
of its 744 (no neutrinos).

### The particle listing

Every final particle of the 63 in-time collisions, 13 456 rows:

```
     k       id  name                px          py          pz           e  mixjet
   713     -211  pi-                 0.5799     -0.2054      6.6442      6.6741     104
```

with `mixjet` the mixed anti-kT jet that holds the particle (-1: none, or outside
abs(eta) < 5, or a neutrino). The out-of-time collisions are not listed; their particles
are reproducible from the library events of the vertex listing, and nothing at this stop
uses them.

### The jet listing and the signal constituent listing

One row per mixed jet, 175 rows, pT-ordered (ours is the second in this crossing):

```
  mixjet        pT       A     rhoA     pTsub       sig        pu       y     eta     phi       m    n  nsig  npu ncol  genjet     dR
 mix   0   136.774   0.489   36.881    99.893    94.329    43.420  -3.429  -3.476   0.580  43.093  101    19   82   33       1  0.014
 mix   1   133.533   0.529   39.891    93.642   100.070    34.573  -1.434  -1.479  -2.701  43.206  115    31   84   33       0  0.012
```

`A` is the active area (ghosts x 0.01), `rhoA` the product with the grid median,
`pTsub = pT - rhoA`, `sig` and `pu` the scalar sums of the pT of the signal and of the
pileup constituents (the truth), `n`, `nsig`, `npu` their numbers, `ncol` the number of
collisions the pileup constituents came from, `genjet` the chapter-7 ak4 jet matched
within dR < 0.2 in (y, phi), the plane of the clustering (-1 if none), and `dR` the
distance to it (in pseudorapidity our jet would seem to have moved by 0.04, only
because its mass grew from 14.8 to 43 GeV at fixed rapidity). The signal constituent listing
has one row per signal particle that entered the clustering (651 rows): its row in the
chapter-7 listing and its mixed jet. `read_pileup_tables(path)` reads the four tables back
(the summary lines, the collisions, the particles, the jets and the signal map).

## 3. Drawing the event and rendering the film

### The (eta, phi) displays

Since 2026-11-14 the chapter shows this display on one page with the CMSSW one below it (`\ledgerdisplays` of `preamble.tex`: standalone on top, [`CMSSW/genJetAnalyzer_08_PU.pdf`](../example/PU/CMSSW/genJetAnalyzer_08_PU.pdf) at the bottom, each on half a page). `journey_PU_signal.pdf`, the same frame without the crowd, is still written by `make figures` but is no longer shown in the chapter, which refers to the photograph of chapter 7 instead.

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
sixteen draws of our event, the 2025 profile of CMSSW and one fill of its header against
the flat 55 ... 75 profile of the production, with the weight. Page 2: the multiplicity of the library by
process type, the plateau dN/deta, the pT per unit area per collision. Page 3: the
density p(z, t) with the in-time vertices and the signal, the z of the 960 vertices
against the Gaussian of the payload and the first 250 vertices of the CMSSW library, the sixteen
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
and multiplicities filling row by row with, below it, the panel of the chapter's sketch
(`chapter/tikz/pu_minbias.tex`) for the kind being told, then the four sketches side by
side under the full table, then the multiplicity of the whole library. The panels are
compiled from the TikZ once (`\def\puminbiaspanel{0..3}` cuts the bounding box to one
panel) and rendered with ghostscript into `build/pu_minbias_panels/`; without pdflatex or gs
the scene is drawn without them. The
still the book prints for it is the library histogram, since the table is in the theory
section. The film picks up where chapter 7's ends and ends at the border of the detector.
The 2025 film (`--jobs 8`, rendered on 15 November 2026) is the one on the playlist.

## 4. Validation from official packages

### 4.1 The library: PYTHIA's cross sections and the 2025 CMSSW library

The catalogue of section 1 is the first comparison: against the summary of the companion's 1000 GEN
events (`genstats_08_PU.txt`: interactions 6.6 mean / 3 median / 1 to 45; stable 188 / 119 / 3 to 857;
charged 87 / 56; central 20 / 9 / 0 to 167; beyond |eta| = 5.5 38 / 35; hardest 2.2 / 1.7 / 0.16 to
19.0 GeV; sum pT 76 / 41 / 0.3 to 498; jets above 10 GeV 0.15 per event; protons at the PPS 0.36 per
event; per kind ND 8.5 interactions / 236 stable / 26 central, SD 2.1 / 70 / 7, DD 1.3 / 55 / 3, CD
1.2 / 24 / 2) the library gives 6.0 / 2 / 1 to 45; 179 / 116 / 3 to 829; 83 / 54; 19 / 8 / 0 to 167;
37 / 34; 2.2 / 1.7 / 0.1 to 17.9; 73 / 38 / 0.5 to 518; 0.15; 0.38; ND 8.1 / 230 / 25, SD 1.0 / 51 /
4, DD 1.0 / 62 / 4, CD 1.0 / 28 / 2. The interactions differ in the counting (CMSSW counts one plus
the status-31 pairs, the library PYTHIA's `nMPI`, at least one). The non-diffractive and
double-diffractive multiplicities agree within the statistics (236 against 230, 0.8 standard errors;
55 against 62, 1.7), and so does almost everything else of 1000 against 2000; the one difference
beyond them is the single-diffractive multiplicity, 70 stable particles per collision in CMSSW
against 51 here (161 and 319 collisions, an r.m.s. of 51 per collision: about 3.8 standard errors),
which this page records and does not explain (PYTHIA 8.311 and the 14 TeV fragment there, 8.315
and `Beams:eCM = 13600` here).

The library is itself PYTHIA's product (`make_pileup_library.py`: `SoftQCD:inelastic`,
the CP5 tune, the CMS common settings, `Beams:eCM = 13600` as the companion's
`process.generator.comEnergy = 13600` does with the 14 TeV fragment, seed 20261), so the
check is that its composition follows the cross sections PYTHIA printed at initialisation
and that its multiplicities agree with the 1000-event GEN library of the CMSSW
exercise:

```bash
cd example/PU/Standalone
../../ME/Standalone/.venv/bin/python validate_pileup_with_fastjet.py --mode library
```

| | PYTHIA sigma [mb] at 13.6 TeV | expected share | library (2000) | CMSSW library (1000) |
|---|---|---|---|---|
| non-diffractive (101) | 55.42 | 70.7% | 1425 = 71.2% | 722 = 72.2% |
| single diffractive (103 + 104) | 2 x 6.43 | 16.4% | 319 = 15.9% | 161 = 16.1% |
| double diffractive (105) | 8.85 | 11.3% | 219 = 10.9% | 105 = 10.5% |
| central diffractive (106) | 1.29 | 1.6% | 37 = 1.9% | 12 = 1.2% |
| inelastic | 78.42 | | 2000 | 1000 |

| per collision | library (2000) | CMSSW library (1000) |
|---|---|---|
| final particles: mean, median, min, max | 179.2, 116, 3, 829 | 187.7, 119, 3, 857 |
| charged | 82.6 | 86.6 (median 56) |
| central (abs(eta) < 2.5, pT > 0.5 GeV): charged / all | 12.2 (median 6, max 97) / 19.3 (median 8, max 167) | 20.4 all particles (median 9, max 167) |
| hardest particle [GeV] | 2.17 (max 17.9) | 2.21 (max 19.0) |
| sum pT [GeV] | 73 (max 518) | 76.5 (0.3 to 498) |
| by type: particles (central charged) | ND 230 (16.1), SD 51 (2.5), DD 62 (2.7), CD 28 (1.6) | stable (central, all species) ND 236 (26), SD 70 (7), DD 55 (3), CD 24 (2) |

The CMSSW numbers are those of the JitJetSW companion (`genstats_08_PU.txt`), whose cross sections are
exactly the ones above (the same fragment, the same tune, the same energy; PYTHIA 8.311
there, 8.315 here; the 14 TeV fragment of the Phase-2 exercise gives 55.36, 6.44, 8.88,
1.29 mb: the tune, not the energy, sets them). The CMSSW library's central count includes
the neutrals. Its shares are all within 1.1 standard deviations of the expectation (the
first 250 events alone had a double-diffractive deficit of 1.7, which the 1000 wash out);
the medians and the extremes agree to what 1000 events allow (the 857 of its largest event
against 829 here is the tail). The first 250 events of the 1000 are the 250-event library
of the earlier exercise, collision by collision (the same seeds give the same events
whatever the length of the run), and they remain the library of chapters 10 to 13 of
JitJetSW until those are moved to the sixteen crossings (chapter 9, Geant4, takes all 1000).

### 4.2 The mixed final state with FastJet

```bash
cd example/PU/Standalone
../../ME/Standalone/.venv/bin/python validate_pileup_with_fastjet.py --mode record
```

The mixed final state of the record (the 651 signal particles and the 10 045 in-time
pileup particles within abs(eta) < 5 without neutrinos, exactly the input of the
standalone clustering) is clustered by FastJet 3.5.2 with anti-kT R = 0.4 and the active
area (`active_area_explicit_ghosts`, ghost area 0.01, abs(y) < 5), the
`GridMedianBackgroundEstimator(5.0, 0.55)` that CMSSW's `fixedGridRhoFastjetAll` wraps,
and the `JetMedianBackgroundEstimator` on kT jets of R = 0.4 with areas within abs(y) <
4.4. The jets agree exactly (the same particles, the same algorithm: all 175 jets with
identical constituents, the momenta to the last digit, max |dpT| = 0.0005 GeV); the grid
median agrees to the last digit (75.4682 both: the standalone grid is FastJet's, in
rapidity with phi from 0; the same grid filled in eta, the older `fixedGridRhoAll`, gives
75.07); the areas differ by the placement of the ghosts (FastJet scatters its own: mean
difference -0.000, r.m.s. 0.026 over the 175 jets, max 0.07; our jet 0.539 against 0.529);
the jet median, which is measured on the ghost jets themselves, moves by a percent with
the ghosts (FastJet 79.53 against the standalone's 78.50). The same placement noise is in
the script's own areas: re-clustering the record with five ghost seeds gives our jet
A = 0.519, 0.529 (the record), 0.539, 0.469 and 0.529, the jets and their constituents
unchanged, and a rigid circle of R = 0.4 on the script's jittered ghost grid measures
0.505 +- 0.021; at rho = 75 that is +- 1.5 GeV on pT - rho A. The result is
`validate_pileup_with_fastjet_record.json`.

### 4.3 CMSSW: the library and the overlay

The CMSSW side of this stop is one job of
[JitJetSW](https://github.com/ravindkv/JitJetSW), `test/08_PU/run_08_PU.sh`
(CMSSW_15_0_5, 13.6 TeV, run 1, event 1 with the default seeds): a library of 1000 minimum-bias collisions, the draw of the mixing over bx = -12 ... +3, and the overlay of the 64 in-time collisions on the signal at generator level.
Its commands, configurations, logs and tables are documented there, and this
page does not repeat them:

- physics details: [`chapter_08_PU.tex`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_08_PU.tex): what the stage does to the event, the numbers of
  the event, its ledger and display;
- technical companion: [`chapter_08_PU.md`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_08_PU.md): the `cmsDriver.py` commands, the `cmsRun` on the
  python configuration, the analyser configurations and every table in full.

```bash
cmsrel CMSSW_15_0_5 && cd CMSSW_15_0_5/src && cmsenv
git clone git@github.com:ravindkv/JitJetSW.git && scram b
cd JitJetSW/Journey/test
./08_PU/run_08_PU.sh        # cmsDriver.py, cmsRun, the analyser, the ledger; outputs in 08_PU/out/
```

The page of the analyser that the chapter shows is copied to
[`example/PU/CMSSW/genJetAnalyzer_08_PU.pdf`](../example/PU/CMSSW/genJetAnalyzer_08_PU.pdf);
the links are also in [`example/PU/CMSSW/README.md`](../example/PU/CMSSW/README.md).

The job has three parts. The library is one GEN job of 1000 minimum-bias collisions
(`MinBias_14TeV_pythia8_TuneCP5_cfi` at 13.6 TeV, `SoftQCD:inelastic`, seeds 20241 /
20242 so that its vertices are not those of the signal); JitJetSW runs it with the
2025 conditions of the release, which at generator level give the same library as the
2024 ones collision by collision (the beam-spot payload of the 2025 global tag is the same tag as in 2024,
`SimBeamSpot_Realistic25ns13p6TeVEarly2024CollisionVtxSmearingParameters_v1_mc`:
sigma_z = 3.6 cm, sigma_t = 120 ps). The draw (`common/pileupDraw_cfg.py` and
`pileupDraw.py`, `draw_08_PU.txt`) runs the `MixingModule` of the overlay with the
settings of the scenario, `AVE_60_BX_25ns` with seed 12345 over the window of every Run-3
scenario of the release, `minBunch = -12` to `maxBunch = 3`, the library read in sequence,
early crossings first: 995 collisions, 746 for the twelve early crossings (library
events 1-746), 64 in time (747-810) and 185 for the three late ones (811-995). The overlay
runs the `MixingModule` with no digitiser, `mixHepMC` only, on the signal of chapter 7
and library events 747 to 810 (`skipEvents = 746`, `type = fixed`, 64 collisions, the
single crossing 0), and clusters the sum as `ak4GenJetsPU`.

What the chapter reads from the output (`test/08_PU/out/`):

| number in the chapter | value | file |
|---|---|---|
| cross sections, process counts of the 1000 | 78.42 mb; 722 ND / 161 SD / 105 DD / 12 CD (706.7 / 164.0 / 112.8 / 16.4 expected) | `step1_GEN_MinBias_08_PU.log` |
| per collision: interactions, stable particles, central ones, hardest | 6.6; 187.7 (median 119, 3 to 857); 20.4; 2.21 GeV (max 18.99) | `genstats_08_PU.txt` |
| the 1000 stable-particle species, pT and ND multiplicity against interactions | 187 684 particles: photons 45.7%, charged pions 37.3%, ... 77 above 5 GeV; ND 1 to 45 interactions, mean 8.5; 87 stable for one, 805 for 45 | `genstats_08_PU.txt` |
| vertices of the 1000: r.m.s. in z and t, range in z | 3.57 cm, 120 ps; -12.4 to +12.9 cm | `genstats_08_PU.txt` |
| signal vertex | z = 3.11 cm, t = +0.121 ns | `step1_GEN_01_ME.root` of chapter 1 (the same in every job) |
| the draw over bx = -12 ... +3 | 71, 68, 54, 59, 52, 61, 76, 59, 65, 59, 68, 54, **64**, 63, 62, 60: 746 early, 64 in time, 185 late, 995 | `draw_08_PU.txt` |
| the 64 in-time collisions | 41 ND / 12 SD / 11 DD, 350 interactions, 10 893 stable particles (4 to 826, median 116) | `genstats_08_PU.txt`, events 747 to 810 |
| library event 747, the first in-time collision | SD (103), 37 stable particles, none central; the surviving proton at eta = -11.97 | `genPartAnalyzer_MinBias_08_PU.log` |
| pileup at abs(eta) < 2.5 | 4252 particles, 1922.1 GeV of scalar pT: 61.2 GeV per unit area, 30.8 GeV per circle of R = 0.4 | `ledger_08_PU.txt` |
| pileup energy | 870 400 GeV, 92% beyond abs(eta) = 5.5 | `ledger_08_PU.txt` |
| the out-of-time pileup (bx -12 ... -1, +1 ... +3; not overlaid) | 931 collisions, 176 266 stable particles, 71 747.8 GeV of scalar pT (189 and 77 per collision) | `ledger_08_PU.txt` (`genLedger.py --events` on the library) |
| jets above 3 / 20 / 30 / 50 GeV | 261 / 99 / 45 / 7 (27 / 2 / 2 / 2 without pileup) | `genJetPileup_08_PU.log` |
| the b jet with the crowd | jet [0], 150.752 GeV, mass 50.53 GeV, 101 constituents (23 + 78 from 33 collisions), pileup pT 34.22 GeV | `genJetPileup_08_PU.log`, FWLite |
| the bbar jet with the crowd | jet [1], 144.688 GeV, mass 42.87 GeV, 86 constituents (18 + 68 from 30 collisions), pileup pT 29.51 GeV | the same |
| dijet mass, signal / overlay | 386.8 / 505.2 GeV | `genJetPileup_08_PU.log` |
| jets without a signal particle; signal particles in a jet | 79 of 261; 626 of 634 (239 before) | FWLite on `step2_GENJETPU_08_PU.root` |

The mixing with the detector simulation (the 193 collisions of the three crossings
-1 ... +1, their 31 060 primaries, 438 043 tracker hits and 103 808 pileup-dominated ECAL
cells quoted in the chapter's theory section) is chapter 10 of JitJetSW
([`chapter_10_SIM.tex`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_10_SIM.tex)),
which has not been moved to the sixteen crossings yet: it reads library events 1 to 193
(in time 72 to 139) of the first 250, and the same seed and mean give it the 71, 68 and
54 collisions that the long window draws for its first three crossings, -12, -11 and
-10. The standalone script takes the conditions of the draw with `--sequential`; the
draws differ (different random numbers and a different library), and the vertex spread,
the particles per collision and the per-collision density are what can be compared.

## 5. The full tables

### 5.1 The crowd on the road obtained four ways

The table of the chapter's first section, column by column: PYTHIA's cross sections from
`pileup_library.log` (the "PYTHIA Process Initialization" block and the sigmaGen lines);
the library from `validate_pileup_with_fastjet.py --mode library`; the standalone numbers
from `standalone_pileup.log` and `standalone_pileup.json`; the FastJet numbers from
`validate_pileup_with_fastjet.py --mode record`; the 400 draws from
`standalone_pileup_stats.json` (`--repeat 400`); the CMSSW numbers from the JitJetSW
job of section 4.3 (the table there names the file of each).

### 5.2 Our jet and the bbar jet before and after the crowd

From the jet listing of the record (rows `mix 0` and `mix 1`), the chapter-7 jet listing
(rows `ak4 0` and `ak4 1`), and the ensemble:

| | pT | m | n | A | rho A | pT - rho A | residual |
|---|---|---|---|---|---|---|---|
| b: photograph (ak4 [0]) | 99.830 | 14.826 | 31 | | | | |
| b: mixed jet [1] | 133.533 | 43.206 | 115 (31 + 84 from 33 collisions) | 0.529 | 39.891 | 93.642 | -6.43 (against 100.070 of signal inside); -5.32 as pileup inside (34.573) minus rho A |
| b: the circle, 400 draws | | | 68.1 pileup from 29.8 collisions | 0.503 | 29.45 +- 5.25 | | -1.11 +- 6.05 (pileup inside 28.34 +- 8.10); +2.01 +- 6.09 with the rho of the crowd alone |
| bbar: photograph (ak4 [1]) | 94.093 | 17.825 | 18 | | | | |
| bbar: mixed jet [0] | 136.774 | 43.093 | 101 (19 + 82 from 33) | 0.489 | 36.881 | 99.893 | +5.56 (94.329 of signal inside); +6.54 as pileup inside (43.420) minus rho A |
| bbar: the circle, 400 draws | | | 64.0 pileup from 28.6 collisions | 0.503 | 29.45 +- 5.25 | | -2.72 +- 6.07 (pileup inside 26.73 +- 7.45); +0.40 +- 6.09 with the rho of the crowd alone |

The two residuals are two definitions. The record's is the jet's pT, a vector sum, minus
rho A, minus the scalar sum of the signal particles inside; the ensemble's is the scalar
sum of the pileup inside the circle minus rho pi R^2. For one jet they differ by what the
vector sum loses to the spread of the particles (1.1 GeV for ours, 0.8 GeV on average
over seeds 1 to 9); the `-v` log and the JSON give both (`residual`, `residual_pileup`).
The circle is drawn in (y, phi) around the rapidity of the photograph's jet, the plane of
the clustering, of the rho grid and of pi R^2; the signal inside it is then exactly the
signal of the mixed jet (100.07, and 94.30 for the bbar jet, whose mixed jet picked up one
more signal particle of 0.03 GeV in this crossing).

### 5.3 The ensemble

`standalone_pileup.py --repeat 400 --jobs 4 --quiet --history standalone_pileup_stats.json`
(15 s): in-time collisions 59.9 +- 7.4, 7882 particles within abs(eta) < 5; rho (grid)
58.59 +- 10.44, 0.979 per collision (a linear fit: 0.967 per collision, offset +0.7;
sqrt(60) collisions of Poisson spread give 7.5 of the 10.4); the same grid on the crowd
alone 52.38 +- 10.25 (fit: 0.952 per collision, offset -4.6): the signal raises the
median by 6.2 +- 0.9 GeV per unit area, 3.1 GeV per jet, which is why the residual is
negative with the full rho and positive with the crowd's own; visible vertices
37.4 +- 6.0 (62%); the nearest crowd vertex to the signal's: median 0.82 mm in z and
2.5 ps in t, at least one within 1 mm in 56% of the draws, 8.3 within 30 ps on average;
the signal vertex r.m.s. 3.61 cm and 117 ps; our jet's circle as in 5.2, 0.47 GeV of
pileup per collision inside it; the pileup inside exceeds 40 GeV in 9% of the draws and
the residual exceeds 10 GeV in magnitude in 9%.

Nine full records (seeds 1 to 9, `--seed N -o` elsewhere, each clustered with ghosts) say
what one crossing can do: rho from 51.6 to 75.5; mixed jets above 30 GeV 82, 54, 48, 33,
37, 37, 64, 41, 30, of which the crowd's own 76, 49, 42, 27, 31, 31, 58, 34, 23 before the
subtraction and 1, 1, 0, 1, 0, 0, 2, 0, 0 after it (five in all, none in five crossings);
the record's residual of our jet -6.43, -5.89, -3.05, +1.81, -1.74, -2.62, -1.16, -6.77,
+0.46 (mean -2.8, r.m.s. 3.1) and of the bbar jet +5.56, -10.81, -13.96, -1.44, -5.27,
+4.04, +1.30, +2.68, -5.35 (mean -2.6, r.m.s. 6.8); the area of our jet from 0.469 to
0.529; our jet keeps its 31 particles in all nine and is the leading mixed jet in seven
(the bbar jet leads in seeds 1 and 8, and picks up one signal particle in seeds 1 and 7).

### 5.4 The energy ledger

| stage | pT [GeV] | m [GeV] | n | delta pT |
|---|---|---|---|---|
| **Standalone** (seed 1, `event_PU.lhe`) | | | | |
| the photograph, ak4 jet [0] (chapter 7) | 99.83 | 14.83 | 31 | |
| with the crowd: mixed jet [1], mu = 60 | 133.53 | 43.21 | 115 | +33.70 |
| after the subtraction, pT - rho A | 93.64 | | 115 | -39.89 |
| **CMSSW** (run 1, event 1; 64 in-time collisions of the draw over bx = -12 ... +3) | | | | |
| the photograph, ak4 jet [0] (chapter 7) | 117.79 | 16.72 | 24 | |
| with the crowd: mixed jet [0], mu = 60 | 150.75 | 50.53 | 101 | +32.96 |
| of which pileup (scalar sum) | 34.22 | | 78 | |
| minus the mean pedestal of a circle, pT - rho pi R^2 | 120.0 | | 101 | -30.8 |

The 33.70 GeV of the second line are the 34.57 GeV of pileup constituents less the
difference between the vector sum of the jet and the scalar sums of its constituents;
the third line is the scalar subtraction, which leaves the mass at its inflated value.
The CMSSW rows are lines of `genJetPileup_08_PU.log` of JitJetSW (`[0] pt=150.752 eta=-1.63804
phi=-2.07282 mass=50.5254 nConst=101 pileupPt=34.2175 (22.6978%)`); no subtraction is run in that
job, and the last row is arithmetic: the mean pileup pT per unit area at abs(eta) < 2.5,
1922.1 / (5 x 2 pi) = 61.2 GeV, times pi R^2 = 0.503 is 30.8 GeV, and 150.75 - 30.8 = 120.0 GeV,
2.2 GeV above the photograph. (Before the CMSSW draw was moved to the sixteen crossings, the
overlay took the 68 in-time collisions of the three-crossing draw, library events 72 to 139:
the b jet was then jet [1], 150.90 GeV with 114 constituents, and the subtracted value 116.0.)

### 5.5 The ledger of the overlay

`overlay_ledger.py` on `event_PU.lhe` (seed 1), the table `tab:crowd_on_the_road:overlay_ledger`
of the chapter; the CMSSW counterpart is section 4.7 of the JitJetSW companion (`ledger_08_PU.txt`).
The file (`overlay_ledger.txt`) in full but for the by-kind and by-region lines of the signal with its
in-time pileup, which are the sums of the first two blocks:

```
== overlay_ledger event_PU.lhe: signal 749 + in-time pileup 13456 final particles from 63 collisions (bx 0); out-of-time pileup 161696 final particles from 897 collisions (bx -12..-1 and 1..3, rebuilt from pileup_library.lhe.gz, not overlaid)
final state of the signal: n=749 sumE=13600.0 GeV sumPt=743.0 GeV
  by kind                  n      E[GeV]   pT[GeV]
  charged hadron         329      7188.6     420.2
  neutral hadron          79      3401.4     156.9
  photon                 327      2735.6     153.7
  electron                 7         6.6       0.9
  muon                     2        38.4       1.0
  neutrino                 5       229.4      10.2
  by region                n      E[GeV]   pT[GeV]      E_nu
  |eta|<2.5              396       982.8     418.6       2.9
  2.5-3.0                 55       378.0      51.3       0.0
  3.0-5.2                216      5180.7     241.7     122.1
  5.2-5.5                 15       827.7       8.2     104.4
  >5.5                    67      6230.9      23.1       0.0
  visible (|eta|<5.2, no neutrinos): n=664 E=6416.4 GeV pT=702.5 GeV; neutrinos: n=5 E=229.4 GeV pT=10.2 GeV
  cone Delta R < 0.4 around the last copy of each b quark (no neutrinos):
    bbar axis [708] pT=72.5 eta=-3.42 phi=0.57: n=18 E=1441.8 GeV vector pT=93.0 GeV scalar pT=93.2 GeV
    b    axis [719] pT=78.3 eta=-1.38 phi=-2.73: n=31 E=220.9 GeV vector pT=99.4 GeV scalar pT=99.7 GeV
final state of the in-time pileup (63 collisions, bx 0): n=13456 sumE=856800.0 GeV sumPt=5725.4 GeV
  by kind                  n      E[GeV]   pT[GeV]
  charged hadron        6035    535374.8    3324.7
  neutral hadron        1075    156113.9     756.3
  photon                6258    164380.6    1610.0
  electron                73       749.6      15.2
  muon                     4        20.7       5.7
  neutrino                11       160.4      13.4
  by region                n      E[GeV]   pT[GeV]      E_nu
  |eta|<2.5             5229      5937.1    2440.0      21.6
  2.5-3.0               1038      3545.6     447.8       0.0
  3.0-5.2               4124     62092.7    1773.6      99.3
  5.2-5.5                466     20954.6     198.4       0.0
  >5.5                  2599    764270.0     865.6      39.5
  visible (|eta|<5.2, no neutrinos): n=10381 E=71454.5 GeV pT=4648.3 GeV; neutrinos: n=11 E=160.4 GeV pT=13.4 GeV
  cone Delta R < 0.4 around the last copy of each b quark (no neutrinos):
    bbar axis [708] pT=72.5 eta=-3.42 phi=0.57: n=81 E=701.2 GeV vector pT=44.1 GeV scalar pT=45.0 GeV
    b    axis [719] pT=78.3 eta=-1.38 phi=-2.73: n=79 E=76.6 GeV vector pT=32.2 GeV scalar pT=33.0 GeV
final state of the out-of-time pileup (897 collisions, bx -12..-1 and 1..3): n=161696 sumE=12199200.0 GeV sumPt=65501.2 GeV
  by kind                  n      E[GeV]   pT[GeV]
  charged hadron       73483   7659197.4   37987.6
  neutral hadron       13209   2278029.0    9150.9
  photon               73978   2244967.9   18162.9
  electron               942     14135.8     131.4
  muon                    31       834.3      21.1
  neutrino                53      2035.7      47.2
  by region                n      E[GeV]   pT[GeV]      E_nu
  |eta|<2.5            60623     66364.4   26412.1      83.0
  2.5-3.0              12613     43641.1    5471.6      17.6
  3.0-5.2              49767    714318.0   20616.9     408.4
  5.2-5.5               5550    229656.0    2184.2     218.7
  >5.5                 33143  11145220.5   10816.4    1308.1
  visible (|eta|<5.2, no neutrinos): n=122960 E=823814.6 GeV pT=52458.2 GeV; neutrinos: n=53 E=2035.7 GeV pT=47.2 GeV
final state of signal + in-time pileup: n=14205 sumE=870400.0 GeV sumPt=6468.3 GeV
  ...
jets: chapter 7's ak4GenJets (the photograph) -> the mixed anti-kT R = 0.4 jets (signal + in-time pileup, |eta| < 5, no neutrinos)
  jets above 3 / 20 / 30 / 50 GeV: 42 / 6 / 6 / 3 -> 175 / 108 / 82 / 20
  signal particles inside a jet: 368 of 749 -> 643 of 749 (651 entered the mixed clustering)
  b    quark pT=78.29: jet [0] pt=99.83 eta=-1.439 phi=-2.712 m=14.83 n=31 dR=0.0622 pt(jet)/pt(quark)=1.2752
       -> mixed jet [1] pt=133.53 eta=-1.479 phi=-2.701 m=43.21 n=115 (31 signal + 84 pileup from 33 collisions) pileupPt=34.57 (25.9%) dR=0.1036 pt(jet)/pt(quark)=1.7057 A=0.529 rhoA=39.89 pTsub=93.64
  bbar quark pT=72.50: jet [1] pt=94.09 eta=-3.446 phi=0.594 m=17.82 n=18 dR=0.0352 pt(jet)/pt(quark)=1.2978
       -> mixed jet [0] pt=136.77 eta=-3.476 phi=0.580 m=43.09 n=101 (19 signal + 82 pileup from 33 collisions) pileupPt=43.42 (31.7%) dR=0.0588 pt(jet)/pt(quark)=1.8865 A=0.489 rhoA=36.88 pTsub=99.89
  m(b jet + bbar jet): 302.9 -> 436.5 GeV; |pT(b jet) + pT(bbar jet)|: 16.9 -> 19.1 GeV
  mixed jets without a single signal particle: 30 of 175; jets above 30 GeV with at least one: 78 of 82
```

The out-of-time block is rebuilt from the library: the record lists the particles of the in-time
crossing only, and its vertex listing gives, for every other collision, its crossing and its library
event, whose particles the script reads from `pileup_library.lhe.gz` (the mixing moves vertices, not
momenta; the script checks that the in-time crossing rebuilt the same way gives back the record's
particle listing). 897 collisions, 161 696 final particles (180.3 per collision, against 13 456 / 63 =
213.6 in our crowded in-time crossing and 179.2 on average in the library), 65 501.2 GeV of scalar pT
(73.0 per collision, against 90.9 in time), 12 199 200 GeV = 897 x 13 600 GeV; 897 / 63 = 14.2 times
as many collisions as in time. CMSSW's 931 (section 4.7 of the JitJetSW companion): 176 266 stable
particles, 189.3 per collision, 71 747.8 GeV of pT, 77.1 per collision. Not overlaid in either: the
jets and the last column are signal + in-time pileup. The chapter says where each crowd is removed
later: the out-of-time one by the pulse fits of the ECAL and HCAL local reconstruction (chapter 14;
the release's `EcalUncalibRecHitWorkerMultiFit.cc`, `activeBXs` -5 ... +4, and
`HBHEMahiParameters_cfi.py`, `activeBXs` -3 ... +4, as quoted by the JitJetSW companion), the in-time
one by CHS and PUPPI (chapter 16), the rest by the L1 offset correction (chapter 19).

Derived, as the chapter quotes them: the pileup energy beyond abs(eta) = 5.5 is
764 270 / 856 800 = 89%; the pileup pT density at abs(eta) < 2.5 is 2440.0 / (5 x 2 pi) =
77.7 GeV per unit of eta-phi area (CMSSW 61.2), the mean pT of a pileup particle there
2440.0 / 5229 = 0.47 GeV (CMSSW 0.45), and a circle of pi R^2 = 0.503 collects 39.0 GeV
(CMSSW 30.8). The standalone crowd is the denser of the two because its 63 collisions
hold 51 non-diffractive ones against 41 of the 64 of CMSSW: 38.7 GeV of central pT per
collision against 30.0. The "electrons, muons" row of the chapter's table is the sum of
the two lines. Three differences of definition against the CMSSW ledger: the mixed
clustering takes only the particles within abs(eta) < 5 and no neutrinos (CMSSW clusters
every stable particle of `genParticlesForJets`), so the jet counts of the two are not
like for like, and 98 of the 749 signal particles cannot be in a mixed jet; the quark is
the last copy before the hadronisation (status below 70), where CMSSW takes the status-71
copy that enters it; and Delta R is in (eta, phi), as CMSSW's.

### 5.6 The HL-LHC crowd, for the open problems

`standalone_pileup.py --scenario ultimate` mixes the same family at mu = 200 in
bx = -3 ... +3 with the luminous region of CMSSW's HLLHCEvtVtxGenerator (two 9 cm bunches
crossing at 510 urad, crab-rotated by 380 urad at 400 MHz, beta* = 20 cm: sigma_z =
4.27 cm, sigma_t = 193 ps against the generator's nominal 4.3 cm and 190 ps; head-on the
region would be 6.36 cm long, with the crossing angle alone 2.14 cm). With the committed
script and the 13.6 TeV library, seed 1 (written to a scratch file, not committed) gives
1405 collisions, 202 of them in time with 35 369 particles, 26 488 + 6300 pseudojets
clustered in 32 788 steps, rho = 182.21 GeV per unit area (kT jets 185.08, the crowd
alone 174.31), 185 jets of which 137 above 30 GeV, our jet (mixed jet [1], A = 0.539)
99.83 -> 187.40 -> 89.26 with 214 strangers from 93 collisions, a residual of -10.81 GeV
(-9.08 as pileup minus rho A), and 136 pileup jets above 30 GeV before the subtraction, 8
after; `--scenario ultimate --repeat 400` gives rho = 187.4 +- 17.8, 94.8 +- 14.2 GeV of
pileup in the circle of our jet and a residual of +0.6 +- 13.1 GeV (+4.4 +- 13.0 with the
rho of the crowd alone; -4.2 +- 12.6 for the bbar jet). The record takes 15 minutes
(`--no-area` one).
