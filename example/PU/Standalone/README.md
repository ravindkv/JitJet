# The crowd on the road: pileup in Run 3 (2024), without CMSSW

`standalone_pileup.py` takes the final state of chapter 7 (`../../GenJet/event_GenJet.lhe`:
the particles the detector will see and the generator-level jets) and puts it into a
bunch crossing of the LHC of 2024, with every piece of CMSSW's mixing written out: the
mean number of collisions per crossing from the luminosity, mu = L sigma_inel / (n_b f_rev);
the Poisson draw of the MixingModule for every crossing of the Run-3 production window
bx = -12 ... +3; the library of minimum-bias collisions it reads (`../pileup_library.lhe.gz`,
2000 collisions at 13.6 TeV made once with PYTHIA by `make_pileup_library.py`, the
generator-level twin of the GEN-SIM library of the CMSSW production); the 2024 beam spot of
CMSSW's BetafuncEvtVtxGenerator (the SimBeamSpot payload of the 2024 global tag: sigma_z =
3.6 cm, sigma_t = 120 ps, beta* = 30 cm, 9 um across) and the 25 ns offsets of the
out-of-time crossings; the stacking of the in-time collisions onto the signal; anti-kT on
the mixed final state with the active area of every jet measured with ghosts (FastJet's
N2Tiled search ported); the median energy density rho two ways (the grid of CMSSW's
fixedGridRhoFastjetAll and the kT-jet median of Cacciari and Salam); the subtraction
pT - rho A against the truth; the vertices the tracker will see; and the flat pileup
profile of the Run-3 production with its reweighting. The output, `../event_PU.lhe`, is the
chapter-7 record followed by the vertex table of the 960 collisions, the particles of the
in-time crossing, the mixed jets with their areas and subtracted momenta, and the signal
rows with the mixed jet that holds each. The HL-LHC crowd of the Phase-2 production
(mu = 200 in bx = -3 ... +3, the luminous region of the crab cavities) is kept as
`--scenario ultimate` for the exercises and the open problems.

```bash
python3 standalone_pileup.py                        # -> ../event_PU.lhe (the book's record, mu = 60, bx = -12 ... +3; 2 min)
python3 standalone_pileup.py -v --log standalone_pileup.log --summary standalone_pileup.json
python3 standalone_pileup.py --no-area              # 20 s: no ghosts, no areas, no jet-median rho
python3 standalone_pileup.py --window 0             # the in-time crossing only
python3 standalone_pileup.py --window 1 --sequential   # the conditions of the CMSSW companion (PUSIM: PU 60 in bx = -1 ... +1)
python3 standalone_pileup.py --scenario ultimate    # the HL-LHC crowd: mu = 200, bx = -3 ... +3, the Phase-2 luminous region (15 min)
python3 standalone_pileup.py --repeat 400 --jobs 4 --quiet --history standalone_pileup_stats.json   # 15 s
```

On this Mac use `/usr/bin/python3` (NumPy); the record takes two minutes, almost all of it
the anti-kT clustering of 10 696 particles and 6300 ghosts (the kT clustering for the jet
median 20 s); `--no-area` brings it to 20 s; the `--repeat 400` ensemble (the in-time crowd
drawn again and again on the photograph, no clustering: the grid rho and the pileup inside
the circle of our jet, drawn in (y, phi)) takes 15 s with four workers. The script imports chapter 7's
`standalone_genjets.py` (the record, the clustering, the jet tables) and needs nothing else.

## What the script reproduces, and the numbers

* **The scenario** (`mu_of`, `SCENARIOS`): L = 2.0e34 cm^-2 s^-1 (2024, levelled), sigma_inel
  = 78.42 mb (PYTHIA, CP5; the tune fixes it, at 13.6 as at 14 TeV), 2350 colliding bunches,
  f_rev = 11245.5 Hz: mu = 59.3 (60.5 with the 80 mb of the bookkeeping); the 2024 simulation
  mixes at mu = 60 (`AVE_60_BX_25ns`), the default, with the production window bx = -12 ... +3:
  960 collisions expected per signal event. The production draws mu_true flat in 55 ... 75
  (`Run3_Flat55To75_PoissonOOTPU`, about 1040 collisions per signal event) and reweights.
* **The draw** (`Stage.draw_crowd`): Poisson(60) per crossing; seed 1 gives sixteen numbers
  from 50 to 68, **63** in time, 960 in all. The library is opened at a random event (430
  for seed 1) and read on in order, wrapping at the end, as CMSSW's
  `RootEmbeddedFileSequence::readOneRandom` does with `sequential = False`: no collision is
  used twice in an event (drawing every position at random would repeat one within the
  in-time crossing in six draws of ten); `--sequential` reads from the first event on, as
  the CMSSW exercise of the companion does.
* **The beam spot** (`BeamSpot`): BetafuncEvtVtxGenerator with the 2024 payload: z Gaussian of
  3.6 cm around 0.42 cm, t an independent Gaussian of sigma_z / c = 120 ps, x and y Gaussians
  of sqrt(eps beta*/2) sqrt(1 + ((z - z0)/beta*)^2) = 9 um at the focus (beta* = 30 cm,
  emittance 5.725e-8 cm). The 960 drawn vertices: r.m.s. 3.56 cm, 124 ps, 9 um; the signal at
  z = +0.16 cm, t = +94 ps. CMSSW's 250-event library: 3.56 cm, 122 ps. (`LuminousRegion`, the
  HL-LHC model of two 9 cm bunches crossing at 510 urad, crab-rotated by 380 urad at 400 MHz
  and focused to beta* = 20 cm, 4.27 cm and 193 ps, is `--region hllhc`.)
* **The stacking** (`Stage.stack`): the signal's 744 particles without neutrinos, 651 within
  |eta| < 5; the 63 in-time collisions bring 13 456 particles, 10 045 within |eta| < 5 and not
  neutrinos; 995 charged particles in the tracker's reach against 93 of the signal.
* **The clustering** (`TiledClustering`, `MixedJets`, `ghost_grid`): chapter 7's anti-kT with
  FastJet's tiled nearest-neighbour search (a pair farther apart than R never merges, so the
  neighbour of every pseudojet is within R and the plane can be tiled in cells of side R);
  6300 ghosts of area 0.01 within |y| < 5, pT = 1e-18; 16 996 steps; 175 jets above 3 GeV, 82
  above 30. Our jet is mixed jet [1] in this crossing (the bbar jet, 136.77 GeV, leads):
  115 particles, the 31 of the photograph and 84 strangers from 33 collisions, 133.53 GeV,
  0.012 from the photograph's axis in (y, phi), A = 0.529 (pi R^2 = 0.503; the ghosts
  measure it to +- 0.02: 0.47 to 0.54 with other ghost seeds, 0.539 with FastJet's).
* **rho** (`grid_rho`, `jet_median_rho`): 18 x 11 cells of 0.556 x 0.571 in |y| < 5, median
  75.47 GeV per unit area (cells from 30 to 361, mean 82.9); the kT-jet median in |y| < 4.4
  (the range of CMSSW's kT-jet rho): 78.50; per collision 1.20 and 1.25 in this crowded
  crossing, 0.97 over the ensemble; the signal alone 3.42 (mean 11.1, 17 of 198 cells
  empty), the crowd alone 69.93: the signal raises the median by 5.5 here, 6.2 on average.
* **The subtraction** (`Stage.subtract`): our jet 133.53 - 75.47 x 0.529 = 93.64 against the
  100.07 GeV of signal inside: residual -6.43 GeV in this crossing (the record's
  definition: vector pT - rho A - scalar signal), -5.32 as pileup inside minus rho A (the
  ensemble's definition: scalar sums on both sides). Over 400 draws the crowd puts
  28.3 +- 8.1 GeV inside the circle, rho pi R^2 estimates 29.5 +- 5.3, the residual is
  -1.1 +- 6.1 GeV, and +2.0 +- 6.1 with the rho of the crowd alone (-2.7 +- 6.1 and +0.4
  for the bbar jet). 76 pileup jets above 30 GeV before the subtraction, one after; over
  seeds 1 to 9 clustered in full, 23 to 76 before and five in all after.
* **The vertices** (`Stage.visible_vertices`): 40 of 63 in-time collisions have at least four
  charged particles of pT > 0.5 GeV within |eta| < 2.5; the nearest to the signal is 0.01 mm
  away in z and 0.3 ps in t; 3 are within 1 mm, 7 within 30 ps.
* **The profile** (`fill_profile`, `year_profile`, `flat_profile`, `reweight`): a 2024-like fill
  (levelled at 60 for 8 h, then decaying with tau = 10 h, 12 h) keeps 72% of its collisions at
  the levelled value (mean 57); a year of such fills levelled between 57 and 67 (a model) has
  mean 59; the production's flat 55 ... 75 profile reweighted to it gets w from 1.6 to 0, 1.54
  at mu_true = 60; a sample at a fixed mu = 60 cannot be reweighted (w = 0.07, one bin).

## The other files

| File | What it is |
|---|---|
| `make_pileup_library.py` | the library: 2000 minimum-bias collisions with PYTHIA (`SoftQCD:inelastic`, CP5, 13.6 TeV, seed 20261), 15 s in the ME venv; writes `../pileup_library.lhe.gz` and `../pileup_library.log` |
| `standalone_pileup.log`, `standalone_pileup.json` | the `-v` narration of the record and its summary |
| `standalone_pileup_stats.json`, `standalone_pileup_stats.log` | the 400-draw ensemble |
| `minbias_catalogue.py` | the catalogue of the library in the format of the CMSSW companion's `minBiasGenStats.py`: one line per collision (kind, interactions, stable, charged, central, beyond \|eta\| = 5.5, lead pT, sum pT, jets > 10 GeV, forward protons, and the vertex drawn for the in-time ones), the summary and the means per kind; writes `.txt`, `.json` and the `.tex` tables the chapter inputs (2 min) |
| `validate_pileup_with_fastjet.py` | the validation: `--mode record` clusters the mixed final state of the record with FastJet (areas, grid and jet-median rho); `--mode library` checks the library's shares and multiplicities against PYTHIA's cross sections and the 2024 CMSSW library |
| `plot_theory_PU.py` | the four pages of `theory_pileup.pdf` behind the theory sections |
| `animate_standalone_pileup.py` | the film (six narrated scenes) and its stills; the stills are a build product, the film is rendered on request |
| `../plot_journey_PU.py` | the (eta, phi) display of the event with the crowd, `../journey_PU.pdf` |
| `../CMSSW/` | the CMSSW side: the 2024 pileup library and the mixing of one signal event at PU 60 (`2024/`), and the Phase-2 exercise at PU 10 (`Phase2/`), from JitJetSW |
