# Chapter 6 companion: The Children Produce Grandchildren (Decay)

This is the technical companion of chapter 6, *The Children Produce
Grandchildren: B-Hadron Decays*. The chapter gives the physics; this page
says how to run the scripts, how to read what they write, how each number
of the official packages was obtained, and lists every table the chapter
quotes with the command that produced it. All files are in
[`example/Decay/`](../example/Decay/) of this repository unless stated
otherwise; the CMSSW analysers are in
[JitJetSW](https://github.com/ravindkv/JitJetSW).

| | |
|---|---|
| Stage | the hadron decays, on the hadron-level record of chapter 5 |
| Standalone script | [`example/Decay/Standalone/standalone_hadron_decays.py`](../example/Decay/Standalone/standalone_hadron_decays.py) |
| Decay tables | [`example/Decay/Standalone/decay_tables.json`](../example/Decay/Standalone/decay_tables.json) (161 particles, 3114 channels, dumped from PYTHIA 8.315 by `dump_decay_tables.py`) |
| Input | [`example/Had/event_Had.lhe`](../example/Had/event_Had.lhe), the 365 primary hadrons of chapter 5 |
| Event record | [`example/Decay/event_Decay.lhe`](../example/Decay/event_Decay.lhe) (seed 1, 2251 rows, 749 final particles, plus the vertex listing) |
| Learning-mode log | [`example/Decay/Standalone/standalone_hadron_decays.log`](../example/Decay/Standalone/standalone_hadron_decays.log) (`-v`, 3193 lines) |
| Validation with PYTHIA | [`example/Decay/Standalone/validate_decays_with_pythia.py`](../example/Decay/Standalone/validate_decays_with_pythia.py) |
| CMSSW figure | [`example/Decay/CMSSW/genPartAnalyzer_ME_PS_MPI_Had_Decay_out.pdf`](../example/Decay/CMSSW/genPartAnalyzer_ME_PS_MPI_Had_Decay_out.pdf) (page 5); the log is that of chapter 2 |
| Film | scene 6 of the [JitJet playlist](https://www.youtube.com/playlist?list=PLbSW20iyZFd4) |

Contents: [1. Running the standalone script](#1-running-the-standalone-script) ·
[2. Reading the record](#2-reading-the-record) ·
[3. Drawing the event and rendering the film](#3-drawing-the-event-and-rendering-the-film) ·
[4. Validation from official packages](#4-validation-from-official-packages) ·
[5. The full tables](#5-the-full-tables)

## 1. Running the standalone script

`standalone_hadron_decays.py` takes the hadron-level record of chapter 5
and decays every unstable one of its 365 primary hadrons, following
PYTHIA 8's `ParticleDecays`: lifetimes and displaced vertices, B0 and B_s0
mixing, the decay tables, the M-generator phase space with the V-A,
omega -> 3 pi, polarisation and Dalitz matrix elements, and the partonic
channels turned into hadrons with chapter 5's `StringFlav`. Thirteen
hundred lines of Python that need NumPy and nothing else; it imports the
hadronisation script of chapter 5 for the event record, the Lorentz frames
and the flavour selection, and reads `decay_tables.json` from its own
directory. The table is regenerated with
`../../ME/Standalone/.venv/bin/python dump_decay_tables.py`, the only step
that needs PYTHIA.

A map of the file, with the steps of the chapter's flow chart:

| Function or method | What it does |
|---|---|
| `open_channels`, `can_decay` | who decays: a table, `mayDecay`, and c tau0 below `tau0Max` = 10 mm |
| `Decayer.draw_lifetime`, `decay_vertex` | the clock, tau = -tau0 ln r, and the walk x_prod + tau p/m |
| the mixing step inside `Decayer.decay` | P(oscillated) = sin^2(x tau / 2 tau0); the id is flipped and the products get status 92 |
| `pick_channel`, `mass_sel`, `pick_hadrons` | the roll of the dice against the cumulative branching ratios with the `onMode` filter; Breit-Wigner masses; the partonic channels paired into singlets, the Poisson multiplicity, the hadrons from chapter 5's flavour machinery |
| `two_body`, `mgenerator`, `me_weight`, `dalitz` | two bodies isotropic; the Raubold-Lynch M-generator; the matrix-element weights by `meMode`; the Kroll-Wada lepton pair |
| `run`, `verify` | the queue of unstable particles in order of laboratory decay time; the conservation checks |
| `summarise` | the two B hadrons, their vertices, tracks and neutrinos; the cones around the two b quarks before and after the decays and with and without the neutrinos |

```bash
cd example/Decay/Standalone
python3 standalone_hadron_decays.py                      # seed 1 -> ../event_Decay.lhe (the book's record)
python3 standalone_hadron_decays.py -v                   # learning mode: the two B hadrons' chains step by step
python3 standalone_hadron_decays.py -vv                  # ... every decay
python3 standalone_hadron_decays.py --no-mix             # no B0 / B_s0 oscillations
python3 standalone_hadron_decays.py --no-limit-tau0      # also decay K_S0, Lambda, Sigma, Xi, Omega (PYTHIA's default)
python3 standalone_hadron_decays.py --stable-b           # keep the b hadrons (everything else decays)
python3 standalone_hadron_decays.py --repeat 400 --quiet --history standalone_decay_stats.json
python3 standalone_hadron_decays.py --single 511 --repeat 20000 --history standalone_single_511.json   # a B0 at rest
```

The first command decays the record in a second and a half and prints

```text
input example/Had/event_Had.lhe: 2251 rows after the decays; seed = 1
  365 primary hadrons -> 376 decays (1 B oscillations) -> 749 final particles: 338 charged, 327 photons, 5 neutrinos (229.4 GeV, |sum pT| = 9.13 GeV), 7 electrons, 2 muons, 408 hadrons; 143 primary hadrons survive, 606 particles come from decays; 396 final particles (176 charged) at |eta| < 2.5
  most common: gamma 327, pi- 134, pi+ 128, K_S0 29, K+ 24, K- 20, K_L0 19, nbar0 12, n0 11, p+ 10, pbar- 6, Lambdabar0 5
  bbar (row 708, pT = 72.50) -> B*0 (row 1069, pT = 82.83) -> Bbar0 -> Xi_c0
     Bbar0 (row 1590): pT = 81.97, E = 1264.18 GeV, gamma = 239.4; tau = 2.046 ps = 1.34 tau0 -> L = 146.88 mm (L_xy = 9.52), vertex (7.97, 5.21, -146.57) mm; oscillated
     channel: Bbar0 -> Xi_c0 pi+ Deltabar- (BR 0.0060, meMode 63, hadronic, partonic); 3 direct products, 9 final particles (6 charged, 2 photons, 0 neutrinos with 0.00 GeV = 0.0% of the B energy); charm hadron Xi_c0 pT = 71.43, flies 21.02 mm
     6 charged tracks from the B: largest impact parameter 417 um, 5 above 100 um; the b system: 9 particles, pT = 81.97 GeV (visible 81.97)
     cone R = 0.4 around the quark: partons pT 93.87 -> 4 primary hadrons pT 91.34 -> 17 final particles pT 92.99 (m 7.59 -> 13.78), visible 17 with pT 92.99, charged 8 with pT 50.61; ...
  b (row 719, pT = 78.29) -> B*bar0 (row 1012, pT = 78.56) -> Bbar0 -> D*_s- -> D_s-
     Bbar0 (row 1527): pT = 78.26, E = 165.47 GeV, gamma = 31.3; tau = 2.237 ps = 1.46 tau0 -> L = 21.01 mm (L_xy = 9.94), vertex (-9.13, -3.93, -18.51) mm
     channel: Bbar0 -> D*_s- D*0 rho+ (BR 0.0740, meMode 43, hadronic, partonic); 3 direct products, 15 final particles (8 charged, 6 photons, 0 neutrinos with 0.00 GeV = 0.0% of the B energy); charm hadron D0 pT = 26.97, flies 10.98 mm
     8 charged tracks from the B: largest impact parameter 1640 um, 8 above 100 um; the b system: 15 particles, pT = 78.26 GeV (visible 78.26)
     cone R = 0.4 around the quark: partons pT 100.67 -> 6 primary hadrons pT 99.23 -> 31 final particles pT 99.43 (m 16.09 -> 14.59), visible 31 with pT 99.43, charged 16 with pT 67.43; neutrino pT in the cone 0.00; the B's products carry 79%, 0 of them left the cone
     cone R = 0.8 around the quark: partons pT 104.09 -> 14 primary hadrons pT 105.97 -> 51 final particles pT 108.73 (m 35.88 -> 42.14), visible 51 with pT 108.73, charged 26 with pT 71.41; ...
wrote example/Decay/event_Decay.lhe (2251 rows, 749 final particles)
```

The output with `--seed 1` is byte-for-byte the committed record. With
`--history` the summary goes to a JSON file as well; with `--repeat N` the
script decays the same primary hadrons N times with seeds 1..N and writes
the per-event quantities of section 5.1 to `<history>.stats.json`
(`standalone_decay_stats.stats.json` is the committed 400-event run, five
minutes; `standalone_decay_stats.log` its printout); with `--single 511` it
decays a B0 at rest instead, twenty thousand times in a minute
(`standalone_single_511.json`).

| Option | Meaning | Default |
|---|---|---|
| `-i`, `-o` | input record, output record | `../../Had/event_Had.lhe`, `../event_Decay.lhe` |
| `--seed` | the random seed | 1 |
| `--tau0max` | `ParticleDecays:tau0Max` [mm]: particles with a longer c tau0 are left to the detector | 10 (CMSSW) |
| `--no-limit-tau0` | decay every particle with a table, as PYTHIA's own default does (K_S0, Lambda, hyperons too) | off |
| `--no-mix` | no B0 / B_s0 oscillations | off |
| `--stable-b` | keep the b hadrons stable, everything else decays | off |
| `--radius` | the cone of the family summary | 0.4 |
| `--repeat N`, `--quiet`, `--history` | N decays with seeds 1..N, no per-event printout, JSON output | |
| `--single ID` | decay this particle at rest `--repeat` times instead (511 is the B0) | |
| `-v`, `-vv` | the learning modes, below | off |

### The learning modes

`-v` prints, before the event, the setting of the scene: the tables, the
rule for who decays with the list of hadrons left to the detector and their
c tau0, the lifetimes of the heavy hadrons, the mixing parameters, the
composition of the B0, B+, Lambda_b0 and D0 tables by class, and the rules
for the partonic channels; then the list of the 222 primary hadrons that
decay; then, for the chains of the two B hadrons, every decay in full, the
lifetime draw with its random number, the vertex, the mixing probability
and its draw, the channel with its number and branching ratio, the parton
pairing and the Poisson draw, the masses, the phase-space tries and the
matrix-element weight, and the products in the laboratory frame with the
conservation check; one line per other decay; the checks; the record with
its vertex table; and the summary. `-vv` does the same for every decay. The
3193 lines of `-v` for seed 1 are committed as
`standalone_hadron_decays.log`, and every number quoted in the theory
section of the chapter is taken from that file.

## 2. Reading the record

### Status codes

The record keeps PYTHIA's conventions and adds the status codes of the
decay level to those of the [chapter 5 companion](chapter_05_Had.md#status-codes).
A particle that decays keeps its row and its status becomes negative, as at
every previous stage; its products are appended with status `91`, or `92`
if the particle oscillated before decaying, with the mother in the first
mother column and the daughter range on the mother's row. Our record has
2251 rows: the 1491 of chapter 5, of which 222 primary hadrons now carry
status `-81` to `-84` and 143 keep their positive status, then 760 decay
products, 154 of which decayed in their turn (152 of status `-91` and 2 of
`-92`) and 606 of which are final (605 of status `91`, 1 of `92`). The final
state is every row with a positive status, 143 + 606 = 749 particles; their
momentum sum is still that of the two protons and their charge sum +2.

### The rows the chapter reads

The B*bar0 of chapter 5 (row 1012), status `-83` now, with daughters 1527
and 1528; the Bbar0 (row 1527, status `-91`) and its photon (row 1528,
final); and the three direct products of the Bbar0, every one of which
decays again:

```text
  1012       -513  (B*bar0)           -83  1006  1011  1527  1528     0     0    -72.156    -31.074   -146.183    166.042      5.325
  1527       -511  (Bbar0)            -91  1012     0  1969  1971     0     0    -71.875    -30.965   -145.695    165.469      5.280
  1528         22  gamma               91  1012     0     0     0     0     0     -0.281     -0.109     -0.488      0.573      0.000
  1969       -433  (D*_s-)            -91  1527     0  2177  2178     0     0    -27.738    -12.283    -56.230     63.926      2.112
  1970        423  (D*0)              -91  1527     0  2179  2180     0     0    -28.764    -11.717    -58.976     66.684      2.007
  1971        213  (rho+)             -91  1527     0  2181  2182     0     0    -15.372     -6.966    -30.489     34.858      0.790
```

The listing ends with the charge and momentum sums over the 749 final
particles, the charge of two protons and their momentum to a few MeV:

```text
                                   Charge sum:  2.000           Momentum sum:     -0.004     -0.000     -0.001  13599.999  13599.999
```

### The vertex listing

After the listing and its sums the file carries a second table, one row for
every particle born more than 0.1 um from the origin and for every particle
that drew a proper lifetime: the production vertex in millimetres, the
production time in mm/c, and the proper lifetime tau drawn for the row (330
rows in our record). The Bbar0 is born at the origin with tau = 0.67066 mm/c
= 2.237 ps; its D*_s- and D*0 are born at the Bbar0's decay vertex:

```text
    no      x_prod      y_prod      z_prod      t_prod         tau
  1527     0.00000     0.00000     0.00000     0.00000     0.67066
  1969    -9.13015    -3.93351   -18.50748    21.01925     0.00000
  1970    -9.13015    -3.93351   -18.50748    21.01925     0.00000
```

The listing has six fields and the particle rows thirteen, so the reader of
chapter 2's script skips it; the reader of `plot_journey_Decay.py` parses
it.

### The learning-mode log, section by section

The log has the sections `setting the scene`, `which particles decay`, one
block `---- row n: X decays` per decay of the two B chains (the other decays
in one line each), `checks` and `summary`. The opening lines the chapter
quotes:

```text
  decay tables: PYTHIA 8.315 ParticleData, dumped by dump_decay_tables.py: 161 particles, 3114 channels
  a particle decays if it has a table, mayDecay (PYTHIA: tau0 < 1000 mm) and tau0 < tau0Max = 10.0 mm
  stable for the detector: pi+ (7.8e+03 mm), K+ (3.71e+03 mm), K_L0 (1.53e+04 mm), K_S0 (26.8 mm), Lambda0 (78.9 mm), Sigma+ (24 mm), Sigma- (44.3 mm), Xi- (49.1 mm), Xi0 (87.1 mm), Omega- (24.6 mm), n0 (2.66e+14 mm), mu- (6.59e+05 mm)
  lifetimes: c tau0 = B0 0.4587 mm (1.530 ps), B+ 0.4911 mm (1.638 ps), B_s0 0.4390 mm (1.464 ps), Lambda_b0 0.4400 mm (1.468 ps), D0 0.1229 mm (0.410 ps), D+ 0.3118 mm (1.040 ps), D_s+ 0.1499 mm (0.500 ps), Lambda_c+ 0.0599 mm (0.200 ps), tau- 0.0871 mm (0.291 ps)
  B mixing on: x_d = 0.776, x_s = 26.05; time-integrated P(oscillated) = x^2/(2(1 + x^2)) = 0.188 (B0), 0.499 (B_s0)
  B0: 870 open channels, BR sum 1.0000: partonic 49.8%, hadronic (explicit) 26.4%, semileptonic (e) 10.4%, semileptonic (mu) 10.4%, semileptonic (tau) 3.0%
  B+: 725 open channels, BR sum 0.9999: partonic 43.1%, hadronic (explicit) 31.3%, semileptonic (e) 11.3%, semileptonic (mu) 11.3%, semileptonic (tau) 3.0%
  Lambda_b0: 41 open channels, BR sum 1.0000: partonic 63.6%, hadronic (explicit) 18.5%, semileptonic (e) 7.7%, semileptonic (mu) 7.7%, semileptonic (tau) 2.5%
  D0: 84 open channels, BR sum 1.0000: hadronic (explicit) 86.7%, semileptonic (e) 6.8%, semileptonic (mu) 6.6%
  365 primary hadrons; 222 of them decay (a decay table, tau0 < 10.0 mm), 143 are stable for the detector: pi+ 43, pi- 37, K+ 13, nbar0 10, K- 10, n0 9, p+ 8, Sigma+ 4, Lambdabar0 3, pbar- 3, Sigmabar- 3
```

The mother's Bbar0, the block the chapter's theory quotes:

```text
---- row 1527: Bbar0 decays  (m = 5.2796, E = 165.469, pT = 78.261 GeV, eta = -1.380, phi = -2.735) ----
  lifetime: c tau0 = 0.4587 mm (1.530 ps); r = 0.7682 -> tau = -tau0 ln(1 - r) = 0.6707 mm/c (2.237 ps, 1.46 lifetimes); gamma beta = p/m = 31.33 -> flight L = 21.009 mm (L_xy = 9.941 mm)
  born at (0.0000, 0.0000, 0.0000) mm, decays at (-9.1302, -3.9335, -18.5075) mm, t = 21.0192 mm/c
  mixing: x = Delta m / Gamma = 0.776: P(oscillated) = sin^2(x tau / 2 tau0) = sin^2(0.567) = 0.289; r = 0.519 -> no oscillation
  channel: r x (BR sum 1.0000) -> channel 850 of 870: Bbar0 -> cbar s c dbar (BR 0.07400 = 7.40% of the decays; meMode 43: partons -> hadrons, Poisson, at least 3)
  partons cbar s c dbar as the singlets (cbar s) (c dbar); mDiff = 5.280 - 0.000 (known) - 4.025 (constituent masses) = 1.255 GeV: <n> = 0/2 + 4/4 + 4 ln(1.255/0.7) = 3.33; Poisson -> 3 hadrons
     cbar ... s -> 1 hadron(s): D*_s-
     c ... dbar -> 2 hadron(s): D*0 rho+
  masses: D*_s- 2.112, D*0 2.007, rho+ 0.790 (Breit-Wigner)
  3-body phase space (M-generator): accepted configuration 1 (phase-space weight 0.329 of its maximum)
  products (lab frame), rows 1969-1971:
     1969 D*_s-        m =  2.1123  E=   63.926 px= -27.738 py= -12.283 pz=  -56.230  | pT= 30.336 eta= -1.376 phi=-2.725
     1970 D*0          m =  2.0070  E=   66.684 px= -28.764 py= -11.717 pz=  -58.976  | pT= 31.059 eta= -1.397 phi=-2.755
     1971 rho+         m =  0.7901  E=   34.858 px= -15.372 py=  -6.966 pz=  -30.489  | pT= 16.877 eta= -1.354 phi=-2.716
  check: sum of the products - mother = (+6.8e-13, -3.0e-13, -1.3e-13, -6.0e-13) GeV
```

The B*bar0 before it (row 1012: tau0 = 0, one channel, p* = 0.0452 GeV,
isotropic), the bbar's B0 (row 1590: r = 0.7374, tau = 1.34 tau0, L = 146.88
mm, P(oscillated) = 0.246, r = 0.061 -> oscillated, channel 837
`Bbar0 -> cs_1 udbar_1`, meMode 63) and the charm decays further down the
chains (the D0 at row 2179: r = 0.9435, 2.87 lifetimes, 10.98 mm, channel 64
of 84 `K- omega pi+`; the D_s- at row 2177: channel 29 of 61 `K*- K*0`; the
omega's three pions accepted after 111 tries) each have a block of the same
shape.

## 3. Drawing the event and rendering the film

### The (eta, phi) display

`plot_journey_Decay.py` draws the record in the (eta, phi) plane in the
layout of the CMSSW `genPartAnalyzer`, reusing the reader and the frame of
the previous stages and parsing the vertex listing. The categories of this
stage are four: every final descendant of the B*bar0 and of the B*0, the b
and bbar systems, in red and blue; their vector sum, the magenta star of
every stage; the primary hadrons that survive, in purple; and every other
decay product in grey, with the neutrinos drawn hollow and marked nu. The
header keeps the two B hadron lines of the CMSSW figure and adds the flight
distance of each weakly decaying B, the vector sum of all final particles,
which must vanish, and the missing transverse momentum carried by the
neutrinos.

```bash
python3 example/Decay/plot_journey_Decay.py                 # -> example/Decay/journey_Decay.pdf
python3 example/Decay/plot_journey_Decay.py -i my.lhe --png
```

### The theory plots

`plot_theory_Decay.py` replays the seed-1 decays with the classes of the
script and reads the `--repeat` statistics and the PYTHIA validation files
when they are present; it writes `theory_hadron_decays.pdf`, one page per
theory subsection:

| Page | Chapter figure | Content |
|---|---|---|
| 1 | the decay law | the proper lifetimes of the two B hadrons over 400 decays against the exponential; the flight distance of the b quark's Bbar0 for the script, PYTHIA and the full chain; the ladder of c tau0 with the 10 mm line |
| 2 | the birthmark | the decay vertices of our event in the transverse plane and along the beam with the tracks; the impact parameters of the tracks from the B over 400 decays |
| 3 | the decay tables | the classes of the tables of B0, B+, B_s0, Lambda_b0, D0, D+; the decay chains of the two B hadrons and of the CMSSW event; the multiplicity per B hadron |
| 4 | semileptonic decays | the V-A lepton spectrum against phase space and PYTHIA; the neutrino's share of the B energy; the pT of the hardest lepton in the jet |
| 5 | our jet at this stage | the cone through the stages; the response after the neutrinos; the multiplicity and the mass of the cone |

`sketch_ordering_trap.py` draws the ordering trap of the V-A weight into
`chapter/tikz/decay_ordering.tex`: the charged-lepton energy of
`D+ -> K0bar e+ nu` and of `B0bar -> D*+ l- nubar` at rest for flat phase
space, for the table's pairing and for the swapped pairing (three million
phase-space draws); `--pythia` decays both channels in PYTHIA at rest in
either order and prints the same means (1.404 / 1.265 and 0.499 / 0.567 GeV).

```bash
python3 example/Decay/Standalone/plot_theory_Decay.py      # -> theory_hadron_decays.pdf
python3 example/Decay/Standalone/sketch_ordering_trap.py   # -> chapter/tikz/decay_ordering.tex
make figures                                               # the same, from the book directory
```

### The film

`animate_standalone_hadron_decays.py` imports the standalone script and
replays the seed-1 decays in seven narrated scenes of about seven minutes
in all (seventy seconds without the voice;
`--scene clock,walk,table,cascade,emigrant,event,ensemble` or `1..7`). The
voice picks up where the chapter-5 film stopped, "the children are
married", and ends at the door of the inn of chapter 15, where the anti-kT
algorithm gathers the grandchildren.

1. **clock**: a hundred B0 mesons born at once decay one by one; the
   survivors follow exp(-t/tau0) and the lifetime drawn for our event's
   Bbar0, 2.24 ps, is marked.
2. **walk**: the b quark's Bbar0 leaves the primary vertex at
   gamma beta = 31, decays after 21.0 mm, its D0 walks on another 11.0 mm,
   and the tracks of the charged grandchildren fan out from the two
   displaced vertices; on the right the same along the beam.
3. **table**: the 870 channels of a Bbar0 as a cumulative bar of branching
   ratios coloured by class; the random number slides down to channel 850
   and the partons turn into hadrons.
4. **cascade**: the decay tree of the mother's hadron grows generation by
   generation in the (eta, phi) plane down to the sixteen stable particles.
5. **emigrant**: the V-A lepton spectrum fills up against phase space over
   four thousand decays; then the semileptonic decay of seed 4,
   `Bbar0 -> D*+ mu- nubar`, in which the neutrino takes 19.8 GeV out of
   the cone and the visible pT drops from 99.2 to 89.7 GeV.
6. **event**: the 365 primary hadrons decay in the order of their
   laboratory decay times until 749 particles are left, with the counters.
7. **ensemble**: the same hadrons decayed with seeds 1, 2, 3, ...; the
   flight distance fills the stretched exponential (mean 14.5 mm) and the
   visible-to-primary ratio of the cone stays at one in four decays out of
   five and drops by 10 to 50% when a neutrino leaves.

```bash
cd example/Decay/Standalone
python3 animate_standalone_hadron_decays.py            # the narrated film, 1080p, ~40 min with --jobs 4
python3 animate_standalone_hadron_decays.py --fast     # low-resolution preview
python3 animate_standalone_hadron_decays.py --print-script   # the narration, one line per phase
python3 animate_standalone_hadron_decays.py --scene walk,cascade -o walk.gif   # silent
python3 animate_standalone_hadron_decays.py --stills animate_standalone_hadron_decays_stills.pdf --no-video
```

Rendering needs matplotlib in addition to NumPy, `ffmpeg` for an `.mp4`,
and `edge-tts` for the voice (`en-GB-RyanNeural`; `--engine say` offline);
the shared `example/soundtrack.py` synthesises the lines, caches the clips
in `build/narration/Decay/` and writes the score and the subtitles. The last
command writes the key frames (one page per scene) that the chapter prints;
`make figures` runs it. The physics options are those of the standalone
script: `--seed 4` films the semileptonic decay of scene 5 as the event,
`--no-mix` switches the oscillations off, `--stable-b` keeps the B hadrons,
`--events` sets the decays of scene 7, and `--no-narration` gives the silent
cut.

## 4. Validation from official packages

Two official sources: PYTHIA driven from a script, and the CMS generator
job of chapter 1, which decayed its hadrons in the same step that showered,
dressed and hadronised them. As at the previous stop the first comparison
is made on identical input, and this time a third mode isolates one
particle. EvtGen, the decay package that CMS production hands its b and c
hadrons to, does not appear: the CMSSW job of chapter 1 decayed its hadrons
with PYTHIA's own tables, and so does the script, with the same tables
dumped to a file.

### 4.1 The same hadrons with PYTHIA 8

`validate_decays_with_pythia.py` has three modes. In mode `record` (the
default) it switches PYTHIA's process level off, fills the event record by
hand with the 365 primary hadrons of `event_Had.lhe` with their status
codes, and lets `next()` run the decays on them with
`ParticleDecays:limitTau0 = on` and `tau0Max = 10`, N times with different
seeds; the standalone script does the same with `--repeat N`. In mode
`single` it decays one particle at rest N times, a B0 by default, which
removes the boost and the rest of the event and tests the tables, the
matrix elements and the mixing alone. In mode `full` the hard event of
chapter 1 is written to a Les Houches file N times and PYTHIA runs the
whole chain with the CP5 settings and decays on, with or without the
impact-parameter window of chapter 4. All three record per event what the
chapter's table lists, in the JSON layout of the standalone statistics so
that `plot_theory_Decay.py` draws both. As before PYTHIA's output goes
through C++ streams and is captured by redirecting the file descriptor.

```bash
cd example/Decay/Standalone
PY=../../ME/Standalone/.venv/bin/python
$PY validate_decays_with_pythia.py -n 400 --standalone-stats standalone_decay_stats.stats.json
$PY validate_decays_with_pythia.py --mode single --id 511 -n 20000 --seed 3 --standalone-stats standalone_single_511.json
$PY validate_decays_with_pythia.py --mode full -n 2000
```

The first two runs take a minute each, the full chain a few minutes. The
results are `validate_decays_with_pythia_{record,single_511,full}.json`
and the printouts `..._{record,single_511,full}.log`; PYTHIA's first decay
of our hadrons is in `validate_decays_with_pythia_record_event1.txt`, its
first B0 at rest in `..._single_event1.txt` and its first full-chain event
in `..._full_event1.txt`, all in the standalone format with the vertex
table, so that `plot_journey_Decay.py` draws them. The tables are in
section 5.1.

| Option | Meaning |
|---|---|
| `--mode record` / `single` / `full` | decay the 365 hadrons of chapter 5 / one particle at rest / run the full CP5 chain from the hard event |
| `-n`, `--seed`, `--id` | events, seed, and the particle of mode single (511 = B0) |
| `--impact`, `--window` | full mode: keep events with abs(b/`<b>` - impact) < window |
| `--no-mix` | `ParticleDecays:mixB = off` |
| `--standalone-stats` | the `.stats.json` of the standalone `--repeat` (or the `--single --history` file), printed side by side |
| `--output`, `--listing` | the JSON file and the first event's listing |

### 4.2 CMSSW: decays inside the GEN step

Nothing new has to be run. The `GEN` step of chapter 1 decayed the event,
since `HadronLevel:all` covers the decays and
`ParticleDecays:limitTau0 = on` with `tau0Max = 10` is in every CMS
configuration, and JitJetSW's analyser drew the stable particles,
`genParticles` with status 1, as its fifth figure. From the setup of the
[chapter 2 companion](chapter_02_PS.md#42-cmssw-the-shower-inside-the-gen-step):

```bash
cd CMSSW_15_0_5/src/JitJetSW/Journey/test
./runCMSDrivers_mc2024_BBbar_GEN.sh          # GEN step + genPartAnalyzer, one PDF per stage
```

produces `genPartAnalyzer_ME_PS_MPI_Had_Decay_out.pdf`, whose fifth page is
the chapter's CMSSW figure; the copy in `CMSSW/` is the one used, and the
log is `example/PS/CMSSW/step1_GEN_PS_cfg.log` of chapter 2. This job
decayed its b and c hadrons with PYTHIA's own tables, the ones the script
embeds; the production configurations of CMS hand them to EvtGen through
the external-decayer interface instead, with PHOTOS for the radiative
corrections and TAUOLA for the tau.

**Reading the log.** The particle listing continues after the primary
hadrons of chapter 5 with the decay products, status 1 for the stable ones
and status 2 for those that decayed again, and its vertex columns, in
centimetres, now vary from row to row. The mother's B- and its daughters:

```text
  829 |  -521 -      B- |  2 |  824  828 1434 1436 |  5  3 |  94.279  -1.567 -2.092 | ... |  0.028 -0.017  3.112 |
 1434 |   -12 - nu_ebar |  1 |  829  829   -1   -1 |  1  0 |  37.457  -1.588 -2.098 | ... | -0.128 -0.288  2.394 |
 1435 |    11 -      e- |  1 |  829  829   -1   -1 |  1  0 |   4.951  -1.431 -2.210 | ... | -0.128 -0.288  2.394 |
 1436 | 10421 -   D*_00 |  2 |  829  829 1725 1726 |  1  2 |  51.912  -1.563 -2.077 | ... | -0.128 -0.288  2.394 |
```

Row 829 is the B- of chapter 5 at 94.279 GeV, status 2 now, with daughters
1434 to 1436: an electron antineutrino of 37.457 GeV, an electron of 4.951,
and a D*_00 of 51.912 GeV that decays further at rows 1725 and 1726. Its
production vertex is the smeared collision point, (0.028, -0.017, 3.112) cm;
its daughters are born at (-0.128, -0.288, 2.394), so that the B- walked
L = sqrt(0.156^2 + 0.271^2 + 0.718^2) cm = 7.83 mm with L_xy = 3.13 mm. The
neutrino carries 96.2 of the hadron's 235.9 GeV of energy (f_nu = 0.41) and
37.5 of its 94.3 GeV of pT. The bbar's B*+, row 968, radiated its photon
(row 1504) and left a B+ of 112.618 GeV (row 1503) whose daughters at rows
1752 to 1756, `Dbar*0 pi+ pi0 K0 Kbar*0`, are born at (0.322, 0.451, -8.604)
cm: L = 117.3 mm down the beam pipe, L_xy = 5.53 mm, sixteen final particles
and no neutrino.

## 5. The full tables

### 5.1 Hadron decays obtained three ways

"Same hadrons" means the 365 primary hadrons of `event_Had.lhe`, decayed
400 times by PYTHIA 8.315 with `ParticleDecays:limitTau0 = on` and by the
standalone script; "full chain" means PYTHIA from the hard event of chapter
1 with the CP5 settings and decays on, 2000 events with the impact
parameter drawn. The spread of a single event follows +-. "The B" is the
weakly decaying hadron of the b quark, Bbar0 in every same-hadron column;
in the full chain it is a B- in 44% of the events, a Bbar0 in 43, a
Bbar_s0 in 10 and a Lambda_b0 in 4. "Visible" excludes the neutrinos.
Commands: `validate_decays_with_pythia.py -n 400` (column 1),
`standalone_hadron_decays.py --repeat 400` (column 2), `--mode full -n 2000`
(column 3), `standalone_hadron_decays.py` (column 4), the drawer's table of
the CMSSW log (column 5); the last three rows are `--mode single --id 511 -n 20000`
and `--single 511 --repeat 20000`.

| quantity | PYTHIA, same hadrons | standalone, same hadrons | PYTHIA full, b drawn | standalone, seed 1 | CMSSW, 1 event |
|---|---|---|---|---|---|
| final particles | 739 +- 12 | 740 +- 12 | 459 +- 175 | 749 | 634 |
| charged / photons / neutrinos | 331 / 329 / 2.9 | 331 / 330 / 3.1 | 210 / 211 / 1.3 | 338 / 327 / 5 | -- |
| decays performed | 371 | 372 | 205 | 376 | -- |
| primary hadrons that survive | 143 | 143 | 86 | 143 | 132 |
| the B: tau [ps] | 1.41 +- 1.40 | 1.55 +- 1.46 | 1.59 +- 1.60 | 2.24 | -- |
| the B: L, L_xy [mm] | 13.2, 6.2 | 14.5, 6.8 | 18.7, 7.9 | 21.0, 9.9 | 7.83, 3.13 |
| the B: final particles (charged) | 11.1 (5.0) | 11.1 (4.9) | 11.0 (4.8) | 15 (8) | 9 |
| the B: semileptonic e / mu / tau [%] | 9.5 / 10.0 / 4.0 | 13.0 / 11.0 / 1.2 | 10.9 / 11.4 / 3.4 | hadronic | e |
| the B: E_nu [GeV], share of E_B | 13.8, 0.084 | 16.4, 0.099 | 18.6, 0.091 | 0, 0 | 96.2, 0.41 |
| the B: tracks with abs(d0) > 100 um, mean abs(d0) [um] | 3.0, 330 | 3.0, 362 | 3.0, 358 | 8, max 1640 | -- |
| the B: oscillated [%] | 3.5 (flagged) | 16.8 | 3.7 (flagged) | no | no |
| b, dR < 0.4: primaries -> final -> visible pT [GeV] | 99.23 -> 98.79 -> 92.3 +- 11.4 | 99.23 -> 98.79 -> 91.1 +- 12.2 | 109.2 -> 109.3 -> 101.4 | 99.23 -> 99.43 -> 99.43 | -- |
| b, dR < 0.4: final particles (charged) | 24.6 (9.6) | 24.7 (9.5) | 21.0 (9.2) | 31 (16) | -- |
| b, dR < 0.4: charged pT [GeV], B share | 51.3, 0.788 | 49.9, 0.788 | 60.6, 0.799 | 67.4, 0.79 | -- |
| b, dR < 0.4: mass, primaries -> final [GeV] | 16.1 -> 14.1 | 16.1 -> 14.2 | 14.2 -> 12.3 | 16.1 -> 14.6 | -- |
| b, dR < 0.8: final -> visible pT [GeV] | 107.2 -> 100.7 | 107.2 -> 99.5 | 119.2 -> 111.3 | 108.73 -> 108.73 | -- |
| bbar: L, L_xy [mm] | 119, 7.7 | 117, 7.6 | 152, 7.4 | 146.9, 9.5 | 117.3, 5.5 |
| bbar, dR < 0.4: final -> visible pT [GeV] | 92.4 -> 85.7 +- 11.3 | 92.4 -> 84.7 +- 12.7 | 106.3 -> 99.1 | 92.99 -> 92.99 | -- |
| B0 at rest (20 000): final / charged / photons | 10.87 / 4.85 / 4.50 | 11.11 / 4.90 / 4.77 | -- | -- | -- |
| B0 at rest: semileptonic (e + mu) [%], `<E_l>` [GeV] | 20.7, 1.383 | 20.5, 1.396 | -- | -- | -- |
| B0 at rest: D- / Dbar0 / D0 / D+ [%] | 35 / 27 / 11 / 10 | 30 / 28 / 12 / 9 | -- | -- | -- |

Everything the detector will see agrees within the statistics of 400
decays: the multiplicities to 0.2%, the flight distances, the neutrino
energy, the impact parameters, the cone momenta to 0.1 GeV (final) and
1 GeV (visible). The one bookkeeping difference: PYTHIA marks an oscillated
B0 with status 92 in only a third of the cases where its charm hadron has
the wrong sign; the script marks all of them.

### 5.2 Kinematics of the two b quarks, their hadrons and their cones after the decays

From `event_Decay.lhe` as printed by `standalone_hadron_decays.py`. The
primary-hadron rows are those of the chapter 5 companion; "final" counts
every particle with a positive status, "visible" excludes the neutrinos.

| | | pT [GeV] | m [GeV] | n | E [GeV] |
|---|---|---:|---:|---:|---:|
| b | its B*bar0 (row 1012, chapter 5) | 78.56 | 5.33 | 1 | 166.04 |
| | its Bbar0 (row 1527, L = 21.0 mm) | 78.26 | 5.28 | 1 | 165.47 |
| | the B*bar0's final descendants | 78.56 | 5.33 | 16 (8 charged) | 166.04 |
| | primary hadrons within dR < 0.4 (chapter 5) | 99.23 | 16.08 | 6 | |
| | final particles within dR < 0.4 | 99.43 | 14.59 | 31 (16 charged) | |
| | visible (no neutrino) | 99.43 | 14.59 | 31 | |
| | charged | 67.43 | | 16 | |
| | primary hadrons within dR < 0.8 | 105.98 | 35.87 | 14 | |
| | final particles within dR < 0.8 | 108.73 | 42.14 | 51 (26 charged) | |
| bbar | its B*0 (row 1069, chapter 5) | 82.83 | 5.33 | 1 | 1276.92 |
| | its B0 (row 1590, L = 146.9 mm, oscillated) | 81.97 | 5.28 | 1 | 1264.18 |
| | primary hadrons within dR < 0.4 | 91.34 | 7.58 | 4 | |
| | final particles within dR < 0.4 | 92.99 | 13.78 | 17 (8 charged) | |
| | primary hadrons within dR < 0.8 | 116.92 | 47.92 | 11 | |
| | final particles within dR < 0.8 | 111.19 | 40.75 | 37 (18 charged) | |
| all final particles | | 0.02 | 13600 | 749 | 13600 |

### 5.3 The two B hadrons of our event, decay by decay

| | b quark's Bbar0 (row 1527) | bbar quark's B0 (row 1590) |
|---|---|---|
| r, tau | 0.7682, 2.237 ps = 1.46 tau0 | 0.7374, 2.046 ps = 1.34 tau0 |
| gamma beta, L, L_xy | 31.33, 21.01 mm, 9.94 mm | 239.4, 146.88 mm, 9.52 mm |
| decay vertex [mm] | (-9.13, -3.93, -18.51) | (7.97, 5.21, -146.57) |
| P(oscillated), r | 0.289, 0.519: no | 0.246, 0.061: oscillated |
| channel | 850 of 870, `cbar s c dbar` (BR 7.40%, meMode 43) -> D*_s- D*0 rho+ | 837, `cs_1 udbar_1` (BR 0.60%, meMode 63) -> Xi_c0 pi+ Deltabar- |
| final particles from the B | 15 (8 charged, 6 photons, 0 neutrinos) | 9 (6 charged, 2 photons, 0 neutrinos) |
| charm hadron and its walk | D0, pT 26.97 GeV, 10.98 mm | Xi_c0, pT 71.43 GeV, 21.02 mm |
| charged tracks from the B | 8, all above 100 um, the largest 1640 um | 6, 5 above 100 um, the largest 417 um |

### 5.4 The mother's hadron in both records

| | standalone (seed 1) | CMSSW |
|---|---|---|
| the weakly decaying hadron | Bbar0, row 1527, 78.26 GeV | B-, row 829, 94.279 GeV |
| its decay | hadronic, D*_s- D*0 rho+ | semileptonic, nu_ebar e- D*_00 |
| L, L_xy | 21.0 mm, 9.9 mm | 7.83 mm, 3.13 mm |
| final particles from the B | 15 | 9 |
| the neutrino | none | 37.457 GeV of pT, 96.2 GeV of energy (f_nu = 0.41) |
| the other B hadron | B0 (oscillated), 146.9 mm along the beam, L_xy 9.5 mm | B+, 117.3 mm along the beam, L_xy 5.53 mm, 16 final particles |

### 5.5 The energy ledger

| Stage | pT [GeV] | m [GeV] | n_const | delta pT [GeV] |
|---|---:|---:|---:|---:|
| b family after hadronisation (dR < 0.4) | 99.23 | 16.08 | 6 | |
| b family after this stage (dR < 0.4, final) | 99.43 | 14.59 | 31 | +0.20 |
| of which visible (no neutrino) | 99.43 | 14.59 | 31 | 0.00 |
| of which charged | 67.43 | | 16 | |
| of which the Bbar0's descendants | 78.26 | 5.28 | 15 | |
| the same within dR < 0.8 | 108.73 | 42.14 | 51 | +2.75 |
