# Chapter 2 companion: The Mother Gives Birth (PS)

This is the technical companion of chapter 2, *The Mother Gives Birth: Parton
Showering*. The chapter gives the physics; this page says how to run the
scripts, how to read what they write, how each number of the official
packages was obtained, and lists every table the chapter quotes with the
command that produced it. All files are in
[`example/PS/`](../example/PS/) of this repository unless stated otherwise;
the CMSSW analysers are in [JitJetSW](https://github.com/ravindkv/JitJetSW).

| | |
|---|---|
| Stage | the interleaved initial- and final-state parton shower of the hard event of chapter 1 |
| Standalone script | [`example/PS/Standalone/standalone_parton_showering.py`](../example/PS/Standalone/standalone_parton_showering.py) |
| Input | [`example/ME/event_ME.lhe`](../example/ME/event_ME.lhe), the hard-process record of chapter 1 |
| Event record | [`example/PS/event_PS.lhe`](../example/PS/event_PS.lhe) (seed 1, 158 rows) |
| Learning-mode log | [`example/PS/Standalone/standalone_parton_showering.log`](../example/PS/Standalone/standalone_parton_showering.log) (`-v`, 1683 lines) |
| Validation with PYTHIA | [`example/PS/Standalone/validate_shower_with_pythia.py`](../example/PS/Standalone/validate_shower_with_pythia.py) |
| CMSSW log and figures | [`example/PS/CMSSW/`](../example/PS/CMSSW/) |
| Film | scene 2 of the [JitJet playlist](https://www.youtube.com/playlist?list=PLbSW20iyZFd4) |

Contents: [1. Running the standalone script](#1-running-the-standalone-script) ·
[2. Reading the record](#2-reading-the-record) ·
[3. Drawing the event and rendering the film](#3-drawing-the-event-and-rendering-the-film) ·
[4. Validation from official packages](#4-validation-from-official-packages) ·
[5. The full tables](#5-the-full-tables)

## 1. Running the standalone script

`standalone_parton_showering.py` takes the hard process of chapter 1 and
dresses it with an interleaved initial- and final-state shower written from
scratch, following the construction of PYTHIA 8's `SimpleTimeShower` and
`SimpleSpaceShower`: about a thousand lines of Python needing NumPy and
SciPy, no LHAPDF, no PYTHIA. The PDF grid (member 0 of NNPDF3.1 on LHAPDF's
own knots, all flavours, x >= 1e-3, Q up to 140 GeV) is embedded between the
markers `BEGIN/END GENERATED TABLES`, produced by `dump_shower_tables.py` in
an environment that has LHAPDF (`../../ME/Standalone/.venv/bin/python
dump_shower_tables.py`).

A map of the file, with the step numbers of the chapter's recipe:

| Function or class | What it does | Recipe step |
|---|---|---|
| `AlphaStrong` | second-order running with flavour thresholds; Lambda_3,4,5 = 0.3820, 0.3282, 0.2262 GeV | 2 |
| `TablePDF` | bicubic spline of x f(x, Q^2) in (ln x, ln Q^2), frozen below the grid | 3 |
| `Particle`, `Event`, `read_event`, `listing` | PYTHIA's record format, in and out | 1, 6 |
| `dipole_ends` | the colour dipoles of the current event and their ends | 1 |
| `Shower.pT2next_fsr`, `Shower.pT2next_isr` | the veto algorithm for a dipole end and for a beam | 3 |
| `fsr_kinematics`, `isr_kinematics` | the recoils | 4 |
| `branch_fsr`, `branch_isr`, `Shower.run` | execute the winner, repeat from its scale | 4, 5 |
| `verify` | energy-momentum, masses and colour lines after every branching (`--check`) | |
| `family`, `summarise`, `main` | the cone summary and the printout | 6 |

From the repository root:

```bash
cd example/PS/Standalone
python3 standalone_parton_showering.py                       # seed 1 -> ../event_PS.lhe
python3 standalone_parton_showering.py --seed 7 --history h.json
python3 standalone_parton_showering.py --no-isr              # the two b quarks only
python3 standalone_parton_showering.py --check               # assert conservation laws
python3 standalone_parton_showering.py --repeat 2000 --quiet --history s.json
python3 standalone_parton_showering.py -v > standalone_parton_showering.log   # learning mode
python3 standalone_parton_showering.py -vv | less                             # ... every trial too
```

The first command takes about a second (0.8 s of it parsing the embedded
tables, 0.3 s for the shower) and prints

```text
input ../../ME/event_ME.lhe: hard process g u -> bbar b, pTmax = 113.32 GeV, seed = 1
shower finished: 5 ISR + 33 FSR branchings, 40 final-state partons (27 gluons); ...
incoming after ISR: g x = 1.7140e-02 (+z), u x = 4.6999e-01 (-z)
parton  pT before  pT after     eta     phi  family n  family pT  family E  family m
bbar       113.22     73.18  -3.409   0.577         3      92.13   1414.97      7.02
b          113.22     78.12  -1.379  -2.728         3     100.45    224.59     13.52
branchings in pT order:
    1 ISR q<-g   side A(+z)  g  -> ubar + u   pT =  53.877  z = 0.779  x -> 2.7914e-03
    2 ISR q<-q   side B(-z)  u  -> u    + g   pT =  51.361  z = 0.850  x -> 4.5920e-01
    ...
```

followed by the 38 branchings and the record, written to `../event_PS.lhe`.
The output with `--seed 1` is byte-for-byte the committed record; any other
seed gives another family. With `--history` the branchings and the summary
go to a JSON file as well; with `--repeat N` the script showers the event N
times with seeds 1..N and writes the per-event quantities of section 5 to
`<history>.stats.json` (`standalone_shower_stats.json` is the committed
2000-shower run, about ten minutes).

| Option | Meaning | Default |
|---|---|---|
| `-i`, `-o` | input record, output record | `../../ME/event_ME.lhe`, `../event_PS.lhe` |
| `--seed` | the random seed | 1 |
| `--alphas`, `--order` | alpha_s(m_Z) of the shower and its running order | 0.118, 2 (CP5) |
| `--ptmin-fsr`, `--ptmin-isr` | `TimeShower:pTmin`, `SpaceShower:pTmin` [GeV]; raised to 1.6 and 1.1 Lambda_3 as PYTHIA does | 0.5, 0.2 |
| `--pt0` | `SpaceShower:pT0Ref` [GeV] | 2.0 |
| `--ptmax` | the starting scale [GeV] | sqrt(pTHat^2 + (m3^2 + m4^2)/2) = 113.32 |
| `--radius` | the family cone | 0.4 |
| `--no-isr`, `--no-fsr` | switch one shower off | |
| `--no-dampen` | switch off `TimeShower:dampenBeamRecoil` | |
| `--repeat N`, `--quiet` | N showers with seeds 1..N, no per-event printout | 0 |
| `--check` | assert the conservation laws after every branching | |
| `--history` | write the branchings and the summary as JSON | |
| `-v`, `-vv` | the learning modes, below | off |

### The learning modes

`-v` prints, before the shower, a table of alpha_s at the scales that matter
(second order, the first-order form used for sampling, and their ratio,
which is the veto weight), the input record with x_A, x_B, s_hat, pTHat, the
starting scale and the cut-offs with their formulas, the parton densities at
the two incoming x and the PDF ratios the backward evolution needs, and the
colour dipoles with their masses and the largest pT each can host. Then, for
every step: the competition (each dipole end and each beam with its proposed
pT, z and channel, or "no emission"), the winner, and the winning branching
worked out in full (the evolution variable turned into the virtuality or the
spacelike Q^2 and x' = x/z, alpha_s at the scale, the PDF ratio, the
dipole-rest-frame numbers, every momentum before and after, and the
conservation checks). At the end: how many trials the veto algorithm made,
the branching list in pT order and the record.

`-vv` adds every trial of the veto algorithm: the overestimate integrals,
the random number and the pT it produces, the chosen z, and each factor of
the acceptance weight (kernel over its overestimate, alpha_s ratio,
`dampenBeamRecoil`, PDF ratio) with the accept/reject decision. About 1700
lines for `-v` and 27 000 for `-vv` with seed 1. The random numbers are not
touched, so the record is identical with and without the flags. The `-v`
output for seed 1 is committed as `standalone_parton_showering.log`, and
every number quoted in the theory section of the chapter is taken from it.

## 2. Reading the record

### Status codes

The record keeps PYTHIA's conventions so that the same reader serves the
standalone and the CMSSW output. Status `-21` and `23` are the hard process
as in chapter 1; `43` a parton emitted by ISR, `-41` the new incoming
parton, `44` a final-state parton shifted by an ISR recoil; `51` the two
daughters of an FSR branching, `52` a final-state recoiler, `-53` an
incoming recoiler. A negative status means the parton has been replaced by
its daughters and is no longer in the event; the final state is every row
with a positive status, and its momentum sum equals that of the two current
incoming partons. Every row names its mother, so the family of any final
parton is found by walking up until a hard-process row (`-23`) or an ISR
emission (`43`) is reached; that is what `plot_journey_PS.py` does.

### The first 28 rows: the hard process and the first five branchings

```text
    no         id  name            status     mothers   daughters     colours      p_x        p_y        p_z         e          m
     0         90  (system)           -11     0     0     0     0     0     0      0.000      0.000      0.000  13600.000  13600.000
     1       2212  (p+)               -12     0     0     3     0     0     0      0.000      0.000   6800.000   6800.000      0.938
     2       2212  (p+)               -12     0     0     4     0     0     0      0.000      0.000  -6800.000   6800.000      0.938
     3         -2  (ubar)             -21     7     0     5     6     0   101      0.000      0.000     14.785     14.785      0.000
     4          2  (u)                -21    11     0     5     6   102     0      0.000      0.000  -2652.934   2652.934      0.000
     5         -5  (bbar)             -23     3     4     9     9     0   101     62.071     94.690  -2413.001   2415.661      4.800
     6          5  (b)                -23     3     4    10    10   102     0    -62.071    -94.690   -225.148    252.059      4.800
     7         21  (g)                -41    16     0     3     8   103   101      0.000      0.000     18.981     18.981      0.000
     8          2  (u)                -43     7     0    13    13   103     0     14.618    -42.814   -169.985    175.903      0.000
     9         -5  (bbar)             -44     5     0    14    14     0   101     60.756     98.541  -2327.506   2330.388      4.800
    10          5  (b)                -44     6     0    15    15   102     0    -75.374    -55.726   -136.461    165.624      4.800
    11          2  (u)                -41    43     0     4    12   104     0      0.000      0.000  -3122.560   3122.560      0.000
    12         21  (g)                -43    11     0    18    18   104   102    -27.723    -23.911   -237.210    240.018      0.000
    13          2  (u)                -44     8     0    19    19   103     0     16.425    -41.256   -185.382    190.626      0.000
    14         -5  (bbar)             -44     9     0    20    20     0   101     85.094    119.531  -2529.619   2533.875      4.800
    15          5  (b)                -44    10     0    21    21   102     0    -73.796    -54.365   -151.368    177.022      4.800
    16         21  (g)                -41    81     0     7    17   103   105      0.000      0.000     92.210     92.210      0.000
    17         21  (g)                -43    16     0    24    24   101   105     41.138     -8.971     66.982     79.117      0.000
    18         21  (g)                -44    12     0    27    27   104   102    -30.767    -23.247   -260.841    263.676      0.000
    19          2  (u)                -44    13     0    45    45   103     0     10.742    -40.016   -159.504    164.797      0.000
    20         -5  (bbar)             -44    14     0    22    23     0   101     80.482    120.537  -2445.151   2449.447      4.800
    21          5  (b)                -44    15     0    25    26   102     0   -101.595    -48.303   -231.838    257.733      4.800
    22         -5  (bbar)             -51    20     0    42    42     0   106     76.769     53.289  -1464.413   1467.400      4.800
    23         21  (g)                -51    20     0    30    30   106   101      3.993     67.187   -980.280    982.588      0.000
    24         21  (g)                -52    17     0    28    29   101   105     40.857     -8.910     66.525     78.577      0.000
    25          5  (b)                -51    21     0    46    46   107     0    -90.766    -41.636   -193.075    217.424      4.800
    26         21  (g)                -51    21     0    47    47   102   107    -31.219    -22.073   -211.624    215.050      0.000
    27         21  (g)                -52    18     0    48    48   104   102    -10.377     -7.841    -87.979     88.935      0.000
```

Rows 7 and 8 are branching 1: the ubar of row 3 has acquired mother 7, a
gluon of x = 2.79e-3 (status `-41`, p_z = +18.98 GeV), and the u quark of
row 8 (status `-43`, later replaced by its own daughters) is the emitted
parton, with the transverse momentum sqrt(14.6^2 + 42.8^2) = 45.2 GeV that
the whole downstream system now balances; rows 9 and 10 are the two b quarks
after the recoil, status `-44`. Rows 11 to 21 repeat the pattern twice. Row
21 is the mother just before her first emission, at 112.5 GeV; rows 25 and
26 are her daughters, a b of 99.9 GeV and a gluon of 38.2 GeV, status `-51`
because both branch again, and row 27 is the recoiling gluon of the dipole.

### The mother's last copy and the momentum sum

```text
   124          5  b                   52    89     0     0     0   113     0    -71.531    -31.404   -145.341    165.076      4.800
   ...
                                   Charge sum:  0.667           Momentum sum:      0.000      0.000  -3079.359   3312.460   1220.630
```

Row 124 is the mother at the end of the shower, at 78.12 GeV. The momentum
sum over the 40 final partons is that of the two current incoming partons:
p_z = -3079.4 GeV and E = 3312.5 GeV, that is x_A = 0.0171 and x_B = 0.470
of 6800 GeV each, and a mass of 1220.6 GeV where the hard process had 396.
The charge sum of 0.667 is the u quark emitted in branching 1.

### The learning-mode log, section by section

Section 1, the running coupling (the ratio is the veto weight):

```text
    nf = 3: Lambda_3 = 0.3820 GeV
    nf = 4: Lambda_4 = 0.3282 GeV
    nf = 5: Lambda_5 = 0.2262 GeV
     Q [GeV]  nf  alpha_s (2nd)  alpha_s (1st)   ratio   (the ratio is the veto weight)
       0.611   3         1.5626         1.4854  1.0520
       1.000   3         0.5304         0.7254  0.7312
       4.800   5         0.2160         0.2683  0.8051
      10.000   5         0.1783         0.2163  0.8242
      91.188   5         0.1180         0.1366  0.8638
     113.323   5         0.1143         0.1318  0.8666
```

Section 2, the hard event and the scales:

```text
  incoming partons: #3 ubar x_A = |p_z|/E_beam = 2.17429e-03 (+z), #4 u x_B = 3.90137e-01 (-z)
  s_hat = x_A x_B s = 156896.66 GeV^2, sqrt(s_hat) = 396.102 GeV; pTHat = 113.221 GeV
  starting scale: pTmax^2 = pTHat^2 + (m3^2 + m4^2)/2 = 12819.005 + 23.040 -> pTmax = 113.323 GeV
  cut-offs: pTmin(FSR) = max(TimeShower:pTmin, 1.6 Lambda_3 = 0.6112) = 0.6112 GeV, pTmin(ISR) = max(SpaceShower:pTmin, 1.1 Lambda_3 = 0.4202) = 0.4202 GeV, pT0(ISR) = 2.00 GeV
  the shower fills the range pT = 113.32 -> 0.611 GeV, 7.5 octaves; alpha_s runs from 0.1143 to 1.563 over that range
```

Section 3, the parton densities and the PDF ratios the backward evolution
needs (at Q^2 = pTmax^2, for z = 0.5 and 0.9):

```text
             x        g        u        d     ubar     dbar        s     sbar        c        b
    2.1743e-03  20.3955   1.2566   1.2124   1.1454   1.1482   1.0654   1.0664   0.8421   0.6075
    3.9014e-01   0.0312   0.2107   0.0663   0.0012   0.0029   0.0053   0.0006   0.0028   0.0004
  backward evolution of side A (ubar, x = 2.1743e-03):
    q<-q  mother ubar  emits g    :    0.7390    0.9565
    q<-g  mother g     emits u    :   12.0640   16.8237
  backward evolution of side B (u, x = 3.9014e-01):
    q<-q  mother u     emits g    :    0.0244    0.7638
    q<-g  mother g     emits ubar :    0.0000    0.0926
```

Section 4 and step 1, the dipoles and the first competition:

```text
    #5 bbar (acol 101) against #3 ubar [beam]  m_dip =  377.900 GeV, largest pT the dipole can host = sqrt(m2DipCorr)/2 =  188.935 GeV
    #6 b (col 102) against #4 u [beam]         m_dip =  377.900 GeV, largest pT the dipole can host = sqrt(m2DipCorr)/2 =  188.935 GeV
==== step 1: evolve downwards from pT = 113.323 GeV ====
  FSR #5 bbar (acol 101) against #3 ubar [beam]  ->  proposes q->qg     pT =    2.159  z = 0.9532
  FSR #6 b (col 102) against #4 u [beam]         ->  proposes q->qg     pT =    5.241  z = 0.9819
  ISR side A(+z) #3 ubar  x = 2.1743e-03         ->  proposes q<-g      pT =   53.877  z = 0.7789  x' = 2.7914e-03
  ISR side B(-z) #4 u     x = 3.9014e-01         ->  proposes q<-q      pT =   12.700  z = 0.7973  x' = 4.8932e-01
  winner: ISR q<-g on side A(+z) at pT = 53.877 GeV
```

Each branching then gets a block headed `---- branching n: ...` with the
kinematics worked out in full; the chapter quotes branching 1 (the x' and
Q^2 of the backward step), branching 4 (the dipole-frame numbers of the
bbar's first emission) and branching 5 (the mother's first emission). The
log ends with the 38 branchings in pT order:

```text
    1 ISR q<-g      side A(+z)  g    -> ubar + u     pT =   53.877  z = 0.779  x -> 2.7914e-03
    2 ISR q<-q      side B(-z)  u    -> u    + g     pT =   51.361  z = 0.850  x -> 4.5920e-01
    3 ISR g<-g      side A(+z)  g    -> g    + g     pT =   42.157  z = 0.206  x -> 1.3565e-02
    4 FSR q->qg     bbar  against g          pT =   34.218  z = 0.594  m_virt =   69.84
    5 FSR q->qg     b     against g          pT =   32.070  z = 0.601  m_virt =   65.67
   ...
   10 FSR g->qqbar  g     against bbar       pT =    5.967  z = 0.350  m_virt =   12.51
   13 FSR q->qg     b     against g          pT =    4.402  z = 0.864  m_virt =   13.71
   17 FSR q->qg     bbar  against b          pT =    1.866  z = 0.804  m_virt =    6.72
   ...
   38 FSR q->qg     dbar  against g          pT =    0.644  z = 0.743  m_virt =    1.47
wrote example/PS/event_PS.lhe (158 rows, 40 final-state partons)
```

The veto-algorithm bookkeeping the chapter quotes (3513 final-state and 212
initial-state trials, 433 and 34 accepted proposals, 33 and 5 winners; 14
weights above one) is printed by `-v` at the end of the shower, and the
mother's round-1 proposal trial by trial (59.7 -> 47.5 -> 5.24 GeV, the four
factors of the weight 0.9818 x 0.8076 x 0.2747 x 0.9810 = 0.2137 against
R' = 0.0623) is in the `-vv` output.

## 3. Drawing the event and rendering the film

### The (eta, phi) display

`plot_journey_PS.py` reads the record and draws every final-state parton in
the (eta, phi) plane in the layout of the CMSSW `genPartAnalyzer`,
classifying each parton by walking its mother chain back to the hard b, the
hard bbar or an ISR emission, and prints a table of all partons with their
origin, the momentum sums of the chapter's "children who run away" section
and the x walk of the incoming partons. It needs matplotlib only:

```bash
python3 example/PS/plot_journey_PS.py                  # -> example/PS/journey_PS.pdf
python3 example/PS/plot_journey_PS.py -i my.lhe --label-min 10 --png
```

### The theory plots

`plot_theory_PS.py` replays the seed-1 shower with the classes of the shower
script, reads the committed record with the parser of `plot_journey_PS.py`
and parses the `-v` log, so that every curve and every marker can be traced
to a line of the log. It writes `theory_parton_showering.pdf`, one page per
theory subsection (about a minute; `--trials 400` is faster):

| Page | Chapter figure | Content |
|---|---|---|
| 1 | why a quark cannot travel alone | the propagator factor against the angle, massless and for our mother; the 38 branchings per octave with the coupling |
| 2 | the splitting functions | the four kernels with the 33 FSR branchings at their z; the z distribution by kind |
| 3 | the Sudakov form factor | the measured survival curves of the four radiators against the leading-log formula; who radiates first |
| 4 | final-state radiation | pT_evol against the laboratory pT of the emitted parton; the beam-recoil damping factor |
| 5 | initial-state radiation | the backward-evolution integrand of both beams; the pT0 regularisation |
| 6 | interleaving | the ladder against the round number; the competitors of every round |
| 7 | the CP5 coupling | alpha_s with its thresholds and the first-order overestimate; their ratio |
| 8 | the shower step by step | every proposal of every round and the winner |
| 9 | the dead cone | the mass correction of the b -> bg kernel; the angular density |
| 10 | children who run away | the mother's pT copy by copy; her descendants in (dR, pT); the family pT against the cone radius |

`plot_veto_algorithm.py` draws the four steps of the veto algorithm for the
mother's first proposal in round 1 (the kernel and its overestimate, the
coupling and its first-order overestimate, the three inversions in a row,
the four factors of the acceptance weight) into `veto_algorithm.pdf`; it
imports `AlphaStrong` from the shower script and uses the random numbers of
the `-vv` log.

```bash
python3 example/PS/Standalone/plot_theory_PS.py        # -> theory_parton_showering.pdf
python3 example/PS/Standalone/plot_veto_algorithm.py   # -> veto_algorithm.pdf
make figures                                           # both, from the book directory
```

### The film

`animate_standalone_parton_showering.py` imports the standalone shower and
renders a narrated four-scene film of about four and a half minutes, so that
every number on the screen, and every number the narrator says, is computed
by the same classes that wrote the record:

1. **shower**: a scale runs down from pTmax = 113.3 GeV to the cut-off and
   each time it passes the pT of one of the 38 branchings the branching
   happens; a ladder on the left fills rung by rung, the (eta, phi) picture
   on the right fills with partons joined by their colour dipoles. The scene
   replays the seed-1 shower and checks itself against the record to half a
   MeV.
2. **sudakov**: the four ends that can radiate from pTmax each propose
   1500 times; the survival curves (the measured no-emission probabilities)
   on the left, the histogram of who radiates first on the right.
3. **recoil**: the shower runs again while the transverse plane keeps the
   momentum accounts and the two incoming partons walk up in x on top of
   the PDFs at the current scale.
4. **ensemble**: the same hard event with seeds 1, 2, 3, ...: twelve
   outcomes side by side and the running means of the branching counts and
   of the family pT settling, after 64 showers, next to the PYTHIA values.

It needs matplotlib in addition to NumPy and SciPy, `ffmpeg` for an `.mp4`,
and for the voice the `edge-tts` package; the narration and the score come
from the shared `example/soundtrack.py`, and each phase of a scene lasts at
least as long as its line, so `--no-narration` gives the silent cut of
three quarters of a minute.

```bash
cd example/PS/Standalone
python3 animate_standalone_parton_showering.py           # the film and its .srt, about an hour
python3 animate_standalone_parton_showering.py --fast    # low-resolution preview
python3 animate_standalone_parton_showering.py --print-script   # the narration, to record it yourself
python3 animate_standalone_parton_showering.py --no-narration   # music only, about 45 s
python3 animate_standalone_parton_showering.py --scene sudakov -o sudakov.gif
python3 animate_standalone_parton_showering.py --stills animate_standalone_parton_showering_stills.pdf --no-video
```

The last command writes the key frames (one page per scene) that the
chapter prints; `make figures` runs it. The physics options are those of
the shower script, so `--seed 7` films another family, `--no-dampen` shows
what the beam-recoil damping does to the survival curves, and `--ptmax 50`
starts the shower lower; `--trials` and `--showers` set the statistics of
scenes 2 and 4, `--jobs` renders the frames in parallel, and the narration
clips are cached in `build/narration/PS/`.

## 4. Validation from official packages

A shower is a random process, so the comparison with PYTHIA is made on
averages over two thousand showers of the same hard event, and the
comparison with the single CMSSW event is made on structure, not on numbers.
MadGraph does not appear: it computes matrix elements and hands its events
to PYTHIA for the shower.

### 4.1 The same shower with PYTHIA 8

`validate_shower_with_pythia.py` writes the hard event of chapter 1 two
thousand times into a Les Houches file, with the scale `SCALUP` set to
pTmax = 113.32 GeV, and showers the copies with PYTHIA 8.315 in the
environment of `example/ME/Standalone/` (`pythia8mc`, with the LHAPDF grid
file read through PYTHIA's own `LHAGrid1` reader, since the Python wheel has
no LHAPDF plugin). Multiparton interactions, hadronisation and QED are
switched off, the CP5 shower couplings are set, and
`SpaceShower:rapidityOrder` is switched off to match the standalone
implementation (`--rapidity-order` keeps it). The script computes the same
per-event quantities as `--repeat`, writes PYTHIA's first showered event in
the standalone record format to `validate_shower_with_pythia_event1.txt`,
the averages to `validate_shower_with_pythia.json`, and prints the two sets
of averages side by side:

```bash
cd example/PS/Standalone
PY=../../ME/Standalone/.venv/bin/python
$PY validate_shower_with_pythia.py -n 2000 --standalone-stats standalone_shower_stats.json
$PY validate_shower_with_pythia.py -n 2000 --no-isr
$PY validate_shower_with_pythia.py -n 2000 --rapidity-order
$PY validate_shower_with_pythia.py -n 2000 --ptmin-fsr 2
```

PYTHIA showers the two thousand copies in about a minute. The results are
the tables of sections 5.1 to 5.3. One trap, found on 2026-10-09: `Tune:pp = 14`
resets `PDF:pSet`, so the PDF must be given after the tune, as CMSSW does.

### 4.2 CMSSW: the shower inside the GEN step

In CMS nobody runs the shower separately. The `GEN` step of chapter 1 hands
the hard process to PYTHIA's `PartonLevel`, which runs the interleaved ISR,
FSR and multiparton interactions, adds the beam remnants and passes the
result to hadronisation, all inside the one `cmsRun` job; the output is the
`genParticles` collection, in which every stage is still visible through
the status codes. The book's own analysis package, JitJetSW, adds an
analyser that draws each stage separately:

```bash
cmsrel CMSSW_15_0_5
cd CMSSW_15_0_5/src
cmsenv
git clone git@github.com:ravindkv/JitJetSW.git
cd JitJetSW && scram b
cd Journey/test
cmsRun step1_GEN_cfg.py 2>&1 | tee step1_GEN_PS_cfg.log
cmsRun genPartAnalyzer_cfg.py
```

The first `cmsRun` is the configuration of chapter 1 with the
`customiseJitJet` line active, which adds a `ParticleListDrawer` that prints
the whole generator record of the first event; the second reads the output
file and writes one figure per stage,
`genPartAnalyzer_stage1_hardProcess_run1_event1.pdf` through stage 5. The
log and the first two figures are in `CMSSW/`.

**Reading the log.** Three things in `step1_GEN_PS_cfg.log` close the loop
with the chapter. At initialisation (line 21) the warning

```text
 PYTHIA Warning in SimpleTimeShower::init: pTmin too low , raised to 0.611
```

is the FSR cut-off max(0.5 GeV, 1.6 Lambda_3), and the settings table lists
the four CP5 shower lines:

```text
 | SpaceShower:alphaSorder                       |                        2 |            1           0           3 |
 | SpaceShower:alphaSvalue                       |                  0.11800 |      0.13650   0.0600000     0.25000 |
 | TimeShower:alphaSorder                        |                        2 |            1           0           3 |
 | TimeShower:alphaSvalue                        |                  0.11800 |      0.13650   0.0600000     0.25000 |
```

In the Info Listing of the first event, after the invariants of chapter 1,
come

```text
 Max pT scale for MPI =  1.133e+02, ISR =  1.133e+02, FSR =  1.133e+02.
 Number of MPI =    32, ISR =    28, FSRproc =    64, FSRreson =     0.
```

The first line is the starting scale, the same number for all three
interleaved processes. The second is not directly comparable with the
standalone counts: with multiparton interactions on, the 32 additional
scatterings radiate too, and the 28 initial-state and 64 final-state
branchings count all of them. The single warning
`SimpleSpaceShower::pT2nextQCD: weight above unity` (line 97) is the same
event as the script's counter of weights above one: the PDF-ratio
overestimate of the veto algorithm was exceeded once, and PYTHIA, like the
script, accepts the branching and says so.

**The drawer's table.** After the hard-process listing comes the
`ParticleListDrawer` table, 1846 rows for our event: PYTHIA's record seen
through CMSSW's `GenParticle` interface. The status column is PYTHIA's,
always positive, and the last column carries the status flags that CMS
analyses use instead of walking the record (`isHardProcess`, `isFirstCopy`,
`isLastCopyBeforeFSR`, `isLastCopy`). The mother's copies are in section
5.5.

## 5. The full tables

### 5.1 The shower of our event obtained three ways

"Family" means the final-state partons within dR < 0.4 of the b quark after
the shower. The first two columns are averages over 2000 showers of the same
hard event (the standard deviation of a single shower is quoted after +-
where it matters; the error of a mean is 45 times smaller); the third column
is the one shower with seed 1 that wrote `event_PS.lhe`; the last is the
single event of the CMSSW log, in which multiparton interactions and
hadronisation were also on. Commands: `validate_shower_with_pythia.py -n 2000`
(column 1), `standalone_parton_showering.py --repeat 2000 --quiet --history s.json`
(column 2), `standalone_parton_showering.py` (column 3), the drawer's table
of `CMSSW/step1_GEN_PS_cfg.log` (column 4).

| quantity | PYTHIA 8.315, 2000 showers | standalone, 2000 showers | standalone, seed 1 | CMSSW, 1 event |
|---|---|---|---|---|
| ISR branchings | 4.93 +- 2.9 | 4.88 +- 2.9 | 5 | -- |
| FSR branchings | 19.1 +- 11 | 20.9 +- 12 | 33 | -- |
| final partons (of which gluons) | 26.0 (20.9) | 27.7 (22.0) | 40 (27) | 57 drawn |
| partons with pT > 20 / 5 / 1 GeV | 2.8 / 7.1 / 18.6 | 2.8 / 7.3 / 19.5 | 8 / 18 / 31 | -- |
| x after ISR, +z / -z parton | 0.042 / 0.446 | 0.042 / 0.446 | 0.017 / 0.470 | -- |
| b: pT after the shower [GeV] | 94.6 +- 30 | 93.9 +- 29 | 78.1 | 91.6 |
| b family: pT / n / m [GeV] | 107.8 / 2.5 / 8.1 | 108.4 / 2.7 / 8.2 | 100.5 / 3 / 13.5 | -- |
| bbar: pT after the shower [GeV] | 92.2 | 90.1 | 73.2 | 118.6 |
| bbar family: pT / n / m [GeV] | 104.8 / 2.4 / 8.0 | 103.4 / 2.7 / 8.0 | 92.1 / 3 / 7.0 | -- |

### 5.2 Final-state radiation alone (`--no-isr` on both sides)

Averages over 2000 showers with ISR off (`PartonLevel:ISR = off` for
PYTHIA), isolating the radiation off the two b quarks:

| quantity | PYTHIA | standalone |
|---|---|---|
| FSR branchings | 6.74 +- 4.65 | 7.66 +- 4.92 |
| final partons with pT > 20 GeV | 2.48 +- 0.72 | 2.49 +- 0.70 |
| final partons with pT > 5 GeV | 4.49 +- 2.11 | 4.84 +- 2.19 |
| final partons with pT > 1 GeV | 7.39 +- 3.76 | 8.14 +- 3.99 |
| bbar: pT after / family pT / family mass [GeV] | 93.7 / 105.2 / 7.56 | 90.9 / 104.6 / 7.78 |
| b: pT after / family pT / family mass [GeV] | 93.8 / 105.9 / 7.54 | 91.8 / 105.1 / 7.74 |

### 5.3 What the options are worth

| setting | effect on the 2000-shower averages |
|---|---|
| `--ptmin-fsr 2` on both sides | a 7% excess of standalone FSR branchings remains (2.16 against 2.01): a normalisation of the soft-emission rate in the last octave above the cut-off |
| `--no-dampen` (standalone) | the FSR multiplicity more than doubles: PYTHIA's soft radiation off partons colour-connected to the beams is governed by `TimeShower:dampenBeamRecoil` |
| `--rapidity-order` (PYTHIA's default `SpaceShower:rapidityOrder = on`) | PYTHIA makes 3.4 instead of 5.1 ISR branchings and 16.3 instead of 19.5 FSR branchings; the b families change by less than 1% |

### 5.4 Kinematics of the two b quarks and their families after the shower

From `event_PS.lhe` as printed by `standalone_parton_showering.py` and
`plot_journey_PS.py`; "family" is the final partons within dR < 0.4.

| | | pT [GeV] | E [GeV] | m [GeV] | eta | phi | n |
|---|---|---:|---:|---:|---:|---:|---:|
| b | hard process (row 6) | 113.22 | 252.06 | 4.80 | -1.439 | -2.151 | 1 |
| | after the shower (row 124) | 78.12 | 165.08 | 4.80 | -1.379 | -2.728 | 1 |
| | family, dR < 0.4 | 100.45 | 224.59 | 13.52 | -1.441 | -2.706 | 3 |
| bbar | hard process (row 5) | 113.22 | 2415.66 | 4.80 | -3.753 | 0.991 | 1 |
| | after the shower (row 95) | 73.18 | 1106.94 | 4.80 | -3.409 | 0.577 | 1 |
| | family, dR < 0.4 | 92.13 | 1414.97 | 7.02 | -3.424 | 0.589 | 3 |
| b + bbar | hard process | 0 | 2667.7 | 396.1 | | | |
| | after the shower | 13.31 | 1272.0 | 236.1 | y = -2.37 | 2.45 | |
| ISR partons | after the shower | 64.39 | | | | -1.04 | 24 |
| FSR partons | after the shower | 52.06 | | | | 2.02 | 14 |
| all final partons | | 0.00 | 3312.5 | 1220.6 | | | 40 |

### 5.5 The mother copy by copy, in both records

Standalone (`event_PS.lhe`, rows and the `-v` log): 113.22 GeV at birth (row
6); 93.74, 91.7 and 112.5 GeV after the recoils of the three initial-state
branchings (rows 10, 15, 21); 99.9 GeV after her own 32 GeV emission (row
25); 80.9 GeV after the 4.4 GeV one; 78.12 GeV at the end (row 124).

CMSSW (the drawer's table of `step1_GEN_PS_cfg.log`, PYTHIA status in the
third column, flags in the last):

| row | status | pT [GeV] | flags | meaning |
|---:|---:|---:|---|---|
| 5 | 23 | 113.22 | HardProcess, FirstCopy | the hard-process b |
| 9 | 44 | 112.6 | | after an ISR recoil |
| 14 | 44 | 122.8 | | after an ISR recoil (the copy that plays `LastCopyBeforeFSR`'s role: she never branched herself) |
| 49 | 52 | 106.0 | | after recoiling against an emission of her dipole partner |
| 182 | 52 | 91.59 | | after a second such recoil |
| 515 | 62 | 91.74 | | after the beam remnants (chapter 4) took their share |
| 824 | 71 | 91.74 | LastCopy | the b handed to hadronisation |

### 5.6 The energy ledger

| Stage | pT [GeV] | m [GeV] | n_const | delta pT [GeV] |
|---|---:|---:|---:|---:|
| b quark in the hard-process record | 113.22 | 4.80 | 1 | |
| b family after the shower (dR < 0.4) | 100.45 | 13.52 | 3 | -12.77 |
| of which the b quark itself | 78.12 | 4.80 | 1 | |
