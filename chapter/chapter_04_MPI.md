# Chapter 4 companion: Cousins Nobody Invited (MPI)

This is the technical companion of chapter 4, *Cousins Nobody Invited:
Underlying Event and Colour Reconnection*. The chapter gives the physics;
this page says how to run the scripts, how to read what they write, how
each number of the official packages was obtained, and lists every table
the chapter quotes with the command that produced it. All files are in
[`example/MPI/`](../example/MPI/) of this repository unless stated
otherwise; the CMSSW analysers are in
[JitJetSW](https://github.com/ravindkv/JitJetSW).

| | |
|---|---|
| Stage | multiparton interactions, beam remnants and colour reconnection, on the showered event of chapter 2 |
| Standalone script | [`example/MPI/Standalone/standalone_multiparton_interactions.py`](../example/MPI/Standalone/standalone_multiparton_interactions.py) |
| Input | [`example/PS/event_PS.lhe`](../example/PS/event_PS.lhe), the showered record of chapter 2 |
| Event record | [`example/MPI/event_MPI.lhe`](../example/MPI/event_MPI.lhe) (seed 1, impact parameter 0.3136 `<b>`, 963 rows) |
| Learning-mode log | [`example/MPI/Standalone/standalone_multiparton_interactions.log`](../example/MPI/Standalone/standalone_multiparton_interactions.log) (`-v`, 1493 lines) |
| Validation with PYTHIA | [`example/MPI/Standalone/validate_mpi_with_pythia.py`](../example/MPI/Standalone/validate_mpi_with_pythia.py) |
| CMSSW figure | [`example/MPI/CMSSW/genPartAnalyzer_ME_PS_MPI_out.pdf`](../example/MPI/CMSSW/genPartAnalyzer_ME_PS_MPI_out.pdf) (page 3); the log is that of chapter 2 |
| Film | scene 4 of the [JitJet playlist](https://www.youtube.com/playlist?list=PLbSW20iyZFd4) |

Contents: [1. Running the standalone script](#1-running-the-standalone-script) ·
[2. Reading the record](#2-reading-the-record) ·
[3. Drawing the event and rendering the film](#3-drawing-the-event-and-rendering-the-film) ·
[4. Validation from official packages](#4-validation-from-official-packages) ·
[5. The full tables](#5-the-full-tables)

## 1. Running the standalone script

`standalone_multiparton_interactions.py` takes the showered record of
chapter 2 and adds what the rest of the two protons does: the secondary
parton-parton scatterings with their own showers, the beam remnants with
primordial transverse momentum, and the MPI-based colour reconnection. It is
a port of four classes of PYTHIA 8, `MultipartonInteractions`,
`BeamParticle`, `BeamRemnants` and `ColourReconnection::reconnectMPIs`,
into about two thousand lines of Python that need NumPy and SciPy and
nothing else. It imports the shower script of chapter 2 for the running
coupling, the event record and the shower itself, which it runs on every
secondary system; its own PDF grid, embedded between `BEGIN/END GENERATED
TABLES` by `dump_mpi_tables.py` (`../../ME/Standalone/.venv/bin/python
dump_mpi_tables.py`), is the same NNPDF3.1 set on LHAPDF's knots but
reaches down to x = 1e-9, because a 0.2 GeV scattering at rapidity 9 takes
x ~ 1e-8 from a proton.

A map of the file, with the step numbers of the chapter's flow chart:

| Class or function | What it does | Step |
|---|---|---|
| `SigmaInt` | the regularised 2 -> 2 cross section, its integral sigma_int and the overestimate of the veto algorithm | 1 |
| `Overlap` | the double-Gaussian proton profile, O(b), f(b), the b distributions | 2 |
| `Beam`, `BeamPDF` | the depleted proton: rescaled x, valence counting, companions, gluon + sea rescaling | 4 |
| `Generator.next_pt`, `trial`, `scatter` | the chain: trial scale, veto, rapidities, flavours, process, colour flow | 3 |
| `shower_system` | the chapter-2 shower on each secondary system with the depleted proton on each side | 3 |
| `remnants` | remnant flavours, momentum shapes, primordial kT, the light-cone solution | 5, 6 |
| `reconnect`, `remnant_colours`, `lambda_measure` | colour reconnection, the chains per proton, the string length | 7 |
| `summarise` | the underlying event in the three regions and the cones around the two b quarks | 8 |

```bash
cd example/MPI/Standalone
python3 standalone_multiparton_interactions.py --impact 0.3136      # seed 1 -> ../event_MPI.lhe (the book's record)
python3 standalone_multiparton_interactions.py                      # draw the impact parameter instead
python3 standalone_multiparton_interactions.py --impact 0.3136 -v   # learning mode: every number, every step
python3 standalone_multiparton_interactions.py --impact 0.3136 -vv  # ... every veto trial and the secondary showers
python3 standalone_multiparton_interactions.py --no-cr --no-kt      # colour reconnection and primordial kT off
python3 standalone_multiparton_interactions.py --repeat 400 --impact 0.3136 --quiet --history s.json
```

The first command integrates the cross section at start-up (three seconds)
and generates one event (two seconds, most of it the 24 secondary showers),
then prints

```text
pT0 = 1.4417 GeV, sigma_int = 497.33 mb, sigma_ND = 55.42 mb, <n> = 8.974; f(b) ranges up to 4.52, <f>_hard = 2.218
input example/PS/event_PS.lhe: 963 rows after MPI; seed = 1
impact parameter b = 0.3136 <b> with enhancement f(b) = 3.627; expected 32.5 interactions, got 25 (1 hard + 24 MPI) from 303 trial scales
  pT of the secondary interactions [GeV]: 11.45 7.68 4.47 3.65 3.42 3.14 2.80 2.19 2.18 1.79 1.76 1.63 1.50 1.43 1.32 1.14 1.10 0.90 0.82 0.71 0.60 0.55 0.49 0.43
  final state: 211 partons, of which 155 from the MPI systems (incl. their showers: 31 ISR + 76 FSR branchings) and 16 beam remnants; sum pT of MPI partons = 230.5 GeV
  underlying event (|eta| < 2.5, w.r.t. the leading b): toward 49.47, transverse 21.99, away 47.67 GeV; transverse density 2.100 GeV per unit eta-phi (21 partons); all-azimuth density 3.792
  bbar: pT = 72.50 GeV at eta = -3.42; cone R = 0.4: 4 partons, pT 91.27 (hard system) -> 93.88 GeV with 1 UE partons adding 2.66 GeV scalar / 2.66 vector, mass 7.02 -> 7.75; R = 0.8: pT 112.28 -> 116.07, 3 UE partons, mass 39.89 -> 41.61
  b: pT = 78.29 GeV at eta = -1.38; cone R = 0.4: 3 partons, pT 100.67 (hard system) -> 100.67 GeV with 0 UE partons adding 0.00 GeV scalar / 0.00 vector, mass 13.52 -> 13.52; R = 0.8: pT 100.67 -> 104.09, 3 UE partons, mass 13.52 -> 20.17
  colour reconnection: 23 systems merged; string length lambda = 879.6 -> 520.8; 33 colour-singlet strings
wrote example/MPI/event_MPI.lhe (963 rows, 211 final-state partons)
```

The output with `--seed 1 --impact 0.3136` is byte-for-byte the committed
record. With `--history` the chain and the summary go to a JSON file as
well; with `--repeat N` the script dresses the showered event N times with
seeds 1..N and writes the per-event quantities of section 5.1 to
`<history>.stats.json` (`standalone_mpi_stats_fixedb.stats.json` and
`standalone_mpi_stats_sampled.stats.json` are the committed 400-event runs,
about fifteen minutes each).

### Why the impact parameter is fixed in the book's record

The first thing the model draws is the impact parameter. For an event with a
hard process it is distributed as the overlap O(b) of the two protons; seed
1 lands at b = 1.21 `<b>` (f = 0.20, one secondary interaction), a quiet
event. The CMSSW event has b = 0.3136 `<b>` (f = 3.709, 32 interactions). To
walk through a rich event and to compare with CMSSW at the same geometry the
Makefile fixes `--impact 0.3136` (PYTHIA offers the same through
`UserHooks::doSetImpactParameter`); the chapter says so, and the ensembles
of section 5.1 are given both ways.

| Option | Meaning | Default |
|---|---|---|
| `-i`, `-o` | input record, output record | `../../PS/event_PS.lhe`, `../event_MPI.lhe` |
| `--seed` | the random seed | 1 |
| `--impact` | the impact parameter in units of `<b>`; drawn when absent | drawn |
| `--pt0ref`, `--ecmpow` | `MultipartonInteractions:pT0Ref` and `ecmPow`, the regulator and its energy dependence | 1.41, 0.03344 (CP5) |
| `--sigma-nd` | the non-diffractive cross section [mb] | 55.42 |
| `--ptmax` | the upper scale of the chain [GeV] | the hard scale of the record |
| `--cr-range` | `ColourReconnection:range` | 5.176 (CP5) |
| `--ptmin-isr`, `--ptmin-fsr`, `--pt0` | the cut-offs of the secondary showers | 0.2, 0.5, 2.0 |
| `--alphas`, `--order` | alpha_s(m_Z) and its running order | 0.118, 2 |
| `--radius` | the cone of the family summary | 0.4 |
| `--no-shower`, `--no-isr`, `--no-fsr` | switch the secondary showers, or one side of them, off | |
| `--no-cr`, `--no-kt` | switch colour reconnection, primordial kT off | |
| `--repeat N`, `--quiet`, `--history` | N dressings with seeds 1..N, no per-event printout, JSON output | |
| `--check` | verify the secondary showers after every branching | |
| `-v`, `-vv` | the learning modes, below | off |

### The learning modes

`-v` prints, before the event, the regularised cross section at a dozen
scales with its integral, the overlap table in b, and the state of the two
protons after the hard process; then, for every secondary interaction, the
accepted scale and its veto weight, the rapidities and momentum fractions,
the depleted densities of both protons at that x, the six largest channel
weights, the chosen process with its colour flow and momenta, and the size
of its shower; then the remnant flavours, the primordial-kT widths and
kicks, the light-cone solution, every remnant row, every colour
reconnection decision with the dipole each gluon was inserted into, the
string length before and after, the conservation checks and the full
record. `-vv` adds every trial scale of the veto algorithm and the
secondary showers' own `-v` output. The 1493 lines of `-v` for seed 1 at
b = 0.3136 `<b>` are committed as `standalone_multiparton_interactions.log`,
and every number quoted in the theory section of the chapter is taken from
that file.

## 2. Reading the record

### Status codes

The record keeps PYTHIA's conventions and adds five status codes to those
of the [chapter 2 companion](chapter_02_PS.md#status-codes). A secondary
scattering is written like the hard process: `-31` for its two incoming
partons and `-33` for its two outgoing ones, negative because every one of
them is replaced by the shower's copies. At the end of the stage every
system is boosted to the frame of its initiators after the primordial kT
and the remnant kinematics; the boosted initiators are written with status
`-61` and the boosted final partons with status `62`, so that the final
state of the record is every row with status `62` plus the beam remnants,
status `63`. Our record has 963 rows, of which 195 are status `62` and 16
are status `63`; the mother of a status-62 row is its last copy of the
shower stage, so the family of any final parton is still found by walking
up the mother column, now through a `-31` or `-33` row if the parton belongs
to a secondary system.

### The rows the chapter reads

Rows 0 to 157 are the showered event of chapter 2, every one of its final
partons now with a negative status because the stage copied them. Rows 158
to 161 are the first secondary scattering, gg -> gg at pT = 11.45 GeV, with
the colour tags of the log:

```text
   158         21  (g)                -31   166     0   160   161   136   137      0.000      0.000      3.317      3.317      0.000
   159         21  (g)                -31   162     0   160   161   137   138      0.000      0.000   -103.979    103.979      0.000
   160         21  (g)                -33   158   159   164   164   139   138      3.302     10.967    -92.564     93.269      0.000
   161         21  (g)                -33   158   159   165   165   136   139     -3.302    -10.967     -8.098     14.027      0.000
```

The outgoing gluons have p_x = +-3.302 and p_y = +-10.967 GeV, that is
pT = 11.453 GeV; the incoming gluons carry p_z = 3.317 and -103.979 GeV,
that is x_1 = 4.88e-4 and x_2 = 1.53e-2 of 6800 GeV. The next hundred rows
are its shower and the 23 further scatterings with theirs. From row 702 the
stage writes the boosted copies; 702 and 703 are the hard system's
initiators after their kicks (status `-61`):

```text
   702         21  (g)                -61   121     0     0     0   246   105     -1.767      1.652    116.537    116.563      0.000
   703          2  (u)                -61   145     0     0     0   177     0     -1.113     -1.838  -3195.905   3195.906      0.000
```

Among the status-62 copies, row 719 is the mother after the primordial-kT
boost, at 78.29 GeV, with her last shower copy (row 124) as her mother:

```text
   719          5  b                   62   124     0     0     0   113     0    -71.735    -31.354   -145.640    165.418      4.800
```

The sixteen remnants close the record (status `63`): the `ud_0` diquark of
proton A with its six companions, the `ud_1` of proton B with its eight, and
the momentum sum over the final state, which is that of the two protons:

```text
   947       2101  ud_0                63     1     0     0     0     0   311      0.549      0.040    206.571    206.572      0.579
   948          2  u                   63     1     0     0     0   311     0      0.196     -0.572     40.998     41.004      0.330
   949          2  u                   63     1     0     0     0   190     0      0.825     -0.263    145.472    145.475      0.330
   950         -2  ubar                63     1     0     0     0     0   246      0.207     -0.470     46.822     46.826      0.330
   951         -3  sbar                63     1     0     0     0     0   269      0.099     -0.207     37.225     37.229      0.500
   952          3  s                   63     1     0     0     0   275     0      0.388      0.017    227.931    227.932      0.500
   953         -2  ubar                63     1     0     0     0     0   236      0.187     -0.066      4.380      4.397      0.330
   954       2103  ud_1                63     2     0     0     0     0   318      1.150     -0.259   -208.615    208.620      0.771
   955          2  u                   63     2     0     0     0   318     0      0.147      1.081   -151.915    151.919      0.330
   956          2  u                   63     2     0     0     0   188     0      0.474     -0.148    -40.728     40.732      0.330
   957         -3  sbar                63     2     0     0     0     0   271     -0.020      0.176     -2.326      2.386      0.500
   958          2  u                   63     2     0     0     0   322     0      0.779      0.483     -4.859      4.956      0.330
   959         -3  sbar                63     2     0     0     0     0   281      0.483     -0.305    -12.055     12.079      0.500
   960         -2  ubar                63     2     0     0     0     0   151      0.543     -0.106      0.195      0.673      0.330
   961         -3  sbar                63     2     0     0     0     0   322     -0.242      0.275   -817.123    817.123      0.500
   962          1  d                   63     2     0     0     0   176     0      0.086     -0.234   -558.650    558.650      0.330
                                   Charge sum:  2.000           Momentum sum:     -0.001     -0.000     -0.002  13600.000  13600.000
```

### The learning-mode log, section by section

The log has eight sections: `setting the scene`, `the impact parameter`,
`the two protons after the hard process and its shower`, `the chain of
interactions below pTmax`, `beam remnants`, `colour reconnection`, `checks`
and `summary`. The lines the chapter quotes:

```text
pT0 = 1.4417 GeV, sigma_int = 497.33 mb, sigma_ND = 55.42 mb, <n> = 8.974; f(b) ranges up to 4.52, <f>_hard = 2.218
  E_cm = 13600 GeV; pT0 = pT0Ref (E_cm/E_ref)^ecmPow = 1.41 x (13600/7000)^0.03344 = 1.4417 GeV
  sigma_ND = sigma_tot - sigma_el - sigma_diff = 100.309 - 21.89 - 23.0 = 55.42 mb
    pT [GeV]  alpha_s   (pT2/(pT2+pT02))^2   dsigma/dpT2  sigma(>pT)
       0.200   0.3702               0.0004         599.3       497.3
       1.000   0.3298               0.1055         100.4       285.6
       1.442   0.3046               0.2500         38.66       217.9
       2.000   0.2783               0.4331         20.93       168.1
       5.000   0.2110               0.8524         1.608       33.95
      10.000   0.1778               0.9597       0.07716       5.171
      20.000   0.1533               0.9897      0.002359       0.561
     113.323   0.1143               0.9997     1.234e-07   0.0007525
  overestimate of the veto algorithm: (pT^2 + pT0^2)^2 dsigma/dpT^2 <= 9751.6 mb GeV^2
```

```text
  double-Gaussian proton (coreRadius 0.7634, coreFraction 0.63): k = 93.005 so that <n> = 8.974; <b>_ND = 1.2398; f(0) = 4.517, <f>_ND = 0.973, <f>_hard = 2.218
     b/<b>     O(b)    f(b)  P_int(b)    n(b)
      0.00   0.4479   4.517    1.0000   41.65
      0.25   0.3895   3.928    1.0000   36.23
      0.50   0.2572   2.594    1.0000   23.92
      0.75   0.1304   1.315    1.0000   12.13
      1.00   0.0518   0.522    0.9919    4.81
      1.25   0.0165   0.166    0.7840    1.53
      1.50   0.0043   0.044    0.3307    0.40
      2.00   0.0002   0.002    0.0164    0.02
  -> b = 0.3136 <b>, f(b) = 3.627 (CMSSW: b = 0.3136, f = 3.709); mean number of interactions 32.55
  proton A(+z) gave g at x = 1.7140e-02 to the hard system (gluon); X = 0.9829 is left
  proton B(-z) gave u at x = 4.6999e-01 to the hard system (valence); X = 0.5300 is left
```

The first rung of the chain, every number of which the chapter's chain
sketch reproduces:

```text
---- interaction 1: evolve downwards from pT = 113.323 GeV ----
  accepted at pT = 11.453 GeV (veto weight 0.174): rapidities y3 = -2.787, y4 = -0.658, so x1 = 4.8785e-04, x2 = 1.5291e-02, sqrt(s_hat) = 37.145 GeV
  alpha_s(pT^2 + pT0^2 = 133.251) = 0.1725; regularisation factor (pT^2/(pT^2 + pT0^2))^2 = 0.9690
  proton A(+z): X = 0.9829 left, so x/X = 4.9635e-04; valence u, d left: 2, 1; gluon + sea rescaled by 1.0000; depleted x f(x): g=21.371 u=1.224 d=1.198 ...
  proton B(-z): X = 0.5300 left, so x/X = 2.8850e-02; valence u, d left: 1, 1; gluon + sea rescaled by 1.1641; depleted x f(x): g=4.255 u=0.487 d=0.537 ...
  channel weights, the six largest: g g -> g g 65.1%, g d -> d g 3.6%, g u -> u g 3.2%, g dbar -> dbar g 2.4%, g ubar -> ubar g 2.1%, u g -> g u 1.6%
  chosen: g g -> g g; colour flow (col, acol): in A (136, 137), in B (137, 138), out3 (139, 138), out4 (136, 139)
  shower of this system from pT = 11.453 GeV: 3 ISR + 10 FSR branchings, 15 final partons
```

The remnants and the colour reconnection:

```text
==== beam remnants ====
  proton A(+z) gave away 25 partons: ... valence left: u x1, d x1; remaining momentum fraction X = 0.1100
    remnant partons: ud_0 (valence, m = 0.579), u, u, ubar, sbar, s, ubar (companions)
  proton B(-z) gave away 25 partons: ... valence left: u x1, d x1; remaining momentum fraction X = 0.2657
    remnant partons: ud_1 (valence, m = 0.771), u, u, sbar, u, sbar, ubar, sbar, d (companions)
  primordial kT: width sigma = [(Q_half sigma_soft + Q sigma_hard)/(Q_half + Q)] m/(m + m_half), Q_half = 1.5, sigma_soft = 0.9, sigma_hard = 1.8, m_half = 1.0
    system 0 (hard, Q = 113.32, m = 1220.63 GeV): sigma_kT = 1.786 GeV; kT given: A (-1.767, +1.652), B (-1.113, -1.838)
    system 1 (MPI, Q = 11.45, m = 110.99 GeV): sigma_kT = 1.676 GeV; kT given: A (-0.897, +1.012), B (-1.960, +0.096)
==== colour reconnection (MPI-based model, range = 5.176) ====
  a system of scale pT is merged into a harder one with probability P = (R pT0)^2/((R pT0)^2 + pT^2), R pT0 = 7.462 GeV; tried from the softest system upwards:
    system 24 (pT =  0.434 GeV): P = 0.9966 -> merged into system 23
    ...
    system 3  (pT =  4.47 GeV):  P = 0.736  -> merged into system 2
    system 2  (pT =  7.68 GeV):  P = 0.486  -> merged into the hard system
    system 1  (pT = 11.45 GeV):  P = 0.298  -> not merged
  string length lambda = 879.6 -> 520.8; 33 colour-singlet strings
```

## 3. Drawing the event and rendering the film

### The (eta, phi) display

`plot_journey_MPI.py` draws the record in the (eta, phi) plane in the
layout of the CMSSW `genPartAnalyzer`, reusing the reader and the frame of
chapter 2's script and adding the two origins of this stage, secondary
scatterings (cyan squares) and beam remnants (grey diamonds). It applies the
analyser's classification rule on purpose: a copy or an FSR radiator
inherits the origin of its mother, so a gluon emitted by a parton of a
secondary system is counted as FSR and an initial-state emission of a
secondary system as ISR, and only the outgoing partons of the scatterings
themselves and their descendants by recoil are counted as "MPI". That is
why the figure says 45 MPI partons while the script's summary says 155
partons from the secondary systems: the two numbers count the same partons
with different rules.

```bash
python3 example/MPI/plot_journey_MPI.py                 # -> example/MPI/journey_MPI.pdf
python3 example/MPI/plot_journey_MPI.py -i my.lhe --png
```

### The theory plots

`plot_theory_MPI.py` replays the seed-1 dressing with the classes of the
script and reads the `--repeat` statistics and the PYTHIA validation files
when they are present; it writes `theory_multiparton_interactions.pdf`, one
page per theory subsection:

| Page | Chapter figure | Content |
|---|---|---|
| 1 | the regularised cross section | the bare and regularised rate against pT; sigma_int against pT0 with the CP5 point and sigma_ND; the share of gg, qg and qq |
| 2 | the impact parameter | the double-Gaussian profile; O(b), P_int(b), f(b); the b distributions for minimum bias and hard events against the ensembles |
| 3 | the chain | the no-interaction probability; the ladder of our event against the CMSSW event's; the multiplicity against Poisson and PYTHIA |
| 4 | the depleted proton | (x1, x2) of all scatterings on the densities; the processes; the momentum fraction X left after each scattering |
| 5 | beam remnants | the remnant momentum shapes; the primordial-kT width and the kicks; the light-cone shares |
| 6 | colour reconnection | the merging probability with the 25 systems; lambda before and after; the string through the b |
| 7 | the underlying event | the transverse and all-azimuth densities of the ensembles; their distribution; `<n>` against pT0 |
| 8 | our jet at this stage | the UE pT inside R = 0.4 and 0.8; its distribution; the ledger against the cone radius |

```bash
python3 example/MPI/Standalone/plot_theory_MPI.py      # -> theory_multiparton_interactions.pdf
make figures                                           # the same, from the book directory
```

### The film

`animate_standalone_multiparton_interactions.py` imports the standalone
script and replays the seed-1 dressing at the CMSSW impact parameter in
five narrated scenes of a little over four minutes in all, once without and
once with colour reconnection:

1. **overlap**: two discs of matter (the CP5 double-Gaussian profile) slide
   towards each other in the transverse plane; the overlap O(b) and the
   enhancement f(b) rise along their curves until the collision stops at
   b = 0.3136 `<b>`, where f = 3.63 and the Poisson mean 32.5 appears.
2. **chain**: a scale runs from 113 GeV to pTmin = 0.2 GeV; every time it
   passes the pT of one of the 24 secondary scatterings the ladder gains a
   rung, the (eta, phi) picture a pair of cyan squares, and the
   no-interaction curve shows how unlikely the quiet stretch down to
   11.5 GeV was.
3. **remnants**: the momentum bars of the two protons are eaten interaction
   by interaction; the valence content changes; the grey remnant partons
   appear with their flavours and the primordial-kT kicks are drawn as
   arrows that sum to zero per proton.
4. **strings**: the colour strings in the (y, phi) plane before and after
   the MPI-based reconnection, the string length dropping from 880 to 521,
   the b quark's own string highlighted.
5. **ensemble**: the same showered event dressed with seeds 1, 2, 3, ... at
   the same impact parameter; the histograms of the number of interactions,
   the transverse density and the extra pT inside dR < 0.4 fill up and
   their running means settle next to the PYTHIA values.

```bash
cd example/MPI/Standalone
python3 animate_standalone_multiparton_interactions.py            # the film and its .srt, about an hour to render
python3 animate_standalone_multiparton_interactions.py --fast     # low-resolution preview
python3 animate_standalone_multiparton_interactions.py --print-script   # the narration, to record it yourself
python3 animate_standalone_multiparton_interactions.py --no-narration   # music only, about 55 s
python3 animate_standalone_multiparton_interactions.py --scene chain -o chain.gif
python3 animate_standalone_multiparton_interactions.py --stills animate_standalone_multiparton_interactions_stills.pdf --no-video
```

Rendering needs matplotlib in addition to NumPy and SciPy, `ffmpeg` for an
`.mp4`, and `edge-tts` for the voice; the narration and the score come from
the shared `example/soundtrack.py` (clips cached in `build/narration/MPI/`).
The last command writes the key frames (one page per scene) that the
chapter prints; `make figures` runs it. The physics options are those of
the standalone script: `--seed 7` films another dressing, `--impact 1.0` a
peripheral collision, `--events` sets the showers of scene 5, and `--no-cr`
leaves the strings as they were.

## 4. Validation from official packages

Two official sources: PYTHIA driven from a script so that it dresses
exactly our hard event, and the CMS generator job of chapter 1, which had
multiparton interactions, remnants and colour reconnection on from the
start. Ensembles are compared on averages and the single CMSSW event on
structure. MadGraph does not appear: multiparton interactions, remnants and
colour reconnection are PYTHIA's job in every CMS sample, whatever produced
the hard process.

### 4.1 The same stage with PYTHIA 8

`validate_mpi_with_pythia.py` writes the hard event of chapter 1 into a Les
Houches file N times with `SCALUP` = 113.32 GeV and lets PYTHIA 8.315, in
the environment of `example/ME/Standalone/`, run its parton level with the
CP5 settings of the CMSSW configuration: showers, multiparton interactions,
remnants and colour reconnection on, hadronisation and QED off. It reads
PYTHIA's initialisation numbers, assigns every final parton to its system
by walking the record (an ISR emission hangs on the new incoming parton,
whose other daughter is the previous one), and measures the same quantities
as the standalone `--repeat`.

```bash
cd example/MPI/Standalone
PY=../../ME/Standalone/.venv/bin/python
$PY validate_mpi_with_pythia.py -n 2000 --standalone-stats standalone_mpi_stats_sampled.stats.json
$PY validate_mpi_with_pythia.py -n 8000 --impact 0.3136 --window 0.03
$PY validate_mpi_with_pythia.py -n 500 --no-cr
$PY validate_mpi_with_pythia.py -n 500 --no-mpi          # the chapter-2 situation
```

PYTHIA dresses two thousand copies in about a minute and prints
pT0 = 1.44 GeV, sigma_int = 496.6 mb and sigma_ND = 55.42 mb at
initialisation, against the script's 497.3; the CMSSW log's 500.67 differs
from both by the version of the integration grid. The results are written
to `validate_mpi_with_pythia_{sampled,fixedb}.json` and PYTHIA's first
dressed event to `validate_mpi_with_pythia_event1.txt` in the standalone
format. Three things to know before writing your own:

- `Tune:pp = 14` resets `PDF:pSet`, so the tune must be set *before* the
  PDF, as the CMSSW configuration does; the chapter-2 validation script had
  the order reversed and its PYTHIA numbers moved by one to two percent
  when it was corrected on 2026-10-09.
- PYTHIA prints its initialisation through C++ streams, invisible to
  Python's `sys.stdout`, so the file descriptor is redirected while `init()`
  runs.
- The impact parameter cannot be fixed from the Python binding (the
  `pythia8mc` wheel offers no `UserHooks` override), so the fixed-b sample
  keeps the events with |b/`<b>` - 0.3136| < 0.03: 537 of 8000.

The tables are in section 5.1; the one real disagreement, 20% fewer
scatterings in the standalone chain, is the sequential rather than
interleaved showering of the systems, as the chapter explains.

### 4.2 CMSSW: the rest of the protons inside the GEN step

Nothing new has to be run: the `GEN` step of chapter 1 already produced
this stage, and JitJetSW's analyser drew it as its third figure. From the
setup of the [chapter 2 companion](chapter_02_PS.md#42-cmssw-the-shower-inside-the-gen-step):

```bash
cd CMSSW_15_0_5/src/JitJetSW/Journey/test
./runCMSDrivers_mc2024_BBbar_GEN.sh          # GEN step + genPartAnalyzer, one PDF per stage
```

produces `genPartAnalyzer_ME_PS_MPI_out.pdf`, whose third page is the
chapter's CMSSW figure; the copy in `CMSSW/` is the one used, and the log is
`example/PS/CMSSW/step1_GEN_PS_cfg.log` of chapter 2.

**Reading the log.** At initialisation (lines 27 and 30),

```text
 |    pT0 =  1.44 gives sigmaInteraction =   500.67 mb: accepted    |
 PYTHIA Warning in MultipartonInteractions::init: maximum increased by factor 1.090
```

PYTHIA integrates the regularised cross section at the start of every job
and accepts the regulator if the integral is large enough for the veto
algorithm's overestimate; the warning is the overestimate being tightened
by a trial scan. The settings table lists the CP5 numbers used throughout:

```text
 | ColourReconnection:range                      |                  5.17600 |      1.80000         0.0    10.00000 |
 | MultipartonInteractions:bProfile              |                        2 |            3           0           4 |
 | MultipartonInteractions:coreFraction          |                  0.63000 |      0.50000         0.0     1.00000 |
 | MultipartonInteractions:coreRadius            |                  0.76340 |      0.40000     0.10000     1.00000 |
 | MultipartonInteractions:ecmPow                |                0.0334400 |      0.21500         0.0     0.50000 |
 | MultipartonInteractions:pT0Ref                |                  1.41000 |      2.28000     0.50000    10.00000 |
```

In the Info Listing of the first event, after the hard-process invariants:

```text
 Impact parameter b =  3.136e-01 gives enhancement factor =  3.709e+00.
 Max pT scale for MPI =  1.133e+02, ISR =  1.133e+02, FSR =  1.133e+02.
 Number of MPI =    32, ISR =    28, FSRproc =    64, FSRreson =     0.
```

The first line is the value the book's record was fixed to; the second is
the common starting scale of the interleaved evolution; the third counts
the hard process among the "MPI", so the event has 31 secondary
scatterings, and its 28 initial-state and 64 final-state branchings belong
to all 32 systems together.

**The drawer's table.** The new status codes appear in the same roles as in
the standalone record: 62 rows of status 31 and 62 of status 33 (the
incoming and outgoing partons of the 31 secondary scatterings); 64 rows of
status 61 (the initiators of all 32 systems after the remnant kinematics);
156 of status 62; 13 of status 63 (the remnants); and then 120 of status 71
and 45 of status 73, the copies PYTHIA makes when it collects each
colour-singlet string and joins very soft partons for hadronisation, which
the standalone record does not have. The mother's copies are in section 5.4.

## 5. The full tables

### 5.1 The rest of the protons obtained three ways

"Secondary" means not belonging to the hard b bbar system and its shower;
"UE" (underlying event) means the partons of the secondary systems and the
remnants; cones are drawn around the final copy of the b quark. The first
four columns are averages over 2000 (PYTHIA), 400 (standalone), 537 (PYTHIA
at fixed b) and 400 dressings of the same showered event, the spread of a
single event after +-; the fifth is the one dressing with seed 1 at
b = 0.3136 `<b>` that wrote `event_MPI.lhe`; the last is the single CMSSW
event. The standalone ensembles keep the seed-1 shower of chapter 2 and
redraw only this stage; PYTHIA redraws everything. Commands:
`validate_mpi_with_pythia.py -n 2000` (column 1), `standalone_multiparton_interactions.py --repeat 400 --quiet --history s.json`
(column 2), `validate_mpi_with_pythia.py -n 8000 --impact 0.3136 --window 0.03`
(column 3), the same `--repeat 400` with `--impact 0.3136` (column 4),
`standalone_multiparton_interactions.py --impact 0.3136` (column 5), the
drawer's table of the CMSSW log (column 6).

| quantity | PYTHIA, b drawn | standalone, b drawn | PYTHIA, b = 0.3136 | standalone, b = 0.3136 | standalone, seed 1 | CMSSW, 1 event |
|---|---|---|---|---|---|---|
| interactions, incl. the hard one | 18.0 +- 9.9 | 14.7 +- 7.2 | 27.5 +- 5.4 | 21.4 +- 4.2 | 25 | 32 |
| b / `<b>`, f(b) | 0.61, 2.26 | 0.63, 2.10 | 0.31, 3.70 | 0.31, 3.63 | 0.31, 3.63 | 0.31, 3.71 |
| final partons | 125 +- 60 | 138 +- 54 | 181 +- 34 | 190 +- 33 | 211 | 169 |
| of which from secondary systems | 92 | 88 | 145 | 137 | 155 | -- |
| of which beam remnants | 10.2 | 9.4 | 13.6 | 12.9 | 16 | 13 |
| sum pT of the secondary partons [GeV] | 134 | 124 | 212 | 196 | 230 | -- |
| hardest secondary interaction [GeV] | 6.8 | 6.5 | 8.6 | 8.4 | 11.45 | 6.65 |
| transverse UE density [GeV per unit eta-phi] | 2.21 +- 1.72 | 2.00 +- 1.59 | 3.48 +- 1.54 | 3.23 +- 1.44 | 2.10 | -- |
| UE sum pT within dR < 0.4 of the b [GeV] | 1.21 +- 2.48 | 0.85 +- 1.31 | 1.80 +- 2.43 | 1.60 +- 2.19 | 0.00 | -- |
| UE sum pT within dR < 0.8 of the b [GeV] | 4.57 +- 5.20 | 3.95 +- 4.05 | 7.32 +- 5.68 | 5.88 +- 4.44 | 3.42 | -- |
| UE partons within dR < 0.4 / 0.8 | 0.75 / 2.99 | 0.72 / 2.87 | 1.20 / 4.75 | 1.07 / 4.18 | 0 / 3 | -- |
| systems merged by colour reconnection | -- | 13.2 | -- | 19.7 | 23 | -- |
| string length lambda before -> after | -- | 563 -> 375 | -- | 769 -> 472 | 880 -> 521 | -- |

Without any showering of the secondary systems (`--no-shower`) the two
chains agree to 4%: 7.2 against 7.5 interactions per unit of f.

### 5.2 Kinematics of the two b quarks and their cones after the MPI stage

From `event_MPI.lhe` as printed by `standalone_multiparton_interactions.py`.
"Hard system" means the partons that descend from the hard process and its
shower; "UE" the partons of the secondary systems and the remnants.

| | | pT [GeV] | m [GeV] | n | of which UE |
|---|---|---:|---:|---:|---|
| b | after the shower (chapter 2, row 124) | 78.12 | 4.80 | 1 | |
| | after the kick (row 719, status 62) | 78.29 | 4.80 | 1 | |
| | hard system within dR < 0.4 | 100.67 | 13.52 | 3 | |
| | all partons within dR < 0.4 | 100.67 | 13.52 | 3 | 0 |
| | hard system within dR < 0.8 | 100.67 | 13.52 | 3 | |
| | all partons within dR < 0.8 | 104.09 | 20.17 | 6 | 3 (3.42 GeV) |
| bbar | after the kick (status 62) | 72.50 | 4.80 | 1 | |
| | all partons within dR < 0.4 | 93.88 | 7.75 | 4 | 1 (2.66 GeV) |
| | all partons within dR < 0.8 | 116.07 | 41.61 | | 3 |
| all final partons | | 0.00 | 13600 | 211 | 171 |

### 5.3 The underlying event of our event in the three regions

Partons not descending from the hard system, |eta| < 2.5, azimuth relative
to the leading b: toward 49.47 GeV, transverse 21.99 GeV (21 partons), away
47.67 GeV; transverse density 21.99 / (5 x 2 pi / 3) = 2.10 GeV per unit
eta-phi; all-azimuth density of the secondary partons 3.79 GeV.

### 5.4 The mother copy by copy, in both records

Standalone: 78.12 GeV at the end of the shower (row 124, status 52);
78.29 GeV after the primordial-kT boost of the hard system (row 719, status
62). The family within dR < 0.4 moves from 100.45 to 100.67 GeV with the
same boost.

CMSSW (the drawer's table of the chapter 2 log): row 182, status 52,
91.59 GeV, her last shower copy; row 515, status 62, 91.744 GeV, after the
beam remnants took their share, the same 0.15 GeV primordial-kT boost; row
824, status 71, 91.74 GeV, the b handed to hadronisation.

### 5.5 The energy ledger

| Stage | pT [GeV] | m [GeV] | n_const | delta pT [GeV] |
|---|---:|---:|---:|---:|
| b family after the shower (dR < 0.4) | 100.45 | 13.52 | 3 | |
| b family after this stage (dR < 0.4) | 100.67 | 13.52 | 3 | +0.22 |
| of which the b quark itself | 78.29 | 4.80 | 1 | |
| the same within dR < 0.8 (3 UE partons) | 104.09 | 20.17 | 6 | +3.42 |
