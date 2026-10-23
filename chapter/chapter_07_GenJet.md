# Chapter 7 companion: The Family Photograph (GenJet)

This is the technical companion of chapter 7, *The Family Photograph:
Generator-Level Jets and Truth Definitions*. The chapter gives the physics;
this page says how to run the scripts, how to read what they write, how
each number of the official packages was obtained, and lists every table
the chapter quotes with the command that produced it. All files are in
[`example/GenJet/`](../example/GenJet/) of this repository unless stated
otherwise; the CMSSW analysers are in
[JitJetSW](https://github.com/ravindkv/JitJetSW).

| | |
|---|---|
| Stage | the generator-level jets and their flavour labels, on the final state of chapter 6 |
| Standalone script | [`example/GenJet/Standalone/standalone_genjets.py`](../example/GenJet/Standalone/standalone_genjets.py) |
| The three photographers on nine particles | [`antiKt.py`](../example/GenJet/Standalone/antiKt.py), [`caKt.py`](../example/GenJet/Standalone/caKt.py), [`hotVR.py`](../example/GenJet/Standalone/hotVR.py) |
| Input | [`example/Decay/event_Decay.lhe`](../example/Decay/event_Decay.lhe), the 749 final particles of chapter 6 |
| Event record | [`example/GenJet/event_GenJet.lhe`](../example/GenJet/event_GenJet.lhe) (the chapter 6 record plus a jet listing and a constituent listing) |
| Learning-mode log | [`example/GenJet/Standalone/standalone_genjets.log`](../example/GenJet/Standalone/standalone_genjets.log) (`-v`, 178 lines) |
| Validation with FastJet and SlowJet | [`example/GenJet/Standalone/validate_genjets_with_fastjet.py`](../example/GenJet/Standalone/validate_genjets_with_fastjet.py) |
| CMSSW figures | [`example/GenJet/CMSSW/genJetAnalyzer_ME_PS_MPI_Had_Decay_GenJet_out.pdf`](../example/GenJet/CMSSW/genJetAnalyzer_ME_PS_MPI_Had_Decay_GenJet_out.pdf) (pages 6 to 8) |
| Film | scene 7 of the [JitJet playlist](https://www.youtube.com/playlist?list=PLbSW20iyZFd4) |

Contents: [1. Running the standalone script](#1-running-the-standalone-script) ·
[2. Reading the record](#2-reading-the-record) ·
[3. Drawing the event and rendering the film](#3-drawing-the-event-and-rendering-the-film) ·
[4. Validation from official packages](#4-validation-from-official-packages) ·
[5. The full tables](#5-the-full-tables)

## 1. Running the standalone script

`standalone_genjets.py` takes the final state of chapter 6 (the 749 rows
with a positive status, CMSSW's genParticles with status 1) and clusters
it the way CMSSW's GenJet producers do, with every piece written out: the
generalised-kT sequential recombination (anti-kT for the GenJets, kT and
Cambridge/Aachen as options) in rapidity and azimuth with the E scheme and
the 3 GeV threshold, for the three collections `ak4GenJets` (R = 0.4),
`ak4GenJetsNoNu` (R = 0.4 without neutrinos) and `ak8GenJets` (R = 0.8);
the ghost association of CMSSW's `JetFlavourClustering`, giving
`hadronFlavour`, `partonFlavour`, a "physics" flavour from the status-23
partons and the legacy dR < 0.3 matching; the net flavour and its infrared
problem; and the two infrared-safe definitions of the literature,
flavour-kT and interleaved flavour neutralisation (IFN). A thousand lines
of Python that need NumPy and nothing else; it imports the decay script of
chapter 6 for the event record and the particle names.

A map of the file, with the steps of the chapter's flow chart:

| Function or class | What it does |
|---|---|
| `input_rows` | the particles that go in (`genParticlesForJets`, and `NoNu` without the neutrinos) |
| `Clustering` | the generalised-kT sequential recombination with the nearest-neighbour bookkeeping of FastJet's N^2 scheme |
| `ghost_rows`, `JetCollection.flavours` | the ghosts (last b and c hadrons, status 71-79 partons, status-23 partons, leptons from heavy hadrons) and the CMSSW labels |
| `stable_heavy_particles`, `IFN`, `flavour_kt` | the safe definitions with the heavy hadrons kept undecayed |
| `soft_pair` | the infrared test: a soft B0 Bbar0 pair between two light jets |
| `stage_summary` | the two b quarks in their jets in each collection, the cone of the same radius, the neutrinos inside, the share of the B hadron and the composition by particle type |
| `read_jet_tables` | the parser of the jet and constituent listings of the record |

```bash
cd example/GenJet/Standalone
python3 standalone_genjets.py                       # -> ../event_GenJet.lhe (the book's record)
python3 standalone_genjets.py -v                    # learning mode: the b jet merge by merge, the labels, IFN
python3 standalone_genjets.py -vv                   # ... every merge of the ak4 clustering
python3 standalone_genjets.py --algorithm kt        # kT jets instead of anti-kT (also ca); --ptmin 10
python3 standalone_genjets.py --soft-pair 0.5       # the infrared test: a soft B0 Bbar0 pair between two light jets
python3 standalone_genjets.py --repeat 400 --jobs 4 --quiet --history standalone_genjet_stats.json
```

The first command clusters the record in fifteen seconds (2 s per
collection, 8 s of it the flavour-kT clustering) and prints

```text
749 final particles (5 neutrinos); ghosts: 4 b hadrons, 9 c hadrons, 163 partons (status 71-79),
  2 hard-process partons, 3 leptons from heavy hadrons
ak4GenJets (antikt, R = 0.4, pT > 3 GeV): 42 jets, 368 of 749 particles clustered; 11 jets above 10 GeV,
  6 above 30; hadronFlavour 5: 3, 4: 2
   jet       pT     eta     phi       m    n  nch  had  part  phys  dR.3  heavy hadrons / partons in the jet
     0    99.83   -1.44   -2.71   14.83   31   17    5    +5     0    +5  Bbar0 D0 D_s- | b g g
     1    94.09   -3.45    0.59   17.83   18    8    5    -5     0    -5  Bbar0 Xi_c0 | bbar g g g
     2    60.45   -3.39    1.43   11.79   29   14    5    -5     0    -5  Lambda_bbar0 Bbar0 D+ Dbar0 | bbar b g
     3    48.12    0.28   -0.77    9.75   17    7    0    +2     0    +2   | u ubar g
  ...
b quark (row 719): pT = 78.29 GeV, eta = -1.38, phi = -2.73; its Bbar0 (row 1527) pT = 78.26
  ak4GenJets      : jet [0] pT = 99.83 GeV, eta = -1.44, phi = -2.71, m = 14.83, n = 31 (17 charged);
    dR(jet, quark) = 0.062, pT(jet)/pT(quark) = 1.275; cone of the same R: 99.43 GeV (31),
    2 in the cone but not the jet, 2 in the jet but not the cone
    the Bbar0's descendants in it: 15 with 78.26 GeV (0 outside), dR(jet, hadron) = 0.063; neutrinos 0
    with 0.00 GeV; charged hadrons 68.09, photons 22.75, neutral hadrons 8.99, leptons 0.00 GeV
    hadronFlavour 5, partonFlavour 5, physics 0, dR<0.3 5, net +1 b; b hadrons: Bbar0; c hadrons: D0 D_s-
  ak8GenJets      : jet [1] pT = 105.79 GeV, eta = -1.46, phi = -2.71, m = 27.43, n = 47 (25 charged); ...
  heavy hadrons kept : anti-kT jet [0] pT = 99.83: net flavour +1 b, IFN flavour +1 b
  flavour-kT         : jet pT = 99.83, eta = -1.44, phi = -2.71, n = 17, dR = 0.054: flavour +1 b
ak4GenJets: b jet + bbar jet: |pT(b) + pT(bbar)| = 16.89 GeV, Sigma|pT| = 193.92, dphi = 2.98, m = 302.9 GeV
IFN: no neutralisation in this event (every flavoured hadron keeps its flavour)
```

The output with the defaults is byte-for-byte the committed record. With
`--history` the summary goes to a JSON file as well (`standalone_genjets.json`);
with `--repeat N` the script decays the primary hadrons of chapter 5 N
times with the script of chapter 6, seeds 1..N, clusters each result and
writes the per-event quantities of section 5.1 to `<history>.stats.json`
(`standalone_genjet_stats.stats.json` is the committed 400-event run, half
an hour with four workers; `standalone_genjet_stats.log` its printout).

| Option | Meaning | Default |
|---|---|---|
| `-i`, `-o` | input record, output record | `../../Decay/event_Decay.lhe`, `../event_GenJet.lhe` |
| `--algorithm` | `antikt`, `kt` or `ca` | antikt |
| `--radius` | use this R for the ak4 collections instead of 0.4 | 0.4 |
| `--ptmin` | jets below this pT are dropped (CMSSW: `GenJetParameters.jetPtMin`) | 3 GeV |
| `--soft-pair PT`, `--soft-sep` | add a soft B0 Bbar0 pair of this pT, this far apart in dR, between two light jets | off, 0.5 |
| `--no-flavour-kt` | skip the flavour-kT clustering (the slowest step) | |
| `--repeat N`, `--jobs`, `--seed`, `--quiet`, `--history` | the ensemble of N decays, in this many worker processes | |
| `-v`, `-vv` | the learning modes, below | off |

### The learning modes

`-v` prints, before the summary, the setting of the scene: the input, the
distance, the three collections and the ghosts with their rows and
momenta; then the 36 merges that built the b quark's ak4 jet in the order
the algorithm made them, each with its d_ij, its dR, the two momenta and
the merged one, and what joined; the declaration of the jet and the two
particles that differ between the jet and the cone; the labels of the two
b jets with the ghosts behind each; the IFN pass over the event with the
neutralisation distance of every candidate; and the flavoured merges of
flavour-kT. `-vv` adds every one of the 930 steps of the ak4 clustering.
The `-v` log for seed 1 is committed as `standalone_genjets.log`, and every
number quoted in the theory section of the chapter is taken from it.

### The three photographers on nine particles

`antiKt.py`, `caKt.py` and `hotVR.py` run one algorithm each on the same
nine particles of the record (massless, pT/eta/phi of the rows): the two
cores of the b jet (A = pi+[2181] 14.3 GeV, B = pi-[2239] 5.4,
C = K-[2211] 10.6, D = pi-[2241] 4.5), three soft particles at its border
(E = pi+[1217] 0.7, F = gamma[1538] 1.9, G = gamma[2183] 1.0), the leading
particle of the gluon jet [4] (H = Lambda0[1546] 8.3 GeV) and of jet [5]
(I = pi+[1592] 24.6 GeV). Every step prints the smallest distances and the
decision; `--no-show -o file.pdf` saves the picture (`make figures`).

```bash
cd example/GenJet/Standalone
python3 antiKt.py                       # anti-kT, R = 0.4: every pair and beam distance, every merge, the picture
python3 caKt.py                         # the same nine particles under Cambridge/Aachen
python3 hotVR.py                        # ... and under HOTVR (variable R, mass jump, grooming)
python3 antiKt.py --no-show -o antiKt.pdf   # the picture as a file (make figures); --radius 0.8 to play
python3 hotVR.py --rho 40 --mu 10 --ptsub 5 # the HOTVR parameters are options
```

The three logs, which the chapter's theory reads step by step:

```text
anti-kT (p = -1, R = 0.4)
step 1: d(C, D) = 2.085e-06 (dR = 0.006) wins: merge -> CD with pT = 15.02 GeV
step 2: d(A, B) = 1.935e-05 (dR = 0.025) wins: merge -> AB with pT = 19.68 GeV
step 3: d((C+D), (A+B)) = 7.557e-05 (dR = 0.068) wins: merge -> ABCD with pT = 34.69 GeV
step 4: d(G, ((C+D)+(A+B))) = 7.218e-04 (dR = 0.373) wins: merge -> ABCDG with pT = 35.65 GeV
step 5: d(F, (G+((C+D)+(A+B)))) = 7.166e-04 (dR = 0.382) wins: merge -> ABCDFG with pT = 37.50 GeV
step 6: d((F+(G+((C+D)+(A+B)))), beam) = 7.111e-04 wins: jet ABCDFG with pT = 37.50 GeV
step 7: d(I, beam) = 1.656e-03 wins: jet I with pT = 24.57 GeV
step 8: d(H, beam) = 1.458e-02 wins: jet H with pT = 8.28 GeV
step 9: d(E, beam) = 2.163e+00 wins: jet E with pT = 0.68 GeV
```

```text
Cambridge/Aachen (p = 0, R = 0.4)
step 1: d(C, D) = 0.000 (dR = 0.006) wins: merge -> CD with pT = 15.02 GeV
step 2: d(A, B) = 0.004 (dR = 0.025) wins: merge -> AB with pT = 19.68 GeV
step 3: d(F, G) = 0.009 (dR = 0.037) wins: merge -> FG with pT = 2.83 GeV
step 4: d((C+D), (A+B)) = 0.029 (dR = 0.068) wins: merge -> ABCD with pT = 34.69 GeV
step 5: d(E, (F+G)) = 0.324 (dR = 0.228) wins: merge -> EFG with pT = 3.49 GeV
step 6: d(((C+D)+(A+B)), (E+(F+G))) = 0.944 (dR = 0.389) wins: merge -> ABCDEFG with pT = 38.18 GeV
step 7: d(H, beam) = 1 wins: jet H with pT = 8.28 GeV
step 8: d(I, beam) = 1 wins: jet I with pT = 24.57 GeV
step 9: d((((C+D)+(A+B))+(E+(F+G))), beam) = 1 wins: jet ABCDEFG with pT = 38.18 GeV
```

```text
HOTVR (rho = 600 GeV, R_eff in [0.1, 1.5], mu = 30 GeV, theta = 0.7, pT_sub = 30 GeV)
[01] merge_below_mu  d= 0.0000  Pair C + D (dR=0.006): m_ij=0.0 <= mu=30.0; merge ...
[02] merge_below_mu  d= 0.0006  Pair A + B (dR=0.025): m_ij=0.2 <= mu=30.0; merge ...
[03] merge_below_mu  d= 0.0014  Pair F + G (dR=0.037): m_ij=0.0 <= mu=30.0; merge ...
[04] merge_below_mu  d= 0.0047  Pair (C+D) + (A+B) (dR=0.068): m_ij=1.2 <= mu=30.0; merge ...
[05] merge_below_mu  d= 0.0519  Pair E + (F+G) (dR=0.228): m_ij=0.3 <= mu=30.0; merge ...
[06] merge_below_mu  d= 0.1511  Pair ((C+D)+(A+B)) + (E+(F+G)) (dR=0.389): m_ij=4.6 <= mu=30.0; merge ...
[07] merge_below_mu  d= 1.0954  Pair H + (...) (dR=1.047): m_ij=20.1 <= mu=30.0; merge ...
[08] discard_soft    d= 2.1285  Pair I + (...) (dR=1.459): mass jump passes (m_ij=52.2 > mu, theta*m_ij=36.5 > 20.1),
                                but I has pT <= pT,sub=30.0; discard the soft branch.
[09] finalize_jet    d= 2.2500  d_iB = R_eff^2 smallest for (...): finalize jet (pT=46.3, m=20.1, R_eff=1.500).
```

Final jets: anti-kT ABCDFG 37.50 GeV + I + H (+ E alone, below 3 GeV);
Cambridge/Aachen ABCDEFG 38.18 GeV + H + I; HOTVR one jet ABCDEFGH of
46.3 GeV with R_eff = 1.5 and I thrown away.

## 2. Reading the record

### The two new listings

Nothing happens to the particles at this stop, so the record keeps every
row of chapter 6, the listing with its sums and the vertex table, and adds
two tables after them. The first is the jet listing, one row per jet and
collection: the collection, the index of the jet in pT order, its pT,
rapidity, pseudorapidity, azimuth, mass and energy, the number of
constituents and of charged ones, and its five labels (`hadronFlavour`,
`partonFlavour`, the physics flavour, the dR < 0.3 flavour and the net b
flavour of its heavy hadrons); then, as text, the heavy hadrons and the
hardest partons whose ghosts it caught, with their rows in the listing.
The three b-flavoured ak4 jets and the two leading ak8 jets:

```text
    coll  jet         pT        y      eta      phi        m          E    n  nch  had  part  phys   dR3  net  heavy hadrons; partons (rows)
     ak4    0     99.830   -1.429   -1.439   -2.712   14.826    222.758   31   17    5     5     0     5    1  Bbar0[1527] D0[2179] D_s-[2177]; b[1006] g[1008] g[1007]
     ak4    1     94.093   -3.428   -3.446    0.594   17.825   1477.443   18    8    5    -5     0    -5    1  Bbar0[1590] Xi_c0[2001]; bbar[1064] g[1063] g[1062] g[1171]
     ak4    2     60.451   -3.376   -3.394    1.431   11.786    901.564   29   14    5    -5     0    -5    0  Bbar0[1584] Lambda_bbar0[1005] D+[2189] Dbar0[1965]; bbar[981] b[1061] g[977]
     ak8    0    147.688   -3.410   -3.525    0.917   75.397   2511.446   57   28    5    -5    -5    21    1  Lambda_bbar0[1005] Bbar0[1584] Bbar0[1590] Dbar0[1965] D+[2189] Xi_c0[2001]; bbar[1064] ...
     ak8    1    105.786   -1.426   -1.455   -2.711   27.434    240.565   47   25    5     5     5     5    1  Bbar0[1527] D0[2179] D_s-[2177]; b[1006] g[1008] g[1007] g[968] g[1169]
```

The second is the constituent listing, one row per final particle with the
index of the jet that holds it in each collection, -1 where it ended in no
jet or in one below the threshold; the K- of row 2211, the first particle
of our jet's core, is jet 0 of ak4, 0 of ak4nonu and 1 of ak8:

```text
    no   ak4  ak4nonu   ak8
  2211     0        0     1
```

A jet row begins with the name of its collection and a constituent row has
four fields, so the readers of the previous chapters skip both; the
function `read_jet_tables` of the script parses them, and
`plot_journey_GenJet.py` draws them.

### The learning-mode log, section by section

The log has the sections `set-up`, `the b quark's ak4 jet, merge by merge`,
`the flavour of the two b jets`, `IFN over the whole event`, `flavour-kT
over the whole event` and `summary`. The ghosts:

```text
  ghosts (momentum x 1e-18): b hadrons Lambda_bbar0 [1005] pT 35.52 eta -3.41 phi 1.49, Bbar0 [1527] pT 78.26 eta -1.38 phi -2.73, Bbar0 [1584] pT 24.93 eta -3.36 phi 1.34, Bbar0 [1590] pT 81.97 eta -3.43 phi 0.58
          c hadrons D- [1337] 1.28, D_s+ [1708] 0.92, D+ [1727] 0.11, Dbar0 [1797] 0.83, Dbar0 [1965] 14.95, Xi_c0 [2001] 71.43, D_s- [2177] 27.28, D0 [2179] 26.98, D+ [2189] 10.07
          163 partons with status 71-79 (the strings' partons), 2 hard-process partons: bbar [5] pT 113.22, b [6] pT 113.22; 3 leptons from heavy hadrons
```

The first merges of the b quark's jet (36 of the 930 steps of the event;
the chapter's theory reads the first, the ghosts' and the last):

```text
   step         merge        d_ij     dR     pT_i     pT_j     pT_k  what joined
     12   715 +   742   1.736e-06  0.006   10.567    4.451   15.018  pi-[2241] 4.45 GeV joins K-[2211] 10.57 GeV
     16   697 +   941    2.73e-06  0.010    3.059   15.018   18.077  gamma[2178] 3.06 GeV joins pseudojet 941
     20   760 +   945   5.953e-06  0.018    0.000   18.077   18.077  ghost chad D0[2179] joins pseudojet 945
     23   698 +   949   7.429e-06  0.020    4.084   18.077   22.160  gamma[2180] 4.08 GeV joins pseudojet 949
     24   750 +   952   1.238e-05  0.031    0.000   22.160   22.160  ghost bhad Bbar0[1527] joins pseudojet 952
     27   781 +   953   1.685e-05  0.036    0.000   22.160   22.160  ghost parton b[1006] joins pseudojet 953
     ...
     33   957 +   961   1.843e-05  0.055   23.273   32.093   55.354  pseudojet 957 joins pseudojet 961
     ...
     56   783 +   984   4.701e-05  0.253    0.000   92.387   92.387  ghost parton g[1008] joins pseudojet 984
```

The labels of the two b jets:

```text
  b jet [0]: ghosts clustered: b hadrons Bbar0; c hadrons D0 D_s-; partons b g g
    hadronFlavour = 5 (a b-hadron ghost is in the jet)
    partonFlavour = 5 (the hardest b parton among the status-71 ghosts, else c, else the hardest)
    physics flavour = 0 (the status-23 partons as ghosts); dR < 0.3 matching: 5
    net flavour of the b/c hadrons in the jet, as recorded: +1 b
    with the heavy hadrons undecayed: net +1 b, IFN +1 b; flavoured constituents Bbar0[1527]
  bbar jet [1]: ghosts clustered: b hadrons Bbar0; c hadrons Xi_c0; partons bbar g g g
    hadronFlavour = 5, partonFlavour = -5, physics flavour = 0, dR < 0.3 matching: -5
    net flavour of the b/c hadrons in the jet, as recorded: +1 b, +1 c
    with the heavy hadrons undecayed: net +1 b, IFN +1 b; flavoured constituents Bbar0[1590]
==== IFN over the whole event (heavy hadrons kept, anti-kT R = 0.4) ====
  u_ik = max(pT_i, pT_k)^2 min(pT_i, pT_k)^0 Omega_ik^2, Omega^2 = 2[(cosh(omega dy) - 1)/omega^2 - (cos dphi - 1)], omega = 1
  no flavoured pseudojet found a partner with u_ik < u_ij: no neutralisation in this event
```

## 3. Drawing the event and rendering the film

### The (eta, phi) displays

`plot_journey_GenJet.py` draws one collection of the record in the
(eta, phi) plane in the layout of the CMSSW `genJetAnalyzer`, reusing the
reader and the frame of the previous stages. Every jet above the threshold
is a circle of radius R around its axis, drawn as a star; the leading
eight are in colour with their constituents in the same colour, the b jet
red and the bbar jet blue, the other jets and their constituents grey, and
the particles that ended in no jet hollow; the last copy of each b quark
is a black star. The header gives the quark-to-jet lines with dR and the
response, the system of the two b jets, and the heavy hadrons behind each
`hadronFlavour` label; the margin lists the leading jets with their two
CMSSW labels.

```bash
python3 example/GenJet/plot_journey_GenJet.py                        # ak4GenJets -> example/GenJet/journey_GenJet.pdf
python3 example/GenJet/plot_journey_GenJet.py --collection ak4nonu   # -> journey_GenJet_nonu.pdf
python3 example/GenJet/plot_journey_GenJet.py --collection ak8 --png # -> journey_GenJet_ak8.pdf
```

### The theory plots

`plot_theory_GenJet.py` clusters the record again with the classes of the
script, scans the radius and the soft pair, and reads the `--repeat`
statistics and the FastJet validation files when they are present (about
five minutes); it writes `theory_genjets.pdf`, one page per theory
subsection:

| Page | Chapter figure | Content |
|---|---|---|
| 1 | the choices | the pT within R of the quark for the cone and the jet, with and without neutrinos; the 36 merges of our jet with the dR of each; the number of jets above a threshold |
| 2 | the ghosts | the ghosts over the jets; the six labels of every jet above 5 GeV; the B hadron against the axis over 400 decays |
| 3 | infrared-safe flavour | the soft-pair scan of the labels of jets [8] and [9]; Omega^2 against dy; flavour-kT against anti-kT |
| 4 | our jet at this stage | the family through the stages and its three photographs; what the ak4 jet is made of; the response over 400 decays and the PYTHIA chain |

`plot_algorithms_GenJet.py` runs the three rules on the 749 particles
(anti-kT and Cambridge/Aachen with the `Clustering` class, HOTVR with a
NumPy copy of `hotVR.py`'s decisions, checked against it on the nine) and
draws them side by side in `algorithms_genjets.pdf`: anti-kT 42 jets (368
particles), Cambridge/Aachen 39 jets (356; the b jet identical, the bbar
jet 94.45 GeV / 20 instead of 94.09 / 18, jet [4] 48.07 / 27 instead of
46.40 / 25), HOTVR 16 jets holding 525 particles with 196 groomed away
(the b family a 170.6 GeV jet of 105 particles and mass 160 GeV with three
subjets, anti-kT's jets [0], [4] and [5]; the bbar family a 151.9 GeV jet
with two).

```bash
python3 example/GenJet/Standalone/plot_theory_GenJet.py        # -> theory_genjets.pdf
python3 example/GenJet/Standalone/plot_algorithms_GenJet.py    # -> algorithms_genjets.pdf, the three panels
python3 example/GenJet/Standalone/plot_algorithms_GenJet.py --rho 40 --mu 10 --ptsub 5   # HOTVR scaled to our scale
make figures                                                   # the five PDFs, from the book directory
```

### The film

`animate_standalone_genjets.py` imports the standalone script and clusters
the seed-1 record in six narrated scenes of about five minutes in all
(eighty seconds without the voice; `--scene photograph,gather,choices,ghosts,whisper,ensemble`
or `1..6`). The voice picks up where the chapter-6 film stopped, the
grandchildren are born, and ends at the next stop, the crowd on the road of
chapter 8.

1. **photograph**: the 749 particles appear in the (eta, phi) plane,
   hardest first, then the 42 circles of the anti-kT jets settle over
   them, the leading eight in colour.
2. **gather**: zoom on the b jet, its 36 merges in order, the K- and pi-
   of the Bbar0 at dR = 0.006 first, the ghosts slipping in, the soft
   periphery last, the running pT and the ladder of distances, and the
   circle drawn when d_iB wins.
3. **choices**: the radius grows from 0.05 to 0.9 around the same axis;
   decay seed 4 of the same hadrons shows the jet without neutrinos at
   90.2 GeV instead of 99.6; the threshold counts 42, 11 and 6 jets.
4. **ghosts**: the Bbar0 shrinks to a ghost and still lands in jet [0];
   the status-23 quark sits outside; the gluon's pair and the oscillated
   B0 at the other end of the event.
5. **whisper**: a B0 Bbar0 pair of half a GeV between two light jets; the
   ghost and net labels flip, the IFN distance cancels the pair, and the
   scan in the pair's pT shows which definitions survive.
6. **ensemble**: the same primary hadrons decayed and clustered four
   hundred times; the response histograms of the three collections.

```bash
cd example/GenJet/Standalone
python3 animate_standalone_genjets.py            # the narrated film, 1080p, ~1 h with --jobs 4
python3 animate_standalone_genjets.py --fast     # low-resolution preview
python3 animate_standalone_genjets.py --print-script   # the narration, one line per phase
python3 animate_standalone_genjets.py --scene gather,ghosts -o g.gif   # silent
python3 animate_standalone_genjets.py --stills animate_standalone_genjets_stills.pdf --no-video
```

Rendering needs matplotlib in addition to NumPy, `ffmpeg` for an `.mp4`,
and `edge-tts` for the voice (the shared `example/soundtrack.py`
synthesises the lines, caches the clips in `build/narration/GenJet/` and
writes the score and the subtitles). The last command writes the key
frames (one page per scene) that the chapter prints; `make figures` runs
it. `--no-narration` gives the silent cut; the ensemble of scene 6 is read
from the statistics file when it exists, and `--events` sets how many
decays it shows.

## 4. Validation from official packages

Two official sources, and this time the first comparison is not
statistical: FastJet, driven from a script on the same particles, must
reproduce every jet exactly; PYTHIA's own clustering must do the same; and
the CMS generator job of chapter 1 drew the three collections of this
chapter.

### 4.1 The same particles with FastJet and SlowJet

`validate_genjets_with_fastjet.py` has two modes. In mode `record` (the
default) it builds the 749 particles of `event_Decay.lhe` and the same 181
ghosts as FastJet `PseudoJet`s, clusters them with
`JetDefinition(antikt_algorithm, R)` and the E scheme for the three
collections, and compares jet by jet with the standalone classes: the
momenta, the constituent lists and the three labels. It then fills a
PYTHIA event record with the same particles and runs `SlowJet`, PYTHIA's
own anti-kT with the `fjcore` of FastJet inside, with `select = 2` for the
collection without neutrinos. In mode `full` the hard event of chapter 1
is written to a Les Houches file N times, PYTHIA runs the whole chain with
the CP5 settings and decays on, and FastJet clusters every event with the
ghosts, taking the status-71 copy of each quark as the axis, the choice of
the CMSSW analyser. Both modes record what the chapter's table lists, in
the JSON layout of the standalone statistics. The environment is the one
of `example/ME/Standalone/`, with FastJet 3.5.2 added by `pip install fastjet`.

```bash
cd example/GenJet/Standalone
PY=../../ME/Standalone/.venv/bin/python
$PY validate_genjets_with_fastjet.py                           # mode record; also --algorithm kt, ca
$PY validate_genjets_with_fastjet.py --mode full -n 2000 --standalone-stats standalone_genjet_stats.stats.json
```

The first run takes a minute, the full chain four. The record mode prints,
for each collection,

```text
ak4GenJets (antikt, R = 0.4, pT > 3):
  jets: standalone 42, FastJet 42, SlowJet 42
  FastJet vs standalone, jet by jet: max |dpT| = 0.00e+00 GeV, max |dm| = 0.00e+00 GeV, identical
    constituent lists 42/42, identical (hadron, parton, physics) flavours 42/42
  SlowJet vs standalone: max |dpT| = 2.13e-14 GeV, identical constituent lists 42/42
```

and the same for ak4GenJetsNoNu, for ak8GenJets (30, 30, 30) and, with
`--algorithm`, for kT (36 jets at R = 0.8) and Cambridge/Aachen (35). The
FastJet jets of our record with their constituent rows are in
`validate_genjets_with_fastjet_record_jets.txt` (and `..._kt_jets.txt`,
`..._ca_jets.txt`), the logs in `validate_genjets_with_fastjet_{record,full}.log`,
the results in `validate_genjets_with_fastjet_{record,record_kt,record_ca,full}.json`,
and the first full-chain event in `validate_genjets_with_fastjet_full_event1.txt`.

| Option | Meaning |
|---|---|
| `--mode record` / `full` | cluster the 749 particles of chapter 6 / run the full CP5 chain from the hard event |
| `-n`, `--seed` | events and seed |
| `--algorithm`, `--ptmin` | `antikt`, `kt` or `ca`; the threshold | 
| `--impact`, `--window` | full mode: keep events with abs(b/`<b>` - impact) < window |
| `--standalone-stats` | full mode: the `.stats.json` of the standalone `--repeat`, printed side by side |
| `--output`, `--listing` | the JSON file and the first event's listing |

### 4.2 CMSSW: the GenJet step

Nothing new has to be generated. The `GEN` step of chapter 1 runs the
`genJetParticles` and `recoGenJets` sequences after the hadron level, so
that every GEN file carries `ak4GenJets`, `ak4GenJetsNoNu` and
`ak8GenJets` already, made by `FastjetJetProducer` with FastJet's
anti-kT, the E scheme and `jetPtMin = 3`; the flavour producers run
later, in the `PAT` step, and are not in the GEN output. JitJetSW's
`genJetAnalyzer` drew the three collections as the sixth, seventh and
eighth figures of the job. From the setup of the
[chapter 2 companion](chapter_02_PS.md#42-cmssw-the-shower-inside-the-gen-step):

```bash
cd CMSSW_15_0_5/src/JitJetSW/Journey/test
./runCMSDrivers_mc2024_BBbar_GEN.sh          # GEN step + genJetAnalyzer, one page per collection
```

produces `genJetAnalyzer_ME_PS_MPI_Had_Decay_GenJet_out.pdf`, whose pages
6 to 8 are the chapter's CMSSW figure; the copy in `CMSSW/` is the one
used.

**Reading the pages.** Page 6 is ak4GenJets: 27 jets, 239 of the 634
stable particles clustered, the b quark of status 71 at pT = 91.74 GeV,
(eta, phi) = (-1.57, -2.08), matched to jet [0] of 117.79 GeV with 24
constituents, dR = 0.019 from the quark and a response of 1.284; the bbar
at 116.36 GeV and eta = -3.75 matched to jet [1] of 115.94, 18
constituents, response 0.996. Page 7 is the same event without the
neutrinos: the bbar's jet is unchanged and the b's has become jet [1] of
79.82 GeV with 22 constituents and a response of 0.870, the 37.5 GeV of
the electron antineutrino of chapter 6 gone, and the two leading jets have
swapped places. Page 8 is ak8GenJets: 28 jets, 488 particles clustered,
the b's jet at 141.10 GeV with 55 constituents and the bbar's at 121.01
with 32.

## 5. The full tables

### 5.1 Generator-level jets obtained four ways

"Same particles" means the 749 final particles of `event_Decay.lhe` with
the same ghosts, clustered by FastJet 3.5.2 and by the standalone script;
"400 decays" means the primary hadrons of chapter 5 decayed four hundred
times with the script of chapter 6 and clustered each time; "full chain"
means PYTHIA from the hard event of chapter 1 with the CP5 settings, 2000
events, clustered with FastJet and the same ghosts, the status-71 copy of
the quark as the axis. The spread of a single event follows +-.
"Response" is pT(jet)/pT(b) with the last copy of the quark before the
hadronisation, which in the full chain fluctuates itself. Commands:
`validate_genjets_with_fastjet.py` (column 1), `standalone_genjets.py`
(column 2), `standalone_genjets.py --repeat 400 --jobs 4` (column 3),
`validate_genjets_with_fastjet.py --mode full -n 2000` (column 4), pages
6 to 8 of the CMSSW analyser's output (column 5).

| quantity | FastJet, same particles | standalone, same particles | standalone, 400 decays | PYTHIA + FastJet, full chain | CMSSW, 1 event |
|---|---|---|---|---|---|
| ak4GenJets: jets above 3 / 10 / 30 GeV | 42 / 11 / 6 | 42 / 11 / 6 | 40.0 +- 2.2 / 10.2 / 6.0 | 18.2 +- 11.5 / 3.9 / 2.4 | 27 / -- / -- |
| particles clustered, of | 368 of 749 | 368 of 749 | 357 +- 16 | 152 +- 88 | 239 of 634 |
| ak8GenJets: jets, particles clustered | 30, 600 | 30, 600 | 30.4, 608 | 20.2, 321 | 28, 488 |
| b quark (last copy): pT [GeV] | 78.29 | 78.29 | 78.29 | 95.5 +- 30.0 | 91.74 |
| b jet, ak4GenJets: pT [GeV], n | 99.830, 31 | 99.830, 31 | 99.63 +- 0.49, 25.7 | 109.9 +- 27.6, 21.3 | 117.79, 24 |
| response, dR to the quark | 1.275, 0.062 | 1.275, 0.062 | 1.273 +- 0.006, 0.060 | 1.22 +- 0.51, 0.028 | 1.284, 0.019 |
| the cone of the same R [GeV] | 99.43 | 99.43 | 98.79 +- 0.68 | 109.7 +- 26.5 | -- |
| neutrino pT inside [GeV] | 0 | 0 | 7.7 (none in 54%) | 7.9 (none in 58%) | 37.5 |
| b jet, ak4GenJetsNoNu: pT [GeV] | 99.830 | 99.830 | 91.93 +- 12.12 | 102.1 +- 28.9 | 79.82 |
| b jet, ak8GenJets: pT [GeV], n | 105.786, 47 | 105.786, 47 | 104.61 +- 0.80, 41.3 | 120.0 +- 24.6, 34.3 | 141.10, 55 |
| b jet: hadronFlavour = 5, abs(partonFlavour) = 5 [%] | 5, 5 | 5, 5 | 100, 100 | 99.2, 99.3 | -- |
| b jet: physics flavour, dR < 0.3 flavour | 0, +5 | 0, +5 | -- | -- | -- |
| b jet: net b flavour +1 / -1 [%] | +1 | +1 | 83.2 / 16.8 | -- | -- |
| b jet: B hadron to the axis, its products outside | 0.06, 0 | 0.06, 0 | 0.062, 0.20 | 0.025, 0.31 | -- |
| bbar jet, ak4GenJets: pT [GeV], response | 94.093, 1.298 | 94.093, 1.298 | 93.86 +- 0.83, 1.295 | 106.5 +- 20.9, 1.23 | 115.94, 0.996 |
| bbar jet, ak8GenJets: pT [GeV], dR | 147.688, 0.361 | 147.688, 0.361 | 113.4 +- 20.4, 0.14 | 115.3 +- 17.0, 0.05 | 121.01, 0.022 |
| bbar jet: net b flavour -1 / +1 [%] | +1 (oscillated) | +1 (oscillated) | 79.2 / 20.8 | -- | -- |
| ak4 jets with two b hadrons per event | 1 | 1 | 1.00 | 0.009 | -- |
| IFN neutralisations per event | 0 | 0 | 0.66 | -- | -- |

### 5.2 Kinematics of the two b quarks, their jets and their cones

From `event_GenJet.lhe` as printed by `standalone_genjets.py`, with the
cone of the same radius around the quark for comparison. "Response" is
pT(jet)/pT(b) with the last copy of the quark; dR is the distance in
(eta, phi) from the quark to the jet axis; the labels are
`hadronFlavour`, `partonFlavour`, the physics and the dR < 0.3
definitions, and the net b flavour.

| | | pT [GeV] | m [GeV] | n | dR | response | labels |
|---|---|---:|---:|---:|---:|---:|---|
| b | the quark (row 719, last copy) | 78.29 | 4.80 | 1 | | | |
| | its Bbar0 (row 1527, chapter 6) | 78.26 | 5.28 | 1 | 0.06 | | |
| | the cone dR < 0.4, final (chapter 6) | 99.43 | 14.59 | 31 | | 1.270 | |
| | ak4GenJets jet [0] | 99.83 | 14.83 | 31 (17 charged) | 0.062 | 1.275 | 5, +5, 0, +5, +1 |
| | charged hadrons / photons / neutral hadrons | 68.1 / 22.8 / 9.0 | | 17 / 13 / 1 | | | |
| | the Bbar0's descendants / neutrinos | 78.26 / 0 | | 15 / 0 | | | |
| | ak4GenJetsNoNu jet [0] | 99.83 | 14.83 | 31 | 0.062 | 1.275 | the same |
| | the cone dR < 0.8 (chapter 6) | 108.73 | 42.14 | 51 | | 1.389 | |
| | ak8GenJets jet [1] | 105.79 | 27.43 | 47 (25 charged) | 0.078 | 1.351 | 5, +5, +5, +5, +1 |
| bbar | the quark (row 708, last copy) | 72.50 | 4.80 | 1 | | | |
| | its Bbar0 (row 1590, oscillated) | 81.97 | 5.28 | 1 | 0.02 | | |
| | the cone dR < 0.4 | 93.02 | | 18 | | 1.283 | |
| | ak4GenJets jet [1] | 94.09 | 17.83 | 18 (8 charged) | 0.035 | 1.298 | 5, -5, 0, -5, +1 |
| | ak4GenJets jet [2], the gluon's pair | 60.45 | 11.79 | 29 (14 charged) | 0.87 | | 5, -5, 0, -5, 0 |
| | ak8GenJets jet [0], both of the above | 147.69 | 75.40 | 57 (28 charged) | 0.361 | 2.037 | 5, -5, -5, 21, +1 |
| b jet + bbar jet (ak4GenJets) | abs(sum pT), Sigma abs(pT), dphi, m | 16.9, 193.9 | 302 | | | 2.98 | |

### 5.3 The three photographers on the whole event

From `plot_algorithms_GenJet.py` on the 749 particles:

| algorithm | jets above 3 GeV | particles clustered | the b jet | the bbar jet | jet [4] |
|---|---:|---:|---|---|---|
| anti-kT, R = 0.4 | 42 | 368 | 99.83 GeV, 31 | 94.09 GeV, 18 | 46.40 GeV, 25 |
| Cambridge/Aachen, R = 0.4 | 39 | 356 | 99.83 GeV, 31 (identical) | 94.45 GeV, 20 | 48.07 GeV, 27 |
| HOTVR (paper's parameters) | 16 | 525 (196 groomed away) | inside a 170.59 GeV jet of 105 particles, m = 160 GeV, subjets = anti-kT [0], [4], [5] | inside a 151.86 GeV jet with [1] and [2] as subjets | a subjet |

### 5.4 The CMSSW pages

| page | collection | jets | particles clustered | the b quark's jet | the bbar quark's jet |
|---|---|---:|---:|---|---|
| 6 | ak4GenJets | 27 | 239 of 634 | jet [0], 117.79 GeV, 24 constituents, dR 0.019, response 1.284 | jet [1], 115.94 GeV, 18, response 0.996 |
| 7 | ak4GenJetsNoNu | 27 | -- | jet [1], 79.82 GeV, 22, response 0.870 (the 37.5 GeV neutrino gone) | unchanged |
| 8 | ak8GenJets | 28 | 488 | 141.10 GeV, 55 constituents, response 1.538 | 121.01 GeV, 32, dR 0.022 |

### 5.5 The energy ledger

| Stage | pT [GeV] | m [GeV] | n_const | delta pT [GeV] |
|---|---:|---:|---:|---:|
| b family after the decays (dR < 0.4, final) | 99.43 | 14.59 | 31 | |
| ak4GenJets jet [0], the photograph | 99.83 | 14.83 | 31 | +0.40 |
| ak4GenJetsNoNu jet [0], the reference | 99.83 | 14.83 | 31 | 0.00 |
| of which charged hadrons | 68.09 | | 17 | |
| of which the Bbar0's descendants | 78.26 | 5.28 | 15 | |
| mean of 400 decays, with neutrinos | 99.63 | 14.11 | 25.7 | |
| mean of 400 decays, without | 91.93 | 13.29 | 25.2 | -7.70 |
| ak8GenJets jet [1], the same family at R = 0.8 | 105.79 | 27.43 | 47 | +5.96 |
