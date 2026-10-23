# Chapter 5 companion: The Children Get Married (Had)

This is the technical companion of chapter 5, *The Children Get Married:
Hadronisation and b-Fragmentation*. The chapter gives the physics; this
page says how to run the scripts, how to read what they write, how each
number of the official packages was obtained, and lists every table the
chapter quotes with the command that produced it. All files are in
[`example/Had/`](../example/Had/) of this repository unless stated
otherwise; the CMSSW analysers are in
[JitJetSW](https://github.com/ravindkv/JitJetSW).

| | |
|---|---|
| Stage | hadronisation with the Lund string model, on the parton-level record of chapter 4 |
| Standalone script | [`example/Had/Standalone/standalone_hadronisation.py`](../example/Had/Standalone/standalone_hadronisation.py) |
| Input | [`example/MPI/event_MPI.lhe`](../example/MPI/event_MPI.lhe), the record of chapter 4 (211 final partons with colour tags) |
| Event record | [`example/Had/event_Had.lhe`](../example/Had/event_Had.lhe) (seed 1, 1491 rows, 365 primary hadrons) |
| Learning-mode log | [`example/Had/Standalone/standalone_hadronisation.log`](../example/Had/Standalone/standalone_hadronisation.log) (`-v`, 2187 lines) |
| Validation with PYTHIA | [`example/Had/Standalone/validate_hadronisation_with_pythia.py`](../example/Had/Standalone/validate_hadronisation_with_pythia.py) |
| CMSSW figure | [`example/Had/CMSSW/genPartAnalyzer_ME_PS_MPI_Had_out.pdf`](../example/Had/CMSSW/genPartAnalyzer_ME_PS_MPI_Had_out.pdf) (page 4); the log is that of chapter 2 |
| Film | scene 5 of the [JitJet playlist](https://www.youtube.com/playlist?list=PLbSW20iyZFd4) |

Contents: [1. Running the standalone script](#1-running-the-standalone-script) ·
[2. Reading the record](#2-reading-the-record) ·
[3. Drawing the event and rendering the film](#3-drawing-the-event-and-rendering-the-film) ·
[4. Validation from official packages](#4-validation-from-official-packages) ·
[5. The full tables](#5-the-full-tables)

## 1. Running the standalone script

`standalone_hadronisation.py` takes the parton-level record of chapter 4
and turns it into primary hadrons with the Lund string model, written from
scratch in Python following PYTHIA 8's `ColourTracing`,
`StringFragmentation`, `StringFlav`, `StringZ`, `StringPT` and
`MiniStringFragmentation`: fifteen hundred lines that need NumPy and
nothing else. It imports the shower script of chapter 2 for the event record
and the Lorentz frames; its hadron table (the names, masses and charges of
the 78 hadrons the string can make and the constituent masses of the quarks
and diquarks) is embedded between `BEGIN/END GENERATED TABLES` by
`dump_hadron_tables.py`, which reads PYTHIA's `ParticleData` once in the
`example/ME/Standalone/.venv` environment.

A map of the file, with the steps of the chapter's flow chart:

| Function or class | What it does |
|---|---|
| `colour_singlets` | follows every colour tag from a quark or antidiquark through the gluons to an antiquark or diquark; joins partons closer than `mJoin` = 0.3 GeV; sorts strings from ministrings |
| `StringFlav`, `StringPT` | what comes out of the vacuum at a break: the flavour, diquark and spin weights, the Gaussian kick |
| `StringZ` | the Lund symmetric, Lund-Bowler and Peterson fragmentation functions |
| `StringSystem`, `place_vertex` | the regions of the world sheet and the proper-time rule for a vertex that crosses a kink |
| `fragment`, `final_two`, `MiniString` | the iteration from both ends, the stop at W_min, the last two hadrons, the ministrings |
| `summarise` | the two B hadrons with their x_E, x_pT, x_+ and the hadron-level cones around the two b quarks |

```bash
cd example/Had/Standalone
python3 standalone_hadronisation.py                      # seed 1 -> ../event_Had.lhe (the book's record)
python3 standalone_hadronisation.py -v                   # learning mode: the b quark's string hadron by hadron
python3 standalone_hadronisation.py -vv                  # ... every string in full
python3 standalone_hadronisation.py --peterson           # Peterson instead of Lund-Bowler for b and c
python3 standalone_hadronisation.py --rfactb 0           # plain Lund for the b quark
python3 standalone_hadronisation.py --repeat 400 --quiet --history standalone_had_stats.json
```

The first command hadronises the record in half a second and prints

```text
input example/MPI/event_MPI.lhe: 1491 rows after hadronisation; seed = 1
  29 strings and 4 ministrings (0 closed gluon loops, 48 partons joined), total string mass 3071 GeV
    -> 365 primary hadrons, 184 charged, 52 baryons, 94 with strangeness; 179 (95 charged) at |eta| < 2.5
  most common: pi0 45, pi+ 43, pi- 37, rho0 24, rho- 23, omega 20, rho+ 18, eta 17, Kbar0 15, K+ 13, K0 12, nbar0 10
  bbar (row 708, pT = 72.50, E = 1106.73 GeV) -> B*0 (row 1069): pT = 82.83, E = 1276.92 GeV, dR = 0.012;
    x_E = 1.154, x_pT = 1.142, x_+ = 1.154 (rank 1 from the - end of a 40 GeV string with 4 partons and 5 hadrons)
     cone R = 0.4 around the quark: 4 partons, pT 93.87, m 7.75 -> 4 hadrons (1 charged), pT 91.34, m 7.58 GeV; the bbar hadron carries 91% of the cone pT; R = 0.8: pT 116.07 -> 116.92, m 41.61 -> 47.92, 11 hadrons
  b (row 719, pT = 78.29, E = 165.42 GeV) -> B*bar0 (row 1012): pT = 78.56, E = 166.04 GeV, dR = 0.005;
    x_E = 1.004, x_pT = 1.004, x_+ = 1.004 (z drawn 0.984, rank 1 from the + end of a 69 GeV string with 6 partons and 16 hadrons)
     cone R = 0.4 around the quark: 3 partons, pT 100.67, m 13.52 -> 6 hadrons (2 charged), pT 99.23, m 16.08 GeV; the b hadron carries 79% of the cone pT; R = 0.8: pT 104.09 -> 105.98, m 20.17 -> 35.87, 14 hadrons
wrote example/Had/event_Had.lhe (1491 rows, 365 primary hadrons)
```

The output with `--seed 1` is byte-for-byte the committed record. With
`--history` the strings and the summary go to a JSON file as well; with
`--repeat N` the script hadronises the same partons N times with seeds 1..N
and writes the per-event quantities of section 5.1 to `<history>.stats.json`
(`standalone_had_stats.stats.json`, `..._rfactb0.stats.json` and
`..._peterson.stats.json` are the committed 400-event runs, three minutes
each; `standalone_had_stats.log` is the printout).

| Option | Meaning | Default (Monash, kept by CP5) |
|---|---|---|
| `-i`, `-o` | input record, output record | `../../MPI/event_MPI.lhe`, `../event_Had.lhe` |
| `--seed` | the random seed | 1 |
| `--alund`, `--blund` | `StringZ:aLund`, `StringZ:bLund` [GeV^-2] | 0.68, 0.98 |
| `--rfactb`, `--rfactc` | `StringZ:rFactB`, `rFactC` (0: plain Lund for the heavy quark) | 0.855, 1.32 |
| `--peterson` | the Peterson function for c and b (epsilon_c = 0.05, epsilon_b = 0.005) | off |
| `--sigma` | `StringPT:sigma` [GeV], the width of the kick for the pair | 0.335 |
| `--prob-s`, `--prob-qq` | `StringFlav:probStoUD`, `probQQtoQ` | 0.217, 0.081 |
| `--radius` | the cone of the family summary | 0.4 |
| `--repeat N`, `--quiet`, `--history` | N hadronisations with seeds 1..N, no per-event printout, JSON output | |
| `-v`, `-vv` | the learning modes, below | off |

### The learning modes

`-v` prints, before the event, the parameters of the model with the mean z
of the fragmentation function for a pion, a rho and a B; then the table of
all 33 colour singlets with their partons, masses and the partons joined;
then, for the string that holds the b quark, every hadron in production
order with the end it came from, the flavour draw and its weights, the
hadron and its mass, the pT kick, the fragmentation function and the z
drawn, the proper time of the vertex, the regions stepped through and the
momentum, then the remaining mass against W_min, the final two and the
orientation draw; one line per hadron for every other string; the
ministrings; the conservation checks and the full record. `-vv` does the
same for every string. The 2187 lines of `-v` for seed 1 are committed as
`standalone_hadronisation.log`, and every number quoted in the theory
section of the chapter is taken from that file.

## 2. Reading the record

### Status codes

The record keeps PYTHIA's conventions and adds the status codes of the
hadron level to those of the [chapter 4 companion](chapter_04_MPI.md#status-codes).
When a colour singlet is collected, every one of its partons is copied with
status `-71`; two partons that are joined because their pair mass is within
`mJoin` = 0.3 GeV of their masses become one row with status `-73` and two
mothers; a string that absorbs the recoil of a collapsing ministring would
be copied with status `-72` (none in our event). The hadrons are positive:
`82` for the two hadrons of a ministring (`81` for a single one), `83` for a
hadron produced from the colour end of a string and `84` from its
anticolour end, with the string's partons as the mother range. Our record
has 1491 rows: the 963 of chapter 4, every final parton of which now has a
negative status because this stage copied it, then 122 rows of status `-71`
and 41 of status `-73` (the 48 joins leave 163 string partons), then 170
hadrons of status `83`, 187 of status `84` and 8 of status `82`, the 365
primary hadrons. The final state of the record is every row with a positive
status; their momentum sum is still that of the two protons, and their
charge sum is +2. Hadrons whose daughters are zero are stable for now;
chapter 6 decays them.

### The rows the chapter reads

The b quark's string, rows 1006 to 1011 (status `-71` and `-73`; row 1009 is
two gluons joined), and its first hadron, the B*bar0 (row 1012, status `83`,
mothers 1006 to 1011):

```text
  1006          5  (b)                -71   719     0  1012  1027   113     0    -71.735    -31.354   -145.640    165.418      4.800
  1007         21  (g)                -71   718     0  1012  1027   127   113     -1.822     -0.281     -4.093      4.489      0.000
  1008         21  (g)                -71   717     0  1012  1027   107   127    -17.773    -10.707    -51.049     55.105      0.000
  1009         21  (g)                -73   713   894  1012  1027   124   107      1.645     -1.675    -26.126     26.232      0.248
  1010         21  (g)                -71   902     0  1012  1027   273   265     -0.980     -2.292    -30.991     31.091      0.000
  1011         -2  (ubar)             -71   733     0  1012  1027     0   273    -14.563    -10.375    -94.693     96.366      0.000
  1012       -513  B*bar0              83  1006  1011     0     0     0     0    -72.156    -31.074   -146.183    166.042      5.325
```

Row 1006 is the mother's string copy at the momentum of row 719; the
B*bar0 has pT = 78.56 GeV against the quark's 78.29. Fifteen more hadrons
of status 83 and 84 follow up to row 1027. The bbar's string copy (row 1064,
mother 708) and its B*0, one of the final two of its string (row 1069,
status `84`, from the anticolour end):

```text
  1064         -5  (bbar)             -71   708     0  1065  1069     0   117     60.914     39.317  -1104.346   1106.734      4.800
  1069        513  B*0                 84  1061  1064     0     0     0     0     69.306     45.360  -1274.223   1276.924      5.325
```

The record ends with the last hadron of the 1036 GeV remnant string and the
sums, the charge of two protons and their momentum to a few MeV:

```text
  1490        113  rho0                84  1471  1472     0     0     0     0     -0.212     -0.357    188.252    188.254      0.775
                                   Charge sum:  2.000           Momentum sum:     -0.001     -0.009     -0.001  13599.995  13599.995
```

### The learning-mode log, section by section

The log has four sections: `setting the scene`, `the colour-singlet strings
of the record` (which holds the per-string blocks), `checks` and `summary`.
The opening lines, the parameters the chapter quotes:

```text
  string tension kappa = 1.0 GeV/fm; Lund a = 0.68, b = 0.98 GeV^-2; a_diquark = a + 0.97; Bowler r_b = 0.855, r_c = 1.32
  pT at a break: Gaussian with sigma = 0.335 GeV for the pair (each component sigma/sqrt2 = 0.237); 1% of the breaks with 2.0 x the width
  flavours: s/u = 0.217 (tunnelling exp(-pi m^2/kappa): m_s = 0.5 GeV would give 0.106), diquark/quark = 0.081, strange diquark x 0.915, spin-1 diquark x 3 x 0.0275
  vector/pseudoscalar: u,d 0.5, s 0.55, c 0.88, b 2.2; eta x 0.6, eta' x 0.12
  stop: W_min = m_q1 + m_q2 + 1.0 + 2.0 m_qlast, smeared +-20%; ministring below m_q1 + m_q2 + 1.0 GeV; partons joined below 0.3 GeV
  f(z) for u -> pi (mT = m_pi)   : <z> = 0.328, maximum at z = 0.118
  f(z) for u -> rho              : <z> = 0.529, maximum at z = 0.449
  f(z) for b -> B (Lund-Bowler)  : <z> = 0.888, maximum at z = 0.931
```

The string table, the heaviest strings first (the b quark's string is
string 1, W = 69.34 GeV, b g g g g ubar; the bbar's is string 3, b g g bbar
with a secondary b at the other end):

```text
  33 systems: 29 strings and 4 ministrings (W < m_q1 + m_q2 + mStringMin = ... + 1.0 GeV); total string mass 3071.1 GeV
    #  partons   W [GeV]  lambda kind    colour order
   32        2   1036.49    16.3 string  d ubar
   15        2    553.64    15.0 string  d cbar
    5       10    369.92    31.6 string  u g g g g g g g g ubar
        joined g (row 730) into g (row 924): pair mass above the quark masses 0.075 < mJoin = 0.3 GeV
   ...
```

Each string then gets a block headed `---- string n: W = ..., k partons`
with its hadrons in rank order from each end: the hadron, the rank, the
end (`+` or `-`), the z drawn and the momentum. For string 1 the block is
the full narration the chapter's theory quotes: the first hadron from the
ubar end (a pi0, the flavour draw with P(quark) = 0.925 and d:u:s =
1:1:0.217, the kick (-0.603, -0.143) GeV, z = 0.1342, Gamma = 2.595 GeV^2),
the B*bar0 from the b end (Lund-Bowler exponent 20.305, b mT^2 = 27.859,
<z> = 0.891, z drawn 0.9843, region (0,1), Gamma = 0.453 GeV^2), the
antidiquark draw that made the Sigmabar0 and its Sigma0 partner, the eta'
across a kink in region (1,3) at Gamma = 1.600 GeV^2, the remaining mass
after fourteen hadrons (1.209 GeV below W_min = 2.750 smeared to 2.998),
the final d dbar pair, and the orientation draw (ln(f+ f-) = -1.74 forward
against -0.02 reversed, reversed with P = 0.848).

## 3. Drawing the event and rendering the film

### The (eta, phi) display

`plot_journey_Had.py` draws the record in the (eta, phi) plane in the
layout of the CMSSW `genPartAnalyzer`, reusing the reader and the frame of
chapter 2's script. The categories of this stage are three: the hadron of
each hard b quark, found as the primary hadron of the quark's string that
carries its flavour, in red and blue; their vector sum, the magenta star of
every stage; and every other primary hadron in purple, filled because a
hadron is what the detector will see. The header keeps the two B hadron
lines of the CMSSW figure and adds x_E and x_pT of each, the number of
strings, and the vector sum of all hadrons, which must vanish.

```bash
python3 example/Had/plot_journey_Had.py                 # -> example/Had/journey_Had.pdf
python3 example/Had/plot_journey_Had.py -i my.lhe --png
```

### The theory plots

`plot_theory_Had.py` replays the seed-1 hadronisation with the classes of
the script and reads the `--repeat` statistics and the PYTHIA validation
files when they are present; it writes `theory_hadronisation.pdf`, one page
per theory subsection:

| Page | Chapter figure | Content |
|---|---|---|
| 1 | the linear potential | the potential against the Cornell form; the break-up vertices of the 27.5 GeV string in light-cone coordinates; the multiplicity of every string against its mass |
| 2 | the break-up | the tunnelling factor against the mass; the kicks of every break against the Gaussian; the species of the primary hadrons |
| 3 | the fragmentation function | the Lund function for a pion, a rho, a proton and a kaon; the z values drawn; the remaining mass of the b string hadron by hadron |
| 4 | gluons as kinks | the b string in (eta, phi); the region map with the paths of the two ends; the hadrons in rapidity |
| 5 | the Bowler factor | Lund-Bowler against plain Lund and Peterson; the x_E distributions of both implementations; the B species |
| 6 | x_B in different settings | the hadron's pT against the quark's; the share of the cone against x_E; the means from ALEPH to the single events |
| 7 | our jet at this stage | the hadron-level cones, the cone mass against the multiplicity, the ledger from quark to hadron to cone |

```bash
python3 example/Had/Standalone/plot_theory_Had.py      # -> theory_hadronisation.pdf
make figures                                           # the same, from the book directory
```

### The film

`animate_standalone_hadronisation.py` imports the standalone script and
replays the seed-1 hadronisation in nine narrated scenes of about seven
minutes in all (`--scene guests,wedding,snapshots,ladder,yoyo,bstring,kinks,event,ensemble`
or `1..9`):

1. **guests**: the combinations of quarks and antiquarks QCD lets out of
   the house, one by one; the string makes mesons and baryons only.
2. **wedding**: the mother's rope, b g g g g ubar, snapping in the order
   the script drew, a pair from the vacuum at every break, and the
   procession of its sixteen hadrons with their quark content.
3. **snapshots**: a generic rope in its rest frame, snapshot by snapshot,
   and the same history on the world sheet in light-cone coordinates.
4. **ladder**: the mother's rope from the `-v` log, one row per break,
   down to the final two, then the chain of sixteen with the fifteen
   vacuum pairs.
5. **yoyo**: the 28 GeV two-parton string of our event breaking in its
   rest frame, the vertices at their proper times, and the hyperbola of
   constant Gamma on the right.
6. **bstring**: the 69 GeV string that holds the mother in the (eta, phi)
   plane; hadrons appear from the two ends in the order drawn and the
   table fills with the end, z and the remaining mass.
7. **kinks**: the world sheet of the same string as a map of regions, the
   paths of the two ends and the bars of each gluon half still uncrossed.
8. **event**: the 211 partons grouped into 33 colour singlets and
   fragmented heaviest first, 365 hadrons appearing with the momentum sum
   unchanged.
9. **ensemble**: the same partons hadronised with seeds 1, 2, 3, ...; the
   x_E histogram and its running mean settling on 0.882 next to PYTHIA's
   0.913; the B hadron species on the right.

```bash
cd example/Had/Standalone
python3 animate_standalone_hadronisation.py            # the film and its .srt, over an hour to render
python3 animate_standalone_hadronisation.py --fast     # low-resolution preview
python3 animate_standalone_hadronisation.py --print-script   # the narration, to record it yourself
python3 animate_standalone_hadronisation.py --no-narration   # music only, about 80 s
python3 animate_standalone_hadronisation.py --scene bstring -o bstring.gif
python3 animate_standalone_hadronisation.py --stills animate_standalone_hadronisation_stills.pdf --no-video
```

Rendering needs matplotlib in addition to NumPy, `ffmpeg` for an `.mp4`,
and `edge-tts` for the voice; the narration and the score come from the
shared `example/soundtrack.py` (clips cached in `build/narration/Had/`).
The last command writes the key frames (one page per scene) that the
chapter prints; `make figures` runs it. The physics options are those of
the standalone script: `--seed 7` films another hadronisation,
`--peterson` another fragmentation function for the b, `--rfactb 0` the
plain Lund function, and `--events` sets the hadronisations of scene 9.

## 4. Validation from official packages

Two official sources: PYTHIA driven from a script, and the CMS generator
job of chapter 1, which hadronised its event in the same step that showered
and dressed it. The new thing at this stop is that the first comparison can
be made on identical input. MadGraph does not appear: hadronisation is
PYTHIA's job in every CMS sample.

### 4.1 The same strings with PYTHIA 8

`validate_hadronisation_with_pythia.py` has two modes. In mode `record`
(the default) it switches PYTHIA's process level off
(`ProcessLevel:all = off`), fills the event record by hand with the 211
final partons of `event_MPI.lhe`, their colour tags and status 23, in the
manner of PYTHIA's `main21` example, and lets `next()` run the hadron level
on them with the Monash fragmentation parameters that CP5 keeps and hadron
decays off, N times with different seeds. The standalone script does the
same with `--repeat N`: same strings, two implementations of one model. In
mode `full` the hard event of chapter 1 is written to a Les Houches file N
times and PYTHIA runs showers, multiparton interactions, remnants, colour
reconnection and hadronisation with the CP5 settings, decays off, with or
without the impact-parameter window of chapter 4. Both modes record per
event what the chapter's table lists, in the JSON layout of the standalone
statistics so that `plot_theory_Had.py` draws both. As before PYTHIA's
output goes through C++ streams and is captured by redirecting the file
descriptor.

```bash
cd example/Had/Standalone
PY=../../ME/Standalone/.venv/bin/python
$PY validate_hadronisation_with_pythia.py -n 400 --standalone-stats standalone_had_stats.stats.json      # mode record
$PY validate_hadronisation_with_pythia.py -n 400 --no-bowler --output validate_hadronisation_with_pythia_record_rfactb0.json
$PY validate_hadronisation_with_pythia.py -n 400 --peterson  --output validate_hadronisation_with_pythia_record_peterson.json
$PY validate_hadronisation_with_pythia.py --mode full -n 2000
$PY validate_hadronisation_with_pythia.py --mode full -n 8000 --impact 0.3136 --window 0.03 --output validate_hadronisation_with_pythia_full_fixedb.json
```

The first three runs take a few seconds each, the full chain a few minutes.
The results are the JSON files `validate_hadronisation_with_pythia_{record,record_rfactb0,record_peterson,full,full_fixedb}.json`
and the printouts `validate_hadronisation_with_pythia{,_record}.log`;
PYTHIA's first hadronisation of our partons is in
`validate_hadronisation_with_pythia_record_event1.txt` and its first
full-chain event in `..._full_event1.txt`, both in the standalone format,
so that `plot_journey_Had.py` draws them. The tables are in section 5.1.

| Option | Meaning |
|---|---|
| `--mode record` / `full` | hadronise the 211 partons of chapter 4 / run the full CP5 chain from the hard event |
| `-n`, `--seed` | events and seed |
| `--impact`, `--window` | full mode: keep events with abs(b/`<b>` - impact) < window |
| `--no-bowler`, `--peterson` | `StringZ:rFactB = 0`, `StringZ:usePetersonB/C = on` |
| `--standalone-stats` | the `.stats.json` of the standalone `--repeat`, printed side by side |
| `--output`, `--listing` | the JSON file and the first event's listing |

### 4.2 CMSSW: hadronisation inside the GEN step

Nothing new has to be run. The `GEN` step of chapter 1 hadronised the
event, since `HadronLevel:all` is on in every CMS configuration, and
JitJetSW's analyser drew the primary hadrons as its fourth figure. From the
setup of the [chapter 2 companion](chapter_02_PS.md#42-cmssw-the-shower-inside-the-gen-step):

```bash
cd CMSSW_15_0_5/src/JitJetSW/Journey/test
./runCMSDrivers_mc2024_BBbar_GEN.sh          # GEN step + genPartAnalyzer, one PDF per stage
```

produces `genPartAnalyzer_ME_PS_MPI_Had_out.pdf`, whose fourth page is the
chapter's CMSSW figure; the copy in `CMSSW/` is the one used, and the log
is `example/PS/CMSSW/step1_GEN_PS_cfg.log` of chapter 2. The hadronisation
settings do not appear in the log's table of changed settings because CP5
changes none of them: `Tune:pp = 14` selects `Tune:ee = 7`, the Monash
values the script embeds.

**Reading the log.** The particle listing has, after the status-62 and 63
rows of chapter 4, 120 rows of status 71 and 45 of status 73, the copies
PYTHIA makes when it collects each colour singlet and joins very soft
partons; grouped by their common daughter range they form 24 colour
singlets. The mother's string and its hadrons:

```text
  515 |     5 -     b | 62 |  182  182  824  824 |  1  1 |  91.744  -1.570 -2.084 | ...
  824 |     5 -     b | 71 |  515  515  829  834 |  1  6 |  91.744  -1.570 -2.084 | ...
  825 |    21 -     g | 71 |  520  520  829  834 |  1  6 |  15.374  -1.579 -2.151 | ...
  828 |    -3 -  sbar | 71 |  516  516  829  834 |  1  6 |   3.510  -1.723 -1.963 | ...
  829 |  -521 -    B- |  2 |  824  828 1434 1436 |  5  3 |  94.279  -1.567 -2.092 | ...
```

Row 515 is her status-62 copy at 91.744 GeV; row 824 is her status-71 copy,
the first of five partons, b g g g sbar, that form a string whose daughters
are rows 829 to 834, six hadrons: a B-, three pi0, a pi+ and a K0. The B-
at 94.279 GeV and eta = -1.567 carries x_pT = 94.279/91.744 = 1.028 and
x_E = 235.85/230.04 = 1.025. The bbar's string is d bbar, rows 963 and 964,
and made four hadrons, pi- K+ K*- B*+, the B*+ at 114.17 GeV keeping
x_E = 0.980 of the 116.36 GeV quark. The cone numbers are in section 5.1.

## 5. The full tables

### 5.1 Hadronisation obtained three ways

"Same partons" means the 211 final partons of `event_MPI.lhe` with their
colour tags, hadronised 400 times by PYTHIA 8.315 with the Monash
fragmentation parameters that CP5 keeps, and 400 times by the standalone
script; "full chain" means PYTHIA from the hard event of chapter 1 with the
CP5 settings, 2000 events with b drawn and 526 with
abs(b/`<b>` - 0.3136) < 0.03; hadron decays are off everywhere, so every
hadron is primary. The spread of a single event follows +-. Cones are drawn
around the final copy of the b quark; the B share is the hadron's pT over
the cone's. Commands: `validate_hadronisation_with_pythia.py -n 400`
(column 1), `standalone_hadronisation.py --repeat 400` (column 2),
`--mode full -n 2000` (column 3), `--mode full -n 8000 --impact 0.3136 --window 0.03`
(column 4), `standalone_hadronisation.py` (column 5), the drawer's table
of the CMSSW log (column 6).

| quantity | PYTHIA, same partons | standalone, same partons | PYTHIA full, b drawn | PYTHIA full, b = 0.3136 | standalone, seed 1 | CMSSW, 1 event |
|---|---|---|---|---|---|---|
| primary hadrons | 383 +- 11 | 402 +- 14 | 226 +- 87 | 306 +- 52 | 365 | 320 |
| charged | 207.5 +- 11.9 | 217.2 +- 12.5 | 122 +- 48 | 165 +- 29 | 184 | 167 |
| baryons | 44.0 +- 7.6 | 38.8 +- 7.6 | 26.8 | 36.3 | 52 | 34 |
| with strangeness | 87.8 +- 10.3 | 92.1 +- 10.5 | 49.6 | 67.8 | 94 | 69 |
| at abs(eta) < 2.5 | 193.8 | 206.3 | 91.9 | 128.3 | 179 | 139 |
| colour singlets, partons joined | 33, 48 | 33, 48 | -- | -- | 33, 48 | 24, -- |
| b: x_E = E_B/E_b (Lund-Bowler) | 0.913 +- 0.062 | 0.882 +- 0.078 | 0.941 +- 0.100 | 0.931 +- 0.089 | 1.004 | 1.025 |
| b: x_E, plain Lund (r_b = 0) | 0.963 +- 0.038 | 0.952 +- 0.048 | -- | -- | -- | -- |
| b: x_E, Peterson (epsilon_b = 0.005) | 0.870 +- 0.120 | 0.839 +- 0.132 | -- | -- | -- | -- |
| bbar: x_E | 1.033 +- 0.049 | 1.025 +- 0.072 | 0.941 +- 0.095 | 0.936 +- 0.094 | 1.154 | 0.980 |
| b: pT of its hadron [GeV] | 71.1 +- 5.0 | 69.4 +- 5.2 | 87.8 +- 26.7 | 87.7 +- 27.3 | 78.56 | 94.28 |
| b: B*bar0 : B*- : Bbar0 : B- : B*_sbar0 [%] | 31:28:14:12:8 | 31:32:13:12:6 | 29:29:13:15:6 | 30:29:14:14:4 | B*bar0 | B- |
| b, dR < 0.4: partons pT [GeV] | 100.67 | 100.67 | 109.7 | 109.7 | 100.67 | -- |
| b, dR < 0.4: hadrons pT [GeV] | 98.0 +- 2.2 | 96.8 +- 3.1 | 109.6 | 109.8 | 99.23 | 114.45 |
| b, dR < 0.4: hadrons, mass [GeV] | 7.6, 17.2 | 8.0, 16.8 | 6.2, 14.3 | 6.7, 15.4 | 6, 16.1 | 9, 20.4 |
| b, dR < 0.4: B share of the cone pT | 0.726 | 0.710 | 0.800 | 0.799 | 0.79 | 0.82 |
| b, dR < 0.8: partons pT [GeV] | 104.09 | 104.09 | 118.4 | 120.0 | 104.09 | -- |
| b, dR < 0.8: hadrons pT [GeV] | 104.2 +- 1.8 | 104.6 +- 2.5 | 118.7 | 120.6 | 105.98 | 141.1 |
| b, dR < 0.8: hadrons, mass [GeV] | 15.1, 31.6 | 16.6, 32.5 | 12.0, 28.1 | 14.2, 32.9 | 14, 35.9 | 28, 49.8 |

On identical partons the standalone script makes 5% more hadrons (the
excess sits in the gluon-rich strings: 46 against 43 hadrons from the
108 GeV string with 21 partons, 24 against 21 from the 95 GeV one; the
two-parton strings agree to 2%), 11% fewer baryons (no popcorn), and its B
hadron keeps 0.03 less of the quark's energy; the Bowler, plain-Lund and
Peterson variants move both implementations the same way.

### 5.2 Kinematics of the two b quarks, their hadrons and their cones

From `event_Had.lhe` as printed by `standalone_hadronisation.py`; the parton
rows are those of the chapter 4 companion.

| | | pT [GeV] | m [GeV] | n | E [GeV] |
|---|---|---:|---:|---:|---:|
| b | after the MPI stage (row 719) | 78.29 | 4.80 | 1 | 165.42 |
| | its B*bar0 (row 1012, status 83) | 78.56 | 5.33 | 1 | 166.04 |
| | partons within dR < 0.4 (chapter 4) | 100.67 | 13.52 | 3 | |
| | hadrons within dR < 0.4 | 99.23 | 16.08 | 6 (2 charged) | |
| | partons within dR < 0.8 | 104.09 | 20.17 | 6 | |
| | hadrons within dR < 0.8 | 105.98 | 35.87 | 14 | |
| bbar | after the MPI stage (row 708) | 72.50 | 4.80 | 1 | 1106.73 |
| | its B*0 (row 1069, status 84) | 82.83 | 5.33 | 1 | 1276.92 |
| | partons within dR < 0.4 | 93.88 | 7.75 | 4 | |
| | hadrons within dR < 0.4 | 91.34 | 7.58 | 4 (1 charged) | |
| | partons within dR < 0.8 | 116.07 | 41.61 | | |
| | hadrons within dR < 0.8 | 116.92 | 47.92 | 11 | |
| all primary hadrons | | 0.00 | 13600 | 365 | 13600 |

### 5.3 The four fractions of our event

| | b (rows 719 -> 1012) | bbar (rows 708 -> 1069) |
|---|---:|---:|
| x_E = E_B / E_b | 1.004 | 1.154 |
| x_pT = pT_B / pT_b | 1.004 | 1.142 |
| x_+ = (E + p_par)_B / (E + p_par)_b | 1.004 | 1.154 |
| pT_B / pT(cone, dR < 0.4) | 0.79 | 0.91 |
| dR(hadron, quark) | 0.005 | 0.012 |

The bbar's B*0 was one of the final two of its string (b g g bbar, W = 40 GeV,
with a secondary b at the other end): after the secondary b's B*bar0, a K0
and a Kbar0 from the other end, the remaining 6.28 GeV was below
W_min = 7.22, and the B*0 took its share of what was left, including the
two gluons' halves.

### 5.4 The mother's string in both records

| | standalone (seed 1) | CMSSW |
|---|---|---|
| the string | b g g g g ubar, W = 69.34 GeV, 6 partons (rows 1006 to 1011) | b g g g sbar, 5 partons (rows 824 to 828) |
| its hadrons | 16 (rows 1012 to 1027) | 6 (rows 829 to 834): B-, pi0, pi0, pi0, pi+, K0 |
| the B hadron | B*bar0, row 1012, pT 78.56 GeV, x_E = 1.004 | B-, row 829, pT 94.279 GeV, x_E = 1.025 |
| the quark before | row 719, pT 78.29 GeV, E 165.42 GeV | row 515, pT 91.744 GeV, E 230.04 GeV |
| hadrons within dR < 0.4 | 6, pT 99.23 GeV, m 16.08 GeV, B share 0.79 | 9, pT 114.45 GeV, m 20.4 GeV, B share 0.82 |
| hadrons within dR < 0.8 | 14, pT 105.98 GeV, m 35.87 GeV | 28, pT 141.1 GeV, m 49.8 GeV |

### 5.5 The energy ledger

| Stage | pT [GeV] | m [GeV] | n_const | delta pT [GeV] |
|---|---:|---:|---:|---:|
| b family after the MPI stage (dR < 0.4, partons) | 100.67 | 13.52 | 3 | |
| b family after this stage (dR < 0.4, hadrons) | 99.23 | 16.08 | 6 | -1.44 |
| of which the B*bar0 itself (x_E = 1.004) | 78.56 | 5.33 | 1 | |
| the same within dR < 0.8 | 105.98 | 35.87 | 14 | +1.89 |
