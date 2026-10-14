# From partons to hadrons: the Lund string without PYTHIA

`standalone_hadronisation.py` takes the parton-level record of chapter 4
(`../../MPI/event_MPI.lhe`: the hard b bbar system with its showers, the
secondary scatterings, the beam remnants, the colour tags after colour
reconnection) and turns it into primary hadrons with the Lund string model,
written from scratch in Python following PYTHIA 8's `ColourTracing`,
`StringFragmentation`, `StringFlav`, `StringZ`, `StringPT` and
`MiniStringFragmentation`. The output, `../event_Had.lhe`, is a PYTHIA-style
listing (status codes, mother/daughter links) that chapter 6 decays.

```bash
python3 standalone_hadronisation.py                      # seed 1 -> ../event_Had.lhe (the book's record)
python3 standalone_hadronisation.py -v                   # learning mode: the b quark's string hadron by hadron
python3 standalone_hadronisation.py -vv                  # ... every string
python3 standalone_hadronisation.py --peterson           # Peterson instead of Lund-Bowler for b and c
python3 standalone_hadronisation.py --rfactb 0           # plain Lund for the b quark
python3 standalone_hadronisation.py --repeat 400 --quiet --history standalone_had_stats.json
```

On this Mac use `/usr/bin/python3` (NumPy); one hadronisation takes 0.5 s, the
`--repeat 400` ensemble three minutes. The script imports chapter 2's
`standalone_parton_showering.py` for the event record and the Lorentz frames;
its hadron table (names, masses, charges of the 78 hadrons the string can
make, the constituent masses of quarks and diquarks) is embedded between
`BEGIN/END GENERATED TABLES` by `dump_hadron_tables.py`, which reads PYTHIA's
`ParticleData` once in the `example/ME/Standalone/.venv` environment.

## What the script reproduces, and the numbers

* **Strings.** The colour tags are followed from every quark or antidiquark
  through the gluons to an antiquark or diquark; the record has 33 colour
  singlets (29 strings and 4 ministrings with W below m_q1 + m_q2 + 1 GeV, no
  closed gluon loops), 3071 GeV of string mass in all. As PYTHIA, two
  colour-connected partons whose pair mass (gluons at half momentum) exceeds
  the endpoint masses by less than mJoin = 0.3 GeV are joined: 48 partons of
  the 211 (PYTHIA on the same partons: 48 as well). Note that chapter 4's "51
  colour-singlet strings" counted lone anticolour ends as chains; the colour
  tags give 33.
* **The string and its break-ups.** Every string is a sequence of regions
  spanned by the massless light-cone pieces of neighbouring partons (a gluon
  carries half its momentum into the region on either side). Hadrons are
  produced from the two ends in random order; a hadron takes a fraction z of
  the remaining momentum of its end's own piece with the Lund symmetric
  function f(z) = (1/z)(1-z)^a exp(-b mT^2/z), a = 0.68, b = 0.98 GeV^-2
  (Monash, kept by CP5), and as much of the other side as its mass needs.
  If that does not fit in the current region, the new vertex is placed in
  the regions further along the string as the point with proper time
  Gamma_new = (1-z)(Gamma_old + mT^2/z) at which the hadron is on shell
  (Sjostrand, Nucl. Phys. B248 (1984) 469): two bilinear equations, a
  quadratic, scanning the end's own side outwards and then the other side;
  the gluon halves in between are swallowed, which is how a hadron forms
  across a kink. The chain stops when the remaining mass falls below
  W_min = m_q1 + m_q2 + stopMass + stopNewFlav m_qlast (1.0, 2.0; smeared by
  20 %), or when an end finds no room; the last two hadrons share what is
  left, with the orientation chosen by the fragmentation functions of both
  ends. A string that fails ten times is split into two hadrons (0.5 % of
  the 400 hadronisations have one such string).
* **Flavour and pT.** Gaussian pT of width sigma = 0.335 GeV for the pair
  (sigma/sqrt 2 per component, 1 % of the breaks with twice the width);
  s : u : d = 0.217 : 1 : 1; diquarks with probQQtoQ = 0.081, strange
  diquarks x 0.915, spin-1 diquarks x 3 x 0.0275; vector/pseudoscalar 0.5
  (u, d), 0.55 (s), 0.88 (c), 2.2 (b); eta and eta' from the mixing angle
  -15 degrees with suppressions 0.6 and 0.12 (a rejected eta means a new
  flavour, as in PYTHIA); octet and decuplet baryons with SU(6) weights
  derived from the spin-flavour wave functions (they are PYTHIA's table in
  the normalisation Delta++ from uu_1 + u = 1; the rejected part of the
  weight also means a new flavour). No popcorn: a baryon and its antibaryon
  are always neighbours.
* **Heavy quarks.** A b or c at a string end fragments with the Lund-Bowler
  function (1/z^(1 + r_Q b m_Q^2)) (1-z)^a exp(-b mT^2/z), r_b = 0.855,
  r_c = 1.32, m_b = 4.8, m_c = 1.5 (the exponent for the b is 20.3; <z> =
  0.89 at mT = m_B*); `--rfactb 0` gives the plain Lund function (<z> =
  0.95, far too hard) and `--peterson` the Peterson function with
  epsilon_b = 0.005.
* **Ministrings** give two hadrons along the first parton's direction with
  the pT of one break, or one hadron whose mass difference is taken from
  the string system with the smallest pair mass (its partons are boosted;
  their copies carry status -72).

Left out on purpose: popcorn, junctions (the record has none), the thermal
pT model, ropes and close packing, hadron decays (chapter 6).

## The record (seed 1)

365 primary hadrons (CMSSW's event: 320), 184 charged, 52 baryons, 94 with
strangeness; 179 hadrons (95 charged) at |eta| < 2.5. The b quark (row 719, pT 78.29,
E 165.42 GeV) sits at one end of a 69 GeV string with four gluons; the
first hadron from that end is a B*bar0 with z = 0.984, x_E = E_B/E_b =
1.004 (a high draw: the ensemble mean is 0.88), rank 1 of 16 hadrons. The
bbar (row 708, pT 72.50) is the end of a 40 GeV string whose other end is a
b quark from a secondary scattering; its B*0 has x_E = 1.154 (the hadron
took momentum from the gluon next to the quark: x_B > 1 is not a mistake,
CMSSW's B- has x_pT = 94.28/91.74 = 1.03). Cones around the b quark: R = 0.4
holds 3 partons with pT 100.67 and mass 13.5 GeV before and 6 hadrons with
pT 99.23 and mass 16.1 GeV after; the B hadron carries 79 % of the cone pT;
R = 0.8: 104.09 -> 105.98 GeV, mass 20.2 -> 35.9. The `-v` log is
`standalone_hadronisation.log` (2200 lines: the strings, the b quark's string
break by break with every draw, the final two, the ministrings, the checks
and the record); chapter 5 quotes its numbers. Momentum is conserved to
1e-12 GeV with respect to the partons handed in.

## Validation against PYTHIA 8 (`validate_hadronisation_with_pythia.py`)

```bash
PY=../../ME/Standalone/.venv/bin/python
$PY validate_hadronisation_with_pythia.py -n 400 --standalone-stats standalone_had_stats.stats.json
$PY validate_hadronisation_with_pythia.py -n 400 --no-bowler --output validate_hadronisation_with_pythia_record_rfactb0.json
$PY validate_hadronisation_with_pythia.py -n 400 --peterson  --output validate_hadronisation_with_pythia_record_peterson.json
$PY validate_hadronisation_with_pythia.py --mode full -n 2000
$PY validate_hadronisation_with_pythia.py --mode full -n 8000 --impact 0.3136 --window 0.03 --output validate_hadronisation_with_pythia_full_fixedb.json
```

Mode `record` (the sharp test): PYTHIA 8.315 hadronises exactly the 211
partons of `event_MPI.lhe` with their colour tags (`ProcessLevel:all = off`,
Monash fragmentation, decays off) 400 times; the standalone script
hadronises them 400 times with `--repeat`. Same strings, two Lund
implementations. Mode `full`: the full CP5 chain from the hard event, as in
chapter 4, decays off, with and without the impact-parameter window.

| quantity | PYTHIA, same partons | standalone, same partons | PYTHIA, full chain, b drawn | PYTHIA, full chain, b = 0.3136 |
|---|---|---|---|---|
| primary hadrons | 383 +- 11 | 402 +- 14 | 226 +- 87 | 306 +- 52 |
| charged | 207.5 +- 11.9 | 217.2 +- 12.5 | 122 +- 48 | 165 +- 29 |
| baryons | 44.0 +- 7.6 | 38.8 +- 7.6 | 26.8 | 36.3 |
| with strangeness | 87.8 +- 10.3 | 92.1 +- 10.5 | 49.6 | 67.8 |
| hadrons at abs(eta) < 2.5 | 193.8 | 206.3 | 91.9 | 128.3 |
| b: x_E = E_B/E_b | 0.913 +- 0.062 | 0.882 +- 0.078 | 0.941 +- 0.100 | 0.931 +- 0.089 |
| b: x_E, plain Lund (rFactB = 0) | 0.963 +- 0.038 | 0.952 +- 0.048 | | |
| b: x_E, Peterson | 0.870 +- 0.120 | 0.839 +- 0.132 | | |
| bbar: x_E | 1.033 +- 0.049 | 1.025 +- 0.072 | 0.941 +- 0.095 | 0.936 +- 0.094 |
| b: B hadron pT [GeV] | 71.1 +- 5.0 | 69.4 +- 5.2 | 87.8 +- 26.7 | 87.7 +- 27.3 |
| b, R = 0.4: partons -> hadrons pT [GeV] | 100.67 -> 98.0 +- 2.2 | 100.67 -> 96.8 +- 3.1 | 109.7 -> 109.6 | 109.7 -> 109.8 |
| b, R = 0.4: hadrons, mass [GeV] | 7.6, 17.2 | 8.0, 16.8 | 6.2, 14.3 | 6.7, 15.4 |
| b, R = 0.4: B share of the cone pT | 0.726 | 0.710 | 0.800 | 0.799 |
| b, R = 0.8: partons -> hadrons pT [GeV] | 104.09 -> 104.2 +- 1.8 | 104.09 -> 104.6 +- 2.5 | 118.4 -> 118.7 | 120.0 -> 120.6 |
| b, R = 0.8: hadrons, mass [GeV] | 15.1, 31.6 | 16.6, 32.5 | 12.0, 28.1 | 14.2, 32.9 |
| b hadron species | B*bar0 31, B*- 28, Bbar0 14, B- 12, B*_sbar0 8, B_sbar0 2, Lambda_b0 1 % | 31, 32, 13, 12, 6, 2, 2 % | 29, 29, 13, 15, 6, 3, 2 % | |

On identical partons the standalone script makes 5 % more hadrons (the
excess sits in the gluon-rich strings: 46 against 43 hadrons from the 108
GeV string with 21 partons, 24 against 21 from the 95 GeV one; the
two-parton strings agree to 2 %), 11 % fewer baryons (no popcorn), and its
B hadron keeps 3 % less of the quark's energy (0.882 against 0.913); the
Bowler, plain-Lund and Peterson variants move both implementations the
same way (plain Lund +0.05 to +0.07, Peterson -0.04). What the detector
will see agrees better than that: the hadron-level cone pT around the b
differs by 1.2 GeV (R = 0.4) and 0.4 GeV (R = 0.8) between the two, both
within 4 GeV of the parton-level cone, and the cone masses agree to 0.5
GeV. The full chain at the CMSSW impact parameter gives 306 +- 52 primary
hadrons; CMSSW's single event has 320 from 26 strings, its B- carries x_pT
= 1.03 of the b quark and 114.45 GeV sit within R = 0.4 of it (9 hadrons,
mass 20.4 GeV). PYTHIA's first hadronisation of our partons is in
`validate_hadronisation_with_pythia_record_event1.txt`, the first
full-chain event in `..._full_event1.txt`, both in the standalone format.

## Figures

* `../plot_journey_Had.py` -> `../journey_Had.pdf`: the stage-4 (eta, phi)
  map in the layout of the CMSSW genPartAnalyzer (page 4 of
  `../CMSSW/genPartAnalyzer_ME_PS_MPI_Had_out.pdf`): the two B hadrons in red
  and blue, the primary hadrons in purple, the vector sum, and the x_E and
  x_pT of both B hadrons in the header.
* `plot_theory_Had.py` -> `theory_hadronisation.pdf`, one page per theory
  section of chapter 5: the linear potential, the break-up vertices of a 28
  GeV string in light-cone coordinates and the multiplicity of every string
  against its mass; tunnelling, the pT kicks and the species; the Lund
  function, the z values drawn and the remaining mass of the b string hadron
  by hadron; the b string in (eta, phi), the region map with the paths of the
  two ends and the hadrons in rapidity; Lund-Bowler against plain Lund and
  Peterson with the x_E distributions of both implementations and the B
  species; x_B in different settings (the hadron against the quark, the
  share of the jet, the means from the same partons, the full chain, ALEPH's
  0.716 in Z decays, our event and the CMSSW event); the hadron-level cones,
  masses and multiplicities and the ledger from quark to B hadron to jet.
  It replays the seed-1 hadronisation and reads the `--repeat` statistics
  (`standalone_had_stats*.stats.json`) and the validation files when present.
* `animate_standalone_hadronisation.py`: nine scenes, about 80 s. Four
  introductory sketches (the guest list of colour singlets QCD allows; the
  mother's rope snapping into its chain of sixteen hadrons with their
  quark content; a generic rope snapshot by snapshot with its world sheet
  in light-cone coordinates; the mother's rope break by break from the -v
  log, down to the final two and the chain with its vacuum pairs), then
  the five scenes that follow the record (a two-parton string breaking in
  its rest frame, the b quark's string hadron by hadron, the two ends
  walking through the regions of the world sheet, the whole event turning
  from partons into hadrons, the x_E ensemble); same CLI as the chapter-2
  film, `--scene guests,wedding,...` or `1..9`; `make figures` writes the
  stills PDF only.

`make records` regenerates `../event_Had.lhe`, `make figures` the three PDFs.
