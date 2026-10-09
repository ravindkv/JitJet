# The rest of the two protons: MPI, beam remnants and colour reconnection without PYTHIA

`standalone_multiparton_interactions.py` takes the showered record of chapter 2
(`../../PS/event_PS.lhe`: the hard b bbar system with its ISR and FSR) and adds
what the rest of the two protons does: the secondary parton-parton scatterings
(multiparton interactions, MPI) with their own showers, the beam remnants with
primordial transverse momentum, and the MPI-based colour reconnection, all
written from scratch in Python following PYTHIA 8's `MultipartonInteractions`,
`BeamParticle`, `BeamRemnants` and `ColourReconnection::reconnectMPIs`. The
output, `../event_MPI.lhe`, is a PYTHIA-style event listing (status codes,
mother/daughter links, colour tags) that chapter 5 hadronises.

```bash
python3 standalone_multiparton_interactions.py --impact 0.3136      # -> ../event_MPI.lhe (the book's record)
python3 standalone_multiparton_interactions.py                      # draw the impact parameter instead
python3 standalone_multiparton_interactions.py --impact 0.3136 -v   # learning mode: every number, every step
python3 standalone_multiparton_interactions.py --impact 0.3136 -vv  # ... every veto trial and the secondary showers
python3 standalone_multiparton_interactions.py --no-cr --no-kt      # switch colour reconnection / primordial kT off
python3 standalone_multiparton_interactions.py --repeat 400 --impact 0.3136 --quiet --history stats.json
```

On this Mac use `/usr/bin/python3` (NumPy + SciPy); the cross-section
integration at start-up takes 3 s and one event about 2 s (25 systems, each
showered with the chapter-2 shower). The script imports chapter 2's
`standalone_parton_showering.py` for the running coupling, the event record,
the frames and the shower itself; its own PDF grid (NNPDF31_nnlo_as_0118 on
LHAPDF's knots from x = 1e-9, embedded by `dump_mpi_tables.py`) goes to much
smaller x than chapter 2's because a 0.2 GeV scattering at rapidity 9 takes
x ~ 1e-8 from a proton.

## Why the impact parameter is fixed in the book's record

The first thing the model draws is the impact parameter. For an event with a
hard process it is distributed as the overlap O(b) of the two protons; seed 1
lands at b = 1.21 <b> (f = 0.20, one secondary interaction), a quiet event.
The CMSSW event of `../CMSSW/genPartAnalyzer_ME_PS_MPI_out.pdf` has b = 0.3136 <b>
(f = 3.709, 32 interactions). To walk through a rich event and to compare
with CMSSW at the same geometry the Makefile fixes `--impact 0.3136` (PYTHIA
offers the same through `UserHooks::doSetImpactParameter`); the chapter says so,
and the ensembles below are given both ways.

## What the script reproduces, and the numbers

* **pT0 and the interaction cross section.** pT0 = 1.41 (13600/7000)^0.03344
  = 1.4417 GeV. The regularised QCD 2 -> 2 cross section (all channels of
  chapter 1 plus g g / q qbar -> c cbar, b bbar; alpha_s^2(pT^2+pT0^2)/(pT^2+pT0^2)^2
  in place of alpha_s^2(pT^2)/pT^4) integrated from pTmin = 0.2 GeV gives
  sigma_int = 497.3 mb; PYTHIA prints 500.67 mb in the CMSSW log and 496.6 mb in
  the validation run here. With sigma_ND = 100.309 - 21.89 - 23.0 = 55.42 mb the
  average non-diffractive event has <n> = sigma_int/sigma_ND = 8.97 interactions.
  Two things had to be right to get this number: a final state of two
  different partons is met twice when both rapidities are integrated over the
  full range (PYTHIA's average of the t- and u-channel-sampled configurations),
  and the PDFs are frozen below the grid edge Q = 1.65 GeV (PYTHIA's LHAPDF6
  and LHAGrid1 readers both do that; letting LHAPDF extrapolate gives 352 mb,
  the chapter-2 grid frozen at x = 1e-3 gives 717 mb).
* **The impact parameter.** Double-Gaussian protons (coreRadius 0.7634,
  coreFraction 0.63); n(b) = k O(b) with k fixed by <n>; f(b) = n(b)/<n> =
  O(b) int P_int d^2b / int O d^2b. f(0.3136 <b>) = 3.63 against PYTHIA's 3.709;
  <f> = 2.22 for events with a hard process (PYTHIA: 2.26 measured), 1 for
  minimum bias by construction; f(0) = 4.52.
* **The chain.** Below pTmax = 113.3 GeV the interactions come from the veto
  algorithm with the overestimate C/(pT^2 + pT0^2)^2, each accepted scale
  giving y3, y4, x1, x2, the flavours (from the depleted densities) and the
  process (PYTHIA's colour flows, massive kinematics for c and b), then the
  chapter-2 shower from the system's own pT with the depleted proton on each
  side (`BeamPDF`, PYTHIA's `xfISR` logic: a valence initiator evolves with the
  valence density only).
* **The depleted proton** is a port of `BeamParticle`: rescaled x, valence
  counting, companion quarks from g -> q qbar with `companionPower = 4`, the
  CTEQ5L-based valence momentum fractions, the gluon + sea rescaling. One
  simplification: at most one valence quark leaves a proton (a second one is
  reclassified as sea), so no junction topology is needed.
* **Beam remnants** follow `remnantFlavours`, `xRemnant` and
  `setKinematics`: quark + diquark (ud_0 : ud_1 = 3 : 1, uu_1), companions, a
  gluon if nothing else is left; x shapes (1-x)^3.5/sqrt(x) (u), (1-x)^2/sqrt(x)
  (d), diquark = sum x 2, companion from the g -> q qbar ansatz, gluon
  (1-x)^4/x; primordial kT with width (1.5 x 0.9 + Q x 1.8)/(1.5 + Q) times the
  mass damping, 0.4 GeV for remnants, the imbalance shared by all partons of
  the proton; every system keeps its mass and rapidity and is boosted to the
  frame of its new initiators (status -61 copies, 62 copies of the final
  partons), the remnants (status 63) get what is left, with the exact
  light-cone solution of PYTHIA. Energy and momentum close to 1e-10.
* **Colour reconnection** (`reconnectMPIs`, range 5.176): P = (R pT0)^2/((R pT0)^2 + pT^2)
  tested from the softest system upwards; the gluons (and q qbar pairs from a
  gluon splitting) of a merged system go onto the dipole of the harder system
  with the smallest (p_g.p_i)(p_g.p_j)/(p_i.p_j); fully merged gluon pairs of
  initiators become colour-singlet pass-throughs. Then `remnantColours` strings
  initiators and remnants into one chain per proton, with PYTHIA's resolution
  of collapses that both protons made on the same (in-in) colour line and its
  treatment of colour-singlet gluons. The string length lambda = sum ln(1 + m_ij^2/m0^2),
  m0 = 0.3 GeV, is printed with and without reconnection.

Left out on purpose: the exact interleaving of MPI, ISR and FSR in one common
pT sequence (here each system is showered to the cut-off before the next,
softer, interaction is drawn, so the depletion seen by later interactions is
somewhat too large: without showers the chain gives 7.2 interactions per unit
of f, PYTHIA 7.5, with showers 6.5), rescattering, diffraction, photon
processes, junctions, the newer QCD-based CR model, hadronisation (chapter 5).

## The record (`--impact 0.3136`, seed 1)

24 secondary interactions (pT from 11.45 down to 0.43 GeV; CMSSW's event: 31,
from 6.65 to 0.24 GeV), 211 final partons (155 from the secondary systems with
their 31 ISR + 76 FSR branchings, 16 beam remnants), 23 systems merged by
colour reconnection, lambda 806 -> 404, 51 colour-singlet strings. The b quark
keeps its chapter-2 family (100.45 GeV, 13.52 GeV mass within R = 0.4: no
underlying-event parton happened to land in that cone); within R = 0.8 the
underlying event adds 3.4 GeV and lifts the mass from 13.5 to 20.2 GeV. The
transverse-region density (partons not from the hard system, |eta| < 2.5,
60-120 degrees from the leading b) is 2.1 GeV per unit eta-phi. The `-v` log
is `standalone_multiparton_interactions.log` (about 1500 lines, the record
included); chapter 4 quotes its numbers.

## Validation against PYTHIA 8 (`validate_mpi_with_pythia.py`)

```bash
../../ME/Standalone/.venv/bin/python validate_mpi_with_pythia.py -n 2000
../../ME/Standalone/.venv/bin/python validate_mpi_with_pythia.py -n 8000 --impact 0.3136 --window 0.03
```

PYTHIA 8.315 (`pythia8mc`, PDFs through `LHAGrid1`) gets the same hard event
from a Les Houches file and the CP5 settings of the CMSSW configuration with
hadronisation and QED off. Two traps, both met here: `Tune:pp = 14` resets
`PDF:pSet`, so the PDF must be set after the tune (CMSSW does; the chapter-2
validation script had it the other way round and was corrected on 2026-10-09,
which moved its PYTHIA numbers by 1-2 %); and PYTHIA's initialisation goes
through C++ streams, so the file descriptor is redirected to read pT0,
sigma_int and sigma_ND. Each final parton is assigned to its system by walking
the record (an ISR emission hangs on the new incoming parton, whose other
daughter is the previous one). The impact parameter cannot be fixed from the
Python binding, so the fixed-b sample keeps the events with |b/<b> - 0.3136| < 0.03
(537 of 8000).

Averages over 400 standalone events (seeds 1..400, the chapter-2 shower kept
fixed, only the MPI stage redrawn) and 2000 / 537 PYTHIA events (everything
redrawn). "UE" = partons not belonging to the hard system; cones around the
final-state copy of the b quark.

| quantity | PYTHIA, b drawn | standalone, b drawn | PYTHIA, b = 0.3136 | standalone, b = 0.3136 |
|---|---|---|---|---|
| interactions (incl. the hard one) | 18.0 +- 9.9 | 14.7 +- 7.2 | 27.5 +- 5.4 | 21.4 +- 4.2 |
| b / <b>, f(b) | 0.61, 2.26 | 0.63, 2.10 | 0.3136, 3.70 | 0.3136, 3.63 |
| final partons | 125 +- 60 | 138 +- 54 | 181 +- 34 | 190 +- 33 |
| of which from secondary systems | 92 | 88 | 145 | 137 |
| beam remnants | 10.2 | 9.4 | 13.6 | 12.9 |
| sum pT of the secondary partons [GeV] | 134 | 124 | 212 | 196 |
| hardest secondary interaction [GeV] | 6.8 | 6.5 | 8.6 | 8.4 |
| transverse UE density [GeV per unit eta-phi] | 2.21 +- 1.72 | 2.00 +- 1.59 | 3.48 +- 1.54 | 3.23 +- 1.44 |
| UE scalar pT in R = 0.4 around the b [GeV] | 1.21 +- 2.48 | 0.85 +- 1.31 | 1.80 +- 2.43 | 1.60 +- 2.19 |
| UE scalar pT in R = 0.8 around the b [GeV] | 4.57 +- 5.20 | 3.95 +- 4.05 | 7.32 +- 5.68 | 5.88 +- 4.44 |
| UE partons in R = 0.4 / 0.8 | 0.75 / 2.99 | 0.72 / 2.87 | 1.20 / 4.75 | 1.07 / 4.18 |

The geometry (b, f), the remnant count, the hardest secondary interaction and
the underlying-event density agree at the 5-10 % level; the number of
interactions is 20 % low, mostly the sequential (not interleaved) showering of
the systems described above, partly the 10 % stronger soft radiation of the
chapter-2 shower (6.4 partons per secondary system against PYTHIA's 5.4). The
extra pT inside the b cones follows the density. The cone mass is not compared:
the standalone sample keeps the seed-1 shower (family mass 13.5 GeV), PYTHIA
reshowers (8.1 GeV on average). Colour reconnection changes no momentum at
parton level; its effect shows up in chapter 5. PYTHIA's record of its first
event is in `validate_mpi_with_pythia_event1.txt`.

## Figures

* `../plot_journey_MPI.py` -> `../journey_MPI.pdf`: the stage-3 (eta, phi) map in
  the layout of the CMSSW genPartAnalyzer, with the two new origins (MPI,
  cyan squares; beam remnants, grey diamonds) and the CMSSW classification
  rules (copies and FSR radiators inherit the origin of their mother).
* `plot_theory_MPI.py` -> `theory_multiparton_interactions.pdf`, one page per
  theory section of chapter 4: the regularised cross section and sigma_int(pT0);
  the proton profile, O(b), f(b) and the b distributions; the no-interaction
  probability, the pT ladder against the CMSSW event and the multiplicity
  against Poisson and PYTHIA; (x1, x2) on the densities, processes and the
  depletion of both protons; remnant x shapes, primordial kT and the
  light-cone bookkeeping; the reconnection probability, lambda before/after
  and the strings through the b quark; the UE densities and <n>(pT0); the UE
  pT inside R = 0.4 and 0.8 and the ledger against the cone radius. It reads
  `standalone_mpi_stats_{sampled,fixedb}.stats.json` (written by `--repeat 400`)
  and `validate_mpi_with_pythia_{sampled,fixedb}.json`.
* `animate_standalone_multiparton_interactions.py`: five scenes (overlap,
  chain, remnants, strings, ensemble), same CLI as the chapter-2 film;
  `make figures` writes the stills PDF only.

`make records` regenerates `../event_MPI.lhe`, `make figures` the three PDFs.
