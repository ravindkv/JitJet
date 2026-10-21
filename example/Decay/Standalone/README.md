# From primary hadrons to what the detector sees: the decays without PYTHIA

`standalone_hadron_decays.py` takes the hadron-level record of chapter 5
(`../../Had/event_Had.lhe`: the 365 primary hadrons of the Lund string, status
81-84) and decays every unstable one of them, following PYTHIA 8's
`ParticleDecays`: lifetimes and displaced vertices, B0 and B_s0 mixing, the
decay tables (161 particles, 3114 channels, dumped from PYTHIA's `ParticleData`
into `decay_tables.json` by `dump_decay_tables.py`), the M-generator phase
space with the V-A, omega -> 3 pi, polarisation and Dalitz matrix elements, and
the partonic channels (b -> c W, W -> q qbar' plus the spectator, J/psi -> g g g)
turned into hadrons with chapter 5's `StringFlav`. The output,
`../event_Decay.lhe`, is a PYTHIA-style listing (a decayed particle keeps its
row with a negative status, its products follow with status 91, or 92 after an
oscillation) plus a second table with the production vertex and the proper
lifetime of every displaced row.

```bash
python3 standalone_hadron_decays.py                      # seed 1 -> ../event_Decay.lhe (the book's record)
python3 standalone_hadron_decays.py -v                   # learning mode: the two B hadrons' chains step by step
python3 standalone_hadron_decays.py -vv                  # ... every decay
python3 standalone_hadron_decays.py --no-mix             # no B0 / B_s0 oscillations
python3 standalone_hadron_decays.py --no-limit-tau0      # also decay K_S0, Lambda, Sigma, Xi, Omega (PYTHIA's default)
python3 standalone_hadron_decays.py --stable-b           # keep the b hadrons (everything else decays)
python3 standalone_hadron_decays.py --repeat 400 --quiet --history standalone_decay_stats.json
python3 standalone_hadron_decays.py --single 511 --repeat 20000 --history standalone_single_511.json   # a B0 at rest
```

On this Mac use `/usr/bin/python3` (NumPy); one event takes 1.5 s, the
`--repeat 400` ensemble five minutes. The script imports chapter 5's
`standalone_hadronisation.py` (the event record, the flavour selection, the
Lorentz frames) and reads `decay_tables.json` from its own directory; the
table is regenerated with `../../ME/Standalone/.venv/bin/python
dump_decay_tables.py` (the only step that needs PYTHIA).

## What the script reproduces, and the numbers

* **Who decays.** Every final particle with a decay table and c tau0 < tau0Max
  = 10 mm (`ParticleDecays:limitTau0 = on`, what CMSSW sets; PYTHIA's own
  default is off): pi0, eta, eta', the vector and tensor mesons, the Delta and
  Sigma* baryons, the charm and bottom hadrons, the tau. pi+-, K+-, K_S0,
  K_L0, Lambda, Sigma+-, Xi, Omega, n, p, mu, e, gamma and the neutrinos are
  left to the detector. Of the 365 primaries 222 decay and 143 survive.
* **Lifetime and flight.** tau = -tau0 ln r (mm/c; 1 ps = 0.2998 mm/c); the
  decay vertex is x_prod + tau p/m. The B hadron of our b quark (E = 165 GeV,
  gamma = 31) walks 14.5 +- 13.6 mm on average, 6.9 mm in the transverse
  plane; the bbar's B0 at eta = -3.4 (E = 1264 GeV, gamma = 240) walks 117 mm
  along the beam and the same 7.6 mm transversely.
* **Mixing.** P(oscillated) = sin^2(x tau / 2 tau0), x_d = 0.776, x_s = 26.05;
  18.8 % of the B0 decay as a Bbar0 (the script flips the id in the record and
  marks the products with status 92).
* **The channel** is drawn in proportion to its branching ratio among the open
  ones (`onMode` 1, or 2 for the particle and 3 for the antiparticle only: the
  two 0.316 partonic channels of the B0 table are such a pair, so that the
  branching ratios of each B sum to one). Broad products (rho, K*, a_1, D*_0
  ...) get a Breit-Wigner mass within [mMin, mMax]; a channel whose products
  do not fit is drawn again.
* **The kinematics.** Two bodies isotropic; three and more with the
  Raubold-Lynch M-generator (accept/reject on the product of two-body momenta
  against its maximum); PYTHIA's matrix elements by `meMode`: the V-A weight
  m_B E_1 (p_2 . p_rest) for semileptonic decays, with the first product of the
  channel contracted with the mother (the neutrino for b -> c l nu, the charged
  lepton for c -> s l nu: the ordering in the table encodes the pairing) and
  the soft damping exp(-p^2/0.5^2) of hadrons beyond the third; |p_1 x p_2|^2
  for omega, phi -> 3 pi; cos^2 theta for a vector meson from a pseudoscalar
  decaying to two pseudoscalars; the Kroll-Wada lepton pair with the rho form
  factor in Dalitz decays.
* **Partonic channels.** Half of the B0 decays (50 %) and 43 % of the B+ ones
  are written at quark level. The partons are paired into one or two colour
  singlets (with probability colRearrange = 0.5 the other allowed pairing);
  in the weak channels (meMode 22, 23) the spectator pair collapses to one
  hadron and only the W pair makes a variable number; the total is drawn from
  a Poissonian with mean (n_known + n_spec)/2 + n_partons/4 + multIncrease
  ln(mDiff/0.7 GeV), multIncrease = 4 (2.5 for meMode 23), at least 2 (3 for
  meMode 23 and 43, fixed for 62-70) and below 10; each singlet becomes a
  chain of hadrons with the flavour machinery of chapter 5 (new flavours,
  vector/pseudoscalar, SU(6) baryons), and the products go into phase space.

Left out on purpose: QED radiation in decays (CMSSW has it on), the
polarised tau decays (phase space here), the parton shower before the
hadronisation of J/psi -> g g g, the mass-dependent widths of meMode 3-7,
EvtGen (this CMSSW sample used PYTHIA's own tables, see the chain below).

## The record (seed 1)

365 primary hadrons -> 376 decays -> 749 final particles (CMSSW's event: 634),
338 charged, 327 photons, 5 neutrinos with 229 GeV (a D- and a D+ from the
shower's charm, and the Lambda_b of the extra b pair; missing pT 9.1 GeV),
7 electrons, 2 muons; 143 primary hadrons survive, 606 particles come from
decays; 396 final particles (176 charged) at |eta| < 2.5. The b quark's
B*bar0 (row 1012) radiates its photon at the origin; the Bbar0 (row 1527)
draws tau = 2.24 ps = 1.46 tau0, flies L = 21.0 mm (L_xy = 9.9) to
(-9.1, -3.9, -18.5) mm and decays through the partonic channel b -> c W,
W -> c sbar (meMode 43, 7.4 %) into D*_s- D*0 rho+; the D0 flies another
11.0 mm; 15 final particles, 8 charged tracks all with |d0| > 100 um (the
largest 1.6 mm), no neutrino. The bbar's B0 (row 1590, gamma = 239)
oscillates into a Bbar0 after 2.05 ps and decays to Xi_c0 pi+ Deltabar-
147 mm down the beam pipe. Cones around the b quark: R = 0.4 holds 99.23 GeV
of primary hadrons (6) and 99.43 GeV of final particles (31, 16 charged
with 67.4 GeV), all of it visible; R = 0.8: 105.97 -> 108.73 GeV, 51
particles. The `-v` log is `standalone_hadron_decays.log` (3200 lines: the
set-up, the two B chains decay by decay with every draw, one line per other
decay, the checks, the record with its vertex table, and the summary);
chapter 6 quotes its numbers. Momentum is conserved to 1e-8 GeV.

## Validation against PYTHIA 8 (`validate_decays_with_pythia.py`)

```bash
PY=../../ME/Standalone/.venv/bin/python
$PY validate_decays_with_pythia.py -n 400 --standalone-stats standalone_decay_stats.stats.json
$PY validate_decays_with_pythia.py --mode single --id 511 -n 20000 --seed 3 --standalone-stats standalone_single_511.json
$PY validate_decays_with_pythia.py --mode full -n 2000
```

Mode `record` (the sharp test): PYTHIA 8.315 decays exactly the 365 primary
hadrons of `event_Had.lhe` 400 times (`ProcessLevel:all = off`, limitTau0
on); the standalone script decays them 400 times with `--repeat`. Mode
`single`: a B0 at rest 20000 times in both. Mode `full`: the full CP5 chain
from the hard event with decays on, as in chapters 4 and 5 (2000 events, b
drawn).

| quantity | PYTHIA, same hadrons | standalone, same hadrons | PYTHIA, full chain |
|---|---|---|---|
| final particles | 739.2 +- 11.9 | 740.3 +- 11.7 | 459 +- 175 |
| charged / photons / neutrinos | 331.1 / 328.8 / 2.85 | 331.2 / 330.0 / 3.10 | 210 / 211 / 1.3 |
| primary hadrons that survive | 143 | 143 | 86 |
| b: tau [ps], L [mm], L_xy [mm] | 1.41, 13.2, 6.2 | 1.55, 14.5, 6.8 | 1.59, 18.7, 7.9 |
| b: final particles from the B (charged) | 11.1 (5.0) | 11.1 (4.9) | 11.0 (4.8) |
| b: semileptonic e / mu / tau [%] | 9.5 / 10.0 / 4.0 | 13.0 / 11.0 / 1.2 | 10.9 / 11.4 / 3.4 |
| b: E_nu from the B [GeV], share of E_B | 13.8, 0.084 | 16.4, 0.090 | 18.6, 0.091 |
| b: tracks with d0 > 100 um, median d0 [um] | 3.0, 330 | 3.0, 362 | 3.0, 358 |
| b, R = 0.4: primaries -> final -> visible pT [GeV] | 99.23 -> 98.79 -> 92.3 +- 11.4 | 99.23 -> 98.79 -> 91.1 +- 12.2 | 109.2 -> 109.3 -> 101.4 |
| b, R = 0.4: charged pT, B share | 51.3, 0.788 | 49.9, 0.788 | 60.6, 0.799 |
| bbar: L [mm], L_xy [mm] | 119, 7.7 | 117, 7.6 | 152, 7.4 |
| bbar, R = 0.4: final -> visible pT | 92.39 -> 85.7 +- 11.3 | 92.42 -> 84.7 +- 12.7 | 106.3 -> 99.1 |
| B0 at rest: final / charged / photons | 10.87 / 4.85 / 4.50 | 11.11 / 4.90 / 4.77 | |
| B0 at rest: semileptonic (e + mu) [%], <E_l> [GeV] | 20.7, 1.383 | 20.5, 1.396 | |
| B0 at rest: charm hadron D- / Dbar0 / D0 / D+ [%] | 35 / 27 / 11 / 10 | 30 / 28 / 12 / 9 | |

Everything the detector will see agrees within the statistics of 400
decays: the multiplicities to 0.2 %, the flight distances, the neutrino
energy, the impact parameters, the cone momenta to 0.1 GeV (final) and 1 GeV
(visible). The lepton energy spectrum of the V-A matrix element agrees bin
by bin. The one bookkeeping difference: PYTHIA marks an oscillated B0 with
status 92 in only two thirds of the cases where its charm hadron has the
wrong sign (13 % of the B0 at rest, against the 19 % of sin^2(x tau/2 tau0)
that both codes use); the script marks all of them. The full chain, with the
b hadron species mixed (B+ 44 %, B0 43 %, B_s0 10 %, Lambda_b 4 %), gives the
same semileptonic fractions and neutrino shares. CMSSW's own event
(`../CMSSW/genPartAnalyzer_ME_PS_MPI_Had_Decay_out.pdf`, page 5, and the GEN
log of chapter 2) decayed its B- semileptonically, B- -> nu_ebar e- D*_00
with a 37.5 GeV neutrino and a 5.0 GeV electron, 7.8 mm from the primary
vertex; its B*+ -> B+ gamma, B+ -> D*bar0 pi+ pi0 K0 K*bar0 117 mm down the
beam. PYTHIA's first decay of our hadrons is in
`validate_decays_with_pythia_record_event1.txt`, the first full-chain event
in `..._full_event1.txt`, both in the standalone format with the vertex table.

## Figures

* `../plot_journey_Decay.py` -> `../journey_Decay.pdf`: the stage-5 (eta, phi)
  map in the layout of the CMSSW genPartAnalyzer (page 5 of the CMSSW PDF):
  the b and bbar systems (every final descendant of the two B* mesons) in
  red and blue, the stable primary hadrons in purple, the other decay
  products in grey, neutrinos hollow; the header adds the flight distance of
  each B and the missing pT.
* `plot_theory_Decay.py` -> `theory_hadron_decays.pdf`, one page per theory
  section of chapter 6: the decay law, the flight distance and the c tau0
  ladder; the vertices of our event with the tracks and the impact-parameter
  distribution; the classes of the decay tables, the decay chains of our
  event and the multiplicity per B; the V-A lepton spectrum against phase
  space and PYTHIA, the neutrino share, the lepton pT in the jet; the cone
  through the stages, the response after the neutrinos, multiplicity and
  mass. It replays the seed-1 decays and reads the `--repeat` statistics and
  the validation files when present.
* `animate_standalone_hadron_decays.py`: seven scenes, narrated, about 7 min
  (the exponential clock, the walk of the B in the transverse plane and along
  the beam, the roll of the dice on the decay table, the cascade generation by
  generation, a semileptonic decay from the ensemble with its neutrino, the
  whole event decaying in laboratory time, the ensemble); same CLI as the
  chapter-5 film, `--scene clock,walk,...` or `1..7`, 1920x1080 by default,
  `--jobs 4` parallel rendering, `--fast` for a preview, `--no-narration` for
  the 72 s silent cut. Each phase of a scene has one line of narration
  (`--print-script`), spoken by edge-tts (`en-GB-RyanNeural`; `--engine say`
  offline) over the score of `example/soundtrack.py`; the clips are cached in
  `build/narration/Decay/`, an `.srt` is written next to the `.mp4`. The film
  opens where the chapter-5 film ended, "the children are married", and ends
  at the door of the inn of chapter 15, where the anti-kT algorithm gathers the
  grandchildren. `make figures` writes the stills PDF only.
* `sketch_ordering_trap.py` -> `../../../chapter/tikz/decay_ordering.tex`: the
  ordering trap of the V-A weight. In the rest frame of the hadron, the
  charged-lepton energy of D+ -> K0bar e+ nu and of B0bar -> D*+ l- nubar for
  flat phase space, for the table's pairing (the first product contracted with
  the mother) and for the swapped pairing: the mother's partner is the soft
  one, so the lepton is hard from a B (1.40 GeV against 1.27 swapped) and soft
  from a D (0.50 against 0.57). `--pythia` (the venv of `example/ME/Standalone`)
  decays both channels in PYTHIA at rest in either order and prints the same
  means (1.404/1.265 and 0.499/0.567).

`make records` regenerates `../event_Decay.lhe`, `make figures` the three PDFs.
