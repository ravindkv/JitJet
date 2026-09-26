# Showering the b bbar event without PYTHIA

`standalone_parton_showering.py` takes the hard process of chapter 1
(`../../ME/event_ME.lhe`, q qbar -> b bbar at pTHat = 113.2 GeV) and dresses it
with an interleaved initial- and final-state parton shower written from
scratch in Python, following the construction of PYTHIA 8's
`SimpleTimeShower` and `SimpleSpaceShower`. The output,
`../event_PS.lhe`, is a PYTHIA-style event listing (status codes,
mother/daughter links, colour tags) that later scripts of chapter 2 read.

```bash
python3 standalone_parton_showering.py                  # -> ../event_PS.lhe, seed 1
python3 standalone_parton_showering.py --seed 7 --history shower_history.json
python3 standalone_parton_showering.py --no-isr         # final-state radiation only
python3 standalone_parton_showering.py --check          # assert E/p, masses, colour lines after every branching
python3 standalone_parton_showering.py --repeat 2000 --quiet --history stats.json
```

On this Mac use `/usr/bin/python3` (NumPy + SciPy); one shower takes about
0.3 s, the start-up (parsing the embedded PDF grid) about 0.8 s.

## What the script does

* **Final-state radiation.** Every final-state coloured parton radiates
  against its colour partner (a dipole end; a gluon has two). The evolution
  variable is `pT_evol^2 = z(1-z)(m^2 - m_rad^2)` with `m` the virtuality of
  the branching parton and `z` the energy fraction kept by it in the dipole
  rest frame; the Sudakov form factor is sampled with the veto algorithm
  (overestimate kernels 2C_F/(1-z), C_A/(1-z) per gluon end, T_R n_f flat for
  g -> q qbar shared between the two ends). The b quark keeps its mass in the
  kinematics and radiates with the quasi-collinear kernel
  `C_F [(1+z^2)/(1-z) - 2 m_b^2 z(1-z)/pT_evol^2]`: this is the dead cone. The
  recoiler takes the longitudinal recoil (a final-state partner) or increases
  its x (an incoming partner, status -53); in the latter case the emission is
  weighted by the PDF ratio at the new x and by PYTHIA's `dampenBeamRecoil`
  factor.
* **Initial-state radiation.** Backward evolution of both incoming partons,
  `dP = alpha_s/2pi dpT^2/(pT^2 + pT0^2) dz P(z) x'f_b(x', pT^2) / x f_a(x, pT^2)`
  with `pT0 = 2 GeV`, the four kernels q <- q, q <- g, g <- g, g <- q, and
  `pT_evol^2 = (1-z) Q^2`. The new incoming parton has `x' = x/z`; the emitted
  parton and the whole downstream system are placed with PYTHIA's frame
  construction (old a+r rest frame -> new a'+r rest frame), so the other
  incoming parton is untouched.
* **Interleaving.** All dipole ends and both beams propose their next emission
  from the current scale; the hardest wins; everything restarts from there.
  Starting scale `sqrt(pTHat^2 + mb^2) = 113.32 GeV` (PYTHIA's factorisation
  scale for a 2 -> 2 QCD process, "Max pT scale for ISR/FSR" in the CMSSW log),
  cut-offs `pTmin = max(0.5, 1.6 Lambda_3) = 0.611 GeV` for FSR (the "pTmin
  too low, raised to 0.611" of the log) and `max(0.2, 1.1 Lambda_3)` for ISR.
* **alpha_s.** PYTHIA's second-order running with `alpha_s(MZ) = 0.118`
  (CP5) and flavour thresholds at 1.5, 4.8 and 171 GeV, reproduced
  analytically: Lambda_3,4,5 = 0.3820, 0.3282, 0.2262 GeV agree with
  `AlphaStrong` to six digits.
* **PDF.** NNPDF31_nnlo_as_0118 member 0 on LHAPDF's own grid knots, embedded
  between the `# BEGIN/END GENERATED TABLES` markers (bicubic in ln x, ln Q^2,
  frozen below Q = 1.65 GeV). Regenerate with
  `../../ME/Standalone/.venv/bin/python dump_shower_tables.py`.

Left out on purpose: multiparton interactions and beam remnants (chapter 5),
hadronisation (chapter 3), the rapidity-ordering option of ISR, QED
radiation, matrix-element corrections and spin correlations.

## Reading the output

`../event_PS.lhe` keeps the seven rows of the hard process untouched and
appends the shower: status 43 (emitted by ISR), -41 (new incoming parton),
44 (final-state parton shifted by an ISR recoil), 51 (the two daughters of an
FSR branching), 52 (final-state recoiler), -53 (incoming recoiler). Negative
status means "replaced by its daughters"; the final state is every row with a
positive status, and its momentum sum equals the momentum of the two current
incoming partons. The `--history` JSON lists every branching in pT order with
its type, z, virtuality and x, and a summary of the "family" of each hard b
quark (final partons within dR < 0.4).

## Validation against PYTHIA 8

`validate_shower_with_pythia.py` writes the hard event 2000 times into a Les
Houches file and showers it with PYTHIA 8.315 (`pythia8mc` in the
`example/ME/Standalone/.venv` environment, PDFs from the LHAPDF grid file
through PYTHIA's `LHAGrid1` reader) with MPI, hadronisation and QED switched
off and the CP5 shower settings, then averages the same quantities as
`--repeat`:

```bash
../../ME/Standalone/.venv/bin/python validate_shower_with_pythia.py -n 2000 \
    --standalone-stats stats.stats.json
../../ME/Standalone/.venv/bin/python validate_shower_with_pythia.py -n 2000 --rapidity-order
```

Averages over 2000 showers of the same hard event (seed 1..2000 for the
standalone script; the standard deviation of a single shower is quoted, the
error of a mean is 45 times smaller). "Family" means the final partons within
dR < 0.4 of the b quark after the shower.

| quantity | PYTHIA 8.315 | standalone |
|---|---|---|
| ISR branchings | 5.05 +- 2.92 | 4.88 +- 2.91 |
| FSR branchings | 19.5 +- 11.2 | 20.9 +- 11.5 |
| final-state partons (gluons) | 26.5 (21.3) | 27.7 (22.0) |
| final partons with pT > 20 / 5 / 1 GeV | 2.86 / 7.12 / 18.8 | 2.79 / 7.25 / 19.5 |
| x of incoming ubar (+z) / u (-z) after ISR | 0.051 / 0.450 | 0.042 / 0.446 |
| bbar: pT after / family pT / family n / family mass [GeV] | 91.6 / 104.8 / 2.40 / 8.04 | 90.1 / 103.4 / 2.65 / 7.97 |
| b: pT after / family pT / family n / family mass [GeV] | 94.8 / 108.4 / 2.46 / 8.10 | 93.9 / 108.4 / 2.72 / 8.15 |

The hard structure of the event is reproduced: partons above 20 GeV, the pT
and mass of each b family, the number of ISR branchings and the x of the
incoming partons after ISR agree within 1-3 percent. The standalone shower
makes about 10 percent more soft partons (1-5 GeV). The same comparison with
ISR switched off on both sides (`--no-isr` for the script,
`--no-isr` for the validation, i.e. `PartonLevel:ISR = off`) isolates the
radiation off the two b quarks:

| quantity | PYTHIA | standalone |
|---|---|---|
| FSR branchings | 6.74 +- 4.65 | 7.66 +- 4.92 |
| final partons with pT > 20 GeV | 2.48 +- 0.72 | 2.49 +- 0.70 |
| final partons with pT > 5 GeV | 4.49 +- 2.11 | 4.84 +- 2.19 |
| final partons with pT > 1 GeV | 7.39 +- 3.76 | 8.14 +- 3.99 |
| bbar: pT after / family pT / family mass [GeV] | 93.7 / 105.2 / 7.56 | 90.9 / 104.6 / 7.78 |
| b: pT after / family pT / family mass [GeV] | 93.8 / 105.9 / 7.54 | 91.8 / 105.1 / 7.74 |

Raising the FSR cut-off to 2 GeV on both sides (`--ptmin-fsr 2`) leaves a
7 percent excess (2.16 vs 2.01 branchings), so the residual is a small
normalisation difference of the soft-emission rate, largest in the last
octave above the cut-off. Switching off the beam-recoil damping
(`--no-dampen`) more than doubles the FSR multiplicity, which shows how much
of PYTHIA's soft radiation off partons colour-connected to the beams is
governed by that one factor. With PYTHIA's default `SpaceShower:rapidityOrder
= on` (not implemented here) PYTHIA itself makes 3.4 instead of 5.1 ISR
branchings and 16.3 instead of 19.5 FSR branchings, with the b families
unchanged within 1 percent: the ordering choice matters more for the soft
activity than the difference between the two implementations.

PYTHIA's record of its first showered event is in
`validate_shower_with_pythia_event1.txt`, in the same column format as
`../event_PS.lhe`.
