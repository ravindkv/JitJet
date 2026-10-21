# The family photograph: generator-level jets and their flavour, without FastJet

`standalone_genjets.py` takes the final state of chapter 6 (`../../Decay/event_Decay.lhe`:
the 749 rows with a positive status, CMSSW's genParticles with status 1) and clusters
it the way CMSSW's GenJet producers do, with every piece written out: the generalised-kT
sequential recombination (anti-kT for the GenJets, kT and Cambridge/Aachen as options) in
rapidity and azimuth with the E scheme and the 3 GeV threshold, for the three collections
ak4GenJets (R = 0.4), ak4GenJetsNoNu (R = 0.4 without neutrinos) and ak8GenJets (R = 0.8);
the ghost association of CMSSW's JetFlavourClustering (the last b and c hadrons of the
decay chains, the partons handed to the strings, the hard-process partons and the leptons
from heavy hadrons, with their momentum times 1e-18), giving hadronFlavour, partonFlavour,
a "physics" flavour from the status-23 partons and the legacy dR < 0.3 matching; the net
flavour and its infrared problem; and the two infrared-safe definitions of the literature,
flavour-kT (Banfi, Salam, Zanderighi) and interleaved flavour neutralisation (IFN: Caola,
Grabarczyk, Hutt, Salam, Scyboz, Thaler), the latter replayed on the anti-kT sequence
with the heavy hadrons kept undecayed. The output, `../event_GenJet.lhe`, is the record of
chapter 6 followed by a jet table (one row per jet and collection with its kinematics and
its labels under every definition) and a constituent table (which jet holds each
particle). This is the photograph every reconstructed jet of chapters 15 to 21 will be
compared against.

```bash
python3 standalone_genjets.py                       # -> ../event_GenJet.lhe (the book's record)
python3 standalone_genjets.py -v                    # learning mode: the b jet merge by merge, the flavours, IFN
python3 standalone_genjets.py -vv                   # ... every merge of the ak4 clustering
python3 standalone_genjets.py --algorithm kt        # kT jets (also ca); --ptmin 10 for a higher threshold
python3 standalone_genjets.py --soft-pair 0.5       # the infrared test: a soft B0 Bbar0 pair between two light jets
python3 standalone_genjets.py --repeat 400 --jobs 4 --quiet --history standalone_genjet_stats.json
```

On this Mac use `/usr/bin/python3` (NumPy); one event takes 15 s (2 s per collection, 8 s
of it the flavour-kT clustering), the `--repeat 400` ensemble half an hour with four
workers (it decays the chapter-5 hadrons again with chapter 6's script, seeds 1 to 400,
and clusters each result). The script imports chapter 6's `standalone_hadron_decays.py`
(the record, the particle names and properties) and needs nothing else.

## What the script reproduces, and the numbers

* **The input** (`genParticlesForJets`): every final particle but the ignoreParticleIDs
  (none here); `NoNu` drops the 5 neutrinos as well. No eta cut, no pT cut: 749 (744)
  particles go in, 43 of them beyond |eta| = 6.
* **The algorithm.** d_ij = min(pT_i^2p, pT_j^2p) dR_ij^2 / R^2, d_iB = pT_i^2p, dR in
  rapidity and azimuth, p = -1 for anti-kT; the smallest distance wins, merged pairs add
  their four-momenta, a pseudojet whose d_iB wins is a jet, jets below 3 GeV are dropped.
  The nearest-neighbour bookkeeping of FastJet's N^2 scheme is written out in `Clustering`.
  Our event: 42 ak4 jets holding 368 of the 749 particles (11 above 10 GeV, 6 above 30),
  42 NoNu jets (366 of 744), 30 ak8 jets (600 particles). CMSSW's event (pages 6-8 of
  `../CMSSW/genJetAnalyzer_ME_PS_MPI_Had_Decay_GenJet_out.pdf`): 27, 27 and 28 jets from
  634 particles.
* **The b quark's jet.** The quark (row 719 of the record, the status-62 copy: pT = 78.29
  GeV, eta = -1.38, phi = -2.73) is matched to ak4 jet [0]: 99.83 GeV, 31 particles (17
  charged), mass 14.8 GeV, axis 0.062 from the quark (dR in eta and phi, as CMSSW prints
  it), response pT(jet)/pT(quark) = 1.275. The cone of R = 0.4 around the quark (the
  ledger of chapters 5 and 6) holds 99.43 GeV in 31 particles: two soft particles of the
  cone are not in the jet and two others are, the border effect of an axis 0.06 away. The
  36 merges that build the jet are listed in `standalone_genjets.log`: the hard core first
  (the Bbar0's K- and pi- at dR = 0.006, its photons), the ghosts of the D0, the Bbar0 and
  the b quark between merges 20 and 27, the soft periphery last at dR up to 0.38, then d_iB
  = 1/pT^2 = 1.0e-4 wins. The jet without neutrinos is the same jet (the Bbar0 decayed
  hadronically); the ak8 jet [1] has 105.79 GeV and 47 particles. The jet is 68.1 GeV of
  charged hadrons, 22.8 of photons, 9.0 of neutral hadrons, no lepton, no neutrino.
* **The bbar quark's jet** (row 708, 72.50 GeV at eta = -3.42): ak4 jet [1], 94.09 GeV, 18
  particles, response 1.298. Its ak8 jet [0] (147.69 GeV, 57 particles, 0.36 from the
  quark) also swallows the gluon-splitting pair of the shower, Lambda_bbar0 (row 1005,
  35.5 GeV) and Bbar0 (row 1584, 24.9 GeV), which in ak4 share jet [2] (60.45 GeV).
* **Ghosts and labels.** Four b-hadron ghosts (the Bbar0 of each hard quark, the pair
  from the gluon), nine c-hadron ghosts, 163 parton ghosts (the rows with status 71-79,
  CMSSW's `Pythia8PartonSelector`), the two status-23 partons, three leptons from heavy
  hadrons. The b jet: hadronFlavour 5 (the Bbar0's ghost), partonFlavour +5 (the b quark's
  status-71 ghost; the hardest b parton, else c, else the hardest parton), dR < 0.3
  matching +5, and **physics flavour 0**: the status-23 b quark points 0.56 away from the
  jet, because ISR gave the hard system a recoil and the shower moved the family; the
  physics definition only finds it in the ak8 jet. The bbar jet: 5 / -5 / 0 / -5, and a
  **net flavour of +1 b** from its oscillated B0 (the record keeps the Bbar0 it decayed
  as). Jet [2]: hadronFlavour 5 from two hadrons, net flavour 0 (b + bbar), partonFlavour
  -5. hadronFlavour 5 is given to 3 ak4 jets, 4 to two more (the shower's D+ and D-).
* **IFN** (anti-kT kinematics, alpha = 2, omega = 1, candidates taken among all pseudojets
  alive at the step, final jets included) finds no neutralisation in the event: every
  flavoured hadron's nearest opposite flavour is further in u_ik than the pseudojet it
  joins. **Flavour-kT** (alpha = 2) builds different, kT-like jets, the b jet with 99.83
  GeV in 17 constituents (the heavy hadrons undecayed), the bbar jet with 94.49 GeV, both
  with the flavour of their hadron.
* **The infrared test** (`--soft-pair PT`): a B0 and a Bbar0 of PT each, 0.5 apart in dR,
  placed between the two nearest light jets above 5 GeV (jets [8] and [9] here), one
  hadron inside each. At PT = 0.5 GeV both jets get hadronFlavour 5 and the net flavours
  +1 b / -1 b, while IFN neutralises the pair before the first of them is merged (u_ik
  of the pair against u_ij of hadron and jet) and leaves both jets flavourless; above a few
  GeV the pair is hard enough to keep its flavour under every definition. Page 3 of
  `theory_genjets.pdf` scans PT from 0.2 to 15 GeV.

Left out on purpose: jet areas and the ghosts of the area (chapter 15); the flavour
dressing and CMP variants of infrared-safe flavour; the sign conventions of
partonFlavour for a jet with a b and a bbar parton (the harder one wins, as in CMSSW);
the CMSSW ghost "leptons" are collected but used for no label. The IFN candidate set is
an assumption of this port: the FastJet contrib IFNPlugin is the reference, and the
python wheel of FastJet carries no contrib.

## Validation against FastJet 3 and PYTHIA 8 (`validate_genjets_with_fastjet.py`)

```bash
PY=../../ME/Standalone/.venv/bin/python     # fastjet 3.5.2 (pip install fastjet), pythia8mc 8.315
$PY validate_genjets_with_fastjet.py                               # mode record (also --algorithm kt, ca)
$PY validate_genjets_with_fastjet.py --mode full -n 2000 --standalone-stats standalone_genjet_stats.stats.json
```

Mode `record` (the sharp test): FastJet (the library CMSSW's FastjetJetProducer calls)
and PYTHIA's SlowJet cluster exactly the 749 particles of the record, FastJet with the
same ghosts. For all three collections, and for kT and Cambridge/Aachen as well, the
jets are identical to the standalone ones: 42 / 42 / 30 jets, max |dpT| = 0 and max |dm|
= 0 jet by jet, identical constituent lists for every jet, identical hadron, parton and
physics flavours; SlowJet (no ghosts, `select = 2` for the NoNu collection) agrees to
2e-14 GeV with the same constituents. The FastJet jets with their constituent rows are
in `validate_genjets_with_fastjet_record_jets.txt`, the log in
`validate_genjets_with_fastjet_record.log`.

Mode `full`: the full CP5 chain from the hard event, 2000 events (4 minutes), FastJet
with ghosts, the status-71 b quark as the axis (CMSSW's choice):

| quantity | PYTHIA + FastJet, full chain | standalone, 400 decays of the record |
|---|---|---|
| ak4 jets above 3 / 10 / 30 GeV | 18.2 / 3.9 / 2.4 | 40.0 / 10.2 / 6.0 |
| b: ak4 jet pT, response, dR to the quark | 109.9 +- 27.6, 1.22 +- 0.51 (median 1.06), 0.028 | 99.63 +- 0.49, 1.273 +- 0.006, 0.060 |
| b: ak4 jet without neutrinos, pT | 102.1 +- 28.9 | 91.93 +- 12.12 |
| b: ak8 jet pT | 120.0 +- 24.6 | 104.61 +- 0.80 |
| b: neutrino pT in the ak4 jet | 7.9 +- 14.3 (none in 58% of the events, above 5 GeV in 34%) | 7.7 (none in 54%, above 5 GeV in 37%) |
| b: hadronFlavour 5 / partonFlavour +-5 | 99.2% / 99.3% | 100% / 100% |
| b: B hadron to jet axis, its products outside the jet | 0.025, 0.33 | 0.062, 0.20 |
| ak4 jets with two b hadrons | 0.009 per event | 1.00 per event (the shower's pair) |

The full chain spreads the response because the shower, the MPI and the status-71 quark
itself fluctuate from event to event (a hard final-state gluon leaves a soft last copy),
while the standalone ensemble decays one fixed set of primary hadrons: its spread is the
decays alone. What the two share is the structure: the AK4 jet keeps the quark's family
plus a quarter of its pT from the gluons and the underlying event around it, the jet
without neutrinos loses 8 GeV on average in a tail from the semileptonic decays, the ghost
labels find the b in 99-100% of the jets, and the B hadron sits a few hundredths from
the axis with a fraction of a particle outside the jet.

## The ensemble: 400 decays of the record, clustered (`--repeat 400`)

`standalone_genjet_stats.log` and `.stats.json`, seeds 1 to 400 (28 minutes with four
workers). The same 365 primary hadrons, so the 40.0 +- 2.2 ak4 jets (10.2 above 10 GeV,
always 6 above 30) and the 30.4 ak8 jets change only through the decays. The b quark's
ak4 jet: 99.63 +- 0.49 GeV (the cone of the same R: 98.79 +- 0.68; the jet exceeds the
cone by 0.84 +- 0.59 GeV, the axis being free to follow the family), response 1.273 +-
0.006, 25.7 particles, dR = 0.060 to the quark and 0.062 to the B hadron, 0.20 of the
B's products outside the jet, 78.2 +- 0.5% of the jet pT from the B's descendants;
without neutrinos 91.93 +- 12.12 GeV (none in 54% of the decays, above 5 GeV in 37%);
ak8 104.61 +- 0.80. hadronFlavour 5 and partonFlavour +5 in every decay. The net flavour
of the b jet with the heavy hadrons kept is +1 b in 83.2% and -1 b in 16.8%: the Bbar0
oscillates in a sixth of the decays and the hadron's sign follows; IFN and flavour-kT
give the same split, since nothing soft is there to neutralise (0.66 IFN neutralisations
per event elsewhere, none in 34% of the decays; the gluon-splitting pair shares one ak4
jet in every decay). The bbar quark's jet: 93.86 +- 0.83 GeV, response 1.295, net
flavour -1 b in 79.2% (its B0 oscillates in a fifth).

## Figures

* `../plot_journey_GenJet.py --collection ak4|ak4nonu|ak8` -> `../journey_GenJet.pdf`,
  `../journey_GenJet_nonu.pdf`, `../journey_GenJet_ak8.pdf`: the stage-6 (eta, phi) maps in
  the layout of the CMSSW genJetAnalyzer (pages 6 to 8 of the CMSSW PDF): every jet a
  circle of radius R, the leading eight in colour with their constituents, the b jet red
  and the bbar jet blue, the unclustered particles hollow, the last copy of each quark a
  black star, the quark -> jet lines with dR and response in the header, the leading jets
  with their hadron/parton flavours in the margin.
* `plot_theory_GenJet.py` -> `theory_genjets.pdf`, one page per theory section of chapter
  7: the pT around the quark against R for the cone and the jet with and without
  neutrinos, the clustering ladder of our jet, the threshold and the number of jets; the
  ghosts over the jets, the six labels of every jet above 5 GeV, the B hadron against the
  jet axis over the ensemble; the soft-pair scan, the IFN distance Omega^2 against dR^2,
  flavour-kT against anti-kT; the cone through the stages now with the three jets, what
  the AK4 jet is made of, the response over the ensemble and the PYTHIA chain. It
  re-clusters the record (an R scan, the soft-pair scan) and reads the `--repeat`
  statistics and the validation files when present (about 5 minutes).
* `animate_standalone_genjets.py`: six scenes, narrated (the photograph, anti-kT gathering
  our jet merge by merge, the choices of R, neutrinos and threshold, the ghosts, the soft
  pair and IFN, the ensemble); same CLI as the chapter-6 film, `--scene photograph,...` or
  `1..6`, 1920x1080 by default, `--jobs 4`, `--fast`, `--no-narration`; the clips are cached
  in `build/narration/GenJet/`, an `.srt` is written next to the `.mp4`. The film opens
  where the chapter-6 film ended (the grandchildren are born) and ends at the crowd on the
  road of chapter 8. `make figures` writes the stills PDF only.

`make records` regenerates `../event_GenJet.lhe`, `make figures` the five PDFs.
