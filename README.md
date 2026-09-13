# JitJet: Journey Is The Jet (from bottom to top)

> A training book for new Master, PhD and postdoc students joining the
> Jet/MET (JME) group of the CMS experiment.

## 1. The idea in one paragraph

In CMS an electron, a photon or a muon is a *destination*: an object that is
essentially there at the end of reconstruction and needs "only" identification
and calibration. A **jet is a journey**. It does not exist at any single point;
it is the accumulated history of a coloured parton from the hard scattering to
the calibrated, tagged, smeared four-vector an analyst finally puts into a
histogram. Every stage of that journey either *adds* energy to the jet, *removes*
energy from it, or *reshuffles* how the energy is distributed. The book follows
one and the same jet through every stage, shows the mathematics of each stage,
measures how much energy that stage added or removed, and points to the
official paper for further reading.

The title and subtitle are meant literally and figuratively:

* **Journey Is The Jet** - the jet is nothing but its journey.
* **from bottom to top** - the journey is told from beginning to end, and the two
  jets we follow are the **b-jet** (bottom) and the **top-jet** (top).

## 2. The physics backbone (the same in every version of the book)

Whatever metaphor is chosen for the chapter names, the underlying chain is fixed.
This is the "spine" every Book*.md must map onto, stage by stage:

| # | Stage | Tool / CMS object | Energy effect on the jet |
|---|-------|-------------------|--------------------------|
| 0 | Proton, PDFs, hard scattering `p p > b b~` | MadGraph5_aMC@NLO (LO / NLO) | defines the "true" parton energy |
| 1 | Matrix-element / parton-shower matching (MLM, FxFx) | MG + Pythia 8 | redistributes energy between jets |
| 2 | Parton shower (ISR, FSR), dead cone for heavy quarks | Pythia 8 | out-of-cone radiation removes energy |
| 3 | Multi-parton interactions, underlying event, colour reconnection | Pythia 8 | adds diffuse energy |
| 4 | Hadronisation (Lund string), b-fragmentation, B-hadron decays, neutrinos | Pythia 8, EvtGen | removes energy (neutrinos, muons) |
| 5 | Generator-level jets (GenJets, with/without neutrinos) | FastJet | the reference "truth" |
| 6 | Pileup mixing (in-time, out-of-time), premixing | CMSSW, MinBias library | adds energy |
| 7 | Detector simulation, digitisation | Geant4, CMSSW | response < 1, non-linearity, noise |
| 8 | Level-1 trigger (calorimeter Layer-1, Layer-2, uGT) | L1T | coarse energy, thresholds, efficiencies |
| 9 | High-Level Trigger (Calo jets, PF jets, PUPPI at HLT) | HLT | online calibration, thresholds |
| 10 | Local reconstruction (RecHits, clusters, tracks, vertices) | CMSSW | ECAL/HCAL calibration, zero suppression |
| 11 | Particle Flow (linking, PF hadron calibration) | PF | restores response towards 1 |
| 12 | Pileup mitigation (CHS, PUPPI, SoftKiller, jet-area rho) | PUPPI | removes energy |
| 13 | Jet clustering (anti-kT R=0.4 and R=0.8, jet area, ghosts) | FastJet | defines what "the jet" is |
| 14 | Jet ID, noise cleaning, jet-veto maps | JME | removes fake jets |
| 15 | JEC L1 (pileup offset) | JME | removes energy |
| 16 | JEC L2L3 (MC-truth, relative eta + absolute pT) | JME | adds energy (response correction) |
| 17 | JEC L2L3Residual (data only: dijet, Z+jet, gamma+jet, multijet) | JME | few-% data/MC correction |
| 18 | Jet energy resolution, JER scale factors, smearing (scaling / stochastic) | JME | widens the distribution |
| 19 | Jet substructure: grooming, trimming, pruning, soft drop, N-subjettiness, Lund plane | JME / substructure | removes energy, sharpens mass |
| 20 | Flavour: b-tagging (secondary vertices, DeepJet, ParticleNet, UParT) | BTV / JME | identifies the b-jet |
| 21 | Boosted top tagging (soft-drop mass, tau32, ParticleNet, HOTVR) | JME | identifies the top-jet |
| 22 | Systematic uncertainties (JES sources, reduced sets, JER, flavour, PU) | JME | error bars |
| 23 | MET (Type-1 correction, PUPPI MET, unclustered energy) | JME | the "missing" part of the journey |
| 24 | Run 2 to Run 3 to HL-LHC / Phase-2 (HGCAL, timing, PF at L1) | CMS | the future of the journey |

## 3. The worked example ("one jet, all the way")

To make the book concrete and reproducible, every chapter revisits the **same
event(s)**:

* **Event A (the b-jet):** `generate p p > b b~` at 13.6 TeV in MadGraph5_aMC@NLO,
  fixed random seed, showered and hadronised in Pythia 8 (CP5 tune), premixed
  with Run 3 pileup, simulated in Geant4, run through L1T and HLT emulation,
  reconstructed with Particle Flow, PUPPI, AK4 clustering, JEC and JER.
* **Event B (the top-jet):** `generate p p > t t~` with a boosted hadronic top
  (pT > 400 GeV), same chain, AK8 clustering, soft drop, N-subjettiness,
  ParticleNet.

For both jets we keep an **energy ledger**: a table with one row per stage
giving the jet pT, mass, constituent multiplicity, and the change with respect
to the previous stage. The ledger is the quantitative red thread of the book.

| Stage | pT [GeV] | m [GeV] | n constituents | delta pT vs previous | Chapter |
|-------|----------|---------|----------------|----------------------|---------|
| parton (MG) | ... | ... | 1 | - | 1 |
| after shower | ... | ... | ... | ... | 2 |
| ... | | | | | |

Each chapter follows the same internal structure:

1. **The stage in the journey** (metaphor + what physically happens)
2. **The mathematics** (the minimal, correct formulae)
3. **Our jet at this stage** (numbers from the ledger, event displays, plots)
4. **How CMS does it in practice** (configs, CMSSW modules, JME tools, NanoAOD branches)
5. **Pitfalls and FAQs for newcomers**
6. **Further reading** (official CMS papers, DP notes, theory papers)

## 4. Candidate metaphors for the chapter names

The physics chain is fixed; the *story* can differ. Three complete tables of
contents are provided, each built on a different real-world metaphor:

| File | Metaphor | One-line pitch |
|------|----------|----------------|
| [Book1.md](Book1.md) | **The Family Pilgrimage** | A mother quark gives birth to children in the shower; the growing family travels through borders, crowds, police checks, barbers and tax offices until they are recognised at the destination. |
| [Book2.md](Book2.md) | **The Expedition: from Base Camp to Summit** | "Bottom to top" taken literally. The b-jet climbs camp by camp; the top-jet is the summit. Each camp costs or gives energy; guides, porters, weather and permits play the roles of the CMS chain. |
| [Book3.md](Book3.md) | **The Energy Economy** | Energy is currency. Mint, inheritance, factories, immigrants, customs, census, police, town hall, tax office with refunds, barber, court of identification, insurance. The ledger is literally a bank passbook. |

Other metaphors considered (can be mixed in as chapter-level flavour):

* **A river from spring to sea**: tributaries (UE, PU) join, dams (thresholds)
  and filters (PUPPI) act, the delta (clustering) collects everything.
* **A postal parcel**: packed (hadronised), stamped (triggered), scanned
  (detector), sorted (PF), re-weighed and charged (JEC), delivered (analysis).
* **A train journey**: stations are stages, passengers board and leave, tickets
  are checked (trigger), the timetable is the CMSSW sequence.
* **A theatre production**: the script (matrix element), the cast grows
  (shower), extras (PU), stage lights (detector), the critics (taggers).

## 5. How to use these files

1. Read the three `Book*.md` files and mark, chapter by chapter, which names and
   framings feel most *apt* (a good metaphor must predict the physics, not just
   decorate it).
2. Merge the winners into a final `TOC.md`.
3. Only then start writing chapters, one directory per chapter
   (`chapters/01_.../`), each with its Markdown text, the CMSSW/MG/Pythia
   configs used for the worked example, and the plots.

## 6. Repository layout (proposed)

```
JitJet/
  README.md          this file: concept, spine, worked example, metaphors
  Book1.md           TOC v1: The Family Pilgrimage
  Book2.md           TOC v2: The Expedition (Base Camp to Summit)
  Book3.md           TOC v3: The Energy Economy
  TOC.md             (later) the merged, final table of contents
  chapters/          (later) one folder per chapter
  example/           (later) MG cards, Pythia fragments, CMSSW configs, ledger
  refs/              (later) bibliography (BibTeX) of all cited papers
```

## 7. Core references (to be cited throughout)

* MadGraph5_aMC@NLO: J. Alwall et al., JHEP 07 (2014) 079.
* Pythia 8.3: C. Bierlich et al., SciPost Phys. Codebases 8 (2022).
* Geant4: S. Agostinelli et al., NIM A 506 (2003) 250.
* FastJet and anti-kT: M. Cacciari, G. Salam, G. Soyez, JHEP 04 (2008) 063; EPJC 72 (2012) 1896.
* Jet areas and rho: M. Cacciari, G. Salam, G. Soyez, JHEP 04 (2008) 005; PLB 659 (2008) 119.
* CMS Particle Flow: CMS Collaboration, JINST 12 (2017) P10003.
* CMS trigger: CMS Collaboration, JINST 12 (2017) P01020; L1 Run 2 upgrade: JINST 15 (2020) P10017.
* CMS JES/JER: CMS Collaboration, JINST 6 (2011) P11002; JINST 12 (2017) P02014.
* CMS pileup mitigation and PUPPI: D. Bertolini et al., JHEP 10 (2014) 059; CMS Collaboration, JINST 15 (2020) P09018.
* Soft drop: A. Larkoski, S. Marzani, G. Soyez, J. Thaler, JHEP 05 (2014) 146; mMDT: M. Dasgupta et al., JHEP 09 (2013) 029.
* Trimming: D. Krohn, J. Thaler, L.-T. Wang, JHEP 02 (2010) 084; Pruning: S. Ellis, C. Vermilion, J. Walsh, PRD 80 (2009) 051501.
* N-subjettiness: J. Thaler, K. Van Tilburg, JHEP 03 (2011) 015.
* Lund jet plane: F. Dreyer, G. Salam, G. Soyez, JHEP 12 (2018) 064.
* CMS b-tagging: CMS Collaboration, JINST 13 (2018) P05011; DeepJet: JINST 15 (2020) P12012.
* ParticleNet: H. Qu, L. Gouskos, PRD 101 (2020) 056019; CMS heavy-object tagging: JINST 15 (2020) P06005.
* CMS jet substructure / boosted top: CMS Collaboration, JINST 15 (2020) P06005; HOTVR: T. Lapsien, R. Kogler, J. Haller, EPJC 76 (2016) 600.
* Dead cone: ALICE Collaboration, Nature 605 (2022) 440.
