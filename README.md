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

## 4. The metaphor: The Family Pilgrimage

Three candidate metaphors were drafted (a family pilgrimage, a mountaineering
expedition, an energy economy). The **Family Pilgrimage** was chosen: a quark is
a *mother*, the parton shower produces her *children*, pileup is the *crowd on
the road*, detector and trigger are *borders and checkpoints*, Particle Flow is
the *census*, CHS/PUPPI are the *police*, clustering is the *inn*, JEC is the
*tax office*, JER is the *wobbly ruler*, b-tagging is the *registry office*.
The other images (barber shop for grooming, shop where we pay more and gain
less) survive at chapter level inside the top-jet part.

**Ordering decision.** The b-jet journey is told first and completely
(chapters 1 to 22, from the hard scattering to b-tagging and MET). Only then do
"three families travel as one": the top-jet gets two chapters (23: the boosted
AK8 jet with grooming, substructure and jet mass scale/resolution; 24: top
tagging). Wrap-up chapters 25 to 27 close the ledger, describe daily life in
JME and look at Run 3 / Phase-2.

## 5. Repository layout

```
JitJet/
  README.md                 this file: concept, spine, worked example
  JitJet.tex                main LaTeX file: title, TOC, \include of every chapter
  preamble.tex              packages, macros (\pt, \antikt, ...), boxes, energy-ledger table
  Makefile                  make -> pdflatex, bibtex, pdflatex x2 -> JitJet.pdf
  chapter/
    chapter_00_frontmatter.tex   preface, how to read, the two pilgrims, the ledger
    chapter_01_ancestral_home.tex ... chapter_27_next_pilgrimage.tex
    appendix_A_kinematics.tex ... appendix_E_exercises.tex
  ref/JitJet.bib            bibliography (INSPIRE-style keys)
  example/                  (later) MG cards, Pythia fragments, CMSSW configs, ledger
  prompt/                   the brainstorming prompts that shaped the book
```

Each chapter file opens with a `journeybox` (the metaphor paragraph), a
`\physicsline` (what physics the chapter covers), then one `\section` per
outline item. The section "Our jet at this stage" carries a `ledger` table; the
section "Further reading" cites the papers in `ref/JitJet.bib`. Every chapter
ends with a `pitfalls` box and an `exercises` list. Red `\todo{}` marks mark
what remains to be written.

## 6. Building the PDF

```bash
make
```

or, equivalently, `latexmk -pdf JitJet.tex`. `make quick` runs a single
pdflatex pass; `make clean` removes auxiliary files.

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
