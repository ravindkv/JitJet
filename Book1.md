# JitJet - Table of Contents, version 1

## Metaphor: The Family Pilgrimage

**The story.** A single quark is a *mother*. In the parton shower she gives
birth to children (gluons, quark pairs); in hadronisation the children are
dressed as hadrons and set out together on a pilgrimage. On the road strangers
join the family (underlying event, pileup), some children run away (out-of-cone
radiation, neutrinos), the family crosses borders and checkpoints (trigger),
is counted by census takers (Particle Flow), screened by police who separate
the true family from hangers-on (CHS, PUPPI), gathered under one roof by the
innkeeper (clustering), visits the barber (grooming), settles accounts with the
tax office (JEC), has its measurements taken with a wobbly ruler (JER), and is
finally recognised at the destination by the registry office (b-tagging,
top-tagging).

**Why this metaphor is apt.** It reproduces the genealogy of the jet (mother,
daughters, granddaughters = the shower history), the notion of *who belongs*
(flavour, ghost association, pileup), and *who is missing* (neutrinos, out-of-
cone). The b-jet is the pilgrimage of one family; the top-jet is the
pilgrimage of three families travelling together (t -> W b -> q q' b).

---

## Front matter

* Preface: why a jet is a journey and not a destination
* How to read this book (the fixed six-part chapter structure, the energy ledger)
* Notation and conventions (pT, eta, phi, y, m, R, Delta R, rho, A, mu, NPV)
* The two pilgrims we will follow: Event A (`p p > b b~`) and Event B (`p p > t t~`)
* Software versions, seeds and how to reproduce every plot

---

## Part I - Before the Journey: The Family Is Born

### Chapter 1 - The Ancestral Home: Protons, PDFs and the Hard Scattering
*Physics: factorisation, PDFs, LO/NLO matrix elements, `p p > b b~` in MadGraph5_aMC@NLO*
1. Two protons meet: the factorisation theorem in one page
2. Parton distribution functions: where the mother comes from
3. The matrix element for b b~ production (gg and qq~ channels, 4FS vs 5FS)
4. Running MadGraph: proc card, run card, seeds, LHE format
5. Our jet at this stage: the b quark four-vector in the LHE file
6. Scale choices and their uncertainties (mu_R, mu_F, PDF sets)
7. Further reading: MG5_aMC@NLO, LHAPDF, PDF4LHC

### Chapter 2 - The Mother Gives Birth: Parton Showering
*Physics: DGLAP splitting functions, Sudakov form factor, angular ordering, ISR/FSR, dead cone*
1. Why a quark cannot travel alone: collinear and soft singularities
2. The splitting functions P(z) and the Sudakov form factor
3. Pythia 8's pT-ordered dipole shower; the CP5 tune
4. The heavy mother: the dead cone for b (and top) quarks
5. Children who run away: out-of-cone radiation and the first energy loss
6. Our jet at this stage: multiplicity, pT and mass of the parton-level "family"
7. Further reading: Pythia 8.3 manual, dead-cone measurement (ALICE)

### Chapter 3 - Whose Child Is It? Matching and Merging
*Physics: MLM, CKKW-L, FxFx, double counting, merging scale*
1. Matrix element or shower: who owns the extra emission?
2. MLM matching and the `xqcut` / `qcut` parameters
3. FxFx merging at NLO
4. Our jet at this stage: the effect of merging on the b-jet spectrum
5. Further reading

### Chapter 4 - Dressing the Children: Hadronisation and Heavy-Flavour Decays
*Physics: Lund string model, b-fragmentation (Peterson, Lund-Bowler), B-hadron decays, semileptonic losses*
1. From partons to hadrons: the Lund string
2. b-fragmentation: how much of the mother's energy the B hadron keeps (x_B)
3. B-hadron lifetimes and displaced vertices: the family's birthmark
4. Semileptonic decays: neutrinos and soft muons leave the family
5. Our jet at this stage: GenJet with and without neutrinos
6. Further reading: Lund model, LEP/SLD x_B measurements, EvtGen

### Chapter 5 - Cousins Nobody Invited: Underlying Event and Colour Reconnection
*Physics: MPI, beam remnants, colour reconnection, UE tunes*
1. The rest of the protons does not stay quiet: multi-parton interactions
2. Colour reconnection and its effect on jet mass and top mass
3. Measuring the UE: transverse-region observables
4. Our jet at this stage: UE energy inside R=0.4 and R=0.8
5. Further reading: CP5 tune paper, UE measurements

### Chapter 6 - The Family Photograph: Generator-Level Jets and Truth Definitions
*Physics: GenJets, jet flavour definitions (ghost association, hadron/parton flavour, flavour-kT, IFN), truth for JEC*
1. What we mean by "the true jet": choices and their consequences
2. Ghost-associated hadron flavour vs parton flavour
3. IR-safe flavour: flavour-kT and interleaved flavour neutralisation
4. Our jet at this stage: the reference GenJet we will compare to forever
5. Further reading

---

## Part II - On the Road: Crowds, Borders and Checkpoints

### Chapter 7 - The Crowd on the Road: Pileup
*Physics: in-time and out-of-time PU, mu, NPV, rho, premixing, PU profiles, PU reweighting*
1. Forty families at once: what pileup is and where it comes from
2. In-time, out-of-time and the 25 ns bunch structure
3. How CMS mixes pileup into simulation: the premixing library
4. rho, the median energy density, and the jet area A
5. Our jet at this stage: how much PU energy fell inside our jet
6. Further reading: CMS PU mitigation paper

### Chapter 8 - Crossing the Border: Detector Simulation and Digitisation
*Physics: Geant4, tracker, ECAL, HCAL (HB/HE/HF), non-compensation e/h, sampling fluctuations, digitisation, noise, zero suppression*
1. What a hadron does inside 1.5 m of lead tungstate and brass
2. The calorimeter response: e/h, non-linearity and the origin of response < 1
3. From energy deposits to ADC counts: digitisation, pedestals, noise
4. Our jet at this stage: SimHits and Digis of our b-jet
5. Further reading: Geant4, CMS detector paper, HCAL performance

### Chapter 9 - The First Checkpoint: Level-1 Trigger
*Physics: calorimeter trigger towers, Layer-1 and Layer-2, jet finding on 9x9 towers, PU subtraction at L1, uGT menu, L1 efficiencies*
1. Forty million times per second: the L1 problem
2. Trigger towers and the L1 jet algorithm
3. Level-1 pileup subtraction and calibration (LUTs)
4. Efficiency turn-on curves: the mathematics of a threshold
5. Our jet at this stage: did our event pass? which L1 seed?
6. Further reading: L1 trigger papers, Phase-2 L1 TDR

### Chapter 10 - The Second Checkpoint: High-Level Trigger
*Physics: HLT paths, Calo jets, PF jets, PUPPI at HLT (Run 3), online JEC, prescales, trigger efficiency measurement (tag-and-probe, orthogonal datasets)*
1. From 100 kHz to 1 kHz: HLT design
2. Calo-jet and PF-jet reconstruction online; PUPPI online in Run 3
3. Online jet energy corrections and why they differ from offline
4. Measuring trigger efficiency for a jet analysis
5. Our jet at this stage: HLT objects and their pT
6. Further reading: CMS trigger paper, Run 3 HLT DP notes

---

## Part III - Taking the Census: Reconstruction

### Chapter 11 - Footprints and Fingerprints: Local Reconstruction
*Physics: RecHits, ECAL clustering, HCAL RecHits (method 2, MAHI), tracks (iterative tracking), primary vertices (DA clustering), b-tagging ingredients (impact parameter)*
1. From Digis to RecHits: calibration, timing, pulse-shape fitting
2. Tracks: iterative tracking and the tracker's role for jets
3. Primary vertices and the choice of the hard-scatter vertex
4. Our jet at this stage: RecHits, tracks and the displaced vertex of our B hadron
5. Further reading: CMS tracking, ECAL and HCAL reconstruction papers

### Chapter 12 - The Census Takers: Particle Flow
*Physics: PF blocks, linking, charged hadrons, photons, neutral hadrons, PF hadron calibration, energy composition of jets*
1. Counting everyone once and only once: the PF philosophy
2. Linking tracks and clusters; the PF hadron calibration a(E), b(E)
3. The jet energy composition: charged hadrons, photons, neutral hadrons, leptons
4. Why PF restores the response towards one
5. Our jet at this stage: the list of PF candidates of our jet
6. Further reading: CMS PF paper

---

## Part IV - Who Belongs to the Family? Pileup Mitigation and Clustering

### Chapter 13 - The Police: Charged-Hadron Subtraction and PUPPI
*Physics: CHS, PUPPI alpha metric, per-particle weights, SoftKiller, area-median rho subtraction, constituent subtraction*
1. Papers, please: charged particles and vertex association (CHS)
2. Neutral particles have no passport: the PUPPI alpha metric and its weights
3. rho x A subtraction, SoftKiller, constituent subtraction: alternatives
4. PUPPI tuning in CMS (Run 2 vs Run 3)
5. Our jet at this stage: the weight of every constituent, before and after
6. Further reading: PUPPI (Bertolini et al.), CMS PU mitigation paper

### Chapter 14 - The Inn Where Everyone Gathers: Jet Clustering
*Physics: IRC safety, kT, Cambridge/Aachen, anti-kT, R choice, jet areas via ghosts, AK4 vs AK8, CA15, HOTVR, recombination schemes*
1. What makes a jet algorithm good: infrared and collinear safety
2. The sequential recombination family: kT, C/A, anti-kT
3. Choosing R: 0.4 for the b-jet, 0.8 for the top-jet (pT > 2m/R)
4. Jet areas, ghosts and the geometry of the inn
5. Our jet at this stage: the clustering sequence of our AK4 and AK8 jets
6. Further reading: anti-kT paper, FastJet manual

### Chapter 15 - Impostors at the Inn: Jet ID, Noise and Veto Maps
*Physics: PF jet ID, noise jets (HCAL, ECAL spikes), beam halo, hot/cold towers, jet veto maps, lepton overlap removal*
1. Jets that were never families: detector noise and beam backgrounds
2. Jet ID variables and working points
3. Jet veto maps and their use in analyses
4. Our jet at this stage: passing jet ID
5. Further reading

---

## Part V - Settling the Accounts: Jet Energy Corrections

### Chapter 16 - Paying for the Uninvited: L1 Pileup Offset Correction
*Physics: L1FastJet, hybrid L1 (rho, A, eta, pT), Random Cone method, offset per PU vertex, L1 for CHS vs PUPPI*
1. The factorised JEC chain: an overview of L1, L2, L3, L2L3Residual
2. The L1 offset: rho x A and the hybrid parametrisation
3. Measuring the offset: Random Cone and zero-bias data
4. Our jet at this stage: how much pT the L1 correction removes
5. Further reading: CMS JES papers

### Chapter 17 - The Fair Price: MC-Truth Corrections (L2Relative, L3Absolute)
*Physics: response R = pT_reco/pT_gen, matching Delta R < R/2, response fitting per eta/pT bin, closure, flavour dependence*
1. Comparing to the family photograph: reco-to-gen matching
2. The response distribution and its mean, median and mode
3. Deriving the correction as a function of pT and eta
4. Closure tests and non-closure
5. Flavour dependence: gluon, light quark, b-quark responses
6. Our jet at this stage: the L2L3 factor for our b-jet
7. Further reading

### Chapter 18 - The Local Tax: Residual Corrections on Data
*Physics: dijet pT-balance and MPF, FSR/ISR extrapolation, Z+jet, gamma+jet, multijet, global fit, time dependence, L2Res eta dependence, L3Res pT dependence*
1. Data is not simulation: why residuals are needed
2. Relative eta corrections from dijet balance (pT-balance and MPF)
3. Absolute scale from Z(mumu, ee)+jet, gamma+jet and multijet
4. The global fit and its correlations
5. Time-dependent corrections and IOVs
6. Our jet at this stage: only if it were data (and how to apply it to data)
7. Further reading: CMS JES papers, JME twiki

### Chapter 19 - The Wobbly Ruler: Jet Energy Resolution
*Physics: JER parametrisation (N, S, C), dijet asymmetry, gamma+jet, JER scale factors, smearing: scaling and stochastic methods, matching requirement*
1. Resolution vs scale: two different questions
2. Noise, stochastic and constant terms
3. Measuring JER in data: dijet asymmetry and the generator-level truth
4. Data/MC scale factors and how to smear simulation
5. Our jet at this stage: smeared pT and the random seed problem
6. Further reading

### Chapter 20 - The Receipt: JES and JER Uncertainties
*Physics: uncertainty sources, correlations across years, reduced sets, flavour uncertainties, PU uncertainties, how to propagate in an analysis, Combine nuisance parameters*
1. Reading a JEC uncertainty file
2. The full source list and the reduced sets
3. Correlations between years and between experiments (ATLAS/CMS)
4. Our jet at this stage: the uncertainty band on our b-jet pT
5. Further reading

---

## Part VI - Inside the Family: Substructure

### Chapter 21 - The Barber Shop: Grooming, Trimming, Pruning and Soft Drop
*Physics: mMDT / soft drop (z_cut, beta), trimming (r_sub, f_cut), pruning, groomed mass, mass drop, recursive soft drop, soft-drop constituent multiplicity*
1. Why the raw mass of a fat jet is dominated by noise
2. Soft drop: the algorithm, the parameters and the analytic mass distribution
3. Trimming and pruning: older haircuts still in use
4. Our top-jet at this stage: mass before and after grooming
5. Further reading: soft-drop, mMDT, trimming, pruning papers

### Chapter 22 - Counting the Heads: N-subjettiness, Energy Correlators and the Lund Plane
*Physics: tau_N, tau_32, tau_21, ECFs, N2, D2, DDT decorrelation, Lund jet plane, jet charge, jet pull*
1. One prong, two prongs, three prongs
2. N-subjettiness and its ratios
3. Energy correlation functions and mass decorrelation (DDT)
4. The Lund jet plane: the shower history made visible
5. Our top-jet at this stage: tau_32, N2 and its Lund plane
6. Further reading

### Chapter 23 - Substructure Calibration: Jet Mass Scale and Resolution
*Physics: JMS/JMR from W-jets in semileptonic ttbar, soft-drop mass corrections, PUPPI mass corrections, jet mass uncertainties*
1. Why mass needs its own calibration
2. Extracting JMS and JMR from merged W jets
3. Our top-jet at this stage: corrected soft-drop mass
4. Further reading

---

## Part VII - Arrival: Recognising the Family

### Chapter 24 - The Registry Office I: b-Tagging
*Physics: impact parameter, secondary vertices (IVF), track counting, CSV history, DeepCSV, DeepJet, ParticleNet, UParT, working points, scale factors, mistag rates*
1. What makes a b-jet recognisable: lifetime, mass, multiplicity, soft leptons
2. From cut-based taggers to graph networks: a short history
3. Working points, efficiencies, scale factors and their measurement
4. Our b-jet at this stage: its tagger score and its secondary vertex
5. Further reading: CMS b-tagging papers

### Chapter 25 - The Registry Office II: Top-Tagging
*Physics: resolved vs boosted top, CMSTopTagger, HOTVR, soft-drop mass window + tau32, ParticleNet-MD, mass-decorrelated taggers, scale factors from semileptonic ttbar, mistag from QCD*
1. When three families travel as one: the boosted regime
2. Cut-based top tagging: mass window and tau_32
3. Machine-learned top tagging and mass decorrelation
4. Calibrating the tagger: scale factors and uncertainties
5. Our top-jet at this stage: is it tagged?
6. Further reading: CMS heavy-object tagging paper

### Chapter 26 - The Families That Never Arrived: MET and Unclustered Energy
*Physics: PF MET, PUPPI MET, Type-1 correction, unclustered energy, MET filters, MET significance*
1. Missing transverse momentum as the sum of all journeys
2. Type-1 correction: propagating JEC to MET
3. MET filters and anomalous events
4. Our event at this stage: MET before and after corrections
5. Further reading

---

## Part VIII - The Complete Pilgrimage

### Chapter 27 - The Energy Ledger: The Whole Journey of Our Two Jets
1. The b-jet ledger from LHE to calibrated AK4 jet
2. The top-jet ledger from LHE to tagged AK8 jet
3. Which stages matter most, and for which observable
4. Common misconceptions among newcomers, answered with the ledger

### Chapter 28 - Practical Life in JME
1. NanoAOD / MiniAOD jet collections and branches
2. JME tools: correctionlib, JEC/JER text files, jet veto maps, JetMET POG recipes
3. Producing your own JEC/JER (a guided tour of the JME workflows)
4. How to ask a good question in the JME hypernews / Mattermost
5. A checklist for a jet-based analysis

### Chapter 29 - The Next Pilgrimage: Run 3, Phase-2 and Beyond
1. Run 3 changes: PUPPI everywhere, new HCAL readout, timing
2. Phase-2: HGCAL, MTD, PF at Level-1, 200 pileup
3. Machine-learned jet calibration and end-to-end reconstruction
4. Open questions in jet physics

---

## Appendices

* A. Kinematic variables, coordinate system and useful identities
* B. The worked-example recipe: MG cards, Pythia fragments, CMSSW configs, step by step
* C. Reading JEC/JER files: formats and correctionlib examples
* D. A glossary of JME acronyms
* E. Solutions to the exercises
* F. Bibliography
