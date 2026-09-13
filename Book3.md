# JitJet - Table of Contents, version 3

## Metaphor: The Energy Economy

**The story.** Energy is money. The jet is a traveller carrying a purse whose
contents change at every stop. The **Mint** (matrix element) issues the
original sum; the **Inheritance** (parton shower) splits it among heirs; the
**Factory** (hadronisation) converts it into goods, losing some to waste
(neutrinos); **Immigrants** (underlying event, pileup) arrive with their own
money and mix it into the purse; **Customs** (detector) takes a cut and
mis-weighs the rest; the **Visa Office** (triggers) decides whether the
traveller may enter at all; the **Census** (Particle Flow) counts every coin
once; the **Police** (CHS, PUPPI) confiscate money that belongs to other people;
the **Town Hall** (clustering) declares who forms one household; the **Tax
Office** (JEC) levies the pileup tax and then pays back the response refund; the
**Auditor** (residuals) reconciles data against the model; the **Assessor**
(JER) measures with a noisy scale; the **Barber** (grooming) removes soft
excess; the **Court** (b-tagging, top-tagging) issues the identity card; the
**Insurance Company** (systematics) prices the risk. The **Bank Passbook** is
the energy ledger.

**Why this metaphor is apt.** It makes the additive/subtractive nature of every
stage explicit: each chapter *is* a transaction with a sign and an amount. It
also naturally hosts the shop/tax/police/barber images and gives a
quantitative red thread (the passbook) that newcomers can compute themselves.

---

## Front matter

* Preface: a jet is a passbook, not a coin
* How to read a transaction chapter (sign, amount, formula, reference)
* The two travellers: the b-jet and the top-jet
* Opening the passbook: the energy ledger and how we fill it

---

## Part I - Minting and Inheritance: Generation

### Transaction 1 - The Mint: Protons, PDFs and the Hard Scattering
*Physics: factorisation, PDFs, `p p > b b~`, LO/NLO, LHE*
1. Where the money comes from: the proton and its PDFs
2. The matrix element as the mint's issuing rule
3. MadGraph5_aMC@NLO in practice
4. Scale and PDF uncertainties: the mint's own error bar
5. Passbook entry: the b quark four-vector
6. Further reading

### Transaction 2 - The Inheritance: Parton Showering
*Physics: DGLAP, Sudakov, pT-ordered shower, ISR/FSR, dead cone*
1. Dividing the estate among heirs: splitting functions
2. The Sudakov form factor
3. Pythia 8 and the CP5 tune
4. The heavy heir: the dead cone
5. Heirs who leave the country: out-of-cone radiation (debit)
6. Passbook entry
7. Further reading

### Transaction 3 - Settling Double Claims: Matching and Merging
*Physics: MLM, FxFx*
1. Two accountants, one emission: double counting
2. Matching and merging schemes
3. Passbook entry
4. Further reading

### Transaction 4 - The Factory: Hadronisation and Heavy-Flavour Decays
*Physics: Lund string, b-fragmentation, B-hadron decays, neutrinos, EvtGen*
1. Converting currency into goods: the Lund string
2. b-fragmentation and x_B
3. Displaced vertices: the factory stamp
4. Waste that leaves the books: neutrinos and muons (debit)
5. Passbook entry: GenJet with and without neutrinos
6. Further reading

### Transaction 5 - Street Vendors: Underlying Event
*Physics: MPI, beam remnants, colour reconnection*
1. Money from the rest of the proton (credit)
2. Colour reconnection
3. Passbook entry: UE in R=0.4 and R=0.8
4. Further reading

### Transaction 6 - The Notarised Copy: Generator Jets and Flavour Truth
*Physics: GenJets, flavour definitions, ghost association, IR-safe flavour*
1. The reference document for all later audits
2. Flavour definitions
3. Passbook entry: the GenJet
4. Further reading

---

## Part II - Immigrants and Customs: Pileup and Detector

### Transaction 7 - Mass Immigration: Pileup
*Physics: in-time/out-of-time PU, premixing, mu, NPV, rho, area*
1. Forty households per crossing (credit)
2. In-time and out-of-time
3. Premixing
4. rho and jet area: estimating the immigrants' average wealth
5. Passbook entry
6. Further reading

### Transaction 8 - Customs: Detector Simulation and Digitisation
*Physics: Geant4, tracker, ECAL, HCAL, e/h, sampling, digitisation, noise*
1. Customs takes a cut: calorimeter response and non-compensation (debit)
2. Customs mis-weighs: non-linearity and resolution
3. Digitisation, pedestals and noise (small credit)
4. Passbook entry: SimHits and Digis
5. Further reading

### Transaction 9 - The Visa Office, Ground Floor: Level-1 Trigger
*Physics: trigger towers, L1 jets, L1 PU subtraction, uGT, turn-ons*
1. Who gets a visa: 40 MHz to 100 kHz
2. L1 jets and calibration LUTs
3. Turn-on curves and the error function
4. Passbook entry: L1 jet pT
5. Further reading

### Transaction 10 - The Visa Office, Upper Floor: High-Level Trigger
*Physics: HLT paths, Calo/PF/PUPPI online, online JEC, prescales, efficiencies*
1. 100 kHz to 1 kHz
2. Online reconstruction
3. Measuring the efficiency of your path
4. Passbook entry
5. Further reading

---

## Part III - The Census and the Police: Reconstruction and Pileup Mitigation

### Transaction 11 - Receipts and Signatures: Local Reconstruction
*Physics: RecHits, clusters, tracks, vertices, impact parameters*
1. RecHits and clusters
2. Tracks and primary vertices
3. Passbook entry: the displaced B vertex
4. Further reading

### Transaction 12 - The Census: Particle Flow
*Physics: PF blocks, linking, hadron calibration, jet energy composition*
1. Count every coin once
2. Linking and hadron calibration (credit: response restored)
3. Jet energy composition
4. Passbook entry: PF candidates
5. Further reading

### Transaction 13 - The Police: CHS and PUPPI
*Physics: CHS, PUPPI alpha/weights, SoftKiller, area subtraction*
1. Identity checks for charged particles: CHS (debit)
2. Profiling neutral particles: the PUPPI metric (debit)
3. Alternatives
4. Passbook entry: constituent weights
5. Further reading

### Transaction 14 - The Town Hall: Jet Clustering
*Physics: IRC safety, kT/CA/anti-kT, R, areas, AK4/AK8*
1. Declaring a household: IRC safety
2. Sequential recombination
3. Household size R
4. Passbook entry: the clustering tree
5. Further reading

### Transaction 15 - Fraudulent Households: Jet ID and Veto Maps
*Physics: PF jet ID, noise, beam halo, veto maps*
1. Households that never existed
2. Jet ID and veto maps
3. Passbook entry
4. Further reading

---

## Part IV - The Tax Office: Jet Energy Corrections

### Transaction 16 - The Immigration Tax: L1 Pileup Offset
*Physics: L1FastJet, hybrid, Random Cone*
1. The JEC chain as a tax code
2. rho x A and the hybrid L1 (debit)
3. Random Cone
4. Passbook entry
5. Further reading

### Transaction 17 - The Refund: L2L3 MC-Truth Corrections
*Physics: response, matching, eta/pT dependence, closure, flavour*
1. Refunding what customs took (credit)
2. Response distributions
3. Fitting the refund table
4. Closure and flavour dependence
5. Passbook entry
6. Further reading

### Transaction 18 - The Audit: L2L3Residual on Data
*Physics: dijet balance, MPF, Z+jet, gamma+jet, multijet, global fit, IOVs*
1. Auditing the model against reality
2. Relative residuals from dijets
3. Absolute residuals from Z/gamma+jet and multijet
4. The global fit
5. Time dependence
6. Further reading

### Transaction 19 - The Noisy Scale: Jet Energy Resolution
*Physics: N/S/C, dijet asymmetry, SFs, smearing*
1. Precision vs accuracy
2. Measuring JER
3. Scale factors and smearing
4. Passbook entry
5. Further reading

### Transaction 20 - The Insurance Premium: JES/JER Uncertainties
*Physics: sources, reduced sets, correlations, propagation*
1. Pricing the risk: uncertainty sources
2. Reduced sets and correlations
3. Propagation in an analysis
4. Passbook entry
5. Further reading

---

## Part V - The Barber: Substructure

### Transaction 21 - The Barber Shop: Grooming
*Physics: soft drop/mMDT, trimming, pruning*
1. Why the fat jet needs a haircut
2. Soft drop (debit, but mass sharpened)
3. Trimming and pruning
4. Passbook entry for the top-jet
5. Further reading

### Transaction 22 - Counting the Household Members: N-subjettiness, ECFs, Lund Plane
*Physics: tau_N, N2/D2, DDT, Lund plane, jet charge*
1. Prong counting
2. N-subjettiness and ECFs
3. Mass decorrelation
4. The Lund plane
5. Passbook entry
6. Further reading

### Transaction 23 - Calibrating the Mass Scale
*Physics: JMS/JMR from W jets*
1. Why mass has its own tax office
2. Extracting JMS/JMR
3. Passbook entry
4. Further reading

---

## Part VI - The Court: Identification

### Transaction 24 - The Identity Card I: b-Tagging
*Physics: IP, SV, DeepCSV/DeepJet/ParticleNet/UParT, WPs, SFs*
1. What identifies a b-jet
2. Tagger evolution
3. Efficiencies, scale factors and mistags
4. Passbook entry
5. Further reading

### Transaction 25 - The Identity Card II: Top-Tagging
*Physics: boosted regime, mass window + tau32, HOTVR, ParticleNet-MD, SFs*
1. Three households as one: the boosted top
2. Cut-based and ML taggers
3. Mass decorrelation
4. Calibration
5. Passbook entry
6. Further reading

### Transaction 26 - Money That Left the Country: MET
*Physics: PF/PUPPI MET, Type-1, unclustered energy, filters*
1. MET as the sum of all passbooks
2. Type-1 correction
3. Filters
4. Passbook entry for the event
5. Further reading

---

## Part VII - Closing the Books

### Chapter 27 - The Final Passbook: Both Jets, All Transactions
1. b-jet passbook
2. top-jet passbook
3. Largest transactions per observable
4. Newcomers' misconceptions answered with numbers

### Chapter 28 - Working in JME
1. Jet collections in MiniAOD / NanoAOD
2. correctionlib, JEC/JER files, veto maps
3. Producing JEC/JER yourself
4. Analysis checklist

### Chapter 29 - The Economy of the Future: Run 3, Phase-2, HL-LHC
1. Run 3 changes
2. Phase-2 and 200 PU
3. Machine-learned calibration and reconstruction

---

## Appendices

* A. Kinematics and identities
* B. The worked example, step by step
* C. JEC/JER file formats and correctionlib examples
* D. Glossary
* E. Exercises and solutions
* F. Bibliography
