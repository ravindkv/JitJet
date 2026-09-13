# JitJet - Table of Contents, version 2

## Metaphor: The Expedition - from Base Camp to Summit

**The story.** "From bottom to top" taken literally. The b quark starts at
**Base Camp**; the fully calibrated, tagged jet is the **Summit**. The route is
divided into camps, each of which costs energy (radiation, neutrinos, detector
response, pileup subtraction, grooming) or gives energy back (underlying event,
pileup, calibration). Between camps the climber meets weather (pileup), passes
permit offices (triggers), is roped up with the rest of the team (clustering),
is checked by rangers who send down the stowaways (PUPPI), sheds unnecessary
gear (grooming), has altitude measured with an imperfect altimeter (JER), and
is finally identified in the summit register (tagging). The b-jet is one
climber; the top-jet is a rope team of three (t -> W b -> q q' b) whose members
are so close together at high altitude (boost) that they are logged as one.

**Why this metaphor is apt.** Mountaineering has a natural *ordering* (camps),
a natural *budget* (oxygen, food = energy), a natural notion of *what you carry*
vs *what you drop*, and a natural *permit system* (triggers). It also matches
the title exactly: bottom to top.

---

## Front matter

* Preface: a jet is not a place, it is a route
* The expedition log: how the energy ledger works and why every chapter uses it
* The two climbers: the b-jet (`p p > b b~`) and the top rope-team (`p p > t t~`)
* Map of the route: the 24-stage spine on one page
* Reproducibility: seeds, software releases, containers

---

## Part I - Base Camp: The Climber Is Chosen

### Camp 0 - Choosing the Route: Kinematics, Coordinates and Conventions
1. pT, eta, phi, rapidity, mass and why we use them
2. Delta R and cones on the (eta, phi) cylinder
3. rho, jet area, mu, NPV: the weather variables
4. Reading the expedition log (the energy ledger)

### Camp 1 - Base Camp: The Hard Scattering in MadGraph
*Physics: PDFs, factorisation, LO/NLO, `p p > b b~`, LHE*
1. The mountain: the proton and its parton distribution functions
2. The matrix element: b b~ from gg and qq~; 4-flavour vs 5-flavour schemes
3. Setting up MadGraph5_aMC@NLO: cards, seeds, cuts, scales
4. Scale and PDF uncertainties as the first entries of the log
5. Our climber at Base Camp: the LHE four-vector
6. Further reading

### Camp 2 - Acclimatisation: The Parton Shower
*Physics: splitting functions, Sudakov factor, pT ordering, ISR/FSR, dead cone, tunes*
1. Why the climber cannot stay a single quark
2. DGLAP splitting functions and the Sudakov "no-emission" probability
3. Pythia 8's shower and the CP5 tune
4. The heavy climber: dead cone for b and t
5. Energy left below: out-of-cone radiation as the first loss in the log
6. Our climber at Camp 2
7. Further reading

### Camp 3 - Dividing the Load: Matching and Merging
*Physics: MLM, CKKW-L, FxFx*
1. Who carries the extra emission: ME or shower?
2. Merging schemes and merging scales
3. Our climber at Camp 3
4. Further reading

### Camp 4 - Packing the Gear: Hadronisation and Heavy-Flavour Decay
*Physics: Lund strings, b-fragmentation, B hadrons, semileptonic decays, EvtGen*
1. From partons to hadrons
2. The fragmentation function and x_B: how much energy the B hadron carries
3. B-hadron flight and the displaced vertex
4. Gear left behind: neutrinos and muons from B decays
5. Our climber at Camp 4: GenJet with and without neutrinos
6. Further reading

### Camp 5 - Porters Nobody Hired: Underlying Event
*Physics: MPI, beam remnants, colour reconnection*
1. The rest of the protons joins the climb
2. Colour reconnection and jet mass
3. Our climber at Camp 5: UE energy in R=0.4 and R=0.8
4. Further reading

### Camp 6 - The Reference Photograph: Generator Jets and Flavour Truth
*Physics: GenJets, hadron vs parton flavour, ghost association, IR-safe flavour*
1. Defining "truth" for calibration
2. Flavour definitions and their pitfalls
3. Our climber at Camp 6: the GenJet used as reference for the rest of the book
4. Further reading

---

## Part II - The Weather Closes In: Pileup and the Detector

### Camp 7 - The Storm: Pileup
*Physics: in-time/out-of-time PU, premixing, mu, NPV, rho, PU reweighting*
1. Forty collisions per crossing
2. In-time vs out-of-time; bunch structure and detector integration times
3. Premixing in CMSSW
4. rho and jet area
5. Our climber at Camp 7: pileup energy inside our jet
6. Further reading

### Camp 8 - Into the Ice: Detector Simulation
*Physics: Geant4, tracker, ECAL, HCAL, e/h non-compensation, sampling, digitisation, noise*
1. What the climber's energy does to lead tungstate and brass
2. Why calorimeter response is below one and non-linear
3. Digitisation, pedestals, zero suppression
4. Our climber at Camp 8: SimHits and Digis
5. Further reading

### Camp 9 - The First Permit Office: Level-1 Trigger
*Physics: trigger towers, L1 jet algorithm, L1 PU subtraction, uGT, turn-on curves*
1. 40 MHz in, 100 kHz out
2. The L1 calorimeter jet
3. Efficiency turn-ons: fitting an error function
4. Our climber at Camp 9
5. Further reading

### Camp 10 - The Second Permit Office: High-Level Trigger
*Physics: HLT paths, Calo/PF/PUPPI jets online, online JEC, prescales, efficiency measurement*
1. HLT architecture
2. Online reconstruction and calibration
3. Measuring trigger efficiency in your analysis
4. Our climber at Camp 10
5. Further reading

---

## Part III - The Rangers Count Everyone: Reconstruction

### Camp 11 - Footprints in the Snow: Local Reconstruction
*Physics: RecHits, clustering, tracking, vertexing, impact parameters*
1. From Digis to RecHits
2. Tracks and primary vertices
3. Our climber at Camp 11: the displaced B vertex
4. Further reading

### Camp 12 - The Head Count: Particle Flow
*Physics: PF blocks, linking, hadron calibration, jet energy composition*
1. The Particle Flow idea
2. Linking and the PF hadron calibration
3. Jet energy composition
4. Our climber at Camp 12: the PF candidates of our jet
5. Further reading

### Camp 13 - Sending the Stowaways Down: CHS and PUPPI
*Physics: CHS, PUPPI alpha and weights, SoftKiller, area subtraction*
1. Charged stowaways: vertex association
2. Neutral stowaways: the PUPPI metric
3. Alternatives and comparisons
4. Our climber at Camp 13: constituent weights
5. Further reading

### Camp 14 - Roping Up: Jet Clustering
*Physics: IRC safety, kT/CA/anti-kT, R, areas, AK4/AK8, recombination*
1. Infrared and collinear safety
2. The sequential recombination algorithms
3. Choosing the rope length R (pT > 2m/R for boosted objects)
4. Our climber at Camp 14: the clustering tree
5. Further reading

### Camp 15 - Rocks That Look Like Climbers: Jet ID and Veto Maps
*Physics: PF jet ID, noise, beam halo, veto maps*
1. Fake jets and their sources
2. Jet ID and veto maps
3. Our climber at Camp 15
4. Further reading

---

## Part IV - Adjusting the Altimeter: Jet Energy Corrections

### Camp 16 - Subtracting the Snow Load: L1 Pileup Offset
*Physics: L1FastJet, hybrid parametrisation, Random Cone*
1. The factorised JEC chain
2. rho x A and the hybrid L1
3. Random Cone measurement
4. Our climber at Camp 16
5. Further reading

### Camp 17 - Calibrating Against the Photograph: L2L3 MC-Truth Corrections
*Physics: response, matching, eta/pT dependence, closure, flavour response*
1. Reco-to-gen matching
2. Response distributions: mean vs median vs mode
3. Fitting the correction
4. Closure and flavour dependence
5. Our climber at Camp 17
6. Further reading

### Camp 18 - Calibrating Against the Real Mountain: L2L3Residual
*Physics: dijet balance, MPF, Z+jet, gamma+jet, multijet, global fit, IOVs*
1. Why data needs residuals
2. Relative (eta) residuals from dijets
3. Absolute (pT) residuals from Z/gamma+jet and multijet
4. The global fit
5. Time dependence
6. Further reading

### Camp 19 - The Shaky Altimeter: Jet Energy Resolution
*Physics: N/S/C, dijet asymmetry, JER scale factors, smearing*
1. Resolution parametrisation
2. Measuring JER in data and simulation
3. Scale factors and smearing methods
4. Our climber at Camp 19
5. Further reading

### Camp 20 - The Error Bar on the Altitude: JES/JER Uncertainties
*Physics: sources, reduced sets, correlations, propagation*
1. Uncertainty sources
2. Reduced sets and correlations
3. Propagating to an analysis
4. Our climber at Camp 20
5. Further reading

---

## Part V - Shedding Weight for the Final Push: Substructure

### Camp 21 - Leaving Gear at High Camp: Grooming
*Physics: soft drop / mMDT, trimming, pruning, groomed mass*
1. Why raw fat-jet mass is unusable
2. Soft drop: algorithm, parameters, analytic behaviour
3. Trimming and pruning
4. Our rope team at Camp 21: mass before and after
5. Further reading

### Camp 22 - Counting the Rope Team: N-subjettiness, ECFs, Lund Plane
*Physics: tau_N, ratios, N2/D2, DDT, Lund plane, jet charge*
1. Prongs and how to count them
2. N-subjettiness
3. Energy correlators and decorrelation
4. The Lund plane
5. Our rope team at Camp 22
6. Further reading

### Camp 23 - Calibrating the Scale at Altitude: Jet Mass Scale and Resolution
*Physics: JMS/JMR from W jets*
1. Why mass needs its own calibration
2. Extracting JMS/JMR
3. Our rope team at Camp 23
4. Further reading

---

## Part VI - The Summit Register: Identification

### Camp 24 - Signing the Register I: b-Tagging
*Physics: IP, SV, DeepCSV/DeepJet/ParticleNet/UParT, working points, SFs*
1. What identifies a b-jet
2. Tagger evolution
3. Efficiencies, scale factors, mistags
4. Our b-climber signs the register
5. Further reading

### The Summit - Signing the Register II: Top-Tagging
*Physics: resolved vs boosted, mass window + tau32, HOTVR, ParticleNet-MD, SFs*
1. Why the top is the summit: three climbers logged as one
2. Cut-based and machine-learned top taggers
3. Mass decorrelation
4. Calibration and uncertainties
5. Our rope team reaches the summit
6. Further reading

### Camp 25 - Those Who Did Not Reach the Register: MET
*Physics: PF/PUPPI MET, Type-1, unclustered energy, filters*
1. MET as the sum of all routes
2. Type-1 correction
3. MET filters
4. Our event's MET
5. Further reading

---

## Part VII - Descent and Debrief

### Chapter 26 - The Expedition Log: The Full Ledger of Both Climbs
1. b-jet: Base Camp to Camp 24
2. top-jet: Base Camp to Summit
3. Which camps cost the most, and for which observable

### Chapter 27 - Working in JME: Tools and Workflows
1. MiniAOD / NanoAOD jet collections
2. correctionlib, JEC/JER files, veto maps
3. Producing JEC/JER yourself: the JME workflows
4. Analysis checklist

### Chapter 28 - The Next Expedition: Run 3, Phase-2, HL-LHC
1. Run 3 changes
2. Phase-2 detector and 200 PU
3. Machine-learned calibration and reconstruction

---

## Appendices

* A. Kinematics and identities
* B. The worked example, step by step (cards, fragments, configs)
* C. JEC/JER file formats and correctionlib examples
* D. Glossary of JME acronyms
* E. Exercises and solutions
* F. Bibliography
