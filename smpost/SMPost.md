# JitJet on social media: one post per video

## Teaching merits

JitJet teaches jet physics the way it actually happens: one simulated event is
followed from the proton to the histogram, and every stage of the journey, from
the hard scattering through the parton shower, hadronisation, decays, pileup,
the detector, Particle Flow, clustering, calibration and tagging, is a stop with
the same shape. The reader sees the stage first in a short narrated film, runs
it second with a standalone Python script of a few hundred lines that
reproduces the algorithm of the official package, derives it third from the
formula on the page, and then lets PYTHIA, FastJet or CMSSW have the last
word in an agreement table. The numbers in the text are the numbers the script
prints, so a student can press a button, see 99.83 GeV in 31 particles come out,
change a parameter and watch the number move. Nothing is hidden behind a
framework: the scripts, records, logs and companions are public, and the
family metaphor of a mother quark, her children and the crowd on the road
holds together in memory what the equations state explicitly.

The merit for teaching is that the book closes the gap between a lecture course
on QCD and a first week in an LHC analysis. Textbooks derive the splitting
functions but stop before the detector; experiment tutorials show the
configuration files but not the physics behind them. JitJet keeps the two in
one event, so that the b quark a student met in a Feynman diagram is the same
b quark whose jet is tagged at the last stop, and every intermediate loss of
energy, every choice of radius or threshold and every systematic uncertainty is
met at the place where it enters. Each chapter stands alone as a lecture with
its film, its exercises and its pitfalls, and together they form a course that
a master's student, a new PhD student or a physicist changing from theory to
experiment can follow at their own pace, running the same code the author ran.


### ---------------------------------
Channel Name: JetMET
Channel Handle: JetMETCMS
Channel Playlist: JitJet Book, JitJet bTour, JitJet tTour, etc
### ---------------------------------

JitJet: Journey is the Jet
Director: Ravindra Verma (University of Helsinki)
Producer: Claude Fable, ChatGPT Astra

JitJet Book: https://github.com/ravindkv
JitJet CMSSW : https://github.com/ravindkv/JitJetSW
JitJet Playlist: https://www.youtube.com/playlist?list=PLbSW20iyZFd4 

#CERN, #LHC, #CMS, #JME, #BTV, #QCD, #ParticlePhysics


### --------------------------------------
(0/30) JitJet | Introduction to the Book

In the CMS experiment an electron, a photon or a muon is a destination: it is simply there at the end of reconstruction, waiting to be identified and calibrated. A jet is different. A jet is its entire journey, from a quark inside the proton to the calibrated, tagged object an analyst puts into a histogram. Journey Is The Jet: JitJet. That journey is long, complex and fascinating, and we set out to document it in a book for one single simulated event. We follow that event at every stage, show a film of what happens there first, and only then explain the physics behind it. The author of the book is the tour guide, the readers are the passengers, and the AI agents are the van that carries us from stop to stop. The metaphor is a family trip, and there are two trips: one with a bottom quark and one with a top quark. Book and code are open at github.com/ravindkv/JitJet.

Path: 
cover/animate_short_cover_timeline.mp4
chapter/chapter_00_frontmatter.tex


### --------------------------------------
(1/30) JitJet | Chapter-1 | The Ancestral Home: Protons, PDFs and the Hard Scattering

Two protons at 13.6 TeV, a u quark from one and a u antiquark from the other, and out come
a b quark and its antiquark, back to back. The film shows what the standalone
integrator does in four scenes: the collision, first in the quark centre of
mass and then boosted into the lab frame, where the pair flies far forward;
the Sobol points that sample the parton densities and the scattering angle;
the integrand as a landscape whose volume is the cross section; and the
running average settling on the answer.

Path: 
example/ME/Standalone/animate_standalone_qqbar_bbbar_xsec.mp4
chapter/chapter_01_ancestral_home.tex


### --------------------------------------
(2/30) JitJet | Chapter-2 | The Mother Gives Birth: Parton Showering

The b quark and its antiquark leave the hard scattering with a colour charge, and a colour
charge radiates. The film follows the shower in four scenes. A pT scale runs down from the
hard scale of 113 GeV and every time it passes the pT of a branching, the branching happens:
a ladder fills rung by rung and the eta-phi plane fills with partons, sized by their pT and
coloured by their family (radiation off the b, off the anti-b, or off the incoming partons),
with the colour dipoles that do the radiating drawn between them; at the end the family of
each b quark within R = 0.4 is circled. Then the Sudakov veto algorithm as an ensemble: the
four ends that can radiate first each propose their next emission many times, the survival
curves are the no-emission probabilities, and the hardest proposal wins the first branching.
A third scene keeps the books: the b anti-b pair acquires a transverse momentum that the
initial-state radiation balances, and the incoming partons walk up in x on top of the parton
densities, backward evolution. The last scene showers the same hard event again and again
and lets the number of branchings and the pT of each b family settle next to PYTHIA 8,
within a few percent.

Path:
example/PS/Standalone/animate_standalone_parton_showering.mp4
chapter/chapter_02_mother_gives_birth.tex


### --------------------------------------
(3/30) JitJet | Chapter-3 | Whose Child Is It? Matching and Merging

Chapter 3 is a stop at which our event does not get off. The short clip shows the road
of the book with its stops, the cart carrying our event, u anti-u -> b anti-b, rolling up to
stop 3, and the sign that says who this stop is for: events with extra-parton subprocesses
or a next-to-leading-order matrix element, where the matrix element and the shower both
want to make the same parton and someone has to decide whose child it is (MLM, CKKW-L,
MC@NLO, POWHEG, FxFx). The passengers that do get off here are the ttbar+jets, W/Z+jets
and multijet samples. Our event is one leading-order 2->2 process: every parton beyond the
b and anti-b has exactly one parent, the shower, and there is nothing to match or merge.
The card gets the stamp "not for us", the ledger row repeats the previous one, 100.45 GeV
with Delta pT = 0, and the cart rolls on to stop 4.

Path:
example/MnM/animate_whose_child.mp4
chapter/chapter_03_whose_child.tex


### --------------------------------------
(4/30) JitJet | Chapter-4 | Cousins Nobody Invited: Underlying Event and Colour Reconnection

Two protons overlap in the transverse plane and the impact parameter sets how many 
extra scatterings to expect. A pT scale runs down from the hard scale and each time it 
passes the pT of a secondary interaction, the interaction happens and a new system 
appears among the partons of the hard event. The protons' momentum is used up 
interaction by interaction, the remnants take what is left, and colour reconnection 
ties the soft systems onto the strings of the hard ones. The last scene repeats the 
dressing hundreds of times and lets the averages settle next to PYTHIA 8.

Path:
example/MPI/Standalone/animate_standalone_multiparton_interactions.mp4
chapter/chapter_04_uninvited_cousins.tex


### --------------------------------------
(5/30) JitJet | Chapter-5 | The Children Get Married: Hadronisation and b-Fragmentation

The guest list of the hadrons QCD allows, the b
quark's rope with its gluon knots snapping into sixteen hadrons, and a string
in its rest frame breaking snapshot by snapshot next to the same history drawn
on the world sheet. Then it follows our event: a real string of the record
breaking vertex by vertex, the b quark's string in the eta-phi plane with
hadrons popping out from both ends, the whole event where 211 partons fade and
365 primary hadrons take their place, and the ensemble where the energy
fraction carried by the B hadron settles on the Lund-Bowler value, with PYTHIA
8 on the same partons alongside.

Path:
example/Had/Standalone/animate_standalone_hadronisation.mp4
chapter/chapter_05_children_getting_married.tex


### --------------------------------------
(6/30) JitJet | Chapter-6| The Children Produce Grandchildren: B-Hadron Decays 

Most of the 365 primary hadrons of the previous stop are unstable. Hundred B
mesons born at once decay one by one, the survivors tracing the
exponential that every lifetime measurement rests on. Then our own B hadron
walks away from the collision point, decays into a charm hadron that walks on
and decays as well, and the charged tracks fan out from two displaced
vertices: the picture a secondary-vertex tagger is looking for. The decay tree grows 
generation by generation down to the particles the detector will see. One scene follows 
a semileptonic decay, where the neutrino leaves the cone and takes jet energy with it: 
the first stage of the journey where energy is lost for good. The film ends with the 
whole event decaying in order of laboratory time and with the ensemble that gives the 
flight distance and the visible share of the jet pT.

Path:
example/Decay/Standalone/animate_standalone_hadron_decays.mp4
chapter/chapter_06_children_producing_grandchildren.tex


### --------------------------------------
(7/30) JitJet | Chapter-7 | The Family Photograph: Generator-Level Jets and Truth Definitions

Nothing happens to the 749 particles of the previous stop; what happens is a
decision. Before the family walks into the detector we take its photograph: the
final state appears in the eta-phi plane, hardest particles first, and the 42
circles of the anti-kT jets settle over it. Then the zoom on the b quark's jet,
the 36 merges in their order, the hard core first and the soft periphery last,
until the circle of R = 0.4 is drawn and the jet holds 99.83 GeV in 31
particles: the reference every reconstructed jet of the journey will be judged
against. The same family in other photographs shows what the three choices of
radius, neutrinos and threshold cost. Then who decides this is a b jet at all:
the B hadron shrinks to a ghost and still lands in the jet, hadronFlavour 5,
while at the other end of the event a gluon-splitting pair shares one jet and
an oscillated B gives the bbar jet a b. A soft B0 Bbar0 pair whispering between
two light jets flips the labels, and interleaved flavour neutralisation cancels
it again: which definitions are infrared safe. The film ends with the ensemble,
the response of the AK4, AK4-no-neutrino and AK8 jets to the b quark.

Path:
example/GenJet/Standalone/animate_standalone_genjets.mp4
chapter/chapter_07_GenJet.tex



### --------------------------------------
(Part I trailer) JitJet | Part I | The Family Is Born: Chapters 1-7 in Two Minutes

One simulated event, followed through the first seven chapters on six pages of
the CMSSW genJetAnalyzer stack. A counter counts what each stage produced: 2
particles after the hard scattering, 57 after the parton shower, 169 with the
multiparton interactions and the beam remnants, 320 hadrons, 634 stable
particles after the decays, and then 239 of the 634 inside 27 anti-kT jets of
R = 0.4. One question per stop asks what the chapter answers: where do two
quarks of 113 GeV come from, why does a colour charge radiate, who invited the
other 112, how does a string break, when does a B meson decay, why does the b
jet carry more than the b quark. Every stage has its chapter, and every chapter
has its film: watch Part I.

Path:
example/trailer/animate_trailer_part1_ch01_07.mp4
example/GenJet/CMSSW/genJetAnalyzer_ME_PS_MPI_Had_Decay_GenJet_out.pdf


### --------------------------------------
(8/30) JitJet | Chapter-8 | The Crowd on the Road: Pileup

Our family is not alone. In a bunch crossing of the LHC in 2025 some sixty
proton-proton collisions happen at once, and the film puts the photograph of
the previous stop into that crowd. The 63 other collisions of our crossing
arrive one by one on the eta-phi plane, a counter running and the energy
density rho rising by about a GeV per unit area with each, until anti-kT finds
175 jets and our b jet reads 133.5 GeV instead of 99.8. Then the road: the
beam spot as the overlap of two bunches, 3.6 cm long and 120 ps wide, our
collision a hundredth of a millimetre from its nearest neighbour, and the
fifteen neighbouring crossings of the production window arriving 25 ns apart.
One member of the crowd is shown in its four kinds, non-diffractive, single-,
double- and central-diffractive, each with its sketch and its row in the
table, before the two thousand collisions of the library pile up into their
multiplicity. The grid cells fill and their median is rho, 75 GeV per unit
area; the crowd pours 35 GeV of strangers from 33 collisions into the circle
of our jet, and rho times the area takes 40 GeV back out, leaving 93.6. Drawn
four hundred times, the crowd puts 28 ± 8 GeV inside the circle and the
subtraction leaves ± 6 GeV: the first price of pileup on a hundred-GeV jet,
and what the chapters ahead are about paying less of.

Path:
example/PU/Standalone/animate_standalone_pileup.mp4
chapter/chapter_08_PU.tex


### --------------------------------------
(9/30) JitJet | Chapter-9 | The Hand-over at the Border: Geant4 Primaries

The road ends at a gate. Before the CMS detector sees a single particle, every
member of the event is handed to Geant4, and the hand-over is a selection. The
film opens with the gate at work: the family, the 63 in-time and the 897
out-of-time collisions stream in on three lanes, and at the gate the members
that are turned away fall out of the stream, down the beam pipe, into the bin
of the soft and the neutrinos, or onto the rail along the beam line to the
Zero Degree Calorimeter and the Roman pots, while the rest walk on and gather
at the centre of the detector. Then the rules, one scene each. The eta cut is
a line, not an angle: a cylinder of 2.9 cm around the beam, 355 cm long, and
every particle followed from its own vertex to its end, first for our family,
then for one collision of the crowd six centimetres off centre, where a pion
beyond 5.5 is kept on the long side and a photon inside 5.5 is dropped on the
short one. Below 40 MeV nothing enters, and the neutrinos are killed before
their first step: the missing transverse momentum of the event is decided
here. The fourteen weak decays of the family all happened inside the cylinder,
so every daughter starts as a primary. What is left waits at the centre of
CMS: 672 of the family's 749 particles, 96 % of its transverse momentum, but
only 52 % of its energy, and a quarter of the crowd's. Our b jet crosses whole,
31 of 31. Nothing has moved yet: the crossing is the next film.

Path:
example/G4/Standalone/animate_standalone_G4.mp4
chapter/chapter_09_G4.tex
