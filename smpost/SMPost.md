# JitJet on social media: one post per video


### ---------------------------------
Channel Name: JitJet
Channel Handle: JitJetLHC
Channel Playlist: JitJet bTour, JitJet tTour, etc
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
