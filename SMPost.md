# JitJet on social media: one post per video

Posts for the six films of *JitJet: Journey Is The Jet (from bottom to top)*,
the CMS Jet/MET training book from the University of Helsinki. Every film is
rendered by the same Python code that produces the numbers in the book, and
the code is public at <https://github.com/ravindkv/JitJet>.

Each video gets three versions:

* **Long** (LinkedIn, Facebook, Mastodon, YouTube description): 100-200 words.
* **Short** (X, Bluesky, Threads): under 280 characters, hashtags included.
* **Caption** (Instagram Reels, YouTube Shorts, TikTok): one line on screen.

Suggested posting order: trailer first, then the five stage films in the order
of the journey (hard scattering, underlying event, hadronisation, decays). The
two cross-section films are one post with the second film as a reply or as the
second clip of a carousel.

Common hashtags (pick 3-5 per platform):
`#ParticlePhysics #CMS #LHC #QCD #Jets #OpenScience #SciComm #PhysicsEducation #Python #HelsinkiUni`

Common sign-off line (optional): *Open code, open book, open invitation:
github.com/ravindkv/JitJet*

Before posting, check the CMS guidelines on public communication: these films
are educational material built from standalone code and PYTHIA validation
runs, not CMS physics results, and the posts say so.

---

## 1. Trailer: the whole journey in one minute

**File:** `cover/animate_short_cover_timeline.mp4` (58 s, 1920x1080, no narration, music only)

**What it shows.** A glow on an empty page is the hard scattering. From it a
cone grows, one part of the book at a time: the b quark and her shower, the
hadrons, the pileup crowd and the detector wall, the two trigger gates, the
Particle-Flow census, PUPPI and the R = 0.4 cone, the jet energy scale, the b
tag, and finally the R = 0.8 top jet that caps the cone. The camera pulls back
as the cone grows and the film ends on the title of the book.

### Long

A jet is not a thing. It is a journey.

An electron or a muon is a destination: it is simply there at the end of
reconstruction, waiting to be identified and calibrated. A jet never exists at
a single point. It is the accumulated history of one coloured quark, from the
hard scattering inside the proton to the calibrated, tagged four-vector an
analyst puts into a histogram. Every stage on the way adds energy, removes
energy, or reshuffles it.

This one-minute film is the cover of JitJet (Journey Is The Jet), a training
book we wrote for new Master's, PhD and postdoc students joining the CMS
Jet/MET group. Watch the cone grow: the quark and her shower, the hadrons, the
pileup crowd, the detector wall, the trigger gates, the Particle-Flow census,
the clustering cone, the energy corrections, the b tag, and the top jet that
closes the story.

Every chapter follows the same jet, with a short film, a standalone Python
script you can run yourself, the minimal mathematics, and a validation against
PYTHIA 8 and CMSSW. Book and code are open: https://github.com/ravindkv/JitJet

More films from the individual stages follow in the coming days.

### Short

A jet is not a thing, it is a journey. One minute, one quark, every stage from
the proton to the tagged jet. The trailer of JitJet, our open training book for
new CMS Jet/MET students. Code and book: github.com/ravindkv/JitJet
#ParticlePhysics #CMS #LHC #Jets #OpenScience

### Caption

From one quark to a tagged jet in 58 seconds. Journey Is The Jet.

---

## 2. Stage 1, the hard scattering: a cross section from scratch

**Files:** `example/ME/Standalone/animate_standalone_qqbar_bbbar_xsec.mp4`
(34 s, 1280x720, pT-hat >= 100 GeV) and
`example/ME/Standalone/animate_standalone_qqbar_bbbar_xsec_minPt30.mp4`
(34 s, 1280x720, the same film with pT-hat >= 30 GeV)

**What it shows.** Four scenes. (1) The event: a u quark and a u antiquark
collide head on and the b and anti-b leave back to back through the wall of a
translucent tube, the pT-hat cut; the system is then boosted into the lab frame
and ends up far forward. (2) Sobol points in the unit cube, coloured by their
weight, morph into the physical variables x1, x2 and cos theta. (3) The
integrand as a surface over the x1-x2 triangle; its volume is the cross
section. (4) One Sobol pass: the cube fills up while the running average
settles on the answer.

**Numbers.** At pT-hat >= 100 GeV the script gives 0.345 nb; PYTHIA 8 in the
CMSSW run gives 345.1 +/- 6.1 pb, a difference of 0.06 %. At pT-hat >= 30 GeV
the script gives 18.5 nb against PYTHIA's 18.44 nb. No normalisation was fitted.

### Long

Where does a jet come from? From a number you can compute yourself.

This is the first stop of the JitJet journey: the hard scattering. Two protons
at 13.6 TeV, a u quark from one and a u antiquark from the other, and out come
a b quark and its antiquark, back to back. The film shows what the standalone
integrator does in four scenes: the collision, first in the quark centre of
mass and then boosted into the lab frame, where the pair flies far forward;
the Sobol points that sample the parton densities and the scattering angle;
the integrand as a landscape whose volume is the cross section; and the
running average settling on the answer.

The answer is 0.345 nb for pT above 100 GeV. PYTHIA 8, run through CMSSW on
the same setup, gives 345.1 +/- 6.1 pb. The difference is 0.06 %, and nothing
was tuned to get there. The second clip repeats the film with the cut lowered
to 30 GeV, where the cross section is fifty times larger and the agreement is
the same.

The script is 100 % Python, uses the same parton densities and running
coupling as PYTHIA, and is in the repository with the command to run it:
https://github.com/ravindkv/JitJet

Next stop: the quark radiates.

### Short

A cross section from scratch: u + anti-u -> b + anti-b at 13.6 TeV, integrated
in plain Python. 0.345 nb. PYTHIA 8 says 345.1 +/- 6.1 pb. Difference 0.06 %,
nothing tuned. Stop 1 of the JitJet journey. github.com/ravindkv/JitJet
#QCD #CMS #LHC #Python

### Caption

Two quarks in, two b quarks out, and a cross section that agrees with PYTHIA to 0.06 %.

### Reply for the second clip

Same film, cut lowered from 100 GeV to 30 GeV. The cross section grows fifty
times, to 18.5 nb, and PYTHIA's 18.44 nb is still within reach. The pictures
change, the code does not.

---

## 3. Stage 3, the underlying event: cousins nobody invited

**File:** `example/MPI/Standalone/animate_standalone_multiparton_interactions.mp4` (53 s, 1280x720)

**What it shows.** Five scenes. (1) Two protons, drawn as the CP5 matter
profile, approach in the transverse plane; the overlap and the expected number
of extra interactions follow the sliding impact parameter and stop at the
impact parameter of our event. (2) The pT scale runs down from the hard scale
of 113 GeV and every time it passes the pT of a secondary scattering, the
scattering happens: a ladder fills rung by rung and new parton systems appear
among the partons of the hard event. (3) The momentum of each proton is eaten
interaction by interaction and what is left goes to the beam remnants.
(4) The colour strings before and after colour reconnection; the total string
length drops. (5) The same event dressed hundreds of times: the number of
interactions and the extra pT inside R = 0.4 around the b quark settle against
PYTHIA 8.

### Long

The two b quarks are not alone. The rest of the two protons comes along.

Stop 3 of the JitJet journey: the underlying event. When two protons collide,
the hard scattering that makes our b quarks is one of several parton-parton
scatterings happening at once. The others are softer, but there are many, and
they drop extra energy inside the jet cone that was never part of the quark.

The film follows the showered b anti-b event of the previous stage while the
rest of the protons joins in. Two protons overlap in the transverse plane and
the impact parameter sets how many extra scatterings to expect. A pT scale runs
down from the hard scale and each time it passes the pT of a secondary
interaction, the interaction happens and a new system appears among the
partons of the hard event. The protons' momentum is used up interaction by
interaction, the remnants take what is left, and colour reconnection ties the
soft systems onto the strings of the hard ones. The last scene repeats the
dressing hundreds of times and lets the averages settle next to PYTHIA 8.

All of it is written from scratch in Python following PYTHIA's own
construction, and the committed event is the one at the same impact parameter
as the CMSSW event, so the book can follow a single event all the way.
https://github.com/ravindkv/JitJet

Next stop: the partons become hadrons.

### Short

The b quarks are not alone: the rest of the two protons scatters too. Stop 3
of JitJet, the underlying event, from the proton overlap to colour
reconnection, in Python, checked against PYTHIA 8. github.com/ravindkv/JitJet
#QCD #CMS #LHC #Jets

### Caption

Cousins nobody invited: what the rest of the protons adds to a jet.

---

## 4. Stage 4, hadronisation: the children get married

**File:** `example/Had/Standalone/animate_standalone_hadronisation.mp4` (76 s, 1280x720)

**What it shows.** Nine scenes. The first four are animated sketches: which
quark combinations QCD lets out of the house (mesons, baryons, the rare
tetraquarks and pentaquarks) and which never leave; the b quark's rope with its
gluon knots snapping into sixteen hadrons; a string in its rest frame breaking
snapshot by snapshot next to its world sheet; and the break-by-break ladder
from the log. The last five follow the record: a two-parton string of the
event breaking at the vertices the script drew, the b quark's string in the
eta-phi plane with hadrons popping out from both ends, the string regions
across the gluon kinks, the whole event where 211 partons fade and 365 primary
hadrons take their place, and the ensemble where the b hadron's energy fraction
settles on the Lund-Bowler value next to PYTHIA 8.

### Long

No quark ever reaches the detector. What arrives are hadrons, and this is how
they are made.

Stop 4 of the JitJet journey: hadronisation. The colour field between a quark
and an antiquark behaves like a string with a constant tension of about one
GeV per femtometre. Pull the ends apart and the string does not stretch
forever: it breaks, a new quark-antiquark pair appears from the vacuum at each
break, and every piece between two breaks becomes a hadron. Gluons are kinks
on the string.

The film starts with sketches: the guest list of the hadrons QCD allows, the b
quark's rope with its gluon knots snapping into sixteen hadrons, and a string
in its rest frame breaking snapshot by snapshot next to the same history drawn
on the world sheet. Then it follows our event: a real string of the record
breaking vertex by vertex, the b quark's string in the eta-phi plane with
hadrons popping out from both ends, the whole event where 211 partons fade and
365 primary hadrons take their place, and the ensemble where the energy
fraction carried by the B hadron settles on the Lund-Bowler value, with PYTHIA
8 on the same partons alongside.

The Lund string model, ported to Python from PYTHIA 8's own classes, with the
same tune parameters, and run on the event the book has followed since the
first stop. https://github.com/ravindkv/JitJet

Next stop: the hadrons decay.

### Short

No quark reaches the detector. A colour string breaks, a pair appears at each
break, every piece is a hadron. 211 partons become 365 hadrons. Stop 4 of
JitJet, the Lund string in Python, validated against PYTHIA 8.
github.com/ravindkv/JitJet #QCD #CMS #Hadrons

### Caption

211 partons go in. 365 hadrons come out. The Lund string, frame by frame.

---

## 5. Stage 4b, B-hadron decays: the grandchildren

**File:** `example/Decay/Standalone/animate_standalone_hadron_decays.mp4` (72 s, 1280x720)

**What it shows.** Seven scenes. (1) A hundred B mesons born at once decay one
by one and the survivors follow an exponential; the lifetime drawn for our
event's B hadron is marked on the curve. (2) The B hadron walks away from the
primary vertex at gamma-beta = 31, one picosecond of proper time is a
centimetre, decays, its charm hadron walks on and decays too, and the charged
tracks fan out from the displaced vertices. (3) The decay table of the B as a
cumulative bar of branching ratios; the random number slides down to the
channel of our event. (4) The decay tree grows generation by generation in the
eta-phi plane down to the stable particles. (5) A semileptonic decay: the
lepton energy spectrum fills up and the neutrino leaves the cone, taking jet pT
with it. (6) The whole event: 365 primary hadrons decay in order of their
laboratory decay time. (7) The ensemble: flight distance of the B and visible
share of the jet pT.

### Long

The B hadron lives for 1.5 picoseconds. In our event that is a centimetre of
flight, and that centimetre is how we tag a b jet.

Stop 5 of the JitJet journey: the hadrons decay. Most of the 365 primary
hadrons of the previous stop are unstable. The film opens with a hundred B
mesons born at once and decaying one by one, the survivors tracing the
exponential that every lifetime measurement rests on. Then our own B hadron
walks away from the collision point, decays into a charm hadron that walks on
and decays as well, and the charged tracks fan out from two displaced
vertices: the picture a secondary-vertex tagger is looking for.

The decay table of the B becomes a bar of branching ratios and a random number
picks the channel of our event. The decay tree grows generation by generation
down to the particles the detector will see. One scene follows a semileptonic
decay, where the neutrino leaves the cone and takes jet energy with it: the
first stage of the journey where energy is lost for good. The film ends with
the whole event decaying in order of laboratory time and with the ensemble
that gives the flight distance and the visible share of the jet pT.

Lifetimes, mixing, 3114 decay channels of 161 particles and the matrix
elements that shape them, in Python following PYTHIA 8's ParticleDecays, on
the same event as every other stop. https://github.com/ravindkv/JitJet

Next stop: the jet is clustered.

### Short

A B hadron lives 1.5 ps. In our event that is a centimetre of flight, and that
centimetre is how we tag b jets. Stop 5 of JitJet: lifetimes, displaced
vertices, decay tables and the neutrino that leaves the cone, in Python.
github.com/ravindkv/JitJet #CMS #LHC #BPhysics

### Caption

One picosecond, one centimetre, one b tag. The grandchildren of the quark.

---

## Thread version (X or Bluesky, one post per film)

1/ A jet is not a thing, it is a journey. We wrote an open training book for
new CMS Jet/MET students that follows one quark from the proton to the tagged
jet, with a film, a Python script and a PYTHIA check at every stop. Trailer
below. github.com/ravindkv/JitJet

2/ Stop 1, the hard scattering. u + anti-u -> b + anti-b at 13.6 TeV,
integrated in plain Python: 0.345 nb. PYTHIA 8 says 345.1 +/- 6.1 pb. Nothing
tuned.

3/ Stop 3, the underlying event. The rest of the two protons scatters too,
and drops energy in the jet cone that was never part of the quark. From the
proton overlap to colour reconnection.

4/ Stop 4, hadronisation. A colour string breaks, a pair appears at each
break, every piece is a hadron. 211 partons become 365 hadrons.

5/ Stop 5, the decays. A B hadron lives 1.5 ps, flies a centimetre, and that
centimetre is a b tag. The neutrino leaves the cone and the jet loses energy
for good.

6/ Every film is rendered by the code that produces the numbers in the book,
and every script runs on a laptop. Clone it, run it, break it:
github.com/ravindkv/JitJet
