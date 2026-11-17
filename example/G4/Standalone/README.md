# The gate at the border: the hand-over to Geant4, without CMSSW

`standalone_G4.py` takes the record of chapter 8 (`../../PU/event_PU.lhe`: the signal of chapter 7 with its vertex
table, the 961 collisions of the window bx = -12 ... +3 with their vertices, the particles of the in-time crossing)
and the library the crowd was drawn from (`../../PU/pileup_library.lhe.gz`, which gives back the 897 out-of-time
collisions), and writes out the decision CMSSW makes before Geant4 takes its first step: the hand-over cuts of
`SimG4Core/Generators/src/Generator.cc` (CMSSW_15_0_5; the very-forward rule at |z| = 52.8 m for n, nbar and photons,
the 40 MeV momentum cut, the eta cut as a straight line from the vertex to |z| = 354.8 cm, the cylinder r < 2.9 cm
that decides whether a generator decay is kept), the neutrinos the `StackingAction` kills, and the protons the PPS
transport carries to the Roman pots (|eta| > 8.2, |pz| > 3 TeV). The output, `../event_G4.lhe`, is the chapter-8
record followed by the summary, the hand-over listing (every stable particle of the signal and the in-time crowd
with its decision), the collision listing and the decay listing.

```bash
python3 standalone_G4.py                          # -> ../event_G4.lhe (6 s)
python3 standalone_G4.py -v --log standalone_G4.log --summary standalone_G4.json --tables handover_ledger
python3 standalone_G4.py --library-scan standalone_G4_library.json   # the gate on the 2000 library collisions (5 s)
python3 ../plot_journey_G4.py --png               # the (eta, phi) display of the hand-over
python3 plot_theory_G4.py                         # the five theory pages
python3 draw_cms3d.py                             # the CMS detector of Run 3 in 3D -> chapter/tikz/g4_cms3d.tex
python3 animate_standalone_G4.py --stills animate_standalone_G4_stills.pdf --no-video --no-narration --no-music
PYTHONPATH=$(root-config --libdir) ../../ME/Standalone/.venv/bin/python validate_G4.py --log validate_G4.log --summary validate_G4.json
```

On this Mac use `/usr/bin/python3` (NumPy, matplotlib) for everything but the validation, which needs PyROOT (ROOT
6.32 from Homebrew, built for Python 3.13) and pythia8mc: the ME virtual environment.

## The numbers (seed 1)

* the family: 749 stable -> 672 primaries: 89.7 % of the particles, **52.5 % of the energy**, 95.5 % of the pT;
  dropped 67 beyond |eta| = 5.5 (6230.9 GeV), 5 below 40 MeV (0.12 GeV), 5 neutrinos (229.4 GeV); our jet [0] crosses
  whole (31 of 31, 99.83 GeV); 14 weak decays, all inside 2.9 cm (the farthest a D0 at r = 1.48 cm);
* the in-time crowd (63 collisions): 13 456 -> 10 825 primaries, 25.3 % of the energy; 124 very forward n, nbar,
  gamma handed over (124 TeV), 27 protons to the Roman pots (132 TeV);
* the out-of-time crowd (897): 161 696 -> 128 894, 25.2 % of the energy; 363 protons to the Roman pots;
* the line against the angle: 114 particles kept beyond |eta| = 5.5, 93 dropped inside it;
* the library: 142.7 primaries per collision, 25.5 % of the energy, 0.39 Roman-pot protons;
* validation: TGeo agrees for 175 895 of 175 901 particles (the six others have pz rounded to zero in the library),
  PYTHIA's particle data for every species.

The chapter is `chapter/chapter_09_G4.tex`, the technical companion `chapter/chapter_09_G4.md`.
