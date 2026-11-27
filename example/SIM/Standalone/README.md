# Chapter 10: the footprints and fingerprints at the crossings (SIM), standalone

The family, the in-time and the out-of-time pileup through the official Geant4 in the CMS geometry, each in its
own file, and the same physics written by hand for comparison. Full documentation:
[`chapter/chapter_10_SIM.md`](../../../chapter/chapter_10_SIM.md).

| file | what it is |
|---|---|
| `geant4_SIM.py` | Geant4 11.4 (`geant4-pybind`, physics list FTFP_BERT) in `cmsSimGeo2021.gdml`, with CMSSW's production cuts by region and Birks' law: the three samples of the record; `--sample signal\|intime\|oot`, `--stage`, `-v`, `--only`, `--scan`, `--no-birks`, `--no-regions` |
| `guns_SIM.py` | single particles through Geant4 for the theory plots (radiography of the tracker, 10 GeV photons, pions of 2 to 100 GeV) |
| `export_gdml_SIM.py`, `field_map_SIM.py`, `dump_shapes_SIM.py` | the geometry as GDML (needs ROOT), the field map from its coil and iron, the shapes for the 3D displays |
| `g4view_SIM.py`, `display_geant4_SIM.py` | the 3D viewer and the five pages `pages/geant4_<magnet,tracker,ecal,hcal,muon>.png` |
| `add_hits_SIM.py` | adds the three samples by hand (25 ns x bx, cell by cell); writes `../event_SIM.lhe`, `sim_ledger.tex/.txt`, `add_hits_SIM.json` |
| `standalone_SIM.py` | the hand-written simulation, the readable version: helices in 3.8 T, decays in flight, EM and hadronic physics, parametrised showers; writes `../simhits_<sample>_fast.npz` |
| `validate_SIM.py` | the hand-written script against Geant4: material, sensors, the family, single particles |
| `plot_simhits_SIM.py` | the hit displays, `pages/simHits_<sample>_10_SIM_<station>_<view>.png` (the crowd binned; r < 450 cm) |
| `plot_theory_SIM.py` | `theory_SIM.pdf`, one page per equation of the chapter, and `plot_theory_SIM.json` |
| `animate_standalone_SIM.py` | the film inside the Geant4 detector (seven scenes, narrated, no music; `--print-script`) and its key frames (`--stills`) |
| `../../compress_png.sh` | every PNG page down to 256 colours |
| `*.log`, `*.json` | the logs and numbers the chapter quotes |

```bash
.venv/bin/python geant4_SIM.py --sample signal -v --log geant4_SIM.log   # 1 min; make venv-sim and make gdml first
.venv/bin/python geant4_SIM.py --sample intime                           # 5 min
.venv/bin/python geant4_SIM.py --sample oot                              # about an hour; ../simhits_oot.npz is not in git
for s in transport B B_Decay B_Decay_EM; do .venv/bin/python geant4_SIM.py --sample signal --stage $s; done
.venv/bin/python guns_SIM.py > guns_SIM.log
python3 add_hits_SIM.py -v > add_hits_SIM.log
python3 plot_simhits_SIM.py && ../../compress_png.sh pages/simHits_*.png
python3 display_geant4_SIM.py && ../../compress_png.sh -w 1600 pages/geant4_*.png
python3 plot_theory_SIM.py -o theory_SIM.pdf
```

The three samples are simulated separately and kept in three files: a change of the signal does not cost the
hour of the crowd again. Only a simulation can do this; in data everything arrives at once.
