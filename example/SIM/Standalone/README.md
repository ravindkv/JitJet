# Chapter 10: the crossing of the border (SIM), standalone

A detector simulation in the spirit of Geant4, small enough to read. Full documentation:
[`chapter/chapter_10_SIM.md`](../../../chapter/chapter_10_SIM.md).

| file | what it is |
|---|---|
| `standalone_SIM.py` | the simulation: helices in 3.8 T, decays in flight, EM and hadronic physics, parametrised showers; `--sample signal\|intime\|oot`, `--stage`, `-v` |
| `add_hits_SIM.py` | adds the three samples by hand (25 ns x bx, cell by cell); writes `../event_SIM.lhe`, `sim_ledger.tex/.txt`, `add_hits_SIM.json` |
| `validate_SIM.py` | the layer model against the CMS simulation geometry with ROOT's TGeo (needs ROOT) |
| `plot_simhits_SIM.py` | the station displays, `pages/simHits_<sample>_10_SIM_<station>_<view>.png` |
| `plot_theory_SIM.py` | `theory_SIM.pdf`, one page per equation of the chapter |
| `animate_standalone_SIM.py` | the narrated film (four scenes; `--print-script` prints the narration) and its key frames (`--stills`) |
| `standalone_SIM.log`, `add_hits_SIM.log`, `validate_SIM.log` | the logs the chapter quotes |

```bash
python3 standalone_SIM.py --sample signal -v     # 6 s
python3 standalone_SIM.py --sample intime        # 5 s
python3 standalone_SIM.py --sample oot           # 75 s; writes ../simhits_oot.npz (27 MB, not in git)
python3 add_hits_SIM.py -v
python3 plot_simhits_SIM.py
```
