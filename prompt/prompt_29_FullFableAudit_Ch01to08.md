Ravindra:
It is high time to review all the numbers/formulae from the Standalone in the chapter/*.tex upto 08_PU:
+ The idea is to check  the consistency and potential pitfalls, bugs. 
+ If any bug spotted,  fix in the corresponding .py file and update in the .tex
+ No need to regenerate .mp4. Just stills should be fine.

Propagate the fixes (if any) to the next chapters (from 01 to 08) if applicable.

Fable:

The audit of chapters 1–8 is finished and the book builds: 248 pages, no LaTeX errors, every stop within its page cap (chapter 2 stays at 40, as before). Nothing is committed; 124 files are modified.

**What the audit found.** Every committed record reproduced byte for byte, so nothing was stale. The problems were in three places: script bugs, text and figures that disagreed with the logs, and formulae printed differently from what the code does.

**Your decision, applied.** You chose to keep the event. The script bugs whose fix would change it are left in the code, marked `KNOWN DEVIATION from PYTHIA (audit 2026-11-14)`, and stated in each chapter's caveats with their measured size:

| Chapter | Deviation | Size |
|---|---|---|
| 2 | ISR cut-off 0.42 GeV where PYTHIA keeps 0.2 | 1.6% of ISR emissions |
| 4 | f(b) normalisation | 3.627 in the record against 3.704 (CMSSW 3.709) |
| 5 | string vertex rule, Γ definition, diquark weights, redraws, masses in W_min | 402.5 against 383.3 hadrons; baryons 17% low |
| 6 | polarisation weight on the wrong decays; spectator pair | rows 2209, 2210 and 1584 |

The chapter 5 deviations were confirmed against PYTHIA's public source. The old "+5% multiplicity, cause unknown" and "fewer baryons because no popcorn" statements are replaced.

**Fixed without moving any particle or jet:**
- **Chapter 1:** the learning mode now works for any flavour and beam orientation. The seven-point grid figure had its corner cells in the wrong rows. The μF sign claim, the "massless is off by 3 in 10⁴" claim and the PDF-set αs comparison were wrong; a `--alpha-backend lhapdf` row (344.99 pb, −0.10%) is added.
- **Records 4–7:** rows 3 and 4 (the hard-process incoming partons) were being overwritten on read. They carry the chapter 2 values again; nothing else differs.
- **Chapter 4:** a secondary scattering below the charm or bottom mass threshold could give a negative-energy parton (seen in other seeds, not seed 1). It is now rejected.
- **Chapter 6:** "PYTHIA flags only a third of oscillations" was a counting bug in the validation script. The fractions agree, 16.5% against 16.8%.
- **Chapter 7:** the flavour ghosts no longer include beam-remnant diquarks (161 partons, 179 ghosts, 928 steps). Running an exercise with another algorithm can no longer overwrite the book's record. The flavour-kT and soft-pair statements now say what the script does.
- **Chapters 2–8 text:** many miscounts and stale numbers corrected in `.tex`, `.md`, TikZ and READMEs together.

**Chapter 8 is the exception: its record changed.** Two bugs were fixed in its script: the same library collision could be stacked twice in one crossing, and the ensemble circle was drawn in η while the jets live in rapidity. The new numbers:

| | before | now |
|---|---|---|
| our jet with the crowd | 132.75 GeV, mixed jet [0] | 133.53 GeV, mixed jet [1] |
| after ρA | 96.66 | 93.64 |
| ρ | 68.3 | 75.5 |
| ensemble residual | −1.9 ± 6.7 | −1.1 ± 6.1 |

One pileup jet now survives the subtraction above 30 GeV, so the "never" statements are replaced by the result over nine crossings.

**Not done or not checked:**
- No `.mp4` or `.srt` was re-rendered, as you asked. The chapter 8 film still speaks the old numbers (132.7, 96.7, ρ 68), the chapter 1 subtitles still say 0.3451 ± 0.0061, and the chapter 7 narration still says the soft pair changes nothing.
- I did not look at the rebuilt pages by eye. The swapped seven-point grid, the moved arrow in `decay_tables.tex` and the relabelled rings in `genjet_irsafety.tex` deserve a glance.
- I read the chapter 2–3 diff myself; for chapters 4–8 I relied on each fixer's report, the record comparisons and the clean build.
- The TOTEM reference I added (79.5 ± 1.8 mb, arXiv:1712.06153) is from memory; please check the bib entry.
- The experiment copy of the hadronisation script (with switches for each chapter 5 deviation) is the natural starting point for the later regeneration, but it sits in a session scratch folder that may be cleared. Say if you want it copied into `example/Had/Standalone/`.
