# Chapter 3 companion: Whose Child Is It? (MnM)

This is the technical companion of chapter 3, *Whose Child Is It? Matching
and Merging*. The chapter is an interlude: our event is one leading-order
2 -> 2 process with no extra-parton subprocess and no NLO matrix element, so
no matching or merging is applied to it, there is no standalone script, no
validation run and no new record. This page says what there is instead: the
film, the ledger row that repeats itself, and the settings of the official
packages that the chapter names. All files are in
[`example/MnM/`](../example/MnM/) unless stated otherwise.

| | |
|---|---|
| Stage | matching and merging: not applied to our event, by construction |
| Standalone script | none at this stop |
| Event record | unchanged: [`example/PS/event_PS.lhe`](../example/PS/event_PS.lhe) of chapter 2 |
| Film | [`example/MnM/animate_whose_child.py`](../example/MnM/animate_whose_child.py), scene 3 of the [JitJet playlist](https://www.youtube.com/playlist?list=PLbSW20iyZFd4) |
| Sketches | `chapter/tikz/mm_double_counting.tex`, `chapter/tikz/mm_recipes.tex` |

Contents: [1. Running the standalone script](#1-running-the-standalone-script) ·
[2. Reading the record](#2-reading-the-record) ·
[3. Drawing the event and rendering the film](#3-drawing-the-event-and-rendering-the-film) ·
[4. Validation from official packages](#4-validation-from-official-packages) ·
[5. The full tables](#5-the-full-tables)

## 1. Running the standalone script

There is none. A `HardQCD:qqbar2bbbar` sample is one leading-order 2 -> 2
process; every parton beyond the b and bbar has exactly one parent, the
shower of chapter 2, and there is nothing to match it to. The two
experiments the chapter's exercises propose are run with the shower script
of chapter 2:

```bash
cd example/PS/Standalone
python3 standalone_parton_showering.py --ptmax 396 --repeat 2000 --quiet --history power.json   # the undamped "power shower"
python3 standalone_parton_showering.py --repeat 2000 --quiet --history default.json
```

`--ptmax 396` starts the shower at the kinematic limit, sqrt(sHat), instead
of the factorisation scale 113.32 GeV; the `--history` JSON lists every
branching of every shower with its pT, so the share of events with a
branching above a merging scale Q_cut is one line of Python over it.

## 2. Reading the record

Nothing changes. The record of this stop is `example/PS/event_PS.lhe`, read
exactly as in the [chapter 2 companion](chapter_02_PS.md#2-reading-the-record).
What the chapter does read from it is the list of the 38 winning branchings
in the `-v` log of chapter 2, against a merging scale of 30 GeV:

```text
    1 ISR q<-g      side A(+z)  g    -> ubar + u     pT =   53.877   above Q_cut: a matrix-element parton in a merged sample
    2 ISR q<-q      side B(-z)  u    -> u    + g     pT =   51.361   above
    3 ISR g<-g      side A(+z)  g    -> g    + g     pT =   42.157   above
    4 FSR q->qg     bbar  against g          pT =   34.218   above
    5 FSR q->qg     b     against g          pT =   32.070   above
    6 FSR g->gg     g     against g          pT =   25.620   below: the shower's
   ...
   38 FSR q->qg     dbar  against g          pT =    0.644   below
```

Five branchings above 30 GeV, 33 below, 32 of them below 6 GeV.

## 3. Drawing the event and rendering the film

There is no new event display; the (eta, phi) picture of the stage is that
of chapter 2. `animate_whose_child.py` is the narrated film of the stop: the
road of the book with its stops, the cart carrying our event, the sign that
says who stop 3 is for (MLM, CKKW-L, MC@NLO, POWHEG, FxFx), the samples that
do get off here, the card "not for us" with the unchanged ledger row, and
the cart rolling on to stop 4. The script clock is 12.5 s in three beats;
each phase of a beat lasts at least its line of narration (the voice and the
score come from `../soundtrack.py`, shared by all the films), so the narrated
film is about 110 s and the silent one 12.5 s. The film imports no physics
script; the numbers on the screen are the ledger row.

```bash
cd example/MnM
python3 animate_whose_child.py                                   # -> animate_whose_child.mp4 + .srt (1920x1080, 24 fps, ~110 s)
python3 animate_whose_child.py --fast                            # low-resolution preview
python3 animate_whose_child.py --print-script                    # the narration, to record it yourself
python3 animate_whose_child.py --no-narration                    # music only, the 12.5 s script clock
python3 animate_whose_child.py --stills animate_whose_child_stills.pdf --no-video   # one frame per beat (make figures)
```

It needs matplotlib and `ffmpeg`, plus `edge-tts` for the voice (the clips
are cached in `build/narration/MnM/`); rendering takes about ten minutes.
The `.mp4` and `.srt` are not committed, the stills PDF is: the chapter
prints its three pages as one figure.

| Option | Meaning | Default |
|---|---|---|
| `-o` | the output, `.mp4` (needs ffmpeg) or `.gif` (silent) | `animate_whose_child.mp4` |
| `--fps`, `--dpi` | frame rate and resolution (150 dpi gives 1920x1080) | 24, 150 |
| `--step`, `--fast` | render every step-th frame; `--fast` is `--step 4 --dpi 60` | 1 |
| `--stills`, `--no-video`, `--frames-dir` | key frames per beat, skip the video, PNG frames instead of a video | |
| `--jobs` | render the frames in this many processes | 4 |
| `--print-script`, `--no-narration`, `--audio-only` | the narration text, music only, the soundtrack alone | |

## 4. Validation from official packages

There is nothing to validate at this stop: neither the standalone script,
nor the CMSSW configuration of chapter 1, nor the MadGraph run of chapter 1
applies any matching or merging to our event. What the official packages do
offer, and the chapter names, are the settings of the recipes used for
other CMS samples; they are listed here so that they can be found in a
fragment.

| Recipe | Where it is set | Settings named in the chapter | CMS use |
|---|---|---|---|
| plain 2 -> 2 showering (our event) | PYTHIA | `PTmaxMatch = 1`: the shower stops at the factorisation scale (the CMSSW default); `= 2` is the power shower to the kinematic limit | `HardQCD` multijet samples |
| MLM | MadGraph `run_card.dat` + PYTHIA | `xqcut` (MadGraph), `JetMatching:scheme = 1`, `JetMatching:qCut` (19 to 30 GeV), `JetMatching:nJetMax` | leading-order MadGraph multijet, Drell-Yan, W+jets |
| CKKW-L, UMEPS | PYTHIA `Merging:` block | `Merging:doKTMerging`, `Merging:TMS`, `Merging:nJetMax`, `Merging:Process` | not in CMS production samples |
| MC@NLO | MadGraph5_aMC@NLO | `--` (the subtraction is built into the NLO event file; negative weights in `genWeight`) | NLO samples of MadGraph5_aMC@NLO |
| POWHEG | POWHEG-BOX + PYTHIA | `SCALUP` of the LHE event bounds the shower; `SpaceShower:pTmaxMatch = 1`, `TimeShower:pTmaxMatch = 1` | inclusive ttbar, single top |
| FxFx | MadGraph5_aMC@NLO + PYTHIA | `JetMatching:doFxFx = on`, `JetMatching:qCut = 30`, `JetMatching:nQmatch` | ttbar+jets, W/Z+jets (e.g. `DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8`) |

The third exercise of the chapter asks for these lines in a CMS
Drell-Yan FxFx fragment; they are in the `processParameters` block of the
`_cfi.py` fragment, next to the CP5 settings read in the chapter 1
companion.

## 5. The full tables

### 5.1 The energy ledger

The row repeats the previous one; it is there so that the chain of stages
is complete.

| Stage | pT [GeV] | m [GeV] | n_const | delta pT [GeV] |
|---|---:|---:|---:|---:|
| b family after the shower (dR < 0.4) | 100.45 | 13.52 | 3 | |
| b family after matching and merging (none applied) | 100.45 | 13.52 | 3 | 0 |

### 5.2 The branchings of our event against a merging scale

From the `-v` log of chapter 2 (`example/PS/Standalone/standalone_parton_showering.log`),
the winning branchings sorted by pT_evol, against Q_cut = 30 GeV:

| region | branchings | which |
|---|---:|---|
| above 30 GeV (a matrix element's in a merged sample) | 5 | ISR 53.9, 51.4, 42.2 GeV; the twins' first gluons 34.2, 32.1 GeV |
| 6 to 30 GeV | 1 | the g -> gg at 25.6 GeV |
| below 6 GeV | 32 | the rest of the shower |
