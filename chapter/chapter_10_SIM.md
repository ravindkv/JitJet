# Chapter 10 companion: The Crowd at the Border (SIM)

> **Skeleton (2026-11-15).** The sections below follow Constitution rule D; every
> *TODO* marks content still to come. The chapter itself is
> [`chapter_10_SIM.tex`](chapter_10_SIM.tex).

This is the technical companion of chapter 10, *The Crowd at the Border: Mixing the Simulated Hits*. The chapter gives
the physics; this page says how to run the scripts, how to read what they write, how
each number of the official packages was obtained, and lists every table the chapter
quotes with the command that produced it. All files are in
[`example/SIM/`](../example/SIM/) of this repository unless stated otherwise; the
CMSSW side (the `MixingModule` with the detector simulation: the 193 collisions of bx = -1, 0, +1, their primaries, tracker hits and pileup-dominated ECAL cells) is in [JitJetSW](https://github.com/ravindkv/JitJetSW) and is
written up there, see section 4.3.

The conditions are those of Run 3 in 2024, as in chapter 8: 13.6 TeV, mu = 60, the
2024 beam spot and the 2024 CMS detector (`CMSSW_15_0_5`). The HL-LHC appears only in
the exercises and the open problems.

| | |
|---|---|
| Stage | the crowd of chapter 8 transported by Geant4 and its SimHits, in time and out of time, mixed with the family's |
| Standalone script | `example/SIM/Standalone/standalone_SIM.py` (TODO: to be written) |
| Input | the SimHits of chapter 9 and the pileup library of chapter 8, [`example/PU/pileup_library.lhe.gz`](../example/PU/pileup_library.lhe.gz) |
| Event record | `example/SIM/event_SIM.*` (TODO) |
| Learning-mode log | `example/SIM/Standalone/standalone_SIM.log` (`-v`, seed 1) and its summary `standalone_SIM.json` (TODO) |
| Validation with official packages | `example/SIM/Standalone/validate_SIM.py`: Geant4 on the same collisions (TODO) |
| CMSSW figure | `example/SIM/CMSSW/*_10_SIM.pdf`, the lower panel of the chapter's display (TODO) |
| CMSSW write-up (JitJetSW) | physics details: [`chapter_10_SIM.tex`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_10_SIM.tex); technical companion: [`chapter_10_SIM.md`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_10_SIM.md) |
| Film | scene 10 of the [JitJet playlist](https://www.youtube.com/playlist?list=PLbSW20iyZFd4) (TODO) |

Contents: [1. Running the standalone script](#1-running-the-standalone-script) ·
[2. Reading the record](#2-reading-the-record) ·
[3. Drawing the event and rendering the film](#3-drawing-the-event-and-rendering-the-film) ·
[4. Validation from official packages](#4-validation-from-official-packages) ·
[5. The full tables](#5-the-full-tables)

## 1. Running the standalone script

TODO (D1): what the script does, its command and options, and what `-v` and `-vv`
print.

```bash
python3 example/SIM/Standalone/standalone_SIM.py -i <input> -o example/SIM/event_SIM.<ext> -v
```

### The learning modes

TODO: what `-v` prints, block by block, and what `-vv` adds; the `-v` log of seed 1 is
the source of every number in the chapter's theory section.

## 2. Reading the record

TODO (D2): the rows the record adds to the chapter-9 record (the crossing frame: every SimHit with its collision, its crossing and its time), with the
excerpts the chapter quotes numbers from.

## 3. Drawing the event and rendering the film

TODO (D3): the commands for the (eta, phi) display `example/SIM/journey_SIM.pdf`,
the theory plots and the film with its stills.

### The (eta, phi) displays

TODO.

### The theory plots

TODO: one plot per equation of the theory section.

### The film

TODO.

## 4. Validation from official packages

### 4.1 The standalone script against Geant4

TODO (D4).

### 4.2 The same event on identical input

TODO.

### 4.3 CMSSW

The CMSSW side of this stop is one job of
[JitJetSW](https://github.com/ravindkv/JitJetSW), `test/10_SIM/run_10_SIM.sh`
(CMSSW_15_0_5, 13.6 TeV, run 1, event 1 with the default seeds): the `MixingModule` with the detector simulation: the 193 collisions of bx = -1, 0, +1, their primaries, tracker hits and pileup-dominated ECAL cells.
Its commands, configurations, logs and tables are documented there, and this page does
not repeat them:

- physics details: [`chapter_10_SIM.tex`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_10_SIM.tex);
- technical companion: [`chapter_10_SIM.md`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_10_SIM.md).

The links are also in [`example/SIM/CMSSW/README.md`](../example/SIM/CMSSW/README.md).

What the chapter reads from the output (`test/10_SIM/out/`):

| number in the chapter | value | file |
|---|---|---|
| TODO | | |

## 5. The full tables

TODO (D5): every number the chapter quotes, with the command that produced it.

### 5.1 The agreement table

TODO.

### 5.2 The energy ledger

TODO.
