# Chapter 13 companion: The Second Checkpoint (HLT)

> **Skeleton (2026-11-15).** The sections below follow Constitution rule D; every
> *TODO* marks content still to come. The chapter itself is
> [`chapter_13_HLT.tex`](chapter_13_HLT.tex).

This is the technical companion of chapter 13, *The Second Checkpoint: High-Level Trigger*. The chapter gives
the physics; this page says how to run the scripts, how to read what they write, how
each number of the official packages was obtained, and lists every table the chapter
quotes with the command that produced it. All files are in
[`example/HLT/`](../example/HLT/) of this repository unless stated otherwise; the
CMSSW side (the 2025 HLT menu run on the L1-accepted event) is in [JitJetSW](https://github.com/ravindkv/JitJetSW) and is
written up there, see section 4.3.

The conditions are those of Run 3 in 2025, as in chapter 8: 13.6 TeV, mu = 60, the
2025 beam spot and the CMS detector of 2025 (`CMSSW_15_0_5`, `auto:phase1_2025_realistic`,
era `Run3_2025`; the geometry is the `Extended2024` one, which the 2025 global tag reads unchanged). The HL-LHC appears only in
the exercises and the open problems.

| | |
|---|---|
| Stage | the L1-accepted event of chapter 12 at the HLT: online calorimeter and Particle-Flow jets, online corrections and the paths fired |
| Standalone script | `example/HLT/Standalone/standalone_HLT.py` (TODO: to be written) |
| Input | the digis of chapter 11 and the L1 decision of chapter 12 |
| Event record | `example/HLT/event_HLT.*` (TODO) |
| Learning-mode log | `example/HLT/Standalone/standalone_HLT.log` (`-v`, seed 1) and its summary `standalone_HLT.json` (TODO) |
| Validation with official packages | `example/HLT/Standalone/validate_HLT.py`: FastJet and the online correction tables on the same inputs (TODO) |
| CMSSW figure | `example/HLT/CMSSW/*_13_HLT.pdf`, the lower panel of the chapter's display (TODO) |
| CMSSW write-up (JitJetSW) | physics details: [`chapter_13_HLT.tex`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_13_HLT.tex); technical companion: [`chapter_13_HLT.md`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_13_HLT.md) |
| Film | scene 13 of the [JitJet playlist](https://www.youtube.com/playlist?list=PLbSW20iyZFd4) (TODO) |

Contents: [1. Running the standalone script](#1-running-the-standalone-script) ·
[2. Reading the record](#2-reading-the-record) ·
[3. Drawing the event and rendering the film](#3-drawing-the-event-and-rendering-the-film) ·
[4. Validation from official packages](#4-validation-from-official-packages) ·
[5. The full tables](#5-the-full-tables)

## 1. Running the standalone script

TODO (D1): what the script does, its command and options, and what `-v` and `-vv`
print.

```bash
python3 example/HLT/Standalone/standalone_HLT.py -i <input> -o example/HLT/event_HLT.<ext> -v
```

### The learning modes

TODO: what `-v` prints, block by block, and what `-vv` adds; the `-v` log of seed 1 is
the source of every number in the chapter's theory section.

## 2. Reading the record

TODO (D2): the rows the record adds to the chapter-12 record (the online jets, raw and corrected, and the trigger bits), with the
excerpts the chapter quotes numbers from.

## 3. Drawing the event and rendering the film

TODO (D3): the commands for the (eta, phi) display `example/HLT/journey_HLT.pdf`,
the theory plots and the film with its stills.

### The (eta, phi) displays

TODO.

### The theory plots

TODO: one plot per equation of the theory section.

### The film

TODO.

## 4. Validation from official packages

### 4.1 The standalone script against FastJet and the online corrections

TODO (D4).

### 4.2 The same event on identical input

TODO.

### 4.3 CMSSW

The CMSSW side of this stop is one job of
[JitJetSW](https://github.com/ravindkv/JitJetSW), `test/13_HLT/run_13_HLT.sh`
(CMSSW_15_0_5, 13.6 TeV, run 1, event 1 with the default seeds): the 2025 HLT menu run on the L1-accepted event.
Its commands, configurations, logs and tables are documented there, and this page does
not repeat them:

- physics details: [`chapter_13_HLT.tex`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_13_HLT.tex);
- technical companion: [`chapter_13_HLT.md`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_13_HLT.md).

The links are also in [`example/HLT/CMSSW/README.md`](../example/HLT/CMSSW/README.md).

What the chapter reads from the output (`test/13_HLT/out/`):

| number in the chapter | value | file |
|---|---|---|
| TODO | | |

## 5. The full tables

TODO (D5): every number the chapter quotes, with the command that produced it.

### 5.1 The agreement table

TODO.

### 5.2 The energy ledger

TODO.
