# Chapter 11 companion: The Customs Form (DIGI)

> **Skeleton (2026-11-15).** The sections below follow Constitution rule D; every
> *TODO* marks content still to come. The chapter itself is
> [`chapter_11_DIGI.tex`](chapter_11_DIGI.tex).

This is the technical companion of chapter 11, *The Customs Form: Digitisation*. The chapter gives
the physics; this page says how to run the scripts, how to read what they write, how
each number of the official packages was obtained, and lists every table the chapter
quotes with the command that produced it. All files are in
[`example/DIGI/`](../example/DIGI/) of this repository unless stated otherwise; the
CMSSW side (the DIGI step of the mixed event) is in [JitJetSW](https://github.com/ravindkv/JitJetSW) and is
written up there, see section 4.3.

The conditions are those of Run 3 in 2024, as in chapter 8: 13.6 TeV, mu = 60, the
2024 beam spot and the 2024 CMS detector (`CMSSW_15_0_5`). The HL-LHC appears only in
the exercises and the open problems.

| | |
|---|---|
| Stage | the mixed SimHits of chapter 10 turned into charge, pulses, pedestals, noise and ADC counts, with zero suppression |
| Standalone script | `example/DIGI/Standalone/standalone_DIGI.py` (TODO: to be written) |
| Input | the mixed SimHits of chapter 10 |
| Event record | `example/DIGI/event_DIGI.*` (TODO) |
| Learning-mode log | `example/DIGI/Standalone/standalone_DIGI.log` (`-v`, seed 1) and its summary `standalone_DIGI.json` (TODO) |
| Validation with official packages | `example/DIGI/Standalone/validate_DIGI.py`: the digitiser models with the 2024 conditions (TODO) |
| CMSSW figure | `example/DIGI/CMSSW/*_11_DIGI.pdf`, the lower panel of the chapter's display (TODO) |
| CMSSW write-up (JitJetSW) | physics details: [`chapter_11_DIGI.tex`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_11_DIGI.tex); technical companion: [`chapter_11_DIGI.md`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_11_DIGI.md) |
| Film | scene 11 of the [JitJet playlist](https://www.youtube.com/playlist?list=PLbSW20iyZFd4) (TODO) |

Contents: [1. Running the standalone script](#1-running-the-standalone-script) ·
[2. Reading the record](#2-reading-the-record) ·
[3. Drawing the event and rendering the film](#3-drawing-the-event-and-rendering-the-film) ·
[4. Validation from official packages](#4-validation-from-official-packages) ·
[5. The full tables](#5-the-full-tables)

## 1. Running the standalone script

TODO (D1): what the script does, its command and options, and what `-v` and `-vv`
print.

```bash
python3 example/DIGI/Standalone/standalone_DIGI.py -i <input> -o example/DIGI/event_DIGI.<ext> -v
```

### The learning modes

TODO: what `-v` prints, block by block, and what `-vv` adds; the `-v` log of seed 1 is
the source of every number in the chapter's theory section.

## 2. Reading the record

TODO (D2): the rows the record adds to the chapter-10 record (the digis of our jet: samples, pedestals, gains and the channels kept), with the
excerpts the chapter quotes numbers from.

## 3. Drawing the event and rendering the film

TODO (D3): the commands for the (eta, phi) display `example/DIGI/journey_DIGI.pdf`,
the theory plots and the film with its stills.

### The (eta, phi) displays

TODO.

### The theory plots

TODO: one plot per equation of the theory section.

### The film

TODO.

## 4. Validation from official packages

### 4.1 The standalone script against the digitiser models

TODO (D4).

### 4.2 The same event on identical input

TODO.

### 4.3 CMSSW

The CMSSW side of this stop is one job of
[JitJetSW](https://github.com/ravindkv/JitJetSW), `test/11_DIGI/run_11_DIGI.sh`
(CMSSW_15_0_5, 13.6 TeV, run 1, event 1 with the default seeds): the DIGI step of the mixed event.
Its commands, configurations, logs and tables are documented there, and this page does
not repeat them:

- physics details: [`chapter_11_DIGI.tex`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_11_DIGI.tex);
- technical companion: [`chapter_11_DIGI.md`](https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_11_DIGI.md).

The links are also in [`example/DIGI/CMSSW/README.md`](../example/DIGI/CMSSW/README.md).

What the chapter reads from the output (`test/11_DIGI/out/`):

| number in the chapter | value | file |
|---|---|---|
| TODO | | |

## 5. The full tables

TODO (D5): every number the chapter quotes, with the command that produced it.

### 5.1 The agreement table

TODO.

### 5.2 The energy ledger

TODO.
