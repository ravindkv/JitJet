# Chapter 1 companion: The Ancestral Home (ME)

This is the technical companion of chapter 1, *The Ancestral Home: Protons,
PDFs and the Hard Scattering*. The chapter gives the physics; this page says
how to run the scripts, how to read what they write, how each number of the
official packages was obtained, and lists every table the chapter quotes with
the command that produced it. All files are in
[`example/ME/`](../example/ME/) of this repository unless stated otherwise;
the CMSSW analysers are in [JitJetSW](https://github.com/ravindkv/JitJetSW).

| | |
|---|---|
| Stage | the hard process `q qbar -> b bbar` at 13.6 TeV, pTHat >= 100 GeV |
| Standalone script | [`example/ME/Standalone/standalone_qqbar_bbbar_xsec.py`](../example/ME/Standalone/standalone_qqbar_bbbar_xsec.py) |
| Full integrator (LHAPDF + PYTHIA alpha_s) | [`example/ME/Standalone/qqbar_bbbar_xsec.py`](../example/ME/Standalone/qqbar_bbbar_xsec.py) |
| Event record | [`example/ME/event_ME.lhe`](../example/ME/event_ME.lhe) |
| Learning-mode log | [`example/ME/Standalone/standalone_qqbar_bbbar_xsec.log`](../example/ME/Standalone/standalone_qqbar_bbbar_xsec.log) |
| CMSSW configuration and log | [`example/ME/CMSSW/`](../example/ME/CMSSW/) |
| MadGraph recipe | [`example/ME/MadGraph/qqbar_bbbar_xsec_MadGraph.md`](../example/ME/MadGraph/qqbar_bbbar_xsec_MadGraph.md) |
| Film | scene 1 of the [JitJet playlist](https://www.youtube.com/playlist?list=PLbSW20iyZFd4) |

Contents: [1. Running the standalone script](#1-running-the-standalone-script) ·
[2. Reading the record](#2-reading-the-record) ·
[3. Drawing the event and rendering the film](#3-drawing-the-event-and-rendering-the-film) ·
[4. Validation from official packages](#4-validation-from-official-packages) ·
[5. The full tables](#5-the-full-tables)

## 1. Running the standalone script

`standalone_qqbar_bbbar_xsec.py` contains the calculation of the chapter's
theory section and nothing else. It needs Python 3.10 or newer with NumPy and
SciPy; no LHAPDF, no PYTHIA. The PDF grid (member 0 of `NNPDF31_nnlo_as_0118`
on LHAPDF's own knots, restricted to the phase space of the default run) and
PYTHIA's second-order alpha_s curve are embedded as numeric tables between the
markers `BEGIN/END GENERATED TABLES`, produced by `dump_standalone_tables.py`
in an environment that has both libraries (section 4.1).

A map of the file, with the step numbers of the chapter's recipe:

| Function | What it does | Recipe step |
|---|---|---|
| `TablePDF` | bicubic spline of x f(x, Q^2) in (ln x, ln Q^2), with PYTHIA's freezing, clipping and symmetric-heavy-sea conventions | 3 |
| `TableAlphaS` | monotone cubic in ln Q^2 of PYTHIA's alpha_s, refusing to extrapolate | 3 |
| `dsigma_dcostheta` | the Born matrix element d sigmaHat / d cos theta, vectorised | 3 |
| `born_weights` | the unit cube to (ln tau, Y, cos theta), the kinematics, the ingredients and the weight w_q | 1 to 4 |
| `integrate` | the scrambled Sobol passes, the average and the error | 5 and 6 |
| `main` | command line, sanity checks and the printout | 7 |

From the repository root:

```bash
python3 example/ME/Standalone/standalone_qqbar_bbbar_xsec.py              # one pass, about a second
python3 example/ME/Standalone/standalone_qqbar_bbbar_xsec.py --repeats 8  # with an integration error
python3 example/ME/Standalone/standalone_qqbar_bbbar_xsec.py --ptmin 150 --ptmax 300
```

The first command ends with

```text
TOTAL = 0.345320 +/- nan nb
      = 345.320 +/- nan pb
  d dbar: 0.118568 +/- nan nb
  u ubar: 0.178315 +/- nan nb
  ...
```

(the `nan` says that one pass has no error estimate). The second gives
345.322 +/- 0.001 pb after eight scrambled passes; the third selects a window
in pTHat, which is how sliced production samples are stitched together. The
standalone and the full integrator agree to 5e-6 for the same Sobol seed, so
the tables lose nothing; if the PDF set, the coupling or the kinematic range
is changed, the tables must be regenerated (section 4.1), and the script says
so.

| Option | Meaning | Default |
|---|---|---|
| `--ecm` | centre-of-mass energy in GeV | 13600 |
| `--mb` | the b mass in the matrix element, GeV | 4.8 |
| `--ptmin`, `--ptmax` | the window in pTHat, GeV | 100, none |
| `--power` | 2^power Sobol points per pass | 16 |
| `--repeats` | number of scrambled passes (the error needs at least 2) | 1 |
| `--seed` | the Sobol scrambling seed | 24680 |
| `-v`, `--verbose` | the learning mode, below | off |
| `--event` | the hard-process record narrated by `-v` | `../event_ME.lhe` |

### The learning mode: where every number of the chapter comes from

With `-v` the script narrates the calculation before it performs it, in
three numbered sections: [1] the constants of the integral; [2] our own event,
read from `event_ME.lhe`, pushed through `born_weights` one factor at a time;
and [3] the first Sobol pass with its running average after 4, 8, ...,
65536 points, the weight statistics and the per-flavour shares.

```bash
python3 example/ME/Standalone/standalone_qqbar_bbbar_xsec.py -v
python3 example/ME/Standalone/standalone_qqbar_bbbar_xsec.py -v --event my_record.lhe
```

The output of the first command is committed as
`standalone_qqbar_bbbar_xsec.log` (about a hundred lines), and every number
quoted in the theory section of the chapter is taken from it. This script has
no `-vv` level: the whole narration fits in one. Section 2 below reproduces
the lines the chapter quotes.

## 2. Reading the record

### The hard-process record

`event_ME.lhe` is PYTHIA's hard-process listing for the event followed through
the book, copied from the log of the CMSSW job of section 4.2 (it is not a
Les Houches file despite the extension; the status codes are PYTHIA's):

```text
    no         id  name            status     mothers   daughters     colours      p_x        p_y        p_z         e          m
     0         90  (system)           -11     0     0     0     0     0     0      0.000      0.000      0.000  13600.000  13600.000
     1       2212  (p+)               -12     0     0     3     0     0     0      0.000      0.000   6800.000   6800.000      0.938
     2       2212  (p+)               -12     0     0     4     0     0     0      0.000      0.000  -6800.000   6800.000      0.938
     3         -2  (ubar)             -21     1     0     5     6     0   101      0.000      0.000     14.785     14.785      0.000
     4          2  (u)                -21     2     0     5     6   102     0      0.000      0.000  -2652.934   2652.934      0.000
     5         -5  bbar                23     3     4     0     0     0   101     62.071     94.690  -2413.001   2415.661      4.800
     6          5  b                   23     3     4     0     0   102     0    -62.071    -94.690   -225.148    252.059      4.800
                                   Charge sum:  0.000           Momentum sum:      0.000      0.000  -2638.149   2667.719    396.101
```

Reading the columns: status `-12` marks the two beam protons, `-21` the two
incoming partons after they have been resolved out of the protons, and `23`
the two outgoing partons of the hard process; a negative status means the
particle has been used up, a positive one that it is still there for the next
stage. The colour columns show the colour flow: the ubar carries anticolour
101, handed to the bbar, and the u carries colour 102, handed to the b; the
gluon in between is internal and never appears. The momentum sum at the
bottom is that of the two incoming partons: p_z = 14.785 - 2652.934 =
-2638.15 GeV, the whole system recoils backward, and its mass, 396.1 GeV, is
sqrt(sHat). Momenta and masses are in GeV.

### The learning-mode log, section by section

Section [1], the constants of the integral:

```text
    s = 13600^2 = 1.8496e+08 GeV^2      mb = 4.8 GeV      pTHat_min = 100 GeV
    sHat_min = 4 (mb^2 + pTHat_min^2) = 40092.16 GeV^2,  sqrt(sHat_min) = 200.23 GeV
    tau_min  = sHat_min / s = 2.1676e-04
    ln(1/tau_min) = 8.4367   (the length of the ln tau interval, the first Jacobian factor)
    Y_max(tau_min) = -ln(tau_min)/2 = 4.2184   (the widest rapidity range, at threshold)
    conversion 1 GeV^-2 = 3.89379372e+08 pb
```

Section [2], our event through `born_weights`, the lines the chapter's theory
quotes (the full section also prints the ten PDF values and their change at
Q^2/4 and 4 Q^2):

```text
    incoming from beam +z: ubar  E = 14.785 GeV  ->  x1 = 2.1743e-03
    incoming from beam -z:    u  E = 2652.934 GeV  ->  x2 = 3.9014e-01
  step 1  the integration variables (eq. tauY, maps)
    tau = x1 x2 = 8.4826e-04        ln tau = -7.0723        Y = ln(x1/x2)/2 = -2.5949
    Y_max(tau) = -ln(tau)/2 = 3.5362   (|Y| = 2.595 uses 73.4% of the allowed range)
    cos theta* of the bbar in the partonic frame = -0.8203  (theta* = 2.533 rad)
    c_max = sqrt(1 - pTHat_min^2/p*^2) = 0.8631
    unit-cube coordinates of this point: u1 = 0.1617, u2 = 0.1331, u3 = |cos theta*|/c_max = 0.9505
  step 2  the kinematics
    sHat = tau s = 156894.5 GeV^2,  sqrt(sHat) = 396.10 GeV   (the record's momentum-sum mass)
    p* = sqrt(sHat/4 - mb^2) = 197.991 GeV  (from the boosted record: 197.995 GeV)
    pTHat = p* sin theta* = 113.229 GeV  (record: 113.221 GeV)
    Q^2 = pTHat^2 + mb^2 = 12842.0 GeV^2,  Q = 113.32 GeV
    rho = 4 mb^2/sHat = 5.8740e-04,  beta = sqrt(1-rho) = 0.999706
    y* = artanh(beta cos theta*) = -1.1571;  y_bbar = Y + y* = -3.7520,  y_other = Y - y* = -1.4378
  step 3  the ingredients at this point
    alpha_s(Q^2) [TableAlphaS, PYTHIA 2nd order] = 0.11425
    alpha_s(Q^2/4) = 0.12715   -> alpha_s^2 ratio to the central value 1.2386
    alpha_s(4 Q^2) = 0.10416   -> alpha_s^2 ratio to the central value 0.8311
    x f(x, Q^2) [TablePDF.xf]           at x1 = 2.1743e-03        at x2 = 0.3901
       u: 1.25664    ubar: 1.14546       u: 0.21068    ubar: 0.00122
    at Q^2/4: x1 f_ubar = 1.04495 (-8.8%),  x2 f_u = 0.23222 (+10.2%),  product +0.6%
    at 4 Q^2: x1 f_ubar = 1.23721 (+8.0%),  x2 f_u = 0.19282 (-8.5%),  product -1.1%
    the Jacobian cancellation (eq. xf): f_ubar(x1) = xf/x1 = 526.83, f_u(x2) = 0.5400;
      f f tau = 0.24133 = [x1 f][x2 f] = 0.24133
    d sigma_hat / d cos theta [dsigma_dcostheta] = 18.916 pb at cos theta* = -0.8203
    sigma_hat (integrated over cos theta) = 4 pi alpha_s^2 beta (2+rho)/(27 sHat) = 30.157 pb
    fraction of sigma_hat inside |cos theta| <= c_max: 0.8081  (the cut keeps 80.8% at this sHat)
    solid-angle fraction removed by the cut: 0.1369
  step 4  the weight (eq. weight)
    Jacobian: ln(1/tau_min) x 2 Y_max x 2 c_max = 8.4367 x 7.0723 x 1.7262 = 102.995
      w_d = 102.995 x 18.916 pb x (1.21237 x 0.00295 + 1.14825 x 0.06629) = 155.268 pb
      w_u = 102.995 x 18.916 pb x (1.25664 x 0.00122 + 1.14546 x 0.21068) = 473.161 pb
      w_s = 102.995 x 18.916 pb x (1.06545 x 0.00060 + 1.06637 x 0.00532) = 12.298 pb
      w_c = 102.995 x 18.916 pb x (0.84207 x 0.00278 + 0.84207 x 0.00278) = 9.114 pb
      w_b = 102.995 x 18.916 pb x (0.60747 x 0.00038 + 0.60747 x 0.00038) = 0.902 pb
    sum_q w_q = 650.742 pb at this point  (the cross section is the average of such weights over the cube)
    the ubar(x1) u(x2) orientation alone: 470.164 pb, 99.4% of w_u
```

Section [3], the first Sobol pass (seed 24680, 2^16 points):

```text
    weights > 0: 65505 (99.95%)
    weight statistics [pb]: min 0, median 1.027, mean 345.3202, max 4576
      the largest 1% of the weights carry 11.3% of the integral
      the largest 10% of the weights carry 74.6% of the integral
    variance of the weights: std/mean = 2.365;  a pseudo-random estimate would have error std/sqrt(N) = 3.1903 pb
    running average after N points:
      N = 2^2  =      4:     26.4846 pb
      N = 2^3  =      8:    401.4254 pb
      N = 2^4  =     16:    394.5023 pb
      N = 2^8  =    256:    338.7253 pb
      N = 2^10 =   1024:    345.2173 pb
      N = 2^14 =  16384:    345.2796 pb
      N = 2^15 =  32768:    345.3190 pb
      N = 2^16 =  65536:    345.3202 pb
    per incoming flavour, this pass:
      d dbar:  118.5680 pb  (34.3%)
      u ubar:  178.3155 pb  (51.6%)
      s sbar:   30.2669 pb  (8.8%)
      c cbar:   12.6964 pb  (3.7%)
      b bbar:    5.4734 pb  (1.6%)
    TOTAL this pass: 345.3202 pb
      u1 in [0, 0.1] (just above threshold)   : sqrt(sHat) from   200.2 to   305.3 GeV,  62.9% of the integral
      u1 in [0.1, 0.3]                        : sqrt(sHat) from   305.3 to   709.8 GeV,  35.3% of the integral
      u1 in [0.3, 1]                          : sqrt(sHat) from   709.8 to 13599.6 GeV,  1.8% of the integral
```

## 3. Drawing the event and rendering the film

### The (eta, phi) display

`plot_journey_ME.py` draws the hard process of `event_ME.lhe` in the
(eta, phi) plane in the layout of the CMSSW `genPartAnalyzer` of the later
chapters (marker size growing with log pT, the tracker, endcap and HF bands,
the incoming partons and their momentum fractions at the two edges), and
prints the kinematics table of section 5.6:

```bash
python3 example/ME/plot_journey_ME.py -i example/ME/event_ME.lhe -o example/ME/journey_ME.pdf
python3 example/ME/plot_journey_ME.py --png          # also a PNG next to the PDF
```

### The theory plots

`plot_theory_ME.py` imports the standalone script and uses nothing but its
tables and its `born_weights`; it writes `theory_qqbar_bbbar.pdf`, one page
per theory subsection of the chapter:

| Page | Chapter figure | Content |
|---|---|---|
| 1 | the factorisation formula | the integrand in sqrt(sHat) and in the pair rapidity Y, by incoming flavour |
| 2 | better variables | the allowed triangle in (log x1, log x2); the share of the integral per bin of u1 for ln tau and for tau sampling |
| 3 | the PDFs | x f(x, Q^2) at the event's scale; the two densities of our event against Q^2 |
| 4 | the matrix element | the angular distribution and the cut; sigmaHat(sHat); c_max and the surviving fraction |
| 5 | the flavour shares | the stacked shares against the pTHat cut; the total and its local power n |
| 6 | scales and the coupling | PYTHIA's alpha_s(Q^2); the distribution of the cross section in the scale Q |
| 7 | from formula to number | the weight distribution; the running average, Sobol against pseudo-random; the error against N |
| 8 | scale uncertainties | the cross section against the scale factor; the seven points on the (mu_R, mu_F) grid |

```bash
python3 example/ME/Standalone/plot_theory_ME.py        # -> theory_qqbar_bbbar.pdf (matplotlib)
make figures                                           # the same, from the book directory
```

Page 8 is the only one that cannot be drawn from the embedded tables: it
reads `scale_scan.json`, written by `scan_scales.py` with the full integrator
of section 4.1 (`.venv/bin/python scan_scales.py`, about two minutes: mu_R,
mu_F and both together for factors 2^(k/4), k = -4..4).

### The film

`animate_standalone_qqbar_bbbar_xsec.py` imports the standalone script and
renders a narrated four-scene film of about two and a half minutes from its
tables, its interpolators and its `born_weights` function, so every number on
the screen, and every number the narrator says, comes from the same code that
prints the cross section. The scenes, each named after the equation of the
chapter that explains it:

1. **collision**: our event in the partonic centre-of-mass frame, the
   pTHat >= 100 GeV cut drawn as a tube around the beam, then the boost by
   the pair rapidity Y = -2.59 into the laboratory (the tips slide along the
   tube, because pT is boost invariant).
2. **cube**: Sobol points in the unit cube (u1, u2, u3), coloured by log10 of
   their weight, morphed into (log10 x1, log10 x2, |cos theta|).
3. **surface**: the integrand with cos theta integrated out over the
   triangle; its volume is the cross section, and our event is marked on it.
4. **converge**: one Sobol pass filling the cube while the running average,
   per incoming flavour and in total, settles on 345 pb, with PYTHIA's
   345.1 +/- 6.1 pb as a band.

It needs matplotlib in addition to NumPy and SciPy, `ffmpeg` for an `.mp4`
(a `.gif` needs only Pillow and is silent), and for the voice the `edge-tts`
package (`pip install edge-tts`; it fetches Microsoft's neural voices over the
network). The narration and the score live in `example/soundtrack.py`, shared
by all the films of the book; each phase of a scene lasts at least as long as
its line, so `--no-narration` gives the silent, half-minute cut.

```bash
cd example/ME/Standalone
python3 animate_standalone_qqbar_bbbar_xsec.py                  # the film and its .srt, a few minutes
python3 animate_standalone_qqbar_bbbar_xsec.py --fast           # low-resolution preview
python3 animate_standalone_qqbar_bbbar_xsec.py --print-script   # the narration, to record it yourself
python3 animate_standalone_qqbar_bbbar_xsec.py --no-narration   # music only, about 35 s
python3 animate_standalone_qqbar_bbbar_xsec.py --scene surface -o surface.gif
python3 animate_standalone_qqbar_bbbar_xsec.py --stills animate_standalone_qqbar_bbbar_xsec_stills.pdf --no-video
```

The last command writes the key frames (one page per scene) that the chapter
prints; `make figures` runs it. The physics options `--ecm`, `--mb`,
`--ptmin`, `--ptmax`, `--power` and `--seed` are those of the standalone
script, and `--event` points scene 1 at another hard-process record, so you
can film your own event. The narration clips are cached in
`build/narration/ME/` (`--cache-dir`), and `--jobs` renders the frames in
parallel.

## 4. Validation from official packages

### 4.1 The same integral with LHAPDF and PYTHIA's alpha_s

`qqbar_bbbar_xsec.py` does the same integral with LHAPDF for the PDFs and
PYTHIA's own `AlphaStrong` class for the coupling, and offers every knob of
the chapter as a command-line option: upper and lower pTHat and mHat cuts,
|y| or |eta| cuts on both quarks, the number of incoming flavours
(`--nquark`), all five PYTHIA scale codes and their multipliers, the
convenience factors `--mur-factor` and `--muf-factor`, a K factor, the PDF
member (`--member`), alpha_s(m_Z) (`--alphas-mz`), the coupling backend and
order, and the option to keep negative PDF values (`--signed-pdfs`). It
needs the LHAPDF Python bindings and either the `pythia8mc` package or an
exported coupling table; the
[`README.md`](../example/ME/Standalone/README.md) of the directory explains
how to build such an environment (the one used here: Python 3.13, LHAPDF
6.5.4, `NNPDF31_nnlo_as_0118`, PYTHIA 8.315, in a `.venv` that Git ignores)
and how to export the actual settings of a running PYTHIA instance with the
header `export_pythia_xsec.h`, so that one integrates precisely what the
generator was configured to do rather than what a log file suggests.

```bash
cd example/ME/Standalone
.venv/bin/python qqbar_bbbar_xsec.py --output result.json      # 345.324 +/- 0.001 pb, flavour breakdown
.venv/bin/python qqbar_bbbar_xsec.py --mur-factor 2 --muf-factor 2
.venv/bin/python test_integrator.py                           # normalisation and phase-space checks
.venv/bin/python dump_standalone_tables.py                    # regenerate the embedded tables
.venv/bin/python scan_scales.py -o scale_scan.json            # the seven-point scan, about 2 min
```

The first command prints 345.324 +/- 0.001 pb and writes the flavour
breakdown of section 5.2; the second is one row of section 5.4; the third
checks the analytic total cross section, the threshold behaviour, the scale
semantics and one-dimensional reference integrals. Run the scripts with the
explicit interpreter path: a shell alias such as `alias python=/usr/bin/python3`
silently overrides a virtual environment.

### 4.2 CMSSW: the GEN step

In CMS the hard process is one step of a production chain driven by
`cmsDriver.py`. The configuration `CMSSW/step1_GEN_cfg.py` was generated with
the command in its header, here broken into lines:

```bash
cmsDriver.py Configuration/GenProduction/python/BBbar_14TeV_TuneCP5_cfi.py \
  --conditions auto:phase1_2024_realistic --era Run3_2024 --geometry DB:Extended \
  -n 1 -s GEN --datatier GEN --eventcontent FEVTDEBUG \
  --fileout file:output_step1_GEN.root --python_filename step1_GEN_cfg.py \
  --no_exec --beamspot DBrealistic \
  --customise_commands "process.generator.comEnergy = cms.double(13600.0)"
```

The pieces: a *fragment* (the `_cfi.py` file, which carries the PYTHIA
settings), the detector *conditions* and *era*, the *step* (`GEN` only), the
output data tier and event content, and a customisation that overrides the
fragment's 14 TeV with the Run 3 energy of 13.6 TeV. To run it:

```bash
cmsrel CMSSW_15_0_5
cd CMSSW_15_0_5/src
cmsenv
cmsRun step1_GEN_cfg.py 2>&1 | tee step1_GEN_cfg.log
```

**Reading the configuration.** Most of the file is boilerplate that every
CMSSW job carries. The physics is in the module `process.generator`, an
`EDFilter` of type `Pythia8ConcurrentGeneratorFilter`, and in particular in
three blocks of its `PythiaParameters`:

- `processParameters`: the three lines that define the process,
  `HardQCD:gg2bbbar = off`, `HardQCD:qqbar2bbbar = on` and
  `PhaseSpace:pTHatMin = 100.`.
- `pythia8CP5Settings`: the CMS tune. It fixes the PDF through `PDF:pSet` to
  NNPDF3.1 NNLO, and sets the four couplings (matrix element, the two showers
  and the multiparton interactions) to alpha_s(m_Z) = 0.118 at second order.
  It also carries the multiparton-interaction and colour-reconnection
  parameters of chapter 4.
- `pythia8CommonSettings`: production defaults such as the tolerance on
  energy-momentum conservation and the lifetime cut-off for particle decays.

Other lines worth noticing: `maxEvents.input = 1000` (the fragment name says
one event, the job was run with a thousand); the `comEnergy` override at the
end; the output module writing `FEVTDEBUG` content to `output_step1_GEN.root`;
and the `customiseJitJet` line from
[JitJetSW](https://github.com/ravindkv/JitJetSW) that hooks in the book's own
analysers, used from chapter 2 on.

**Reading the log.** PYTHIA writes two things to `CMSSW/step1_GEN_cfg.log`
that close the loop with the chapter's recipe. For the first event it prints
the Info Listing: the two incoming partons with their x and PDF values, the
partonic invariants, pTHat and thetaHat, and the coupling at the scale Q^2,
which are the quantities of section 5.5:

```text
 In 1: id =   -2, x =  2.174e-03, pdf =  1.145e+00 at Q2 =  1.284e+04.
 In 2: id =    2, x =  3.901e-01, pdf =  2.107e-01 at same Q2.

 Subprocess q qbar -> b bbar with code 124 is 2 -> 2.
 It has sHat =  1.569e+05,    tHat = -1.428e+05,    uHat = -1.409e+04,
       pTHat =  1.132e+02,   m3Hat =  4.800e+00,   m4Hat =  4.800e+00,
    thetaHat =  2.533e+00,  phiHat =  9.905e-01.
     alphaEM =  7.837e-03,  alphaS =  1.143e-01    at Q2 =  1.284e+04.
```

Immediately after it comes the hard-process listing, which is the record of
section 2. At the end of the job it prints the Event and Cross Section
Statistics table: how many phase-space points were tried, how many selected
and accepted, and the cross section with its error, in mb. About three
phase-space points are tried for each unweighted event that is kept; the cross
section is the average weight of the tried points, not the acceptance ratio,
exactly as in the chapter's recipe.

```text
 | Subprocess                                    Code |            Number of events       |      sigma +- delta    |
 |                                                    |       Tried   Selected   Accepted |     (estimated) (mb)   |
 | q qbar -> b bbar                               124 |        2910       1000       1000 |   3.451e-07  6.078e-09 |
```

With 1 mb = 10^9 pb this is 345.1 +/- 6.1 pb, 0.04 standard deviations from
the standalone 345.32 pb. The 1.8% statistical error of a thousand-event run
is the largest number in the agreement table, and still ten times smaller
than the theoretical uncertainty of the chapter.

### 4.3 MadGraph5_aMC@NLO

A second generator is the best check of the first, because it shares none of
its code. The recipe in
[`MadGraph/qqbar_bbbar_xsec_MadGraph.md`](../example/ME/MadGraph/qqbar_bbbar_xsec_MadGraph.md)
reproduces the number with MadGraph5_aMC@NLO 3.8.0. Download and unpack the
release, start the interpreter, and type:

```text
MG5_aMC> define p = u c d s u~ c~ d~ s~
MG5_aMC> generate p p > b b~
MG5_aMC> output qq_to_bbar
MG5_aMC> launch
```

The first line is the whole physics choice of the chapter in one command: it
redefines the proton as a bag of light quarks and antiquarks, without the
gluon, so that only `q qbar -> b bbar` is generated (and without the b, which
is MadGraph's default). With the default Standard Model there is exactly one
diagram, the s-channel gluon. When `launch` offers to edit the run card, set
the beam energies, the PDF and the b pT cut:

```text
 6800.0   = ebeam1   ! beam 1 total energy in GeV
 6800.0   = ebeam2   ! beam 2 total energy in GeV
 nn23lo1   = pdlabel  ! PDF set
 303600   = lhaid    ! NNPDF31_nnlo_as_0118
 100.0    = ptb      ! minimum pT for the b quark
```

The results summary of the run reads

```text
=== Results Summary for run: run_03 tag: tag_1 ===

Cross-section : 344.3 +- 0.9587 pb
Nb of events  : 10000
```

Conventions that differ from the PYTHIA run, and what each is worth with the
full integrator of section 4.1: four incoming flavours instead of five
(`--nquark 4`: -1.6%); m_b = 4.7 GeV instead of 4.8 (`--mb 4.7`: +0.01%);
MadGraph's default dynamical scale for a 2 -> 2 process, built from the
transverse masses after a kT clustering, instead of pTHat^2 + m_b^2 (a factor
two in mu_R moves the result by about 20%, section 5.4); and alpha_s from
the PDF set instead of PYTHIA's own running, which at this scale agree to
four digits. With `ptb = 30` instead, the same recipe gives about 19.4 nb
against PYTHIA's 18.4 nb: at the lower cut the four-flavour proton, the b
mass and the scale convention matter more.

MadGraph stores the result as weighted or unweighted events in the Les
Houches Event format, whose `<event>` blocks carry the same information as
PYTHIA's hard-process listing: one line per particle with its PDG code, status
(-1 incoming, 1 outgoing), mother indices, colour labels, four-momentum and
mass, plus the event weight and the values of alpha_s and the scale that were
used. `plot_journey_ME.py` reads both formats.

## 5. The full tables

### 5.1 The cross section obtained four ways

The agreement table of the chapter. The first three rows use PYTHIA's
conventions exactly (m_b = 4.8 GeV, five incoming flavours, NNPDF3.1 NNLO,
mu_R^2 = mu_F^2 = pTHat^2 + m_b^2, PYTHIA's own alpha_s running); the
MadGraph row uses that program's defaults for the b mass, the scale and the
coupling, and four incoming flavours. Quoted errors are statistical or
numerical only.

| Method | Details | sigma [pb] | Command |
|---|---|---:|---|
| Standalone script, one pass | LO integral, 2^16 Sobol points | 345.320 | `standalone_qqbar_bbbar_xsec.py` |
| Standalone script, 8 repeats | LO integral, scrambled seeds | 345.322 +/- 0.001 | `standalone_qqbar_bbbar_xsec.py --repeats 8` |
| Full script (LHAPDF, PYTHIA alpha_s) | LO integral, 8 repeats | 345.324 +/- 0.001 | `.venv/bin/python qqbar_bbbar_xsec.py --output result.json` |
| PYTHIA 8.311 in CMSSW_15_0_5 | Monte Carlo, 1000 events | 345.1 +/- 6.1 | `cmsRun step1_GEN_cfg.py`, statistics table of the log |
| MadGraph5_aMC@NLO 3.8.0 | Monte Carlo, 10 000 events | 344.3 +/- 1.0 | the recipe of section 4.3 |

### 5.2 Contribution of each incoming flavour

From the full script, eight repeats of 2^16 Sobol points (`result.json`):

| Incoming pair | sigma [pb] | fraction |
|---|---:|---:|
| u ubar | 178.316 | 51.6% |
| d dbar | 118.568 | 34.3% |
| s sbar | 30.269 | 8.8% |
| c cbar | 12.697 | 3.7% |
| b bbar | 5.474 | 1.6% |
| total | 345.324 +/- 0.001 | |

### 5.3 Dependence on the generator-level cut

Single passes of the standalone script with `--ptmin`, all other settings at
their defaults. The CMSSW and MadGraph references of 5.1 apply to the 100 GeV
row only; the points of a scan share the same Sobol seeds and are not
statistically independent.

| pTHat_min [GeV] | sigma [pb] |
|---:|---:|
| 30 | 18 506 |
| 50 | 3 650.5 |
| 100 | 345.32 |
| 150 | 79.74 |
| 200 | 26.96 |
| 300 | 5.350 |

### 5.4 Scale, flavour, mass and PDF-convention variations

Computed with `qqbar_bbbar_xsec.py` (2^15 Sobol points, two repeats;
integration errors below 0.002%). The central value is 345.32 pb; the
envelope of the seven-point scale variation is 345.3 +80.0 -56.5 pb.

| Variation | Option | sigma [pb] | change |
|---|---|---:|---:|
| mu_R x 2 | `--mur-factor 2` | 288.83 | -16.4% |
| mu_R x 1/2 | `--mur-factor 0.5` | 425.34 | +23.2% |
| mu_F x 2 | `--muf-factor 2` | 357.19 | +3.4% |
| mu_F x 1/2 | `--muf-factor 0.5` | 330.19 | -4.4% |
| mu_R, mu_F x 2 | `--mur-factor 2 --muf-factor 2` | 298.70 | -13.5% |
| mu_R, mu_F x 1/2 | `--mur-factor 0.5 --muf-factor 0.5` | 406.61 | +17.7% |
| no incoming b | `--nquark 4` | 339.85 | -1.6% |
| m_b = 4.7 GeV | `--mb 4.7` | 345.37 | +0.01% |
| negative PDFs kept | `--signed-pdfs` | 345.04 | -0.08% |

### 5.5 PDF values and coupling at our event

Evaluated with LHAPDF and with PYTHIA's alpha_s running at the momentum
fractions of `event_ME.lhe` and at the scale Q^2 = pTHat^2 + m_b^2 =
12 842 GeV^2 (section [2] of the `-v` log). PYTHIA prints the same quantities
in the Info Listing of the CMSSW log (section 4.2), where they agree to the
printed digits.

| Parton | x | x f(x, Q^2) |
|---|---:|---:|
| ubar from beam A | 2.174e-3 | 1.1455 |
| u from beam B | 0.3901 | 0.2107 |
| alpha_s(Q^2), PYTHIA second order | | 0.11425 |
| alpha_s(Q^2) from the PDF set | | 0.11425 |

### 5.6 Kinematics of the two quarks

Computed by `plot_journey_ME.py` from `event_ME.lhe`; the b quark is the
mother we follow.

| | p_x | p_y | p_z | E | pT | eta | y | phi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b | -62.071 | -94.690 | -225.148 | 252.059 | 113.22 | -1.439 | -1.438 | -2.151 |
| bbar | 62.071 | 94.690 | -2413.001 | 2415.661 | 113.22 | -3.753 | -3.752 | 0.991 |
| pair | 0 | 0 | -2638.149 | 2667.719 | 0 | | Y = -2.595 | |

Momenta in GeV, phi in rad. Checks: sHat = x1 x2 s = 2.174e-3 x 0.3901 x
13600^2 = 156 895 GeV^2 against the momentum-sum mass 396.1 GeV; the decay
rapidity y* = artanh(beta cos thetaHat) = 1.157 gives y_b = Y + y* = -1.438
and y_bbar = Y - y* = -3.752.

### 5.7 The energy ledger

The ledger opens with a single entry; every later row of the book is compared
with it.

| Stage | pT [GeV] | m [GeV] | n_const | delta pT [GeV] |
|---|---:|---:|---:|---:|
| before this stage | | | | |
| b quark in the hard-process record | 113.22 | 4.80 | 1 | |
