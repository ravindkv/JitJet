# Reproducing PYTHIA's q qbar -> b bbar cross section

The independent Python integration gives **18.50662 nb** with the settings
below. Your PYTHIA result is **18.440 +/- 0.09717 nb**: the difference is
**0.36%**, or **0.69 times PYTHIA's statistical uncertainty**. No normalization
was fitted to the PYTHIA result.

The numerical integration uses a separate analytic LO matrix element.
PYTHIA's native `AlphaStrong` class is used only to evaluate the hard-process
coupling. Alternatively, export that coupling from your own initialized C++
generator using the supplied header; no Python PYTHIA installation is then
required.

## Files

- `qqbar_bbbar_xsec.py`: main integrator and command-line options.
- `export_pythia_xsec.h`: exports the actual generator settings and alpha_s.
- `pythia_log_defaults.json`: parameters inferred from your log and standard defaults.
- `validated_default_result.json`: full numerical result, flavour breakdown and configuration.
- `example_pt_scan.json`: example lower-cut scan with numerical errors.
- `test_integrator.py`: independent normalization and phase-space checks.

## Dependencies

### Project-local environment on this Mac

Run with the explicit interpreter path from this directory:

```bash
.venv/bin/python qqbar_bbbar_xsec.py --output result.json
.venv/bin/python test_integrator.py
```

The local environment contains Python 3.13, LHAPDF 6.5.4, the
`NNPDF31_nnlo_as_0118` set, and PYTHIA 8.315. It is ignored by Git.
An existing shell alias such as `alias python=/usr/bin/python3` overrides
virtual-environment activation, so the explicit path is preferable. The old
`LHAPDFSYS=/Users/rverma/Ravindra/fnal/lhapdf_ps` shell setting points to a
directory that does not exist on this Mac and is unnecessary for this setup.

To rebuild the environment (requires a C++ compiler, `make`, and network
access), use an installed Python 3.13 and follow the
[LHAPDF source-build workflow](https://www.lhapdf.org/install.html):

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install numpy scipy 'Cython>=3.0.11' setuptools pythia8mc==8.315.0
ASTRA_ENV="$PWD/.venv"
mkdir -p /tmp/astra-lhapdf-build
cd /tmp/astra-lhapdf-build
curl -fL 'https://lhapdf.hepforge.org/downloads/?f=LHAPDF-6.5.4.tar.gz' -o LHAPDF-6.5.4.tar.gz
tar -xzf LHAPDF-6.5.4.tar.gz
cd LHAPDF-6.5.4
./configure --prefix="$ASTRA_ENV" PYTHON="$ASTRA_ENV/bin/python"
# Regenerate the shipped wrapper for the newer Python interpreter (Cython 3 is
# required: the Cython 0.29 output does not compile against Python 3.13).
"$ASTRA_ENV/bin/cython" -I wrappers/python wrappers/python/lhapdf.pyx --cplus -3 -o wrappers/python/lhapdf.cpp
make -j4
make install
curl -fL https://lhapdfsets.web.cern.ch/current/NNPDF31_nnlo_as_0118.tar.gz -o /tmp/astra-lhapdf-build/NNPDF31_nnlo_as_0118.tar.gz
tar -xzf /tmp/astra-lhapdf-build/NNPDF31_nnlo_as_0118.tar.gz -C "$ASTRA_ENV/share/LHAPDF"
cd "$ASTRA_ENV/.."
.venv/bin/python -c 'import lhapdf; print(lhapdf.version()); lhapdf.mkPDF("NNPDF31_nnlo_as_0118", 0)'
```

### Other environments

Use **Python 3.10 or newer**, with NumPy, SciPy, and LHAPDF6 Python bindings.
Activate the environment in which your LHAPDF bindings are installed. The
official [LHAPDF installation guide](https://www.lhapdf.org/install.html)
describes building the library and bindings. `lhapdf` is not an ordinary
`pip install lhapdf` dependency.

```bash
python -m pip install numpy scipy
python -c "import lhapdf; print(lhapdf.version())"
lhapdf install NNPDF31_nnlo_as_0118
```

The last command downloads a PDF set if it is not already installed. The
existing LHAPDF data directory can instead be exposed with `LHAPDF_DATA_PATH`.

For the default native coupling backend, use existing `pythia8` Python bindings,
or install the version tested here:

```bash
python -m pip install pythia8mc==8.315.0
```

That extra installation is unnecessary with the exported table described below.
The tested environment used PYTHIA 8.315, LHAPDF 6.5.4, and member 0 of
`NNPDF31_nnlo_as_0118` (data version 1, LHAPDF ID 303600).

## Standalone reproduction without LHAPDF or PYTHIA

`standalone_qqbar_bbbar_xsec.py` needs only NumPy and SciPy. The
NNPDF31_nnlo_as_0118 member 0 grid (LHAPDF's own x and Q knots over the
default phase space) and PYTHIA's second-order alpha_s are embedded as numeric
tables between the `# BEGIN/END GENERATED TABLES` markers and interpolated with
a bicubic spline in (log x, log Q^2) and a monotone cubic in log Q^2. The
integrand is vectorised, so one Sobol pass of 2^16 points takes about a second:

```bash
python3 standalone_qqbar_bbbar_xsec.py              # 18.5064 nb, single pass
python3 standalone_qqbar_bbbar_xsec.py --repeats 8  # adds an integration error
```

With the same Sobol seed the standalone and full-environment integrals agree
to 5e-6 relative (18.50638 vs 18.50647 nb for the first repeat). Regenerate the
tables with the full environment if the PDF set, alpha_s settings, or the
kinematic range change:

```bash
.venv/bin/python dump_standalone_tables.py
```

## Run with the supplied settings

```bash
python qqbar_bbbar_xsec.py --output result.json
```

Equivalently:

```bash
python qqbar_bbbar_xsec.py --config pythia_log_defaults.json --output result.json
```

Defaults: sqrt(s)=13600 GeV, mb=4.8 GeV, pTHatMin=30 GeV, no upper cut,
alpha_s(MZ)=0.118, PYTHIA alphaSorder=2, alphaSnfmax=6, five incoming flavours,
muR^2=muF^2=pTHat^2+mb^2, Kfactor=1, PDF central member 0.

The included run used 8 independently scrambled Sobol sequences, each with
65536 integration points. Its estimated integration error is 0.000047 nb.
This describes numerical integration only, not the physical prediction's
uncertainty. The output includes each repeat and each initial-flavour result.

## Take the actual settings from your PYTHIA run

Include the helper in your existing C++ generator:

```cpp
#include "export_pythia_xsec.h"
```

Immediately after successful `pythia.init()`, add:

```cpp
exportPythiaXsec(pythia, "pythia_snapshot.json");
```

Use the hard-process generator instance (your log also contained an instance
with `ProcessLevel:all = off`). Then run:

```bash
python qqbar_bbbar_xsec.py --config pythia_snapshot.json --output result.json
```

The export records the PDF name/member (including `PDF:useHard`), bottom mass,
beam energy, pTHat and mHat bounds, incoming-flavour count, scale settings,
K factor and coupling settings. It samples the initialized generator's
`coupSM.alphaS(Q2)` on a fine grid; Python interpolates in log(Q2).
The helper supports equal-energy pp beams with the same named LHAPDF6 set
for both protons. It rejects unsupported beam/PDF/scale configurations.

Exporting after initialization resolves unprinted defaults and avoids guessing
the PYTHIA version's alpha_s convention. The log-derived JSON is a separate,
explicitly approximate reconstruction of those settings; it is not an export
of your unseen generator instance.

## Change cuts and parameters

Command-line options override the JSON values. At Born level both outgoing
quarks have the same transverse momentum, which is pTHat.

```bash
# Raise the lower pTHat cut.
python qqbar_bbbar_xsec.py --ptmin 50

# Integrate one pTHat bin, 30 to 50 GeV.
python qqbar_bbbar_xsec.py --ptmin 30 --ptmax 50

# Scan lower cuts and save all results.
python qqbar_bbbar_xsec.py --scan-ptmin 20,30,50,100 --output scan.json

# Change energy or mass.
python qqbar_bbbar_xsec.py --ecm 13000 --mb 4.75

# Double muR and muF (the code applies factors of 4 to their squares).
python qqbar_bbbar_xsec.py --mur-factor 2 --muf-factor 2

# Fixed muR=muF=100 GeV. Fixed-scale arguments are Q^2, not Q.
python qqbar_bbbar_xsec.py --renorm-scale 5 --renorm-fix-scale 10000 \
                         --factor-scale 5 --factor-fix-scale 10000

# Restrict sqrt(sHat)=m_bb and require BOTH Born quarks to have |y|<2.5.
python qqbar_bbbar_xsec.py --mhatmin 100 --mhatmax 500 --ymax 2.5

# Alternatively impose pseudorapidity cuts on BOTH Born quarks.
python qqbar_bbbar_xsec.py --etamax 2.5

# Change PDF member; install that member first.
python qqbar_bbbar_xsec.py --member 1

# Change native hard-process alpha_s parameters.
python qqbar_bbbar_xsec.py --alpha-backend pythia --alphas-mz 0.120 --alpha-order 2

# Change incoming-flavour count; 4 removes the incoming b contribution.
python qqbar_bbbar_xsec.py --nquark 4

# Increase integration precision, especially with extra y/eta cuts.
python qqbar_bbbar_xsec.py --power 18 --repeats 12

python qqbar_bbbar_xsec.py --help
```

When using a snapshot, add `--config pythia_snapshot.json` to these commands.
For a cut inherited from JSON, use `null` in the JSON to remove its upper bound.
Do not change `alphas_mz` or `alpha_order` while retaining an old exported
coupling table: select the native backend or export a new table.

An illustrative scan with the same remaining defaults gives:

| pTHatMin (GeV) | sigma (nb), approximately |
|---:|---:|
| 20 | 62.086 |
| 30 | 18.507 |
| 50 | 3.6505 |
| 100 | 0.34532 |

The 18.44 nb PYTHIA benchmark applies to the original 30 GeV cut and original
parameters. Regenerate PYTHIA for comparisons after changing them.

## What is integrated

For c=cos(theta) in the partonic CM frame, rho=4 mb^2/sHat and
beta=sqrt(1-rho), the spin/colour-averaged Born matrix element gives

```text
d sigmaHat / dc = pi alpha_s(muR^2)^2 beta / (9 sHat)
                 * [1 + c^2 + rho (1-c^2)].
```

The angular integral at fixed alpha_s and without cuts recovers the formula
from your original image. For the hadronic result, the code integrates

```text
sum_q integral dx1 dx2 dc
  [ f_q(x1,muF^2) f_qbar(x2,muF^2)
  + f_qbar(x1,muF^2) f_q(x2,muF^2) ]
  * d sigmaHat/dc * cuts,

sHat = x1*x2*s,
pTHat^2 = (sHat/4 - mb^2) * (1-c^2).
```

It uses log(tau), Y=log(x1/x2)/2, and the allowed absolute scattering cosine.
Since dx1 dx2 = tau dlog(tau) dY and LHAPDF supplies x*f(x), the tau factor
cancels the denominator in the PDF product. Both beam orientations are
included. There is no extra factor for two final quarks: this is a pair-event
cross section. The GeV^-2 to pb factor is 3.89379372e8.

PYTHIA scale codes 1,2,3 all give pTHat^2+mb^2 for two equal masses;
code 4 uses sHat, and code 5 uses the explicitly fixed Q^2. Scale multipliers
in the configuration multiply Q^2. Fixed-scale choices ignore those
multipliers, matching PYTHIA; the convenient `--mur-factor`/`--muf-factor`
options instead scale the selected Q or fixed Q appropriately.

## Details that matter for matching

- `PDFinProcess:nQuarkIn=5` includes d,u,s,c,b initial PDFs in process 124's
  flavour sum. Its incoming b term follows that channel's s-channel
  matrix element; this script does not calculate the complete same-flavour
  elastic b bbar scattering matrix element.
- Hard-process alpha_s is configured separately from shower/MPI alpha_s and
  from the coupling supplied by a PDF set. In particular, choosing an NNLO
  PDF does not make this matrix element NNLO. The `lhapdf` coupling backend
  is an explicit alternative, not the default for PYTHIA matching.
- Default PDF handling clips negative densities to zero, freezes Q^2 at grid
  boundaries, and uses symmetric c/cbar and b/bbar seas as in the PYTHIA
  proton wrapper. It also freezes x below the grid unless extrapolation is
  enabled. Boundary and negative-value counts are reported.
- This is the unweighted, unsuppressed Born hard-process cross section with
  the listed cuts. Custom UserHooks, process weights/vetoes, additional
  phase-space suppression, nuclear PDFs, and post-shower selections are not
  reproduced. The |y|/|eta| cuts act on the two Born quarks, not reconstructed
  jets. Additional channels, showers, MPI and hadronization are not integrated.
- Integration errors come from the spread of independently scrambled Sobol
  integrals, including correlations among initial flavours. Increase power
  and repeats to check convergence. Results in a scan share sampling seeds
  and should not be treated as statistically independent bins.

## Validation

```bash
python test_integrator.py
```

Checks cover the analytic total cross section (including 11.5215613 pb for
the original inputs), threshold behaviour, scale semantics and independent
one-dimensional reference integrals for a test density f(x)=1, including
lower/upper pTHat and invariant-mass cuts. That density is used only in tests.

The physical PDF/coupling checks at your printed event give:

| Quantity | Evaluated value | Rounded PYTHIA output |
|---|---:|---:|
| x f_ubar at x=0.0002752, Q^2=1178 | 1.9218395 | 1.922 |
| x f_u at x=0.2841, Q^2=1178 | 0.4074175 | 0.4075 |
| alpha_s(Q^2=1178) | 0.13853399 | 0.1385 |

The small PDF difference reflects rounded input x and Q^2 values. The coupling
table interpolation was checked against native alpha_s over Q=30..13600 GeV;
the maximum relative difference was below 6e-6. The C++ helper passed a syntax
check against PYTHIA 8.315 headers; it has not been executed inside your own
generator. Its JSON reader/table integration path was separately exercised.

## References

- [PDG cross-section formulae, Eq. (51.20), p. 3](https://pdg.lbl.gov/2024/reviews/rpp2024-rev-cross-section-formulae.pdf#page=3): massive quark-pair differential cross section.
- [PYTHIA manual: couplings and scales](https://portal.nersc.gov/project/alice/pythia8doc/htmldoc/CouplingsAndScales.html): scale options and the separate hard-process coupling.
- [PYTHIA manual: hard QCD processes](https://portal.nersc.gov/project/alice/pythia8doc/htmldoc/QCDProcesses.html): process 124 and heavy-flavour channels.
- [LHAPDF examples](https://www.lhapdf.org/codeexamples.html): PDF evaluation interfaces.
- Implementation cross-check: PYTHIA 8.315 `Sigma2qqbar2QQbar::sigmaKin`,
  `SigmaProcess::initFlux`, `Sigma2Process::store2Kin`, and `AlphaStrong`.
