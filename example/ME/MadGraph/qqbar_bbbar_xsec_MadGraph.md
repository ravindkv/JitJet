# How to get the same cross section in MadGraph

This reproduces the q qbar -> b bbar Born cross section of chapter 1
(pTHat >= 100 GeV, sqrt(s) = 13.6 TeV) with MadGraph5_aMC@NLO instead of
PYTHIA. Note the conventions that differ from the PYTHIA run: the proton is
defined without the b quark (four incoming flavours), the `sm` model has
m_b = 4.7 GeV, alpha_s is taken from the PDF set, and the scale is MadGraph's
default dynamical choice. The chapter discusses what each is worth.

Download **MadGraph5_aMC@NLO** from:

https://launchpad.net/mg5amcnlo/+download

For example, using version:

```bash
MG5_aMC_v3.8.0.tar.gz
```

Untar the package and launch MadGraph:

```bash
tar -xzf MG5_aMC_v3.8.0.tar.gz

cd MG5_aMC_v3_8_0/bin
python3.12 mg5_aMC
```

We want only the quark-antiquark contribution, so redefine the proton
content to exclude the gluon (the default `p` also has no b quark):

```text
MG5_aMC> define p = u c d s u~ c~ d~ s~
```

Generate the process:

```text
MG5_aMC> generate p p > b b~
```

Create the output directory:

```text
MG5_aMC> output qq_to_bbar
```

Then launch the run:

```text
MG5_aMC> launch
```

During the launch, edit `run_card.dat`.

For a centre-of-mass energy of 13.6 TeV, set the two beam energies to 6800 GeV:

```text
6800.0 = ebeam1  ! beam 1 total energy in GeV
6800.0 = ebeam2  ! beam 2 total energy in GeV
```

Select the NNPDF3.1 NNLO set through LHAPDF (`lhaid` is only read when
`pdlabel = lhapdf`):

```text
     nn23lo1    = pdlabel     ! PDF set
     303600    = lhaid       ! if pdlabel=lhapdf, this is the lhapdf number
```

Require a minimum b-quark transverse momentum of 100 GeV, the cut of the
book's event:

```text
100.0 = ptb  ! minimum pT for the b quark
```

Save the run card and continue the run. The results summary reads

```text
=== Results Summary for run: run_03 tag: tag_1 ===

Cross-section : 344.3 +- 0.9587 pb
Nb of events  : 10000
```

Since 1 nb = 1000 pb this is 0.3443 nb, to be compared with the 345.1 +- 6.1 pb
of the CMSSW/PYTHIA run and the 345.32 pb of the standalone integrator. With
`ptb = 30` instead, the same recipe gives about 19.4 nb against PYTHIA's
18.4 nb; at the lower cut the four-flavour proton, the b mass and the scale
convention matter more.
