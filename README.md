# JitJet: Journey Is The Jet (from bottom to top)

> A training book for new Master, PhD and postdoc students joining the
> Jet/MET (JME) group of the CMS experiment.

## Important Links
JitJet CMS Implementation: https://github.com/ravindkv/JitJetSW
JitJet Tour Playlist: https://www.youtube.com/playlist?list=PLbSW20iyZFd4 

## 1. The idea in one paragraph

In CMS an electron, a photon or a muon is a *destination*: an object that is
essentially there at the end of reconstruction and needs "only" identification
and calibration. A **jet is a journey**. It does not exist at any single point;
it is the accumulated history of a coloured parton from the hard scattering to
the calibrated, tagged, smeared four-vector an analyst finally puts into a
histogram. Every stage of that journey either *adds* energy to the jet, *removes*
energy from it, or *reshuffles* how the energy is distributed. The book follows
one and the same jet through every stage, shows the mathematics of each stage,
measures how much energy that stage added or removed, and points to the
official paper for further reading.

## 2. The metaphor: The Family Pilgrimage

Three candidate metaphors were drafted (a family pilgrimage, a mountaineering
expedition, an energy economy). The **Family Pilgrimage** was chosen: a quark is
a *mother*, the parton shower produces her *children*, pileup is the *crowd on
the road*, detector and trigger are *borders and checkpoints*, Particle Flow is
the *census*, CHS/PUPPI are the *police*, clustering is the *inn*, JEC is the
*tax office*, JER is the *wobbly ruler*, b-tagging is the *registry office*.
The other images (barber shop for grooming, shop where we pay more and gain
less) survive at chapter level inside the top-jet part.

## 3. Building the PDF

```bash
make
```

or, equivalently, `latexmk -pdf JitJet.tex` (the `.latexmkrc` adds the same
options). `make quick` runs a single pdflatex pass; `make clean` removes
auxiliary files.

Source files from `example/` are printed in the book with the `minted`
package, which calls `pygmentize` while pdflatex runs. The build therefore
needs `-shell-escape` (set in the Makefile) and Pygments on the PATH
(`pip install Pygments`). Long files are pre-processed into `build/` by
`make listings`, which `make` runs automatically; for instance the standalone
integrator's 250 kB line of embedded tables is replaced by a placeholder.

Figures derived from `example/` (such as `example/ME/journey_ME.pdf`) are
committed, so building the book does not need matplotlib. After changing a
plotting script or its input, regenerate them with

```bash
make figures PYTHON=/usr/bin/python3
```
