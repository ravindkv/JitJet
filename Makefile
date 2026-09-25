# Build JitJet.pdf with pdflatex + bibtex.
#   make           -> JitJet.pdf
#   make quick     -> single pdflatex pass (no bibliography refresh)
#   make figures   -> regenerate the figures made from example/ (needs matplotlib)
#   make clean     -> remove auxiliary files
#   make distclean -> also remove the PDF
#
# Short records and command snippets are typeset with the minted package, which
# calls pygmentize at build time; hence -shell-escape and a working `pygmentize`.
# The Python sources themselves are not printed: the book points to GitHub.
MAIN   := JitJet
TEX    := pdflatex -shell-escape -interaction=nonstopmode -halt-on-error
PYTHON ?= python3

SRC := $(MAIN).tex preamble.tex $(wildcard chapter/*.tex) reference/JitJet.bib

$(MAIN).pdf: $(SRC)
	$(TEX) $(MAIN).tex
	bibtex $(MAIN)
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex

quick:
	$(TEX) $(MAIN).tex

# Figures drawn from the example files. The PDFs are committed, so building the
# book does not need matplotlib; run `make figures` after changing a script.
FIGURES := example/ME/journey_ME.pdf \
           example/ME/Standalone/animate_standalone_qqbar_bbbar_xsec_stills.pdf

figures: $(FIGURES)

example/ME/journey_ME.pdf: example/ME/plot_journey_ME.py example/ME/event_ME.lhe
	$(PYTHON) example/ME/plot_journey_ME.py -i example/ME/event_ME.lhe -o $@

# Key frames of the 3D animation (one page per scene). The film itself takes
# minutes to render and is not a build product: run the script without
# --no-video to make the .mp4.
example/ME/Standalone/animate_standalone_qqbar_bbbar_xsec_stills.pdf: \
		example/ME/Standalone/animate_standalone_qqbar_bbbar_xsec.py \
		example/ME/Standalone/standalone_qqbar_bbbar_xsec.py example/ME/event_ME.lhe
	$(PYTHON) example/ME/Standalone/animate_standalone_qqbar_bbbar_xsec.py --stills $@ --no-video

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc \
	      $(MAIN).lof $(MAIN).lot $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).pyg chapter/*.aux
	rm -rf _minted-$(MAIN) build

distclean: clean
	rm -f $(MAIN).pdf

.PHONY: quick figures clean distclean
