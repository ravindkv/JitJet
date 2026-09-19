# Build JitJet.pdf with pdflatex + bibtex.
#   make           -> JitJet.pdf
#   make quick     -> single pdflatex pass (no bibliography refresh)
#   make listings  -> regenerate the print copies of long source files (build/)
#   make figures   -> regenerate the figures made from example/ (needs matplotlib)
#   make clean     -> remove auxiliary files
#   make distclean -> also remove the PDF
#
# Source files are included in the book with the minted package, which calls
# pygmentize at build time; hence -shell-escape and a working `pygmentize`.
MAIN   := JitJet
TEX    := pdflatex -shell-escape -interaction=nonstopmode -halt-on-error
PYTHON ?= python3

# Long source files that are pre-processed before printing. The standalone
# integrator embeds ~250 kB of numeric tables on one line; the print copy
# replaces that line with a placeholder (the code is otherwise identical).
LISTINGS := build/standalone_qqbar_bbbar_xsec.py

SRC := $(MAIN).tex preamble.tex $(wildcard chapter/*.tex) reference/JitJet.bib $(LISTINGS)

$(MAIN).pdf: $(SRC)
	$(TEX) $(MAIN).tex
	bibtex $(MAIN)
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex

quick: $(LISTINGS)
	$(TEX) $(MAIN).tex

listings: $(LISTINGS)

build/standalone_qqbar_bbbar_xsec.py: example/ME/Standalone/standalone_qqbar_bbbar_xsec.py
	@mkdir -p build
	awk '/^TABLES = \{/ { print "TABLES = {...}  # ~250 kB of PDF and alpha_s tables: see dump_standalone_tables.py"; next } { print }' $< > $@

# Figures drawn from the example files. The PDFs are committed, so building the
# book does not need matplotlib; run `make figures` after changing a script.
FIGURES := example/ME/journey_ME.pdf

figures: $(FIGURES)

example/ME/journey_ME.pdf: example/ME/plot_journey_ME.py example/ME/event_ME.lhe
	$(PYTHON) example/ME/plot_journey_ME.py -i example/ME/event_ME.lhe -o $@

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc \
	      $(MAIN).lof $(MAIN).lot $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).pyg chapter/*.aux
	rm -rf _minted-$(MAIN) build

distclean: clean
	rm -f $(MAIN).pdf

.PHONY: quick listings figures clean distclean
