# Build JitJet.pdf with pdflatex + bibtex.
#   make        -> JitJet.pdf
#   make quick  -> single pdflatex pass (no bibliography refresh)
#   make clean  -> remove auxiliary files
#   make distclean -> also remove the PDF
MAIN   := JitJet
TEX    := pdflatex -interaction=nonstopmode -halt-on-error
SRC    := $(MAIN).tex preamble.tex $(wildcard chapter/*.tex) reference/JitJet.bib

$(MAIN).pdf: $(SRC)
	$(TEX) $(MAIN).tex
	bibtex $(MAIN)
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex

quick:
	$(TEX) $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc \
	      $(MAIN).lof $(MAIN).lot $(MAIN).fls $(MAIN).fdb_latexmk chapter/*.aux

distclean: clean
	rm -f $(MAIN).pdf

.PHONY: quick clean distclean
