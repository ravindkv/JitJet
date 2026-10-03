# Build JitJet.pdf with pdflatex + bibtex.
#   make           -> JitJet.pdf
#   make quick     -> single pdflatex pass (no bibliography refresh)
#   make cover     -> cover/cover.pdf, the TikZ front cover (first page of the book)
#   make cover-timeline -> cover/cover_timeline.pdf, the alternative timeline-cone cover
#   make cover-timeline-minimal -> cover/cover_timeline_minimal.pdf, the same cone without part labels
#   make cover-video -> cover/animate_cover_timeline.mp4, the narrated trailer of the cover
#   make cover-video-short -> cover/animate_short_cover_timeline.mp4, the one-minute cone-growth trailer
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

COVER := cover/cover.pdf
SRC := $(MAIN).tex preamble.tex $(wildcard chapter/*.tex) reference/JitJet.bib $(COVER)

$(MAIN).pdf: $(SRC)
	$(TEX) $(MAIN).tex
	bibtex $(MAIN)
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex

quick: $(COVER)
	$(TEX) $(MAIN).tex

# The cover is a standalone TikZ document (no shell escape needed).
cover: $(COVER)

$(COVER): cover/cover.tex
	cd cover && pdflatex -interaction=nonstopmode -halt-on-error cover.tex

# The alternative cover: the jet as a timeline cone (not included in the book
# unless the \includepdf line in JitJet.tex is pointed at it).
cover-timeline: cover/cover_timeline.pdf

cover/cover_timeline.pdf: cover/cover_timeline.tex
	cd cover && pdflatex -interaction=nonstopmode -halt-on-error cover_timeline.tex

# The minimal variant: a slightly smaller cone, no part labels, no caption.
cover-timeline-minimal: cover/cover_timeline_minimal.pdf

cover/cover_timeline_minimal.pdf: cover/cover_timeline_minimal.tex
	cd cover && pdflatex -interaction=nonstopmode -halt-on-error cover_timeline_minimal.tex

# The trailer: a narrated flight along the timeline cone (about 5 min, 1080p).
# Needs numpy + Pillow (on this Mac: PYTHON=/usr/bin/python3), ffmpeg, gs and
# the macOS `say` voice; rendering takes a while, so it is not a build product.
cover-video: cover/cover_timeline.pdf
	$(PYTHON) cover/animate_cover_timeline.py

# The short trailer (about 1 min, no narration): the cone grows part by part.
# Compiles cover_timeline.tex once per growth step into build/short_cover_frames.
cover-video-short: cover/cover_timeline.tex
	$(PYTHON) cover/animate_short_cover_timeline.py

# Figures drawn from the example files. The PDFs are committed, so building the
# book does not need matplotlib; run `make figures` after changing a script.
FIGURES := example/ME/journey_ME.pdf \
           example/ME/Standalone/animate_standalone_qqbar_bbbar_xsec_stills.pdf \
           example/PS/journey_PS.pdf \
           example/PS/Standalone/animate_standalone_parton_showering_stills.pdf \
           example/PS/Standalone/veto_algorithm.pdf \
           example/ME/Standalone/theory_qqbar_bbbar.pdf

figures: $(FIGURES)

example/ME/journey_ME.pdf: example/ME/plot_journey_ME.py example/ME/event_ME.lhe
	$(PYTHON) example/ME/plot_journey_ME.py -i example/ME/event_ME.lhe -o $@

example/PS/journey_PS.pdf: example/PS/plot_journey_PS.py example/PS/event_PS.lhe
	$(PYTHON) example/PS/plot_journey_PS.py -i example/PS/event_PS.lhe -o $@

# Key frames of the 3D animation (one page per scene). The film itself takes
# minutes to render and is not a build product: run the script without
# --no-video to make the .mp4.
example/ME/Standalone/animate_standalone_qqbar_bbbar_xsec_stills.pdf: \
		example/ME/Standalone/animate_standalone_qqbar_bbbar_xsec.py \
		example/ME/Standalone/standalone_qqbar_bbbar_xsec.py example/ME/event_ME.lhe
	$(PYTHON) example/ME/Standalone/animate_standalone_qqbar_bbbar_xsec.py --stills $@ --no-video

example/PS/Standalone/animate_standalone_parton_showering_stills.pdf: \
		example/PS/Standalone/animate_standalone_parton_showering.py \
		example/PS/Standalone/standalone_parton_showering.py example/PS/plot_journey_PS.py example/ME/event_ME.lhe
	$(PYTHON) example/PS/Standalone/animate_standalone_parton_showering.py --stills $@ --no-video

# The plots behind the theory sections of chapter 1 (one page per section),
# all computed with the tables and born_weights() of the standalone script;
# page 8 reads scale_scan.json, written by scan_scales.py in the .venv.
example/ME/Standalone/theory_qqbar_bbbar.pdf: example/ME/Standalone/plot_theory_ME.py \
		example/ME/Standalone/standalone_qqbar_bbbar_xsec.py example/ME/Standalone/scale_scan.json example/ME/event_ME.lhe
	$(PYTHON) example/ME/Standalone/plot_theory_ME.py -o $@

# The seven-point scale scan with the full integrator (LHAPDF + PYTHIA alpha_s, about 2 min).
example/ME/Standalone/scale_scan.json: example/ME/Standalone/scan_scales.py example/ME/Standalone/qqbar_bbbar_xsec.py
	example/ME/Standalone/.venv/bin/python example/ME/Standalone/scan_scales.py -o $@

# The four steps of the Sudakov veto algorithm with the numbers of the seed-1
# shower (imports AlphaStrong from the shower script; needs matplotlib).
example/PS/Standalone/veto_algorithm.pdf: example/PS/Standalone/plot_veto_algorithm.py \
		example/PS/Standalone/standalone_parton_showering.py
	$(PYTHON) example/PS/Standalone/plot_veto_algorithm.py -o $@

# Event records produced by the standalone scripts. They are committed; run
# `make records` after changing a script (needs numpy + scipy).
RECORDS := example/PS/event_PS.lhe

records: $(RECORDS)

example/PS/event_PS.lhe: example/PS/Standalone/standalone_parton_showering.py example/ME/event_ME.lhe
	$(PYTHON) example/PS/Standalone/standalone_parton_showering.py -i example/ME/event_ME.lhe -o $@

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc \
	      $(MAIN).lof $(MAIN).lot $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).pyg chapter/*.aux \
	      cover/cover.aux cover/cover.log cover/cover_timeline.aux cover/cover_timeline.log \
	      cover/cover_timeline_minimal.aux cover/cover_timeline_minimal.log
	rm -rf _minted-$(MAIN) build

distclean: clean
	rm -f $(MAIN).pdf $(COVER) cover/cover_timeline.pdf cover/cover_timeline_minimal.pdf

.PHONY: quick cover cover-timeline cover-timeline-minimal cover-video cover-video-short figures records clean distclean
