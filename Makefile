# Build JitJet.pdf with pdflatex + bibtex.
#   make           -> JitJet.pdf
#   make quick     -> single pdflatex pass (no bibliography refresh)
#   make cover     -> cover/cover.pdf, the TikZ front cover (first page of the book)
#   make cover-timeline -> cover/cover_timeline.pdf, the alternative timeline-cone cover
#   make cover-timeline-minimal -> cover/cover_timeline_minimal.pdf, the same cone without part labels
#   make cover-video -> cover/animate_cover_timeline.mp4, the narrated trailer of the cover
#   make cover-video-short -> cover/animate_short_cover_timeline.mp4, the narrated cone-growth trailer (~80 s)
#   make cover-youtube -> smpost/cover_photo_youtube.png, the 2560x1440 YouTube channel banner
#   make profile-youtube -> smpost/profile_photo_youtube.png, the 800x800 YouTube profile picture
#   make thumbnails-youtube -> smpost/thumbnail/chapter_*.png, the 1280x720 thumbnails of the videos
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

# The short trailer (about 80 s): the cone grows part by part, one narrated line
# per part (neural voice via edge-tts, needs the network; --engine say for the
# macOS voice, --no-narration for music only) over a synthesised score, plus an
# .srt. Compiles cover_timeline.tex once per growth step into
# build/short_cover_frames, where the narration clips are cached as well.
cover-video-short: cover/cover_timeline.tex
	$(PYTHON) cover/animate_short_cover_timeline.py

# The YouTube channel banner: the timeline cone laid out for YouTube's safe
# areas, compiled with pdflatex and rasterised with gs to exactly 2560x1440.
cover-youtube: smpost/cover_photo_youtube.png

smpost/cover_photo_youtube.pdf: smpost/cover_photo_youtube.tex
	cd smpost && pdflatex -interaction=nonstopmode -halt-on-error cover_photo_youtube.tex

smpost/cover_photo_youtube.png: smpost/cover_photo_youtube.pdf
	gs -q -dBATCH -dNOPAUSE -sDEVICE=png16m -r169.3333 -g2560x1440 -dPDFFitPage \
	   -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -sOutputFile=$@ $<

# The YouTube profile picture: a square 800x800 drawn for the circular crop.
profile-youtube: smpost/profile_photo_youtube.png

smpost/profile_photo_youtube.pdf: smpost/profile_photo_youtube.tex
	cd smpost && pdflatex -interaction=nonstopmode -halt-on-error profile_photo_youtube.tex

smpost/profile_photo_youtube.png: smpost/profile_photo_youtube.pdf
	gs -q -dBATCH -dNOPAUSE -sDEVICE=png16m -r254 -g800x800 -dPDFFitPage \
	   -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -sOutputFile=$@ $<

# The video thumbnails: one standalone TikZ page per video of smpost/SMPost.md,
# sharing smpost/thumbnail/thumbnail_common.tex, rasterised to 1280x720.
THUMBS := $(patsubst %.tex,%.png,$(wildcard smpost/thumbnail/chapter_*.tex))

thumbnails-youtube: $(THUMBS)

smpost/thumbnail/%.pdf: smpost/thumbnail/%.tex smpost/thumbnail/thumbnail_common.tex
	cd smpost/thumbnail && pdflatex -interaction=nonstopmode -halt-on-error $(notdir $<)

smpost/thumbnail/%.png: smpost/thumbnail/%.pdf
	gs -q -dBATCH -dNOPAUSE -sDEVICE=png16m -r169.3333 -g1280x720 -dPDFFitPage \
	   -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -sOutputFile=$@ $<

# Figures drawn from the example files. The PDFs are committed, so building the
# book does not need matplotlib; run `make figures` after changing a script.
FIGURES := example/ME/journey_ME.pdf \
           example/ME/Standalone/animate_standalone_qqbar_bbbar_xsec_stills.pdf \
           example/PS/journey_PS.pdf \
           example/PS/Standalone/animate_standalone_parton_showering_stills.pdf \
           example/PS/Standalone/veto_algorithm.pdf \
           example/ME/Standalone/theory_qqbar_bbbar.pdf \
           example/PS/Standalone/theory_parton_showering.pdf \
           example/MnM/animate_whose_child_stills.pdf \
           example/MPI/journey_MPI.pdf \
           example/MPI/Standalone/animate_standalone_multiparton_interactions_stills.pdf \
           example/MPI/Standalone/theory_multiparton_interactions.pdf \
           example/Had/journey_Had.pdf \
           example/Had/Standalone/animate_standalone_hadronisation_stills.pdf \
           example/Had/Standalone/theory_hadronisation.pdf \
           example/Decay/journey_Decay.pdf \
           example/Decay/Standalone/animate_standalone_hadron_decays_stills.pdf \
           example/Decay/Standalone/theory_hadron_decays.pdf

figures: $(FIGURES)

example/ME/journey_ME.pdf: example/ME/plot_journey_ME.py example/ME/event_ME.lhe
	$(PYTHON) example/ME/plot_journey_ME.py -i example/ME/event_ME.lhe -o $@

example/PS/journey_PS.pdf: example/PS/plot_journey_PS.py example/PS/event_PS.lhe
	$(PYTHON) example/PS/plot_journey_PS.py -i example/PS/event_PS.lhe -o $@

# Stage 3: the event after multiparton interactions, beam remnants and colour reconnection.
example/MPI/journey_MPI.pdf: example/MPI/plot_journey_MPI.py example/PS/plot_journey_PS.py example/MPI/event_MPI.lhe
	$(PYTHON) example/MPI/plot_journey_MPI.py -i example/MPI/event_MPI.lhe -o $@

# Stage 4: the primary hadrons after the Lund string fragmentation.
example/Had/journey_Had.pdf: example/Had/plot_journey_Had.py example/PS/plot_journey_PS.py example/Had/event_Had.lhe
	$(PYTHON) example/Had/plot_journey_Had.py -i example/Had/event_Had.lhe -o $@

# Stage 5: everything the detector will see, after the hadron decays.
example/Decay/journey_Decay.pdf: example/Decay/plot_journey_Decay.py example/PS/plot_journey_PS.py example/Had/plot_journey_Had.py \
		example/Decay/event_Decay.lhe
	$(PYTHON) example/Decay/plot_journey_Decay.py -i example/Decay/event_Decay.lhe -o $@

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

# The plots behind the theory sections of chapter 2 (one page per section): the
# seed-1 shower replayed with the script's classes, the committed record and the
# -v log (about a minute: 1500 proposals per radiator for the survival curves).
example/PS/Standalone/theory_parton_showering.pdf: example/PS/Standalone/plot_theory_PS.py \
		example/PS/Standalone/standalone_parton_showering.py example/PS/plot_journey_PS.py \
		example/PS/event_PS.lhe example/PS/Standalone/standalone_parton_showering.log
	$(PYTHON) example/PS/Standalone/plot_theory_PS.py -o $@

# The plots behind the theory sections of chapter 4 (one page per section): the
# seed-1 MPI event at the CMSSW impact parameter replayed with the script's classes,
# plus the --repeat statistics and the PYTHIA validation files when present (about 30 s).
example/MPI/Standalone/theory_multiparton_interactions.pdf: example/MPI/Standalone/plot_theory_MPI.py \
		example/MPI/Standalone/standalone_multiparton_interactions.py example/MPI/plot_journey_MPI.py \
		example/PS/event_PS.lhe
	$(PYTHON) example/MPI/Standalone/plot_theory_MPI.py -o $@

# Key frames of the MPI animation (one page per scene; the film is not a build product).
example/MPI/Standalone/animate_standalone_multiparton_interactions_stills.pdf: \
		example/MPI/Standalone/animate_standalone_multiparton_interactions.py \
		example/MPI/Standalone/standalone_multiparton_interactions.py example/MPI/plot_journey_MPI.py example/PS/event_PS.lhe
	$(PYTHON) example/MPI/Standalone/animate_standalone_multiparton_interactions.py --stills $@ --no-video

# The plots behind the theory sections of chapter 5 (one page per section): the seed-1
# hadronisation replayed with the script's classes, the --repeat statistics and the PYTHIA
# validation files when present (about 20 s).
example/Had/Standalone/theory_hadronisation.pdf: example/Had/Standalone/plot_theory_Had.py \
		example/Had/Standalone/standalone_hadronisation.py example/Had/plot_journey_Had.py example/MPI/event_MPI.lhe
	$(PYTHON) example/Had/Standalone/plot_theory_Had.py -o $@

# Key frames of the hadronisation animation (one page per scene; the film is not a build product).
example/Had/Standalone/animate_standalone_hadronisation_stills.pdf: \
		example/Had/Standalone/animate_standalone_hadronisation.py \
		example/Had/Standalone/standalone_hadronisation.py example/Had/plot_journey_Had.py example/MPI/event_MPI.lhe
	$(PYTHON) example/Had/Standalone/animate_standalone_hadronisation.py --stills $@ --no-video

# The plots behind the theory sections of chapter 6: the seed-1 decays replayed with the script's
# classes, the --repeat statistics and the PYTHIA validation files when present (about 30 s).
example/Decay/Standalone/theory_hadron_decays.pdf: example/Decay/Standalone/plot_theory_Decay.py \
		example/Decay/Standalone/standalone_hadron_decays.py example/Decay/Standalone/decay_tables.json example/Had/event_Had.lhe
	$(PYTHON) example/Decay/Standalone/plot_theory_Decay.py -o $@

# Key frames of the hadron-decay animation (one page per scene; the film is not a build product).
example/Decay/Standalone/animate_standalone_hadron_decays_stills.pdf: \
		example/Decay/Standalone/animate_standalone_hadron_decays.py \
		example/Decay/Standalone/standalone_hadron_decays.py example/Decay/Standalone/decay_tables.json example/Had/event_Had.lhe
	$(PYTHON) example/Decay/Standalone/animate_standalone_hadron_decays.py --stills $@ --no-video

# Chapter 3: one frame per beat of the ten-second film (the film itself is not a build product).
example/MnM/animate_whose_child_stills.pdf: example/MnM/animate_whose_child.py
	$(PYTHON) example/MnM/animate_whose_child.py --stills $@ --no-video

# The four steps of the Sudakov veto algorithm with the numbers of the seed-1
# shower (imports AlphaStrong from the shower script; needs matplotlib).
example/PS/Standalone/veto_algorithm.pdf: example/PS/Standalone/plot_veto_algorithm.py \
		example/PS/Standalone/standalone_parton_showering.py
	$(PYTHON) example/PS/Standalone/plot_veto_algorithm.py -o $@

# Event records produced by the standalone scripts. They are committed; run
# `make records` after changing a script (needs numpy + scipy).
RECORDS := example/PS/event_PS.lhe example/MPI/event_MPI.lhe example/Had/event_Had.lhe example/Decay/event_Decay.lhe

records: $(RECORDS)

example/PS/event_PS.lhe: example/PS/Standalone/standalone_parton_showering.py example/ME/event_ME.lhe
	$(PYTHON) example/PS/Standalone/standalone_parton_showering.py -i example/ME/event_ME.lhe -o $@

# Chapter 4 adds the rest of the two protons to the showered record. The impact parameter is
# fixed to the one of the CMSSW event (b = 0.3136 <b>), so that the two can be compared;
# without --impact the script draws b itself (seed 1 then gives a peripheral collision).
example/MPI/event_MPI.lhe: example/MPI/Standalone/standalone_multiparton_interactions.py example/PS/event_PS.lhe
	$(PYTHON) example/MPI/Standalone/standalone_multiparton_interactions.py -i example/PS/event_PS.lhe -o $@ --impact 0.3136

# Chapter 5 hadronises the record of chapter 4 (seed 1; the hadron table is embedded, no PYTHIA needed).
example/Had/event_Had.lhe: example/Had/Standalone/standalone_hadronisation.py example/MPI/event_MPI.lhe
	$(PYTHON) example/Had/Standalone/standalone_hadronisation.py -i example/MPI/event_MPI.lhe -o $@

# Chapter 6 decays the primary hadrons of chapter 5 (seed 1; the decay tables are in decay_tables.json, no PYTHIA needed).
example/Decay/event_Decay.lhe: example/Decay/Standalone/standalone_hadron_decays.py example/Decay/Standalone/decay_tables.json example/Had/event_Had.lhe
	$(PYTHON) example/Decay/Standalone/standalone_hadron_decays.py -i example/Had/event_Had.lhe -o $@

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc \
	      $(MAIN).lof $(MAIN).lot $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).pyg chapter/*.aux \
	      cover/cover.aux cover/cover.log cover/cover_timeline.aux cover/cover_timeline.log \
	      cover/cover_timeline_minimal.aux cover/cover_timeline_minimal.log \
	      smpost/cover_photo_youtube.aux smpost/cover_photo_youtube.log \
	      smpost/profile_photo_youtube.aux smpost/profile_photo_youtube.log \
	      smpost/thumbnail/*.aux smpost/thumbnail/*.log
	rm -rf _minted-$(MAIN) build

distclean: clean
	rm -f $(MAIN).pdf $(COVER) cover/cover_timeline.pdf cover/cover_timeline_minimal.pdf

.PHONY: quick cover cover-timeline cover-timeline-minimal cover-video cover-video-short cover-youtube profile-youtube thumbnails-youtube figures records clean distclean
