# latexmk configuration: `latexmk -pdf JitJet.tex` is equivalent to `make`.
# minted needs shell escape (pygmentize is called at build time), and the print
# copies of long source files live in build/ (see the Makefile target `listings`).
$pdf_mode = 1;
$pdflatex = 'pdflatex -shell-escape -interaction=nonstopmode -halt-on-error %O %S';
$bibtex_use = 2;
system('make listings');
push @generated_exts, 'pyg';
$clean_ext .= ' %R.pyg';
