# Latex Makefile using latexmk
# Modified by Dogukan Cagatay <dcagatay@gmail.com>
# Originally from : http://tex.stackexchange.com/a/40759
#
# Change only the variable below to the name of the main tex file.
PROJNAME=Main
bibfile=setup/BibSettings.tex

# You want latexmk to *always* run, because make does not have all the info.
# Also, include non-file targets in .PHONY so they are run regardless of any
# file of the given name existing.
.PHONY: $(PROJNAME).pdf all clean

# The first rule in a Makefile is the one executed by default ("make"). It
# should always be the "all" rule, so that "make" and "make all" are identical.
all: $(PROJNAME).pdf

# MAIN LATEXMK RULE

$(PROJNAME).pdf: $(PROJNAME).tex
	texfot --no-stderr latexmk -pdflatex --shell-escape -outdir=.build --interaction=nonstopmode $<
	mv .build/Main.pdf $@

clean:
	latexmk -c -auxdir=.build/ -quiet
	rm .build/**/*.aux

# CONVERT TO .docx
citestring := $(shell printf -- '--bibliography %s ' $(shell ggrep -oPe '\\addbibresource{\K.*.bib' $(bibfile)))
$(PROJNAME).docx: $(PROJNAME).tex
	pandoc $< -s -o $@ $(citestring) --citeproc --resource-path draft_figs --resource-path figs -F pandoc-acronym-filter.py

# Figures
figures:
	## Intro figures: require additional annotation in inkscape after generating
	# python3 graphics/kspace.py
	# python3 graphics/relaxation.py
	## Results figures
	python3 graphics/ts_boxplots.py
	python3 graphics/TI_boxplots.py
	python3 graphics/score_mats.py
	python3 graphics/registration_boxplots.py
	python3 graphics/hcp_sbd_box.py
	python3 graphics/btcd_boxplots.py
	## Methods figures
	python3 graphics/fa_tumour.py
	python3 graphics/oneshot_bar.py
	python3 graphics/plot_k.py
	python3 graphics/ntrain.py
