# Lab-meeting marp deck — run from inside a meeting folder (e.g. 260617/).
# Concatenates the numbered slide files [0-9]*.md into .build.md, then runs marp.
# Output basename = folder name, so this same Makefile works in every meeting unchanged.
#
#   make            # build PDF with the default theme (indigo)
#   make html       # build HTML
#   make pptx       # build PowerPoint
#   make THEME=serif        # switch theme (indigo | serif | teal), no file edits
#   make THEME=teal html
#   make watch      # live-reload HTML preview of the combined deck
#   make serve      # serve folder in browser
#   make clean      # remove generated files

MARP   ?= marp
THEME  ?= indigo
THEMES := $(wildcard ../theme/*.css)
SRC    := $(sort $(wildcard [0-9]*.md))
BUILD  := .build.md
BASE   := $(shell basename "$(CURDIR)")
COMMON := --theme-set $(THEMES) --theme $(THEME) --allow-local-files

.PHONY: all pdf html pptx watch serve clean
all: pdf

# concatenate slide files, inserting a slide separator between each.
# Each slide file carries its own `marp: true` frontmatter (for per-file VS Code
# preview); we keep the first file's frontmatter as the deck's, and strip the
# leading frontmatter block from every subsequent file so the build has exactly one.
$(BUILD): $(SRC)
	@rm -f $@
	@i=0; for f in $(SRC); do \
	  [ $$i -ne 0 ] && printf '\n\n---\n\n' >> $@; \
	  if [ $$i -eq 0 ]; then cat "$$f" >> $@; \
	  else awk 'NR==1&&$$0=="---"{f=1;next} f==1&&$$0=="---"{f=2;next} f==1{next} {print}' "$$f" >> $@; fi; \
	  i=1; \
	done
	@echo "built $@ from: $(SRC)"

pdf: $(BUILD)
	$(MARP) $(BUILD) $(COMMON) --pdf -o $(BASE).pdf

html: $(BUILD)
	$(MARP) $(BUILD) $(COMMON) --html -o $(BASE).html

pptx: $(BUILD)
	$(MARP) $(BUILD) $(COMMON) --pptx -o $(BASE).pptx

watch: $(BUILD)
	$(MARP) -w $(BUILD) $(COMMON) -o $(BASE).html

serve:
	$(MARP) -s . $(COMMON)

clean:
	rm -f $(BUILD) $(BASE).pdf $(BASE).html $(BASE).pptx
