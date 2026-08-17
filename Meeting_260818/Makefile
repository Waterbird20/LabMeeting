# Makefile for Marp presentation
# Meeting_260818.md -> HTML, PDF
#
# Image paths in the source are ROOT-RELATIVE (./Meeting_260818/src/Presentation/media/...)
# so they resolve when the built HTML is served as the repo-root index.html (GitHub Pages).

MARP := marp
SOURCE_DIR := src/Presentation
SOURCE_FILE := Meeting_260818.md
THEME := ../../theme/serif.css
OUTPUT_DIR := $(CURDIR)
BASENAME := Meeting_260818
ROOT := $(CURDIR)/..

HTML := $(OUTPUT_DIR)/$(BASENAME).html
PDF := $(OUTPUT_DIR)/$(BASENAME).pdf

.PHONY: all
all: html pdf

# HTML output, then publish as the repo-root index.html
.PHONY: html
html: $(HTML)

$(HTML): $(SOURCE_DIR)/$(SOURCE_FILE)
	cd $(SOURCE_DIR) && $(MARP) $(SOURCE_FILE) --html --theme-set $(THEME) -o $(HTML) --allow-local-files
	cp $(HTML) $(ROOT)/index.html

# PDF: build from the repo root so the root-relative image paths resolve.
.PHONY: pdf
pdf: $(PDF)

$(PDF): $(SOURCE_DIR)/$(SOURCE_FILE)
	cd $(ROOT) && $(MARP) $(BASENAME)/$(SOURCE_DIR)/$(SOURCE_FILE) --html \
	  --theme-set $(BASENAME)/theme/serif.css --pdf -o $(PDF) --allow-local-files

.PHONY: clean
clean:
	rm -f $(HTML) $(PDF)
