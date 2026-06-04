# Makefile for Marp presentation
# Meeting_260605.md -> HTML, PDF, PPTX

# Variables
MARP := marp
SOURCE_DIR := src/Presentation
SOURCE_FILE := Meeting_260605.md
THEME := ../../../KIST.css
OUTPUT_DIR := $(CURDIR)
BASENAME := Meeting_260605

# Output files
HTML := $(OUTPUT_DIR)/$(BASENAME).html
PDF := $(OUTPUT_DIR)/$(BASENAME).pdf
PPTX := $(OUTPUT_DIR)/$(BASENAME).pptx

# Default target
.PHONY: all
all: html pdf

# HTML output
.PHONY: html
html: $(HTML)

$(HTML): $(SOURCE_DIR)/$(SOURCE_FILE)
	cd $(SOURCE_DIR) && $(MARP) $(SOURCE_FILE) --html --theme-set $(THEME) -o $(HTML) --allow-local-files
	cp $(OUTPUT_DIR)/$(BASENAME).html ../index.html

# PDF output
.PHONY: pdf
pdf: $(PDF)

$(PDF): $(SOURCE_DIR)/$(SOURCE_FILE)
	cd $(SOURCE_DIR) && $(MARP) $(SOURCE_FILE) --html --theme-set $(THEME) --pdf -o $(PDF) --allow-local-files

# Clean generated files
.PHONY: clean
clean:
	rm -f $(HTML) $(PDF) $(PPTX)

# Watch mode for development (HTML only)
.PHONY: watch
watch:
	cd $(SOURCE_DIR) && $(MARP) -w $(SOURCE_FILE) --html --theme-set $(THEME) -o $(HTML)

# Serve presentation locally
.PHONY: serve
serve:
	cd $(SOURCE_DIR) && $(MARP) -s $(SOURCE_FILE) --html --theme-set $(THEME)

# Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all      - Build HTML, PDF, and PPTX (default)"
	@echo "  html     - Build HTML presentation"
	@echo "  pdf      - Build PDF presentation"
	@echo "  pptx     - Build PowerPoint presentation"
	@echo "  clean    - Remove all generated files"
	@echo "  watch    - Watch mode for live HTML updates"
	@echo "  serve    - Serve presentation locally in browser"
	@echo "  help     - Show this help message"
