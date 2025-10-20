# Makefile for Marp presentation
# Meeting_251024.md -> HTML, PDF, PPTX

# Variables
MARP := marp
SOURCE := src/Presentation/Meeting_251024.md
THEME := ../KIST.css
OUTPUT_DIR := .
BASENAME := Meeting_251024

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

$(HTML): $(SOURCE) $(THEME)
	$(MARP) $(SOURCE) --html --theme-set $(THEME) -o $(HTML) --allow-local-files main.md

# PDF output
.PHONY: pdf
pdf: $(PDF)

$(PDF): $(SOURCE) $(THEME)
	$(MARP) $(SOURCE) --html --theme-set $(THEME) --pdf -o $(PDF) --allow-local-files main.md

# Clean generated files
.PHONY: clean
clean:
	rm -f $(HTML) $(PDF) $(PPTX)

# Watch mode for development (HTML only)
.PHONY: watch
watch:
	$(MARP) -w $(SOURCE) --html --theme-set $(THEME) -o $(HTML)

# Serve presentation locally
.PHONY: serve
serve:
	$(MARP) -s $(SOURCE) --html --theme-set $(THEME)

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
