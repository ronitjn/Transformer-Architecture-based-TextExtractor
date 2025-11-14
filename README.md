# PDF Text Extractor

A simple and efficient tool to extract text from PDF files for dataset creation and model training.

## Features

- 📄 Extract plain text from PDF files
- 🧹 Automatic text cleaning (whitespace normalization, hyphenation fixes)
- 💾 Save output as `.txt` or `.json` with metadata
- 🔄 Support for multiple PDF libraries (PyMuPDF, pdfplumber)
- 📊 Page-by-page extraction with character counts

## Installation

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
python pdf_text_extractor.py <path_to_your_pdf>
```

**Example:**
```bash
python pdf_text_extractor.py research_paper.pdf
```

This will generate:
- `research_paper_extracted.txt` - Plain text output
- `research_paper_extracted.json` - JSON with metadata

### Python Script

```python
from pdf_text_extractor import PDFTextExtractor

# Create extractor
extractor = PDFTextExtractor("document.pdf")

# Extract text
extractor.extract()

# Save outputs
extractor.save_as_text("output.txt")
extractor.save_as_json("output.json")

# Or get full text as string
full_text = extractor.get_full_text()
```

## Output Formats

### Text File (.txt)
- Simple plain text format
- Page separators included
- Easy to read and process

### JSON File (.json)
```json
{
  "source_pdf": "document.pdf",
  "total_pages": 10,
  "total_characters": 25000,
  "pages": [
    {
      "page_number": 1,
      "text": "Page content here...",
      "char_count": 2500
    }
  ]
}
```

## Text Cleaning Features

- Removes excessive whitespace
- Fixes hyphenated words across line breaks
- Normalizes spacing
- Preserves paragraph structure
- Skips empty pages

## Notes

- Tables, images, and references are skipped (text-only extraction)
- Works best with text-based PDFs (not scanned documents)
- For OCR support, consider adding pytesseract

## Troubleshooting

If you encounter import errors:
```bash
pip install pymupdf pdfplumber
```

If extraction quality is poor, try switching libraries in the code by specifying the method:
```python
extractor.extract(method='pdfplumber')  # or 'pymupdf'
```
