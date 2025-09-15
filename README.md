# Knowledge Base Data Extractor

A Python tool for extracting and converting PDF documents into structured Markdown format with image extraction capabilities. The project automatically classifies PDFs as scanned (image-based) or normal (text-based) and processes them accordingly using IBM's Docling library and Tesseract OCR.

## 🚀 Features

- **Automatic PDF Classification**: Distinguishes between scanned (image-based) and normal (text-based) PDFs
- **PDF to Markdown Conversion**: Converts PDFs to clean, structured Markdown
- **Image Extraction**: Extracts and references images from PDFs
- **OCR Support**: Processes scanned PDFs with Tesseract OCR
- **Batch Processing**: Processes entire folders of PDF files
- **Organized Output**: Creates a separate directory for each document

## 📋 Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Tesseract OCR (for scanned PDF processing)

## 🛠️ Installation

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd Knowledge-Base-Data-Extractor
   ```

2. **Install Dependencies**

   This project uses `uv` for dependency management. If you don't have `uv` installed:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Then install project dependencies:

   ```bash
   uv sync
   ```

3. **Install Tesseract OCR** (Required for OCR functionality)

   - **Ubuntu/Debian:**
     ```bash
     sudo apt-get update
     sudo apt-get install tesseract-ocr
     ```
   - **macOS:**
     ```bash
     brew install tesseract
     ```
   - **Windows:**
     Download and install from [Tesseract GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

   **Important:**

   - You may need to set the `TESSDATA_PREFIX` environment variable to the directory containing Tesseract's `tessdata` folder (usually `/usr/share/tesseract-ocr/4.00/tessdata` or similar). For example:
     ```bash
     export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/
     ```
   - This ensures Tesseract can find its language data files.

## 🎯 Usage

### Main Pipeline (Recommended)

The main entry point is `main.py`, which will:

1. Scan a folder for PDF files
2. Classify each PDF as scanned or normal using `is_scanned` from `classify_pdfs.py`
3. Process each PDF with the unified `pdf_parser` from `docling_parser.py`, enabling OCR only for scanned files
4. Save output in a structured directory under `data_docling_output/` (default)

#### Run the pipeline:

```bash
python main.py
```

By default, this processes all PDFs in the `data/` folder and saves results in `data_docling_output/`.

#### Custom Input/Output Folders

Edit the last lines of `main.py`:

```python
if __name__ == "__main__":
    workspace_path = Path(__file__).parent
    pdf_path = workspace_path / "your_input_folder"
    process_pdfs(str(pdf_path), output_base_dir="your_output_folder")
```

### Programmatic Usage

You can use the core functions directly in your own scripts:

```python
from data_extractor.classify_pdfs import is_scanned
from data_extractor.docling_parser import pdf_parser
from pathlib import Path

pdf_file = Path("path/to/file.pdf")
output_dir = Path("output/directory")
scanned = is_scanned(str(pdf_file))
pdf_parser(pdf_file, output_dir, do_ocr=scanned)
```

## 📁 Project Structure

```
Knowledge-Base-Data-Extractor/
├── main.py                     # Main entry point: classification + processing
├── pyproject.toml              # Project configuration
├── uv.lock                     # Dependency lock file
├── README.md                   # This file
├── data_extractor/             # Core extraction modules
│   ├── classify_pdfs.py        # PDF classification (scanned/normal)
│   ├── docling_parser.py       # Unified PDF parser (with/without OCR)
│   ├── docling_pdfs.py         # (Legacy) Standard PDF processing
│   └── doclings_ocr.py         # (Legacy) OCR-enabled PDF processing
├── data/                       # Input folder for PDFs
├── data_docling_output/        # Output folder for processed PDFs
└── ...
```

## 📖 How It Works

1. **PDF Classification**: `classify_pdfs.py` provides `is_scanned` and `is_scanned_folder` to determine if a PDF is scanned (image-based) or normal (text-based) using heuristics.
2. **Unified Parsing**: `docling_parser.py` provides `pdf_parser`, which takes a PDF, output directory, and a `do_ocr` flag. If `do_ocr` is `True`, Tesseract OCR is used; otherwise, text is extracted directly.
3. **Main Pipeline**: `main.py` ties it together: for each PDF, it classifies and then parses with the correct settings.
4. **Output**: Each PDF gets its own folder with a Markdown file and referenced images.

## 📝 Example Output Structure

```
data_docling_output/
└── sample_document/
    ├── sample_document.md      # Converted Markdown
    ├── data_docling_output/sample_document_artifacts               # Extracted images (if any)
    └── ...
```

## 🔧 Advanced Usage

You can still use the legacy scripts for only-normal or only-OCR processing:

- `python data_extractor/docling_pdfs.py` — process all PDFs in `data/` as normal (no OCR)
- `python data_extractor/doclings_ocr.py` — process all PDFs in `data_ocr/` with OCR

But the recommended approach is to use `main.py` for automatic classification and unified output.

## 🐛 Troubleshooting

1. **Tesseract Not Found or Language Data Error**
   - Ensure Tesseract is installed and in PATH
   - Try `tesseract --version` to verify installation
   - If you see errors about missing language data, set the `TESSDATA_PREFIX` environment variable to the correct path (see Installation section above).
2. **Memory Issues with Large PDFs**
   - Process files individually
   - Reduce `images_scale` parameter in `pdf_parser`
3. **OCR Quality Issues**
   - Check PDF quality and resolution
   - Consider preprocessing images before conversion

## 🤝 Contributing

1. **Fork and Clone**
   ```bash
   git fork <repository-url>
   git clone <your-fork-url>
   cd Knowledge-Base-Data-Extractor
   ```
2. **Set Up Development Environment**
   ```bash
   uv sync --dev
   source .venv/bin/activate
   ```
3. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Code Style**
   - Follow PEP 8
   - Use type hints and docstrings
   - Keep functions modular
5. **Testing**
   - Test with both normal and scanned PDFs
   - Check output structure and error handling

## 🙏 Acknowledgments

- [IBM Docling](https://github.com/DS4SD/docling) - Core PDF processing library
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - OCR engine

---

For questions or support, please open an issue on GitHub.
