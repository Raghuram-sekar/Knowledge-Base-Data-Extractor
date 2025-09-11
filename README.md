# Knowledge Base Data Extractor

A Python tool for extracting and converting PDF documents into structured Markdown format with image extraction capabilities. This project supports both regular PDFs and scanned PDFs with OCR functionality using IBM's Docling library.

## 🚀 Features

- **PDF to Markdown Conversion**: Convert PDF documents to clean, structured Markdown
- **Image Extraction**: Automatically extract and reference images from PDFs
- **OCR Support**: Process scanned PDFs with Tesseract OCR integration
- **Batch Processing**: Process entire folders of PDF files
- **Structured Output**: Organized output with separate directories per document

## 📋 Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Tesseract OCR (for scanned PDF processing)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Knowledge-Base-Data-Extractor
```

### 2. Install Dependencies

This project uses `uv` for dependency management. If you don't have `uv` installed:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install project dependencies:

```bash
# Install dependencies using uv
uv sync
```

### 3. Install Tesseract OCR (Required for OCR functionality)

#### Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

#### macOS:

```bash
brew install tesseract
```

#### Windows:

Download and install from [Tesseract GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

## 🎯 Usage

### Basic PDF Processing

For regular PDF documents (non-scanned):

```bash
# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run the basic PDF processor
python data_extractor/docling_pdfs.py
```

This will process PDFs from the `data/` folder and save output to `data_docling/`.

### OCR PDF Processing

For scanned PDF documents that require OCR:

```bash
# Run the OCR PDF processor
python data_extractor/doclings_ocr.py
```

This will process PDFs from the `data_ocr/` folder and save output to `data_docling_ocr/`.

### Custom Usage

You can also use the functions programmatically:

```python
from data_extractor.docling_pdfs import process_pdf_folder
from data_extractor.doclings_ocr import process_scanned_pdfs

# Process regular PDFs
process_pdf_folder("path/to/pdf/folder", "output/directory")

# Process scanned PDFs with OCR
process_scanned_pdfs("path/to/scanned/pdfs", "ocr/output/directory")
```

## 📁 Project Structure

```
Knowledge-Base-Data-Extractor/
├── main.py                     # Main entry point
├── pyproject.toml             # Project configuration
├── uv.lock                    # Dependency lock file
├── README.md                  # This file
├── data_extractor/            # Core extraction modules
│   ├── docling_pdfs.py       # Standard PDF processing
│   └── doclings_ocr.py       # OCR-enabled PDF processing
├── data/                      # Input folder for regular PDFs
├── data_ocr/                  # Input folder for scanned PDFs
├── data_docling/              # Output folder for processed PDFs
└── data_docling_ocr/          # Output folder for OCR-processed PDFs
```

## 📖 How It Works

### Standard PDF Processing ([docling_pdfs.py](data_extractor/docling_pdfs.py))

1. Scans the input folder for PDF files
2. Converts each PDF to Markdown using Docling
3. Extracts images and saves them with references
4. Creates organized output structure

### OCR PDF Processing ([doclings_ocr.py](data_extractor/doclings_ocr.py))

1. Enables OCR using Tesseract
2. Processes scanned or image-based PDFs
3. Extracts text through OCR with full-page processing
4. Generates high-resolution images (2x scale)

## 🔧 Configuration

### Customizing Input/Output Paths

Edit the `main()` function in either processor file:

```python
def main():
    workspace_path = Path(__file__).parent.parent

    # Change these paths as needed
    input_folder = workspace_path / "your_input_folder"

    # Process with custom output directory
    process_pdf_folder(str(input_folder), "your_output_folder")
```

### Pipeline Options

Modify pipeline options for different processing needs:

```python
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True  # Enable/disable OCR
pipeline_options.generate_picture_images = True  # Extract images
pipeline_options.images_scale = 2.0  # Image resolution multiplier
```

## 🤝 Contributing

### Development Setup

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

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings for all functions
- Keep functions focused and modular

### Testing

Before submitting a PR:

1. Test with sample PDF files
2. Verify OCR functionality with scanned documents
3. Check that output structure is correct
4. Ensure error handling works properly

### Submitting Changes

1. **Commit Changes**

   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

2. **Push to Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request**
   - Provide clear description of changes
   - Include examples if adding new features
   - Reference any related issues

## 📝 Example Output Structure

```
data_docling/
└── sample_document/
    ├── sample_document.md      # Converted Markdown
    ├── images/                 # Extracted images
    │   ├── image_1.png
    │   └── image_2.png
    └── ...
```

## 🐛 Troubleshooting

### Common Issues

1. **Tesseract Not Found**

   - Ensure Tesseract is installed and in PATH
   - Try `tesseract --version` to verify installation

2. **Memory Issues with Large PDFs**

   - Process files individually
   - Reduce `images_scale` parameter

3. **OCR Quality Issues**
   - Check PDF quality and resolution
   - Consider preprocessing images before conversion

## 🙏 Acknowledgments

- [IBM Docling](https://github.com/DS4SD/docling) - Core PDF processing library
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - OCR engine

---

For questions or support, please open an issue on GitHub.
