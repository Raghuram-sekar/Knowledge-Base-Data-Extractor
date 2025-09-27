# Knowledge Base Data Extractor

A Python tool for extracting and converting PDF documents into structured Markdown format with image extraction capabilities. The project automatically classifies PDFs as scanned (image-based) or normal (text-based) and processes them accordingly using IBM's Docling library and Tesseract OCR.

## 📋 Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Tesseract OCR (for scanned PDF processing)

## 🛠️ Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Telesto-Amrita/Knowledge-Base-Data-Extractor.git
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

   Then activate the venv

   ```bash
   source .venv/bin/activate
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

   - You may need to set the `TESSDATA_PREFIX` environment variable to the directory containing Tesseract's `tessdata` folder (usually `/usr/share/tesseract/tessdata/` or similar). For example:
     ```bash
     export TESSDATA_PREFIX=/usr/share/tesseract/tessdata/
     ```
   - This ensures Tesseract can find its language data files.

4. Set Environment Variables
   ```bash
   cp env.example .env
   ```

5. To run Parser
   ```bash
   uv run src/run_parser.py
   ```

---
