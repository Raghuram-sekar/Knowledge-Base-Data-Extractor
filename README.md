# 🌍 Telesto Knowledge Base Data Extractor
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [What This Project Does](#-what-this-project-does)
- [Key Innovation](#-key-innovation)
- [Performance Highlights](#-performance-highlights)
- [Architecture](#-architecture)
- [Methodology & Technical Details](#-methodology--technical-details)
- [Original Documentation & Setup Guide](#-original-documentation--setup-guide)


---

## 🎯 Project Overview
# 🌍 Telesto Knowledge Base Data Extractor

An advanced Python tool for extracting and processing geological documents and well log data. The system automatically classifies PDFs as scanned or normal documents, processes them using IBM's Docling library and Tesseract OCR, and provides AI-powered geological analysis capabilities.

## 🎯 Project Overvi... (Refer to the Original Documentation section below for full details).

---

## 🚀 What This Project Does
This project implements a secure, high-efficiency data intelligence pipeline, enabling local processing, edge decisions, or automated agentic API workflows.

---

## 🔬 Key Innovation
| Feature | Traditional Parsing ❌ | Telesto Extractor ✅ | Benefit |
|---------|------------------------|----------------------|---------|
| **PDF Extraction** | Basic pdf-plumber text readers | **IBM Docling + Tesseract OCR** | Parses both scanned maps and normal texts |
| **Well logs** | Ignored or parsed manually | **Automated LAS file parser** | Direct depositional environment detection |
| **Package Manager** | Slow pip installations | **Fast Astral `uv` tool sync** | Instant virtual environment synchronization |

---

## 📊 Performance Highlights
- ✅ **Processes geological data** and stratigraphic maps.
- ✅ **Fast workspace sync** using `uv` packaging.
- ✅ **Classifies PDF styles** dynamically to select OCR paths.

---

## 🏗️ Architecture
```mermaid
graph TD
    PDF[Geological PDF / LAS File] -->|OCR Scan detection| Classify[PDF Classifier]
    Classify -->|Scanned PDF| Tesseract[Tesseract OCR Engine]
    Classify -->|Digital PDF| Docling[IBM Docling Library]
    Classify -->|Well Log LAS file| LAS[LAS File Reader]
    Tesseract -->|Extract Text| Extracted[Geological Data Stream]
    Docling -->|Extract Text| Extracted
    LAS -->|depositional environment| Extracted
    Extracted -->|AI geological interpretation| AI[AI Insights scoring]
```


---

## 📖 Original Documentation & Setup Guide
# 🌍 Telesto Knowledge Base Data Extractor

An advanced Python tool for extracting and processing geological documents and well log data. The system automatically classifies PDFs as scanned or normal documents, processes them using IBM's Docling library and Tesseract OCR, and provides AI-powered geological analysis capabilities.

## 🎯 Project Overview

This project is designed for **geological data analysis** and **oil & gas exploration** with capabilities for:
- **Document Processing**: Extract knowledge from research papers, technical reports, and geological documents
- **Well Log Analysis**: Process LAS files for depositional environment identification
- **AI-Powered Insights**: Automated geological interpretation with confidence scoring
- **Knowledge Base Construction**: Build searchable geological knowledge repositories

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

4. **Set Environment Variables**
   ```bash
   cp env.example .env
   ```

5. **Run Parser**
   ```bash
   uv run src/run_parser.py
   ```

## 📁 Project Structure

```
Knowledge-Base-Data-Extractor/
├── docs/                           # 📚 Organized documentation
│   ├── project-planning/           # Project SOW, strategy, approach
│   ├── technical-specifications/   # Data pipelines, formats, specs
│   ├── cost-analysis/             # Financial analysis and estimates
│   ├── research-data/             # Research materials and analysis
│   └── system-architecture/       # System design and deployment
├── src/                           # 💻 Source code
├── data/                          # 📊 Input/output data
├── Telesto Data/                  # 🏺 Geological data (well logs, papers)
├── services/                      # 🔧 Enhanced microservices architecture
├── frontend/                      # 🖥️ Web dashboard interface
└── archived-data/                 # 📦 Archived and legacy files
```

## 📖 Documentation

All project documentation has been organized into logical categories:

- **📋 Project Planning**: [docs/project-planning/](./docs/project-planning/) - SOW, project approach, strategy
- **🔧 Technical Specs**: [docs/technical-specifications/](./docs/technical-specifications/) - Data pipelines, formats
- **💰 Cost Analysis**: [docs/cost-analysis/](./docs/cost-analysis/) - Financial planning and estimates  
- **📊 Research Data**: [docs/research-data/](./docs/research-data/) - Research materials and analysis
- **🏗️ System Architecture**: [docs/system-architecture/](./docs/system-architecture/) - System design guides

See [docs/README.md](./docs/README.md) for complete documentation index.

---


---
