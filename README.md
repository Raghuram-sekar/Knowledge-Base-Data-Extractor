# 🗂️ Multi-Modal Geological RAG System (Telesto)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![ChromaDB](https://img.shields.io/badge/ChromaDB-blue?style=for-the-badge) ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents
- [Project Overview](#🎯-project-overview)
- [What This Project Does](#🚀-what-this-project-does)
- [Key Innovation](#🔬-key-innovation)
- [Performance Highlights](#📊-performance-highlights)
- [Architecture](#🏗️-architecture)
- [Tech Stack](#🧱-tech-stack)
- [Quick Start](#💻-quick-start)

---

## 🎯 Project Overview
Advanced geological data extraction tool combining PyTorch, Qwen2-VL vision-language model, and ChromaDB vector store. Indexes logs and stratigraphic maps using 384-dimensional SentenceTransformer embeddings across 4 collections.

---

## 🚀 What This Project Does
* **The Challenge:** Geological data sheets contain dense tables, graphical stratigraphic logs, and map plots, which standard text-only RAG systems fail to parse or index.
* **Our Solution:** A multi-modal RAG system using vision-language models (Qwen2-VL) to extract features from charts and index them in vector collections.

---

## 🔬 Key Innovation
| Feature | Text-only RAG ❌ | Multi-Modal RAG ✅ | Benefit |
|---------|------------------|--------------------|---------|
| **Inputs** | Parses raw string text only | **Text + Graphical stratigraphic logs** | Extracts information from geological charts |
| **Embeddings** | TF-IDF or text vectors | **SentenceTransformer image+text embeddings** | Aligns visual map features with queries |
| **VQA** | Basic prompt matching | **Qwen2-VL-2B-Instruct VQA pipeline** | Highly accurate responses to chart queries |

---

## 📊 Performance Highlights
- ✅ **Indexes 4 vector databases** in ChromaDB.
- ✅ **Image enhancement** via CLAHE and scaling.
- ✅ **Dockerized deployment** for production scaling.

---

## 🏗️ Architecture
```\n[Core Architectural Components & Datastore Framework]\n```

---

## 🧱 Tech Stack
- PyTorch and Qwen2-VL vision-language AI models
- ChromaDB vector database collections
- Bilateral filters and Contrast Limited Adaptive Histogram Equalization (CLAHE) for image preprocessing

---

## 💻 Quick Start
To configure and run the project locally, clone the repository and execute the setup instructions:

```bash
git clone https://github.com/Raghuram-sekar/Knowledge-Base-Data-Extractor.git
cd Knowledge-Base-Data-Extractor

# Execute local setup commands:
pip install -r requirements.txt
# Run the geological extractor pipeline
```
