# Multi-Modal Geological RAG System
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![ChromaDB](https://img.shields.io/badge/ChromaDB-blue?style=for-the-badge) ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

## Overview
Advanced geological data extraction tool combining PyTorch, Qwen2-VL vision-language model, and ChromaDB vector store. Indexes logs and stratigraphic maps using 384-dimensional SentenceTransformer embeddings across 4 collections.

## System Architecture
```\n[Relational Database / Core API Architecture]\n```

## Features
- Ingests PDFs containing mixed text and graphical logs.
- Indexes text and image metadata using ChromaDB vector database with SentenceTransformer embeddings.
- Visual Question Answering (VQA) using Qwen2-VL vision-language model.
- Image enhancement preprocessing (CLAHE, scaling) to improve OCR parsing.

## Tech Stack
- PyTorch and Qwen2-VL vision-language AI models
- ChromaDB vector database collections
- Bilateral filters and Contrast Limited Adaptive Histogram Equalization (CLAHE) for image preprocessing

## Getting Started
To configure and run the project locally, clone the repository and execute the setup instructions:

```bash
git clone https://github.com/Raghuram-sekar/Knowledge-Base-Data-Extractor.git
cd Knowledge-Base-Data-Extractor

# Execute local setup commands:
pip install -r requirements.txt
# Run the geological extractor pipeline
```
