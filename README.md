# Multi-Modal Geological RAG System

Advanced geological data extraction tool combining PyTorch, Qwen2-VL vision-language model, and ChromaDB vector store. Indexes logs and stratigraphic maps using 384-dimensional SentenceTransformer embeddings across 4 collections.

## Features
- Ingests PDFs containing mixed text and graphical logs.
- Indexes text and image metadata using ChromaDB vector database with SentenceTransformer embeddings.
- Visual Question Answering (VQA) using Qwen2-VL vision-language model.
- Image enhancement preprocessing (CLAHE, scaling) to improve OCR parsing.

## Tech Stack
- PyTorch
- Qwen2-VL-2B-Instruct
- ChromaDB
- SentenceTransformers
- uv
- Docker

## Getting Started
To configure and run the project locally, clone the repository and execute the setup instructions:

```bash
git clone https://github.com/Raghuram-sekar/Knowledge-Base-Data-Extractor.git
cd Knowledge-Base-Data-Extractor

# Execute local setup commands:
pip install -r requirements.txt
# Run the geological extractor pipeline
```
