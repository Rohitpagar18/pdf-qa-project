# PDF Q&A Project
A documents-based question-answering system using LangChain, Ollama, and ChromaDB.


## Project Overview
Retrieval-Augmented Generation pipeline for querying PDF documents using:
- Ollama LLMs
- ChromaDB vector storage
- Semantic search with nomic-embed-text

## Features
✅ PDF Document Ingestion  
✅ Chunking Strategy (1100 char windows)  
✅ CLI Interface for Querying  
✅ Docker Support  

## Installation
```bash
git clone https://github.com/Rohitpagar18/pdf-qa-project.git
cd pdf-qa-rag-system
pip install -r requirements.txt


## Usage
```bash
# Populate database
python populate_database.py --reset

# Query documents
python query_data.py \"Your question here\"
