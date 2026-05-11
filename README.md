# Local Multimodal RAG System

A fully local, end-to-end Retrieval-Augmented Generation (RAG) pipeline for complex PDF documents. Built specifically to run efficiently on Apple Silicon (M-series) hardware, this project enables private, secure document querying with zero API costs.

## Key Features

* 100% Private Execution: Runs lightweight AI models completely on-device using llama.cpp and sentence-transformers, with aggressive MPS/CPU thread optimizations for macOS stability.
* Advanced PDF Parsing: Utilizes PyMuPDF with custom geometric column sorting to accurately extract text from multi-column layouts while simultaneously extracting and linking embedded document images.
* Parent-Child Semantic Chunking: Generates dense LLM summaries ("child" vectors) to improve retrieval math, while passing the raw, structural text ("parent" chunks) to the generator for maximum factual accuracy.
* Multi-Stage Hybrid Search: 
  * Generates semantic variations of the user prompt via query expansion.
  * Combines FAISS (dense vector semantic search) and BM25 (sparse keyword search).
  * Fuses results via Reciprocal Rank Fusion (RRF) and reranks using a cross-encoder for maximum relevance.
* Automated MLOps Testing: Includes a custom synthetic data generator and an "LLM-as-a-Judge" evaluation suite to automatically grade the system's retrieval accuracy.
* Interactive Web UI: A Gradio chat interface that provides cited sources, confidence scores, and processing traces for every answer.

## Evaluation Metrics

Evaluated on a synthetically generated 100-question Golden Set using the automated LLM-as-a-Judge pipeline (Retrieval Depth / Top-K = 3).

| Metric |              | Score | Description |
| Semantic Hit Rate     | 81.0% | The search engine successfully retrieved chunks with the factual answer. |
| Exact Match Hit Rate  | 60.0% | The search engine retrieved the exact original source chunk ID. |
| Faithfulness          | 60.0% | The generated response was fully grounded in the retrieved text, avoiding hallucinations. |

## Tech Stack

* Generator Models: Qwen2-1.5B-Instruct (GGUF Quantized) or google/flan-t5-large
* Embeddings: sentence-transformers/paraphrase-MiniLM-L3-v2
* Vector Database: FAISS
* Keyword Search: rank_bm25
* Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
* PDF Processing: PyMuPDF (fitz)
* Frontend UI: Gradio

## Setup & Installation

1. Clone the repository and set up the environment:
    git clone https://github.com/YOUR_USERNAME/multimodal-rag-system.git
    cd multimodal-rag-system
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .

2. Download the local AI model:
    Download qwen2-1_5b-instruct-q8_0.gguf and place it in the models/ folder at the root of the project. If no GGUF model is found, the system will fall back to the Hugging Face pipeline model specified in your environment variables.

3. Configure your Environment:
    Copy the .env.example file to a new file named .env. The default parameters are pre-configured for Apple Silicon stability.

4. Add your PDFs:
    Place any PDF documents you want to analyze inside the data/ folder.

## Quickstart

1. Build the Search Index:
    This process extracts text and images, generates AI summaries, and builds the local FAISS and BM25 databases.
    
    rag-index

2. Launch the Web UI:
    Starts the Gradio application. Access the provided local URL in your browser to start querying your documents.
    
    rag-app

## Evaluation Pipeline

To test the system against your own documents or replicate the benchmark scores:

1. Generate synthetic test questions:
    python src/multimodal_rag/scripts/generate_eval.py

2. Run the LLM Judge:
    Paste your generated queries into the EVAL_DATA list inside the evaluate script, then run:
    
    python src/multimodal_rag/scripts/evaluate.py

## Project Layout

    src/multimodal_rag/      # Core package source code
    data/                    # PDF input directory
    embeddings/              # Generated FAISS index, BM25, and metadata
    models/                  # Local GGUF model weights
    cache/                   # Disk cache for embeddings and responses

## Configuration Options

Key environment variables available in .env:

* EMBEDDING_MODEL – Sentence embedding model (Default: paraphrase-MiniLM-L3-v2)
* RERANKER_MODEL – Cross-encoder reranker model
* GENERATOR_MODEL – Text generation fallback model (Default: google/flan-t5-large)
* RETRIEVAL_TOP_K – Initial candidate pool size
* RERANK_TOP_K – Final reranked context size
* MAX_NEW_TOKENS – Maximum generation length
* MIN_CHUNK_WORDS – Filter threshold for very short chunks

## Notes

* Embeddings must be rebuilt using the rag-index command whenever PDFs are added or removed from the data/ directory.