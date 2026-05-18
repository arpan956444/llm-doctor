import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

DATA_PATH = "data/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DB_FAISS_PATH = "vectorstore/db_faiss"

# Configuration for Model Combinations (Fully Updated & Swapped)
MODEL_COMBINATIONS = {
    "Setup_Default": {
        # Swapped to Llama 4 Scout: Extremely fast Mixture of Experts (MoE) with a 128k context window.
        "llm": "meta-llama/llama-4-scout-17b-16e-instruct",
        # Swapped to Nomic: Just as light as MiniLM (~280MB) but supports a massive 8k token input context.
        "embeddings": "nomic-ai/nomic-embed-text-v1.5",
        "vectorstore": "vectorstore/db_faiss_default",
        "rerank": False
    },
    "Setup_Enhanced": {
        # Swapped to GPT-OSS 20B: Built specifically for high-throughput, advanced reasoning on Groq LPUs.
        "llm": "openai/gpt-oss-20b",
        # Swapped to DistilRoBERTa: A highly accurate, lightweight (~290MB) semantic upgraded encoder.
        "embeddings": "sentence-transformers/all-distilroberta-v1",
        "vectorstore": "vectorstore/db_faiss_enhanced",
        "rerank": False
    },
    "Setup_Medical": {
        # Swapped to Qwen3: Best-in-class open-source model for intense code syntax, structured outputs, and logic.
        "llm": "qwen/qwen3-32b",
        # Swapped to Multilingual MiniLM: Lightweight tool optimizing semantic tracking across diverse scripts.
        "embeddings": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "vectorstore": "vectorstore/db_faiss_medical",
        "rerank": True,
        "reranker_model": "BAAI/bge-reranker-base"
    }
}