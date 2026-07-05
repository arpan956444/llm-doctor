import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

DATA_PATH = "data/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DB_FAISS_PATH = "vectorstore/db_faiss"

# Configuration for Model Combinations
MODEL_COMBINATIONS = {
    "Setup_Default": { # Corresponds to Model 1 in the paper (High Speed & Efficiency)
        "llm": "meta-llama/llama-4-scout-17b-16e-instruct",
        "embeddings": "nomic-ai/nomic-embed-text-v1.5",
        "vectorstore": "vectorstore/db_faiss_default",
        "rerank": False,
        "answer_style": "concise"
    },
    "Setup_Enhanced": { # Corresponds to Model 2 in the paper (Advanced Reasoning & Agentic Tasks)
        "llm": "openai/gpt-oss-20b",
        "embeddings": "sentence-transformers/all-distilroberta-v1",
        "vectorstore": "vectorstore/db_faiss_enhanced",
        "rerank": False,
        "answer_style": "concise"
    },
    "Setup_Medical": { # Corresponds to Model 3 in the paper (High Precision & Multilingual Support, with Reranking)
        "llm": "qwen/qwen3-32b",
        "embeddings": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "vectorstore": "vectorstore/db_faiss_medical",
        "rerank": True,
        "reranker_model": "BAAI/bge-reranker-base",
        "answer_style": "detailed" # Example: this config provides more detailed answers
    },
    "Setup_Model4": { # Corresponds to Model 4 in the paper (LLM Doctor Main System)
        "llm": "llama-3.1-8b-instant",
        "embeddings": "sentence-transformers/all-mpnet-base-v2",
        "vectorstore": "vectorstore/db_faiss_model4",
        "rerank": False,
        "answer_style": "concise"
    }
}