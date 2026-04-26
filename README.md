# 🩺 LLM Doctor: AI Medical Chatbot

This project is a Retrieval-Augmented Generation (RAG) system designed to answer medical questions using specific local PDF documentation. It leverages high-performance components: **Groq (Llama 3.1)** for fast inference, **HuggingFace** for local embeddings, and **FAISS** for efficient vector storage.

---

## 🚀 Workflow Overview

The system follows a standard RAG pipeline:

1.  **Data Ingestion:** PDF files are loaded from the `data/` directory.
2.  **Chunking:** Documents are split into smaller text segments (chunks) to fit within the LLM's context window.
3.  **Embedding:** Each chunk is converted into a numerical vector using a HuggingFace transformer model.
4.  **Vector Storage:** These vectors are stored in a local FAISS index for high-speed similarity searching.
5.  **Retrieval:** When a user asks a question, the system searches the FAISS index for the most relevant text chunks.
6.  **Generation:** The relevant chunks + the user's question are sent to the Llama-3.1 model via Groq, which generates a concise medical answer.
7.  **Evaluation:** The `evaluator.py` script compares the AI's answers against ground truth data using BERTScore, Token F1, and Ragas metrics.

---

## 🛠️ Step-by-Step Setup Guide

### 1. Prerequisites
*   Python 3.9 or higher installed.
*   A **Groq API Key** (Get one at [console.groq.com](https://console.groq.com/)).

### 2. Installation
Clone this repository and navigate into the project folder:

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements-old.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY="your_api_key_here"
```

### 4. Prepare Your Data
1.  Create a folder named `data/` in the project root.
2.  Place any medical PDF books or documents you want the chatbot to "read" inside this folder.

### 5. Initialize the Vector Database
Before running the app, you must process the PDFs and create the FAISS index:
```bash
python -m app.components.data_loader
```
*What this does:* It creates a `vectorstore/` folder containing your processed document embeddings.

---

## 🏃 Running the Project

### Start the Web Application
Launch the Flask-based chat interface:
```bash
python -m app.application
```
*   **URL:** Open `http://127.0.0.1:5000` in your browser.
*   **Interaction:** Ask any medical question based on the PDFs you uploaded.

### Run Performance Evaluations
To see how accurate your chatbot is, run the evaluator:
```bash
python evaluator.py
```
*   **Result:** This will print a summary table of scores (F1, BERTScore) and generate a `full_evaluation.png` chart showing Faithfulness and Relevancy.

---

## 📂 Project Structure

| File/Folder | Purpose |
| :--- | :--- |
| `app/application.py` | The main Flask entry point for the web UI. |
| `app/components/` | Core logic: `pdf_loader`, `embeddings`, `llm`, and `vector_store`. |
| `app/common/` | Shared utilities: Custom logging and exception handling. |
| `app/config/` | Configuration settings (Chunk size, Paths, etc.). |
| `evaluator.py` | Testing suite using Ragas and BERTScore. |
| `data/` | (User Created) Folder for source PDF files. |
| `vectorstore/` | (Auto-Generated) Local FAISS database files. |

---

## 📊 Evaluation Metrics Explained

*   **Token F1:** Measures the overlap between the AI's answer and the ground truth.
*   **BERTScore:** Uses embeddings to measure semantic similarity (better for paraphrasing).
*   **Faithfulness (Ragas):** Ensures the AI isn't "hallucinating" (making things up not found in the context).
*   **Answer Relevancy (Ragas):** Measures how well the answer addresses the specific question asked.

---

## ⚠️ Important Notes
*   **Safety:** This is an AI project for educational purposes. It is NOT a substitute for professional medical advice.
*   **FAISS Deserialization:** The project uses `allow_dangerous_deserialization=True`. Only load `.faiss` files you have generated yourself.
*   **API Limits:** Groq has rate limits; if you run the evaluator too quickly, you may encounter 429 errors.