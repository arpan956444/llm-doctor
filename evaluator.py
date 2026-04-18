import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from datasets import Dataset
from bert_score import score as bert_score_func

# Updated Ragas imports to avoid Deprecation Warnings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Internal imports
from app.components.retriever import create_qa_chain
from app.components.llm import load_llm
from app.components.embeddings import get_embedding_model

load_dotenv()

test_questions = [
    {
        "question": "What is the common treatment for a cold?",
        "ground_truth": "Rest, hydration, and over-the-counter medicines like decongestants."
    },
    {
        "question": "What are symptoms of hypertension?",
        "ground_truth": "Hypertension often has no symptoms but can cause headaches, shortness of breath, or nosebleeds."
    }
]

def calculate_token_f1(prediction, ground_truth):
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common: return 0, 0, 0
    prec = len(common) / len(pred_tokens) if pred_tokens else 0
    rec = len(common) / len(gt_tokens) if gt_tokens else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1

def run_evaluation():
    qa_chain = create_qa_chain()
    llm_obj = load_llm()
    embed_model = get_embedding_model()

    # Wrapper for Ragas
    ragas_llm = LangchainLLMWrapper(llm_obj)
    ragas_emb = LangchainEmbeddingsWrapper(embed_model)
    
    results_data = []
    print("--- Starting RAG Retrieval and BERTScore ---")
    
    for item in test_questions:
        print(f"Testing Question: {item['question']}")
        response = qa_chain.invoke({"query": item["question"]})
        
        answer = response.get("result", "")
        source_docs = response.get("source_documents", [])
        context = [doc.page_content for doc in source_docs]
        if not context: context = ["No context found in vectorstore"]

        p, r, f1 = calculate_token_f1(answer, item["ground_truth"])
        P_b, R_b, F1_b = bert_score_func([answer], [item["ground_truth"]], lang="en")

        results_data.append({
            "question": item["question"],
            "answer": answer,
            "contexts": context,
            "ground_truth": item["ground_truth"],
            "f1": f1,
            "bert_f1": F1_b.item()
        })

    dataset_dict = {
        "question": [item["question"] for item in results_data],
        "answer": [item["answer"] for item in results_data],
        "contexts": [item["contexts"] for item in results_data],
        "ground_truth": [item["ground_truth"] for item in results_data]
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    print("\n--- Calculating Ragas Metrics ---")
    
    # Note: Groq often fails on answer_relevancy because it doesn't support n > 1
    # Faithfulness is generally safer with Groq.
    ragas_results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_emb
    )
    
    df = pd.DataFrame(results_data)
    return df, ragas_results

def plot_evaluations(df, rag_results):
    plt.figure(figsize=(14, 6))

    # Graph 1: Token F1 vs BERTScore
    plt.subplot(1, 2, 1)
    # Ensure values are float for plotting
    df["f1"] = df["f1"].astype(float)
    df["bert_f1"] = df["bert_f1"].astype(float)
    
    df_melted = df.melt(id_vars="question", value_vars=["f1", "bert_f1"])
    sns.barplot(data=df_melted, x="variable", y="value", hue="question")
    plt.title("F1 vs BERTScore per Question")
    plt.ylim(0, 1.1)
    plt.ylabel("Score")

    # Graph 2: Ragas Overall Scores
    plt.subplot(1, 2, 2)
    
    metric_names = ["faithfulness", "answer_relevancy"]
    scores = []
    
    # EvaluationResult object behaves like a dict but doesn't have .get()
    # It also might return NaN if the LLM provider (Groq) doesn't support 'n' variations
    for name in metric_names:
        try:
            # Access score via indexing which EvaluationResult supports
            val = rag_results[name]
            
            # Handle potential array/list results or NaN
            if isinstance(val, (list, pd.Series, pd.Index)):
                val = val[0]
            
            val_float = float(val)
            if pd.isna(val_float):
                print(f"Warning: Metric '{name}' returned NaN. Likely due to LLM provider limitations.")
                scores.append(0.0)
            else:
                scores.append(val_float)
        except Exception as e:
            print(f"Warning: Could not retrieve metric '{name}': {e}")
            scores.append(0.0)

    rag_df = pd.DataFrame({"Metric": metric_names, "Score": scores})
    
    sns.barplot(data=rag_df, x="Metric", y="Score", hue="Metric", palette="viridis", legend=False)
    plt.title("Overall RAG Quality (Ragas)")
    plt.ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig("full_evaluation.png")
    print("\nSUCCESS: Graph saved as 'full_evaluation.png' in project root.")
    plt.show()

if __name__ == "__main__":
    try:
        df, rag_scores = run_evaluation()
        print("\n--- Summary Table ---")
        print(df[['question', 'f1', 'bert_f1']])
        print("\n--- Ragas Overall Scores ---")
        print(rag_scores)
        plot_evaluations(df, rag_scores)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()