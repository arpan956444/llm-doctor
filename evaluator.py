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
    {"question": "What is normal body temperature?", "ground_truth": "Normal body temperature averages 37°C (98.6°F). It varies slightly with time of day, activity, and age, but remains tightly regulated by hypothalamic control."},
    {"question": "What is blood pressure?", "ground_truth": "Blood pressure is the force exerted by circulating blood on arterial walls, expressed as systolic over diastolic pressure, reflecting cardiac output and vascular resistance."},
    {"question": "What is hypertension?", "ground_truth": "Hypertension is a chronic condition where blood pressure persistently exceeds 140/90 mmHg, increasing risk of heart disease, stroke, kidney damage, and other complications."},
    {"question": "What is hemoglobin?", "ground_truth": "Hemoglobin is an iron-containing protein in red blood cells responsible for transporting oxygen from lungs to tissues and returning carbon dioxide to lungs."},
    {"question": "What is anemia?", "ground_truth": "Anemia is a condition characterized by reduced hemoglobin or red blood cells, leading to decreased oxygen delivery, causing fatigue, weakness, pallor, and shortness of breath."},
    {"question": "What is heart rate?", "ground_truth": "Heart rate is the number of heartbeats per minute. Normal resting rate in adults ranges from 60 to 100 beats per minute."},
    {"question": "What is myocardial infarction?", "ground_truth": "Myocardial infarction occurs when blood flow to heart muscle is blocked, causing tissue death due to oxygen deprivation, commonly known as a heart attack."},
    {"question": "What is angina pectoris?", "ground_truth": "Angina is chest discomfort caused by reduced blood supply to heart muscle, often triggered by exertion or stress, relieved by rest or medication."},
    {"question": "What is atherosclerosis?", "ground_truth": "Atherosclerosis is buildup of fatty plaques in arteries, narrowing them and reducing blood flow, increasing risk of heart attack, stroke, and peripheral vascular disease."},
    {"question": "What is respiration?", "ground_truth": "Respiration is the process of inhaling oxygen and exhaling carbon dioxide, involving lungs and blood circulation to maintain cellular metabolism and energy production."},
    {"question": "What is asthma?", "ground_truth": "Asthma is a chronic inflammatory disease of airways causing reversible narrowing, leading to wheezing, breathlessness, chest tightness, and coughing, especially at night or early morning."},
    {"question": "What is pneumonia?", "ground_truth": "Pneumonia is infection of lung tissue causing inflammation of alveoli, often filled with fluid or pus, resulting in fever, cough, difficulty breathing, and chest pain."},
    {"question": "What is hypoxia?", "ground_truth": "Hypoxia is a condition where body tissues receive insufficient oxygen, leading to cellular dysfunction and potential organ damage if prolonged or severe."},
    {"question": "What is digestion?", "ground_truth": "Digestion is the process of breaking down food into smaller molecules for absorption and use by the body, involving mechanical and chemical processes."},
    {"question": "What is gastritis?", "ground_truth": "Gastritis is inflammation of stomach lining caused by infection, stress, alcohol, or drugs, leading to pain, nausea, vomiting, and possible bleeding."},
    {"question": "What is jaundice?", "ground_truth": "Jaundice is yellow discoloration of skin and eyes due to increased bilirubin levels, often caused by liver disease, bile duct obstruction, or hemolysis."},
    {"question": "What is diarrhea?", "ground_truth": "Diarrhea is frequent passage of loose or watery stools, commonly due to infection, food intolerance, or gastrointestinal disorders, leading to dehydration and electrolyte imbalance."},
    {"question": "What is constipation?", "ground_truth": "Constipation is infrequent or difficult bowel movements, often caused by low fiber intake, dehydration, or reduced physical activity, leading to discomfort and straining."},
    {"question": "What is neuron?", "ground_truth": "Neuron is the basic functional unit of nervous system responsible for transmitting electrical and chemical signals between different parts of body."},
    {"question": "What is stroke?", "ground_truth": "Stroke occurs when blood supply to part of brain is interrupted or reduced, causing brain cell death and leading to neurological deficits like paralysis or speech loss."},
    {"question": "What is epilepsy?", "ground_truth": "Epilepsy is a neurological disorder characterized by recurrent seizures due to abnormal electrical activity in brain, affecting consciousness, movement, or behavior."},
    {"question": "What is reflex action?", "ground_truth": "Reflex action is an automatic, rapid response to a stimulus without conscious control, mediated through spinal cord pathways for protection and quick reaction."},
    {"question": "What is coma?", "ground_truth": "Coma is a prolonged state of unconsciousness where a person cannot be awakened and does not respond to stimuli due to severe brain dysfunction."},
    {"question": "What is hormone?", "ground_truth": "Hormones are chemical messengers secreted by endocrine glands that regulate body functions such as growth, metabolism, reproduction, and homeostasis."},
    {"question": "What is diabetes mellitus?", "ground_truth": "Diabetes mellitus is a metabolic disorder characterized by high blood glucose levels due to insufficient insulin production or impaired insulin action."},
    {"question": "What is insulin?", "ground_truth": "Insulin is a hormone produced by pancreas that regulates blood glucose by facilitating uptake of glucose into cells for energy and storage."},
    {"question": "What is hyperthyroidism?", "ground_truth": "Hyperthyroidism is condition where thyroid gland produces excess hormones, increasing metabolism, causing weight loss, rapid heartbeat, nervousness, and heat intolerance."},
    {"question": "What is hypothyroidism?", "ground_truth": "Hypothyroidism is condition of reduced thyroid hormone production, leading to slowed metabolism, fatigue, weight gain, cold intolerance, and depression."},
    {"question": "What is infection?", "ground_truth": "Infection occurs when microorganisms like bacteria, viruses, or fungi invade body tissues, multiply, and cause disease by damaging cells or triggering immune responses."},
    {"question": "What is immunity?", "ground_truth": "Immunity is the body’s ability to resist infections through defense mechanisms including physical barriers, immune cells, and antibodies that recognize and eliminate pathogens."},
    {"question": "What is vaccination?", "ground_truth": "Vaccination is the process of administering weakened or inactive pathogens to stimulate immune response, providing protection against specific infectious diseases."},
    {"question": "What is antibiotic?", "ground_truth": "Antibiotics are drugs used to treat bacterial infections by killing bacteria or inhibiting their growth, but they are ineffective against viral infections."},
    {"question": "What is virus?", "ground_truth": "Virus is a microscopic infectious agent that requires host cells to replicate, causing diseases by invading cells and disrupting their normal functions."},
    {"question": "What is dehydration?", "ground_truth": "Dehydration is a condition resulting from excessive loss of body fluids, leading to imbalance in electrolytes and impaired physiological functions."},
    {"question": "What is electrolyte?", "ground_truth": "Electrolytes are charged minerals like sodium and potassium that regulate nerve function, muscle contraction, hydration, and acid-base balance in body."},
    {"question": "What is edema?", "ground_truth": "Edema is swelling caused by accumulation of excess fluid in tissues, commonly seen in conditions like heart failure, kidney disease, or inflammation."},
    {"question": "What is acidosis?", "ground_truth": "Acidosis is a condition where blood pH falls below normal due to excess acid or loss of bicarbonate, affecting enzyme activity and organ function."},
    {"question": "What is alkalosis?", "ground_truth": "Alkalosis is a condition where blood pH rises above normal due to loss of acid or excess bicarbonate, affecting normal cellular processes."},
    {"question": "What is diagnosis?", "ground_truth": "Diagnosis is the process of identifying a disease by evaluating symptoms, medical history, physical examination, and laboratory or imaging investigations."},
    {"question": "What is prognosis?", "ground_truth": "Prognosis refers to the expected outcome or course of a disease, including chances of recovery, complications, and survival based on clinical evaluation."},
    {"question": "What is treatment?", "ground_truth": "Treatment is the management of disease through medications, surgery, lifestyle changes, or therapies aimed at curing, controlling, or relieving symptoms."},
    {"question": "What is pathology?", "ground_truth": "Pathology is the study of diseases, including their causes, development, structural changes, and effects on body tissues and organs."},
    {"question": "What is inflammation?", "ground_truth": "Inflammation is a protective response of body to injury or infection, characterized by redness, swelling, heat, pain, and loss of function."},
    {"question": "What is metabolism?", "ground_truth": "Metabolism includes all chemical reactions in body that maintain life, including processes of breaking down nutrients and producing energy for cellular activities."},
    {"question": "What is homeostasis?", "ground_truth": "Homeostasis is the ability of body to maintain stable internal environment despite external changes, regulating temperature, pH, fluids, and other vital parameters."},
    {"question": "What is oxygen saturation?", "ground_truth": "Oxygen saturation is the percentage of hemoglobin carrying oxygen in blood, normally between 95% and 100%, indicating adequate oxygenation of tissues."},
    {"question": "What is pulse?", "ground_truth": "Pulse is rhythmic expansion of arteries due to heartbeats, used clinically to assess heart rate, rhythm, and strength of blood circulation."},
    {"question": "What is organ failure?", "ground_truth": "Organ failure occurs when an organ loses its ability to function properly, often due to disease, injury, or insufficient blood supply."},
    {"question": "What is infection control?", "ground_truth": "Infection control involves measures like hygiene, sterilization, and isolation to prevent spread of infections in healthcare and community settings."},
    {"question": "What is public health?", "ground_truth": "Public health focuses on prevention of disease, promotion of health, and prolonging life through organized community efforts, policies, education, and healthcare services."},
    {"question": "What is Cardiac Tamponade?", "ground_truth": "Life-threatening disorder occurring when pericardial fluid accumulates under pressure; effusions rapidly increasing in size may cause an elevated intrapericardial pressure."},
    {"question": "What defines Cor Pulmonale?", "ground_truth": "Heart failure resulting from pulmonary disease. Most commonly due to COPD; other causes include pulmonary fibrosis, pneumoconioses, recurrent pulmonary emboli..."},
    {"question": "What is Celiac Disease?", "ground_truth": "Celiac disease is a disease of the digestive system that damages the small intestine and interferes with the absorption of nutrients from food. It occurs when the body reacts abnormally to gluten."},
    {"question": "What are the symptoms of Tetanus?", "ground_truth": "Jaw stiffness followed by spasms (trismus). Stiffness of neck or other muscles, dysphagia, irritability, hyperreflexia; late, painful convulsions precipitated by minimal stimuli."},
    {"question": "What is Dilated Cardiomyopathy?", "ground_truth": "A cause of systolic dysfunction, this represents a group of disorders that lead to congestive heart failure. Symptoms and signs of congestive heart failure: Exertional dyspnea, cough, fatigue, paroxysmal nocturnal dyspnea..."},
    {"question": "What defines Atrial Septal Defect?", "ground_truth": "Patients with small defects are usually asymptomatic and have a normal life span. Large shunts symptomatic by age 40, including exertional dyspnea, fatigue, and palpitations."},
    {"question": "What is Candidiasis?", "ground_truth": "Candidiasis is an infection caused by a species of the yeast Candida, usually Candida albicans. This is a common cause of vaginal infections in women."},
    {"question": "What are the symptoms of Tuberculosis?", "ground_truth": "Symptoms progressive and include cough, dyspnea, fever, night sweats, weight loss, and hemoptysis. In primary infection, mid-lung field infiltrates with regional lymphadenopathy; pleural effusion common."},
    {"question": "What are Calcium Channel Blockers?", "ground_truth": "Calcium channel blockers are medicines that slow the movement of calcium into the cells of the heart and blood vessels. This, in turn, relaxes blood vessels, increases the supply of oxygen-rich blood to the heart, and reduces the heart's workload."},
    {"question": "What is Carbon Monoxide Poisoning?", "ground_truth": "Carbon monoxide (CO) poisoning occurs when carbon monoxide gas is inhaled. CO is a colorless, odorless, highly poisonous gas that is produced by incomplete combustion."},
    {"question": "What characterizes Tinea Versicolor?", "ground_truth": "Finely scaling patches on upper trunk and upper arms, usually asymptomatic. Lesions yellowish or brownish on pale skin, or hypopigmented on dark skin."},
    {"question": "What characterizes Urticaria (Hives)?", "ground_truth": "Pale or red, evanescent, edematous papules or plaques surrounded by red halo with severe itching or stinging; wheals appear suddenly and resolve in hours."},
    {"question": "What characterizes Vitiligo?", "ground_truth": "Depigmented white patches surrounded by a normal, hyperpigmented, or occasionally inflamed border. Hairs in affected area usually turn white."}
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