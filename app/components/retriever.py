from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.components.reranker import get_reranker_model, rerank_documents # NEW IMPORT
logger = get_logger(__name__)

DEFAULT_PROMPT_TEMPLATE = """
You are an expert medical assistant. Your goal is to answer medical questions factually and concisely, using ONLY the provided context.
If the context does not contain enough information to answer the question, state "I couldn't find a definitive answer in the provided medical documents for this question. Please provide more context or rephrase your query." Do not try to make up an answer.
Keep your answer to a maximum of 4-5 sentences.

Context:
{context}

Question:
{question}

Answer:
"""

DETAIL_PROMPT_TEMPLATE = """
You are a highly knowledgeable medical researcher. Your goal is to provide a comprehensive and factual answer to the medical question, drawing ONLY from the provided context.
Elaborate on the key aspects and details found in the context to give a thorough explanation.
If the context does not provide sufficient information, clearly state that you cannot fully answer based on the given documents. Avoid external knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt(style: str = "concise"):
    if style == "detailed":
        template = DETAIL_PROMPT_TEMPLATE
    else: # Default to concise
        template = DEFAULT_PROMPT_TEMPLATE
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

from app.config.config import MODEL_COMBINATIONS

def create_qa_chain(config_name="Setup_Default"):
    try:
        config = MODEL_COMBINATIONS.get(config_name)
        if not config:
            raise CustomException(f"Config {config_name} not found in config.py")

        logger.info(f"Creating QA chain for {config_name}")
        
        db = load_vector_store(config["vectorstore"], config["embeddings"])
        if db is None:
            raise CustomException(f"Vector store not found for {config_name}. Run data_loader first.")

        llm = load_llm(model_name=config["llm"])
        if llm is None:
            raise CustomException(f"LLM {config['llm']} failed to load.")

        # If reranking is enabled, retrieve more documents initially
        initial_k = 10 if config.get("rerank") else 2
        
        # Define a custom retriever that incorporates reranking
        class RerankingRetriever:
            def __init__(self, base_retriever, reranker_model, query_llm, top_n=2):
                self.base_retriever = base_retriever
                self.reranker_model = reranker_model
                self.query_llm = query_llm # LLM to generate query if needed
                self.top_n = top_n

            def get_relevant_documents(self, query):
                retrieved_docs = self.base_retriever.get_relevant_documents(query)
                if self.reranker_model and retrieved_docs:
                    # rerank_documents returns Documents, potentially with scores as metadata
                    ranked_docs = rerank_documents(query, retrieved_docs, self.reranker_model, self.top_n)
                    return ranked_docs
                return retrieved_docs

            # Adding a method to potentially get documents with scores, if rerank_documents supports it
            def get_relevant_documents_with_scores(self, query):
                retrieved_docs_with_scores = self.base_retriever.get_relevant_documents(query) # This actually returns Documents
                
                if self.reranker_model and retrieved_docs_with_scores:
                    # rerank_documents would ideally return documents with scores attached as metadata
                    # For simplicity, let's assume rerank_documents *can* return scores directly or via metadata.
                    # As currently implemented, rerank_documents returns Documents.
                    # We need to modify rerank_documents to return (Document, score) pairs if we want to use scores here.
                    # For this step, let's modify rerank_documents to add scores to doc.metadata.
                    
                    # Temporarily, to demonstrate, we can infer a "score" from the reranked list order
                    reranked_docs_pure = rerank_documents(query, retrieved_docs_with_scores, self.reranker_model, self.top_n)
                    # For a real implementation, rerank_documents should return (doc, score)
                    # Let's mock a score based on presence in the top_n
                    
                    # Modify `rerank_documents` to return (Document, score) pairs
                    pairs = [(query, doc.page_content) for doc in retrieved_docs_with_scores]
                    if not pairs:
                        return []
                    scores = self.reranker_model.predict(pairs)
                    
                    # Combine documents with their scores and sort
                    scored_documents_raw = sorted(zip(scores, retrieved_docs_with_scores), key=lambda x: x[0], reverse=True)
                    
                    # Take top_n and attach score to metadata
                    final_scored_docs = []
                    for i, (score, doc) in enumerate(scored_documents_raw[:self.top_n]):
                        doc.metadata['relevance_score'] = float(score) # Attach the score
                        final_scored_docs.append(doc)
                    return final_scored_docs
                
                # If no reranker or no docs, just return original documents (without scores)
                return retrieved_docs_with_scores
                

        base_retriever = db.as_retriever(search_kwargs={'k': initial_k})
        
        if config.get("rerank"):
            reranker = get_reranker_model(config["reranker_model"])
            # The custom_retriever now needs to be able to pass docs *with* scores
            class CustomRetrievalWithScores(RerankingRetriever):
                def __init__(self, base_retriever, reranker_model, top_n):
                    super().__init__(base_retriever, reranker_model, None, top_n) # No query_llm needed here for this simple confidence
                
                def get_relevant_documents(self, query):
                    # This method is what ConversationalRetrievalChain calls.
                    # It will use the get_relevant_documents_with_scores internally.
                    return self.get_relevant_documents_with_scores(query)

            custom_retriever_instance = CustomRetrievalWithScores(
                base_retriever=base_retriever, 
                reranker_model=reranker, 
                top_n=2 # Final number of documents to pass to LLM after reranking
            )
            retriever_to_use = custom_retriever_instance
        else:
            retriever_to_use = base_retriever

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever_to_use, # Use the custom (possibly reranking) retriever
            return_source_documents=True,
            combine_docs_chain_kwargs={'prompt': set_custom_prompt(config.get("answer_style", "concise"))} # Pass answer_style from config
        )
        
        logger.info(f"Successfully created QA chain for {config_name}")
        return qa_chain

    except Exception as e:
        error_message = CustomException(f"Failed to create QA chain for {config_name}", e)
        logger.error(str(error_message))
        return None