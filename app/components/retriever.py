from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever # NEW IMPORT
from langchain_core.documents import Document # NEW IMPORT
from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.components.reranker import get_reranker_model, rerank_documents # NEW IMPORT
from typing import Any # NEW IMPORT
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
                
        base_retriever = db.as_retriever(search_kwargs={'k': initial_k})
        
        if config.get("rerank"):
            reranker = get_reranker_model(config["reranker_model"])
            retriever_to_use = RerankingRetriever(
                base_retriever=base_retriever, 
                reranker_model=reranker, 
                top_n=2 # Final number of documents to pass to LLM after reranking
            )
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

# Define a custom retriever that incorporates reranking
class RerankingRetriever(BaseRetriever):
    """
    A custom retriever that first retrieves documents from a base retriever
    and then reranks them using a CrossEncoder model.
    """
    # These are Pydantic fields and must be passed during initialization
    base_retriever: BaseRetriever
    reranker_model: Any # CrossEncoder model instance
    top_n: int = 2

    def __init__(self, base_retriever: BaseRetriever, reranker_model: Any, top_n: int = 2, **kwargs: Any):
        # Pass all Pydantic fields to the super constructor
        super().__init__(base_retriever=base_retriever, reranker_model=reranker_model, top_n=top_n, **kwargs)

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        """
        Retrieve and rerank documents.
        """
        # Step 1: Initial retrieval from the base retriever
        # We call the base retriever's public method here.
        retrieved_docs = self.base_retriever.get_relevant_documents(query)
        
        if self.reranker_model and retrieved_docs:
            # Step 2: Rerank the retrieved documents
            # rerank_documents function already adds 'relevance_score' to metadata
            ranked_docs = rerank_documents(query, retrieved_docs, self.reranker_model, self.top_n)
            return ranked_docs
        
        # If no reranker or no docs, just return the initially retrieved documents (or an empty list)
        return retrieved_docs

    async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        """
        Async version of retrieve and rerank documents.
        """
        # For simplicity, we can just call the sync version if async isn't strictly needed for the reranker
        return self._get_relevant_documents(query, **kwargs)