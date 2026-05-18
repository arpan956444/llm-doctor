from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """
Answer the following medical question in 4-5 lines maximum using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
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

        # Configuration for retrieval
        search_kwargs = {'k': 10 if config.get("rerank") else 2}

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(search_kwargs=search_kwargs),
            return_source_documents=True,
            combine_docs_chain_kwargs={'prompt': set_custom_prompt()}
        )

        # Note: True Reranking requires a custom chain. 
        # For this implementation, we return the base chain and 
        # the evaluator will handle the reranking logic if config['rerank'] is True.
        
        logger.info(f"Successfully created QA chain for {config_name}")
        return qa_chain

    except Exception as e:
        error_message = CustomException(f"Failed to create QA chain for {config_name}", e)
        logger.error(str(error_message))
        return None