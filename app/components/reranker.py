# app\components\reranker.py
from sentence_transformers import CrossEncoder
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from langchain_core.documents import Document

logger = get_logger(__name__)

_reranker_models = {} # Cache reranker models

def get_reranker_model(model_name: str = "BAAI/bge-reranker-base"):
    """
    Loads and caches a CrossEncoder reranker model.
    """
    if model_name not in _reranker_models:
        try:
            logger.info(f"Loading Reranker: {model_name}")
            reranker = CrossEncoder(model_name)
            _reranker_models[model_name] = reranker
            logger.info(f"Reranker model {model_name} loaded successfully.")
        except Exception as e:
            error_message = CustomException(f"Error loading reranker model {model_name}", e)
            logger.error(str(error_message))
            raise error_message
    return _reranker_models[model_name]

def rerank_documents(query: str, documents: list[Document], reranker, top_n: int = 2) -> list[Document]:
    """
    Reranks a list of documents based on the query using a CrossEncoder reranker.
    Attaches the relevance score to the metadata of each returned document.
    Returns the top_n documents with scores in metadata.
    """
    if not documents:
        return []

    if not reranker:
        logger.warning("No reranker provided, returning original documents.")
        return documents

    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)

    # Combine documents with their scores
    scored_documents = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

    final_ranked_docs = []
    for score, doc in scored_documents[:top_n]:
        # Create a new Document object or modify existing one to add the score to metadata
        # It's safer to create a new one to avoid modifying documents that might be referenced elsewhere
        new_metadata = doc.metadata.copy()
        new_metadata['relevance_score'] = float(score)
        final_ranked_docs.append(Document(page_content=doc.page_content, metadata=new_metadata))

    logger.info(f"Reranked {len(documents)} documents, selecting top {top_n} with scores attached.")
    return final_ranked_docs