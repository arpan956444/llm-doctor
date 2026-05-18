from langchain_huggingface import HuggingFaceEmbeddings
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Huggingface embedding model: {model_name} on {device}")
        
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': False}

        model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        logger.info(f"Embedding model {model_name} loaded successfully.")
        return model

    except Exception as e:
        error_message = CustomException(f"Error loading embedding model {model_name}", e)
        logger.error(str(error_message))
        raise error_message