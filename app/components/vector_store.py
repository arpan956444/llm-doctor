from langchain_community.vectorstores import FAISS
import os
from app.components.embeddings import get_embedding_model

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_vector_store(db_path, embedding_model_name):
    try:
        embedding_model = get_embedding_model(embedding_model_name)

        if os.path.exists(db_path):
            logger.info(f"Loading vectorstore from {os.path.abspath(db_path)}...")
            return FAISS.load_local(
                db_path,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            logger.warning(f"No vectorstore found at {os.path.abspath(db_path)}")
            return None
    except Exception as e:
        error_message = CustomException(f"Failed to load vectorstore at {db_path}", e)
        logger.error(str(error_message))
        return None


def save_vector_store(text_chunks, db_path, embedding_model_name):
    try:
        if not text_chunks:
            raise CustomException("No chunks provided for saving.")
        
        logger.info(f"Generating vectorstore for {embedding_model_name} at {db_path}")
        embedding_model = get_embedding_model(embedding_model_name)
        db = FAISS.from_documents(text_chunks, embedding_model)

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        db.save_local(db_path)
        logger.info(f"Vectorstore saved successfully to {db_path}")
        return db
    except Exception as e:
        error_message = CustomException(f"Failed to create vectorstore at {db_path}", e)
        logger.error(str(error_message))
        raise error_message