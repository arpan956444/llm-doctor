import os
from dotenv import load_dotenv
load_dotenv()

from app.components.pdf_loader import load_pdf_files, create_text_chunks
from app.components.vector_store import save_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
logger = get_logger(__name__)

from app.config.config import MODEL_COMBINATIONS

import sys

def process_and_store_pdfs(target_config=None):
    try:
        logger.info("Starting vectorstore generation pipeline...")
        documents = load_pdf_files()
        if not documents:
            logger.error("No documents found to process.")
            return

        text_chunks = create_text_chunks(documents)

        unique_configs = {}
        if target_config and target_config in MODEL_COMBINATIONS:
            cfg = MODEL_COMBINATIONS[target_config]
            unique_configs[cfg["embeddings"]] = cfg["vectorstore"]
            logger.info(f"Targeting specific configuration: {target_config}")
        else:
            logger.info("Processing all unique embedding models from MODEL_COMBINATIONS...")
            for config_name, config in MODEL_COMBINATIONS.items():
                unique_configs[config["embeddings"]] = config["vectorstore"]

        for embed_model, store_path in unique_configs.items():
            logger.info(f"Processing embedding model: {embed_model}")
            save_vector_store(text_chunks, store_path, embed_model)

        logger.info("Vectorstore processing completed successfully.")
    except Exception as e:
        error_message = CustomException("Pipeline failed", e)
        logger.error(str(error_message))

if __name__ == "__main__":
    # Usage: python -m app.components.data_loader [optional_config_name]
    target = sys.argv[1] if len(sys.argv) > 1 else None
    process_and_store_pdfs(target)