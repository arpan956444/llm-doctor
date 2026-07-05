import os

import os

# Consider more advanced loaders for richer parsing:
# from langchain_community.document_loaders import UnstructuredPDFLoader # Requires 'unstructured' library and its dependencies
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader # Keep as fallback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Ensure Document is imported for consistency with reranker.py

from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

def load_pdf_files():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException(f"Data path '{DATA_PATH}' does not exist.")
        
        logger.info(f"Loading files from {DATA_PATH}")

        documents = []
        for file_name in os.listdir(DATA_PATH):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(DATA_PATH, file_name)
                logger.info(f"Processing PDF: {file_name}")
                try:
                    # Option 1: Use PyPDFLoader for basic text extraction
                    loader = PyPDFLoader(file_path)
                    docs_from_file = loader.load()
                    documents.extend(docs_from_file)
                    logger.info(f"Loaded {len(docs_from_file)} pages from {file_name} using PyPDFLoader.")
                    
                    # --- RESEARCH CONTRIBUTION OPPORTUNITY ---
                    # Option 2 (Advanced): Integrate UnstructuredPDFLoader for richer parsing
                    # This would require installing `unstructured` and its dependencies (e.g., `poppler-utils`, `tesseract-ocr`)
                    # from unstructured.partition.pdf import partition_pdf
                    # elements = partition_pdf(filename=file_path, strategy="hi_res")
                    # for element in elements:
                    #     # Convert elements to Langchain Document format if needed
                    #     # This allows preserving semantic structure (tables, figures, etc.)
                    #     # which can be stored in metadata or processed differently.
                    #     documents.append(Document(page_content=str(element), metadata={"source": file_name, "type": str(type(element))}))
                    # logger.info(f"Loaded {len(elements)} elements from {file_name} using UnstructuredPDFLoader.")

                except Exception as file_e:
                    logger.error(f"Failed to load specific PDF {file_name}: {file_e}")

        if not documents:
            logger.warning("No PDFs were successfully loaded or processed.")
        else:
            logger.info(f"Successfully fetched a total of {len(documents)} document pages/elements.")

        return documents
    
    except Exception as e:
        error_message = CustomException("Failed to load PDF files from directory", e)
        logger.error(str(error_message))
        return []

def create_text_chunks(documents):
    try:
        if not documents:
            raise CustomException("No documents were found")
        
        logger.info(f"Splitting {len(documents)} documents into chunks")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        text_chunks = text_splitter.split_documents(documents)

        logger.info(f"Generated {len(text_chunks)} text chunks")

        return text_chunks
    
    except Exception as e:
        error_message = CustomException("Failed to generate chunks", e)
        logger.error(str(error_message))
        return []