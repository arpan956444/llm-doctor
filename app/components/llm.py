from langchain_groq import ChatGroq
from app.config.config import GROQ_API_KEY
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
logger= get_logger(__name__)

def load_llm(model_name: str= "llama-3.1-8b-instant",groq_api_key: str = GROQ_API_KEY):
    try:
        logger.info("loadong llm from groq using llama-3.1-8b-instant model .......")

        llm = ChatGroq(
            groq_api_key = groq_api_key,
            model_name = model_name,
            temperature=0.3,
            max_tokens=500,
        )
        logger.info("LLM loaded Successfully from Groq.....")

        return llm
    except Exception as e:
        error_massage= CustomException("Failed to load LLM From Groq",e)
        logger.error(str(error_massage))

        return None