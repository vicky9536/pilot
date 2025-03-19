import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# LLM CONFIGURATION
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))

# SERVER CONFIGURATION
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", 8000))
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# FAISS CONFIGURATION
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")

# LOGGING CONFIGURATION
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# PRINT CONFIGURATION SUMMARY
def print_config():
    """
    Prints the current configuration for debugging purposes.
    """
    config_summary = f"""
    **FastAPI Configuration**
    - Host: {FASTAPI_HOST}
    - Port: {FASTAPI_PORT}
    - Debug Mode: {DEBUG_MODE}

    **LLM Configuration**
    - Provider: {LLM_PROVIDER}
    - Model: {LLM_MODEL_NAME}
    - Temperature: {LLM_TEMPERATURE}

    **FAISS Configuration**
    - FAISS Index Path: {FAISS_INDEX_PATH}

    **Logging Configuration**
    - Log Level: {LOG_LEVEL}
    """
    print(config_summary)

if DEBUG_MODE:
    print_config()
