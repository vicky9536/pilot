import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PINECONE CONFIGURATION
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Ensure essential Pinecone variables are set
if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise ValueError("ERROR: Missing Pinecone API key or index name. Check your .env file.")

# LLM CONFIGURATION
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))

# SERVER CONFIGURATION
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", 8000))
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# LOGGING CONFIGURATION
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# STORAGE CONFIGURATION (Local)
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp")

# Ensure the temporary directory exists
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

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

    **Pinecone Configuration**
    - API Key: {'✔️ Set' if PINECONE_API_KEY else 'Not Set'}
    - Index Name: {PINECONE_INDEX_NAME}

    **LLM Configuration**
    - Provider: {LLM_PROVIDER}
    - Model: {LLM_MODEL_NAME}
    - Temperature: {LLM_TEMPERATURE}

    **Logging Configuration**
    - Log Level: {LOG_LEVEL}

    **Storage Configuration**
    - Temporary Directory: {TEMP_DIR}
    """
    print(config_summary)

if DEBUG_MODE:
    print_config()
