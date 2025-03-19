import os
import logging
from typing import Dict, Any

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Directory for storing temporary files
TEMP_DIR = "/tmp"


# LOCAL FILE MANAGEMENT UTILITIES
def save_temp_file(file_bytes: bytes, file_name: str) -> str:
    """
    Saves an uploaded file temporarily in the `/tmp/` directory.

    Args:
        file_bytes (bytes): The file content as bytes.
        file_name (str): The name of the file.

    Returns:
        str: The temporary file path.
    """
    temp_path = os.path.join(TEMP_DIR, file_name)
    try:
        with open(temp_path, "wb") as temp_file:
            temp_file.write(file_bytes)
        logging.info(f"Temporary file saved: {temp_path}")
        return temp_path

    except Exception as e:
        logging.error(f"Error saving temporary file: {e}")
        return ""


def delete_temp_file(file_path: str) -> bool:
    """
    Deletes a temporary file after processing.

    Args:
        file_path (str): Path of the file to delete.

    Returns:
        bool: True if deleted successfully, False otherwise.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Temporary file deleted: {file_path}")
            return True
        return False

    except Exception as e:
        logging.error(f"Error deleting file: {e}")
        return False


def ensure_temp_dir():
    """
    Ensures that the temporary directory exists.
    """
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        logging.info(f"Created temporary directory: {TEMP_DIR}")


# ERROR HANDLING UTILITIES
def handle_exception(error: Exception, message: str = "An error occurred") -> Dict[str, Any]:
    """
    Handles exceptions and logs errors.

    Args:
        error (Exception): The exception object.
        message (str, optional): Custom error message. Defaults to "An error occurred".

    Returns:
        Dict[str, Any]: A dictionary containing the error message.
    """
    logging.error(f"{message}: {error}")
    return {"error": str(error)}
