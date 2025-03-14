import os
import logging
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API keys from .env file
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Ensure environment variables are set
if not PINECONE_API_KEY or not INDEX_NAME:
    raise ValueError("ERROR: Missing Pinecone API key or index name. Check your .env file.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize HuggingFace Embeddings Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create Pinecone Vector Store
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding_function=embedding_model)

# VECTOR STORE UTILITIES
def add_documents_to_pinecone(docs, file_name):
    """
    Adds processed text documents to Pinecone for vector storage.
    """
    if vectorstore is None:
        return "Error: Pinecone vectorstore is not initialized."

    try:
        ids = [f"{file_name}_chunk_{i}" for i in range(len(docs))]
        vectorstore.add_documents(docs, ids=ids)
        logging.info(f"Successfully added {len(docs)} documents from {file_name} to Pinecone.")
        return "Documents added successfully to Pinecone."

    except Exception as e:
        logging.error(f"Error adding documents to Pinecone: {e}")
        return f"Error: {e}"


def search_pinecone(query, top_k=5):
    """
    Performs a semantic search on the Pinecone vector database.
    """
    if vectorstore is None:
        return ["Error: Pinecone vectorstore is not initialized."]

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        search_results = retriever.get_relevant_documents(query)
        return [doc.page_content for doc in search_results]

    except Exception as e:
        logging.error(f"Error performing search in Pinecone: {e}")
        return [f"Error: {e}"]


def delete_pinecone_index():
    """
    Deletes the Pinecone index (for resetting storage).
    """
    try:
        pc.delete_index(INDEX_NAME)
        pc.create_index(INDEX_NAME, dimension=embedding_model.client.embed_query("test").shape[0])
        logging.info("Pinecone index deleted and recreated successfully.")
        return "Pinecone index deleted and recreated successfully."
    except Exception as e:
        logging.error(f"Error deleting Pinecone index: {e}")
        return f"Error: {e}"
