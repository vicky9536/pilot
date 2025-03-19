import os
import logging
import faiss
import pickle
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.storage import InMemoryStore
from langchain.docstore import InMemoryDocstore
from app.config import FAISS_INDEX_PATH

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FAISS_METADATA_PATH = FAISS_INDEX_PATH + "_metadata.pkl"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

sample_embedding = embedding_model.embed_query("test text")
embedding_dimension = len(sample_embedding)

# Try to load an existing FAISS index
if os.path.exists(FAISS_INDEX_PATH):
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings=embedding_model)
    logging.info("Loaded existing FAISS index.")
else:
    index = faiss.IndexFlatL2(embedding_dimension)
    docstore = InMemoryDocstore()
    index_to_docstore_id = InMemoryStore()
    
    vectorstore = FAISS(
        index=index,
        embedding_function=embedding_model,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    logging.info("Initialized new FAISS index.")

# Load document metadata (tracking doc IDs)
if os.path.exists(FAISS_METADATA_PATH):
    with open(FAISS_METADATA_PATH, "rb") as f:
        doc_metadata = pickle.load(f)
else:
    doc_metadata = {}

# VECTOR STORE UTILITIES
def add_documents_to_faiss(docs, file_name):
    """
    Adds processed text documents to FAISS for vector storage.
    """
    try:
        vectorstore.add_documents(docs)
        vectorstore.save_local(FAISS_INDEX_PATH)
        logging.info(f"Successfully added {len(docs)} documents from {file_name} to FAISS.")
        return "Documents added successfully to FAISS."

    except Exception as e:
        logging.error(f"Error adding documents to FAISS: {e}")
        return f"Error: {e}"


def search_faiss(query, top_k=5):
    """
    Performs a semantic search on the FAISS vector database.
    """
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        search_results = retriever.get_relevant_documents(query)
        return [doc.page_content for doc in search_results]

    except Exception as e:
        logging.error(f"Error performing search in FAISS: {e}")
        return [f"Error: {e}"]


def delete_document_from_faiss(file_name):
    """
    Deletes all vectors associated with a given file from FAISS.
    Since FAISS does not support direct deletion, we must rebuild the index.
    """
    try:
        if file_name not in doc_metadata:
            logging.warning(f"No records found for {file_name}. Nothing to delete.")
            return f"No documents found for {file_name}."

        # Remove document IDs from metadata
        ids_to_remove = doc_metadata.pop(file_name, [])

        # Rebuild FAISS index excluding the deleted documents
        all_docs = []
        for remaining_file, doc_ids in doc_metadata.items():
            all_docs.extend([(vectorstore.index.reconstruct(i), i) for i in doc_ids])

        # Create a new FAISS index
        new_index = faiss.IndexFlatL2(embedding_model.model.embed_dim)
        vectorstore.index = new_index

        # Reinsert remaining documents
        for vector, doc_id in all_docs:
            new_index.add_with_ids(vector.reshape(1, -1), [doc_id])

        # Save updated FAISS index
        vectorstore.save_local(FAISS_INDEX_PATH)

        # Update metadata file
        with open(FAISS_METADATA_PATH, "wb") as f:
            pickle.dump(doc_metadata, f)

        logging.info(f"Deleted {len(ids_to_remove)} documents from {file_name} in FAISS.")
        return f"Deleted {len(ids_to_remove)} documents from {file_name}."

    except Exception as e:
        logging.error(f"Error deleting documents from FAISS: {e}")
        return f"Error: {e}"