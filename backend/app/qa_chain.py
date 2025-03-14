import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains.qa_with_sources import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from pinecone import Pinecone
from typing import List, Dict

# Load environment variables (Optional: If you store API keys in .env)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your_pinecone_api_key")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ohanadata")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
vectorstore = Pinecone.from_documents([], embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"), index_name=INDEX_NAME)

# Load LLM for question answering
llm = OpenAI(model_name="gpt-4", temperature=0)

# ------------------------------
# DOCUMENT INDEXING FUNCTION
# ------------------------------
def process_pdf(file_path: str, file_name: str):
    """
    Loads a PDF, splits it into chunks, and indexes it into Pinecone for retrieval.
    """
    try:
        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)

        # Store in Pinecone with unique IDs
        vectorstore.add_documents(split_docs, ids=[f"{file_name}_chunk_{i}" for i in range(len(split_docs))])

        return {"message": "PDF processed and indexed successfully"}

    except Exception as e:
        return {"error": str(e)}


# ------------------------------
# SEMANTIC SEARCH FUNCTION
# ------------------------------
def search_documents(query: str) -> List[str]:
    """
    Performs a semantic search over indexed PDF documents.
    """
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        search_results = retriever.get_relevant_documents(query)

        return [doc.page_content for doc in search_results]
    
    except Exception as e:
        return {"error": str(e)}


# ------------------------------
# QUESTION ANSWERING FUNCTION
# ------------------------------
def answer_question(query: str) -> Dict[str, List[str]]:
    """
    Answers a question based on retrieved PDF content.
    """
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        qa_chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

        response = qa_chain.run(query)
        return {
            "answer": response["answer"],
            "sources": response["sources"]
        }
    
    except Exception as e:
        return {"error": str(e)}
