import os
import faiss
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.storage import InMemoryStore
from app.config import FAISS_INDEX_PATH
from typing import List, Dict

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

sample_embedding = embedding_model.embed_query("test text")
embedding_dimension = len(sample_embedding)

if os.path.exists(FAISS_INDEX_PATH):
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings=embedding_model)
else:
    index = faiss.IndexFlatL2(embedding_dimension)
    docstore = InMemoryStore()
    index_to_docstore_id = InMemoryStore()

    vectorstore = FAISS(
        index=index,
        embedding_function=embedding_model,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

# Load LLM for question answering
llm = OpenAI(model_name="gpt-4", temperature=0)

# Create QA chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# DOCUMENT INDEXING FUNCTION
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

        # Store in FAISS
        vectorstore.add_documents(split_docs)
        vectorstore.save_local(FAISS_INDEX_PATH)

        return {"message": "PDF processed and indexed successfully"}

    except Exception as e:
        return {"error": str(e)}


# SEMANTIC SEARCH FUNCTION
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


# QUESTION ANSWERING FUNCTION
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
