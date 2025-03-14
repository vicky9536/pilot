import logging
from fastapi import FastAPI, UploadFile, File
from app.config import FASTAPI_HOST, FASTAPI_PORT, DEBUG_MODE
from app.database import add_documents_to_pinecone, search_pinecone
from app.qa_chain import answer_question, process_pdf
from app.utils import save_temp_file, delete_temp_file, ensure_temp_dir, handle_exception
from app.models import PDFUploadResponse, SearchRequest, SearchResult, AnswerRequest, AnswerResponse

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI App
app = FastAPI()

# Ensure /tmp directory exists for local file storage
ensure_temp_dir()


# PDF UPLOAD & INDEXING
@app.post("/upload_pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF, extract text, and store embeddings for semantic search.
    """
    try:
        file_path = save_temp_file(await file.read(), file.filename)
        if not file_path:
            return handle_exception(Exception("File save failed"), "Failed to save temporary file")

        # Process PDF (extract text & store in Pinecone)
        documents = process_pdf(file_path, file.filename)
        add_documents_to_pinecone(documents, file.filename)

        # Delete local file after processing
        delete_temp_file(file_path)

        return {"message": "PDF processed and indexed successfully"}

    except Exception as e:
        return handle_exception(e, "Error processing PDF")


# SEMANTIC SEARCH ENDPOINT
@app.post("/search", response_model=SearchResult)
def search_text(request_data: SearchRequest):
    """
    Endpoint to perform semantic search on uploaded PDF documents.
    """
    try:
        results = search_pinecone(request_data.query)
        return {"results": results}

    except Exception as e:
        return handle_exception(e, "Error performing semantic search")


# QUESTION ANSWERING ENDPOINT
@app.post("/answer", response_model=AnswerResponse)
def answer_question_api(request_data: AnswerRequest):
    """
    Endpoint to answer questions based on uploaded PDF documents.
    """
    try:
        response = answer_question(request_data.query)
        return response

    except Exception as e:
        return handle_exception(e, "Error answering question")


# RUN FASTAPI SERVER (FOR DEBUGGING PURPOSES)
if __name__ == "__main__":
    import uvicorn
    logging.info(f"Starting FastAPI on {FASTAPI_HOST}:{FASTAPI_PORT} (Debug Mode: {DEBUG_MODE})")
    uvicorn.run(app, host=FASTAPI_HOST, port=FASTAPI_PORT, reload=DEBUG_MODE)
