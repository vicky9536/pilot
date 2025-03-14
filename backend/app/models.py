from pydantic import BaseModel
from typing import List, Optional


# PDF Upload Request Model
class PDFUploadResponse(BaseModel):
    message: str


# Semantic Search Models
class SearchRequest(BaseModel):
    """
    Request model for performing semantic search on uploaded PDF documents.
    """
    query: str


class SearchResult(BaseModel):
    """
    Response model for search results.
    """
    results: List[str]


# Question Answering Models
class AnswerRequest(BaseModel):
    """
    Request model for asking questions based on uploaded PDF documents.
    """
    query: str


class AnswerResponse(BaseModel):
    """
    Response model for answering questions, including the sources of the answer.
    """
    answer: str
    sources: Optional[List[str]] = []


# Firestore Models (For Internal Processing)
class FirestoreRequest(BaseModel):
    """
    Request model for Firestore listener processing search and answer requests.
    """
    id: str
    type: str
    query: str


class FirestoreResponse(BaseModel):
    """
    Response model for storing processed Firestore responses.
    """
    request_id: str
    response_data: dict
