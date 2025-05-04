
import os
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from starlette.config import Config

# Load environment variables from .env

current_file_dir = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(current_file_dir, ".env")
config = Config(env_path)

# ──────────────── Application Metadata ────────────────

class AppSettings(BaseSettings):
    """
    Metadata for API documentation and app configuration.
    """
    APP_NAME: str = config("APP_NAME", default="RAG Chatbot API")
    APP_DESCRIPTION: Optional[str] = config("APP_DESCRIPTION", default="API for RAG Chatbot")
    APP_URL: str = config("APP_URL", default="http://localhost:8000")
    APP_VERSION: Optional[str] = config("APP_VERSION", default="0.1.0")
    LICENSE_NAME: Optional[str] = config("LICENSE_NAME", default=None)
    CONTACT_NAME: Optional[str] = config("CONTACT_NAME", default=None)
    CONTACT_EMAIL: Optional[str] = config("CONTACT_EMAIL", default=None)

# Instantiate application settings

app_settings = AppSettings()

# ──────────────── Service Configuration ────────────────

class Settings(BaseSettings):
    """
    Core settings for vector store and LLM integrations.
    """
    QDRANT_URL: str = config("QDRANT_URL", default="http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = config("QDRANT_API_KEY", default=None)
    OPENAI_API_KEY: str = config("OPENAI_API_KEY", default="")
    OPENAI_API_BASE_URL: str = config("OPENAI_API_BASE_URL", default="https://api.openai.com/v1")
    HF_TOKEN: Optional[str] = config("HF_TOKEN", default=None)
    COLLECTION_NAME: str = config("QDRANT_COLLECTION", default="documents")
    MAX_TOKENS: int = config("MAX_TOKENS", cast=int, default=500)
    OVERLAP: int = config("OVERLAP", cast=int, default=50)
    EMBEDDING_MODEL_NAME: str = config("EMBEDDING_MODEL", default="BAAI/bge-small-en-v1.5")

# Instantiate service settings

db_settings = Settings()

# ──────────────── Request/Response Schemas ────────────────

class DocumentIngestionRequest(BaseModel):
    """
    Payload for requesting document ingestion.
    """
    file_path: str
document_id: Optional[str] = None

class QueryRequest(BaseModel):
    """
    Payload for querying the RAG system.
    """
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    """
    Response from the RAG query endpoint.
    """
    query: str
    results: List[Dict[str, Any]]

# ──────────────── Core Data Models ────────────────

class RawEntry(BaseModel):
    """
    A single extracted block of text from a document.
    """
    document_name: str
    page: Optional[int]  # None for DOCX/TXT
    text: str
    is_ocr: bool  # True if extracted via OCR
    source: str  # e.g. 'page', 'paragraph', 'image'
    chunk_index: int  # order within this entry list

class Chunk(RawEntry):
    """
    A token-based sub-chunk of a RawEntry with overlap.
    """
    sub_chunk_index: int  # index within the RawEntry chunks
