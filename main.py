from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tempfile
import os
import uuid
import asyncio
import logging
from pydantic import BaseModel

from services.ingest_service import ingest_and_store
from services.rag_assistant import ChatbotManager
from models import QueryRequest, QueryResponse
from fastapi.responses import JSONResponse
from typing import Dict, List

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="API for ingesting documents, embedding them, and querying with RAG",
    version="0.1.0",
)

# CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChatbotManager
chatbot_manager = ChatbotManager()

conversation_store: Dict[str, List[Dict[str, str]]] = {}  # {conv_id: [{q,r}, …]}

logger = logging.getLogger(__name__)

ALLOWED_EXT = {"pdf", "docx", "txt"}
MAX_SIZE_MB = 100  # hard limit


class EmbeddingResponse(BaseModel):
    document_id: str
    chunks_stored: int
    status: str = "success"


@app.post("/api/embedding", response_model=EmbeddingResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    # ◇ 1. Validate extension
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, f"File type .{ext} not supported")

    # ◇ 2. Size check (StreamingUpload already in memory / spooled to disk)
    size_mb = file.size / (1024 * 1024) if hasattr(file, "size") else None
    if size_mb and size_mb > MAX_SIZE_MB:
        raise HTTPException(413, f"File larger than {MAX_SIZE_MB} MB")

    document_id = str(uuid.uuid4())

    # ◇ 3. Save to a secure temp file
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False, dir="temp") as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name

    async def cleanup(path: str):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    background_tasks.add_task(cleanup, temp_path)

    # ◇ 4. Off‑load CPU‑heavy ingest to thread pool
    try:
        chunks_stored = await asyncio.to_thread(
            ingest_and_store, temp_path, document_id
        )
    except ValueError as ve:
        # Parser raised unsupported / empty etc.
        logger.warning("Ingest error: %s", ve)
        raise HTTPException(400, str(ve))
    except Exception as exc:
        # Log full traceback, return generic 500
        logger.exception("Fatal ingest error")
        raise HTTPException(500, "Embedding failed, see server logs")

    if chunks_stored == 0:
        raise HTTPException(400, "No valid text chunks found")

    return EmbeddingResponse(document_id=document_id, chunks_stored=chunks_stored)

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the indexed documents and return an answer (with citations if requested),
    while maintaining optional conversation history.
    """
    # 1️⃣  conversation‑id handling
    if request.conversation_id:
        if request.conversation_id not in conversation_store:
            raise HTTPException(
                400, "Invalid conversation ID. Please start a new session."
            )
        conv_id = request.conversation_id
    else:
        conv_id = str(uuid.uuid4())
        conversation_store[conv_id] = []

    # 2️⃣  pass document‑id filter to ChatbotManager
    chatbot_manager.retriever.search_kwargs["filter"] = {
        "document_id": request.document_id
    }

    # 3️⃣  include prior messages in prompt (simple linear memory)
    history = conversation_store[conv_id]
    if history:
        prior_context = (
            "\n".join(f"User: {h['query']}\nAssistant: {h['answer']}" for h in history)
            + "\n\n"
        )
    else:
        prior_context = ""

    # 4️⃣  get answer from ChatbotManager (it already produces citations list)
    # answer, citations = chatbot_manager.get_response(
    #     prior_context + request.query,
    #     top_k=request.top_k,
    # )
    answer, citations = chatbot_manager.get_response(
        query=prior_context + request.query,
        top_k=request.top_k,
        document_id=request.document_id,
        require_citations=request.require_citations,
    )


    # 5️⃣  update conversation memory
    history.append({"query": request.query, "answer": answer})

    # 6️⃣  build response payload
    resp_payload = {
        "answer": answer,
    }
    if request.require_citations:
        resp_payload["citations"] = citations

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "response": resp_payload,
            "conversation_id": conv_id,
        },
    )


@app.get("/api/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
