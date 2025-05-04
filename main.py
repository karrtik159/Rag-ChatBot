# src/rag_chatbot/main.py
import os
import uuid
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.ingest_service import ingest_and_store
from services.chatbot_manager import ChatbotManager
from models import QueryRequest, QueryResponse

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="API for ingesting documents, embedding them, and querying with RAG",
    version="0.1.0"
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


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Endpoint to ingest a document and store its embeddings in Qdrant.

    Returns:
        dict: {document_id, chunks_stored, status}
    """
    # Generate a document_id and temporary file path
    document_id = str(uuid.uuid4())
    filename = file.filename
    ext = filename.rsplit('.', 1)[-1].lower()
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{document_id}.{ext}")

    try:
        # Save upload to disk
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        # Ingest, chunk, embed, and store
        chunks_count = ingest_and_store(temp_path, document_id)
        if chunks_count == 0:
            raise HTTPException(status_code=400, detail="No valid text chunks found.")

        return JSONResponse(
            status_code=200,
            content={
                "document_id": document_id,
                "chunks_stored": chunks_count,
                "status": "success"
            }
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Endpoint to query the RAG-indexed documents and return answers with citations.
    """
    try:
        answer, citations = chatbot_manager.get_response(request.query, request.top_k)
        # Build results list
        results: List[dict] = [{
            "answer": answer,
            "citations": citations
        }]
        return QueryResponse(query=request.query, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
