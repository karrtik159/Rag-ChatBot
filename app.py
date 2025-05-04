# import os
# import uuid
# from typing import Optional, List, Dict

# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from langchain.chains import RetrievalQA

# from langchain.schema import Document as LCDocument
# from services.rag_assistant import ChatbotManager
# from utils.document_ingestion_pipelines import helpers

# # Initialize ChatbotManager
# chatbot_manager = ChatbotManager()

# # Create FastAPI app
# app = FastAPI(
#     title="RAG Document Embedding API",
#     description="API for embedding documents and querying them",
#     version="0.1.0"
# )


# @app.post("/api/embedding", response_model=dict)
# async def embedding(document: UploadFile = File(...)):
#     try:
#         document_id = str(uuid.uuid4())
#         file_ext = document.filename.rsplit(".", 1)[-1].lower()
#         temp_dir = "temp"
#         os.makedirs(temp_dir, exist_ok=True)
#         temp_path = os.path.join(temp_dir, f"{document_id}.{file_ext}")
#         with open(temp_path, "wb") as buf:
#             buf.write(await document.read())

#         # ingest_pdf / ingest_docx / ingest_txt all return:
#         # List[{"document_name","page","text","is_ocr",...}]
#         if file_ext == "pdf":
#             chunks = helpers.ingest_pdf(temp_path)
#         elif file_ext == "docx":
#             chunks = helpers.ingest_docx(temp_path)
#         elif file_ext == "txt":
#             chunks = helpers.ingest_txt(temp_path)
#         else:
#             raise HTTPException(400, f"Unsupported file type: {file_ext}")

#         if not chunks:
#             os.remove(temp_path)
#             return JSONResponse(400, {
#                 "status": "error",
#                 "message": "Failed to embed document.",
#                 "error_details": "Document content is empty."
#             })

#         # Build LangChain Documents with metadata
#         docs = []
#         for c in chunks:
#             text = c.get("text", "").strip()
#             if not text:
#                 continue
#             metadata = {
#                 "document_id": document_id,
#                 "document_name": c.get("document_name"),
#                 "page": c.get("page"),
#                 "chunk_index": c.get("chunk_index"),
#                 "is_ocr": c.get("is_ocr", False),
#             }
#             docs.append(LCDocument(page_content=text, metadata=metadata))

#         os.remove(temp_path)

#         if not docs:
#             return JSONResponse(400, {
#                 "status": "error",
#                 "message": "Failed to embed document.",
#                 "error_details": "No valid text chunks."
#             })

#         # Upsert all at once, preserving metadata for later citations
#         chatbot_manager.db.add_documents(docs)

#         return JSONResponse(200, {
#             "status": "success",
#             "message": "Document embedded successfully.",
#             "document_id": document_id,
#             "chunks_embedded": len(docs)
#         })

#     except Exception as e:
#         return JSONResponse(500, {
#             "status": "error",
#             "message": f"Error embedding document: {e}"
#         })


# # Conversation State Management
# conversation_store: Dict[str, List[Dict]] = {}

# class Citation(BaseModel):
#     page: Optional[int] = None
#     document_name: str

# class QueryRequest(BaseModel):
#     query: str
#     document_id: str
#     conversation_id: Optional[str] = None
#     require_citations: bool = False

# class QueryResponse(BaseModel):
#     status: str
#     response: Dict[str, str]
#     conversation_id: Optional[str] = None

# # Optional: Add a health check endpoint
# @app.get("/health")
# async def health_check():
#     return {"status": "healthy"}

# @app.post("/api/query", response_model=QueryResponse)
# async def query_documents(request: QueryRequest):
#     """Query embedded documents with optional conversation history and citations."""
#     try:
#         # Generate or validate conversation ID
#         if not request.conversation_id:
#             conversation_id = str(uuid.uuid4())
#             conversation_store[conversation_id] = []
#         else:
#             # Validate existing conversation ID
#             if request.conversation_id not in conversation_store:
#                 return JSONResponse(
#                     status_code=400,
#                     content={
#                         "status": "error",
#                         "message": "Invalid conversation ID. Please start a new session."
#                     }
#                 )
#             conversation_id = request.conversation_id
        
#         # Prepare conversation history
#         conversation_history = conversation_store[conversation_id]
        
#         # Modify ChatbotManager to support citations and conversation history
#         chatbot_manager.retriever = chatbot_manager.db.as_retriever(
#             search_kwargs={
#                 "k": 3 if request.require_citations else 1
#             }
#         )
        
#         # Modify RetrievalQA to return source documents if citations are required
#         chatbot_manager.qa = RetrievalQA.from_chain_type(
#             llm=chatbot_manager.llm,
#             chain_type="stuff",
#             retriever=chatbot_manager.retriever,
#             return_source_documents=request.require_citations,
#             chain_type_kwargs=chatbot_manager.chain_type_kwargs,
#             verbose=False
#         )
        
#         # Process query with conversation history
#         full_query = "\n".join([
#             f"Previous context: {history['query']}\nPrevious response: {history['response']}"
#             for history in conversation_history
#         ] + [request.query])
        
#         # Get response from chatbot
#         result = chatbot_manager.qa(full_query)
        
#         # Prepare response
#         response_data = {
#             "answer": result['result']
#         }
        
#         # Add citations if requested
#         if request.require_citations and 'source_documents' in result:
#             citations = []
#             for doc in result['source_documents']:
#                 # Extract page and document name from metadata
#                 page = doc.metadata.get('page')
#                 document_name = doc.metadata.get('document_name', 'Unknown Document')
#                 citations.append({
#                     "page": page,
#                     "document_name": document_name
#                 })
#             response_data["citations"] = citations
        
#         # Store conversation history
#         conversation_store[conversation_id].append({
#             "query": request.query,
#             "response": response_data.get("answer", "")
#         })
        
#         return JSONResponse(
#             status_code=200,
#             content={
#                 "status": "success",
#                 "response": response_data,
#                 "conversation_id": conversation_id
#             }
#         )
#     except Exception as e:
#         # Log the error and return a generic error response
#         return JSONResponse(
#             status_code=500,
#             content={
#                 "status": "error",
#                 "message": f"Error processing query: {str(e)}"
#             }
#         )