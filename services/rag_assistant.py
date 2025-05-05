# rag_chatbot/services/chatbot_manager.py
"""ChatbotManager – retrieval‑augmented generation with reranking and citations."""
from __future__ import annotations

import os
from typing import List, Tuple

from langchain import PromptTemplate, OpenAI
from langchain.schema import Document as LCDocument
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from langchain.embeddings import HuggingFaceBgeEmbeddings

from models import db_settings


class ChatbotManager:
    """Wraps retrieval, reranking, and LLM generation logic."""

    def __init__(
        self,
        llm_model: str | None = None,
        llm_temperature: float = 0.7,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        # ------------------------------------------------------------------
        # LLM (OpenAI‑compatible) ------------------------------------------------
        # ------------------------------------------------------------------
        self.llm = OpenAI(
            model_name=llm_model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=llm_temperature,
            openai_api_key=db_settings.OPENAI_API_KEY,
            openai_api_base=db_settings.OPENAI_API_BASE_URL,
        )

        # ------------------------------------------------------------------
        # Prompt template ---------------------------------------------------
        # ------------------------------------------------------------------
        template = (
            "Use the following context to answer the question. "
            "If you don't know the answer, just say you don't know.\n"  # noqa: E501
            "\nContext:\n{context}\n---\nQuestion: {question}\nAnswer:"
        )
        self.prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

        # ------------------------------------------------------------------
        # Embeddings --------------------------------------------------------
        # ------------------------------------------------------------------
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=db_settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": os.getenv("EMBED_DEVICE", "cpu")},
            encode_kwargs={"normalize_embeddings": True},
        )

        # ------------------------------------------------------------------
        # Qdrant vector store ----------------------------------------------
        # ------------------------------------------------------------------
        self.client = QdrantClient(
            url=db_settings.QDRANT_URL,
            api_key=db_settings.QDRANT_API_KEY,
            prefer_grpc=False,
        )
        self.db = Qdrant(
            client=self.client,
            collection_name=db_settings.COLLECTION_NAME,
            embeddings=self.embeddings,
        )
        # Retriever: quick recall (k≈max_tokens/100) then rerank
        k_initial = max(1, db_settings.MAX_TOKENS // 100)
        self.retriever = self.db.as_retriever(search_kwargs={"k": k_initial})

        # ------------------------------------------------------------------
        # Cross‑encoder reranker -------------------------------------------
        # ------------------------------------------------------------------
        self.reranker = CrossEncoder(reranker_model)

    # ----------------------------------------------------------------------
    # Public interface -----------------------------------------------------
    # ----------------------------------------------------------------------
    def get_response(
        self, query: str, top_k: int = 3
    ) -> Tuple[str, List[dict]]:
        """Return LLM answer and citations for user query."""
        # 1️⃣ Retrieve coarse candidates
        docs: List[LCDocument] = self.retriever.get_relevant_documents(query)
        if not docs:
            return "I couldn't find relevant information.", []

        # 2️⃣ Rerank with cross‑encoder scores
        pairs = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [d for d, _ in ranked[:top_k]]

        # 3️⃣ Compose context for LLM
        context = "\n\n".join(
            f"Source: {d.metadata.get('document_name')} (page {d.metadata.get('page')})\n{d.page_content}"
            for d in top_docs
        )
        prompt_str = self.prompt.format(context=context, question=query)

        # 4️⃣ Generate answer
        answer = self.llm(prompt_str)

        # 5️⃣ Build citations payload
        citations = [
            {
                "document_name": d.metadata.get("document_name"),
                "page": d.metadata.get("page"),
            }
            for d in top_docs
        ]

        return answer, citations
