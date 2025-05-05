from __future__ import annotations

from typing import List, Tuple, Optional

from langchain_openai import ChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document as LCDocument
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient  
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import CrossEncoder

from models import db_settings


class ChatbotManager:
    """Retrieval‑augmented generation with reranker & citations."""

    # ─────────────────────────── init ────────────────────────────
    def __init__(
        self,
        llm_model: str | None = None,
        llm_temperature: float = 0.7,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        # LLM
        self.llm = ChatOpenAI(
            model_name=llm_model or "gpt-4o-mini",
            temperature=llm_temperature,
            api_key=db_settings.OPENAI_API_KEY,
            base_url=db_settings.OPENAI_API_BASE_URL,
        )

        # Prompt
        self.prompt = PromptTemplate.from_template(
            "Use the following context to answer the question. "
            "If you don't know the answer, say you don't know.\n\n"
            "Context:\n{context}\n---\nQuestion: {question}\nAnswer:"
        )

        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=db_settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Qdrant vector store
        self.client = QdrantClient(
            url=db_settings.QDRANT_URL,
            api_key=db_settings.QDRANT_API_KEY,
            prefer_grpc=False,
        )
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=db_settings.COLLECTION_NAME,
            embedding=self.embeddings,
        )

        # Default retriever (no filter)
        self.retriever = self._build_retriever()

        # Cross‑encoder reranker
        self.reranker = CrossEncoder(reranker_model)

    # ───────────────────── helper: build retriever ─────────────────────
    def _build_retriever(self, filt: Optional[Filter] = None):
        k_initial = max(1, db_settings.MAX_TOKENS // 100)
        kwargs = {"k": k_initial}
        if filt:
            kwargs["filter"] = filt
        return self.store.as_retriever(search_kwargs=kwargs)

    # ───────────────────── public API ─────────────────────
    def get_response(
        self,
        query: str,
        top_k: int = 3,
        document_id: Optional[str] = None,
        require_citations: bool = True,
    ) -> Tuple[str, List[dict]]:
        """Return answer & citations for a user query."""

        # 1️⃣  Build / reuse retriever with correct filter
        if document_id:
            filt = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )
            retriever = self._build_retriever(filt)
        else:
            retriever = self.retriever

        docs: List[LCDocument] = retriever.invoke(query)
        # print(docs)
        if not docs:
            return "I couldn't find relevant information.", []

        # 2️⃣  Rerank
        scores = self.reranker.predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [d for d, _ in ranked[:top_k]]

        # 3️⃣  Prepare LLM context
        context = "\n\n".join(
            f"Source: {d.metadata.get('document_name')} "
            f"(page {d.metadata.get('page')})\n{d.page_content}"
            for d in top_docs
        )
        prompt_str = self.prompt.format(context=context, question=query)

        # 4️⃣  Call LLM (extract .content from AIMessage)
        answer = self.llm.invoke(prompt_str).content

        # 5️⃣  Citations
        citations = (
            [
                {
                    "document_name": d.metadata.get("document_name"),
                    "page": d.metadata.get("page"),
                }
                for d in top_docs
            ]
            if require_citations
            else []
        )

        return answer, citations
