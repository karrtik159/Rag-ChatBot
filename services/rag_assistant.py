import os
from typing import List, Tuple

from langchain import OpenAI, PromptTemplate
from langchain.schema import Document as LCDocument
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from langchain.embeddings import HuggingFaceBgeEmbeddings

from ..models import settings


class ChatbotManager:
    def __init__(
        self,
        llm_model: str = "Base_model",
        llm_temperature: float = 0.7,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Initializes the ChatbotManager with OpenAI LLM, Qdrant vector store, and a reranker.
        """
        # OpenAI configuration
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

        self.llm = OpenAI(
            model_name=llm_model,
            temperature=llm_temperature,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
        )

        # Prompt template for RAG
        template = (
            "Use the following context to answer the question. "
            "If you don't know, just say you don't know.\n"  
            "Context:\n{context}\n---\nQuestion: {question}\nAnswer:"
        )
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

        # Embeddings (HuggingFace BGE)
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Qdrant client and vector store
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=False,
        )
        self.db = Qdrant(
            client=self.client,
            collection_name=settings.collection_name,
            embeddings=self.embeddings,
        )
        # Default retriever: fetch top_k raw candidates
        self.retriever = self.db.as_retriever(
            search_kwargs={"k": settings.max_tokens // 100}
        )

        # Reranker for fine-tuning top results
        self.reranker = CrossEncoder(reranker_model)

    def get_response(
        self, query: str, top_k: int = 3
    ) -> Tuple[str, List[dict]]:
        """
        Executes a RAG query with initial retrieval, reranking, LLM answer, and citations.

        Args:
            query (str): The user question.
            top_k (int): Number of final context passages to include.

        Returns:
            answer (str): LLM-generated answer.
            citations (List[dict]): List of {{document_name, page}} for each passage used.
        """
        # 1) Retrieve initial candidates
        docs: List[LCDocument] = self.retriever.get_relevant_documents(query)
        if not docs:
            return "I couldn't find relevant information.", []

        # 2) Rerank using cross-encoder
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        # Order docs by descending score
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in ranked[:top_k]]

        # 3) Build context string
        context = "\n\n".join(
            f"Source: {d.metadata.get('document_name')} (page {d.metadata.get('page')})\n{d.page_content}"
            for d in top_docs
        )

        # 4) Format prompt and get answer
        prompt_str = self.prompt.format(context=context, question=query)
        answer = self.llm(prompt_str)

        # 5) Prepare citations
        citations = [
            {"document_name": d.metadata.get('document_name'), "page": d.metadata.get('page')}
            for d in top_docs
        ]

        return answer, citations
