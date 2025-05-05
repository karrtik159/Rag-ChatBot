import itertools
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

from models import db_settings

# ──────────────── Init client & collection ────────────────
client = QdrantClient(
    url=db_settings.QDRANT_URL,
    api_key=db_settings.QDRANT_API_KEY,
    prefer_grpc=False,
)

VECTOR_SIZE = 384  # BGE‑small‑en v1.5

client = QdrantClient(
    url=db_settings.QDRANT_URL,
    api_key=db_settings.QDRANT_API_KEY,
    prefer_grpc=False,
)

VECTOR_SIZE = 384
COL = db_settings.COLLECTION_NAME

if COL not in [c.name for c in client.get_collections().collections]:
    # 1) create collection
    client.create_collection(
        collection_name=COL,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    # 2) add payload indexes we care about
    client.create_payload_index(COL, field_name="document_id", field_schema="keyword")
    client.create_payload_index(COL, field_name="page", field_schema="integer")
    client.create_payload_index(COL, field_name="is_ocr", field_schema="boolean")


# ──────────────── Helper APIs ────────────────
def upsert_points(points: List[PointStruct], batch: int = 1000) -> None:
    """
    Upsert in batches to avoid request-size limits.
    """
    for chunk in _grouper(points, batch):
        client.upsert(collection_name=db_settings.COLLECTION_NAME, points=chunk)


def search_points(
    query_vector: List[float],
    limit: int = 3,
    document_id: Optional[str] = None,
):
    payload_filter = None
    if document_id:
        payload_filter = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        )

    return client.search(
        collection_name=db_settings.COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit,
        query_filter=payload_filter,
        with_payload=True,
    )


def delete_by_document(document_id: str) -> None:
    client.delete(
        collection_name=db_settings.COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        ),
    )


# ──────────────── Utils ────────────────
def _grouper(iterable, n):
    """Yield chunks of size n."""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            break
        yield chunk
