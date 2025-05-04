import logging
from typing import List
from parsers import pdf_parser, docx_parser, txt_parser
from chunker import chunk_text
from embedder import embed_text
from storage.qdrant_client import upsert_points
from qdrant_client.http.models import PointStruct

logger = logging.getLogger(__name__)


def ingest_and_store(path: str, document_id: str) -> int:
    """
    Ingests a document from the given path, chunks its content, embeds each chunk,
    and upserts the resulting vectors with metadata to Qdrant.

    Args:
        path (str): Filesystem path of the document to ingest.
        document_id (str): Unique identifier to tag all chunks of this document.

    Returns:
        int: Total number of vector points upserted to Qdrant.

    Raises:
        ValueError: If the file extension is unsupported or ingestion yields no entries.
    """
    # Determine which parser to use
    ext = path.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':
        entries = pdf_parser.ingest_pdf(path)
    elif ext == 'docx':
        entries = docx_parser.ingest_docx(path)
    elif ext == 'txt':
        entries = txt_parser.ingest_txt(path)
    else:
        logger.error(f"Unsupported file extension: {ext}")
        raise ValueError(f"Unsupported document type: {ext}")

    if not entries:
        logger.warning(f"No entries extracted from document: {path}")
        return 0

    # Split raw entries into overlapping chunks
    chunks = chunk_text(entries)
    if not chunks:
        logger.warning(f"Chunking produced no data for document: {path}")
        return 0

    # Prepare points for Qdrant upsert
    points: List[PointStruct] = []
    for chunk in chunks:
        vector = embed_text(chunk.text)
        payload = chunk.dict()
        payload['document_id'] = document_id
        # Construct a deterministic unique ID for each sub-chunk
        point_id = f"{document_id}-{chunk.chunk_index}-{chunk.sub_chunk_index}"
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
        )

    # Upsert in a single batch (Qdrant client can batch internally)
    upsert_points(points)
    logger.info(f"Upserted {len(points)} chunks for document {document_id}")

    return len(points)
