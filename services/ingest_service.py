import logging
import itertools
from typing import List

from qdrant_client.http.models import PointStruct

from parsers import pdf_parser, docx_parser, txt_parser
from chunker import chunk_text
from embedder import embed_text
from storage.qdrant_client import upsert_points
from models import RawEntry, Chunk

logger = logging.getLogger(__name__)


def _dict_to_raw(entry: dict, fallback_index: int) -> RawEntry:
    """Coerce legacy parser dicts into `RawEntry` objects."""
    return RawEntry(
        document_name=entry.get("document_name", "unknown"),
        page=entry.get("page"),
        text=entry.get("text", ""),
        is_ocr=entry.get("is_ocr", False),
        source=entry.get("source", "unknown"),
        chunk_index=entry.get("chunk_index", fallback_index),
    )


def _grouper(iterable, size):
    """Yield successive lists of length ≤ size."""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def ingest_and_store(path: str, document_id: str, batch: int = 1000) -> int:
    """Parse file → chunk → embed → upsert to Qdrant.

    Returns the number of vectors stored.
    """

    # 1️⃣  Detect file type & parse -------------------------------------------------
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        raw_entries = pdf_parser.ingest_pdf(path)
    elif ext == "docx":
        raw_entries = docx_parser.ingest_docx(path)
    elif ext == "txt":
        raw_entries = txt_parser.ingest_txt(path)
    else:
        logger.error("Unsupported extension: %s", ext)
        raise ValueError(f"Unsupported document type: {ext}")

    if not raw_entries:
        logger.warning("No extractable text in %s", path)
        return 0

    # Coerce to RawEntry (parsers may still return dicts)
    entries: List[RawEntry] = [
        _dict_to_raw(e, idx) if not isinstance(e, RawEntry) else e
        for idx, e in enumerate(raw_entries)
    ]
    # after raw_entries fetched
    # entries: List[RawEntry] = [
    #     _dict_to_raw(e, idx) if not isinstance(e, RawEntry) else e
    #     for idx, e in enumerate(raw_entries)
    # ]

    # 2️⃣  Chunk -------------------------------------------------------------------
    chunks: List[Chunk] = chunk_text(entries)
    if not chunks:
        logger.warning("Chunker produced zero output for %s", path)
        return 0

    # 3️⃣  Embed + build PointStructs -----------------------------------
    points: List[PointStruct] = []

    for idx, ch in enumerate(chunks):
        vec = embed_text(ch.text)

        # convert Pydantic → dict, then rename `text` → `page_content`
        payload = ch.model_dump()
        payload["page_content"] = payload.pop("text")          # ← crucial
        payload["document_id"] = document_id

        points.append(
            PointStruct(
                id=idx,         # unsigned‑int ID
                vector=vec,
                payload=payload,
            )
        )




    # 4️⃣  Batched upsert ----------------------------------------------------------
    for chunk in _grouper(points, batch):
        upsert_points(chunk)
    logger.info("Ingested %d chunks for %s", len(points), document_id)

    return len(points)
