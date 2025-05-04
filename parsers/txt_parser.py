from models import RawEntry
from typing import List

def ingest_txt(path: str) -> List[RawEntry]:
    """
    Reads a TXT file in whole; splits by newline blocks.
    """
    doc_name = path.split("/")[-1]
    results = []
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # Treat each paragraph (separated by blank lines) as one chunk
    chunk = []
    chunk_idx = 0
    for line in lines:
        if line:
            chunk.append(line)
        else:
            if chunk:
                results.append(RawEntry(
                    document_name=doc_name,
                    page=None,
                    text=" ".join(chunk),
                    is_ocr=False,
                    source="paragraph",
                    chunk_index=chunk_idx
                ))
                chunk_idx += 1
                chunk = []
    # final chunk
    if chunk:
        results.append(RawEntry(
            document_name=doc_name,
            page=None,
            text=" ".join(chunk),
            is_ocr=False,
            source="paragraph",
            chunk_index=chunk_idx
        ))
    return results
