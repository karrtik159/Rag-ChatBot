from pathlib import Path
from typing import List
from models import RawEntry

def ingest_txt(path: str) -> List[RawEntry]:
    doc_name = Path(path).name
    results: List[RawEntry] = []
    para: list[str] = []
    chunk_idx = 0

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.strip():           # non‑blank
                para.append(line.strip())
            else:                      # blank ⇒ end paragraph
                if para:
                    results.append(
                        RawEntry(
                            document_name=doc_name,
                            page=None,
                            text=" ".join(para),
                            is_ocr=False,
                            source="paragraph",
                            chunk_index=chunk_idx,
                        )
                    )
                    chunk_idx += 1
                    para = []
        # flush tail
        if para:
            results.append(
                RawEntry(
                    document_name=doc_name,
                    page=None,
                    text=" ".join(para),
                    is_ocr=False,
                    source="paragraph",
                    chunk_index=chunk_idx,
                )
            )
    return results
