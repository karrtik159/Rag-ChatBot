import io
from pathlib import Path
from typing import List

from docx import Document
from PIL import Image
import pytesseract

from models import RawEntry

def ingest_docx(path: str) -> List[RawEntry]:
    results: List[RawEntry] = []
    doc_name = Path(path).name
    doc = Document(path)

    # paragraphs
    for idx, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text:
            results.append(
                RawEntry(
                    document_name=doc_name,
                    page=None,
                    text=text,
                    is_ocr=False,
                    source="paragraph",
                    chunk_index=idx,
                )
            )

    # images (unique chunk_index continues from current len)
    for rel in doc.part._rels.values():
        if "image" in rel.target_ref:
            img_bytes = rel.target_part.blob
            img = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(img)
            results.append(
                RawEntry(
                    document_name=doc_name,
                    page=None,
                    text=ocr_text,
                    is_ocr=True,
                    source="image",
                    chunk_index=len(results),   # ensure uniqueness
                )
            )

    return results
