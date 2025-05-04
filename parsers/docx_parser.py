import io
from docx import Document
from PIL import Image
import pytesseract
from models import RawEntry
from typing import List

def ingest_docx(path: str) -> List[RawEntry]:
    """
    Extracts paragraphs from DOCX and OCRs any inline images.
    Returns a list of RawEntry objects: {
      document_name: str,
      page: None,                # DOCX has no pages
      text: str,
      is_ocr: bool,
      source: "paragraph"|"image"
    }
    """
    results = []
    doc_name = path.split("/")[-1]
    doc = Document(path)

    # 1. Extract all text paragraphs
    for idx, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text:
            results.append(RawEntry(
                document_name=doc_name,
                page=None,
                text=text,
                is_ocr=False,
                source="paragraph",
                chunk_index=idx
            ))

    # 2. Extract and OCR images
    for idx, rel in enumerate(doc.part._rels):
        rel = doc.part._rels[rel]
        if "image" in rel.target_ref:
            img_bytes = rel.target_part.blob
            img = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(img)
            results.append(RawEntry(
                document_name=doc_name,
                page=None,
                text=ocr_text,
                is_ocr=True,
                source="image",
                chunk_index=idx
            ))

    return results
