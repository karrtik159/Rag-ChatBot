import pdfplumber, pytesseract
from typing import List
from models import RawEntry
import os


def ingest_pdf(path: str) -> List[dict]:
    results = []
    name = os.path.basename(path)
    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            is_ocr = False
            if not text.strip():
                text = pytesseract.image_to_string(page.to_image(300).original)
                is_ocr = True
            results.append(
                {
                    "document_name": name,
                    "page": idx,
                    "text": text,
                    "is_ocr": is_ocr,
                    "source": "page",  # <- add
                    "chunk_index": idx - 1,  # <- add
                }
            )
    return results
