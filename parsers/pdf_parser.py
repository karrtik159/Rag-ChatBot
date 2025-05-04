import pdfplumber, pytesseract
from typing import List
from models import RawEntry

def ingest_pdf(path: str) -> List[RawEntry]:
    """
    Extracts text from each PDF page; if empty, falls back to OCR.
    Returns a list of dicts: {
      document_name: str,
      page: int,
      text: str,
      is_ocr: bool
    }
    """
    results = []
    doc_name = path.split("/")[-1]
    with pdfplumber.open(path) as pdf:
        for pg_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                results.append(RawEntry(
                    document_name=doc_name,
                    page=pg_num,
                    text=text,
                    is_ocr=False
                ))
            else:
                # OCR fallback
                pil_img = page.to_image(resolution=300).original
                ocr_text = pytesseract.image_to_string(pil_img)
                results.append(RawEntry(
                    document_name=doc_name,
                    page=pg_num,
                    text=ocr_text,
                    is_ocr=True
                ))
    return results

