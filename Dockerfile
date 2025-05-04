# ────────── requirements stage ──────────
FROM python:3.11-slim AS requirements-stage

WORKDIR /tmp

# System libs needed by pdfplumber / Tesseract OCR
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      poppler-utils \
      build-essential \
 && rm -rf /var/lib/apt/lists/*

# Poetry + export plugin
RUN pip install --no-cache-dir poetry poetry-plugin-export

# Copy lock‑files from repo
COPY ./pyproject.toml ./poetry.lock* /tmp/

# Export production‑only requirements
RUN poetry export -f requirements.txt --without-hashes -o requirements.txt


# ────────── final image ──────────
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /code

# Install system libs again (runtime)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY --from=requirements-stage /tmp/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy source
COPY ./ /code
ENV PYTHONPATH=/code

# Default port
ENV PORT=8000

# ────────── launch ──────────
# Swap to gunicorn line for prod as needed
CMD ["uvicorn", "rag_chatbot.main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["gunicorn", "rag_chatbot.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
