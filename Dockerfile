###############################################################################
# Stage 1 – build requirements with Poetry
###############################################################################
FROM python:3.11-slim AS requirements-stage

WORKDIR /tmp

# ----- OS libraries needed for pdfplumber & Tesseract OCR ------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Poetry + export plug‑in
RUN pip install --no-cache-dir poetry poetry-plugin-export

# Copy lock files only
COPY pyproject.toml poetry.lock* /tmp/

# Export prod requirements (no dev, no hashes → smaller image)
RUN poetry export -f requirements.txt -o requirements.txt --without-hashes


###############################################################################
# Stage 2 – runtime image
###############################################################################
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /code

# Runtime system libraries (OCR & PDF rendering)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY --from=requirements-stage /tmp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ----- Copy application package -------------------------------------
# Tree root already contains rag_chatbot/, pyproject.toml, etc.
COPY ./ /code/

# Make package discoverable
ENV PYTHONPATH="/code"

# Expose default port
EXPOSE 8000
ENV PORT=8000

# ─────────── entry‑point ───────────
# Use uvicorn for dev; switch to gunicorn for prod if desired
CMD ["uvicorn", "rag_chatbot.main:app", "--host", "0.0.0.0", "--port", "8000"]
# For production, uncomment the line below and comment uvicorn:
# CMD ["gunicorn", "rag_chatbot.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
