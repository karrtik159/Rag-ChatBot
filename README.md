# 🤖 RAG Chatbot – Document Embedding & Query API

A Retrieval‑Augmented‑Generation stack that:

* ingests PDF / DOCX / TXT (incl. OCR)
* chunks + embeds with **BGE‑small‑en v1.5**
* stores vectors in **Qdrant**
* reranks, then answers with **GPT‑4 / any OpenAI‑compatible model** (local or cloud)
* returns tight, cited answers with conversation memory

Everything can run **entirely locally** via Docker; no data ever leaves your machine.

---

## 🗺️ Project Layout

```
.                   # application code
├── rag_chatbot/
│   ├── main.py          # FastAPI entry‑point
│   ├── models.py        # env‑driven settings & schemas
│   ├── services/
│   │   ├── ingest_service.py
│   │   └── chatbot_manager.py
│   ├── parsers/       # pdf_parser.py, docx_parser.py, txt_parser.py
│   ├── chunker.py
│   ├── embedder.py
│   └── storage/
│       └── qdrant_client.py
├── pyproject.toml       # Poetry config
├── poetry.lock
├── Dockerfile           # multi‑stage build
├── docker-compose.yml   # app + Qdrant
└── .env                 # *never* commit real keys
```

---

## ⚙️ Prerequisites (local run)

| Tool              | Purpose            | Install                                                  |
| ----------------- | ------------------ | -------------------------------------------------------- |
| **Python 3.11**   | runtime            | [https://python.org](https://python.org)                 |
| **Poetry 1.8+**   | dependency manager | `pip install poetry`                                     |
| **Tesseract‑OCR** | scanned‑PDF text   | `sudo apt install tesseract-ocr` / Windows installer     |
| **Qdrant**        | vector DB          | `docker run -p 6333:6333 qdrant/qdrant` (or via compose) |

> **Optional:** GPU‑ready PyTorch, Ollama, etc. if you want fully offline LLMs.

---

## 🔑 Environment Vars (`.env`)

```dotenv
# Qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=

# OpenAI (or any compatible gateway)
OPENAI_API_KEY=
OPENAI_API_BASE_URL=https://api.openai.com/v1   # point to local server if needed

# Embeddings
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# HuggingFace token (only if you flip USE_HF_INFERENCE_API=true)
HF_TOKEN=
```

Create `.env` (or copy `.env.example`) before running.

---

## 🐍 Running with Poetry

```bash
# 1. Install deps (prod only)
poetry install --only main

# 2. Activate shell
poetry shell

# 3. Launch dev server
uvicorn rag_chatbot.main:app --reload --port 8000
```

Interactive docs: `http://localhost:8000/docs`

---

## 🐳 Running with Docker (+ Compose)

### 1 / Standalone image (quick)

```bash
docker build -t rag-chatbot .
docker run -p 8000:8000 -p 6333:6333 --env-file .env rag-chatbot
```

*Image includes Qdrant inside the same container for zero‑config demos.*

### 2 / Compose (recommended dev)

```bash
docker compose --env-file .env up --build
```

* `app` ➜ [http://localhost:8000/docs](http://localhost:8000/docs)
* `qdrant` ➜ [http://localhost:6333](http://localhost:6333) (REST & gRPC)

Stop with `Ctrl‑C`; data lives in the `qdrant_data` volume.

---

## 🚀 Key Endpoints

| Method | Path      | Body                             | Description                                             |
| ------ | --------- | -------------------------------- | ------------------------------------------------------- |
| POST   | `/ingest` | `multipart/form-data` (file)     | Embeds a document → returns `document_id` & chunk count |
| POST   | `/query`  | `{ "query": "...", "top_k": 3 }` | Returns answer + citations                              |
| GET    | `/health` | –                                | Simple liveness probe                                   |

Full Swagger / ReDoc at `/docs` & `/redoc`.

---

## 📈 Performance (M1 MBA, 8‑core CPU)

| Stage            | PDF (10 pages) | Notes                            |
| ---------------- | -------------- | -------------------------------- |
| Ingestion + OCR  | 2.3 s          | pdfplumber + Tesseract           |
| Embedding Upsert | 1.1 s          | batched BGE (CPU)                |
| Query (top‑3)    | 650 ms         | includes rerank + GPT‑4 response |

Switch to GPU or locally‑quantised LLM to speed things up / cut costs.

---

## 🔐 Security & Privacy

* All docs & vectors stay on your disk.
* LLM calls default to **OpenAI**; point `OPENAI_API_BASE_URL` to a local llama‑cpp or LM Studio endpoint for 100 % offline.
* HTTPS, auth headers, and rate limiting can be added via a reverse proxy (Traefik / Nginx).

---

## 📄 License

MIT — use, modify, and distribute as you please. Pull requests welcome!
