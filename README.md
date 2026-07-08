# Enterprise AI Documentation Assistant

An AI-powered application that lets developers search and understand technical documentation using natural language, powered by a local Retrieval-Augmented Generation (RAG) pipeline — no cloud LLM required.

Upload API docs, user manuals, technical guides, or Markdown files, then ask questions like *"How do I authenticate a user?"* and get accurate, source-grounded answers in seconds.

---

## Project Overview

- **Problem:** Developers waste time manually searching through hundreds of pages of documentation.
- **Solution:** Upload documentation once, index it, then query it conversationally. The assistant retrieves the most relevant chunks and generates an answer using a **locally hosted LLM (Ollama + Llama 3.2)** — nothing leaves your machine.
- **Core technique:** Retrieval-Augmented Generation (RAG) — the LLM is instructed to answer *only* from retrieved context, and to say so honestly when the answer isn't in the docs.

---

## Architecture

```
┌─────────────────┐        HTTP        ┌──────────────────┐
│   Streamlit UI   │ ─────────────────▶ │   FastAPI Backend │
│  (frontend/app.py)│ ◀───────────────── │   (backend/api.py) │
└─────────────────┘                     └─────────┬─────────┘
                                                    │
                     ┌──────────────────────────────┼───────────────────────────┐
                     ▼                              ▼                           ▼
             ┌───────────────┐            ┌──────────────────┐        ┌────────────────┐
             │  ingest.py     │            │   embeddings.py   │        │    rag.py       │
             │  load/split docs│            │  sentence-transf. │        │  retrieve + LLM │
             └───────┬───────┘            └─────────┬────────┘        └───────┬────────┘
                     │                               │                        │
                     ▼                               ▼                        ▼
             ┌───────────────┐            ┌──────────────────┐        ┌────────────────┐
             │   docs/        │            │  FAISS vectorstore│        │  Ollama (local) │
             │  (raw uploads) │            │  (vectorstore/)   │        │  llama3.2        │
             └───────────────┘            └──────────────────┘        └────────────────┘
```

**Flow:**
1. User uploads PDF / Markdown / TXT documentation via the Streamlit sidebar.
2. Backend saves files to `docs/`.
3. "Index Documents" triggers `ingest.py`: load → split into 800-char chunks (150 overlap) → embed with `all-MiniLM-L6-v2` → store in a FAISS index under `vectorstore/`.
4. User asks a question → backend embeds the query, retrieves the top-3 relevant chunks, builds a grounded prompt, and sends it to Ollama (`llama3.2`).
5. The answer and its source chunks are returned and displayed in the UI.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Backend | FastAPI |
| Frontend | Streamlit |
| LLM | Ollama — Llama 3.2 (3B), run locally |
| AI Framework | LangChain |
| Vector Database | FAISS |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Document Processing | PyPDF, Markdown/Text loaders |
| Config | python-dotenv |

---

## Folder Structure

```
Enterprise-AI-Documentation-Assistant/
├── backend/
│   ├── api.py          # FastAPI app & endpoints
│   ├── rag.py           # Retrieval + LLM generation pipeline
│   ├── ingest.py         # Document loading, chunking, FAISS indexing
│   ├── embeddings.py      # sentence-transformers embedding model
│   ├── prompts.py         # Prompt template construction
│   ├── config.py          # Centralized settings
│   └── utils.py           # Logging, custom exceptions, helpers
├── frontend/
│   └── app.py            # Streamlit UI
├── docs/                  # Uploaded documentation (runtime)
├── vectorstore/           # FAISS index (runtime)
├── screenshots/           # UI screenshots for this README
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── start.sh
├── .env.example
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Enterprise-AI-Documentation-Assistant.git
cd Enterprise-AI-Documentation-Assistant
```

### 2. Create a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env if you need to change ports, model names, or chunking settings
```

---

## Running Ollama

Ollama runs the LLM locally, so no API keys or cloud calls are needed.

1. Install Ollama: https://ollama.com/download
2. Pull the model:
   ```bash
   ollama pull llama3.2
   ```
3. Start the Ollama server (usually starts automatically after install; otherwise):
   ```bash
   ollama serve
   ```
4. Verify it's running:
   ```bash
   curl http://localhost:11434
   ```

---

## Running the Backend

```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

The API docs (Swagger UI) will be available at `http://localhost:8000/docs`.

## Running the Frontend

In a separate terminal (with the virtual environment activated):

```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser.

---

## Using the Application

1. **Upload Documentation** — use the sidebar to upload PDF, Markdown, or TXT files.
2. **Index Documents** — click "Index Documents" to build the FAISS vector index.
3. **Ask a question** — type a question in the main text box and click "Generate Answer".
4. **Review the answer and sources** — the generated answer appears first, followed by the retrieved source chunks (Chunk 1, Chunk 2, Chunk 3) so you can verify where the answer came from.
5. **Clear Database** — use the sidebar button to wipe uploaded documents and the index and start fresh.

### Sample Questions

- How do I authenticate?
- How do I upload a file?
- Which endpoint creates users?
- How do I reset password?
- Explain JWT authentication.
- Which API returns user profile?
- How do I generate API keys?
- What are required request parameters?
- Which HTTP method is used?
- Show example request.

### Screenshots

Add screenshots of the running application here:

```
screenshots/
├── upload.png
├── query.png
└── answer.png
```

---

## API Reference

### `POST /upload`
Upload one or more documentation files (multipart form, field name `files`).

### `POST /ingest`
Runs the ingestion pipeline (load → chunk → embed → build FAISS index) over all uploaded documents.

### `POST /query`
```json
{ "question": "How do I login?" }
```
Response:
```json
{
  "answer": "....",
  "sources": ["...", "...", "..."]
}
```

### `DELETE /clear`
Clears uploaded documents and the FAISS index.

### `GET /health`
Basic health check.

Full interactive API documentation is available at `/docs` (Swagger) and `/redoc` once the backend is running.

---

## Error Handling

The application surfaces friendly, actionable messages instead of raw stack traces:

| Situation | Message |
|---|---|
| No documents uploaded | "Please upload documentation first." |
| No relevant chunks found | "No relevant information found." |
| Ollama offline / unreachable | "Unable to connect to local LLM." |

---

## Docker

Build and run the whole application (backend + frontend) in a single container:

```bash
docker build -t doc-assistant .
docker run -p 8000:8000 -p 8501:8501 --env-file .env doc-assistant
```

Or, to also spin up Ollama alongside the app in one command:

```bash
docker compose up --build
```

This starts:
- `ollama` — the local LLM runtime (port `11434`)
- `app` — the FastAPI backend (port `8000`) and Streamlit frontend (port `8501`)

Remember to pull the model into the Ollama container once it's running:

```bash
docker exec -it doc-assistant-ollama ollama pull llama3.2
```

---

## Performance

Average response time for small documentation sets is under 3 seconds, dominated by embedding the query, FAISS similarity search, and local LLM generation.

---

## Development Notes

- **Code style:** PEP8-compliant, fully type-hinted, with docstrings and inline comments explaining intent.
- **Logging:** structured logging throughout the ingestion and RAG pipelines via `backend/utils.py`.
- **Exception handling:** custom exception hierarchy (`NoDocumentsError`, `NoRelevantChunksError`, `OllamaConnectionError`, etc.) mapped to appropriate HTTP status codes in the API layer.
- **Extensibility:** the document loader dispatch in `ingest.py` is structured so DOCX and HTML support can be added by registering new loaders, per the "future-ready" requirement.

### Suggested Git History

```
Initial FastAPI setup
Added document ingestion
Integrated FAISS
Added RAG pipeline
Implemented Streamlit UI
Integrated Ollama
Dockerized application
```

---

## What This Project Demonstrates

Python · FastAPI · REST APIs · LangChain · Retrieval-Augmented Generation (RAG) · FAISS vector search · LLM integration via Ollama · Prompt engineering · Semantic search · Streamlit · Backend development · AI application development · Software engineering practices · Git/GitHub · Docker

---

## License

MIT — feel free to use this project as a portfolio piece or starting point for your own documentation assistant.
