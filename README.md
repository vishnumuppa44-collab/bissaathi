# BIS Saathi — Full-stack AI Assistant

BIS Saathi is a full-stack, multilingual web application for answering questions about Bureau of Indian Standards (BIS) services and documents. It uses an LLM plus retrieval-augmented generation (RAG): it searches the documents you place in `backend/data/` and asks the model to answer only from those sources.

## Important scope

- This starter is a working application, but it does **not** include BIS standards or copyrighted documents.
- Add public/authorized BIS PDFs, circulars, FAQs, scheme guides, and your own permitted content to `backend/data/`.
- The quality and coverage of answers depend on the documents you ingest.
- For a real deployment, obtain appropriate rights/access for standards and verify operational advice against the official BIS website.

## What you need

1. Python 3.10 or newer
2. An OpenAI-compatible API key. The default configuration is OpenAI; any provider with a compatible `/chat/completions` API can be configured in `.env`.
3. Internet access while the server calls your selected AI API.

## Quick start — Windows / macOS / Linux

Open a terminal in this project folder and run:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install packages:

```bash
pip install -r backend/requirements.txt
```

Create your environment file:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# macOS/Linux
cp .env.example .env
```

Open `.env` and add `OPENAI_API_KEY=your_key_here`.

Start the server:

```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Now open this address in a browser:

`http://127.0.0.1:8000`

## Add BIS documents

1. Put `.pdf`, `.txt`, or `.md` files inside `backend/data/`.
2. In the running app, open the **Admin** panel and click **Index documents**.
3. Or call this endpoint: `POST http://127.0.0.1:8000/api/admin/reindex`.

The app will extract text, break it into chunks, store a local index at `backend/vector_store.json`, and use matching chunks as citations in answers.

## Configuration

See `.env.example`.

- `OPENAI_API_KEY`: API key for your model provider.
- `OPENAI_BASE_URL`: default is `https://api.openai.com/v1`.
- `OPENAI_MODEL`: default is `gpt-4o-mini`; change this to a model your provider supports.
- `ADMIN_TOKEN`: optional token required by the indexing endpoint. If blank, indexing is open locally.

## API endpoints

- `GET /api/health` — server/index status
- `POST /api/chat` — send a question and receive a cited answer
- `POST /api/admin/reindex` — read and index permitted documents in `backend/data/`

## Example questions

- What is the process for BIS CRS registration?
- How can a consumer verify a BIS licence?
- Explain the documents required in the certification process.

## Production notes

Before public deployment, add authentication, rate limiting, formal source validation, a real vector database, a database for audit logs, language-specific evaluation, and official-process review. Never present the AI answer as a substitute for the latest official BIS rule or legal advice.
