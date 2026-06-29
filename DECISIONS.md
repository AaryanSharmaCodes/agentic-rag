# Decision Log — Agentic RAG Project

Every meaningful "X over Y because Z" choice, with the tradeoff and the interview angle.
This is the document to skim the night before an interview.

Format for each entry: **Decision → Why → Tradeoff → If it changed**

---

## Day 1 — Foundations & setup

- **`venv` over Anaconda/conda**
  Why: lighter footprint, maps cleanly to Docker later (the container builds the same env from `requirements.txt`).
  Tradeoff: conda handles non-Python system deps better; we don't need that here.
  If it changed: if we needed heavy native ML libs (CUDA, etc.), conda would be worth it.

- **FastAPI for the backend**
  Why: async, auto-generated `/docs` (Swagger), Pydantic validation built in, standard for ML-serving APIs.
  Tradeoff: Flask is simpler for trivial apps; FastAPI's async adds concepts.
  If it changed: a non-API tool could skip the web layer entirely.

- **Public GitHub repo, push over HTTPS + Personal Access Token (PAT)**
  Why: PAT in macOS Keychain avoids SSH key setup for a beginner; public repo = resume visibility.
  Tradeoff: SSH is smoother long-term; private repo if the corpus were sensitive.

- **LLM model kept swappable, default to current Claude models**
  Why: targeting ML roles; don't hard-code one provider. Default Claude for quality.
  Tradeoff: provider-agnostic code is slightly more abstraction up front.

---

## Day 2 — PDF extraction & chunking

- **`pypdf` over PyMuPDF / pdfplumber for text extraction**
  Why: pure-Python, zero system dependencies, installs clean and behaves identically inside Docker (Day 9).
  Tradeoff: no OCR (can't read scanned/image PDFs), weaker on complex multi-column layouts.
  If it changed: scanned contracts → switch to an OCR-capable tool (e.g. PyMuPDF + Tesseract).

- **Defensive `page.extract_text() or ""`**
  Why: an image-only page returns `None`; joining `None` into a string throws `TypeError`. The `or ""` converts it to empty string so extraction never crashes mid-document.

- **Chunk size measured in CHARACTERS, not tokens**
  Why: simpler, needs no tokenizer library right now. We size chars to map to sensible token counts (~4 chars ≈ 1 token).
  Tradeoff: less precise than true token counting; a chunk could slightly over/undershoot a model's token limit.
  If it changed: precision/limits demand it → move to token-based + recursive splitting (LangChain `RecursiveCharacterTextSplitter` style, splitting on paragraph/sentence separators first).

- **`chunk_size=1000`, `overlap=200` (20% overlap)**
  Why: ~250 tokens per chunk — small enough for precise retrieval, big enough to hold a complete idea; 20% overlap preserves context across boundaries.
  Tradeoff: overlap duplicates text → more chunks, more storage/embedding cost. Bigger overlap = safer context but more redundancy.

- **Step forward by `chunk_size - overlap`, with a `overlap >= chunk_size` guard**
  Why: stepping short of a full window is *what creates* the overlap. The guard prevents an infinite loop (if step were 0 the window never advances).

- **Separate `ingest.py` module (not in `main.py`)**
  Why: separation of concerns — ingestion logic is independent of the web layer and easier to test in isolation.

- **`*.pdf` git-ignored**
  Why: uploaded/test PDFs are data, not code — same reasoning as ignoring `.env`. Keeps the repo clean and avoids committing user documents.

---

## Open interview questions to answer (write your own answers here)

**Day 2:**
1. Walk me through your chunking strategy — how did you pick chunk size and overlap, and what breaks if each is too big or too small?
   - *(your answer)*
2. Your extracted text has whitespace artifacts and can't read scanned PDFs — how would you handle those, and when would it actually matter?
   - *(your answer)*
