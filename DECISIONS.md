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

## Day 3 — Embeddings & vector math

- **No Anthropic embeddings API — use Voyage AI (`voyage-3`) for embeddings, Claude for generation**
  Why: Anthropic does not offer an embeddings endpoint; Voyage is their recommended embeddings partner. Real RAG mixes an embedding model + a chat model — they're different jobs.
  Tradeoff: one more API key + provider; embeddings happen over the network (latency) vs a local model.
  If it changed: privacy/offline needs → swap to a local `sentence-transformers` model (only `embeddings.py` changes).

- **Voyage (cloud) over a local embedding model**
  Why: keeps the deployed app small + low-memory for a free-tier host (Day 10); tiny dependency footprint at runtime; fits the Claude/Anthropic stack story.
  Tradeoff: local would be free + offline + no rate limits, but pulls in PyTorch (~hundreds of MB) and needs ~1–2GB RAM — tight on free-tier deploy.

- **All embedding access behind one `embed()` function, model name in one constant**
  Why: the "keep the model swappable" Day 1 decision, made real — the rest of the codebase never imports `voyageai` directly, so swapping providers touches one file.

- **Cosine similarity implemented by hand in pure Python first (`vectors.py`), no numpy**
  Why: it's the core retrieval primitive and a common interview question — owning the dot-product/magnitude math beats calling a library blind. numpy comes later for speed.
  Tradeoff: hand-rolled is slower than numpy's vectorized version; fine at this scale.

- **Cosine (angle) over Euclidean (straight-line) distance**
  Why: we care about *direction* (meaning), not vector *length*. `[1,2,2]` and `[2,4,4]` mean the same thing and score 1.0 — cosine ignores magnitude by design.

- **`input_type="query"` for questions vs `"document"` for stored chunks**
  Why: Voyage tunes the vector differently so a question and its answering passage land closer together, improving retrieval.

- **Key insight: retrieval needs correct *ranking*, not perfect scores**
  Unrelated English sentences still scored ~0.56 (not 0) because everything is "a little similar." What matters is that relevant chunks outscore irrelevant ones — the ordering, not the absolute number.

- **Environment lesson (learned the hard way): always confirm `(.venv)` is active before `pip install`**
  Installed into Anaconda `base` by accident, which also polluted `requirements.txt` with hundreds of packages. Fix: `source .venv/bin/activate`, verify with `which pip` (must point inside the project), reinstall, re-freeze. Clean `requirements.txt` is project-only (~64 lines).

- **Bug caught: `=+` is not `+=`**
  `dot =+ x*y` parses as `dot = (+x*y)` — a plain assignment keeping only the last value, not an accumulation. Ran with no error but gave wrong answers (0.666 instead of 1.0). Same gotcha exists in C++.

---

## Open interview questions to answer (write your own answers here)

**Day 3:**
1. Anthropic has no embeddings API — explain your two-model setup (Voyage for embeddings, Claude for generation) and why that split exists.
   - *(your answer)*
2. Walk me through cosine similarity: what does it measure geometrically, and why use the angle instead of straight-line distance?
   - *(your answer)*
3. Two unrelated sentences still scored 0.56. Why isn't that a problem for retrieval?
   - *(your answer)*

**Day 2:**
1. Walk me through your chunking strategy — how did you pick chunk size and overlap, and what breaks if each is too big or too small?
   - *(your answer)*
2. Your extracted text has whitespace artifacts and can't read scanned PDFs — how would you handle those, and when would it actually matter?
   - *(your answer)*
