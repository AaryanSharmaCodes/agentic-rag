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

## Day 4 — Retrieval / semantic search

- **A `VectorStore` class holding two parallel lists (`chunks` + `vectors`), position `i` linking them**
  Why: retrieval needs the chunk text AND its vector kept together and searchable. Parallel lists indexed by position is the simplest structure that does the job and stays readable for a beginner.
  Tradeoff: parallel lists can drift out of sync if edited carelessly; a list of objects each holding both would be safer. Fine at this size.
  If it changed: swap the internals for a real index (FAISS / Chroma / pgvector) without changing the `add`/`search` interface.

- **In-memory, hand-rolled search (loop + cosine over every chunk) — NOT a vector DB yet**
  Why: it makes the core idea unmissable — retrieval is literally "score every chunk against the question, sort, take the top k," powered by the cosine function we wrote by hand on Day 3. Zero new dependencies.
  Tradeoff: O(n) per query — compares against every chunk. Unusable at millions of chunks.
  If it changed: production scale → an Approximate Nearest Neighbour index (FAISS/HNSW) or a hosted vector DB. Same `search()` signature, different guts.

- **`search()` returns top-`k` (default 3), not top-1 and not everything**
  Why: top-1 is fragile (the single best chunk may hold only part of the answer); returning everything defeats retrieval (we'd dump the whole doc back into the model). A small handful balances recall vs noise.
  Tradeoff: bigger k = more context/recall but more tokens + more chance of irrelevant chunks diluting the answer.

- **Each chunk stored with an `id` and a `source` label**
  Why: Day 5 citations need to point back at *which* chunk from *which* document an answer came from. Storing that at index time makes grounding + citations possible later.

- **Sort with `key=lambda pair: pair[0]` (score only), not a bare sort**
  Why: a plain `sort()` on `(score, chunk_dict)` tuples would, on a score tie, try to compare the two dicts with `<` and crash. Sorting on the score alone never touches the dict.

- **Observation: real-document scores clustered ~0.34–0.37 (low, and close together)**
  Confirms the Day 3 insight on real data: absolute similarity numbers are low and bunched because all English text is "a little similar." Retrieval quality is about *ordering* (relevant chunks outranking irrelevant ones), not hitting a high score.

---

## Day 5 — Answer generation with citations

- **Switched generation provider Claude → Groq (Llama 3.3 70B) mid-project**
  Why: could not pay for the Anthropic API; Groq is free with no credit card and runs a strong open-weight model. The Day 1 "keep the model swappable" decision made this a ONE-FILE change (`generate.py` only) — retrieval/embedding/ingestion untouched. This is the project's headline "provider-independence" story.
  Tradeoff: Llama 70B is slightly less capable than Opus 4.8 on hard reasoning; also a second vendor/rate-limit to reason about. Embeddings stayed on Voyage.
  If it changed: pay for Claude later → swap `Groq()` back to `anthropic.Anthropic()` and adjust the message shape (~10 lines).

- **Grounding via a strict system prompt: "use ONLY the numbered context, cite it, say 'I don't know' otherwise"**
  Why: this instruction is what turns "a chatbot" into RAG. Constraining the model to the retrieved passages is THE mechanism that reduces hallucination; the "I don't know" clause stops it inventing answers when retrieval finds nothing relevant.
  Tradeoff: an over-strict prompt can make the model refuse to answer things that ARE implied by the context. Prompt wording is a real lever.

- **Numbered context `[1] … [2] …` with a display number separate from the chunk's internal `id`**
  Why: the model cites human-friendly "passage 1/2", while we keep our own stable `id`/`source` mapping so a UI (Day 8) can turn `[1]` into a real citation back to the document. Citations = traceability = trust.

- **OpenAI-style chat API (system prompt as the first `messages` entry) vs Anthropic (separate `system=` arg)**
  Why worth noting: same concept, different plumbing. Groq/OpenAI put system as `messages[0]` with role "system" and return text at `choices[0].message.content`; Anthropic takes `system=` top-level and returns a list of content blocks. Knowing both shapes is useful.

- **`temperature=0` for generation**
  Why: RAG answers should be faithful to the source, not creative. Temperature 0 makes the model as deterministic/grounded as possible — we don't want stylistic variance, we want it to stick to the passages.
  Tradeoff: 0 is repetitive/less fluent; fine (desirable) here, wrong for creative writing.

- **`max_tokens=1024`**
  Why: a grounded answer over a few passages is short; 1024 is ample headroom. Bump only if an answer ever truncates mid-sentence.

---

## Open interview questions to answer (write your own answers here)

**Day 5:**
1. Your system prompt has three rules (use-only-context, cite, say-I-don't-know). Which one actually reduces hallucination, and why that one?
   - *(your answer)*
2. How do citations work end to end — how does the model "know" a passage is number [1], and how would you turn that back into a clickable source?
   - *(your answer)*
3. You set temperature=0. What does temperature do, and why is 0 the right call for RAG specifically?
   - *(your answer)*
4. You swapped Claude for Llama/Groq in one file. What made that possible, and what did NOT have to change?
   - *(your answer)*

**Day 4:**
1. Walk me through what happens, step by step, when a user asks a question — from the raw question to the top-k chunks coming back.
   - *(your answer)*
2. Your search compares the query against every stored chunk. Why is that a problem at scale, and what would you replace it with?
   - *(your answer)*
3. Why embed the question with `input_type="query"` but the chunks with `"document"`? What would break if you used the same type for both?
   - *(your answer)*

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
