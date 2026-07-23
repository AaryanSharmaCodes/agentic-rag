"""
generate.py — turn retrieved chunks into a grounded, cited answer.

This is the "G" in RAG (Generation). Day 4 gave us retrieval: ask a question,
get the most relevant chunks. But chunks aren't an answer. Here we hand those
chunks to an LLM with a strict instruction:

    "Answer using ONLY these passages, cite which ones you used, and if the
     answer isn't here, say you don't know."

That instruction is what makes this RAG and not just 'ask a chatbot'. The
model is GROUNDED in our documents, and every answer is traceable to a source.

PROVIDER NOTE (Day 5 decision): we use Groq (running Meta's Llama 3.3 70B)
instead of Claude, purely because Groq is free with no card required. Thanks
to the Day 1 "keep the model swappable" decision, this swap only touched THIS
file — the rest of the pipeline never knew. Groq speaks the OpenAI-style chat
API, which is the shape most LLM code uses in industry.
"""

import os

from groq import Groq
from dotenv import load_dotenv

# Pull GROQ_API_KEY out of .env into the environment. The Groq client picks it
# up automatically — the key never appears in our code. Same pattern as the
# Voyage key in embeddings.py.
load_dotenv()

# One client, created once at import time (cheaper than rebuilding per call).
_client = Groq()

# The model name lives in ONE place — the "keep it swappable" habit again.
# llama-3.3-70b-versatile is a strong open-weight model, free on Groq.
MODEL = "llama-3.3-70b-versatile"

# The SYSTEM prompt sets the rules of the game — the model's job description,
# separate from the user's actual question. Every rule maps to a real RAG
# concern: grounding, citations, and honest "I don't know" over guessing.
SYSTEM_PROMPT = (
    "You are a document assistant. Answer the user's question using ONLY the "
    "numbered context passages provided. Cite the passage number(s) you used "
    "in square brackets, like [1] or [2]. If the answer is not contained in the "
    "context, say you don't know — do NOT use outside knowledge or guess."
)


def _build_context(results: list[tuple[float, dict]]) -> tuple[str, list[dict]]:
    """
    Turn search() results into (a) a numbered text block for the prompt and
    (b) a matching list of sources we can show the user.

    The DISPLAY number (1, 2, 3...) is what the model cites; it's separate from
    the chunk's internal `id`, so the model only reasons about "passage 1", etc.
    """
    lines = []
    sources = []
    # enumerate(..., start=1) gives (1, first_item), (2, second_item)...
    for number, (score, chunk) in enumerate(results, start=1):
        lines.append(f"[{number}] {chunk['text']}")
        sources.append({
            "number": number,
            "id": chunk["id"],
            "source": chunk["source"],
            "score": score,
        })
    # "\n\n".join glues the passages together with a blank line between each.
    return "\n\n".join(lines), sources


def answer(question: str, store, k: int = 3) -> dict:
    """
    Retrieve, then generate a grounded answer.

    Args:
        question: the user's question.
        store:    a VectorStore that already has documents added.
        k:        how many chunks to retrieve as context.

    Returns:
        {"answer": <text>, "sources": [...]} — the answer plus the passages it
        was allowed to use, so a UI (Day 8) can render citations.
    """
    # Step 1 — RETRIEVAL (Day 4). Find the chunks most likely to hold the answer.
    results = store.search(question, k=k)

    # Step 2 — assemble the numbered context and the source list.
    context, sources = _build_context(results)

    # Step 3 — GENERATION. Send context + question to the LLM.
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    # KEY CONTRAST with the Anthropic style: OpenAI/Groq put the system prompt
    # as the FIRST message in the `messages` list (role "system"), whereas
    # Anthropic takes `system=` as a separate top-level argument. Same idea,
    # different plumbing — worth knowing both.
    response = _client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,       # a grounded answer is short; bump if it truncates
        temperature=0,         # 0 = as deterministic/factual as possible — we
                               # want it to stick to the context, not get creative
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    # OpenAI-style responses put the text at choices[0].message.content
    # (Anthropic instead returns a LIST of content blocks). Another good
    # both-worlds detail for interviews.
    answer_text = response.choices[0].message.content

    return {"answer": answer_text, "sources": sources}


# Manual test: full pipeline end to end — PDF in, cited answer out.
# Run with:  python generate.py
if __name__ == "__main__":
    from ingest import extract_text_from_pdf, chunk_text
    from store import VectorStore

    # Build the store from the sample PDF (Days 2 + 3 + 4).
    text = extract_text_from_pdf("sample.pdf")
    store = VectorStore()
    store.add(chunk_text(text), source="sample.pdf")

    question = "What problem does RAG solve?"
    result = answer(question, store)

    print(f"Q: {question}\n")
    print(f"A: {result['answer']}\n")
    print("Sources the model was given:")
    for s in result["sources"]:
        print(f"  [{s['number']}] {s['source']} (chunk {s['id']}, score {s['score']:.3f})")
