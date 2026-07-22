"""
store.py — a tiny in-memory vector store + semantic search.

This is the "shoebox of index cards" of the RAG pipeline. Earlier days built
the pieces: ingest.py cuts a PDF into chunks, embeddings.py turns each chunk
into a vector. But those vectors were computed and thrown away. Here we finally
KEEP them, and add the one operation the whole project is built around:

    search(question) -> the chunks most relevant to that question

That searching step is RETRIEVAL — the "R" in RAG. It reuses the cosine
similarity you hand-wrote on Day 3, so the function you built IS the engine.
"""

from embeddings import embed
from vectors import cosine_similarity


class VectorStore:
    """
    Holds chunks and their vectors together, and can search them.

    (C++ note: `class` in Python needs no header/.cpp split, no access
    keywords by default — everything is public. `self` is like `this`, but
    Python makes you write it as the first parameter of every method.)
    """

    def __init__(self):
        # __init__ is the constructor — it runs when you write VectorStore().
        # We keep TWO lists that always stay lined up by position:
        #   self.chunks[i]  is the text (plus bookkeeping) of card i
        #   self.vectors[i] is that same card's vector
        # Position i is the invisible thread tying a chunk to its vector.
        self.chunks = []    # list of dicts: {"id", "text", "source"}
        self.vectors = []   # list of list[float]

    def add(self, texts: list[str], source: str) -> None:
        """
        Embed a batch of chunk texts and store them.

        Args:
            texts:  the chunks from chunk_text() for one document.
            source: a label for where they came from (e.g. the filename),
                    so that on Day 5 we can say "this answer came from X".
        """
        # ONE network call embeds the whole batch. input_type defaults to
        # "document" inside embed() — correct, because these are stored passages.
        new_vectors = embed(texts)

        # zip() walks two lists together: first text with first vector, etc.
        # (C++: like iterating two vectors with one index, but no index needed.)
        for text, vector in zip(texts, new_vectors):
            self.chunks.append({
                "id": len(self.chunks),   # a stable number we can cite later
                "text": text,
                "source": source,
            })
            self.vectors.append(vector)

    def search(self, query: str, k: int = 3) -> list[tuple[float, dict]]:
        """
        Return the top-k chunks most similar to `query`, best first.

        Args:
            query: the user's question.
            k:     how many chunks to return. Not 1 (too fragile — the single
                   best chunk might miss part of the answer) and not all
                   (that's just dumping the whole doc back into the model,
                   which defeats the point of retrieval). A small handful.

        Returns:
            A list of (score, chunk) pairs, sorted highest score first.
        """
        # Embed the QUESTION as a "query", not a "document". Voyage tunes the
        # two slightly differently so a question lands NEAR the passage that
        # answers it. embed() returns a list of vectors; we asked for one, so
        # we take element [0].
        query_vector = embed([query], input_type="query")[0]

        # Score EVERY stored chunk against the question. At this scale a plain
        # loop is fine; a production store (FAISS, Chroma, pgvector) uses a
        # smarter index so it doesn't compare against millions one-by-one.
        scored = []
        for chunk, vector in zip(self.chunks, self.vectors):
            score = cosine_similarity(query_vector, vector)
            scored.append((score, chunk))

        # Sort by the score (the first item of each pair), biggest first.
        # WHY the `key=` bit: if two scores tie, Python would then try to
        # compare the two chunk DICTS to break the tie — and comparing dicts
        # with < throws an error. Telling sort to look ONLY at the score
        # (pair[0]) avoids ever touching the dict. `reverse=True` = descending.
        scored.sort(key=lambda pair: pair[0], reverse=True)

        return scored[:k]   # slice off just the first k


# Manual test: build a store from the sample PDF and run a real query.
# Run it with:  python store.py
if __name__ == "__main__":
    from ingest import extract_text_from_pdf, chunk_text

    # 1. Ingest the sample document (reusing Day 2's code).
    text = extract_text_from_pdf("sample.pdf")
    chunks = chunk_text(text)
    print(f"Ingested {len(chunks)} chunks from sample.pdf.\n")

    # 2. Build the store (this makes ONE embedding call for all chunks).
    store = VectorStore()
    store.add(chunks, source="sample.pdf")
    print(f"Stored {len(store.chunks)} chunks with vectors.\n")

    # 3. Ask a question and show what retrieval hands back.
    question = "What is this document about?"
    results = store.search(question, k=3)

    print(f"Top {len(results)} chunks for: {question!r}\n")
    for rank, (score, chunk) in enumerate(results, start=1):
        preview = chunk["text"][:150].replace("\n", " ")
        print(f"#{rank}  score={score:.3f}  (chunk id {chunk['id']})")
        print(f"     {preview}...\n")
