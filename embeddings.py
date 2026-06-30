"""
embeddings.py — turn text into vectors using Voyage AI.

This is the "get the pin's coordinates" step. We send text to Voyage's
servers; they run their embedding model and hand back a list of numbers
for each piece of text. We keep this behind ONE function so the rest of
the project never cares *which* provider we use — swap Voyage for another
model later and only this file changes. (That's the "keep the model
swappable" decision from Day 1, made real.)
"""

import os

import voyageai
from dotenv import load_dotenv

# load_dotenv() reads the .env file and copies VOYAGE_API_KEY into the
# environment, so the line below can find it. The key NEVER appears in
# our code — it lives only in .env, which is git-ignored.
load_dotenv()

# Create the Voyage client once, at import time. It automatically picks up
# VOYAGE_API_KEY from the environment. Reusing one client is cheaper than
# building a new one on every call.
_client = voyageai.Client()

# Name the model in ONE place. If we ever change models, we edit this string
# and nothing else. "voyage-3" is a solid general-purpose embedding model.
MODEL = "voyage-3"


def embed(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """
    Turn a list of strings into a list of embedding vectors.

    Args:
        texts:      the pieces of text to embed (e.g. your document chunks).
        input_type: "document" when embedding stored text, or "query" when
                    embedding a user's question. Voyage tunes the vector
                    slightly differently for each so questions and documents
                    land in compatible spots. (We'll use "query" on Day 4.)

    Returns:
        One vector (list of floats) per input string, in the same order.
    """
    # One network call embeds the whole batch — far faster than one call per
    # string. result.embeddings is a list of vectors, aligned to `texts`.
    result = _client.embed(texts, model=MODEL, input_type=input_type)
    return result.embeddings


# Manual test: run `python embeddings.py` to SEE semantic search work.
if __name__ == "__main__":
    from vectors import cosine_similarity

    sentences = [
        "How do I fix my car?",        # 0
        "automobile repair guide",     # 1  — different words, same idea as 0
        "best recipe for banana bread" # 2  — totally unrelated
    ]

    vectors = embed(sentences)

    # Compare sentence 0 against 1 (should be HIGH) and 0 against 2 (LOW).
    print(f"Each vector has {len(vectors[0])} numbers.\n")
    print("car-fix  vs  automobile-repair :",
          round(cosine_similarity(vectors[0], vectors[1]), 3))
    print("car-fix  vs  banana-bread      :",
          round(cosine_similarity(vectors[0], vectors[2]), 3))
