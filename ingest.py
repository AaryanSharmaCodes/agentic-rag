"""
ingest.py — turning a PDF file into clean, chunk-able text.

This module is the "read the book and copy it onto index cards" step
of the RAG pipeline. Today it does the FIRST half: extraction.
(Chunking gets added in block 2.)
"""

from pypdf import PdfReader


def extract_text_from_pdf(path: str) -> str:
    """
    Open a PDF and pull all of its text out into one big string.

    Args:
        path: location of the PDF file on disk, e.g. "sample.pdf"

    Returns:
        The full text of the document, pages joined together.
    """
    # PdfReader opens the file and understands the PDF format for us.
    reader = PdfReader(path)

    # A PDF is a list of pages. We walk them one by one and collect text.
    pages = []
    for page in reader.pages:
        # extract_text() can return None for a page with no readable text
        # (e.g. a page that's just an image). `or ""` turns that None into
        # an empty string so we never crash when joining.
        pages.append(page.extract_text() or "")

    # Join pages with a newline so words from page 1 and page 2 don't
    # get glued together into a fake word at the seam.
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Cut a long string into overlapping, card-sized pieces.

    Args:
        text:       the full document text from extract_text_from_pdf().
        chunk_size: how many CHARACTERS each chunk holds (1000 chars ~ 250 tokens).
        overlap:    how many characters the end of one chunk repeats at the
                    start of the next, so ideas don't get cut in half.

    Returns:
        A list of text chunks.
    """
    # Guard rail: if overlap is as big as the chunk, the window never moves
    # forward (see the `start +=` line below) and we'd loop FOREVER.
    # Fail loudly and early instead.
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        # Grab a window of `chunk_size` characters starting at `start`.
        # Slicing past the end of a string is safe in Python — it just stops
        # at the end — so the final (short) chunk needs no special handling.
        end = start + chunk_size
        chunks.append(text[start:end])

        # THE KEY LINE. Move the window forward by (chunk_size - overlap),
        # NOT by the full chunk_size. Stepping a little short is what makes
        # the next chunk re-include the tail of this one. That short step
        # IS the overlap.
        start += chunk_size - overlap

    return chunks


# A tiny manual test: run `python ingest.py` directly to eyeball the output.
# This block only runs when you execute the file itself, NOT when another
# file imports it. (That's what __name__ == "__main__" means.)
if __name__ == "__main__":
    text = extract_text_from_pdf("sample.pdf")
    print(f"Extracted {len(text)} characters.")

    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks.\n")

    # Print the END of chunk 0 and the START of chunk 1 so you can SEE the
    # overlap with your own eyes — the same text should appear in both.
    print("--- last 120 chars of chunk 0 ---")
    print(repr(chunks[0][-120:]))
    print("\n--- first 120 chars of chunk 1 ---")
    print(repr(chunks[1][:120]))
