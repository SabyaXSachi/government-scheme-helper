"""Text cleaning and chunking shared by the classifier and the RAG pipeline."""
import re

_PUNCT = re.compile(r"[^a-z0-9\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace.

    The same function is used at training time and at prediction time so
    the classifier always sees text shaped the same way.
    """
    text = (text or "").lower()
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def chunk_text(text: str, max_words: int = 350, overlap: int = 60) -> list[str]:
    """Split long text into chunks of at most `max_words` words, keeping a
    small tail overlap between consecutive chunks so meaning is not lost at
    the edges. Splits prefer sentence boundaries."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        words = len(sentence.split())
        if current and count + words > max_words:
            chunks.append(" ".join(current))
            tail: list[str] = []
            tail_words = 0
            for sent in reversed(current):
                tail_words += len(sent.split())
                tail.insert(0, sent)
                if tail_words >= overlap:
                    break
            current, count = tail, tail_words
        current.append(sentence)
        count += words
    if current:
        chunks.append(" ".join(current))
    return chunks


def scheme_text(record: dict) -> str:
    """Combine a scheme record's fields into one section of text used for
    chunking and embedding."""
    parts = [
        f"Scheme: {record['scheme_name']}. Category: {record['category']}.",
        f"Eligibility: {record['eligibility']}",
        f"Benefits: {record['benefits']}",
        f"How to apply: {record['how_to_apply']}",
        f"Documents required: {record['documents_required']}",
    ]
    return " ".join(parts)
