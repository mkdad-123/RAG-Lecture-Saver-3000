import re

# --------------------------------------
# Arabic-safe normalization (RAG oriented)
# --------------------------------------

ARABIC_DIACRITICS = re.compile(r"[ًٌٍَُِّْـ]")

def normalize_arabic_text(text: str) -> str:
    """
    Safe normalization for Arabic PDFs:
    - Remove diacritics
    - Normalize punctuation
    - Preserve word boundaries
    - DO NOT hallucinate spaces
    """

    # 1. Remove diacritics
    text = re.sub(ARABIC_DIACRITICS, "", text)

    # 2. Normalize Arabic punctuation spacing
    text = re.sub(r"\s*([،؛؟!])\s*", r"\1 ", text)
    text = re.sub(r"\s*([.:])\s*", r"\1 ", text)

    # 3. Fix obvious header glue (##عنوان)
    text = re.sub(r"(#+)([^\s#])", r"\1 \2", text)

    # 4. Separate bullets safely
    text = re.sub(r"\*\s*", "* ", text)

    # 5. Collapse excessive whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_text(text: str) -> str:
    text = normalize_arabic_text(text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines)
