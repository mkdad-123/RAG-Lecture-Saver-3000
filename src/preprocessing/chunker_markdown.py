from typing import List, Dict
import re

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# --------------------------------------
# Settings
# --------------------------------------
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

SEPARATORS = ["\n\n", "\n", " ", ""]

MIN_CHUNK_LENGTH = 40  


# --------------------------------------
# Main API 
# --------------------------------------
def chunk_document(
    document: Dict,
    chunk_size: int = 800,
    overlap: int = 100,
) -> List[str]:
    """
    Final structure-aware chunking for Markdown (Docling output)

    Input:
        document = {
            "text": "... markdown ...",
            "metadata": {
                "source": "...",
                "page": int,
                "format": "markdown"
            }
        }

    Output:
        List[str]  
    """

    text = document

    # 1) Header-based splitting
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    header_sections = header_splitter.split_text(text)

    # 2) Recursive splitter (size-based)
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
    )

    final_chunks: List[str] = []

    for section in header_sections:
        section_text = section.page_content.strip()

        # -----------------------------
        # Noise filtering 
        # -----------------------------
        if _is_noise(section_text):
            continue

        # -----------------------------
        # Table handling
        # -----------------------------
        if _looks_like_markdown_table(section_text):
            final_chunks.append(section_text)
            continue

        # -----------------------------
        # Merge header-only chunks
        # -----------------------------
        if _is_header_only(section_text):
            continue

        # -----------------------------
        # Size-based splitting
        # -----------------------------
        sub_chunks = recursive_splitter.split_text(section_text)

        for sub in sub_chunks:
            sub = sub.strip()
            if len(sub) < MIN_CHUNK_LENGTH:
                continue

            final_chunks.append(sub)

    return final_chunks


# --------------------------------------
# Helpers
# --------------------------------------
def _looks_like_markdown_table(text: str) -> bool:
    lines = text.splitlines()
    if len(lines) < 2:
        return False

    has_pipes = any("|" in line for line in lines)
    has_separator = any(
        re.match(r"^\s*\|?[\s:-]+\|", line) for line in lines
    )

    return has_pipes and has_separator


def _is_header_only(text: str) -> bool:
    """
    Detect chunks that are only headers (e.g. '## العنوان')
    """
    lines = text.splitlines()
    if len(lines) != 1:
        return False

    return lines[0].lstrip().startswith("#")


def _is_noise(text: str) -> bool:
    """
    Remove garbage chunks: symbols, single letters, etc.
    """
    stripped = text.strip()

    if len(stripped) < 10:
        return True

    if re.fullmatch(r"[■S\s]+", stripped):
        return True

    return False
