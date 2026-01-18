import pdfplumber
from pathlib import Path
from typing import List, Dict


def load_txt(file_path : Path) -> str:
    """
    load text from a .txt file
    
    """
    with open(file_path , "r" , encoding="utf-8" , errors="ignore") as f:
        text = f.read()

    return normalize_text(text)


def load_pdf(pdf_path: Path) -> List[Dict]:
    """
    Extract text from a PDF file page by page
    and preserve page-level metadata.
    """
    documents = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            tables = page.extract_tables()

            texts = []

            if page_text:
                texts.append(page_text)

            if tables:
                for table in tables:
                    table_text = " | ".join(
                        " , ".join(cell for cell in row if cell)
                        for row in table
                    )
                    texts.append(f"[TABLE]\n{table_text}")

            if texts:
                full_text = normalize_text("\n".join(texts))
                documents.append({
                    "text": full_text,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_number
                    }
                })

    return documents


def normalize_text(text: str)-> str:
    """
    Clean text : remove extra spaces and empty lines from text
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    return "\n".join(lines)