import json
from pathlib import Path
from typing import List , Dict
from ingestion.pdf_to_markdown import MarkdownLoader
from pathlib import Path

def load_documents(path: Path):

    documents: List[Dict] = []
    loader = MarkdownLoader()

    if path.is_file():
        if path.suffix.lower() == ".pdf" :

            documents.extend(loader.load_pdf(path))

        else :
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
    elif path.is_dir():
        pdf_files = sorted(path.rglob("*.pdf"))

        if not pdf_files:
            raise ValueError("No PDF files found in the directory")
        
        for pdf_path in pdf_files :
            documents.extend(loader.load_pdf(pdf_path))

    else:
        raise ValueError(f"Invalid path: {path}")
    
    return documents

    