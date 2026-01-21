from dataclasses import dataclass

@dataclass
class MarkdownDocument:
    content: str
    source: str

def load_markdown(md_text: str , source: str)->MarkdownDocument:
    return MarkdownDocument(
        content=md_text,
        source=source
    )