import re
import logging
from pathlib import Path
from typing import List, Dict
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel, TableItem, TextItem
from  src.preprocessing.cleaner import normalize_text
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarkdownLoader:
    def __init__(self):
        options = PdfPipelineOptions()
        options.do_ocr = False           
        options.do_table_structure = True 

        self.converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=options)
            }
        )


    def get_element_markdown(self, element) -> str:

        if isinstance(element, TableItem):
            try:
                return element.export_to_markdown()
            except:
                return element.text if hasattr(element, 'text') else ""
        
        elif isinstance(element, TextItem):
            text = element.text
            label = element.label
            if label == DocItemLabel.TITLE:
                return f"# {text}"
            elif label == DocItemLabel.SECTION_HEADER:
                           
                return f"## {text}"
            elif label == DocItemLabel.LIST_ITEM:
                return f"* {text}"
            return text
        
        return ""

    def load_pdf(self, pdf_path: Path) -> List[Dict]:
        documents = []
        try:
            logger.info(f"Processing.... {pdf_path.name}")
            result = self.converter.convert(str(pdf_path))
            doc = result.document
            
            pages_content = {}

            for element, level in doc.iterate_items():
                if hasattr(element, 'prov') and element.prov:
                    page_no = element.prov[0].page_no
                    
                    element_md = self.get_element_markdown(element)
                    
                    if element_md:
                        if page_no not in pages_content:
                            pages_content[page_no] = []
                        pages_content[page_no].append(element_md)

            for page_number in sorted(pages_content.keys()):
                raw_text = "\n\n".join(pages_content[page_number])
                
                final_text = normalize_text(raw_text)
                
                if final_text.strip():
                    documents.append({
                        "text": final_text,
                        "metadata": {
                            "source": pdf_path.name,
                            "page": page_number,
                            "format": "markdown"
                        }
                    })
            
            logger.info(f" {len(documents)} successfully")
            return documents

        except Exception as e:
            logger.error(f"error in processing {pdf_path.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
