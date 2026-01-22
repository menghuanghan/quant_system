
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class CleaningMixin:
    """
    Data cleaning and extraction mixin for unstructured collectors
    """
    
    def extract_pdf_text(self, pdf_bytes: bytes, max_pages: int = 20) -> str:
        """
        Extract text from PDF bytes using multiple backends
        
        尝试顺序:
        1. pdfplumber (推荐，支持表格和复杂布局)
        2. PyMuPDF (快速，支持加密PDF)
        3. PyPDF2 (备选)
        
        Args:
            pdf_bytes: PDF file content in bytes
            max_pages: Maximum pages to extract
            
        Returns:
            Extracted text string
        """
        if not pdf_bytes:
            return ""
        
        # 尝试方案1: pdfplumber
        text = self._extract_with_pdfplumber(pdf_bytes, max_pages)
        if text and len(text.strip()) > 50:
            return text
        
        # 尝试方案2: PyMuPDF (fitz)
        text = self._extract_with_pymupdf(pdf_bytes, max_pages)
        if text and len(text.strip()) > 50:
            return text
        
        # 尝试方案3: PyPDF2
        text = self._extract_with_pypdf(pdf_bytes, max_pages)
        if text and len(text.strip()) > 50:
            return text
        
        logger.warning("All PDF extraction methods failed or returned insufficient text")
        return ""
    
    def _extract_with_pdfplumber(self, pdf_bytes: bytes, max_pages: int) -> str:
        """使用pdfplumber提取文本"""
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text_parts = []
                for i, page in enumerate(pdf.pages[:max_pages]):
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                return "\n".join(text_parts)
        except Exception as e:
            logger.debug(f"pdfplumber extraction failed: {e}")
            return ""
    
    def _extract_with_pymupdf(self, pdf_bytes: bytes, max_pages: int) -> str:
        """使用PyMuPDF (fitz)提取文本"""
        try:
            import fitz
            pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_parts = []
            for i in range(min(len(pdf), max_pages)):
                page = pdf[i]
                text = page.get_text()
                if text:
                    text_parts.append(text)
            pdf.close()
            return "\n".join(text_parts)
        except Exception as e:
            logger.debug(f"PyMuPDF extraction failed: {e}")
            return ""
    
    def _extract_with_pypdf(self, pdf_bytes: bytes, max_pages: int) -> str:
        """使用PyPDF2提取文本"""
        try:
            from PyPDF2 import PdfReader
            pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
            text_parts = []
            for i in range(min(len(pdf_reader.pages), max_pages)):
                page = pdf_reader.pages[i]
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n".join(text_parts)
        except Exception as e:
            logger.debug(f"PyPDF2 extraction failed: {e}")
            return ""

    def is_pdf_scanned(self, pdf_bytes: bytes) -> bool:
        """
        Check if PDF is likely scanned (no text layer)
        
        Args:
            pdf_bytes: PDF file content in bytes
            
        Returns:
            True if PDF seems to be scanned (no extractable text)
        """
        # Simple heuristic: if we can't extract text from the first page, assume scanned
        # or encrypted/protected without copy rights.
        text = self.extract_pdf_text(pdf_bytes, max_pages=1)
        # If text is very short, it might be an image-only PDF
        return len(text.strip()) < 10
