"""
PDF Text Extractor
Extracts plain text from PDF files for dataset creation
"""

import os
import json
import re
from typing import List, Dict
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


class PDFTextExtractor:
    """Extract text from PDF files using PyMuPDF or pdfplumber"""
    
    def __init__(self, pdf_path: str):
        """
        Initialize the PDF text extractor
        
        Args:
            pdf_path: Path to the PDF file
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.pdf_path = pdf_path
        self.text_data = []
        
    def extract_with_pymupdf(self) -> List[Dict]:
        """
        Extract text using PyMuPDF (fitz)
        
        Returns:
            List of dictionaries containing page text and metadata
        """
        if fitz is None:
            raise ImportError("PyMuPDF not installed. Install with: pip install pymupdf")
        
        print(f"Extracting text from: {self.pdf_path}")
        doc = fitz.open(self.pdf_path)
        extracted_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Clean the text
            cleaned_text = self._clean_text(text)
            
            if cleaned_text.strip():  # Only add non-empty pages
                extracted_data.append({
                    'page_number': page_num + 1,
                    'text': cleaned_text,
                    'char_count': len(cleaned_text)
                })
                print(f"Page {page_num + 1}: Extracted {len(cleaned_text)} characters")
        
        doc.close()
        self.text_data = extracted_data
        return extracted_data
    
    def extract_with_pdfplumber(self) -> List[Dict]:
        """
        Extract text using pdfplumber
        
        Returns:
            List of dictionaries containing page text and metadata
        """
        if pdfplumber is None:
            raise ImportError("pdfplumber not installed. Install with: pip install pdfplumber")
        
        print(f"Extracting text from: {self.pdf_path}")
        extracted_data = []
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                
                if text:
                    # Clean the text
                    cleaned_text = self._clean_text(text)
                    
                    if cleaned_text.strip():
                        extracted_data.append({
                            'page_number': page_num + 1,
                            'text': cleaned_text,
                            'char_count': len(cleaned_text)
                        })
                        print(f"Page {page_num + 1}: Extracted {len(cleaned_text)} characters")
        
        self.text_data = extracted_data
        return extracted_data
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing excessive whitespace and fixing common issues
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace while preserving paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove trailing/leading whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Fix hyphenated words at line breaks
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Normalize multiple spaces to single space
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def extract(self, method: str = 'auto') -> List[Dict]:
        """
        Extract text using the specified method
        
        Args:
            method: 'pymupdf', 'pdfplumber', or 'auto' (default)
            
        Returns:
            List of dictionaries containing page text and metadata
        """
        if method == 'auto':
            # Try PyMuPDF first, fall back to pdfplumber
            if fitz is not None:
                return self.extract_with_pymupdf()
            elif pdfplumber is not None:
                return self.extract_with_pdfplumber()
            else:
                raise ImportError("No PDF library available. Install PyMuPDF or pdfplumber")
        elif method == 'pymupdf':
            return self.extract_with_pymupdf()
        elif method == 'pdfplumber':
            return self.extract_with_pdfplumber()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def save_as_text(self, output_path: str):
        """
        Save extracted text as a plain text file
        
        Args:
            output_path: Path to save the text file
        """
        if not self.text_data:
            raise ValueError("No text data to save. Run extract() first.")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for page_data in self.text_data:
                f.write(f"{'='*60}\n")
                f.write(f"PAGE {page_data['page_number']}\n")
                f.write(f"{'='*60}\n\n")
                f.write(page_data['text'])
                f.write(f"\n\n")
        
        print(f"Text saved to: {output_path}")
    
    def save_as_json(self, output_path: str):
        """
        Save extracted text as a JSON file with metadata
        
        Args:
            output_path: Path to save the JSON file
        """
        if not self.text_data:
            raise ValueError("No text data to save. Run extract() first.")
        
        output_data = {
            'source_pdf': os.path.basename(self.pdf_path),
            'total_pages': len(self.text_data),
            'total_characters': sum(page['char_count'] for page in self.text_data),
            'pages': self.text_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"JSON saved to: {output_path}")
    
    def get_full_text(self) -> str:
        """
        Get all extracted text as a single string
        
        Returns:
            Combined text from all pages
        """
        if not self.text_data:
            raise ValueError("No text data available. Run extract() first.")
        
        return '\n\n'.join(page['text'] for page in self.text_data)


def main():
    """Example usage of PDFTextExtractor"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_text_extractor.py <path_to_pdf>")
        print("\nExample:")
        print("  python pdf_text_extractor.py document.pdf")
        return
    
    pdf_path = sys.argv[1]
    
    try:
        # Create extractor
        extractor = PDFTextExtractor(pdf_path)
        
        # Extract text
        extractor.extract()
        
        # Generate output filenames
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        txt_output = f"{base_name}_extracted.txt"
        json_output = f"{base_name}_extracted.json"
        
        # Save in both formats
        extractor.save_as_text(txt_output)
        extractor.save_as_json(json_output)
        
        print("\n" + "="*60)
        print("Extraction complete!")
        print(f"Total pages: {len(extractor.text_data)}")
        print(f"Total characters: {sum(p['char_count'] for p in extractor.text_data)}")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
