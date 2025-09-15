import pdfplumber
from pathlib import Path
import re

def is_scanned(pdf_path: str) -> bool:
    """
    Check if a PDF is scanned (image-based) or normal (text-based).
    Returns True if scanned, False otherwise.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists() or pdf_file.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"Invalid PDF file: {pdf_path}")
    
    try:
        with pdfplumber.open(pdf_file) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                return True  # empty PDFs considered scanned
            
            scanned_pages = 0
            checked_pages = min(10, total_pages)
            step = max(1, total_pages // checked_pages)
            
            for i in range(0, total_pages, step):
                page = pdf.pages[i]
                text = page.extract_text() or ""
                text = re.sub(r"\s+", " ", text).strip()
                
                # Heuristics
                text_chars = len(text)
                img_count = len(page.images or [])
                page_area = float(page.width) * float(page.height)
                img_area = sum(
                    (float(im.get("x1", 0)) - float(im.get("x0", 0))) *
                    (float(im.get("bottom", 0)) - float(im.get("top", 0)))
                    for im in page.images or []
                )
                img_ratio = img_area / page_area if page_area > 0 else 0
                
                # Decide per page
                if (text_chars < 50 and img_ratio > 0.2) or (img_ratio > 0.5):
                    scanned_pages += 1
            
            # If majority of sampled pages look scanned → classify as scanned
            return scanned_pages / checked_pages >= 0.6
    except Exception:
        return True  # If PDF can't be read, treat as scanned
    
    
def is_scanned_folder(input_folder_path: str) -> dict:
    """
    Check all PDFs in a folder and classify them as scanned or normal.
    Returns a dictionary with PDF filenames as keys and boolean values indicating if they are scanned.
    """
    input_folder = Path(input_folder_path)
    if not input_folder.exists() or not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    
    pdf_files = list(input_folder.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {input_folder}")
    
    results = {}
    for pdf_file in pdf_files:
        try:
            results[pdf_file.name] = is_scanned(str(pdf_file))
        except Exception as e:
            results[pdf_file.name] = f"Error: {str(e)}"
    
    return results

if __name__ == "__main__":
    workspace_path = Path(__file__).parent.parent
    pdf_path = workspace_path / "data"  # Change to your PDF path
    print(f"{is_scanned_folder(str(pdf_path))}")
