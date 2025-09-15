from pathlib import Path
from data_extractor.classify_pdfs import is_scanned
from data_extractor.docling_parser import pdf_parser


def process_pdfs(input_folder_path: str, output_base_dir: str = "data_docling_output"):
    """
    Process all PDF files in a folder, classify them as scanned or normal,
    and call pdf_parser for each file with appropriate OCR settings.
    Args:
        input_folder_path: Path to folder containing PDF files
        output_base_dir: Base directory for organized output
    """
    from pathlib import Path
    import logging

    logging.basicConfig(level=logging.INFO)
    input_folder = Path(input_folder_path)
    output_base = Path(output_base_dir)
    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Input folder does not exist: {input_folder}")
        return
    pdf_files = list(input_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in: {input_folder}")
        return
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        try:
            scanned = is_scanned(str(pdf_file))
            print(f"\nClassified '{pdf_file.name}' as {'scanned' if scanned else 'normal'}")
            pdf_parser(pdf_file, output_base, do_ocr=scanned)
        except Exception as e:
            print(f"Error classifying {pdf_file.name}: {str(e)}")
    
    print(f"\nProcessing complete! Output saved in: {output_base}")
    
if __name__ == "__main__":
    workspace_path = Path(__file__).parent
    pdf_path = workspace_path / "data"  # Change to your PDF path
    process_pdfs(str(pdf_path))