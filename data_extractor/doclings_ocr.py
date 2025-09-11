import logging
from pathlib import Path
from docling_core.types.doc import ImageRefMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def process_scanned_pdfs(input_folder_path: str, output_base_dir: str = "data_docling_ocr"):
    """
    Process all scanned PDF files in a folder with OCR and convert them to Markdown + images.
    
    Args:
        input_folder_path: Path to folder containing PDF files
        output_base_dir: Base directory for organized output
    """
    logging.basicConfig(level=logging.INFO)
    
    input_folder = Path(input_folder_path)
    output_base = Path(output_base_dir)
    
    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Input folder does not exist: {input_folder}")
        return
    
    # Find all PDF files in the folder
    pdf_files = list(input_folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in: {input_folder}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Pipeline options for OCR
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
    pipeline_options.generate_picture_images = True  # export figures
    pipeline_options.images_scale = 2.0  # higher resolution
    
    # Create converter
    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    
    # Process each PDF file
    for pdf_file in pdf_files:
        try:
            print(f"\nProcessing (OCR): {pdf_file.name}")
            
            # Create structured output directory for this PDF
            pdf_output_dir = output_base / pdf_file.stem
            pdf_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert PDF with OCR
            conv_res = doc_converter.convert(pdf_file)
            
            # Save Markdown with referenced images
            md_filename = pdf_output_dir / f"{pdf_file.stem}.md"
            conv_res.document.save_as_markdown(
                md_filename,
                image_mode=ImageRefMode.REFERENCED
            )
            
            print(f"OCR Markdown saved: {md_filename}")
            
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {str(e)}")
            continue
    
    print(f"\nOCR Processing complete! Output saved in: {output_base}")


def main():
    # Modify these paths as needed
    workspace_path = Path(__file__).parent.parent
    
    # Folder containing scanned PDFs
    input_folder = workspace_path / "data_ocr"
    
    # Process all PDFs in the folder with OCR
    process_scanned_pdfs(str(input_folder))


if __name__ == "__main__":
    main()
