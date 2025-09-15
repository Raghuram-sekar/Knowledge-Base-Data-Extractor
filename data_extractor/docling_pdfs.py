import logging
from pathlib import Path
from docling_core.types.doc import ImageRefMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

def process_pdf_folder(input_folder_path: str, output_base_dir: str = "data_docling"):
    """
    Process all PDF files in a folder and call process_pdf for each file.
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
    pdf_files = list(input_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in: {input_folder}")
        return
    print(f"Found {len(pdf_files)} PDF files to process")
    for pdf_file in pdf_files:
        process_pdf(pdf_file, output_base)
    print(f"\n Processing complete! Output saved in: {output_base}")

def process_pdf(pdf_file: Path, output_base: Path):
    """
    Process a single PDF file using the docling pipeline and save the output.
    Args:
        pdf_file: Path to the PDF file
        output_base: Base directory for organized output
    """
    try:
        print(f"\nProcessing: {pdf_file.name}")
        # Pipeline options (needed for image extraction)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 2.0  # better resolution
        # Create converter
        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        # Create structured output directory for this PDF
        pdf_output_dir = output_base / pdf_file.stem
        pdf_output_dir.mkdir(parents=True, exist_ok=True)
        # Convert PDF
        conv_res = doc_converter.convert(pdf_file)
        # Save Markdown with referenced images
        md_filename = pdf_output_dir / f"{pdf_file.stem}.md"
        conv_res.document.save_as_markdown(
            md_filename,
            image_mode=ImageRefMode.REFERENCED
        )
        print(f"  Markdown saved: {md_filename}")
    except Exception as e:
        print(f"Error processing {pdf_file.name}: {str(e)}")
        return
def main():
    # Configuration - modify these paths as needed
    workspace_path = Path(__file__).parent.parent
    
    # You can change this to any folder containing PDFs
    input_folder = workspace_path / "data"
    
    # Process all PDFs in the folder
    process_pdf_folder(str(input_folder))

if __name__ == "__main__":
    main()
