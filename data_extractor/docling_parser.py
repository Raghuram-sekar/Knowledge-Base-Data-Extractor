import logging
from pathlib import Path
from docling_core.types.doc import ImageRefMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

def pdf_parser(pdf_file: Path, output_base: Path, do_ocr: bool = False):
	"""
	Process a single PDF file, optionally with OCR, and save the output as Markdown with referenced images.
	Args:
		pdf_file: Path to the PDF file
		output_base: Base directory for organized output
		do_ocr: Whether to enable OCR processing
	"""
	try:
		print(f"\nProcessing: {pdf_file.name} (OCR: {do_ocr})")
		pipeline_options = PdfPipelineOptions()
		if do_ocr:
			pipeline_options.do_ocr = True
			pipeline_options.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
		pipeline_options.generate_picture_images = True
		pipeline_options.images_scale = 2.0
		doc_converter = DocumentConverter(
			format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
		)
		pdf_output_dir = output_base / pdf_file.stem
		pdf_output_dir.mkdir(parents=True, exist_ok=True)
		conv_res = doc_converter.convert(pdf_file)
		md_filename = pdf_output_dir / f"{pdf_file.stem}.md"
		conv_res.document.save_as_markdown(
			md_filename,
			image_mode=ImageRefMode.REFERENCED
		)
		print(f"  Markdown saved: {md_filename}")
	except Exception as e:
		print(f"  Error processing {pdf_file.name}: {str(e)}")
		return

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    input_folder_path = "data_ocr"  # specify your input folder here
    output_base_dir = "data_docling_output_new"  # specify your output folder here
    do_ocr = True  # set to True if OCR is needed

    input_folder = Path(input_folder_path)
    output_base = Path(output_base_dir)

    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Input folder does not exist: {input_folder}")
    else:
        pdf_files = list(input_folder.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in: {input_folder}")
        else:
            print(f"Found {len(pdf_files)} PDF files to process")
            for pdf_file in pdf_files:
                pdf_parser(pdf_file, output_base, do_ocr=do_ocr)
            print(f"\nProcessing complete! Output saved in: {output_base}")