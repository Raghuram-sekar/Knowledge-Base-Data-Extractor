import os
# Set tokenizers parallelism to false to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import psutil
from pydantic.types import SecretType
from .storage import Storage, LocalStorage, S3Storage
from .engine import OCREngineFactory, OCREngine, OCRResult
from pathlib import Path
import re
from typing import Optional, Union, Dict, Any
from core import logger
from core.config import config

import pdfplumber
from docling_core.types.doc import ImageRefMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

"""
Parser class for extracting information from PDF files.

This class provides a modular approach to PDF parsing with:
- Configurable storage backends (Local, S3)
- Multiple OCR engine support (Tesseract, EasyOCR)
- Automatic document classification (scanned vs text-based)
- Comprehensive logging and error handling

Attributes:
    storage (Storage): The storage backend for saving processed files.
    ocr_engine (OCREngine): The OCR engine for processing scanned documents.

Methods:
    parse(filepath: Path, output_base: Path) -> Dict[str, Any]:
        Parse a PDF file and extract relevant information.

    classify_document(filepath: Path) -> bool:
        Classify whether a PDF is a scanned document requiring OCR.
"""


class Parser:
    """
    Advanced PDF parser with configurable storage and OCR engines.

    This class provides a complete solution for PDF document processing,
    supporting both text-based and scanned documents with various output formats.
    """

    def __init__(self, 
                 storage: Optional[Storage] = None,
                 ocr_engine_name: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Parser with configurable backends.

        Args:
            storage: Storage backend instance (LocalStorage or S3Storage)
            ocr_engine_name: Name of OCR engine ('tesseract' or 'easyocr')
            **kwargs: Additional configuration options
                - output_storage: Override config OUTPUT_STORAGE
                - ocr_engine: Override config OCR_ENGINE
        """
        self.logger = logger.get_logger(__name__)

        # Initialize storage backend
        self._init_storage(storage, **kwargs)

        # Initialize OCR engine
        self._init_ocr_engine(ocr_engine_name, **kwargs)
        
        # Configuration
        self.config = kwargs
        
        self.logger.info(f"Parser initialized with storage: {type(self.storage).__name__}, "
                        f"OCR engine: {type(self.ocr_engine).__name__}")
    
    def _init_storage(self, storage: Optional[Storage], **kwargs) -> None:
        """Initialize the storage backend."""
        if storage:
            self.storage = storage
        else:
            # Use config to determine storage type
            output_storage = kwargs.get('output_storage', config.OUTPUT_STORAGE).upper()
            
            if output_storage == 'S3':
                try:
                    self.storage = S3Storage()
                    self.logger.info("Using S3Storage backend")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize S3Storage: {e}, falling back to LocalStorage")
                    self.storage = LocalStorage()
            else:
                self.storage = LocalStorage()
                self.logger.info("Using LocalStorage backend")
    
    def _init_ocr_engine(self, ocr_engine_name: Optional[str], **kwargs) -> None:
        """Initialize the OCR engine."""
        if ocr_engine_name is None:
            ocr_engine_name = kwargs.get('ocr_engine', config.OCR_ENGINE).lower()
        
        try:
            # Try to create the requested engine
            self.ocr_engine = OCREngineFactory.create_engine(ocr_engine_name, **kwargs)
            
            # Check if engine is available
            if not self.ocr_engine.is_available():
                raise RuntimeError(f"OCR engine '{ocr_engine_name}' is not available")
            
            self.logger.info(f"Using {ocr_engine_name} OCR engine")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize {ocr_engine_name} OCR engine: {e}")
            
            # Try to fall back to available engines
            available_engines = OCREngineFactory.get_available_and_ready_engines()
            
            if available_engines:
                fallback_engine = available_engines[0]
                self.logger.info(f"Falling back to {fallback_engine} OCR engine")
                self.ocr_engine = OCREngineFactory.create_engine(fallback_engine, **kwargs)
            else:
                self.logger.error("No OCR engines are available")
                raise RuntimeError("No OCR engines are available")

    def _pdf_parser(self,
                   pdf_file: Path,
                   output_base: Path,
                   do_ocr: bool = False,
                   **kwargs) -> Dict[str, Any]:
        """
        Process a single PDF file, optionally with OCR, and save the output.
        
        Args:
            pdf_file: Path to the PDF file
            output_base: Base directory for organized output
            do_ocr: Whether to enable OCR processing
            **kwargs: Additional processing options
            
        Returns:
            Dict containing processing results and metadata
        """
        import time
        
        # Start overall timing
        start_time = time.time()
        timing_data = {}
        
        try:
            self.logger.info(f"Processing: {pdf_file.name} (OCR: {do_ocr})")
            
            # Configure docling pipeline with comprehensive extraction
            pipeline_setup_start = time.time()
            pipeline_options = PdfPipelineOptions()
            
            # OCR Configuration using the initialized engine
            pipeline_options.do_ocr = do_ocr or config.FORCE_FULL_PAGE_OCR
            if pipeline_options.do_ocr:
                # Use Tesseract options but leverage our OCR engine for post-processing
                pipeline_options.ocr_options = TesseractCliOcrOptions(
                    force_full_page_ocr=config.FORCE_FULL_PAGE_OCR
                )
            
            # Enhanced extraction options using config
            pipeline_options.do_table_structure = config.EXTRACT_TABLES
            pipeline_options.do_formula_enrichment = config.EXTRACT_FORMULAS
            pipeline_options.do_picture_description = config.GENERATE_IMAGE_DESCRIPTIONS
            
            # Image generation settings from config
            pipeline_options.generate_picture_images = config.SAVE_IMAGES
            pipeline_options.images_scale = config.IMAGE_SCALE
            
            # Additional comprehensive extraction options
            # Note: generate_table_images is deprecated, using generate_page_images instead
            if config.SAVE_IMAGES and config.EXTRACT_TABLES:
                pipeline_options.generate_page_images = True
            
            # Create document converter
            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            timing_data['pipeline_setup'] = round(time.time() - pipeline_setup_start, 3)
            
            # Process document
            conversion_start = time.time()
            conv_res = doc_converter.convert(pdf_file)
            timing_data['document_conversion'] = round(time.time() - conversion_start, 3)
            
            # Prepare output directory structure
            # Use exact PDF filename (without extension) as folder name, replace spaces with underscores
            pdf_folder_name = pdf_file.stem.replace(" ", "_")
            pdf_output_dir = output_base / pdf_folder_name
            images_dir = pdf_output_dir / "images"
            
            # Generate markdown content with enhanced image handling
            markdown_start = time.time()
            md_content = conv_res.document.export_to_markdown(
                image_mode=ImageRefMode.REFERENCED,
                image_placeholder="![{image_name}](images/{image_name})"
            )
            timing_data['markdown_generation'] = round(time.time() - markdown_start, 3)
            
            # Save markdown file
            save_start = time.time()
            md_filename = f"{pdf_file.stem}.md"
            md_path = pdf_output_dir / md_filename
            
            save_success = self.storage.save(
                data=md_content,
                path=md_path,
                format='text',
                ensure_dir=True
            )
            
            if not save_success:
                raise RuntimeError("Failed to save markdown file")
            
            timing_data['markdown_save'] = round(time.time() - save_start, 3)
            
            # Process and save images if enabled
            images_saved = []
            image_descriptions = {}
            
            if config.SAVE_IMAGES and hasattr(conv_res.document, 'pictures'):
                image_processing_start = time.time()
                images_saved, image_descriptions = self._process_and_save_images(
                    conv_res.document.pictures, 
                    images_dir, 
                    pdf_folder_name
                )
                timing_data['image_processing'] = round(time.time() - image_processing_start, 3)
            
            # Enhanced OCR processing on extracted text using our engine
            ocr_enhanced_content = None
            if do_ocr and self.ocr_engine.is_available():
                ocr_enhancement_start = time.time()
                try:
                    # Get raw text from document
                    raw_text = conv_res.document.export_to_text()
                    
                    # Use our OCR engine for additional processing if needed
                    # This could be useful for improving text quality or extracting from images
                    ocr_enhanced_content = self._enhance_text_with_ocr(raw_text, pdf_file)
                except Exception as e:
                    self.logger.warning(f"OCR enhancement failed: {e}")
                timing_data['ocr_enhancement'] = round(time.time() - ocr_enhancement_start, 3)
            
            # Calculate total processing time
            total_time = round(time.time() - start_time, 3)
            timing_data['total_processing'] = total_time
            
            # Calculate file size for performance metrics
            file_size_mb = round(pdf_file.stat().st_size / (1024 * 1024), 2) if pdf_file.exists() else 0
            pages_per_second = round(len(conv_res.document.pages) / total_time, 2) if total_time > 0 else 0
            mb_per_second = round(file_size_mb / total_time, 2) if total_time > 0 else 0
            
            # Save comprehensive document metadata
            metadata_start = time.time()
            metadata = {
                'source_file': str(pdf_file),
                'processing_timestamp': self._get_timestamp(),
                'ocr_enabled': pipeline_options.do_ocr,
                'ocr_engine': type(self.ocr_engine).__name__,
                'storage_backend': type(self.storage).__name__,
                'output_directory': str(pdf_output_dir),
                'folder_name': pdf_folder_name,
                'markdown_file': md_filename,
                'page_count': len(conv_res.document.pages) if conv_res.document.pages else 0,
                'images_generated': pipeline_options.generate_picture_images,
                'images_saved': len(images_saved),
                'image_files': images_saved,
                'image_descriptions': image_descriptions,
                'tables_extracted': pipeline_options.do_table_structure,
                'formulas_extracted': pipeline_options.do_formula_enrichment,
                'figures_extracted': config.EXTRACT_FIGURES,
                'processing_options': {
                    'image_format': config.IMAGE_FORMAT,
                    'image_quality': config.IMAGE_QUALITY,
                    'image_dpi': config.IMAGE_DPI,
                    'image_scale': config.IMAGE_SCALE,
                    'extract_tables': config.EXTRACT_TABLES,
                    'extract_formulas': config.EXTRACT_FORMULAS,
                    'extract_figures': config.EXTRACT_FIGURES,
                    'force_full_page_ocr': config.FORCE_FULL_PAGE_OCR,
                    **kwargs
                },
                'document_structure': {
                    'has_tables': bool(getattr(conv_res.document, 'tables', [])),
                    'has_figures': bool(getattr(conv_res.document, 'pictures', [])),
                    'has_formulas': pipeline_options.do_formula_enrichment,
                    'text_length': len(md_content),
                    'images_count': len(images_saved)
                },
                # Performance and timing data
                'performance_metrics': {
                    'file_size_mb': file_size_mb,
                    'processing_time_seconds': total_time,
                    'pages_per_second': pages_per_second,
                    'mb_per_second': mb_per_second,
                    'timing_breakdown': timing_data  # Always include timing data
                }
            }
            
            # Add OCR enhanced content to metadata if available
            if ocr_enhanced_content:
                metadata['ocr_enhanced_available'] = True
                # Optionally save OCR enhanced content as separate file
                ocr_path = pdf_output_dir / f"{pdf_file.stem}_ocr_enhanced.txt"
                self.storage.save(
                    data=ocr_enhanced_content,
                    path=ocr_path,
                    format='text'
                )
                metadata['ocr_enhanced_file'] = f"{pdf_file.stem}_ocr_enhanced.txt"
            
            # Save comprehensive metadata
            metadata_path = pdf_output_dir / f"{pdf_file.stem}_metadata.json"
            metadata_success = self.storage.save(
                data=metadata,
                path=metadata_path,
                format='json'
            )
            
            timing_data['metadata_save'] = round(time.time() - metadata_start, 3)
            
            # Update total time with metadata save
            total_time = round(time.time() - start_time, 3)
            timing_data['total_processing'] = total_time
            
            if metadata_success:
                self.logger.info(f"✅ Successfully processed {pdf_file.name} in {total_time}s")
                # Log performance stats (always enabled)
                self.logger.info(f"   📊 Performance: {pages_per_second} pages/sec, {mb_per_second} MB/sec")
                self.logger.info(f"   ⏱️  Timing breakdown: {timing_data}")
            else:
                self.logger.warning(f"Failed to save metadata for {pdf_file.name}")
            
            return {
                'status': 'success',
                'markdown_saved': save_success,
                'metadata_saved': metadata_success,
                'output_path': str(pdf_output_dir),
                'images_saved': len(images_saved),
                'processing_time': total_time,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {pdf_file.name}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'source_file': str(pdf_file),
                'processing_time': round(time.time() - start_time, 3) if 'start_time' in locals() else 0
            }
        finally:
            # Clean up memory after processing
            self._cleanup_memory()

    def _pdf_parser_with_original_name(self,
                                     pdf_file: Path,
                                     output_base: Path,
                                     original_filename: str,
                                     do_ocr: bool = False,
                                     **kwargs) -> Dict[str, Any]:
        """
        Process a PDF file using original filename for folder structure (used for S3 files).
        
        Args:
            pdf_file: Path to the temporary PDF file
            output_base: Base directory for organized output
            original_filename: Original filename from S3 path
            do_ocr: Whether to enable OCR processing
            **kwargs: Additional processing options
            
        Returns:
            Dict containing processing results and metadata
        """
        import time
        
        # Start overall timing
        start_time = time.time()
        timing_data = {}
        
        # Extract original filename stem (without extension)
        original_stem = Path(original_filename).stem
        
        try:
            self.logger.info(f"Processing: {original_filename} (OCR: {do_ocr})")
            
            # Configure docling pipeline with comprehensive extraction
            pipeline_setup_start = time.time()
            pipeline_options = PdfPipelineOptions()
            
            # OCR Configuration using the initialized engine
            pipeline_options.do_ocr = do_ocr or config.FORCE_FULL_PAGE_OCR
            if pipeline_options.do_ocr:
                # Use Tesseract options but leverage our OCR engine for post-processing
                pipeline_options.ocr_options = TesseractCliOcrOptions(
                    force_full_page_ocr=config.FORCE_FULL_PAGE_OCR
                )
            
            # Enhanced extraction options using config
            pipeline_options.do_table_structure = config.EXTRACT_TABLES
            pipeline_options.do_formula_enrichment = config.EXTRACT_FORMULAS
            pipeline_options.do_picture_description = config.GENERATE_IMAGE_DESCRIPTIONS
            
            # Image generation settings from config
            pipeline_options.generate_picture_images = config.SAVE_IMAGES
            pipeline_options.images_scale = config.IMAGE_SCALE
            
            # Additional comprehensive extraction options
            # Note: generate_table_images is deprecated, using generate_page_images instead
            if config.SAVE_IMAGES and config.EXTRACT_TABLES:
                pipeline_options.generate_page_images = True
            
            # Create document converter
            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            timing_data['pipeline_setup'] = round(time.time() - pipeline_setup_start, 3)
            
            # Process document
            conversion_start = time.time()
            conv_res = doc_converter.convert(pdf_file)
            timing_data['document_conversion'] = round(time.time() - conversion_start, 3)
            
            # Prepare output directory structure using ORIGINAL filename
            # Use exact original PDF filename (without extension) as folder name, replace spaces with underscores
            pdf_folder_name = original_stem.replace(" ", "_")
            pdf_output_dir = Path(output_base) / pdf_folder_name
            images_dir = pdf_output_dir / "images"
            
            # Generate markdown content with enhanced image handling
            markdown_start = time.time()
            md_content = conv_res.document.export_to_markdown(
                image_mode=ImageRefMode.REFERENCED,
                image_placeholder="![{image_name}](images/{image_name})"
            )
            timing_data['markdown_generation'] = round(time.time() - markdown_start, 3)
            
            # Save markdown file using ORIGINAL filename
            save_start = time.time()
            md_filename = f"{original_stem}.md"
            md_path = pdf_output_dir / md_filename
            
            save_success = self.storage.save(
                data=md_content,
                path=md_path,
                format='text',
                ensure_dir=True
            )
            
            if not save_success:
                raise RuntimeError("Failed to save markdown file")
            
            timing_data['markdown_save'] = round(time.time() - save_start, 3)
            
            # Process and save images if enabled
            images_saved = []
            image_descriptions = {}
            
            if config.SAVE_IMAGES and hasattr(conv_res.document, 'pictures'):
                image_processing_start = time.time()
                images_saved, image_descriptions = self._process_and_save_images(
                    conv_res.document.pictures, 
                    images_dir, 
                    pdf_folder_name
                )
                timing_data['image_processing'] = round(time.time() - image_processing_start, 3)
            
            # Enhanced OCR processing on extracted text using our engine
            ocr_enhanced_content = None
            if do_ocr and self.ocr_engine.is_available():
                ocr_enhancement_start = time.time()
                try:
                    # Get raw text from document
                    raw_text = conv_res.document.export_to_text()
                    
                    # Use our OCR engine for additional processing if needed
                    # This could be useful for improving text quality or extracting from images
                    ocr_enhanced_content = self._enhance_text_with_ocr(raw_text, Path(original_filename))
                except Exception as e:
                    self.logger.warning(f"OCR enhancement failed: {e}")
                timing_data['ocr_enhancement'] = round(time.time() - ocr_enhancement_start, 3)
            
            # Calculate total processing time
            total_time = round(time.time() - start_time, 3)
            timing_data['total_processing'] = total_time
            
            # Calculate file size for performance metrics
            file_size_mb = round(pdf_file.stat().st_size / (1024 * 1024), 2) if pdf_file.exists() else 0
            pages_per_second = round(len(conv_res.document.pages) / total_time, 2) if total_time > 0 else 0
            mb_per_second = round(file_size_mb / total_time, 2) if total_time > 0 else 0
            
            # Save comprehensive document metadata
            metadata_start = time.time()
            metadata = {
                'source_file': original_filename,  # Use original filename
                'processing_timestamp': self._get_timestamp(),
                'ocr_enabled': pipeline_options.do_ocr,
                'ocr_engine': type(self.ocr_engine).__name__,
                'storage_backend': type(self.storage).__name__,
                'output_directory': str(pdf_output_dir),
                'folder_name': pdf_folder_name,
                'markdown_file': md_filename,
                'page_count': len(conv_res.document.pages) if conv_res.document.pages else 0,
                'images_generated': pipeline_options.generate_picture_images,
                'images_saved': len(images_saved),
                'image_files': images_saved,
                'image_descriptions': image_descriptions,
                'tables_extracted': pipeline_options.do_table_structure,
                'formulas_extracted': pipeline_options.do_formula_enrichment,
                'figures_extracted': config.EXTRACT_FIGURES,
                'processing_options': {
                    'image_format': config.IMAGE_FORMAT,
                    'image_quality': config.IMAGE_QUALITY,
                    'image_dpi': config.IMAGE_DPI,
                    'image_scale': config.IMAGE_SCALE,
                    'extract_tables': config.EXTRACT_TABLES,
                    'extract_formulas': config.EXTRACT_FORMULAS,
                    'extract_figures': config.EXTRACT_FIGURES,
                    'force_full_page_ocr': config.FORCE_FULL_PAGE_OCR,
                    **kwargs
                },
                'document_structure': {
                    'has_tables': bool(getattr(conv_res.document, 'tables', [])),
                    'has_figures': bool(getattr(conv_res.document, 'pictures', [])),
                    'has_formulas': pipeline_options.do_formula_enrichment,
                    'text_length': len(md_content),
                    'images_count': len(images_saved)
                },
                # Performance and timing data
                'performance_metrics': {
                    'file_size_mb': file_size_mb,
                    'processing_time_seconds': total_time,
                    'pages_per_second': pages_per_second,
                    'mb_per_second': mb_per_second,
                    'timing_breakdown': timing_data  # Always include timing data
                }
            }
            
            # Add OCR enhanced content to metadata if available
            if ocr_enhanced_content:
                metadata['ocr_enhanced_available'] = True
                # Optionally save OCR enhanced content as separate file
                ocr_path = pdf_output_dir / f"{original_stem}_ocr_enhanced.txt"
                self.storage.save(
                    data=ocr_enhanced_content,
                    path=ocr_path,
                    format='text'
                )
                metadata['ocr_enhanced_file'] = f"{original_stem}_ocr_enhanced.txt"
            
            # Save comprehensive metadata using ORIGINAL filename
            metadata_path = pdf_output_dir / f"{original_stem}_metadata.json"
            metadata_success = self.storage.save(
                data=metadata,
                path=metadata_path,
                format='json'
            )
            
            timing_data['metadata_save'] = round(time.time() - metadata_start, 3)
            
            # Update total time with metadata save
            total_time = round(time.time() - start_time, 3)
            timing_data['total_processing'] = total_time
            
            if metadata_success:
                self.logger.info(f"✅ Successfully processed {original_filename} in {total_time}s")
                # Log performance stats (always enabled)
                self.logger.info(f"   📊 Performance: {pages_per_second} pages/sec, {mb_per_second} MB/sec")
                self.logger.info(f"   ⏱️  Timing breakdown: {timing_data}")
            else:
                self.logger.warning(f"Failed to save metadata for {original_filename}")
            
            return {
                'status': 'success',
                'markdown_saved': save_success,
                'metadata_saved': metadata_success,
                'output_path': str(pdf_output_dir),
                'images_saved': len(images_saved),
                'processing_time': total_time,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {original_filename}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'source_file': original_filename,
                'processing_time': round(time.time() - start_time, 3) if 'start_time' in locals() else 0
            }
        finally:
            # Clean up memory after processing
            self._cleanup_memory()

    def parse(self, 
              filepath: Union[str, Path], 
              output_base: Optional[Union[str, Path]] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Parse a PDF file and extract relevant information.
        
        Args:
            filepath: The path to the PDF file to parse
            output_base: The base directory for output files (default: "./data/processed")
            **kwargs: Additional processing options
                - force_ocr: Force OCR even if document is not classified as scanned
                - skip_classification: Skip document classification and use force_ocr
                
        Returns:
            Dict containing processing results and metadata
        """
        # Convert to Path objects
        pdf_path = Path(filepath)
        if output_base is None:
            output_base = Path("./data/processed")
        else:
            output_base = Path(output_base)
        
        # Validate input
        if not pdf_path.exists():
            error_msg = f"PDF file not found: {filepath}"
            self.logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'source_file': str(pdf_path)
            }
        
        if pdf_path.suffix.lower() != '.pdf':
            error_msg = f"File is not a PDF: {filepath}"
            self.logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'source_file': str(pdf_path)
            }
        
        try:
            # Determine if OCR is needed
            force_ocr = kwargs.get('force_ocr', False)
            skip_classification = kwargs.get('skip_classification', False)
            
            if skip_classification:
                use_ocr = force_ocr
            else:
                is_scanned = self.classify_document(pdf_path)
                use_ocr = is_scanned or force_ocr
            
            self.logger.info(f"Processing {pdf_path.name}: OCR={'enabled' if use_ocr else 'disabled'}")
            
            # Process the document
            result = self._pdf_parser(pdf_path, output_base, use_ocr, **kwargs)
            
            # Add classification result to metadata
            if result['status'] == 'success' and 'metadata' in result:
                result['metadata']['document_classified_as_scanned'] = not skip_classification and self.classify_document(pdf_path)
                result['metadata']['ocr_forced'] = force_ocr
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to parse PDF {pdf_path.name}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'source_file': str(pdf_path)
            }

    def classify_document(self, filepath: Union[str, Path]) -> bool:
        """
        Classify whether a PDF is a scanned document requiring OCR.
        
        Uses heuristic analysis to determine if a PDF contains primarily
        scanned images or extractable text.
        
        Args:
            filepath: Path to the PDF file
            
        Returns:
            bool: True if document appears to be scanned, False otherwise
        """
        pdf_file = Path(filepath)
        
        if not pdf_file.exists() or pdf_file.suffix.lower() != ".pdf":
            self.logger.error(f"Invalid PDF file: {filepath}")
            raise FileNotFoundError(f"Invalid PDF file: {filepath}")

        try:
            with pdfplumber.open(pdf_file) as pdf:
                total_pages = len(pdf.pages)
                
                if total_pages == 0:
                    self.logger.info(f"Empty PDF classified as scanned: {pdf_file.name}")
                    return True  # Empty PDFs considered scanned

                scanned_pages = 0
                checked_pages = min(10, total_pages)  # Sample up to 10 pages
                step = max(1, total_pages // checked_pages)

                for i in range(0, total_pages, step):
                    page = pdf.pages[i]
                    text = page.extract_text() or ""
                    text = re.sub(r"\s+", " ", text).strip()

                    # Heuristic analysis
                    text_chars = len(text)
                    img_count = len(page.images or [])
                    page_area = float(page.width) * float(page.height)
                    
                    # Calculate image coverage area
                    img_area = sum(
                        (float(im.get("x1", 0)) - float(im.get("x0", 0))) *
                        (float(im.get("bottom", 0)) - float(im.get("top", 0)))
                        for im in page.images or []
                    )
                    img_ratio = img_area / page_area if page_area > 0 else 0

                    # Classification logic
                    if (text_chars < 50 and img_ratio > 0.2) or (img_ratio > 0.5):
                        scanned_pages += 1

                # Final classification
                scanned_ratio = scanned_pages / checked_pages
                is_scanned = scanned_ratio >= 0.6
                
                self.logger.info(f"Document classification for {pdf_file.name}: "
                               f"{'SCANNED' if is_scanned else 'TEXT-BASED'} "
                               f"({scanned_pages}/{checked_pages} pages appear scanned)")
                
                return is_scanned
                
        except Exception as e:
            self.logger.error(f"Error classifying document {pdf_file.name}: {e}")
            # Default to scanned if classification fails
            return True

    def batch_parse(self, 
                   directory: Union[str, Path],
                   output_base: Optional[Union[str, Path]] = None,
                   pattern: str = "*.pdf",
                   **kwargs) -> Dict[str, Any]:
        """
        Parse multiple PDF files in a directory (supports both local and S3 paths with parallel processing).
        
        Args:
            directory: Directory containing PDF files (local path or S3 prefix)
            output_base: Base directory for output files (local path or S3 prefix)
            pattern: Glob pattern for file matching (default: "*.pdf")
            **kwargs: Additional processing options
                - skip_processed: Skip files that are already processed (default: True)
                - input_storage_type: Override INPUT_STORAGE config ('local' or 's3')
                - parallel: Enable/disable parallel processing (default: True)
                - max_workers: Override MAX_WORKERS config
            
        Returns:
            Dict containing batch processing results
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
        import os
        
        batch_start_time = time.time()
        
        try:
            # Determine input storage type
            input_storage_type = kwargs.get('input_storage_type', config.INPUT_STORAGE).upper()
            is_s3_input = input_storage_type == 'S3'
            
            # Parallel processing configuration (hardcoded defaults)
            enable_parallel = kwargs.get('parallel', True)  # Default: enabled
            max_workers = kwargs.get('max_workers', 4)  # Default: 4 workers
            if max_workers == 0:
                max_workers = min(32, (os.cpu_count() or 1) + 4)  # Default from ThreadPoolExecutor
            
            self.logger.info(f"🚀 Starting batch processing with parallel={'ON' if enable_parallel else 'OFF'}")
            self.logger.info(f"📊 Config: Input={input_storage_type}, Output={config.OUTPUT_STORAGE}, Workers={max_workers if enable_parallel else 1}")
            
            # Create input storage backend if different from output storage
            input_storage = None
            if is_s3_input and not isinstance(self.storage, S3Storage):
                # Need S3 storage for input but have local storage for output
                try:
                    input_storage = S3Storage()
                    self.logger.info("Created separate S3 storage backend for input")
                except Exception as e:
                    self.logger.error(f"Failed to create S3 storage for input: {e}")
                    return {'status': 'error', 'error': f'Failed to create S3 storage for input: {e}'}
            elif not is_s3_input and isinstance(self.storage, S3Storage):
                # Need local storage for input but have S3 storage for output
                input_storage = LocalStorage()
                self.logger.info("Created separate local storage backend for input")
            else:
                # Input and output use same storage type
                input_storage = self.storage
            
            if is_s3_input:
                # Handle S3 input
                pdf_files = self._list_s3_files_with_storage(directory, pattern, input_storage)
                if not pdf_files:
                    self.logger.warning(f"No PDF files found matching pattern '{pattern}' in S3: {directory}")
                    return {
                        'status': 'success',
                        'processed': 0,
                        'successful': 0,
                        'failed': 0,
                        'skipped': 0,
                        'results': [],
                        'message': 'No files found',
                        'processing_time': 0
                    }
            else:
                # Handle local input
                directory_path = Path(directory)
                if not directory_path.exists() or not directory_path.is_dir():
                    error_msg = f"Invalid local directory: {directory}"
                    self.logger.error(error_msg)
                    return {'status': 'error', 'error': error_msg}
                
                pdf_files = [str(f) for f in directory_path.glob(pattern)]
                if not pdf_files:
                    self.logger.warning(f"No PDF files found matching pattern '{pattern}' in {directory}")
                    return {
                        'status': 'success',
                        'processed': 0,
                        'successful': 0,
                        'failed': 0,
                        'skipped': 0,
                        'results': [],
                        'message': 'No files found',
                        'processing_time': 0
                    }
            
            self.logger.info(f"📁 Found {len(pdf_files)} PDF files to process")
            
            # Set default output base
            if output_base is None:
                output_base = Path("./data/processed") if config.OUTPUT_STORAGE.upper() == 'LOCAL' else "data/processed"
            
            # Process files
            results = []
            successful = 0
            failed = 0
            skipped = 0
            skip_processed = kwargs.get('skip_processed', True)
            
            if enable_parallel and len(pdf_files) > 1:
                # Parallel processing
                self.logger.info(f"🔄 Processing {len(pdf_files)} files in parallel with {max_workers} workers")
                
                # Choose executor based on hardcoded default (thread)
                parallel_method = kwargs.get('parallel_method', 'thread')
                if parallel_method == 'process':
                    executor_class = ProcessPoolExecutor
                    self.logger.info("Using ProcessPoolExecutor for parallel processing")
                else:
                    executor_class = ThreadPoolExecutor
                    self.logger.info("Using ThreadPoolExecutor for parallel processing")
                
                # Process files in batches to manage memory
                batch_size = kwargs.get('batch_size', 10)  # Default: 10 files per batch
                batches = [pdf_files[i:i+batch_size] for i in range(0, len(pdf_files), batch_size)]
                
                for batch_num, batch_files in enumerate(batches, 1):
                    self.logger.info(f"📦 Processing batch {batch_num}/{len(batches)} ({len(batch_files)} files)")
                    
                    with executor_class(max_workers=max_workers) as executor:
                        # Submit tasks
                        future_to_file = {}
                        
                        for pdf_file_path in batch_files:
                            pdf_file = Path(pdf_file_path)
                            
                            # Check if already processed (if skip_processed is enabled)
                            if skip_processed and self._is_already_processed(pdf_file, output_base):
                                self.logger.info(f"⏭️  Skipping already processed: {pdf_file.name}")
                                skipped += 1
                                results.append({
                                    'status': 'skipped',
                                    'source_file': str(pdf_file),
                                    'message': 'Already processed'
                                })
                                continue
                            
                            # Submit parsing task
                            if is_s3_input:
                                future = executor.submit(self._parse_s3_file_wrapper, pdf_file_path, output_base, input_storage, **kwargs)
                            else:
                                future = executor.submit(self._parse_file_wrapper, pdf_file_path, output_base, **kwargs)
                            
                            future_to_file[future] = pdf_file_path
                        
                        # Collect results as they complete
                        for future in as_completed(future_to_file):
                            pdf_file_path = future_to_file[future]
                            pdf_name = Path(pdf_file_path).name
                            
                            try:
                                result = future.result()
                                results.append(result)
                                
                                if result['status'] == 'success':
                                    successful += 1
                                    processing_time = result.get('processing_time', 0)
                                    self.logger.info(f"✅ {pdf_name} ({processing_time}s)")
                                    
                                    # Log performance stats periodically (every 5 files)
                                    if successful % 5 == 0:
                                        avg_time = sum(r.get('processing_time', 0) for r in results if r.get('processing_time')) / successful
                                        self.logger.info(f"📊 Progress: {successful}/{len(pdf_files)} completed, avg {avg_time:.2f}s per file")
                                else:
                                    failed += 1
                                    self.logger.error(f"❌ {pdf_name}: {result.get('error', 'Unknown error')}")
                                    
                            except Exception as e:
                                failed += 1
                                error_result = {
                                    'status': 'error',
                                    'error': str(e),
                                    'source_file': str(pdf_file_path)
                                }
                                results.append(error_result)
                                self.logger.error(f"❌ {pdf_name}: {e}")
            
            else:
                # Sequential processing
                self.logger.info(f"🔄 Processing {len(pdf_files)} files sequentially")
                
                for i, pdf_file_path in enumerate(pdf_files, 1):
                    pdf_file = Path(pdf_file_path)
                    
                    # Check if already processed (if skip_processed is enabled)
                    if skip_processed and self._is_already_processed(pdf_file, output_base):
                        self.logger.info(f"⏭️  ({i}/{len(pdf_files)}) Skipping already processed: {pdf_file.name}")
                        skipped += 1
                        results.append({
                            'status': 'skipped',
                            'source_file': str(pdf_file),
                            'message': 'Already processed'
                        })
                        continue
                    
                    self.logger.info(f"🔄 ({i}/{len(pdf_files)}) Processing: {pdf_file.name}")
                    
                    try:
                        # Parse the document
                        if is_s3_input:
                            result = self._parse_s3_file(pdf_file_path, output_base, input_storage, **kwargs)
                        else:
                            result = self.parse(pdf_file_path, output_base, **kwargs)
                        
                        results.append(result)
                        
                        if result['status'] == 'success':
                            successful += 1
                            processing_time = result.get('processing_time', 0)
                            self.logger.info(f"✅ Successfully processed: {pdf_file.name} ({processing_time}s)")
                            
                            # Log performance stats periodically (every 5 files)
                            if successful % 5 == 0:
                                avg_time = sum(r.get('processing_time', 0) for r in results if r.get('processing_time')) / successful
                                self.logger.info(f"📊 Progress: {successful}/{len(pdf_files)} completed, avg {avg_time:.2f}s per file")
                        else:
                            failed += 1
                            self.logger.error(f"❌ Processing failed for {pdf_file.name}: {result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        error_result = {
                            'status': 'error',
                            'error': str(e),
                            'source_file': str(pdf_file_path)
                        }
                        results.append(error_result)
                        failed += 1
                        self.logger.error(f"❌ Error processing {pdf_file.name}: {e}")
            
            # Calculate batch processing metrics
            total_batch_time = round(time.time() - batch_start_time, 3)
            total_processing_time = sum(r.get('processing_time', 0) for r in results if r.get('processing_time'))
            efficiency = round((total_processing_time / total_batch_time) * 100, 1) if total_batch_time > 0 else 0
            
            self.logger.info(f"🏁 Batch processing complete in {total_batch_time}s")
            self.logger.info(f"📊 Results: {successful} successful, {failed} failed, {skipped} skipped")
            if enable_parallel:
                self.logger.info(f"⚡ Efficiency: {efficiency}% (parallel speedup)")
            
            return {
                'status': 'success',
                'processed': len(pdf_files),
                'successful': successful,
                'failed': failed,
                'skipped': skipped,
                'results': results,
                'processing_time': total_batch_time,
                'parallel_enabled': enable_parallel,
                'workers_used': max_workers if enable_parallel else 1,
                'batch_metrics': {
                    'total_batch_time': total_batch_time,
                    'total_processing_time': total_processing_time,
                    'efficiency_percent': efficiency,
                    'average_file_time': round(total_processing_time / max(1, successful), 3)
                }
            }
            
        except Exception as e:
            error_msg = f"Batch processing failed: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'error', 'error': error_msg}

    def _check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': round(memory_info.rss / 1024 / 1024, 2),
                'vms_mb': round(memory_info.vms / 1024 / 1024, 2),
                'percent': round(process.memory_percent(), 2)
            }
        except Exception:
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}

    def _cleanup_memory(self):
        """Force garbage collection to free memory."""
        try:
            gc.collect()
            # Always log memory stats in debug mode
            memory = self._check_memory_usage()
            self.logger.debug(f"🧹 Memory after cleanup: {memory['rss_mb']} MB ({memory['percent']}%)")
        except Exception as e:
            self.logger.debug(f"Memory cleanup failed: {e}")

    def _should_reduce_batch_size(self, current_memory: Dict[str, float]) -> bool:
        """Check if batch size should be reduced due to memory pressure."""
        memory_limit_mb = 2048  # Default: 2GB limit
        if current_memory['rss_mb'] > memory_limit_mb:
            return True
        if current_memory['percent'] > 80:  # More than 80% memory usage
            return True
        return False

    def _parse_file_wrapper(self, pdf_file_path: str, output_base: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Wrapper for parse method to use in parallel processing."""
        try:
            return self.parse(pdf_file_path, output_base, **kwargs)
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'source_file': str(pdf_file_path)
            }

    def _parse_s3_file_wrapper(self, s3_file_path: str, output_base: Union[str, Path], input_storage: Storage, **kwargs) -> Dict[str, Any]:
        """Wrapper for _parse_s3_file method to use in parallel processing."""
        try:
            return self._parse_s3_file(s3_file_path, output_base, input_storage, **kwargs)
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'source_file': str(s3_file_path)
            }

    def _list_s3_files_with_storage(self, s3_path: Union[str, Path], pattern: str = "*.pdf", storage_backend: Storage = None) -> list[str]:
        """List files in S3 bucket matching the pattern using specified storage backend."""
        try:
            if storage_backend is None:
                storage_backend = self.storage
                
            if isinstance(storage_backend, S3Storage):
                # Use the S3 storage backend to list files
                files = storage_backend.list_files(s3_path, pattern)
                # Convert to full paths and filter for PDFs
                return [str(f) for f in files if str(f).lower().endswith('.pdf')]
            else:
                self.logger.warning("S3 path provided but storage backend is not S3Storage")
                return []
        except Exception as e:
            self.logger.error(f"Error listing S3 files: {e}")
            return []

    def _parse_s3_file(self, s3_file_path: str, output_base: Union[str, Path], input_storage: Storage, **kwargs) -> Dict[str, Any]:
        """Parse a PDF file stored in S3."""
        try:
            import tempfile
            
            # Extract the original filename from S3 path
            original_filename = Path(s3_file_path).name
            
            # Create a temporary file to download the S3 file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                temp_path = Path(tmp_file.name)
                
            try:
                # Download file from S3 to temp location
                s3_data = input_storage.load(s3_file_path, format='binary')
                
                # Write to temporary file
                with open(temp_path, 'wb') as f:
                    f.write(s3_data)
                
                self.logger.info(f"Downloaded S3 file {s3_file_path} to temporary location for processing")
                
                # Create a new Path object with the original filename for processing
                # This ensures the folder structure uses the original PDF name, not the temp name
                original_path = temp_path.parent / original_filename
                
                # Process the temporary file but use custom kwargs to preserve original naming
                processing_kwargs = {**kwargs}
                # Remove original_filename from kwargs to avoid duplicate parameter error
                processing_kwargs.pop('original_filename', None)
                processing_kwargs.pop('original_s3_path', None)
                
                # Determine if OCR is needed (same logic as regular parse method)
                force_ocr = kwargs.get('force_ocr', False)
                skip_classification = kwargs.get('skip_classification', False)
                
                if skip_classification:
                    use_ocr = force_ocr
                else:
                    # Classify using the temporary file
                    is_scanned = self.classify_document(temp_path)
                    use_ocr = is_scanned or force_ocr
                
                result = self._pdf_parser_with_original_name(temp_path, Path(output_base), original_filename, use_ocr, **processing_kwargs)
                
                # Update the source file reference to the original S3 path
                if 'metadata' in result:
                    result['metadata']['source_file'] = s3_file_path
                    result['metadata']['original_location'] = 'S3'
                    result['metadata']['temp_file_used'] = str(temp_path)
                
                return result
                
            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()
                    self.logger.debug(f"Cleaned up temporary file: {temp_path}")
                    
        except Exception as e:
            error_msg = f"Failed to process S3 file {s3_file_path}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'source_file': s3_file_path
            }

    def _list_s3_files(self, s3_path: Union[str, Path], pattern: str = "*.pdf") -> list[str]:
        """List files in S3 bucket matching the pattern."""
        try:
            if isinstance(self.storage, S3Storage):
                # Use the S3 storage backend to list files
                files = self.storage.list_files(s3_path, pattern)
                # Convert to full S3 paths or keep as relative paths based on how storage works
                return [str(f) for f in files if str(f).lower().endswith('.pdf')]
            else:
                self.logger.warning("S3 path provided but storage backend is not S3Storage")
                return []
        except Exception as e:
            self.logger.error(f"Error listing S3 files: {e}")
            return []

    def _is_already_processed(self, pdf_file: Path, output_base: Union[str, Path]) -> bool:
        """Check if a PDF file has already been processed."""
        try:
            # Use same naming convention as _pdf_parser
            pdf_folder_name = pdf_file.stem.replace(" ", "_")
            pdf_output_dir = Path(output_base) / pdf_folder_name
            
            # Check for markdown file and metadata file
            md_file = pdf_output_dir / f"{pdf_file.stem}.md"
            metadata_file = pdf_output_dir / f"{pdf_file.stem}_metadata.json"
            
            # Check if both files exist using storage backend
            md_exists = self.storage.exists(md_file)
            metadata_exists = self.storage.exists(metadata_file)
            
            return md_exists and metadata_exists
        except Exception as e:
            self.logger.debug(f"Error checking if file is processed: {e}")
            return False

    def get_storage_backend(self) -> Storage:
        """Get the current storage backend."""
        return self.storage
    
    def get_ocr_engine(self) -> OCREngine:
        """Get the current OCR engine."""
        return self.ocr_engine
    
    def switch_storage_backend(self, storage: Storage) -> None:
        """Switch to a different storage backend."""
        old_storage = type(self.storage).__name__
        self.storage = storage
        new_storage = type(self.storage).__name__
        self.logger.info(f"Switched storage backend: {old_storage} -> {new_storage}")
    
    def switch_ocr_engine(self, engine_name: str, **kwargs) -> None:
        """Switch to a different OCR engine."""
        old_engine = type(self.ocr_engine).__name__
        self.ocr_engine = OCREngineFactory.create_engine(engine_name, **kwargs)
        
        if not self.ocr_engine.is_available():
            raise RuntimeError(f"OCR engine '{engine_name}' is not available")
        
        new_engine = type(self.ocr_engine).__name__
        self.logger.info(f"Switched OCR engine: {old_engine} -> {new_engine}")

    def _process_and_save_images(self, pictures, images_dir: Path, pdf_stem: str) -> tuple[list, dict]:
        """
        Process and save extracted images with descriptions.
        
        Args:
            pictures: Document pictures from docling
            images_dir: Directory to save images
            pdf_stem: PDF filename stem for naming
            
        Returns:
            Tuple of (saved_image_files, image_descriptions)
        """
        saved_images = []
        descriptions = {}
        
        try:
            if not pictures:
                return saved_images, descriptions
            
            for i, picture in enumerate(pictures):
                try:
                    # Generate image filename
                    image_name = f"{pdf_stem}_image_{i+1:03d}.{config.IMAGE_FORMAT.lower()}"
                    image_path = images_dir / image_name
                    
                    # Extract actual image data from docling ImageRef
                    image_data = None
                    if hasattr(picture, 'image') and picture.image:
                        # Check if it's an ImageRef object from docling
                        if hasattr(picture.image, 'pil_image') and picture.image.pil_image:
                            # Convert PIL image to bytes with comprehensive error handling
                            import io
                            buffer = io.BytesIO()
                            
                            # Try multiple fallback strategies for PIL image processing
                            success = False
                            pil_img = picture.image.pil_image
                            
                            # Strategy 1: Try to load and copy the image safely
                            try:
                                # Verify the image is valid before processing
                                if hasattr(pil_img, 'verify'):
                                    # Create a copy for verification (verify destroys the image)
                                    verify_img = pil_img.copy()
                                    verify_img.verify()
                                
                                # Force load the image data to resolve any lazy loading
                                pil_img.load()
                                # Create a copy to avoid tile issues
                                safe_img = pil_img.copy()
                                safe_img.save(buffer, format='PNG')
                                image_data = buffer.getvalue()
                                success = True
                                
                            except Exception as img_error:
                                self.logger.warning(f"PIL image copy/load failed: {img_error}, trying format conversion")
                                
                                # Strategy 2: Try converting to RGB without copy
                                try:
                                    buffer = io.BytesIO()  # Reset buffer
                                    rgb_img = pil_img.convert('RGB')
                                    rgb_img.save(buffer, format='PNG')
                                    image_data = buffer.getvalue()
                                    success = True
                                    
                                except Exception as img_error2:
                                    self.logger.warning(f"RGB conversion failed: {img_error2}, trying direct save")
                                    
                                    # Strategy 3: Try direct save without any preprocessing
                                    try:
                                        buffer = io.BytesIO()  # Reset buffer
                                        pil_img.save(buffer, format='PNG')
                                        image_data = buffer.getvalue()
                                        success = True
                                        
                                    except Exception as img_error3:
                                        self.logger.warning(f"Direct save failed: {img_error3}, trying bitmap fallback")
                                        
                                        # Strategy 4: Try creating a new image from getdata()
                                        try:
                                            from PIL import Image
                                            buffer = io.BytesIO()  # Reset buffer
                                            
                                            # Get image properties
                                            size = pil_img.size
                                            mode = pil_img.mode if pil_img.mode in ['RGB', 'RGBA', 'L'] else 'RGB'
                                            
                                            # Try to get pixel data safely
                                            if hasattr(pil_img, 'getdata'):
                                                pixel_data = list(pil_img.getdata())
                                                new_img = Image.new(mode, size)
                                                new_img.putdata(pixel_data)
                                                new_img.save(buffer, format='PNG')
                                                image_data = buffer.getvalue()
                                                success = True
                                            else:
                                                # Last resort: create blank image with size info
                                                self.logger.warning(f"Creating placeholder image for corrupted image {i}")
                                                placeholder_img = Image.new('RGB', size, color='white')
                                                placeholder_img.save(buffer, format='PNG')
                                                image_data = buffer.getvalue()
                                                success = True
                                                
                                        except Exception as img_error4:
                                            self.logger.error(f"All image conversion strategies failed: {img_error4}")
                                            success = False
                            
                            if not success:
                                self.logger.error(f"Cannot process PIL image {i}, skipping")
                                continue
                        elif hasattr(picture.image, 'bytes') and picture.image.bytes:
                            # Direct bytes access
                            image_data = picture.image.bytes
                        elif hasattr(picture.image, 'get_bytes'):
                            # Method to get bytes
                            image_data = picture.image.get_bytes()
                        else:
                            # Try to convert the object to bytes
                            self.logger.debug(f"Image object type: {type(picture.image)}")
                            # Skip this image if we can't extract data
                            self.logger.warning(f"Cannot extract image data from {type(picture.image)}")
                            continue
                    
                    if image_data:
                        # Convert image format if needed
                        processed_image_data = self._convert_image_format(image_data, config.IMAGE_FORMAT, config.IMAGE_QUALITY)
                        
                        success = self.storage.save(
                            data=processed_image_data,
                            path=image_path,
                            format='binary',
                            ensure_dir=True
                        )
                        
                        if success:
                            saved_images.append(image_name)
                            
                            # Generate description if enabled
                            if config.GENERATE_IMAGE_DESCRIPTIONS:
                                description = self._generate_image_description(picture, processed_image_data)
                                if description:
                                    descriptions[image_name] = description
                                    
                            self.logger.info(f"Saved image: {image_name}")
                        else:
                            self.logger.warning(f"Failed to save image: {image_name}")
                    else:
                        self.logger.warning(f"No image data found for picture {i}")
                            
                except Exception as e:
                    self.logger.error(f"Error processing image {i}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error in image processing: {e}")
            
        return saved_images, descriptions
    
    def _convert_image_format(self, image_data: bytes, target_format: str, quality: int = 95) -> bytes:
        """
        Convert image to specified format and quality.
        
        Args:
            image_data: Original image bytes
            target_format: Target format (PNG, JPEG, WEBP)
            quality: Image quality for lossy formats
            
        Returns:
            Converted image bytes
        """
        try:
            from PIL import Image
            import io
            
            # Ensure we have bytes data
            if not isinstance(image_data, bytes):
                raise ValueError(f"Expected bytes, got {type(image_data)}")
            
            # Load image from bytes with comprehensive error handling
            try:
                # Try to open the image
                image = Image.open(io.BytesIO(image_data))
                
                # Try to load the image data safely with multiple strategies
                try:
                    # Strategy 1: Direct load
                    image.load()
                except Exception as load_error:
                    self.logger.debug(f"Direct image load failed: {load_error}, trying verification")
                    try:
                        # Strategy 2: Verify and reload
                        # Make a copy first since verify destroys the image
                        verify_copy = image.copy()
                        verify_copy.verify()
                        # Reload original image
                        image = Image.open(io.BytesIO(image_data))
                        image.load()
                    except Exception as verify_error:
                        self.logger.debug(f"Image verification failed: {verify_error}, using image as-is")
                        # Continue with the image even if load/verify failed
                        pass
                        
            except Exception as open_error:
                self.logger.warning(f"Image opening failed: {open_error}")
                return image_data  # Return original if we can't open it
            
            # Convert to RGB if necessary for JPEG with safe error handling
            if target_format.upper() == 'JPEG' and image.mode in ('RGBA', 'LA', 'P'):
                try:
                    # Create white background for transparent images
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    if image.mode == 'P':
                        # Safe palette conversion
                        try:
                            image = image.convert('RGBA')
                        except Exception as palette_error:
                            self.logger.debug(f"Palette conversion failed: {palette_error}, trying direct RGB")
                            image = image.convert('RGB')
                            background = image  # Skip masking if palette conversion fails
                    
                    if image.mode == 'RGBA':
                        try:
                            # Safe alpha channel handling
                            alpha_mask = image.split()[-1] if len(image.split()) > 3 else None
                            background.paste(image, mask=alpha_mask)
                        except Exception as paste_error:
                            self.logger.debug(f"Alpha paste failed: {paste_error}, using direct conversion")
                            background = image.convert('RGB')
                    
                    image = background
                    
                except Exception as rgb_convert_error:
                    self.logger.warning(f"RGB conversion failed: {rgb_convert_error}, keeping original format")
                    # Keep original image and let the save operation handle it
                    pass
            
            # Save to bytes with specified format and quality
            output_buffer = io.BytesIO()
            save_kwargs = {'format': target_format.upper()}
            
            if target_format.upper() in ['JPEG', 'WEBP']:
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = True
            elif target_format.upper() == 'PNG':
                save_kwargs['optimize'] = True
                
            try:
                image.save(output_buffer, **save_kwargs)
                return output_buffer.getvalue()
                
            except Exception as save_error:
                self.logger.warning(f"Image save with format {target_format} failed: {save_error}, trying fallbacks")
                
                # Fallback 1: PNG without optimization
                try:
                    output_buffer = io.BytesIO()
                    image.save(output_buffer, format='PNG')
                    return output_buffer.getvalue()
                    
                except Exception as png_error:
                    self.logger.warning(f"PNG save failed: {png_error}, trying copy-based save")
                    
                    # Fallback 2: Create a copy and save
                    try:
                        output_buffer = io.BytesIO()
                        safe_image = image.copy()
                        safe_image.save(output_buffer, format='PNG')
                        return output_buffer.getvalue()
                        
                    except Exception as copy_error:
                        self.logger.warning(f"Copy-based save failed: {copy_error}, trying pixel reconstruction")
                        
                        # Fallback 3: Reconstruct image from pixel data
                        try:
                            output_buffer = io.BytesIO()
                            # Get basic image info
                            size = image.size
                            mode = image.mode if image.mode in ['RGB', 'RGBA', 'L'] else 'RGB'
                            
                            # Create new image from pixel data
                            pixel_data = list(image.getdata())
                            new_image = Image.new(mode, size)
                            new_image.putdata(pixel_data)
                            new_image.save(output_buffer, format='PNG')
                            return output_buffer.getvalue()
                            
                        except Exception as reconstruct_error:
                            self.logger.error(f"All image save strategies failed: {reconstruct_error}")
                            # Return original data as last resort
                            return image_data
            
        except Exception as e:
            self.logger.warning(f"Image format conversion failed: {e}, using original")
            return image_data
    
    def _generate_image_description(self, picture, image_data: bytes) -> str:
        """
        Generate description for an image using available methods.
        
        Args:
            picture: Picture object from docling
            image_data: Image data bytes
            
        Returns:
            Image description string
        """
        description = ""
        
        try:
            # Try to get description from docling first
            if hasattr(picture, 'caption') and picture.caption:
                description = picture.caption
            elif hasattr(picture, 'text') and picture.text:
                description = picture.text
                
            # If no description and OCR engine available, try OCR on image
            if not description and self.ocr_engine.is_available():
                try:
                    # Use OCR engine to extract text from image
                    ocr_result = self.ocr_engine.extract_text_from_bytes(image_data)
                    if ocr_result and ocr_result.text.strip():
                        description = f"OCR extracted text: {ocr_result.text.strip()}"
                except Exception as e:
                    self.logger.debug(f"OCR on image failed: {e}")
            
            # Add metadata if available
            metadata_parts = []
            if hasattr(picture, 'bbox') and picture.bbox:
                metadata_parts.append(f"Position: {picture.bbox}")
            if hasattr(picture, 'page_no') and picture.page_no is not None:
                metadata_parts.append(f"Page: {picture.page_no + 1}")
                
            if metadata_parts:
                description = f"{description} [{', '.join(metadata_parts)}]" if description else f"[{', '.join(metadata_parts)}]"
                
        except Exception as e:
            self.logger.error(f"Error generating image description: {e}")
            
        return description or "Image extracted from PDF"
    
    def _enhance_text_with_ocr(self, text: str, pdf_file: Path) -> str:
        """
        Enhance extracted text using OCR engine for better accuracy.
        
        Args:
            text: Original extracted text
            pdf_file: Path to PDF file
            
        Returns:
            Enhanced text string
        """
        try:
            if not self.ocr_engine.is_available():
                return text
            
            # This is a placeholder for advanced OCR enhancement
            # In practice, you might want to:
            # 1. Extract pages as images
            # 2. Run OCR on images
            # 3. Compare and merge results with original text
            
            enhanced_text = f"=== OCR ENHANCED VERSION ===\n\n{text}\n\n=== OCR ENGINE: {type(self.ocr_engine).__name__} ==="
            
            # Here you could implement more sophisticated enhancement
            # For now, we'll just return the original text with OCR metadata
            return enhanced_text
            
        except Exception as e:
            self.logger.error(f"Text enhancement with OCR failed: {e}")
            return text

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        return re.sub(r'[^\w\-_.]', '_', filename).lower()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()


# Factory function for easy parser creation
def create_parser(storage_type: str = "local", 
                 ocr_engine: str = "tesseract",
                 **kwargs) -> Parser:
    """
    Factory function to create a Parser with specified backends.
    
    Args:
        storage_type: "local" or "s3"
        ocr_engine: "tesseract" or "easyocr"
        **kwargs: Additional configuration options
        
    Returns:
        Parser: Configured parser instance
    """
    # Create storage backend
    if storage_type.lower() == "s3":
        storage = S3Storage(**kwargs)
    else:
        storage = LocalStorage(**kwargs)
    
    # Pass config values as default parameters
    enhanced_kwargs = {
        'save_images': config.SAVE_IMAGES,
        'image_format': config.IMAGE_FORMAT,
        'image_quality': config.IMAGE_QUALITY,
        'image_dpi': config.IMAGE_DPI,
        'image_scale': config.IMAGE_SCALE,
        'generate_descriptions': config.GENERATE_IMAGE_DESCRIPTIONS,
        'extract_tables': config.EXTRACT_TABLES,
        'extract_formulas': config.EXTRACT_FORMULAS,
        'extract_figures': config.EXTRACT_FIGURES,
        'force_full_page_ocr': config.FORCE_FULL_PAGE_OCR,
        **kwargs  # Allow override of defaults
    }
    
    return Parser(storage=storage, ocr_engine_name=ocr_engine, **enhanced_kwargs)
