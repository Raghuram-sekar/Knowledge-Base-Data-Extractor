"""
Simplified PDF Parser with configurable storage and OCR engines.

This module provides a streamlined PDF parsing solution with:
- Automatic storage backend selection from config
- OCR engine selection from config
- Parallel batch processing with configurable workers
- Comprehensive document extraction (text, images, tables, formulas)
"""

import os
import time
import re
import json
import psutil
import gc
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set environment variables to reduce noise and optimize performance
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MallocStackLogging"] = "0"  # Suppress malloc stack logging on macOS
os.environ["MALLOC_CHECK_"] = "0"      # Disable malloc debugging
os.environ["PYTHONHASHSEED"] = "0"     # Make Python hash reproducible
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow warnings

from .storage import Storage, LocalStorage, S3Storage
from .engine import OCREngineFactory, OCREngine
from core import logger
from core.config import config

# Docling imports
from docling_core.types.doc import ImageRefMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


class Parser:
    """Simplified PDF parser with automatic configuration from config file."""

    def __init__(self):
        """Initialize parser with settings from config."""
        self.logger = logger.get_logger(__name__)
        
        # Set OCR environment variables for optimization
        os.environ["TESSDATA_PREFIX"] = config.TESSDATA_PREFIX or "/usr/local/share/tessdata"
        os.environ["OMP_NUM_THREADS"] = str(config.OMP_NUM_THREADS)
        
        # Initialize storage backends from config
        self.input_storage = self._create_input_storage_backend()
        self.output_storage = self._create_output_storage_backend()
        
        # Initialize OCR engine from config
        self.ocr_engine = self._create_ocr_engine()
        
        self.logger.info(f"Parser initialized with input storage: {type(self.input_storage).__name__}, "
                        f"output storage: {type(self.output_storage).__name__}, "
                        f"OCR engine: {type(self.ocr_engine).__name__}")

    def _create_input_storage_backend(self) -> Storage:
        """Create input storage backend based on config."""
        if config.INPUT_STORAGE == "S3":
            return S3Storage(
                bucket_name=config.S3_BUCKET_NAME,
                access_key=config.S3_ACCESS_KEY,
                secret_key=config.S3_SECRET_KEY,
                region=config.S3_REGION,
                endpoint_url=config.S3_ENDPOINT_URL
            )
        else:
            return LocalStorage()
    
    def _create_output_storage_backend(self) -> Storage:
        """Create output storage backend based on config."""
        if config.OUTPUT_STORAGE == "S3":
            return S3Storage(
                bucket_name=config.S3_BUCKET_NAME,
                access_key=config.S3_ACCESS_KEY,
                secret_key=config.S3_SECRET_KEY,
                region=config.S3_REGION,
                endpoint_url=config.S3_ENDPOINT_URL
            )
        else:
            return LocalStorage()
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename while preserving the original name as much as possible."""
        # Only replace spaces with underscores and remove truly invalid characters
        sanitized = filename.replace(' ', '_')
        # Remove only truly problematic characters for filesystem/S3
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove multiple underscores but preserve the original structure
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores or dots only if they exist
        sanitized = sanitized.strip('_.')
        # Ensure it's not empty (fallback only if completely invalid)
        if not sanitized:
            sanitized = "document"
        return sanitized

    def _create_ocr_engine(self) -> OCREngine:
        """Create OCR engine based on config."""
        try:
            engine = OCREngineFactory.create_engine(config.OCR_ENGINE.lower())
            
            if not engine.is_available():
                # Fall back to any available engine
                available_engines = OCREngineFactory.get_available_and_ready_engines()
                if available_engines:
                    engine_name = available_engines[0]
                    self.logger.warning(f"Requested engine {config.OCR_ENGINE} not available, using {engine_name}")
                    engine = OCREngineFactory.create_engine(engine_name)
                else:
                    raise RuntimeError("No OCR engines available")
            
            return engine
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OCR engine: {e}")
            raise

    def parse(self, filepath: Union[str, Path], output_base: Optional[Union[str, Path]] = None, original_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse a single PDF file.
        
        Args:
            filepath: Path to PDF file
            output_base: Output directory (defaults to config.OUTPUT_PATH)
            original_filename: Original filename to use for output (for S3 temp files)
            
        Returns:
            Dict containing parsing results and metadata
        """
        pdf_path = Path(filepath)
        if output_base is None:
            output_base = Path(config.OUTPUT_PATH)
        else:
            output_base = Path(output_base)

        # Validate input
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")

        # Classify document to determine if OCR is needed
        needs_ocr = self._classify_document(pdf_path)
        
        # Process the document
        return self._process_pdf(pdf_path, output_base, needs_ocr, original_filename)

    def _classify_document(self, filepath: Path) -> bool:
        """
        Classify whether a PDF needs OCR.
        Simple heuristic: if we can extract meaningful text, it doesn't need OCR.
        """
        try:
            import pdfplumber
            
            with pdfplumber.open(filepath) as pdf:
                text_length = 0
                pages_checked = min(3, len(pdf.pages))  # Check first 3 pages
                
                for i in range(pages_checked):
                    page_text = pdf.pages[i].extract_text() or ""
                    text_length += len(page_text.strip())
                
                # If we got less than 50 characters per page on average, likely scanned
                avg_text_per_page = text_length / pages_checked if pages_checked > 0 else 0
                needs_ocr = avg_text_per_page < 50
                
                self.logger.debug(f"Document classification: {filepath.name} - "
                                f"Avg text per page: {avg_text_per_page:.1f}, "
                                f"Needs OCR: {needs_ocr}")
                
                return needs_ocr
                
        except Exception as e:
            self.logger.warning(f"Failed to classify document {filepath}: {e}")
            return True  # Default to OCR if classification fails

    def _process_pdf(self, pdf_file: Path, output_base: Path, do_ocr: bool, original_filename: Optional[str] = None) -> Dict[str, Any]:
        """Process a single PDF file with docling."""
        start_time = time.time()
        timing_breakdown = {}
        
        try:
            # Use original filename if provided, otherwise use the current file's name
            display_name = original_filename if original_filename else pdf_file.name
            self.logger.info(f"Processing: {display_name} (OCR: {do_ocr})")
            
            # Get file size for performance metrics
            file_size_mb = pdf_file.stat().st_size / (1024 * 1024)
            
            # Pipeline setup timing
            pipeline_start = time.time()
            
            # Configure docling pipeline
            pipeline_options = PdfPipelineOptions()
            
            # OCR Configuration
            pipeline_options.do_ocr = do_ocr or config.FORCE_FULL_PAGE_OCR
            if pipeline_options.do_ocr:
                # Configure Tesseract with optimized options
                ocr_options = TesseractCliOcrOptions()
                # Use PSM mode from config (default 6 for uniform text blocks)
                ocr_options.psm = config.OCR_PSM_MODE
                # Skip OSD if configured to reduce warnings
                if config.SKIP_OSD:
                    ocr_options.force_full_page_ocr = True
                pipeline_options.ocr_options = ocr_options
            
            # Extraction options from config
            pipeline_options.do_table_structure = config.EXTRACT_TABLES
            pipeline_options.do_formula_enrichment = config.EXTRACT_FORMULAS
            pipeline_options.do_picture_description = config.GENERATE_IMAGE_DESCRIPTIONS
            pipeline_options.generate_picture_images = config.SAVE_IMAGES
            pipeline_options.images_scale = config.IMAGE_SCALE
            
            # Create document converter
            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            timing_breakdown['pipeline_setup'] = time.time() - pipeline_start
            
            # Document conversion timing
            conversion_start = time.time()
            result = doc_converter.convert(pdf_file)
            timing_breakdown['document_conversion'] = time.time() - conversion_start
            
            # Create sanitized folder name based on original filename
            if original_filename:
                # Extract filename without extension from original S3 path
                original_stem = Path(original_filename).stem
            else:
                original_stem = pdf_file.stem
            
            pdf_stem = self._sanitize_filename(original_stem)
            
            # Build output paths
            if config.OUTPUT_STORAGE == "S3":
                # For S3, use forward slashes and no absolute paths
                output_folder = f"{str(output_base).strip('/')}/{pdf_stem}"
                markdown_file_path = f"{output_folder}/{pdf_stem}.md"
                json_file_path = f"{output_folder}/{pdf_stem}.json"
                metadata_file_path = f"{output_folder}/metadata.json"
                images_folder = f"{output_folder}/images"
            else:
                # For local storage
                output_dir = output_base / pdf_stem
                output_dir.mkdir(parents=True, exist_ok=True)
                markdown_file_path = str(output_dir / f"{pdf_stem}.md")
                json_file_path = str(output_dir / f"{pdf_stem}.json")
                metadata_file_path = str(output_dir / "metadata.json")
                images_folder = str(output_dir / "images")
                # Create images directory if local
                if config.SAVE_IMAGES:
                    (output_dir / "images").mkdir(exist_ok=True)
            
            # Markdown generation and save timing
            markdown_start = time.time()
            markdown_content = result.document.export_to_markdown()
            timing_breakdown['markdown_generation'] = time.time() - markdown_start
            
            markdown_save_start = time.time()
            self.output_storage.save(markdown_content, markdown_file_path, format='text')
            timing_breakdown['markdown_save'] = time.time() - markdown_save_start
            
            # Save JSON metadata
            doc_dict = result.document.export_to_dict()
            self.output_storage.save(doc_dict, json_file_path, format='json')
            
            # Process and save images if enabled
            image_processing_start = time.time()
            saved_images = []
            image_descriptions = {}
            image_files = []
            
            if config.SAVE_IMAGES and result.document.pictures:
                saved_images, image_descriptions = self._process_images(
                    result.document.pictures, images_folder, pdf_stem
                )
                # Extract just the filenames for metadata
                image_files = [Path(img_path).name for img_path in saved_images]
            
            timing_breakdown['image_processing'] = time.time() - image_processing_start
            
            # Calculate total processing time
            total_processing_time = time.time() - start_time
            timing_breakdown['total_processing'] = total_processing_time
            
            # Extract text for analysis
            text_content = markdown_content
            text_length = len(text_content.strip())
            
            # Create comprehensive metadata
            metadata = {
                "source_file": original_filename if original_filename else str(pdf_file),
                "processing_timestamp": datetime.now().isoformat(),
                "ocr_enabled": pipeline_options.do_ocr,
                "ocr_engine": type(self.ocr_engine).__name__,
                "storage_backend": type(self.output_storage).__name__,
                "output_directory": output_folder if config.OUTPUT_STORAGE == "S3" else str(output_dir),
                "markdown_file": f"{pdf_stem}.md",
                "page_count": len(result.document.pages),
                "images_generated": config.SAVE_IMAGES and len(result.document.pictures) > 0,
                "images_saved": len(saved_images),
                "image_files": image_files,
                "image_descriptions": image_descriptions,
                "tables_extracted": config.EXTRACT_TABLES,
                "formulas_extracted": config.EXTRACT_FORMULAS,
                "figures_extracted": config.EXTRACT_FIGURES,
                "processing_options": {
                    "image_format": config.IMAGE_FORMAT,
                    "image_quality": config.IMAGE_QUALITY,
                    "image_dpi": config.IMAGE_DPI,
                    "image_scale": config.IMAGE_SCALE,
                    "extract_tables": config.EXTRACT_TABLES,
                    "extract_formulas": config.EXTRACT_FORMULAS,
                    "extract_figures": config.EXTRACT_FIGURES,
                    "force_full_page_ocr": config.FORCE_FULL_PAGE_OCR,
                    "input_storage_type": config.INPUT_STORAGE
                },
                "document_structure": {
                    "has_tables": bool(result.document.tables),
                    "has_figures": bool(result.document.pictures),
                    "has_formulas": any(hasattr(page, 'equations') and page.equations for page in result.document.pages),
                    "text_length": text_length,
                    "images_count": len(result.document.pictures) if result.document.pictures else 0
                },
                "performance_metrics": {
                    "file_size_mb": round(file_size_mb, 2),
                    "processing_time_seconds": round(total_processing_time, 3),
                    "pages_per_second": round(len(result.document.pages) / total_processing_time, 2) if total_processing_time > 0 else 0,
                    "mb_per_second": round(file_size_mb / total_processing_time, 2) if total_processing_time > 0 else 0,
                    "timing_breakdown": {k: round(v, 3) for k, v in timing_breakdown.items()}
                }
            }
            
            # Save metadata.json file
            self.output_storage.save(metadata, metadata_file_path, format='json')
            
            self.logger.info(f"✅ Successfully processed {display_name} in {total_processing_time:.2f}s")
            
            return {
                'status': 'success',
                'source_file': original_filename if original_filename else str(pdf_file),
                'output_files': {
                    'markdown': markdown_file_path,
                    'json': json_file_path,
                    'metadata': metadata_file_path,
                    'images': saved_images
                },
                'metadata': metadata
            }
            
        except Exception as e:
            display_name = original_filename if original_filename else pdf_file.name
            error_msg = f"Failed to process {display_name}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'status': 'error',
                'source_file': original_filename if original_filename else str(pdf_file),
                'error': error_msg,
                'processing_time': time.time() - start_time
            }

    def _process_images(self, pictures, images_folder: str, pdf_stem: str) -> tuple[List[str], Dict[str, str]]:
        """Process and save extracted images to the appropriate storage backend."""
        saved_images = []
        descriptions = {}
        
        try:
            from PIL import Image
            import io
            
            for i, picture in enumerate(pictures):
                try:
                    # Get image data
                    if hasattr(picture, 'get_image'):
                        pil_image = picture.get_image(ImageRefMode.EMBEDDED)
                        if pil_image:
                            # Generate filename
                            image_filename = f"{pdf_stem}_image_{i+1}.{config.IMAGE_FORMAT.lower()}"
                            
                            # Build image path
                            if config.OUTPUT_STORAGE == "S3":
                                image_path = f"{images_folder.strip('/')}/{image_filename}"
                            else:
                                image_path = str(Path(images_folder) / image_filename)
                            
                            # Convert and save image
                            if config.IMAGE_FORMAT.upper() == "JPEG":
                                # Convert to RGB for JPEG
                                if pil_image.mode in ('RGBA', 'LA', 'P'):
                                    pil_image = pil_image.convert('RGB')
                            
                            # Save image using appropriate storage backend
                            if config.OUTPUT_STORAGE == "S3":
                                # Save to S3 - convert PIL image to bytes
                                img_buffer = io.BytesIO()
                                save_kwargs = {'dpi': (config.IMAGE_DPI, config.IMAGE_DPI)}
                                
                                if config.IMAGE_FORMAT.upper() == "JPEG":
                                    save_kwargs['quality'] = config.IMAGE_QUALITY
                                    save_kwargs['format'] = 'JPEG'
                                else:
                                    save_kwargs['format'] = config.IMAGE_FORMAT
                                
                                pil_image.save(img_buffer, **save_kwargs)
                                img_buffer.seek(0)
                                
                                # Upload to S3
                                self.output_storage.save(img_buffer.getvalue(), image_path, format='binary')
                            else:
                                # Save to local filesystem
                                save_kwargs = {'dpi': (config.IMAGE_DPI, config.IMAGE_DPI)}
                                
                                if config.IMAGE_FORMAT.upper() == "JPEG":
                                    save_kwargs['quality'] = config.IMAGE_QUALITY
                                    pil_image.save(image_path, format='JPEG', **save_kwargs)
                                else:
                                    pil_image.save(image_path, format=config.IMAGE_FORMAT, **save_kwargs)
                            
                            saved_images.append(image_path)
                            
                            # Generate description if enabled
                            if config.GENERATE_IMAGE_DESCRIPTIONS:
                                description = getattr(picture, 'text', '') or f"Image {i+1} extracted from PDF"
                                descriptions[image_filename] = description
                                
                        else:
                            self.logger.warning(f"Could not get image data for picture {i+1}")
                            
                except Exception as e:
                    self.logger.error(f"Failed to save image {i+1}: {e}")
                    
        except ImportError:
            self.logger.error("PIL not available for image processing")
            
        return saved_images, descriptions

    def batch_parse(self, 
                   directory: Union[str, Path] = None,
                   output_base: Optional[Union[str, Path]] = None,
                   pattern: str = "*.pdf") -> Dict[str, Any]:
        """
        Parse multiple PDF files with parallel processing.
        
        Args:
            directory: Directory containing PDF files (defaults to config.INPUT_PATH)
            output_base: Base output directory (defaults to config.OUTPUT_PATH)
            pattern: File pattern to match
            
        Returns:
            Dict containing batch processing results
        """
        # Use config paths if not provided
        if directory is None:
            directory = config.INPUT_PATH
        if output_base is None:
            output_base = config.OUTPUT_PATH
            
        input_path = Path(directory) if directory else Path(config.INPUT_PATH)
        output_path = Path(output_base) if output_base else Path(config.OUTPUT_PATH)
            
        batch_start_time = time.time()
        
        try:
            self.logger.info(f"Starting batch processing - Input: {input_path}, Output: {output_path}")
            self.logger.info(f"Storage types - Input: {config.INPUT_STORAGE}, Output: {config.OUTPUT_STORAGE}")
            
            # Handle different input storage types
            if config.INPUT_STORAGE == "S3":
                pdf_files = self._list_s3_files(input_path, pattern)
                process_func = self._process_s3_file
            else:
                pdf_files = list(input_path.glob(pattern))
                process_func = self._process_local_file
            
            if not pdf_files:
                self.logger.warning(f"No PDF files found in {input_path} with pattern {pattern}")
                return {
                    'status': 'success',
                    'processed': 0,
                    'successful': 0,
                    'failed': 0,
                    'processing_time': 0,
                    'results': []
                }
            
            self.logger.info(f"Found {len(pdf_files)} PDF files to process")
            
            results = []
            successful = 0
            failed = 0
            
            # Process files
            if config.PARALLEL_PROCESSING and len(pdf_files) > 1:
                # Parallel processing
                max_workers = min(config.MAX_WORKERS, len(pdf_files))
                self.logger.info(f"Using parallel processing with {max_workers} workers")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_file = {
                        executor.submit(process_func, pdf_file, output_path): pdf_file 
                        for pdf_file in pdf_files
                    }
                    
                    # Collect results
                    for future in as_completed(future_to_file):
                        pdf_file = future_to_file[future]
                        try:
                            result = future.result()
                            results.append(result)
                            
                            if result['status'] == 'success':
                                successful += 1
                                self.logger.info(f"✅ Completed: {result['source_file']}")
                            else:
                                failed += 1
                                self.logger.error(f"❌ Failed: {result['source_file']} - {result.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            self.logger.error(f"Parallel processing error for {pdf_file}: {e}")
                            failed += 1
                            results.append({
                                'status': 'error',
                                'source_file': str(pdf_file),
                                'error': str(e)
                            })
            else:
                # Sequential processing
                self.logger.info("Using sequential processing")
                for idx, pdf_file in enumerate(pdf_files, 1):
                    try:
                        self.logger.info(f"Processing {idx}/{len(pdf_files)}: {pdf_file}")
                        result = process_func(pdf_file, output_path)
                        results.append(result)
                        
                        if result['status'] == 'success':
                            successful += 1
                            self.logger.info(f"✅ Completed: {result['source_file']}")
                        else:
                            failed += 1
                            self.logger.error(f"❌ Failed: {result['source_file']} - {result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        self.logger.error(f"Sequential processing error for {pdf_file}: {e}")
                        failed += 1
                        results.append({
                            'status': 'error',
                            'source_file': str(pdf_file),
                            'error': str(e)
                        })
                        
                    # Memory cleanup between files for large batches
                    if len(pdf_files) > config.BATCH_SIZE:
                        self._cleanup_memory()
            
            processing_time = time.time() - batch_start_time
            
            self.logger.info(f"Batch processing completed: {successful} successful, {failed} failed in {processing_time:.2f}s")
            
            return {
                'status': 'success',
                'processed': len(pdf_files),
                'successful': successful,
                'failed': failed,
                'processing_time': processing_time,
                'parallel_enabled': config.PARALLEL_PROCESSING and len(pdf_files) > 1,
                'workers_used': min(config.MAX_WORKERS, len(pdf_files)) if config.PARALLEL_PROCESSING else 1,
                'input_storage': config.INPUT_STORAGE,
                'output_storage': config.OUTPUT_STORAGE,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - batch_start_time
            }

    def _process_local_file(self, pdf_file: Path, output_base: Path) -> Dict[str, Any]:
        """Process a local PDF file."""
        return self.parse(pdf_file, output_base)

    def _process_s3_file(self, s3_file_path: str, output_base: Path) -> Dict[str, Any]:
        """Process a PDF file from S3."""
        try:
            # Extract original filename from S3 path
            original_filename = Path(s3_file_path).name
            
            # Download to temporary file with proper extension
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                content = self.input_storage.load(s3_file_path, format='binary')
                tmp_file.write(content)
                tmp_file.flush()  # Ensure all data is written
                tmp_file_path = Path(tmp_file.name)
            
            try:
                # Process the temporary file with original filename
                result = self.parse(tmp_file_path, output_base, original_filename)
                # Ensure source file shows the original S3 path
                result['source_file'] = s3_file_path
                return result
                
            finally:
                # Clean up temporary file
                try:
                    tmp_file_path.unlink(missing_ok=True)
                except Exception as cleanup_e:
                    self.logger.warning(f"Failed to cleanup temporary file {tmp_file_path}: {cleanup_e}")
                
        except Exception as e:
            self.logger.error(f"Error processing S3 file {s3_file_path}: {e}")
            return {
                'status': 'error',
                'source_file': s3_file_path,
                'error': str(e)
            }

    def _list_s3_files(self, s3_path: Path, pattern: str) -> List[str]:
        """List PDF files in S3."""
        try:
            # List files with .pdf extension
            files = self.input_storage.list_files(str(s3_path))
            pdf_files = [f for f in files if f.lower().endswith('.pdf')]
            
            self.logger.info(f"Found {len(pdf_files)} PDF files in S3 path: {s3_path}")
            return pdf_files
            
        except Exception as e:
            self.logger.error(f"Failed to list S3 files: {e}")
            return []

    def _cleanup_memory(self):
        """Clean up memory between processing large batches."""
        try:
            gc.collect()
            
            # Log memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                self.logger.warning(f"High memory usage: {memory.percent:.1f}%")
                
        except Exception as e:
            self.logger.debug(f"Memory cleanup error: {e}")


# Factory function for easy parser creation
def create_parser() -> Parser:
    """
    Create a Parser instance using configuration from config file.
    
    Returns:
        Parser: Configured parser instance
    """
    return Parser()