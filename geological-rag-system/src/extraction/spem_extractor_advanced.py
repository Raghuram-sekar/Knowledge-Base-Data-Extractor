"""
ADVANCED SPEM DOCUMENT EXTRACTOR
================================================================================
Extracts ALL content from SPEM document including:
- Text content with structure preservation
- Images (figures, diagrams, charts, photos)
- Tables and structured data
- Metadata and page information
================================================================================
"""

import os
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import base64
import json
import hashlib
from datetime import datetime
import re
from loguru import logger


@dataclass
class SPEMImage:
    """Represents an extracted image from SPEM document"""
    id: str
    page_number: int
    image_type: str  # 'figure', 'diagram', 'chart', 'photo', 'table', 'other'
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    width: int
    height: int
    file_path: str
    base64_data: str
    caption: str
    context_text: str  # Surrounding text for context
    metadata: Dict[str, Any]
    geological_content: str  # Detected geological terms
    

@dataclass
class SPEMTextBlock:
    """Represents a text block with structure information"""
    id: str
    page_number: int
    text: str
    block_type: str  # 'header', 'paragraph', 'caption', 'list', 'table'
    bbox: Tuple[float, float, float, float]
    font_info: Dict[str, Any]
    structure_level: int  # Hierarchy level (1=main header, 2=subheader, etc.)
    related_images: List[str]  # IDs of related images
    geological_terms: List[str]


@dataclass
class SPEMPage:
    """Represents a complete page with all extracted content"""
    page_number: int
    text_blocks: List[SPEMTextBlock]
    images: List[SPEMImage]
    tables: List[Dict[str, Any]]
    page_metadata: Dict[str, Any]
    geological_keywords: List[str]
    content_summary: str


class AdvancedSPEMExtractor:
    """Advanced extractor for SPEM geological documents"""
    
    def __init__(self, output_dir: str = "extracted_spem_content"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.text_dir = self.output_dir / "text"
        self.tables_dir = self.output_dir / "tables"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.images_dir, self.text_dir, self.tables_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Geological keyword dictionary
        self.geological_keywords = self._load_geological_keywords()
        
        logger.info(f"Advanced SPEM extractor initialized. Output: {self.output_dir}")
    
    def _load_geological_keywords(self) -> Dict[str, List[str]]:
        """Load comprehensive geological keyword dictionary"""
        return {
            'log_curves': [
                'gamma ray', 'GR', 'resistivity', 'RT', 'RD', 'RS', 'spontaneous potential', 'SP',
                'neutron', 'NPHI', 'density', 'RHOB', 'photoelectric', 'PEF', 'sonic', 'DT',
                'caliper', 'CAL', 'deep resistivity', 'shallow resistivity', 'laterolog',
                'microresistivity', 'MSFL', 'formation micro imager', 'FMI', 'borehole imaging'
            ],
            'patterns': [
                'funnel pattern', 'bell pattern', 'boxcar pattern', 'cylindrical pattern',
                'bow pattern', 'serrated pattern', 'cleaning up', 'dirtying up', 'coarsening upward',
                'fining upward', 'aggradational', 'progradational', 'retrogradational',
                'stacking pattern', 'log signature', 'log response', 'log character'
            ],
            'sequence_stratigraphy': [
                'sequence boundary', 'SB', 'maximum flooding surface', 'MFS', 'transgressive surface', 'TS',
                'ravinement surface', 'RS', 'lowstand systems tract', 'LST', 'transgressive systems tract', 'TST',
                'highstand systems tract', 'HST', 'falling stage systems tract', 'FSST',
                'depositional sequence', 'parasequence', 'systems tract', 'relative sea level',
                'accommodation space', 'base level', 'sequence stratigraphic', 'SPEM'
            ],
            'depositional_environments': [
                'shoreface', 'foreshore', 'backshore', 'delta', 'deltaic', 'delta front', 'delta plain',
                'fluvial', 'channel', 'point bar', 'levee', 'floodplain', 'crevasse splay',
                'tidal', 'estuarine', 'lagoonal', 'barrier', 'beach', 'aeolian', 'dune',
                'marine', 'offshore', 'shelf', 'slope', 'basin', 'turbidite', 'submarine fan'
            ],
            'lithology': [
                'sandstone', 'sand', 'shale', 'mudstone', 'siltstone', 'limestone', 'dolomite',
                'conglomerate', 'breccia', 'coal', 'evaporite', 'anhydrite', 'salt',
                'clean sand', 'dirty sand', 'shaly sand', 'calcareous', 'siliceous',
                'arkose', 'quartz arenite', 'sublitharenite', 'litharenite'
            ],
            'structural_geological': [
                'fault', 'fracture', 'fold', 'anticline', 'syncline', 'unconformity',
                'angular unconformity', 'disconformity', 'nonconformity', 'paraconformity',
                'structural dip', 'bedding', 'lamination', 'cross-bedding', 'bioturbation'
            ],
            'reservoir_terms': [
                'reservoir', 'porosity', 'permeability', 'net pay', 'gross pay', 'water saturation',
                'hydrocarbon saturation', 'seal', 'cap rock', 'source rock', 'migration',
                'trap', 'structural trap', 'stratigraphic trap', 'combination trap'
            ]
        }
    
    def extract_complete_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract ALL content from SPEM document
        
        Args:
            pdf_path: Path to SPEM PDF document
            
        Returns:
            Complete extraction results
        """
        logger.info(f"Starting complete SPEM document extraction: {pdf_path}")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"SPEM document not found: {pdf_path}")
        
        # Open PDF document
        doc = fitz.open(str(pdf_path))
        
        extraction_results = {
            'document_info': {
                'file_path': str(pdf_path),
                'file_size': pdf_path.stat().st_size,
                'extraction_date': datetime.now().isoformat(),
                'total_pages': len(doc),
                'document_metadata': doc.metadata
            },
            'pages': [],
            'summary': {
                'total_images': 0,
                'total_text_blocks': 0,
                'total_tables': 0,
                'geological_keywords_found': set(),
                'content_types': set()
            }
        }
        
        # Extract content from each page
        for page_num in range(len(doc)):
            logger.info(f"Processing page {page_num + 1}/{len(doc)}")
            
            page = doc[page_num]
            page_data = self._extract_page_content(page, page_num + 1)
            
            extraction_results['pages'].append(page_data)
            
            # Update summary statistics
            extraction_results['summary']['total_images'] += len(page_data.images)
            extraction_results['summary']['total_text_blocks'] += len(page_data.text_blocks)
            extraction_results['summary']['total_tables'] += len(page_data.tables)
            extraction_results['summary']['geological_keywords_found'].update(page_data.geological_keywords)
            
            # Determine content types
            if page_data.images:
                extraction_results['summary']['content_types'].add('images')
            if page_data.text_blocks:
                extraction_results['summary']['content_types'].add('text')
            if page_data.tables:
                extraction_results['summary']['content_types'].add('tables')
        
        doc.close()
        
        # Convert sets to lists for JSON serialization
        extraction_results['summary']['geological_keywords_found'] = list(extraction_results['summary']['geological_keywords_found'])
        extraction_results['summary']['content_types'] = list(extraction_results['summary']['content_types'])
        
        # Save complete extraction results
        self._save_extraction_results(extraction_results)
        
        logger.info(f"Complete extraction finished. Results saved to: {self.output_dir}")
        logger.info(f"Extracted: {extraction_results['summary']['total_images']} images, "
                   f"{extraction_results['summary']['total_text_blocks']} text blocks, "
                   f"{extraction_results['summary']['total_tables']} tables")
        
        return extraction_results
    
    def _extract_page_content(self, page: fitz.Page, page_num: int) -> SPEMPage:
        """Extract all content from a single page"""
        
        # Extract text blocks with structure
        text_blocks = self._extract_structured_text(page, page_num)
        
        # Extract images
        images = self._extract_images(page, page_num)
        
        # Extract tables
        tables = self._extract_tables(page, page_num)
        
        # Extract geological keywords from page
        page_text = ' '.join([block.text for block in text_blocks])
        geological_keywords = self._extract_geological_terms(page_text)
        
        # Create content summary
        content_summary = self._create_page_summary(page_text, images, tables)
        
        # Page metadata
        page_metadata = {
            'page_number': page_num,
            'page_size': page.rect,
            'rotation': page.rotation,
            'has_images': len(images) > 0,
            'has_tables': len(tables) > 0,
            'word_count': len(page_text.split()),
            'geological_content_density': len(geological_keywords) / max(len(page_text.split()), 1)
        }
        
        return SPEMPage(
            page_number=page_num,
            text_blocks=text_blocks,
            images=images,
            tables=tables,
            page_metadata=page_metadata,
            geological_keywords=geological_keywords,
            content_summary=content_summary
        )
    
    def _extract_structured_text(self, page: fitz.Page, page_num: int) -> List[SPEMTextBlock]:
        """Extract text with structure preservation"""
        text_blocks = []
        
        # Get text with detailed formatting
        text_dict = page.get_text("dict")
        
        block_id = 0
        for block in text_dict["blocks"]:
            if "lines" in block:  # Text block
                block_text = ""
                font_info = {}
                
                # Collect text from all lines in block
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"]
                        
                        # Capture font information
                        if not font_info:
                            font_info = {
                                'font': span.get('font', ''),
                                'size': span.get('size', 0),
                                'color': span.get('color', 0),
                                'flags': span.get('flags', 0)
                            }
                
                if block_text.strip():
                    # Determine block type
                    block_type = self._classify_text_block(block_text, font_info)
                    structure_level = self._determine_structure_level(block_text, font_info, block_type)
                    
                    # Extract geological terms
                    geological_terms = self._extract_geological_terms(block_text)
                    
                    text_block = SPEMTextBlock(
                        id=f"text_block_{page_num}_{block_id:03d}",
                        page_number=page_num,
                        text=block_text.strip(),
                        block_type=block_type,
                        bbox=tuple(block["bbox"]),
                        font_info=font_info,
                        structure_level=structure_level,
                        related_images=[],  # Will be populated later
                        geological_terms=geological_terms
                    )
                    
                    text_blocks.append(text_block)
                    block_id += 1
        
        return text_blocks
    
    def _extract_images(self, page: fitz.Page, page_num: int) -> List[SPEMImage]:
        """Extract all images from page with enhanced processing"""
        images = []
        
        # Get image list from page
        image_list = page.get_images()
        
        for img_index, img_ref in enumerate(image_list):
            try:
                # Extract image data
                img_dict = page.parent.extract_image(img_ref[0])
                image_data = img_dict["image"]
                
                # Get image properties
                img_ext = img_dict["ext"]
                img_width = img_dict["width"]
                img_height = img_dict["height"]
                
                # Create PIL image for processing
                import io
                pil_image = Image.open(io.BytesIO(image_data))
                
                # Enhance image quality
                enhanced_image = self._enhance_image(pil_image)
                
                # Generate unique image ID
                image_hash = hashlib.md5(image_data).hexdigest()
                image_id = f"spem_img_{page_num:03d}_{img_index:03d}_{image_hash[:8]}"
                
                # Save image file
                image_filename = f"{image_id}.{img_ext}"
                image_path = self.images_dir / image_filename
                enhanced_image.save(str(image_path), quality=95, optimize=True)
                
                # Convert to base64 for storage
                import io
                buffer = io.BytesIO()
                enhanced_image.save(buffer, format=img_ext.upper() if img_ext.upper() != 'JPG' else 'JPEG')
                base64_data = base64.b64encode(buffer.getvalue()).decode()
                
                # Get image bounding box (approximate)
                img_bbox = self._find_image_bbox(page, img_ref)
                
                # Extract surrounding text for context
                context_text = self._extract_image_context(page, img_bbox)
                
                # Extract caption
                caption = self._extract_image_caption(page, img_bbox, context_text)
                
                # Classify image type
                image_type = self._classify_image_type(enhanced_image, caption, context_text)
                
                # Extract geological content
                geological_content = self._analyze_image_geological_content(enhanced_image, caption, context_text)
                
                # Create image metadata
                metadata = {
                    'original_ref': img_ref,
                    'file_size': len(image_data),
                    'color_space': img_dict.get('colorspace', ''),
                    'bits_per_component': img_dict.get('bpc', 0),
                    'extraction_method': 'pymupdf',
                    'enhancement_applied': True,
                    'geological_indicators': self._detect_geological_indicators(enhanced_image, caption)
                }
                
                spem_image = SPEMImage(
                    id=image_id,
                    page_number=page_num,
                    image_type=image_type,
                    bbox=img_bbox,
                    width=img_width,
                    height=img_height,
                    file_path=str(image_path),
                    base64_data=base64_data,
                    caption=caption,
                    context_text=context_text,
                    metadata=metadata,
                    geological_content=geological_content
                )
                
                images.append(spem_image)
                
                logger.debug(f"Extracted image {image_id}: {image_type} ({img_width}x{img_height})")
                
            except Exception as e:
                logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                continue
        
        return images
    
    def _enhance_image(self, pil_image: Image.Image) -> Image.Image:
        """Enhance image quality for better analysis"""
        try:
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Apply enhancement filters
            enhanced = pil_image
            
            # Increase contrast slightly
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.2)
            
            # Increase sharpness slightly
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.1)
            
            # Apply slight noise reduction
            enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return pil_image
    
    def _find_image_bbox(self, page: fitz.Page, img_ref: Tuple) -> Tuple[float, float, float, float]:
        """Find bounding box of image on page"""
        try:
            # Try to find image placement on page
            img_rects = page.get_image_rects(img_ref[0])
            if img_rects:
                return tuple(img_rects[0])
            else:
                # Fallback: estimate based on page size
                page_rect = page.rect
                return (page_rect.width * 0.1, page_rect.height * 0.1,
                       page_rect.width * 0.9, page_rect.height * 0.9)
        except:
            # Default bbox
            return (0, 0, page.rect.width, page.rect.height)
    
    def _extract_image_context(self, page: fitz.Page, img_bbox: Tuple[float, float, float, float]) -> str:
        """Extract text surrounding an image for context"""
        try:
            # Expand bbox to capture surrounding text
            expanded_bbox = fitz.Rect(
                max(0, img_bbox[0] - 50),
                max(0, img_bbox[1] - 100),
                min(page.rect.width, img_bbox[2] + 50),
                min(page.rect.height, img_bbox[3] + 200)
            )
            
            # Extract text from expanded area
            context_text = page.get_textbox(expanded_bbox)
            return context_text.strip()
            
        except Exception as e:
            logger.debug(f"Context extraction failed: {e}")
            return ""
    
    def _extract_image_caption(self, page: fitz.Page, img_bbox: Tuple[float, float, float, float], context_text: str) -> str:
        """Extract image caption"""
        try:
            # Look for figure/table references in context
            caption_patterns = [
                r'Figure\s+\d+[:\.\-\s]+([^\.]+)',
                r'Fig\.\s+\d+[:\.\-\s]+([^\.]+)',
                r'Table\s+\d+[:\.\-\s]+([^\.]+)',
                r'Plate\s+\d+[:\.\-\s]+([^\.]+)'
            ]
            
            for pattern in caption_patterns:
                match = re.search(pattern, context_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    return match.group(0).strip()
            
            # Fallback: look for text below image
            below_bbox = fitz.Rect(img_bbox[0], img_bbox[3], img_bbox[2], img_bbox[3] + 100)
            below_text = page.get_textbox(below_bbox)
            
            if below_text and len(below_text.strip()) < 200:
                return below_text.strip()
            
            return ""
            
        except Exception as e:
            logger.debug(f"Caption extraction failed: {e}")
            return ""
    
    def _classify_image_type(self, pil_image: Image.Image, caption: str, context_text: str) -> str:
        """Classify image type based on content and context"""
        # Combine caption and context for analysis
        text_content = f"{caption} {context_text}".lower()
        
        # Image type classification based on geological content
        if any(term in text_content for term in ['log', 'curve', 'track', 'gr', 'resistivity']):
            return 'well_log'
        elif any(term in text_content for term in ['seismic', 'reflection', 'amplitude']):
            return 'seismic'
        elif any(term in text_content for term in ['core', 'sample', 'thin section']):
            return 'core_photo'
        elif any(term in text_content for term in ['map', 'contour', 'structure', 'isopach']):
            return 'geological_map'
        elif any(term in text_content for term in ['chart', 'diagram', 'schematic', 'model']):
            return 'diagram'
        elif any(term in text_content for term in ['table', 'data', 'list']):
            return 'table'
        elif any(term in text_content for term in ['outcrop', 'field', 'exposure']):
            return 'field_photo'
        elif any(term in text_content for term in ['figure', 'fig']):
            return 'figure'
        else:
            return 'other'
    
    def _analyze_image_geological_content(self, pil_image: Image.Image, caption: str, context_text: str) -> str:
        """Analyze geological content in image"""
        geological_content = []
        
        # Extract geological terms from caption and context
        text_content = f"{caption} {context_text}"
        geological_terms = self._extract_geological_terms(text_content)
        
        if geological_terms:
            geological_content.extend(geological_terms)
        
        # Basic image analysis for geological indicators
        geological_indicators = self._detect_geological_indicators(pil_image, caption)
        geological_content.extend(geological_indicators)
        
        return '; '.join(geological_content)
    
    def _detect_geological_indicators(self, pil_image: Image.Image, caption: str) -> List[str]:
        """Detect visual geological indicators in image"""
        indicators = []
        
        try:
            # Convert to numpy array for analysis
            img_array = np.array(pil_image)
            
            # Basic color analysis
            if len(img_array.shape) == 3:  # Color image
                # Check for common geological color patterns
                avg_colors = np.mean(img_array, axis=(0, 1))
                
                # Brown/red tones might indicate sandstone/shale
                if avg_colors[0] > avg_colors[1] and avg_colors[0] > avg_colors[2]:
                    indicators.append('reddish_coloration')
                
                # Gray tones might indicate logs or seismic
                if np.std(avg_colors) < 20:  # Low color variation
                    indicators.append('grayscale_pattern')
            
            # Size-based classification
            height, width = img_array.shape[:2]
            aspect_ratio = width / height
            
            if aspect_ratio > 3:  # Very wide images might be log tracks
                indicators.append('horizontal_log_pattern')
            elif aspect_ratio < 0.5:  # Tall images might be vertical logs
                indicators.append('vertical_log_pattern')
            
            # Check caption for specific indicators
            caption_lower = caption.lower()
            if 'gamma ray' in caption_lower or 'gr' in caption_lower:
                indicators.append('gamma_ray_log')
            if 'resistivity' in caption_lower:
                indicators.append('resistivity_log')
            if 'sequence' in caption_lower:
                indicators.append('sequence_stratigraphic_content')
            
        except Exception as e:
            logger.debug(f"Geological indicator detection failed: {e}")
        
        return indicators
    
    def _extract_tables(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables from page"""
        tables = []
        
        try:
            # Try to find tables using pymupdf
            table_finder = page.find_tables()
            
            for table_index, table in enumerate(table_finder):
                table_data = table.extract()
                
                if table_data and len(table_data) > 1:  # Valid table with header and data
                    table_info = {
                        'id': f"table_{page_num}_{table_index}",
                        'page_number': page_num,
                        'bbox': table.bbox,
                        'data': table_data,
                        'row_count': len(table_data),
                        'col_count': len(table_data[0]) if table_data else 0,
                        'geological_content': self._analyze_table_geological_content(table_data)
                    }
                    
                    tables.append(table_info)
                    
                    # Save table as CSV
                    table_filename = f"table_{page_num}_{table_index}.json"
                    table_path = self.tables_dir / table_filename
                    with open(table_path, 'w') as f:
                        json.dump(table_info, f, indent=2)
        
        except Exception as e:
            logger.debug(f"Table extraction failed for page {page_num}: {e}")
        
        return tables
    
    def _analyze_table_geological_content(self, table_data: List[List[str]]) -> List[str]:
        """Analyze geological content in table"""
        geological_content = []
        
        # Flatten table data for analysis
        table_text = ' '.join([' '.join(row) for row in table_data])
        geological_terms = self._extract_geological_terms(table_text)
        
        return geological_terms
    
    def _classify_text_block(self, text: str, font_info: Dict[str, Any]) -> str:
        """Classify text block type"""
        text_lower = text.lower().strip()
        
        # Check for headers
        if font_info.get('size', 0) > 14 and font_info.get('flags', 0) & 2**4:  # Bold and large
            return 'header'
        
        # Check for captions
        if text_lower.startswith(('figure', 'fig.', 'table', 'plate')):
            return 'caption'
        
        # Check for lists
        if text_lower.startswith(('•', '-', '1.', 'a)', 'i)')):
            return 'list'
        
        # Default to paragraph
        return 'paragraph'
    
    def _determine_structure_level(self, text: str, font_info: Dict[str, Any], block_type: str) -> int:
        """Determine hierarchical structure level"""
        if block_type == 'header':
            font_size = font_info.get('size', 0)
            if font_size > 18:
                return 1  # Main header
            elif font_size > 14:
                return 2  # Subheader
            else:
                return 3  # Sub-subheader
        elif block_type == 'caption':
            return 4
        else:
            return 5  # Regular content
    
    def _extract_geological_terms(self, text: str) -> List[str]:
        """Extract geological terms from text"""
        found_terms = []
        text_lower = text.lower()
        
        for category, terms in self.geological_keywords.items():
            for term in terms:
                if term.lower() in text_lower:
                    found_terms.append(term)
        
        return list(set(found_terms))  # Remove duplicates
    
    def _create_page_summary(self, page_text: str, images: List[SPEMImage], tables: List[Dict]) -> str:
        """Create summary of page content"""
        summary_parts = []
        
        # Text summary
        word_count = len(page_text.split())
        summary_parts.append(f"{word_count} words of text")
        
        # Image summary
        if images:
            image_types = [img.image_type for img in images]
            summary_parts.append(f"{len(images)} images ({', '.join(set(image_types))})")
        
        # Table summary
        if tables:
            summary_parts.append(f"{len(tables)} tables")
        
        # Geological content summary
        geological_terms = self._extract_geological_terms(page_text)
        if geological_terms:
            summary_parts.append(f"Geological content: {', '.join(geological_terms[:5])}")
        
        return "; ".join(summary_parts)
    
    def _save_extraction_results(self, results: Dict[str, Any]):
        """Save complete extraction results"""
        # Save main results file
        results_file = self.metadata_dir / "extraction_results.json"
        
        # Convert SPEMPage objects to dictionaries for JSON serialization
        serializable_results = {
            'document_info': results['document_info'],
            'summary': results['summary'],
            'pages': []
        }
        
        for page in results['pages']:
            page_dict = {
                'page_number': page.page_number,
                'text_blocks': [asdict(block) for block in page.text_blocks],
                'images': [asdict(img) for img in page.images],
                'tables': page.tables,
                'page_metadata': page.page_metadata,
                'geological_keywords': page.geological_keywords,
                'content_summary': page.content_summary
            }
            serializable_results['pages'].append(page_dict)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save individual page summaries
        pages_summary_file = self.metadata_dir / "pages_summary.txt"
        with open(pages_summary_file, 'w', encoding='utf-8') as f:
            f.write("SPEM DOCUMENT EXTRACTION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Document: {results['document_info']['file_path']}\n")
            f.write(f"Total Pages: {results['document_info']['total_pages']}\n")
            f.write(f"Extraction Date: {results['document_info']['extraction_date']}\n\n")
            
            f.write("CONTENT SUMMARY:\n")
            f.write(f"- Total Images: {results['summary']['total_images']}\n")
            f.write(f"- Total Text Blocks: {results['summary']['total_text_blocks']}\n")
            f.write(f"- Total Tables: {results['summary']['total_tables']}\n")
            f.write(f"- Content Types: {', '.join(results['summary']['content_types'])}\n\n")
            
            f.write("GEOLOGICAL KEYWORDS FOUND:\n")
            for keyword in sorted(results['summary']['geological_keywords_found']):
                f.write(f"- {keyword}\n")
            f.write("\n")
            
            f.write("PAGE-BY-PAGE SUMMARY:\n")
            f.write("-" * 30 + "\n")
            for page in results['pages']:
                f.write(f"Page {page.page_number}: {page.content_summary}\n")
        
        logger.info(f"Extraction results saved to: {results_file}")


def main():
    """Test the advanced SPEM extractor"""
    extractor = AdvancedSPEMExtractor("extracted_spem_complete")
    
    # Extract from SPEM document
    spem_path = "data/SPEM_Strata_Log_Signature_SS.pdf"
    
    try:
        results = extractor.extract_complete_document(spem_path)
        
        print(f"\nEXTRACTION COMPLETE!")
        print(f"Total Pages: {results['document_info']['total_pages']}")
        print(f"Total Images: {results['summary']['total_images']}")
        print(f"Total Text Blocks: {results['summary']['total_text_blocks']}")
        print(f"Total Tables: {results['summary']['total_tables']}")
        print(f"Geological Keywords: {len(results['summary']['geological_keywords_found'])}")
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()