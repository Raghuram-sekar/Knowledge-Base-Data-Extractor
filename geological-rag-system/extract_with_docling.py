"""
Extract SPEM images using Docling for superior quality
Combines Docling's AI-powered extraction with existing enhancement pipeline
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

# Docling imports
from docling_core.types.doc import ImageRefMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from PIL import Image
import io


class DoclingImageExtractor:
    """Extract images from PDF using Docling's AI-powered document understanding"""
    
    def __init__(self, output_dir: str = "extracted_spem_docling"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.metadata_dir = self.output_dir / "metadata"
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_images_from_pdf(self, pdf_path: str, 
                               image_scale: float = 3.0,
                               image_dpi: int = 300,
                               generate_descriptions: bool = True) -> Dict[str, Any]:
        """
        Extract images using Docling with high-quality settings
        
        Args:
            pdf_path: Path to SPEM PDF file
            image_scale: Scale factor for images (3.0 = 3x resolution)
            image_dpi: DPI for saved images
            generate_descriptions: Generate AI descriptions for images
            
        Returns:
            Dictionary with extraction results
        """
        print(f"Starting Docling extraction for: {pdf_path}")
        print(f"Settings: scale={image_scale}x, dpi={image_dpi}, descriptions={generate_descriptions}")
        
        # Configure Docling pipeline for optimal image extraction
        pipeline_options = PdfPipelineOptions()
        
        # Image extraction settings
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = image_scale  # Higher resolution
        pipeline_options.do_picture_description = generate_descriptions
        
        # Additional extraction features
        pipeline_options.do_table_structure = True
        pipeline_options.do_ocr = False  # We'll do OCR separately if needed
        
        # Create converter
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        # Convert document
        print("Converting document with Docling...")
        result = doc_converter.convert(pdf_path)
        
        # Extract images
        extraction_results = {
            'pdf_name': Path(pdf_path).name,
            'extraction_time': datetime.now().isoformat(),
            'settings': {
                'image_scale': image_scale,
                'image_dpi': image_dpi,
                'generate_descriptions': generate_descriptions
            },
            'images': [],
            'total_images': 0
        }
        
        # Get document pictures
        pictures = result.document.pictures if hasattr(result.document, 'pictures') else []
        extraction_results['total_images'] = len(pictures)
        
        print(f"Found {len(pictures)} images in document")
        
        # Process each image
        for i, picture in enumerate(pictures):
            try:
                # Get high-quality image
                pil_image = picture.get_image(ImageRefMode.EMBEDDED)
                
                if pil_image:
                    # Generate filename
                    image_filename = f"spem_docling_img_{i+1:03d}.png"
                    image_path = self.images_dir / image_filename
                    
                    # Convert RGBA to RGB if needed
                    if pil_image.mode in ('RGBA', 'LA', 'P'):
                        pil_image = pil_image.convert('RGB')
                    
                    # Save with high quality
                    pil_image.save(
                        image_path,
                        format='PNG',
                        dpi=(image_dpi, image_dpi),
                        optimize=False  # Maximum quality
                    )
                    
                    # Get image metadata
                    image_info = {
                        'filename': image_filename,
                        'index': i + 1,
                        'size': pil_image.size,
                        'mode': pil_image.mode,
                        'format': 'PNG',
                        'path': str(image_path)
                    }
                    
                    # Add description if available
                    if generate_descriptions and hasattr(picture, 'text'):
                        image_info['description'] = picture.text or f"Image {i+1} from SPEM document"
                    
                    # Add page reference if available
                    if hasattr(picture, 'prov'):
                        image_info['page_info'] = str(picture.prov)
                    
                    extraction_results['images'].append(image_info)
                    
                    print(f"  ✓ Extracted image {i+1}: {pil_image.size[0]}x{pil_image.size[1]} -> {image_filename}")
                    
                else:
                    print(f"  ✗ Could not get image data for picture {i+1}")
                    
            except Exception as e:
                print(f"  ✗ Error processing image {i+1}: {e}")
        
        # Save metadata
        metadata_file = self.metadata_dir / "docling_extraction_results.json"
        with open(metadata_file, 'w') as f:
            json.dump(extraction_results, f, indent=2)
        
        print(f"\n✓ Extraction complete!")
        print(f"  - Images saved to: {self.images_dir}")
        print(f"  - Metadata saved to: {metadata_file}")
        print(f"  - Total extracted: {len(extraction_results['images'])} images")
        
        return extraction_results


def main():
    """Extract SPEM images using Docling"""
    
    # Path to SPEM document
    spem_pdf = "data/SPEM_Strata_Log_Signature_SS.pdf"
    
    if not Path(spem_pdf).exists():
        print(f"Error: SPEM PDF not found at {spem_pdf}")
        print("Please update the path to your SPEM document")
        return
    
    # Create extractor
    extractor = DoclingImageExtractor(output_dir="extracted_spem_docling")
    
    # Extract with high-quality settings
    results = extractor.extract_images_from_pdf(
        pdf_path=spem_pdf,
        image_scale=3.0,      # 3x resolution boost
        image_dpi=300,        # High DPI
        generate_descriptions=True
    )
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Check extracted images in: extracted_spem_docling/images/")
    print("2. Compare quality with previous extraction")
    print("3. Optionally apply your ultimate enhancement on these images")
    print("4. Use Qwen2-VL for visual analysis")
    print("="*80)


if __name__ == "__main__":
    main()
