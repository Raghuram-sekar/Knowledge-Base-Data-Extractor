"""
Simple wrapper to use Telesto's Docling parser for SPEM extraction
This avoids dependency issues by using the already-configured environment
"""

import sys
import os
from pathlib import Path

# Set Tesseract path for subprocess
os.environ['PATH'] = os.environ['PATH'] + r";C:\Program Files\Tesseract-OCR"

# Disable AI picture descriptions to avoid model compatibility issues
os.environ['GENERATE_IMAGE_DESCRIPTIONS'] = 'false'
os.environ['SAVE_IMAGES'] = 'true'  # But keep image extraction enabled

# Add Telesto parser to path
telesto_path = Path(r"C:\Users\Raghuram S\Project\Telesto\Knowledge-Base-Data-Extractor")
sys.path.insert(0, str(telesto_path / "src"))

def extract_spem_with_docling():
    """Extract SPEM images using the Telesto Docling parser"""
    
    # Import from Telesto (it has Docling properly configured)
    from parser.parser import Parser
    
    # Paths
    spem_pdf = Path("data/SPEM_Strata_Log_Signature_SS.pdf")
    output_dir = Path("extracted_spem_docling")
    
    if not spem_pdf.exists():
        print(f"Error: SPEM PDF not found at {spem_pdf}")
        return
    
    print("="*80)
    print("DOCLING-BASED SPEM IMAGE EXTRACTION")
    print("="*80)
    print(f"Input: {spem_pdf}")
    print(f"Output: {output_dir}")
    print("="*80)
    
    # Create parser instance
    parser = Parser()
    
    # Process PDF with high-quality settings
    print("\nProcessing with Docling (this may take a few minutes)...")
    
    result = parser.parse(
        filepath=spem_pdf,
        output_base=str(output_dir),
        original_filename=None
    )
    
    if result and result.get('status') == 'success':
        print("\n✓ Extraction Complete!")
        print(f"  - Pages processed: {result.get('pages_processed', 0)}")
        print(f"  - Images saved: {result.get('images_saved', 0)}")
        print(f"  - Processing time: {result.get('processing_time', 0):.2f}s")
        print(f"\nImages location: {output_dir / 'images'}")
        print("\nNext steps:")
        print("  1. Check image quality in extracted_spem_docling/images/")
        print("  2. Compare with previous PyMuPDF extraction")
        print("  3. Apply ultimate enhancement if needed")
        print("  4. Use Qwen2-VL for analysis")
    else:
        print(f"\n✗ Extraction failed: {result.get('error', 'Unknown error')}")
    
    print("="*80)

if __name__ == "__main__":
    try:
        extract_spem_with_docling()
    except ImportError as e:
        print(f"\nError: Could not import Telesto parser: {e}")
        print("\nAlternative: Copy the parser from Telesto folder")
        print("Or install Docling dependencies properly")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
