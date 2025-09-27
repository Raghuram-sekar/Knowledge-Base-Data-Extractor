
from .engine import OCREngine, OCRResult, OCREngineFactory
from pathlib import Path
from typing import List, Union, Dict, Any
import time
from core.logger import logger


class TesseractEngine(OCREngine):
    """Tesseract OCR engine implementation."""
    
    def __init__(self, **kwargs):
        """
        Initialize Tesseract OCR engine.
        
        Args:
            **kwargs: Configuration options
                - lang: Language code (default: 'eng')
                - oem: OCR Engine Mode (default: 3)
                - psm: Page Segmentation Mode (default: 6)
                - config: Additional tesseract config options
        """
        self.logger = logger.get_logger(__name__)
        super().__init__(**kwargs)
    
    def _initialize(self) -> None:
        """Initialize Tesseract OCR engine."""
        try:
            import pytesseract
            from PIL import Image
            
            self.pytesseract = pytesseract
            self.Image = Image
            
            # Set default configuration
            self.lang = self.config.get('lang', 'eng')
            self.oem = self.config.get('oem', 3)  # LSTM OCR Engine
            self.psm = self.config.get('psm', 6)  # Uniform block of text
            self.tesseract_config = self.config.get('config', '')
            
            # Test if tesseract is available
            self._test_availability()
            
            self.logger.info(f"Tesseract OCR engine initialized with lang={self.lang}, oem={self.oem}, psm={self.psm}")
            
        except ImportError as e:
            self.logger.error(f"Failed to import required libraries for Tesseract: {e}")
            raise ImportError("pytesseract and PIL are required for TesseractEngine")
        except Exception as e:
            self.logger.error(f"Failed to initialize Tesseract OCR engine: {e}")
            raise
    
    def _test_availability(self) -> None:
        """Test if Tesseract is properly installed and available."""
        try:
            # Try to get tesseract version
            version = self.pytesseract.get_tesseract_version()
            self.logger.debug(f"Tesseract version: {version}")
        except Exception as e:
            raise RuntimeError(f"Tesseract not available: {e}")
    
    def _build_config_string(self, **kwargs) -> str:
        """Build tesseract configuration string."""
        config_parts = []
        
        # OCR Engine Mode
        oem = kwargs.get('oem', self.oem)
        config_parts.append(f'--oem {oem}')
        
        # Page Segmentation Mode
        psm = kwargs.get('psm', self.psm)
        config_parts.append(f'--psm {psm}')
        
        # Additional config
        extra_config = kwargs.get('config', self.tesseract_config)
        if extra_config:
            config_parts.append(extra_config)
        
        return ' '.join(config_parts)
    
    def extract_text(self, image_path: Union[str, Path], **kwargs) -> OCRResult:
        """Extract text from image file using Tesseract."""
        start_time = time.time()
        
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if not self.supports_format(image_path):
                raise ValueError(f"Unsupported image format: {image_path.suffix}")
            
            # Open image
            image = self.Image.open(image_path)
            
            # Extract text
            lang = kwargs.get('lang', self.lang)
            config = self._build_config_string(**kwargs)
            
            # Get text with confidence
            text = self.pytesseract.image_to_string(image, lang=lang, config=config)
            
            # Get detailed data with bounding boxes if requested
            bounding_boxes = None
            confidence = 0.0
            
            if kwargs.get('include_boxes', False):
                data = self.pytesseract.image_to_data(
                    image, lang=lang, config=config, output_type=self.pytesseract.Output.DICT
                )
                
                # Calculate average confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                # Extract bounding boxes
                bounding_boxes = []
                for i in range(len(data['text'])):
                    if int(data['conf'][i]) > 0:
                        bbox = {
                            'text': data['text'][i],
                            'confidence': int(data['conf'][i]),
                            'left': int(data['left'][i]),
                            'top': int(data['top'][i]),
                            'width': int(data['width'][i]),
                            'height': int(data['height'][i])
                        }
                        bounding_boxes.append(bbox)
            
            processing_time = time.time() - start_time
            
            result = OCRResult(
                text=text.strip(),
                confidence=confidence,
                bounding_boxes=bounding_boxes,
                language=lang,
                processing_time=processing_time,
                metadata={
                    'engine': 'tesseract',
                    'config': config,
                    'image_path': str(image_path),
                    'image_size': image.size
                }
            )
            
            self.logger.info(f"Text extraction completed in {processing_time:.2f}s, confidence: {confidence:.1f}")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Text extraction failed after {processing_time:.2f}s: {e}")
            raise
    
    def extract_text_from_bytes(self, image_bytes: bytes, **kwargs) -> OCRResult:
        """Extract text from image bytes using Tesseract."""
        start_time = time.time()
        
        try:
            # Convert bytes to PIL Image
            from io import BytesIO
            image = self.Image.open(BytesIO(image_bytes))
            
            # Extract text
            lang = kwargs.get('lang', self.lang)
            config = self._build_config_string(**kwargs)
            
            # Get text with confidence
            text = self.pytesseract.image_to_string(image, lang=lang, config=config)
            
            # Get detailed data with bounding boxes if requested
            bounding_boxes = None
            confidence = 0.0
            
            if kwargs.get('include_boxes', False):
                data = self.pytesseract.image_to_data(
                    image, lang=lang, config=config, output_type=self.pytesseract.Output.DICT
                )
                
                # Calculate average confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                # Extract bounding boxes
                bounding_boxes = []
                for i in range(len(data['text'])):
                    if int(data['conf'][i]) > 0:
                        bbox = {
                            'text': data['text'][i],
                            'confidence': int(data['conf'][i]),
                            'left': int(data['left'][i]),
                            'top': int(data['top'][i]),
                            'width': int(data['width'][i]),
                            'height': int(data['height'][i])
                        }
                        bounding_boxes.append(bbox)
            
            processing_time = time.time() - start_time
            
            result = OCRResult(
                text=text.strip(),
                confidence=confidence,
                bounding_boxes=bounding_boxes,
                language=lang,
                processing_time=processing_time,
                metadata={
                    'engine': 'tesseract',
                    'config': config,
                    'image_size': image.size
                }
            )
            
            self.logger.info(f"Text extraction from bytes completed in {processing_time:.2f}s, confidence: {confidence:.1f}")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Text extraction from bytes failed after {processing_time:.2f}s: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Tesseract is available."""
        try:
            self._test_availability()
            return True
        except Exception:
            return False
    
    def get_supported_formats(self) -> List[str]:
        """Get supported image formats."""
        return ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp']


# Register the engine
OCREngineFactory.register_engine('tesseract', TesseractEngine)