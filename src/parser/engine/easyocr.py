
from .engine import OCREngine, OCRResult, OCREngineFactory
from pathlib import Path
from typing import List, Union, Dict, Any
import time
from core.logger import logger


class EasyOCREngine(OCREngine):
    """EasyOCR engine implementation."""
    
    def __init__(self, **kwargs):
        """
        Initialize EasyOCR engine.
        
        Args:
            **kwargs: Configuration options
                - lang_list: List of language codes (default: ['en'])
                - gpu: Use GPU if available (default: True)
                - model_storage_directory: Custom model storage path
                - user_network_directory: Custom network directory
                - download_enabled: Enable model downloads (default: True)
        """
        self.logger = logger.get_logger()
        self.reader = None
        super().__init__(**kwargs)
    
    def _initialize(self) -> None:
        """Initialize EasyOCR engine."""
        try:
            import easyocr
            
            self.easyocr = easyocr
            
            # Set default configuration
            self.lang_list = self.config.get('lang_list', ['en'])
            self.gpu = self.config.get('gpu', True)
            self.model_storage_directory = self.config.get('model_storage_directory', None)
            self.user_network_directory = self.config.get('user_network_directory', None)
            self.download_enabled = self.config.get('download_enabled', True)
            
            # Initialize EasyOCR Reader
            self._initialize_reader()
            
            self.logger.info(f"EasyOCR engine initialized with languages={self.lang_list}, gpu={self.gpu}")
            
        except ImportError as e:
            self.logger.error(f"Failed to import EasyOCR: {e}")
            raise ImportError("easyocr is required for EasyOCREngine")
        except Exception as e:
            self.logger.error(f"Failed to initialize EasyOCR engine: {e}")
            raise
    
    def _initialize_reader(self) -> None:
        """Initialize the EasyOCR Reader object."""
        try:
            self.reader = self.easyocr.Reader(
                self.lang_list,
                gpu=self.gpu,
                model_storage_directory=self.model_storage_directory,
                user_network_directory=self.user_network_directory,
                download_enabled=self.download_enabled
            )
            self.logger.debug("EasyOCR Reader initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize EasyOCR Reader: {e}")
            raise
    
    def extract_text(self, image_path: Union[str, Path], **kwargs) -> OCRResult:
        """Extract text from image file using EasyOCR."""
        start_time = time.time()
        
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if not self.supports_format(image_path):
                raise ValueError(f"Unsupported image format: {image_path.suffix}")
            
            # Extract parameters
            paragraph = kwargs.get('paragraph', False)
            width_ths = kwargs.get('width_ths', 0.7)
            height_ths = kwargs.get('height_ths', 0.7)
            decoder = kwargs.get('decoder', 'greedy')
            beamWidth = kwargs.get('beamWidth', 5)
            batch_size = kwargs.get('batch_size', 1)
            
            # Perform OCR
            results = self.reader.readtext(
                str(image_path),
                paragraph=paragraph,
                width_ths=width_ths,
                height_ths=height_ths,
                decoder=decoder,
                beamWidth=beamWidth,
                batch_size=batch_size
            )
            
            # Process results
            full_text = []
            bounding_boxes = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                full_text.append(text)
                confidences.append(confidence)
                
                if kwargs.get('include_boxes', False):
                    # Convert bbox coordinates
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    bbox_dict = {
                        'text': text,
                        'confidence': confidence * 100,  # Convert to percentage
                        'left': int(min(x_coords)),
                        'top': int(min(y_coords)),
                        'width': int(max(x_coords) - min(x_coords)),
                        'height': int(max(y_coords) - min(y_coords)),
                        'bbox_coords': bbox
                    }
                    bounding_boxes.append(bbox_dict)
            
            # Calculate average confidence
            avg_confidence = (sum(confidences) / len(confidences) * 100) if confidences else 0.0
            
            processing_time = time.time() - start_time
            
            result = OCRResult(
                text=' '.join(full_text) if not paragraph else '\n'.join(full_text),
                confidence=avg_confidence,
                bounding_boxes=bounding_boxes if kwargs.get('include_boxes', False) else None,
                language='+'.join(self.lang_list),
                processing_time=processing_time,
                metadata={
                    'engine': 'easyocr',
                    'image_path': str(image_path),
                    'languages': self.lang_list,
                    'paragraph_mode': paragraph,
                    'total_detections': len(results)
                }
            )
            
            self.logger.info(f"Text extraction completed in {processing_time:.2f}s, confidence: {avg_confidence:.1f}")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Text extraction failed after {processing_time:.2f}s: {e}")
            raise
    
    def extract_text_from_bytes(self, image_bytes: bytes, **kwargs) -> OCRResult:
        """Extract text from image bytes using EasyOCR."""
        start_time = time.time()
        
        try:
            import numpy as np
            import cv2
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image from bytes")
            
            # Extract parameters
            paragraph = kwargs.get('paragraph', False)
            width_ths = kwargs.get('width_ths', 0.7)
            height_ths = kwargs.get('height_ths', 0.7)
            decoder = kwargs.get('decoder', 'greedy')
            beamWidth = kwargs.get('beamWidth', 5)
            batch_size = kwargs.get('batch_size', 1)
            
            # Perform OCR
            results = self.reader.readtext(
                image,
                paragraph=paragraph,
                width_ths=width_ths,
                height_ths=height_ths,
                decoder=decoder,
                beamWidth=beamWidth,
                batch_size=batch_size
            )
            
            # Process results
            full_text = []
            bounding_boxes = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                full_text.append(text)
                confidences.append(confidence)
                
                if kwargs.get('include_boxes', False):
                    # Convert bbox coordinates
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    bbox_dict = {
                        'text': text,
                        'confidence': confidence * 100,  # Convert to percentage
                        'left': int(min(x_coords)),
                        'top': int(min(y_coords)),
                        'width': int(max(x_coords) - min(x_coords)),
                        'height': int(max(y_coords) - min(y_coords)),
                        'bbox_coords': bbox
                    }
                    bounding_boxes.append(bbox_dict)
            
            # Calculate average confidence
            avg_confidence = (sum(confidences) / len(confidences) * 100) if confidences else 0.0
            
            processing_time = time.time() - start_time
            
            result = OCRResult(
                text=' '.join(full_text) if not paragraph else '\n'.join(full_text),
                confidence=avg_confidence,
                bounding_boxes=bounding_boxes if kwargs.get('include_boxes', False) else None,
                language='+'.join(self.lang_list),
                processing_time=processing_time,
                metadata={
                    'engine': 'easyocr',
                    'languages': self.lang_list,
                    'paragraph_mode': paragraph,
                    'total_detections': len(results),
                    'image_shape': image.shape
                }
            )
            
            self.logger.info(f"Text extraction from bytes completed in {processing_time:.2f}s, confidence: {avg_confidence:.1f}")
            return result
            
        except ImportError as e:
            self.logger.error(f"Missing dependencies for EasyOCR: {e}")
            raise ImportError("opencv-python and numpy are required for processing image bytes")
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Text extraction from bytes failed after {processing_time:.2f}s: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if EasyOCR is available."""
        try:
            if self.reader is None:
                self._initialize_reader()
            return True
        except Exception:
            return False
    
    def get_supported_formats(self) -> List[str]:
        """Get supported image formats."""
        return ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp']
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        try:
            # Common EasyOCR supported languages
            return [
                'en', 'ch_sim', 'ch_tra', 'ja', 'ko', 'th', 'vi', 'ar', 'bg', 'cs', 'da', 'de',
                'el', 'es', 'et', 'fi', 'fr', 'hr', 'hu', 'id', 'is', 'it', 'lt', 'lv', 'mt',
                'nl', 'no', 'pl', 'pt', 'ro', 'rs_cyrillic', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
                'tr', 'uk', 'ru', 'be', 'bg', 'cn', 'hi', 'mr', 'ne', 'ta', 'bn', 'as', 'or'
            ]
        except Exception:
            return ['en']  # Default fallback


# Register the engine
OCREngineFactory.register_engine('easyocr', EasyOCREngine)