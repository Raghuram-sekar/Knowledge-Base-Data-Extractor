
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass


@dataclass
class OCRResult:
    """Data class to hold OCR results."""
    text: str
    confidence: float
    bounding_boxes: Optional[List[Dict[str, Any]]] = None
    language: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class OCREngine(ABC):
    """Abstract base class for OCR engines."""
    
    def __init__(self, **kwargs):
        """Initialize OCR engine with configuration."""
        self.config = kwargs
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize the OCR engine with specific configurations."""
        pass
    
    @abstractmethod
    def extract_text(self, image_path: Union[str, Path], **kwargs) -> OCRResult:
        """
        Extract text from an image file.
        
        Args:
            image_path: Path to the image file
            **kwargs: Additional parameters specific to the OCR engine
            
        Returns:
            OCRResult: Object containing extracted text and metadata
        """
        pass
    
    @abstractmethod
    def extract_text_from_bytes(self, image_bytes: bytes, **kwargs) -> OCRResult:
        """
        Extract text from image bytes.
        
        Args:
            image_bytes: Image data as bytes
            **kwargs: Additional parameters specific to the OCR engine
            
        Returns:
            OCRResult: Object containing extracted text and metadata
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the OCR engine is available and properly configured.
        
        Returns:
            bool: True if available, False otherwise
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported image formats.
        
        Returns:
            List[str]: List of supported file extensions (e.g., ['.png', '.jpg', '.tiff'])
        """
        pass
    
    def supports_format(self, file_path: Union[str, Path]) -> bool:
        """
        Check if the OCR engine supports the given file format.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            bool: True if supported, False otherwise
        """
        path = Path(file_path)
        return path.suffix.lower() in self.get_supported_formats()
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        self.config.update(kwargs)
        self._initialize()  # Re-initialize with new config


class OCREngineFactory:
    """Factory class for creating OCR engine instances."""
    
    _engines: Dict[str, type] = {}
    
    @classmethod
    def register_engine(cls, name: str, engine_class: type) -> None:
        """Register an OCR engine class."""
        if not issubclass(engine_class, OCREngine):
            raise ValueError(f"Engine class must inherit from OCREngine")
        cls._engines[name.lower()] = engine_class
    
    @classmethod
    def create_engine(cls, name: str, **kwargs) -> OCREngine:
        """Create an OCR engine instance."""
        name_lower = name.lower()
        if name_lower not in cls._engines:
            available = ', '.join(cls._engines.keys())
            raise ValueError(f"Unknown OCR engine '{name}'. Available engines: {available}")
        
        engine_class = cls._engines[name_lower]
        return engine_class(**kwargs)
    
    @classmethod
    def get_available_engines(cls) -> List[str]:
        """Get list of available OCR engine names."""
        return list(cls._engines.keys())
    
    @classmethod
    def get_available_and_ready_engines(cls) -> List[str]:
        """Get list of available and ready OCR engines."""
        ready_engines = []
        for name in cls._engines.keys():
            try:
                engine = cls.create_engine(name)
                if engine.is_available():
                    ready_engines.append(name)
            except Exception:
                continue  # Engine not ready
        return ready_engines