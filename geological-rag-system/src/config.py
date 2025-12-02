"""
Geological RAG System Configuration
SPEM-grounded well log interpretation system
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """System configuration settings"""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
    SPEM_DOCUMENT_PATH: Path = DATA_DIR / "SPEM_Strata_Log_Signature_SS.pdf"
    
    # Vector Database Settings
    VECTOR_DB_PATH: Path = DATA_DIR / "spem_vector_db"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # LLM Settings
    LLM_PROVIDER: str = "ollama"  # "openai", "anthropic", or "ollama"
    LLM_MODEL: str = "gemma3:1b"  # Use available model
    MAX_TOKENS: int = 4000
    LLM_TEMPERATURE: float = 0.1  # Low temperature for factual geological interpretation
    
    # API Keys (from environment) - Not needed for Ollama
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Ollama Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma3:1b"  # Use available model
    
    # LAS Processing Settings
    SUPPORTED_LOG_TYPES: list = ["GR", "SP", "RES", "NPHI", "RHOB", "DT", "CALIPER"]
    MIN_LOG_LENGTH: float = 10.0  # meters
    MAX_MISSING_DATA_PERCENT: float = 20.0
    
    # Pattern Recognition Settings
    PATTERN_DETECTION_WINDOW: int = 20  # Number of data points for pattern analysis
    SIMILARITY_THRESHOLD: float = -0.1  # Minimum similarity for SPEM knowledge retrieval (very permissive for cosine similarity)
    
    # Visualization Settings
    PLOT_DPI: int = 300
    PLOT_FORMAT: str = "png"
    FIGURE_SIZE: tuple = (12, 8)
    
    # Image Processing Settings
    IMAGE_EXTRACTION_DPI: int = 300  # High DPI for better quality
    IMAGE_SCALE_FACTOR: float = 3.0  # Higher scale for geological diagrams
    IMAGE_FORMAT: str = "PNG"  # Lossless format for technical diagrams
    IMAGE_ENHANCE_CONTRAST: bool = True  # Enhance contrast for better readability
    IMAGE_SHARPEN: bool = True  # Apply sharpening filter
    IMAGE_MIN_SIZE: tuple = (800, 600)  # Minimum size for extracted images
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = PROJECT_ROOT / "geological_rag.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(exist_ok=True)
settings.OUTPUTS_DIR.mkdir(exist_ok=True)
settings.VECTOR_DB_PATH.mkdir(exist_ok=True)