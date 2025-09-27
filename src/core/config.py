from typing import Literal, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from dotenv import load_dotenv
import os

load_dotenv()


class Config(BaseSettings):
    """Configuration class using Pydantic BaseSettings for validation and environment variable management."""
    
    # Project Configurations
    PROJECT_NAME: str = Field(
        default="Knowledge Base Data Extractor",
        description="Name of the project"
    )
    PROJECT_DESCRIPTION: str = Field(
        default="A tool for extracting data from knowledge bases",
        description="Description of the project"
    )
    VERSION: str = Field(
        default="1.0.0",
        description="Version of the application"
    )

    # Database Configuration
    DATABASE_NAME: Optional[str] = Field(
        default=None,
        env="DATABASE_NAME",
        description="Database name"
    )
    DATABASE_USERNAME: Optional[str] = Field(
        default=None,
        env="DATABASE_USERNAME",
        description="Database username"
    )
    DATABASE_PASSWORD: Optional[str] = Field(
        default=None,
        env="DATABASE_PASSWORD",
        description="Database password"
    )
    DATABASE_PORT: Optional[int] = Field(
        default=5432,
        env="DATABASE_PORT",
        description="Database port"
    )

    # S3 Configuration
    S3_BUCKET_NAME: Optional[str] = Field(
        default=None,
        env="S3_BUCKET_NAME",
        description="S3 bucket name"
    )
    S3_ACCESS_KEY: Optional[str] = Field(
        default=None,
        env="S3_ACCESS_KEY",
        description="S3 access key"
    )
    S3_SECRET_KEY: Optional[str] = Field(
        default=None,
        env="S3_SECRET_KEY",
        description="S3 secret key"
    )
    S3_REGION: Optional[str] = Field(
        default="us-east-1",
        env="S3_REGION",
        description="S3 region"
    )
    S3_ENDPOINT_URL: Optional[str] = Field(
        default=None,
        env="S3_ENDPOINT_URL",
        description="S3 endpoint URL (for MinIO or other S3-compatible services)"
    )

    # Logging Configuration
    LOG_LEVEL: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level"
    )
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT",
        description="Logging format"
    )

    # Neo4j Configuration
    NEO4J_URI: Optional[str] = Field(
        default=None,
        env="NEO4J_URI",
        description="Neo4j URI"
    )
    NEO4J_USERNAME: Optional[str] = Field(
        default=None,
        env="NEO4J_USERNAME",
        description="Neo4j username"
    )
    NEO4J_PASSWORD: Optional[str] = Field(
        default=None,
        env="NEO4J_PASSWORD",
        description="Neo4j password"
    )
    NEO4J_PORT: Optional[int] = Field(
        default=7687,
        env="NEO4J_PORT",
        description="Neo4j port"
    )

    # Parsing Configuration
    INPUT_STORAGE: Literal["LOCAL", "S3"] = Field(
        default="LOCAL",
        env="INPUT_STORAGE",
        description="Input storage type"
    )
    OUTPUT_STORAGE: Literal["LOCAL", "S3"] = Field(
        default="LOCAL",
        env="OUTPUT_STORAGE",
        description="Output storage type"
    )
    OCR_ENGINE: Literal["TESSERACT", "EASYOCR"] = Field(
        default="TESSERACT",
        env="OCR_ENGINE",
        description="OCR engine to use"
    )

    # Input/Output Path Configuration
    INPUT_PATH: str = Field(
        default="data/raw",
        env="INPUT_PATH",
        description="Input directory path (local or S3)"
    )
    OUTPUT_PATH: str = Field(
        default="data/processed",
        env="OUTPUT_PATH",
        description="Output directory path (local or S3)"
    )

    # Image Processing Configuration
    SAVE_IMAGES: bool = Field(
        default=True,
        env="SAVE_IMAGES",
        description="Whether to save extracted images"
    )
    IMAGE_FORMAT: Literal["PNG", "JPEG", "WEBP"] = Field(
        default="PNG",
        env="IMAGE_FORMAT",
        description="Format for extracted images"
    )
    IMAGE_QUALITY: int = Field(
        default=95,
        env="IMAGE_QUALITY",
        ge=1,
        le=100,
        description="Quality for JPEG images (1-100)"
    )
    IMAGE_DPI: int = Field(
        default=300,
        env="IMAGE_DPI",
        ge=72,
        le=600,
        description="DPI for image extraction"
    )
    IMAGE_SCALE: float = Field(
        default=2.0,
        env="IMAGE_SCALE",
        ge=0.1,
        le=5.0,
        description="Scale factor for images"
    )
    GENERATE_IMAGE_DESCRIPTIONS: bool = Field(
        default=True,
        env="GENERATE_IMAGE_DESCRIPTIONS",
        description="Whether to generate descriptions for images"
    )

    # Advanced Processing Configuration
    EXTRACT_TABLES: bool = Field(
        default=True,
        env="EXTRACT_TABLES",
        description="Whether to extract tables"
    )
    EXTRACT_FORMULAS: bool = Field(
        default=True,
        env="EXTRACT_FORMULAS",
        description="Whether to extract formulas"
    )
    EXTRACT_FIGURES: bool = Field(
        default=True,
        env="EXTRACT_FIGURES",
        description="Whether to extract figures"
    )
    FORCE_FULL_PAGE_OCR: bool = Field(
        default=True,
        env="FORCE_FULL_PAGE_OCR",
        description="Whether to force full page OCR"
    )

    # Batch Processing Configuration
    PARALLEL_PROCESSING: bool = Field(
        default=True,
        env="PARALLEL_PROCESSING",
        description="Enable parallel processing for batch operations"
    )
    MAX_WORKERS: int = Field(
        default=4,
        env="MAX_WORKERS",
        ge=1,
        le=16,
        description="Maximum number of worker threads/processes for parallel processing"
    )
    BATCH_SIZE: int = Field(
        default=10,
        env="BATCH_SIZE",
        ge=1,
        le=100,
        description="Number of files to process in each batch"
    )
    MEMORY_LIMIT_MB: int = Field(
        default=2048,
        env="MEMORY_LIMIT_MB",
        ge=512,
        le=8192,
        description="Memory limit in MB before reducing batch size"
    )

    # OCR Optimization Configuration
    TESSERACT_LOG_LEVEL: int = Field(
        default=1,
        env="TESSERACT_LOG_LEVEL",
        ge=0,
        le=3,
        description="Tesseract logging level (0=errors only, 1=warnings, 2=info, 3=debug)"
    )
    OCR_PSM_MODE: int = Field(
        default=6,
        env="OCR_PSM_MODE", 
        ge=0,
        le=13,
        description="Tesseract Page Segmentation Mode (6=uniform block of text)"
    )
    SKIP_OSD: bool = Field(
        default=True,
        env="SKIP_OSD",
        description="Skip Orientation and Script Detection to reduce OCR warnings"
    )
    TESSDATA_PREFIX: Optional[str] = Field(
        default="/usr/local/share/tessdata",
        env="TESSDATA_PREFIX",
        description="Path to Tesseract language data files"
    )
    OMP_NUM_THREADS: int = Field(
        default=1,
        env="OMP_NUM_THREADS",
        ge=1,
        le=16,
        description="Number of OpenMP threads for Tesseract"
    )

    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level is one of the standard Python logging levels."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @validator("DATABASE_PORT", "NEO4J_PORT")
    def validate_port(cls, v):
        """Validate port numbers are within valid range."""
        if v is not None and (v < 1 or v > 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True


config: Config = Config()
