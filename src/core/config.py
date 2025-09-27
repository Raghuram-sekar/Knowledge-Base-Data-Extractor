from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    # Project Configurations
    PROJECT_NAME = "Knowledge Base Data Extractor"
    PROJECT_DESCRIPTION = "A tool for extracting data from knowledge bases"
    VERSION = "1.0.0"

    # Database Configuration
    DATABASE_NAME = os.getenv("DATABASE_NAME")
    DATABASE_USERNAME = os.getenv("DATABASE_USERNAME")
    DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")
    DATABASE_PORT = os.getenv("DATABASE_PORT")

    # S3 Configuration
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
    S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
    S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
    S3_REGION = os.getenv("S3_REGION")
    S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Neo4j Configuration
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_PORT = os.getenv("NEO4J_PORT")

    # Parsing Configuration
    INPUT_STORAGE = os.getenv("INPUT_STORAGE", "LOCAL")  # Options: LOCAL, S3
    OUTPUT_STORAGE = os.getenv("OUTPUT_STORAGE", "LOCAL")  # Options: LOCAL, S3
    OCR_ENGINE = os.getenv("OCR_ENGINE", "TESSERACT")  # Options: TESSERACT, EASYOCR

    # Input/Output Path Configuration
    INPUT_PATH = os.getenv("INPUT_PATH", "data/raw")  # Input directory path (local or S3)
    OUTPUT_PATH = os.getenv("OUTPUT_PATH", "data/processed")  # Output directory path (local or S3)

    # Image Processing Configuration
    SAVE_IMAGES = os.getenv("SAVE_IMAGES", "true").lower() == "true"
    IMAGE_FORMAT = os.getenv("IMAGE_FORMAT", "PNG")  # PNG, JPEG, WEBP
    IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY", "95"))  # 1-100 for JPEG
    IMAGE_DPI = int(os.getenv("IMAGE_DPI", "300"))  # DPI for image extraction
    IMAGE_SCALE = float(os.getenv("IMAGE_SCALE", "2.0"))  # Scale factor for images
    GENERATE_IMAGE_DESCRIPTIONS = os.getenv("GENERATE_IMAGE_DESCRIPTIONS", "true").lower() == "true"

    # Advanced Processing Configuration
    EXTRACT_TABLES = os.getenv("EXTRACT_TABLES", "true").lower() == "true"
    EXTRACT_FORMULAS = os.getenv("EXTRACT_FORMULAS", "true").lower() == "true"
    EXTRACT_FIGURES = os.getenv("EXTRACT_FIGURES", "true").lower() == "true"
    FORCE_FULL_PAGE_OCR = os.getenv("FORCE_FULL_PAGE_OCR", "true").lower() == "true"



config: Config = Config()
