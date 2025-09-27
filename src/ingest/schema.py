from typing import List, Optional
from pydantic import BaseModel, HttpUrl, Field
from datetime import date

# -----------------------------
# Document Identification Schema
# -----------------------------
class DocumentIdentification(BaseModel):
    Document_ID: str = Field(..., description="Unique internal identifier for the document (e.g., DOC-0001)")
    External_ID: Optional[str] = Field(..., description="Standardized external identifier like DOI, ISBN, or URL")
    External_ID_Type: Optional[str] = Field(..., description="Type of external ID (DOI, ISBN, URL, arXiv ID, etc.)")
    Title: str = Field(..., description="Title of the book or article")
    Authors: List[str] = Field(..., description="List of authors formatted consistently")
    Publication_Date: Optional[date] = Field(..., description="Publication date (YYYY-MM-DD)")
    Publisher: Optional[str] = Field(..., description="Publisher name")
    Source_Origin: Optional[str] = Field(..., description="Where the document was obtained from (e.g., JSTOR)")

# -----------------------------
# Document Classification Schema
# -----------------------------
class DocumentClassification(BaseModel):
    Document_Type: str = Field(..., description="Type of document (Book, Journal Article, Thesis, etc.)")
    Source_Format: str = Field(..., description="Format of the source (Born-Digital, Scanned, Scanned-OCR)")
    Language: str = Field(..., description="Primary language code, e.g., 'en', 'fr'")
    Keywords: Optional[List[str]] = Field(default_factory=list, description="Keywords or tags for searching")

# -----------------------------
# Extracted Data Schema
# -----------------------------
class ExtractedData(BaseModel):
    Extraction_ID: str = Field(..., description="Unique identifier for extracted data (e.g., EXT-00001)")
    Parent_Document_ID: str = Field(..., description="Link back to the source document's Document_ID")
    Extraction_Type: str = Field(..., description="Type of extracted content (Paragraph, Abstract, Table, etc.)")
    Location_in_Source: Optional[str] = Field(None, description="Precise location of the extracted content")
    Page_Number: Optional[int] = Field(None, description="Page number in the source")
    Section_Title: Optional[str] = Field(None, description="Section or chapter title")
    Paragraph_Number: Optional[int] = Field(None, description="Paragraph number in the section or page")
    Figure_Table_Number: Optional[str] = Field(None, description="Figure or table identifier (e.g., Figure 3)")
    Content_Caption: Optional[str] = Field(None, description="Caption or title for figure/table")
    Content_Text: Optional[str] = Field(None, description="Extracted text or structured data")
    Confidence_Score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score (0.0 - 1.0)")

# -----------------------------
# Master Schema (links everything)
# -----------------------------
class DocumentMetadata(BaseModel):
    identification: DocumentIdentification
    classification: DocumentClassification
    extractions: List[ExtractedData] = Field(default_factory=list)
