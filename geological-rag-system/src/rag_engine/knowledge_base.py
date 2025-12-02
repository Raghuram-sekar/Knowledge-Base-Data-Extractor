"""
SPEM Knowledge Base Builder
Processes the SPEM document and creates a vectorized knowledge base
"""

import os
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger
import chromadb
from sentence_transformers import SentenceTransformer
import re
from collections import defaultdict

try:
    from ..config import settings
except ImportError:
    # Fallback for when running as a script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import settings


@dataclass
class SPEMChunk:
    """Individual knowledge chunk from SPEM document"""
    id: str
    content: str
    page_number: int
    section: str
    chunk_type: str  # 'pattern', 'surface', 'environment', 'general'
    metadata: Dict[str, Any]


class SPEMKnowledgeBuilder:
    """Builds and manages SPEM knowledge base"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.vector_db = None
        self.collection = None
    
    def build_knowledge_base(self) -> bool:
        """
        Complete pipeline to build SPEM knowledge base
        
        Returns:
            Success status
        """
        logger.info("Building SPEM knowledge base...")
        
        try:
            # Extract text from SPEM PDF
            spem_text = self._extract_spem_text()
            
            # Create knowledge chunks
            chunks = self._create_knowledge_chunks(spem_text)
            
            # Initialize vector database
            self._initialize_vector_db()
            
            # Add chunks to vector database
            self._add_chunks_to_db(chunks)
            
            logger.info(f"Successfully built knowledge base with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build knowledge base: {str(e)}")
            return False
    
    def _extract_spem_text(self) -> Dict[int, str]:
        """Extract text from SPEM PDF document"""
        spem_path = settings.SPEM_DOCUMENT_PATH
        
        if not spem_path.exists():
            raise FileNotFoundError(f"SPEM document not found: {spem_path}")
        
        logger.info(f"Extracting text from: {spem_path}")
        
        doc_text = {}
        doc = fitz.open(spem_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():  # Only store pages with content
                doc_text[page_num + 1] = text.strip()
                logger.debug(f"Extracted {len(text)} characters from page {page_num + 1}")
        
        doc.close()
        logger.info(f"Extracted text from {len(doc_text)} pages")
        
        return doc_text
    
    def _create_knowledge_chunks(self, doc_text: Dict[int, str]) -> List[SPEMChunk]:
        """Create structured knowledge chunks from SPEM text"""
        chunks = []
        chunk_id = 0
        
        for page_num, text in doc_text.items():
            # Clean and preprocess text
            cleaned_text = self._clean_text(text)
            
            # Split into logical sections
            sections = self._split_into_sections(cleaned_text, page_num)
            
            for section in sections:
                # Create overlapping chunks within each section
                section_chunks = self._create_overlapping_chunks(
                    section['content'], 
                    section, 
                    page_num
                )
                
                for chunk_content in section_chunks:
                    chunk_type = self._classify_chunk_type(chunk_content)
                    
                    chunk = SPEMChunk(
                        id=f"spem_chunk_{chunk_id:04d}",
                        content=chunk_content,
                        page_number=page_num,
                        section=section['title'],
                        chunk_type=chunk_type,
                        metadata={
                            'source': 'SPEM',
                            'page': page_num,
                            'section': section['title'],
                            'type': chunk_type,
                            'length': len(chunk_content),
                            'keywords': ', '.join(self._extract_keywords(chunk_content))  # Convert list to string
                        }
                    )
                    
                    chunks.append(chunk)
                    chunk_id += 1
        
        logger.info(f"Created {len(chunks)} knowledge chunks")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Fix common OCR issues
        text = text.replace('•', '* ')
        text = text.replace('–', '-')
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def _split_into_sections(self, text: str, page_num: int) -> List[Dict]:
        """Split text into logical sections"""
        sections = []
        
        # Define section patterns based on SPEM document structure
        section_patterns = [
            r'Well logs? and sequence stratigraphic interpretation',
            r'Well Log Interpretation',
            r'Log Response Patterns?',
            r'Depositional Environment',
            r'Sequence Stratigraphy',
            r'Systems Tract',
            r'Maximum Flooding Surface',
            r'Transgressive Surface',
            r'Sequence Boundary',
            r'Gamma Ray Response',
            r'Resistivity',
            r'Spontaneous Potential',
            r'Neutron',
            r'Density',
            r'Sonic',
            r'Funnel.{0,20}Pattern',
            r'Bell.{0,20}Pattern',
            r'Boxcar.{0,20}Pattern',
            r'La Pascua Formation',
            r'Venezuela',
            r'Guarico',
        ]
        
        # Try to identify sections by headers
        current_section = {'title': 'General', 'content': ''}
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            is_header = False
            for pattern in section_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Save previous section if it has content
                    if current_section['content'].strip():
                        sections.append(current_section.copy())
                    
                    # Start new section
                    current_section = {'title': line[:100], 'content': ''}
                    is_header = True
                    break
            
            if not is_header:
                current_section['content'] += line + ' '
        
        # Add the last section
        if current_section['content'].strip():
            sections.append(current_section)
        
        # If no sections were found, treat entire page as one section
        if not sections:
            sections.append({
                'title': f'Page {page_num}',
                'content': text
            })
        
        return sections
    
    def _create_overlapping_chunks(self, text: str, section: Dict, page_num: int) -> List[str]:
        """Create overlapping text chunks"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the current chunk
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                sentence_ends = [i for i, char in enumerate(text[max(0, end-100):end], max(0, end-100)) 
                                if char in '.!?']
                if sentence_ends:
                    end = sentence_ends[-1] + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
            
            # Avoid infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def _classify_chunk_type(self, content: str) -> str:
        """Classify chunk type based on content"""
        content_lower = content.lower()
        
        # Pattern-related keywords
        pattern_keywords = ['funnel', 'bell', 'boxcar', 'cylindrical', 'pattern', 'trend', 
                           'coarsening', 'fining', 'upward', 'cleaning', 'dirtying']
        
        # Surface-related keywords
        surface_keywords = ['maximum flooding', 'transgressive surface', 'sequence boundary',
                           'mfs', 'ts', 'sb', 'unconformity', 'flooding', 'boundary']
        
        # Environment-related keywords
        environment_keywords = ['shoreface', 'delta', 'channel', 'tidal', 'barrier', 'beach',
                               'marine', 'fluvial', 'depositional', 'environment', 'facies']
        
        # Count keyword matches
        pattern_score = sum(1 for kw in pattern_keywords if kw in content_lower)
        surface_score = sum(1 for kw in surface_keywords if kw in content_lower)
        environment_score = sum(1 for kw in environment_keywords if kw in content_lower)
        
        # Classify based on highest score
        if pattern_score > surface_score and pattern_score > environment_score:
            return 'pattern'
        elif surface_score > environment_score:
            return 'surface'
        elif environment_score > 0:
            return 'environment'
        else:
            return 'general'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key geological terms from text"""
        geological_terms = [
            'gamma ray', 'resistivity', 'spontaneous potential', 'neutron', 'density',
            'funnel', 'bell', 'boxcar', 'pattern', 'trend',
            'maximum flooding surface', 'transgressive surface', 'sequence boundary',
            'lowstand', 'transgressive', 'highstand', 'systems tract',
            'shoreface', 'delta', 'channel', 'tidal', 'barrier', 'beach',
            'coarsening', 'fining', 'upward', 'prograding', 'retrograding',
            'sand', 'shale', 'limestone', 'reservoir', 'seal'
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in geological_terms:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms
    
    def _initialize_vector_db(self):
        """Initialize ChromaDB vector database"""
        try:
            # Create persistent client
            self.vector_db = chromadb.PersistentClient(
                path=str(settings.VECTOR_DB_PATH)
            )
            
            # Create or get collection
            collection_name = "spem_geological_knowledge"
            
            try:
                # Try to delete existing collection first
                try:
                    existing_collection = self.vector_db.get_collection(name=collection_name)
                    self.vector_db.delete_collection(name=collection_name)
                    logger.info("Deleted existing collection")
                except Exception:
                    logger.info("No existing collection to delete")
                
                # Create new collection
                self.collection = self.vector_db.create_collection(
                    name=collection_name,
                    embedding_function=None  # We'll provide embeddings manually
                )
                
            except Exception as e:
                logger.error(f"Failed to create collection: {e}")
                raise
            
            logger.info("Initialized vector database")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {str(e)}")
            raise
    
    def _add_chunks_to_db(self, chunks: List[SPEMChunk]):
        """Add knowledge chunks to vector database"""
        if not self.collection:
            raise RuntimeError("Vector database not initialized")
        
        logger.info("Generating embeddings and adding chunks to database...")
        
        # Prepare data for batch insertion
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        batch_size = 100
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk.content)
            metadatas.append(chunk.metadata)
            ids.append(chunk.id)
            
            # Process in batches
            if len(documents) >= batch_size or i == len(chunks) - 1:
                # Generate embeddings
                batch_embeddings = self.embedding_model.encode(
                    documents, 
                    convert_to_numpy=True,
                    show_progress_bar=True
                ).tolist()
                
                embeddings.extend(batch_embeddings)
                
                # Add to database
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.debug(f"Added batch of {len(documents)} chunks to database")
                
                # Reset batch
                documents = []
                embeddings = []
                metadatas = []
                ids = []
        
        logger.info(f"Successfully added {len(chunks)} chunks to vector database")


class SPEMKnowledgeRetriever:
    """Retrieves relevant SPEM knowledge for queries"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.vector_db = None
        self.collection = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize connection to existing vector database"""
        try:
            self.vector_db = chromadb.PersistentClient(
                path=str(settings.VECTOR_DB_PATH)
            )
            
            self.collection = self.vector_db.get_collection(
                name="spem_geological_knowledge"
            )
            
            logger.info("Connected to SPEM knowledge base")
            
        except Exception as e:
            logger.error(f"Failed to connect to knowledge base: {str(e)}")
            raise
    
    def search_knowledge(self, query: str, n_results: int = 5, 
                        filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant SPEM knowledge
        
        Args:
            query: Natural language query
            n_results: Number of results to return
            filter_criteria: Optional metadata filters
            
        Returns:
            List of relevant knowledge chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Build where clause for filtering
            where_clause = {"source": "SPEM"}
            if filter_criteria:
                where_clause.update(filter_criteria)
            
            # Search vector database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            
            for i in range(len(results['documents'][0])):
                result = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'page': results['metadatas'][0][i].get('page', 'Unknown'),
                    'section': results['metadatas'][0][i].get('section', 'Unknown'),
                    'type': results['metadatas'][0][i].get('type', 'general')
                }
                formatted_results.append(result)
            
            # Filter by similarity threshold
            filtered_results = [
                r for r in formatted_results 
                if r['similarity'] >= settings.SIMILARITY_THRESHOLD
            ]
            
            logger.debug(f"Found {len(filtered_results)} relevant knowledge chunks for query: {query[:50]}...")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {str(e)}")
            return []
    
    def search_by_pattern_type(self, pattern_type: str) -> List[Dict[str, Any]]:
        """Search for knowledge about specific log patterns"""
        pattern_queries = {
            'funnel': 'funnel pattern cleaning up trend coarsening upward prograding shoreline',
            'bell': 'bell pattern dirtying up trend fining upward transgressive deepening',
            'boxcar': 'boxcar cylindrical pattern channel sand turbidite fluvial',
            'bow': 'bow pattern symmetrical barrel progradation retrogradation',
            'irregular': 'irregular pattern aggradation shale silt'
        }
        
        query = pattern_queries.get(pattern_type, pattern_type)
        
        return self.search_knowledge(
            query=query,
            filter_criteria={"type": "pattern"}
        )
    
    def search_by_surface_type(self, surface_type: str) -> List[Dict[str, Any]]:
        """Search for knowledge about sequence stratigraphic surfaces"""
        surface_queries = {
            'mfs': 'maximum flooding surface radioactive shale marine condensed',
            'ts': 'transgressive surface sea level rise marine flooding',
            'sb': 'sequence boundary unconformity erosion incised valley lowstand',
            'rs': 'ravinement surface wave erosion transgressive lag'
        }
        
        query = surface_queries.get(surface_type.lower(), surface_type)
        
        return self.search_knowledge(
            query=query,
            filter_criteria={"type": "surface"}
        )
    
    def search_by_environment(self, environment: str) -> List[Dict[str, Any]]:
        """Search for knowledge about depositional environments"""
        return self.search_knowledge(
            query=f"{environment} depositional environment facies characteristics",
            filter_criteria={"type": "environment"}
        )