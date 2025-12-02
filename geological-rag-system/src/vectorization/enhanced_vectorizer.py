"""
ENHANCED SPEM VECTORIZATION ENGINE
================================================================================
Creates comprehensive vector database from extracted SPEM content including:
- Text content with geological context
- Image metadata and descriptions
- Geological keywords and relationships
- Multi-modal search capabilities
================================================================================
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import hashlib
from datetime import datetime
from loguru import logger
import re


@dataclass
class VectorizedChunk:
    """Represents a vectorized content chunk"""
    id: str
    content: str
    content_type: str  # 'text', 'image_description', 'geological_keyword', 'combined'
    page_number: int
    chunk_type: str  # 'geological_text', 'image_metadata', 'keyword_context', 'mixed'
    source_file: str
    geological_terms: List[str]
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    relationships: List[str]  # IDs of related chunks


class EnhancedSPEMVectorizer:
    """Enhanced vectorizer for SPEM geological content with image support"""
    
    def __init__(self, output_path: str = "data/enhanced_spem_vector_db"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.output_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create collections for different content types
        self.collections = {}
        self._initialize_collections()
        
        # Geological knowledge categories
        self.geological_categories = self._load_geological_categories()
        
        logger.info(f"Enhanced SPEM vectorizer initialized. Output: {self.output_path}")
    
    def _initialize_collections(self):
        """Initialize ChromaDB collections for different content types"""
        collection_configs = {
            'spem_text_content': 'Primary text content from SPEM document pages',
            'spem_image_metadata': 'Image descriptions and geological analysis',
            'spem_geological_terms': 'Geological keywords and their contexts',
            'spem_combined_content': 'Multi-modal content combining text and images'
        }
        
        for collection_name, description in collection_configs.items():
            try:
                # Delete existing collection if it exists
                try:
                    existing = self.client.get_collection(collection_name)
                    self.client.delete_collection(collection_name)
                    logger.info(f"Deleted existing collection: {collection_name}")
                except:
                    pass
                
                # Create new collection
                collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=None,  # We'll provide embeddings manually
                    metadata={"description": description}
                )
                self.collections[collection_name] = collection
                logger.info(f"Created collection: {collection_name}")
                
            except Exception as e:
                logger.error(f"Failed to create collection {collection_name}: {e}")
                raise
    
    def _load_geological_categories(self) -> Dict[str, List[str]]:
        """Load geological term categories for enhanced context"""
        return {
            'well_logs': [
                'gamma ray', 'GR', 'resistivity', 'RT', 'RD', 'RS', 'spontaneous potential', 'SP',
                'neutron', 'NPHI', 'density', 'RHOB', 'photoelectric', 'PEF', 'sonic', 'DT',
                'caliper', 'CAL', 'deep resistivity', 'shallow resistivity'
            ],
            'sequence_stratigraphy': [
                'sequence boundary', 'SB', 'maximum flooding surface', 'MFS', 'transgressive surface', 'TS',
                'lowstand systems tract', 'LST', 'transgressive systems tract', 'TST',
                'highstand systems tract', 'HST', 'systems tract', 'parasequence', 'stacking pattern',
                'accommodation space', 'base level', 'relative sea level', 'sequence stratigraphic'
            ],
            'depositional_environments': [
                'marine', 'fluvial', 'delta', 'deltaic', 'channel', 'shoreface', 'beach', 'tidal',
                'barrier', 'shelf', 'basin', 'turbidite', 'submarine fan', 'aeolian', 'lagoonal'
            ],
            'lithology': [
                'sandstone', 'sand', 'shale', 'mudstone', 'limestone', 'dolomite', 'conglomerate',
                'coal', 'anhydrite', 'evaporite', 'calcareous', 'siliceous', 'arkose'
            ],
            'patterns': [
                'funnel pattern', 'bell pattern', 'boxcar pattern', 'bow pattern', 'cleaning up',
                'dirtying up', 'coarsening upward', 'fining upward', 'progradational', 'retrogradational'
            ]
        }
    
    def create_enhanced_vector_database(self, extracted_content_path: str) -> Dict[str, Any]:
        """
        Create comprehensive vector database from extracted SPEM content
        
        Args:
            extracted_content_path: Path to extracted content directory
            
        Returns:
            Vectorization results and statistics
        """
        logger.info("Starting enhanced SPEM vectorization...")
        
        content_path = Path(extracted_content_path)
        if not content_path.exists():
            raise FileNotFoundError(f"Extracted content not found: {content_path}")
        
        # Load extraction results
        extraction_file = content_path / "metadata" / "extraction_results.json"
        with open(extraction_file, 'r', encoding='utf-8') as f:
            extraction_data = json.load(f)
        
        vectorization_stats = {
            'start_time': datetime.now().isoformat(),
            'text_chunks': 0,
            'image_chunks': 0,
            'keyword_chunks': 0,
            'combined_chunks': 0,
            'total_embeddings': 0,
            'collections_created': len(self.collections)
        }
        
        # Process different content types
        text_chunks = self._process_text_content(extraction_data)
        image_chunks = self._process_image_content(extraction_data)
        keyword_chunks = self._process_geological_keywords(extraction_data)
        combined_chunks = self._create_combined_content(text_chunks, image_chunks)
        
        # Add to vector database
        self._add_chunks_to_database(text_chunks, 'spem_text_content')
        self._add_chunks_to_database(image_chunks, 'spem_image_metadata')
        self._add_chunks_to_database(keyword_chunks, 'spem_geological_terms')
        self._add_chunks_to_database(combined_chunks, 'spem_combined_content')
        
        # Update statistics
        vectorization_stats.update({
            'text_chunks': len(text_chunks),
            'image_chunks': len(image_chunks),
            'keyword_chunks': len(keyword_chunks),
            'combined_chunks': len(combined_chunks),
            'total_embeddings': len(text_chunks) + len(image_chunks) + len(keyword_chunks) + len(combined_chunks),
            'end_time': datetime.now().isoformat()
        })
        
        # Save vectorization metadata
        self._save_vectorization_metadata(vectorization_stats, extraction_data)
        
        logger.info(f"Enhanced vectorization complete!")
        logger.info(f"Created {vectorization_stats['total_embeddings']} embeddings across {vectorization_stats['collections_created']} collections")
        
        return vectorization_stats
    
    def _process_text_content(self, extraction_data: Dict) -> List[VectorizedChunk]:
        """Process and chunk text content for vectorization"""
        text_chunks = []
        chunk_id = 0
        
        for page_data in extraction_data['pages']:
            page_num = page_data['page_number']
            
            # Combine text blocks from page
            page_text = ""
            geological_terms = set()
            
            for text_block in page_data['text_blocks']:
                page_text += text_block['text'] + " "
                geological_terms.update(text_block['geological_terms'])
            
            if page_text.strip():
                # Create semantic chunks from page text
                page_chunks = self._create_semantic_chunks(
                    page_text.strip(), 
                    page_num,
                    list(geological_terms)
                )
                
                for chunk_text in page_chunks:
                    extracted_terms = self._extract_geological_terms(chunk_text)
                    chunk = VectorizedChunk(
                        id=f"text_chunk_{chunk_id:04d}",
                        content=chunk_text,
                        content_type='text',
                        page_number=page_num,
                        chunk_type='geological_text',
                        source_file='SPEM_Strata_Log_Signature_SS.pdf',
                        geological_terms=extracted_terms,
                        embedding=None,  # Will be generated later
                        metadata={
                            'page': page_num,
                            'content_length': len(chunk_text),
                            'geological_density': len(extracted_terms) / max(len(chunk_text.split()), 1),
                            'chunk_type': 'text',
                            'geological_terms': ', '.join(extracted_terms),  # Convert list to string
                            'creation_date': datetime.now().isoformat()
                        },
                        relationships=[]
                    )
                    
                    text_chunks.append(chunk)
                    chunk_id += 1
        
        return text_chunks
    
    def _process_image_content(self, extraction_data: Dict) -> List[VectorizedChunk]:
        """Process image metadata and descriptions for vectorization"""
        image_chunks = []
        chunk_id = 0
        
        for page_data in extraction_data['pages']:
            page_num = page_data['page_number']
            
            for image_data in page_data['images']:
                # Create comprehensive image description
                image_description = self._create_image_description(image_data)
                
                extracted_terms = self._extract_geological_terms(image_description)
                chunk = VectorizedChunk(
                    id=f"image_chunk_{chunk_id:04d}",
                    content=image_description,
                    content_type='image_description',
                    page_number=page_num,
                    chunk_type='image_metadata',
                    source_file=image_data['file_path'],
                    geological_terms=extracted_terms,
                    embedding=None,
                    metadata={
                        'page': page_num,
                        'image_id': image_data['id'],
                        'image_type': image_data['image_type'],
                        'image_size': f"{image_data['width']}x{image_data['height']}",
                        'caption': image_data['caption'],
                        'geological_content': image_data['geological_content'],
                        'file_path': image_data['file_path'],
                        'geological_terms': ', '.join(extracted_terms),  # Convert list to string
                        'creation_date': datetime.now().isoformat()
                    },
                    relationships=[]
                )
                
                image_chunks.append(chunk)
                chunk_id += 1
        
        return image_chunks
    
    def _process_geological_keywords(self, extraction_data: Dict) -> List[VectorizedChunk]:
        """Process geological keywords with enhanced context"""
        keyword_chunks = []
        chunk_id = 0
        
        # Get all geological keywords found
        all_keywords = extraction_data['summary']['geological_keywords_found']
        
        # Create keyword context chunks
        for category, terms in self.geological_categories.items():
            found_terms = [kw for kw in all_keywords if kw.lower() in [t.lower() for t in terms]]
            
            if found_terms:
                # Create context for this category
                keyword_context = self._create_keyword_context(category, found_terms, extraction_data)
                
                chunk = VectorizedChunk(
                    id=f"keyword_chunk_{chunk_id:04d}",
                    content=keyword_context,
                    content_type='geological_keyword',
                    page_number=0,  # Keywords span multiple pages
                    chunk_type='keyword_context',
                    source_file='SPEM_geological_keywords',
                    geological_terms=found_terms,
                    embedding=None,
                    metadata={
                        'keyword_category': category,
                        'terms_count': len(found_terms),
                        'terms': ', '.join(found_terms),  # Convert list to string
                        'creation_date': datetime.now().isoformat()
                    },
                    relationships=[]
                )
                
                keyword_chunks.append(chunk)
                chunk_id += 1
        
        return keyword_chunks
    
    def _create_combined_content(self, text_chunks: List[VectorizedChunk], 
                                image_chunks: List[VectorizedChunk]) -> List[VectorizedChunk]:
        """Create combined content chunks linking text and images by page"""
        combined_chunks = []
        chunk_id = 0
        
        # Group chunks by page
        text_by_page = {}
        images_by_page = {}
        
        for chunk in text_chunks:
            page = chunk.page_number
            if page not in text_by_page:
                text_by_page[page] = []
            text_by_page[page].append(chunk)
        
        for chunk in image_chunks:
            page = chunk.page_number
            if page not in images_by_page:
                images_by_page[page] = []
            images_by_page[page].append(chunk)
        
        # Create combined content for each page
        for page_num in range(1, 24):  # 23 pages total
            if page_num in text_by_page or page_num in images_by_page:
                combined_content = self._create_page_combined_content(
                    page_num,
                    text_by_page.get(page_num, []),
                    images_by_page.get(page_num, [])
                )
                
                if combined_content:
                    extracted_terms = self._extract_geological_terms(combined_content)
                    chunk = VectorizedChunk(
                        id=f"combined_chunk_{chunk_id:04d}",
                        content=combined_content,
                        content_type='combined',
                        page_number=page_num,
                        chunk_type='mixed',
                        source_file='SPEM_combined_content',
                        geological_terms=extracted_terms,
                        embedding=None,
                        metadata={
                            'page': page_num,
                            'has_text': page_num in text_by_page,
                            'has_images': page_num in images_by_page,
                            'text_chunks': len(text_by_page.get(page_num, [])),
                            'image_chunks': len(images_by_page.get(page_num, [])),
                            'geological_terms': ', '.join(extracted_terms),  # Convert list to string
                            'creation_date': datetime.now().isoformat()
                        },
                        relationships=[]
                    )
                    
                    combined_chunks.append(chunk)
                    chunk_id += 1
        
        return combined_chunks
    
    def _create_semantic_chunks(self, text: str, page_num: int, 
                               geological_terms: List[str]) -> List[str]:
        """Create semantic chunks from text based on geological concepts"""
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_geological_terms = set()
        
        for sentence in sentences:
            sentence_terms = self._extract_geological_terms(sentence)
            
            # If adding this sentence would make chunk too long or change geological focus
            if (len(current_chunk) + len(sentence) > 500 or 
                (chunk_geological_terms and 
                 len(set(sentence_terms) & chunk_geological_terms) == 0 and 
                 len(sentence_terms) > 0)):
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                current_chunk = sentence
                chunk_geological_terms = set(sentence_terms)
            else:
                current_chunk += " " + sentence
                chunk_geological_terms.update(sentence_terms)
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _create_image_description(self, image_data: Dict) -> str:
        """Create comprehensive description for image"""
        description_parts = [
            f"Geological image from SPEM document: {image_data['image_type']}",
            f"Page {image_data.get('page_number', 'unknown')}",
            f"Dimensions: {image_data['width']}x{image_data['height']}"
        ]
        
        if image_data['caption']:
            description_parts.append(f"Caption: {image_data['caption']}")
        
        if image_data['geological_content']:
            description_parts.append(f"Geological content: {image_data['geological_content']}")
        
        if image_data['context_text']:
            context = image_data['context_text'][:200] + "..." if len(image_data['context_text']) > 200 else image_data['context_text']
            description_parts.append(f"Context: {context}")
        
        # Add metadata insights
        if 'metadata' in image_data and 'geological_indicators' in image_data['metadata']:
            indicators = image_data['metadata']['geological_indicators']
            if indicators:
                description_parts.append(f"Visual indicators: {', '.join(indicators)}")
        
        return " | ".join(description_parts)
    
    def _create_keyword_context(self, category: str, terms: List[str], 
                               extraction_data: Dict) -> str:
        """Create context for geological keywords"""
        context_parts = [
            f"Geological concept category: {category.replace('_', ' ').title()}",
            f"Terms found: {', '.join(terms)}"
        ]
        
        # Find context from pages where these terms appear
        term_contexts = []
        for page_data in extraction_data['pages']:
            for text_block in page_data['text_blocks']:
                text = text_block['text'].lower()
                for term in terms:
                    if term.lower() in text:
                        # Extract sentence containing the term
                        sentences = re.split(r'[.!?]+', text_block['text'])
                        for sentence in sentences:
                            if term.lower() in sentence.lower():
                                term_contexts.append(sentence.strip())
                                break
        
        if term_contexts:
            # Use first few unique contexts
            unique_contexts = list(dict.fromkeys(term_contexts))[:3]
            context_parts.append("Usage contexts: " + " | ".join(unique_contexts))
        
        return " | ".join(context_parts)
    
    def _create_page_combined_content(self, page_num: int, 
                                     text_chunks: List[VectorizedChunk],
                                     image_chunks: List[VectorizedChunk]) -> str:
        """Create combined content description for a page"""
        combined_parts = [f"SPEM Document Page {page_num}"]
        
        if text_chunks:
            # Combine text from all chunks on page
            page_text = " ".join([chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content 
                                 for chunk in text_chunks])
            combined_parts.append(f"Text content: {page_text}")
        
        if image_chunks:
            # Describe images on page
            image_descriptions = []
            for img_chunk in image_chunks:
                img_type = img_chunk.metadata.get('image_type', 'unknown')
                img_geological = img_chunk.metadata.get('geological_content', '')
                image_descriptions.append(f"{img_type} ({img_geological})")
            
            combined_parts.append(f"Images: {', '.join(image_descriptions)}")
        
        return " | ".join(combined_parts)
    
    def _extract_geological_terms(self, text: str) -> List[str]:
        """Extract geological terms from text"""
        found_terms = []
        text_lower = text.lower()
        
        for category, terms in self.geological_categories.items():
            for term in terms:
                if term.lower() in text_lower:
                    found_terms.append(term)
        
        return list(set(found_terms))
    
    def _add_chunks_to_database(self, chunks: List[VectorizedChunk], collection_name: str):
        """Add vectorized chunks to ChromaDB collection"""
        if not chunks:
            logger.warning(f"No chunks to add to collection: {collection_name}")
            return
        
        collection = self.collections[collection_name]
        
        logger.info(f"Adding {len(chunks)} chunks to collection: {collection_name}")
        
        # Generate embeddings in batches
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # Generate embeddings for batch
            batch_texts = [chunk.content for chunk in batch_chunks]
            batch_embeddings = self.embedding_model.encode(
                batch_texts, 
                convert_to_numpy=True,
                show_progress_bar=True
            ).tolist()
            
            # Prepare data for ChromaDB
            documents = []
            embeddings = []
            metadatas = []
            ids = []
            
            for j, chunk in enumerate(batch_chunks):
                chunk.embedding = batch_embeddings[j]
                
                documents.append(chunk.content)
                embeddings.append(batch_embeddings[j])
                metadatas.append(chunk.metadata)
                ids.append(chunk.id)
            
            # Add to collection
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.debug(f"Added batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} to {collection_name}")
        
        logger.info(f"Successfully added {len(chunks)} chunks to {collection_name}")
    
    def _save_vectorization_metadata(self, stats: Dict, extraction_data: Dict):
        """Save vectorization metadata and statistics"""
        metadata_file = self.output_path / "vectorization_metadata.json"
        
        metadata = {
            'vectorization_stats': stats,
            'source_document': extraction_data['document_info'],
            'collections': {
                name: {
                    'description': collection.metadata,
                    'count': collection.count()
                }
                for name, collection in self.collections.items()
            },
            'embedding_model': 'all-MiniLM-L6-v2',
            'embedding_dimension': self.embedding_dimension,
            'geological_categories': list(self.geological_categories.keys()),
            'creation_date': datetime.now().isoformat()
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Vectorization metadata saved to: {metadata_file}")


def main():
    """Test the enhanced SPEM vectorizer"""
    vectorizer = EnhancedSPEMVectorizer("data/enhanced_spem_vector_db")
    
    try:
        # Vectorize extracted content
        stats = vectorizer.create_enhanced_vector_database("extracted_spem_complete")
        
        print("\nENHANCED VECTORIZATION COMPLETE!")
        print("=" * 60)
        print(f"Text chunks: {stats['text_chunks']}")
        print(f"Image chunks: {stats['image_chunks']}")
        print(f"Keyword chunks: {stats['keyword_chunks']}")
        print(f"Combined chunks: {stats['combined_chunks']}")
        print(f"Total embeddings: {stats['total_embeddings']}")
        print(f"Collections: {stats['collections_created']}")
        
    except Exception as e:
        print(f"Vectorization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()