"""
ENHANCED SPEM RAG QUERY ENGINE
================================================================================
Multi-modal RAG system for querying vectorized SPEM geological content.
Supports searching across:
- Text content with geological context
- Image descriptions and metadata
- Geological keywords and relationships
- Combined multi-modal content
================================================================================
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
from loguru import logger
import re
from dataclasses import dataclass


@dataclass
class QueryResult:
    """Represents a query result with context"""
    content: str
    content_type: str  # 'text', 'image', 'keyword', 'combined'
    page_number: int
    relevance_score: float
    geological_terms: List[str]
    metadata: Dict[str, Any]
    source_collection: str


class EnhancedSPEMQueryEngine:
    """Enhanced RAG query engine for SPEM geological content"""
    
    def __init__(self, vector_db_path: str = "data/enhanced_spem_vector_db"):
        self.vector_db_path = Path(vector_db_path)
        
        if not self.vector_db_path.exists():
            raise FileNotFoundError(f"Vector database not found: {vector_db_path}")
        
        # Initialize embedding model (same as used for indexing)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.vector_db_path))
        
        # Load collections
        self.collections = {}
        self._load_collections()
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        logger.info(f"Enhanced SPEM query engine initialized with {len(self.collections)} collections")
    
    def _load_collections(self):
        """Load all available collections"""
        collection_names = [
            'spem_text_content',
            'spem_image_metadata', 
            'spem_geological_terms',
            'spem_combined_content'
        ]
        
        for name in collection_names:
            try:
                collection = self.client.get_collection(name)
                self.collections[name] = collection
                logger.info(f"Loaded collection: {name} ({collection.count()} items)")
            except Exception as e:
                logger.warning(f"Could not load collection {name}: {e}")
    
    def _load_metadata(self) -> Dict:
        """Load vectorization metadata"""
        metadata_file = self.vector_db_path / "vectorization_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def search_all_content(self, query: str, max_results: int = 50, 
                          similarity_threshold: float = 0.2) -> List[QueryResult]:
        """
        Search across all collections for relevant content with improved accuracy
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score for results
            
        Returns:
            List of QueryResult objects ranked by relevance
        """
        logger.info(f"Searching all content for: '{query}'")
        
        all_results = []
        
        # First, try exact string matching for better accuracy
        exact_results = self._exact_string_search(query, max_results)
        all_results.extend(exact_results)
        
        # Generate query embedding for vector search
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search each collection
        for collection_name, collection in self.collections.items():
            try:
                # Query collection
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(max_results, collection.count()),
                    include=['documents', 'metadatas', 'distances']
                )
                
                # Convert to QueryResult objects
                if results['documents'][0]:  # Check if results exist
                    for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0], 
                        results['distances'][0]
                    )):
                        # STRICT FILTERING: Only include results that actually contain the search term
                        query_lower = query.lower().strip()
                        doc_lower = doc.lower()
                        
                        # Skip if document doesn't contain the search term
                        if query_lower not in doc_lower:
                            continue
                            
                        # Use real ChromaDB vector similarity instead of complex calculation
                        similarity = max(0.0, 1.0 - (distance / 2.0)) if distance < 2.0 else 0.0
                        
                        # Only include results with meaningful similarity
                        if similarity >= similarity_threshold:
                            query_result = QueryResult(
                                content=doc,
                                content_type=metadata.get('chunk_type', 'unknown'),
                                page_number=metadata.get('page', 0),
                                relevance_score=similarity,
                                geological_terms=self._extract_geological_terms(doc),
                                metadata=metadata,
                                source_collection=collection_name
                            )
                            all_results.append(query_result)
                
            except Exception as e:
                if "hnsw segment reader" in str(e).lower():
                    logger.debug(f"HNSW index not available for {collection_name}, using fallback search")
                else:
                    logger.error(f"Error searching collection {collection_name}: {e}")
                # Try to continue with other collections instead of failing completely
                continue
        
        # Remove duplicates based on content
        unique_results = self._remove_duplicate_results(all_results)
        
        # Sort by relevance score
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Apply content-type diversity
        final_results = self._diversify_results(unique_results, max_results)
        
        logger.info(f"Found {len(final_results)} relevant results")
        return final_results
    
    def _exact_string_search(self, query: str, max_results: int) -> List[QueryResult]:
        """Exact string search for better accuracy with short queries"""
        exact_results = []
        query_lower = query.lower().strip()
        
        # Skip exact search for very long queries
        if len(query.split()) > 3:
            return exact_results
        
        # Create intelligent variations for any geological term
        query_variations = [query_lower]
        
        # Add intelligent geological term variations
        query_variations.extend(self._generate_geological_variations(query_lower))
            
        for collection_name, collection in self.collections.items():
            try:
                all_docs = collection.get()
                
                for doc, metadata in zip(all_docs['documents'], all_docs['metadatas']):
                    doc_lower = doc.lower()
                    
                    # Different matching logic for images vs text
                    is_image_collection = "image" in collection_name
                    matched = False
                    similarity = 0.0
                    
                    if is_image_collection:
                        # For images: use intelligent semantic matching
                        matched, similarity = self._intelligent_geological_match(query_lower, doc_lower)
                    else:
                        # For text: use flexible variation matching
                        for variation in query_variations:
                            if variation in doc:  # Exact case match
                                similarity = 0.95
                                matched = True
                                break
                            elif variation in doc_lower:  # Case-insensitive match
                                similarity = 0.85
                                matched = True
                                break
                    
                    if matched:
                        # Try multiple metadata keys for page number
                        page_num = (metadata.get('page_number') or 
                                  metadata.get('page') or 
                                  self._extract_page_from_content(doc) or 0)
                        
                        # Map collection name to content type
                        content_type_map = {
                            'spem_text_content': 'text',
                            'spem_image_metadata': 'image', 
                            'spem_geological_terms': 'geological',
                            'spem_combined_content': 'combined'
                        }
                        content_type = content_type_map.get(collection_name, 'text')
                                  
                        query_result = QueryResult(
                            content=doc,
                            content_type=content_type,
                            page_number=page_num,
                            relevance_score=similarity,
                            geological_terms=self._extract_geological_terms(doc),
                            metadata=metadata,
                            source_collection=collection_name
                        )
                        exact_results.append(query_result)
                    
            except Exception as e:
                logger.error(f"Error in exact search for {collection_name}: {e}")
                continue
        
        return exact_results
    
    def _extract_page_from_content(self, content: str) -> int:
        """Extract page number from content text as fallback"""
        import re
        
        # Look for patterns like "Page 10", "page 12", etc.
        page_patterns = [
            r'Page (\d+)',
            r'page (\d+)', 
            r'\| Page (\d+) \|',
            r'Page: (\d+)'
        ]
        
        for pattern in page_patterns:
            match = re.search(pattern, content)
            if match:
                return int(match.group(1))
                
        return 0

    def _generate_geological_variations(self, term: str) -> List[str]:
        """Intelligently generate variations by analyzing the database content"""
        variations = []
        
        # Basic linguistic variations
        words = term.split()
        if len(words) > 1:
            # Add underscore version
            variations.append("_".join(words))
            # Add individual words for partial matching
            variations.extend(words)
        
        # Generate potential abbreviations
        if len(words) > 1:
            # First letter of each word
            abbrev = "".join(word[0].lower() for word in words)
            variations.append(abbrev)
            
        # Analyze database to find related terms
        related_terms = self._analyze_database_for_related_terms(term)
        variations.extend(related_terms)
        
        # Remove duplicates and original
        variations = list(set(variations))
        if term in variations:
            variations.remove(term)
            
        return variations
    
    def _analyze_database_for_related_terms(self, query_term: str) -> List[str]:
        """Analyze the database to find terms that appear in similar contexts"""
        related_terms = []
        query_words = set(query_term.lower().split())
        
        try:
            # Search across all collections to find contextually related terms
            for collection_name, collection in self.collections.items():
                all_docs = collection.get()
                
                for doc in all_docs['documents']:
                    doc_lower = doc.lower()
                    
                    # If document contains any word from our query
                    if any(word in doc_lower for word in query_words):
                        # Extract potential related terms from the same document
                        # Look for patterns like abbreviations in parentheses
                        import re
                        
                        # Find abbreviations in parentheses
                        abbrev_pattern = r'\(([A-Z]{1,4})\)'
                        abbreviations = re.findall(abbrev_pattern, doc)
                        related_terms.extend([abbrev.lower() for abbrev in abbreviations])
                        
                        # Find underscore separated terms
                        underscore_pattern = r'\b\w+_\w+\b'
                        underscore_terms = re.findall(underscore_pattern, doc_lower)
                        related_terms.extend(underscore_terms)
                        
                        # Find terms that appear near our query words
                        for query_word in query_words:
                            if query_word in doc_lower:
                                # Find words within 20 characters of our query word
                                start_pos = doc_lower.find(query_word)
                                if start_pos != -1:
                                    context_start = max(0, start_pos - 20)
                                    context_end = min(len(doc_lower), start_pos + len(query_word) + 20)
                                    context = doc_lower[context_start:context_end]
                                    
                                    # Extract meaningful terms from context
                                    context_words = re.findall(r'\b[a-z_]{2,10}\b', context)
                                    related_terms.extend(context_words)
        
        except Exception as e:
            # If analysis fails, return empty list
            pass
        
        # Remove duplicates and common words
        common_words = {'and', 'the', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from'}
        related_terms = list(set(related_terms))
        related_terms = [term for term in related_terms if term not in common_words and len(term) > 1]
        
        return related_terms[:10]  # Return top 10 most relevant
    
    def _intelligent_geological_match(self, query_term: str, document_content: str) -> tuple[bool, float]:
        """Intelligent geological matching by analyzing content patterns"""
        variations = self._generate_geological_variations(query_term)
        all_terms = [query_term] + variations
        
        # Dynamic scoring based on match quality
        exact_match_score = 0.95
        variation_match_score = 0.85
        contextual_match_score = 0.70
        partial_match_score = 0.60
        
        # Check for exact matches first
        if query_term in document_content:
            return True, exact_match_score
        
        # Check for variation matches with context analysis
        for variation in variations:
            if variation in document_content:
                # Analyze the context around the match
                match_pos = document_content.find(variation)
                if match_pos != -1:
                    # Get context around the match
                    context_start = max(0, match_pos - 30)
                    context_end = min(len(document_content), match_pos + len(variation) + 30)
                    context = document_content[context_start:context_end]
                    
                    # Score based on context quality
                    if len(variation) <= 2:
                        # For short terms, require geological context
                        geological_indicators = ['log', 'curve', 'response', 'well', 'formation', 
                                               'geology', 'stratigraphy', 'sequence', 'content']
                        if any(indicator in context.lower() for indicator in geological_indicators):
                            return True, contextual_match_score
                    else:
                        # Longer variations are more reliable
                        return True, variation_match_score
        
        # Check for partial semantic matches by analyzing word co-occurrence
        query_words = query_term.split()
        if len(query_words) > 1:
            # Count how many query words appear in the document
            word_matches = sum(1 for word in query_words if word.lower() in document_content.lower())
            match_ratio = word_matches / len(query_words)
            
            if match_ratio >= 0.7:  # Most words match
                return True, partial_match_score * match_ratio
            elif match_ratio >= 0.5:  # Half words match
                # Check if the matching words appear close together
                positions = []
                for word in query_words:
                    pos = document_content.lower().find(word.lower())
                    if pos != -1:
                        positions.append(pos)
                
                if len(positions) >= 2:
                    # If matching words are within 50 characters, it's likely related
                    max_distance = max(positions) - min(positions)
                    if max_distance < 50:
                        return True, partial_match_score * 0.8
        
        return False, 0.0
    
    def _calculate_similarity(self, distance: float, query: str, document: str) -> float:
        """Improved similarity calculation with geological term variation matching"""
        query_lower = query.lower().strip()
        doc_lower = document.lower()
        
        # Create intelligent variations for any geological term
        query_variations = [query_lower, query]  # Include original case
        query_variations.extend(self._generate_geological_variations(query_lower))
        
        # For short queries (1-2 words), require any variation to be present
        if len(query.split()) <= 2:
            term_found = any(variation.lower() in doc_lower for variation in query_variations)
            if not term_found:
                return 0.0  # Reject if no variation found
        
        # Base vector similarity (ChromaDB uses cosine distance usually)
        base_similarity = max(0.0, 1.0 - (distance / 2.0))
        
        # Strong boost for exact matches with any variation
        exact_boost = 0.0
        for variation in query_variations:
            if variation in document:  # Exact case match
                exact_boost = 0.5
                break
            elif variation.lower() in doc_lower:  # Case-insensitive match
                exact_boost = 0.4
                break
        
        # Keyword density boost
        query_words = query.split()
        if len(query_words) > 1:
            matches = sum(1 for word in query_words if word.lower() in doc_lower)
            if matches == 0:  # No keywords found
                return 0.0
            keyword_boost = (matches / len(query_words)) * 0.2
        else:
            keyword_boost = 0.0
            
        final_similarity = base_similarity + exact_boost + keyword_boost
        return min(1.0, final_similarity)
    
    def search_by_content_type(self, query: str, content_types: List[str], 
                              max_results: int = 50) -> Dict[str, List[QueryResult]]:
        """
        Search specific content types
        
        Args:
            query: Search query
            content_types: Types to search ('text', 'image', 'geological', 'combined')
            max_results: Max results per type
            
        Returns:
            Dictionary mapping content type to results
        """
        type_collection_map = {
            'text': 'spem_text_content',
            'image': 'spem_image_metadata',
            'geological': 'spem_geological_terms',
            'combined': 'spem_combined_content'
        }
        
        results_by_type = {}
        
        # Use ONLY vector search for proper similarity ranking
        # Skip exact string search that gives hardcoded scores
        
        # Vector search for each requested content type
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        for content_type in content_types:
            if content_type in type_collection_map:
                collection_name = type_collection_map[content_type]
                
                if collection_name in self.collections:
                    collection = self.collections[collection_name]
                    
                    try:
                        results = collection.query(
                            query_embeddings=[query_embedding],
                            n_results=min(max_results, collection.count()),
                            include=['documents', 'metadatas', 'distances']
                        )
                        
                        # Start with exact results for this content type
                        type_results = results_by_type.get(content_type, [])
                        if results['documents'][0]:
                            for doc, metadata, distance in zip(
                                results['documents'][0],
                                results['metadatas'][0],
                                results['distances'][0]
                            ):  
                                # STRICT FILTERING with geological term variations
                                query_lower = query.lower().strip()
                                doc_lower = doc.lower()
                                
                                # Create intelligent variations for any geological term
                                query_variations = [query_lower]
                                query_variations.extend(self._generate_geological_variations(query_lower))
                                
                                # Different filtering for images vs text
                                is_image_collection = "image" in collection_name
                                term_found = False
                                
                                if is_image_collection:
                                    # For images: use intelligent semantic matching but preserve ChromaDB scores
                                    term_found, confidence = self._intelligent_geological_match(query_lower, doc_lower)
                                    if term_found:
                                        # Use real ChromaDB vector similarity, not hardcoded scores
                                        similarity = max(0.0, 1.0 - (distance / 2.0)) if distance < 2.0 else 0.0
                                        # Apply confidence boost but preserve relative ranking
                                        similarity = similarity * (1.0 + confidence * 0.2)
                                    else:
                                        continue
                                else:
                                    # For text: use flexible variation matching
                                    term_found = any(variation in doc_lower for variation in query_variations)
                                    if not term_found:
                                        continue
                                    # Use ChromaDB vector similarity for text too
                                    similarity = max(0.0, 1.0 - (distance / 2.0)) if distance < 2.0 else 0.0
                                
                                # Only include results with meaningful similarity (term must be present)
                                if similarity > 0.0:
                                    # Try multiple metadata keys for page number
                                    page_num = (metadata.get('page_number') or 
                                              metadata.get('page') or 
                                              self._extract_page_from_content(doc) or 0)
                                    
                                    query_result = QueryResult(
                                        content=doc,
                                        content_type=content_type,
                                        page_number=page_num,
                                        relevance_score=similarity,
                                        geological_terms=self._extract_geological_terms(doc),
                                        metadata=metadata,
                                        source_collection=collection_name
                                    )
                                    type_results.append(query_result)
                        
                        # Remove duplicates and sort by relevance
                        type_results = self._remove_duplicate_results(type_results)
                        type_results.sort(key=lambda x: x.relevance_score, reverse=True)
                        
                        results_by_type[content_type] = type_results[:max_results]
                        
                    except Exception as e:
                        if "hnsw segment reader" in str(e).lower():
                            logger.debug(f"HNSW index not available for {content_type}, using fallback search")
                        else:
                            logger.error(f"Error searching {content_type}: {e}")
                        results_by_type[content_type] = []
        
        return results_by_type
    
    def search_geological_concepts(self, geological_terms: List[str], 
                                  max_results: int = 10) -> List[QueryResult]:
        """
        Search for specific geological concepts
        
        Args:
            geological_terms: List of geological terms to search for
            max_results: Maximum results to return
            
        Returns:
            List of relevant results containing the geological terms
        """
        # Create query from geological terms
        query = " ".join(geological_terms)
        
        # Search with emphasis on geological content
        results = self.search_all_content(query, max_results * 2, similarity_threshold=0.2)
        
        # Filter and rank by geological term relevance
        geological_results = []
        
        for result in results:
            # Calculate geological relevance
            result_terms = set([term.lower() for term in result.geological_terms])
            query_terms = set([term.lower() for term in geological_terms])
            
            # Calculate geological term overlap
            geological_overlap = len(result_terms & query_terms)
            geological_relevance = geological_overlap / len(query_terms) if query_terms else 0
            
            # Combine with semantic similarity
            combined_score = (result.relevance_score * 0.6) + (geological_relevance * 0.4)
            
            if geological_overlap > 0 or result.relevance_score > 0.5:
                result.relevance_score = combined_score
                geological_results.append(result)
        
        # Re-sort by combined score
        geological_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return geological_results[:max_results]
    
    def search_by_page(self, page_numbers: List[int], 
                       query: Optional[str] = None) -> List[QueryResult]:
        """
        Search content from specific pages
        
        Args:
            page_numbers: List of page numbers to search
            query: Optional query to filter results
            
        Returns:
            List of results from specified pages
        """
        all_results = []
        
        if query:
            query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        for collection_name, collection in self.collections.items():
            try:
                if query:
                    # Query with semantic search
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=collection.count(),
                        include=['documents', 'metadatas', 'distances'],
                        where={"page": {"$in": page_numbers}}
                    )
                else:
                    # Get all content from pages
                    results = collection.get(
                        include=['documents', 'metadatas'],
                        where={"page": {"$in": page_numbers}}
                    )
                
                if results['documents']:
                    documents = results['documents'][0] if query else results['documents']
                    metadatas = results['metadatas'][0] if query else results['metadatas']
                    distances = results.get('distances', [None] * len(documents))
                    if distances and len(distances) > 0:
                        distances = distances[0] if query else [0.0] * len(documents)
                    else:
                        distances = [0.0] * len(documents)
                    
                    for doc, metadata, distance in zip(documents, metadatas, distances):
                        if metadata.get('page') in page_numbers:
                            # STRICT FILTERING: Only include results that contain search term (if query provided)
                            if query:
                                query_lower = query.lower().strip()
                                doc_lower = doc.lower()
                                
                                # Skip if document doesn't contain the search term
                                if query_lower not in doc_lower:
                                    continue
                            
                            # Use improved similarity calculation
                            similarity = self._calculate_similarity(distance, query or "", doc)
                            
                            # Try multiple metadata keys for page number
                            page_num = (metadata.get('page_number') or 
                                      metadata.get('page') or 
                                      self._extract_page_from_content(doc) or 0)
                            
                            query_result = QueryResult(
                                content=doc,
                                content_type=metadata.get('chunk_type', 'unknown'),
                                page_number=page_num,
                                relevance_score=similarity,
                                geological_terms=self._extract_geological_terms(doc),
                                metadata=metadata,
                                source_collection=collection_name
                            )
                            all_results.append(query_result)
                
            except Exception as e:
                logger.error(f"Error searching pages in {collection_name}: {e}")
        
        # Sort by page number, then by relevance
        all_results.sort(key=lambda x: (x.page_number, -x.relevance_score))
        
        return all_results
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        stats = {
            'database_path': str(self.vector_db_path),
            'total_collections': len(self.collections),
            'collections': {},
            'total_documents': 0,
            'metadata': self.metadata
        }
        
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats['collections'][name] = {
                    'document_count': count,
                    'description': collection.metadata
                }
                stats['total_documents'] += count
            except Exception as e:
                stats['collections'][name] = {'error': str(e)}
        
        return stats
    
    def _remove_duplicate_results(self, results: List[QueryResult]) -> List[QueryResult]:
        """Remove duplicate results based on content similarity"""
        unique_results = []
        seen_content = set()
        
        for result in results:
            # Use first 100 characters as content fingerprint
            content_fingerprint = result.content[:100].strip()
            
            if content_fingerprint not in seen_content:
                seen_content.add(content_fingerprint)
                unique_results.append(result)
                
        return unique_results
    
    def _diversify_results(self, results: List[QueryResult], 
                          max_results: int) -> List[QueryResult]:
        """Ensure diversity across content types and pages"""
        if len(results) <= max_results:
            return results
        
        # Group by content type
        by_type = {}
        for result in results:
            content_type = result.content_type
            if content_type not in by_type:
                by_type[content_type] = []
            by_type[content_type].append(result)
        
        # Select diverse results
        diverse_results = []
        remaining_slots = max_results
        
        # First, take top result from each type
        for content_type, type_results in by_type.items():
            if remaining_slots > 0 and type_results:
                diverse_results.append(type_results[0])
                remaining_slots -= 1
        
        # Fill remaining slots with highest scoring results
        used_results = set(id(r) for r in diverse_results)
        for result in results:
            if remaining_slots > 0 and id(result) not in used_results:
                diverse_results.append(result)
                remaining_slots -= 1
        
        return diverse_results
    
    def _extract_geological_terms(self, text: str) -> List[str]:
        """Extract geological terms from text (same logic as vectorizer)"""
        geological_categories = {
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
        
        found_terms = []
        text_lower = text.lower()
        
        for category, terms in geological_categories.items():
            for term in terms:
                if term.lower() in text_lower:
                    found_terms.append(term)
        
        return list(set(found_terms))
    
    def format_results(self, results: List[QueryResult], 
                      max_content_length: int = 200) -> str:
        """Format results for display"""
        if not results:
            return "No results found."
        
        formatted = []
        formatted.append(f"Found {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            content = result.content
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            geological_terms = ", ".join(result.geological_terms[:5])
            if len(result.geological_terms) > 5:
                geological_terms += f" (+{len(result.geological_terms) - 5} more)"
            
            formatted.append(f"""
{i}. [{result.content_type.upper()}] Page {result.page_number} (Score: {result.relevance_score:.3f})
   Content: {content}
   Geological Terms: {geological_terms}
   Source: {result.source_collection}
""")
        
        return "\n".join(formatted)


def main():
    """Test the enhanced query engine"""
    try:
        # Initialize query engine
        engine = EnhancedSPEMQueryEngine("data/enhanced_spem_vector_db")
        
        # Get database statistics
        stats = engine.get_database_statistics()
        print("DATABASE STATISTICS")
        print("=" * 60)
        print(f"Total collections: {stats['total_collections']}")
        print(f"Total documents: {stats['total_documents']}")
        for name, info in stats['collections'].items():
            print(f"  {name}: {info.get('document_count', 'Error')}")
        
        # Test searches
        test_queries = [
            "gamma ray log interpretation",
            "sequence stratigraphy patterns",
            "well log signatures",
            "depositional environments",
            "stacking patterns"
        ]
        
        for query in test_queries:
            print(f"\n\nSEARCH: '{query}'")
            print("=" * 60)
            results = engine.search_all_content(query, max_results=3)
            print(engine.format_results(results))
        
    except Exception as e:
        print(f"Query engine test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()