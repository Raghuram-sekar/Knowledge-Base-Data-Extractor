"""
ENHANCED GEOLOGICAL RAG SYSTEM - MAIN INTERFACE
================================================================================
Complete RAG system integrating:
- LAS file processing with geological analysis
- SPEM document vectorization with image support
- Multi-modal semantic search capabilities
- Intelligent geological query processing
================================================================================
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
from loguru import logger

# Import our components
from src.vectorization.enhanced_vectorizer import EnhancedSPEMVectorizer
from src.vectorization.rag_query_engine import EnhancedSPEMQueryEngine, QueryResult

# Import visual components
try:
    from src.visual.visual_analyzer import VisualGeologicalAnalyzer
    VISUAL_AVAILABLE = True
except ImportError:
    VISUAL_AVAILABLE = False
    logger.warning("Visual analysis components not available")


@dataclass
class RAGSystemConfig:
    """Configuration for the geological RAG system"""
    vector_db_path: str = "data/enhanced_spem_vector_db"
    extracted_content_path: str = "extracted_spem_complete"
    las_data_path: str = "data"
    output_path: str = "rag_output"
    similarity_threshold: float = 0.3  # Good threshold for natural language questions
    max_results_per_query: int = 50  # Increased to show all available results
    enable_geological_analysis: bool = True
    include_images: bool = True
    enable_visual_analysis: bool = True
    visual_confidence_threshold: float = 0.3
    
    # Dynamic threshold configuration
    high_relevance_threshold: float = 0.8  # Very high confidence threshold
    search_expansion_factor: int = 2
    default_search_results: int = 10
    page_match_score_factor: float = 0.8
    
    # Additional thresholds for specific terms
    specific_term_threshold: float = -0.2
    secondary_threshold: float = -0.6
    
    # Relevance labels
    high_relevance_label: str = "🔥 Highly relevant"
    moderate_relevance_label: str = "⭐ Relevant" 
    medium_relevance_label: str = "⭐ Relevant"
    low_relevance_label: str = "📝 Related"


@dataclass
class ImageReference:
    """Reference to a specific image in the SPEM document"""
    image_id: str
    page_number: int
    file_path: str
    geological_content: str
    context_text: str
    image_type: str
    relevance_score: float = 0.0


@dataclass
class VisualAnalysisResult:
    """Result of visual image analysis"""
    image_ref: ImageReference
    question: str
    answer: str
    confidence_score: float
    geological_concepts: List[str]
    analysis_details: Dict[str, Any]
    timestamp: str


class EnhancedGeologicalRAGSystem:
    """
    Complete geological RAG system with multi-modal capabilities
    """
    
    def __init__(self, config: RAGSystemConfig = None):
        self.config = config or RAGSystemConfig()
        
        # Initialize configurable parameters
        self.similarity_thresholds = {
            'strict': 0.7,     # High confidence matches
            'moderate': 0.4,   # Good matches  
            'relaxed': 0.2     # Broader search
        }
        
        self.search_limits = {
            'specific_term': 5,
            'general_query': 10,
            'broad_search': 50
        }
        
        self.content_limits = {
            'min_length': 20,
            'preview_length': 100,
            'summary_length': 200,
            'max_length': 1500
        }
        
        self.api_settings = {
            'timeout': 30,
            'num_predict': 800,
            'temperature': 0.1,
            'top_p': 0.9
        }
        
        self.output_path = Path(self.config.output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.query_engine = None
        self.visual_analyzer = None
        self.images_metadata = {}
        self.last_search_results = []  # Store last search results for follow-up questions
        
        self._initialize_system()
        
        # Learn optimal thresholds after system is initialized
        # Temporarily disabled - using fixed improved thresholds
        # if self.query_engine:
        #     learned_thresholds = self._learn_optimal_thresholds()
        #     self.similarity_thresholds.update(learned_thresholds)
        #     logger.info(f"Updated similarity thresholds: {self.similarity_thresholds}")
        
        logger.info(f"Enhanced Geological RAG System initialized")
        logger.info(f"Vector DB: {self.config.vector_db_path}")
        logger.info(f"Output: {self.config.output_path}")
        logger.info(f"Visual Analysis: {'Enabled' if self.visual_analyzer else 'Disabled'}")
        logger.info(f"Available Images: {len(self.images_metadata)}")
    
    def _initialize_system(self):
        """Initialize RAG system components"""
        try:
            # Check if vector database exists
            if not Path(self.config.vector_db_path).exists():
                logger.warning("Vector database not found. Creating new one...")
                self._create_vector_database()
            
            # Initialize query engine
            self.query_engine = EnhancedSPEMQueryEngine(self.config.vector_db_path)
            
            # Initialize visual components if enabled
            if self.config.enable_visual_analysis and VISUAL_AVAILABLE:
                self._initialize_visual_components()
            
            # Load image metadata
            self.images_metadata = self._load_image_metadata()
            
            logger.info("RAG system components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def _create_vector_database(self):
        """Create vector database from extracted content"""
        logger.info("Creating vector database from extracted SPEM content...")
        
        vectorizer = EnhancedSPEMVectorizer(self.config.vector_db_path)
        stats = vectorizer.create_enhanced_vector_database(self.config.extracted_content_path)
        
        logger.info(f"Vector database created with {stats['total_embeddings']} embeddings")
        return stats
    
    def _initialize_visual_components(self):
        """Initialize visual analysis components"""
        try:
            self.visual_analyzer = VisualGeologicalAnalyzer()
            logger.info("Visual analysis system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize visual analyzer: {e}")
            self.visual_analyzer = None
    
    def _load_image_metadata(self) -> Dict[str, ImageReference]:
        """Load image metadata from extracted content"""
        images = {}
        
        try:
            # Load extraction results
            extraction_file = Path(self.config.extracted_content_path) / "metadata" / "extraction_results.json"
            if not extraction_file.exists():
                logger.warning(f"Image metadata file not found: {extraction_file}")
                return images
                
            with open(extraction_file, 'r', encoding='utf-8') as f:
                extraction_data = json.load(f)
            
            # Check for ultimate enhanced images directory (highest quality)
            base_images_dir = Path(self.config.extracted_content_path) / "images"
            ultimate_enhanced_dir = base_images_dir / "ultimate_enhanced"
            
            use_ultimate_images = ultimate_enhanced_dir.exists()
            images_dir = ultimate_enhanced_dir if use_ultimate_images else base_images_dir
            use_enhanced_images = use_ultimate_images  # Fix undefined variable
            
            if use_ultimate_images:
                logger.info("Using ultimate enhanced images for highest quality analysis")
            else:
                logger.info("Using base images directory (ultimate enhanced images not found)")
            
            # Extract image references
            for page_data in extraction_data['pages']:
                page_num = page_data['page_number']
                
                for image_data in page_data['images']:
                    original_file_path = image_data['file_path']
                    
                    if use_enhanced_images:
                        # Update file path to point to enhanced image
                        original_name = Path(original_file_path).name
                        ultimate_name = f"ultimate_{original_name}"
                        # Convert JPEG to PNG (ultimate enhancer saves as PNG)
                        if ultimate_name.lower().endswith(('.jpg', '.jpeg')):
                            ultimate_name = ultimate_name.rsplit('.', 1)[0] + '.png'
                        
                        ultimate_file_path = str(images_dir / ultimate_name)
                        
                        # Check if ultimate enhanced file exists, fallback to original if not
                        if Path(ultimate_file_path).exists():
                            file_path = ultimate_file_path
                        else:
                            file_path = original_file_path
                            logger.debug(f"Ultimate enhanced image not found for {original_name}, using original")
                    else:
                        file_path = original_file_path
                    
                    image_ref = ImageReference(
                        image_id=image_data['id'],
                        page_number=page_num,
                        file_path=file_path,
                        geological_content=image_data['geological_content'],
                        context_text=image_data.get('context_text', ''),
                        image_type=image_data['image_type'],
                        relevance_score=0.0
                    )
                    images[image_data['id']] = image_ref
            
            enhancement_status = "enhanced" if use_enhanced_images else "original"
            logger.info(f"Loaded metadata for {len(images)} images ({enhancement_status} quality)")
            
        except Exception as e:
            logger.error(f"Failed to load image metadata: {e}")
        
        return images
    
    def search(self, query: str, search_type: str = "all", 
               max_results: int = None) -> List[QueryResult]:
        """
        Perform intelligent geological search
        
        Args:
            query: Natural language query
            search_type: "all", "text", "image", "geological", or "combined"
            max_results: Maximum results to return
            
        Returns:
            List of QueryResult objects
        """
        if not self.query_engine:
            raise RuntimeError("Query engine not initialized")
        
        max_results = max_results or self.config.max_results_per_query
        
        logger.info(f"Searching for: '{query}' (type: {search_type})")
        
        if search_type == "all":
            # Use learned moderate threshold for general searches
            threshold = self.similarity_thresholds.get('moderate', self.config.similarity_threshold)
            results = self.query_engine.search_all_content(
                query, 
                max_results=max_results,
                similarity_threshold=threshold
            )
        elif search_type == "text":
            # For text search, include all collections with textual content
            content_types = ["text", "combined", "geological"]  # Include all text-related types
            results_by_type = self.query_engine.search_by_content_type(
                query, 
                content_types, 
                max_results=max_results
            )
            # Combine all text-related results and normalize content_type display
            results = []
            for content_type in content_types:
                type_results = results_by_type.get(content_type, [])
                # Normalize all text-related types to show as "text" for display
                for result in type_results:
                    result.content_type = "text"
                results.extend(type_results)
            
            # Filter out Page 0 (metadata/concept definitions) for cleaner text search results
            results = [r for r in results if r.page_number > 0]
        elif search_type in ["image", "geological", "combined"]:
            results_by_type = self.query_engine.search_by_content_type(
                query, 
                [search_type], 
                max_results=max_results
            )
            results = results_by_type.get(search_type, [])
        else:
            raise ValueError(f"Invalid search type: {search_type}")
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def search_images_by_concept(self, query: str, max_results: int = 5) -> List[ImageReference]:
        """
        Discovery Phase: Search for images related to geological concepts
        
        Args:
            query: Geological concept or keyword to search for
            max_results: Maximum number of image references to return
            
        Returns:
            List of ImageReference objects ranked by relevance
        """
        logger.info(f"Searching images for concept: '{query}'")
        
        if not self.query_engine:
            logger.error("Query engine not available")
            return []
        
        # Search using base RAG system for image content
        base_results = self.search(query, search_type="image", max_results=max_results*self.config.search_expansion_factor)
        
        # Convert to ImageReference objects and rank
        image_refs = []
        
        for result in base_results:
            # Extract image ID from metadata
            image_id = result.metadata.get('image_id', '')
            
            if image_id in self.images_metadata:
                image_ref = self.images_metadata[image_id]
                image_ref.relevance_score = result.relevance_score
                image_refs.append(image_ref)
        
        # Only add text-based page matches if we have few direct image matches
        if len(image_refs) < max_results:
            text_results = self.search(query, search_type="text", max_results=self.config.default_search_results)
            
            # Use learned strict threshold instead of config value
            learned_threshold = self.similarity_thresholds.get('strict', self.config.high_relevance_threshold)
            
            for result in text_results:
                # Only consider high-relevance text matches using learned threshold
                if result.relevance_score > learned_threshold:
                    page_num = result.page_number
                    # Find images on the same page
                    for img_id, img_ref in self.images_metadata.items():
                        if img_ref.page_number == page_num and img_ref not in image_refs:
                            img_ref.relevance_score = result.relevance_score * self.config.page_match_score_factor
                            image_refs.append(img_ref)
        
        # Sort by relevance and apply better filtering
        image_refs.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Enhanced relevance filtering for specific terms
        is_specific_term = self._is_specific_term(query)
        
        if is_specific_term:
            # For specific terms (names, places, etc.), use much stricter thresholds
            image_threshold = self.config.specific_term_threshold
            secondary_threshold = self.config.specific_term_threshold + 0.1
            logger.info(f"Detected specific term '{query}' - using strict relevance filtering")
        else:
            # For general geological concepts, use standard thresholds  
            image_threshold = 0.3  # Good threshold for image search
            secondary_threshold = self.config.secondary_threshold
        
        filtered_refs = [ref for ref in image_refs if ref.relevance_score > image_threshold]
        
        # If we have fewer good matches, check if we should show any results for specific terms
        if len(filtered_refs) < max_results:
            if is_specific_term and len(image_refs) > 0:
                best_score = image_refs[0].relevance_score if image_refs else self.config.secondary_threshold - 0.1
                poor_relevance_threshold = self.config.specific_term_threshold + 0.1
                if best_score < poor_relevance_threshold:
                    logger.warning(f"No relevant images found for specific term: '{query}' (best score: {best_score:.3f})")
                    return []  # Return empty rather than irrelevant results
            
            logger.warning(f"Only {len(filtered_refs)} images above threshold {image_threshold}, found {len(image_refs)} total")
            # Take the best available from secondary threshold
            remaining = [ref for ref in image_refs if ref not in filtered_refs and ref.relevance_score > secondary_threshold]
            filtered_refs.extend(remaining[:max_results - len(filtered_refs)])
        
        result_refs = filtered_refs[:max_results]
        
        # Log more detailed info about filtering
        above_threshold = len([ref for ref in image_refs if ref.relevance_score > image_threshold])
        logger.info(f"Found {len(result_refs)} relevant images (filtered from {len(image_refs)} total, {above_threshold} above threshold {image_threshold})")
        return result_refs
    
    def _learn_optimal_thresholds(self) -> dict:
        """Learn optimal similarity thresholds from data performance"""
        try:
            if not self.query_engine:
                return self.similarity_thresholds
            
            # Test different thresholds on sample queries
            test_queries = ["formation", "well log", "gamma ray", "sequence"]
            threshold_performance = {}
            
            for threshold in [-0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]:
                total_results = 0
                quality_score = 0
                
                for query in test_queries:
                    try:
                        results = self.query_engine.search_all_content(
                            query, max_results=10, 
                            similarity_threshold=self.similarity_thresholds['relaxed']
                        )
                        # Filter by threshold
                        filtered_results = [r for r in results if r.relevance_score >= threshold]
                        total_results += len(filtered_results)
                        
                        # Simple quality metric based on result count and relevance
                        if filtered_results:
                            avg_relevance = sum(r.relevance_score for r in filtered_results) / len(filtered_results)
                            quality_score += avg_relevance * len(filtered_results)
                    except:
                        continue
                
                if total_results > 0:
                    threshold_performance[threshold] = quality_score / total_results
            
            # Select optimal thresholds based on performance
            if threshold_performance:
                sorted_thresholds = sorted(threshold_performance.items(), key=lambda x: x[1], reverse=True)
                best_threshold = sorted_thresholds[0][0]
                
                return {
                    'strict': max(best_threshold + 0.2, -0.2),
                    'moderate': best_threshold,
                    'relaxed': min(best_threshold - 0.2, -0.8)
                }
            
        except Exception as e:
            logger.error(f"Failed to learn optimal thresholds: {e}")
        
        return self.similarity_thresholds
    
    def _learn_geological_terms(self) -> set:
        """Learn geological terms dynamically from the vector database without hardcoding"""
        try:
            if not self.query_engine:
                return set()
            
            geological_terms = set()
            
            # NO HARDCODING - Extract terms directly from all geological content in database
            try:
                # Get all geological terms from the geological collection
                geological_collection = self.query_engine.client.get_collection('spem_geological_terms')
                all_geological_docs = geological_collection.get(include=['documents', 'metadatas'])
                
                for doc, metadata in zip(all_geological_docs['documents'], all_geological_docs['metadatas']):
                    # Extract terms from geological content
                    if metadata and 'geological_terms' in metadata:
                        terms = str(metadata['geological_terms']).split(', ')
                        geological_terms.update(term.lower().strip() for term in terms if len(term.strip()) > 2)
                    
                    # Extract from document content
                    if doc:
                        words = doc.lower().split()
                        geological_terms.update(word.strip(';:,') for word in words if len(word.strip(';:,')) > 3)
                            
            except Exception as e:
                logger.debug(f"Error learning geological terms from database: {e}")
            
            # Filter to keep only likely geological terms
            filtered_terms = set()
            for term in geological_terms:
                if (len(term) >= 3 and term.isalpha() and 
                    not term in ['the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'been']):
                    filtered_terms.add(term)
            
            logger.info(f"Learned {len(filtered_terms)} geological terms from data")
            return filtered_terms
            
        except Exception as e:
            logger.error(f"Failed to learn geological terms: {e}")
            return {'log', 'well', 'formation', 'sequence'}

    def _is_specific_term(self, query: str) -> bool:
        """Determine if a query is a specific term using learned geological vocabulary"""
        query_lower = query.lower().strip()
        words = query.split()
        
        # Learn geological terms if not already done
        if not hasattr(self, '_learned_geological_terms'):
            self._learned_geological_terms = self._learn_geological_terms()
        
        # Short queries are often specific
        if len(words) <= 2:
            # Check if it contains capitalized words that aren't in learned geological terms
            for word in words:
                if (word and word[0].isupper() and 
                    word.lower() not in self._learned_geological_terms):
                    return True  # Likely a proper name or specific term
        
        # Check for CamelCase or specific patterns (names, places)
        import re
        specific_patterns = [
            r'[A-Z][a-z]+[A-Z][a-z]+',  # CamelCase
            r'^[A-Z][a-z]+$',           # Single capitalized word
        ]
        
        for pattern in specific_patterns:
            if re.match(pattern, query.strip()):
                # Additional check: is it a known geological term?
                if query.lower() not in self._learned_geological_terms:
                    return True
        
        # Very short queries that aren't common geological terms
        if (len(query_lower) <= 4 and 
            query_lower not in self._learned_geological_terms):
            return True
            
        return False

    def search_images_with_disambiguation(self, query: str, max_results: int = 5) -> dict:
        """
        Enhanced image search with multi-image disambiguation support
        
        Args:
            query: Geological concept or keyword to search for
            max_results: Maximum number of image references to return
            
        Returns:
            Dict with 'images' list and 'disambiguation_needed' bool
            If disambiguation_needed=True, includes 'ambiguous_groups' for user clarification
        """
        logger.info(f"Searching images with disambiguation for: '{query}'")
        
        # Get all relevant images
        image_refs = self.search_images_by_concept(query, max_results * 2)  # Get reasonable amount for disambiguation
        
        if len(image_refs) <= 1:
            return {
                'images': image_refs,
                'disambiguation_needed': False
            }
        
        # Group images by page to detect potential ambiguity
        page_groups = {}
        for img_ref in image_refs:
            page_key = f"{img_ref.file_path}_page_{img_ref.page_number}"
            if page_key not in page_groups:
                page_groups[page_key] = []
            page_groups[page_key].append(img_ref)
        
        # Check for pages with multiple relevant images
        ambiguous_groups = []
        final_images = []
        
        for page_key, images_on_page in page_groups.items():
            if len(images_on_page) > 1:
                # Multiple images on same page - potential ambiguity
                ambiguous_groups.append({
                    'page_info': f"Page {images_on_page[0].page_number} in {images_on_page[0].file_path.split('/')[-1]}",
                    'images': images_on_page[:3],  # Limit to top 3 for clarity
                    'query': query
                })
            else:
                final_images.extend(images_on_page)
        
        # If we have ambiguous groups, return them for user clarification
        if ambiguous_groups:
            return {
                'images': final_images[:max_results//2] if final_images else [],  # Some non-ambiguous results
                'disambiguation_needed': True,
                'ambiguous_groups': ambiguous_groups,
                'message': f"Found multiple images related to '{query}' on the same page(s). Please specify which image you're asking about."
            }
        
        return {
            'images': image_refs[:max_results],
            'disambiguation_needed': False
        }
    
    def search_geological_concepts(self, geological_terms: List[str]) -> List[QueryResult]:
        """Search for specific geological concepts"""
        if not self.query_engine:
            raise RuntimeError("Query engine not initialized")
        
        logger.info(f"Searching geological concepts: {geological_terms}")
        results = self.query_engine.search_geological_concepts(
            geological_terms, 
            max_results=self.config.max_results_per_query
        )
        
        logger.info(f"Found {len(results)} geological results")
        return results
    
    def search_by_page(self, page_numbers: List[int], 
                       query: Optional[str] = None) -> List[QueryResult]:
        """Search content from specific pages"""
        if not self.query_engine:
            raise RuntimeError("Query engine not initialized")
        
        logger.info(f"Searching pages {page_numbers}")
        results = self.query_engine.search_by_page(page_numbers, query)
        
        logger.info(f"Found {len(results)} page results")
        return results
    
    def display_disambiguation_options(self, ambiguous_groups: List[dict], query: str) -> str:
        """
        Display disambiguation options for multiple images matching the same query
        
        Args:
            ambiguous_groups: List of groups containing multiple images on same pages
            query: Original search query
            
        Returns:
            Formatted string showing disambiguation options
        """
        output = [f"\nMultiple images found for '{query}' - Please specify which one:\n"]
        
        option_count = 1
        for group in ambiguous_groups:
            output.append(f"[PAGE] {group['page_info']}:")
            
            for img_ref in group['images']:
                # Create short description based on image content
                img_desc = (img_ref.geological_content[:self.content_limits['preview_length']] + "..." 
                           if len(img_ref.geological_content) > self.content_limits['preview_length'] 
                           else img_ref.geological_content)
                output.append(f"  {option_count}. Image {img_ref.image_id} - {img_desc or 'Geological diagram/chart'}")
                output.append(f"     Type: {img_ref.image_type} | Relevance: {img_ref.relevance_score:.2f}")
                option_count += 1
            
            output.append("")  # Empty line between groups
        
        output.append("USAGE: Use 'analyze-image <image_id> <your_question>' to analyze a specific image")
        output.append("   Example: analyze-image img_123 what type of geological formation is this?")
        
        return "\n".join(output)

    def analyze_image(self, image_identifier: str, question: str = "What does this image show?") -> VisualAnalysisResult:
        """
        Explanation Phase: Provide detailed analysis of a specific image using both visual AI and document context
        
        Args:
            image_identifier: Either image_id or partial identifier
            question: Question to ask about the image
            
        Returns:
            VisualAnalysisResult with detailed geological explanation
        """
        logger.info(f"Analyzing image: {image_identifier} with question: '{question}'")
        
        # Find the image reference
        image_ref = self._resolve_image_identifier(image_identifier)
        
        if not image_ref:
            return VisualAnalysisResult(
                image_ref=ImageReference(
                    image_id=image_identifier,
                    page_number=0,
                    file_path="",
                    geological_content="",
                    context_text="",
                    image_type="unknown",
                    relevance_score=0.0
                ),
                question=question,
                answer=f"Image not found: {image_identifier}. Use 'list-images' to see available images.",
                confidence_score=0.0,
                geological_concepts=[],
                analysis_details={'error': 'Image not found'},
                timestamp=datetime.now().isoformat()
            )
        
        try:
            # Get document context for the image
            document_context = self._get_document_context_for_image(image_ref, question)
            
            # Get visual analysis if available - do this ONCE with the user's question
            enhanced_answer = ""
            if self.visual_analyzer:
                visual_result = self.visual_analyzer.analyze_image_with_question(
                    image_ref.file_path,
                    question,  # Use the actual user question, not a generic one
                    image_ref.context_text  # Pass context for intelligent prompting
                )
                enhanced_answer = visual_result or "Unable to analyze image content."
            else:
                enhanced_answer = "Visual analysis not available."
            
            # Extract geological concepts from the enhanced answer
            geological_concepts = self._extract_geological_concepts_from_answer(enhanced_answer)
            
            # Calculate confidence with detailed reasoning
            confidence_data = self._calculate_analysis_confidence(
                image_ref, question, enhanced_answer, "", document_context
            )
            
            return VisualAnalysisResult(
                image_ref=image_ref,
                question=question,
                answer=enhanced_answer,
                confidence_score=confidence_data['score'],
                geological_concepts=geological_concepts,
                analysis_details={
                    'visual_description': enhanced_answer,  # The enhanced answer IS the visual description
                    'document_context': (document_context[:self.content_limits['summary_length']] + "..." 
                                       if len(document_context) > self.content_limits['summary_length'] 
                                       else document_context),
                    'confidence_reasoning': confidence_data['reasoning'],
                    'confidence_breakdown': confidence_data['breakdown']
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Enhanced image analysis failed: {e}")
            return VisualAnalysisResult(
                image_ref=image_ref,
                question=question,
                answer=f"Analysis failed: {str(e)}",
                confidence_score=0.0,
                geological_concepts=[],
                analysis_details={'error': str(e)},
                timestamp=datetime.now().isoformat()
            )
    
    def _resolve_image_identifier(self, identifier: str) -> Optional[ImageReference]:
        """Resolve image identifier to ImageReference"""
        # Direct match
        if identifier in self.images_metadata:
            return self.images_metadata[identifier]
        
        # Partial match
        for img_id, img_ref in self.images_metadata.items():
            if identifier.lower() in img_id.lower() or identifier in str(img_ref.page_number):
                return img_ref
        
        # Try to find by page reference
        if "page_" in identifier.lower():
            for img_id, img_ref in self.images_metadata.items():
                if f"page_{img_ref.page_number}" in identifier.lower():
                    return img_ref
        
        return None
    
    def _extract_geological_concepts_from_answer(self, answer: str) -> List[str]:
        """Extract geological concepts from Qwen-2 analysis without hardcoded terms"""
        if not answer:
            return []
        
        # Simple approach: extract meaningful words from Qwen-2's response
        # that appear to be geological or technical terms
        import re
        
        # Extract words that are likely geological terms (capitalized, technical, etc.)
        words = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]{4,}\b', answer)
        
        # Filter out common words and keep potential geological terms
        common_words = {'this', 'that', 'with', 'from', 'have', 'been', 'they', 'were', 'what', 'when', 'where', 'would', 'could', 'should', 'also', 'such', 'which', 'their', 'these', 'those', 'there', 'then'}
        
        geological_concepts = []
        for word in words:
            word_lower = word.lower()
            if (len(word) >= 3 and 
                word_lower not in common_words and
                word.isalpha()):
                geological_concepts.append(word_lower)
        
        # Remove duplicates and limit to top 10
        return list(set(geological_concepts))[:10]
    
    def _calculate_analysis_confidence(self, image_ref: ImageReference, question: str, 
                                      enhanced_analysis: str, visual_description: str, document_context: str) -> dict:
        """Calculate confidence score with transparent reasoning"""
        confidence_factors = {}
        reasoning = []
        
        # Factor 1: Image Quality Assessment (20%)
        quality_score = 0.8  # Default good quality
        if hasattr(image_ref, 'image_quality'):
            quality_score = image_ref.image_quality / 100
        confidence_factors['image_quality'] = quality_score * 0.20
        reasoning.append(f"Image Quality: {quality_score*100:.0f}% (contributes {confidence_factors['image_quality']:.1%})")
        
        # Factor 2: Question-Image Alignment (25%)
        question_lower = question.lower()
        geological_content_lower = image_ref.geological_content.lower()
        
        # Check for geological term alignment using learned terms
        if not hasattr(self, '_learned_geological_terms'):
            self._learned_geological_terms = self._learn_geological_terms()
        
        alignment_score = 0.0
        
        for term in self._learned_geological_terms:
            if term in question_lower and term in geological_content_lower:
                alignment_score += 0.3
            elif term in question_lower or term in geological_content_lower:
                alignment_score += 0.1
        
        alignment_score = min(alignment_score, 1.0)
        confidence_factors['question_alignment'] = alignment_score * 0.25
        reasoning.append(f"Question-Image Alignment: {alignment_score*100:.0f}% (contributes {confidence_factors['question_alignment']:.1%})")
        
        # Factor 3: Technical Terms Presence (20%)
        technical_terms_found = 0
        visual_desc_lower = visual_description.lower()
        enhanced_analysis_lower = enhanced_analysis.lower()
        
        for term in self._learned_geological_terms:
            if term in visual_desc_lower or term in enhanced_analysis_lower:
                technical_terms_found += 1
        
        learned_terms_count = len(self._learned_geological_terms) if self._learned_geological_terms else 1
        tech_score = min(technical_terms_found / learned_terms_count, 1.0)
        confidence_factors['technical_terms'] = tech_score * 0.20
        reasoning.append(f"Technical Terms Found: {technical_terms_found}/{learned_terms_count} (contributes {confidence_factors['technical_terms']:.1%})")
        
        # Factor 4: Context Integration (15%)
        context_score = 0.5  # Default moderate
        if document_context and len(document_context) > 100:
            context_score = 0.8
        elif document_context and len(document_context) > 50:
            context_score = 0.6
        
        confidence_factors['context_integration'] = context_score * 0.15
        reasoning.append(f"Context Integration: {context_score*100:.0f}% (contributes {confidence_factors['context_integration']:.1%})")
        
        # Factor 5: Analysis Completeness (20%)
        completeness_score = 0.7  # Default
        if len(enhanced_analysis) > 200 and any(term in enhanced_analysis_lower for term in self._learned_geological_terms):
            completeness_score = 0.9
        elif len(enhanced_analysis) > 100:
            completeness_score = 0.8
        
        confidence_factors['completeness'] = completeness_score * 0.20
        reasoning.append(f"Analysis Completeness: {completeness_score*100:.0f}% (contributes {confidence_factors['completeness']:.1%})")
        
        # Calculate total confidence
        total_confidence = sum(confidence_factors.values())
        
        # Add summary
        reasoning.append(f"Total Confidence: {total_confidence:.1%} = " + 
                        " + ".join([f"{v:.1%}" for v in confidence_factors.values()]))
        
        return {
            'score': total_confidence,
            'reasoning': reasoning,
            'breakdown': confidence_factors
        }

    def _get_document_context_for_image(self, image_ref: ImageReference, user_question: str = "") -> str:
        """Intelligently get relevant document context for an image using RAG"""
        if not self.query_engine:
            return image_ref.geological_content
        
        try:
            # Always get page context first
            page_context = self.query_engine.search_by_page([image_ref.page_number])
            
            # Intelligently enhance vague questions using available context
            if user_question:
                enhanced_question = self._intelligently_enhance_question(
                    user_question, image_ref, page_context
                )
                
                # Search with the enhanced question
                if enhanced_question != user_question:
                    logger.info(f"Enhanced question from '{user_question}' to '{enhanced_question}'")
                    question_results = self.query_engine.search_all_content(
                        enhanced_question,
                        max_results=self.search_limits['general_query'],
                        similarity_threshold=self.config.similarity_threshold
                    )
                    page_context.extend(question_results)
            
            # Combine context intelligently
            context_parts = []
            for result in page_context[:8]:
                context_parts.append(f"Page {result.page_number}: {result.content}")
            
            return " | ".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to get document context: {e}")
            return image_ref.geological_content
    
    def _intelligently_enhance_question(self, question: str, image_ref: ImageReference, page_context: list) -> str:
        """Intelligently enhance vague questions using available context"""
        question_lower = question.lower().strip()
        
        # For result-specific questions, use page content to enhance
        if any(word in question_lower for word in ['result', 'figure', 'image', 'show', 'display']):
            # Extract meaningful terms from page context and image metadata
            context_terms = set()
            
            # Get terms from page content
            for result in page_context[:3]:
                words = result.content.lower().split()
                for word in words:
                    if (len(word) > 3 and 
                        word.isalpha() and 
                        word not in {'the', 'and', 'for', 'with', 'this', 'that', 'from'}):
                        context_terms.add(word)
            
            # Get terms from image geological content
            if image_ref.geological_content:
                geo_terms = image_ref.geological_content.split(';')
                for term in geo_terms[:5]:
                    if term.strip():
                        context_terms.add(term.strip().lower())
            
            # Get terms from context text
            if image_ref.context_text:
                context_words = image_ref.context_text.lower().split()
                for word in context_words:
                    if (len(word) > 4 and 
                        word.isalpha() and 
                        any(geo in word for geo in ['geo', 'rock', 'sed', 'strat', 'form'])):
                        context_terms.add(word)
            
            # Intelligently build enhanced question
            if context_terms:
                # Prioritize geological and technical terms
                priority_terms = []
                for term in context_terms:
                    if any(geo in term for geo in ['rock', 'sed', 'geo', 'strat', 'pale', 'meso', 'cret', 'tert', 'quat']):
                        priority_terms.append(term)
                
                if priority_terms:
                    enhanced = f"geological {' '.join(list(priority_terms)[:3])} classification"
                    return enhanced
                elif len(context_terms) > 0:
                    enhanced = f"geological {' '.join(list(context_terms)[:3])}"
                    return enhanced
        
        # For very generic questions, use image type and geological content
        if len(question.split()) <= 4:
            base_terms = []
            if image_ref.geological_content:
                geo_parts = image_ref.geological_content.split(';')[:2]
                for part in geo_parts:
                    if part.strip():
                        base_terms.append(part.strip())
            
            if base_terms:
                return f"geological {' '.join(base_terms)} analysis"
        
        return question
    
    def _analyze_with_ollama(self, image_ref: ImageReference, question: str, visual_description: str, document_context: str) -> str:
        """Use Ollama LLM for enhanced geological image analysis"""
        try:
            import requests
            
            # Prepare the prompt with all available context
            prompt = f"""You are a geological expert analyzing an image from a SPEM (Society of Petroleum Engineers of Mexico) document about Venezuelan geology.

Image Details:
- Page: {image_ref.page_number}
- Type: {image_ref.image_type}
- Image ID: {image_ref.image_id}
- Geological Content: {image_ref.geological_content}

Question: {question}

Visual Description: {visual_description}

Document Context: {document_context}

Based on the document context and image details, provide a detailed geological analysis focusing on:
1. What geological features or data are shown
2. The geological significance 
3. How this relates to Venezuelan geology and petroleum exploration
4. Any specific geological formations, well logs, or depositional environments depicted

Answer in a factual, professional manner based on the provided context."""

            # Call Ollama API with better parameters
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'gemma3:1b',
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'num_predict': self.api_settings['num_predict'],
                        'temperature': self.api_settings['temperature'],
                        'top_p': self.api_settings['top_p'],
                        'stop': ['**In Conclusion:**', 'Image Details:', '\n\n---']  # Stop tokens
                    }
                },
                timeout=self.api_settings['timeout']
            )
            
            if response.status_code == 200:
                result = response.json()
                ollama_response = result.get('response', '')
                
                # Clean up the response to remove repetitions and corruption
                if ollama_response:
                    # Remove excessive repetition patterns
                    cleaned_response = self._clean_ollama_response(ollama_response)
                    if len(cleaned_response.strip()) > 50:  # Ensure meaningful content
                        return cleaned_response
                
                # Fall back if response is too short or corrupted
                logger.warning("Ollama response was corrupted or too short, using fallback")
                return self._fallback_analysis(image_ref, question, document_context)
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return self._fallback_analysis(image_ref, question, document_context)
                
        except requests.exceptions.RequestException as e:
            logger.debug(f"Network request failed, using fallback: {e}")
            return self._fallback_analysis(image_ref, question, document_context)
        except Exception as e:
            logger.debug(f"Analysis failed, using fallback: {e}")
            return self._fallback_analysis(image_ref, question, document_context)
    
    def _clean_ollama_response(self, response: str) -> str:
        """Clean up corrupted or repetitive Ollama responses"""
        import re
        
        # Split into paragraphs and remove duplicates
        paragraphs = []
        seen_paragraphs = set()
        
        for paragraph in response.split('\n\n'):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Remove excessive asterisks and formatting
            paragraph = re.sub(r'\*{3,}', '**', paragraph)
            
            # Skip if this paragraph is too similar to one we've already seen
            # (simple check: if 80% of words are the same)
            words = set(paragraph.lower().split())
            is_duplicate = False
            for seen_para in seen_paragraphs:
                seen_words = set(seen_para.lower().split())
                if len(words) > 0 and len(seen_words) > 0:
                    overlap = len(words.intersection(seen_words)) / max(len(words), len(seen_words))
                    if overlap > 0.8:
                        is_duplicate = True
                        break
            
            if not is_duplicate and len(paragraph) > 20:  # Skip very short paragraphs
                paragraphs.append(paragraph)
                seen_paragraphs.add(paragraph)
                
                # Stop if we have enough content (prevent runaway responses)
                if len('\n\n'.join(paragraphs)) > 1500:
                    break
        
        # Join the cleaned paragraphs
        cleaned = '\n\n'.join(paragraphs)
        
        # Final cleanup: remove incomplete sentences at the end
        if cleaned and not cleaned[-1] in '.!?':
            # Find the last complete sentence
            last_punct = max(
                cleaned.rfind('.'), 
                cleaned.rfind('!'), 
                cleaned.rfind('?')
            )
            if last_punct > len(cleaned) // 2:  # Only if we're not cutting too much
                cleaned = cleaned[:last_punct + 1]
        
        return cleaned
    
    def _fallback_analysis(self, image_ref: ImageReference, question: str, document_context: str) -> str:
        """Enhanced geological analysis using integrated Qwen2-VL capabilities"""
        
        # Get actual visual analysis from Qwen2-VL
        if self.visual_analyzer:
            try:
                # Use Qwen2-VL to analyze the image with the specific question
                # Use the more direct analysis method that takes a specific question
                visual_result = self.visual_analyzer.analyze_image_with_question(
                    image_ref.file_path,
                    question,
                    image_ref.context_text  # Pass context for intelligent prompting
                )
                
                # Return the actual AI-generated response
                if visual_result:
                    return visual_result
                    
            except Exception as e:
                logger.debug(f"Visual analysis failed: {e}")
                # Fallback to generic analysis
                try:
                    visual_result = self.visual_analyzer.analyze_image_content(
                        image_ref.file_path,
                        f"{question} Context: {image_ref.geological_content}"
                    )
                    if visual_result and visual_result.get('detailed_caption'):
                        return visual_result['detailed_caption']
                except:
                    pass
        
        # Only use metadata-based fallback if visual analysis completely fails
        return f"Unable to analyze the image content for: {question}"
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            'system_info': {
                'config': asdict(self.config),
                'initialization_time': datetime.now().isoformat(),
                'components_status': {
                    'query_engine': self.query_engine is not None,
                    'vector_db_exists': Path(self.config.vector_db_path).exists()
                }
            }
        }
        
        if self.query_engine:
            db_stats = self.query_engine.get_database_statistics()
            stats['database_statistics'] = db_stats
        
        return stats
    
    def interactive_query_session(self):
        """Start an interactive query session"""
        print("\n" + "="*80)
        print("ENHANCED GEOLOGICAL RAG SYSTEM - INTERACTIVE SESSION")
        print("="*80)
        
        # Show system status
        stats = self.get_system_statistics()
        db_stats = stats.get('database_statistics', {})
        
        print(f"Vector Database: {db_stats.get('total_documents', 0)} documents")
        print(f"Collections: {db_stats.get('total_collections', 0)}")
        for collection, info in db_stats.get('collections', {}).items():
            count = info.get('document_count', 'Unknown')
            print(f"  - {collection}: {count} items")
        
        print("\nAvailable commands:")
        print("  search <query>           - Search all content")
        print("  text <query>            - Search text content only")
        print("  image <query>           - Search image content only")
        print("  geo <terms>             - Search geological concepts")
        print("  page <numbers> [query]  - Search specific pages")
        print("  find-images <concept>   - Find images by geological concept")
        print("  ask-image <image_id> <question> - Ask question about specific image")
        print("  list-images             - List all available images")
        print("  stats                   - Show system statistics")
        print("  help                    - Show this help")
        print("  quit                    - Exit session")
        print("\n" + "-"*80)
        
        while True:
            try:
                try:
                    user_input = input("\nRAG> ").strip()
                except EOFError:
                    print("\nGoodbye!")
                    break
                
                if not user_input:
                    continue
                
                parts = user_input.split()
                command = parts[0].lower()
                
                if command in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                elif command == 'help':
                    print(self._get_help_text())
                
                elif command == 'stats':
                    self._print_statistics()
                
                elif command == 'search':
                    if len(parts) > 1:
                        query = ' '.join(parts[1:])
                        results = self.search(query)
                        self._print_results(results, f"Search: '{query}'")
                    else:
                        print("Usage: search <query>")
                
                elif command == 'text':
                    if len(parts) > 1:
                        query = ' '.join(parts[1:])
                        results = self.search(query, "text")
                        self._print_results(results, f"Text Search: '{query}'")
                    else:
                        print("Usage: text <query>")
                
                elif command == 'image':
                    if len(parts) > 1:
                        query = ' '.join(parts[1:])
                        # Use disambiguation-aware image search
                        search_result = self.search_images_with_disambiguation(query)
                        
                        if search_result['disambiguation_needed']:
                            print(f"\nImage Search: '{query}' - Disambiguation Required")
                            print("="*50)
                            print(self.display_disambiguation_options(search_result['ambiguous_groups'], query))
                            
                            # Also show any non-ambiguous results
                            if search_result['images']:
                                print(f"\nOther relevant images:")
                                self._print_image_discovery(search_result['images'], query + " (other)")
                        else:
                            self._print_image_discovery(search_result['images'], query)
                    else:
                        print("Usage: image <query>")
                
                elif command == 'geo':
                    if len(parts) > 1:
                        terms = ' '.join(parts[1:]).split(',')
                        terms = [t.strip() for t in terms]
                        results = self.search_geological_concepts(terms)
                        self._print_results(results, f"Geological Search: {terms}")
                    else:
                        print("Usage: geo <term1,term2,...>")
                
                elif command == 'page':
                    if len(parts) > 1:
                        try:
                            page_nums = [int(x) for x in parts[1].split(',')]
                            query = ' '.join(parts[2:]) if len(parts) > 2 else None
                            results = self.search_by_page(page_nums, query)
                            search_desc = f"Page Search: {page_nums}"
                            if query:
                                search_desc += f" with query '{query}'"
                            self._print_results(results, search_desc)
                        except ValueError:
                            print("Usage: page <page1,page2,...> [optional_query]")
                    else:
                        print("Usage: page <page1,page2,...> [optional_query]")
                
                elif command == 'find-images':
                    if len(parts) > 1:
                        concept = ' '.join(parts[1:])
                        image_refs = self.search_images_by_concept(concept)
                        self._print_image_discovery(image_refs, concept)
                    else:
                        print("Usage: find-images <geological_concept>")
                
                elif command == 'ask-image':
                    if len(parts) > 2:
                        image_id = parts[1]
                        question = ' '.join(parts[2:])
                        analysis = self.analyze_image(image_id, question)
                        self._print_visual_analysis(analysis)
                    else:
                        print("Usage: ask-image <image_id> <question>")
                
                elif command == 'list-images':
                    self._list_available_images()
                
                else:
                    # Check for shorthand commands with attached parameters
                    shorthand_handled = self._handle_shorthand_command(command, parts)
                    if not shorthand_handled:
                        # Check if this is a follow-up question about search results
                        if self._is_followup_question(user_input):
                            self._handle_followup_question(user_input)
                        else:
                            print(f"Unknown command: {command}. Type 'help' for available commands.")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                logger.error(f"Interactive session error: {e}")
    
    def _print_statistics(self):
        """Print system statistics"""
        stats = self.get_system_statistics()
        
        print("\nSYSTEM STATISTICS")
        print("-" * 40)
        
        # Database stats
        db_stats = stats.get('database_statistics', {})
        print(f"Total Documents: {db_stats.get('total_documents', 0)}")
        print(f"Total Collections: {db_stats.get('total_collections', 0)}")
        
        print("\nCollections:")
        for name, info in db_stats.get('collections', {}).items():
            count = info.get('document_count', 'Error')
            print(f"  {name}: {count}")
        
        # Configuration
        config = stats.get('system_info', {}).get('config', {})
        print(f"\nConfiguration:")
        print(f"  Similarity Threshold: {config.get('similarity_threshold', 'N/A')}")
        print(f"  Max Results: {config.get('max_results_per_query', 'N/A')}")
        print(f"  Include Images: {config.get('include_images', 'N/A')}")
    
    def _print_results(self, results: List[QueryResult], title: str):
        """Print search results"""
        print(f"\n{title}")
        print("-" * len(title))
        
        if not results:
            print("No results found.")
            return
        
        # Store results for follow-up questions
        self.last_search_results = results
        
        for i, result in enumerate(results, 1):
            content = result.content
            if len(content) > 150:
                content = content[:150] + "..."
            
            geological_terms = ", ".join(result.geological_terms[:3])
            if len(result.geological_terms) > 3:
                geological_terms += f" (+{len(result.geological_terms) - 3} more)"
            
            print(f"\n{i}. [{result.content_type.upper()}] Page {result.page_number}")
            print(f"   Score: {result.relevance_score:.3f}")
            print(f"   Content: {content}")
            if geological_terms:
                print(f"   Geological Terms: {geological_terms}")
        
        # Show follow-up guidance
        print(f"\nFollow-up options: Ask 'what does result 1 show?' or 'explain page {results[0].page_number}'")
        
        # Show follow-up options for image results
        image_results = [r for r in results if r.content_type.upper() == 'IMAGE']
    
    def _print_image_discovery(self, image_refs: List[ImageReference], concept: str):
        """Print image discovery results"""
        print(f"\nIMAGE DISCOVERY RESULTS for '{concept}':")
        print("-" * 60)
        
        if not image_refs:
            print("No relevant images found.")
            return
        
        # Convert ImageReference objects to QueryResult for follow-up compatibility
        query_results = []
        for img_ref in image_refs:
            query_results.append(QueryResult(
                content=f"Geological image from SPEM document: {img_ref.image_type} | "
                       f"Page {img_ref.page_number} | Geological content: {img_ref.geological_content[:self.content_limits['preview_length']]}...",
                page_number=img_ref.page_number,
                relevance_score=img_ref.relevance_score,
                source_collection="image_metadata",
                content_type="IMAGE",
                metadata={'image_id': img_ref.image_id, 'image_type': img_ref.image_type},
                geological_terms=[]
            ))
        
        # Store results for follow-up questions
        self.last_search_results = query_results
        
        print(f"Found {len(image_refs)} relevant images:")
        
        for i, img_ref in enumerate(image_refs, 1):
            # Add relevance indicator based on learned thresholds
            strict_threshold = self.similarity_thresholds.get('strict', self.config.high_relevance_threshold)
            moderate_threshold = self.similarity_thresholds.get('moderate', self.config.similarity_threshold)
            
            if img_ref.relevance_score > strict_threshold:
                relevance_indicator = self.config.high_relevance_label
            elif img_ref.relevance_score > moderate_threshold:
                relevance_indicator = self.config.medium_relevance_label
            else:
                relevance_indicator = self.config.low_relevance_label
                
            print(f"\n{i}. {img_ref.image_id}")
            print(f"   Page {img_ref.page_number}")
            print(f"   Score: {img_ref.relevance_score:.3f} {relevance_indicator}")
            print(f"   Content: {img_ref.geological_content}")
            print(f"   Type: {img_ref.image_type} - {self._get_content_type_description(img_ref.image_type)}")
            print(f"   Usage: ask-{img_ref.image_id} <your question> OR ask-image {img_ref.image_id} <your question>")
        
        if image_refs:
            print(f"\nExample usage: ask-{image_refs[0].image_id} explain this diagram")
            print(f"Follow-up options: 'what does result 1 show?' or 'explain the image on page {image_refs[0].page_number}'")
    
    def _print_visual_analysis(self, analysis: VisualAnalysisResult):
        """Print detailed visual analysis results"""
        print(f"\nVISUAL ANALYSIS RESULT:")
        print("=" * 60)
        
        img_ref = analysis.image_ref
        print(f"Image: {img_ref.image_id} (Page {img_ref.page_number})")
        print(f"Question: {analysis.question}")
        print(f"Confidence: {analysis.confidence_score:.1%}")
        
        print(f"\nANSWER:")
        # Format answer for better readability
        answer_lines = analysis.answer.split('|')
        for line in answer_lines:
            if line.strip():
                print(f"   {line.strip()}")
        
        if analysis.geological_concepts:
            print(f"\nGEOLOGICAL CONCEPTS:")
            for concept in analysis.geological_concepts[:8]:
                print(f"   - {concept}")
        
        if analysis.analysis_details and 'error' not in analysis.analysis_details:
            print(f"\nImage Details:")
            print(f"   File: {img_ref.file_path}")
            print(f"   Type: {img_ref.image_type}")
            if img_ref.geological_content:
                print(f"   Geological Content: {img_ref.geological_content[:self.content_limits['preview_length']]}...")
        
        if analysis.analysis_details.get('error'):
            print(f"\nError: {analysis.analysis_details['error']}")
    
    def _list_available_images(self):
        """List all available images organized by page"""
        print(f"\nAVAILABLE IMAGES ({len(self.images_metadata)}):")
        print("-" * 60)
        
        if not self.images_metadata:
            print("No images available. Make sure the extraction process completed successfully.")
            return
        
        # Group by page
        by_page = {}
        for img_id, img_ref in self.images_metadata.items():
            page = img_ref.page_number
            if page not in by_page:
                by_page[page] = []
            by_page[page].append((img_id, img_ref))
        
        for page in sorted(by_page.keys()):
            print(f"\nPage {page}:")
            for img_id, img_ref in by_page[page]:
                content_preview = img_ref.geological_content[:60] + "..." if len(img_ref.geological_content) > 60 else img_ref.geological_content
                print(f"   {img_id}")
                print(f"      Type: {img_ref.image_type} | Content: {content_preview}")
        
        print(f"\nUse 'ask-image <image_id> <question>' to analyze any image")
        print(f"Use 'find-images <concept>' to search for images by geological concept")
    
    def _handle_shorthand_command(self, command: str, parts: list) -> bool:
        """Handle shorthand commands with attached parameters
        
        Returns True if shorthand was handled, False otherwise
        """
        # Handle ask-<image_id> shorthand
        if command.startswith('ask-') and len(command) > 4:
            image_id = command[4:]
            if len(parts) > 1:
                question = ' '.join(parts[1:])
                analysis = self.analyze_image(image_id, question)
                self._print_visual_analysis(analysis)
            else:
                print(f"Usage: ask-{image_id} <question>")
            return True
        
        # Handle image-<query> shorthand
        elif command.startswith('image-') and len(command) > 6:
            query = command[6:].replace('-', ' ')
            if len(parts) > 1:
                query += ' ' + ' '.join(parts[1:])
            search_result = self.search_images_with_disambiguation(query)
            if search_result['disambiguation_needed']:
                print(f"\nImage Search: '{query}' - Disambiguation Required")
                print("="*50)
                print(self.display_disambiguation_options(search_result['ambiguous_groups'], query))
                if search_result['images']:
                    print(f"\nOther relevant images:")
                    self._print_image_discovery(search_result['images'], query + " (other)")
            else:
                self._print_image_discovery(search_result['images'], query)
            return True
        
        # Handle text-<query> shorthand
        elif command.startswith('text-') and len(command) > 5:
            query = command[5:].replace('-', ' ')
            if len(parts) > 1:
                query += ' ' + ' '.join(parts[1:])
            results = self.search(query, "text")
            self._print_results(results, f"Text Search: '{query}'")
            return True
        
        # Handle search-<query> shorthand
        elif command.startswith('search-') and len(command) > 7:
            query = command[7:].replace('-', ' ')
            if len(parts) > 1:
                query += ' ' + ' '.join(parts[1:])
            results = self.search(query)
            self._print_results(results, f"Search: '{query}'")
            return True
        
        # Handle page-<number> shorthand
        elif command.startswith('page-') and len(command) > 5:
            page_part = command[5:]
            try:
                # Handle page-5,10,15 or page-5
                page_nums = [int(x) for x in page_part.split(',')]
                query = ' '.join(parts[1:]) if len(parts) > 1 else None
                results = self.search_by_page(page_nums, query)
                search_desc = f"Page Search: {page_nums}"
                if query:
                    search_desc += f" with query '{query}'"
                self._print_results(results, search_desc)
                return True
            except ValueError:
                print(f"Invalid page format: {page_part}. Use page-5 or page-5,10,15")
                return True
        
        # Handle geo-<terms> shorthand
        elif command.startswith('geo-') and len(command) > 4:
            terms_part = command[4:]
            if len(parts) > 1:
                terms_part += ' ' + ' '.join(parts[1:])
            
            # Handle both comma-separated and hyphen-separated terms
            if ',' in terms_part:
                terms = [t.strip() for t in terms_part.split(',')]
            else:
                terms = [t.strip() for t in terms_part.replace('-', ' ').split()]
            
            results = self.search_geological_concepts(terms)
            self._print_results(results, f"Geological Search: {terms}")
            return True
        
        # Handle find-images-<concept> shorthand (alternative to find-images)
        elif command.startswith('find-images-') and len(command) > 12:
            concept = command[12:].replace('-', ' ')
            if len(parts) > 1:
                concept += ' ' + ' '.join(parts[1:])
            image_refs = self.search_images_by_concept(concept)
            self._print_image_discovery(image_refs, concept)
            return True
        
        return False
    
    def _is_followup_question(self, user_input: str) -> bool:
        """Check if user input is a follow-up question about previous search results"""
        if not self.last_search_results:
            return False
        
        question_keywords = ['what', 'how', 'why', 'explain', 'describe', 'show', 'convey', 'mean', 'tell']
        reference_keywords = ['result', 'image', 'page', 'first', 'second', 'third', 'that', 'this']
        
        input_lower = user_input.lower()
        
        has_question = any(keyword in input_lower for keyword in question_keywords)
        has_reference = any(keyword in input_lower for keyword in reference_keywords)
        
        return has_question and (has_reference or len(user_input.split()) > 3)
    
    def _handle_followup_question(self, user_input: str):
        """Handle follow-up questions about previous search results"""
        if not self.last_search_results:
            print("No previous search results to reference. Please search first.")
            return
        
        # Look for result number reference or page reference
        import re
        result_match = re.search(r'result\s+(\d+)', user_input.lower())
        page_match = re.search(r'page\s+(\d+)', user_input.lower())
        number_match = re.search(r'\b(\d+)\s+(result|page)', user_input.lower())
        standalone_number = re.search(r'\b(\d+)\b', user_input.lower())
        
        target_result = None
        result_description = ""
        
        if result_match:
            # "what does result 1 show?"
            result_num = int(result_match.group(1))
            if 1 <= result_num <= len(self.last_search_results):
                target_result = self.last_search_results[result_num - 1]
                result_description = f"result {result_num}"
        elif page_match:
            # "explain page 9" or "i meant the page 9"
            page_num = int(page_match.group(1))
            # Find the result from that page
            for i, result in enumerate(self.last_search_results):
                if result.page_number == page_num:
                    target_result = result
                    result_description = f"result {i+1} (page {page_num})"
                    break
        elif number_match:
            # "the 1 result" or "1 page"
            num = int(number_match.group(1))
            ref_type = number_match.group(2)
            if ref_type == "result" and 1 <= num <= len(self.last_search_results):
                target_result = self.last_search_results[num - 1]
                result_description = f"result {num}"
            elif ref_type == "page":
                for i, result in enumerate(self.last_search_results):
                    if result.page_number == num:
                        target_result = result
                        result_description = f"result {i+1} (page {num})"
                        break
        elif standalone_number:
            # Just "1" or "page 9" - try to interpret contextually
            num = int(standalone_number.group(1))
            if 1 <= num <= len(self.last_search_results) and len(self.last_search_results) <= 10:
                # Likely referring to result number if we have few results
                target_result = self.last_search_results[num - 1]
                result_description = f"result {num}"
            else:
                # Might be a page number
                for i, result in enumerate(self.last_search_results):
                    if result.page_number == num:
                        target_result = result
                        result_description = f"result {i+1} (page {num})"
                        break
        else:
            # Default to first image result if no specific reference
            image_results = [r for r in self.last_search_results if r.content_type.upper() == 'IMAGE']
            if image_results:
                target_result = image_results[0]
                result_description = "first image result"
            elif self.last_search_results:
                target_result = self.last_search_results[0]
                result_description = "first result"
        
        if not target_result:
            print("Could not identify which result you're asking about.")
            print("Try: 'what does result 1 show?' or 'explain page 9'")
            print(f"Available results: {len(self.last_search_results)} items from pages {', '.join(str(r.page_number) for r in self.last_search_results[:5])}")
            return
        
        print(f"Explaining {result_description}...")
        
        # If it's an image, analyze it
        if target_result.content_type.upper() == 'IMAGE':
            # Extract image ID from metadata
            image_id = target_result.metadata.get('image_id', '')
            if image_id and image_id in self.images_metadata:
                analysis = self.analyze_image(image_id, user_input)
                self._print_visual_analysis(analysis)
            else:
                print(f"Could not find image details for page {target_result.page_number}")
        else:
            # For text results, provide the content and context
            print(f"\nCONTENT FROM PAGE {target_result.page_number}:")
            print(f"Score: {target_result.relevance_score:.3f}")
            print(f"Content: {target_result.content}")
            if target_result.geological_terms:
                print(f"Geological Terms: {', '.join(target_result.geological_terms[:5])}")
            print(f"\nThis content relates to: {user_input}")
            
            # Use RAG context to provide more detailed answer
            related_context = self._get_related_context(target_result, user_input)
            if related_context:
                print(f"\nRelated Context:")
                for ctx in related_context[:2]:
                    print(f"   - Page {ctx.page_number}: {ctx.content[:self.content_limits['preview_length']]}...")
    
    def _get_related_context(self, target_result: QueryResult, question: str) -> List[QueryResult]:
        """Get related context for a text result using RAG"""
        if not self.query_engine:
            return []
        
        # Search for related content based on the question and result content
        search_query = f"{question} {' '.join(target_result.geological_terms[:3])}"
        try:
            related_results = self.query_engine.search_all_content(
                search_query, 
                max_results=self.search_limits['general_query'],
                similarity_threshold=self.config.similarity_threshold
            )
            # Filter out the original result
            return [r for r in related_results if r != target_result]
        except Exception as e:
            logger.error(f"Failed to get related context: {e}")
            return []
    
    def _get_content_type_description(self, image_type: str) -> str:
        """Get a human-readable description of what an image type contains"""
        descriptions = {
            'well_log': 'well logging data (gamma ray curves, resistivity logs, SP logs)',
            'figure': 'geological diagrams, maps, or cross-sections', 
            'core_photo': 'photographs of rock core samples',
            'seismic': 'seismic survey data and interpretations',
            'stratigraphic': 'stratigraphic columns and sequence charts',
            'other': 'miscellaneous geological content',
            'unknown': 'unclassified geological content'
        }
        return descriptions.get(image_type.lower(), f'{image_type} geological content')
    
    def _get_help_text(self) -> str:
        """Get help text"""
        return """
ENHANCED GEOLOGICAL RAG SYSTEM HELP
====================================

Two-Step Visual RAG Workflow:
1. DISCOVERY: Find relevant images and content
2. EXPLANATION: Ask detailed questions about specific images

Commands:
  search <query>           - Search all content types
  text <query>            - Search only text content
  image <query>           - Search only image descriptions
  geo <term1,term2,...>   - Search for geological concepts
  page <1,2,3> [query]    - Search specific pages (optionally with query)
  find-images <concept>   - Find images related to geological concept
  ask-image <id> <question> - Ask question about specific image
  ask-<image_id> <question> - Shorthand for ask-image command
  list-images             - List all available images
  stats                   - Show system statistics
  help                    - Show this help
  quit                    - Exit session

Shorthand Commands (attach parameters with hyphens):
  search-<query>          - Same as search <query>
  text-<query>            - Same as text <query>
  image-<query>           - Same as image <query>
  page-<numbers>          - Same as page <numbers> (e.g., page-5,10,15)
  geo-<terms>             - Same as geo <terms> (e.g., geo-gamma-ray,resistivity)
  find-images-<concept>   - Same as find-images <concept>

Examples:
  search gamma ray interpretation
  find-images sequence stratigraphy
  ask-image page_5_image_1 explain this diagram
  ask-spem_img_003 what depositional environment?
  ask-spem_img_016_002_d8afd457 what does this convey?
  text sequence stratigraphy
  image well log patterns
  geo gamma ray,resistivity,SP
  page 5,10,15 depositional environment

Shorthand Examples:
  search-gamma-ray-interpretation
  text-sequence-stratigraphy
  image-well-log-patterns
  page-5,10,15 depositional environment
  geo-gamma-ray,resistivity,SP
  find-images-systems-tract

Visual RAG Workflow Example:
  RAG> find-images systems tract
  Found: page_5_image_2 (sequence diagram), page_8_image_1 (stacking patterns)
  
  RAG> ask-image page_5_image_2 explain the systems tracts
  Answer: This diagram shows three systems tracts (LST, TST, HST)...

The system searches across:
- Text content from SPEM document pages
- Image descriptions and metadata with visual analysis
- Geological terminology and concepts
- Combined multi-modal content with AI-powered image understanding

Results are ranked by semantic similarity and geological relevance.
        """


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced Geological RAG System")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Start interactive query session")
    parser.add_argument("--query", "-q", type=str, help="Single query to run")
    parser.add_argument("--search-type", "-t", choices=["all", "text", "image", "geological", "combined"],
                       default="all", help="Type of search to perform")
    parser.add_argument("--max-results", "-m", type=int, default=5, 
                       help="Maximum results to return")
    parser.add_argument("--create-db", "-c", action="store_true",
                       help="Force recreation of vector database")
    parser.add_argument("--stats", "-s", action="store_true",
                       help="Show system statistics")
    
    args = parser.parse_args()
    
    # Initialize system
    config = RAGSystemConfig()
    config.max_results_per_query = args.max_results
    
    try:
        rag_system = EnhancedGeologicalRAGSystem(config)
        
        if args.create_db:
            print("Recreating vector database...")
            rag_system._create_vector_database()
        
        if args.stats:
            rag_system._print_statistics()
        
        if args.query:
            print(f"Searching for: '{args.query}'")
            results = rag_system.search(args.query, args.search_type, args.max_results)
            rag_system._print_results(results, f"Search Results for '{args.query}'")
        
        if args.interactive:
            rag_system.interactive_query_session()
        
        if not any([args.query, args.interactive, args.stats, args.create_db]):
            print("No action specified. Use --help for options or --interactive for query session.")
    
    except Exception as e:
        logger.error(f"RAG system error: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())