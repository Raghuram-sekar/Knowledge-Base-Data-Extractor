"""
VISUAL RAG ENHANCEMENT - IMAGE ANALYSIS ENGINE
================================================================================
Adds deep image understanding and visual question answering capabilities
to the existing geological RAG system.
================================================================================
"""

import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import requests
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from loguru import logger
import re
from datetime import datetime


class VisualGeologicalAnalyzer:
    """Enhanced image analysis for geological content understanding"""
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize vision-language models
        self._initialize_models()
        
        # Geological knowledge for image interpretation
        self.geological_patterns = self._load_geological_patterns()
        
        logger.info(f"Visual Geological Analyzer initialized on {self.device}")
    
    def _initialize_models(self):
        """Initialize vision-language models for image understanding"""
        try:
            # Qwen2-VL for advanced technical document analysis
            logger.info("Loading Qwen2-VL model for image understanding...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
            
            if self.device == "cpu":
                self.model.to(self.device)
            
            # Check if Ollama is available for enhanced geological analysis
            self.ollama_available = self._check_ollama_availability()
            
            logger.info("Vision-language models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load vision models: {e}")
            # Fallback to basic analysis
            self.model = None
            self.processor = None
            self.ollama_available = False
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama is running and available"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _load_geological_patterns(self) -> Dict[str, Dict]:
        """No hardcoded patterns - rely entirely on Qwen-2 visual analysis"""
        return {
            'note': 'Using pure AI visual analysis without predefined patterns'
        }
    
    def analyze_image_with_question(self, image_path: str, question: str, context_text: str = "") -> str:
        """
        Analyze image content with a specific question using Qwen2-VL
        
        Args:
            image_path: Path to the image file
            question: Specific question to answer about the image
            context_text: Additional context about the image
            
        Returns:
            Direct answer to the question based on image analysis
        """
        try:
            # Load image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Create professional, analytical prompt for detailed geological analysis
            base_prompt = f"Analyze this geological/technical image and {question.lower()}"
            
            # Minimal, open-ended prompt that lets the model naturally interpret any image
            comprehensive_prompt = f"""You are an expert geologist analyzing technical geological data. {base_prompt}

CRITICAL: Your goal is to INTERPRET and EXPLAIN what this data means geologically, not just describe what you see.

Think like an expert presenting findings to colleagues:
• What geological story does this tell?
• What processes created what you observe?
• What are the implications for petroleum exploration or geological understanding?
• What insights can be drawn from the patterns and relationships you see?

Write naturally in flowing paragraphs, as you would in a professional geological report. Connect observations to geological processes and explain the significance of what you're analyzing.

Focus on interpretation and meaning, not surface-level description. Show your geological reasoning and expertise."""
            
            conversation = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": comprehensive_prompt}
                    ]
                }
            ]
            
            # Apply chat template
            text_prompt = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text_prompt], 
                images=[image], 
                padding=True, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response optimized for natural, insightful interpretation
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1000,     # Sufficient for comprehensive analysis
                    do_sample=True,          # Enable natural language generation
                    temperature=0.5,         # Balanced - not too rigid, not too creative
                    top_p=0.92,              # Allow diverse vocabulary and phrasing
                    repetition_penalty=1.15, # Strongly discourage repetition
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Extract generated text
            generated_ids = [
                output_ids[i][len(inputs["input_ids"][i]):] 
                for i in range(len(output_ids))
            ]
            
            response = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            # Log the response and return it directly (remove double-generation fallback)
            if response:
                logger.debug(f"Question-specific analysis complete for {image_path}")
                return response
            else:
                logger.warning(f"Empty response from visual analysis for {image_path}")
                return "Unable to analyze image content."
            
        except Exception as e:
            logger.error(f"Question-specific analysis failed for {image_path}: {e}")
            return f"Unable to analyze image for question: {question}"

    def analyze_image_content(self, image_path: str, context_text: str = "") -> Dict[str, Any]:
        """
        Perform pure Qwen-2 analysis of geological image content
        
        Args:
            image_path: Path to the image file
            context_text: Surrounding text context from the document
            
        Returns:
            Analysis results based entirely on Qwen-2 visual understanding
        """
        try:
            # Load image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            analysis = {
                'image_path': str(image_path),
                'image_size': image.size,
                'timestamp': datetime.now().isoformat(),
                'analysis_methods': ['Qwen2-VL pure visual analysis']
            }
            
            # Generate detailed analysis using Qwen-2 only with comprehensive prompting
            if self.model is not None:
                caption = self.analyze_image_with_question(
                    image_path,
                    "What does this image contain and what information does it present?",
                    context_text
                )
                analysis['detailed_caption'] = caption
            else:
                analysis['detailed_caption'] = "Visual analysis model not available"
            
            logger.debug(f"Pure visual analysis complete for {image_path}")
            return analysis
            
        except Exception as e:
            logger.error(f"Pure visual analysis failed for {image_path}: {e}")
            return {
                'image_path': str(image_path),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_detailed_caption(self, image: Image.Image) -> str:
        """Generate detailed caption using comprehensive intelligent prompting"""
        try:
            # Use the same comprehensive prompt structure for consistency
            comprehensive_prompt = """Analyze this image thoroughly and describe what it contains.

Please examine every visible element and provide:
1. EXACT TEXT: Read and transcribe any visible text, labels, titles, or annotations exactly as they appear
2. CLASSIFICATIONS: Identify any classification systems, legends, keys, or organizational structures shown
3. TECHNICAL TERMS: Note all scientific, geological, or technical terminology present
4. VISUAL ELEMENTS: Describe charts, graphs, diagrams, maps, or data visualizations
5. RELATIONSHIPS: Explain how different elements relate to each other
6. CONTEXT: Determine what type of scientific or technical document this appears to be from

DETAILED EXPLANATION: After listing the technical details, provide a thorough, flowing explanation that:
- Explains the overall purpose and significance of what's shown
- Describes how the different elements work together
- Discusses the scientific or technical implications
- Explains what insights or information this image provides
- Contextualizes the findings within the broader field
- Discusses any patterns, trends, or relationships visible
- Explains the practical applications or interpretations possible
- Provides a comprehensive understanding of what makes this image important or informative

Focus on precision and completeness. If this shows any kind of legend, key, classification system, time periods, rock types, formations, or scientific data, capture ALL the specific details and terminology visible. End with a detailed, comprehensive explanation that fully describes the significance and implications of what you've observed."""
            
            # Prepare the conversation for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": comprehensive_prompt}
                    ]
                }
            ]
            
            # Prepare inputs for the model
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=200)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                caption = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            
            return caption
            
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return "Failed to generate image caption"
    
    def _analyze_geological_patterns(self, image: Image.Image, context_text: str) -> Dict[str, Any]:
        """Use pure Qwen-2 visual analysis instead of hardcoded patterns"""
        analysis = {
            'pattern_type': 'ai_analyzed',
            'geological_interpretation': 'Analysis performed by Qwen-2 visual AI',
            'identified_features': ['AI visual analysis'],
            'depositional_environment': 'determined_by_ai'
        }
        
        # Return minimal analysis since we're relying entirely on Qwen-2 
        # for genuine visual understanding instead of hardcoded patterns
        return analysis
    
    def _is_well_log_pattern(self, img_array: np.ndarray, context: str) -> bool:
        """Detect if image contains well log patterns"""
        # Check context for well log keywords
        well_log_keywords = ['gamma ray', 'resistivity', 'neutron', 'density', 'SP', 'log', 'curve']
        context_lower = context.lower()
        
        keyword_count = sum(1 for keyword in well_log_keywords if keyword in context_lower)
        
        # Check image characteristics (vertical orientation, curve-like patterns)
        height, width = img_array.shape[:2]
        aspect_ratio = height / width
        
        return keyword_count >= 2 or aspect_ratio > 1.5
    
    def _is_sequence_stratigraphy_diagram(self, img_array: np.ndarray, context: str) -> bool:
        """Detect sequence stratigraphy diagrams"""
        seq_strat_keywords = ['sequence', 'TST', 'LST', 'HST', 'systems tract', 'MFS', 'SB', 'boundary']
        context_lower = context.lower()
        
        return sum(1 for keyword in seq_strat_keywords if keyword in context_lower) >= 2
    
    def _is_depositional_diagram(self, img_array: np.ndarray, context: str) -> bool:
        """Detect depositional environment diagrams"""
        dep_env_keywords = ['depositional', 'environment', 'facies', 'marine', 'fluvial', 'deltaic']
        context_lower = context.lower()
        
        return sum(1 for keyword in dep_env_keywords if keyword in context_lower) >= 1
    
    def _interpret_well_log_pattern(self, img_array: np.ndarray, context: str) -> str:
        """Interpret well log patterns"""
        interpretations = []
        
        # Basic pattern recognition based on context
        if 'funnel' in context.lower():
            interpretations.append(self.geological_patterns['well_log_patterns']['funnel']['geological_meaning'])
        elif 'bell' in context.lower():
            interpretations.append(self.geological_patterns['well_log_patterns']['bell']['geological_meaning'])
        elif 'boxcar' in context.lower() or 'uniform' in context.lower():
            interpretations.append(self.geological_patterns['well_log_patterns']['boxcar']['geological_meaning'])
        
        if 'coarsening upward' in context.lower():
            interpretations.append("Progradational sequence indicating deltaic or shallow marine environment")
        elif 'fining upward' in context.lower():
            interpretations.append("Retrogradational sequence indicating channel fill or transgressive environment")
        
        return "; ".join(interpretations) if interpretations else "Well log pattern requiring detailed analysis"
    
    def _interpret_sequence_diagram(self, img_array: np.ndarray, context: str) -> str:
        """Interpret sequence stratigraphy diagrams"""
        interpretations = []
        
        # Identify systems tracts
        for tract, meaning in self.geological_patterns['sequence_stratigraphy']['systems_tracts'].items():
            if tract in context.upper():
                interpretations.append(f"{tract}: {meaning}")
        
        # Identify key surfaces
        for surface, meaning in self.geological_patterns['sequence_stratigraphy']['surfaces'].items():
            if surface in context.upper():
                interpretations.append(f"{surface}: {meaning}")
        
        return "; ".join(interpretations) if interpretations else "Sequence stratigraphy diagram showing systems tracts and bounding surfaces"
    
    def _interpret_depositional_diagram(self, img_array: np.ndarray, context: str) -> str:
        """Interpret depositional environment diagrams"""
        for env, characteristics in self.geological_patterns['depositional_environments'].items():
            if env in context.lower():
                return f"Depositional environment diagram showing {env} characteristics: {characteristics}"
        
        return "Depositional environment diagram requiring detailed analysis"
    
    def _extract_image_text(self, image: Image.Image) -> str:
        """Extract text from image using OCR"""
        try:
            import pytesseract
            import os
            
            # Add tesseract to PATH if not already there
            tesseract_path = r"C:\Program Files\Tesseract-OCR"
            if tesseract_path not in os.environ.get('PATH', ''):
                os.environ['PATH'] += f";{tesseract_path}"
                pytesseract.pytesseract.tesseract_cmd = os.path.join(tesseract_path, 'tesseract.exe')
            
            # Convert image to grayscale for better OCR
            gray_image = image.convert('L')
            
            # Extract text
            extracted_text = pytesseract.image_to_string(gray_image)
            
            # Clean extracted text
            cleaned_text = re.sub(r'\s+', ' ', extracted_text).strip()
            
            return cleaned_text
            
        except ImportError:
            logger.warning("pytesseract not available for OCR")
            return ""
        except Exception as e:
            logger.warning(f"OCR extraction not available: {e}. See README file for more information.")
            return ""
    
    def _identify_geological_concepts(self, caption: str, image_text: str, context_text: str) -> List[str]:
        """Identify geological concepts from all text sources"""
        all_text = f"{caption} {image_text} {context_text}".lower()
        
        concepts = set()
        
        # Check all geological pattern categories
        for category, patterns in self.geological_patterns.items():
            if isinstance(patterns, dict):
                for pattern_key, pattern_info in patterns.items():
                    if pattern_key in all_text:
                        concepts.add(pattern_key)
                    
                    # Check within nested structures
                    if isinstance(pattern_info, dict):
                        for sub_key in pattern_info.keys():
                            if sub_key in all_text:
                                concepts.add(sub_key)
        
        # Add common geological terms
        geological_terms = [
            'gamma ray', 'resistivity', 'neutron', 'density', 'SP', 'log',
            'sequence', 'stratigraphy', 'systems tract', 'depositional',
            'environment', 'marine', 'fluvial', 'deltaic', 'facies',
            'LST', 'TST', 'HST', 'SB', 'MFS', 'TS'
        ]
        
        for term in geological_terms:
            if term in all_text:
                concepts.add(term)
        
        return list(concepts)
    
    def answer_visual_question(self, image_path: str, question: str, context_text: str = "") -> Dict[str, Any]:
        """
        Answer questions about specific images
        
        Args:
            image_path: Path to the image
            question: Question about the image
            context_text: Surrounding text context
            
        Returns:
            Structured answer with geological explanation
        """
        try:
            # Perform comprehensive image analysis
            analysis = self.analyze_image_content(image_path, context_text)
            
            # Generate contextual answer based on question type
            answer = self._generate_contextual_answer(question, analysis, context_text)
            
            return {
                'question': question,
                'answer': answer,
                'image_analysis': analysis,
                'confidence': analysis.get('enhanced_confidence', self._calculate_answer_confidence(analysis, question)),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Visual question answering failed: {e}")
            return {
                'question': question,
                'answer': f"Unable to analyze image: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_contextual_answer(self, question: str, analysis: Dict, context_text: str) -> str:
        """Generate contextual answer based on question and image analysis"""
        # First get basic description from BLIP
        basic_description = analysis.get('description', 'Unable to analyze image')
        geological_content = analysis.get('geological_interpretation', '')
        
        # Use Ollama for enhanced geological analysis if available
        enhanced_answer, confidence = self._enhance_with_ollama(
            basic_description, question, context_text, geological_content
        )
        
        if enhanced_answer != basic_description:
            # Store confidence from Ollama enhancement
            analysis['enhanced_confidence'] = confidence
            return enhanced_answer
        
        # Fallback to original logic for specific question types
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what', 'describe', 'explain']):
            return self._explain_image_content(analysis, context_text)
        
        elif any(word in question_lower for word in ['pattern', 'signature', 'shape']):
            return self._explain_geological_pattern(analysis)
        
        elif any(word in question_lower for word in ['environment', 'depositional']):
            return self._explain_depositional_environment(analysis)
        
        elif any(word in question_lower for word in ['sequence', 'stratigraphy', 'systems tract']):
            return self._explain_sequence_stratigraphy(analysis)
        
        elif any(word in question_lower for word in ['log', 'curve', 'gamma ray', 'resistivity']):
            return self._explain_well_logs(analysis)
        
        else:
            return self._provide_general_explanation(analysis, context_text)
    
    def _explain_image_content(self, analysis: Dict, context_text: str) -> str:
        """Provide general explanation of image content"""
        explanation_parts = []
        
        if 'detailed_caption' in analysis:
            explanation_parts.append(f"Image description: {analysis['detailed_caption']}")
        
        if 'geological_interpretation' in analysis and analysis['geological_interpretation']:
            explanation_parts.append(f"Geological interpretation: {analysis['geological_interpretation']}")
        
        if 'geological_concepts' in analysis and analysis['geological_concepts']:
            concepts = ", ".join(analysis['geological_concepts'][:5])
            explanation_parts.append(f"Key geological concepts: {concepts}")
        
        if 'extracted_text' in analysis and analysis['extracted_text']:
            text = analysis['extracted_text'][:100] + "..." if len(analysis['extracted_text']) > 100 else analysis['extracted_text']
            explanation_parts.append(f"Text in image: {text}")
        
        return " | ".join(explanation_parts) if explanation_parts else "This geological image requires detailed expert analysis."
    
    def _explain_geological_pattern(self, analysis: Dict) -> str:
        """Explain geological patterns identified in the image"""
        pattern_type = analysis.get('pattern_type', 'unknown')
        
        if pattern_type == 'well_log':
            return f"This shows well log patterns. {analysis.get('geological_interpretation', 'Pattern analysis indicates specific depositional characteristics.')}"
        elif pattern_type == 'sequence_stratigraphy':
            return f"This displays sequence stratigraphic patterns. {analysis.get('geological_interpretation', 'Shows systems tracts and sequence boundaries.')}"
        else:
            return f"Geological pattern type: {pattern_type}. {analysis.get('geological_interpretation', 'Requires detailed geological analysis.')}"
    
    def _explain_depositional_environment(self, analysis: Dict) -> str:
        """Explain depositional environment aspects"""
        interpretation = analysis.get('geological_interpretation', '')
        if 'environment' in interpretation or 'depositional' in interpretation:
            return interpretation
        else:
            concepts = analysis.get('geological_concepts', [])
            env_concepts = [c for c in concepts if c in ['marine', 'fluvial', 'deltaic', 'aeolian']]
            if env_concepts:
                return f"This image relates to {', '.join(env_concepts)} depositional environments."
            return "Depositional environment characteristics visible in geological patterns."
    
    def _explain_sequence_stratigraphy(self, analysis: Dict) -> str:
        """Explain sequence stratigraphic elements"""
        concepts = analysis.get('geological_concepts', [])
        seq_concepts = [c for c in concepts if c in ['LST', 'TST', 'HST', 'SB', 'MFS', 'TS']]
        
        if seq_concepts:
            return f"Sequence stratigraphic elements visible: {', '.join(seq_concepts)}. {analysis.get('geological_interpretation', 'These represent different phases of relative sea-level change and sediment supply.')}"
        else:
            return analysis.get('geological_interpretation', 'Sequence stratigraphic framework showing systems tracts and key surfaces.')
    
    def _explain_well_logs(self, analysis: Dict) -> str:
        """Explain well log characteristics"""
        interpretation = analysis.get('geological_interpretation', '')
        if 'log' in interpretation or 'curve' in interpretation:
            return interpretation
        else:
            concepts = analysis.get('geological_concepts', [])
            log_concepts = [c for c in concepts if c in ['gamma ray', 'resistivity', 'neutron', 'density', 'SP']]
            if log_concepts:
                return f"Well log types visible: {', '.join(log_concepts)}. These provide insights into lithology and depositional characteristics."
            return "Well log patterns indicating specific geological and petrophysical properties."
    
    def _provide_general_explanation(self, analysis: Dict, context_text: str) -> str:
        """Provide general geological explanation"""
        return self._explain_image_content(analysis, context_text)
    
    def _calculate_answer_confidence(self, analysis: Dict, question: str) -> float:
        """Calculate confidence score for the answer"""
        confidence_factors = 0
        total_factors = 5
        
        # Factor 1: Successful image analysis
        if 'error' not in analysis:
            confidence_factors += 1
        
        # Factor 2: Geological concepts identified
        if analysis.get('geological_concepts') and len(analysis['geological_concepts']) > 0:
            confidence_factors += 1
        
        # Factor 3: Pattern recognition successful
        if analysis.get('pattern_type') and analysis['pattern_type'] != 'unknown':
            confidence_factors += 1
        
        # Factor 4: Geological interpretation available
        if analysis.get('geological_interpretation') and len(analysis['geological_interpretation']) > 20:
            confidence_factors += 1
        
        # Factor 5: Question keywords match analysis
        question_lower = question.lower()
        analysis_text = str(analysis).lower()
        question_words = set(question_lower.split())
        common_words = question_words.intersection(set(analysis_text.split()))
        if len(common_words) >= 2:
            confidence_factors += 1
        
        return confidence_factors / total_factors
    
    def _enhance_with_ollama(self, basic_description: str, question: str, context: str, geological_content: str) -> tuple[str, float]:
        """Use Ollama LLM for enhanced geological interpretation"""
        if not self.ollama_available:
            return basic_description, 0.4
        
        try:
            # Create comprehensive prompt for geological analysis
            prompt = f"""You are a geological expert analyzing well logs and geological images. 

Context from document: {context}

Geological content indicators: {geological_content}

Basic image description: {basic_description}

User question: {question}

Based on the geological context and image description, provide a detailed geological interpretation. Focus on:
1. What geological features or patterns are visible
2. What this tells us about depositional environment  
3. Any sequence stratigraphic significance
4. Well log responses and their geological meaning

Provide a factual, detailed geological analysis in 2-3 sentences."""

            # Call Ollama API
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma3:1b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                enhanced_answer = result.get('response', '').strip()
                
                if enhanced_answer and len(enhanced_answer) > 50:
                    # Higher confidence for Ollama-enhanced analysis
                    return enhanced_answer, 0.8
                
        except Exception as e:
            logger.error(f"Ollama analysis failed: {e}")
        
        # Fallback to basic description
        return basic_description, 0.4


def main():
    """Test the visual geological analyzer"""
    analyzer = VisualGeologicalAnalyzer()
    
    # Test image paths (update with actual extracted images)
    test_images = [
        "extracted_spem_complete/images/page_5_image_2.png",
        "extracted_spem_complete/images/page_6_image_1.png"
    ]
    
    for image_path in test_images:
        if Path(image_path).exists():
            print(f"\nAnalyzing: {image_path}")
            
            # Test visual question answering
            questions = [
                "What does this image show?",
                "Explain the geological patterns in this image",
                "What depositional environment does this represent?"
            ]
            
            for question in questions:
                result = analyzer.answer_visual_question(image_path, question)
                print(f"Q: {question}")
                print(f"A: {result['answer']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print("-" * 50)


if __name__ == "__main__":
    main()