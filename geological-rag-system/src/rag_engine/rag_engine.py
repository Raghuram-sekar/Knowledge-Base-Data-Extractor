"""
RAG Engine for Geological Interpretation
Combines SPEM knowledge retrieval with LLM inference for well log interpretation
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from loguru import logger
import openai
import requests
from pathlib import Path
import re
from datetime import datetime

try:
    from .knowledge_base import SPEMKnowledgeRetriever
    from ..config import settings
    from ..data_processing.las_processor import ProcessedWellData, LogPattern
except ImportError:
    # Fallback for when running as a script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from rag_engine.knowledge_base import SPEMKnowledgeRetriever
    from config import settings
    try:
        from data_processing.las_processor import ProcessedWellData, LogPattern
    except ImportError:
        # Define minimal classes if not available
        class ProcessedWellData:
            pass
        class LogPattern:
            pass


@dataclass
class GeologicalInterpretation:
    """Structured geological interpretation result"""
    well_name: str
    depth_interval: Tuple[float, float]
    patterns_detected: List[LogPattern]
    sequence_stratigraphy: Dict[str, Any]
    depositional_environment: str
    confidence_score: float
    supporting_evidence: List[str]
    spem_references: List[Dict[str, Any]]
    interpretation_text: str
    recommendations: List[str]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class GeologicalRAGEngine:
    """Main RAG engine for geological interpretation"""
    
    def __init__(self):
        self.knowledge_retriever = SPEMKnowledgeRetriever()
        self.llm_client = self._initialize_llm()
        self.interpretation_templates = self._load_interpretation_templates()
    
    def _initialize_llm(self):
        """Initialize LLM client"""
        if settings.LLM_PROVIDER.lower() == 'openai':
            openai.api_key = settings.OPENAI_API_KEY
            return openai
        elif settings.LLM_PROVIDER.lower() == 'ollama':
            # For Ollama, we'll use requests directly
            return None  # We'll handle this in _query_llm
        else:
            # For future Anthropic integration
            raise NotImplementedError(f"LLM provider {settings.LLM_PROVIDER} not yet implemented")
    
    def _load_interpretation_templates(self) -> Dict[str, str]:
        """Load interpretation templates for different geological contexts"""
        return {
            'pattern_analysis': """
Based on the SPEM document, analyze this well log pattern:

Pattern Type: {pattern_type}
Depth Interval: {depth_start:.1f} - {depth_end:.1f} ft
Log Characteristics:
- Gamma Ray: {gr_trend}
- Resistivity: {res_trend}
- SP: {sp_trend}

SPEM Knowledge Context:
{spem_context}

Provide interpretation following SPEM principles:
1. Pattern description and geological significance
2. Depositional environment implications
3. Sequence stratigraphic context
4. Reservoir potential assessment
""",
            
            'sequence_stratigraphy': """
Using SPEM sequence stratigraphic principles, interpret this well section:

Well: {well_name}
Depth Range: {depth_start:.1f} - {depth_end:.1f} ft

Detected Patterns:
{patterns_summary}

Key Surfaces Identified:
{surfaces_summary}

SPEM Guidance:
{spem_context}

Provide comprehensive sequence stratigraphic interpretation:
1. Systems tract identification
2. Key surface recognition
3. Relative sea level implications
4. Depositional sequence architecture
""",
            
            'environmental_analysis': """
Based on SPEM environmental indicators, analyze this depositional setting:

Log Response Summary:
{log_summary}

Pattern Analysis:
{pattern_analysis}

SPEM Environmental Context:
{spem_context}

Determine:
1. Most likely depositional environment
2. Energy conditions and water depth
3. Sediment supply characteristics
4. Paleoenvironmental evolution
"""
        }
    
    def interpret_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """Interpret a list of geological patterns using SPEM knowledge"""
        try:
            # Build prompt for pattern interpretation
            prompt = self._build_patterns_prompt(patterns)
            
            # Query LLM for interpretation
            interpretation = self._query_llm(prompt)
            
            return interpretation.strip()
            
        except Exception as e:
            logger.error(f"Pattern interpretation failed: {str(e)}")
            return f"Error generating interpretation: {str(e)}"
    
    def _build_patterns_prompt(self, patterns: List[Dict[str, Any]]) -> str:
        """Build prompt for pattern interpretation"""
        
        # Get relevant SPEM knowledge
        pattern_types = [p.get('type', 'unknown') for p in patterns]
        
        spem_context = ""
        if self.knowledge_retriever:
            try:
                # Search for relevant SPEM knowledge about these patterns
                search_terms = " ".join(pattern_types) + " gamma ray log pattern geological interpretation"
                results = self.knowledge_retriever.search_knowledge(search_terms, n_results=3)
                
                if results:
                    spem_context = "SPEM Knowledge Context:\n"
                    for i, result in enumerate(results, 1):
                        spem_context += f"{i}. {result['content'][:200]}...\n"
                    spem_context += "\n"
            except Exception as e:
                logger.warning(f"Could not retrieve SPEM context: {e}")
        
        # Build pattern description
        pattern_description = "Detected Well Log Patterns:\n"
        for i, pattern in enumerate(patterns, 1):
            pattern_description += f"{i}. {pattern.get('type', 'unknown').title()} Pattern:\n"
            pattern_description += f"   Depth: {pattern.get('start_depth', 0):.1f} - {pattern.get('end_depth', 0):.1f}m\n"
            pattern_description += f"   Intensity: {pattern.get('intensity', 0):.2f}\n"
            pattern_description += f"   Logs: {', '.join(pattern.get('logs_involved', []))}\n"
            
            if 'gr_values' in pattern:
                gr = pattern['gr_values']
                pattern_description += f"   GR Values: Top={gr.get('top', 0):.1f}, Middle={gr.get('middle', 0):.1f}, Bottom={gr.get('bottom', 0):.1f} API\n"
            
            pattern_description += "\n"
        
        prompt = f"""You are a geological expert specializing in well log interpretation using SPEM sequence stratigraphic principles.

{spem_context}

{pattern_description}

Based on the SPEM knowledge above and these detected patterns, provide a geological interpretation that includes:

1. Depositional Environment: What depositional setting do these patterns suggest?
2. Sequence Stratigraphic Context: Are these patterns associated with transgressive, regressive, or highstand conditions?
3. Lithology Implications: What rock types and textures are implied by these gamma ray responses?
4. Hydrocarbon Potential: Comment on reservoir and seal potential based on the patterns.

Provide a comprehensive but concise geological interpretation (200-300 words):"""

        return prompt

    def interpret_well_data(self, processed_data: ProcessedWellData, 
                          interpretation_mode: str = 'comprehensive') -> GeologicalInterpretation:
        """
        Generate geological interpretation for processed well data
        
        Args:
            processed_data: Processed well log data
            interpretation_mode: 'patterns', 'sequence', 'environment', or 'comprehensive'
            
        Returns:
            Complete geological interpretation
        """
        logger.info(f"Generating geological interpretation for {processed_data.well_info.name}")
        
        try:
            # Extract key information from processed data
            patterns = processed_data.patterns
            depth_range = (processed_data.depth_range[0], processed_data.depth_range[1])
            
            # Retrieve relevant SPEM knowledge
            spem_context = self._gather_spem_context(processed_data)
            
            # Generate interpretation based on mode
            if interpretation_mode == 'comprehensive':
                interpretation = self._comprehensive_interpretation(processed_data, spem_context)
            elif interpretation_mode == 'patterns':
                interpretation = self._pattern_interpretation(processed_data, spem_context)
            elif interpretation_mode == 'sequence':
                interpretation = self._sequence_interpretation(processed_data, spem_context)
            elif interpretation_mode == 'environment':
                interpretation = self._environmental_interpretation(processed_data, spem_context)
            else:
                raise ValueError(f"Unknown interpretation mode: {interpretation_mode}")
            
            logger.info(f"Successfully generated {interpretation_mode} interpretation")
            return interpretation
            
        except Exception as e:
            logger.error(f"Interpretation failed: {str(e)}")
            raise
    
    def _gather_spem_context(self, processed_data: ProcessedWellData) -> Dict[str, List[Dict[str, Any]]]:
        """Gather relevant SPEM knowledge based on processed data"""
        spem_context = {
            'patterns': [],
            'sequences': [],
            'environments': [],
            'surfaces': []
        }
        
        # Search for pattern-specific knowledge
        for pattern in processed_data.patterns:
            pattern_knowledge = self.knowledge_retriever.search_by_pattern_type(pattern.pattern_type)
            spem_context['patterns'].extend(pattern_knowledge[:2])  # Top 2 results per pattern
        
        # Search for sequence stratigraphic knowledge
        sequence_query = self._build_sequence_query(processed_data)
        sequence_knowledge = self.knowledge_retriever.search_knowledge(sequence_query, n_results=3)
        spem_context['sequences'].extend(sequence_knowledge)
        
        # Search for environmental knowledge
        env_query = self._build_environmental_query(processed_data)
        env_knowledge = self.knowledge_retriever.search_knowledge(env_query, n_results=3)
        spem_context['environments'].extend(env_knowledge)
        
        # Search for surface identification
        surface_query = "maximum flooding surface transgressive surface sequence boundary identification"
        surface_knowledge = self.knowledge_retriever.search_knowledge(surface_query, n_results=2)
        spem_context['surfaces'].extend(surface_knowledge)
        
        logger.debug(f"Gathered SPEM context: {len(sum(spem_context.values(), []))} knowledge chunks")
        
        return spem_context
    
    def _build_sequence_query(self, processed_data: ProcessedWellData) -> str:
        """Build sequence stratigraphic query based on data characteristics"""
        query_parts = ["sequence stratigraphy systems tract"]
        
        # Add pattern-based terms
        pattern_terms = {
            'funnel': 'prograding lowstand forced regression',
            'bell': 'transgressive retrogradation flooding',
            'boxcar': 'aggradation highstand channel'
        }
        
        for pattern in processed_data.patterns:
            if pattern.pattern_type in pattern_terms:
                query_parts.append(pattern_terms[pattern.pattern_type])
        
        # Add log response characteristics
        if hasattr(processed_data, 'log_summary'):
            if 'high_gamma' in processed_data.log_summary:
                query_parts.append('condensed section maximum flooding')
            if 'resistive' in processed_data.log_summary:
                query_parts.append('sand reservoir lowstand')
        
        return ' '.join(query_parts)
    
    def _build_environmental_query(self, processed_data: ProcessedWellData) -> str:
        """Build environmental query based on log characteristics"""
        query_parts = ["depositional environment"]
        
        # Pattern-based environmental indicators
        env_indicators = {
            'funnel': 'shoreface deltaic progradational',
            'bell': 'transgressive marine flooding',
            'boxcar': 'channel fluvial turbidite'
        }
        
        for pattern in processed_data.patterns:
            if pattern.pattern_type in env_indicators:
                query_parts.append(env_indicators[pattern.pattern_type])
        
        return ' '.join(query_parts)
    
    def _comprehensive_interpretation(self, processed_data: ProcessedWellData, 
                                   spem_context: Dict[str, List[Dict[str, Any]]]) -> GeologicalInterpretation:
        """Generate comprehensive geological interpretation"""
        
        # Combine all SPEM knowledge
        all_spem_refs = []
        for category, refs in spem_context.items():
            all_spem_refs.extend(refs)
        
        # Create comprehensive prompt
        prompt = self._build_comprehensive_prompt(processed_data, all_spem_refs)
        
        # Get LLM interpretation
        interpretation_text = self._query_llm(prompt)
        
        # Extract structured information
        sequence_strat = self._extract_sequence_stratigraphy(interpretation_text, processed_data.patterns)
        environment = self._extract_environment(interpretation_text)
        confidence = self._calculate_confidence_score(processed_data, all_spem_refs)
        evidence = self._extract_supporting_evidence(interpretation_text)
        recommendations = self._extract_recommendations(interpretation_text)
        
        return GeologicalInterpretation(
            well_name=processed_data.well_info.name,
            depth_interval=(processed_data.depth_range[0], processed_data.depth_range[1]),
            patterns_detected=processed_data.patterns,
            sequence_stratigraphy=sequence_strat,
            depositional_environment=environment,
            confidence_score=confidence,
            supporting_evidence=evidence,
            spem_references=all_spem_refs,
            interpretation_text=interpretation_text,
            recommendations=recommendations
        )
    
    def _pattern_interpretation(self, processed_data: ProcessedWellData, 
                              spem_context: Dict[str, List[Dict[str, Any]]]) -> GeologicalInterpretation:
        """Generate pattern-focused interpretation"""
        
        pattern_refs = spem_context.get('patterns', [])
        
        # Create pattern analysis prompt
        prompt = self._build_pattern_prompt(processed_data, pattern_refs)
        
        # Get LLM interpretation
        interpretation_text = self._query_llm(prompt)
        
        # Extract information with focus on patterns
        sequence_strat = {'focus': 'pattern_analysis', 'patterns': [asdict(p) for p in processed_data.patterns]}
        environment = self._extract_environment(interpretation_text)
        confidence = self._calculate_confidence_score(processed_data, pattern_refs)
        evidence = self._extract_supporting_evidence(interpretation_text)
        recommendations = self._extract_recommendations(interpretation_text)
        
        return GeologicalInterpretation(
            well_name=processed_data.well_info.name,
            depth_interval=(processed_data.depth_range[0], processed_data.depth_range[1]),
            patterns_detected=processed_data.patterns,
            sequence_stratigraphy=sequence_strat,
            depositional_environment=environment,
            confidence_score=confidence,
            supporting_evidence=evidence,
            spem_references=pattern_refs,
            interpretation_text=interpretation_text,
            recommendations=recommendations
        )
    
    def _sequence_interpretation(self, processed_data: ProcessedWellData, 
                               spem_context: Dict[str, List[Dict[str, Any]]]) -> GeologicalInterpretation:
        """Generate sequence stratigraphic interpretation"""
        
        sequence_refs = spem_context.get('sequences', []) + spem_context.get('surfaces', [])
        
        # Build sequence stratigraphy prompt
        prompt = self._build_sequence_prompt(processed_data, sequence_refs)
        
        # Get LLM interpretation
        interpretation_text = self._query_llm(prompt)
        
        # Extract detailed sequence stratigraphic information
        sequence_strat = self._extract_detailed_sequence_stratigraphy(interpretation_text, processed_data.patterns)
        environment = self._extract_environment(interpretation_text)
        confidence = self._calculate_confidence_score(processed_data, sequence_refs)
        evidence = self._extract_supporting_evidence(interpretation_text)
        recommendations = self._extract_recommendations(interpretation_text)
        
        return GeologicalInterpretation(
            well_name=processed_data.well_info.name,
            depth_interval=(processed_data.depth_range[0], processed_data.depth_range[1]),
            patterns_detected=processed_data.patterns,
            sequence_stratigraphy=sequence_strat,
            depositional_environment=environment,
            confidence_score=confidence,
            supporting_evidence=evidence,
            spem_references=sequence_refs,
            interpretation_text=interpretation_text,
            recommendations=recommendations
        )
    
    def _environmental_interpretation(self, processed_data: ProcessedWellData, 
                                    spem_context: Dict[str, List[Dict[str, Any]]]) -> GeologicalInterpretation:
        """Generate environment-focused interpretation"""
        
        env_refs = spem_context.get('environments', [])
        
        # Build environmental analysis prompt
        prompt = self._build_environmental_prompt(processed_data, env_refs)
        
        # Get LLM interpretation
        interpretation_text = self._query_llm(prompt)
        
        # Extract environmental information
        sequence_strat = {'focus': 'environmental_analysis'}
        environment = self._extract_detailed_environment(interpretation_text)
        confidence = self._calculate_confidence_score(processed_data, env_refs)
        evidence = self._extract_supporting_evidence(interpretation_text)
        recommendations = self._extract_recommendations(interpretation_text)
        
        return GeologicalInterpretation(
            well_name=processed_data.well_info.name,
            depth_interval=(processed_data.depth_range[0], processed_data.depth_range[1]),
            patterns_detected=processed_data.patterns,
            sequence_stratigraphy=sequence_strat,
            depositional_environment=environment,
            confidence_score=confidence,
            supporting_evidence=evidence,
            spem_references=env_refs,
            interpretation_text=interpretation_text,
            recommendations=recommendations
        )
    
    def _build_comprehensive_prompt(self, processed_data: ProcessedWellData, 
                                  spem_refs: List[Dict[str, Any]]) -> str:
        """Build comprehensive interpretation prompt"""
        
        # Summarize patterns
        patterns_summary = "\n".join([
            f"- {p.pattern_type} pattern from {p.depth_start:.1f} to {p.depth_end:.1f} ft"
            for p in processed_data.patterns
        ])
        
        # Summarize SPEM context
        spem_summary = "\n".join([
            f"- {ref['content'][:200]}... (Page {ref['page']}, {ref['type']})"
            for ref in spem_refs[:5]  # Top 5 references
        ])
        
        # Build log characteristics summary
        log_summary = self._build_log_summary(processed_data)
        
        prompt = f"""
You are a geological expert using the SPEM (Society for Sedimentary Geology) sequence stratigraphic principles to interpret well logs. 
Provide a comprehensive geological interpretation based SOLELY on SPEM document knowledge.

WELL INFORMATION:
Well Name: {processed_data.well_info.name}
Depth Range: {processed_data.depth_range[0]:.1f} - {processed_data.depth_range[1]:.1f} ft

LOG CHARACTERISTICS:
{log_summary}

PATTERNS DETECTED:
{patterns_summary}

SPEM KNOWLEDGE CONTEXT:
{spem_summary}

REQUIRED INTERPRETATION (based strictly on SPEM principles):

1. PATTERN ANALYSIS
   - Describe each log pattern and its geological significance according to SPEM
   - Explain the depositional processes that created these patterns

2. SEQUENCE STRATIGRAPHIC INTERPRETATION
   - Identify systems tracts (Lowstand, Transgressive, Highstand)
   - Recognize key surfaces (SB, TS, MFS)
   - Describe relative sea level implications

3. DEPOSITIONAL ENVIRONMENT
   - Determine the most likely depositional setting
   - Assess energy conditions and water depth
   - Evaluate sediment supply characteristics

4. RESERVOIR ASSESSMENT
   - Evaluate reservoir potential of sand bodies
   - Assess seal effectiveness
   - Consider structural and stratigraphic trapping

5. CONFIDENCE ASSESSMENT
   - Rate interpretation confidence (1-10)
   - List supporting evidence from log patterns
   - Identify areas of uncertainty

6. RECOMMENDATIONS
   - Suggest additional data needs
   - Recommend drilling or completion strategies
   - Propose further analysis requirements

Base ALL interpretations strictly on SPEM document principles and the provided SPEM knowledge context.
"""
        
        return prompt
    
    def _build_pattern_prompt(self, processed_data: ProcessedWellData, 
                             pattern_refs: List[Dict[str, Any]]) -> str:
        """Build pattern-focused prompt"""
        
        patterns_detail = []
        for pattern in processed_data.patterns:
            patterns_detail.append(
                f"Pattern: {pattern.pattern_type}\n"
                f"Depth: {pattern.depth_start:.1f} - {pattern.depth_end:.1f} ft\n"
                f"Confidence: {pattern.confidence:.2f}\n"
                f"Characteristics: {pattern.characteristics}\n"
            )
        
        spem_pattern_context = "\n".join([
            f"- {ref['content']}"
            for ref in pattern_refs
        ])
        
        prompt = f"""
Using SPEM sequence stratigraphic principles, analyze these well log patterns in detail:

WELL: {processed_data.well_info.name}

DETECTED PATTERNS:
{chr(10).join(patterns_detail)}

SPEM PATTERN KNOWLEDGE:
{spem_pattern_context}

Provide detailed pattern analysis following SPEM methodology:

1. Pattern Description
   - Geometric characteristics of each pattern
   - Log response trends (GR, resistivity, SP, neutron, density)
   - Thickness and lateral continuity implications

2. Genetic Interpretation
   - Depositional processes that created each pattern
   - Energy regime and accommodation space changes
   - Sediment supply characteristics

3. Sequence Stratigraphic Context
   - Systems tract assignment for each pattern
   - Relationship to base level changes
   - Position within depositional sequences

4. Environmental Implications
   - Water depth and energy conditions
   - Depositional setting for each pattern
   - Paleoenvironmental evolution

Use ONLY the SPEM document knowledge provided above for this interpretation.
"""
        
        return prompt
    
    def _build_sequence_prompt(self, processed_data: ProcessedWellData, 
                             sequence_refs: List[Dict[str, Any]]) -> str:
        """Build sequence stratigraphic prompt"""
        
        spem_sequence_context = "\n".join([
            f"- {ref['content']}"
            for ref in sequence_refs
        ])
        
        patterns_summary = ", ".join([p.pattern_type for p in processed_data.patterns])
        
        prompt = f"""
Apply SPEM sequence stratigraphic principles to interpret this well section:

WELL: {processed_data.well_info.name}
DEPTH: {processed_data.depth_range[0]:.1f} - {processed_data.depth_range[1]:.1f} ft
PATTERNS: {patterns_summary}

SPEM SEQUENCE STRATIGRAPHIC KNOWLEDGE:
{spem_sequence_context}

Provide comprehensive sequence stratigraphic interpretation:

1. SYSTEMS TRACT IDENTIFICATION
   - Lowstand Systems Tract (LST) intervals
   - Transgressive Systems Tract (TST) intervals  
   - Highstand Systems Tract (HST) intervals
   - Evidence for each systems tract assignment

2. KEY SURFACE RECOGNITION
   - Sequence Boundaries (SB) identification and characteristics
   - Transgressive Surfaces (TS) recognition
   - Maximum Flooding Surfaces (MFS) identification
   - Supporting evidence for surface picks

3. DEPOSITIONAL SEQUENCE ARCHITECTURE
   - Complete depositional sequences
   - Incomplete sequences and missing intervals
   - Stacking patterns and hierarchy

4. RELATIVE SEA LEVEL INTERPRETATION
   - Sea level trends during deposition
   - Rate of change implications
   - Accommodation space evolution

Base interpretation strictly on SPEM principles provided in the knowledge context.
"""
        
        return prompt
    
    def _build_environmental_prompt(self, processed_data: ProcessedWellData, 
                                   env_refs: List[Dict[str, Any]]) -> str:
        """Build environmental analysis prompt"""
        
        spem_env_context = "\n".join([
            f"- {ref['content']}"
            for ref in env_refs
        ])
        
        log_summary = self._build_log_summary(processed_data)
        
        prompt = f"""
Using SPEM environmental interpretation principles, analyze this depositional setting:

WELL: {processed_data.well_info.name}

LOG RESPONSE SUMMARY:
{log_summary}

PATTERNS: {', '.join([p.pattern_type for p in processed_data.patterns])}

SPEM ENVIRONMENTAL KNOWLEDGE:
{spem_env_context}

Provide detailed environmental analysis:

1. DEPOSITIONAL ENVIRONMENT DETERMINATION
   - Most likely primary depositional setting
   - Sub-environment variations within the section
   - Evidence supporting environmental interpretation

2. ENERGY CONDITIONS AND WATER DEPTH
   - Energy regime assessment (high, moderate, low energy)
   - Water depth estimates and trends
   - Wave, tidal, or current influence

3. SEDIMENT SUPPLY CHARACTERISTICS  
   - Sediment supply rates and variations
   - Grain size trends and implications
   - Source area characteristics

4. PALEOENVIRONMENTAL EVOLUTION
   - Environmental changes through the section
   - Controls on environmental shifts
   - Regional vs local influences

Base all environmental interpretations strictly on SPEM document principles.
"""
        
        return prompt
    
    def _build_log_summary(self, processed_data: ProcessedWellData) -> str:
        """Build summary of log characteristics"""
        summary_parts = []
        
        # Add curve summaries if available
        for curve in processed_data.curves:
            if curve.quality_score > 0.7:  # Good quality curves only
                stats = {
                    'min': np.min(curve.values),
                    'max': np.max(curve.values), 
                    'mean': np.mean(curve.values)
                }
                
                summary_parts.append(
                    f"{curve.mnemonic}: {stats['min']:.1f} to {stats['max']:.1f} "
                    f"(avg: {stats['mean']:.1f}) {curve.unit}"
                )
        
        return "\n".join(summary_parts) if summary_parts else "Log summary not available"
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM with the given prompt"""
        try:
            if settings.LLM_PROVIDER.lower() == 'openai':
                response = self.llm_client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a geological expert specializing in well log interpretation using SPEM sequence stratigraphic principles. Provide detailed, technically accurate interpretations based strictly on the SPEM document knowledge provided."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.MAX_TOKENS
                )
                return response.choices[0].message.content
            
            elif settings.LLM_PROVIDER.lower() == 'ollama':
                # Query Ollama directly
                response = requests.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": settings.OLLAMA_MODEL,
                        "prompt": f"""You are a geological expert specializing in well log interpretation using SPEM sequence stratigraphic principles. Provide detailed, technically accurate interpretations based strictly on the SPEM document knowledge provided.

{prompt}""",
                        "stream": False,
                        "options": {
                            "temperature": settings.LLM_TEMPERATURE,
                            "num_predict": settings.MAX_TOKENS
                        }
                    },
                    timeout=120  # 2 minute timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    raise Exception(f"Ollama request failed: {response.status_code} - {response.text}")
            
            else:
                raise NotImplementedError(f"LLM provider {settings.LLM_PROVIDER} not implemented")
                
        except Exception as e:
            logger.error(f"LLM query failed: {str(e)}")
            raise
    
    def _extract_sequence_stratigraphy(self, interpretation_text: str, 
                                     patterns: List[LogPattern]) -> Dict[str, Any]:
        """Extract sequence stratigraphic information from interpretation"""
        sequence_strat = {
            'systems_tracts': [],
            'key_surfaces': [],
            'depositional_sequences': [],
            'interpretation_confidence': 'moderate'
        }
        
        # Look for systems tract mentions
        if 'lowstand' in interpretation_text.lower():
            sequence_strat['systems_tracts'].append('LST')
        if 'transgressive' in interpretation_text.lower():
            sequence_strat['systems_tracts'].append('TST')
        if 'highstand' in interpretation_text.lower():
            sequence_strat['systems_tracts'].append('HST')
        
        # Look for surface mentions
        if 'maximum flooding' in interpretation_text.lower():
            sequence_strat['key_surfaces'].append('MFS')
        if 'transgressive surface' in interpretation_text.lower():
            sequence_strat['key_surfaces'].append('TS')
        if 'sequence boundary' in interpretation_text.lower():
            sequence_strat['key_surfaces'].append('SB')
        
        return sequence_strat
    
    def _extract_detailed_sequence_stratigraphy(self, interpretation_text: str, 
                                              patterns: List[LogPattern]) -> Dict[str, Any]:
        """Extract detailed sequence stratigraphic information"""
        sequence_strat = self._extract_sequence_stratigraphy(interpretation_text, patterns)
        
        # Add pattern-based systems tract assignments
        pattern_systems_tracts = []
        for pattern in patterns:
            if pattern.pattern_type == 'funnel':
                pattern_systems_tracts.append({
                    'depth_range': (pattern.depth_start, pattern.depth_end),
                    'systems_tract': 'LST/Forced Regression',
                    'pattern': pattern.pattern_type,
                    'confidence': pattern.confidence
                })
            elif pattern.pattern_type == 'bell':
                pattern_systems_tracts.append({
                    'depth_range': (pattern.depth_start, pattern.depth_end),
                    'systems_tract': 'TST',
                    'pattern': pattern.pattern_type,
                    'confidence': pattern.confidence
                })
            elif pattern.pattern_type == 'boxcar':
                pattern_systems_tracts.append({
                    'depth_range': (pattern.depth_start, pattern.depth_end),
                    'systems_tract': 'HST/LST Channel',
                    'pattern': pattern.pattern_type,
                    'confidence': pattern.confidence
                })
        
        sequence_strat['pattern_based_systems_tracts'] = pattern_systems_tracts
        
        return sequence_strat
    
    def _extract_environment(self, interpretation_text: str) -> str:
        """Extract depositional environment from interpretation"""
        environments = [
            'shoreface', 'delta', 'fluvial', 'channel', 'marine', 'tidal',
            'barrier', 'beach', 'lagoon', 'shelf', 'slope', 'turbidite'
        ]
        
        text_lower = interpretation_text.lower()
        
        for env in environments:
            if env in text_lower:
                return env.title()
        
        return "Undetermined"
    
    def _extract_detailed_environment(self, interpretation_text: str) -> str:
        """Extract detailed environmental information"""
        base_env = self._extract_environment(interpretation_text)
        
        # Add more context from interpretation
        text_lower = interpretation_text.lower()
        
        modifiers = []
        if 'shallow' in text_lower:
            modifiers.append('shallow')
        if 'deep' in text_lower:
            modifiers.append('deep')
        if 'high energy' in text_lower:
            modifiers.append('high energy')
        if 'low energy' in text_lower:
            modifiers.append('low energy')
        if 'progradational' in text_lower:
            modifiers.append('progradational')
        if 'transgressive' in text_lower:
            modifiers.append('transgressive')
        
        if modifiers:
            return f"{' '.join(modifiers)} {base_env.lower()}"
        
        return base_env
    
    def _calculate_confidence_score(self, processed_data: ProcessedWellData, 
                                   spem_refs: List[Dict[str, Any]]) -> float:
        """Calculate interpretation confidence score"""
        confidence_factors = []
        
        # Pattern confidence
        if processed_data.patterns:
            pattern_confidences = [p.confidence for p in processed_data.patterns]
            confidence_factors.append(np.mean(pattern_confidences))
        
        # SPEM knowledge relevance
        if spem_refs:
            similarities = [ref['similarity'] for ref in spem_refs if 'similarity' in ref]
            if similarities:
                confidence_factors.append(np.mean(similarities))
        
        # Data quality
        if processed_data.curves:
            quality_scores = [c.quality_score for c in processed_data.curves]
            confidence_factors.append(np.mean(quality_scores))
        
        # Overall confidence
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.5  # Default moderate confidence
    
    def _extract_supporting_evidence(self, interpretation_text: str) -> List[str]:
        """Extract supporting evidence from interpretation"""
        evidence = []
        
        # Look for evidence patterns in the text
        evidence_patterns = [
            r'evidence.*?(?=\.|$)',
            r'supported by.*?(?=\.|$)',
            r'indicated by.*?(?=\.|$)',
            r'shown by.*?(?=\.|$)'
        ]
        
        for pattern in evidence_patterns:
            matches = re.findall(pattern, interpretation_text, re.IGNORECASE)
            evidence.extend(matches)
        
        # Clean and format evidence
        cleaned_evidence = []
        for item in evidence:
            cleaned = item.strip()
            if len(cleaned) > 20:  # Filter out very short fragments
                cleaned_evidence.append(cleaned)
        
        return cleaned_evidence[:5]  # Return top 5 evidence items
    
    def _extract_recommendations(self, interpretation_text: str) -> List[str]:
        """Extract recommendations from interpretation"""
        recommendations = []
        
        # Look for recommendation patterns
        rec_patterns = [
            r'recommend.*?(?=\.|$)',
            r'suggest.*?(?=\.|$)',
            r'should.*?(?=\.|$)',
            r'consider.*?(?=\.|$)'
        ]
        
        for pattern in rec_patterns:
            matches = re.findall(pattern, interpretation_text, re.IGNORECASE)
            recommendations.extend(matches)
        
        # Clean and format recommendations
        cleaned_recs = []
        for rec in recommendations:
            cleaned = rec.strip()
            if len(cleaned) > 15:
                cleaned_recs.append(cleaned)
        
        return cleaned_recs[:3]  # Return top 3 recommendations
    
    def save_interpretation(self, interpretation: GeologicalInterpretation, 
                          output_dir: Path = None) -> Path:
        """Save interpretation to file"""
        if output_dir is None:
            output_dir = settings.OUTPUT_DIR
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        filename = f"{interpretation.well_name}_interpretation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = output_dir / filename
        
        # Convert to dict for JSON serialization
        interpretation_dict = asdict(interpretation)
        
        # Handle non-serializable objects
        if 'patterns_detected' in interpretation_dict:
            interpretation_dict['patterns_detected'] = [
                asdict(pattern) for pattern in interpretation.patterns_detected
            ]
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(interpretation_dict, f, indent=2, default=str)
        
        logger.info(f"Saved interpretation to: {output_path}")
        return output_path