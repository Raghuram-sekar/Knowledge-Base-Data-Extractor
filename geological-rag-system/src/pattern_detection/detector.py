#!/usr/bin/env python3
"""
Geological Pattern Detection Module

This module provides comprehensive pattern detection for geological well logs,
identifying key log signature patterns used in sequence stratigraphy interpretation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import pearsonr
import logging

try:
    from ..data_processing.las_processor import ProcessedWellData
except ImportError:
    from data_processing.las_processor import ProcessedWellData


class PatternDetector:
    """
    Advanced geological pattern detection for well log analysis.
    
    Detects and classifies geological patterns in well logs including:
    - Funnel patterns (fining upward sequences)
    - Bell patterns (coarsening upward sequences) 
    - Boxcar patterns (uniform sequences)
    - Irregular patterns (complex sequences)
    """
    
    def __init__(self, smoothing_window: int = 5, min_pattern_length: float = 5.0):
        """
        Initialize the pattern detector.
        
        Args:
            smoothing_window: Window size for data smoothing
            min_pattern_length: Minimum pattern length in meters
        """
        self.smoothing_window = smoothing_window
        self.min_pattern_length = min_pattern_length
        self.logger = logging.getLogger(__name__)
        
        # Pattern type mapping
        self.pattern_types = {
            'funnel': 'Fining Upward (Funnel)',
            'bell': 'Coarsening Upward (Bell)', 
            'boxcar': 'Uniform (Boxcar)',
            'irregular': 'Irregular/Complex'
        }
        
    def detect_patterns(self, well_data: ProcessedWellData) -> Dict[str, Any]:
        """
        Main pattern detection method that identifies all geological patterns in well data.
        
        Args:
            well_data: Processed well log data
            
        Returns:
            Dictionary containing detected patterns and metadata
        """
        
        self.logger.info(f"Starting pattern detection for well: {well_data.well_info.well_name}")
        
        # Initialize results
        results = {
            'well_name': well_data.well_info.well_name,
            'total_depth_range': well_data.depth_range,
            'patterns': [],
            'summary': {},
            'metadata': {
                'detection_method': 'gradient_correlation_analysis',
                'smoothing_window': self.smoothing_window,
                'min_pattern_length': self.min_pattern_length
            }
        }
        
        # Use GR (Gamma Ray) as primary curve for pattern detection
        primary_curve = 'GR'
        if primary_curve not in well_data.curves:
            # Fallback to other available curves
            available_curves = list(well_data.curves.keys())
            if available_curves:
                primary_curve = available_curves[0]
                self.logger.warning(f"GR curve not found, using {primary_curve} instead")
            else:
                self.logger.error("No curves available for pattern detection")
                return results
        
        # Extract curve data
        curve_data = well_data.curves[primary_curve]
        depth = curve_data.depth
        values = curve_data.data
        
        # Remove NaN values
        valid_mask = ~np.isnan(values)
        depth_clean = depth[valid_mask]
        values_clean = values[valid_mask]
        
        if len(values_clean) < 10:
            self.logger.warning("Insufficient clean data for pattern detection")
            return results
        
        self.logger.info(f"Analyzing {len(values_clean):,} data points over {depth_clean[-1] - depth_clean[0]:.1f}m")
        
        # Smooth the data
        if len(values_clean) > self.smoothing_window:
            values_smooth = savgol_filter(values_clean, 
                                        min(self.smoothing_window, len(values_clean)-1 if len(values_clean) % 2 == 0 else len(values_clean)), 
                                        polyorder=2)
        else:
            values_smooth = values_clean
        
        # Detect patterns using multiple methods
        patterns = self._detect_gradient_patterns(depth_clean, values_smooth)
        patterns.extend(self._detect_correlation_patterns(depth_clean, values_smooth))
        patterns.extend(self._detect_statistical_patterns(depth_clean, values_smooth))
        
        # Remove overlapping patterns and keep the best ones
        patterns = self._filter_overlapping_patterns(patterns)
        
        # Add patterns to results
        results['patterns'] = patterns
        
        # Generate summary
        results['summary'] = self._generate_pattern_summary(patterns)
        
        self.logger.info(f"Detected {len(patterns)} patterns: {results['summary']}")
        
        return results
    
    def _detect_gradient_patterns(self, depth: np.ndarray, values: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect patterns based on gradient analysis.
        
        Args:
            depth: Depth array
            values: Log values array
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        # Calculate gradient
        gradient = np.gradient(values, depth)
        
        # Find segments with consistent gradient direction
        window_size = max(10, len(depth) // 20)  # Adaptive window size
        
        for i in range(0, len(depth) - window_size, window_size // 2):
            end_idx = min(i + window_size, len(depth))
            
            if end_idx - i < window_size // 2:
                continue
                
            segment_depth = depth[i:end_idx]
            segment_gradient = gradient[i:end_idx]
            
            # Skip if segment too short
            if segment_depth[-1] - segment_depth[0] < self.min_pattern_length:
                continue
            
            # Analyze gradient characteristics
            mean_gradient = np.mean(segment_gradient)
            gradient_std = np.std(segment_gradient)
            gradient_consistency = 1 - (gradient_std / (abs(mean_gradient) + 1e-6))
            
            pattern_type = None
            confidence = max(0, min(100, gradient_consistency * 100))
            
            if abs(mean_gradient) > 0.1 and gradient_consistency > 0.3:
                if mean_gradient > 0:
                    pattern_type = 'funnel'  # Increasing values upward = fining upward
                else:
                    pattern_type = 'bell'    # Decreasing values upward = coarsening upward
            elif gradient_std < 0.5:
                pattern_type = 'boxcar'  # Low variability = uniform
                confidence = max(confidence, 60)
            else:
                pattern_type = 'irregular'
                confidence = max(confidence, 40)
            
            patterns.append({
                'type': pattern_type,
                'start_depth': float(segment_depth[0]),
                'end_depth': float(segment_depth[-1]),
                'thickness': float(segment_depth[-1] - segment_depth[0]),
                'confidence': float(confidence),
                'method': 'gradient_analysis',
                'gradient_mean': float(mean_gradient),
                'gradient_std': float(gradient_std)
            })
        
        return patterns
    
    def _detect_correlation_patterns(self, depth: np.ndarray, values: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect patterns using correlation analysis with ideal patterns.
        
        Args:
            depth: Depth array  
            values: Log values array
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        window_size = max(15, len(depth) // 15)
        
        for i in range(0, len(depth) - window_size, window_size // 2):
            end_idx = min(i + window_size, len(depth))
            
            if end_idx - i < window_size // 2:
                continue
                
            segment_depth = depth[i:end_idx]
            segment_values = values[i:end_idx]
            
            # Skip if segment too short
            if segment_depth[-1] - segment_depth[0] < self.min_pattern_length:
                continue
            
            # Normalize segment
            seg_normalized = (segment_values - segment_values.min()) / (segment_values.max() - segment_values.min() + 1e-6)
            depth_normalized = (segment_depth - segment_depth[0]) / (segment_depth[-1] - segment_depth[0])
            
            # Create ideal patterns
            funnel_ideal = depth_normalized  # Linear increase with depth
            bell_ideal = 1 - depth_normalized  # Linear decrease with depth
            boxcar_ideal = np.ones_like(depth_normalized) * 0.5  # Constant
            
            # Calculate correlations
            try:
                funnel_corr, _ = pearsonr(seg_normalized, funnel_ideal)
                bell_corr, _ = pearsonr(seg_normalized, bell_ideal)
                boxcar_corr = 1 - np.std(seg_normalized)  # Uniformity measure
            except:
                continue
            
            # Determine best pattern match
            correlations = {
                'funnel': abs(funnel_corr),
                'bell': abs(bell_corr), 
                'boxcar': boxcar_corr
            }
            
            best_pattern = max(correlations.keys(), key=lambda k: correlations[k])
            best_correlation = correlations[best_pattern]
            
            # Confidence based on correlation strength
            confidence = max(0, min(100, best_correlation * 100))
            
            if confidence > 30:  # Only keep patterns with reasonable confidence
                patterns.append({
                    'type': best_pattern,
                    'start_depth': float(segment_depth[0]),
                    'end_depth': float(segment_depth[-1]), 
                    'thickness': float(segment_depth[-1] - segment_depth[0]),
                    'confidence': float(confidence),
                    'method': 'correlation_analysis',
                    'correlation_value': float(best_correlation)
                })
        
        return patterns
    
    def _detect_statistical_patterns(self, depth: np.ndarray, values: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect patterns using statistical analysis.
        
        Args:
            depth: Depth array
            values: Log values array
            
        Returns:
            List of detected patterns  
        """
        patterns = []
        
        window_size = max(12, len(depth) // 12)
        
        for i in range(0, len(depth) - window_size, window_size // 2):
            end_idx = min(i + window_size, len(depth))
            
            if end_idx - i < window_size // 2:
                continue
                
            segment_depth = depth[i:end_idx]
            segment_values = values[i:end_idx]
            
            # Skip if segment too short
            if segment_depth[-1] - segment_depth[0] < self.min_pattern_length:
                continue
            
            # Statistical measures
            mean_val = np.mean(segment_values)
            std_val = np.std(segment_values)
            skewness = self._calculate_skewness(segment_values)
            
            # Trend analysis
            trend_slope = np.polyfit(range(len(segment_values)), segment_values, 1)[0]
            
            # Pattern classification based on statistics
            pattern_type = 'irregular'
            confidence = 50
            
            coefficient_of_variation = std_val / (abs(mean_val) + 1e-6)
            
            if coefficient_of_variation < 0.15:  # Low variability
                pattern_type = 'boxcar'
                confidence = max(60, min(90, (1 - coefficient_of_variation) * 100))
            elif abs(trend_slope) > std_val * 0.1:  # Strong trend
                if trend_slope > 0:
                    pattern_type = 'funnel'
                else:
                    pattern_type = 'bell'
                confidence = max(40, min(85, abs(trend_slope) / std_val * 50))
            elif abs(skewness) > 1:  # High skewness indicates irregular pattern
                pattern_type = 'irregular'
                confidence = max(30, min(70, abs(skewness) * 30))
            
            patterns.append({
                'type': pattern_type,
                'start_depth': float(segment_depth[0]),
                'end_depth': float(segment_depth[-1]),
                'thickness': float(segment_depth[-1] - segment_depth[0]),
                'confidence': float(confidence),
                'method': 'statistical_analysis',
                'coefficient_of_variation': float(coefficient_of_variation),
                'trend_slope': float(trend_slope),
                'skewness': float(skewness)
            })
        
        return patterns
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 3)
        except:
            return 0
    
    def _filter_overlapping_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove overlapping patterns, keeping the ones with highest confidence.
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            Filtered list of non-overlapping patterns
        """
        if not patterns:
            return []
        
        # Sort by confidence (highest first)
        patterns = sorted(patterns, key=lambda p: p['confidence'], reverse=True)
        
        filtered = []
        
        for pattern in patterns:
            # Check if this pattern overlaps with any already accepted pattern
            overlap = False
            
            for accepted in filtered:
                # Check for depth overlap
                if not (pattern['end_depth'] <= accepted['start_depth'] or 
                       pattern['start_depth'] >= accepted['end_depth']):
                    overlap = True
                    break
            
            if not overlap:
                filtered.append(pattern)
        
        # Sort by depth for final result
        filtered = sorted(filtered, key=lambda p: p['start_depth'])
        
        return filtered
    
    def _generate_pattern_summary(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics for detected patterns.
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_patterns': len(patterns),
            'pattern_types': {},
            'average_thickness': 0,
            'total_thickness_analyzed': 0,
            'average_confidence': 0
        }
        
        if not patterns:
            return summary
        
        # Count pattern types
        for pattern in patterns:
            ptype = pattern['type']
            summary['pattern_types'][ptype] = summary['pattern_types'].get(ptype, 0) + 1
        
        # Calculate averages
        thicknesses = [p['thickness'] for p in patterns]
        confidences = [p['confidence'] for p in patterns]
        
        summary['average_thickness'] = float(np.mean(thicknesses))
        summary['total_thickness_analyzed'] = float(np.sum(thicknesses))
        summary['average_confidence'] = float(np.mean(confidences))
        
        return summary