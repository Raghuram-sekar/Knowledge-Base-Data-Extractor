"""
LAS File Processing Module
Handles parsing, validation, and preprocessing of well log data
"""

import pandas as pd
import numpy as np
import lasio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger

try:
    from ..config import settings
except ImportError:
    # Fallback for when running as a script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import settings


@dataclass
class WellInfo:
    """Well metadata information"""
    well_name: str
    uwi: Optional[str] = None
    field: Optional[str] = None
    country: Optional[str] = None
    operator: Optional[str] = None
    api_number: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    kb_elevation: Optional[float] = None
    ground_elevation: Optional[float] = None


@dataclass
class LogCurve:
    """Individual log curve data"""
    name: str
    mnemonic: str
    unit: str
    description: str
    data: np.ndarray
    depth: np.ndarray
    quality_score: float = 0.0


@dataclass
class ProcessedWellData:
    """Complete processed well log dataset"""
    well_info: WellInfo
    curves: Dict[str, LogCurve]
    depth_range: Tuple[float, float]
    depth_interval: float
    quality_report: Dict[str, Any]
    raw_las: Any  # Original lasio object


class LASProcessor:
    """Main LAS file processing class"""
    
    def __init__(self):
        self.supported_mnemonics = {
            'GR': ['GR', 'GAMMA', 'GAPI', 'GRC', 'GRDE'],
            'SP': ['SP', 'SPONTPOT', 'SPT'],
            'RES': ['RES', 'RILD', 'RILM', 'RLA1', 'RLA2', 'RLA3', 'RT'],
            'NPHI': ['NPHI', 'NEU', 'NEUTRON', 'NPOR'],
            'RHOB': ['RHOB', 'DEN', 'DENSITY', 'DENB'],
            'DT': ['DT', 'SONIC', 'AC', 'TRANSIT'],
            'CALIPER': ['CAL', 'CALIPER', 'CALI']
        }
    
    def process_las_file(self, file_path: str) -> ProcessedWellData:
        """
        Complete LAS file processing pipeline
        
        Args:
            file_path: Path to LAS file
            
        Returns:
            ProcessedWellData object containing all processed data
        """
        logger.info(f"Processing LAS file: {file_path}")
        
        # Parse LAS file
        las = self._parse_las_file(file_path)
        
        # Extract well information
        well_info = self._extract_well_info(las)
        
        # Process log curves
        curves = self._process_log_curves(las)
        
        # Quality assessment
        quality_report = self._assess_data_quality(curves)
        
        # Calculate depth info
        depth_range, depth_interval = self._calculate_depth_info(las)
        
        processed_data = ProcessedWellData(
            well_info=well_info,
            curves=curves,
            depth_range=depth_range,
            depth_interval=depth_interval,
            quality_report=quality_report,
            raw_las=las
        )
        
        logger.info(f"Successfully processed well: {well_info.well_name}")
        return processed_data
    
    def _parse_las_file(self, file_path: str) -> lasio.LASFile:
        """Parse LAS file using lasio library"""
        try:
            las = lasio.read(file_path)
            logger.debug(f"LAS file parsed successfully: {len(las.curves)} curves found")
            return las
        except Exception as e:
            logger.error(f"Failed to parse LAS file {file_path}: {str(e)}")
            raise ValueError(f"Invalid LAS file: {str(e)}")
    
    def _extract_well_info(self, las: lasio.LASFile) -> WellInfo:
        """Extract well metadata from LAS header"""
        header = las.header
        
        # Helper function to safely get header value
        def get_header_value(section: str, key: str, default: Any = None) -> Any:
            try:
                if hasattr(las, section.lower()):
                    section_obj = getattr(las, section.lower())
                    if hasattr(section_obj, key.upper()):
                        return getattr(section_obj, key.upper()).value
                return default
            except:
                return default
        
        well_info = WellInfo(
            well_name=get_header_value('well', 'WELL', 'Unknown'),
            uwi=get_header_value('well', 'UWI'),
            field=get_header_value('well', 'FLD'),
            country=get_header_value('well', 'CTRY'),
            operator=get_header_value('well', 'COMP'),
            api_number=get_header_value('well', 'API'),
            latitude=self._safe_float(get_header_value('well', 'LAT')),
            longitude=self._safe_float(get_header_value('well', 'LON')),
            kb_elevation=self._safe_float(get_header_value('well', 'EKB')),
            ground_elevation=self._safe_float(get_header_value('well', 'EGL'))
        )
        
        return well_info
    
    def _process_log_curves(self, las: lasio.LASFile) -> Dict[str, LogCurve]:
        """Process and normalize log curves"""
        curves = {}
        depth = las.depth_m  # Get depth array
        
        for curve in las.curves:
            # Skip depth curve (usually DEPT, DEPTH, MD, etc.)
            if curve.mnemonic.upper() in ['DEPT', 'DEPTH', 'MD', 'MDEPTH']:
                continue
            
            # Identify curve type
            curve_type = self._identify_curve_type(curve.mnemonic)
            if not curve_type:
                continue
            
            # Get curve data
            curve_data = las[curve.mnemonic]
            
            # Handle missing values
            curve_data = self._handle_missing_data(curve_data)
            
            # Calculate quality score
            quality_score = self._calculate_curve_quality(curve_data)
            
            log_curve = LogCurve(
                name=curve_type,
                mnemonic=curve.mnemonic,
                unit=curve.unit,
                description=curve.descr,
                data=curve_data,
                depth=depth,
                quality_score=quality_score
            )
            
            curves[curve_type] = log_curve
            logger.debug(f"Processed curve: {curve_type} ({curve.mnemonic})")
        
        return curves
    
    def _identify_curve_type(self, mnemonic: str) -> Optional[str]:
        """Identify standardized curve type from mnemonic"""
        mnemonic_upper = mnemonic.upper()
        
        for curve_type, mnemonics in self.supported_mnemonics.items():
            if any(mn in mnemonic_upper for mn in mnemonics):
                return curve_type
        
        return None
    
    def _handle_missing_data(self, data: np.ndarray) -> np.ndarray:
        """Handle missing/null values in log data"""
        # Convert common null values to NaN
        data = np.where(data == -999.25, np.nan, data)
        data = np.where(data == -9999, np.nan, data)
        data = np.where(np.isinf(data), np.nan, data)
        
        return data
    
    def _calculate_curve_quality(self, data: np.ndarray) -> float:
        """Calculate quality score for log curve"""
        if len(data) == 0:
            return 0.0
        
        # Calculate percentage of valid data
        valid_data_percent = (1 - np.isnan(data).sum() / len(data)) * 100
        
        # Calculate data range (normalized)
        if valid_data_percent > 0:
            valid_data = data[~np.isnan(data)]
            data_range = np.max(valid_data) - np.min(valid_data)
            range_score = min(data_range / 100, 1.0)  # Normalize to 0-1
        else:
            range_score = 0.0
        
        # Combined quality score
        quality_score = (valid_data_percent / 100) * 0.8 + range_score * 0.2
        
        return quality_score
    
    def _assess_data_quality(self, curves: Dict[str, LogCurve]) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        quality_report = {
            'overall_quality': 'Good',
            'curve_quality': {},
            'missing_data': {},
            'recommendations': [],
            'warnings': []
        }
        
        total_quality = 0
        curve_count = 0
        
        for curve_type, curve in curves.items():
            quality = curve.quality_score
            missing_percent = np.isnan(curve.data).sum() / len(curve.data) * 100
            
            quality_report['curve_quality'][curve_type] = quality
            quality_report['missing_data'][curve_type] = missing_percent
            
            total_quality += quality
            curve_count += 1
            
            # Add warnings for poor quality curves
            if quality < 0.5:
                quality_report['warnings'].append(
                    f"{curve_type} curve has poor quality (score: {quality:.2f})"
                )
            
            if missing_percent > settings.MAX_MISSING_DATA_PERCENT:
                quality_report['warnings'].append(
                    f"{curve_type} curve has {missing_percent:.1f}% missing data"
                )
        
        # Overall quality assessment
        if curve_count > 0:
            avg_quality = total_quality / curve_count
            if avg_quality > 0.8:
                quality_report['overall_quality'] = 'Excellent'
            elif avg_quality > 0.6:
                quality_report['overall_quality'] = 'Good'
            elif avg_quality > 0.4:
                quality_report['overall_quality'] = 'Fair'
            else:
                quality_report['overall_quality'] = 'Poor'
        
        # Add recommendations
        if 'GR' not in curves:
            quality_report['recommendations'].append(
                "Gamma Ray log missing - essential for geological interpretation"
            )
        
        if len(curves) < 3:
            quality_report['recommendations'].append(
                "Limited log suite available - interpretation may be constrained"
            )
        
        return quality_report
    
    def _calculate_depth_info(self, las: lasio.LASFile) -> Tuple[Tuple[float, float], float]:
        """Calculate depth range and interval"""
        depth = las.depth_m
        depth_range = (float(depth.min()), float(depth.max()))
        
        # Calculate depth interval
        depth_diff = np.diff(depth)
        depth_interval = float(np.median(depth_diff[depth_diff > 0]))
        
        return depth_range, depth_interval
    
    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Safely convert value to float"""
        try:
            if value is None or value == '':
                return None
            return float(value)
        except (ValueError, TypeError):
            return None


class PatternDetector:
    """Detect geological patterns in log curves"""
    
    def __init__(self):
        self.window_size = settings.PATTERN_DETECTION_WINDOW
    
    def detect_log_patterns(self, processed_data: ProcessedWellData) -> Dict[str, List[Dict]]:
        """
        Detect geological patterns in log curves
        
        Returns:
            Dictionary mapping curve types to detected patterns
        """
        patterns = {}
        
        for curve_type, curve in processed_data.curves.items():
            if curve_type == 'GR':  # Focus on Gamma Ray for pattern detection
                curve_patterns = self._detect_gr_patterns(curve)
                patterns[curve_type] = curve_patterns
        
        return patterns
    
    def _detect_gr_patterns(self, curve: LogCurve) -> List[Dict]:
        """Detect patterns in Gamma Ray log"""
        patterns = []
        data = curve.data
        depth = curve.depth
        
        # Remove NaN values for analysis
        valid_mask = ~np.isnan(data)
        clean_data = data[valid_mask]
        clean_depth = depth[valid_mask]
        
        if len(clean_data) < self.window_size:
            return patterns
        
        # Sliding window pattern detection
        for i in range(0, len(clean_data) - self.window_size, self.window_size // 2):
            window_data = clean_data[i:i + self.window_size]
            window_depth = clean_depth[i:i + self.window_size]
            
            pattern = self._classify_pattern(window_data)
            if pattern:
                patterns.append({
                    'type': pattern,
                    'depth_start': float(window_depth[0]),
                    'depth_end': float(window_depth[-1]),
                    'thickness': float(window_depth[-1] - window_depth[0]),
                    'data_range': (float(window_data.min()), float(window_data.max())),
                    'confidence': self._calculate_pattern_confidence(window_data, pattern)
                })
        
        return patterns
    
    def _classify_pattern(self, data: np.ndarray) -> Optional[str]:
        """Classify log pattern type based on SPEM definitions"""
        if len(data) < 5:
            return None
        
        # Calculate trend using linear regression
        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]
        
        # Calculate pattern characteristics
        data_range = data.max() - data.min()
        relative_slope = slope / (data_range + 1e-10)  # Avoid division by zero
        
        # Pattern classification based on SPEM document
        if abs(relative_slope) < 0.1:
            return 'boxcar'  # Cylindrical/block shape
        elif relative_slope < -0.1:
            return 'funnel'  # Cleaning-up trend (decreasing GR upward)
        elif relative_slope > 0.1:
            return 'bell'    # Dirtying-up trend (increasing GR upward)
        else:
            return 'irregular'
    
    def _calculate_pattern_confidence(self, data: np.ndarray, pattern_type: str) -> float:
        """Calculate confidence score for pattern classification"""
        x = np.arange(len(data))
        slope, intercept = np.polyfit(x, data, 1)
        
        # Calculate R-squared for linear trend
        y_pred = slope * x + intercept
        ss_res = np.sum((data - y_pred) ** 2)
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        
        # Adjust confidence based on pattern type
        if pattern_type in ['funnel', 'bell']:
            return max(0.0, min(1.0, r_squared))
        elif pattern_type == 'boxcar':
            # For boxcar, low variation is good
            cv = np.std(data) / (np.mean(data) + 1e-10)
            return max(0.0, min(1.0, 1.0 - cv))
        else:
            return 0.5  # Default confidence for irregular patterns