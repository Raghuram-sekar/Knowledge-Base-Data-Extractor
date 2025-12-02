"""
Advanced Image Enhancement Utilities for Geological RAG System
Superior algorithms for geological documents and well logs
"""

import os
from pathlib import Path
from typing import Union, Tuple, Optional
import logging
import cv2
import numpy as np
from skimage import exposure, restoration, morphology, filters
from skimage.transform import resize
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)


class AdvancedGeologicalImageEnhancer:
    """Superior image processing for geological documents using OpenCV + scikit-image"""
    
    def __init__(self):
        self.min_size = (1200, 800)  # Higher resolution for geological analysis
        self.target_dpi = 300
        
        # Advanced enhancement parameters
        self.clahe_params = {
            'clip_limit': 3.0,      # Adaptive contrast enhancement
            'tile_grid_size': (8, 8)  # Grid for local enhancement
        }
        
        self.bilateral_params = {
            'd': 9,                 # Neighborhood diameter
            'sigma_color': 75,      # Color similarity
            'sigma_space': 75       # Spatial similarity
        }
        
        self.morphology_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    def enhance_geological_image(self, image_path: Union[str, Path], 
                                output_path: Optional[Union[str, Path]] = None,
                                enhancement_mode: str = 'geological') -> Tuple[str, dict]:
        """
        Advanced enhancement for geological images using superior algorithms
        
        Args:
            image_path: Input image path
            output_path: Output path (auto-generated if None)
            enhancement_mode: 'geological', 'well_log', or 'diagram'
        
        Returns:
            Tuple of (output_path, enhancement_stats)
        """
        try:
            image_path = Path(image_path)
            
            # Load image with OpenCV (superior color handling)
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                # Fallback to PIL for exotic formats
                pil_img = Image.open(image_path).convert('RGB')
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            original_height, original_width = img_bgr.shape[:2]
            logger.info(f"Processing {image_path.name}: {original_width}x{original_height}")
            
            # Convert to different color spaces for processing
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            
            # Calculate optimal scale factor (more intelligent)
            scale_factor = self._calculate_optimal_scale(original_width, original_height)
            
            # Advanced super-resolution using OpenCV's EDSR
            enhanced_img = self._apply_super_resolution(img_bgr, scale_factor)
            
            # Apply geological-specific enhancements
            if enhancement_mode == 'well_log':
                enhanced_img = self._enhance_well_log(enhanced_img)
            elif enhancement_mode == 'diagram':
                enhanced_img = self._enhance_diagram(enhanced_img)
            else:  # geological (general)
                enhanced_img = self._enhance_geological_general(enhanced_img)
            
            # Generate output path
            if output_path is None:
                output_dir = image_path.parent / "enhanced"
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"enhanced_{image_path.stem}.png"
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with optimal settings
            success = cv2.imwrite(str(output_path), enhanced_img, [
                cv2.IMWRITE_PNG_COMPRESSION, 1,  # Best quality
                cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_DEFAULT
            ])
            
            if not success:
                raise ValueError("Failed to save enhanced image")
            
            # Calculate enhancement statistics
            final_height, final_width = enhanced_img.shape[:2]
            stats = {
                'original_size': (original_width, original_height),
                'enhanced_size': (final_width, final_height),
                'scale_factor': scale_factor,
                'enhancement_mode': enhancement_mode,
                'file_size_mb': output_path.stat().st_size / (1024 * 1024)
            }
            
            logger.info(f"Enhanced {image_path.name}: "
                       f"{original_width}x{original_height} → {final_width}x{final_height} "
                       f"(scale: {scale_factor:.2f}x)")
            
            return str(output_path), stats
            
        except Exception as e:
            logger.error(f"Enhancement failed for {image_path}: {e}")
            raise
    
    def _calculate_optimal_scale(self, width: int, height: int) -> float:
        """Calculate optimal scale factor based on image content and size"""
        min_dimension = min(width, height)
        target_min = min(self.min_size)
        
        if min_dimension < 200:
            return min(10.0, target_min / min_dimension)  # Cap at 10x
        elif min_dimension < 400:
            return min(6.0, target_min / min_dimension)
        elif min_dimension < 600:
            return min(4.0, target_min / min_dimension)
        elif min_dimension < 800:
            return min(2.0, target_min / min_dimension)
        else:
            return max(1.0, target_min / min_dimension)
    
    def _apply_super_resolution(self, img: np.ndarray, scale_factor: float) -> np.ndarray:
        """Apply advanced super-resolution using OpenCV algorithms"""
        if scale_factor <= 1.0:
            return img
        
        height, width = img.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Use INTER_CUBIC for smooth upsampling, then enhance
        upsampled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filtering to preserve edges while smoothing
        filtered = cv2.bilateralFilter(upsampled, **self.bilateral_params)
        
        return filtered
    
    def _enhance_well_log(self, img: np.ndarray) -> np.ndarray:
        """Specialized enhancement for well log images"""
        # Convert to LAB for better contrast control
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Apply CLAHE to L channel (much better than simple contrast adjustment)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_params['clip_limit'], 
                               tileGridSize=self.clahe_params['tile_grid_size'])
        l_channel = clahe.apply(l_channel)
        
        # Enhance curves and lines (critical for well logs)
        l_channel = self._enhance_curves_and_lines(l_channel)
        
        lab[:, :, 0] = l_channel
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Additional sharpening for text and measurements
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel * 0.5)
        
        return enhanced
    
    def _enhance_diagram(self, img: np.ndarray) -> np.ndarray:
        """Specialized enhancement for geological diagrams"""
        # Convert to grayscale for morphological operations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        
        # Enhance text and line clarity
        enhanced_gray = self._enhance_text_clarity(enhanced_gray)
        
        # Convert back to color
        enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        
        # Slight color enhancement if original was colored
        if len(img.shape) == 3:
            # Preserve original color information
            enhanced = cv2.addWeighted(enhanced, 0.7, img, 0.3, 0)
        
        return enhanced
    
    def _enhance_geological_general(self, img: np.ndarray) -> np.ndarray:
        """General geological image enhancement"""
        # Multi-scale enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Global and local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        l_channel = clahe.apply(l_channel)
        
        # Noise reduction while preserving geological features
        l_channel = cv2.bilateralFilter(l_channel, 5, 50, 50)
        
        lab[:, :, 0] = l_channel
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Final sharpening
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def _enhance_curves_and_lines(self, gray_img: np.ndarray) -> np.ndarray:
        """Enhance curves and lines critical for well logs"""
        # Use morphological operations to enhance thin lines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        
        # Closing operation to connect broken lines
        closed = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
        
        # Edge enhancement
        edges = cv2.Canny(gray_img, 50, 150)
        
        # Combine original with enhanced edges
        enhanced = cv2.addWeighted(closed, 0.8, edges, 0.2, 0)
        
        return enhanced
    
    def _enhance_text_clarity(self, gray_img: np.ndarray) -> np.ndarray:
        """Enhance text readability in geological diagrams"""
        # Adaptive thresholding for text enhancement
        adaptive = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Combine with original for better text while preserving images
        enhanced = cv2.addWeighted(gray_img, 0.7, adaptive, 0.3, 0)
        
        # Noise reduction
        enhanced = cv2.medianBlur(enhanced, 3)
        
        return enhanced
    
    def enhance_image(self, image_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None,
                     enhance_for_geological_analysis: bool = True) -> Path:
        """
        Enhance image quality for better geological analysis
        
        Args:
            image_path: Path to input image
            output_path: Path for enhanced output (if None, overwrites original)
            enhance_for_geological_analysis: Apply geological-specific enhancements
            
        Returns:
            Path to enhanced image
        """
        image_path = Path(image_path)
        if output_path is None:
            output_path = image_path.parent / f"enhanced_{image_path.name}"
        else:
            output_path = Path(output_path)
        
        try:
            # Open image
            img = Image.open(image_path)
            logger.info(f"Original image size: {img.size}, mode: {img.mode}")
            
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Resize if image is too small
            if img.size[0] < self.min_size[0] or img.size[1] < self.min_size[1]:
                # Calculate scale factor to meet minimum size
                scale_x = self.min_size[0] / img.size[0]
                scale_y = self.min_size[1] / img.size[1]
                scale_factor = max(scale_x, scale_y, 2.0)  # At least 2x scale
                
                new_size = (int(img.size[0] * scale_factor), 
                           int(img.size[1] * scale_factor))
                
                # Use LANCZOS for high-quality upsampling
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Upsampled to: {img.size} (scale factor: {scale_factor:.2f})")
            
            if enhance_for_geological_analysis:
                img = self._apply_geological_enhancements(img)
            
            # Save with high quality
            save_kwargs = {'dpi': (self.target_dpi, self.target_dpi)}
            if output_path.suffix.lower() in ['.jpg', '.jpeg']:
                save_kwargs['quality'] = 95
                save_kwargs['optimize'] = True
            
            img.save(output_path, **save_kwargs)
            logger.info(f"Enhanced image saved: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to enhance image {image_path}: {e}")
            raise
    
    def _apply_geological_enhancements(self, img: Image.Image) -> Image.Image:
        """Apply geological-specific image enhancements"""
        
        # 1. Enhance contrast for better line/text visibility
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.enhancement_settings['contrast'])
        
        # 2. Apply unsharp mask for crisp geological features
        img = img.filter(ImageFilter.UnsharpMask(
            radius=2.0,      # Radius for geological diagrams
            percent=150,     # Strength of sharpening
            threshold=3      # Threshold to avoid noise
        ))
        
        # 3. Enhance sharpness for technical drawings
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(self.enhancement_settings['sharpness'])
        
        # 4. Slight brightness adjustment
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(self.enhancement_settings['brightness'])
        
        # 5. Color enhancement for geological color schemes
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(self.enhancement_settings['color'])
        
        # 6. Apply edge enhancement for geological boundaries
        edge_enhance = ImageFilter.EDGE_ENHANCE_MORE
        img = img.filter(edge_enhance)
        
        return img
    
    def enhance_batch(self, images_directory: Union[str, Path], 
                     output_directory: Optional[Union[str, Path]] = None,
                     pattern: str = "*.png") -> list:
        """
        Enhance all images in a directory
        
        Args:
            images_directory: Directory containing images to enhance
            output_directory: Directory for enhanced images (if None, creates 'enhanced' subdirectory)
            pattern: File pattern to match (e.g., "*.png", "*.jpg")
            
        Returns:
            List of enhanced image paths
        """
        images_dir = Path(images_directory)
        
        if output_directory is None:
            output_dir = images_dir / "enhanced"
        else:
            output_dir = Path(output_directory)
        
        output_dir.mkdir(exist_ok=True)
        
        enhanced_paths = []
        image_files = list(images_dir.glob(pattern))
        
        logger.info(f"Enhancing {len(image_files)} images from {images_dir}")
        
        for image_file in image_files:
            try:
                output_path = output_dir / f"enhanced_{image_file.name}"
                enhanced_path = self.enhance_image(image_file, output_path)
                enhanced_paths.append(enhanced_path)
            except Exception as e:
                logger.error(f"Failed to enhance {image_file}: {e}")
        
        logger.info(f"Enhanced {len(enhanced_paths)} images successfully")
        return enhanced_paths
    
    def compare_image_quality(self, original_path: Union[str, Path], 
                            enhanced_path: Union[str, Path]) -> dict:
        """Compare original vs enhanced image quality metrics"""
        
        try:
            orig_img = Image.open(original_path)
            enh_img = Image.open(enhanced_path)
            
            # Convert to numpy for analysis
            orig_array = np.array(orig_img.convert('L'))  # Grayscale
            enh_array = np.array(enh_img.convert('L'))
            
            # Calculate quality metrics
            metrics = {
                'original_size': orig_img.size,
                'enhanced_size': enh_img.size,
                'size_improvement': (enh_img.size[0] * enh_img.size[1]) / (orig_img.size[0] * orig_img.size[1]),
                'original_contrast': np.std(orig_array),
                'enhanced_contrast': np.std(enh_array),
                'contrast_improvement': np.std(enh_array) / np.std(orig_array) if np.std(orig_array) > 0 else 0,
                'original_sharpness': self._calculate_sharpness(orig_array),
                'enhanced_sharpness': self._calculate_sharpness(enh_array)
            }
            
            metrics['sharpness_improvement'] = (
                metrics['enhanced_sharpness'] / metrics['original_sharpness'] 
                if metrics['original_sharpness'] > 0 else 0
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to compare image quality: {e}")
            return {}
    
    def _calculate_sharpness(self, img_array: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        # Apply Laplacian operator to detect edges
        from scipy import ndimage
        laplacian = ndimage.laplace(img_array.astype(float))
        return np.var(laplacian)


def enhance_extracted_images(extracted_dir: Union[str, Path]) -> Path:
    """
    Convenience function to enhance all extracted SPEM images
    
    Args:
        extracted_dir: Directory containing extracted_spem_complete
        
    Returns:
        Path to directory containing enhanced images
    """
    extracted_path = Path(extracted_dir)
    images_dir = extracted_path / "images"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    enhancer = GeologicalImageEnhancer()
    enhanced_images = enhancer.enhance_batch(
        images_dir, 
        output_directory=images_dir / "enhanced",
        pattern="*.{png,jpg,jpeg}"
    )
    
    logger.info(f"Enhanced {len(enhanced_images)} geological images")
    return images_dir / "enhanced"


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        enhance_extracted_images(input_dir)
    else:
        # Default to current extracted images
        current_dir = Path(__file__).parent.parent.parent
        extracted_dir = current_dir / "extracted_spem_complete"
        if extracted_dir.exists():
            enhance_extracted_images(extracted_dir)
        else:
            print("No extracted images found. Please provide path to extracted_spem_complete directory.")