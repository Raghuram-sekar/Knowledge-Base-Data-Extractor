"""
ULTIMATE Image Enhancement for Geological Documents
Using state-of-the-art algorithms for maximum clarity
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, Dict
import logging
from PIL import Image
from skimage import restoration, exposure, morphology
from scipy import ndimage

logger = logging.getLogger(__name__)


class UltimateGeologicalEnhancer:
    """
    Ultimate enhancement using the best available algorithms
    Specifically tuned for geological well log diagrams
    """
    
    def __init__(self):
        self.target_resolution = (1600, 1200)  # Professional quality
        
    def enhance_geological_diagram(self, image_path: Union[str, Path], 
                                  output_path: Optional[Union[str, Path]] = None) -> Tuple[str, Dict]:
        """
        Ultimate enhancement using best-in-class algorithms
        """
        try:
            image_path = Path(image_path)
            
            # Load image with maximum quality preservation
            img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                # Fallback with PIL
                pil_img = Image.open(image_path).convert('RGB')
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            h, w = img_bgr.shape[:2]
            logger.info(f"Processing {image_path.name}: {w}x{h}")
            
            # Step 1: Intelligent super-resolution using Lanczos (better than cubic)
            enhanced_img = self._apply_lanczos_upsampling(img_bgr)
            
            # Step 2: Geological-specific enhancement pipeline
            enhanced_img = self._apply_geological_enhancement_pipeline(enhanced_img)
            
            # Step 3: Professional sharpening (unsharp mask)
            enhanced_img = self._apply_professional_sharpening(enhanced_img)
            
            # Step 4: Final quality optimization
            enhanced_img = self._apply_final_optimization(enhanced_img)
            
            # Generate output path
            if output_path is None:
                output_dir = image_path.parent / "ultimate_enhanced"
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"ultimate_{image_path.stem}.png"
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with maximum quality settings
            cv2.imwrite(str(output_path), enhanced_img, [
                cv2.IMWRITE_PNG_COMPRESSION, 0,  # No compression for maximum quality
            ])
            
            final_h, final_w = enhanced_img.shape[:2]
            stats = {
                'original_size': (w, h),
                'enhanced_size': (final_w, final_h),
                'scale_factor': max(final_w/w, final_h/h),
                'algorithm': 'Ultimate-Lanczos+CLAHE+UnsharpMask',
                'file_size_mb': output_path.stat().st_size / (1024 * 1024)
            }
            
            logger.info(f"✨ Ultimate enhanced {image_path.name}: {w}x{h} → {final_w}x{final_h}")
            return str(output_path), stats
            
        except Exception as e:
            logger.error(f"❌ Ultimate enhancement failed for {image_path}: {e}")
            raise
    
    def _apply_lanczos_upsampling(self, img: np.ndarray) -> np.ndarray:
        """Apply Lanczos upsampling - superior to cubic interpolation"""
        h, w = img.shape[:2]
        
        # Calculate target size maintaining aspect ratio
        scale_w = self.target_resolution[0] / w
        scale_h = self.target_resolution[1] / h
        scale = min(scale_w, scale_h, 8.0)  # Cap at 8x to avoid over-processing
        
        if scale <= 1.0:
            return img
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Convert to PIL for Lanczos (highest quality resampling)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Apply Lanczos resampling (best quality for upsampling)
        pil_upsampled = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Convert back to OpenCV
        img_upsampled = cv2.cvtColor(np.array(pil_upsampled), cv2.COLOR_RGB2BGR)
        
        logger.info(f"🔍 Lanczos upsampling: {w}x{h} → {new_w}x{new_h} ({scale:.1f}x)")
        return img_upsampled
    
    def _apply_geological_enhancement_pipeline(self, img: np.ndarray) -> np.ndarray:
        """Professional geological enhancement pipeline"""
        
        # Convert to LAB for professional color processing
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Apply CLAHE with optimized parameters for geological diagrams
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        l_enhanced = clahe.apply(l_channel)
        
        # Enhance geological features using morphological operations
        l_enhanced = self._enhance_geological_features(l_enhanced)
        
        # Reconstruct color image
        lab[:, :, 0] = l_enhanced
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Enhance colors for better geological visualization
        enhanced = self._enhance_geological_colors(enhanced)
        
        return enhanced
    
    def _enhance_geological_features(self, gray_img: np.ndarray) -> np.ndarray:
        """Enhance lines, text, and geological boundaries"""
        
        # Create different enhancement kernels
        # Horizontal lines (for well log data)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        # Vertical lines (for depth scales)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        # Small details
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Enhance different types of features
        enhanced_h = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel_h)
        enhanced_v = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel_v)
        enhanced_small = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel_small)
        
        # Combine enhancements
        enhanced = cv2.addWeighted(enhanced_h, 0.3, enhanced_v, 0.3, 0)
        enhanced = cv2.addWeighted(enhanced, 0.6, enhanced_small, 0.4, 0)
        enhanced = cv2.addWeighted(gray_img, 0.7, enhanced, 0.3, 0)
        
        # Edge enhancement for geological boundaries
        edges = cv2.Canny(gray_img, 30, 100)
        enhanced = cv2.addWeighted(enhanced, 0.9, edges, 0.1, 0)
        
        return enhanced
    
    def _enhance_geological_colors(self, img: np.ndarray) -> np.ndarray:
        """Enhance colors specifically for geological visualization"""
        
        # Convert to HSV for color enhancement
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Enhance saturation for better geological color distinction
        s = cv2.multiply(s, 1.2)  # Boost saturation by 20%
        s = np.clip(s, 0, 255)
        
        # Slight value enhancement for brightness
        v = cv2.multiply(v, 1.1)
        v = np.clip(v, 0, 255)
        
        # Merge and convert back
        hsv_enhanced = cv2.merge([h, s, v])
        color_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        return color_enhanced
    
    def _apply_professional_sharpening(self, img: np.ndarray) -> np.ndarray:
        """Apply professional unsharp mask sharpening"""
        
        # Create Gaussian blur for unsharp mask
        gaussian = cv2.GaussianBlur(img, (0, 0), 1.0)
        
        # Unsharp mask: original + amount * (original - blurred)
        unsharp_strength = 0.8  # Professional strength
        sharpened = cv2.addWeighted(img, 1 + unsharp_strength, gaussian, -unsharp_strength, 0)
        
        return sharpened
    
    def _apply_final_optimization(self, img: np.ndarray) -> np.ndarray:
        """Final optimization for professional quality"""
        
        # Noise reduction while preserving details
        denoised = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
        
        # Blend original with denoised (preserve detail while reducing noise)
        optimized = cv2.addWeighted(img, 0.8, denoised, 0.2, 0)
        
        return optimized
    
    def batch_ultimate_enhance(self, input_dir: Union[str, Path], 
                              output_dir: Optional[Union[str, Path]] = None) -> Dict:
        """Batch ultimate enhancement"""
        input_dir = Path(input_dir)
        if output_dir is None:
            output_dir = input_dir / "ultimate_enhanced"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Find images
        supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in input_dir.iterdir() 
                      if f.suffix.lower() in supported_formats and f.is_file()]
        
        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return {"processed": 0, "errors": 0, "files": []}
        
        logger.info(f"🚀 Ultimate enhancement starting: {len(image_files)} images")
        
        results = {"processed": 0, "errors": 0, "files": [], "total_improvement": 0.0}
        
        for image_file in image_files:
            try:
                output_path = output_dir / f"ultimate_{image_file.stem}.png"
                
                if output_path.exists():
                    logger.debug(f"⏭️ Skipping {image_file.name} (already enhanced)")
                    continue
                
                enhanced_path, stats = self.enhance_geological_diagram(image_file, output_path)
                
                results["processed"] += 1
                results["total_improvement"] += stats["scale_factor"]
                results["files"].append({
                    "original": str(image_file),
                    "enhanced": enhanced_path,
                    "stats": stats
                })
                
            except Exception as e:
                logger.error(f"❌ Failed to enhance {image_file.name}: {e}")
                results["errors"] += 1
        
        avg_improvement = (results["total_improvement"] / results["processed"] 
                          if results["processed"] > 0 else 0)
        
        logger.info(f"✨ Ultimate enhancement complete: {results['processed']} processed, "
                   f"{results['errors']} errors, average: {avg_improvement:.1f}x")
        
        return results


def test_ultimate_enhancement():
    """Test the ultimate enhancement on one image"""
    logging.basicConfig(level=logging.INFO)
    
    enhancer = UltimateGeologicalEnhancer()
    images_dir = Path('extracted_spem_complete/images')
    
    if not images_dir.exists():
        print("❌ Images directory not found")
        return
    
    # Test on first image
    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    if not image_files:
        print("❌ No images found")
        return
    
    test_image = image_files[0]
    print(f"🧪 Testing ultimate enhancement on: {test_image.name}")
    
    try:
        enhanced_path, stats = enhancer.enhance_geological_diagram(test_image)
        
        print(f"✨ ULTIMATE SUCCESS!")
        print(f"📊 {stats['original_size']} → {stats['enhanced_size']}")
        print(f"📊 Scale: {stats['scale_factor']:.1f}x")
        print(f"📊 Algorithm: {stats['algorithm']}")
        print(f"📊 Output: {enhanced_path}")
        
    except Exception as e:
        print(f"❌ Ultimate enhancement failed: {e}")


if __name__ == "__main__":
    test_ultimate_enhancement()