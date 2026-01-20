import numpy as np
from PIL import Image
from typing import List, Tuple
import logging
from src.config import NORMALIZE_PERCENTILE, IMAGE_SIZE

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    @staticmethod
    def select_slices(dwi_vol: np.ndarray, top_n: int = 3) -> List[int]:
        """
        Selects axial slices with the most 'lesion-like' activity using a global threshold heuristic.
        
        Args:
            dwi_vol: 3D numpy array (H, W, D)
            top_n: Number of slices to select
            
        Returns:
            List of indices of the selected axial slices.
        """
        if dwi_vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {dwi_vol.ndim}D")
            
        num_slices = dwi_vol.shape[2]
        slice_scores = []
        
        # Skip top/bottom 20% to focus on brain parenchyma
        start_idx = int(num_slices * 0.20)
        end_idx = int(num_slices * 0.80)
        
        # Robust clipping to suppress hyperintense artifacts
        foreground = dwi_vol[dwi_vol > 0]
        if foreground.size > 0:
            clip_thresh = np.percentile(foreground, 99.5)
            dwi_clipped = np.clip(dwi_vol, 0, clip_thresh)
        else:
            dwi_clipped = dwi_vol
        
        for i in range(start_idx, end_idx):
            slice_data = dwi_clipped[:, :, i]
            
            # Heuristic: Variance of the slice
            # Stroke lesions often introduce significant local variance compared to uniform brain/background
            # This proved more robust than max intensity for small lesions (e.g. case 0002)
            score = np.var(slice_data)
            
            # Secondary tie-breaker: Max intensity
            max_intensity = np.max(slice_data)
            
            slice_scores.append((i, score, max_intensity))
            
        # Sort by score (Variance) descending
        slice_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        logger.debug(f"Top 5 slices by score: {slice_scores[:5]}")
        
        # Select Top N
        selected_indices = [x[0] for x in slice_scores[:top_n]]
        # Return in anatomical order
        selected_indices.sort()
        
        logger.info(f"Selected slices indices: {selected_indices}")
        return selected_indices

    @staticmethod
    def process_slice(slice_data: np.ndarray) -> Image.Image:
        """
        Normalizes a slice and converts it to a PIL Image compatible with MedGemma.
        """
        # Robust min-max normalization
        min_val = np.min(slice_data)
        max_val = np.percentile(slice_data, NORMALIZE_PERCENTILE)
        
        if max_val - min_val < 1e-6:
            norm_data = np.zeros_like(slice_data)
        else:
            norm_data = (slice_data - min_val) / (max_val - min_val)
            norm_data = np.clip(norm_data, 0, 1)
            
        # Convert to 8-bit uint
        img_uint8 = (norm_data * 255).astype(np.uint8)
        
        # Create PIL Image
        img = Image.fromarray(img_uint8)
        
        # Resize/Pad to expected input size
        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to RGB (Model expects 3 channels even for grayscale)
        img = img.convert("RGB")
        
        return img
