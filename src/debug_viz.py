import argparse
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import ImagePreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_case(case_id):
    loader = DataLoader()
    case = loader.load_case(case_id)
    
    print(f"=== Case: {case_id} ===")
    print(f"DWI Shape: {case.dwi_volume.shape}, Range: {case.dwi_volume.min()} - {case.dwi_volume.max()}")
    print(f"ADC Shape: {case.adc_volume.shape}, Range: {case.adc_volume.min()} - {case.adc_volume.max()}")
    print(f"FLAIR Shape: {case.flair_volume.shape}, Range: {case.flair_volume.min()} - {case.flair_volume.max()}")
    
    if case.mask_volume is not None:
         print(f"Mask Shape: {case.mask_volume.shape}, Unique values: {np.unique(case.mask_volume)}")
         total_mask_voxels = np.sum(case.mask_volume > 0)
         print(f"Total positive mask voxels: {total_mask_voxels}")
    else:
        print("No Mask Found.")

    # Slice Selection Debug
    print("\n--- Heuristic Debug ---")
    foreground = case.dwi_volume[case.dwi_volume > 10]
    clip_thresh = np.percentile(foreground, 99.5)
    dwi_clipped = np.clip(case.dwi_volume, 0, clip_thresh)
    
    scores = []
    for i in range(15, 60): # Search range
        slice_data = dwi_clipped[:, :, i]
        
        # Intensity Score
        flat = slice_data.flatten()
        flat.sort()
        intensity_score = np.mean(flat[-200:])
        
        # Variance Score
        var_score = np.var(slice_data)
        
        scores.append((i, intensity_score, var_score))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    print("Top 5 by Intensity (Mean Top 200):")
    for s in scores[:5]: print(f"Slice {s[0]}: Score={s[1]:.1f}")
    
    scores.sort(key=lambda x: x[2], reverse=True)
    print("\nTop 5 by Variance:")
    for s in scores[:5]: print(f"Slice {s[0]}: Var={s[2]:.1f}") 
    
    # Check score at lesion center (approx 32)
    lesion_slice = [s for s in scores if s[0] == 32][0]
    print(f"\nLesion Slice (32): Int={lesion_slice[1]:.1f}, Var={lesion_slice[2]:.1f}")

if __name__ == "__main__":
    inspect_case("sub-strokecase0002")
