import sys
import numpy as np
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_gt(case_id):
    loader = DataLoader()
    case = loader.load_case(case_id)
    
    if case.mask_volume is None:
        print(f"No mask found for {case_id}")
        return

    # Find slices with lesions (where mask > 0)
    # volume shape is (H, W, D) -> index 2 is slices
    mask_indices = np.where(case.mask_volume > 0)
    affected_slices = np.unique(mask_indices[2])
    
    print(f"\n=== Ground Truth Verification for {case_id} ===")
    print(f"Total Slices in Volume: {case.mask_volume.shape[2]}")
    print(f"Slices containing Lesion: {affected_slices}")
    print(f"Number of affected slices: {len(affected_slices)}")
    
    # Calculate volume center
    if len(affected_slices) > 0:
        center = int(np.mean(affected_slices))
        print(f"Lesion Center (approx): Slice {center}")

if __name__ == "__main__":
    verify_gt("sub-strokecase0001")
