"""
Territory Atlas Verification Script
Runs atlas-based territory lookup on 5 test cases and compares with MedGemma output.
"""
import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.territory.registration import register_to_mni, apply_transform_to_mask
from src.territory.lookup import get_vascular_territory
from src.config import DATASET_ROOT
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_PATH = Path(DATASET_ROOT)

# Test cases
CASES = [
    "sub-strokecase0001",
    "sub-strokecase0002",
    "sub-strokecase0003",
    "sub-strokecase0004",
    "sub-strokecase0005",
]

# MedGemma results from consistency test (for comparison)
MEDGEMMA_TERRITORIES = {
    "sub-strokecase0001": "MCA",
    "sub-strokecase0002": "MCA",
    "sub-strokecase0003": "MCA",
    "sub-strokecase0004": "MCA",
    "sub-strokecase0005": "PCA",
}


def get_paths(case_id: str):
    """Get DWI and mask paths for a case."""
    dwi_path = DATASET_PATH / case_id / "ses-0001" / "dwi" / f"{case_id}_ses-0001_dwi.nii.gz"
    mask_path = DATASET_PATH / "derivatives" / case_id / "ses-0001" / f"{case_id}_ses-0001_msk.nii.gz"
    return dwi_path, mask_path


def run_territory_verification():
    """Run territory lookup on all test cases."""
    print("=" * 70)
    print("VASCULAR TERRITORY ATLAS VERIFICATION")
    print("=" * 70)
    print()
    
    results = []
    
    for case_id in CASES:
        print(f"Processing {case_id}...")
        
        dwi_path, mask_path = get_paths(case_id)
        
        if not dwi_path.exists():
            print(f"  ERROR: DWI not found at {dwi_path}")
            continue
        if not mask_path.exists():
            print(f"  ERROR: Mask not found at {mask_path}")
            continue
        
        try:
            # Step 1: Register to MNI
            logger.info(f"  Registering {case_id} to MNI...")
            reg_result = register_to_mni(str(dwi_path), type_of_transform='Affine')
            
            # Step 2: Warp mask
            logger.info(f"  Warping lesion mask...")
            warped_mask = apply_transform_to_mask(str(mask_path), reg_result['forward_transforms'])
            
            # Step 3: Lookup territory
            logger.info(f"  Looking up territory...")
            territory_result = get_vascular_territory(warped_mask, include_laterality=True)
            
            # Extract major territory (without laterality)
            atlas_territory = territory_result['primary_territory']
            atlas_major = atlas_territory.split('-')[0] if '-' in atlas_territory else atlas_territory
            
            medgemma_territory = MEDGEMMA_TERRITORIES.get(case_id, "Unknown")
            
            match = "✅" if atlas_major == medgemma_territory else "⚠️"
            
            results.append({
                'case_id': case_id,
                'atlas_territory': atlas_territory,
                'atlas_major': atlas_major,
                'medgemma_territory': medgemma_territory,
                'overlap_pct': territory_result['overlap_percentage'],
                'match': atlas_major == medgemma_territory
            })
            
            print(f"  Atlas: {atlas_territory} ({territory_result['overlap_percentage']:.1f}%)")
            print(f"  MedGemma: {medgemma_territory}")
            print(f"  Match: {match}")
            print()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Case':<25} {'Atlas':<12} {'MedGemma':<12} {'Match'}")
    print("-" * 60)
    
    for r in results:
        match_str = "✅" if r['match'] else "⚠️"
        print(f"{r['case_id']:<25} {r['atlas_territory']:<12} {r['medgemma_territory']:<12} {match_str}")
    
    matches = sum(1 for r in results if r['match'])
    print()
    print(f"Agreement: {matches}/{len(results)} cases")
    print()


if __name__ == "__main__":
    run_territory_verification()
