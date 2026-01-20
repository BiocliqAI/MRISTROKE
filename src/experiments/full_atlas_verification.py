"""
Full Atlas Verification Script
Runs both Arterial Territory and Harvard-Oxford Anatomical lookup on all 5 test cases.
"""
import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.territory.registration import register_to_mni, apply_transform_to_mask
from src.territory.lookup import get_vascular_territory
from src.territory.anatomical import get_anatomical_location, get_location_simple
from src.config import DATASET_ROOT
from pathlib import Path

logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)

DATASET_PATH = Path(DATASET_ROOT)

CASES = [
    "sub-strokecase0001",
    "sub-strokecase0002",
    "sub-strokecase0003",
    "sub-strokecase0004",
    "sub-strokecase0005",
]


def get_paths(case_id: str):
    dwi_path = DATASET_PATH / case_id / "ses-0001" / "dwi" / f"{case_id}_ses-0001_dwi.nii.gz"
    mask_path = DATASET_PATH / "derivatives" / case_id / "ses-0001" / f"{case_id}_ses-0001_msk.nii.gz"
    return dwi_path, mask_path


def run_full_atlas_verification():
    print("=" * 80)
    print("FULL ATLAS VERIFICATION (Vascular Territory + Anatomical Location)")
    print("=" * 80)
    print()
    
    results = []
    
    for case_id in CASES:
        print(f"Processing {case_id}...", end=" ", flush=True)
        
        dwi_path, mask_path = get_paths(case_id)
        
        if not dwi_path.exists() or not mask_path.exists():
            print("SKIP (files not found)")
            continue
        
        try:
            # Step 1: Register to MNI
            reg_result = register_to_mni(str(dwi_path), type_of_transform='Affine')
            
            # Step 2: Warp mask
            warped_mask = apply_transform_to_mask(str(mask_path), reg_result['forward_transforms'])
            
            # Step 3: Vascular Territory Lookup
            territory_result = get_vascular_territory(warped_mask, include_laterality=True)
            
            # Step 4: Anatomical Location Lookup
            anat_result = get_anatomical_location(warped_mask)
            
            # Extract key info
            vascular_territory = territory_result['primary_territory']
            territory_pct = territory_result['overlap_percentage']
            
            # Get top anatomical locations (>5%)
            top_locations = [
                loc['name'] for loc in anat_result['locations'] 
                if loc['percentage'] > 5
            ]
            primary_location = anat_result['primary_location']
            
            results.append({
                'case_id': case_id,
                'territory': vascular_territory,
                'territory_pct': territory_pct,
                'primary_location': primary_location,
                'all_locations': top_locations,
            })
            
            print(f"OK ✓")
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Summary Table
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Case':<22} {'Territory':<12} {'Conf':<8} {'Primary Location':<25} {'Other Locations'}")
    print("-" * 100)
    
    for r in results:
        other_locs = ", ".join(r['all_locations'][:3]) if r['all_locations'] else "-"
        print(f"{r['case_id']:<22} {r['territory']:<12} {r['territory_pct']:.1f}%    {r['primary_location']:<25} {other_locs}")
    
    print()
    print("=" * 80)
    print("LEGEND:")
    print("  Territory: Vascular territory from Arterial Atlas (ACA/MCA/PCA/VB + Left/Right)")
    print("  Conf: Percentage of lesion voxels in primary territory")
    print("  Location: Anatomical region from Harvard-Oxford Atlas")
    print("=" * 80)


if __name__ == "__main__":
    run_full_atlas_verification()
