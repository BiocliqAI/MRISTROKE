"""
End-to-End Atlas-Augmented Pipeline
Combines MedGemma (VLM) inference with Atlas-based localization.
"""
import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from pathlib import Path
import numpy as np

from config import DATASET_ROOT
from data.loader import DataLoader
from data.preprocessor import ImagePreprocessor
from model.medgemma import MedGemmaPredictor
from report.parser import parse_medgemma_output
from utils.adc_gate import check_adc_confirmation
from territory.registration import register_to_mni, apply_transform_to_mask
from territory.lookup import get_vascular_territory
from territory.anatomical import get_anatomical_location
from report.atlas_generator import generate_atlas_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_PATH = Path(DATASET_ROOT)


def get_representative_slab(case):
    """Get the slice with max lesion area (using GT mask for now)."""
    if case.mask_volume is None:
        return None, None
    
    lesion_area_per_slice = np.sum(case.mask_volume, axis=(0, 1))
    max_slice_idx = int(np.argmax(lesion_area_per_slice))
    
    if lesion_area_per_slice[max_slice_idx] == 0:
        return None, None
    
    num_slices = case.dwi_volume.shape[2]
    start_idx = max(0, max_slice_idx - 1)
    end_idx = min(num_slices - 1, max_slice_idx + 1)
    
    if start_idx == max_slice_idx:
        slice_indices = [0, 1, 2]
    elif end_idx == max_slice_idx:
        slice_indices = [end_idx - 2, end_idx - 1, end_idx]
    else:
        slice_indices = [max_slice_idx - 1, max_slice_idx, max_slice_idx + 1]
    
    return max_slice_idx, slice_indices


def run_end_to_end_pipeline(case_id: str, output_dir: Path = None):
    """
    Run the full atlas-augmented pipeline on a single case.
    
    Pipeline:
    1. Load case data
    2. Get representative slices (GT-guided)
    3. Run MedGemma inference → Detection + Stage
    4. Register to MNI space with ANTs
    5. Look up Vascular Territory (Arterial Atlas)
    6. Look up Anatomical Location (Harvard-Oxford Atlas)
    7. Generate combined report
    """
    logger.info(f"=" * 60)
    logger.info(f"Processing {case_id}")
    logger.info(f"=" * 60)
    
    # Step 1: Load case
    logger.info("Step 1: Loading case data...")
    loader = DataLoader()
    case = loader.load_case(case_id)
    
    # Step 2: Get representative slices
    logger.info("Step 2: Getting representative slices (GT-guided)...")
    max_slice_idx, slice_indices = get_representative_slab(case)
    if max_slice_idx is None:
        logger.error("No lesion found in GT mask. Cannot proceed.")
        return None
    logger.info(f"  Representative slice: {max_slice_idx}, Slab: {slice_indices}")
    
    # Prepare images for VLM
    dwi_images = [ImagePreprocessor.process_slice(case.dwi_volume[:, :, idx]) for idx in slice_indices]
    adc_image = ImagePreprocessor.process_slice(case.adc_volume[:, :, max_slice_idx])
    flair_image = ImagePreprocessor.process_slice(case.flair_volume[:, :, max_slice_idx])
    
    adc_confirmed = check_adc_confirmation(
        case.dwi_volume[:, :, max_slice_idx],
        case.adc_volume[:, :, max_slice_idx]
    )
    
    # Step 3: MedGemma inference
    logger.info("Step 3: Running MedGemma inference...")
    predictor = MedGemmaPredictor()
    vlm_response = predictor.predict(dwi_images, adc_image, flair_image, adc_confirmed)
    parsed = parse_medgemma_output(vlm_response)
    
    vlm_findings = {
        'infarction_present': parsed.get('infarction_present', False),
        'stage': parsed.get('Stage', 'Acute')
    }
    logger.info(f"  VLM Detection: {'POSITIVE' if vlm_findings['infarction_present'] else 'NEGATIVE'}")
    logger.info(f"  VLM Stage: {vlm_findings['stage']}")
    
    # Step 4: Register to MNI
    logger.info("Step 4: Registering to MNI space (ANTs)...")
    dwi_path = DATASET_PATH / case_id / "ses-0001" / "dwi" / f"{case_id}_ses-0001_dwi.nii.gz"
    mask_path = DATASET_PATH / "derivatives" / case_id / "ses-0001" / f"{case_id}_ses-0001_msk.nii.gz"
    
    reg_result = register_to_mni(str(dwi_path), type_of_transform='Affine')
    warped_mask = apply_transform_to_mask(str(mask_path), reg_result['forward_transforms'])
    
    # Step 5: Vascular Territory Lookup
    logger.info("Step 5: Looking up vascular territory...")
    territory_result = get_vascular_territory(warped_mask, include_laterality=True)
    logger.info(f"  Territory: {territory_result['primary_territory']} ({territory_result['overlap_percentage']:.1f}%)")
    
    # Step 6: Anatomical Location Lookup
    logger.info("Step 6: Looking up anatomical location...")
    anat_result = get_anatomical_location(warped_mask)
    top_locations = [loc['name'] for loc in anat_result['locations'] if loc['percentage'] > 5]
    logger.info(f"  Location: {anat_result['primary_location']} ({', '.join(top_locations[:3])})")
    
    # Step 7: Generate Report
    logger.info("Step 7: Generating atlas-augmented report...")
    report = generate_atlas_report(
        case_id=case_id,
        vlm_findings=vlm_findings,
        atlas_territory=territory_result,
        atlas_location=anat_result,
        adc_confirmed=adc_confirmed
    )
    
    # Save report
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{case_id}_atlas_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")
    
    print()
    print(report)
    
    return {
        'case_id': case_id,
        'vlm_detection': vlm_findings['infarction_present'],
        'vlm_stage': vlm_findings['stage'],
        'atlas_territory': territory_result['primary_territory'],
        'atlas_locations': top_locations,
        'report': report
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Atlas-Augmented Pipeline")
    parser.add_argument("--case", type=str, default="sub-strokecase0003", help="Case ID")
    parser.add_argument("--output", type=str, default="reports/atlas_reports", help="Output directory")
    args = parser.parse_args()
    
    run_end_to_end_pipeline(args.case, Path(args.output))
