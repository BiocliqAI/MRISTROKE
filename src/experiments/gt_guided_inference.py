import sys
import os
import argparse
import logging
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.config import OUTPUT_DIR
from src.data.loader import DataLoader
from src.data.preprocessor import ImagePreprocessor
from src.model.medgemma import MedGemmaPredictor
from src.report.parser import parse_medgemma_output
from src.utils.adc_gate import check_adc_confirmation
from src.report.generator import generate_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_gt_guided_inference(case_id: str):
    logger.info(f"Starting GT-Guided Inference for {case_id}")
    
    # 1. Load Data
    loader = DataLoader()
    try:
        case = loader.load_case(case_id)
    except Exception as e:
        logger.error(f"Failed to load case {case_id}: {e}")
        return

    if case.mask_volume is None:
        logger.error("No Ground Truth mask found for this case. Cannot simulate guided strategy.")
        return

    # 2. Find Representative Slice (Max Area)
    # Sum pixels per slice (axis 0 and 1 are H, W; axis 2 is Depth)
    # mask_volume shape: (H, W, D)
    lesion_area_per_slice = np.sum(case.mask_volume, axis=(0, 1))
    
    max_slice_idx = np.argmax(lesion_area_per_slice)
    max_area = lesion_area_per_slice[max_slice_idx]
    
    if max_area == 0:
        logger.warning("Ground Truth mask is empty! No lesion found in annotation.")
        return

    logger.info(f"Max lesion area found on Slice {max_slice_idx} (Area: {max_area} pixels)")
    
    # 3. Define Context Slab (Slice +/- 1)
    # Ensure bounds
    start_idx = max(0, max_slice_idx - 1)
    end_idx = min(case.dwi_volume.shape[2] - 1, max_slice_idx + 1)
    
    # If standard 3-slice slab is possible
    if start_idx == max_slice_idx: # At slice 0
         slice_indices = [0, 1, 2]
    elif end_idx == max_slice_idx: # At last slice
         slice_indices = [end_idx-2, end_idx-1, end_idx]
    else:
         slice_indices = [max_slice_idx - 1, max_slice_idx, max_slice_idx + 1]
         
    logger.info(f"Selected Representative Slab: {slice_indices}")
    
    # 4. Prepare Images
    dwi_images = [ImagePreprocessor.process_slice(case.dwi_volume[:, :, idx]) for idx in slice_indices]
    
    # Use the max slice for ADC/FLAIR context
    adc_slice_idx = max_slice_idx
    adc_image = ImagePreprocessor.process_slice(case.adc_volume[:, :, adc_slice_idx])
    flair_image = ImagePreprocessor.process_slice(case.flair_volume[:, :, adc_slice_idx])
    
    # 5. ADC Confirmation
    adc_confirmed = check_adc_confirmation(
        case.dwi_volume[:, :, adc_slice_idx],
        case.adc_volume[:, :, adc_slice_idx]
    )
    logger.info(f"ADC Confirmation: {adc_confirmed}")
    
    # 6. Run Inference
    predictor = MedGemmaPredictor()
    response = predictor.predict(dwi_images, adc_image, flair_image, adc_confirmed)
    
    # 7. Generate Report
    parsed = parse_medgemma_output(response)
    report_text = generate_report(case_id, parsed, adc_confirmed)
    
    # Save Report
    output_path = OUTPUT_DIR / f"{case_id}_gt_guided_report.txt"
    with open(output_path, "w") as f:
        f.write(f"STRATEGY: GT-Guided (Representative Slice {max_slice_idx})\n")
        f.write("="*60 + "\n")
        f.write(report_text)
        
    logger.info(f"Report saved to {output_path}")
    print("\n--- GENERATED REPORT SUMMARY ---")
    print(f"Lesion Found on Slice: {max_slice_idx}")
    print(f"ADC Confirmed: {adc_confirmed}")
    print(f"AI Diagnosis: {'POSITIVE' if parsed['infarction_present'] else 'NEGATIVE'}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GT-Guided Inference Simulation")
    parser.add_argument("--case", type=str, default="sub-strokecase0005", help="Case ID")
    args = parser.parse_args()
    
    run_gt_guided_inference(args.case)
