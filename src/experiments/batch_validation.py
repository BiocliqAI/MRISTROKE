import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
from tqdm import tqdm
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.config import PROJECT_ROOT, OUTPUT_DIR, SLICES_PER_CASE, DATASET_ROOT
from src.data.loader import DataLoader
from src.data.preprocessor import ImagePreprocessor
from src.model.medgemma import MedGemmaPredictor
from src.report.parser import parse_medgemma_output
from src.report.generator import generate_report
from src.utils.adc_gate import check_adc_confirmation

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output Directory for Batch
BATCH_OUTPUT_DIR = OUTPUT_DIR / "batch_validation_sliding"
BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def verify_gt(case, mask_data):
    """
    Returns a list of slice indices where the mask has a lesion.
    """
    if mask_data is None:
        return []
    
    # Check each slice
    positive_slices = []
    for i in range(mask_data.shape[2]):
        if np.any(mask_data[:, :, i] > 0.5): # Binary mask
            positive_slices.append(i)
    return positive_slices

def run_batch():
    logger.info("Initializing Sliding Window Batch Validation (5 Cases)...")
    
    # 1. Load Model Once
    predictor = MedGemmaPredictor()
    loader = DataLoader() # Cases are loaded one by one
    
    results = []
    
    # Select cases 2 to 6 (5 cases) for rapid validation
    case_ids = [f"sub-strokecase{i:04d}" for i in range(2, 7)]
    
    for case_id in tqdm(case_ids, desc="Processing Cases"):
        try:
            logger.info(f"--- Processing {case_id} [Sliding Window] ---")
            
            # Load Data
            try:
                case = loader.load_case(case_id)
            except Exception as e:
                logger.error(f"Failed to load {case_id}: {e}")
                results.append({"CaseID": case_id, "Status": "Load Failed", "Error": str(e)})
                continue
                
            # Ground Truth Validation
            gt_slices = verify_gt(case, case.mask_volume)
            has_gt_lesion = len(gt_slices) > 0
            
            # --- Sliding Window Logic ---
            num_slices = case.dwi_volume.shape[2]
            start_idx = int(num_slices * 0.2)
            end_idx = int(num_slices * 0.8)
            stride = 3
            window_size = 3
            
            best_slab = None
            any_positive_slab = False
            
            logger.info(f"Scanning slices [{start_idx}, {end_idx}] with stride {stride}...")
            
            # We want to find IF any slab is positive. 
            # If multiple are positive, we might want to pick the "most confident" or just the first robust one.
            # Ideally, we collect all positives. For this summary, let's just flag if ANY was positive.
            
            findings = []
            
            for i in range(start_idx, end_idx, stride):
                slice_indices = [i + j for j in range(window_size) if (i + j) < end_idx]
                if not slice_indices: continue
                
                # Prepare Images
                mid_slice = slice_indices[len(slice_indices)//2]
                
                dwi_images = [ImagePreprocessor.process_slice(case.dwi_volume[:, :, idx]) for idx in slice_indices]
                adc_image = ImagePreprocessor.process_slice(case.adc_volume[:, :, mid_slice])
                flair_image = ImagePreprocessor.process_slice(case.flair_volume[:, :, mid_slice])
                
                adc_confirmed = check_adc_confirmation(
                    case.dwi_volume[:, :, mid_slice],
                    case.adc_volume[:, :, mid_slice]
                )
                
                # Predict
                generated_text = predictor.predict(
                    dwi_images=dwi_images, 
                    adc_image=adc_image, 
                    flair_image=flair_image, 
                    adc_confirmed=adc_confirmed
                )
                parsed = parse_medgemma_output(generated_text)
                
                is_positive = parsed['infarction_present']
                
                if is_positive:
                    any_positive_slab = True
                    # Keep track of this finding
                    findings.append({
                        "slices": slice_indices,
                        "adc_confirmed": adc_confirmed,
                        "text": generated_text,
                        "parsed": parsed
                    })
                    # Optional: Break early if we just want "Screening Positive"? 
                    # User asked to "create AI reports", usually implies finding the BEST positive/negative.
                    # Let's verify all slabs and pick the one with ADC confirmation + Positive if available.
            
            # Decide Final Report content
            # Strategy: 
            # 1. Prefer Positive + ADC Confirmed
            # 2. Else Positive 
            # 3. Else Negative (last slab or middle slab)
            
            final_selection = None
            if findings:
                # Filter for ADC confirmed
                confirmed = [f for f in findings if f['adc_confirmed']]
                if confirmed:
                    final_selection = confirmed[0] # Pick first confirmed
                else:
                    final_selection = findings[0] # Pick first positive (even if not confirmed)
            else:
                # No positive findings -> Pick a middle slab for the negative report
                mid_idx_global = num_slices // 2
                slice_indices = [mid_idx_global, mid_idx_global+1, mid_idx_global+2]
                # Re-run for report generation if needed, or we just say "Negative"
                final_selection = {
                    "slices": slice_indices,
                    "adc_confirmed": False, # Likelihood
                    "text": "Negative Screening",
                    "parsed": {"infarction_present": False, "findings": [], "impression": "No acute infarct detected."}
                }
                
                # To get a real negative report text, we might need to re-run predict on this slab if we didn't save it.
                # Simplification: Just run it now.
                mid_slice = slice_indices[1]
                dwi_images = [ImagePreprocessor.process_slice(case.dwi_volume[:, :, idx]) for idx in slice_indices]
                adc_image = ImagePreprocessor.process_slice(case.adc_volume[:, :, mid_slice])
                flair_image = ImagePreprocessor.process_slice(case.flair_volume[:, :, mid_slice])
                
                adc_confirmed = check_adc_confirmation(case.dwi_volume[:, :, mid_slice], case.adc_volume[:, :, mid_slice])
                
                generated_text = predictor.predict(dwi_images, adc_image, flair_image, adc_confirmed)
                parsed = parse_medgemma_output(generated_text)
                final_selection = {
                    "slices": slice_indices,
                    "adc_confirmed": adc_confirmed,
                    "text": generated_text,
                    "parsed": parsed
                }
            
            # Save Report
            report_text = generate_report(case_id, final_selection['parsed'], final_selection['adc_confirmed'])
            report_path = BATCH_OUTPUT_DIR / f"{case_id}_sliding_report.txt"
            with open(report_path, "w") as f:
                f.write(report_text)
                
            # Check Hit logic
            # Did the FINAL selected slices hit the GT?
            final_slices = final_selection['slices']
            slice_hit = any(s in gt_slices for s in final_slices)
            
            # Append Result
            results.append({
                "CaseID": case_id,
                "Status": "Success",
                "GT_Lesion_Present": has_gt_lesion,
                "GT_Slices": str(gt_slices),
                "Selected_Slices": str(final_slices),
                "Slice_Hit": slice_hit,
                "AI_Result_Positive": final_selection['parsed']['infarction_present'], 
                "Any_Positive_Slab": any_positive_slab,
                "ADC_Confirmed": final_selection['adc_confirmed'],
                "Validation_Match": (has_gt_lesion == final_selection['parsed']['infarction_present']),
                "Report_Path": str(report_path)
            })
            
        except Exception as e:
            logger.error(f"Error processing {case_id}: {e}")
            results.append({
                "CaseID": case_id,
                "Status": "Error",
                "Error": str(e)
            })

    # Save Summary
    df = pd.DataFrame(results)
    summary_path = BATCH_OUTPUT_DIR / "validation_summary_sliding.csv"
    df.to_csv(summary_path, index=False)
    logger.info(f"Sliding Window Batch Validation (5 Cases) Complete. Summary saved to {summary_path}")

if __name__ == "__main__":
    run_batch()
