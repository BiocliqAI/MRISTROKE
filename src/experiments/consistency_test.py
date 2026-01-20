"""
Consistency Testing Experiment
Runs GT-guided inference N times on M cases to check for hallucination and consistency.
"""
import sys
import os
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.config import OUTPUT_DIR
from src.data.loader import DataLoader
from src.data.preprocessor import ImagePreprocessor
from src.model.medgemma import MedGemmaPredictor
from src.report.parser import parse_medgemma_output
from src.utils.adc_gate import check_adc_confirmation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Output Directory
CONSISTENCY_OUTPUT_DIR = OUTPUT_DIR / "consistency_test"
CONSISTENCY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_representative_slab(case):
    """Returns the slab indices centered on the max lesion area slice."""
    if case.mask_volume is None:
        return None, None
    
    lesion_area_per_slice = np.sum(case.mask_volume, axis=(0, 1))
    max_slice_idx = int(np.argmax(lesion_area_per_slice))
    max_area = lesion_area_per_slice[max_slice_idx]
    
    if max_area == 0:
        return None, None
    
    # Get context slab
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


def run_single_inference(predictor, case, slice_indices, max_slice_idx):
    """Runs a single inference and returns parsed results."""
    dwi_images = [ImagePreprocessor.process_slice(case.dwi_volume[:, :, idx]) for idx in slice_indices]
    adc_image = ImagePreprocessor.process_slice(case.adc_volume[:, :, max_slice_idx])
    flair_image = ImagePreprocessor.process_slice(case.flair_volume[:, :, max_slice_idx])
    
    adc_confirmed = check_adc_confirmation(
        case.dwi_volume[:, :, max_slice_idx],
        case.adc_volume[:, :, max_slice_idx]
    )
    
    response = predictor.predict(dwi_images, adc_image, flair_image, adc_confirmed)
    parsed = parse_medgemma_output(response)
    
    return {
        "raw_response": response,
        "infarction_present": parsed.get("infarction_present", False),
        "stage": parsed.get("Stage", "Unknown"),
        "laterality": parsed.get("Laterality", "Unknown"),
        "territory": parsed.get("Territory", "Unknown"),
        "location": ", ".join(parsed.get("location", [])),
        "adc_confirmed": adc_confirmed
    }


def run_consistency_test(case_ids: list, num_runs: int = 5):
    logger.info(f"Starting Consistency Test: {len(case_ids)} cases x {num_runs} runs each")
    
    loader = DataLoader()
    predictor = MedGemmaPredictor()
    
    all_results = []
    
    for case_id in case_ids:
        logger.info(f"--- Processing {case_id} ---")
        
        try:
            case = loader.load_case(case_id)
        except Exception as e:
            logger.error(f"Failed to load {case_id}: {e}")
            continue
        
        max_slice_idx, slice_indices = get_representative_slab(case)
        if max_slice_idx is None:
            logger.warning(f"{case_id}: No lesion in GT mask. Skipping.")
            continue
        
        logger.info(f"{case_id}: Representative Slice = {max_slice_idx}, Slab = {slice_indices}")
        
        case_findings = []
        for run_num in range(num_runs):
            logger.info(f"  Run {run_num + 1}/{num_runs}...")
            result = run_single_inference(predictor, case, slice_indices, max_slice_idx)
            result["case_id"] = case_id
            result["run"] = run_num + 1
            result["rep_slice"] = max_slice_idx
            case_findings.append(result)
            all_results.append(result)
        
        # Log quick consistency summary for this case
        diagnoses = [r["infarction_present"] for r in case_findings]
        stages = [r["stage"] for r in case_findings]
        territories = [r["territory"] for r in case_findings]
        
        logger.info(f"  {case_id} Summary:")
        logger.info(f"    Diagnoses: {diagnoses} (Consistent: {len(set(diagnoses)) == 1})")
        logger.info(f"    Stages: {set(stages)}")
        logger.info(f"    Territories: {set(territories)}")
    
    # Save Results
    df = pd.DataFrame(all_results)
    summary_path = CONSISTENCY_OUTPUT_DIR / "consistency_results.csv"
    df.to_csv(summary_path, index=False)
    logger.info(f"Results saved to {summary_path}")
    
    # Generate Summary Report
    report_lines = ["CONSISTENCY TEST REPORT", "=" * 60, ""]
    
    for case_id in case_ids:
        case_df = df[df["case_id"] == case_id]
        if case_df.empty:
            continue
        
        report_lines.append(f"Case: {case_id}")
        report_lines.append("-" * 40)
        
        # Diagnosis Consistency
        diagnoses = case_df["infarction_present"].tolist()
        diag_consistent = len(set(diagnoses)) == 1
        report_lines.append(f"  Diagnosis: {'CONSISTENT' if diag_consistent else 'INCONSISTENT'} ({diagnoses})")
        
        # Stage Consistency
        stages = case_df["stage"].tolist()
        stage_consistent = len(set(stages)) == 1
        report_lines.append(f"  Stage: {'CONSISTENT' if stage_consistent else 'VARIABLE'} ({set(stages)})")
        
        # Territory Consistency
        territories = case_df["territory"].tolist()
        territory_consistent = len(set(territories)) == 1
        report_lines.append(f"  Territory: {'CONSISTENT' if territory_consistent else 'VARIABLE'} ({set(territories)})")
        
        # Location Consistency
        locations = case_df["location"].tolist()
        location_consistent = len(set(locations)) == 1
        report_lines.append(f"  Location: {'CONSISTENT' if location_consistent else 'VARIABLE'} ({set(locations)})")
        
        report_lines.append("")
    
    # Overall Summary
    overall_diag_consistent = all(
        len(set(df[df["case_id"] == cid]["infarction_present"].tolist())) == 1
        for cid in case_ids if not df[df["case_id"] == cid].empty
    )
    report_lines.append("=" * 60)
    report_lines.append(f"OVERALL DIAGNOSIS CONSISTENCY: {'PASS' if overall_diag_consistent else 'FAIL'}")
    
    report_path = CONSISTENCY_OUTPUT_DIR / "consistency_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"Report saved to {report_path}")
    print("\n".join(report_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Consistency Test")
    parser.add_argument("--cases", type=str, nargs="+", 
                        default=["sub-strokecase0001", "sub-strokecase0002", "sub-strokecase0003", 
                                 "sub-strokecase0004", "sub-strokecase0005"],
                        help="List of case IDs")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per case")
    args = parser.parse_args()
    
    run_consistency_test(args.cases, args.runs)
