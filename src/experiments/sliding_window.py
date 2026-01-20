import sys
import logging
import argparse
from pathlib import Path
import time
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import OUTPUT_DIR, SLICES_PER_CASE
from src.data.loader import DataLoader
from src.data.preprocessor import ImagePreprocessor
from src.utils.adc_gate import check_adc_confirmation
from src.model.medgemma import MedGemmaPredictor
from src.report.parser import parse_medgemma_output

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_experiment(case_id: str, window_size: int = 3, stride: int = 3):
    logger.info(f"Starting Sliding Window Experiment for {case_id}")
    logger.info(f"Config: Window Size={window_size}, Stride={stride}")
    
    loader = DataLoader()
    case = loader.load_case(case_id)
    predictor = MedGemmaPredictor()
    
    num_slices = case.dwi_volume.shape[2]
    # Define window: 3 slices, stride 3 (non-overlapping coverage)
    # Restrict to middle 60% of brain to save time/noise
    start_idx = int(num_slices * 0.2)
    end_idx = int(num_slices * 0.8)
    
    logger.info(f"Volume Depth: {num_slices}. Scanning range [{start_idx}, {end_idx}] with stride {stride}.")
    
    findings = []
    
    start_time = time.time()
    
    for i in range(start_idx, end_idx, stride):
        # Current window indices
        slice_indices = [i + j for j in range(window_size) if (i + j) < end_idx]
        if not slice_indices: continue
        
        logger.info(f"Processing Slab: {slice_indices}")
        
        # Prepare Images
        dwi_images = []
        for idx in slice_indices:
            dwi_images.append(ImagePreprocessor.process_slice(case.dwi_volume[:, :, idx]))
            
        # Use middle slice for ADC/FLAIR context
        mid_slice = slice_indices[len(slice_indices)//2]
        adc_img = ImagePreprocessor.process_slice(case.adc_volume[:, :, mid_slice])
        flair_img = ImagePreprocessor.process_slice(case.flair_volume[:, :, mid_slice])
        
        # ADC Gate
        adc_confirmed = check_adc_confirmation(
            case.dwi_volume[:, :, mid_slice],
            case.adc_volume[:, :, mid_slice]
        )
        
        # Inference
        try:
            response = predictor.predict(dwi_images, adc_img, flair_img, adc_confirmed)
            parsed = parse_medgemma_output(response)
            
            # Simple check for positive findings
            is_positive = "Acute" in parsed.get("Stage", "") or "Infarction" in response
            
            finding_entry = {
                "slices": slice_indices,
                "adc_confirmed": adc_confirmed,
                "is_positive": is_positive,
                "raw": response
            }
            findings.append(finding_entry)
            logger.info(f"Result for {slice_indices}: Positive={is_positive}, ADC_Gate={adc_confirmed}")
            
        except Exception as e:
            logger.error(f"Failed on slab {slice_indices}: {e}")
            
    duration = time.time() - start_time
    logger.info(f"Experiment finished in {duration:.1f}s")
    
    # --- Comparison Report ---
    report_path = OUTPUT_DIR / f"{case_id}_sliding_window_w{window_size}_s{stride}_results.txt"
    with open(report_path, "w") as f:
        f.write(f"SLIDING WINDOW EXPERIMENT REPORT: {case_id}\n")
        f.write(f"Config: Window={window_size}, Stride={stride}\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total Slabs Processed: {len(findings)}\n")
        f.write(f"Total Time: {duration:.1f}s\n\n")
        
        positive_slabs = [f for f in findings if f['is_positive']]
        f.write(f"Positive Slabs Found: {len(positive_slabs)}\n")
        for p in positive_slabs:
            f.write(f" - Slices {p['slices']}: ADC_Gate={p['adc_confirmed']}\n")
            f.write(f"   Reasoning: {p['raw'][:200]}...\n\n")
            
        f.write("="*60 + "\n")
        f.write("FULL LOG:\n")
        for fgy in findings:
             f.write(f"Slices {fgy['slices']} -> Positive: {fgy['is_positive']} | ADC: {fgy['adc_confirmed']}\n")

    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sliding Window Experiment on a specific case.")
    parser.add_argument("--case", type=str, default="sub-strokecase0001", help="Case ID")
    parser.add_argument("--window", type=int, default=3, help="Number of slices per window")
    parser.add_argument("--stride", type=int, default=3, help="Stride step size")
    args = parser.parse_args()
    
    run_experiment(args.case, args.window, args.stride)
