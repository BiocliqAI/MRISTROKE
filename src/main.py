import argparse
import logging
import sys
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import OUTPUT_DIR, SLICES_PER_CASE
from src.data.loader import DataLoader
from src.data.preprocessor import ImagePreprocessor
from src.utils.adc_gate import check_adc_confirmation
from src.model.medgemma import MedGemmaPredictor
from src.report.parser import parse_medgemma_output
from src.report.generator import generate_report
from src.report.visualizer import Visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="MedGemma Stroke Reporting Pipeline")
    parser.add_argument("--case", type=str, required=True, help="Case ID (e.g., sub-strokecase0001)")
    parser.add_argument("--save", action="store_true", help="Save report to text file")
    parser.add_argument("--slices", type=str, help="Comma-separated slice indices (override auto-selection)")
    args = parser.parse_args()
    
    try:
        # 0. Setup
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. Load Data
        loader = DataLoader()
        case = loader.load_case(args.case)
        
        # 2. Preprocessing & Slice Selection
        if args.slices:
            logger.info(f"Using manual slice selection: {args.slices}")
            dwi_slice_indices = [int(s.strip()) for s in args.slices.split(",")]
        else:
            logger.info("Selecting slices...")
            dwi_slice_indices = ImagePreprocessor.select_slices(case.dwi_volume, top_n=SLICES_PER_CASE)
            
        logger.info(f"Final Slice Selection: {dwi_slice_indices}")
        
        # Prepare Images
        dwi_images = []
        # We need corresponding ADC/FLAIR slices. 
        # For simplicity, we use the BEST single slice index from DWI for the ADC/FLAIR reference,
        # but we pass multiple DWI slices to giving 'volumetric' context.
        # Let's take the middle one of the selected slices as the "primary" one for ADC/FLAIR.
        primary_slice_idx = dwi_slice_indices[len(dwi_slice_indices)//2]
        
        for idx in dwi_slice_indices:
            dwi_images.append(ImagePreprocessor.process_slice(case.dwi_volume[:, :, idx]))
            
        adc_image = ImagePreprocessor.process_slice(case.adc_volume[:, :, primary_slice_idx])
        flair_image = ImagePreprocessor.process_slice(case.flair_volume[:, :, primary_slice_idx])
        
        # 3. ADC Gate
        logger.info(f"Checking ADC confirmation on slice {primary_slice_idx}...")
        adc_confirmed = check_adc_confirmation(
            case.dwi_volume[:, :, primary_slice_idx],
            case.adc_volume[:, :, primary_slice_idx]
        )
        logger.info(f"ADC Confirmation: {adc_confirmed}")
        
        # 4. Inference
        predictor = MedGemmaPredictor()
        raw_response = predictor.predict(dwi_images, adc_image, flair_image, adc_confirmed)
        
        # 5. Reporting
        parsed_data = parse_medgemma_output(raw_response)
        report = generate_report(args.case, parsed_data, adc_confirmed)
        
        print(report)
        print("\n" + "="*80 + "\n")
        print("RAW MEDGEMMA OUTPUT:\n", raw_response)
        
        if args.save:
            # Save Text Report
            txt_path = OUTPUT_DIR / f"{args.case}_report.txt"
            with open(txt_path, "w") as f:
                f.write(report)
            logger.info(f"Text report saved to {txt_path}")
            
            # Save Visual Report
            viz_path = OUTPUT_DIR / f"{args.case}_visual_report.png"
            
            # Get raw slice data for visualization
            dwi_disp = case.dwi_volume[:, :, primary_slice_idx]
            adc_disp = case.adc_volume[:, :, primary_slice_idx]
            flair_disp = case.flair_volume[:, :, primary_slice_idx]
            mask_disp = case.mask_volume[:, :, primary_slice_idx] if case.mask_volume is not None else None
            
            Visualizer.save_report_image(
                case_id=args.case,
                report_text=report,
                dwi_slice=dwi_disp,
                adc_slice=adc_disp,
                flair_slice=flair_disp,
                mask_slice=mask_disp,
                output_path=viz_path
            )
            logger.info(f"Visual report saved to {viz_path}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
