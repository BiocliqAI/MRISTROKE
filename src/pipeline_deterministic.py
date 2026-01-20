"""
Fully Deterministic Pipeline with MedGemma Report Writing

All clinical findings are determined algorithmically:
- Detection: From segmentation mask (GT for now)
- Staging: ADC/FLAIR signal ratios
- Territory: Arterial Atlas
- Location: Harvard-Oxford Atlas

MedGemma is ONLY used for natural language report generation,
grounded to the deterministic findings.
"""
import sys
import os
import argparse
import logging
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from pathlib import Path
import numpy as np

from config import DATASET_ROOT
from data.loader import DataLoader
from data.preprocessor import ImagePreprocessor
from staging.deterministic import stage_from_volumes
from territory.registration import register_to_mni, apply_transform_to_mask
from territory.lookup import get_vascular_territory
from territory.anatomical import get_anatomical_location, get_location_simple
from model.medgemma import MedGemmaPredictor
from model.prompts import STROKE_ANALYSIS_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_PATH = Path(DATASET_ROOT)

# Cases to process
CASES = [
    "sub-strokecase0001",
    "sub-strokecase0002",
    "sub-strokecase0003",
    "sub-strokecase0004",
    "sub-strokecase0005",
]


def get_representative_slice(case) -> int:
    """Get the slice with maximum lesion area."""
    if case.mask_volume is None:
        return case.dwi_volume.shape[2] // 2
    
    lesion_area = np.sum(case.mask_volume, axis=(0, 1))
    return int(np.argmax(lesion_area))


def run_deterministic_findings(case_id: str) -> dict:
    """
    Run fully deterministic analysis on a case.
    
    Returns structured findings dict.
    """
    logger.info(f"Processing {case_id} - Deterministic Analysis")
    
    # Load case
    loader = DataLoader()
    case = loader.load_case(case_id)
    
    # 1. DETECTION (from GT mask)
    lesion_voxels = np.sum(case.mask_volume > 0)
    detection = lesion_voxels > 10  # More than 10 voxels = positive
    
    if not detection:
        return {
            'case_id': case_id,
            'detection': False,
            'stage': None,
            'territory': None,
            'location': None,
        }
    
    # 2. STAGING (deterministic from ADC/FLAIR ratios)
    staging_result = stage_from_volumes(
        lesion_mask=case.mask_volume,
        adc_volume=case.adc_volume,
        flair_volume=case.flair_volume,
        dwi_volume=case.dwi_volume
    )
    
    # 3. TERRITORY (Atlas-based)
    dwi_path = DATASET_PATH / case_id / "ses-0001" / "dwi" / f"{case_id}_ses-0001_dwi.nii.gz"
    mask_path = DATASET_PATH / "derivatives" / case_id / "ses-0001" / f"{case_id}_ses-0001_msk.nii.gz"
    
    reg_result = register_to_mni(str(dwi_path), type_of_transform='Affine')
    warped_mask = apply_transform_to_mask(str(mask_path), reg_result['forward_transforms'])
    
    territory_result = get_vascular_territory(warped_mask, include_laterality=True)
    territory = territory_result['primary_territory']
    territory_major = territory.split('-')[0]
    
    # Determine laterality from territory
    if '-L' in territory:
        laterality = "Left"
    elif '-R' in territory:
        laterality = "Right"
    else:
        laterality = "Bilateral"
    
    # 4. LOCATION (Atlas-based)
    anat_result = get_anatomical_location(warped_mask)
    locations = [
        loc['name'] for loc in anat_result['locations']
        if loc['percentage'] > 5 and 'Cerebral' not in loc['name']
    ][:3]  # Top 3 locations
    
    if not locations:
        locations = [anat_result['primary_location']]
    
    # Check ADC confirmation
    adc_confirmed = staging_result['ratios'].get('adc_ratio', 1.0) < 0.85
    
    return {
        'case_id': case_id,
        'detection': True,
        'stage': staging_result['stage'],
        'stage_confidence': staging_result['confidence'],
        'stage_ratios': staging_result['ratios'],
        'territory': territory,
        'territory_major': territory_major,
        'territory_overlap': territory_result['overlap_percentage'],
        'laterality': laterality,
        'locations': locations,
        'adc_confirmed': adc_confirmed,
        'lesion_voxels': int(lesion_voxels),
    }


def generate_report_with_medgemma(findings: dict, predictor: MedGemmaPredictor) -> str:
    """
    Use MedGemma to generate a natural language report grounded to deterministic findings.
    """
    if not findings['detection']:
        return f"""
================================================================================
                    MRI BRAIN STROKE ANALYSIS PROTOCOL
================================================================================
CASE ID: {findings['case_id']}
ANALYSIS METHOD: Fully Deterministic + VLM Report Writing

FINDINGS:
• No lesion detected in segmentation mask.
• Brain parenchyma demonstrates normal signal intensity.

IMPRESSION:
No MRI evidence of acute ischemic stroke.

--------------------------------------------------------------------------------
AI DISCLAIMER: This report was generated by an AI system.
NOT FOR CLINICAL DIAGNOSIS. All findings must be verified by a radiologist.
--------------------------------------------------------------------------------
"""
    
    # Build grounding prompt
    grounding_prompt = f"""You are a neuroradiology AI assistant. Generate a professional MRI brain stroke report based ONLY on the following deterministic findings. Do NOT add or infer any information not provided.

DETERMINISTIC FINDINGS:
- Detection: Infarction PRESENT
- Stage: {findings['stage']} (confidence: {findings['stage_confidence']:.0%})
- Vascular Territory: {findings['territory']} ({findings['territory_overlap']:.0f}% overlap)
- Laterality: {findings['laterality']}
- Anatomical Location(s): {', '.join(findings['locations'])}
- ADC Confirmation: {'Yes - restricted diffusion confirmed' if findings['adc_confirmed'] else 'No - ADC signals equivocal'}
- ADC Ratio: {findings['stage_ratios'].get('adc_ratio', 0):.2f}
- FLAIR Ratio: {findings['stage_ratios'].get('flair_ratio', 0):.2f}
- Lesion Volume: {findings['lesion_voxels']} voxels

Generate a professional radiologist-style report with:
1. FINDINGS section (bullet points)
2. IMPRESSION section (1-2 sentences)

Use standard radiology terminology. Be concise."""

    # Get MedGemma response (text-only, no images needed for report writing)
    response = predictor.generate_text_only(grounding_prompt)
    
    report = f"""
================================================================================
                    MRI BRAIN STROKE ANALYSIS PROTOCOL
================================================================================
CASE ID: {findings['case_id']}
ANALYSIS METHOD: Fully Deterministic + VLM Report Writing
LOCALIZATION: Atlas-Based (Arterial + Harvard-Oxford)

--- DETERMINISTIC FINDINGS ---
Detection: POSITIVE
Stage: {findings['stage']} (ADC={findings['stage_ratios'].get('adc_ratio', 0):.2f}, FLAIR={findings['stage_ratios'].get('flair_ratio', 0):.2f})
Territory: {findings['territory']} ({findings['territory_overlap']:.0f}% overlap)
Location: {', '.join(findings['locations'])}
Laterality: {findings['laterality']}
ADC Confirmed: {findings['adc_confirmed']}

--- VLM-GENERATED REPORT ---
{response}

--------------------------------------------------------------------------------
AI DISCLAIMER: 
Clinical findings determined algorithmically. Report text generated by AI.
- Staging: ADC/FLAIR ratio analysis
- Territory: Arterial Atlas (Johns Hopkins, 2022)
- Location: Harvard-Oxford Atlas (FSL)
NOT FOR CLINICAL DIAGNOSIS. All findings must be verified by a radiologist.
--------------------------------------------------------------------------------
"""
    return report


def run_deterministic_pipeline(output_dir: Path = None):
    """Run the full deterministic pipeline on all cases."""
    
    print("=" * 70)
    print("FULLY DETERMINISTIC PIPELINE")
    print("Detection: GT Segmentation | Staging: ADC/FLAIR Ratios")
    print("Territory: Arterial Atlas | Location: Harvard-Oxford Atlas")
    print("Report: MedGemma (grounded to findings)")
    print("=" * 70)
    print()
    
    results = []
    
    # Load MedGemma once for report writing
    logger.info("Loading MedGemma for report writing...")
    predictor = MedGemmaPredictor()
    
    for case_id in CASES:
        print(f"\n{'='*60}")
        print(f"Processing {case_id}")
        print(f"{'='*60}")
        
        try:
            # Get deterministic findings
            findings = run_deterministic_findings(case_id)
            
            # Generate report with MedGemma
            report = generate_report_with_medgemma(findings, predictor)
            
            # Store results
            findings['report'] = report
            results.append(findings)
            
            # Save report
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                report_path = output_dir / f"{case_id}_deterministic_report.txt"
                with open(report_path, 'w') as f:
                    f.write(report)
            
            print(f"  Detection: {'POSITIVE' if findings['detection'] else 'NEGATIVE'}")
            if findings['detection']:
                print(f"  Stage: {findings['stage']}")
                print(f"  Territory: {findings['territory']}")
                print(f"  Location: {', '.join(findings['locations'])}")
            
        except Exception as e:
            logger.error(f"Error processing {case_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Case':<25} {'Stage':<15} {'Territory':<12} {'Location'}")
    print("-" * 70)
    for r in results:
        if r['detection']:
            print(f"{r['case_id']:<25} {r['stage']:<15} {r['territory']:<12} {', '.join(r['locations'][:2])}")
        else:
            print(f"{r['case_id']:<25} {'NEGATIVE':<15} {'-':<12} -")
    
    # Save summary JSON
    if output_dir:
        summary_path = output_dir / "deterministic_summary.json"
        # Remove non-serializable items
        for r in results:
            r.pop('report', None)
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSummary saved to {summary_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully Deterministic Pipeline")
    parser.add_argument("--output", type=str, default="reports/deterministic", help="Output directory")
    args = parser.parse_args()
    
    run_deterministic_pipeline(Path(args.output))
