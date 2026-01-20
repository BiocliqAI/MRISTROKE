"""
Deterministic Staging Module
Stages infarcts based on ADC/FLAIR signal ratios without using VLM.
"""
import numpy as np
import nibabel as nib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def get_contralateral_reference(volume: np.ndarray, lesion_mask: np.ndarray) -> np.ndarray:
    """
    Get a reference mask from the contralateral hemisphere (opposite side of lesion).
    
    Assumes:
    - Volume is in standard orientation (RAS or LAS)
    - Left/Right is along axis 0
    """
    # Determine which hemisphere has the lesion
    midline = volume.shape[0] // 2
    
    left_lesion_voxels = np.sum(lesion_mask[:midline, :, :])
    right_lesion_voxels = np.sum(lesion_mask[midline:, :, :])
    
    # Create reference mask on contralateral side
    reference_mask = np.zeros_like(lesion_mask)
    
    if left_lesion_voxels > right_lesion_voxels:
        # Lesion on left, use right hemisphere for reference
        # Mirror the lesion mask to right side
        reference_mask[midline:, :, :] = lesion_mask[:midline, :, :][::-1, :, :]
    else:
        # Lesion on right, use left hemisphere for reference
        reference_mask[:midline, :, :] = lesion_mask[midline:, :, :][::-1, :, :]
    
    # Make sure we have some reference voxels
    if np.sum(reference_mask) < 10:
        # Fallback: use central 20% of brain as reference
        logger.warning("Could not find contralateral reference, using central fallback")
        center = [s // 2 for s in volume.shape]
        size = [s // 10 for s in volume.shape]  # 10% of each dimension
        reference_mask[
            center[0]-size[0]:center[0]+size[0],
            center[1]-size[1]:center[1]+size[1],
            center[2]-size[2]:center[2]+size[2]
        ] = 1
        # Exclude lesion area
        reference_mask[lesion_mask > 0] = 0
    
    return reference_mask


def calculate_signal_ratios(
    lesion_mask: np.ndarray,
    adc_volume: np.ndarray,
    flair_volume: np.ndarray,
    dwi_volume: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate signal ratios for staging.
    
    Returns dict with:
        - adc_ratio: lesion_ADC / reference_ADC (< 1 = restricted diffusion)
        - flair_ratio: lesion_FLAIR / reference_FLAIR (> 1 = bright)
        - dwi_ratio: lesion_DWI / reference_DWI (> 1 = bright)
    """
    # Get reference region
    reference_mask = get_contralateral_reference(adc_volume, lesion_mask)
    
    # Ensure masks are boolean
    lesion_mask = lesion_mask > 0
    reference_mask = reference_mask > 0
    
    # Calculate mean intensities
    lesion_adc = np.mean(adc_volume[lesion_mask])
    reference_adc = np.mean(adc_volume[reference_mask])
    adc_ratio = lesion_adc / reference_adc if reference_adc > 0 else 1.0
    
    lesion_flair = np.mean(flair_volume[lesion_mask])
    reference_flair = np.mean(flair_volume[reference_mask])
    flair_ratio = lesion_flair / reference_flair if reference_flair > 0 else 1.0
    
    result = {
        'adc_ratio': adc_ratio,
        'flair_ratio': flair_ratio,
        'lesion_adc': lesion_adc,
        'reference_adc': reference_adc,
        'lesion_flair': lesion_flair,
        'reference_flair': reference_flair,
    }
    
    if dwi_volume is not None:
        lesion_dwi = np.mean(dwi_volume[lesion_mask])
        reference_dwi = np.mean(dwi_volume[reference_mask])
        result['dwi_ratio'] = lesion_dwi / reference_dwi if reference_dwi > 0 else 1.0
    
    return result


def determine_stage(adc_ratio: float, flair_ratio: float) -> Tuple[str, float]:
    """
    Determine stroke stage based on signal ratios.
    
    Decision tree based on established MRI patterns:
    - Hyperacute (<6h): DWI+, ADC-, FLAIR-
    - Acute (6h-7d): DWI+, ADC-, FLAIR+
    - Subacute (1-3w): DWI+/-, ADC normalized, FLAIR+
    - Chronic (>3w): DWI-, ADC+, FLAIR+ with volume loss
    
    Returns:
        (stage_name, confidence_score)
    """
    # ADC thresholds (ratio relative to normal tissue)
    ADC_RESTRICTED = 0.75   # Below this = definitely restricted
    ADC_LOW = 0.85          # Below this = likely restricted
    ADC_NORMAL = 1.15       # Below this = normal range
    # Above ADC_NORMAL = elevated (chronic/T2 shine-through)
    
    # FLAIR thresholds
    FLAIR_NORMAL = 1.1      # Below this = not bright
    FLAIR_BRIGHT = 1.3      # Above this = definitely bright
    
    # Decision logic
    if adc_ratio < ADC_RESTRICTED:
        # Definitely restricted diffusion
        if flair_ratio < FLAIR_NORMAL:
            return ("Hyperacute", 0.85)  # DWI+, ADC-, FLAIR- 
        elif flair_ratio >= FLAIR_BRIGHT:
            return ("Acute", 0.95)       # DWI+, ADC-, FLAIR+
        else:
            return ("Acute", 0.75)       # DWI+, ADC-, FLAIR equivocal
    
    elif adc_ratio < ADC_LOW:
        # Likely restricted
        if flair_ratio >= FLAIR_BRIGHT:
            return ("Acute", 0.80)
        else:
            return ("Acute/Subacute", 0.60)
    
    elif adc_ratio < ADC_NORMAL:
        # ADC pseudonormalized
        if flair_ratio >= FLAIR_BRIGHT:
            return ("Subacute", 0.75)
        else:
            return ("Indeterminate", 0.40)
    
    else:
        # ADC elevated (>=1.15)
        if flair_ratio >= FLAIR_BRIGHT:
            return ("Chronic", 0.70)
        else:
            return ("Indeterminate", 0.30)


def stage_from_volumes(
    lesion_mask: np.ndarray,
    adc_volume: np.ndarray,
    flair_volume: np.ndarray,
    dwi_volume: np.ndarray = None
) -> Dict[str, Any]:
    """
    Full staging workflow from volumes.
    
    Returns dict with:
        - stage: Stage name
        - confidence: Confidence score (0-1)
        - ratios: Signal ratio details
        - decision_factors: Explanation of decision
    """
    # Validate lesion exists
    if np.sum(lesion_mask > 0) < 10:
        return {
            'stage': 'No Lesion',
            'confidence': 1.0,
            'ratios': {},
            'decision_factors': 'Lesion mask has < 10 voxels'
        }
    
    # Calculate ratios
    ratios = calculate_signal_ratios(lesion_mask, adc_volume, flair_volume, dwi_volume)
    
    # Determine stage
    stage, confidence = determine_stage(ratios['adc_ratio'], ratios['flair_ratio'])
    
    # Build decision explanation
    decision_factors = []
    if ratios['adc_ratio'] < 0.85:
        decision_factors.append(f"ADC restricted (ratio={ratios['adc_ratio']:.2f})")
    else:
        decision_factors.append(f"ADC not restricted (ratio={ratios['adc_ratio']:.2f})")
    
    if ratios['flair_ratio'] > 1.2:
        decision_factors.append(f"FLAIR bright (ratio={ratios['flair_ratio']:.2f})")
    else:
        decision_factors.append(f"FLAIR not bright (ratio={ratios['flair_ratio']:.2f})")
    
    logger.info(f"Deterministic staging: {stage} (confidence={confidence:.2f})")
    logger.info(f"  ADC ratio: {ratios['adc_ratio']:.2f}, FLAIR ratio: {ratios['flair_ratio']:.2f}")
    
    return {
        'stage': stage,
        'confidence': confidence,
        'ratios': ratios,
        'decision_factors': "; ".join(decision_factors)
    }
