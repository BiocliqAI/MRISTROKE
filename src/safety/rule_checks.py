"""
Rule-Based Safety Checks for Stroke Detection Pipeline

These checks run ONLY on uncertain cases:
- Small predictions (1-100 voxels)
- No predictions (0 voxels)

They detect potential misses by analyzing DWI/ADC signal characteristics.
"""
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class SafetyCheckResult:
    """Result of safety check analysis."""
    triggered: bool  # Was a safety check triggered?
    alert: bool  # Should this case be flagged for review?
    reason: str  # Explanation of the finding
    confidence: str  # HIGH, MEDIUM, LOW
    details: Dict[str, Any]  # Additional metrics


def get_brain_mask(dwi: np.ndarray, percentile: float = 10) -> np.ndarray:
    """
    Create approximate brain mask from DWI intensity.
    Excludes background/air voxels.
    """
    threshold = np.percentile(dwi, percentile)
    return dwi > threshold


def check_dwi_hotspot(dwi: np.ndarray, brain_mask: np.ndarray, 
                      threshold_std: float = 3.5) -> Tuple[bool, float, np.ndarray]:
    """
    Check if there's an unusually bright region in DWI.
    
    Returns: (has_hotspot, max_zscore, hotspot_mask)
    """
    dwi_brain = dwi[brain_mask]
    mean = np.mean(dwi_brain)
    std = np.std(dwi_brain)
    
    if std < 1e-6:
        return False, 0.0, np.zeros_like(dwi, dtype=bool)
    
    # Z-score the entire volume
    zscore = (dwi - mean) / std
    
    # Find hotspots
    hotspot_mask = (zscore > threshold_std) & brain_mask
    max_zscore = np.max(zscore[brain_mask]) if np.any(brain_mask) else 0.0
    
    has_hotspot = np.sum(hotspot_mask) > 10  # At least 10 voxels
    
    return has_hotspot, float(max_zscore), hotspot_mask


def check_adc_coldspot(adc: np.ndarray, brain_mask: np.ndarray,
                       threshold_ratio: float = 0.7) -> Tuple[bool, float, np.ndarray]:
    """
    Check if there's a region with restricted diffusion (low ADC).
    
    Returns: (has_coldspot, min_ratio, coldspot_mask)
    """
    adc_brain = adc[brain_mask]
    mean = np.mean(adc_brain)
    
    if mean < 1e-10:
        return False, 1.0, np.zeros_like(adc, dtype=bool)
    
    # Ratio relative to mean
    ratio = adc / (mean + 1e-10)
    
    # Find coldspots
    coldspot_mask = (ratio < threshold_ratio) & brain_mask
    min_ratio = np.min(ratio[brain_mask]) if np.any(brain_mask) else 1.0
    
    has_coldspot = np.sum(coldspot_mask) > 10  # At least 10 voxels
    
    return has_coldspot, float(min_ratio), coldspot_mask


def check_dwi_adc_mismatch(dwi: np.ndarray, adc: np.ndarray, brain_mask: np.ndarray,
                           dwi_threshold_std: float = 2.5,
                           adc_threshold_ratio: float = 0.8) -> Tuple[bool, int, np.ndarray]:
    """
    Check for DWI-bright + ADC-dark regions (classic stroke pattern).
    
    This is the most specific check for acute infarct.
    
    Returns: (has_mismatch, mismatch_voxels, mismatch_mask)
    """
    # DWI bright regions
    dwi_brain = dwi[brain_mask]
    dwi_mean = np.mean(dwi_brain)
    dwi_std = np.std(dwi_brain)
    
    if dwi_std < 1e-6:
        return False, 0, np.zeros_like(dwi, dtype=bool)
    
    dwi_bright = ((dwi - dwi_mean) / dwi_std > dwi_threshold_std) & brain_mask
    
    # ADC dark regions
    adc_brain = adc[brain_mask]
    adc_mean = np.mean(adc_brain)
    
    if adc_mean < 1e-10:
        return False, 0, np.zeros_like(adc, dtype=bool)
    
    adc_dark = (adc / (adc_mean + 1e-10) < adc_threshold_ratio) & brain_mask
    
    # Mismatch = both conditions
    mismatch_mask = dwi_bright & adc_dark
    mismatch_voxels = int(np.sum(mismatch_mask))
    
    has_mismatch = mismatch_voxels > 5  # At least 5 voxels with classic pattern
    
    return has_mismatch, mismatch_voxels, mismatch_mask


def run_safety_checks(dwi: np.ndarray, adc: np.ndarray, 
                      pred_voxels: int,
                      check_threshold: int = 100) -> SafetyCheckResult:
    """
    Run rule-based safety checks on a case.
    
    Args:
        dwi: DWI volume
        adc: ADC volume
        pred_voxels: Number of voxels in segmentation prediction
        check_threshold: Only run checks if pred_voxels <= this value
    
    Returns:
        SafetyCheckResult with findings
    """
    # Determine if checks should run
    if pred_voxels > check_threshold:
        return SafetyCheckResult(
            triggered=False,
            alert=False,
            reason="Adequate lesion detected, no safety check needed",
            confidence="HIGH",
            details={"pred_voxels": pred_voxels}
        )
    
    # Create brain mask
    brain_mask = get_brain_mask(dwi)
    
    # Run individual checks
    has_hotspot, max_zscore, hotspot_mask = check_dwi_hotspot(dwi, brain_mask)
    has_coldspot, min_ratio, coldspot_mask = check_adc_coldspot(adc, brain_mask)
    has_mismatch, mismatch_voxels, mismatch_mask = check_dwi_adc_mismatch(dwi, adc, brain_mask)
    
    # Aggregate findings
    details = {
        "pred_voxels": pred_voxels,
        "dwi_hotspot": has_hotspot,
        "dwi_max_zscore": round(max_zscore, 2),
        "adc_coldspot": has_coldspot,
        "adc_min_ratio": round(min_ratio, 3),
        "dwi_adc_mismatch": has_mismatch,
        "mismatch_voxels": mismatch_voxels
    }
    
    # Decision logic
    if pred_voxels == 0:
        # No lesion predicted - check for potential miss
        if has_mismatch:
            return SafetyCheckResult(
                triggered=True,
                alert=True,
                reason=f"⚠️ POSSIBLE MISS: DWI-ADC mismatch detected ({mismatch_voxels} voxels) but segmentation found nothing",
                confidence="HIGH",
                details=details
            )
        elif has_hotspot and has_coldspot:
            return SafetyCheckResult(
                triggered=True,
                alert=True,
                reason=f"⚠️ SUSPICIOUS: DWI hotspot (z={max_zscore:.1f}) + ADC coldspot (ratio={min_ratio:.2f}) detected",
                confidence="MEDIUM",
                details=details
            )
        elif has_hotspot:
            return SafetyCheckResult(
                triggered=True,
                alert=False,
                reason=f"DWI hotspot detected (z={max_zscore:.1f}) but no ADC restriction. Likely T2 shine-through.",
                confidence="LOW",
                details=details
            )
        else:
            return SafetyCheckResult(
                triggered=True,
                alert=False,
                reason="No suspicious findings on DWI/ADC analysis. Likely true negative.",
                confidence="HIGH",
                details=details
            )
    
    else:
        # Small lesion predicted (1-100 voxels)
        if has_mismatch and mismatch_voxels > pred_voxels * 2:
            return SafetyCheckResult(
                triggered=True,
                alert=True,
                reason=f"⚠️ UNDER-SEGMENTED: Mismatch region ({mismatch_voxels} vox) >> prediction ({pred_voxels} vox)",
                confidence="HIGH",
                details=details
            )
        elif pred_voxels < 50:
            return SafetyCheckResult(
                triggered=True,
                alert=True,
                reason=f"⚠️ VERY SMALL LESION: Only {pred_voxels} voxels predicted. Manual verification recommended.",
                confidence="MEDIUM",
                details=details
            )
        else:
            return SafetyCheckResult(
                triggered=True,
                alert=False,
                reason=f"Small lesion detected ({pred_voxels} voxels). Pattern appears consistent.",
                confidence="MEDIUM",
                details=details
            )


def validate_with_safety_checks(dwi: np.ndarray, adc: np.ndarray,
                                 pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, Any]:
    """
    Run safety checks and evaluate against ground truth.
    
    Returns dict with:
        - safety_result: SafetyCheckResult
        - would_have_helped: True if safety check would have caught a miss
        - gt_voxels: Ground truth lesion size
        - dice: Dice score
    """
    pred_voxels = int(np.sum(pred_mask > 0))
    gt_voxels = int(np.sum(gt_mask > 0))
    
    # Calculate Dice
    pred_bin = pred_mask > 0
    gt_bin = gt_mask > 0
    intersection = np.sum(pred_bin & gt_bin)
    dice = 2 * intersection / (np.sum(pred_bin) + np.sum(gt_bin) + 1e-8)
    
    # Run safety checks
    safety_result = run_safety_checks(dwi, adc, pred_voxels)
    
    # Determine if safety check would have helped
    # "Helped" = flagged a case that had poor Dice or was a miss
    would_have_helped = False
    if safety_result.alert:
        if dice < 0.5 and gt_voxels > 0:
            would_have_helped = True
    
    return {
        "safety_result": safety_result,
        "would_have_helped": would_have_helped,
        "gt_voxels": gt_voxels,
        "pred_voxels": pred_voxels,
        "dice": float(dice)
    }
