import numpy as np
import logging

from src.config import DWI_HIGH_PERCENTILE, ADC_LOW_PERCENTILE

logger = logging.getLogger(__name__)

def check_adc_confirmation(dwi_slice: np.ndarray, adc_slice: np.ndarray) -> bool:
    """
    Checks if a hyperintense area in DWI corresponds to a hypointense area in ADC.
    This is the radiological definition of 'restricted diffusion' (acute stroke).
    
    Args:
        dwi_slice: 2D numpy array of DWI signal
        adc_slice: 2D numpy array of ADC signal
        
    Returns:
        bool: True if restricted diffusion pattern is confirmed.
    """
    # 1. Identify "bright" areas on DWI
    dwi_thresh = np.percentile(dwi_slice, DWI_HIGH_PERCENTILE)
    dwi_mask = dwi_slice > dwi_thresh
    
    if not np.any(dwi_mask):
        return False # No significant hyperintensity
        
    # 2. Check corresponding pixels on ADC
    # We look at the median value of ADC within the DWI mask provided region
    adc_values_in_roi = adc_slice[dwi_mask]
    
    # 3. Determine if "dark" on ADC
    # We compare the ROI ADC values to the background brain ADC values
    whole_adc_median = np.median(adc_slice[adc_slice > 0]) # approximate brain tissue median
    roi_adc_median = np.median(adc_values_in_roi)
    
    logger.debug(f"ADC Check: Whole median={whole_adc_median:.1f}, ROI median={roi_adc_median:.1f}")
    
    # Heuristic: ROI should be significantly darker than general brain tissue
    # E.g. < 80% of normal tissue value
    is_restricted = roi_adc_median < (whole_adc_median * 0.85)
    
    return bool(is_restricted)
