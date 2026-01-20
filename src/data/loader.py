import nibabel as nib
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import logging

from src.config import DATASET_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MRICase:
    case_id: str
    dwi_path: Path
    adc_path: Path
    flair_path: Path
    dwi_volume: np.ndarray
    adc_volume: np.ndarray
    flair_volume: np.ndarray
    affine: np.ndarray
    mask_path: Optional[Path] = None
    mask_volume: Optional[np.ndarray] = None

class DataLoader:
    def __init__(self, data_root: Path = Path(DATASET_ROOT)):
        self.data_root = data_root

    def load_case(self, case_id: str) -> MRICase:
        """
        Loads DWI, ADC, and FLAIR volumes for a given case ID.
        Also attempts to load ground truth mask from derivatives if available.
        Expected structure: 'sub-strokecaseXXXX/ses-0001/...'
        """
        case_dir = self.data_root / case_id / "ses-0001"
        if not case_dir.exists():
            raise FileNotFoundError(f"Case directory not found: {case_dir}")

        # Construct paths
        dwi_files = list(case_dir.glob("dwi/*_dwi.nii.gz"))
        adc_files = list(case_dir.glob("dwi/*_adc.nii.gz"))
        flair_files = list(case_dir.glob("anat/*_FLAIR.nii.gz"))

        if not dwi_files: raise FileNotFoundError(f"DWI file missing for {case_id}")
        if not adc_files: raise FileNotFoundError(f"ADC file missing for {case_id}")
        if not flair_files: raise FileNotFoundError(f"FLAIR file missing for {case_id}")

        dwi_path = dwi_files[0]
        adc_path = adc_files[0]
        flair_path = flair_files[0]
        
        # Look for mask in derivatives
        # derivatives/sub-strokecaseXXXX/ses-0001/sub-strokecaseXXXX_ses-0001_msk.nii.gz
        deriv_dir = self.data_root / "derivatives" / case_id / "ses-0001"
        mask_path = None
        mask_vol = None
        
        if deriv_dir.exists():
            mask_files = list(deriv_dir.glob("*_msk.nii.gz"))
            if mask_files:
                mask_path = mask_files[0]

        logger.info(f"Loading case {case_id}...")
        
        try:
            dwi_img = nib.load(dwi_path)
            adc_img = nib.load(adc_path)
            flair_img = nib.load(flair_path)
            
            # Helper to retrieve data, potentially resampling
            def get_aligned_volume(target_img, source_img, name):
                import nibabel.processing
                
                # Check shapes and affines
                if (target_img.shape == source_img.shape) and np.allclose(target_img.affine, source_img.affine):
                    return source_img.get_fdata()
                
                logger.info(f"Resampling {name} to match DWI geometry...")
                # Resample source to match target (DWI)
                resampled_img = nibabel.processing.resample_from_to(source_img, target_img, order=1) # Linear interpolation
                return resampled_img.get_fdata()

            dwi_vol = dwi_img.get_fdata()
            adc_vol = get_aligned_volume(dwi_img, adc_img, "ADC")
            flair_vol = get_aligned_volume(dwi_img, flair_img, "FLAIR")
            
            mask_path = None
            mask_vol = None
            
            # Look for mask in derivatives
            deriv_dir = self.data_root / "derivatives" / case_id / "ses-0001"
            if deriv_dir.exists():
                mask_files = list(deriv_dir.glob("*_msk.nii.gz"))
                if mask_files:
                    mask_path = mask_files[0]
                    logger.info(f"Found mask at {mask_path}")
                    mask_img = nib.load(mask_path)
                    # Resample mask (nearest neighbor for labels)
                    # Note: We need to be careful with mask resampling to preserve 0/1, so order=0
                    if (dwi_img.shape == mask_img.shape) and np.allclose(dwi_img.affine, mask_img.affine):
                         mask_vol = mask_img.get_fdata()
                    else:
                         import nibabel.processing
                         logger.info("Resampling Mask to match DWI geometry...")
                         mask_res = nibabel.processing.resample_from_to(mask_img, dwi_img, order=0)
                         mask_vol = mask_res.get_fdata()
            
            self._validate_volumes(dwi_vol, adc_vol, flair_vol, case_id)

            return MRICase(
                case_id=case_id,
                dwi_path=dwi_path,
                adc_path=adc_path,
                flair_path=flair_path,
                dwi_volume=dwi_vol,
                adc_volume=adc_vol,
                flair_volume=flair_vol,
                affine=dwi_img.affine,
                mask_path=mask_path,
                mask_volume=mask_vol
            )

        except Exception as e:
            logger.error(f"Failed to load case {case_id}: {e}")
            raise

    def _validate_volumes(self, dwi: np.ndarray, adc: np.ndarray, flair: np.ndarray, case_id: str):
        """
        Validates shape consistency and content integrity.
        """
        if dwi.shape != adc.shape:
             # This is critical for pixel-wise correspondence
             # FLAIR might be different resolution in some datasets, but for this pipeline
             # we ideally want correspondence. If FLAIR is diff, we might need resampling (future work).
             # For now, we enforce strict checking or log warnings.
             logger.warning(f"Shape mismatch: DWI/ADC {dwi.shape} vs FLAIR {flair.shape} for case {case_id}")
             # We assume DWI and ADC MUST match exactly.
        
        if dwi.size == 0 or adc.size == 0:
            raise ValueError(f"Empty volume detected for case {case_id}")

        if np.var(dwi) == 0:
            raise ValueError(f"Zero variance (blank scan) detected in DWI for {case_id}")
