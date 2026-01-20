"""
ANTs-based Registration for MNI Space Alignment
Registers patient DWI to MNI152 template for atlas lookup.
"""
import ants
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# MNI152 template (bundled with ANTsPy)
MNI_TEMPLATE = None


def get_mni_template():
    """Get the MNI152 template image from ANTsPy."""
    global MNI_TEMPLATE
    if MNI_TEMPLATE is None:
        MNI_TEMPLATE = ants.get_ants_data('mni')
        MNI_TEMPLATE = ants.image_read(MNI_TEMPLATE)
    return MNI_TEMPLATE


def register_to_mni(moving_image_path: str, type_of_transform: str = "Affine"):
    """
    Register a patient image to MNI152 space using ANTs.
    
    Args:
        moving_image_path: Path to the patient image (NIfTI)
        type_of_transform: Registration type. Options:
            - "Affine": Fast, 12 DOF (recommended for speed)
            - "SyN": Nonlinear, more accurate but slower (~2-5 min)
            - "SyNRA": Rigid + Affine + SyN
    
    Returns:
        dict with:
            - 'warped_image': ANTs image in MNI space
            - 'forward_transforms': List of transform files
            - 'inverse_transforms': List of inverse transform files
    
    Raises:
        RuntimeError: If registration fails
    """
    logger.info(f"Registering {moving_image_path} to MNI152 space using {type_of_transform}...")
    
    # Load images
    fixed = get_mni_template()
    moving = ants.image_read(str(moving_image_path))
    
    # Perform registration
    try:
        registration = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=type_of_transform,
            verbose=False
        )
    except Exception as e:
        raise RuntimeError(f"ANTs registration failed: {e}")
    
    logger.info("Registration complete.")
    
    return {
        'warped_image': registration['warpedmovout'],
        'forward_transforms': registration['fwdtransforms'],
        'inverse_transforms': registration['invtransforms'],
    }


def apply_transform_to_mask(mask_path: str, transforms: list, reference_image=None):
    """
    Apply the forward transforms to warp a lesion mask to MNI space.
    
    Args:
        mask_path: Path to patient lesion mask (NIfTI)
        transforms: List of transform files from register_to_mni()
        reference_image: Reference image (MNI template if None)
    
    Returns:
        ANTs image of the mask in MNI space
    """
    if reference_image is None:
        reference_image = get_mni_template()
    
    mask = ants.image_read(str(mask_path))
    
    # Apply transforms (use nearest neighbor for masks)
    warped_mask = ants.apply_transforms(
        fixed=reference_image,
        moving=mask,
        transformlist=transforms,
        interpolator='nearestNeighbor'
    )
    
    return warped_mask
