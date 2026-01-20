"""
Vascular Territory Lookup
Determines the vascular territory of a lesion based on atlas overlap.
"""
import numpy as np
import logging
from collections import Counter
from .atlas_loader import load_atlas, get_label_name, MAJOR_TERRITORIES

logger = logging.getLogger(__name__)


def get_vascular_territory(lesion_mask_mni, include_laterality: bool = True) -> dict:
    """
    Determine the vascular territory of a lesion based on atlas overlap.
    
    Args:
        lesion_mask_mni: ANTs image of lesion mask in MNI space (binary)
        include_laterality: If True, returns "MCA-L", else "MCA"
    
    Returns:
        dict with:
            - 'primary_territory': Most common territory (e.g., "MCA-L" or "MCA")
            - 'territories': List of all overlapping territories with voxel counts
            - 'total_lesion_voxels': Total number of lesion voxels
            - 'overlap_percentage': Percentage of lesion in primary territory
    """
    # Load atlas
    atlas = load_atlas()
    
    # Convert to numpy
    mask_array = lesion_mask_mni.numpy()
    atlas_array = atlas.numpy()
    
    # Ensure same shape (may need resampling in practice)
    if mask_array.shape != atlas_array.shape:
        logger.warning(f"Shape mismatch: mask {mask_array.shape} vs atlas {atlas_array.shape}. Resampling...")
        import ants
        lesion_mask_mni = ants.resample_image_to_target(lesion_mask_mni, atlas, interp_type='nearestNeighbor')
        mask_array = lesion_mask_mni.numpy()
    
    # Get lesion voxel coordinates
    lesion_voxels = np.where(mask_array > 0)
    total_lesion_voxels = len(lesion_voxels[0])
    
    if total_lesion_voxels == 0:
        return {
            'primary_territory': "No Lesion",
            'territories': [],
            'total_lesion_voxels': 0,
            'overlap_percentage': 0.0
        }
    
    # Get atlas labels at lesion locations
    territory_labels = atlas_array[lesion_voxels]
    
    # Filter out background (0) and ventricles (9, 10)
    valid_labels = territory_labels[(territory_labels > 0) & (territory_labels < 9)]
    
    if len(valid_labels) == 0:
        return {
            'primary_territory': "Unknown (no atlas overlap)",
            'territories': [],
            'total_lesion_voxels': total_lesion_voxels,
            'overlap_percentage': 0.0
        }
    
    # Count territories
    label_counts = Counter(valid_labels.astype(int))
    
    # Get primary (most common) territory
    primary_label = label_counts.most_common(1)[0][0]
    primary_count = label_counts.most_common(1)[0][1]
    
    primary_territory = get_label_name(primary_label, include_laterality)
    overlap_percentage = (primary_count / total_lesion_voxels) * 100
    
    # Build territory list
    territories = [
        {
            'label': label,
            'name': get_label_name(label, include_laterality),
            'voxel_count': count,
            'percentage': (count / total_lesion_voxels) * 100
        }
        for label, count in label_counts.most_common()
    ]
    
    logger.info(f"Territory lookup: Primary={primary_territory} ({overlap_percentage:.1f}%)")
    
    return {
        'primary_territory': primary_territory,
        'territories': territories,
        'total_lesion_voxels': total_lesion_voxels,
        'overlap_percentage': overlap_percentage
    }


def get_territory_simple(lesion_mask_mni) -> str:
    """
    Simplified lookup that just returns the major territory name.
    
    Args:
        lesion_mask_mni: ANTs image of lesion mask in MNI space
    
    Returns:
        Territory name string (e.g., "MCA", "PCA", "ACA", "VB")
    """
    result = get_vascular_territory(lesion_mask_mni, include_laterality=False)
    
    # Remove laterality suffix if present
    territory = result['primary_territory']
    if territory.endswith('-L') or territory.endswith('-R'):
        territory = territory[:-2]
    
    return territory
