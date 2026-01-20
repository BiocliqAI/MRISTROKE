"""
Anatomical Location Lookup using Harvard-Oxford Atlas
Provides lobar and subcortical region labels for lesions.
"""
import numpy as np
import logging
from collections import Counter
from nilearn import datasets
import nibabel as nib

logger = logging.getLogger(__name__)

# Cached atlases
_ho_cortical = None
_ho_subcortical = None


def load_harvard_oxford_atlases():
    """Load Harvard-Oxford cortical and subcortical atlases."""
    global _ho_cortical, _ho_subcortical
    
    if _ho_cortical is None:
        logger.info("Loading Harvard-Oxford Cortical Atlas...")
        _ho_cortical = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    
    if _ho_subcortical is None:
        logger.info("Loading Harvard-Oxford Subcortical Atlas...")
        _ho_subcortical = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    
    return _ho_cortical, _ho_subcortical


# Simplified region groupings for clinical reporting
LOBE_MAPPING = {
    # Frontal Lobe
    'Frontal Pole': 'Frontal',
    'Insular Cortex': 'Insular',
    'Superior Frontal Gyrus': 'Frontal',
    'Middle Frontal Gyrus': 'Frontal',
    'Inferior Frontal Gyrus, pars triangularis': 'Frontal',
    'Inferior Frontal Gyrus, pars opercularis': 'Frontal',
    'Precentral Gyrus': 'Frontal',
    'Frontal Medial Cortex': 'Frontal',
    'Subcallosal Cortex': 'Frontal',
    'Paracingulate Gyrus': 'Frontal',
    'Cingulate Gyrus, anterior division': 'Frontal',
    'Frontal Orbital Cortex': 'Frontal',
    'Frontal Operculum Cortex': 'Frontal',
    
    # Temporal Lobe
    'Temporal Pole': 'Temporal',
    'Superior Temporal Gyrus, anterior division': 'Temporal',
    'Superior Temporal Gyrus, posterior division': 'Temporal',
    'Middle Temporal Gyrus, anterior division': 'Temporal',
    'Middle Temporal Gyrus, posterior division': 'Temporal',
    'Middle Temporal Gyrus, temporooccipital part': 'Temporal',
    'Inferior Temporal Gyrus, anterior division': 'Temporal',
    'Inferior Temporal Gyrus, posterior division': 'Temporal',
    'Inferior Temporal Gyrus, temporooccipital part': 'Temporal',
    'Planum Polare': 'Temporal',
    'Heschl\'s Gyrus (includes H1 and H2)': 'Temporal',
    'Planum Temporale': 'Temporal',
    'Parahippocampal Gyrus, anterior division': 'Temporal',
    'Parahippocampal Gyrus, posterior division': 'Temporal',
    'Temporal Fusiform Cortex, anterior division': 'Temporal',
    'Temporal Fusiform Cortex, posterior division': 'Temporal',
    'Temporal Occipital Fusiform Cortex': 'Temporal',
    
    # Parietal Lobe
    'Postcentral Gyrus': 'Parietal',
    'Superior Parietal Lobule': 'Parietal',
    'Supramarginal Gyrus, anterior division': 'Parietal',
    'Supramarginal Gyrus, posterior division': 'Parietal',
    'Angular Gyrus': 'Parietal',
    'Parietal Operculum Cortex': 'Parietal',
    'Central Opercular Cortex': 'Parietal',
    'Precuneous Cortex': 'Parietal',
    'Cingulate Gyrus, posterior division': 'Parietal',
    
    # Occipital Lobe
    'Lateral Occipital Cortex, superior division': 'Occipital',
    'Lateral Occipital Cortex, inferior division': 'Occipital',
    'Intracalcarine Cortex': 'Occipital',
    'Cuneal Cortex': 'Occipital',
    'Lingual Gyrus': 'Occipital',
    'Supracalcarine Cortex': 'Occipital',
    'Occipital Pole': 'Occipital',
    'Occipital Fusiform Gyrus': 'Occipital',
}

SUBCORTICAL_MAPPING = {
    'Left Thalamus': 'Thalamus',
    'Right Thalamus': 'Thalamus',
    'Left Caudate': 'Basal Ganglia',
    'Right Caudate': 'Basal Ganglia',
    'Left Putamen': 'Basal Ganglia',
    'Right Putamen': 'Basal Ganglia',
    'Left Pallidum': 'Basal Ganglia',
    'Right Pallidum': 'Basal Ganglia',
    'Left Hippocampus': 'Hippocampus',
    'Right Hippocampus': 'Hippocampus',
    'Left Amygdala': 'Amygdala',
    'Right Amygdala': 'Amygdala',
    'Left Accumbens': 'Basal Ganglia',
    'Right Accumbens': 'Basal Ganglia',
    'Brain-Stem': 'Brainstem',
}


def get_anatomical_location(lesion_mask_mni) -> dict:
    """
    Determine anatomical locations of a lesion using Harvard-Oxford Atlas.
    
    Args:
        lesion_mask_mni: ANTs image of lesion mask in MNI space (binary)
    
    Returns:
        dict with:
            - 'primary_location': Most common region (e.g., "Frontal")
            - 'locations': List of all anatomical regions with voxel counts
            - 'detailed_regions': Raw Harvard-Oxford labels for detailed reporting
    """
    import ants
    
    ho_cort, ho_sub = load_harvard_oxford_atlases()
    
    # Get atlas data as numpy
    # nilearn returns Nifti1Image, need to get data
    cort_img = ho_cort.maps
    sub_img = ho_sub.maps
    
    if isinstance(cort_img, nib.Nifti1Image):
        cort_data = cort_img.get_fdata()
        cort_affine = cort_img.affine
    else:
        cort_nii = nib.load(cort_img)
        cort_data = cort_nii.get_fdata()
        cort_affine = cort_nii.affine
    
    if isinstance(sub_img, nib.Nifti1Image):
        sub_data = sub_img.get_fdata()
    else:
        sub_nii = nib.load(sub_img)
        sub_data = sub_nii.get_fdata()
    
    # Resample lesion mask to atlas space (2mm)
    # Create ANTs image from atlas for reference
    # The H-O atlas is 91x109x91 at 2mm, our mask may be at 1mm
    mask_array = lesion_mask_mni.numpy()
    
    # Simple nearest-neighbor resampling if shapes differ
    if mask_array.shape != cort_data.shape:
        # Resample mask to atlas resolution using scipy
        from scipy.ndimage import zoom
        zoom_factors = [cort_data.shape[i] / mask_array.shape[i] for i in range(3)]
        mask_resampled = zoom(mask_array, zoom_factors, order=0)  # nearest neighbor
    else:
        mask_resampled = mask_array
    
    # Get lesion voxel coordinates
    lesion_voxels = np.where(mask_resampled > 0)
    total_voxels = len(lesion_voxels[0])
    
    if total_voxels == 0:
        return {
            'primary_location': "No Lesion",
            'locations': [],
            'detailed_regions': []
        }
    
    # Get cortical labels
    cort_labels_at_lesion = cort_data[lesion_voxels].astype(int)
    sub_labels_at_lesion = sub_data[lesion_voxels].astype(int)
    
    # Count regions
    all_regions = []
    
    # Process cortical labels
    cort_label_names = ho_cort.labels
    for label_id in np.unique(cort_labels_at_lesion):
        if label_id == 0:  # Background
            continue
        count = np.sum(cort_labels_at_lesion == label_id)
        region_name = cort_label_names[label_id] if label_id < len(cort_label_names) else f"Unknown-{label_id}"
        lobe = LOBE_MAPPING.get(region_name, region_name)
        all_regions.append({
            'raw_label': region_name,
            'simplified': lobe,
            'count': count,
            'type': 'cortical'
        })
    
    # Process subcortical labels
    sub_label_names = ho_sub.labels
    for label_id in np.unique(sub_labels_at_lesion):
        if label_id == 0:  # Background
            continue
        count = np.sum(sub_labels_at_lesion == label_id)
        region_name = sub_label_names[label_id] if label_id < len(sub_label_names) else f"Unknown-{label_id}"
        simplified = SUBCORTICAL_MAPPING.get(region_name, region_name)
        all_regions.append({
            'raw_label': region_name,
            'simplified': simplified,
            'count': count,
            'type': 'subcortical'
        })
    
    # Sort by count
    all_regions.sort(key=lambda x: x['count'], reverse=True)
    
    # Group by simplified location
    location_counts = Counter()
    for r in all_regions:
        location_counts[r['simplified']] += r['count']
    
    # Get primary location
    if location_counts:
        primary_location = location_counts.most_common(1)[0][0]
    else:
        primary_location = "Unknown"
    
    # Build location list
    locations = [
        {'name': loc, 'voxel_count': cnt, 'percentage': (cnt / total_voxels) * 100}
        for loc, cnt in location_counts.most_common()
    ]
    
    logger.info(f"Anatomical lookup: Primary={primary_location}")
    
    return {
        'primary_location': primary_location,
        'locations': locations,
        'detailed_regions': all_regions[:10],  # Top 10 detailed regions
        'total_lesion_voxels': total_voxels
    }


def get_location_simple(lesion_mask_mni) -> list:
    """
    Simplified lookup that returns a list of affected lobes/regions.
    
    Args:
        lesion_mask_mni: ANTs image of lesion mask in MNI space
    
    Returns:
        List of location strings (e.g., ["Frontal", "Parietal"])
    """
    result = get_anatomical_location(lesion_mask_mni)
    
    # Return locations with >5% involvement
    significant_locations = [
        loc['name'] for loc in result['locations'] 
        if loc['percentage'] > 5
    ]
    
    return significant_locations if significant_locations else [result['primary_location']]
