"""
Atlas Loader for Vascular Territory Mapping
Loads the Johns Hopkins Arterial Territory Atlas (Level 2 = Major Territories)
"""
import nibabel as nib
from pathlib import Path

# Path to atlas files
ATLAS_DIR = Path(__file__).parent.parent.parent / "data" / "atlases" / "arterial_atlas" / "data" / "Atlas_182"
ATLAS_LEVEL2_PATH = ATLAS_DIR / "ArterialAtlas_level2.nii"

# Label mapping for Level 2 (Major Territories)
# Based on ArterialAtlasLables.txt
TERRITORY_LABELS = {
    1: "ACA-L",   # Anterior Cerebral Artery - Left
    2: "ACA-R",   # Anterior Cerebral Artery - Right
    3: "MCA-L",   # Middle Cerebral Artery - Left
    4: "MCA-R",   # Middle Cerebral Artery - Right
    5: "PCA-L",   # Posterior Cerebral Artery - Left
    6: "PCA-R",   # Posterior Cerebral Artery - Right
    7: "VB-L",    # Vertebrobasilar - Left
    8: "VB-R",    # Vertebrobasilar - Right
    9: "LV-L",    # Lateral Ventricle - Left (ignored)
    10: "LV-R",   # Lateral Ventricle - Right (ignored)
}

# Simplified major territory names (without laterality suffix for reporting)
MAJOR_TERRITORIES = {
    1: "ACA", 2: "ACA",
    3: "MCA", 4: "MCA",
    5: "PCA", 6: "PCA",
    7: "VB",  8: "VB",
}

_cached_atlas = None


def load_atlas():
    """
    Load the arterial territory atlas (Level 2 - Major Territories).
    Returns an ANTs image object.
    
    The atlas is in MNI152 space (182x218x182 mm^3, FSL compatible).
    """
    global _cached_atlas
    
    if _cached_atlas is not None:
        return _cached_atlas
    
    import ants
    
    if not ATLAS_LEVEL2_PATH.exists():
        raise FileNotFoundError(f"Atlas not found at {ATLAS_LEVEL2_PATH}")
    
    _cached_atlas = ants.image_read(str(ATLAS_LEVEL2_PATH))
    return _cached_atlas


def get_label_name(label_id: int, include_laterality: bool = True) -> str:
    """
    Get the territory name for a given label ID.
    
    Args:
        label_id: Integer label from atlas
        include_laterality: If True, returns "MCA-L", else returns "MCA"
    """
    if label_id == 0:
        return "Background"
    
    if include_laterality:
        return TERRITORY_LABELS.get(label_id, f"Unknown-{label_id}")
    else:
        return MAJOR_TERRITORIES.get(label_id, f"Unknown-{label_id}")
