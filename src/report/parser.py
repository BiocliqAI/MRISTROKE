import re
from typing import Dict, Any, List

def parse_medgemma_output(text: str) -> Dict[str, Any]:
    """
    Parses the structured text response from MedGemma into structured fields.
    Handles both markdown-formatted numbered lists and plain text.
    """
    parsed = {
        "raw_text": text,
        "infarction_present": False,
        "restricted_diffusion": False,
        "Stage": "Unknown",
        "Laterality": "Unknown",
        "location": [],
        "Territory": "Unknown",
        "limitations": []
    }

    text_lower = text.lower()

    # --- Structured Field Extraction (for numbered lists) ---
    
    # Stage: Look for "**Stage**:" pattern
    stage_match = re.search(r'\*\*Stage\*\*:\s*([^\n\*]+)', text, re.IGNORECASE)
    if stage_match:
        stage_val = stage_match.group(1).strip().rstrip('.')
        if "acute" in stage_val.lower():
            parsed["Stage"] = "Acute"
        elif "subacute" in stage_val.lower():
            parsed["Stage"] = "Subacute"
        elif "chronic" in stage_val.lower():
            parsed["Stage"] = "Chronic"
    
    # Laterality: Look for "**Laterality**:" pattern
    lat_match = re.search(r'\*\*Laterality\*\*:\s*([^\n\*]+)', text, re.IGNORECASE)
    if lat_match:
        lat_val = lat_match.group(1).strip().rstrip('.')
        if "left" in lat_val.lower() and "right" not in lat_val.lower():
            parsed["Laterality"] = "Left"
        elif "right" in lat_val.lower() and "left" not in lat_val.lower():
            parsed["Laterality"] = "Right"
        elif "bilateral" in lat_val.lower() or ("left" in lat_val.lower() and "right" in lat_val.lower()):
            parsed["Laterality"] = "Bilateral"
    
    # Territory: Look for "**Vascular Territory**:" pattern
    terr_match = re.search(r'\*\*Vascular Territory\*\*:\s*([^\n\*]+)', text, re.IGNORECASE)
    if terr_match:
        terr_val = terr_match.group(1).strip().rstrip('.')
        # Normalize common territory names
        if "mca" in terr_val.lower() or "middle cerebral" in terr_val.lower():
            parsed["Territory"] = "MCA"
        elif "aca" in terr_val.lower() or "anterior cerebral" in terr_val.lower():
            parsed["Territory"] = "ACA"
        elif "pca" in terr_val.lower() or "posterior cerebral" in terr_val.lower():
            parsed["Territory"] = "PCA"
        elif "thalamic" in terr_val.lower():
            parsed["Territory"] = "PCA (Thalamic)"
        elif "watershed" in terr_val.lower():
            parsed["Territory"] = "Watershed"
        elif "lacunar" in terr_val.lower():
            parsed["Territory"] = "Lacunar"
        else:
            parsed["Territory"] = terr_val  # Use raw value if no match

    # --- Fallback to keyword matching if structured extraction fails ---
    
    # 1. Infarction Presence
    if "no acute" in text_lower or "unremarkable" in text_lower or "no evidence of" in text_lower:
        # Check if there's a conflicting positive finding
        if "infarction" in text_lower or "infarct" in text_lower:
            parsed["infarction_present"] = True  # Positive signal overrides negative header
        else:
            parsed["infarction_present"] = False
    elif "infarction" in text_lower or "stroke" in text_lower or "restricted diffusion" in text_lower or "infarct" in text_lower:
        parsed["infarction_present"] = True

    # 2. Restricted Diffusion
    if "restricted diffusion" in text_lower or ("hypointens" in text_lower and "adc" in text_lower):
        parsed["restricted_diffusion"] = True

    # 3. Stage fallback (if structured extraction failed)
    if parsed["Stage"] == "Unknown":
        if "acute" in text_lower:
            parsed["Stage"] = "Acute"
        elif "subacute" in text_lower:
            parsed["Stage"] = "Subacute"
        elif "chronic" in text_lower:
            parsed["Stage"] = "Chronic"

    # 4. Laterality fallback
    if parsed["Laterality"] == "Unknown":
        if "left" in text_lower and "right" not in text_lower:
            parsed["Laterality"] = "Left"
        elif "right" in text_lower and "left" not in text_lower:
            parsed["Laterality"] = "Right"
        elif "bilateral" in text_lower or ("left" in text_lower and "right" in text_lower):
            parsed["Laterality"] = "Bilateral"

    # 5. Location Extraction
    locations = [
        "frontal", "temporal", "parietal", "occipital", "insula",
        "basal ganglia", "thalamus", "caudate", "putamen", "capsule",
        "cerebellum", "brainstem", "pons", "medulla", "midbrain"
    ]
    found_locs = [loc.title() for loc in locations if loc in text_lower]
    parsed["location"] = found_locs if found_locs else ["Unspecified"]

    # 6. Territory fallback
    if parsed["Territory"] == "Unknown":
        territories = ["mca", "aca", "pca", "watershed", "lacunar"]
        found_terr = [t.upper() for t in territories if t in text_lower]
        parsed["Territory"] = ", ".join(found_terr) if found_terr else "Indeterminate"

    return parsed
