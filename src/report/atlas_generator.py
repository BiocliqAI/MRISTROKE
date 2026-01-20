"""
Atlas-Based Report Generator
Generates stroke reports using atlas-based territory and anatomical lookup.
MedGemma is only used for infarction detection and stage assessment.
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_atlas_report(
    case_id: str,
    vlm_findings: Dict[str, Any],  # From MedGemma (infarction_present, stage)
    atlas_territory: Dict[str, Any],  # From get_vascular_territory()
    atlas_location: Dict[str, Any],  # From get_anatomical_location()
    adc_confirmed: bool
) -> str:
    """
    Generates a formatted text report combining VLM detection with atlas-based localization.
    
    Args:
        case_id: Case identifier
        vlm_findings: Dict with 'infarction_present' (bool) and 'stage' (str)
        atlas_territory: Result from get_vascular_territory()
        atlas_location: Result from get_anatomical_location()
        adc_confirmed: Whether ADC confirmed restricted diffusion
    
    Returns:
        Formatted report string
    """
    findings_list = []
    
    # Extract atlas-based info
    territory = atlas_territory.get('primary_territory', 'Indeterminate')
    territory_pct = atlas_territory.get('overlap_percentage', 0)
    
    # Get anatomical locations with >5% involvement
    locations = [
        loc['name'] for loc in atlas_location.get('locations', [])
        if loc['percentage'] > 5 and loc['name'] not in ['Left Cerebral White Matter', 'Right Cerebral White Matter', 
                                                           'Left Cerebral Cortex', 'Right Cerebral Cortex']
    ]
    primary_location = atlas_location.get('primary_location', 'Unknown')
    
    # Filter generic labels
    if primary_location in ['Left Cerebral White Matter', 'Right Cerebral White Matter',
                            'Left Cerebral Cortex', 'Right Cerebral Cortex']:
        if locations:
            primary_location = locations[0]
        else:
            primary_location = "Subcortical/White Matter"
    
    # Determine laterality from territory
    if '-L' in territory:
        laterality = "Left"
    elif '-R' in territory:
        laterality = "Right"
    else:
        laterality = "Bilateral/Midline"
    
    # Major territory (without laterality)
    major_territory = territory.split('-')[0] if '-' in territory else territory
    
    # Construct Findings Section
    infarction_present = vlm_findings.get('infarction_present', False)
    stage = vlm_findings.get('stage', 'Acute')
    
    if infarction_present:
        location_str = ", ".join(locations[:3]) if locations else primary_location
        
        findings_list.append(f"• DIFFUSION: Abnormal high signal on DWI in the {laterality.lower()} {location_str.lower()}.")
        
        if adc_confirmed:
            findings_list.append("• ADC: Corresponding hypointensity confirms restricted diffusion.")
        else:
            findings_list.append("• ADC: No definite corresponding hypointensity (T2 shine-through cannot be excluded).")
        
        findings_list.append(f"• STAGE: Findings are consistent with {stage.lower()} evolution.")
        findings_list.append(f"• TERRITORY: {major_territory} vascular distribution ({territory}, {territory_pct:.0f}% overlap).")
        
    else:
        findings_list.append("• No definite areas of restricted diffusion identified.")
        findings_list.append("• Brain parenchyma demonstrates normal signal intensity.")
    
    # Site of Infarct Checklist
    sites = [
        "Frontal", "Parietal", "Temporal", "Occipital",
        "Brainstem", "Cerebellar", "Thalamus", "Basal Ganglia",
        "Insular", "Hippocampus"
    ]
    
    checklist_lines = []
    for i in range(0, len(sites), 4):
        row_sites = sites[i:i+4]
        row_items = []
        for site in row_sites:
            # Check if this site is in our atlas-detected locations
            is_checked = any(site.lower() in loc.lower() for loc in locations + [primary_location])
            marker = "[x]" if is_checked else "[ ]"
            row_items.append(f"{marker} {site:<12}")
        checklist_lines.append("  ".join(row_items))
    
    site_checklist = "\n".join(checklist_lines)
    
    findings_list.append("\nSITE OF INFARCT (Atlas-Based):")
    findings_list.append(site_checklist + "\n")
    
    findings_text = "\n".join(findings_list)
    
    # Construct Impression
    if infarction_present:
        impression = f"{stage} {laterality.lower()} hemispheric infarction in the {major_territory} territory"
        if locations:
            impression += f" involving the {', '.join(locations[:2]).lower()}"
        impression += "."
        if not adc_confirmed:
            impression += " (LIMITATION: ADC signals ambiguous; T2 shine-through possible)."
    else:
        impression = "No MRI evidence of acute ischemic stroke."
    
    # Disclaimer
    disclaimer = """
--------------------------------------------------------------------------------
AI DISCLAIMER: 
This report was generated by an AI system using atlas-based localization.
- Vascular Territory: Arterial Atlas (Johns Hopkins, 2022)
- Anatomical Location: Harvard-Oxford Atlas (FSL)
NOT FOR CLINICAL DIAGNOSIS. All findings must be verified by a radiologist.
--------------------------------------------------------------------------------
"""
    
    report = f"""
================================================================================
                    MRI BRAIN STROKE ANALYSIS PROTOCOL
================================================================================
CASE ID: {case_id}
SEQUENCES ANALYZED: Axial DWI, ADC, FLAIR
LOCALIZATION METHOD: Atlas-Based (ANTs Registration)

FINDINGS:
{findings_text}

IMPRESSION:
{impression}
{disclaimer}
"""
    return report


def generate_atlas_report_simple(
    case_id: str,
    is_positive: bool,
    stage: str,
    territory: str,
    locations: list,
    adc_confirmed: bool
) -> str:
    """
    Simplified report generation with pre-extracted values.
    """
    vlm_findings = {
        'infarction_present': is_positive,
        'stage': stage
    }
    
    # Determine laterality
    laterality = "Left" if "-L" in territory else ("Right" if "-R" in territory else "Bilateral")
    territory_pct = 100.0  # Placeholder
    
    atlas_territory = {
        'primary_territory': territory,
        'overlap_percentage': territory_pct
    }
    
    atlas_location = {
        'primary_location': locations[0] if locations else "Unknown",
        'locations': [{'name': loc, 'percentage': 20} for loc in locations]
    }
    
    return generate_atlas_report(case_id, vlm_findings, atlas_territory, atlas_location, adc_confirmed)
