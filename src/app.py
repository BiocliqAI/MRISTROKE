import streamlit as st
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import DATASET_ROOT, SLICES_PER_CASE
from src.data.loader import DataLoader
from src.data.preprocessor import ImagePreprocessor
from src.utils.adc_gate import check_adc_confirmation
from src.model.medgemma import MedGemmaPredictor
from src.report.parser import parse_medgemma_output
from src.report.generator import generate_report
from src.report.visualizer import Visualizer

st.set_page_config(layout="wide", page_title="MedGemma Stroke Analyst")

@st.cache_resource
def load_loader():
    return DataLoader()

@st.cache_resource
def load_model():
    return MedGemmaPredictor()

def main():
    st.title("MedGemma 1.5 Stroke Validation Dashboard")
    
    loader = load_loader()
    
    # Sidebar: Case Selection
    data_root = Path(DATASET_ROOT)
    cases = sorted([d.name for d in data_root.iterdir() if d.is_dir() and d.name.startswith("sub-strokecase")])
    if not cases:
        st.error(f"No cases found in {DATASET_ROOT}")
        return

    selected_case_id = st.sidebar.selectbox("Select Case", cases)
    
    try:
        case = loader.load_case(selected_case_id)
    except Exception as e:
        st.error(f"Error loading case: {e}")
        return

    # Slice Selection Logic (Cached for UI speed)
    dwi_slice_indices = ImagePreprocessor.select_slices(case.dwi_volume, top_n=SLICES_PER_CASE)
    primary_ai_slice = dwi_slice_indices[len(dwi_slice_indices)//2]
    
    # Main Layout
    col_viz, col_report = st.columns([2, 1])
    
    with col_viz:
        st.subheader("Image Viewer")
        
        # Slice Slider
        max_slice = case.dwi_volume.shape[2] - 1
        default_slice = primary_ai_slice # Default to AI's top pick
        slice_idx = st.slider("Axial Slice Index", 0, max_slice, default_slice)
        
        # Display AI Selection Markers
        st.caption(f"AI Selected Slices: {dwi_slice_indices}")
        
        # Sync Views
        # Prepare arrays
        dwi_slice = case.dwi_volume[:, :, slice_idx]
        adc_slice = case.adc_volume[:, :, slice_idx]
        flair_slice = case.flair_volume[:, :, slice_idx]
        
        # Options
        show_mask = st.checkbox("Show Ground Truth Mask (Red Overlay)", value=True)
        
        # Create Plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # DWI + Mask
        axes[0].set_title("DWI (b=1000)")
        if show_mask and case.mask_volume is not None:
            mask_slice = case.mask_volume[:, :, slice_idx]
            overlay = Visualizer.create_overlay(dwi_slice.T, mask_slice.T, alpha=0.4)
            axes[0].imshow(overlay, origin='lower')
        else:
            axes[0].imshow(dwi_slice.T, cmap='gray', origin='lower')
        axes[0].axis('off')
        
        # ADC
        axes[1].set_title("ADC Map")
        axes[1].imshow(adc_slice.T, cmap='gray', origin='lower')
        axes[1].axis('off')

        # FLAIR
        axes[2].set_title("FLAIR")
        axes[2].imshow(flair_slice.T, cmap='gray', origin='lower')
        axes[2].axis('off')
        
        st.pyplot(fig)
        
        # Slice Level logic check
        st.markdown("---")
        st.subheader("Slice Analysis")
        adc_check = check_adc_confirmation(dwi_slice, adc_slice)
        st.info(f"Restricted Diffusion Check (DWI > 90% & ADC < 10%): **{adc_check}**")

    with col_report:
        st.subheader("AI Analysis")
        
        if st.button("Generate MedGemma Report"):
            with st.spinner("Running Inference (may take ~10s)..."):
                try:
                    # Prepare inputs exactly as pipeline does
                    dwi_images = []
                    for idx in dwi_slice_indices:
                        dwi_images.append(ImagePreprocessor.process_slice(case.dwi_volume[:, :, idx]))
                    
                    # AI uses the text-selected "primary" slice for ADC/FLAIR context
                    adc_img = ImagePreprocessor.process_slice(case.adc_volume[:, :, primary_ai_slice])
                    flair_img = ImagePreprocessor.process_slice(case.flair_volume[:, :, primary_ai_slice])
                    
                    # ADC Gate on primary slice
                    adc_confirmed = check_adc_confirmation(
                        case.dwi_volume[:, :, primary_ai_slice],
                        case.adc_volume[:, :, primary_ai_slice]
                    )
                    
                    predictor = load_model()
                    raw_response = predictor.predict(dwi_images, adc_img, flair_img, adc_confirmed)
                    parsed_data = parse_medgemma_output(raw_response)
                    final_report = generate_report(selected_case_id, parsed_data, adc_confirmed)
                    
                    st.success("Analysis Complete")
                    st.text_area("Generated Report", final_report, height=400)
                    
                    with st.expander("Show Raw Model Output"):
                        st.write(raw_response)
                        
                except Exception as e:
                    st.error(f"Inference Failed: {e}")

if __name__ == "__main__":
    main()
