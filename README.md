# MedGemma Stroke MRI Reporting Pipeline

An AI-assisted clinical reasoning tool using **MedGemma 1.5 4B** to generate radiologist-style text reports from ISLES2022 brain MRI data.

## ⚠️ Disclaimer
**NOT FOR CLINICAL USE.** This tool is a demonstration of AI-assisted reporting. It is not a medical device and should not be used for diagnosis or treatment decisions. All outputs must be verified by a board-certified radiologist.

## Features
- **Dataset Support**: Native support for ISLES2022 BIDS structure (DWI, ADC, FLAIR).
- **Intelligent Slice Selection**: Heuristic-based selection of most relevant slices from DWI.
- **ADC Confirmation Gate**: Validates "restricted diffusion" by cross-referencing DWI hyperintensity with ADC hypointensity.
- **MedGemma 1.5 Integration**: Uses Google's open weights medical VLM for image analysis.
- **Radiologist-Style Output**: Generates structured text reports with findings and impressions.

## Setup

1. **Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Hugging Face Auth**:
   MedGemma 1.5 is gated. You must accept the license at [huggingface.co/google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) and log in:
   ```bash
   huggingface-cli login
   ```

3. **Data**:
   Ensure ISLES2022 dataset is located at `/Users/rengarajanbashyam/Desktop/mristroke/ISLES-2022` (or set via `src/config.py`).

## Usage

Run the pipeline on a specific case:

```bash
python src/main.py --case sub-strokecase0001 --save
```

### Interactive Dashboard (New)
To scroll through images, toggle masks, and verify AI results interactively:
```bash
streamlit run src/app.py
```
This opens a web interface where you can:
- Select cases.
- Scroll slices synchronously.
- View Ground Truth overlays.
- Trigger AI generation.

Reports are saved to `reports/`.

## Architecture

- **`src/data`**: NIfTI loading (`nibabel`) and image preprocessing.
- **`src/model`**: MedGemma inference wrapper and prompt engineering.
- **`src/utils`**: Clinical logic checks (e.g., ADC confirmation).
- **`src/report`**: Output parsing and report templating.

## Scope & Limitations

### In Scope
- Qualitative infarct detection
- Stroke aging classification (Acute/Subacute/Chronic)
- Anatomical localization (Lobe/Hemisphere)
- DWI-FLAIR mismatch assessment

### Out of Scope (Anti-Goals)
- Segmentation masks (using heuristics instead)
- Quantitative volume measurements
- ASPECTS scoring
- Treatment recommendations (tPA/Thrombectomy)
