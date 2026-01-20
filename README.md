# MRI Stroke Analysis Pipeline

An AI-powered clinical decision support system for stroke MRI analysis, combining **nnU-Net segmentation**, **deterministic staging**, **atlas-based localization**, and **MedGemma** for natural language report generation.

## ⚠️ Disclaimer
**NOT FOR CLINICAL USE.** This is a research tool for AI-assisted reporting. All outputs must be verified by a board-certified radiologist.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│               INPUT: DWI, ADC, FLAIR                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│           nnU-Net 3D Segmentation                           │
│           (Trained on ISLES-2022, Dice ~0.82)               │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │ Deterministic│ │ Atlas-Based  │ │ Rule-Based   │
   │ Staging      │ │ Localization │ │ Safety Checks│
   │ (ADC/FLAIR)  │ │ (Territory)  │ │              │
   └──────────────┘ └──────────────┘ └──────────────┘
          │               │               │
          └───────────────┴───────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│           MedGemma 1.5 Report Generation                    │
│           (Text-only, grounded to deterministic findings)   │
└─────────────────────────────────────────────────────────────┘
```

## Features

| Component | Description |
|:----------|:------------|
| **nnU-Net Segmentation** | 3D full-resolution segmentation (ResidualEncoderUNet) |
| **Deterministic Staging** | ADC/FLAIR ratio-based acute/subacute/chronic classification |
| **Territory Localization** | Johns Hopkins Arterial Atlas (MCA, ACA, PCA, VB) |
| **Anatomical Location** | Harvard-Oxford Atlas (lobe, subcortical structures) |
| **Safety Checks** | Rule-based DWI/ADC mismatch detection for uncertain cases |
| **Report Generation** | MedGemma VLM for natural language reports |

## Validation Results (50 cases)

| Metric | Value |
|:-------|:------|
| Mean Dice | 0.78 |
| Median Dice | 0.82 |
| Detection Rate | 98% |

## Setup

1. **Environment**:
   ```bash
   python3.11 -m venv .venv311
   source .venv311/bin/activate
   pip install -r requirements.txt
   pip install nnunetv2  # For segmentation
   ```

2. **Hugging Face Auth** (for MedGemma):
   ```bash
   huggingface-cli login
   ```

3. **Model Checkpoint**:
   Place `checkpoint_best.pth` (nnU-Net trained model) in `src/model/`

4. **Arterial Atlas** (for vascular territory mapping):
   
   Download the Johns Hopkins Arterial Atlas:
   ```bash
   # Create directory
   mkdir -p data/atlases/arterial_atlas/data
   
   # Download from Johns Hopkins (requires registration)
   # URL: https://www.nitrc.org/projects/arterialatlas
   # Or use direct link if available:
   # wget -O data/atlases/arterial_atlas/data/ArterialAtlas.nii https://...
   ```
   
   Required files in `data/atlases/arterial_atlas/data/`:
   - `ArterialAtlas.nii` (30 sub-territories)
   - `ArterialAtlas_level2.nii` (4 major territories)
   - `ArterialAtlasLabels.txt`
   
   > **Note**: Harvard-Oxford atlas (for anatomical location) is auto-downloaded by `nilearn`.

5. **Data**:
   Set `ISLES2022_PATH` in `src/config.py` or use environment variable.


## Usage

### Full Pipeline
```bash
python src/pipeline_deterministic.py --case sub-strokecase0001
```

### Interactive Dashboard
```bash
streamlit run src/app.py
```

## Modules

| Module | Purpose |
|:-------|:--------|
| `src/model/segmentation.py` | nnU-Net inference wrapper |
| `src/model/medgemma.py` | MedGemma VLM for reports |
| `src/staging/deterministic.py` | ADC/FLAIR ratio staging |
| `src/territory/` | Atlas-based localization |
| `src/safety/rule_checks.py` | DWI/ADC safety checks |
| `src/report/` | Report templating |

## Scope & Limitations

### In Scope
- ✅ Lesion detection via 3D segmentation
- ✅ Stroke staging (Acute/Subacute/Chronic)
- ✅ Vascular territory mapping
- ✅ Anatomical localization
- ✅ Safety checks for small/missed lesions

### Out of Scope
- ❌ Quantitative volume measurements
- ❌ ASPECTS scoring
- ❌ Treatment recommendations

## License
Research use only. Not for clinical diagnosis.
