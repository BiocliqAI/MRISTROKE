import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = os.environ.get("ISLES2022_PATH", "/Users/rengarajanbashyam/Desktop/mristroke/ISLES-2022")
OUTPUT_DIR = PROJECT_ROOT / "reports"

# Model Settings
MODEL_ID = "google/medgemma-1.5-4b-it" 
DEVICE = "mps"  # Metal Performance Shaders for Apple Silicon
USE_QUANTIZATION = False # 4-bit quantization can be unstable on MPS, defaulting to full/16-bit
TORCH_DTYPE = "bfloat16" # Recommended for Gemma on MPS

# Data Processing
SLICES_PER_CASE = 3 # Top N slices to select
MAX_IMAGES_INPUT = 4 # Max images to send to MedGemma (Constraint: context window & performance)
IMAGE_SIZE = (896, 896) # MedGemma expects 896x896 or similar high-res
NORMALIZE_PERCENTILE = 99.5 # Robust scaling

# ADC Gating
DWI_HIGH_PERCENTILE = 90
ADC_LOW_PERCENTILE = 10
ADC_CONFIRMATION_THRESHOLD_RATIO = 1.5 # DWI intensity / ADC intensity heuristic (simplified)

# Generation
DEFAULT_TEMPERATURE = 0.2
MAX_NEW_TOKENS = 1024
