import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from typing import List, Dict, Any
import logging

from src.config import MODEL_ID, DEVICE, TORCH_DTYPE, DEFAULT_TEMPERATURE, MAX_NEW_TOKENS
from src.model.prompts import STROKE_ANALYSIS_SYSTEM_PROMPT, STROKE_ANALYSIS_USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class MedGemmaPredictor:
    def __init__(self):
        self.device = torch.device(DEVICE)  # Auto-detected in config.py
        self.dtype = getattr(torch, TORCH_DTYPE)
        
        logger.info(f"Loading MedGemma {MODEL_ID} on {self.device} with {self.dtype}...")
        
        try:
            self.processor = AutoProcessor.from_pretrained(MODEL_ID)
            self.model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                torch_dtype=self.dtype,
                device_map=self.device,
            )
            self.model.eval()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, 
                dwi_images: List[Image.Image], 
                adc_image: Image.Image, 
                flair_image: Image.Image, 
                adc_confirmed: bool) -> str:
        """
        Runs inference on a set of images for a single case.
        """
        # Prepare inputs
        images = dwi_images + [adc_image, flair_image]
        num_dwi = len(dwi_images)
        idx_adc = num_dwi + 1
        idx_flair = num_dwi + 2
        
        # Format prompt
        prompt_text = STROKE_ANALYSIS_USER_PROMPT_TEMPLATE.format(
            num_dwi=num_dwi,
            idx_adc=idx_adc,
            idx_flair=idx_flair,
            adc_confirmed="Confirmed (Restricted Diffusion present)" if adc_confirmed else "NOT Confirmed (Use caution)"
        )

        # Construct chat messages
        # MedGemma 1.5 expects interleaved image/text content
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        
        content.append({"type": "text", "text": prompt_text})

        messages = [
            #{"role": "system", "content": [{"type": "text", "text": STROKE_ANALYSIS_SYSTEM_PROMPT}]}, # System prompt might not be supported in all templates
            {"role": "user", "content": content}
        ]

        logger.info("Running inference...")
        
        # Process inputs
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False  # Deterministic output for consistency
            )
            
        # Decode
        # We need to slice off the input tokens to get just the new text
        input_len = inputs["input_ids"].shape[-1]
        generated_text_ids = generated_ids[0][input_len:]
        response = self.processor.decode(generated_text_ids, skip_special_tokens=True)
        
        return response

    def generate_text_only(self, prompt: str) -> str:
        """
        Generate text response without images (for grounded report writing).
        """
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )
        
        input_len = inputs["input_ids"].shape[-1]
        generated_text_ids = generated_ids[0][input_len:]
        response = self.processor.decode(generated_text_ids, skip_special_tokens=True)
        
        return response
