import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional
import textwrap

class Visualizer:
    @staticmethod
    def create_overlay(background: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Creates an RGB overlay of a binary mask (red) on a grayscale background.
        """
        # Normalize background to 0-1
        bg_norm = (background - np.min(background)) / (np.max(background) - np.min(background) + 1e-8)
        bg_rgb = np.stack([bg_norm]*3, axis=-1)
        
        if mask is None:
            return bg_rgb
            
        # Create red mask
        # Mask is usually 0 and 1
        mask_binary = mask > 0.5
        
        overlay = bg_rgb.copy()
        # Set red channel to 1 where mask is present, blend other channels
        # Simple blending:
        # OUT = (1-alpha)*BG + alpha*FG
        
        # Red Color: [1, 0, 0]
        
        red_channel = overlay[:, :, 0]
        green_channel = overlay[:, :, 1]
        blue_channel = overlay[:, :, 2]
        
        red_channel[mask_binary] = (1 - alpha) * red_channel[mask_binary] + alpha * 1.0
        green_channel[mask_binary] = (1 - alpha) * green_channel[mask_binary] + alpha * 0.0
        blue_channel[mask_binary] = (1 - alpha) * blue_channel[mask_binary] + alpha * 0.0
        
        overlay[:, :, 0] = red_channel
        overlay[:, :, 1] = green_channel
        overlay[:, :, 2] = blue_channel
        
        return overlay

    @staticmethod
    def save_report_image(
        case_id: str,
        report_text: str,
        dwi_slice: np.ndarray,
        adc_slice: np.ndarray,
        flair_slice: np.ndarray,
        mask_slice: Optional[np.ndarray],
        output_path: Path
    ):
        """
        Creates a composite image with the report text and selected visualized slices.
        """
        # Portrait Layout: Images Top, Text Bottom
        fig = plt.figure(figsize=(12, 16), constrained_layout=True)
        gs = fig.add_gridspec(6, 3) # 6 rows, 3 columns
        
        # Image Row (Row 0)
        ax_dwi = fig.add_subplot(gs[0, 0])
        ax_dwi.imshow(dwi_slice.T, cmap='gray', origin='lower')
        ax_dwi.set_title("DWI")
        ax_dwi.axis('off')

        ax_adc = fig.add_subplot(gs[0, 1])
        ax_adc.imshow(adc_slice.T, cmap='gray', origin='lower')
        ax_adc.set_title("ADC")
        ax_adc.axis('off')

        ax_flair = fig.add_subplot(gs[0, 2])
        ax_flair.imshow(flair_slice.T, cmap='gray', origin='lower')
        ax_flair.set_title("FLAIR")
        ax_flair.axis('off')
        
        # Report Text Panel (Rows 1-5, spanning all columns)
        ax_text = fig.add_subplot(gs[1:, :])
        ax_text.axis('off')
        
        # Add text
        ax_text.text(0.01, 0.98, report_text, transform=ax_text.transAxes, fontsize=11, 
                     verticalalignment='top', family='monospace')
        
        plt.suptitle(f"AI Analysis Case: {case_id}", fontsize=16)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
