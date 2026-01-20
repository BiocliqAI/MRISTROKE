"""
nnU-Net Segmentation Model Inference Wrapper
Loads the trained nnU-Net model and performs lesion segmentation.
"""
import torch
import numpy as np
import nibabel as nib
import logging
from pathlib import Path
from typing import Tuple, Optional, Union
import os

# Suppress nnU-Net's verbose output
os.environ['nnUNet_compile'] = 'F'

logger = logging.getLogger(__name__)

# Path to checkpoint
CHECKPOINT_PATH = Path(__file__).parent / "checkpoint_best.pth"


class NNUNetPredictor:
    """
    Wrapper for nnU-Net inference using the trained checkpoint.
    """
    
    def __init__(self, checkpoint_path: Union[str, Path] = None, device: str = None):
        """
        Initialize the predictor.
        
        Args:
            checkpoint_path: Path to checkpoint_best.pth
            device: 'cpu', 'cuda', or 'mps'. Auto-detected if None.
        """
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Loading nnU-Net model from {self.checkpoint_path} on {self.device}")
        
        # Load checkpoint
        self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract configuration
        self.init_args = self.checkpoint['init_args']
        self.plans = self.init_args['plans']
        self.dataset_json = self.init_args['dataset_json']
        self.configuration = self.init_args['configuration']
        
        # Build network
        self._build_network()
        
        logger.info("nnU-Net model loaded successfully")
    
    def _build_network(self):
        """Build the network architecture from plans."""
        from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
        
        # Get architecture config from plans
        config = self.plans['configurations'][self.configuration]
        arch = config.get('architecture', {})
        arch_kwargs = arch.get('arch_kwargs', {})
        
        # Get number of input channels and classes
        num_input_channels = len(self.dataset_json.get('channel_names', {0: 'MR'}))
        num_classes = len(self.dataset_json.get('labels', {'background': 0, 'Stroke': 1}))
        
        # Build the network using exact kwargs from checkpoint
        self.network = ResidualEncoderUNet(
            input_channels=num_input_channels,
            n_stages=arch_kwargs.get('n_stages', 5),
            features_per_stage=arch_kwargs.get('features_per_stage', [32, 64, 128, 256, 320]),
            conv_op=torch.nn.Conv3d,
            kernel_sizes=arch_kwargs.get('kernel_sizes', [[3, 3, 3]] * 5),
            strides=arch_kwargs.get('strides', [[1, 1, 1]] + [[2, 2, 2]] * 4),
            n_blocks_per_stage=arch_kwargs.get('n_blocks_per_stage', [1, 3, 4, 6, 6]),
            n_conv_per_stage_decoder=arch_kwargs.get('n_conv_per_stage_decoder', [1, 1, 1, 1]),
            conv_bias=arch_kwargs.get('conv_bias', True),
            norm_op=torch.nn.InstanceNorm3d,
            norm_op_kwargs=arch_kwargs.get('norm_op_kwargs', {'eps': 1e-5, 'affine': True}),
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs=arch_kwargs.get('nonlin_kwargs', {'inplace': True}),
            deep_supervision=True,
            num_classes=num_classes,
        )
        
        # Load weights
        self.network.load_state_dict(self.checkpoint['network_weights'])
        self.network.to(self.device)
        self.network.eval()
        
        # Store preprocessing params
        self.patch_size = config.get('patch_size', [80, 112, 112])
        self.target_spacing = config.get('spacing', [2.0, 2.0, 2.0])
        
        # Store foreground intensity properties for normalization (from training)
        self.intensity_properties = self.plans.get('foreground_intensity_properties_per_channel', {})
    
    def resample_volume(self, volume: np.ndarray, original_spacing: tuple, 
                        target_spacing: tuple, order: int = 3) -> np.ndarray:
        """
        Resample volume to target spacing.
        
        Args:
            volume: Input volume
            original_spacing: Current spacing (z, y, x) or (x, y, z)
            target_spacing: Target spacing
            order: Interpolation order (3 for data, 0/1 for masks)
        
        Returns:
            Resampled volume
        """
        from scipy.ndimage import zoom
        
        # Calculate zoom factors
        zoom_factors = [o / t for o, t in zip(original_spacing, target_spacing)]
        
        # Resample
        resampled = zoom(volume, zoom_factors, order=order)
        
        return resampled
    
    def preprocess(self, dwi: np.ndarray, adc: np.ndarray, flair: np.ndarray,
                   original_spacing: tuple = None) -> Tuple[torch.Tensor, tuple]:
        """
        Preprocess input volumes for the network with proper nnU-Net preprocessing.
        
        Preprocessing steps:
        1. Resample to target spacing (2mm isotropic)
        2. ZScore normalization per channel
        
        Args:
            dwi: DWI volume (H, W, D)
            adc: ADC volume (H, W, D)
            flair: FLAIR volume (H, W, D)
            original_spacing: Original voxel spacing. If None, assumes same as target.
        
        Returns:
            Tuple of (Preprocessed tensor (1, 3, H, W, D), original_shape)
        """
        original_shape = dwi.shape
        
        # 1. Resample to target spacing if needed
        if original_spacing is not None:
            dwi = self.resample_volume(dwi, original_spacing, self.target_spacing, order=3)
            adc = self.resample_volume(adc, original_spacing, self.target_spacing, order=3)
            flair = self.resample_volume(flair, original_spacing, self.target_spacing, order=3)
        
        # Stack channels: order matches training (DWI=0, ADC=1, FLAIR=2)
        data = np.stack([dwi, adc, flair], axis=0)  # (3, H, W, D)
        
        # 2. ZScore normalization per channel
        for c in range(data.shape[0]):
            channel_data = data[c]
            
            # Use foreground-only statistics for normalization (nnU-Net default)
            # Create mask of non-zero voxels (approximate foreground)
            mask = channel_data > np.percentile(channel_data, 1)
            
            if np.sum(mask) > 0:
                mean = np.mean(channel_data[mask])
                std = np.std(channel_data[mask])
            else:
                mean = np.mean(channel_data)
                std = np.std(channel_data)
            
            if std > 0:
                data[c] = (channel_data - mean) / std
            else:
                data[c] = channel_data - mean
        
        # Convert to tensor
        data = torch.from_numpy(data.astype(np.float32))
        data = data.unsqueeze(0)  # (1, 3, H, W, D)
        
        return data, original_shape
    
    def sliding_window_inference(self, data: torch.Tensor) -> torch.Tensor:
        """
        Perform sliding window inference for large volumes.
        
        Args:
            data: Input tensor (1, C, H, W, D)
        
        Returns:
            Segmentation probabilities (1, num_classes, H, W, D)
        """
        from torch.nn.functional import pad
        
        data = data.to(self.device)
        _, C, H, W, D = data.shape
        pH, pW, pD = self.patch_size
        
        # Pad if needed
        pad_h = max(0, pH - H)
        pad_w = max(0, pW - W)
        pad_d = max(0, pD - D)
        
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            data = pad(data, (0, pad_d, 0, pad_w, 0, pad_h), mode='constant', value=0)
            _, _, H, W, D = data.shape
        
        # Initialize output
        output = torch.zeros((1, 2, H, W, D), device=self.device)
        count = torch.zeros((1, 1, H, W, D), device=self.device)
        
        # Sliding window with 50% overlap
        step_h = pH // 2
        step_w = pW // 2
        step_d = pD // 2
        
        for h in range(0, H - pH + 1, step_h):
            for w in range(0, W - pW + 1, step_w):
                for d in range(0, D - pD + 1, step_d):
                    patch = data[:, :, h:h+pH, w:w+pW, d:d+pD]
                    
                    with torch.no_grad():
                        pred = self.network(patch)
                        if isinstance(pred, (list, tuple)):
                            pred = pred[0]  # Deep supervision - use highest resolution
                    
                    output[:, :, h:h+pH, w:w+pW, d:d+pD] += pred
                    count[:, :, h:h+pH, w:w+pW, d:d+pD] += 1
        
        # Average overlapping predictions
        output = output / count.clamp(min=1)
        
        # Remove padding
        output = output[:, :, :H - pad_h, :W - pad_w, :D - pad_d]
        
        return output
    
    def predict(self, dwi: np.ndarray, adc: np.ndarray, flair: np.ndarray, 
                original_spacing: tuple = None, threshold: float = 0.5) -> np.ndarray:
        """
        Predict lesion segmentation.
        
        Args:
            dwi: DWI volume (H, W, D)
            adc: ADC volume (H, W, D)  
            flair: FLAIR volume (H, W, D)
            original_spacing: Original voxel spacing (optional, for resampling)
            threshold: Probability threshold for binary mask
        
        Returns:
            Binary segmentation mask (H, W, D) in original resolution
        """
        # Preprocess (includes resampling if spacing provided)
        data, original_shape = self.preprocess(dwi, adc, flair, original_spacing)
        
        # Inference
        with torch.no_grad():
            probs = self.sliding_window_inference(data)
            probs = torch.softmax(probs, dim=1)
        
        # Get stroke class (class 1)
        stroke_prob = probs[0, 1].cpu().numpy()
        
        # Threshold
        mask = (stroke_prob > threshold).astype(np.uint8)
        
        # Resample back to original resolution if needed
        if original_spacing is not None and mask.shape != original_shape:
            mask = self.resample_volume(
                mask.astype(np.float32), 
                self.target_spacing, 
                original_spacing, 
                order=0  # Nearest neighbor for masks
            ).astype(np.uint8)
            
            # Crop/pad to exact original shape if needed
            if mask.shape != original_shape:
                result = np.zeros(original_shape, dtype=np.uint8)
                slices = tuple(slice(0, min(m, o)) for m, o in zip(mask.shape, original_shape))
                result[slices] = mask[slices]
                mask = result
        
        return mask
    
    def predict_from_case(self, case, use_spacing: bool = True) -> np.ndarray:
        """
        Predict from a loaded Case object.
        
        Args:
            case: Case object with dwi_volume, adc_volume, flair_volume
            use_spacing: Whether to use case spacing for resampling
        
        Returns:
            Binary segmentation mask
        """
        # Get spacing from case if available
        original_spacing = None
        if use_spacing and hasattr(case, 'spacing') and case.spacing is not None:
            original_spacing = case.spacing
        
        return self.predict(
            case.dwi_volume, 
            case.adc_volume, 
            case.flair_volume,
            original_spacing=original_spacing
        )


# Singleton instance
_predictor = None

def get_predictor() -> NNUNetPredictor:
    """Get or create the singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = NNUNetPredictor()
    return _predictor
