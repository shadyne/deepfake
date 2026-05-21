import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.fftpack import dct


class ResidualNoiseExtractor:
    def __init__(self, kernel_size=5, sigma=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma
        print(f"Residual Noise Extractor (kernel={kernel_size}, sigma={sigma})")
    
    def extract(self, image):

        is_tensor = isinstance(image, torch.Tensor)
        
        if is_tensor:
            device = image.device
            image_np = image.cpu().numpy()
            
            # (C, H, W) -> (H, W, C)
            if image_np.ndim == 3 and image_np.shape[0] == 3:
                image_np = np.transpose(image_np, (1, 2, 0))
        else:
            image_np = image
        
        image_np = (image_np * 0.5) + 0.5  # [-1,1] -> [0,1]
        image_np = (image_np * 255).astype(np.uint8)  # [0,1] -> [0,255]
        
        # Apply Gaussian blur (low-pass filter)
        blurred = cv2.GaussianBlur(
            image_np, 
            (self.kernel_size, self.kernel_size), 
            self.sigma
        )
        
        # Residual = Original - Blurred
        residual = image_np.astype(np.float32) - blurred.astype(np.float32)
        
        residual = residual / 127.5  # [-255, 255] -> [-2, 2]
        residual = np.clip(residual, -1, 1)  # Clip ke [-1, 1]
        
        if is_tensor:
            # Convert back ke tensor
            if residual.ndim == 3:
                residual = np.transpose(residual, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            residual = torch.from_numpy(residual).float().to(device)
        
        return residual
    
    def extract_batch(self, batch):

        residuals = []
        
        for i in range(batch.size(0)):
            residual = self.extract(batch[i])
            residuals.append(residual)
        
        return torch.stack(residuals)


class DCTExtractor:
    
    def __init__(self, block_size=8):
        self.block_size = block_size
        print(f"DCT Extractor (block_size={block_size}x{block_size})")
    
    def extract_dct_block(self, block):
        """Ekstrak DCT dari satu block"""
        # Apply DCT 2D
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
        return dct_block
    
    def extract(self, image):

        is_tensor = isinstance(image, torch.Tensor)
        
        if is_tensor:
            device = image.device
            image_np = image.cpu().numpy()
            
            if image_np.ndim == 3 and image_np.shape[0] == 3:
                image_np = np.transpose(image_np, (1, 2, 0))
        else:
            image_np = image
        
        image_np = (image_np + 1) * 127.5  # [-1, 1] -> [0, 255]
        image_np = np.clip(image_np, 0, 255).astype(np.float32)
        
        H, W, C = image_np.shape
        
        # Padding agar bisa dibagi block_size
        pad_h = (self.block_size - H % self.block_size) % self.block_size
        pad_w = (self.block_size - W % self.block_size) % self.block_size
        
        if pad_h > 0 or pad_w > 0:
            image_np = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), 
                            mode='reflect')
        
        H_new, W_new, _ = image_np.shape
        
        # Proses per channel
        dct_channels = []
        
        for c in range(C):
            channel = image_np[:, :, c]
            dct_channel = np.zeros_like(channel)
            
            # Proses per block
            for i in range(0, H_new, self.block_size):
                for j in range(0, W_new, self.block_size):
                    block = channel[i:i+self.block_size, j:j+self.block_size]
                    dct_block = self.extract_dct_block(block)
                    dct_channel[i:i+self.block_size, j:j+self.block_size] = dct_block
            
            dct_channels.append(dct_channel)
        
        dct_image = np.stack(dct_channels, axis=-1)
        
        # Crop kembali ke ukuran asli
        dct_image = dct_image[:H, :W, :]
        

        p_low = np.percentile(dct_image, 1)
        p_high = np.percentile(dct_image, 99)
        
        # Scale ke [-1, 1]
        if p_high - p_low > 0:
            dct_image = 2 * (dct_image - p_low) / (p_high - p_low) - 1
            dct_image = np.clip(dct_image, -1, 1)
        else:
            dct_image = np.zeros_like(dct_image)
        
        if is_tensor:
            if dct_image.ndim == 3:
                dct_image = np.transpose(dct_image, (2, 0, 1))
            dct_image = torch.from_numpy(dct_image).float().to(device)
        
        return dct_image
    
    def extract_batch(self, batch):

        dct_results = []
        
        for i in range(batch.size(0)):
            dct_img = self.extract(batch[i])
            dct_results.append(dct_img)
        
        return torch.stack(dct_results)



def visualize_features(rgb, residual, dct, save_path):

    import matplotlib.pyplot as plt
    
    # Convert tensor ke numpy untuk visualisasi
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
        residual = residual.cpu().numpy()
        dct = dct.cpu().numpy()
    
    # Transpose jika perlu (C, H, W) -> (H, W, C)
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
        residual = np.transpose(residual, (1, 2, 0))
        dct = np.transpose(dct, (1, 2, 0))
    
    # Normalisasi untuk visualisasi
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    residual = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
    dct = (dct - dct.min()) / (dct.max() - dct.min() + 1e-8)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(residual)
    axes[1].set_title('Residual Noise (Spatial)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(dct)
    axes[2].set_title('DCT Coefficients', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature visualization saved: {save_path}")


def get_feature_extractor(method, config):
    """Factory function untuk feature extractor"""
    
    if method == 'baseline':
        return None
    
    elif method == 'residual_spatial':
        return ResidualNoiseExtractor(
            kernel_size=config.RESIDUAL_SPATIAL_KERNEL_SIZE,
            sigma=config.RESIDUAL_SPATIAL_SIGMA
        )
    
    elif method == 'residual_dct':
        # DCT diterapkan pada residual
        residual_extractor = ResidualNoiseExtractor(
            kernel_size=config.RESIDUAL_SPATIAL_KERNEL_SIZE,
            sigma=config.RESIDUAL_SPATIAL_SIGMA
        )
        dct_extractor = DCTExtractor(block_size=config.DCT_BLOCK_SIZE)
        
        return {
            'residual': residual_extractor,
            'dct': dct_extractor
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")