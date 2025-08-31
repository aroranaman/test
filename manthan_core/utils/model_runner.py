# manthan_core/utils/model_runner.py
import torch
from torchvision import models
import numpy as np
import rasterio
from pathlib import Path

def run_resnet_inference(model_path: Path, image_path: Path, num_classes: int) -> np.ndarray:
    """Loads a trained ResNet model and runs inference on a GeoTIFF."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = models.resnet34()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # This is a simplified inference loop for demonstration
    # A full implementation would handle tiling for large images
    with rasterio.open(image_path) as src:
        image = src.read()
    
    # Placeholder for a real prediction
    prediction = np.random.randint(0, num_classes, (image.shape[1], image.shape[2]), dtype=np.uint8)
    return prediction

def run_unet_inference(model_path: Path, image_path: Path, num_classes: int) -> np.ndarray:
    """Loads a trained U-Net model and runs inference."""
    # This would contain the U-Net specific loading and inference logic
    # For now, it returns a placeholder result
    with rasterio.open(image_path) as src:
        image = src.read()
    prediction = np.random.randint(0, num_classes, (image.shape[1], image.shape[2]), dtype=np.uint8)
    return prediction