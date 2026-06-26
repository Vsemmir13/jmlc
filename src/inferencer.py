"""Evaluation module for multilabel image classification."""
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, IterableDataset
from typing import Dict, List, Iterable, Any, Tuple, Optional
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from .dataset import MultilabelImageDataset
from .model import create_model
from .config import Config


import base64
from io import BytesIO
from PIL import Image

def base64_to_pil(base64_bytes):
    """
    Convert base64 bytes to PIL Image
    
    Args:
        base64_bytes: Base64 encoded bytes or bytearray
        
    Returns:
        PIL Image object
    """
    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_bytes)
    
    # Create BytesIO buffer from bytes
    image_buffer = BytesIO(image_bytes)
    
    # Open image with PIL
    image = Image.open(image_buffer).convert('RGB')
    
    return image


class MultilabelImageInferenceDataset(IterableDataset):
    """Iterable dataset for batch inference from in-memory rows."""

    def __init__(
        self,
        rows: List[Any],
        image_field: str,
        image_id: str,
        transform: Optional[transforms.Compose],
        decode_image=None,
    ):
        """
        Args:
            rows: Iterable of dict-like records with image payload and id
            image_field: Key for base64-encoded image bytes in each row
            image_id: Key for sample identifier in each row
            decode_image: Optional decoder(image_field value) -> PIL Image
        """
        self.rows = rows
        self.transform = transform
        self.image_field = image_field
        self.image_id = image_id
        self.decode_image = decode_image or base64_to_pil
    
    def __iter__(self) -> Tuple[str, torch.Tensor]:
        for row in self.rows:
            try:
                image = self.decode_image(row[self.image_field])
            except Exception as e:
                print(f"Error loading image {row[self.image_id]}: {e}")
                image = Image.new('RGB', (256, 256), color='black')

            image = self.transform(image)
            yield row[self.image_id], image


class MultilabelInferencer:
    """Evaluator for multilabel classification tasks."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        use_bf16: bool = False
    ):
        """
        Initialize evaluator.
        
        Args:
            model: PyTorch model
            device: Device to run evaluation on
            use_bf16: Whether to use bfloat16 mixed precision
        """
        self.model = model
        self.device = device
        self.use_bf16 = use_bf16
        self.model.eval()
    
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            data_loader: DataLoader for evaluation
                    
        Returns:
            Dictionary of metric names and values
        """
        
        all_preds = []
        all_paths = []
        
        for paths, images in tqdm(data_loader, desc='Eval...'):
            all_paths.extend(paths)
            images = images.to(self.device)
            
            # Forward pass with bfloat16 mixed precision if enabled
            if self.use_bf16:
                with autocast(dtype=torch.bfloat16):
                    preds = self.model(images)
            else:
                preds = self.model(images)

            all_preds.append(preds.float().cpu().numpy())
        
        # Concatenate all predictions and labels
        all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
        return all_paths, all_preds
    
def run_eval(
    rows,
    config,
    checkpoint_path,
    image_field='base64',
    image_id='image_id',
    device='cpu',
    batch_size=16,
    num_workers=0,
    prefetch_factor=1,
    ):
    config = Config(config)
    
    model = create_model(
        num_classes=config.get('data.num_classes'),
        model_name=config.get('model.name'),
        fc_activation=config.get('model.fc_activation'),
        pretrained=False,
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    inferencer = MultilabelInferencer(model=model, device=device, use_bf16=(device == 'cuda'))
    
    test_dataset = MultilabelImageInferenceDataset(
        rows=rows,
        image_field=image_field,
        image_id=image_id,
        transform=MultilabelImageDataset.eval_transform(config.get('data.resize'))
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )
    
    all_paths, all_preds = inferencer.evaluate(test_loader)
    return all_paths, all_preds




