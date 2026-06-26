"""Dataset module for multilabel image classification."""
import os
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from tqdm import tqdm

DEFAULT_RESIZE = 480


class MultilabelImageDataset(Dataset):
    """Dataset class for multilabel image classification."""

    def __init__(
        self,
        data_path: str,
        transform: Optional[transforms.Compose] = None,
        num_classes: Optional[int] = None,
        resize: int = DEFAULT_RESIZE,
        train: bool = False,
    ):
        """
        Initialize multilabel image dataset.

        Args:
            data_path: Path to directory containing images
            transform: Optional transform to apply to images
            num_classes: Number of classes/labels
            resize: Image size (height and width) for transforms
            train: If True and transform is None, use training augmentations
        """
        self.data_path = Path(data_path)
        self.resize = resize
        if transform is None:
            transform = (
                self._get_train_transform() if train else self._get_default_transform()
            )
        self.transform = transform
        self.num_classes = num_classes

        # Load image paths and labels
        if Path(data_path).exists():
            self.image_paths, self.labels = self._load_from_csv(data_path)

        if self.num_classes is None:
            self.num_classes = len(self.labels[0]) if len(self.labels) > 0 else 1

    def _get_train_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(
                (self.resize, self.resize),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomRotation(10),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
            transforms.RandomChoice([
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),
                transforms.RandomAutocontrast(p=0.1),
            ]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

    @classmethod
    def eval_transform(cls, resize) -> transforms.Compose:
        """Build eval transforms without loading dataset data."""
        obj = cls.__new__(cls)
        obj.resize = resize
        return obj._get_default_transform()

    def _get_default_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(
                (self.resize, self.resize),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

    def _load_from_csv(self, data_path: str) -> Tuple[List[Path], List[torch.Tensor]]:
        """Load images and labels from CSV file."""
        df = pd.read_csv(data_path)

        image_paths = []
        labels = []

        print(f'{data_path} with shape {df.shape}')

        if self.num_classes is None:
            self.num_classes = df.shape[1] - 1

        image_paths = df.values[:, 0]

        labels = torch.FloatTensor(
            df.values[:, 1:(self.num_classes + 1)].astype(np.float64)
        )

        print(f'{data_path} OK')

        return image_paths, labels

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.

        Returns:
            Tuple of (image, labels) where labels is a binary tensor
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = Image.open(
                img_path
            ).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (self.resize, self.resize), color='black')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return img_path, image, label


def get_data_loaders(
    train_path: str,
    test_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    num_classes: Optional[int] = None,
    resize: int = DEFAULT_RESIZE,
    use_ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    """
    Create data loaders for train and test sets.

    Args:
        train_path: Path to training data
        test_path: Path to test data
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        num_classes: Number of classes
        resize: Image size (height and width) for transforms
        use_ddp: Whether to use distributed data parallel
        rank: Rank of current process
        world_size: Total number of processes

    Returns:
        Tuple of (train_loader, test_loader)
    """

    train_dataset = MultilabelImageDataset(
        data_path=train_path,
        num_classes=num_classes,
        resize=resize,
        train=True,
    )

    test_dataset = MultilabelImageDataset(
        data_path=test_path,
        num_classes=num_classes,
        resize=resize,
        train=False,
    )

    # Create samplers for DDP
    if use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=True  # for easy all gather
        )
        shuffle = False  # Shuffle is handled by sampler
    else:
        train_sampler = None
        test_sampler = None
        shuffle = True

    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size // world_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        **loader_kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=(batch_size // world_size) * 2,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        **loader_kwargs,
    )

    return train_loader, test_loader

