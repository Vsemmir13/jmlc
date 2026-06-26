"""Evaluation module for multilabel image classification."""
import torch
import torch.distributed as dist
from torch.amp import autocast
from torch.utils.data import DataLoader
from typing import Dict, List
from tqdm import tqdm
import numpy as np


class MultilabelEvaluator:
    """Evaluator for multilabel classification tasks."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        use_ddp: bool = False,
        rank: int = 0,
        world_size: int = 1,
        use_bf16: bool = False
    ):
        """
        Initialize evaluator.
        
        Args:
            model: PyTorch model
            device: Device to run evaluation on
            use_ddp: Whether to use distributed data parallel
            rank: Rank of current process
            world_size: Total number of processes
            use_bf16: Whether to use bfloat16 mixed precision
        """
        self.model = model
        self.device = device
        self.use_ddp = use_ddp
        self.rank = rank
        self.world_size = world_size
        self.use_bf16 = use_bf16
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            data_loader: DataLoader for evaluation
        
        Returns:
            Dictionary of metric names and values
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        # Set epoch for DistributedSampler if using DDP
        if self.use_ddp and hasattr(data_loader.sampler, 'set_epoch'):
            data_loader.sampler.set_epoch(0)
        
        # Only show progress bar on rank 0
        if self.rank == 0:
            pbar = tqdm(data_loader, desc='Eval...')
        else:
            pbar = data_loader
        
        for _, images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with bfloat16 mixed precision if enabled
            if self.use_bf16:
                with autocast(dtype=torch.bfloat16, device_type='cuda'):
                    preds = self.model(images)
            else:
                preds = self.model(images)

            # Gather predictions and labels from all processes if using DDP
            if self.use_ddp:
                preds = self.all_gather(preds)
                labels = self.all_gather(labels)

            # Convert to binary predictions
            preds_np = preds.float().cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_preds.append(preds_np)
            all_labels.append(labels_np)
        
        # Concatenate all predictions and labels
        all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
        all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
        return all_labels, all_preds
    
    def all_gather(self, tensor_for_gather):
        gathered_tensor = [torch.zeros_like(tensor_for_gather) for _ in range(self.world_size)]
        dist.all_gather(gathered_tensor, tensor_for_gather)
        return torch.cat(gathered_tensor, dim=0)
        
    def count_metrics(self, all_labels, all_preds, threshold, metrics):
        from sklearn.metrics import (
            f1_score,
            precision_score,
            recall_score,
            hamming_loss,
            roc_auc_score,
            precision_recall_curve,
            auc
        )
        # Compute metrics (only on rank 0, but all processes return results for consistency)
        results = {}
        
        if metrics is None:
            metrics = ["accuracy", "f1_score", "precision", "recall", "hamming_loss"]
        
        all_preds_binary = (all_preds > threshold).astype(np.int64)
        
        if "accuracy" in metrics:
            # Exact match accuracy (all labels must match)
            exact_match = np.all(all_preds_binary == all_labels, axis=1)
            results["accuracy"] = exact_match.mean()
        
        if "f1_score" in metrics:
            # F1 score (macro average across labels)
            results["f1_score"] = f1_score(
                all_labels, all_preds_binary, average='macro', zero_division=0
            )
            # Also compute micro F1
            results["f1_score_micro"] = f1_score(
                all_labels, all_preds_binary, average='micro', zero_division=0
            )
        
        if "precision" in metrics:
            results["precision"] = precision_score(
                all_labels, all_preds_binary, average='macro', zero_division=0
            )
            results["precision_micro"] = precision_score(
                all_labels, all_preds_binary, average='micro', zero_division=0
            )
        
        if "recall" in metrics:
            results["recall"] = recall_score(
                all_labels, all_preds_binary, average='macro', zero_division=0
            )
            results["recall_micro"] = recall_score(
                all_labels, all_preds_binary, average='micro', zero_division=0
            )
        
        if "hamming_loss" in metrics:
            results["hamming_loss"] = hamming_loss(all_labels, all_preds)

        if "roc_auc" in metrics:
            results["roc_auc"] = roc_auc_score(all_labels.flatten(), all_preds.flatten())

        if "pr_auc" in metrics:
            precision, recall, _ = precision_recall_curve(all_labels.flatten(), all_preds.flatten())
            results["pr_auc"] = auc(recall, precision)
        
        # Per-class metrics
        if "per_class_f1" in metrics or "per_class" in metrics:
            per_class_f1 = f1_score(
                all_labels, all_preds_binary, average=None, zero_division=0
            )
            results["per_class_f1"] = per_class_f1.tolist()
        
        return results
    
    def evaluate_epoch(
        self,
        data_loader: DataLoader,
        threshold: float = 0.0,
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate model for one epoch (alias for evaluate).
        
        Args:
            data_loader: DataLoader for evaluation
            threshold: Threshold for binary predictions
            metrics: List of metrics to compute
        
        Returns:
            Dictionary of metric names and values
        """
        all_labels, all_preds = self.evaluate(data_loader)
        return self.count_metrics(all_labels, all_preds, threshold, metrics)
