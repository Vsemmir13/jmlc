"""Training module with wandb logging."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.amp import autocast
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import math
import numpy as np

from .evaluator import MultilabelEvaluator
from .losses import create_loss


class CosineAnnealingLRWithWarmup(optim.lr_scheduler._LRScheduler):
    """Cosine annealing LR scheduler with linear warmup."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_max: int,
        warmup_steps: int = 0,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            T_max: Maximum number of iterations (after warmup)
            warmup_steps: Number of warmup steps
            warmup_start_lr: Starting learning rate for warmup
            eta_min: Minimum learning rate for cosine annealing
            last_epoch: The index of last epoch
        """
        self.T_max = T_max
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: linear increase from warmup_start_lr to base_lr
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * warmup_factor
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            step = self.last_epoch - self.warmup_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * step / self.T_max))
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_factor
                for base_lr in self.base_lrs
            ]


class MultilabelTrainer:
    """Trainer for multilabel image classification with wandb logging."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        device: torch.device,
        use_ddp: bool = False,
        rank: int = 0,
        world_size: int = 1
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            test_loader: Test data loader
            config: Configuration dictionary
            device: Device to train on
            use_ddp: Whether to use distributed data parallel
            rank: Rank of current process
            world_size: Total number of processes
        """
        self.use_ddp = use_ddp
        self.rank = rank
        self.world_size = world_size
        
        # Move model to device and wrap with DDP if needed
        self.model = model.to(device)
        if use_ddp:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[device.index] if device.type == 'cuda' else None,
                output_device=device.index if device.type == 'cuda' else None,
                find_unused_parameters=False
            )
            # For evaluation, we'll use model.module to access the underlying model
            self.model_for_eval = self.model.module
        else:
            self.model_for_eval = self.model
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss function (move to same device as model so internal buffers
        # like class-balanced weights reside on the correct device)
        self.criterion = self._create_loss().to(device)
        
        # Setup bfloat16 mixed precision training
        self.use_bf16 = config.get('training.use_bf16', False)
        if self.use_bf16 and device.type == 'cuda':
            # Check if bfloat16 is supported
            if torch.cuda.is_bf16_supported():
                self.use_bf16 = True
                if rank == 0:
                    print("bfloat16 mixed precision training enabled")
            else:
                self.use_bf16 = False
                if rank == 0:
                    print("Warning: bfloat16 not supported on this device, disabling mixed precision")
        else:
            self.use_bf16 = False
        
        # Setup evaluator (use model_for_eval for evaluation)
        self.evaluator = MultilabelEvaluator(
            self.model_for_eval, device, use_ddp, rank, world_size, use_bf16=self.use_bf16
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.checkpoint_dir = Path(config.get('checkpoint.save_dir', 'checkpoints'))
        if rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.config.save(self.checkpoint_dir / 'config.yaml')
        # Wandb setup (only on rank 0)
        if rank == 0:
            self._setup_wandb()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config."""
        opt_config = self.config.get('optimizer', {})
        opt_name = opt_config.get('name', 'adam').lower()
        lr = opt_config.get('lr', 0.001)
        weight_decay = opt_config.get('weight_decay', 0.0001)
        
        if opt_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'sgd':
            momentum = opt_config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _create_loss(self) -> nn.Module:
        loss_name = self.config.get("loss.name")
        loss_params = self.config.get("loss")
        class_freq = None

        if loss_name == "db_loss":
            freq_path = loss_params.get("class_freq_path")
            if freq_path and os.path.exists(freq_path):
                freq_np = np.load(freq_path)
                class_freq = torch.from_numpy(freq_np).float()
                if class_freq.ndim != 1:
                    raise ValueError("class_freq must be 1D array")
            else:
                if self.rank == 0:
                    print("Computing class frequencies from training data")
                all_labels = self.train_loader.dataset.labels.float()
                class_freq = all_labels.sum(dim=0)
                print(f"Class frequencies: {class_freq.shape}")
                class_freq = torch.clamp(class_freq, min=1.0)

        criterion = create_loss(
            loss_name=loss_name,
            class_freq=class_freq,
            loss_config=loss_params,
        )
        return criterion

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from config."""
        sched_config = self.config.get('scheduler', {})
        sched_name = sched_config.get('name', '').lower()
        
        if not sched_name:
            return None
        
        if sched_name == 'step':
            step_size = sched_config.get('step_size', 10)
            gamma = sched_config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif sched_name == 'cosine':
            warmup_start_lr = sched_config.get('warmup_start_lr', 0.0)
            eta_min = sched_config.get('eta_min', 0.0)
            
            # Support both warmup_steps (in batches) and warmup_epochs
            warmup_steps = sched_config.get('warmup_steps', 0)
            warmup_epochs = sched_config.get('warmup_epochs', 0)
            
            # Calculate total training steps
            num_epochs = self.config.get('training.num_epochs', 50)
            batches_per_epoch = len(self.train_loader)
            total_steps = num_epochs * batches_per_epoch
            
            if warmup_epochs > 0:
                # Convert epochs to batches
                warmup_steps = int(warmup_epochs * batches_per_epoch)
            
            # Calculate T_max: if not specified or None, use all remaining steps after warmup
            T_max = sched_config.get('T_max', None)
            if T_max is None:
                T_max = total_steps - warmup_steps if warmup_steps > 0 else total_steps
            
            if warmup_steps > 0:
                return CosineAnnealingLRWithWarmup(
                    self.optimizer,
                    T_max=T_max,
                    warmup_steps=warmup_steps,
                    warmup_start_lr=warmup_start_lr,
                    eta_min=eta_min
                )
            else:
                return optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=T_max,
                    eta_min=eta_min
                )
        elif sched_name == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
            )
        else:
            return None
    
    def _setup_wandb(self):
        """Setup wandb logging."""
        wandb_config = self.config.get('wandb', {})

        if not wandb_config or not wandb_config.get('enabled', False):
            return

        if not os.environ.get('WANDB_API_KEY'):
            print("Wandb disabled: set WANDB_API_KEY to enable logging")
            return

        try:
            import wandb
        except ImportError:
            print("Wandb disabled: install wandb to enable logging")
            return

        wandb.init(
                project=wandb_config.get('project', 'multilabel_classification'),
                name=wandb_config.get('name', 'experiment'),
                tags=wandb_config.get('tags', []),
                config={
                    'model': self.config.get('model', {}),
                    'training': self.config.get('training', {}),
                    'optimizer': self.config.get('optimizer', {}),
                    'scheduler': self.config.get('scheduler', {}),
                },
            )
    
    def eval_if_need(self):
        # Evaluate based on steps
        eval_frequency = self.config.get('evaluation.eval_frequency', 1000)
        eval_metrics = self.config.get('evaluation.metrics', ['accuracy', 'f1_score'])
        
        save_best = self.config.get('checkpoint.save_best', True)
        monitor_metric = self.config.get('checkpoint.monitor_metric', 'f1_score')

        should_eval = self.global_step % eval_frequency
        if should_eval == 0:
            # Synchronize all processes after evaluation
            if self.use_ddp:
                dist.barrier()
            
            last_eval_step = self.global_step
            eval_results = self.evaluator.evaluate_epoch(data_loader=self.test_loader, metrics=eval_metrics)
            
            # Update scheduler for ReduceLROnPlateau (only on rank 0)
            if self.rank == 0 and self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(eval_results.get(monitor_metric, 0))
            
            # Log metrics (only on rank 0)
            if self.rank == 0:
                log_dict = {
                    'step': self.global_step,
                }
                
                for metric_name, metric_value in eval_results.items():
                    log_dict[f'test/{metric_name}'] = metric_value
                
                if self._wandb_run():
                    import wandb
                    wandb.log(log_dict, step=self.global_step)
                
                for metric_name, metric_value in eval_results.items():
                    if isinstance(metric_value, (int, float)):
                        print(f"Test {metric_name}: {metric_value:.4f}")
                
                # Save best model
                if save_best:
                    current_metric = eval_results.get(monitor_metric, 0)
                    if current_metric > self.best_metric:
                        self.best_metric = current_metric
                        self.save_checkpoint(
                            self.checkpoint_dir / f'best_model_{self.global_step}ba.pth',
                            metrics=eval_results
                        )
                        print(f"Saved best model with {monitor_metric}: {current_metric:.4f}")
        
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        # Set epoch for DistributedSampler to ensure proper shuffling
        if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.current_epoch)
        
        running_loss = 0.0
        
        
        log_frequency = self.config.get('wandb.log_frequency', 10)
        
        # Only show progress bar on rank 0
        if self.rank == 0:
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        else:
            pbar = self.train_loader
        
        for batch_idx, (_, images, labels) in enumerate(pbar):
            self.eval_if_need()
            
            self.model.train()

            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with bfloat16 mixed precision if enabled
            self.optimizer.zero_grad()
            if self.use_bf16:
                with autocast(dtype=torch.bfloat16, device_type='cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()

            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            
            running_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar (only on rank 0)
            if self.rank == 0:
                pbar.set_postfix({'loss': loss.item()})
            
            # Log to wandb (only on rank 0)
            if self.rank == 0 and self._wandb_run() and (batch_idx + 1) % log_frequency == 0:
                import wandb
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/batch': batch_idx + 1,
                    'train/epoch': self.current_epoch
                }, step=self.global_step)
        
    
    def train(self):
        """Main training loop."""
        num_epochs = self.config.get('training.num_epochs', 50)
        eval_frequency = self.config.get('evaluation.eval_frequency', 1000)  # Now in steps
        
        if self.rank == 0:
            print(f"Starting training for {num_epochs} epochs...")
            print(f"Evaluation will run every {eval_frequency} steps")
        
        last_eval_step = -1
        eval_results = {}
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            self.train_epoch()
            
            # Synchronize all processes at end of epoch
            if self.use_ddp:
                dist.barrier()
        
        # Save final model (only on rank 0)
        if self.rank == 0:
            final_path = Path(self.config.get('checkpoint.final_checkpoint_path', 'checkpoints/final_model.pth'))
            final_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_checkpoint(final_path, is_final=True)
            print(f"\nTraining completed! Final model saved to {final_path}")
        
        if self.rank == 0 and self._wandb_run():
            import wandb
            wandb.finish()

    def _wandb_run(self) -> bool:
        try:
            import wandb
        except ImportError:
            return False
        return wandb.run is not None
    
    def save_checkpoint(
        self,
        path: Path,
        metrics: Optional[Dict] = None,
        is_final: bool = False
    ):
        """Save model checkpoint."""
        # Get model state dict (unwrap DDP if needed)
        if self.use_ddp:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            'step': self.global_step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'metrics': metrics or {}
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        
        if is_final:
            print(f"Saved final checkpoint to {path}")
