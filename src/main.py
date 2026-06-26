"""Main entry point for multilabel image classification training."""
import argparse
from pathlib import Path
import torch
import torch.distributed as dist
import os
import sys

# Handle both relative and absolute imports
# If running as a script directly (not as a module), use absolute imports
if __name__ == '__main__' and __package__ is None:
    # Add parent directory to path for absolute imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.config import Config
    from src.model import create_model
    from src.dataset import DEFAULT_RESIZE, get_data_loaders
    from src.trainer import MultilabelTrainer
else:
    # Use relative imports when run as a module
    from .config import Config
    from .model import create_model
    from .dataset import DEFAULT_RESIZE, get_data_loaders
    from .trainer import MultilabelTrainer


def setup_ddp(rank: int, world_size: int, backend: str = 'nccl'):
    """
    Initialize the distributed process group.
    
    Args:
        rank: Rank of the current process
        world_size: Total number of processes
        backend: Backend to use (nccl for GPU, gloo for CPU)
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Initialize the process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    print(f"Process {rank} initialized for DDP")


def cleanup_ddp():
    """Cleanup the distributed process group."""
    dist.destroy_process_group()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train multilabel image classifier')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=None,
        help='Local rank for distributed training (set automatically by torchrun)'
    )
    parser.add_argument(
        '--world_size',
        type=int,
        default=None,
        help='Total number of processes (set automatically by torchrun)'
    )
    
    args = parser.parse_args()
    
    # Check if DDP should be used
    use_ddp = False
    if 'RANK' in os.environ or 'LOCAL_RANK' in os.environ:
        # Using torchrun or similar
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank or 0))
        world_size = int(os.environ.get('WORLD_SIZE', args.world_size or 1))
        use_ddp = world_size > 1
    elif args.local_rank is not None:
        # Manual DDP setup
        rank = args.local_rank
        local_rank = args.local_rank
        world_size = args.world_size or torch.cuda.device_count()
        use_ddp = world_size > 1
    else:
        # Single GPU/CPU
        rank = 0
        local_rank = 0
        world_size = 1
        use_ddp = False
    
    # Initialize DDP if needed
    if use_ddp:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        setup_ddp(rank, world_size, backend)
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configuration
    if rank == 0:
        print("Loading configuration...")
    config = Config(args.config)
    
    # Override device from config if using DDP
    if use_ddp:
        config.device = device
    
    if rank == 0:
        print(f"Using device: {device}")
        if use_ddp:
            print(f"Distributed training with {world_size} processes")
    
    # Get data loaders
    if rank == 0:
        print("Loading datasets...")
    train_loader, test_loader = get_data_loaders(
        train_path=config.get('data.train_path'),
        test_path=config.get('data.test_path'),
        batch_size=config.get('training.batch_size', 32),
        num_workers=config.get('training.num_workers', 4),
        pin_memory=config.get('training.pin_memory', True),
        prefetch_factor=config.get('training.prefetch_factor', 2),
        num_classes=config.get('data.num_classes'),
        resize=config.get('data.resize', DEFAULT_RESIZE),
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size
    )
    
    if rank == 0:
        print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Create model
    if rank == 0:
        print("Creating model...")
    model = create_model(
        num_classes=config.get('data.num_classes'),
        model_name=config.get('model.name', 'resnet50'),
        pretrained=config.get('model.pretrained', True),
        checkpoint_path=config.get('model.checkpoint_path'),
        fc_activation=config.get('model.fc_activation', 'silu'),
        device=device,
    )
    
    if rank == 0:
        print(f"Model created: {config.get('model.name')}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    if rank == 0:
        print("Initializing trainer...")
    trainer = MultilabelTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size
    )
    
    # Train
    trainer.train()
    
    # Cleanup DDP
    if use_ddp:
        cleanup_ddp()
    
    if rank == 0:
        print("Training completed successfully!")


if __name__ == '__main__':
    main()