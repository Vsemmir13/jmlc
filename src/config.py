"""Configuration loader module."""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import torch


class Config:
    """Configuration class to load and manage YAML config."""
    
    def __init__(self, config_path: str):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Validate and set device
        self.device = self._get_device()
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device (cuda or cpu)."""
        device_str = self.config.get('device', 'cuda')
        if device_str == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dot notation)."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access."""
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return self.get(key) is not None
    
    def save(self, config_path: str):
        """Save configuration to YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        print(f"Configuration saved to {config_path}")
