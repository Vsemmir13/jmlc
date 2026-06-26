#!/usr/bin/env python3
"""Quick end-to-end smoke test on CPU with demo data (no GPU, no pretrained download)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def run_training(config_path: Path) -> None:
    print("==> Training smoke run")
    subprocess.run(
        [sys.executable, "-m", "src.main", "--config", str(config_path)],
        cwd=ROOT,
        check=True,
    )


def run_forward_pass(config_path: Path) -> None:
    print("==> Forward pass on one demo image")
    sys.path.insert(0, str(ROOT))
    from src.config import Config
    from src.dataset import MultilabelImageDataset
    from src.model import create_model

    config = Config(str(config_path))
    model = create_model(
        num_classes=config.get("data.num_classes"),
        model_name=config.get("model.name"),
        fc_activation=config.get("model.fc_activation"),
        pretrained=config.get("model.pretrained", False),
        device=torch.device("cpu"),
    )
    model.eval()

    dataset = MultilabelImageDataset(
        data_path=str(ROOT / config.get("data.test_path")),
        num_classes=config.get("data.num_classes"),
        resize=config.get("data.resize"),
        train=False,
    )
    _, image, labels = dataset[0]
    with torch.no_grad():
        logits = model(image.unsqueeze(0))
    probs = torch.sigmoid(logits)[0]

    print(f"Input shape: {tuple(image.shape)}")
    print(f"Labels:      {labels.tolist()}")
    print(f"Pred probs:  {[round(x, 4) for x in probs.tolist()]}")
    print("Forward pass OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JMLC smoke demo")
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "smoke.yaml"),
        help="Path to smoke YAML config",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only run forward pass (expects checkpoint if model changed)",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    print(f"Project root: {ROOT}")
    print(f"Config:       {config_path}")
    with open(config_path) as f:
        print(yaml.safe_load(f))

    if not args.skip_train:
        run_training(config_path)
    run_forward_pass(config_path)
    print("\nSmoke demo completed successfully.")


if __name__ == "__main__":
    main()
