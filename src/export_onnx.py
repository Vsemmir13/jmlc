"""Export a trained multilabel classifier checkpoint to ONNX."""
import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import torch


if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.config import Config
    from src.dataset import DEFAULT_RESIZE
    from src.model import create_model
else:
    from .config import Config
    from .dataset import DEFAULT_RESIZE
    from .model import create_model


PathLike = Union[str, Path]


def load_model_from_config(
    checkpoint_path: PathLike,
    config_path: PathLike,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.nn.Module:
    """Load a model from a checkpoint and YAML config."""
    config = Config(str(config_path))
    device = torch.device(device) if device is not None else torch.device("cpu")

    model = create_model(
        num_classes=config.get("data.num_classes"),
        model_name=config.get("model.name", "resnet50"),
        pretrained=False,
        checkpoint_path=str(checkpoint_path),
        fc_activation=config.get("model.fc_activation", "silu"),
        device=device,
    )
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def export_model_to_onnx(
    checkpoint_path: PathLike,
    config_path: PathLike,
    output_path: PathLike,
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = 1,
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Load a checkpoint from config and export it to ONNX."""
    config = Config(str(config_path))
    device = torch.device(device) if device is not None else torch.device("cpu")
    input_size = config.get("data.resize", DEFAULT_RESIZE)

    model = load_model_from_config(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(
        batch_size,
        3,
        input_size,
        input_size,
        device=device,
        dtype=torch.float32,
    )

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a multilabel classifier checkpoint to ONNX."
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to the PyTorch checkpoint to export.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        type=str,
        help="Path to the YAML config used to build the model.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path where the ONNX model will be written.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device to use for loading/exporting, for example cpu or cuda.",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Dummy input batch size used during export.",
    )
    parser.add_argument(
        "--opset_version",
        default=17,
        type=int,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--static_batch",
        action="store_true",
        help="Disable dynamic batch axis in the exported ONNX graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_model_to_onnx(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
        opset_version=args.opset_version,
        dynamic_batch=not args.static_batch,
    )
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    main()
