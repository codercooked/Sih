"""Train and validate an IBVAP custom Ultralytics detector.

Example:
    python tools/train_detector.py --data datasets/ibvap/data.yaml --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an IBVAP YOLO detector")
    parser.add_argument("--data", required=True, help="Path to a YOLO data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained model/checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="cpu, 0, or a CUDA device index")
    parser.add_argument("--project", default="runs/ibvap")
    parser.add_argument("--name", default="detector")
    parser.add_argument("--export", choices=["none", "onnx"], default="none")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.is_file():
        raise SystemExit(f"Dataset YAML not found: {data_path}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
    }
    if args.device:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)
    metrics = model.val(data=str(data_path), imgsz=args.imgsz)
    print(f"Validation mAP50: {getattr(metrics.box, 'map50', 'unavailable')}")
    print(f"Validation mAP50-95: {getattr(metrics.box, 'map', 'unavailable')}")

    if args.export != "none":
        exported = model.export(format=args.export)
        print(f"Exported model: {exported}")


if __name__ == "__main__":
    main()
