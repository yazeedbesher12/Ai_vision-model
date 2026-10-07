import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLOv8n classifier for plant vs soil.")
    parser.add_argument("--data", default="Dataset", help="Dataset folder with train/val/test class folders.")
    parser.add_argument("--model", default="yolov8n-cls.pt", help="Base YOLO classification model.")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs. Kept modest for a first run.")
    parser.add_argument("--imgsz", type=int, default=224, help="Classifier image size.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for the GPU.")
    parser.add_argument("--device", default="0", help="GPU id. Use 0 for the first NVIDIA GPU.")
    parser.add_argument("--workers", type=int, default=0, help="Use 0 on Windows to avoid dataloader issues.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_path.resolve()}")
    project_root = Path(__file__).resolve().parent

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_root / "runs" / "classify"),
        name="plant_classifier",
        exist_ok=True,
        pretrained=True,
        patience=4,
        seed=42,
    )

    best_candidates = sorted((project_root / "runs").glob("**/plant_classifier/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    best = best_candidates[-1] if best_candidates else project_root / "runs" / "classify" / "plant_classifier" / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, "plant_best.pt")
        print(f"\nBest model saved to: {best.resolve()}")
        print(f"Easy camera path copied to: {Path('plant_best.pt').resolve()}")


if __name__ == "__main__":
    main()
