"""
main.py
-------
Command-line inference for the fruit detection + classification pipeline.

Usage examples
--------------
Detect only:
    python main.py detect --source path/to/image_or_dir --conf 0.25

Classify only (whole image):
    python main.py classify --source path/to/image_or_dir

Combined (detect boxes, then classify each crop for a finer label):
    python main.py both --source path/to/image_or_dir --conf 0.25

Weights default to weights/best_detect.pt and weights/best_cls.pt. Override with:
    python main.py both --source img.jpg \
        --detector-weights path/to/best.pt \
        --classifier-weights path/to/best_cls.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inference import (
    DEFAULT_CLASSIFIER_WEIGHTS,
    DEFAULT_DETECTOR_WEIGHTS,
    FruitPipeline,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(source: str) -> list[str]:
    path = Path(source)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        return sorted(
            str(p) for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"Source not found: {source}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fruit detection + classification inference")
    parser.add_argument("mode", choices=["detect", "classify", "both"], help="Inference mode")
    parser.add_argument("--source", required=True, help="Image file or directory of images")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold")
    parser.add_argument("--det-imgsz", type=int, default=512, help="Detector inference image size")
    parser.add_argument("--cls-imgsz", type=int, default=224, help="Classifier inference image size")
    parser.add_argument("--detector-weights", default=DEFAULT_DETECTOR_WEIGHTS)
    parser.add_argument("--classifier-weights", default=DEFAULT_CLASSIFIER_WEIGHTS)
    parser.add_argument(
        "--out-dir", default="outputs", help="Directory to save annotated images / results JSON"
    )
    parser.add_argument(
        "--no-annotate", action="store_true", help="Skip saving annotated images (detect/both modes)"
    )
    return parser


def print_friendly_usage() -> None:
    print(
        "No arguments given.\n"
        "\n"
        "Usage: python main.py {detect,classify,both} --source path/to/image_or_dir [options]\n"
        "\n"
        "Examples:\n"
        "  python main.py detect   --source path/to/image_or_dir --conf 0.25\n"
        "  python main.py classify --source path/to/image_or_dir\n"
        "  python main.py both     --source path/to/image_or_dir --conf 0.25\n"
        "\n"
        "Run 'python main.py --help' for the full list of options.",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    effective_argv = sys.argv[1:] if argv is None else argv
    if not effective_argv:
        print_friendly_usage()
        return 1

    args = build_parser().parse_args(argv)

    try:
        images = collect_images(args.source)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not images:
        print(f"No images found at {args.source}", file=sys.stderr)
        return 1

    need_detector = args.mode in ("detect", "both")
    need_classifier = args.mode in ("classify", "both")

    try:
        pipeline = FruitPipeline(
            detector_weights=args.detector_weights,
            classifier_weights=args.classifier_weights,
            load_detector=need_detector,
            load_classifier=need_classifier,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for img_path in images:
        print(f"\n{img_path}")

        if args.mode == "detect":
            result = pipeline.detect(img_path, conf=args.conf, imgsz=args.det_imgsz)
            for det in result.detections:
                print(f"  {det.detector_label}: {det.detector_conf:.3f}  box={det.box_xyxy}")
            record = {
                "image": img_path,
                "detections": [
                    {"label": d.detector_label, "conf": d.detector_conf, "box": d.box_xyxy}
                    for d in result.detections
                ],
            }

        elif args.mode == "classify":
            label, conf = pipeline.classify(img_path, imgsz=args.cls_imgsz)
            print(f"  {label}: {conf:.3f}")
            record = {"image": img_path, "label": label, "conf": conf}
            result = None

        else:  # both
            result = pipeline.detect_and_classify(
                img_path, conf=args.conf, det_imgsz=args.det_imgsz, cls_imgsz=args.cls_imgsz
            )
            for det in result.detections:
                label = det.classifier_label or det.detector_label
                conf = det.classifier_conf if det.classifier_conf is not None else det.detector_conf
                print(f"  detector={det.detector_label} ({det.detector_conf:.3f})  "
                      f"-> classifier={det.classifier_label} ({det.classifier_conf:.3f})"
                      if det.classifier_label else f"  {label}: {conf:.3f}")
            record = {
                "image": img_path,
                "detections": [
                    {
                        "box": d.box_xyxy,
                        "detector_label": d.detector_label,
                        "detector_conf": d.detector_conf,
                        "classifier_label": d.classifier_label,
                        "classifier_conf": d.classifier_conf,
                    }
                    for d in result.detections
                ],
            }

        if result is not None and result.detections and not args.no_annotate:
            out_path = out_dir / f"{Path(img_path).stem}_annotated.jpg"
            pipeline.annotate(img_path, result, str(out_path))
            record["annotated_image"] = str(out_path)
            print(f"  Saved annotated image: {out_path}")

        all_results.append(record)

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results summary to {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())