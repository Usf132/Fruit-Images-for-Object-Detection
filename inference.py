"""
inference.py
------------
Shared inference helpers for the fruit detection + classification models
trained in fruit_detection_yolo11_improved.ipynb.

Two models are used together:
  1. Detector  (yolo11n, trained on Roboflow data)  -> finds WHERE fruits/veg are (bounding boxes)
  2. Classifier (yolo11n-cls, trained on Kaggle data) -> refines WHAT each crop is

This module is intentionally UI-agnostic so both main.py (CLI) and app.py
(Streamlit) can share the exact same logic.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Default weight locations. Override via environment variables or by passing
# explicit paths to load_models(). These defaults assume you've copied
# best.pt / best_cls.pt out of the Colab runs/ directories into ./weights/.
# ---------------------------------------------------------------------------
DEFAULT_DETECTOR_WEIGHTS = os.environ.get("DETECTOR_WEIGHTS", "weights/best_detect.pt")
DEFAULT_CLASSIFIER_WEIGHTS = os.environ.get("CLASSIFIER_WEIGHTS", "weights/best_cls.pt")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


@dataclass
class Detection:
    """A single detected box, optionally refined by the classifier."""
    box_xyxy: tuple[float, float, float, float]
    detector_label: str
    detector_conf: float
    classifier_label: str | None = None
    classifier_conf: float | None = None


@dataclass
class InferenceResult:
    image_path: str
    detections: list[Detection] = field(default_factory=list)
    annotated_image_path: str | None = None


@dataclass
class VideoResult:
    """Summary of running inference over every (sampled) frame of a video."""
    video_path: str
    annotated_video_path: str
    total_frames: int
    processed_frames: int
    fps: float
    class_counts: dict[str, int] = field(default_factory=dict)


class FruitPipeline:
    """Loads both models once and runs detect / classify / combined inference."""

    def __init__(
        self,
        detector_weights: str = DEFAULT_DETECTOR_WEIGHTS,
        classifier_weights: str = DEFAULT_CLASSIFIER_WEIGHTS,
        load_detector: bool = True,
        load_classifier: bool = True,
    ):
        self.detector = None
        self.classifier = None

        if load_detector:
            if not Path(detector_weights).exists():
                raise FileNotFoundError(
                    f"Detector weights not found at '{detector_weights}'. "
                    "Copy best.pt from your Colab training run into that path, "
                    "or set the DETECTOR_WEIGHTS environment variable."
                )
            self.detector = YOLO(detector_weights)

        if load_classifier:
            if not Path(classifier_weights).exists():
                raise FileNotFoundError(
                    f"Classifier weights not found at '{classifier_weights}'. "
                    "Copy best.pt from your Colab classification run into that "
                    "path, or set the CLASSIFIER_WEIGHTS environment variable."
                )
            self.classifier = YOLO(classifier_weights)

    # -- Detection only -----------------------------------------------------
    def detect(self, image_path: str, conf: float = 0.25, imgsz: int = 512) -> InferenceResult:
        if self.detector is None:
            raise RuntimeError("Detector model was not loaded.")

        results = self.detector.predict(source=image_path, conf=conf, imgsz=imgsz, verbose=False)
        r = results[0]

        detections = []
        for box in r.boxes:
            xyxy = tuple(box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            detections.append(
                Detection(
                    box_xyxy=xyxy,
                    detector_label=self.detector.names[cls_id],
                    detector_conf=float(box.conf[0]),
                )
            )

        return InferenceResult(image_path=image_path, detections=detections)

    # -- Classification only (whole image) -----------------------------------
    def classify(self, image_path: str, imgsz: int = 224) -> tuple[str, float]:
        if self.classifier is None:
            raise RuntimeError("Classifier model was not loaded.")

        results = self.classifier.predict(source=image_path, imgsz=imgsz, verbose=False)
        r = results[0]
        top1_idx = int(r.probs.top1)
        top1_conf = float(r.probs.top1conf)
        label = self.classifier.names[top1_idx]
        return label, top1_conf

    # -- Combined: detect boxes, then classify each crop ---------------------
    def detect_and_classify(
        self, image_path: str, conf: float = 0.25, det_imgsz: int = 512, cls_imgsz: int = 224
    ) -> InferenceResult:
        result = self.detect(image_path, conf=conf, imgsz=det_imgsz)

        if self.classifier is None or not result.detections:
            return result

        image = Image.open(image_path).convert("RGB")
        for det in result.detections:
            x1, y1, x2, y2 = [int(v) for v in det.box_xyxy]
            crop = image.crop((x1, y1, x2, y2))

            cls_results = self.classifier.predict(source=crop, imgsz=cls_imgsz, verbose=False)
            r = cls_results[0]
            top1_idx = int(r.probs.top1)
            det.classifier_label = self.classifier.names[top1_idx]
            det.classifier_conf = float(r.probs.top1conf)

        return result

    # -- Drawing --------------------------------------------------------------
    def annotate(self, image_path: str, result: InferenceResult, out_path: str) -> str:
        """Draw boxes + labels (using the classifier label when available,
        falling back to the detector label) and save to out_path."""
        from PIL import ImageDraw, ImageFont

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=16)
        except Exception:
            font = ImageFont.load_default()

        for det in result.detections:
            x1, y1, x2, y2 = det.box_xyxy
            label = det.classifier_label or det.detector_label
            conf = det.classifier_conf if det.classifier_conf is not None else det.detector_conf
            text = f"{label} {conf:.2f}"

            draw.rectangle([x1, y1, x2, y2], outline=(255, 87, 34), width=3)
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            draw.rectangle(
                [text_bbox[0], text_bbox[1] - 4, text_bbox[2] + 4, text_bbox[3] + 2],
                fill=(255, 87, 34),
            )
            draw.text((x1 + 2, y1 - 2), text, fill="white", font=font)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        result.annotated_image_path = out_path
        return out_path

    # -- Video: run detect / classify / both over every (sampled) frame ------
    def process_video(
        self,
        video_path: str,
        out_path: str,
        mode: str = "both",
        conf: float = 0.25,
        det_imgsz: int = 512,
        cls_imgsz: int = 224,
        frame_skip: int = 1,
        progress_callback: Callable[[float], None] | None = None,
    ) -> VideoResult:
        """Run inference over a video, writing an annotated copy to out_path.

        frame_skip=1 processes every frame; frame_skip=3 processes every 3rd
        frame (faster, boxes/labels are held over on skipped frames for a
        smoother-looking output).
        """
        import cv2

        if mode in ("detect", "both") and self.detector is None:
            raise RuntimeError("Detector model was not loaded.")
        if mode in ("classify", "both") and self.classifier is None and mode == "classify":
            raise RuntimeError("Classifier model was not loaded.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        writer = None
        for fourcc_str in ("avc1", "H264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            candidate = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            if candidate.isOpened():
                writer = candidate
                break
        if writer is None:
            cap.release()
            raise RuntimeError("Could not open a video writer for any supported codec.")

        class_counts: Counter[str] = Counter()
        frame_idx = 0
        processed = 0
        # Boxes carried over onto skipped frames so the overlay doesn't flicker.
        last_boxes: list[tuple[tuple[int, int, int, int], str, float]] = []
        last_whole_label: tuple[str, float] | None = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % max(frame_skip, 1) == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)

                if mode == "classify":
                    r = self.classifier.predict(source=pil_image, imgsz=cls_imgsz, verbose=False)[0]
                    label = self.classifier.names[int(r.probs.top1)]
                    label_conf = float(r.probs.top1conf)
                    class_counts[label] += 1
                    last_whole_label = (label, label_conf)
                    last_boxes = []
                else:
                    det = self.detector.predict(source=pil_image, conf=conf, imgsz=det_imgsz, verbose=False)[0]
                    boxes = []
                    for box in det.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        det_label = self.detector.names[int(box.cls[0])]
                        det_conf = float(box.conf[0])

                        label, label_conf = det_label, det_conf
                        if mode == "both" and self.classifier is not None:
                            crop = pil_image.crop((x1, y1, x2, y2))
                            cr = self.classifier.predict(source=crop, imgsz=cls_imgsz, verbose=False)[0]
                            label = self.classifier.names[int(cr.probs.top1)]
                            label_conf = float(cr.probs.top1conf)

                        class_counts[label] += 1
                        boxes.append(((x1, y1, x2, y2), label, label_conf))
                    last_boxes = boxes
                    last_whole_label = None

                processed += 1

            # Draw whatever the most recent inference produced (current or carried-over frame).
            if last_whole_label is not None:
                label, label_conf = last_whole_label
                cv2.putText(
                    frame, f"{label} {label_conf:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
                )
            for (x1, y1, x2, y2), label, label_conf in last_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 87, 255), 2)
                text = f"{label} {label_conf:.2f}"
                text_y = max(y1 - 8, 14)
                cv2.putText(
                    frame, text, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 87, 255), 2, cv2.LINE_AA
                )

            writer.write(frame)
            frame_idx += 1
            if progress_callback and total_frames:
                progress_callback(min(frame_idx / total_frames, 1.0))

        cap.release()
        writer.release()

        return VideoResult(
            video_path=video_path,
            annotated_video_path=out_path,
            total_frames=frame_idx,
            processed_frames=processed,
            fps=fps,
            class_counts=dict(class_counts),
        )