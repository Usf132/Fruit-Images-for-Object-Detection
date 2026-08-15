# 🍎 Produce Vision — Fruit & Vegetable Detection + Classification

![Produce Vision Cover](cover.png)

A two-stage computer-vision pipeline built on **YOLO11**:

1. **Detector** (`yolo11n`) — locates fruits/vegetables in an image and draws bounding boxes.
2. **Classifier** (`yolo11n-cls`) — takes each detected crop and refines the label for higher precision.

The two models are trained and evaluated in the included notebook, then served through a CLI (`main.py`) and a Streamlit web app (`app.py`) that both share the same inference logic (`inference.py`).

Works on **still images** and **video**.

---

## Project structure

```
.
├── fruit_detection_yolo11.ipynb   # Training + evaluation notebook (Colab-ready)
├── main.py                        # CLI inference entry point
├── inference.py                   # Shared pipeline logic (used by main.py and app.py)
├── app.py                         # Streamlit web UI
├── weights/
│   ├── best_detect.pt             # Trained detector weights
│   └── best_cls.pt                # Trained classifier weights
├── results/                       # Training/evaluation plots exported from the notebook
│   ├── detection_loss_curves.png
│   ├── detection_map_vs_epoch.png
│   ├── detection_confusion_matrix.png
│   ├── detection_confusion_matrix_normalized.png
│   ├── detection_pr_curve.png
│   ├── classification_accuracy_loss.png
│   ├── classification_confusion_matrix.png
│   └── classification_confusion_matrix_normalized.png
└── outputs/                       # Default output folder for CLI runs (annotated images, results.json)
```

---

## How it was trained

The notebook (`fruit_detection_yolo11.ipynb`) does the full pipeline:

1. **Detection data** — two Roboflow bounding-box datasets are downloaded and merged into a single YOLO-format dataset. Dataset 0 contributes 11 base classes; Dataset 1 contributes ripe/unripe variants of 15 classes, which are folded into the same final class (e.g. `ripe apple` + `unripe apple` → `apple`).
2. **Detector training** — `yolo11n.pt` is fine-tuned on the merged dataset, then validated and evaluated on a held-out test split.
3. **Classification data** — a Kaggle fruit/vegetable classification dataset is downloaded.
4. **Classifier training** — `yolo11n-cls.pt` is fine-tuned on the classification dataset.
5. **Diagnostics** — weak-performing detector classes are inspected (sample counts, visual review, confusion-matrix mix-ups), with optional targeted re-augmentation.
6. **Results export** — loss curves, mAP curves, PR curves, and confusion matrices for both models are saved to `results/`.

To retrain or reproduce the results, open the notebook in Google Colab (recommended, GPU runtime) or a local Jupyter environment with a CUDA-capable GPU.

### Detector classes (21)

`apple`, `banana`, `broccoli`, `cauliflower`, `corn`, `cucumber`, `lemon`, `pepper`, `pineapple`, `strawberry`, `tomato`, `bell fruit`, `dragon fruit`, `grape`, `mango`, `mangosteen`, `orange`, `papaya`, `passion fruit`, `pomegranate`, `star fruit`

**Dataset split:** 4,825 train / 839 valid / 389 test images.

> **Known open issue:** the merge script flags `pepper` (Dataset 0) and `bell fruit` (Dataset 1) as low string-similarity but potentially the same real-world object — if Dataset 1's "bell fruit" label actually means bell pepper, these should be merged into one class rather than trained separately. This hasn't been visually verified yet; check a sample of "bell fruit" images before deciding whether to merge.

---

## Performance

### Detector
(Test split, 21 classes — from the notebook's evaluation cells)

- **mAP@50:** 91.0%
- **mAP@50-95:** 74.0%
- **Precision:** 87.4%
- **Recall:** 85.8%

Per-class performance varies — `broccoli` and `corn` are the weakest classes (mAP@50 of 59% and 72% respectively), while most other classes score above 90% mAP@50. See the "Diagnosing weak detector classes" section of the notebook for the investigation into this, and `results/detection_confusion_matrix.png` for the full per-class breakdown.

### Classifier
(Test split, 5 classes)

- **Top-1 accuracy:** 95.0%
- **Top-5 accuracy:** 100.0%

> Precision/recall/F1 aren't currently computed for the classifier — the notebook's evaluation only reports top-1/top-5 accuracy via Ultralytics' built-in `.val()`. Adding per-class precision/recall/F1 would need a small additional cell that runs predictions over the test set and builds a classification report (e.g. via `sklearn.metrics.classification_report`) — happy to add that if you want those numbers.

---

## Web App Demo

The Streamlit interface provides three inference modes:
- Detect only
- Classify only
- Detect + Classify

Example output:

![Produce Vision Streamlit App](results/app_demo.png)

```bash
streamlit run app.py
```

Sidebar controls:
- Detector confidence threshold
- Detector / classifier inference image size
- Video frame skip (process every Nth frame, for faster video processing — boxes/labels hold over on skipped frames to avoid flicker)
- Detector / classifier weights paths (override the `weights/` defaults)

Upload either a still image or a video, and the app returns:
- An annotated image or video with boxes and labels drawn
- Per-object confidence scores
- A class-count bar chart summarizing what was found

If the weights aren't found at the configured paths, the app shows a clear warning with the expected paths rather than failing silently.

---

## Setup

```bash
pip install ultralytics streamlit pandas pillow opencv-python
```

Place your trained weights at:

```
weights/best_detect.pt
weights/best_cls.pt
```

(Copy these out of the notebook's Colab `runs/` directories after training.) Paths can be overridden via environment variables:

```bash
export DETECTOR_WEIGHTS=/path/to/best_detect.pt
export CLASSIFIER_WEIGHTS=/path/to/best_cls.pt
```

---

## Usage

### CLI (`main.py`)

```bash
# Detect only
python main.py detect --source path/to/image_or_dir --conf 0.25

# Classify only (whole image)
python main.py classify --source path/to/image_or_dir

# Combined: detect boxes, then classify each crop
python main.py both --source path/to/image_or_dir --conf 0.25
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--source` | — | Image file or directory of images (required) |
| `--conf` | `0.25` | Detector confidence threshold |
| `--det-imgsz` | `512` | Detector inference image size |
| `--cls-imgsz` | `224` | Classifier inference image size |
| `--detector-weights` | `weights/best_detect.pt` | Path to detector weights |
| `--classifier-weights` | `weights/best_cls.pt` | Path to classifier weights |
| `--out-dir` | `outputs` | Where annotated images and `results.json` are saved |
| `--no-annotate` | off | Skip saving annotated images |

Each run writes annotated images (for `detect`/`both`) and a `results.json` summary into `--out-dir`.

### Web app (`app.py`)

See the [Web App Demo](#web-app-demo) section above for details.

---

## Results

Training and evaluation plots for both models are in [`results/`](results/):

- **Detector**: loss curves, mAP vs. epoch, confusion matrix (raw + normalized), PR curve
- **Classifier**: accuracy/loss curves, confusion matrix (raw + normalized)

These are generated by the "Results Diagrams" section at the end of the notebook, which pulls the plots from the Ultralytics training/validation runs and saves clean copies here.

---

## Notes

- `inference.py` is UI-agnostic — it's the single source of truth for model loading and prediction, shared by both `main.py` and `app.py`.
- Video support (in both the CLI pipeline and the app) carries the last detected boxes/labels over skipped frames to avoid flicker when `frame_skip > 1`.
- Supported image types: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`. Supported video types: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`.
