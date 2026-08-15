"""
app.py
------
Streamlit UI for the fruit detection + classification pipeline.

Run with:
    streamlit run app.py

Accepts either a still IMAGE or a VIDEO and runs the same detector /
classifier pipeline on whichever type you upload.

Expects trained weights at weights/best_detect.pt and weights/best_cls.pt
(override via the DETECTOR_WEIGHTS / CLASSIFIER_WEIGHTS environment
variables, or the sidebar paths below).
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from inference import (
    DEFAULT_CLASSIFIER_WEIGHTS,
    DEFAULT_DETECTOR_WEIGHTS,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    FruitPipeline,
    is_image_file,
    is_video_file,
)

st.set_page_config(
    page_title="Produce Vision — Detection & Classification",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit's file_uploader wants extensions without the leading dot.
IMAGE_TYPES = sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS)
VIDEO_TYPES = sorted(ext.lstrip(".") for ext in VIDEO_EXTENSIONS)

# --------------------------------------------------------------------------- #
# Design system
# --------------------------------------------------------------------------- #

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:            #0b0f14;
  --surface:       #111823;
  --surface-2:     #151e2b;
  --border:        rgba(255,255,255,.08);
  --border-strong: rgba(255,255,255,.16);
  --text:          #eef2f7;
  --muted:         #93a1b3;
  --brand:         #34d399;
  --brand-2:       #22d3ee;
  --warn:          #fbbf24;
  --danger:        #fb7185;
  --radius:        16px;
  --shadow:        0 18px 40px -22px rgba(0,0,0,.85);
}

html, body, [class*="css"], .stApp {
  font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
}

.stApp {
  background:
    radial-gradient(1000px 520px at 12% -10%, rgba(52,211,153,.14), transparent 60%),
    radial-gradient(900px 480px at 92% 0%, rgba(34,211,238,.12), transparent 60%),
    var(--bg);
  color: var(--text);
}

/* hide default chrome */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1360px;}

h1, h2, h3, h4 {color: var(--text); letter-spacing: -.02em; font-weight: 700;}

/* ---------- hero ---------- */
.pv-hero {
  position: relative;
  border: 1px solid var(--border);
  border-radius: 22px;
  padding: 30px 34px;
  background: linear-gradient(135deg, rgba(52,211,153,.10), rgba(34,211,238,.06) 45%, transparent 80%), var(--surface);
  box-shadow: var(--shadow);
  overflow: hidden;
  animation: pv-rise .5s ease both;
}
.pv-hero:after {
  content: ""; position: absolute; inset: -40% -10% auto auto; width: 340px; height: 340px;
  background: radial-gradient(circle, rgba(52,211,153,.22), transparent 62%); filter: blur(6px);
}
.pv-eyebrow {
  display:inline-flex; align-items:center; gap:8px;
  font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
  color: var(--brand); background: rgba(52,211,153,.10);
  border:1px solid rgba(52,211,153,.28); padding:6px 12px; border-radius:999px;
}
.pv-title {font-size: clamp(1.7rem, 3.2vw, 2.5rem); font-weight: 800; margin:.65rem 0 .4rem; line-height:1.1;}
.pv-title span {background: linear-gradient(92deg, var(--brand), var(--brand-2)); -webkit-background-clip:text; background-clip:text; color:transparent;}
.pv-sub {color: var(--muted); font-size: 1rem; max-width: 62ch; line-height:1.6; margin:0;}
.pv-chips {display:flex; flex-wrap:wrap; gap:8px; margin-top:18px;}
.pv-chip {
  font-size:.78rem; color:var(--muted); background:var(--surface-2);
  border:1px solid var(--border); padding:7px 12px; border-radius:999px;
}
.pv-chip b {color:var(--text); font-weight:600;}

/* ---------- cards ---------- */
.pv-card {
  border:1px solid var(--border); border-radius: var(--radius);
  background: var(--surface); padding: 20px 22px; box-shadow: var(--shadow);
  animation: pv-rise .45s ease both;
}
.pv-card-head {display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:14px;}
.pv-card-title {font-size:.95rem; font-weight:700; letter-spacing:-.01em; display:flex; align-items:center; gap:9px;}
.pv-dot {width:8px; height:8px; border-radius:50%; background:var(--brand); box-shadow:0 0 0 4px rgba(52,211,153,.16);}
.pv-tag {
  font-size:.7rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase;
  color:var(--muted); border:1px solid var(--border); border-radius:999px; padding:4px 10px; background:var(--surface-2);
}

/* ---------- stat tiles ---------- */
.pv-stats {display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin:6px 0 4px;}
.pv-stat {
  border:1px solid var(--border); border-radius:14px; padding:16px 18px;
  background: linear-gradient(180deg, var(--surface-2), var(--surface));
  transition: transform .18s ease, border-color .18s ease;
}
.pv-stat:hover {transform: translateY(-2px); border-color: var(--border-strong);}
.pv-stat-label {font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); font-weight:700;}
.pv-stat-value {font-size:1.7rem; font-weight:800; margin-top:6px; letter-spacing:-.03em;}
.pv-stat-value.mono {font-family:'JetBrains Mono', monospace; font-size:1.4rem;}
.pv-stat-foot {font-size:.76rem; color:var(--muted); margin-top:2px;}

/* ---------- prediction rows ---------- */
.pv-pred {
  display:flex; align-items:center; gap:14px; padding:13px 4px;
  border-bottom:1px dashed var(--border);
}
.pv-pred:last-child {border-bottom:none;}
.pv-rank {
  width:26px; height:26px; flex:0 0 26px; border-radius:8px; display:grid; place-items:center;
  font-size:.75rem; font-weight:700; color:#04140d;
  background: linear-gradient(135deg, var(--brand), var(--brand-2));
}
.pv-pred-body {flex:1; min-width:0;}
.pv-pred-top {display:flex; align-items:baseline; justify-content:space-between; gap:10px;}
.pv-pred-label {font-weight:600; font-size:.95rem; text-transform:capitalize; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
.pv-pred-val {font-family:'JetBrains Mono',monospace; font-size:.85rem; color:var(--brand);}
.pv-pred-meta {font-size:.75rem; color:var(--muted); margin-top:3px;}
.pv-bar {height:6px; border-radius:999px; background:rgba(255,255,255,.07); margin-top:8px; overflow:hidden;}
.pv-bar > i {
  display:block; height:100%; border-radius:999px;
  background: linear-gradient(90deg, var(--brand), var(--brand-2));
  animation: pv-grow .7s cubic-bezier(.22,1,.36,1) both;
}

/* ---------- states ---------- */
.pv-state {
  border:1px dashed var(--border-strong); border-radius:var(--radius);
  background: linear-gradient(180deg, var(--surface-2), var(--surface));
  padding:52px 28px; text-align:center; animation: pv-rise .45s ease both;
}
.pv-state-icon {font-size:2.4rem; line-height:1;}
.pv-state h4 {margin:14px 0 6px; font-size:1.1rem;}
.pv-state p {color:var(--muted); font-size:.9rem; margin:0 auto; max-width:46ch; line-height:1.6;}
.pv-banner {
  border-radius:var(--radius); padding:18px 20px; border:1px solid;
  display:flex; gap:14px; align-items:flex-start; animation: pv-rise .4s ease both;
}
.pv-banner.err {background:rgba(251,113,133,.08); border-color:rgba(251,113,133,.34);}
.pv-banner.ok  {background:rgba(52,211,153,.08);  border-color:rgba(52,211,153,.32);}
.pv-banner.warn{background:rgba(251,191,36,.08);  border-color:rgba(251,191,36,.30);}
.pv-banner h4 {margin:0 0 4px; font-size:.95rem;}
.pv-banner p, .pv-banner code {color:var(--muted); font-size:.85rem; margin:2px 0;}
.pv-banner code {font-family:'JetBrains Mono',monospace; color:var(--text); background:rgba(255,255,255,.06); padding:2px 6px; border-radius:6px;}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d141d, #0b0f14);
  border-right:1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container {padding-top:1.6rem;}
.pv-side-brand {display:flex; align-items:center; gap:10px; margin-bottom:6px;}
.pv-side-mark {
  width:34px; height:34px; border-radius:10px; display:grid; place-items:center; font-size:1.05rem;
  background: linear-gradient(135deg, rgba(52,211,153,.9), rgba(34,211,238,.85));
}
.pv-side-name {font-weight:800; letter-spacing:-.02em;}
.pv-side-role {font-size:.72rem; color:var(--muted);}
.pv-side-label {
  font-size:.7rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
  color:var(--muted); margin:20px 0 6px;
}

/* ---------- widgets ---------- */
.stRadio [role="radiogroup"] label,
div[data-testid="stFileUploaderDropzone"] {transition: all .18s ease;}
div[data-testid="stFileUploaderDropzone"] {
  background: var(--surface-2) !important; border:1.5px dashed var(--border-strong) !important;
  border-radius:14px !important; padding:26px !important;
}
div[data-testid="stFileUploaderDropzone"]:hover {border-color: var(--brand) !important; background: rgba(52,211,153,.06) !important;}
.stSlider [data-baseweb="slider"] div[role="slider"] {box-shadow:0 0 0 4px rgba(52,211,153,.2);}
.stTextInput input, .stNumberInput input {
  background: var(--surface-2) !important; border:1px solid var(--border) !important;
  color: var(--text) !important; border-radius:10px !important;
  font-family:'JetBrains Mono',monospace !important; font-size:.82rem !important;
}
.stTextInput input:focus {border-color: var(--brand) !important;}
.stButton > button, .stDownloadButton > button {
  border-radius:11px; border:1px solid var(--border-strong); font-weight:600;
  background: var(--surface-2); color: var(--text); padding:.5rem 1.1rem; transition: all .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--brand); color: var(--brand); transform: translateY(-1px);
}
[data-testid="stImage"] img {border-radius:14px; border:1px solid var(--border);}
video {border-radius:14px; border:1px solid var(--border);}
[data-testid="stDataFrame"] {border:1px solid var(--border); border-radius:14px; overflow:hidden;}
.stTabs [data-baseweb="tab-list"] {gap:6px; border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"] {
  background:transparent; border-radius:10px 10px 0 0; padding:10px 16px;
  font-weight:600; font-size:.88rem; color:var(--muted);
}
.stTabs [aria-selected="true"] {color:var(--text) !important; background:var(--surface) !important;}
.stSpinner > div {border-top-color: var(--brand) !important;}
hr {border-color: var(--border) !important;}

@keyframes pv-rise {from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:none;}}
@keyframes pv-grow {from{width:0;} }

@media (max-width: 820px) {
  .pv-hero {padding:22px;}
  .block-container {padding-top:1.2rem;}
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

MODE_DETECT = "Detect only"
MODE_CLASSIFY = "Classify whole image"
MODE_BOTH = "Detect + classify each box"

MODE_META = {
    MODE_DETECT: ("Localisation", "YOLO11 detector draws bounding boxes around every fruit it finds."),
    MODE_CLASSIFY: ("Single label", "YOLO11-cls assigns one label to the entire image / each frame."),
    MODE_BOTH: ("Two-stage", "Detector proposes regions, classifier refines each crop's label."),
}


def esc(value) -> str:
    return str(value).replace("<", "&lt;").replace(">", "&gt;")


def card_open(title: str, tag: str | None = None) -> None:
    tag_html = f'<span class="pv-tag">{esc(tag)}</span>' if tag else ""
    st.markdown(
        f'<div class="pv-card"><div class="pv-card-head">'
        f'<div class="pv-card-title"><span class="pv-dot"></span>{esc(title)}</div>{tag_html}'
        f"</div>",
        unsafe_allow_html=True,
    )


def card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def stat(label: str, value: str, foot: str = "", mono: bool = False) -> str:
    cls = "pv-stat-value mono" if mono else "pv-stat-value"
    return (
        f'<div class="pv-stat"><div class="pv-stat-label">{esc(label)}</div>'
        f'<div class="{cls}">{esc(value)}</div>'
        f'<div class="pv-stat-foot">{esc(foot)}</div></div>'
    )


def prediction_row(rank: int, label: str, conf: float | None, meta: str = "") -> str:
    pct = 0.0 if conf is None else max(0.0, min(1.0, float(conf))) * 100
    conf_txt = "—" if conf is None else f"{pct:.1f}%"
    meta_html = f'<div class="pv-pred-meta">{esc(meta)}</div>' if meta else ""
    return (
        f'<div class="pv-pred"><div class="pv-rank">{rank}</div><div class="pv-pred-body">'
        f'<div class="pv-pred-top"><div class="pv-pred-label">{esc(label)}</div>'
        f'<div class="pv-pred-val">{conf_txt}</div></div>{meta_html}'
        f'<div class="pv-bar"><i style="width:{pct:.1f}%"></i></div></div></div>'
    )


def count_row(rank: int, label: str, count: int, max_count: int, meta: str = "") -> str:
    """Same visual style as prediction_row, but bar length reflects a raw
    count relative to the largest count (used for aggregated video stats)."""
    pct = 0.0 if max_count <= 0 else (count / max_count) * 100
    meta_html = f'<div class="pv-pred-meta">{esc(meta)}</div>' if meta else ""
    return (
        f'<div class="pv-pred"><div class="pv-rank">{rank}</div><div class="pv-pred-body">'
        f'<div class="pv-pred-top"><div class="pv-pred-label">{esc(label)}</div>'
        f'<div class="pv-pred-val">{count}</div></div>{meta_html}'
        f'<div class="pv-bar"><i style="width:{pct:.1f}%"></i></div></div></div>'
    )


@st.cache_resource(show_spinner="Loading models...")
def load_pipeline(detector_weights: str, classifier_weights: str, need_detector: bool, need_classifier: bool):
    return FruitPipeline(
        detector_weights=detector_weights,
        classifier_weights=classifier_weights,
        load_detector=need_detector,
        load_classifier=need_classifier,
    )


def save_upload_to_tmp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


# --------------------------------------------------------------------------- #
# Image flow
# --------------------------------------------------------------------------- #
def run_on_image(pipeline: FruitPipeline, tmp_path: str, mode: str, conf: float, det_imgsz: int, cls_imgsz: int) -> None:
    with st.spinner("Running inference..."):
        if mode == MODE_DETECT:
            result = pipeline.detect(tmp_path, conf=conf, imgsz=det_imgsz)
            rows = [
                {"Label": d.detector_label, "Confidence": round(d.detector_conf, 3)}
                for d in result.detections
            ]
            preds = [(d.detector_label, d.detector_conf, "Detector") for d in result.detections]

        elif mode == MODE_CLASSIFY:
            label, cls_conf = pipeline.classify(tmp_path, imgsz=cls_imgsz)
            result = None
            rows = [{"Label": label, "Confidence": round(cls_conf, 3)}]
            preds = [(label, cls_conf, "Whole-image classifier")]

        else:  # Detect + classify each box
            result = pipeline.detect_and_classify(
                tmp_path, conf=conf, det_imgsz=det_imgsz, cls_imgsz=cls_imgsz
            )
            rows = [
                {
                    "Detector label": d.detector_label,
                    "Detector conf": round(d.detector_conf, 3),
                    "Classifier label": d.classifier_label,
                    "Classifier conf": round(d.classifier_conf, 3) if d.classifier_conf else None,
                }
                for d in result.detections
            ]
            preds = [
                (
                    d.classifier_label or d.detector_label,
                    d.classifier_conf if d.classifier_conf else d.detector_conf,
                    f"detector: {d.detector_label} · {d.detector_conf:.2f}",
                )
                for d in result.detections
            ]

    counts = Counter(label for label, _, _ in preds)
    ranked_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    max_count = ranked_counts[0][1] if ranked_counts else 0

    annotated_path = None
    if result is not None and result.detections:
        annotated_path = os.path.join(tempfile.gettempdir(), f"annotated_{Path(tmp_path).name}")
        pipeline.annotate(tmp_path, result, annotated_path)

    # ---------------------------- status banner ----------------------------- #
    if preds:
        top = max(preds, key=lambda p: p[1] or 0)
        st.markdown(
            f"""
            <div class="pv-banner ok">
              <div style="font-size:1.3rem">✅</div>
              <div>
                <h4>Inference complete</h4>
                <p>{len(preds)} object(s) resolved · top prediction
                <code>{esc(top[0])}</code> at {((top[1] or 0) * 100):.1f}% confidence.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="pv-banner warn">
              <div style="font-size:1.3rem">🔍</div>
              <div>
                <h4>No detections above threshold</h4>
                <p>Try lowering the confidence threshold (currently {conf:.2f}) or using a larger detector image size.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.write("")

    # ------------------------------- metrics -------------------------------- #
    confs = [c for _, c, _ in preds if c]
    avg = sum(confs) / len(confs) if confs else 0.0
    best = max(confs) if confs else 0.0
    top_label, top_count = ranked_counts[0] if ranked_counts else ("—", 0)
    st.markdown(
        '<div class="pv-stats">'
        + stat("Objects found", str(len(preds)), "above threshold")
        + stat("Most detected", f"{top_count}× {top_label}" if preds else "—", "highest object count")
        + stat("Top confidence", f"{best * 100:.1f}%", "highest scoring object", mono=True)
        + stat("Mean confidence", f"{avg * 100:.1f}%", "across all objects", mono=True)
        + stat("Unique labels", str(len({p[0] for p in preds})), "distinct classes")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # ----------------------------- visual output ---------------------------- #
    col1, col2 = st.columns(2, gap="large")

    with col1:
        card_open("Input", "original")
        st.image(Image.open(tmp_path), use_container_width=True)
        card_close()

    with col2:
        card_open("Output", "annotated" if annotated_path else "unannotated")
        st.image(Image.open(annotated_path or tmp_path), use_container_width=True)
        if result is not None and not result.detections:
            st.caption("No detections above the confidence threshold.")
        card_close()

    st.write("")

    # -------------------------------- results ------------------------------- #
    tab_counts, tab_pred, tab_table, tab_chart = st.tabs(
        ["Object counts", "Predictions", "Raw table", "Confidence chart"]
    )

    with tab_counts:
        card_open("Object counts", f"{len(preds)} object(s) · {len(counts)} class(es)")
        if ranked_counts:
            st.markdown(
                "".join(
                    count_row(i, label, cnt, max_count, meta=f"{cnt} of {len(preds)} object(s)")
                    for i, (label, cnt) in enumerate(ranked_counts, start=1)
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="pv-state"><div class="pv-state-icon">🫥</div>'
                "<h4>Nothing to show</h4><p>No objects were detected in this image.</p></div>",
                unsafe_allow_html=True,
            )
        card_close()

    with tab_pred:
        card_open("Predictions", f"{len(preds)} result(s)")
        if preds:
            ranked = sorted(preds, key=lambda p: p[1] or 0, reverse=True)
            st.markdown(
                "".join(
                    prediction_row(i, label, conf_v, meta)
                    for i, (label, conf_v, meta) in enumerate(ranked, start=1)
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="pv-state"><div class="pv-state-icon">🫥</div>'
                "<h4>Nothing to show</h4><p>No predictions were returned for this image.</p></div>",
                unsafe_allow_html=True,
            )
        card_close()

    with tab_table:
        card_open("Raw results", "dataframe")
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
        else:
            st.caption("No results.")
        card_close()

    with tab_chart:
        card_open("Confidence distribution", "per object")
        if preds:
            chart_df = pd.DataFrame(
                {"Confidence": [c or 0 for _, c, _ in preds]},
                index=[f"{i}. {label}" for i, (label, _, _) in enumerate(preds, start=1)],
            )
            st.bar_chart(chart_df, color="#34d399", use_container_width=True)
        else:
            st.caption("No results to plot.")
        card_close()


# --------------------------------------------------------------------------- #
# Video flow
# --------------------------------------------------------------------------- #
def run_on_video(
    pipeline: FruitPipeline,
    tmp_path: str,
    mode: str,
    conf: float,
    det_imgsz: int,
    cls_imgsz: int,
    frame_skip: int,
) -> None:
    process_mode = {MODE_DETECT: "detect", MODE_CLASSIFY: "classify"}.get(mode, "both")
    out_path = os.path.join(tempfile.gettempdir(), f"annotated_{Path(tmp_path).stem}.mp4")

    card_open("Processing", f"frame skip = {frame_skip}")
    progress_bar = st.progress(0.0, text="Processing video...")

    def _on_progress(frac: float) -> None:
        progress_bar.progress(min(max(frac, 0.0), 1.0), text=f"Processing video... {frac * 100:.0f}%")

    with st.spinner("Running inference over video frames..."):
        video_result = pipeline.process_video(
            tmp_path,
            out_path,
            mode=process_mode,
            conf=conf,
            det_imgsz=det_imgsz,
            cls_imgsz=cls_imgsz,
            frame_skip=frame_skip,
            progress_callback=_on_progress,
        )
    progress_bar.empty()
    card_close()
    st.write("")

    counts = video_result.class_counts
    total_hits = sum(counts.values())

    # ---------------------------- status banner ----------------------------- #
    if counts:
        top_label, top_count = max(counts.items(), key=lambda kv: kv[1])
        st.markdown(
            f"""
            <div class="pv-banner ok">
              <div style="font-size:1.3rem">✅</div>
              <div>
                <h4>Video processed</h4>
                <p>{video_result.processed_frames} of {video_result.total_frames} frame(s) analysed
                at {video_result.fps:.1f} fps · most frequent label
                <code>{esc(top_label)}</code> ({top_count} hit(s)).</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="pv-banner warn">
              <div style="font-size:1.3rem">🔍</div>
              <div>
                <h4>No detections above threshold</h4>
                <p>Try lowering the confidence threshold (currently {conf:.2f}) or a smaller frame skip.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.write("")

    # ------------------------------- metrics -------------------------------- #
    st.markdown(
        '<div class="pv-stats">'
        + stat("Frames processed", f"{video_result.processed_frames}/{video_result.total_frames}", "sampled at this frame skip")
        + stat("Video fps", f"{video_result.fps:.1f}", "source frame rate", mono=True)
        + stat("Total hits", str(total_hits), "labelled instances across frames")
        + stat("Unique labels", str(len(counts)), "distinct classes seen")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # ----------------------------- visual output ---------------------------- #
    col1, col2 = st.columns(2, gap="large")

    with col1:
        card_open("Input", "original")
        st.video(tmp_path)
        card_close()

    with col2:
        card_open("Output", "annotated")
        st.video(video_result.annotated_video_path)
        with open(video_result.annotated_video_path, "rb") as f:
            st.download_button(
                "Download annotated video",
                data=f.read(),
                file_name=f"annotated_{Path(tmp_path).stem}.mp4",
                mime="video/mp4",
            )
        card_close()

    st.write("")

    # -------------------------------- results ------------------------------- #
    tab_pred, tab_table, tab_chart = st.tabs(["Label frequency", "Raw table", "Chart"])

    ranked_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    max_count = ranked_counts[0][1] if ranked_counts else 0

    with tab_pred:
        card_open("Label frequency", f"{len(counts)} label(s)")
        if ranked_counts:
            st.markdown(
                "".join(
                    count_row(i, label, cnt, max_count, meta=f"{(cnt / total_hits * 100):.1f}% of hits")
                    for i, (label, cnt) in enumerate(ranked_counts, start=1)
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="pv-state"><div class="pv-state-icon">🫥</div>'
                "<h4>Nothing to show</h4><p>No labels were detected in this video.</p></div>",
                unsafe_allow_html=True,
            )
        card_close()

    with tab_table:
        card_open("Raw results", "dataframe")
        if ranked_counts:
            df = pd.DataFrame(ranked_counts, columns=["Label", "Count"])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="video_label_counts.csv",
                mime="text/csv",
            )
        else:
            st.caption("No results.")
        card_close()

    with tab_chart:
        card_open("Label distribution", "count per class")
        if ranked_counts:
            chart_df = pd.DataFrame(
                {"Count": [c for _, c in ranked_counts]},
                index=[label for label, _ in ranked_counts],
            )
            st.bar_chart(chart_df, color="#34d399", use_container_width=True)
        else:
            st.caption("No results to plot.")
        card_close()


def main():
    # ----------------------------- sidebar ---------------------------------- #
    with st.sidebar:
        st.markdown(
            '<div class="pv-side-brand"><div class="pv-side-mark">🍎</div>'
            '<div><div class="pv-side-name">Produce Vision</div>'
            '<div class="pv-side-role">Inference console</div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="pv-side-label">Pipeline mode</div>', unsafe_allow_html=True)
        mode = st.radio(
            "Pipeline mode",
            [MODE_DETECT, MODE_CLASSIFY, MODE_BOTH],
            index=0,  # default to "Detect only" — fastest, single-model path
            label_visibility="collapsed",
        )
        st.caption(MODE_META[mode][1])

        st.markdown('<div class="pv-side-label">Inference parameters</div>', unsafe_allow_html=True)
        conf = st.slider("Detector confidence threshold", 0.05, 0.95, 0.25, 0.05)
        det_imgsz = st.select_slider("Detector image size", options=[320, 416, 512, 640], value=512)
        cls_imgsz = st.select_slider("Classifier image size", options=[128, 224, 320], value=224)
        frame_skip = st.slider(
            "Video frame skip",
            1,
            10,
            1,
            help="Process every Nth frame for speed (video only); boxes/labels hold over on skipped frames.",
        )

        st.markdown('<div class="pv-side-label">Model weights</div>', unsafe_allow_html=True)
        detector_weights = st.text_input("Detector weights path", value=DEFAULT_DETECTOR_WEIGHTS)
        classifier_weights = st.text_input("Classifier weights path", value=DEFAULT_CLASSIFIER_WEIGHTS)

    need_detector = mode in (MODE_DETECT, MODE_BOTH)
    need_classifier = mode in (MODE_CLASSIFY, MODE_BOTH)

    # ------------------------------- hero ----------------------------------- #
    st.markdown(
        f"""
        <div class="pv-hero">
          <span class="pv-eyebrow">● {esc(MODE_META[mode][0])} · YOLO11</span>
          <div class="pv-title">Fruit &amp; vegetable <span>detection and classification</span></div>
          <p class="pv-sub">A two-stage computer-vision pipeline: the detector locates produce in the frame,
          then a dedicated classifier refines the label of every crop for higher precision.
          Works on still images and video.</p>
          <div class="pv-chips">
            <div class="pv-chip">Mode <b>{esc(mode)}</b></div>
            <div class="pv-chip">Confidence <b>{conf:.2f}</b></div>
            <div class="pv-chip">Detector <b>{det_imgsz}px</b></div>
            <div class="pv-chip">Classifier <b>{cls_imgsz}px</b></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # --------------------------- weights guard ------------------------------ #
    missing = []
    if need_detector and not Path(detector_weights).exists():
        missing.append(detector_weights)
    if need_classifier and not Path(classifier_weights).exists():
        missing.append(classifier_weights)

    if missing:
        items = "".join(f"<p><code>{esc(m)}</code></p>" for m in missing)
        st.markdown(
            f"""
            <div class="pv-banner err">
              <div style="font-size:1.3rem">⚠️</div>
              <div>
                <h4>Model weights not found</h4>
                {items}
                <p>Copy <code>best.pt</code> from your Colab training run(s) to these paths,
                or update the paths in the sidebar.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    pipeline = load_pipeline(detector_weights, classifier_weights, need_detector, need_classifier)

    # ------------------------------ upload ---------------------------------- #
    card_open("Input image or video", " · ".join(IMAGE_TYPES + VIDEO_TYPES))
    uploaded_file = st.file_uploader(
        "Upload an image or a video",
        type=IMAGE_TYPES + VIDEO_TYPES,
        label_visibility="collapsed",
    )
    card_close()
    st.write("")

    if uploaded_file is None:
        st.markdown(
            """
            <div class="pv-state">
              <div class="pv-state-icon">🖼️</div>
              <h4>Nothing loaded yet</h4>
              <p>Drop a photo or a video of fruit or vegetables above to run the pipeline.
              Annotated output and per-object confidence scores appear here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    tmp_path = save_upload_to_tmp(uploaded_file)

    if is_image_file(tmp_path):
        run_on_image(pipeline, tmp_path, mode, conf, det_imgsz, cls_imgsz)
    elif is_video_file(tmp_path):
        run_on_video(pipeline, tmp_path, mode, conf, det_imgsz, cls_imgsz, frame_skip)
    else:
        st.markdown(
            f"""
            <div class="pv-banner err">
              <div style="font-size:1.3rem">⚠️</div>
              <div>
                <h4>Unsupported file type</h4>
                <p>Could not recognise <code>{esc(Path(uploaded_file.name).suffix)}</code> as an image or video.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()