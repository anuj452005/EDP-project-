# 🌿 Plant Disease Detection & Severity Estimation

[![Python](https://img.shields.io/badge/Python-3.9--3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)

An intelligent plant disease detection pipeline using **YOLOv8** for leaf detection, **MobileNetV2** for disease classification, and **YOLOv8-seg** for accurate severity estimation with spray recommendations.

---
# ![Hugging Face Deployed](https://huggingface.co/spaces/anuj5666449/edp_project)

## 🔄 Pipeline Architecture

![Pipeline Architecture](docs/pipeline_architecture.png)

---

## 🎯 Severity Estimation

### Heuristic vs Segmentation Model

![Severity Comparison](docs/severity_comparison.png)

![Visual Example](docs/visual_example.png)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Leaf Detection** | YOLOv8 detects individual leaves |
| 🏷️ **Disease Classification** | 15 disease classes (Tomato, Potato, Pepper) |
| 🔬 **Segmentation Severity** | YOLOv8-seg for accurate (~85-95%) severity |
| 📊 **Heuristic Fallback** | Color-based analysis if no seg model |
| 🎯 **Spray Tiers** | NO_ACTION → LOW → MEDIUM → HIGH |

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run Web UI
streamlit run scripts/streamlit_app.py
```

**Models Required:**
| Model | Path | Purpose |
|-------|------|---------|
| YOLOv8 Detection | `models/yolov8_leaf.pt` | Leaf detection |
| MobileNetV2 | `models/mobilenetv2_disease.keras` | Disease classification |
| YOLOv8 Segmentation | `models/yolov8_seg.pt` | Severity estimation ⭐ |

---

## 🎚️ Settings Guide

| Setting | Value | Use Case |
|---------|-------|----------|
| **YOLO Confidence** | 0.35 | General use ✅ |
| **Segmentation Toggle** | ON | More accurate severity |

---

## 👨‍💻 Author

**Anuj Tripathi** | EDP Project

---
