# 🌿 Plant Disease Detection - System Working

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    INPUT     │ →  │   DETECT     │ →  │   ANALYZE    │ →  │    OUTPUT    │
│  Full Plant  │    │   YOLOv8     │    │  Classification│   │ Spray Action │
│    Image     │    │   + Seg      │    │  + Severity   │    │  + Dosage    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

---

## Step-by-Step Process

### 1️⃣ Image Input
- Upload full plant image (JPG/PNG) or paste from clipboard
- Converted to BGR format for OpenCV

### 2️⃣ Leaf Detection (YOLOv8)
| Property | Value |
|----------|-------|
| Model | `models/yolov8_leaf.pt` |
| Input | Full plant image |
| Output | Bounding boxes `[x1, y1, x2, y2]` per leaf |
| Confidence | 0.35 (adjustable) |

### 3️⃣ Disease Segmentation (YOLOv8-seg)
| Property | Value |
|----------|-------|
| Model | `models/yolov8_seg.pt` |
| Input | Full plant image |
| Output | Binary mask of diseased regions |
| Accuracy | ~85-95% |

### 4️⃣ Disease Classification (MobileNetV2)
| Property | Value |
|----------|-------|
| Model | `models/mobilenetv2_disease.keras` |
| Input | Cropped leaf (224×224) |
| Output | Disease class + confidence |

**15 Classes:** Pepper (2), Potato (3), Tomato (10)

### 5️⃣ Severity Calculation
```
Segmentation: (disease_pixels / leaf_pixels) × 100
Heuristic:    Color analysis (brown/yellow/black/white)
```

### 6️⃣ Spray Decision
| Condition | Tier |
|-----------|------|
| No infection OR avg <3% | 🟢 NO_ACTION |
| <2 leaves AND avg <10% | 🟡 LOW |
| 2-3 leaves OR avg 10-25% | 🟠 MEDIUM |
| ≥4 leaves AND avg ≥25% | 🔴 HIGH |
| avg >35% OR >70% infected | 🟣 CRITICAL |

### 7️⃣ Pesticide Dosage
| Tier | Dosage | Action |
|------|--------|--------|
| NO_ACTION | 0% | Skip |
| LOW | 25% | Preventive mist |
| MEDIUM | 50% | Standard spray |
| HIGH | 75% | Thorough spray |
| CRITICAL | 100% | Full dosage + alert |

---

## Output Format (JSON)

```json
{
  "total_leaves": 5,
  "infected_count": 5,
  "avg_severity": 20.2,
  "max_severity": 31.0,
  "spray_tier": "HIGH",
  "dosage_percent": 75,
  "primary_disease": "Tomato - Late blight"
}
```

---

## Models Summary

| Model | File | Size | Purpose |
|-------|------|------|---------|
| YOLOv8 | `yolov8_leaf.pt` | 6.3 MB | Leaf detection |
| YOLOv8-seg | `yolov8_seg.pt` | 23.8 MB | Disease segmentation |
| MobileNetV2 | `mobilenetv2_disease.keras` | 13.6 MB | Classification |

---

## Run Command

```bash
cd "c:\d-driver\DL palnt disease"
streamlit run scripts/streamlit_app.py
```
