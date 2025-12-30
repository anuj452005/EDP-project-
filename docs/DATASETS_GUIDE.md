# Pre-Annotated Datasets for Plant Disease Segmentation

This document lists the best available datasets for training a YOLOv8-seg model for your plant disease severity estimation project.

---

## 🏆 Top Recommended Datasets

### 1. PlantSeg (Best Choice)
| Attribute | Details |
|-----------|---------|
| **Images** | 11,400+ |
| **Categories** | 115 disease types |
| **Plant Species** | 34 |
| **Annotation Type** | Pixel-level segmentation masks |
| **Quality** | High-quality, real-world images |

**Download Links:**
- 📦 [Zenodo](https://zenodo.org/records/8413916)
- 💻 [GitHub](https://github.com/mtang724/PlantSeg-dataset)

**Why Choose This:**
- ✅ Largest dataset specifically for plant disease segmentation
- ✅ Covers diseases matching your `class_names.txt` (Tomato, Potato, Pepper)
- ✅ Real-world images (not lab conditions)
- ✅ Ready-to-use segmentation masks

---

### 2. Kaggle Leaf Disease Segmentation
| Attribute | Details |
|-----------|---------|
| **Images** | 588 |
| **Categories** | Apple Scab, Apple Rust, Corn Blight, Potato Blight, Bell Pepper Spot |
| **Annotation Type** | Binary masks (diseased/healthy) |
| **Quality** | Good, based on PlantDoc |

**Download Link:**
- 📦 [Kaggle Dataset](https://www.kaggle.com/datasets/fakhrealam9537/leaf-disease-segmentation-dataset)

**Why Choose This:**
- ✅ Simple to use (binary masks)
- ✅ Smaller download size (~500 images)
- ✅ Good for quick prototyping

---

### 3. Roboflow Universe Datasets
| Dataset | Images | Link |
|---------|--------|------|
| Leaf Disease Segmentation | 500+ | [Roboflow](https://universe.roboflow.com/search?q=leaf+disease+segmentation&type=instance-segmentation) |
| Plant Pathology | Various | [Roboflow](https://universe.roboflow.com/search?q=plant+pathology) |

**Why Choose This:**
- ✅ Pre-processed for YOLOv8 format
- ✅ Easy export to multiple formats
- ✅ Can combine multiple datasets

---

## 📥 How to Download and Prepare

### For PlantSeg Dataset:
```bash
# Option 1: Clone from GitHub
git clone https://github.com/mtang724/PlantSeg-dataset.git

# Option 2: Download from Zenodo
# Go to: https://zenodo.org/records/8413916
# Download the zip file
```

### For Kaggle Dataset:
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d fakhrealam9537/leaf-disease-segmentation-dataset
unzip leaf-disease-segmentation-dataset.zip -d data/leaf_seg
```

### For Roboflow:
1. Create free account at [roboflow.com](https://roboflow.com)
2. Find dataset → Click "Download"
3. Select format: "YOLOv8" → Download zip

---

## 🔧 Converting to YOLOv8 Format

### PlantSeg (PNG masks → YOLO polygons):
```python
import cv2
import numpy as np
from pathlib import Path

def mask_to_yolo_polygon(mask_path, class_id=1):
    """Convert binary mask to YOLO polygon format."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lines = []
    for contour in contours:
        if len(contour) < 3:
            continue
        
        # Simplify contour
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Normalize coordinates
        points = approx.squeeze()
        if len(points.shape) < 2:
            continue
            
        normalized = []
        for x, y in points:
            normalized.append(f"{x/w:.6f}")
            normalized.append(f"{y/h:.6f}")
        
        line = f"{class_id} " + " ".join(normalized)
        lines.append(line)
    
    return "\n".join(lines)

# Usage:
# for each mask file, generate corresponding .txt label file
```

---

## 🎯 Dataset Selection Guide

| Your Situation | Recommended Dataset |
|----------------|---------------------|
| Quick prototype | Kaggle (588 images) |
| Production model | PlantSeg (11,400 images) |
| Specific crops only | Roboflow (filter by crop) |
| Limited compute | Kaggle + augmentation |

---

## 📊 After Training: Pesticide Dosage Mapping

Once you have severity percentage from the model, map to pesticide dosage:

| Severity (%) | Spray Tier | Pesticide Concentration | Notes |
|--------------|------------|------------------------|-------|
| 0-5% | NO_ACTION | - | Healthy, no spray needed |
| 5-15% | LOW | 50% of label rate | Early intervention |
| 15-40% | MEDIUM | 75% of label rate | Standard treatment |
| 40-70% | HIGH | 100% of label rate | Aggressive treatment |
| >70% | CRITICAL | 100% + repeat in 7 days | Severe infection |

> ⚠️ **Important**: Always consult local agricultural guidelines for actual pesticide rates. These are example thresholds only.
