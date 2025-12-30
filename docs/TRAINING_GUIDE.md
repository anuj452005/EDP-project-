# YOLOv8-Seg Training Guide for Plant Disease Segmentation

This guide walks you through training a YOLOv8 segmentation model to detect and segment diseased areas on leaves for **precise severity estimation**.

---

## 📋 Prerequisites

```bash
pip install ultralytics labelme albumentations
```

---

## Step 1: Collect & Annotate Images

### 1.1 Image Collection
You need **100-500+ images** of plant leaves showing various disease states:
- Healthy leaves (multiple angles, lighting conditions)
- Diseased leaves at different severity levels
- Multiple disease types if applicable

### 1.2 Annotation with LabelMe
[LabelMe](https://github.com/labelmeai/labelme) is a free polygon annotation tool.

**Install:**
```bash
pip install labelme
labelme   # Launches the GUI
```

**Annotation Process:**
1. Open LabelMe → File → Open Dir → Select your images folder
2. For each image:
   - Draw **polygon** around healthy leaf areas → Label: `healthy`
   - Draw **polygon** around diseased areas → Label: `diseased` (or specific disease name)
3. Save (creates `.json` file per image)

**Example class structure:**
| Class ID | Label | Description |
|----------|-------|-------------|
| 0 | healthy | Healthy green leaf tissue |
| 1 | bacterial_spot | Bacterial spot lesions |
| 2 | early_blight | Early blight symptoms |
| 3 | late_blight | Late blight symptoms |

> [!TIP]
> For simpler training, use just 2 classes: `healthy` and `diseased`

---

## Step 2: Prepare Dataset Structure

```
data/
├── disease_segmentation/
│   ├── images/
│   │   ├── train/     (80% of images)
│   │   └── val/       (20% of images)
│   └── labels/
│       ├── train/     (.txt files, same names as images)
│       └── val/
└── disease_seg.yaml
```

### 2.1 Convert LabelMe JSON to YOLO Format

```python
# In Python or add to train_yolov8_seg.py
from scripts.train_yolov8_seg import convert_labelme_to_yolo

# Your class names (must match LabelMe labels exactly)
classes = ['healthy', 'bacterial_spot', 'early_blight', 'late_blight']

# Convert training annotations
convert_labelme_to_yolo(
    labelme_json_dir='data/annotations/train',
    output_dir='data/disease_segmentation/labels/train',
    class_names=classes
)

# Convert validation annotations
convert_labelme_to_yolo(
    labelme_json_dir='data/annotations/val',
    output_dir='data/disease_segmentation/labels/val',
    class_names=classes
)
```

### 2.2 Create Dataset YAML

Create `data/disease_seg.yaml`:
```yaml
path: data/disease_segmentation
train: images/train
val: images/val

names:
  0: healthy
  1: bacterial_spot
  2: early_blight
  3: late_blight
```

---

## Step 3: Train the Model

### Option A: Command Line
```bash
# Nano model (fastest, lower accuracy) - good for testing
python scripts/train_yolov8_seg.py \
  --data data/disease_seg.yaml \
  --epochs 100 \
  --model n \
  --batch 16

# Medium model (balanced) - recommended for production
python scripts/train_yolov8_seg.py \
  --data data/disease_seg.yaml \
  --epochs 200 \
  --model m \
  --batch 8

# Large model (highest accuracy, slower)
python scripts/train_yolov8_seg.py \
  --data data/disease_seg.yaml \
  --epochs 300 \
  --model l \
  --batch 4
```

### Option B: Python Script
```python
from ultralytics import YOLO

# Load pretrained segmentation model
model = YOLO('yolov8m-seg.pt')

# Train
results = model.train(
    data='data/disease_seg.yaml',
    epochs=200,
    imgsz=640,
    batch=8,
    augment=True
)

# Best model saved at: runs/segment/train/weights/best.pt
```

### Training Tips
| Issue | Solution |
|-------|----------|
| Out of Memory | Reduce `--batch` (try 8, 4, or 2) |
| Low accuracy | Increase epochs, use larger model (m, l, x) |
| Overfitting | Add more training data, enable augmentation |
| Slow training | Use smaller model (n, s), reduce image size |

---

## Step 4: Validate & Test

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/segment/disease_seg/weights/best.pt')

# Validate on test set
metrics = model.val(data='data/disease_seg.yaml')
print(f"Mask mAP50: {metrics.seg.map50:.4f}")

# Test on single image
results = model.predict('test_leaf.jpg', save=True)
```

---

## Step 5: Deploy to Your App

### 5.1 Copy Model
```bash
copy runs\segment\disease_seg\weights\best.pt models\yolov8_disease_seg.pt
```

### 5.2 Update Streamlit App

Add to `streamlit_app.py`:
```python
def estimate_severity_segmentation(seg_model, crop_bgr):
    """
    Use YOLOv8-seg for precise severity estimation.
    """
    import cv2
    import numpy as np
    
    results = seg_model.predict(source=crop_bgr, conf=0.25, verbose=False)
    
    for r in results:
        if r.masks is None:
            return 0.0
        
        h, w = crop_bgr.shape[:2]
        masks = r.masks.data.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        
        diseased_area = 0
        healthy_area = 0
        
        for mask, cls in zip(masks, classes):
            mask_resized = cv2.resize(mask, (w, h))
            pixel_count = np.sum(mask_resized > 0.5)
            
            if int(cls) == 0:  # healthy class
                healthy_area += pixel_count
            else:  # diseased classes
                diseased_area += pixel_count
        
        total = healthy_area + diseased_area
        if total == 0:
            return 0.0
        
        severity = (diseased_area / total) * 100.0
        return float(severity)
    
    return 0.0
```

---

## 📊 Expected Results

| Model | mAP50 (Mask) | Inference Time | Recommended Use |
|-------|--------------|----------------|-----------------|
| YOLOv8n-seg | 0.70-0.80 | ~5ms | Mobile, real-time |
| YOLOv8s-seg | 0.75-0.85 | ~8ms | Edge devices |
| YOLOv8m-seg | 0.80-0.90 | ~15ms | **Production** |
| YOLOv8l-seg | 0.85-0.92 | ~25ms | High accuracy needs |

---

## 🔗 Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [LabelMe GitHub](https://github.com/labelmeai/labelme)
- [Roboflow (Online Annotation)](https://roboflow.com/) - Alternative to LabelMe
- [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) - Public plant disease images
