#!/usr/bin/env python3
"""
pipeline.py

Usage:
python pipeline.py \
  --yolo_path models/yolov8_leaf.pt \
  --mobilenet_path models/mobilenetv2_disease.keras \
  --class_names data/class_names.txt \
  --input_dir data/full_plants \
  --output_dir results \
  --conf 0.25

This script:
- Loads YOLOv8 detector (ultralytics)
- Loads MobileNetV2 .keras classifier (tf.keras)
- Runs detection -> crops -> classification -> simple severity heuristic
- Saves annotated images and results.csv
"""

import os
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from ultralytics import YOLO

# -------------- Utility helpers --------------
def ensure_file_exists(p, name):
    if not Path(p).exists():
        raise FileNotFoundError(f"{name} not found: {p}")

def load_class_names(path):
    ensure_file_exists(path, "class_names")
    with open(path, 'r') as f:
        classes = [l.strip() for l in f if l.strip()]
    if not classes:
        raise ValueError("class_names file is empty")
    return classes

# -------------- Model loaders --------------
def load_yolo(yolo_path):
    ensure_file_exists(yolo_path, "YOLO model")
    print(f"[INFO] Loading YOLOv8 model from: {yolo_path}")
    yolo = YOLO(str(yolo_path))
    return yolo

def load_mobilenet_kmodel(kmodel_path):
    ensure_file_exists(kmodel_path, "MobileNet .keras model")
    print(f"[INFO] Loading MobileNet model from: {kmodel_path}")
    # Prevent TF from printing too much and configure GPU memory growth if available
    tf.get_logger().setLevel('ERROR')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    # load model (compile=False handles custom objects absence)
    model = tf.keras.models.load_model(str(kmodel_path), compile=False)
    return model

# -------------- Detection & cropping --------------
def detect_leaves(yolo_model, image_bgr, conf=0.25, iou=0.45):
    # ultralytics YOLO predict expects RGB
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Predict returns a list (batch)
    results = yolo_model.predict(source=img_rgb, conf=conf, iou=iou, verbose=False)
    detections = []
    for r in results:
        if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)  # [x1,y1,x2,y2]
                conf_score = float(box.conf[0].cpu().numpy())
                cls_idx = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                detections.append({'xyxy': xyxy.tolist(), 'conf': conf_score, 'cls': cls_idx})
    return detections

def crop_box(image_bgr, box, expand_ratio=0.03):
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    dx = int((x2 - x1) * expand_ratio)
    dy = int((y2 - y1) * expand_ratio)
    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(w-1, x2 + dx)
    ny2 = min(h-1, y2 + dy)
    crop = image_bgr[ny1:ny2, nx1:nx2]
    return crop, (nx1, ny1, nx2, ny2)

# -------------- Preprocessing & classification --------------
def preprocess_for_mobilenet(crop_bgr, target_size=(224,224)):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(crop_rgb, target_size, interpolation=cv2.INTER_AREA)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, 0)
    return arr

def classify_crop(mobilenet_model, crop_bgr, class_names):
    x = preprocess_for_mobilenet(crop_bgr)
    preds = mobilenet_model.predict(x, verbose=0)[0]
    # If raw logits -> softmax
    if preds.max() > 1.01 or preds.min() < -0.01:
        probs = tf.nn.softmax(preds).numpy()
    else:
        probs = preds
    idx = int(np.argmax(probs))
    label = class_names[idx] if idx < len(class_names) else f"class_{idx}"
    conf = float(probs[idx])
    return label, conf, probs

# -------------- Improved severity estimation --------------
def estimate_severity_heuristic(crop_bgr):
    """
    Improved severity estimation using multi-stage color analysis.
    
    Detects:
    1. Brown/tan lesions (bacterial spots, blight)
    2. Yellow/chlorotic areas (nutrient deficiency, viral)
    3. Black/necrotic tissue (advanced disease)
    4. White/powdery areas (mildew)
    
    Returns severity percentage (0-100%)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    
    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 10:
        return 0.0
    
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    
    # STEP 1: Detect leaf area (exclude background)
    lower_leaf = np.array([15, 20, 20])
    upper_leaf = np.array([95, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)
    
    # Include brown/tan areas as part of the leaf
    lower_brown = np.array([5, 30, 30])
    upper_brown = np.array([25, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    full_leaf_mask = cv2.bitwise_or(leaf_mask, brown_mask)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    full_leaf_mask = cv2.morphologyEx(full_leaf_mask, cv2.MORPH_CLOSE, kernel)
    full_leaf_mask = cv2.morphologyEx(full_leaf_mask, cv2.MORPH_OPEN, kernel)
    
    leaf_pixels = cv2.countNonZero(full_leaf_mask)
    if leaf_pixels < 100:
        return 0.0
    
    # STEP 2: Detect healthy green areas
    lower_healthy = np.array([30, 40, 40])
    upper_healthy = np.array([85, 255, 255])
    healthy_mask = cv2.inRange(hsv, lower_healthy, upper_healthy)
    healthy_mask = cv2.bitwise_and(healthy_mask, full_leaf_mask)
    
    # STEP 3: Detect diseased areas
    # Brown/tan spots
    brown_disease_mask = cv2.inRange(hsv, np.array([5, 50, 30]), np.array([25, 255, 180]))
    # Yellow/chlorotic areas
    yellow_mask = cv2.inRange(hsv, np.array([18, 80, 80]), np.array([35, 255, 255]))
    # Black/necrotic tissue
    necrotic_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
    # White/powdery areas
    white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 40, 255]))
    
    # STEP 4: Combine disease masks
    disease_mask = cv2.bitwise_or(brown_disease_mask, yellow_mask)
    disease_mask = cv2.bitwise_or(disease_mask, necrotic_mask)
    disease_mask = cv2.bitwise_or(disease_mask, white_mask)
    disease_mask = cv2.bitwise_and(disease_mask, full_leaf_mask)
    
    # Cleanup noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel_small)
    
    # STEP 5: Calculate severity
    disease_pixels = cv2.countNonZero(disease_mask)
    healthy_pixels = cv2.countNonZero(healthy_mask)
    
    total_analyzed = disease_pixels + healthy_pixels
    if total_analyzed < 50:
        severity = (disease_pixels / leaf_pixels) * 100.0
    else:
        severity = (disease_pixels / total_analyzed) * 100.0
    
    return float(max(0.0, min(100.0, severity)))

# -------------- Aggregation & decision --------------
def aggregate_and_decide(per_leaf, config=None):
    if config is None:
        config = {
            'severity_min_for_infected': 5.0,
            'low_count': 3,
            'medium_count': 7,
            'low_severity': 15.0,
            'high_severity': 40.0
        }
    infected = [p for p in per_leaf if p['severity'] >= config['severity_min_for_infected']]
    infected_count = len(infected)
    avg_sev = float(np.mean([p['severity'] for p in infected]) if infected else 0.0)
    if infected_count == 0 or avg_sev < 5.0:
        tier = 'NO_ACTION'
    elif infected_count < config['low_count'] and avg_sev < config['low_severity']:
        tier = 'LOW'
    elif infected_count < config['medium_count'] or avg_sev < config['high_severity']:
        tier = 'MEDIUM'
    else:
        tier = 'HIGH'
    return {'infected_count': infected_count, 'avg_severity': avg_sev, 'spray_tier': tier}

# -------------- Visualization --------------
def visualize_and_save(image_bgr, detections, per_leaf, out_path):
    img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for det, info in zip(detections, per_leaf):
        x1,y1,x2,y2 = det['xyxy']
        label = f"{info['class']} {info['conf']:.2f} S:{info['severity']:.0f}%"
        draw.rectangle([x1,y1,x2,y2], outline="red", width=2)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, max(0,y1-text_h-4), x1+text_w+4, y1], fill="red")
        draw.text((x1+2, max(0,y1-text_h-2)), label, fill="white", font=font)
    img.save(out_path)

# -------------- Process single image --------------
def process_single_image(img_path, yolo_model, mobilenet_model, class_names, conf_thresh):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    detections = detect_leaves(yolo_model, img_bgr, conf=conf_thresh)
    per_leaf = []
    for d in detections:
        crop, absbox = crop_box(img_bgr, d['xyxy'])
        if crop.size == 0:
            lbl, conf, probs = "UNKNOWN", 0.0, None
            severity = 0.0
        else:
            try:
                lbl, conf, probs = classify_crop(mobilenet_model, crop, class_names)
            except Exception as e:
                lbl, conf, probs = "ERROR", 0.0, None
            severity = estimate_severity_heuristic(crop)
        per_leaf.append({
            'class': lbl,
            'conf': conf,
            'probs': probs,
            'severity': severity,
            'box': absbox
        })
    agg = aggregate_and_decide(per_leaf)
    return {
        'image_path': str(img_path),
        'detections': detections,
        'per_leaf': per_leaf,
        'aggregation': agg,
        'raw': img_bgr
    }

# -------------- Batch runner --------------
def run_batch(yolo_path, mobilenet_path, class_names_path, input_dir, output_dir, conf_thresh=0.25):
    # validate inputs
    ensure_file_exists(yolo_path, "YOLO model")
    ensure_file_exists(mobilenet_path, "MobileNet model")
    ensure_file_exists(class_names_path, "class_names file")
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        raise NotADirectoryError(f"Input dir not found: {input_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = out_dir / "annotated"
    annotated_dir.mkdir(exist_ok=True)
    # load models
    yolo = load_yolo(yolo_path)
    mobilenet = load_mobilenet_kmodel(mobilenet_path)
    class_names = load_class_names(class_names_path)

    # iterate images
    rows = []
    images = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg')])
    if not images:
        print("[WARN] No images found in input dir.")
    for img_path in images:
        try:
            res = process_single_image(img_path, yolo, mobilenet, class_names, conf_thresh)
        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")
            continue
        # build rows
        for idx, leaf in enumerate(res['per_leaf']):
            rows.append({
                'image': res['image_path'],
                'leaf_idx': idx,
                'label': leaf['class'],
                'label_conf': leaf['conf'],
                'severity_pct': leaf['severity'],
                'box_x1': int(leaf['box'][0]),
                'box_y1': int(leaf['box'][1]),
                'box_x2': int(leaf['box'][2]),
                'box_y2': int(leaf['box'][3]),
                'spray_tier_for_image': res['aggregation']['spray_tier'],
                'infected_count_for_image': res['aggregation']['infected_count'],
                'avg_severity_for_image': res['aggregation']['avg_severity']
            })
        # visualization
        out_img_path = annotated_dir / f"{img_path.stem}_annotated.jpg"
        visualize_and_save(res['raw'], res['detections'], res['per_leaf'], str(out_img_path))
        print(f"[INFO] Processed {img_path.name} -> {len(res['per_leaf'])} leaves, tier={res['aggregation']['spray_tier']}")

    # save CSV
    df = pd.DataFrame(rows)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] CSV saved to: {csv_path}")
    print(f"[INFO] Annotated images in: {annotated_dir}")
    return df

# -------------- CLI --------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--yolo_path', required=True, help="Path to YOLOv8 .pt model")
    p.add_argument('--mobilenet_path', required=True, help="Path to MobileNetV2 .keras model")
    p.add_argument('--class_names', required=True, help="Path to class_names.txt")
    p.add_argument('--input_dir', required=True, help="Folder with full-plant images")
    p.add_argument('--output_dir', default="results", help="Output folder")
    p.add_argument('--conf', type=float, default=0.25, help="YOLO detection confidence threshold")
    args = p.parse_args()

    run_batch(args.yolo_path, args.mobilenet_path, args.class_names, args.input_dir, args.output_dir, conf_thresh=args.conf)

if __name__ == '__main__':
    main()
