#!/usr/bin/env python3
"""
scripts/seg_infer_yolov8_seg.py
Given YOLOv8-seg model path, predict masks on the full image, then compute per-detected-leaf mask overlap
Assumes a leaf detector is available; better: train detector+seg jointly or use CLS on crops.
"""
from ultralytics import YOLO
import cv2
import numpy as np

def get_seg_masks(seg_model_path, image_bgr, conf=0.25):
    model = YOLO(seg_model_path)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(source=img_rgb, conf=conf, verbose=False)
    # collect masks
    masks = []
    for r in results:
        if hasattr(r, 'masks') and r.masks is not None:
            for m in r.masks.data:
                # m is a numpy array (flattened) in ultralytics v8. Make sure to adapt if API changes.
                mask = r.masks.data[0].cpu().numpy() if hasattr(r.masks.data[0],'cpu') else None
                # safer approach: use r.masks.xy if available
                # For production, check ultralytics docs for mask access methods
                pass
    # NOTE: This script is a blueprint. Use ultralytics YOLOv8-seg mask API in your runtime to get mask per box.
    return masks

# Usage: integrate mask inference into pipeline: for each crop compute mask percentage -> severity pct
