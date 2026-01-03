import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import pandas as pd

# --- Global Load ---
# Loading models globally for efficiency
yolo_model = YOLO("models/yolov8_leaf.pt")
seg_model = YOLO("models/yolov8_seg.pt")
mobilenet_model = tf.keras.models.load_model("models/mobilenetv2_disease.keras", compile=False)

with open("data/class_names.txt", "r") as f:
    class_names = [l.strip() for l in f if l.strip()]

# --- Functions ---

def get_full_image_disease_mask(seg_model, image_bgr, conf=0.25):
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = seg_model.predict(source=img_rgb, conf=conf, verbose=False)
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for r in results:
        if hasattr(r, 'masks') and r.masks is not None:
            masks_data = r.masks.data.cpu().numpy()
            for mask in masks_data:
                mask_resized = cv2.resize(mask.astype(np.float32), (w, h))
                binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                combined_mask = cv2.bitwise_or(combined_mask, binary_mask)
    return combined_mask

def estimate_severity_from_mask(disease_mask, box, image_bgr):
    x1, y1, x2, y2 = box
    h, w = disease_mask.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1: return 0.0
    mask_crop = disease_mask[y1:y2, x1:x2]
    leaf_crop = image_bgr[y1:y2, x1:x2]
    disease_pixels = cv2.countNonZero(mask_crop)
    gray = cv2.cvtColor(leaf_crop, cv2.COLOR_BGR2GRAY)
    _, leaf_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    leaf_pixels = cv2.countNonZero(leaf_mask)
    if leaf_pixels < 100: leaf_pixels = (x2 - x1) * (y2 - y1)
    return float(np.clip((disease_pixels / leaf_pixels) * 100.0, 0.0, 100.0))

def aggregate_and_decide(per_leaf):
    config = {
        'severity_min_for_infected': 3.0,
        'low_count': 2,
        'medium_count': 4,
        'low_severity': 10.0,
        'high_severity': 25.0,
        'critical_severity': 35.0,
        'critical_infected_ratio': 0.7
    }
    total_leaves = len(per_leaf)
    infected = [p for p in per_leaf if p['severity'] >= config['severity_min_for_infected']]
    count = len(infected)
    avg = float(np.mean([p['severity'] for p in infected]) if infected else 0.0)
    infected_ratio = count / total_leaves if total_leaves > 0 else 0
    
    if count == 0 or avg < 3.0: tier = 'NO_ACTION'
    elif avg >= config['critical_severity'] or infected_ratio >= config['critical_infected_ratio']: tier = 'CRITICAL'
    elif count >= config['medium_count'] and avg >= config['high_severity']: tier = 'HIGH'
    elif count >= config['low_count'] or avg >= config['low_severity']: tier = 'MEDIUM'
    else: tier = 'LOW'
    
    dosage = {'NO_ACTION': 0, 'LOW': 25, 'MEDIUM': 50, 'HIGH': 75, 'CRITICAL': 100}.get(tier, 50)
    return tier, dosage, count, avg

def infer(image):
    # image is a PIL image from Gradio
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 1. Detect Leaves
    results = yolo_model.predict(source=image, conf=0.35, verbose=False)
    dets = []
    for r in results:
        for box in r.boxes:
            dets.append(box.xyxy[0].cpu().numpy().astype(int).tolist())
    
    # 2. Get Overall Disease Mask
    disease_mask = get_full_image_disease_mask(seg_model, img_bgr)
    
    # 3. Process
    per_leaf = []
    if len(dets) == 0:
        # Fallback to whole image
        h, w = img_bgr.shape[:2]
        crop = cv2.resize(img_bgr, (224, 224)) / 255.0
        preds = mobilenet_model.predict(np.expand_dims(crop, axis=0), verbose=False)
        lbl = class_names[np.argmax(preds)]
        sev = estimate_severity_from_mask(disease_mask, [0, 0, w, h], img_bgr)
        per_leaf.append({'severity': sev})
    else:
        for box in dets:
            x1, y1, x2, y2 = box
            leaf_crop = img_bgr[y1:y2, x1:x2]
            if leaf_crop.size == 0: continue
            crop_resized = cv2.resize(leaf_crop, (224, 224)) / 255.0
            # Note: For simplicity, we just use the first leaf's class as primary or could aggregate
            sev = estimate_severity_from_mask(disease_mask, box, img_bgr)
            per_leaf.append({'severity': sev})

    tier, dosage, infected, avg_sev = aggregate_and_decide(per_leaf)
    
    return {
        "spray_tier": tier,
        "dosage_percent": dosage,
        "infected_leaves": infected,
        "average_severity": f"{avg_sev:.1f}%"
    }

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌿 Plant Disease Precision Spray API")
    gr.Markdown("This space provides a **JSON API** for hardware integration (Raspberry Pi/Jetson) and a simple UI for testing.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Upload Plant Image")
            run_btn = gr.Button("Analyze Image", variant="primary")
        
        with gr.Column():
            output_json = gr.JSON(label="Analysis Results (JSON)")
            
    gr.Markdown("### 📜 Hardware Integration Note")
    gr.Markdown("Use the **'Use via API'** link at the bottom of this page to see the code snippets for your Raspberry Pi.")
    
    run_btn.click(fn=infer, inputs=input_img, outputs=output_json)
    
    gr.Examples(
        examples=[], # Add example images if you want
        inputs=input_img
    )

if __name__ == "__main__":
    demo.launch()
