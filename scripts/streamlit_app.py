# streamlit_app.py
import streamlit as st
from pathlib import Path
import tempfile
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import tensorflow as tf
from ultralytics import YOLO
import io
import base64
from streamlit_paste_button import paste_image_button as pbutton

st.set_page_config(page_title="Plant Disease Pipeline", layout="wide")

# ---------------- Helpers ----------------
@st.cache_resource
def load_yolo_model(path: str):
    return YOLO(str(path))

@st.cache_resource
def load_mobilenet_model(path: str):
    # Ensure reproducible light logging
    tf.get_logger().setLevel('ERROR')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    model = tf.keras.models.load_model(str(path), compile=False)
    return model

def read_imagefile_to_bgr(uploaded_file) -> np.ndarray:
    # uploaded_file: BytesIO from streamlit
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def detect_leaves(yolo_model, image_bgr, conf=0.25, iou=0.45):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model.predict(source=img_rgb, conf=conf, iou=iou, verbose=False)
    detections = []
    for r in results:
        if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf_score = float(box.conf[0].cpu().numpy())
                cls_idx = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                detections.append({'xyxy': xyxy, 'conf': conf_score, 'cls': cls_idx})
    return detections

def crop_box(image_bgr, box, expand_ratio=0.03):
    h,w = image_bgr.shape[:2]
    x1,y1,x2,y2 = box
    dx = int((x2-x1) * expand_ratio)
    dy = int((y2-y1) * expand_ratio)
    nx1, ny1 = max(0,x1-dx), max(0,y1-dy)
    nx2, ny2 = min(w-1,x2+dx), min(h-1,y2+dy)
    crop = image_bgr[ny1:ny2, nx1:nx2]
    return crop, (nx1, ny1, nx2, ny2)

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
    if preds.max() > 1.01 or preds.min() < -0.01:
        probs = tf.nn.softmax(preds).numpy()
    else:
        probs = preds
    idx = int(np.argmax(probs))
    label = class_names[idx] if idx < len(class_names) else f"class_{idx}"
    conf = float(probs[idx])
    return label, conf, probs

def estimate_severity_heuristic(crop_bgr):
    """
    Improved severity estimation using multi-stage color analysis.
    
    This heuristic detects:
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
    
    # Convert to multiple color spaces for robust detection
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    
    # ===== STEP 1: Detect leaf area (exclude background) =====
    # Green + Yellow-green range (healthy leaf tissue)
    lower_leaf = np.array([15, 20, 20])
    upper_leaf = np.array([95, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)
    
    # Also include brown/tan areas as part of the leaf (diseased but still leaf)
    lower_brown = np.array([5, 30, 30])
    upper_brown = np.array([25, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Combine leaf masks
    full_leaf_mask = cv2.bitwise_or(leaf_mask, brown_mask)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    full_leaf_mask = cv2.morphologyEx(full_leaf_mask, cv2.MORPH_CLOSE, kernel)
    full_leaf_mask = cv2.morphologyEx(full_leaf_mask, cv2.MORPH_OPEN, kernel)
    
    leaf_pixels = cv2.countNonZero(full_leaf_mask)
    if leaf_pixels < 100:  # Too few pixels for reliable analysis
        return 0.0
    
    # ===== STEP 2: Detect healthy green areas =====
    lower_healthy = np.array([30, 40, 40])
    upper_healthy = np.array([85, 255, 255])
    healthy_mask = cv2.inRange(hsv, lower_healthy, upper_healthy)
    healthy_mask = cv2.bitwise_and(healthy_mask, full_leaf_mask)
    
    # ===== STEP 3: Detect diseased areas =====
    # 3a. Brown/tan spots (bacterial spot, early blight)
    lower_brown_disease = np.array([5, 50, 30])
    upper_brown_disease = np.array([25, 255, 180])
    brown_disease_mask = cv2.inRange(hsv, lower_brown_disease, upper_brown_disease)
    
    # 3b. Yellow/chlorotic areas (chlorosis, mosaic virus)
    lower_yellow = np.array([18, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 3c. Black/dark necrotic tissue
    lower_necrotic = np.array([0, 0, 0])
    upper_necrotic = np.array([180, 255, 50])
    necrotic_mask = cv2.inRange(hsv, lower_necrotic, upper_necrotic)
    
    # 3d. White/gray powdery areas (powdery mildew)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # ===== STEP 4: Combine disease masks =====
    disease_mask = cv2.bitwise_or(brown_disease_mask, yellow_mask)
    disease_mask = cv2.bitwise_or(disease_mask, necrotic_mask)
    disease_mask = cv2.bitwise_or(disease_mask, white_mask)
    
    # Only count disease pixels within the leaf area
    disease_mask = cv2.bitwise_and(disease_mask, full_leaf_mask)
    
    # Clean up small noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel_small)
    
    # ===== STEP 5: Calculate severity =====
    disease_pixels = cv2.countNonZero(disease_mask)
    healthy_pixels = cv2.countNonZero(healthy_mask)
    
    # Severity = diseased / (diseased + healthy)
    total_analyzed = disease_pixels + healthy_pixels
    if total_analyzed < 50:
        # Fallback: use full leaf area
        severity = (disease_pixels / leaf_pixels) * 100.0
    else:
        severity = (disease_pixels / total_analyzed) * 100.0
    
    # Clamp to valid range
    return float(np.clip(severity, 0.0, 100.0))

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
    count = len(infected)
    avg = float(np.mean([p['severity'] for p in infected]) if infected else 0.0)
    if count == 0 or avg < 5.0:
        tier = 'NO_ACTION'
    elif count < config['low_count'] and avg < config['low_severity']:
        tier = 'LOW'
    elif count < config['medium_count'] or avg < config['high_severity']:
        tier = 'MEDIUM'
    else:
        tier = 'HIGH'
    return {'infected_count': count, 'avg_severity': avg, 'spray_tier': tier}

def draw_annotated(image_bgr, detections, per_leaf):
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
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, max(0,y1-th-4), x1+tw+4, y1], fill="red")
        draw.text((x1+2, max(0,y1-th-2)), label, fill="white", font=font)
    return img

# ---------------- Streamlit UI ----------------

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2e7d32, #66bb6a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #4caf50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
    .stButton > button:hover {
        background-color: #388e3c;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🌿 Plant Disease Detection & Severity</p>', unsafe_allow_html=True)
st.markdown("**Upload a full-plant image** → The pipeline will detect leaves, classify disease per leaf, estimate severity, and propose a spray tier.")

# Initialize session state for models
if 'yolo_model' not in st.session_state:
    st.session_state.yolo_model = None
if 'mobilenet_model' not in st.session_state:
    st.session_state.mobilenet_model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Sidebar with organized sections
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Model Configuration Section
    with st.expander("📁 Model Paths", expanded=True):
        yolopath = st.text_input(
            "YOLOv8 model path (.pt)", 
            value="models/yolov8_leaf.pt",
            help="Path to the trained YOLOv8 model for leaf detection"
        )
        mobpath = st.text_input(
            "MobileNet .keras path", 
            value="models/mobilenetv2_disease.keras",
            help="Path to the trained MobileNetV2 model for disease classification"
        )
        class_names_path = st.text_input(
            "class_names.txt path", 
            value="data/class_names.txt",
            help="Path to the text file containing class names (one per line)"
        )
    
    # Detection Settings Section
    with st.expander("🎯 Detection Settings", expanded=True):
        st.markdown("#### YOLO Confidence Threshold")
        
        # Confidence guidance info box
        st.info("""
        **📊 Recommended Value: 0.25 - 0.50**
        
        • **0.25** (Default): Detects more leaves, may include some false positives
        • **0.35** (Balanced): Good balance between precision and recall
        • **0.50** (Conservative): High confidence only, may miss some leaves
        
        💡 *Start with 0.35 for most cases*
        """)
        
        conf_thresh = st.slider(
            "Confidence Threshold",
            min_value=0.05, 
            max_value=0.95, 
            value=0.35,  # Changed default to recommended value
            step=0.05,
            help="Higher = fewer detections but more confident. Lower = more detections but may include false positives."
        )
        
        # Visual indicator for current setting
        if conf_thresh < 0.25:
            st.warning("⚠️ Very low threshold - may detect too many false positives")
        elif conf_thresh <= 0.50:
            st.success("✅ Optimal range for most use cases")
        else:
            st.warning("⚠️ High threshold - may miss some leaves")
    
    st.markdown("---")
    
    # Load Models Button with better styling
    run_button = st.button("🚀 Load Models", use_container_width=True)
    
    # Model Status Indicator
    if st.session_state.models_loaded:
        st.success("✅ All models loaded and ready!")
    else:
        st.info("👆 Click 'Load Models' to start")
    
    # Help Section
    with st.expander("❓ Help & Tips", expanded=False):
        st.markdown("""
        **Quick Start Guide:**
        1. Verify model paths are correct
        2. Adjust confidence if needed
        3. Click 'Load Models'
        4. Upload or paste an image
        
        **Confidence Tips:**
        - Use **lower values (0.25-0.35)** for images with many small leaves
        - Use **higher values (0.40-0.50)** for close-up shots
        - If seeing false positives, increase the threshold
        - If missing leaves, decrease the threshold
        
        **Spray Tier Meanings:**
        - 🟢 **NO_ACTION**: Plant is healthy
        - 🟡 **LOW**: Minor infection, monitor
        - 🟠 **MEDIUM**: Moderate infection, treat soon
        - 🔴 **HIGH**: Severe infection, immediate action
        """)

# Load models on demand
yolo_model = None
mobilenet_model = None
class_names = []

if run_button:
    with st.sidebar:
        progress = st.progress(0, text="Initializing...")
        
        try:
            progress.progress(10, text="Loading YOLO model...")
            st.session_state.yolo_model = load_yolo_model(yolopath)
            progress.progress(40, text="YOLO loaded ✓")
        except Exception as e:
            st.error(f"❌ YOLO load error: {e}")
            st.session_state.models_loaded = False
            
        try:
            progress.progress(50, text="Loading MobileNet model...")
            st.session_state.mobilenet_model = load_mobilenet_model(mobpath)
            progress.progress(80, text="MobileNet loaded ✓")
        except Exception as e:
            st.error(f"❌ MobileNet load error: {e}")
            st.session_state.models_loaded = False
            
        try:
            progress.progress(90, text="Loading class names...")
            with open(class_names_path, 'r') as f:
                st.session_state.class_names = [l.strip() for l in f if l.strip()]
            if not st.session_state.class_names:
                st.warning("⚠️ class_names file is empty")
            else:
                progress.progress(100, text="All models loaded!")
                st.session_state.models_loaded = True
        except Exception as e:
            st.error(f"❌ class_names load error: {e}")
            st.session_state.models_loaded = False
        
        # Clear progress after short delay
        import time
        time.sleep(1)
        progress.empty()

# Use models from session state
yolo_model = st.session_state.yolo_model
mobilenet_model = st.session_state.mobilenet_model
class_names = st.session_state.class_names

st.markdown("---")
st.subheader("📷 Input Image")
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload full plant image (jpg/png)", type=["jpg","jpeg","png"])

with col2:
    st.write("**Or paste from clipboard:**")
    paste_result = pbutton(label="📋 Paste Image", key="paste_btn")

# Determine which image source to use
image_data = None
if paste_result and paste_result.image_data:
    image_data = paste_result.image_data
    st.success("✅ Image pasted from clipboard!")
elif uploaded_file is not None:
    image_data = uploaded_file

if image_data is not None:
    # require models loaded
    if yolo_model is None or mobilenet_model is None or not class_names:
        st.warning("Please enter model paths and click 'Load models' in the sidebar first.")
    else:
        try:
            # Handle both file upload and pasted image
            if hasattr(image_data, 'read'):
                # File upload
                img_bgr = read_imagefile_to_bgr(image_data)
            else:
                # Pasted image (PIL Image)
                img_rgb = np.array(image_data)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Input image", use_container_width=True)
            with st.spinner("Running detection + classification..."):
                dets = detect_leaves(yolo_model, img_bgr, conf=conf_thresh)
                per_leaf = []
                for d in dets:
                    crop, absbox = crop_box(img_bgr, d['xyxy'])
                    if crop is None or crop.size == 0:
                        lbl, conf, probs = "UNKNOWN", 0.0, None
                        sev = 0.0
                    else:
                        lbl, conf, probs = classify_crop(mobilenet_model, crop, class_names)
                        sev = estimate_severity_heuristic(crop)
                    per_leaf.append({
                        'class': lbl,
                        'conf': conf,
                        'severity': sev,
                        'box': absbox
                    })
                agg = aggregate_and_decide(per_leaf)
            
            # Results Section with better styling
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")
            
            # Spray Tier with color-coded badge
            tier = agg['spray_tier']
            tier_colors = {
                'NO_ACTION': ('🟢', '#4caf50', 'Healthy - No action needed'),
                'LOW': ('🟡', '#ffeb3b', 'Low risk - Monitor closely'),
                'MEDIUM': ('🟠', '#ff9800', 'Medium risk - Treatment recommended'),
                'HIGH': ('🔴', '#f44336', 'High risk - Immediate action required')
            }
            tier_emoji, tier_color, tier_desc = tier_colors.get(tier, ('⚪', '#9e9e9e', 'Unknown'))
            
            # Metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="🎯 Spray Tier",
                    value=f"{tier_emoji} {tier}",
                    help=tier_desc
                )
            with col2:
                st.metric(
                    label="🍃 Leaves Detected",
                    value=len(dets),
                    delta=f"{agg['infected_count']} infected" if agg['infected_count'] > 0 else None,
                    delta_color="inverse" if agg['infected_count'] > 0 else "normal"
                )
            with col3:
                st.metric(
                    label="📈 Avg Severity",
                    value=f"{agg['avg_severity']:.1f}%",
                    help="Average severity across infected leaves"
                )
            
            # Recommendation based on tier
            if tier == 'NO_ACTION':
                st.success(f"✅ **Recommendation:** {tier_desc}")
            elif tier == 'LOW':
                st.info(f"💡 **Recommendation:** {tier_desc}")
            elif tier == 'MEDIUM':
                st.warning(f"⚠️ **Recommendation:** {tier_desc}")
            else:
                st.error(f"🚨 **Recommendation:** {tier_desc}")
            
            st.markdown("---")
            
            # Annotated image with better caption
            st.markdown("### 🖼️ Annotated Detection")
            annotated = draw_annotated(img_bgr, dets, per_leaf)
            st.image(annotated, caption="Detected leaves with disease classification and severity", use_container_width=True)
            
            # per-leaf table with improved styling
            st.markdown("### 📋 Per-Leaf Analysis")
            rows = []
            for i, p in enumerate(per_leaf):
                x1,y1,x2,y2 = p['box']
                # Add severity indicator
                sev = p['severity']
                if sev < 5:
                    sev_status = "✅ Healthy"
                elif sev < 20:
                    sev_status = "🟡 Low"
                elif sev < 50:
                    sev_status = "🟠 Medium"
                else:
                    sev_status = "🔴 High"
                    
                rows.append({
                    'Leaf #': i + 1,
                    'Disease Class': p['class'],
                    'Confidence': f"{p['conf']*100:.1f}%",
                    'Severity': f"{sev:.1f}%",
                    'Status': sev_status
                })
            df = pd.DataFrame(rows)
            if df.empty:
                st.info("ℹ️ No leaves detected. Try adjusting the confidence threshold or using a clearer image.")
            else:
                # Display with custom styling
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Leaf #": st.column_config.NumberColumn("Leaf #", width="small"),
                        "Disease Class": st.column_config.TextColumn("Disease Class", width="medium"),
                        "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                        "Severity": st.column_config.TextColumn("Severity", width="small"),
                        "Status": st.column_config.TextColumn("Status", width="small"),
                    }
                )
                
                # Download button with better styling
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    csv_bytes = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Results as CSV",
                        data=csv_bytes,
                        file_name="plant_disease_analysis.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"Processing error: {e}")

st.markdown("---")
st.caption("Notes: severity uses a simple heuristic (non-green pixel ratio). For production replace with segmentation-based severity model and fine-tune classifier on cropped field leaves.")
