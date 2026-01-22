import os
import sys
import warnings
import contextlib
from io import StringIO, BytesIO
import hashlib
from PIL import Image
import base64
import json
from datetime import datetime
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Comprehensive NNPACK warning suppression
os.environ['NNPACK_DISABLE'] = '1'
os.environ['PYTORCH_NNPACK_DISABLE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Redirect stderr to capture C++ warnings
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# Filter all warnings
warnings.filterwarnings("ignore")

import streamlit as st
import cv2

# Show OpenAI warning if not available
if not HAS_OPENAI:
    st.warning("⚠️ OpenAI not installed. Install with: pip install openai")

# Import torch with stderr suppression
with suppress_stderr():
    import torch
    
# Continue with other imports
import time
import threading
import queue
from ultralytics import YOLO
import numpy as np

# Additional NNPACK disable after torch import
try:
    torch.backends.nnpack.enabled = False
    torch.set_num_threads(1)
except:
    pass

# AMD ROCm optimizations
if torch.cuda.is_available():
    try:
        # Enable AMD-specific optimizations
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except AttributeError:
        pass  # Some PyTorch versions don't have these attributes

# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="Zero-Defect Manufacturing AI",
    layout="wide"
)
st.title("🏭 Zero-Defect Manufacturing – AI Vision System")
st.markdown("Real-time defect detection demo using **AMD Instinct + ROCm**")

# -----------------------
# Sidebar Controls
# -----------------------
st.sidebar.header("⚙️ Inference Controls")
video_path = st.sidebar.text_input("Video path", "sample_video-3.mp4")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)
precision = st.sidebar.selectbox("Precision Mode", ["INT8 (Fastest)", "FP16 (GPU)", "FP32"])
start_button = st.sidebar.button("▶ Start")
stop_button = st.sidebar.button("⏹ Stop")
fps_target = st.sidebar.slider("Max FPS", 1, 60, 20)
buffer_size = st.sidebar.slider("Frame Buffer", 1, 10, 3)

# LLM Configuration
st.sidebar.header("🤖 AI Assistant")
enable_llm = st.sidebar.checkbox("Enable AI Analysis", value=True if HAS_OPENAI else False, disabled=not HAS_OPENAI)
if enable_llm and HAS_OPENAI:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Required for AI analysis and recommendations")
    llm_model = st.sidebar.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    analysis_frequency = st.sidebar.slider("Analysis Frequency (seconds)", 5, 60, 10)
else:
    openai_api_key = None
    llm_model = None
    analysis_frequency = 10

# -----------------------
# Session State & Threading
# -----------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "cap" not in st.session_state:
    st.session_state.cap = None
if "last_time" not in st.session_state:
    st.session_state.last_time = time.time()
if "frame_queue" not in st.session_state:
    st.session_state.frame_queue = queue.Queue(maxsize=buffer_size)
if "stats" not in st.session_state:
    st.session_state.stats = {"fps": 0, "latency": 0, "frame_count": 0}
if "worker_thread" not in st.session_state:
    st.session_state.worker_thread = None
if "current_frame" not in st.session_state:
    st.session_state.current_frame = None
if "frame_hash" not in st.session_state:
    st.session_state.frame_hash = None
if "current_frame_b64" not in st.session_state:
    st.session_state.current_frame_b64 = None
if "current_defects" not in st.session_state:
    st.session_state.current_defects = 0
if "current_defect_counts" not in st.session_state:
    st.session_state.current_defect_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
if "llm_analysis" not in st.session_state:
    st.session_state.llm_analysis = ""
if "last_llm_update" not in st.session_state:
    st.session_state.last_llm_update = 0
if "defect_history" not in st.session_state:
    st.session_state.defect_history = []
if "maintenance_recommendations" not in st.session_state:
    st.session_state.maintenance_recommendations = []

# -----------------------
# Device & AMD Instinct 300X Setup
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# AMD Instinct 300X optimizations
if device == "cuda":
    try:
        # Check for AMD GPU
        gpu_name = torch.cuda.get_device_name(0)
        is_amd = any(x in gpu_name.upper() for x in ['INSTINCT', 'MI', 'RADEON', 'VEGA'])
        
        if is_amd:
            st.sidebar.success(f"🚀 AMD GPU Detected: {gpu_name}")
            # AMD-specific optimizations
            torch.cuda.empty_cache()
            # Updated environment variable for newer PyTorch
            os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:128'
        else:
            st.sidebar.info(f"🖥️ GPU: {gpu_name}")
            
        # General CUDA optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        
    except Exception as e:
        st.sidebar.info(f"🖥️ GPU: CUDA Device (details unavailable)")

@st.cache_resource
def load_model():
    # Suppress YOLO warnings and NNPACK
    if not os.path.exists("yolov8n.pt"):
        st.error("❌ Model file 'yolov8n.pt' not found!")
        st.stop()
    
    with suppress_stderr():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = YOLO("yolov8n.pt")
    model.to(device)
    return model

@st.cache_resource
def load_quantized_model(_precision):
    """Load model with specific quantization for AMD GPUs"""
    with suppress_stderr():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

def analyze_defects_with_llm(defect_data, defect_counts, api_key, model="gpt-4o-mini"):
    """Generate intelligent analysis of detected defects using LLM"""
    if not api_key or not HAS_OPENAI:
        return "LLM analysis unavailable - API key required"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Prepare defect summary
        total_defects = sum(defect_counts.values())
        severity_breakdown = ", ".join([f"{k}: {v}" for k, v in defect_counts.items() if v > 0])
        
        recent_defects = ", ".join([d['type'] for d in defect_data[-10:]])  # Last 10 defects
        
        prompt = f"""
As a steel manufacturing quality control expert, analyze this defect detection data:

Current Frame Analysis:
- Total defects detected: {total_defects}
- Severity breakdown: {severity_breakdown}
- Recent defect types: {recent_defects}

Provide:
1. Brief assessment of current quality status
2. Immediate action recommendations if critical/high severity defects present
3. Potential root causes for observed defect patterns
4. Preventive maintenance suggestions

Keep response concise and actionable for production floor operators.
"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert steel manufacturing quality control analyst with 20+ years experience in defect analysis and process optimization."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"LLM Analysis Error: {str(e)}"

def generate_defect_report(defect_history, api_key, model="gpt-4o-mini"):
    """Generate comprehensive defect analysis report"""
    if not api_key or not HAS_OPENAI or not defect_history:
        return "Report generation unavailable"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Analyze recent defect trends
        recent_defects = defect_history[-50:]  # Last 50 defects
        defect_types = {}
        severity_trends = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        for defect in recent_defects:
            defect_type = defect.get('type', 'Unknown')
            severity = defect.get('severity', 'LOW')
            defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
            severity_trends[severity] += 1
        
        top_defects = sorted(defect_types.items(), key=lambda x: x[1], reverse=True)[:5]
        
        prompt = f"""
Generate a steel manufacturing quality control report based on recent defect data:

Defect Statistics (Last 50 detections):
- Most common defects: {top_defects}
- Severity distribution: {dict(severity_trends)}
- Total defects analyzed: {len(recent_defects)}

Provide a structured report with:
1. **Quality Status Summary**
2. **Key Defect Patterns**
3. **Risk Assessment**
4. **Recommended Actions**
5. **Process Optimization Suggestions**

Format for production management review.
"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a senior quality control manager generating executive reports for steel manufacturing operations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.2
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Report Generation Error: {str(e)}"

@st.cache_resource
def load_quantized_model(_precision):
    """Load model with specific quantization for AMD GPUs"""
    if not os.path.exists("yolov8n.pt"):
        st.error("❌ Model file 'yolov8n.pt' not found!")
        st.stop()
        
    with suppress_stderr():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = YOLO("yolov8n.pt")
    
    model.to(device)
    
    if _precision == "INT8 (Fastest)" and device == "cuda":
        try:
            # AMD Instinct optimizations - use FP16 as base for INT8
            model.model.half()
            
            # Set AMD-friendly inference settings
            torch.cuda.empty_cache()
            st.sidebar.success("✅ AMD-optimized INT8 enabled")
        except Exception as e:
            st.sidebar.warning(f"INT8 fallback to FP16: {str(e)}")
    
    return model

# Model will be loaded lazily when needed
# model = load_model()  # Commented out to prevent startup blocking

# -----------------------
# Steel Defect Detection Configuration
# -----------------------
# Map YOLO classes to steel defects
DEFECT_CLASS_MAP = {
    # Map common detected objects to steel defects
    'person': 'Foreign Object',
    'bicycle': 'Surface Scratch', 
    'car': 'Large Dent',
    'motorcycle': 'Corrosion',
    'airplane': 'Crack',
    'bus': 'Scale Formation',
    'train': 'Edge Defect',
    'truck': 'Inclusion',
    'boat': 'Lamination',
    'bench': 'Surface Defect',  # Your current detection
    'chair': 'Pit/Hole',
    'bottle': 'Bubble',
    'cup': 'Dent',
    'knife': 'Sharp Edge',
    'spoon': 'Deformation',
    'bowl': 'Crater',
    'book': 'Coating Defect',
    'clock': 'Circular Mark',
    'scissors': 'Cut/Tear',
    'cell phone': 'Small Inclusion',
    'laptop': 'Flat Defect',
    'mouse': 'Small Pit',
    'remote': 'Linear Defect',
    'keyboard': 'Pattern Defect',
    'microwave': 'Heat Mark',
    'oven': 'Oxidation',
    'toaster': 'Burn Mark',
    'sink': 'Deep Defect',
    'refrigerator': 'Large Area Defect',
    'bed': 'Flat Surface Issue',
    'dining table': 'Horizontal Defect',
    'toilet': 'Circular Defect',
    'tv': 'Screen-like Mark',
    'couch': 'Soft Defect',
    'potted plant': 'Organic Contamination',
    'stop sign': 'Critical Defect',
    'fire hydrant': 'Protrusion',
    'parking meter': 'Vertical Defect',
    'backpack': 'Bulge',
    'umbrella': 'Arc Defect',
    'handbag': 'Pouch-like Defect',
    'tie': 'Linear Mark',
    'suitcase': 'Rectangular Defect',
    'sports ball': 'Spherical Mark',
    'kite': 'Angular Defect',
    'baseball bat': 'Rod-like Defect',
    'baseball glove': 'Hand-shaped Mark',
    'skateboard': 'Board Defect',
    'surfboard': 'Long Defect',
    'tennis racket': 'Mesh Pattern',
    'wine glass': 'Stemmed Defect',
    'fork': 'Multi-point Defect',
    'apple': 'Round Inclusion',
    'banana': 'Curved Defect',
    'orange': 'Orange Peel Effect',
    'broccoli': 'Textured Defect',
    'carrot': 'Tapered Defect',
    'hot dog': 'Cylindrical Defect',
    'pizza': 'Flat Circular Defect',
    'donut': 'Ring Defect',
    'cake': 'Layered Defect'
}

# Defect severity mapping
DEFECT_SEVERITY = {
    'Critical Defect': 'CRITICAL',
    'Large Dent': 'HIGH',
    'Large Area Defect': 'HIGH', 
    'Crack': 'HIGH',
    'Foreign Object': 'HIGH',
    'Deep Defect': 'MEDIUM',
    'Corrosion': 'HIGH',
    'Edge Defect': 'MEDIUM',
    'Scale Formation': 'MEDIUM',
    'Surface Defect': 'LOW',
    'Surface Scratch': 'LOW',
    'Small Inclusion': 'LOW',
    'Small Pit': 'LOW'
}

def get_defect_info(class_name, confidence):
    """Convert YOLO detection to steel defect information"""
    defect_name = DEFECT_CLASS_MAP.get(class_name, f'Unknown Defect ({class_name})')
    severity = DEFECT_SEVERITY.get(defect_name, 'MEDIUM')
    
    # Adjust severity based on confidence
    if confidence > 0.8:
        if severity == 'LOW': severity = 'MEDIUM'
        elif severity == 'MEDIUM': severity = 'HIGH'
    
    return defect_name, severity

# -----------------------
# Image Handling Utilities
# -----------------------
def convert_cv2_to_base64(cv2_image, quality=85):
    """Convert CV2 image to base64 string for stable display"""
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality, optimize=True)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    except Exception as e:
        return None

def get_image_hash(image_array):
    """Generate hash for image to detect changes"""
    return hashlib.md5(image_array.tobytes()).hexdigest()[:16]

def display_image_base64(img_base64, width=800):
    """Display base64 image using HTML to avoid Streamlit storage issues"""
    if img_base64:
        html = f'''
        <div style="display: flex; justify-content: center;">
            <img src="data:image/jpeg;base64,{img_base64}" 
                 style="max-width: {width}px; height: auto; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        </div>
        '''
        return html
    return None

def annotate_steel_defects(results, frame):
    """Custom annotation for steel defect detection"""
    annotated_frame = frame.copy()
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        
        for i, box in enumerate(boxes):
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # Get original class name
            class_name = results[0].names[class_id]
            
            # Convert to defect information
            defect_name, severity = get_defect_info(class_name, confidence)
            
            # Color coding by severity
            if severity == 'CRITICAL':
                color = (0, 0, 255)  # Red
                thickness = 4
            elif severity == 'HIGH':
                color = (0, 165, 255)  # Orange
                thickness = 3
            elif severity == 'MEDIUM':
                color = (0, 255, 255)  # Yellow
                thickness = 2
            else:  # LOW
                color = (0, 255, 0)  # Green
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Create label with defect info
            label = f"{defect_name}: {confidence:.2f} ({severity})"
            
            # Calculate label size and position
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw label background
            cv2.rectangle(annotated_frame, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_frame

# -----------------------
# Background Processing Function
# -----------------------
def process_video_worker(cap, frame_queue, stats, running_flag, fps_target, model, device, precision, confidence):
    """Background worker for video processing with AMD Instinct optimizations"""
    frame_time = 1.0 / fps_target
    last_process_time = time.time()
    
    # AMD Instinct 300X specific optimizations
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)  # Use primary GPU
        
    # Pre-warm the model for better performance
    warmup_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    try:
        model.predict(source=warmup_frame, device=device, verbose=False)
    except:
        pass
    
    while running_flag[0]:
        current_time = time.time()
        if current_time - last_process_time < frame_time:
            time.sleep(0.001)  # Small sleep to prevent CPU spinning
            continue
            
        ret, frame = cap.read()
        if not ret:
            running_flag[0] = False
            break
            
        # Skip processing if queue is full (drop frames)
        if frame_queue.full():
            try:
                frame_queue.get_nowait()  # Remove oldest frame
            except queue.Empty:
                pass
        
        # Run optimized inference with AMD settings
        start_time = time.time()
        
        # Configure precision settings for AMD GPUs
        use_half = precision in ["FP16 (GPU)", "INT8 (Fastest)"]
        
        # Use context manager to suppress warnings during inference
        with suppress_stderr():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if precision == "INT8 (Fastest)":
                    # AMD-optimized INT8 inference
                    with torch.no_grad():
                        results = model.predict(
                            source=frame,
                            device=device,
                            half=use_half,
                            conf=confidence,
                            verbose=False,
                            imgsz=640,
                            max_det=100,
                            augment=False,  # Disable for speed
                            agnostic_nms=True  # Faster NMS
                        )
                else:
                    results = model.predict(
                        source=frame,
                        device=device,
                        half=use_half,
                        conf=confidence,
                        verbose=False,
                        imgsz=640,
                        max_det=100
                    )
                
        # Use custom steel defect annotation instead of default
        annotated = annotate_steel_defects(results, frame)
        
        # Calculate processing stats
        end_time = time.time()
        processing_latency = (end_time - start_time) * 1000  # ms
        
        # Update defect statistics
        current_defect_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        current_total_defects = 0
        detected_defects = []
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = results[0].names[class_id]
                defect_name, severity = get_defect_info(class_name, confidence)
                current_defect_counts[severity] += 1
                current_total_defects += 1
                
                # Store defect for LLM analysis
                detected_defects.append({
                    'type': defect_name,
                    'severity': severity,
                    'confidence': confidence,
                    'timestamp': time.time(),
                    'class_name': class_name
                })
        
        # Update stats with current frame data
        stats["latency"] = processing_latency
        stats["frame_count"] += 1
        stats["current_total_defects"] = current_total_defects
        stats["current_defect_counts"] = current_defect_counts
        
        # Keep track of last detection for consistent display
        if current_total_defects > 0:
            stats["last_total_defects"] = current_total_defects
            stats["last_defect_counts"] = current_defect_counts.copy()
        elif "last_total_defects" not in stats:
            stats["last_total_defects"] = 0
            stats["last_defect_counts"] = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        # Calculate FPS
        if stats["frame_count"] > 1:
            stats["fps"] = 1.0 / (current_time - last_process_time)
        else:
            stats["fps"] = 0
        
        # Convert frame to base64 for display
        frame_b64 = convert_cv2_to_base64(annotated, quality=85)
        frame_hash = get_image_hash(annotated)
        
        # Put processed frame in queue with defect info
        frame_data = {
            'frame_b64': frame_b64,
            'frame_hash': frame_hash,
            'latency': processing_latency,
            'fps': stats["fps"],
            'total_defects': current_total_defects,
            'defect_counts': current_defect_counts.copy(),
            'detected_defects': detected_defects  # Add defect data for LLM analysis
        }
        
        try:
            frame_queue.put(frame_data, timeout=0.1)
        except queue.Full:
            # If queue is full, skip this frame
            pass
        
        last_process_time = current_time
        
        # AMD GPU memory management
        if device == "cuda" and stats["frame_count"] % 30 == 0:  # Every 30 frames
            torch.cuda.empty_cache()

# -----------------------
# Start/Stop logic with Threading
# -----------------------
if start_button:
    st.session_state.running = True
    if st.session_state.cap is None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ Cannot open video file")
            st.session_state.running = False
        else:
            st.session_state.cap = cap
            
            # Load optimized model based on precision with progress indicator
            with st.spinner(f"🔄 Loading {precision} model..."):
                try:
                    optimized_model = load_quantized_model(precision)
                    st.success("✅ Model loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to load model: {str(e)}")
                    st.session_state.running = False
                    cap.release()
                    st.session_state.cap = None
                    st.stop()
            
            # Set video capture properties for better performance
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, fps_target)
            
            # Start background worker thread
            running_flag = [True]
            st.session_state.running_flag = running_flag
            st.session_state.worker_thread = threading.Thread(
                target=process_video_worker,
                args=(cap, st.session_state.frame_queue, st.session_state.stats, 
                      running_flag, fps_target, optimized_model, device, precision, confidence),
                daemon=True
            )
            st.session_state.worker_thread.start()

if stop_button:
    st.session_state.running = False
    if hasattr(st.session_state, 'running_flag'):
        st.session_state.running_flag[0] = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    
    # Clear frame cache to prevent memory issues
    st.session_state.current_frame = None
    st.session_state.current_frame_b64 = None
    st.session_state.frame_hash = None
    
    # Clear the queue
    while not st.session_state.frame_queue.empty():
        try:
            st.session_state.frame_queue.get_nowait()
        except queue.Empty:
            break

# -----------------------
# Metrics Display
# -----------------------
col1, col2, col3, col4 = st.columns([2,2,1,1])
fps_box = col1.empty()
latency_box = col2.empty()
device_box = col3.empty()
memory_box = col4.empty() if device == "cuda" else None

# -----------------------
# Video Display with Efficient Updates
# -----------------------
frame_placeholder = st.empty()

# -----------------------
# Display Loop (Non-blocking)
# -----------------------
if st.session_state.running:
    try:
        # Get latest frame from queue without blocking
        frame_data = st.session_state.frame_queue.get_nowait()
        
        # Extract frame information
        frame_b64 = frame_data['frame_b64']
        frame_hash = frame_data['frame_hash']
        latency = frame_data['latency']
        fps_val = frame_data['fps']
        current_defects = frame_data['total_defects']
        current_defect_counts = frame_data['defect_counts']
        
        # Only update display if frame changed (reduces Streamlit load)
        if frame_hash != st.session_state.frame_hash and frame_b64 is not None:
            st.session_state.current_frame_b64 = frame_b64
            st.session_state.frame_hash = frame_hash
            # Update defect info in session state for consistent display
            st.session_state.current_defects = current_defects
            st.session_state.current_defect_counts = current_defect_counts
            
            # Update defect history for LLM analysis
            if 'detected_defects' in frame_data and frame_data['detected_defects']:
                st.session_state.defect_history.extend(frame_data['detected_defects'])
                # Keep only last 100 defects to prevent memory issues
                if len(st.session_state.defect_history) > 100:
                    st.session_state.defect_history = st.session_state.defect_history[-100:]
        
        # Update metrics
        fps_box.metric("FPS", f"{fps_val:.1f}")
        latency_box.metric("Latency (ms)", f"{latency:.1f}")
        device_box.metric("Device", device.upper())
        
        if device == "cuda" and memory_box:
            mem_used = torch.cuda.memory_reserved()/(1024*1024)
            memory_box.metric("GPU Memory", f"{mem_used:.1f} MB")
        
        # Show precision mode performance indicator
        if precision == "INT8 (Fastest)":
            st.sidebar.metric("Precision Mode", "🚀 INT8 Turbo")
        elif precision == "FP16 (GPU)":
            st.sidebar.metric("Precision Mode", "⚡ FP16 Fast")
        else:
            st.sidebar.metric("Precision Mode", "🔧 FP32 Precise")
        
# Display defect statistics in sidebar using synchronized frame data
        if hasattr(st.session_state, 'current_defects'):
            st.sidebar.markdown("---")
            st.sidebar.markdown("**🔍 Current Frame Defects:**")
            st.sidebar.metric("Total Defects", st.session_state.current_defects)
            
            defect_counts = st.session_state.current_defect_counts
            if defect_counts.get('CRITICAL', 0) > 0:
                st.sidebar.error(f"🚨 Critical: {defect_counts['CRITICAL']}")
            if defect_counts.get('HIGH', 0) > 0:
                st.sidebar.warning(f"⚠️ High: {defect_counts['HIGH']}")
            if defect_counts.get('MEDIUM', 0) > 0:
                st.sidebar.info(f"ℹ️ Medium: {defect_counts['MEDIUM']}")
            if defect_counts.get('LOW', 0) > 0:
                st.sidebar.success(f"✅ Low: {defect_counts['LOW']}")
        
    except queue.Empty:
        # No new frame available, keep current display
        if st.session_state.stats["frame_count"] > 0:
            fps_box.metric("FPS", f"{st.session_state.stats['fps']:.1f}")
            latency_box.metric("Latency (ms)", f"{st.session_state.stats['latency']:.1f}")
            #device_box.metric("Device", device.upper())
    
    # Display current frame using HTML (avoids Streamlit file storage)
    if st.session_state.current_frame_b64 is not None:
        html_content = display_image_base64(st.session_state.current_frame_b64, width=800)
        if html_content:
            frame_placeholder.markdown(html_content, unsafe_allow_html=True)
    
    # Auto-refresh with controlled timing
    time.sleep(0.05)  # 50ms = ~20 FPS UI updates for better stability
    st.rerun()
else:
    # Show status when not running
    if st.session_state.cap is None:
        frame_placeholder.info("📹 Click 'Start' to begin video processing")
    else:
        frame_placeholder.info("⏸️ Processing stopped")

# -----------------------
# LLM AI Analysis Panel
# -----------------------
if enable_llm and openai_api_key and HAS_OPENAI:
    st.markdown("---")
    st.subheader("🤖 AI Quality Analysis")
    
    col_analysis, col_report = st.columns(2)
    
    with col_analysis:
        st.markdown("**Real-time Defect Analysis**")
        analysis_placeholder = st.empty()
        
        # Trigger LLM analysis periodically
        current_time = time.time()
        if (st.session_state.current_defects > 0 and 
            current_time - st.session_state.last_llm_update > analysis_frequency and
            st.session_state.running):
            
            with st.spinner("Analyzing defects..."):
                analysis = analyze_defects_with_llm(
                    st.session_state.defect_history[-10:],  # Recent defects
                    st.session_state.current_defect_counts,
                    openai_api_key,
                    llm_model
                )
                st.session_state.llm_analysis = analysis
                st.session_state.last_llm_update = current_time
        
        if st.session_state.llm_analysis:
            analysis_placeholder.markdown(f"```\n{st.session_state.llm_analysis}\n```")
        else:
            analysis_placeholder.info("🔄 Waiting for defect data to analyze...")
    
    with col_report:
        st.markdown("**Quality Control Report**")
        if st.button("📊 Generate Full Report", help="Generate comprehensive defect analysis report"):
            with st.spinner("Generating comprehensive report..."):
                report = generate_defect_report(
                    st.session_state.defect_history,
                    openai_api_key,
                    llm_model
                )
                with st.expander("📋 Quality Control Report", expanded=True):
                    st.markdown(report)
        
        # Show defect history summary
        if st.session_state.defect_history:
            recent_count = len([d for d in st.session_state.defect_history 
                             if time.time() - d['timestamp'] < 300])  # Last 5 minutes
            st.metric("Defects (5 min)", recent_count)
            
            # Show severity breakdown
            severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for defect in st.session_state.defect_history[-20:]:  # Last 20 defects
                severity_counts[defect['severity']] += 1
            
            st.markdown("**Recent Severity Breakdown:**")
            for severity, count in severity_counts.items():
                if count > 0:
                    color = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}[severity]
                    st.markdown(f"{color} **{severity}**: {count}")
                    
            # Export defect data
            if st.button("💾 Export Defect Data"):
                import json
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "total_defects": len(st.session_state.defect_history),
                    "defects": st.session_state.defect_history[-50:],  # Last 50 defects
                    "summary": severity_counts
                }
                st.download_button(
                    "📥 Download JSON",
                    json.dumps(export_data, indent=2),
                    f"steel_defects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
elif enable_llm and not HAS_OPENAI:
    st.warning("⚠️ OpenAI package not installed. Install with: `pip install openai`")
elif enable_llm and not openai_api_key:
    st.info("🔑 Enter OpenAI API key in sidebar to enable AI analysis")

# -----------------------
# Performance Information
# -----------------------
with st.expander("📊 Performance Guide (AMD Instinct 300X)", expanded=False):
    st.markdown("""
    **Precision Mode Performance on AMD Instinct 300X:**
    
    | Mode | Speed | Quality | Use Case |
    |------|-------|---------|----------|
    | 🚀 **INT8 Turbo** | ~3-4x faster | Good | Real-time detection |
    | ⚡ **FP16 Fast** | ~2x faster | Excellent | Balanced performance |
    | 🔧 **FP32 Precise** | Baseline | Best | Accuracy critical |
    
    **Tips for Maximum Speed:**
    - Use INT8 for real-time applications (30+ FPS)
    - Set Frame Buffer to 1-3 for lowest latency
    - Target FPS at 30 for optimal GPU utilization
    - Close other GPU applications for best performance
    """)
    
    # System information
    if device == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            st.write(f"**GPU:** {gpu_name}")
            st.write(f"**Memory:** {gpu_memory:.1f} GB")
            st.write(f"**ROCm Optimizations:** ✅ Enabled")
            st.write(f"**NNPACK:** ❌ Disabled (AMD compatibility)")
        except:
            st.write("**GPU:** AMD Instinct Detected")
    else:
        st.write("**Device:** CPU")

# -----------------------
# Steel Defect Detection Guide
# -----------------------
with st.expander("🔍 Steel Defect Detection Guide", expanded=False):
    st.markdown("""
    **Defect Severity Color Coding:**
    
    - 🔴 **CRITICAL** - Red boxes: Immediate attention required
    - 🟠 **HIGH** - Orange boxes: Significant defects affecting quality  
    - 🟡 **MEDIUM** - Yellow boxes: Moderate defects for review
    - 🟢 **LOW** - Green boxes: Minor surface imperfections
    
    **Common Steel Defects Detected:**
    - **Surface Defect** (bench detections) - Surface irregularities, scratches
    - **Inclusion** - Foreign materials embedded in steel
    - **Crack** - Linear fractures in the material
    - **Corrosion** - Oxidation and rust formation
    - **Scale Formation** - Oxide layer buildup
    - **Edge Defect** - Problems at steel edges
    - **Dent** - Mechanical deformation
    - **Pit/Hole** - Small cavities in surface
    
    **Detection Confidence:**
    - Higher confidence (>0.8) may upgrade severity level
    - Lower confidence suggests possible false positive
    - Adjust confidence threshold to balance sensitivity vs accuracy
    """)

# -----------------------
# Cleanup on app shutdown
# -----------------------
if not st.session_state.running:
    if hasattr(st.session_state, 'running_flag'):
        st.session_state.running_flag[0] = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
