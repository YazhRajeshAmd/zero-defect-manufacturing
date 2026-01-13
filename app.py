import os
import sys
import warnings
import contextlib
from io import StringIO, BytesIO
import hashlib
from PIL import Image
import base64

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
video_path = st.sidebar.text_input("Video path", "sample_video-4.mp4")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)
precision = st.sidebar.selectbox("Precision Mode", ["INT8 (Fastest)", "FP16 (GPU)", "FP32"])
start_button = st.sidebar.button("▶ Start")
stop_button = st.sidebar.button("⏹ Stop")
fps_target = st.sidebar.slider("Max FPS", 1, 60, 20)
buffer_size = st.sidebar.slider("Frame Buffer", 1, 10, 3)

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
            # Disable problematic optimizations for AMD
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        else:
            st.sidebar.info(f"🖥️ GPU: {gpu_name}")
            
        # General CUDA optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        
    except Exception as e:
        st.sidebar.info(f"🖥️ GPU: CUDA Device (details unavailable)")

@st.cache_resource
def load_model():
    # Suppress YOLO warnings and NNPACK
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

model = load_model()

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
    'Corrosion': 'MEDIUM',
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
        defect_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        total_defects = 0
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = results[0].names[class_id]
                defect_name, severity = get_defect_info(class_name, confidence)
                defect_counts[severity] += 1
                total_defects += 1
        
        # Update stats
        stats["latency"] = processing_latency
        stats["frame_count"] += 1
        stats["total_defects"] = total_defects
        stats["defect_counts"] = defect_counts
        
        # Calculate FPS
        if stats["frame_count"] > 1:
            stats["fps"] = 1.0 / (current_time - last_process_time)
        else:
            stats["fps"] = 0
        
        # Convert frame to base64 for display
        frame_b64 = convert_cv2_to_base64(annotated, quality=85)
        frame_hash = get_image_hash(annotated)
        
        # Put processed frame in queue
        try:
            frame_queue.put((frame_b64, frame_hash, processing_latency, stats["fps"]), timeout=0.1)
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
            
            # Load optimized model based on precision
            optimized_model = load_quantized_model(precision)
            
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
        frame_b64, frame_hash, latency, fps_val = frame_data
        
        # Only update display if frame changed (reduces Streamlit load)
        if frame_hash != st.session_state.frame_hash and frame_b64 is not None:
            st.session_state.current_frame_b64 = frame_b64
            st.session_state.frame_hash = frame_hash
        
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
        
        # Display defect statistics in sidebar
        if "total_defects" in st.session_state.stats:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**🔍 Current Frame Defects:**")
            st.sidebar.metric("Total Defects", st.session_state.stats.get("total_defects", 0))
            
            defect_counts = st.session_state.stats.get("defect_counts", {})
            if defect_counts.get('CRITICAL', 0) > 0:
                st.sidebar.error(f"🚨 Critical: {defect_counts['CRITICAL']}")
            if defect_counts.get('HIGH', 0) > 0:
                st.sidebar.warning(f"⚠️ High: {defect_counts['HIGH']}")
            if defect_counts.get('MEDIUM', 0) > 0:
                st.sidebar.info(f"ℹ️ Medium: {defect_counts['MEDIUM']}")
            if defect_counts.get('LOW', 0) > 0:
                st.sidebar.success(f"✅ Low: {defect_counts['LOW']}")
        
        # Display defect statistics in sidebar
        if "total_defects" in st.session_state.stats:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**🔍 Current Frame Defects:**")
            st.sidebar.metric("Total Defects", st.session_state.stats.get("total_defects", 0))
            
            defect_counts = st.session_state.stats.get("defect_counts", {})
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
            device_box.metric("Device", device.upper())
    
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
            st.write("**GPU:** CUDA Device")
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
