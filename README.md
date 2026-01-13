# 🏭 Zero-Defect Manufacturing – AI Vision System

A real-time steel defect detection system powered by **AMD Instinct GPUs + ROCm** for industrial quality control applications.

## 🔬 What This Application Does

This Streamlit application processes video streams to detect and classify defects in steel manufacturing using computer vision and AI. It maps generic YOLO object detections to specific steel manufacturing defects with severity classification.

### Core Functionality

1. **Real-Time Video Processing**: Processes video files frame-by-frame using background threading
2. **AI-Powered Defect Detection**: Uses YOLOv8 model optimized for AMD GPUs
3. **Steel-Specific Defect Mapping**: Converts generic object detections into relevant steel defects
4. **Severity Classification**: Categorizes defects by severity (CRITICAL, HIGH, MEDIUM, LOW)
5. **Performance Optimization**: Multiple precision modes for different speed/accuracy trade-offs

## 🎯 Key Features

### 🚀 AMD GPU Optimization
- **AMD Instinct GPU Detection**: Automatically detects and optimizes for AMD Instinct cards
- **ROCm Integration**: Native AMD GPU acceleration support
- **Memory Management**: Efficient CUDA memory handling with periodic cleanup
- **NNPACK Disable**: Compatibility optimizations for AMD hardware

### ⚡ Performance Modes
| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| 🚀 **INT8 Turbo** | ~3-4x faster | Good | Real-time detection |
| ⚡ **FP16 Fast** | ~2x faster | Excellent | Balanced performance |
| 🔧 **FP32 Precise** | Baseline | Best | Accuracy critical |

### 🔍 Steel Defect Detection
The system maps YOLO's 80 object classes to steel manufacturing defects:

**Critical Defects** (Red boxes):
- Large structural damage
- Critical safety issues

**High Severity** (Orange boxes):
- Large Dent, Crack, Foreign Object
- Edge Defect

**Medium Severity** (Yellow boxes):
- Scale Formation, Deep Defect
- Corrosion, Inclusion, Surface irregularities

**Low Severity** (Green boxes):
- Surface Defect, Surface Scratch
- Small Inclusion, Small Pit

## 🛠️ Technical Architecture

### Threading Model
```
Main Thread (Streamlit UI)
├── Background Worker Thread
│   ├── Video Frame Capture
│   ├── AI Model Inference
│   ├── Defect Classification
│   └── Frame Queue Management
└── Display Thread
    ├── Metrics Updates
    ├── Frame Display
    └── UI Refresh
```

### AMD GPU Optimizations
```python
# GPU Detection and Optimization
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    is_amd = any(x in gpu_name.upper() for x in ['INSTINCT', 'MI', 'RADEON'])
    
    if is_amd:
        # AMD-specific optimizations
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
```

### Defect Mapping System
```python
# Example mapping from YOLO classes to steel defects
DEFECT_CLASS_MAP = {
    'bench': 'Surface Defect',      # Your current detection
    'car': 'Large Dent',
    'knife': 'Sharp Edge',
    'crack': 'Crack',
    # ... 80+ mappings
}

# Severity classification
DEFECT_SEVERITY = {
    'Critical Defect': 'CRITICAL',
    'Large Dent': 'HIGH',
    'Surface Defect': 'LOW',
    # ... severity rules
}
```

## 🚀 Usage Instructions

### 1. Prerequisites
```bash
# Ensure you have these files:
- app.py (main application)
- yolov8n.pt (YOLO model weights)
- sample_video.mp4 (test video)
- requirements.txt (dependencies)
```

### 2. Running the Application
```bash
# Activate virtual environment
source steel_venv/bin/activate

# Run the application
./run_app.sh
# or
streamlit run app.py
```

### 3. Using the Interface

**Sidebar Controls:**
- **Video path**: Set path to your video file (default: `sample_video.mp4`)
- **Confidence Threshold**: Adjust detection sensitivity (0.1-1.0)
- **Precision Mode**: Choose speed vs accuracy trade-off
- **Max FPS**: Control processing speed (1-60 FPS)
- **Frame Buffer**: Set buffer size for smoother playback

**Main Interface:**
- Click **▶ Start** to begin processing
- Monitor real-time metrics (FPS, Latency, GPU Memory)
- View defect statistics in sidebar
- Click **⏹ Stop** to halt processing

## 📊 Performance Metrics

The application displays real-time performance data:

- **FPS**: Frames processed per second
- **Latency**: Processing time per frame (milliseconds)
- **GPU Memory**: VRAM usage (MB)
- **Defect Counts**: Live count by severity level

## 🔧 Configuration Options

### Video Processing
- **Frame Buffer Size**: 1-10 frames (lower = less latency)
- **Target FPS**: 1-60 (match your video's native FPS)
- **Quality Settings**: JPEG compression for display optimization

### AI Model Settings
- **Confidence Threshold**: 0.1-1.0 (higher = fewer false positives)
- **Image Size**: 640x640 (optimized for YOLOv8n)
- **Max Detections**: 100 per frame
- **NMS Settings**: Agnostic NMS for faster processing

### AMD GPU Settings
- **Half Precision**: FP16 mode for 2x speed improvement
- **Memory Allocation**: Optimized chunking for AMD architecture
- **Cache Management**: Periodic cleanup every 30 frames

## 🐛 Troubleshooting

### Common Issues

**"Cannot open video file"**
- Check video file path is correct
- Ensure video file exists and is readable
- Try absolute path instead of relative path

**No video display after clicking Start**
- Verify video file format is supported by OpenCV
- Check if background thread is running properly
- Look for error messages in terminal/console

**Low FPS Performance**
- Switch to INT8 Turbo mode
- Reduce frame buffer size to 1-2
- Lower confidence threshold
- Close other GPU applications

**GPU Not Detected**
- Verify ROCm installation
- Check CUDA compatibility
- Ensure proper GPU drivers

### Performance Optimization Tips

1. **For Maximum Speed**:
   - Use INT8 Turbo mode
   - Set frame buffer to 1
   - Target 30 FPS
   - Enable aggressive NMS

2. **For Best Accuracy**:
   - Use FP32 Precise mode
   - Lower confidence threshold
   - Increase frame buffer size
   - Disable speed optimizations

3. **For Balanced Performance**:
   - Use FP16 Fast mode
   - Set frame buffer to 3
   - Target 20-25 FPS
   - Enable standard optimizations

## 🔬 Steel Manufacturing Integration

This system can be integrated into real manufacturing environments:

1. **Quality Control Stations**: Real-time inspection during production
2. **Automated Sorting**: Classify products by defect severity
3. **Process Monitoring**: Track defect trends over time
4. **Maintenance Alerts**: Trigger maintenance based on defect patterns

The defect classification system is designed to be extensible - you can modify the `DEFECT_CLASS_MAP` and `DEFECT_SEVERITY` dictionaries to match your specific manufacturing requirements.

## 📁 File Structure

```
steel_manufacturing/
├── app.py                 # Main Streamlit application
├── README.md             # This documentation
├── requirements.txt      # Python dependencies
├── run_app.sh           # Launch script
├── yolov8n.pt          # YOLO model weights
├── sample_video.mp4    # Test video file
└── steel_venv/         # Python virtual environment
```

## 🔮 Future Enhancements

- Custom trained models for specific steel defects
- Historical defect tracking and analytics
- Integration with manufacturing execution systems (MES)
- Real-time camera feed support
- Multi-camera synchronization
- Defect trend analysis and reporting