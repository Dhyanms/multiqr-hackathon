# Multi-QR Code Recognition for Medicine Packs

## 1. Introduction

This project addresses the challenge of detecting and decoding multiple QR codes on medicine packaging. Medicine packs typically contain 2-6 QR codes containing information about batch numbers, manufacturer details, distributor information, and regulatory compliance. Our solution uses YOLOv8s to detect all QR codes in a single image, even under challenging conditions like tilting, blur, or partial occlusion. The system also decodes and classifies each QR code automatically.

## 2. Why YOLOv8s?

**YOLOv8s was chosen for its optimal balance between speed and accuracy.** 

Compared to larger models like YOLOv8x or YOLOv8l, YOLOv8s is 2x faster with only a 2% accuracy drop, making it ideal for real-time applications. Unlike Faster R-CNN which takes 200-300ms per image, YOLOv8s achieves 30-50ms inference time. YOLOv5, while popular, uses an older architecture that struggles with small objects like QR codes. EfficientDet offers good accuracy but has complex training procedures and slower inference. YOLOv8n is too lightweight and misses QR codes in challenging conditions.

**Key advantages of YOLOv8s:**
- Single-stage detector handles multiple QR codes simultaneously
- Superior feature pyramid network for small object detection
- Pre-trained weights on COCO enable faster convergence
- Anchor-free design reduces hyperparameter tuning
- Production-ready with 30+ FPS capability on standard GPUs

## 3. Project Structure

```
multiqr-hackathon/
├── train.py                           # Training pipeline
├── infer.py                           # Inference script (Stage 1 & 2)
├── evaluate.py                        # Evaluation script
├── requirements.txt                   # Python dependencies
│
├── src/
│   ├── __init__.py
│   └── utils/
│       ├── __init__.py
│       ├── qr_decoder.py             # QR decoding with 7 preprocessing variants
│       ├── nms_utils.py              # Custom Non-Maximum Suppression
│       └── data_utils.py             # Visualization utilities
│
├── data/
│   ├── train/
│   │   └── train/
│   │       ├── images/               # 582 training images
│   │       └── labels/               # 582 YOLO format labels
│   └── test/
│   │    ├── images/                   # 50 test images
│   │    └── labels/                 # 50 test labels (optional)
│   └── data.yaml  
└── outputs/
    ├── qr_detection/
    │   └── weights/
    │       └── best.pt               # Trained model weights
    ├── submission_detection_1.json   # Stage 1 output
    └── submission_decoding_2.json    # Stage 2 output (bonus)
```

## 4. Dataset Details

### Training Data
- **Location**: `data/train/train/`
- **Total Images**: 582 (200 original images × 3 augmented versions each)
- **Labels**: `data/train/train/labels/` (582 .txt files)
- **Source**: Roboflow Universe - 1pharma/annotation-k7xrm
- **QR Codes per Image**: 2-4 (average)
- **Total Annotations**: ~1,200+ QR code bounding boxes

### Test Data
- **Location**: `data/test/images/`
- **Total Images**: 50
- **Labels**: `data/test/labels/` (50 .txt files - optional for evaluation)

### Preprocessing Applied
1. **Auto-orientation**: Automatic EXIF-based rotation correction
2. **Resize**: All images resized to 640×640 pixels (stretch mode)

### Augmentation Strategy
Each original image was augmented to create 3 versions (200 → 582 images):
1. **Horizontal Flip**: 50% probability to handle different orientations
2. **Random Rotation**: -15° to +15° to simulate tilted packages
3. **Brightness Adjustment**: -8% to +8% for varying lighting conditions
4. **Gaussian Blur**: 0 to 0.2 pixels to simulate motion blur

### Label Format
**YOLO v8 Format** (normalized coordinates):
```
class_id x_center y_center width height
```
Example:
```
0 0.342 0.487 0.123 0.156
0 0.678 0.234 0.098 0.134
```
- `class_id`: Always 0 (single class: QR code)
- `x_center, y_center`: Center coordinates (normalized 0-1)
- `width, height`: Box dimensions (normalized 0-1)

Sample Image:
<img width="1442" height="1137" alt="image" src="https://github.com/user-attachments/assets/68042812-effd-4c87-a911-b752ff6d527b" />


## 5. Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

### Steps

```bash
# Clone repository
git clone https://github.com/yourusername/multiqr-hackathon.git
cd multiqr-hackathon

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install system dependency for QR decoding
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install libzbar0

# macOS:
brew install zbar

# Windows: Download from https://sourceforge.net/projects/zbar/

# Verify installation
python -c "from pyzbar import pyzbar; print('✅ All dependencies installed successfully')"
```

## 6. Training

### Basic Training

```bash
python train.py --dataset data/train \
                --epochs 100 \
                --batch 16 \
                --model yolov8s.pt
```

### Parameters Explained

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--dataset` | `data/train` | Path to dataset directory |
| `--epochs` | `100` | Number of training iterations (100 recommended) |
| `--batch` | `16` | Batch size (16 for T4 GPU, 8 for smaller GPUs) |
| `--model` | `yolov8s.pt` | Model variant (s/m/l/x) |

### Quick Training (for testing)

```bash
python train.py --dataset data/train\
                --epochs 20 \
                --batch 16
```

### Training Output

After training completes:
- Model weights saved at: `outputs/qr_detection/weights/best.pt`
- Training metrics saved in: `outputs/qr_detection/`
- Validation metrics displayed in console

**Expected Training Time:**
- GPU (T4): ~1.5 hours (100 epochs)
- GPU (RTX 3090): ~45 minutes (100 epochs)
- CPU: Not recommended (20+ hours)

## 7. Inference

### Stage 1: Detection Only (Mandatory)

**Images**: 50 test images  
**Location**: `data/test/images/`

```bash
python infer.py --input data/test/images \
                --output outputs/submission_detection_1.json \
                --stage 1 \
                --conf 0.25
```

**Output**: `outputs/submission_detection_1.json`

**Format:**
```json
[
  {
    "image_id": "img001",
    "qrs": [
      {"bbox": [100, 150, 250, 300]},
      {"bbox": [400, 200, 550, 350]}
    ]
  }
]
```

### Stage 2: Detection + Decoding + Classification (Bonus)

**Images**: Same 50 test images  
**Location**: `data/test/images/`

```bash
python infer.py --input data/test/images \
                --output outputs/submission_decoding_2.json \
                --stage 2 \
                --conf 0.25
```

**Output**: `outputs/submission_decoding_2.json`

**Format:**
```json
[
  {
    "image_id": "img001",
    "qrs": [
      {
        "bbox": [100, 150, 250, 300],
        "value": "BATCH-12345",
        "type": "batch_number"
      },
      {
        "bbox": [400, 200, 550, 350],
        "value": "MFR-PHARMA-2024",
        "type": "manufacturer"
      }
    ]
  }
]
```

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | Required | Directory containing test images |
| `--output` | Required | Output JSON file path |
| `--stage` | `1` | 1 for detection only, 2 for detection+decoding |
| `--conf` | `0.25` | Confidence threshold (0.20-0.35 recommended) |
| `--model` | `outputs/qr_detection/weights/best.pt` | Path to trained model |

### Adjusting Confidence Threshold

```bash
# Lower threshold = Higher recall (fewer missed QR codes)
python infer.py --input data/test/images --output outputs/results.json --conf 0.20

# Higher threshold = Higher precision (fewer false positives)
python infer.py --input data/test/images --output outputs/results.json --conf 0.35
```

## 8. Results

Validation Metrics:
   mAP50: 0.9921
   mAP50-95: 0.8748
   Precision: 0.9696
   Recall: 0.9682

Test Set:

Stage 1 (Detection):
   Total images: 50
   Total QR codes detected: 186
   Average QR per image: 3.72
Stage 2 (Decoding - Bonus):
   Successfully decoded: 155/186
   Decode success rate: 83.3%
QR Code Types Detected:
   general: 155

## 9. Limitations

### Manual Labeling Challenge

The primary limitation encountered during this project was the **absence of pre-labeled training data**. While the dataset images were available, bounding box annotations for QR codes were not provided initially.

**Solution Implemented:**
We manually annotated all 200 training images using two approaches:
1. **Roboflow**: Web-based annotation platform for batch processing
2. **LabelImg**: Desktop application for precise bounding box creation

**Time Investment:**
- Annotation time: Approximately **2 hours**
- Images labeled: 200 medicine pack images
- Average time per image: ~36 seconds
- QR codes annotated: 400+ bounding boxes

**Annotation Process:**
1. Load image in annotation tool
2. Draw bounding box around each QR code
3. Assign class label (class 0: QR)
4. Export in YOLO format
5. Verify annotation accuracy

This manual labeling process, while time-consuming, ensured high-quality ground truth data essential for training an accurate detection model. The annotations were then used with Roboflow's augmentation pipeline to generate the final 582-image training set.

## 10. Future Improvements

### Model Enhancements
1. **Ensemble Approach**: Combine YOLOv8s with YOLOv8m for improved accuracy through model averaging
2. **Test-Time Augmentation (TTA)**: Apply multiple transformations during inference and average predictions
3. **Attention Mechanisms**: Integrate CBAM or SE blocks to focus on QR code features
4. **Custom Loss Function**: Design QR-specific loss emphasizing small object detection

### Data Improvements
1. **Expand Dataset**: Collect 500+ more medicine pack images from diverse sources
2. **Hard Negative Mining**: Add images without QR codes to reduce false positives
3. **Synthetic Data Generation**: Create artificial medicine packs with varied QR placements
4. **Real-world Conditions**: Include more examples with extreme blur, occlusion, and lighting

### Decoding Enhancements
1. **Custom QR Decoder**: Build specialized decoder beyond pyzbar for damaged codes
2. **Error Correction**: Implement Reed-Solomon error correction for partial QR codes
3. **Multi-stage Decoding**: Use detection confidence to prioritize decoding attempts
4. **Deep Learning Decoder**: Train CNN to decode QR codes directly from images

### System Optimization
1. **Model Quantization**: INT8 quantization for 4x faster inference
2. **TensorRT Integration**: Optimize for NVIDIA GPUs (10x speedup potential)
3. **ONNX Export**: Enable deployment across different platforms
4. **Edge Deployment**: Optimize for mobile/embedded devices (Raspberry Pi, Jetson Nano)

### Production Features
1. **REST API**: Flask/FastAPI endpoint for web service deployment
2. **Batch Processing**: Handle multiple images in parallel
3. **Real-time Video**: Extend to video stream processing (webcam/scanner)
4. **Quality Checks**: Add image quality assessment before processing
5. **Database Integration**: Store detections and decoded values in database
6. **Confidence Thresholding**: Auto-adjust confidence based on image quality

### Evaluation & Monitoring
1. **Cross-validation**: K-fold validation for robust performance metrics
2. **A/B Testing**: Compare multiple model versions in production
3. **Performance Dashboard**: Real-time monitoring of inference speed and accuracy
4. **Error Analysis**: Automatic logging of failed detections for retraining

---

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [pyzbar Library](https://github.com/NaturalHistoryMuseum/pyzbar)
- [Roboflow Dataset](https://universe.roboflow.com/1pharma/annotation-k7xrm/dataset/3)

---

**Made for Multi-QR Code Recognition Hackathon 2025** 🏆
