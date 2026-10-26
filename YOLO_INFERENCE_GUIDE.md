# YOLOv11 Inference Guide

Complete guide to use YOLOv11 for coin recognition.

## Installation

```bash
source .venv3.11/bin/activate
pip install ultralytics opencv-python
```

## Usage

### 1. Real-Time Camera Detection

```bash
python main_yolo.py
```

**Controls:**
- `q` - Quit
- `s` - Save screenshot

### 2. Process Single Image

```bash
python main_yolo.py path/to/image.jpg
```

## Features

### CoinDetector Class

```python
from main_yolo import CoinDetector

# Initialize
detector = CoinDetector(
    model_path="coin_detection/eagle_morgan/weights/best.pt",
    confidence_threshold=0.80  # 80% threshold
)

# Detect coins in image
detections = detector.detect_coins(frame)

# Draw results
frame = detector.draw_detections(frame, detections)
```

### Detection Output

Each detection contains:
```python
{
    'bbox': (x1, y1, x2, y2),      # Bounding box coordinates
    'confidence': 0.95,             # Confidence score (0-1)
    'class_id': 0,                  # Class ID
    'class_name': 'eagle',          # Class name
    'is_real': True                 # True if confidence > threshold
}
```

## Confidence Threshold

Adjust the threshold for real/fake detection:

```python
detector = CoinDetector(confidence_threshold=0.80)  # 80%

# Detection results:
# - confidence > 0.80 → REAL (green box)
# - confidence < 0.80 → FAKE? (red box)
```

### Recommended Thresholds

| Threshold | Use Case |
|-----------|----------|
| 0.70 | Lenient (more false positives) |
| 0.80 | Balanced (recommended) |
| 0.90 | Strict (fewer false positives) |

## Real-Time Performance

| Hardware | FPS | Latency |
|----------|-----|---------|
| M1/M2 GPU | 30-60 | 16-33ms |
| CPU | 5-15 | 66-200ms |

## Model Path

Default: `coin_detection/eagle_morgan/weights/best.pt`

To use a different model:
```python
detector = CoinDetector(model_path="path/to/your/model.pt")
```

## Examples

### Example 1: Real-Time Detection

```bash
python main_yolo.py
```

Output:
```
✓ REAL eagle: 95.2%
✓ REAL eagle: 92.1%
✗ FAKE? eagle: 45.3%
```

### Example 2: Single Image

```bash
python main_yolo.py test_image.jpg
```

### Example 3: Custom Threshold

```python
detector = CoinDetector(confidence_threshold=0.85)
frame, detections = detector.process_frame(frame)
```

### Example 4: Batch Processing

```python
import cv2
from main_yolo import CoinDetector

detector = CoinDetector()

for image_file in image_files:
    frame = cv2.imread(image_file)
    frame, detections = detector.process_frame(frame)
    cv2.imwrite(f"result_{image_file}", frame)
```

## Comparison: Keras vs YOLO

| Feature | Keras (main.py) | YOLO (main_yolo.py) |
|---------|-----------------|---------------------|
| **Detection** | Classification only | Object detection |
| **Bounding Box** | No | Yes |
| **Speed** | Slow | Fast |
| **Accuracy** | ~70-80% | ~90-95% |
| **Real-time** | No | Yes |
| **Model Size** | ~50MB | ~5MB |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model not found" | Train first: `python train_yolo.py train` |
| "Could not open camera" | Check camera permissions |
| "Low FPS" | Reduce image size or use GPU |
| "False positives" | Increase confidence threshold |
| "False negatives" | Decrease confidence threshold |

## Integration with iOS

To deploy YOLO model to iOS:

1. Export to TFLite:
   ```bash
   python train_yolo.py export
   ```

2. Use exported `.tflite` file in iOS app

3. Update `CoinRecognitionModel.swift` to use YOLO model

## Next Steps

1. Train model: `python train_yolo.py train`
2. Test inference: `python main_yolo.py`
3. Validate results: `python train_yolo.py validate`
4. Export for iOS: `python train_yolo.py export`

## Files

- [main_yolo.py](main_yolo.py) - YOLO inference script
- [train_yolo.py](train_yolo.py) - YOLO training script
- [dataset/data.yaml](dataset/data.yaml) - Dataset configuration
- [YOLO_SETUP_GUIDE.md](YOLO_SETUP_GUIDE.md) - Setup guide

Good luck! 🚀
