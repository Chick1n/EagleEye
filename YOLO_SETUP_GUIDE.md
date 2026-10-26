# YOLOv11 Coin Recognition Setup Guide

Complete guide to train YOLOv11 for coin detection.

## Quick Start

### 1. Install Dependencies

```bash
source .venv3.11/bin/activate
pip install ultralytics
```

### 2. Prepare Dataset

#### Directory Structure
```
dataset/
├── images/
│   ├── train/          # Training images (80%)
│   │   ├── eagle_001.jpg
│   │   ├── eagle_002.jpg
│   │   └── morgan_001.jpg
│   ├── val/            # Validation images (10%)
│   │   ├── eagle_050.jpg
│   │   └── morgan_050.jpg
│   └── test/           # Test images (10%)
│       ├── eagle_100.jpg
│       └── morgan_100.jpg
├── labels/
│   ├── train/          # YOLO format labels (.txt)
│   │   ├── eagle_001.txt
│   │   ├── eagle_002.txt
│   │   └── morgan_001.txt
│   ├── val/
│   │   ├── eagle_050.txt
│   │   └── morgan_050.txt
│   └── test/
│       ├── eagle_100.txt
│       └── morgan_100.txt
└── data.yaml           # Dataset configuration
```

#### Label Images in YOLO Format

For each image, create a `.txt` file with the same name:

```
# eagle_001.txt
# Format: class_id x_center y_center width height (normalized 0-1)
0 0.5 0.5 0.3 0.3
```

**Class IDs:**
- `0` = eagle
- `1` = morgan

**Labeling Tools:**
- [LabelImg](https://github.com/tzutalin/labelImg) - Free, offline
- [Roboflow](https://roboflow.com/) - Online, auto-augmentation
- [CVAT](https://cvat.org/) - Professional, collaborative

### 3. Train the Model

```bash
# Activate environment
source .venv3.11/bin/activate

# Train
python train_yolo.py train
```

**Training Output:**
```
coin_detection/eagle_morgan/
├── weights/
│   ├── best.pt        # Best model
│   └── last.pt        # Last epoch
├── results.png        # Training curves
├── confusion_matrix.png
└── ...
```

### 4. Validate the Model

```bash
python train_yolo.py validate
```

**Metrics:**
- mAP50: Mean Average Precision at IoU 0.5
- mAP75: Mean Average Precision at IoU 0.75
- mAP50-95: Mean Average Precision averaged over IoU 0.5-0.95

### 5. Test Inference

```bash
python train_yolo.py test
```

### 6. Export for iOS

```bash
python train_yolo.py export
```

**Output:** `coin_detection/eagle_morgan/weights/best_saved_model/best_float16.tflite`

## Dataset Preparation Tips

### Collecting Images

**Per coin type (eagle, morgan):**
- 200-300 images minimum
- Various angles, lighting, backgrounds
- Both front and back faces
- Different distances

### Labeling Best Practices

1. **Tight bounding boxes** - Box should touch the coin edges
2. **Consistent labeling** - Same class name for same coin type
3. **Include difficult cases** - Blurry, dark, partially occluded
4. **Balance classes** - Similar number of eagle and morgan images

### Data Augmentation (Already in training script)

- Rotation: ±10 degrees
- Translation: ±10%
- Scale: ±50%
- Flip: 50% chance left-right
- Mosaic: Combine 4 images
- HSV: Color variations

## Training Parameters

Edit `train_yolo.py` to customize:

```python
results = model.train(
    data="dataset/data.yaml",
    epochs=100,              # More = better, but slower
    imgsz=640,               # Image size (640x640)
    device=0,                # GPU 0, or 'cpu'
    batch=16,                # Batch size (reduce if GPU memory error)
    patience=20,             # Early stopping patience
    save=True,
    project="coin_detection",
    name="eagle_morgan",
)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "CUDA out of memory" | Reduce batch size to 8 or 4 |
| "No module named 'ultralytics'" | Run `pip install ultralytics` |
| "data.yaml not found" | Check path in data.yaml |
| "No labels found" | Ensure .txt files match image names |
| "Low mAP" | Collect more data, train longer, check labels |

## Comparison: Teachable Machine vs YOLOv11

| Metric | Teachable Machine | YOLOv11 |
|--------|-------------------|---------|
| Accuracy | ~70-80% | ~90-95% |
| Speed | Slow (1-2 FPS) | Fast (30+ FPS) |
| Custom classes | Limited | Unlimited |
| False positives | High | Low |
| Detail recognition | Poor | Good |
| Model size | ~50MB | ~5MB (nano) |

## Next Steps

1. **Label your dataset** using LabelImg or Roboflow
2. **Organize into train/val/test** splits
3. **Run training** with `python train_yolo.py train`
4. **Validate results** with `python train_yolo.py validate`
5. **Export to TFLite** with `python train_yolo.py export`
6. **Deploy to iOS** using the exported model

## Resources

- [Ultralytics YOLOv11 Docs](https://docs.ultralytics.com/)
- [YOLO Format Guide](https://roboflow.com/formats/yolo-darknet-txt)
- [LabelImg Tutorial](https://github.com/tzutalin/labelImg)

Good luck! 🚀
