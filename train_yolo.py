"""
YOLOv11 Training Script for Coin Recognition
Train a custom model to detect eagle and morgan coins
"""

from ultralytics import YOLO
import os

def train_coin_model():
    """
    Train YOLOv11 model on custom coin dataset
    """
    
    # Check if dataset exists
    data_yaml = "dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"Error: {data_yaml} not found!")
        print("Please create dataset structure first")
        return False
    
    # Load pretrained nano model (fastest)
    print("Loading YOLOv11 nano model...")
    model = YOLO("yolo11n.pt")
    
    # Train the model
    print("Starting training...")
    print(f"Dataset: {data_yaml}")
    
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        device="mps",  # Use Metal Performance Shaders (GPU on M1/M2/M3)
        batch=8,       # Smaller batch for GPU memory
        patience=20,
        save=True,
        project="coin_detection",
        name="eagle_morgan",
        verbose=True,
        workers=4,     # Parallel data loading
        # Augmentation for better generalization
        hsv_h=0.015,      # HSV Hue augmentation
        hsv_s=0.7,        # HSV Saturation augmentation
        hsv_v=0.4,        # HSV Value augmentation
        degrees=10,       # Rotation augmentation
        translate=0.1,    # Translation augmentation
        scale=0.5,        # Scale augmentation
        shear=0.0,        # Shear augmentation
        perspective=0.0,  # Perspective augmentation
        flipud=0.0,       # Flip up-down
        fliplr=0.5,       # Flip left-right
        mosaic=1.0,       # Mosaic augmentation
        mixup=0.0,        # Mixup augmentation
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best model saved at: {results.best}")
    print(f"Results directory: {results.save_dir}")
    
    return True

def validate_model():
    """
    Validate the trained model
    """
    model_path = "runs/detect/coin_detection/eagle_morgan/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return False
    
    print("Loading trained model...")
    model = YOLO(model_path)
    
    # Validate on test set
    print("Running validation...")
    results = model.val(data="dataset/data.yaml", split="test")
    
    print("\nValidation Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP75: {results.box.map75:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    return True

def export_to_tflite():
    """
    Export trained model to TensorFlow Lite for iOS deployment
    """
    model_path = "runs/detect/coin_detection/eagle_morgan/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return False
    
    print("Exporting to TensorFlow Lite...")
    model = YOLO(model_path)
    
    # Export to TFLite
    model.export(format="tflite", imgsz=640)
    
    print("✓ Model exported successfully!")
    print("  File: coin_detection/eagle_morgan/weights/best_saved_model/best_float16.tflite")
    
    return True

def test_inference():
    """
    Test the trained model on a sample image
    """
    model_path = "coin_detection/eagle_morgan/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return False
    
    model = YOLO(model_path)
    
    # Test on a sample image
    test_image = "dataset/images/test/eagle_001.jpg"  # Update with your test image
    
    if os.path.exists(test_image):
        print(f"Testing on {test_image}...")
        results = model(test_image)
        
        # Print results
        for result in results:
            print(f"\nDetections: {len(result.boxes)}")
            for box in result.boxes:
                print(f"  Class: {result.names[int(box.cls)]}")
                print(f"  Confidence: {box.conf:.2f}")
                print(f"  BBox: {box.xyxy}")
    else:
        print(f"Test image not found: {test_image}")
    
    return True

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("YOLOv11 Coin Recognition Training")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "train":
            train_coin_model()
        elif command == "validate":
            validate_model()
        elif command == "export":
            export_to_tflite()
        elif command == "test":
            test_inference()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: train, validate, export, test")
    else:
        # Default: run training
        print("\nUsage:")
        print("  python train_yolo.py train     # Train the model")
        print("  python train_yolo.py validate  # Validate trained model")
        print("  python train_yolo.py export    # Export to TFLite")
        print("  python train_yolo.py test      # Test inference")
        print("\nRunning training by default...")
        train_coin_model()
