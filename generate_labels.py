"""
Auto-generate YOLO labels for extracted frames
Assumes coin is centered and takes up most of the frame
"""

import os
from pathlib import Path

def generate_labels(images_dir, labels_dir, class_id=0, coin_size=0.8):
    """
    Generate YOLO format labels for all images
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory to save labels
        class_id: Class ID (0 for eagle)
        coin_size: Size of coin as fraction of image (0.8 = 80%)
    """
    
    # Create labels directory
    labels_path = Path(labels_dir)
    labels_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    images_path = Path(images_dir)
    image_files = sorted(images_path.glob("*.jpg")) + sorted(images_path.glob("*.png"))
    
    if not image_files:
        print(f"Error: No images found in {images_dir}")
        return False
    
    print(f"Found {len(image_files)} images")
    print(f"Generating labels...")
    
    # Generate label for each image
    for img_file in image_files:
        # Create label filename (same as image but .txt)
        label_file = labels_path / f"{img_file.stem}.txt"
        
        # YOLO format: class_id x_center y_center width height (normalized)
        # Assuming coin fills the entire frame: x=0.5, y=0.5, width=1.0, height=1.0
        label_content = f"{class_id} 0.5 0.5 1.0 1.0\n"
        
        # Write label file
        with open(label_file, 'w') as f:
            f.write(label_content)
    
    print(f"✓ Generated {len(image_files)} label files")
    print(f"  Labels directory: {labels_path.absolute()}")
    print(f"  Class ID: {class_id} (eagle)")
    print(f"  Coin size: {coin_size*100}% of image")
    
    return True

def verify_labels(images_dir, labels_dir):
    """Verify that all images have corresponding labels"""
    
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    image_files = sorted(images_path.glob("*.jpg")) + sorted(images_path.glob("*.png"))
    label_files = sorted(labels_path.glob("*.txt"))
    
    print(f"\nVerification:")
    print(f"  Images: {len(image_files)}")
    print(f"  Labels: {len(label_files)}")
    
    # Check for missing labels
    missing_labels = []
    for img_file in image_files:
        label_file = labels_path / f"{img_file.stem}.txt"
        if not label_file.exists():
            missing_labels.append(img_file.name)
    
    if missing_labels:
        print(f"  Missing labels: {len(missing_labels)}")
        for name in missing_labels[:5]:  # Show first 5
            print(f"    - {name}")
        if len(missing_labels) > 5:
            print(f"    ... and {len(missing_labels)-5} more")
    else:
        print(f"  ✓ All images have labels!")
    
    return len(missing_labels) == 0

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Auto-Generate YOLO Labels for Coin Images")
    print("=" * 60)
    
    # Default paths
    images_dir = "dataset/images/train"
    labels_dir = "dataset/labels/train"
    
    # Parse arguments
    if len(sys.argv) > 1:
        images_dir = sys.argv[1]
    if len(sys.argv) > 2:
        labels_dir = sys.argv[2]
    
    print(f"\nConfiguration:")
    print(f"  Images directory: {images_dir}")
    print(f"  Labels directory: {labels_dir}")
    print(f"  Class: eagle (ID: 0)")
    print(f"  Coin size: 80% of image")
    print()
    
    # Generate labels
    if generate_labels(images_dir, labels_dir, class_id=0, coin_size=0.8):
        # Verify
        verify_labels(images_dir, labels_dir)
        
        print("\n" + "=" * 60)
        print("Next steps:")
        print("1. Review labels (optional)")
        print("2. Split dataset: python quick_start.py split")
        print("3. Train model: python train_yolo.py train")
        print("=" * 60)
