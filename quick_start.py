"""
Quick Start Script for YOLOv11 Coin Recognition
This script helps you set up the dataset structure and start training
"""

import os
import shutil
from pathlib import Path

def create_dataset_structure():
    """Create the required YOLO dataset directory structure"""
    
    base_path = Path("dataset")
    
    # Create directories
    directories = [
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test"
    ]
    
    print("Creating dataset structure...")
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")
    
    print("\nDataset structure created!")
    print(f"Location: {base_path.absolute()}")
    
    return True

def split_dataset(source_images_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Split images into train/val/test sets
    
    Args:
        source_images_dir: Directory containing all images
        train_ratio: Proportion for training (default 0.8)
        val_ratio: Proportion for validation (default 0.1)
    """
    
    source_path = Path(source_images_dir)
    if not source_path.exists():
        print(f"Error: Source directory not found: {source_images_dir}")
        return False
    
    # Get all image files
    image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
    
    if not image_files:
        print(f"No images found in {source_images_dir}")
        return False
    
    # Shuffle and split
    import random
    random.shuffle(image_files)
    
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    print(f"\nDataset split:")
    print(f"  Total: {n_total}")
    print(f"  Train: {len(train_files)} ({len(train_files)/n_total*100:.1f}%)")
    print(f"  Val: {len(val_files)} ({len(val_files)/n_total*100:.1f}%)")
    print(f"  Test: {len(test_files)} ({len(test_files)/n_total*100:.1f}%)")
    
    # Copy files
    print("\nCopying files...")
    
    for file in train_files:
        shutil.copy(file, f"dataset/images/train/{file.name}")
        # Copy corresponding label if exists
        label_file = file.with_suffix('.txt')
        if label_file.exists():
            shutil.copy(label_file, f"dataset/labels/train/{label_file.name}")
    
    for file in val_files:
        shutil.copy(file, f"dataset/images/val/{file.name}")
        label_file = file.with_suffix('.txt')
        if label_file.exists():
            shutil.copy(label_file, f"dataset/labels/val/{label_file.name}")
    
    for file in test_files:
        shutil.copy(file, f"dataset/images/test/{file.name}")
        label_file = file.with_suffix('.txt')
        if label_file.exists():
            shutil.copy(label_file, f"dataset/labels/test/{label_file.name}")
    
    print("✓ Files copied successfully!")
    return True

def verify_dataset():
    """Verify that the dataset is properly structured"""
    
    print("\nVerifying dataset structure...")
    
    required_dirs = [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/images/test",
        "dataset/labels/train",
        "dataset/labels/val",
        "dataset/labels/test"
    ]
    
    all_ok = True
    for directory in required_dirs:
        if os.path.exists(directory):
            count = len(os.listdir(directory))
            print(f"  ✓ {directory} ({count} files)")
        else:
            print(f"  ✗ {directory} (missing)")
            all_ok = False
    
    # Check data.yaml
    if os.path.exists("dataset/data.yaml"):
        print(f"  ✓ dataset/data.yaml")
    else:
        print(f"  ✗ dataset/data.yaml (missing)")
        all_ok = False
    
    return all_ok

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("YOLOv11 Coin Recognition - Quick Start")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "setup":
            create_dataset_structure()
        elif command == "split" and len(sys.argv) > 2:
            source_dir = sys.argv[2]
            split_dataset(source_dir)
        elif command == "verify":
            verify_dataset()
        else:
            print("Unknown command")
    else:
        print("\nUsage:")
        print("  python quick_start.py setup              # Create dataset structure")
        print("  python quick_start.py split <source_dir> # Split images into train/val/test")
        print("  python quick_start.py verify             # Verify dataset structure")
        print("\nExample:")
        print("  python quick_start.py setup")
        print("  python quick_start.py split /path/to/all/images")
        print("  python quick_start.py verify")
        print("  python train_yolo.py train")
