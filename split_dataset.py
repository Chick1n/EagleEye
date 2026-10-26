"""
Split training dataset into train/val/test sets
Maintains class balance between real and fake coins
"""

import os
import shutil
from pathlib import Path
import random

def split_dataset(train_images_dir="dataset/images/train", 
                  train_labels_dir="dataset/labels/train",
                  val_ratio=0.1, 
                  test_ratio=0.1,
                  seed=42):
    """
    Split training data into train/val/test sets
    
    Args:
        train_images_dir: Directory containing training images
        train_labels_dir: Directory containing training labels
        val_ratio: Proportion for validation (default 0.1 = 10%)
        test_ratio: Proportion for test (default 0.1 = 10%)
        seed: Random seed for reproducibility
    """
    
    random.seed(seed)
    
    # Create output directories
    val_images_dir = Path("dataset/images/val")
    val_labels_dir = Path("dataset/labels/val")
    test_images_dir = Path("dataset/images/test")
    test_labels_dir = Path("dataset/labels/test")
    
    for dir_path in [val_images_dir, val_labels_dir, test_images_dir, test_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all training images
    train_images_path = Path(train_images_dir)
    all_images = sorted(train_images_path.glob("*.jpg"))
    
    if not all_images:
        print(f"Error: No images found in {train_images_dir}")
        return False
    
    print(f"Total training images: {len(all_images)}")
    
    # Separate real and fake images
    real_images = [img for img in all_images if not img.name.startswith("fake_")]
    fake_images = [img for img in all_images if img.name.startswith("fake_")]
    
    print(f"Real images: {len(real_images)}")
    print(f"Fake images: {len(fake_images)}")
    
    # Shuffle both lists
    random.shuffle(real_images)
    random.shuffle(fake_images)
    
    # Calculate split indices for real images
    n_real = len(real_images)
    n_real_val = int(n_real * val_ratio)
    n_real_test = int(n_real * test_ratio)
    n_real_train = n_real - n_real_val - n_real_test
    
    # Calculate split indices for fake images
    n_fake = len(fake_images)
    n_fake_val = int(n_fake * val_ratio)
    n_fake_test = int(n_fake * test_ratio)
    n_fake_train = n_fake - n_fake_val - n_fake_test
    
    print(f"\nReal images split:")
    print(f"  Train: {n_real_train}")
    print(f"  Val: {n_real_val}")
    print(f"  Test: {n_real_test}")
    
    print(f"\nFake images split:")
    print(f"  Train: {n_fake_train}")
    print(f"  Val: {n_fake_val}")
    print(f"  Test: {n_fake_test}")
    
    # Split real images
    real_train = real_images[:n_real_train]
    real_val = real_images[n_real_train:n_real_train + n_real_val]
    real_test = real_images[n_real_train + n_real_val:]
    
    # Split fake images
    fake_train = fake_images[:n_fake_train]
    fake_val = fake_images[n_fake_train:n_fake_train + n_fake_val]
    fake_test = fake_images[n_fake_train + n_fake_val:]
    
    # Move files
    print("\nMoving files...")
    
    # Move real validation images
    for img in real_val:
        label = Path(train_labels_dir) / f"{img.stem}.txt"
        
        shutil.move(str(img), str(val_images_dir / img.name))
        if label.exists():
            shutil.move(str(label), str(val_labels_dir / label.name))
    
    # Move real test images
    for img in real_test:
        label = Path(train_labels_dir) / f"{img.stem}.txt"
        
        shutil.move(str(img), str(test_images_dir / img.name))
        if label.exists():
            shutil.move(str(label), str(test_labels_dir / label.name))
    
    # Move fake validation images
    for img in fake_val:
        label = Path(train_labels_dir) / f"{img.stem}.txt"
        
        shutil.move(str(img), str(val_images_dir / img.name))
        if label.exists():
            shutil.move(str(label), str(val_labels_dir / label.name))
    
    # Move fake test images
    for img in fake_test:
        label = Path(train_labels_dir) / f"{img.stem}.txt"
        
        shutil.move(str(img), str(test_images_dir / img.name))
        if label.exists():
            shutil.move(str(label), str(test_labels_dir / label.name))
    
    print("✓ Files moved successfully!")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Split Complete!")
    print("=" * 60)
    
    train_count = len(list(Path(train_images_dir).glob("*.jpg")))
    val_count = len(list(val_images_dir.glob("*.jpg")))
    test_count = len(list(test_images_dir.glob("*.jpg")))
    
    print(f"\nFinal split:")
    print(f"  Train: {train_count} images")
    print(f"  Val: {val_count} images")
    print(f"  Test: {test_count} images")
    print(f"  Total: {train_count + val_count + test_count} images")
    
    print(f"\nReal/Fake balance:")
    print(f"  Train: {len(real_train)} real + {len(fake_train)} fake")
    print(f"  Val: {len(real_val)} real + {len(fake_val)} fake")
    print(f"  Test: {len(real_test)} real + {len(fake_test)} fake")
    
    print("\nDirectories:")
    print(f"  Train images: {train_images_dir}")
    print(f"  Val images: {val_images_dir}")
    print(f"  Test images: {test_images_dir}")
    
    return True

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Split Training Dataset into Train/Val/Test")
    print("=" * 60)
    print()
    
    # Parse arguments
    val_ratio = 0.1
    test_ratio = 0.1
    
    if len(sys.argv) > 1:
        val_ratio = float(sys.argv[1])
    if len(sys.argv) > 2:
        test_ratio = float(sys.argv[2])
    
    print(f"Configuration:")
    print(f"  Validation ratio: {val_ratio*100}%")
    print(f"  Test ratio: {test_ratio*100}%")
    print(f"  Train ratio: {(1-val_ratio-test_ratio)*100}%")
    print()
    
    # Split dataset
    if split_dataset(val_ratio=val_ratio, test_ratio=test_ratio):
        print("\n✓ Dataset split successful!")
        print("\nNext steps:")
        print("1. Verify: python quick_start.py verify")
        print("2. Train: python train_yolo.py train")
    else:
        print("\n✗ Dataset split failed!")
