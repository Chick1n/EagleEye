"""
Data Augmentation Script for Fake Coin Images
Generates augmented images with class 1 (eagle_fake) labels
"""

import os
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import cv2
from PIL import Image
import pillow_heif

# Register HEIF opener
pillow_heif.register_heif_opener()

def augment_fake_images(input_dir, output_dir, labels_output_dir, 
                        num_augmented_per_image=20, target_size=(640, 640)):
    """
    Augment fake coin images with class 1 labels
    
    Args:
        input_dir: Directory containing fake coin images
        output_dir: Directory to save augmented images
        labels_output_dir: Directory to save YOLO labels
        num_augmented_per_image: Number of augmented versions per original image
        target_size: Resize images to this size (width, height)
    """
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    labels_path = Path(labels_output_dir)
    labels_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files (JPG, PNG, HEIC)
    input_path = Path(input_dir)
    image_files = (sorted(input_path.glob("*.jpg")) + 
                   sorted(input_path.glob("*.png")) + 
                   sorted(input_path.glob("*.heic")) +
                   sorted(input_path.glob("*.HEIC")))
    
    if not image_files:
        print(f"Error: No images found in {input_dir}")
        return False
    
    print(f"Found {len(image_files)} fake coin images")
    print(f"Generating {num_augmented_per_image} augmented versions per image")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    print()
    
    # Create ImageDataGenerator with augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=20,              # Rotate up to 20 degrees
        width_shift_range=0.0,          # NO horizontal shift
        height_shift_range=0.0,         # NO vertical shift
        brightness_range=[0.8, 1.2],    # Brightness 80%-120%
        zoom_range=0.1,                 # Zoom up to 10%
        horizontal_flip=True,           # Flip horizontally
        vertical_flip=False,            # Don't flip vertically
        shear_range=0.1,                # Shear transformation
        fill_mode='nearest'             # Fill pixels
    )
    
    total_augmented = 0
    
    # Process each image
    for img_file in image_files:
        print(f"Processing: {img_file.name}")
        
        # Load image (supports JPG, PNG, HEIC)
        if img_file.suffix.lower() in ['.heic', '.heif']:
            # Load HEIC using PIL
            img = Image.open(img_file)
            img = img.convert('RGB')
            img = img.resize(target_size)
        else:
            # Load JPG/PNG using Keras
            img = load_img(img_file, target_size=target_size)
        
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Generate augmented images
        aug_iter = datagen.flow(img_array, batch_size=1)
        
        for i in range(num_augmented_per_image):
            # Generate augmented image
            aug_img = next(aug_iter)[0].astype(np.uint8)
            
            # Create output filename
            output_filename = f"fake_{img_file.stem}_aug_{i:03d}.jpg"
            output_file = output_path / output_filename
            
            # Save augmented image
            cv2.imwrite(str(output_file), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            
            # Generate YOLO label for fake coin (class 1)
            # Coin stays centered and covers full image
            x_center = 0.5  # Always centered
            y_center = 0.5  # Always centered
            width = 1.0     # Full width
            height = 1.0    # Full height
            
            # YOLO format: class_id x_center y_center width height
            # class_id = 1 for fake coins
            label_content = f"1 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n"
            
            # Save label
            label_file = labels_path / f"fake_{img_file.stem}_aug_{i:03d}.txt"
            with open(label_file, 'w') as f:
                f.write(label_content)
            
            total_augmented += 1
        
        print(f"  ✓ Generated {num_augmented_per_image} augmented images with fake labels")
    
    print()
    print("=" * 60)
    print("Fake Coin Augmentation Complete!")
    print("=" * 60)
    print(f"Original fake images: {len(image_files)}")
    print(f"Augmented per image: {num_augmented_per_image}")
    print(f"Total augmented fake images: {total_augmented}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Labels directory: {labels_path.absolute()}")
    print()
    print("Labels use class 1 (eagle_fake)")
    print()
    
    return True

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Fake Coin Image Augmentation with Labels")
    print("=" * 60)
    print()
    
    # Default paths
    input_dir = "dataset/images/fake_eagle"
    output_dir = "dataset/images/train"
    labels_output_dir = "dataset/labels/train"
    num_augmented = 20
    
    # Parse arguments
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    if len(sys.argv) > 3:
        num_augmented = int(sys.argv[3])
    
    print(f"Configuration:")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Labels directory: {labels_output_dir}")
    print(f"  Augmented per image: {num_augmented}")
    print(f"  Target size: 640x640")
    print(f"  Class: eagle_fake (ID: 1)")
    print()
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        print()
        print("Usage:")
        print(f"  python augment_fake_images.py <input_dir> [output_dir] [num_augmented]")
        print()
        print("Example:")
        print(f"  python augment_fake_images.py dataset/images/fake_eagle dataset/images/train 20")
        sys.exit(1)
    
    # Augment fake images with labels
    if augment_fake_images(input_dir, output_dir, labels_output_dir, num_augmented):
        print("✓ Fake coin augmentation successful!")
        print()
        print("Next steps:")
        print("1. Verify dataset: python quick_start.py verify")
        print("2. Split dataset: python quick_start.py split")
        print("3. Train model: python train_yolo.py train")
    else:
        print("✗ Fake coin augmentation failed!")
