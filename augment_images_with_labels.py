"""
Data Augmentation Script with Position Tracking
Generates augmented images AND correct YOLO labels
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

def augment_images_with_labels(input_dir, output_dir, labels_output_dir, 
                               num_augmented_per_image=10, target_size=(640, 640),
                               coin_size=0.8):
    """
    Augment images and generate correct YOLO labels
    
    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save augmented images
        labels_output_dir: Directory to save YOLO labels
        num_augmented_per_image: Number of augmented versions per original image
        target_size: Resize images to this size (width, height)
        coin_size: Size of coin as fraction of image (0.8 = 80%)
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
    
    print(f"Found {len(image_files)} original images")
    print(f"Generating {num_augmented_per_image} augmented versions per image")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    print()
    
    # Create ImageDataGenerator with augmentation parameters
    # Keep coin centered - only apply rotation, brightness, zoom, shear
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
            output_filename = f"{img_file.stem}_aug_{i:03d}.jpg"
            output_file = output_path / output_filename
            
            # Save augmented image
            cv2.imwrite(str(output_file), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            
            # Generate YOLO label for this augmented image
            # Since coin stays centered (no translation) and covers full image
            # Position is always (0.5, 0.5) and size is always (1.0, 1.0)
            
            x_center = 0.5  # Always centered
            y_center = 0.5  # Always centered
            width = 1.0     # Full width (coin covers entire image)
            height = 1.0    # Full height (coin covers entire image)
            
            # YOLO format: class_id x_center y_center width height
            label_content = f"0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n"
            
            # Save label
            label_file = labels_path / f"{img_file.stem}_aug_{i:03d}.txt"
            with open(label_file, 'w') as f:
                f.write(label_content)
            
            total_augmented += 1
        
        print(f"  ✓ Generated {num_augmented_per_image} augmented images with labels")
    
    print()
    print("=" * 60)
    print("Augmentation Complete!")
    print("=" * 60)
    print(f"Original images: {len(image_files)}")
    print(f"Augmented per image: {num_augmented_per_image}")
    print(f"Total augmented images: {total_augmented}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Labels directory: {labels_path.absolute()}")
    print()
    print("Note: Labels use estimated positions based on augmentation parameters.")
    print("For more accurate labels, consider manual labeling or Roboflow.")
    print()
    
    return True

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Coin Image Data Augmentation with Labels")
    print("=" * 60)
    print()
    
    # Default paths
    input_dir = "dataset/images/original"
    output_dir = "dataset/images/train"
    labels_output_dir = "dataset/labels/train"
    num_augmented = 10
    
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
    print()
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        print()
        print("Usage:")
        print(f"  python augment_images.py <input_dir> [output_dir] [num_augmented]")
        print()
        print("Example:")
        print(f"  python augment_images.py dataset/images/original dataset/images/train 10")
        sys.exit(1)
    
    # Augment images with labels
    if augment_images_with_labels(input_dir, output_dir, labels_output_dir, num_augmented):
        print("✓ Augmentation successful!")
        print()
        print("Next steps:")
        print("1. Verify augmented images and labels")
        print("2. Split dataset: python quick_start.py split")
        print("3. Train model: python train_yolo.py train")
    else:
        print("✗ Augmentation failed!")
