"""
Extract frames from video for YOLO training
Converts 2-minute video into 200+ training images
"""

import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir, num_frames=200, target_size=(640, 640)):
    """
    Extract frames from video and save as images
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract
        target_size: Resize images to this size (width, height)
    """
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        return False
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"Video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {duration:.2f} seconds")
    
    # Calculate frame skip interval
    frame_skip = max(1, total_frames // num_frames)
    print(f"  Extracting every {frame_skip} frame(s)")
    print(f"  Target frames: {num_frames}")
    
    # Extract frames
    extracted_count = 0
    frame_number = 0
    
    print("\nExtracting frames...")
    
    while extracted_count < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Extract every Nth frame
        if frame_number % frame_skip == 0:
            # Resize to target size
            frame_resized = cv2.resize(frame, target_size)
            
            # Save frame
            output_file = output_path / f"frame_{extracted_count:04d}.jpg"
            cv2.imwrite(str(output_file), frame_resized)
            
            extracted_count += 1
            
            if extracted_count % 20 == 0:
                print(f"  Extracted: {extracted_count}/{num_frames}")
        
        frame_number += 1
    
    cap.release()
    
    print(f"\n✓ Extraction complete!")
    print(f"  Total extracted: {extracted_count} frames")
    print(f"  Output directory: {output_path.absolute()}")
    print(f"  Image size: {target_size[0]}x{target_size[1]}")
    
    return True

def extract_frames_with_labels(video_path, output_dir, num_frames=200, target_size=(640, 640)):
    """
    Extract frames and create corresponding YOLO label files
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract
        target_size: Resize images to this size (width, height)
    """
    
    # Extract frames
    if not extract_frames(video_path, output_dir, num_frames, target_size):
        return False
    
    # Create label files
    output_path = Path(output_dir)
    labels_dir = output_path.parent / "labels" / output_path.name
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nCreating YOLO label files...")
    
    # Create a label file for each frame
    # Assuming coin is centered and takes up most of the frame
    for i in range(num_frames):
        label_file = labels_dir / f"frame_{i:04d}.txt"
        
        # YOLO format: class_id x_center y_center width height (normalized)
        # Assuming coin is centered: x=0.5, y=0.5, width=0.8, height=0.8
        label_content = "0 0.5 0.5 0.8 0.8\n"
        
        with open(label_file, 'w') as f:
            f.write(label_content)
    
    print(f"✓ Created {num_frames} label files")
    print(f"  Labels directory: {labels_dir.absolute()}")
    
    return True

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Video to Frames Extractor for YOLO Training")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python extract_frames.py <video_path> [num_frames] [output_dir]")
        print("\nExample:")
        print("  python extract_frames.py coin_video.mp4 200 dataset/images/train")
        print("  python extract_frames.py coin_video.mp4 200 dataset/images/train --with-labels")
        sys.exit(1)
    
    video_path = sys.argv[1]
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "dataset/images/train"
    with_labels = "--with-labels" in sys.argv
    
    print(f"\nConfiguration:")
    print(f"  Video: {video_path}")
    print(f"  Frames to extract: {num_frames}")
    print(f"  Output directory: {output_dir}")
    print(f"  Create labels: {with_labels}")
    print()
    
    if with_labels:
        extract_frames_with_labels(video_path, output_dir, num_frames)
    else:
        extract_frames(video_path, output_dir, num_frames)
