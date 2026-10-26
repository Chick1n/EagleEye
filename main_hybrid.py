"""
Hybrid Coin Recognition: YOLO Model + Ellipse Visualization
Uses YOLO for detection accuracy and Keras-style ellipse for visualization
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from src import PreProcessing, Ellipse

class HybridCoinDetector:
    def __init__(self, yolo_model_path="runs/detect/coin_detection/eagle_morgan-4/weights/best.pt", 
                 confidence_threshold=0.80):
        """
        Initialize hybrid detector: YOLO for detection + Ellipse for visualization
        
        Args:
            yolo_model_path: Path to trained YOLO model
            confidence_threshold: Confidence threshold (0.0-1.0)
        """
        self.confidence_threshold = confidence_threshold
        
        # Check if YOLO model exists
        if not Path(yolo_model_path).exists():
            raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")
        
        # Load YOLO model
        print(f"Loading YOLO model: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        self.class_names = self.yolo_model.names
        print(f"Classes: {self.class_names}")
        
    def detect_and_visualize(self, frame):
        """
        Detect coins using YOLO and visualize with ellipses
        
        Args:
            frame: Input image (numpy array)
            
        Returns:
            frame: Frame with ellipses drawn
            detections: List of detections
        """
        # Run YOLO detection
        results = self.yolo_model(frame, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]
                
                # Determine if real or fake
                is_real = confidence > self.confidence_threshold
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'is_real': is_real,
                    'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                    'size': (x2 - x1, y2 - y1)
                })
        
        # Draw ellipses instead of rectangles
        for detection in detections:
            center_x, center_y = detection['center']
            width, height = detection['size']
            
            # Calculate ellipse parameters
            # Use the smaller dimension for more accurate circle
            radius = min(width, height) / 2
            
            # Choose color based on real/fake
            if detection['is_real']:
                color = (0, 255, 0)  # Green for real
                status = "REAL"
            else:
                color = (0, 0, 255)  # Red for fake
                status = "FAKE?"
            
            # Draw ellipse (circle)
            cv2.circle(frame, 
                      (int(center_x), int(center_y)), 
                      int(radius), 
                      color, 
                      3)
            
            # Prepare label
            label = f"{status} {detection['class_name']}: {detection['confidence']*100:.1f}%"
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                frame,
                (int(center_x - text_width/2), int(center_y - radius - text_height - 15)),
                (int(center_x + text_width/2), int(center_y - radius - 5)),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (int(center_x - text_width/2), int(center_y - radius - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        return frame, detections


def main():
    """Main inference loop with hybrid detection"""
    
    # Initialize hybrid detector
    try:
        detector = HybridCoinDetector(
            yolo_model_path="runs/detect/coin_detection/eagle_morgan-4/weights/best.pt",
            confidence_threshold=0.80  # 80% threshold
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease train the model first:")
        print("  python train_yolo.py train")
        return
    
    # Open camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened successfully")
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    
    frame_count = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Detect and visualize
        frame, detections = detector.detect_and_visualize(frame)
        
        # Print results
        if detections:
            for detection in detections:
                status = "✓ REAL" if detection['is_real'] else "✗ FAKE?"
                print(f"{status} {detection['class_name']}: {detection['confidence']*100:.1f}%")
        
        # Display frame
        cv2.imshow("Coin Recognition - YOLO + Ellipse", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            # Save screenshot
            screenshot_path = f"screenshot_{frame_count:04d}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"Screenshot saved: {screenshot_path}")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


def main_image(image_path):
    """Process a single image"""
    
    # Initialize detector
    try:
        detector = HybridCoinDetector(
            yolo_model_path="runs/detect/coin_detection/eagle_morgan-4/weights/best.pt",
            confidence_threshold=0.80
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Read image
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not read image: {image_path}")
        return
    
    # Detect and visualize
    frame, detections = detector.detect_and_visualize(frame)
    
    # Print results
    if detections:
        for detection in detections:
            status = "✓ REAL" if detection['is_real'] else "✗ FAKE?"
            print(f"{status} {detection['class_name']}: {detection['confidence']*100:.1f}%")
    
    # Display result
    cv2.imshow("Coin Recognition - YOLO + Ellipse", frame)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process image file
        image_path = sys.argv[1]
        main_image(image_path)
    else:
        # Process camera feed
        main()
