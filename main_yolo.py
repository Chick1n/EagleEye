"""
YOLOv11 Inference Script for Coin Recognition
Real-time coin detection using YOLOv11 model
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

class CoinDetector:
    def __init__(self, model_path="runs/detect/coin_detection/eagle_morgan/weights/best.pt", confidence_threshold=0.80):
        """
        Initialize YOLO coin detector
        
        Args:
            model_path: Path to trained YOLO model
            confidence_threshold: Confidence threshold (0.0-1.0)
        """
        self.confidence_threshold = confidence_threshold
        
        # Check if model exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load YOLO model
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        # Get class names
        self.class_names = self.model.names
        print(f"Classes: {self.class_names}")
        
    def detect_coins(self, frame):
        """
        Detect coins in a frame
        
        Args:
            frame: Input image (numpy array)
            
        Returns:
            List of detections with bounding boxes and confidence scores
        """
        # Run inference
        results = self.model(frame, verbose=False)
        
        detections = []
        
        for result in results:
            # Get bounding boxes
            boxes = result.boxes
            
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence
                confidence = float(box.conf[0])
                
                # Get class ID
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]
                
                # Determine if real or fake based on confidence
                is_real = confidence > self.confidence_threshold
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'is_real': is_real
                })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input image
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            is_real = detection['is_real']
            
            # Choose color based on real/fake
            if is_real:
                color = (0, 255, 0)  # Green for real
                status = "REAL"
            else:
                color = (0, 0, 255)  # Red for fake
                status = "FAKE?"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{status} {class_name}: {confidence*100:.1f}%"
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return frame
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: Input image
            
        Returns:
            Processed frame with detections
        """
        # Detect coins
        detections = self.detect_coins(frame)
        
        # Draw detections
        frame = self.draw_detections(frame, detections)
        
        # Print results
        if detections:
            for detection in detections:
                status = "✓ REAL" if detection['is_real'] else "✗ FAKE?"
                print(f"{status} {detection['class_name']}: {detection['confidence']*100:.1f}%")
        
        return frame, detections


def main():
    """Main inference loop"""
    
    # Initialize detector
    try:
        detector = CoinDetector(
            model_path="runs/detect/coin_detection/eagle_morgan-4/weights/best.pt",
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
        
        # Process frame
        frame, detections = detector.process_frame(frame)
        
        # Display frame
        cv2.imshow("Coin Recognition - YOLOv11", frame)
        
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
        detector = CoinDetector(
            model_path="runs/detect/coin_detection/eagle_morgan-4/weights/best.pt",
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
    
    # Process image
    frame, detections = detector.process_frame(frame)
    
    # Display result
    cv2.imshow("Coin Recognition - YOLOv11", frame)
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
