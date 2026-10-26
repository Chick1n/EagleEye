"""
Convert Keras model to TensorFlow Lite format for iOS deployment
"""

import tensorflow as tf
import os

def convert_to_tflite():
    """Convert keras_model.h5 to TensorFlow Lite format"""
    
    model_path = 'model/keras_model_3.h5'
    output_path = 'model/coin_recognition_3.tflite'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    try:
        print(f"Loading Keras model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        
        print("Converting to TensorFlow Lite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optional: Enable optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        print(f"Saving TensorFlow Lite model to {output_path}...")
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Conversion successful!")
        print(f"  Model size: {model_size_mb:.2f} MB")
        print(f"  Output file: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False

if __name__ == "__main__":
    convert_to_tflite()
