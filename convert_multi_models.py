"""
Train and convert multiple specialized coin recognition models for iOS deployment.
Example: Separate models for US coins and Euro coins.
"""

import tensorflow as tf
import os

def convert_model_to_tflite(model_path: str, output_name: str):
    """Convert a Keras model to TensorFlow Lite format"""
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    try:
        print(f"\nLoading Keras model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        
        print(f"Converting to TensorFlow Lite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        output_path = f'model/{output_name}.tflite'
        print(f"Saving to {output_path}...")
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Conversion successful!")
        print(f"  Model size: {model_size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False


def create_labels_file(coin_types: list, output_name: str):
    """Create a labels file for the model output classes"""
    
    output_path = f'model/{output_name}_labels.txt'
    
    try:
        with open(output_path, 'w') as f:
            for idx, coin_type in enumerate(coin_types):
                f.write(f"{idx} {coin_type}\n")
        
        print(f"✓ Created labels file: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating labels file: {str(e)}")
        return False


def setup_multi_model_project():
    """
    Example setup for multiple coin recognition models.
    
    Train separate models:
    1. US Coins Model
    2. Euro Coins Model
    
    Then convert each to TensorFlow Lite.
    """
    
    print("=" * 60)
    print("Multi-Model Coin Recognition Setup")
    print("=" * 60)
    
    # Configuration for different models
    models_config = [
        {
            'name': 'US Coins',
            'keras_model': 'model/us_coins_model.h5',
            'tflite_output': 'us_coins',
            'labels': ['penny', 'nickel', 'dime', 'quarter']
        },
        {
            'name': 'Euro Coins',
            'keras_model': 'model/euro_coins_model.h5',
            'tflite_output': 'euro_coins',
            'labels': ['1_cent', '2_cent', '5_cent', '10_cent', '20_cent', '50_cent', '1_euro', '2_euro']
        }
    ]
    
    print("\nProcessing models...")
    print("-" * 60)
    
    for config in models_config:
        print(f"\n{config['name']}:")
        
        # Convert model
        success = convert_model_to_tflite(
            config['keras_model'],
            config['tflite_output']
        )
        
        if success:
            # Create labels file
            create_labels_file(
                config['labels'],
                config['tflite_output']
            )
        else:
            print(f"⚠ Skipping {config['name']} - model file not found")
            print(f"  Expected: {config['keras_model']}")
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("\nFor iOS App:")
    print("1. Copy .tflite files to Xcode project")
    print("2. Copy _labels.txt files to Xcode project")
    print("3. Update model configurations in MultiModelCameraViewController.swift")
    print("=" * 60)


# Example: How to manually convert if you have existing models
def convert_existing_models():
    """
    Convert existing Keras models you may already have.
    
    Usage:
        If you already trained:
        - us_coins_model.h5
        - euro_coins_model.h5
        
        Just run:
        python convert_to_tflite.py
    """
    
    # Check what models exist
    model_dir = 'model'
    h5_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    
    print("Found Keras models:")
    for f in h5_files:
        print(f"  - {f}")
    
    # Convert all .h5 files
    for h5_file in h5_files:
        model_path = os.path.join(model_dir, h5_file)
        output_name = h5_file.replace('.h5', '')
        convert_model_to_tflite(model_path, output_name)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--setup':
        # Run full setup with example config
        setup_multi_model_project()
    else:
        # Convert existing models
        convert_existing_models()
