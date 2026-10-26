# Multi-Model Coin Recognition Guide

Using separate specialized models for different coin types (US coins, Euro coins, etc.)

## Overview

Instead of one model for all coins, use **specialized models** for better accuracy:

```
US Coins Model (US-optimized)
├── Trained on: penny, nickel, dime, quarter
├── File: us_coins.tflite
└── Labels: us_coins_labels.txt

Euro Coins Model (Euro-optimized)
├── Trained on: 1¢, 2¢, 5¢, 10¢, 20¢, 50¢, €1, €2
├── File: euro_coins.tflite
└── Labels: euro_coins_labels.txt
```

**Benefits:**
- ✅ Higher accuracy per model (specialized training)
- ✅ Smaller model files
- ✅ Faster inference
- ✅ User selects correct model type
- ✅ Easy to add more coin types

---

## Step 1: Train Separate Models (Python)

Train one model per coin type:

```python
# us_coins_model.py
import tensorflow as tf
from tensorflow import keras

# Load training data for US coins only
train_images = load_us_coins_training_data()
train_labels = ['penny', 'nickel', 'dime', 'quarter']

# Build and train model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    # ... more layers ...
    keras.layers.Dense(4, activation='softmax')  # 4 US coin types
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(train_images, train_labels, epochs=50)

# Save
model.save('us_coins_model.h5')
```

Repeat for `euro_coins_model.h5`, etc.

---

## Step 2: Convert Models to TensorFlow Lite

```bash
cd /Users/dongyuliu/wkspace/CoinRecognition

# Convert all models at once
python convert_multi_models.py

# Or setup with example config
python convert_multi_models.py --setup
```

**Output:**
```
model/
├── us_coins.tflite
├── us_coins_labels.txt
├── euro_coins.tflite
└── euro_coins_labels.txt
```

---

## Step 3: Add Files to Xcode

### 3.1 Copy to Xcode Project
```bash
cp model/us_coins.tflite /Users/dongyuliu/xcode/CoinRecognition/CoinRecognition/
cp model/us_coins_labels.txt /Users/dongyuliu/xcode/CoinRecognition/CoinRecognition/
cp model/euro_coins.tflite /Users/dongyuliu/xcode/CoinRecognition/CoinRecognition/
cp model/euro_coins_labels.txt /Users/dongyuliu/xcode/CoinRecognition/CoinRecognition/
```

### 3.2 In Xcode
1. Select each .tflite and .txt file
2. File Inspector → Check "CoinRecognition" under Target Membership

---

## Step 4: Update SceneDelegate (iOS)

Replace `CameraViewController` with `MultiModelCameraViewController`:

```swift
// SceneDelegate.swift
func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options connectionOptions: UIScene.ConnectionOptions) {
    guard let scene = (scene as? UIWindowScene) else { return }
    
    let window = UIWindow(windowScene: scene)
    // Use multi-model version
    let rootViewController = MultiModelCameraViewController()
    window.rootViewController = UINavigationController(rootViewController: rootViewController)
    self.window = window
    window.makeKeyAndVisible()
}
```

---

## Step 5: Build and Run

1. Open `CoinRecognition.xcworkspace`
2. Select device/simulator
3. Build: Cmd+B
4. Run: Cmd+R

**App Features:**
- Top: Segmented control to switch between models
- Center: Coin detection result
- Shows which model is active in confidence label

---

## Adding More Models

To add another model (e.g., Canadian coins):

### 1. Python Side
```python
# Create new model
model = train_canadian_coins_model()
model.save('canadian_coins_model.h5')
```

### 2. Conversion
```bash
python convert_multi_models.py
# Creates:
# - canadian_coins.tflite
# - canadian_coins_labels.txt
```

### 3. iOS Update
In `MultiModelCameraViewController.swift`:
```swift
let modelConfigs = [
    (name: "US Coins", modelFile: "us_coins.tflite", labelsFile: "us_coins_labels.txt"),
    (name: "Euro Coins", modelFile: "euro_coins.tflite", labelsFile: "euro_coins_labels.txt"),
    (name: "Canadian Coins", modelFile: "canadian_coins.tflite", labelsFile: "canadian_coins_labels.txt"),  // NEW
]

modelRecognition = MultiModelCoinRecognition(modelConfigs: modelConfigs)
```

4. Copy files to Xcode and set Target Membership
5. Rebuild

---

## File Structure

```
/Users/dongyuliu/wkspace/CoinRecognition/
├── model/
│   ├── us_coins.tflite
│   ├── us_coins_labels.txt
│   ├── euro_coins.tflite
│   ├── euro_coins_labels.txt
│   ├── us_coins_model.h5        (training model)
│   └── euro_coins_model.h5      (training model)
├── convert_multi_models.py       (conversion script)
└── ...

/Users/dongyuliu/xcode/CoinRecognition/CoinRecognition/
├── MultiModelCoinRecognition.swift
├── MultiModelCameraViewController.swift
├── us_coins.tflite
├── us_coins_labels.txt
├── euro_coins.tflite
└── euro_coins_labels.txt
```

---

## Code: How It Works

### Loading Multiple Models
```swift
let modelConfigs = [
    (name: "US Coins", modelFile: "us_coins.tflite", labelsFile: "us_coins_labels.txt"),
    (name: "Euro Coins", modelFile: "euro_coins.tflite", labelsFile: "euro_coins_labels.txt"),
]

let recognizer = MultiModelCoinRecognition(modelConfigs: modelConfigs)
```

### Switching Models
```swift
recognizer.setActiveModel(name: "Euro Coins")  // Switch to Euro model
```

### Running Inference
```swift
let result = recognizer.runInference(onPixelBuffer: pixelBuffer)
// Returns: (label: "penny", confidence: 0.95, modelName: "US Coins")
```

### Getting Available Models
```swift
let models = recognizer.getAvailableModels()
// Returns: ["US Coins", "Euro Coins"]
```

---

## Performance Tips

1. **Model Size**: Smaller models = faster inference
   - Monitor: Each model should be <1MB if possible

2. **Accuracy vs Speed**: 
   - Simpler models (fewer layers) = faster but less accurate
   - Complex models = slower but more accurate

3. **Testing**:
   - Test each model separately first
   - Then test switching between models
   - Check for memory leaks with Xcode Instruments

4. **Optimization**:
   - Use quantization in conversion script
   - Remove unused features from models
   - Profile inference time with device

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model not found" | Check file names in Xcode exactly match config |
| "Labels mismatch" | Ensure model outputs match number of label lines |
| "App crashes on switch" | Deallocate old model properly |
| "Slow switching" | Models may load on demand - optimize memory |
| "Inference always wrong" | Check correct model is active |

---

## Next Steps

1. Train your US coins model
2. Train your Euro coins model
3. Convert both to TensorFlow Lite
4. Add to Xcode with Target Membership
5. Update SceneDelegate to use MultiModelCameraViewController
6. Build and test!

Questions? Check `MultiModelCoinRecognition.swift` for implementation details.
