# Coin Recognition iOS App

A native iOS app that uses TensorFlow Lite to recognize coins in real-time using the device camera.

## Prerequisites

- macOS with Xcode 14+ installed
- CocoaPods (`sudo gem install cocoapods`)
- iOS 14.0+

## Setup Instructions

### Step 1: Convert Keras Model to TensorFlow Lite

From the project root directory, run the conversion script:

```bash
python convert_to_tflite.py
```

This will create `model/coin_recognition.tflite` from your Keras model.

### Step 2: Install Dependencies

Navigate to the iOS directory and install pods:

```bash
cd ios
pod install
```

### Step 3: Add Model Files to Xcode

1. Open `CoinRecognition.xcworkspace` in Xcode (NOT the `.xcodeproj`)
2. Copy `model/coin_recognition.tflite` to the project
3. Copy `model/labels.txt` to the project
4. In Xcode: Select both files → Target Membership → Check "CoinRecognition"

### Step 4: Build and Run

1. Select an iOS device or simulator in Xcode
2. Press `Cmd + R` to build and run
3. Grant camera permission when prompted
4. Point the camera at a coin to test

## Project Structure

```
ios/
├── CoinRecognition/
│   ├── AppDelegate.swift           # App entry point
│   ├── SceneDelegate.swift         # Scene setup
│   ├── CameraViewController.swift   # Camera & UI handling
│   ├── CoinRecognitionModel.swift   # Model inference logic
│   └── Info.plist                  # App configuration (camera permission)
└── Podfile                         # CocoaPods dependencies
```

## Files to Update in Xcode

### Info.plist

Add camera permission description:
```xml
<key>NSCameraUsageDescription</key>
<string>We need access to your camera to recognize coins</string>
```

## Model Details

- **Input Size**: 224x224 pixels
- **Input Format**: BGRA pixel buffer
- **Output**: Class probabilities for each coin type
- **Model Format**: TensorFlow Lite (.tflite)

## Troubleshooting

### "Model file not found"
- Ensure `coin_recognition.tflite` is added to the Xcode project
- Verify it's included in Target Membership

### "Labels file not found"
- Ensure `labels.txt` is added to the Xcode project
- Verify it's included in Target Membership

### Camera permission denied
- Check `Info.plist` has `NSCameraUsageDescription`
- Reset app permissions in Settings → General → iPhone Storage → App

### Model inference is slow
- Test on a physical device (simulators run slower)
- Check model size with `ls -lh model/coin_recognition.tflite`

## Building for Release (IPA)

1. In Xcode: Product → Scheme → Edit Scheme
2. Select "Release" build configuration
3. Product → Archive
4. In Organizer window: "Distribute App"
5. Follow the export steps to generate IPA

## Next Steps

- Customize the UI in `CameraViewController.swift`
- Add real-time statistics and history
- Implement confidence threshold filtering
- Add vibration/sound feedback for detections

## Support

For issues with TensorFlow Lite: https://www.tensorflow.org/lite/guide/swift
For iOS development: https://developer.apple.com/ios/
