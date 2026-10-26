# iOS App Deployment Guide

Complete step-by-step guide to deploy your Coin Recognition app to iPhone.

## Phase 1: Convert Model (Python - Your Mac)

### Step 1.1: Install TensorFlow Lite Converter
```bash
pip install tensorflow
```

### Step 1.2: Run Conversion Script
```bash
cd /Users/dongyuliu/wkspace/CoinRecognition
python convert_to_tflite.py
```

**Expected output:**
```
Loading Keras model from model/keras_model.h5...
Converting to TensorFlow Lite...
Saving TensorFlow Lite model to model/coin_recognition.tflite...
✓ Conversion successful!
  Model size: X.XX MB
  Output file: model/coin_recognition.tflite
```

**Verify:** Check that `model/coin_recognition.tflite` was created:
```bash
ls -lh model/coin_recognition.tflite
```

---

## Phase 2: Create Xcode Project (Xcode)

### Step 2.1: Create New Project
1. Open Xcode
2. File → New → Project
3. Choose "App"
4. Fill in:
   - Product Name: `CoinRecognition`
   - Team: Select your team
   - Organization: Your name
   - Bundle Identifier: `com.yourname.coinrecognition`
   - Interface: SwiftUI (or Storyboard)
   - Language: Swift
5. Click Create

### Step 2.2: Replace Swift Files
In Xcode project:
1. Delete the auto-generated SwiftUI files
2. Copy these files from `/Users/dongyuliu/wkspace/CoinRecognition/ios/CoinRecognition/`:
   - `AppDelegate.swift`
   - `SceneDelegate.swift`
   - `CameraViewController.swift`
   - `CoinRecognitionModel.swift`
   - `Info.plist`

---

## Phase 3: Add Dependencies (CocoaPods)

### Step 3.1: Install CocoaPods (if not already installed)
```bash
sudo gem install cocoapods
```

### Step 3.2: Setup CocoaPods
```bash
cd ios
pod install
```

### Step 3.3: Close Xcode
- Close the `.xcodeproj` file

### Step 3.4: Open Workspace
- Open `CoinRecognition.xcworkspace` (not `.xcodeproj`)

---

## Phase 4: Add Model Files to Xcode

### Step 4.1: Copy Model Files
In Xcode:
1. Right-click project → Add Files to "CoinRecognition"
2. Navigate to `/Users/dongyuliu/wkspace/CoinRecognition/model/`
3. Select both:
   - `coin_recognition.tflite`
   - `labels.txt`
4. Check "Copy items if needed"
5. Check target: `CoinRecognition`
6. Click Add

### Step 4.2: Verify Files
- Both files should appear in Xcode project tree
- Check Target Membership for both files

---

## Phase 5: Configure App Settings

### Step 5.1: Set Deployment Target
1. Select project in Xcode
2. Select target "CoinRecognition"
3. General tab:
   - Minimum Deployment: iOS 14.0
   - Supported Orientations: Portrait, Landscape Left, Landscape Right

### Step 5.2: Configure Info.plist
- Ensure `NSCameraUsageDescription` is set
- Value: "We need access to your camera to recognize coins"

### Step 5.3: Enable Camera Permission
1. Select target → Signing & Capabilities
2. + Capability → Camera
3. A camera entitlement should be added

---

## Phase 6: Build and Test

### Step 6.1: Build
1. Select device/simulator (recommend physical device)
2. Xcode → Product → Build (Cmd+B)

### Step 6.2: Run
1. Xcode → Product → Run (Cmd+R)
2. Grant camera permission when prompted
3. Point camera at a coin

### Step 6.3: Troubleshoot
If build fails:
- Clean build folder: Cmd+Shift+K
- Delete Derived Data: Xcode → Preferences → Locations → Derived Data
- Delete Pods folder and run `pod install` again

---

## Phase 7: Generate IPA for Distribution

### Step 7.1: Create Signing Certificate
1. Xcode → Preferences → Accounts
2. Add your Apple ID
3. Manage Certificates → Create Certificate

### Step 7.2: Archive
1. Select "Generic iOS Device" or your device
2. Product → Archive
3. Wait for archive to complete

### Step 7.3: Export IPA
1. In Organizer window, select archive
2. Click "Distribute App"
3. Choose "Custom"
4. Follow export wizard steps
5. Save IPA file

---

## File Checklist

- [x] `convert_to_tflite.py` - Model conversion script
- [x] `model/coin_recognition.tflite` - Converted model (generated after running conversion script)
- [x] `model/labels.txt` - Class labels
- [x] `ios/CoinRecognition/AppDelegate.swift` - App entry
- [x] `ios/CoinRecognition/SceneDelegate.swift` - Scene setup
- [x] `ios/CoinRecognition/CameraViewController.swift` - UI & camera
- [x] `ios/CoinRecognition/CoinRecognitionModel.swift` - Model inference
- [x] `ios/CoinRecognition/Info.plist` - App configuration
- [x] `ios/Podfile` - Dependencies
- [x] `ios/README.md` - Setup guide

---

## Quick Reference Commands

### Python Setup
```bash
# Convert model
python convert_to_tflite.py

# Verify model
ls -lh model/coin_recognition.tflite
```

### Xcode Setup
```bash
# Install pods
cd ios && pod install

# Clean build (if needed)
rm -rf Pods
rm Podfile.lock
pod install
```

---

## Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| "Module not found" | Run `pod install` again, close and reopen workspace |
| "Camera permission denied" | Reinstall app on device, check Info.plist |
| "Model file not found" | Add files to Xcode and check Target Membership |
| "Build fails with TFLite error" | Delete Pods folder, run `pod install` |
| "Slow inference" | Test on real device, check model size |

---

## Next Steps After Launch

1. **Customize UI**: Edit `CameraViewController.swift`
2. **Add Features**: History, confidence thresholds, sound/vibration
3. **Test Extensively**: Different lighting, coin angles, distances
4. **Optimize**: Profile inference speed, optimize model size
5. **Publish**: TestFlight or App Store (requires paid developer account)

Good luck! 🎉
