import UIKit
import AVFoundation

class CameraViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    private let captureSession = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let previewLayer = AVCaptureVideoPreviewLayer()
    private let overlayLayer = CALayer()
    private var model: CoinRecognitionModel?
    private let ciContext = CIContext()
    private let processingQueue = DispatchQueue(label: "coin.processing.queue")
    private var isProcessingFrame = false
    private var frameCounter = 0

    private let processEveryNFrames = 3
    private let minConfidence: Float = 0.50
    private let realThreshold: Float = 0.90
    private let maxDetections = 1  // Match main.py - process only largest coin
    
    // UI Elements
    private let resultLabel = UILabel()
    private let confidenceLabel = UILabel()
    private let cameraButton = UIButton()
    private let debugLabel = UILabel()
    private var debugLines: [String] = []
    private let showDebugOverlay = false

    private struct DetectionResult {
        let rect: CGRect
        let label: String
        let confidence: Float
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        title = "Coin Recognition"
        view.backgroundColor = .black
        
        setupUI()
        setupDebugObservers()

        // Initialize model, but do not block camera startup if model loading fails.
        model = CoinRecognitionModel()
        if model == nil {
            logDebug("model load failed")
            resultLabel.text = "Camera ready, model failed"
            confidenceLabel.text = "Check .tflite and labels bundle"
        } else {
            logDebug("model load ok")
        }

        configureCameraAccessAndStart()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
        overlayLayer.frame = view.bounds
    }
    
    private func setupUI() {
        // Result Label
        resultLabel.textColor = .white
        resultLabel.font = UIFont.systemFont(ofSize: 24, weight: .bold)
        resultLabel.textAlignment = .center
        resultLabel.text = "Point at a coin"
        resultLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(resultLabel)
        
        // Confidence Label
        confidenceLabel.textColor = .lightGray
        confidenceLabel.font = UIFont.systemFont(ofSize: 16)
        confidenceLabel.textAlignment = .center
        confidenceLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(confidenceLabel)
        
        // Camera Button
        cameraButton.setTitle("Toggle Camera", for: .normal)
        cameraButton.backgroundColor = .systemBlue
        cameraButton.layer.cornerRadius = 8
        cameraButton.addTarget(self, action: #selector(toggleCamera), for: .touchUpInside)
        cameraButton.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(cameraButton)

        if showDebugOverlay {
            // Debug Label
            debugLabel.textColor = .systemYellow
            debugLabel.backgroundColor = UIColor.black.withAlphaComponent(0.55)
            debugLabel.font = UIFont.monospacedSystemFont(ofSize: 12, weight: .regular)
            debugLabel.numberOfLines = 0
            debugLabel.text = "Debug: app launched"
            debugLabel.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview(debugLabel)
        }
        
        // Constraints
        NSLayoutConstraint.activate([
            resultLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            resultLabel.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            
            confidenceLabel.topAnchor.constraint(equalTo: resultLabel.bottomAnchor, constant: 16),
            confidenceLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            
            cameraButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            cameraButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            cameraButton.widthAnchor.constraint(equalToConstant: 150),
            cameraButton.heightAnchor.constraint(equalToConstant: 50)
        ])

        if showDebugOverlay {
            NSLayoutConstraint.activate([
                debugLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
                debugLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 12),
                debugLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -12)
            ])
        }
    }
    
    private func setupCamera() {
        logDebug("setupCamera start")
        captureSession.sessionPreset = .high
        
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            logDebug("back camera unavailable")
            showError("Camera not available")
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: camera)
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
                logDebug("input added")
            } else {
                logDebug("cannot add camera input")
                showError("Unable to add camera input")
                return
            }
            
            videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            if captureSession.canAddOutput(videoDataOutput) {
                captureSession.addOutput(videoDataOutput)
                logDebug("output added")
            } else {
                logDebug("cannot add video output")
                showError("Unable to add camera output")
                return
            }

            if let connection = videoDataOutput.connection(with: .video), connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
            
            previewLayer.session = captureSession
            previewLayer.videoGravity = .resizeAspectFill
            view.layer.insertSublayer(previewLayer, at: 0)
            previewLayer.frame = view.bounds

            overlayLayer.frame = view.bounds
            overlayLayer.masksToBounds = true
            view.layer.insertSublayer(overlayLayer, above: previewLayer)
            logDebug("preview attached")

            // Ensure overlays stay visible above the camera preview.
            view.bringSubviewToFront(resultLabel)
            view.bringSubviewToFront(confidenceLabel)
            view.bringSubviewToFront(cameraButton)
            if showDebugOverlay {
                view.bringSubviewToFront(debugLabel)
            }
            
            DispatchQueue.global(qos: .background).async {
                self.captureSession.startRunning()
                DispatchQueue.main.async {
                    self.logDebug("session running=\(self.captureSession.isRunning)")
                }
            }
        } catch {
            logDebug("setupCamera error: \(error.localizedDescription)")
            showError("Camera setup error: \(error.localizedDescription)")
        }
    }

    private func configureCameraAccessAndStart() {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        logDebug("camera auth: \(status.rawValue)")
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            setupCamera()
        case .notDetermined:
            logDebug("requesting camera permission")
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    self.logDebug("camera permission granted=\(granted)")
                    if granted {
                        self.setupCamera()
                    } else {
                        self.showError("Camera permission denied")
                    }
                }
            }
        case .denied, .restricted:
            logDebug("camera denied/restricted")
            showError("Camera permission denied. Enable it in Settings > Privacy & Security > Camera.")
        @unknown default:
            logDebug("camera status unknown")
            showError("Unable to access camera")
        }
    }

    private func setupDebugObservers() {
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(sessionRuntimeError(_:)),
                                               name: .AVCaptureSessionRuntimeError,
                                               object: captureSession)
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(sessionInterrupted(_:)),
                                               name: .AVCaptureSessionWasInterrupted,
                                               object: captureSession)
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(sessionInterruptionEnded(_:)),
                                               name: .AVCaptureSessionInterruptionEnded,
                                               object: captureSession)
    }

    @objc private func sessionRuntimeError(_ notification: Notification) {
        let err = notification.userInfo?[AVCaptureSessionErrorKey] as? NSError
        logDebug("runtime error: \(err?.localizedDescription ?? "unknown")")
    }

    @objc private func sessionInterrupted(_ notification: Notification) {
        if let reasonValue = notification.userInfo?[AVCaptureSessionInterruptionReasonKey] as? Int,
           let reason = AVCaptureSession.InterruptionReason(rawValue: reasonValue) {
            logDebug("session interrupted: \(reason.rawValue)")
        } else {
            logDebug("session interrupted")
        }
    }

    @objc private func sessionInterruptionEnded(_ notification: Notification) {
        logDebug("session interruption ended")
    }

    private func logDebug(_ message: String) {
        let line = "[\(timestamp())] \(message)"
        debugLines.append(line)
        if debugLines.count > 8 {
            debugLines.removeFirst(debugLines.count - 8)
        }
        if showDebugOverlay {
            debugLabel.text = debugLines.joined(separator: "\n")
        }
        print("CameraDebug: \(line)")
    }

    private func timestamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter.string(from: Date())
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let model = model,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        frameCounter += 1
        if frameCounter % processEveryNFrames != 0 || isProcessingFrame {
            return
        }

        isProcessingFrame = true

        processingQueue.async {
            defer { self.isProcessingFrame = false }

            guard let frameImage = self.cgImage(from: pixelBuffer) else {
                return
            }

            let candidateRects = self.detectCoinCandidateRects(in: frameImage)
            var detections: [DetectionResult] = []

            for rect in candidateRects.prefix(self.maxDetections) {
                guard let crop = frameImage.cropping(to: rect),
                      let result = model.runInference(onCGImage: crop),
                      result.confidence >= self.minConfidence else {
                    continue
                }

                detections.append(DetectionResult(rect: rect, label: result.label, confidence: result.confidence))
            }

            DispatchQueue.main.async {
                self.renderDetections(detections, imageSize: CGSize(width: frameImage.width, height: frameImage.height))
            }
        }
    }
    
    @objc private func toggleCamera() {
        if captureSession.isRunning {
            captureSession.stopRunning()
            logDebug("camera stopped by user")
            cameraButton.setTitle("Start Camera", for: .normal)
        } else {
            DispatchQueue.global(qos: .background).async {
                self.captureSession.startRunning()
                DispatchQueue.main.async {
                    self.logDebug("camera start requested, running=\(self.captureSession.isRunning)")
                }
            }
            cameraButton.setTitle("Stop Camera", for: .normal)
        }
    }
    
    private func showError(_ message: String) {
        logDebug("error shown: \(message)")
        let alert = UIAlertController(title: "Error", message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
    
    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()
        previewLayer.frame = view.bounds
        overlayLayer.frame = view.bounds
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
    }

    private func cgImage(from pixelBuffer: CVPixelBuffer) -> CGImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let rect = CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
        return ciContext.createCGImage(ciImage, from: rect)
    }

    private func detectCoinCandidateRects(in image: CGImage) -> [CGRect] {
        let rectValues = OpenCVBridge.detectCoinRects(in: image)
        let rects = rectValues.map { $0.cgRectValue }
        let sorted = rects.sorted { ($0.width * $0.height) > ($1.width * $1.height) }
        return deduplicated(rects: sorted)
    }

    private func deduplicated(rects: [CGRect]) -> [CGRect] {
        var kept: [CGRect] = []
        for rect in rects {
            if kept.allSatisfy({ iou($0, rect) < 0.4 }) {
                kept.append(rect)
            }
            if kept.count >= maxDetections {
                break
            }
        }
        return kept
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        if inter.isNull {
            return 0
        }
        let interArea = inter.width * inter.height
        let unionArea = (a.width * a.height) + (b.width * b.height) - interArea
        guard unionArea > 0 else { return 0 }
        return interArea / unionArea
    }

    private func renderDetections(_ detections: [DetectionResult], imageSize: CGSize) {
        overlayLayer.sublayers?.forEach { $0.removeFromSuperlayer() }

        if detections.isEmpty {
            resultLabel.text = "Point at a coin"
            confidenceLabel.text = "No confident detections"
            return
        }

        let sorted = detections.sorted { $0.confidence > $1.confidence }
        //resultLabel.text = sorted.first?.label
        // Show top result with real/fake indicator
        if let top = sorted.first {
            let isReal = top.confidence > realThreshold
            let statusIcon = isReal ? "✓" : "✗"
            let statusText = isReal ? "REAL" : "FAKE?"
            resultLabel.text = "\(statusIcon) \(statusText) \(top.label)"
            resultLabel.textColor = isReal ? .green : .red
        }
        confidenceLabel.text = String(format: "Detected %d coin(s) • Top %.2f%%", sorted.count, (sorted.first?.confidence ?? 0) * 100)

        for detection in sorted {
            let normalized = CGRect(
                x: detection.rect.minX / imageSize.width,
                y: detection.rect.minY / imageSize.height,
                width: detection.rect.width / imageSize.width,
                height: detection.rect.height / imageSize.height
            )

            let previewRect = previewLayer.layerRectConverted(fromMetadataOutputRect: normalized)

            let diameter = min(previewRect.width, previewRect.height)
            let circleFrame = CGRect(
                x: previewRect.midX - (diameter / 2.0),
                y: previewRect.midY - (diameter / 2.0),
                width: diameter,
                height: diameter
            )

            let boxLayer = CAShapeLayer()
            boxLayer.frame = circleFrame
            boxLayer.borderWidth = 2
            //boxLayer.borderColor = UIColor.systemGreen.cgColor
            let isReal = detection.confidence > self.realThreshold
            boxLayer.borderColor = isReal ? UIColor.systemGreen.cgColor : UIColor.systemRed.cgColor
            boxLayer.cornerRadius = diameter / 2.0
            boxLayer.backgroundColor = UIColor.clear.cgColor
            overlayLayer.addSublayer(boxLayer)

            let textLayer = CATextLayer()
            textLayer.contentsScale = UIScreen.main.scale
            textLayer.alignmentMode = .center
            textLayer.fontSize = 17
            textLayer.foregroundColor = UIColor.white.cgColor
            textLayer.backgroundColor = UIColor.black.withAlphaComponent(0.6).cgColor
            textLayer.cornerRadius = 4
            //textLayer.string = String(format: "%@ %.0f%%", detection.label, detection.confidence * 100)
            let statusIcon = detection.confidence > self.realThreshold ? "✓" : "✗"
            textLayer.string = String(format: "%@ %@ %.0f%%", statusIcon, detection.label, detection.confidence * 100)
            textLayer.frame = CGRect(x: circleFrame.minX,
                                     y: max(0, circleFrame.minY - 30),
                                     width: max(140, circleFrame.width),
                                     height: 26)
            overlayLayer.addSublayer(textLayer)
        }

        view.bringSubviewToFront(resultLabel)
        view.bringSubviewToFront(confidenceLabel)
        view.bringSubviewToFront(cameraButton)
        if showDebugOverlay {
            view.bringSubviewToFront(debugLabel)
        }
    }
}
