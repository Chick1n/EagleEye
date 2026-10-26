import UIKit
import AVFoundation

class MainPyStyleViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
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
    
    // UI Elements
    private let resultLabel = UILabel()
    private let confidenceLabel = UILabel()
    private let cameraButton = UIButton()
    private let debugLabel = UILabel()
    private var debugLines: [String] = []
    private let showDebugOverlay = true  // Show debug window
    
    private struct DetectionResult {
        let rect: CGRect
        let label: String
        let confidence: Float
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        title = "Coin Recognition (main.py style)"
        view.backgroundColor = .black
        
        setupUI()
        setupDebugObservers()
        model = CoinRecognitionModel()
        if model == nil {
            logDebug("model load failed")
            resultLabel.text = "Camera ready, model failed"
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
        
        // Debug Label (like main.py preprocessing window)
        debugLabel.textColor = .systemYellow
        debugLabel.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        debugLabel.font = UIFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        debugLabel.numberOfLines = 0
        debugLabel.text = "Debug: Initializing..."
        debugLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(debugLabel)
        
        // Constraints
        NSLayoutConstraint.activate([
            resultLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            resultLabel.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            
            confidenceLabel.topAnchor.constraint(equalTo: resultLabel.bottomAnchor, constant: 16),
            confidenceLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            
            cameraButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            cameraButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            cameraButton.widthAnchor.constraint(equalToConstant: 150),
            cameraButton.heightAnchor.constraint(equalToConstant: 50),
            
            debugLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            debugLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 12),
            debugLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -12),
            debugLabel.heightAnchor.constraint(equalToConstant: 120)
        ])
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
            }
            
            videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            if captureSession.canAddOutput(videoDataOutput) {
                captureSession.addOutput(videoDataOutput)
                logDebug("output added")
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
            
            view.bringSubviewToFront(resultLabel)
            view.bringSubviewToFront(confidenceLabel)
            view.bringSubviewToFront(cameraButton)
            view.bringSubviewToFront(debugLabel)
            
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
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    if granted {
                        self.setupCamera()
                    } else {
                        self.showError("Camera permission denied")
                    }
                }
            }
        case .denied, .restricted:
            showError("Camera permission denied")
        @unknown default:
            showError("Unable to access camera")
        }
    }
    
    private func setupDebugObservers() {
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(sessionRuntimeError(_:)),
                                               name: .AVCaptureSessionRuntimeError,
                                               object: captureSession)
    }
    
    @objc private func sessionRuntimeError(_ notification: Notification) {
        let err = notification.userInfo?[AVCaptureSessionErrorKey] as? NSError
        logDebug("runtime error: \(err?.localizedDescription ?? "unknown")")
    }
    
    private func logDebug(_ message: String) {
        let line = "[\(timestamp())] \(message)"
        debugLines.append(line)
        if debugLines.count > 6 {
            debugLines.removeFirst(debugLines.count - 6)
        }
        debugLabel.text = debugLines.joined(separator: "\n")
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
            
            // Preprocess (like main.py)
            let preprocessed = self.preprocessImage(frameImage)
            
            // Detect ellipses (like main.py)
            let ellipses = self.detectEllipses(in: preprocessed)
            
            // Process largest ellipse
            if let largestEllipse = ellipses.max(by: { $0.width * $0.height < $1.width * $1.height }) {
                // Crop and classify
                if let crop = frameImage.cropping(to: largestEllipse),
                   let result = model.runInference(onCGImage: crop),
                   result.confidence >= self.minConfidence {
                    
                    let isReal = result.confidence > self.realThreshold
                    
                    DispatchQueue.main.async {
                        self.renderResult(
                            ellipse: largestEllipse,
                            label: result.label,
                            confidence: result.confidence,
                            isReal: isReal
                        )
                    }
                }
            }
        }
    }
    
    private func cgImage(from pixelBuffer: CVPixelBuffer) -> CGImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let rect = CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
        return ciContext.createCGImage(ciImage, from: rect)
    }
    
    private func preprocessImage(_ image: CGImage) -> CGImage {
        // Simplified preprocessing (Gaussian blur + edge detection)
        // In real implementation, you'd use OpenCV or similar
        return image
    }
    
    private func detectEllipses(in image: CGImage) -> [CGRect] {
        // Simplified ellipse detection
        // In real implementation, you'd use OpenCV or similar
        return []
    }
    
    private func renderResult(ellipse: CGRect, label: String, confidence: Float, isReal: Bool) {
        overlayLayer.sublayers?.forEach { $0.removeFromSuperlayer() }
        
        let color = isReal ? UIColor.systemGreen : UIColor.systemRed
        let status = isReal ? "✓ REAL" : "✗ FAKE?"
        
        // Draw ellipse (like main.py)
        let boxLayer = CAShapeLayer()
        boxLayer.frame = ellipse
        boxLayer.borderWidth = 3
        boxLayer.borderColor = color.cgColor
        boxLayer.cornerRadius = ellipse.width / 2
        boxLayer.backgroundColor = UIColor.clear.cgColor
        overlayLayer.addSublayer(boxLayer)
        
        // Update labels
        resultLabel.text = "\(status) \(label)"
        resultLabel.textColor = color
        confidenceLabel.text = String(format: "Confidence: %.2f%%", confidence * 100)
        
        logDebug("Detected: \(label) (\(confidence*100)%)")
    }
    
    @objc private func toggleCamera() {
        if captureSession.isRunning {
            captureSession.stopRunning()
            logDebug("camera stopped")
            cameraButton.setTitle("Start Camera", for: .normal)
        } else {
            DispatchQueue.global(qos: .background).async {
                self.captureSession.startRunning()
                DispatchQueue.main.async {
                    self.logDebug("camera started")
                }
            }
            cameraButton.setTitle("Stop Camera", for: .normal)
        }
    }
    
    private func showError(_ message: String) {
        logDebug("error: \(message)")
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
}
