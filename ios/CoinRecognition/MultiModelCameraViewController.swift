import UIKit
import AVFoundation

class MultiModelCameraViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    private let captureSession = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let previewLayer = AVCaptureVideoPreviewLayer()
    private var modelRecognition: MultiModelCoinRecognition?
    
    // UI Elements
    private let resultLabel = UILabel()
    private let confidenceLabel = UILabel()
    private let modelLabel = UILabel()
    private let modelSegmentControl = UISegmentedControl()
    private let cameraButton = UIButton()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        title = "Multi-Model Coin Recognition"
        view.backgroundColor = .black
        
        // Initialize multi-model system
        // Example with US and Euro coins
        let modelConfigs = [
            (name: "US Coins", modelFile: "us_coins.tflite", labelsFile: "us_coins_labels.txt"),
            (name: "Euro Coins", modelFile: "euro_coins.tflite", labelsFile: "euro_coins_labels.txt")
        ]
        
        modelRecognition = MultiModelCoinRecognition(modelConfigs: modelConfigs)
        guard modelRecognition != nil else {
            showError("Failed to load models")
            return
        }
        
        setupUI()
        setupCamera()
    }
    
    private func setupUI() {
        // Model Selection Label
        modelLabel.text = "Select Model:"
        modelLabel.textColor = .white
        modelLabel.font = UIFont.systemFont(ofSize: 12)
        modelLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(modelLabel)
        
        // Model Selection Segment Control
        let models = modelRecognition?.getAvailableModels() ?? []
        modelSegmentControl.insertSegment(withTitle: "", at: 0)
        for (index, model) in models.enumerated() {
            modelSegmentControl.insertSegment(withTitle: model, at: index)
        }
        modelSegmentControl.selectedSegmentIndex = 0
        modelSegmentControl.addTarget(self, action: #selector(modelChanged(_:)), for: .valueChanged)
        modelSegmentControl.backgroundColor = .systemGray6
        modelSegmentControl.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(modelSegmentControl)
        
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
        
        // Constraints
        NSLayoutConstraint.activate([
            modelLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            modelLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            
            modelSegmentControl.topAnchor.constraint(equalTo: modelLabel.bottomAnchor, constant: 8),
            modelSegmentControl.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            modelSegmentControl.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            
            resultLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            resultLabel.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            
            confidenceLabel.topAnchor.constraint(equalTo: resultLabel.bottomAnchor, constant: 16),
            confidenceLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            
            cameraButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            cameraButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            cameraButton.widthAnchor.constraint(equalToConstant: 150),
            cameraButton.heightAnchor.constraint(equalToConstant: 50)
        ])
    }
    
    private func setupCamera() {
        captureSession.sessionPreset = .high
        
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            showError("Camera not available")
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: camera)
            captureSession.addInput(input)
            
            videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
            captureSession.addOutput(videoDataOutput)
            
            previewLayer.session = captureSession
            previewLayer.videoGravity = .resizeAspectFill
            view.layer.insertSublayer(previewLayer, below: resultLabel.layer)
            previewLayer.frame = view.bounds
            
            DispatchQueue.global(qos: .background).async {
                self.captureSession.startRunning()
            }
        } catch {
            showError("Camera setup error: \(error.localizedDescription)")
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let modelRecognition = modelRecognition,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let result = modelRecognition.runInference(onPixelBuffer: pixelBuffer)
        
        DispatchQueue.main.async {
            if let result = result {
                self.resultLabel.text = result.label
                self.confidenceLabel.text = String(format: "Confidence: %.2f%% (%s)", result.confidence * 100, result.modelName)
            }
        }
    }
    
    @objc private func modelChanged(_ sender: UISegmentedControl) {
        guard let selectedModel = sender.titleForSegment(at: sender.selectedSegmentIndex) else {
            return
        }
        
        if modelRecognition?.setActiveModel(name: selectedModel) == true {
            resultLabel.text = "Model switched to \(selectedModel)"
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                self.resultLabel.text = "Point at a coin"
            }
        }
    }
    
    @objc private func toggleCamera() {
        if captureSession.isRunning {
            captureSession.stopRunning()
            cameraButton.setTitle("Start Camera", for: .normal)
        } else {
            DispatchQueue.global(qos: .background).async {
                self.captureSession.startRunning()
            }
            cameraButton.setTitle("Stop Camera", for: .normal)
        }
    }
    
    private func showError(_ message: String) {
        let alert = UIAlertController(title: "Error", message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
    
    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()
        previewLayer.frame = view.bounds
    }
    
    deinit {
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
    }
}
