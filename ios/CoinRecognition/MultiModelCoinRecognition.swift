import UIKit
import TensorFlowLite
import Vision

// MARK: - Model Definition
struct CoinModel {
    let name: String              // e.g., "US Coins", "Euro Coins"
    let modelFileName: String     // e.g., "us_coins.tflite"
    let labelsFileName: String    // e.g., "us_coins_labels.txt"
    var interpreter: Interpreter?
    var labels: [String] = []
}

// MARK: - Multi-Model Coin Recognition
class MultiModelCoinRecognition {
    
    private var models: [String: CoinModel] = [:]  // Key: model name
    private var activeModelName: String = ""
    
    private let inputImageWidth = 224
    private let inputImageHeight = 224
    
    // Initialize with multiple models
    init?(modelConfigs: [(name: String, modelFile: String, labelsFile: String)]) {
        for config in modelConfigs {
            var model = CoinModel(
                name: config.name,
                modelFileName: config.modelFile,
                labelsFileName: config.labelsFile
            )
            
            // Load labels
            guard let labels = loadLabels(fileName: config.labelsFile) else {
                print("Failed to load labels for \(config.name)")
                return nil
            }
            model.labels = labels
            
            // Load model interpreter
            guard let interpreter = loadModel(fileName: config.modelFile) else {
                print("Failed to load model for \(config.name)")
                return nil
            }
            model.interpreter = interpreter
            
            models[config.name] = model
            
            // Set first model as active
            if activeModelName.isEmpty {
                activeModelName = config.name
            }
        }
        
        guard !models.isEmpty else {
            print("No models loaded")
            return nil
        }
    }
    
    // MARK: - Model Management
    
    func getAvailableModels() -> [String] {
        return Array(models.keys)
    }
    
    func setActiveModel(name: String) -> Bool {
        guard models[name] != nil else {
            print("Model '\(name)' not found")
            return false
        }
        activeModelName = name
        return true
    }
    
    func getActiveModel() -> String {
        return activeModelName
    }
    
    // MARK: - Inference
    
    func runInference(onPixelBuffer pixelBuffer: CVPixelBuffer) -> (label: String, confidence: Float, modelName: String)? {
        guard let model = models[activeModelName],
              let interpreter = model.interpreter else {
            return nil
        }
        
        // Preprocess the image
        guard let resizedPixelBuffer = pixelBuffer.resized(to: CGSize(width: inputImageWidth, height: inputImageHeight)) else {
            return nil
        }
        
        do {
            // Copy resized image data to input tensor
            try interpreter.copy(resizedPixelBuffer, toInputAt: 0)
            
            // Run inference
            try interpreter.invoke()
            
            // Get output tensor
            let outputTensor = try interpreter.output(at: 0)
            
            // Get the results
            let results = UnsafeMutableBufferPointer<Float32>(
                start: outputTensor.data.assumingMemoryBound(to: Float32.self),
                count: outputTensor.shape.dimensions.reduce(1, { $0 * $1 })
            )
            
            // Find the label with highest confidence
            var maxConfidence: Float = 0
            var maxIndex = 0
            
            for (index, confidence) in results.enumerated() {
                if confidence > maxConfidence {
                    maxConfidence = confidence
                    maxIndex = index
                }
            }
            
            let label = maxIndex < model.labels.count ? model.labels[maxIndex] : "Unknown"
            return (label: label, confidence: maxConfidence, modelName: activeModelName)
            
        } catch {
            print("Error running inference: \(error)")
            return nil
        }
    }
    
    // Run inference on all models and return best result
    func runInferenceAllModels(onPixelBuffer pixelBuffer: CVPixelBuffer) -> [(label: String, confidence: Float, modelName: String)] {
        var results: [(label: String, confidence: Float, modelName: String)] = []
        
        for modelName in models.keys {
            setActiveModel(name: modelName)
            if let result = runInference(onPixelBuffer: pixelBuffer) {
                results.append(result)
            }
        }
        
        return results
    }
    
    // MARK: - Helper Methods
    
    private func loadLabels(fileName: String) -> [String]? {
        guard let labelsPath = Bundle.main.path(forResource: fileName.replacingOccurrences(of: ".txt", with: ""), ofType: "txt") else {
            print("Labels file not found: \(fileName)")
            return nil
        }
        
        do {
            let labelsContent = try String(contentsOfFile: labelsPath, encoding: .utf8)
            return labelsContent.split(separator: "\n").map(String.init)
        } catch {
            print("Error loading labels: \(error)")
            return nil
        }
    }
    
    private func loadModel(fileName: String) -> Interpreter? {
        guard let modelPath = Bundle.main.path(forResource: fileName.replacingOccurrences(of: ".tflite", with: ""), ofType: "tflite") else {
            print("Model file not found: \(fileName)")
            return nil
        }
        
        do {
            let interpreter = try Interpreter(modelPath: modelPath)
            try interpreter.allocateTensors()
            return interpreter
        } catch {
            print("Error loading model: \(error)")
            return nil
        }
    }
}

// MARK: - CVPixelBuffer Extension
extension CVPixelBuffer {
    func resized(to size: CGSize) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(self)
        let height = CVPixelBufferGetHeight(self)
        
        let scale = CGAffineTransform(scaleX: size.width / CGFloat(width),
                                     y: size.height / CGFloat(height))
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(size.width),
            Int(size.height),
            kCVPixelFormatType_32BGRA,
            nil,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let pixelBuffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(self, .readOnly)
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        
        defer {
            CVPixelBufferUnlockBaseAddress(self, .readOnly)
            CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
        }
        
        let context = CIContext()
        let ciImage = CIImage(cvPixelBuffer: self)
        let transformed = ciImage.transformed(by: scale)
        
        context.render(transformed, to: pixelBuffer)
        
        return pixelBuffer
    }
}
