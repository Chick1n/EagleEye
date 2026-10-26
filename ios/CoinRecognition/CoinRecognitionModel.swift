import UIKit
import TensorFlowLite
import Vision

class CoinRecognitionModel {
    private var interpreter: Interpreter?
    private let inputImageWidth = 224
    private let inputImageHeight = 224
    private let labels: [String]
    private let ciContext = CIContext()
    
    init?() {
        // Load labels
        guard let labelsPath = Bundle.main.path(forResource: "labels_3", ofType: "txt") else {
            print("Labels file not found")
            return nil
        }
        
        do {
            let labelsContent = try String(contentsOfFile: labelsPath, encoding: .utf8)
            self.labels = labelsContent
                .split(separator: "\n")
                .map(String.init)
                .map { raw in
                    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                    let parts = trimmed.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
                    if parts.count == 2, Int(parts[0]) != nil {
                        return String(parts[1])
                    }
                    return trimmed
                }
        } catch {
            print("Error loading labels: \(error)")
            return nil
        }
        
        // Load and initialize the TensorFlow Lite model
        guard let modelPath = Bundle.main.path(forResource: "coin_recognition_3", ofType: "tflite") else {
            print("Model file not found")
            return nil
        }
        
        do {
            self.interpreter = try Interpreter(modelPath: modelPath)
            try self.interpreter?.allocateTensors()
        } catch {
            print("Error loading model: \(error)")
            return nil
        }
    }
    
    func runInference(onPixelBuffer pixelBuffer: CVPixelBuffer) -> (label: String, confidence: Float)? {
        guard let interpreter = interpreter else { return nil }
        
        // Preprocess the image
        guard let inputTensor = try? interpreter.input(at: 0),
              let resizedPixelBuffer = pixelBuffer.resized(to: CGSize(width: inputImageWidth, height: inputImageHeight)) else {
            return nil
        }
        
        do {
            // Convert pixel buffer to tensor data matching model input shape/type.
            guard let imageData = pixelBufferToModelInputData(resizedPixelBuffer, inputTensor: inputTensor) else {
                print("Failed to convert pixel buffer to data")
                return nil
            }
            return try executeInference(inputData: imageData, interpreter: interpreter)
            
        } catch {
            print("Error running inference: \(error)")
            return nil
        }
    }

    func runInference(onCGImage image: CGImage) -> (label: String, confidence: Float)? {
        guard let interpreter = interpreter,
              let inputTensor = try? interpreter.input(at: 0),
              let inputData = cgImageToModelInputData(image, inputTensor: inputTensor) else {
            return nil
        }

        do {
            return try executeInference(inputData: inputData, interpreter: interpreter)
        } catch {
            print("Error running inference on CGImage: \(error)")
            return nil
        }
    }

    private func executeInference(inputData: Data, interpreter: Interpreter) throws -> (label: String, confidence: Float) {
        try interpreter.copy(inputData, toInputAt: 0)
        try interpreter.invoke()

        let outputTensor = try interpreter.output(at: 0)

        var maxConfidence: Float = 0
        var maxIndex = 0

        if outputTensor.dataType == .float32 {
            let results = outputTensor.data.withUnsafeBytes {
                Array($0.bindMemory(to: Float32.self))
            }

            for (index, confidence) in results.enumerated() {
                if confidence > maxConfidence {
                    maxConfidence = confidence
                    maxIndex = index
                }
            }
        } else {
            let results = outputTensor.data.withUnsafeBytes {
                Array($0.bindMemory(to: UInt8.self))
            }

            for (index, confidence) in results.enumerated() {
                let normalized = Float(confidence) / 255.0
                if normalized > maxConfidence {
                    maxConfidence = normalized
                    maxIndex = index
                }
            }
        }

        let label = maxIndex < labels.count ? labels[maxIndex] : "Unknown"
        return (label: label, confidence: maxConfidence)
    }
    
    private func pixelBufferToModelInputData(_ pixelBuffer: CVPixelBuffer, inputTensor: Tensor) -> Data? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return nil
        }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        // Match the Python pipeline used in this project: BGR order and [-1, 1] normalization.
        if inputTensor.dataType == .float32 {
            var bgrFloats = [Float32]()
            bgrFloats.reserveCapacity(width * height * 3)

            for y in 0..<height {
                let row = baseAddress.advanced(by: y * bytesPerRow)
                    .assumingMemoryBound(to: UInt8.self)
                for x in 0..<width {
                    let pixelOffset = x * 4
                    let b = Float32(row[pixelOffset])
                    let g = Float32(row[pixelOffset + 1])
                    let r = Float32(row[pixelOffset + 2])

                    bgrFloats.append((b / 127.5) - 1.0)
                    bgrFloats.append((g / 127.5) - 1.0)
                    bgrFloats.append((r / 127.5) - 1.0)
                }
            }

            return bgrFloats.withUnsafeBufferPointer { Data(buffer: $0) }
        }

        // Fallback path for quantized uint8 models.
        var bgrBytes = [UInt8]()
        bgrBytes.reserveCapacity(width * height * 3)

        for y in 0..<height {
            let row = baseAddress.advanced(by: y * bytesPerRow)
                .assumingMemoryBound(to: UInt8.self)
            for x in 0..<width {
                let pixelOffset = x * 4
                let b = row[pixelOffset]
                let g = row[pixelOffset + 1]
                let r = row[pixelOffset + 2]
                bgrBytes.append(b)
                bgrBytes.append(g)
                bgrBytes.append(r)
            }
        }
        
        return Data(bgrBytes)
    }

    private func cgImageToModelInputData(_ image: CGImage, inputTensor: Tensor) -> Data? {
        let ciImage = CIImage(cgImage: image)
        let sx = CGFloat(inputImageWidth) / CGFloat(image.width)
        let sy = CGFloat(inputImageHeight) / CGFloat(image.height)
        let scaledImage = ciImage.transformed(by: CGAffineTransform(scaleX: sx, y: sy))

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            inputImageWidth,
            inputImageHeight,
            kCVPixelFormatType_32BGRA,
            nil,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let pixelBuffer = pixelBuffer else {
            return nil
        }

        ciContext.render(scaledImage, to: pixelBuffer)
        return pixelBufferToModelInputData(pixelBuffer, inputTensor: inputTensor)
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
