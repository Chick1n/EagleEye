#import <opencv2/core.hpp>
#import <opencv2/imgproc.hpp>
#import <opencv2/imgcodecs/ios.h>

#import <UIKit/UIKit.h>
#import "OpenCVBridge.h"

@implementation OpenCVBridge

static float biggestDifference(const std::vector<float>& values) {
    if (values.size() != 4) {
        return 0.0f;
    }

    float biggest = std::abs(values[0] - values[1]);
    for (size_t i = 0; i < values.size(); ++i) {
        for (size_t j = i + 1; j < values.size(); ++j) {
            float diff = std::abs(values[i] - values[j]);
            if (diff > biggest) {
                biggest = diff;
            }
        }
    }
    return biggest;
}

+ (NSArray<NSValue *> *)detectCoinRectsInCGImage:(CGImageRef)image {
    if (image == nil) {
        return @[];
    }

    UIImage *uiImage = [UIImage imageWithCGImage:image];
    if (uiImage == nil) {
        return @[];
    }

    cv::Mat source;
    UIImageToMat(uiImage, source);
    if (source.empty()) {
        return @[];
    }

    cv::Mat bgr;
    if (source.channels() == 4) {
        cv::cvtColor(source, bgr, cv::COLOR_RGBA2BGR);
    } else if (source.channels() == 3) {
        bgr = source;
    } else {
        cv::cvtColor(source, bgr, cv::COLOR_GRAY2BGR);
    }

    // Match Python preprocessing in src/PreProcessing.py.
    cv::Mat blurred;
    cv::GaussianBlur(bgr, blurred, cv::Size(5, 5), 3.0, 3.0);

    cv::Mat edges;
    cv::Canny(blurred, edges, 90, 140);

    cv::Mat kernel = cv::Mat::ones(4, 4, CV_8U);
    cv::Mat dilated;
    cv::dilate(edges, dilated, kernel, cv::Point(-1, -1), 2);

    cv::Mat preprocessed;
    cv::erode(dilated, preprocessed, kernel, cv::Point(-1, -1), 1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(preprocessed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    NSMutableArray<NSValue *> *rects = [NSMutableArray array];
    for (const auto& contour : contours) {
        if (contour.size() < 5) {
            continue;
        }

        double area = cv::contourArea(contour);
        if (area <= 2000.0) {
            continue;
        }
        
        // Additional check: minimum radius for a real coin
        double minRadius = std::sqrt(area / M_PI);
        if (minRadius < 30.0) {
            continue;
        }

        cv::RotatedRect ellipse = cv::fitEllipse(contour);
        cv::Point2f vertices[4];
        ellipse.points(vertices);

        std::vector<float> distances;
        distances.reserve(4);
        for (int i = 0; i < 4; ++i) {
            int j = (i + 1) % 4;
            float dx = vertices[i].x - vertices[j].x;
            float dy = vertices[i].y - vertices[j].y;
            distances.push_back(std::sqrt(dx * dx + dy * dy));
        }

        float maxDistance = *std::max_element(distances.begin(), distances.end());
        float diff = biggestDifference(distances);

        // Same 10% circle-likeness tolerance as Python Ellipse.py.
        if ((maxDistance * 0.1f) < diff) {
            continue;
        }

        // Use a circular ROI (inscribed from fitted ellipse) so detection stays truly coin-shaped.
        float radius = std::min(ellipse.size.width, ellipse.size.height) * 0.5f;
        if (radius < 30.0f) {
            continue;
        }

        CGRect rect = CGRectMake(ellipse.center.x - radius,
                                 ellipse.center.y - radius,
                                 2.0f * radius,
                                 2.0f * radius);
        rect = CGRectIntersection(rect, CGRectMake(0, 0, bgr.cols, bgr.rows));
        if (CGRectIsNull(rect) || rect.size.width < 60.0 || rect.size.height < 60.0) {
            continue;
        }

        [rects addObject:[NSValue valueWithCGRect:CGRectIntegral(rect)]];
    }

    return rects;
}

@end
