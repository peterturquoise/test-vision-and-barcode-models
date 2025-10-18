#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <ZXing/ReadBarcode.h>
#include <ZXing/ImageView.h>
#include <vector>
#include <future>
#include <thread>

extern "C" {

// C structure for barcode results
typedef struct {
    char* value;
    char* format;
    int bbox[4];
    int rotation;
    char* approach;
    int has_error;
    char* error_type;
    char* error_level;
    int lines;
} C_BarcodeResult;

// C structure for processing results
typedef struct {
    C_BarcodeResult* results;
    int count;
} C_ProcessingResult;

// Helper function to convert std::string to C string
char* string_to_cstr(const std::string& str) {
    char* cstr = (char*)malloc(str.length() + 1);
    strcpy(cstr, str.c_str());
    return cstr;
}

// Helper function to free C string
void free_cstr(char* cstr) {
    if (cstr) free(cstr);
}

// Process image with a specific approach
std::vector<ZXing::Barcode> processWithApproach(const cv::Mat& image, const std::string& approach) {
    cv::Mat processed;
    
    if (approach == "Original") {
        processed = image.clone();
    } else if (approach == "Grayscale") {
        if (image.channels() == 3) {
            cv::cvtColor(image, processed, cv::COLOR_BGR2GRAY);
        } else {
            processed = image.clone();
        }
    } else if (approach == "Higher Contrast") {
        cv::convertScaleAbs(image, processed, 1.5, 0);
    } else if (approach == "Scale 1.5x") {
        int new_width = static_cast<int>(image.cols * 1.5);
        int new_height = static_cast<int>(image.rows * 1.5);
        cv::resize(image, processed, cv::Size(new_width, new_height), 0, 0, cv::INTER_CUBIC);
    } else if (approach == "Scale 2.0x") {
        int new_width = static_cast<int>(image.cols * 2.0);
        int new_height = static_cast<int>(image.rows * 2.0);
        cv::resize(image, processed, cv::Size(new_width, new_height), 0, 0, cv::INTER_CUBIC);
    } else if (approach == "Threshold Binary") {
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }
        cv::threshold(gray, processed, 127, 255, cv::THRESH_BINARY);
    } else if (approach == "CLAHE Enhancement") {
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(gray, processed);
    } else {
        processed = image.clone();
    }
    
    // Try both color and grayscale formats for ZXing
    std::vector<ZXing::Barcode> allBarcodes;
    
    // Try grayscale first
    cv::Mat gray_for_zxing;
    if (processed.channels() == 3) {
        cv::cvtColor(processed, gray_for_zxing, cv::COLOR_BGR2GRAY);
    } else {
        gray_for_zxing = processed;
    }
    
    auto grayImageView = ZXing::ImageView(gray_for_zxing.data, gray_for_zxing.cols, gray_for_zxing.rows, ZXing::ImageFormat::Lum);
    auto grayOptions = ZXing::ReaderOptions()
        .setFormats(ZXing::BarcodeFormat::Any)
        .setTryHarder(true)
        .setMaxNumberOfSymbols(10);
    
    auto grayBarcodes = ZXing::ReadBarcodes(grayImageView, grayOptions);
    allBarcodes.insert(allBarcodes.end(), grayBarcodes.begin(), grayBarcodes.end());
    
    // Try color format if grayscale didn't work
    if (processed.channels() == 3) {
        auto colorImageView = ZXing::ImageView(processed.data, processed.cols, processed.rows, ZXing::ImageFormat::BGR);
        auto colorOptions = ZXing::ReaderOptions()
            .setFormats(ZXing::BarcodeFormat::Any)
            .setTryHarder(true)
            .setMaxNumberOfSymbols(10);
        
        auto colorBarcodes = ZXing::ReadBarcodes(colorImageView, colorOptions);
        allBarcodes.insert(allBarcodes.end(), colorBarcodes.begin(), colorBarcodes.end());
    }
    
    return allBarcodes;
}

// Main processing function
C_ProcessingResult* processImageWithAllApproaches(unsigned char* imageData, int imageSize) {
    printf("🔧 C++: FUNCTION CALLED!\n");
    fflush(stdout);
    
    try {
        printf("🔧 C++: Starting image processing with %d bytes\n", imageSize);
        fflush(stdout);
        
        // Decode image from memory
        std::vector<unsigned char> buffer(imageData, imageData + imageSize);
        printf("🔧 C++: Created buffer with %zu bytes\n", buffer.size());
        fflush(stdout);
        
        cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
        printf("🔧 C++: Called cv::imdecode\n");
        fflush(stdout);
        
        if (image.empty()) {
            printf("❌ C++: Failed to decode image\n");
            fflush(stdout);
            return nullptr;
        }
        
        printf("🔧 C++: Image decoded successfully: %dx%d channels=%d\n", image.cols, image.rows, image.channels());
        fflush(stdout);
        
        // Define all preprocessing approaches
        std::vector<std::string> approaches = {
            "Original",
            "Grayscale", 
            "Higher Contrast",
            "Scale 1.5x",
            "Scale 2.0x",
            "Threshold Binary",
            "CLAHE Enhancement"
        };
        
        printf("🔧 C++: Defined %zu preprocessing approaches\n", approaches.size());
        fflush(stdout);
        printf("🔧 C++: Starting %zu preprocessing approaches in parallel\n", approaches.size());
        fflush(stdout);
        
        // Process all approaches in parallel
        printf("🔧 C++: Creating futures for parallel processing\n");
        fflush(stdout);
        
        std::vector<std::future<std::vector<ZXing::Barcode>>> futures;
        for (const auto& approach : approaches) {
            printf("🔧 C++: Creating future for approach: %s\n", approach.c_str());
            fflush(stdout);
            
            futures.push_back(std::async(std::launch::async, [&image, &approach]() {
                printf("🔧 C++: Processing approach: %s\n", approach.c_str());
                fflush(stdout);
                auto barcodes = processWithApproach(image, approach);
                printf("🔧 C++: %s found %zu barcodes\n", approach.c_str(), barcodes.size());
                fflush(stdout);
                return barcodes;
            }));
        }
        
        printf("🔧 C++: Created %zu futures, waiting for results\n", futures.size());
        fflush(stdout);
        
        // Collect all results
        std::vector<C_BarcodeResult> allResults;
        for (size_t i = 0; i < futures.size(); ++i) {
            auto barcodes = futures[i].get();
            std::string approach = approaches[i];
            
            printf("🔧 C++: Collecting results from %s: %zu barcodes\n", approach.c_str(), barcodes.size());
            
            for (const auto& barcode : barcodes) {
                C_BarcodeResult result;
                result.value = string_to_cstr(barcode.text());
                result.format = string_to_cstr(ZXing::ToString(barcode.format()));
                result.approach = string_to_cstr(approach);
                result.rotation = 0; // ZXing doesn't provide rotation info
                result.has_error = 0; // ZXing doesn't provide error info
                result.error_type = nullptr;
                result.error_level = string_to_cstr("none");
                result.lines = 0; // ZXing doesn't provide lines info
                
                // Extract bounding box
                auto position = barcode.position();
                if (position.size() >= 4) {
                    result.bbox[0] = static_cast<int>(position[0].x);
                    result.bbox[1] = static_cast<int>(position[0].y);
                    result.bbox[2] = static_cast<int>(position[2].x);
                    result.bbox[3] = static_cast<int>(position[2].y);
                } else {
                    result.bbox[0] = 0;
                    result.bbox[1] = 0;
                    result.bbox[2] = 100;
                    result.bbox[3] = 100;
                }
                
                allResults.push_back(result);
            }
        }
        
        printf("🔧 C++: Total results collected: %zu barcodes\n", allResults.size());
        
        // Create result structure
        C_ProcessingResult* cResult = (C_ProcessingResult*)malloc(sizeof(C_ProcessingResult));
        cResult->count = allResults.size();
        cResult->results = (C_BarcodeResult*)malloc(sizeof(C_BarcodeResult) * allResults.size());
        
        // Copy results
        for (size_t i = 0; i < allResults.size(); ++i) {
            cResult->results[i] = allResults[i];
        }
        
        printf("🔧 C++: Returning %d results to Go\n", cResult->count);
        return cResult;
        
    } catch (const std::exception& e) {
        // Return empty result on error
        C_ProcessingResult* cResult = (C_ProcessingResult*)malloc(sizeof(C_ProcessingResult));
        cResult->count = 0;
        cResult->results = nullptr;
        return cResult;
    }
}

// Free memory allocated by C++ code
void freeProcessingResult(C_ProcessingResult* result) {
    if (result) {
        if (result->results) {
            for (int i = 0; i < result->count; ++i) {
                free_cstr(result->results[i].value);
                free_cstr(result->results[i].format);
                free_cstr(result->results[i].approach);
                free_cstr(result->results[i].error_type);
                free_cstr(result->results[i].error_level);
            }
            free(result->results);
        }
        free(result);
    }
}

} // extern "C"
