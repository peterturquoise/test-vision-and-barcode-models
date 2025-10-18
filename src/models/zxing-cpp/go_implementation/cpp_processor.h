#ifndef CPP_PROCESSOR_H
#define CPP_PROCESSOR_H

#ifdef __cplusplus
extern "C" {
#endif

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

typedef struct {
    C_BarcodeResult* results;
    int count;
} C_ProcessingResult;

C_ProcessingResult* processImageWithAllApproaches(unsigned char* imageData, int imageSize);
void freeProcessingResult(C_ProcessingResult* result);

#ifdef __cplusplus
}
#endif

#endif // CPP_PROCESSOR_H
