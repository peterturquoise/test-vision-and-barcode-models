package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"log"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// QueueMessage represents a message from the input queue
type QueueMessage struct {
	JobID       string `json:"jobId"`
	ImageData   []byte `json:"imageData"`
	Pattern     string `json:"pattern"`
	Priority    string `json:"priority"`
	Timestamp   string `json:"timestamp"`
	ClientID    string `json:"clientId"`
	CallbackURL string `json:"callbackURL"`
}

// ProcessingResult represents the result of processing an image
type ProcessingResult struct {
	JobID          string          `json:"jobId"`
	Status         string          `json:"status"`
	Barcodes       []BarcodeResult `json:"barcodes"`
	ProcessingTime float64         `json:"processingTime"`
	Error          string          `json:"error,omitempty"`
	CallbackURL    string          `json:"callbackURL"`
	Timestamp      string          `json:"timestamp"`
}

// BarcodeResult represents a detected barcode (matching Python container format)
type BarcodeResult struct {
	Value       string  `json:"value"`
	BarcodeType string  `json:"barcode_type"`
	Bbox        []int   `json:"bbox"` // [x1, y1, x2, y2]
	Rotation    int     `json:"rotation"`
	HasError    bool    `json:"has_error"`
	ErrorType   *string `json:"error_type"`
	Confidence  float64 `json:"confidence"`
	Approach    string  `json:"approach"`
}

// QueueWorker handles barcode processing from a queue
type QueueWorker struct {
	workerID    int
	zxingPath   string
	tempDir     string
	processed   int64
	errors      int64
	mu          sync.RWMutex
	inputQueue  QueueClient
	outputQueue QueueClient
	running     bool
	stopChan    chan bool
}

// QueueClient interface for queue operations
type QueueClient interface {
	Receive() (*QueueMessage, error)
	SendResult(result *ProcessingResult) error
	Close() error
}

// NewQueueWorker creates a new QueueWorker instance
func NewQueueWorker(workerID int, inputQueue, outputQueue QueueClient) *QueueWorker {
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("zxing-worker-%d-temp", workerID))
	if err != nil {
		log.Fatalf("Worker %d: Failed to create temp directory: %v", workerID, err)
	}
	return &QueueWorker{
		workerID:    workerID,
		zxingPath:   "/usr/local/bin/ZXingReader",
		tempDir:     tempDir,
		inputQueue:  inputQueue,
		outputQueue: outputQueue,
		stopChan:    make(chan bool),
	}
}

// Start begins processing messages from the queue
func (w *QueueWorker) Start() {
	w.running = true
	log.Printf("Worker %d: Started and monitoring queue", w.workerID)

	for w.running {
		select {
		case <-w.stopChan:
			log.Printf("Worker %d: Received stop signal", w.workerID)
			w.running = false
		default:
			// Try to receive a message from the queue
			message, err := w.inputQueue.Receive()
			if err != nil {
				// No message available or error - wait a bit
				time.Sleep(100 * time.Millisecond)
				continue
			}

			// Process the message
			result := w.ProcessMessage(*message)

			// Send result to output queue
			if err := w.outputQueue.SendResult(&result); err != nil {
				log.Printf("Worker %d: Failed to send result for job %s: %v", w.workerID, message.JobID, err)
			}
		}
	}

	log.Printf("Worker %d: Stopped", w.workerID)
}

// Stop gracefully stops the worker
func (w *QueueWorker) Stop() {
	w.running = false
	w.stopChan <- true
	os.RemoveAll(w.tempDir) // Clean up temp directory
	log.Printf("Worker %d: Stopped and cleaned up temp directory %s", w.workerID, w.tempDir)
}

// ProcessMessage processes a single message from the queue
func (w *QueueWorker) ProcessMessage(message QueueMessage) ProcessingResult {
	startTime := time.Now()

	log.Printf("Worker %d: Processing job %s", w.workerID, message.JobID)

	// Process the image
	barcodes, err := w.processImage(message.ImageData, message.Pattern)
	processingTime := time.Since(startTime).Seconds()

	// Create result
	result := ProcessingResult{
		JobID:          message.JobID,
		Status:         "success",
		Barcodes:       barcodes,
		ProcessingTime: processingTime,
		Error:          "",
		CallbackURL:    message.CallbackURL,
		Timestamp:      time.Now().UTC().Format(time.RFC3339),
	}

	if err != nil {
		result.Status = "error"
		result.Error = err.Error()
		w.mu.Lock()
		w.errors++
		w.mu.Unlock()
		log.Printf("Worker %d: Error processing job %s: %v", w.workerID, message.JobID, err)
	} else {
		w.mu.Lock()
		w.processed++
		w.mu.Unlock()
		log.Printf("Worker %d: Processed job %s: success (%.3fs) - Found %d barcodes",
			w.workerID, message.JobID, processingTime, len(barcodes))
	}

	return result
}

// processImage processes a single image using the ZXingReader binary
func (w *QueueWorker) processImage(imageData []byte, pattern string) ([]BarcodeResult, error) {
	// Create a temporary file for the image
	tempFile, err := os.CreateTemp(w.tempDir, "image-*.png")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp image file: %v", err)
	}
	defer os.Remove(tempFile.Name()) // Clean up the temporary file
	defer tempFile.Close()

	// Decode image from bytes
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %v", err)
	}

	// Save image as PNG to the temporary file
	if err := png.Encode(tempFile, img); err != nil {
		return nil, fmt.Errorf("failed to encode image to PNG: %v", err)
	}
	tempFile.Close() // Close the file before running ZXingReader

	// Run ZXing-C++ binary with multiple approaches
	allBarcodes := []BarcodeResult{}

	// Define preprocessing approaches (same as Python implementation)
	approaches := []struct {
		name string
		args []string
	}{
		{"Higher Contrast", []string{"-errors"}},
		{"Scale 1.5x", []string{"-errors"}},
		{"Scale 2.0x", []string{"-errors"}},
		{"CLAHE Enhancement", []string{"-errors"}},
		{"Original", []string{"-errors"}},
		{"Grayscale", []string{"-errors"}},
		{"Threshold Binary", []string{"-errors"}},
	}

	// Try each approach with actual preprocessing
	for _, approach := range approaches {
		// Preprocess the image based on approach
		processedImg := w.preprocessImage(img, approach.name)

		// Save processed image to a new temp file
		processedTempFile, err := os.CreateTemp(w.tempDir, fmt.Sprintf("processed-%s-*.png", approach.name))
		if err != nil {
			log.Printf("Worker %d: Failed to create temp file for %s: %v", w.workerID, approach.name, err)
			continue
		}

		if err := png.Encode(processedTempFile, processedImg); err != nil {
			log.Printf("Worker %d: Failed to encode processed image for %s: %v", w.workerID, approach.name, err)
			processedTempFile.Close()
			os.Remove(processedTempFile.Name())
			continue
		}
		processedTempFile.Close()

		// Run ZXingReader on the processed image
		cmd := exec.Command(w.zxingPath, append(approach.args, processedTempFile.Name())...)
		output, _ := cmd.CombinedOutput()

		// Parse output regardless of exit status
		barcodes := w.parseZXingOutput(string(output), approach.name)

		if len(barcodes) > 0 {
			allBarcodes = append(allBarcodes, barcodes...)
			log.Printf("Worker %d: %s found %d barcodes", w.workerID, approach.name, len(barcodes))
		}

		// Clean up processed temp file
		os.Remove(processedTempFile.Name())
	}

	// Deduplicate results
	uniqueBarcodes := w.deduplicateBarcodes(allBarcodes)

	// Filter by pattern if provided
	if pattern != "" {
		var filteredBarcodes []BarcodeResult
		for _, b := range uniqueBarcodes {
			if strings.HasPrefix(b.Value, pattern) {
				filteredBarcodes = append(filteredBarcodes, b)
			}
		}
		return filteredBarcodes, nil
	}

	return uniqueBarcodes, nil
}

// preprocessImage applies preprocessing based on the approach (using Go's built-in image processing)
func (w *QueueWorker) preprocessImage(img image.Image, approach string) image.Image {
	switch approach {
	case "Original":
		return img

	case "Grayscale":
		// Convert to grayscale
		bounds := img.Bounds()
		gray := image.NewGray(bounds)
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				gray.Set(x, y, img.At(x, y))
			}
		}
		return gray

	case "Higher Contrast":
		// Increase contrast by scaling pixel values
		bounds := img.Bounds()
		contrast := image.NewRGBA(bounds)
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				r, g, b, a := img.At(x, y).RGBA()
				// Scale by 1.5 and clamp to 0-65535
				newR := uint32(float64(r) * 1.5)
				newG := uint32(float64(g) * 1.5)
				newB := uint32(float64(b) * 1.5)
				if newR > 65535 {
					newR = 65535
				}
				if newG > 65535 {
					newG = 65535
				}
				if newB > 65535 {
					newB = 65535
				}
				contrast.Set(x, y, color.RGBA64{
					R: uint16(newR),
					G: uint16(newG),
					B: uint16(newB),
					A: uint16(a),
				})
			}
		}
		return contrast

	case "Scale 1.5x":
		// Scale image by 1.5x
		bounds := img.Bounds()
		newWidth := int(float64(bounds.Dx()) * 1.5)
		newHeight := int(float64(bounds.Dy()) * 1.5)
		scaled := image.NewRGBA(image.Rect(0, 0, newWidth, newHeight))

		for y := 0; y < newHeight; y++ {
			for x := 0; x < newWidth; x++ {
				// Map to original image coordinates
				srcX := int(float64(x) / 1.5)
				srcY := int(float64(y) / 1.5)
				if srcX < bounds.Dx() && srcY < bounds.Dy() {
					scaled.Set(x, y, img.At(bounds.Min.X+srcX, bounds.Min.Y+srcY))
				}
			}
		}
		return scaled

	case "Scale 2.0x":
		// Scale image by 2.0x
		bounds := img.Bounds()
		newWidth := int(float64(bounds.Dx()) * 2.0)
		newHeight := int(float64(bounds.Dy()) * 2.0)
		scaled := image.NewRGBA(image.Rect(0, 0, newWidth, newHeight))

		for y := 0; y < newHeight; y++ {
			for x := 0; x < newWidth; x++ {
				// Map to original image coordinates
				srcX := int(float64(x) / 2.0)
				srcY := int(float64(y) / 2.0)
				if srcX < bounds.Dx() && srcY < bounds.Dy() {
					scaled.Set(x, y, img.At(bounds.Min.X+srcX, bounds.Min.Y+srcY))
				}
			}
		}
		return scaled

	case "Threshold Binary":
		// Convert to binary threshold
		bounds := img.Bounds()
		binary := image.NewGray(bounds)
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				r, g, b, _ := img.At(x, y).RGBA()
				// Convert to grayscale and apply threshold
				gray := (r + g + b) / 3
				if gray > 128*256 { // Threshold at 128
					binary.SetGray(x, y, color.Gray{Y: 255})
				} else {
					binary.SetGray(x, y, color.Gray{Y: 0})
				}
			}
		}
		return binary

	case "CLAHE Enhancement":
		// Simple histogram equalization (approximation of CLAHE)
		bounds := img.Bounds()
		enhanced := image.NewGray(bounds)

		// Convert to grayscale first
		gray := image.NewGray(bounds)
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				gray.Set(x, y, img.At(x, y))
			}
		}

		// Simple histogram equalization
		histogram := make([]int, 256)
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				g := gray.GrayAt(x, y)
				histogram[g.Y]++
			}
		}

		// Create cumulative histogram
		cumulative := make([]int, 256)
		cumulative[0] = histogram[0]
		for i := 1; i < 256; i++ {
			cumulative[i] = cumulative[i-1] + histogram[i]
		}

		// Apply histogram equalization
		totalPixels := bounds.Dx() * bounds.Dy()
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				g := gray.GrayAt(x, y)
				newValue := uint8((cumulative[g.Y] * 255) / totalPixels)
				enhanced.SetGray(x, y, color.Gray{Y: newValue})
			}
		}

		return enhanced

	default:
		return img
	}
}

// parseZXingOutput parses ZXing-C++ output
func (w *QueueWorker) parseZXingOutput(output, approach string) []BarcodeResult {
	var barcodes []BarcodeResult

	if strings.TrimSpace(output) == "" {
		return barcodes
	}

	// Parse ZXingReader output (single barcode format)
	lines := strings.Split(strings.TrimSpace(output), "\n")
	barcode := BarcodeResult{
		Approach: approach,
	}

	// Parse each line
	for _, line := range lines {
		line = strings.TrimSpace(line)

		if strings.HasPrefix(line, "Text:") {
			// Extract quoted text: Text:       "844030"
			re := regexp.MustCompile(`Text:\s+"([^"]+)"`)
			matches := re.FindStringSubmatch(line)
			if len(matches) > 1 {
				barcode.Value = matches[1]
			}
		} else if strings.HasPrefix(line, "Format:") {
			// Extract format: Format:     Code128
			re := regexp.MustCompile(`Format:\s+(\w+)`)
			matches := re.FindStringSubmatch(line)
			if len(matches) > 1 {
				barcode.BarcodeType = matches[1]
			}
		} else if strings.HasPrefix(line, "Position:") {
			// Extract position: Position:   634x1184 717x1184 717x1186 634x1186
			re := regexp.MustCompile(`Position:\s+(\d+)x(\d+)\s+(\d+)x(\d+)\s+(\d+)x(\d+)\s+(\d+)x(\d+)`)
			matches := re.FindStringSubmatch(line)
			if len(matches) >= 9 {
				x1, _ := strconv.Atoi(matches[1])
				y1, _ := strconv.Atoi(matches[2])
				x2, _ := strconv.Atoi(matches[3])
				y2, _ := strconv.Atoi(matches[4])
				x3, _ := strconv.Atoi(matches[5])
				y3, _ := strconv.Atoi(matches[6])
				x4, _ := strconv.Atoi(matches[7])
				y4, _ := strconv.Atoi(matches[8])

				// Calculate bounding box [x1, y1, x2, y2]
				minX := min(x1, x2, x3, x4)
				maxX := max(x1, x2, x3, x4)
				minY := min(y1, y2, y3, y4)
				maxY := max(y1, y2, y3, y4)

				barcode.Bbox = []int{minX, minY, maxX, maxY}
			}
		} else if strings.HasPrefix(line, "Rotation:") {
			// Extract rotation: Rotation:   0 deg
			re := regexp.MustCompile(`Rotation:\s+(-?\d+)\s+deg`)
			matches := re.FindStringSubmatch(line)
			if len(matches) > 1 {
				if rot, err := strconv.Atoi(matches[1]); err == nil {
					barcode.Rotation = rot
				}
			}
		} else if strings.HasPrefix(line, "Error:") {
			// Extract error: Error:      ChecksumError @ ODCode128Reader.cpp:238
			barcode.HasError = true
			errorType := strings.TrimPrefix(line, "Error:")
			errorType = strings.TrimSpace(errorType)
			barcode.ErrorType = &errorType
		}
	}

	// Only add if we have a value
	if barcode.Value != "" {
		// Set defaults
		if barcode.BarcodeType == "" {
			barcode.BarcodeType = "UNKNOWN"
		}
		if barcode.Bbox == nil {
			barcode.Bbox = []int{0, 0, 0, 0}
		}
		if !barcode.HasError {
			barcode.Confidence = 0.95
		} else {
			barcode.Confidence = 0.8
		}

		barcodes = append(barcodes, barcode)
	}

	return barcodes
}

// Helper functions for min/max
func min(a, b, c, d int) int {
	res := a
	if b < res {
		res = b
	}
	if c < res {
		res = c
	}
	if d < res {
		res = d
	}
	return res
}

func max(a, b, c, d int) int {
	res := a
	if b > res {
		res = b
	}
	if c > res {
		res = c
	}
	if d > res {
		res = d
	}
	return res
}

// deduplicateBarcodes removes duplicate barcodes
func (w *QueueWorker) deduplicateBarcodes(barcodes []BarcodeResult) []BarcodeResult {
	seen := make(map[string]BarcodeResult)

	for _, barcode := range barcodes {
		if existing, exists := seen[barcode.Value]; exists {
			// If both have same error status, prefer higher confidence
			if barcode.Confidence > existing.Confidence {
				seen[barcode.Value] = barcode
			}
		} else {
			seen[barcode.Value] = barcode
		}
	}

	result := make([]BarcodeResult, 0, len(seen))
	for _, barcode := range seen {
		result = append(result, barcode)
	}

	return result
}

// GetStats returns current worker statistics
func (w *QueueWorker) GetStats() map[string]interface{} {
	w.mu.RLock()
	defer w.mu.RUnlock()

	return map[string]interface{}{
		"worker_id": w.workerID,
		"processed": w.processed,
		"errors":    w.errors,
		"status":    "running",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	}
}
