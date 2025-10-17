package main

import (
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/streadway/amqp"
	"gocv.io/x/gocv"
)

// SimpleWorker processes messages from RabbitMQ
type SimpleWorker struct {
	workerID  int
	conn      *amqp.Connection
	channel   *amqp.Channel
	processed int64
	errors    int64
	mu        sync.RWMutex
	running   bool
}

func NewSimpleWorker(workerID int, rabbitmqURL string) (*SimpleWorker, error) {
	conn, err := amqp.Dial(rabbitmqURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to RabbitMQ: %v", err)
	}

	channel, err := conn.Channel()
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to open channel: %v", err)
	}

	// Declare queues
	_, err = channel.QueueDeclare("image-processing", true, false, false, false, nil)
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to declare input queue: %v", err)
	}

	_, err = channel.QueueDeclare("results", true, false, false, false, nil)
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to declare output queue: %v", err)
	}

	// Set QoS
	err = channel.Qos(1, 0, false)
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to set QoS: %v", err)
	}

	return &SimpleWorker{
		workerID: workerID,
		conn:     conn,
		channel:  channel,
		running:  true,
	}, nil
}

func (w *SimpleWorker) Start() {
	log.Printf("Worker %d: Started", w.workerID)

	// Consume messages
	msgs, err := w.channel.Consume("image-processing", "", false, false, false, false, nil)
	if err != nil {
		log.Printf("Worker %d: Failed to register consumer: %v", w.workerID, err)
		return
	}

	for w.running {
		select {
		case msg := <-msgs:
			w.processMessage(msg)
		case <-time.After(100 * time.Millisecond):
			// Check if still running
		}
	}

	log.Printf("Worker %d: Stopped", w.workerID)
}

func (w *SimpleWorker) processMessage(delivery amqp.Delivery) {
	startTime := time.Now()

	// Parse message
	var imgMsg ImageMessage
	if err := json.Unmarshal(delivery.Body, &imgMsg); err != nil {
		log.Printf("Worker %d: Failed to unmarshal message: %v", w.workerID, err)
		delivery.Nack(false, false)
		w.mu.Lock()
		w.errors++
		w.mu.Unlock()
		return
	}

	log.Printf("Worker %d: Processing job %s", w.workerID, imgMsg.JobID)

	// Process image
	barcodes, err := w.processImage(imgMsg.ImageData)
	processingTime := time.Since(startTime).Seconds()

	// Create result
	result := ResultMessage{
		JobID:          imgMsg.JobID,
		Status:         "success",
		Barcodes:       barcodes,
		ProcessingTime: processingTime,
		Timestamp:      time.Now().UTC().Format(time.RFC3339),
	}

	if err != nil {
		result.Status = "error"
		result.Error = err.Error()
		w.mu.Lock()
		w.errors++
		w.mu.Unlock()
		log.Printf("Worker %d: Error processing job %s: %v", w.workerID, imgMsg.JobID, err)
	} else {
		w.mu.Lock()
		w.processed++
		w.mu.Unlock()
		log.Printf("Worker %d: Processed job %s: success (%.3fs) - Found %d barcodes",
			w.workerID, imgMsg.JobID, processingTime, len(barcodes))
	}

	// Send result
	resultBody, _ := json.Marshal(result)
	err = w.channel.Publish("", "results", false, false, amqp.Publishing{
		ContentType:   "application/json",
		Body:          resultBody,
		DeliveryMode:  amqp.Persistent,
		CorrelationId: imgMsg.JobID,
	})

	if err != nil {
		log.Printf("Worker %d: Failed to publish result: %v", w.workerID, err)
	}

	// Acknowledge original message
	delivery.Ack(false)
}

func (w *SimpleWorker) processImage(imageData []byte) ([]BarcodeResult, error) {
	// Convert image bytes to OpenCV Mat
	img, err := gocv.IMDecode(imageData, gocv.IMReadColor)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %v", err)
	}
	defer img.Close()

	// Run multiple preprocessing approaches like Python version
	allBarcodes := []BarcodeResult{}

	// Define approaches (same as Python implementation)
	approaches := []string{
		"Higher Contrast",
		"Scale 1.5x",
		"Scale 2.0x",
		"CLAHE Enhancement",
		"Original",
		"Grayscale",
		"Threshold Binary",
	}

	// Try each approach
	for _, approach := range approaches {
		processedImg := w.preprocessImage(img, approach)
		if processedImg.Empty() {
			log.Printf("Worker %d: Failed to preprocess image with approach %s", w.workerID, approach)
			continue
		}

		// Save processed image to temporary file
		tempFile, err := os.CreateTemp("", fmt.Sprintf("processed-%s-*.png", approach))
		if err != nil {
			log.Printf("Worker %d: Failed to create temp file for %s: %v", w.workerID, approach, err)
			processedImg.Close()
			continue
		}
		tempPath := tempFile.Name()
		tempFile.Close()

		// Save processed image as PNG
		if !gocv.IMWrite(tempPath, processedImg) {
			log.Printf("Worker %d: Failed to save processed image for %s", w.workerID, approach)
			processedImg.Close()
			os.Remove(tempPath)
			continue
		}
		processedImg.Close()

		// Run ZXingReader on processed image
		cmd := exec.Command("/usr/local/bin/ZXingReader", "-errors", tempPath)
		output, _ := cmd.CombinedOutput()

		barcodes := w.parseZXingOutput(string(output), approach)
		if len(barcodes) > 0 {
			allBarcodes = append(allBarcodes, barcodes...)
			log.Printf("Worker %d: %s found %d barcodes", w.workerID, approach, len(barcodes))
		}

		// Clean up temp file
		os.Remove(tempPath)
	}

	// Deduplicate results
	uniqueBarcodes := w.deduplicateBarcodes(allBarcodes)
	return uniqueBarcodes, nil
}

// preprocessImage applies preprocessing based on the approach (same as Python version)
func (w *SimpleWorker) preprocessImage(img gocv.Mat, approach string) gocv.Mat {
	switch approach {
	case "Original":
		return img.Clone()

	case "Grayscale":
		if img.Channels() == 3 {
			gray := gocv.NewMat()
			gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)
			return gray
		}
		return img.Clone()

	case "Higher Contrast":
		// cv2.convertScaleAbs(image, alpha=1.5, beta=0)
		result := gocv.NewMat()
		img.ConvertTo(&result, gocv.MatTypeCV8U, 1.5, 0)
		return result

	case "Scale 1.5x":
		// cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
		height := img.Rows()
		width := img.Cols()
		newWidth := int(float64(width) * 1.5)
		newHeight := int(float64(height) * 1.5)

		result := gocv.NewMat()
		gocv.Resize(img, &result, image.Pt(newWidth, newHeight), 0, 0, gocv.InterpolationCubic)
		return result

	case "Scale 2.0x":
		height := img.Rows()
		width := img.Cols()
		newWidth := int(float64(width) * 2.0)
		newHeight := int(float64(height) * 2.0)

		result := gocv.NewMat()
		gocv.Resize(img, &result, image.Pt(newWidth, newHeight), 0, 0, gocv.InterpolationCubic)
		return result

	case "Threshold Binary":
		// cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
		var gray gocv.Mat
		if img.Channels() == 3 {
			gray = gocv.NewMat()
			gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)
		} else {
			gray = img.Clone()
		}

		result := gocv.NewMat()
		gocv.Threshold(gray, &result, 128, 255, gocv.ThresholdBinary)
		gray.Close()
		return result

	case "CLAHE Enhancement":
		// cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		var gray gocv.Mat
		if img.Channels() == 3 {
			gray = gocv.NewMat()
			gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)
		} else {
			gray = img.Clone()
		}

		clahe := gocv.NewCLAHE()
		clahe.SetClipLimit(2.0)
		clahe.SetTilesGridSize(image.Pt(8, 8))

		result := gocv.NewMat()
		clahe.Apply(gray, &result)

		gray.Close()
		clahe.Close()
		return result

	default:
		return img.Clone()
	}
}

func (w *SimpleWorker) parseZXingOutput(output, approach string) []BarcodeResult {
	var barcodes []BarcodeResult

	if strings.TrimSpace(output) == "" {
		return barcodes
	}

	lines := strings.Split(strings.TrimSpace(output), "\n")
	barcode := BarcodeResult{
		Approach: approach,
	}

	for _, line := range lines {
		line = strings.TrimSpace(line)

		if strings.HasPrefix(line, "Text:") {
			re := regexp.MustCompile(`Text:\s+"([^"]+)"`)
			matches := re.FindStringSubmatch(line)
			if len(matches) > 1 {
				barcode.Value = matches[1]
			}
		} else if strings.HasPrefix(line, "Format:") {
			re := regexp.MustCompile(`Format:\s+(\w+)`)
			matches := re.FindStringSubmatch(line)
			if len(matches) > 1 {
				barcode.BarcodeType = matches[1]
			}
		} else if strings.HasPrefix(line, "Position:") {
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

				minX := min(x1, x2, x3, x4)
				maxX := max(x1, x2, x3, x4)
				minY := min(y1, y2, y3, y4)
				maxY := max(y1, y2, y3, y4)

				barcode.Bbox = []int{minX, minY, maxX, maxY}
			}
		} else if strings.HasPrefix(line, "Error:") {
			barcode.HasError = true
			errorType := strings.TrimPrefix(line, "Error:")
			errorType = strings.TrimSpace(errorType)
			barcode.ErrorType = &errorType
		}
	}

	if barcode.Value != "" {
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

// deduplicateBarcodes removes duplicate barcodes
func (w *SimpleWorker) deduplicateBarcodes(barcodes []BarcodeResult) []BarcodeResult {
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

// Main function for the simple worker
func main() {
	// Get number of workers from environment or default to 4
	workers := 4
	if envWorkers := os.Getenv("WORKERS"); envWorkers != "" {
		if n, err := fmt.Sscanf(envWorkers, "%d", &workers); err != nil || n != 1 {
			log.Printf("Invalid WORKERS value, using default: %d", workers)
		}
	}

	rabbitmqURL := os.Getenv("RABBITMQ_URL")
	if rabbitmqURL == "" {
		rabbitmqURL = "amqp://admin:admin123@localhost:5672/"
	}

	inputQueue := "image-processing"
	outputQueue := "results"

	log.Printf("🚀 Starting Simple ZXing Worker with %d workers", workers)
	log.Printf("RabbitMQ URL: %s", rabbitmqURL)

	// Create and start workers
	workerList := make([]*SimpleWorker, workers)
	for i := 0; i < workers; i++ {
		worker, err := NewSimpleWorker(i, rabbitmqURL)
		if err != nil {
			log.Fatalf("Failed to create worker %d: %v", i, err)
		}
		workerList[i] = worker
		go worker.Start()
	}

	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Println("Shutting down workers...")
	for _, worker := range workerList {
		worker.Stop()
	}
	log.Println("All workers stopped.")
}

func (w *SimpleWorker) Stop() {
	w.running = false
	w.channel.Close()
	w.conn.Close()
}

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
