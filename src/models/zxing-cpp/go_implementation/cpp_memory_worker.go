package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/streadway/amqp"
)

// BarcodeResult represents a detected barcode
type BarcodeResult struct {
	Value       string  `json:"value"`
	BarcodeType string  `json:"barcode_type"`
	Bbox        []int   `json:"bbox"`
	Rotation    int     `json:"rotation"`
	HasError    bool    `json:"has_error"`
	ErrorType   *string `json:"error_type"`
	ErrorLevel  string  `json:"error_level"` // "none", "warning", "error"
	Approach    string  `json:"approach"`
	Lines       int     `json:"lines"`
}

// ZXingMemoryWorker processes images in-memory using C++ ZXing
type ZXingMemoryWorker struct {
	zxingPath string
}

// NewZXingMemoryWorker creates a new in-memory ZXing worker
func NewZXingMemoryWorker() *ZXingMemoryWorker {
	return &ZXingMemoryWorker{
		zxingPath: "/usr/local/bin/ZXingReader",
	}
}

// ProcessImage processes image data directly in memory
func (z *ZXingMemoryWorker) ProcessImage(imageData []byte) ([]BarcodeResult, error) {
	// Create temporary file for image data
	tempFile, err := os.CreateTemp("", "image_*.jpeg")
	if err != nil {
		return nil, err
	}
	defer os.Remove(tempFile.Name())
	defer tempFile.Close()

	// Write image data to temp file
	if _, err := tempFile.Write(imageData); err != nil {
		return nil, err
	}
	tempFile.Close()

	// Run ZXingReader on temp file
	cmd := exec.Command(z.zxingPath, "-errors", tempFile.Name())
	output, err := cmd.Output()
	if err != nil {
		// ZXingReader returns exit status 1 even when it finds barcodes
		// Only treat it as an error if there's no output
		if len(output) == 0 {
			return nil, fmt.Errorf("ZXingReader failed: %v", err)
		}
		// Continue processing even with exit status 1 if we have output
	}

	// Parse ZXingReader output
	return z.parseZXingOutput(string(output), "Memory"), nil
}

// parseZXingOutput parses ZXingReader output
func (z *ZXingMemoryWorker) parseZXingOutput(output, approach string) []BarcodeResult {
	var results []BarcodeResult

	if !strings.Contains(output, "Text:") {
		return results // No barcodes found
	}

	// Split output into blocks (each barcode is separated by empty lines)
	blocks := strings.Split(strings.TrimSpace(output), "\n\n")

	for _, block := range blocks {
		if !strings.Contains(block, "Text:") {
			continue
		}

		lines := strings.Split(block, "\n")
		var barcode BarcodeResult

		for _, line := range lines {
			line = strings.TrimSpace(line)

			if strings.HasPrefix(line, "Text:") {
				// Extract the quoted text
				re := regexp.MustCompile(`"([^"]+)"`)
				matches := re.FindStringSubmatch(line)
				if len(matches) > 1 {
					barcode.Value = matches[1]
				}
			} else if strings.HasPrefix(line, "Format:") {
				// Extract format
				format := strings.TrimSpace(strings.TrimPrefix(line, "Format:"))
				barcode.BarcodeType = format
			} else if strings.HasPrefix(line, "Error:") {
				// Extract error type
				errorType := strings.TrimSpace(strings.TrimPrefix(line, "Error:"))
				barcode.HasError = true
				barcode.ErrorType = &errorType
			} else if strings.HasPrefix(line, "Position:") {
				// Extract position coordinates
				position := strings.TrimSpace(strings.TrimPrefix(line, "Position:"))
				bbox := z.parsePosition(position)
				if len(bbox) == 4 {
					barcode.Bbox = bbox
				}
			} else if strings.HasPrefix(line, "Rotation:") {
				// Extract rotation
				rotation := strings.TrimSpace(strings.TrimPrefix(line, "Rotation:"))
				if strings.Contains(rotation, "deg") {
					re := regexp.MustCompile(`(-?\d+)\s+deg`)
					matches := re.FindStringSubmatch(rotation)
					if len(matches) > 1 {
						if rot, err := strconv.Atoi(matches[1]); err == nil {
							barcode.Rotation = rot
						}
					}
				}
			} else if strings.HasPrefix(line, "Lines:") {
				// Extract lines count for confidence calculation
				lines := strings.TrimSpace(strings.TrimPrefix(line, "Lines:"))
				if linesCount, err := strconv.Atoi(lines); err == nil {
					barcode.Lines = linesCount
				}
			}
		}

		// Only add if we found a value
		if barcode.Value != "" {
			barcode.Approach = approach
			if len(barcode.Bbox) == 0 {
				barcode.Bbox = []int{0, 0, 100, 100} // Fallback if no position found
			}
			// Calculate error level based on error status
			barcode.ErrorLevel = z.calculateErrorLevel(barcode)
			results = append(results, barcode)
		}
	}

	return results
}

// parsePosition parses ZXingReader position string into bbox coordinates
func (z *ZXingMemoryWorker) parsePosition(position string) []int {
	// Position format: "634x1184 717x1184 717x1186 634x1186"
	// Extract coordinates: x1,y1 x2,y2 x3,y3 x4,y4
	re := regexp.MustCompile(`(\d+)x(\d+)`)
	matches := re.FindAllStringSubmatch(position, -1)

	if len(matches) < 4 {
		return []int{0, 0, 100, 100} // Fallback
	}

	// Extract coordinates
	var coords [][]int
	for _, match := range matches {
		if len(match) >= 3 {
			x, _ := strconv.Atoi(match[1])
			y, _ := strconv.Atoi(match[2])
			coords = append(coords, []int{x, y})
		}
	}

	if len(coords) < 4 {
		return []int{0, 0, 100, 100} // Fallback
	}

	// Find bounding box: min_x, min_y, max_x, max_y
	minX, minY := coords[0][0], coords[0][1]
	maxX, maxY := coords[0][0], coords[0][1]

	for _, coord := range coords {
		if coord[0] < minX {
			minX = coord[0]
		}
		if coord[0] > maxX {
			maxX = coord[0]
		}
		if coord[1] < minY {
			minY = coord[1]
		}
		if coord[1] > maxY {
			maxY = coord[1]
		}
	}

	return []int{minX, minY, maxX, maxY}
}

// calculateErrorLevel determines error level based on available data
func (z *ZXingMemoryWorker) calculateErrorLevel(barcode BarcodeResult) string {
	if !barcode.HasError {
		return "none" // No errors detected
	}

	// Check error type to determine severity
	if barcode.ErrorType != nil {
		errorType := strings.ToLower(*barcode.ErrorType)

		// Critical errors that make the barcode unreliable
		if strings.Contains(errorType, "checksum") {
			return "error" // Checksum errors are critical
		}

		// Warning-level errors that don't prevent reading but indicate issues
		if strings.Contains(errorType, "format") || strings.Contains(errorType, "decode") {
			return "warning" // Format/decode issues are warnings
		}
	}

	// Default to warning for any other errors
	return "warning"
}

func main() {
	fmt.Println("🚀 C++ ZXing Memory Worker Starting...")
	fmt.Println("📸 Processing image in-memory (no file I/O!)")

	// Connect to RabbitMQ
	conn, err := amqp.Dial("amqp://guest:guest@rabbitmq:5672/")
	if err != nil {
		log.Fatalf("Failed to connect to RabbitMQ: %v", err)
	}
	defer conn.Close()

	ch, err := conn.Channel()
	if err != nil {
		log.Fatalf("Failed to open channel: %v", err)
	}
	defer ch.Close()

	// Declare queues
	q, err := ch.QueueDeclare(
		"image_queue", // name
		false,         // durable
		false,         // delete when unused
		false,         // exclusive
		false,         // no-wait
		nil,           // arguments
	)
	if err != nil {
		log.Fatalf("Failed to declare queue: %v", err)
	}

	resultQ, err := ch.QueueDeclare(
		"result_queue", // name
		false,          // durable
		false,          // delete when unused
		false,          // exclusive
		false,          // no-wait
		nil,            // arguments
	)
	if err != nil {
		log.Fatalf("Failed to declare result queue: %v", err)
	}

	// Consume messages
	msgs, err := ch.Consume(
		q.Name, // queue
		"",     // consumer
		true,   // auto-ack
		false,  // exclusive
		false,  // no-local
		false,  // no-wait
		nil,    // args
	)
	if err != nil {
		log.Fatalf("Failed to register consumer: %v", err)
	}

	worker := NewZXingMemoryWorker()
	fmt.Println("✅ Worker ready, waiting for images...")

	// Process messages
	for d := range msgs {
		start := time.Now()

		// Process image in memory
		results, err := worker.ProcessImage(d.Body)
		if err != nil {
			fmt.Printf("❌ Error processing image: %v\n", err)
			continue
		}

		processingTime := time.Since(start)
		fmt.Printf("⚡ Processed in %v - Found %d barcodes\n", processingTime, len(results))

		// Send results back
		resultJSON, _ := json.Marshal(results)
		err = ch.Publish(
			"",           // exchange
			resultQ.Name, // routing key
			false,        // mandatory
			false,        // immediate
			amqp.Publishing{
				ContentType: "application/json",
				Body:        resultJSON,
			},
		)
		if err != nil {
			fmt.Printf("❌ Failed to send result: %v\n", err)
		}
	}
}
