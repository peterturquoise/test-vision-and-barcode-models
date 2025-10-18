package main

/*
#cgo LDFLAGS: -L. -lcpp_processor -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_objdetect -lZXing
#include <stdlib.h>
#include "cpp_processor.h"
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"log"
	"time"
	"unsafe"

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
	ErrorLevel  string  `json:"error_level"`
	Approach    string  `json:"approach"`
	Lines       int     `json:"lines"`
}

// ZXingCGOWorker processes images using CGO + C++ OpenCV + ZXing-CPP
type ZXingCGOWorker struct{}

// NewZXingCGOWorker creates a new CGO-based ZXing worker
func NewZXingCGOWorker() *ZXingCGOWorker {
	return &ZXingCGOWorker{}
}

// ProcessImage processes image data using CGO + C++ (memory-only)
func (z *ZXingCGOWorker) ProcessImage(imageData []byte) ([]BarcodeResult, error) {
	fmt.Printf("🔧 CGO: Processing image with %d bytes\n", len(imageData))

	// Convert Go byte slice to C array
	cImageData := C.CBytes(imageData)
	defer C.free(cImageData)

	// Call C++ processing function
	fmt.Printf("🔧 CGO: Calling C++ processImageWithAllApproaches\n")
	cResult := C.processImageWithAllApproaches((*C.uchar)(cImageData), C.int(len(imageData)))
	defer C.freeProcessingResult(cResult)

	if cResult == nil {
		fmt.Printf("❌ CGO: C++ function returned nil\n")
		return nil, fmt.Errorf("failed to process image")
	}

	fmt.Printf("🔧 CGO: C++ function returned %d results\n", int(cResult.count))

	// Convert C results to Go
	var results []BarcodeResult
	for i := 0; i < int(cResult.count); i++ {
		// Use unsafe pointer arithmetic to access array elements
		cBarcode := (*C.C_BarcodeResult)(unsafe.Pointer(uintptr(unsafe.Pointer(cResult.results)) + uintptr(i)*unsafe.Sizeof(C.C_BarcodeResult{})))

		result := BarcodeResult{
			Value:       C.GoString(cBarcode.value),
			BarcodeType: C.GoString(cBarcode.format),
			Bbox: []int{
				int(cBarcode.bbox[0]),
				int(cBarcode.bbox[1]),
				int(cBarcode.bbox[2]),
				int(cBarcode.bbox[3]),
			},
			Rotation:   int(cBarcode.rotation),
			HasError:   cBarcode.has_error != 0,
			ErrorLevel: C.GoString(cBarcode.error_level),
			Approach:   C.GoString(cBarcode.approach),
			Lines:      int(cBarcode.lines),
		}

		// Handle error type
		if cBarcode.error_type != nil {
			errorType := C.GoString(cBarcode.error_type)
			result.ErrorType = &errorType
		}

		results = append(results, result)
	}

	return results, nil
}

func main() {
	fmt.Println("🚀 C++ ZXing CGO Memory Worker Starting...")
	fmt.Println("📸 Processing images with OpenCV + ZXing-CPP (memory-only!)")

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

	worker := NewZXingCGOWorker()
	fmt.Println("✅ CGO Worker ready, waiting for images...")

	// Process messages
	for d := range msgs {
		start := time.Now()
		fmt.Printf("📥 Received image message: %d bytes\n", len(d.Body))

		// Process image using CGO + C++
		results, err := worker.ProcessImage(d.Body)
		if err != nil {
			fmt.Printf("❌ Error processing image: %v\n", err)
			continue
		}

		processingTime := time.Since(start)
		fmt.Printf("⚡ Processed in %v - Found %d barcodes\n", processingTime, len(results))

		// Log each result with its approach
		for i, result := range results {
			fmt.Printf("  📊 Result %d: %s (%s) - %s\n", i+1, result.Value, result.BarcodeType, result.Approach)
		}

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
