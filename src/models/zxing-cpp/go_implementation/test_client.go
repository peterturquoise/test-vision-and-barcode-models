package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strings"
	"time"
)

// TestClient puts images onto the queue for testing
type TestClient struct {
	inputQueue  *MemoryQueue
	outputQueue *MemoryQueue
	imageData   []byte
}

// NewTestClient creates a new test client
func NewTestClient(inputQueue, outputQueue *MemoryQueue, imagePath string) (*TestClient, error) {
	imageData, err := ioutil.ReadFile(imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to load test image: %v", err)
	}

	return &TestClient{
		inputQueue:  inputQueue,
		outputQueue: outputQueue,
		imageData:   imageData,
	}, nil
}

// RunTest runs the test for the specified duration
func (c *TestClient) RunTest(duration time.Duration) {
	log.Printf("🚀 Starting test client for %v", duration)

	startTime := time.Now()
	endTime := startTime.Add(duration)
	jobCounter := 0

	// Start result collector
	resultChan := make(chan ProcessingResult, 1000)
	go c.collectResults(resultChan)

	// Submit jobs continuously for test duration
	for time.Now().Before(endTime) {
		message := QueueMessage{
			JobID:       fmt.Sprintf("job_%d_%d", time.Now().UnixNano(), jobCounter),
			ImageData:   c.imageData,
			Pattern:     "",
			Priority:    "normal",
			Timestamp:   time.Now().UTC().Format(time.RFC3339),
			ClientID:    "test_client",
			CallbackURL: "",
		}

		if err := c.inputQueue.SendMessage(message); err != nil {
			log.Printf("Failed to send message: %v", err)
			time.Sleep(100 * time.Millisecond)
			continue
		}

		jobCounter++

		// Small delay to prevent overwhelming the queue
		time.Sleep(10 * time.Millisecond)
	}

	log.Printf("📤 Submitted %d jobs total", jobCounter)

	// Wait a bit for remaining results
	time.Sleep(2 * time.Second)
	close(resultChan)

	// Wait for result collection to finish
	time.Sleep(1 * time.Second)
}

// collectResults collects results from the output queue
func (c *TestClient) collectResults(resultChan chan ProcessingResult) {
	var results []ProcessingResult

	for {
		result, err := c.outputQueue.GetResult()
		if err != nil {
			// No more results available
			break
		}

		results = append(results, *result)
		resultChan <- *result

		log.Printf("📊 Result: Job %s - %s (%.3fs) - %d barcodes",
			result.JobID, result.Status, result.ProcessingTime, len(result.Barcodes))
	}

	// Print final statistics
	c.printStatistics(results)
}

// printStatistics prints test statistics
func (c *TestClient) printStatistics(results []ProcessingResult) {
	if len(results) == 0 {
		log.Println("No results received")
		return
	}

	successful := 0
	failed := 0
	totalBarcodes := 0
	var totalProcessingTime float64

	for _, result := range results {
		if result.Status == "success" {
			successful++
			totalBarcodes += len(result.Barcodes)
		} else {
			failed++
		}
		totalProcessingTime += result.ProcessingTime
	}

	avgProcessingTime := totalProcessingTime / float64(len(results))

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("📊 QUEUE-BASED ARCHITECTURE TEST RESULTS")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("📈 Total Jobs: %d\n", len(results))
	fmt.Printf("✅ Successful: %d\n", successful)
	fmt.Printf("❌ Failed: %d\n", failed)
	fmt.Printf("📊 Success Rate: %.1f%%\n", float64(successful)/float64(len(results))*100)
	fmt.Printf("🔍 Total Barcodes Found: %d\n", totalBarcodes)
	fmt.Printf("⏱️  Average Processing Time: %.3fs\n", avgProcessingTime)
	fmt.Println(strings.Repeat("=", 60))
}

func main() {
	// Check for command-line arguments
	if len(os.Args) < 2 {
		log.Println("Usage: go run test_client.go <image_path> [duration]")
		log.Println("Example: go run test_client.go test_image.jpg 30s")
		return
	}

	imagePath := os.Args[1]
	duration := 20 * time.Second

	if len(os.Args) > 2 {
		if parsedDuration, err := time.ParseDuration(os.Args[2]); err == nil {
			duration = parsedDuration
		} else {
			log.Printf("Invalid duration '%s', using default 20s", os.Args[2])
		}
	}

	// Create in-memory queues (in production, these would be Azure Service Bus)
	inputQueue := NewMemoryQueue(1000)
	outputQueue := NewMemoryQueue(1000)

	// Create test client
	client, err := NewTestClient(inputQueue, outputQueue, imagePath)
	if err != nil {
		log.Fatalf("Failed to create test client: %v", err)
	}

	// Run the test
	client.RunTest(duration)
}
